"""Stage-1 grasp policy of the IKER shoe task: hand action law, grasp quality and reward (design 2026-09-14 §4-§6).

Pure torch, importable without Isaac Sim. The environment computes the geometry (link positions, the shoe's surface
points, speeds) and feeds it here; every function returns new tensors and ``Stage1State`` is immutable, so the
environment owns the only copy of the episode state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

HAND_EMA_ALPHA = 1.0 - 0.9**6  # grasp_fj's 0.1 per 60 Hz step, same time constant at 10 Hz (= 0.468559)
SURFACE_POINT_COUNT = 256
KERNEL_TAU_M = 0.02
SOFTMIN_TAU = 0.1
PALM_COS_MIN = 0.0
RACK_MARGIN_M = 0.01  # a shoe hull point this close to the rack's footprint counts as supported by the rack


@dataclass
class Stage1RewardCfg:
    """Design §6. ``g_min`` defaults to phase A (g == 1, design §5); ``q_lo``/``q_hi`` are then inert, and whenever
    ``g_min < 1`` the environment replaces them with the values measured on the shoe (grasp_quality_calibration.json).

    Not frozen: Isaac Lab's hydra round trip assigns every nested config field with ``setattr`` and a frozen
    dataclass raises there (the same reason as ``IkerRewardCfg``). ``setattr`` skips ``__post_init__``, so the
    environment calls ``validate`` after the round trip; ``stage1_step`` never mutates it.
    """

    palm_scale: float = 50.0
    lift_progress_scale: float = 2000.0
    # The shoe's surface points sit ~4 mm below the table in its settled pose and restitution is randomised up to 1, so
    # a reset bounces the shoe a few mm: lift progress starts only past this dead band (grasp smoke, 2026-09-14).
    lift_deadband_m: float = 0.01
    lift_height_m: float = 0.05
    lift_bonus: float = 300.0
    success_bonus: float = 1000.0
    hold_radius_m: float = 0.15
    hold_rel_speed: float = 0.05
    latch_steps: int = 3
    success_steps: int = 20
    success_speed: float = 0.05
    hand_floor_scale: float = 10.0
    hand_floor_cap: float = 5.0
    arm_vel_scale: float = 0.0
    hand_vel_scale: float = 0.0
    g_min: float = 1.0
    q_lo: float = 0.0  # placeholders, not a calibration: the whole q range, inert while g_min is 1
    q_hi: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not 0.0 < self.g_min <= 1.0:
            raise ValueError(f"g_min must be in (0, 1], got {self.g_min}")
        if not self.q_lo < self.q_hi:
            raise ValueError(f"q_lo {self.q_lo} must be < q_hi {self.q_hi}")
        if not 1 <= self.latch_steps <= self.success_steps:
            raise ValueError(f"need 1 <= latch_steps {self.latch_steps} <= success_steps {self.success_steps}")
        if not 0.0 <= self.lift_deadband_m < self.lift_height_m:
            raise ValueError(f"need 0 <= lift_deadband_m {self.lift_deadband_m} < lift_height_m {self.lift_height_m}")
        if min(self.palm_scale, self.lift_progress_scale, self.lift_bonus, self.success_bonus, self.hand_floor_scale,
               self.hand_floor_cap, self.arm_vel_scale, self.hand_vel_scale) < 0.0:
            raise ValueError("reward scales and bonuses must be non-negative")


REWARD_TERMS = ("palm_progress", "lift_progress", "lift_bonus", "success_bonus", "hand_floor", "arm_vel", "hand_vel")


def hand_action_limits(
    joint_names: Sequence[str], hard_lo: torch.Tensor, hard_hi: torch.Tensor, override: Mapping[str, tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(lo, hi) per hand joint: the URDF hard limits narrowed by the profile override (regex -> (lo|None, hi|None)).

    An override can only narrow a range. A regex that matches no joint and a range narrowed to zero width are
    errors, so a typo cannot silently leave a joint at its full (back-bending) range.
    """
    names = list(joint_names)
    if hard_lo.shape != (len(names),) or hard_hi.shape != (len(names),):
        raise ValueError(f"limits must be ({len(names)},), got {tuple(hard_lo.shape)} and {tuple(hard_hi.shape)}")
    lo, hi = hard_lo.clone(), hard_hi.clone()
    for pattern, (olo, ohi) in override.items():
        regex = re.compile(pattern)
        mask = torch.tensor([regex.fullmatch(n) is not None for n in names], device=lo.device)
        if not bool(mask.any()):
            raise ValueError(f"hand limit override {pattern!r} matches no joint of {names}")
        if olo is not None:
            lo = torch.where(mask, torch.maximum(lo, torch.full_like(lo, float(olo))), lo)
        if ohi is not None:
            hi = torch.where(mask, torch.minimum(hi, torch.full_like(hi, float(ohi))), hi)
    dead = [n for n, width in zip(names, (hi - lo).tolist()) if width <= 1e-6]
    if dead:
        raise ValueError(f"hand action range has zero width for {dead}")
    return lo, hi


def hand_targets(
    action: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, previous: torch.Tensor, alpha: float = HAND_EMA_ALPHA
) -> torch.Tensor:
    """One step of the full-joint hand law: raw = lo + (a+1)/2 (hi-lo), EMA toward raw, clamp to [lo, hi]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    raw = lo + 0.5 * (action.clamp(-1.0, 1.0) + 1.0) * (hi - lo)
    return torch.max(torch.min(alpha * raw + (1.0 - alpha) * previous, hi), lo)


def normalized_targets(targets: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Post-EMA hand targets mapped to [-1, 1], the observation of the policy's own filter state."""
    return 2.0 * (targets - lo) / (hi - lo) - 1.0


def surface_subsample(points: Sequence[Sequence[float]], count: int = SURFACE_POINT_COUNT) -> torch.Tensor:
    """(count, 3) evenly spaced rows of the shoe's surface point list — deterministic, no RNG."""
    table = torch.as_tensor(points, dtype=torch.float32)
    if table.dim() != 2 or table.shape[1] != 3 or table.shape[0] < count:
        raise ValueError(f"need at least {count} points of shape (P, 3), got {tuple(table.shape)}")
    index = torch.linspace(0, table.shape[0] - 1, count).round().long()
    return table[index]


def nearest_distance(query: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    """(N, L) distance from each query point (N, L, 3) to its nearest surface point (N, P, 3)."""
    return torch.cdist(query, surface).min(dim=-1).values


def grasp_quality(
    link_pos: torch.Tensor,
    finger_sizes: tuple[int, ...],
    surface: torch.Tensor,
    palm_pos: torch.Tensor,
    palm_normal: torch.Tensor,
    shoe_center: torch.Tensor,
    tau: float = KERNEL_TAU_M,
    tau_q: float = SOFTMIN_TAU,
    palm_cos_min: float = PALM_COS_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(q (N,), w_f (N, F), palm_cos (N,)).

    k_l = exp(-d_l / tau) with d_l the nearest-surface distance of finger link l; w_f = mean over the finger's links;
    q = soft-min over fingers (min <= q <= mean), zero unless the palm normal points at the shoe centre.
    """
    sizes = tuple(int(s) for s in finger_sizes)
    if not sizes or min(sizes) < 1 or sum(sizes) != link_pos.shape[1]:
        raise ValueError(f"finger_sizes {sizes} must be positive and sum to {link_pos.shape[1]} links")
    if min(tau, tau_q) <= 0.0:
        raise ValueError("tau and tau_q must be positive")
    kernel = torch.exp(-nearest_distance(link_pos, surface) / tau)
    w_f = torch.stack([part.mean(dim=1) for part in torch.split(kernel, list(sizes), dim=1)], dim=1)
    q = -tau_q * (torch.logsumexp(-w_f / tau_q, dim=1) - math.log(len(sizes)))
    q = torch.minimum(torch.maximum(q, w_f.min(dim=1).values), w_f.mean(dim=1))
    toward = shoe_center - palm_pos
    palm_cos = (palm_normal * toward).sum(dim=-1) / (palm_normal.norm(dim=-1) * toward.norm(dim=-1)).clamp(min=1e-9)
    return q * (palm_cos > palm_cos_min).float(), w_f, palm_cos


def g_factor(q: torch.Tensor, cfg: Stage1RewardCfg) -> torch.Tensor:
    """Grasp factor on the one-shot bonuses: g_min for a flat grasp rising linearly to 1 between q_lo and q_hi."""
    ramp = ((q - cfg.q_lo) / (cfg.q_hi - cfg.q_lo)).clamp(0.0, 1.0)
    return cfg.g_min + (1.0 - cfg.g_min) * ramp


def free_lift_height(
    surface: torch.Tensor,
    start_bottom_z: torch.Tensor,
    rack_x: tuple[float, float],
    rack_y: tuple[float, float],
    margin: float = RACK_MARGIN_M,
) -> torch.Tensor:
    """(N,) rise of the shoe's lowest surface point since the episode start, or 0 while any surface point lies over the
    rack footprint widened by ``margin`` — height gained by pushing the shoe onto or against the rack is not a lift."""
    x, y = surface[..., 0], surface[..., 1]
    over_rack = ((x >= rack_x[0] - margin) & (x <= rack_x[1] + margin) & (y >= rack_y[0] - margin) & (y <= rack_y[1] + margin)).any(dim=-1)
    rise = surface[..., 2].min(dim=-1).values - start_bottom_z
    return torch.where(over_rack, torch.zeros_like(rise), rise)


@dataclass(frozen=True)
class Stage1State:
    closest_palm: torch.Tensor  # (N,) best palm-to-surface gap so far, -1 before the first step
    best_lift: torch.Tensor  # (N,) best clamp(dz_free, 0, lift_height) so far
    hold_count: torch.Tensor  # (N,) consecutive held steps
    latched: torch.Tensor  # (N,) bool, the lift bonus has been paid
    succeeded: torch.Tensor  # (N,) bool, the success bonus has been paid

    @classmethod
    def start(cls, num_envs: int, device: str | torch.device = "cpu") -> "Stage1State":
        return cls(
            closest_palm=torch.full((num_envs,), -1.0, device=device),
            best_lift=torch.zeros(num_envs, device=device),
            hold_count=torch.zeros(num_envs, dtype=torch.long, device=device),
            latched=torch.zeros(num_envs, dtype=torch.bool, device=device),
            succeeded=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )

    def reset_rows(self, env_ids: torch.Tensor) -> "Stage1State":
        """A new state with the rows of ``env_ids`` back at their start values."""
        fresh = Stage1State.start(self.closest_palm.shape[0], self.closest_palm.device)

        def mix(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            out = old.clone()
            out[env_ids] = new[env_ids]
            return out

        return Stage1State(*(mix(getattr(self, f), getattr(fresh, f)) for f in ("closest_palm", "best_lift", "hold_count", "latched", "succeeded")))


@dataclass(frozen=True)
class Stage1Step:
    reward: torch.Tensor
    terms: dict[str, torch.Tensor]
    state: Stage1State
    held: torch.Tensor
    just_latched: torch.Tensor
    success: torch.Tensor


def stage1_step(
    state: Stage1State,
    cfg: Stage1RewardCfg,
    *,
    palm_gap: torch.Tensor,
    dz_free: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    rel_speed: torch.Tensor,
    shoe_speed: torch.Tensor,
    q: torch.Tensor,
    hand_floor_depth: torch.Tensor,
    arm_speed_sum: torch.Tensor,
    hand_speed_sum: torch.Tensor,
) -> Stage1Step:
    """One policy step of the stage-1 reward (design §6). All inputs are (N,) tensors."""
    n = state.closest_palm.shape[0]
    for name, value in dict(palm_gap=palm_gap, dz_free=dz_free, palm_shoe_dist=palm_shoe_dist, rel_speed=rel_speed,
                            shoe_speed=shoe_speed, q=q, hand_floor_depth=hand_floor_depth,
                            arm_speed_sum=arm_speed_sum, hand_speed_sum=hand_speed_sum).items():
        if value.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(value.shape)}")

    held = (dz_free >= cfg.lift_height_m) & (palm_shoe_dist <= cfg.hold_radius_m) & (rel_speed < cfg.hold_rel_speed)
    hold_count = torch.where(held, state.hold_count + 1, torch.zeros_like(state.hold_count))
    before_latch = (~state.latched).float()
    just_latched = ~state.latched & (hold_count >= cfg.latch_steps)
    latched = state.latched | just_latched
    success = ~state.succeeded & latched & (hold_count >= cfg.success_steps) & (shoe_speed < cfg.success_speed)

    first = state.closest_palm < 0.0
    palm_delta = torch.where(first, torch.zeros_like(palm_gap), (state.closest_palm - palm_gap).clamp(min=0.0))
    closest_palm = torch.where(first, palm_gap, torch.minimum(state.closest_palm, palm_gap))
    lift_level = (dz_free - cfg.lift_deadband_m).clamp(0.0, cfg.lift_height_m - cfg.lift_deadband_m)
    lift_delta = (lift_level - state.best_lift).clamp(min=0.0)
    best_lift = torch.maximum(state.best_lift, lift_level)
    g = g_factor(q, cfg)

    terms = {
        "palm_progress": cfg.palm_scale * palm_delta * before_latch,
        "lift_progress": cfg.lift_progress_scale * lift_delta * before_latch,
        "lift_bonus": cfg.lift_bonus * g * just_latched.float(),
        "success_bonus": cfg.success_bonus * g * success.float(),
        "hand_floor": -(cfg.hand_floor_scale * hand_floor_depth.clamp(min=0.0)).clamp(max=cfg.hand_floor_cap),
        "arm_vel": -cfg.arm_vel_scale * arm_speed_sum,
        "hand_vel": -cfg.hand_vel_scale * hand_speed_sum,
    }
    if tuple(terms) != REWARD_TERMS:
        raise RuntimeError(f"term order drifted: {tuple(terms)}")
    reward = torch.nan_to_num(torch.stack(list(terms.values())).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    new_state = replace(state, closest_palm=closest_palm, best_lift=best_lift, hold_count=hold_count,
                        latched=latched, succeeded=state.succeeded | success)
    return Stage1Step(reward=reward, terms=terms, state=new_state, held=held, just_latched=just_latched, success=success)
