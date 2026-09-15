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

from openarm.agnostic.modules.iker import run_files

HAND_EMA_ALPHA = 1.0 - 0.9**6  # grasp_fj's 0.1 per 60 Hz step, same time constant at 10 Hz (= 0.468559)
SURFACE_POINT_COUNT = 256
KERNEL_TAU_M = 0.02
SOFTMIN_TAU = 0.1
PALM_COS_MIN = 0.0
RACK_MARGIN_M = 0.01  # a shoe hull point this close to the rack's footprint counts as supported by the rack
FROZEN_HALFWIDTH_RAD = 0.01  # a pinned hand joint keeps this much action range (zero width is rejected)


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
    # revision 3-4: lift progress needs the shoe slower than this against the palm — looser than the hold's hold_rel_speed, because
    # an exploring policy almost never lifts with the shoe fully still (phase A r5 lost the lift signal at 0.05 m/s)
    progress_rel_speed: float = 0.20
    latch_steps: int = 3
    success_steps: int = 20
    success_speed: float = 0.05
    # learned-grasp spec §16 (revision 3): a hold counts only near the pick location, below a lift ceiling and with the thumb
    # closing; a negative thumb_curl_min_rad switches the thumb condition off (grasp_smoke's wiring check)
    lift_max_m: float = 0.15
    hold_xy_radius_m: float = 0.10
    thumb_curl_min_rad: float = 0.05
    hand_floor_scale: float = 10.0
    hand_floor_cap: float = 5.0
    # revision 3-5: motion penalties act only while the shoe is lifted (free lift >= lift_deadband_m); arm joint speed sum and the
    # hand command rate sum |a_hand(t) - a_hand(t-1)| (phase A r6: fingers and arm kept moving while lifted, user video 2026-09-15)
    arm_vel_scale: float = 0.1
    hand_vel_scale: float = 0.0
    hand_rate_scale: float = 0.1
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
        if not self.lift_height_m < self.lift_max_m:
            raise ValueError(f"need lift_height_m {self.lift_height_m} < lift_max_m {self.lift_max_m}")
        if not self.hold_xy_radius_m > 0.0:
            raise ValueError(f"hold_xy_radius_m must be positive, got {self.hold_xy_radius_m}")
        if not 0.0 < self.hold_rel_speed <= self.progress_rel_speed:
            raise ValueError(f"need 0 < hold_rel_speed {self.hold_rel_speed} <= progress_rel_speed {self.progress_rel_speed}")
        if min(self.palm_scale, self.lift_progress_scale, self.lift_bonus, self.success_bonus, self.hand_floor_scale,
               self.hand_floor_cap, self.arm_vel_scale, self.hand_vel_scale, self.hand_rate_scale) < 0.0:
            raise ValueError("reward scales and bonuses must be non-negative")


REWARD_TERMS = ("palm_progress", "lift_progress", "lift_bonus", "success_bonus", "hand_floor", "arm_vel", "hand_vel", "hand_rate")


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


def role_joint_index(joint_names: Sequence[str], role: str) -> int:
    """Index of the one hand joint named ``<side>_hj_<role>`` (``thumb_3``) — no side is spelled out."""
    matches = [i for i, name in enumerate(joint_names) if name.endswith(f"_hj_{role}")]
    if len(matches) != 1:
        raise ValueError(f"hand joint role {role!r} matches {len(matches)} joints of {list(joint_names)}")
    return matches[0]


def frozen_hand_override(
    joint_names: Sequence[str], open_pose: Sequence[float], roles: Sequence[str], halfwidth: float = FROZEN_HALFWIDTH_RAD
) -> dict[str, tuple[float, float]]:
    """``hand_action_limits`` overrides pinning hand joints at the profile's open pose (+-halfwidth).

    A role is a joint name without its side prefix (``thumb_2``, the thumb's opposition joint), so no side is spelled
    out here; each role must name exactly one joint.
    """
    names = list(joint_names)
    if len(open_pose) != len(names):
        raise ValueError(f"open pose has {len(open_pose)} values for {len(names)} hand joints")
    override = {}
    for role in roles:
        index = role_joint_index(names, role)
        value = float(open_pose[index])
        override[re.escape(names[index]) + "$"] = (value - halfwidth, value + halfwidth)
    return override


def stage1_hand_limits(
    joint_names: Sequence[str],
    hard_lo: torch.Tensor,
    hard_hi: torch.Tensor,
    profile_override: Mapping[str, tuple],
    open_pose: Sequence[float],
    frozen_roles: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage-1 hand action limits: the hard limits narrowed by the profile override, then narrowed again around the open
    pose for the frozen roles. Two passes, so a frozen joint keeps the profile's narrowing — a merged override dict lets
    a frozen entry replace a profile entry that uses the same pattern."""
    lo, hi = hand_action_limits(joint_names, hard_lo, hard_hi, profile_override)
    return hand_action_limits(joint_names, lo, hi, frozen_hand_override(joint_names, open_pose, frozen_roles))


def backstop_limits(
    joint_names: Sequence[str],
    hard_lo: torch.Tensor,
    hard_hi: torch.Tensor,
    open_pose: Sequence[float],
    grip_pose: Sequence[float],
    roles: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """Joint position limits that keep the hand joint ``roles`` from bending back past the profile's open pose (spec §16):
    the limit opposite each role's closing direction (open -> grip) moves to the open pose, the other stays hard.

    ``joint_names`` are the profile's hand joints, ``hard_lo``/``hard_hi`` their (J,) simulator limits, the poses (J,)
    profile values. Returns ``{joint name: (lo, hi)}`` in ``roles`` order.
    """
    names = list(joint_names)
    if hard_lo.shape != (len(names),) or hard_hi.shape != (len(names),):
        raise ValueError(f"limits must be ({len(names)},), got {tuple(hard_lo.shape)} and {tuple(hard_hi.shape)}")
    limits = {}
    for role in roles:
        index = role_joint_index(names, role)
        lo, hi = float(hard_lo[index]), float(hard_hi[index])
        q_open, q_grip = float(open_pose[index]), float(grip_pose[index])
        if q_grip == q_open:
            raise ValueError(f"{names[index]} has no closing direction: its open and grip poses are both {q_open}")
        if not lo <= q_open <= hi:
            raise ValueError(f"{names[index]} open pose {q_open} lies outside its limits [{lo}, {hi}]")
        limits[names[index]] = (lo, q_open) if q_grip < q_open else (q_open, hi)
    return limits


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


def closing_travel(joint_pos: torch.Tensor, open_pose: torch.Tensor, grip_pose: torch.Tensor) -> torch.Tensor:
    """How far joints moved from the open pose toward the grip pose; negative = bent back past the open pose.

    Shapes broadcast (``joint_pos`` (N,) or (N, J) against (J,) or scalar poses). A joint whose open and grip poses coincide
    has no closing direction and is rejected.
    """
    direction = torch.sign(grip_pose - open_pose)
    if bool((direction == 0).any()):
        raise ValueError("closing_travel needs a closing direction: the open and grip poses coincide")
    return (joint_pos - open_pose) * direction


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
    best_lift: torch.Tensor  # (N,) best clamp(dz_free - deadband, 0, lift_height - deadband) inside the hold zone (thumb closing, shoe not sliding fast) so far
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
    lost: torch.Tensor  # revision 3-5: latched, not succeeded, shoe back below lift_deadband_m — the environment ends the episode


def stage1_step(
    state: Stage1State,
    cfg: Stage1RewardCfg,
    *,
    palm_gap: torch.Tensor,
    dz_free: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    shoe_shift_xy: torch.Tensor,
    thumb_curl: torch.Tensor,
    rel_speed: torch.Tensor,
    shoe_speed: torch.Tensor,
    q: torch.Tensor,
    hand_floor_depth: torch.Tensor,
    arm_speed_sum: torch.Tensor,
    hand_speed_sum: torch.Tensor,
    hand_command_rate: torch.Tensor,
) -> Stage1Step:
    """One policy step of the stage-1 reward (design §6, held narrowed by §16, motion penalties and lost grasps by revision 3-5).
    All inputs are (N,) tensors."""
    n = state.closest_palm.shape[0]
    for name, value in dict(palm_gap=palm_gap, dz_free=dz_free, palm_shoe_dist=palm_shoe_dist, shoe_shift_xy=shoe_shift_xy,
                            thumb_curl=thumb_curl, rel_speed=rel_speed, shoe_speed=shoe_speed, q=q,
                            hand_floor_depth=hand_floor_depth, arm_speed_sum=arm_speed_sum, hand_speed_sum=hand_speed_sum,
                            hand_command_rate=hand_command_rate).items():
        if value.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(value.shape)}")

    held = (
        (dz_free >= cfg.lift_height_m)
        & (dz_free <= cfg.lift_max_m)
        & (shoe_shift_xy <= cfg.hold_xy_radius_m)
        & (thumb_curl >= cfg.thumb_curl_min_rad)
        & (palm_shoe_dist <= cfg.hold_radius_m)
        & (rel_speed < cfg.hold_rel_speed)
    )
    hold_count = torch.where(held, state.hold_count + 1, torch.zeros_like(state.hold_count))
    before_latch = (~state.latched).float()
    just_latched = ~state.latched & (hold_count >= cfg.latch_steps)
    latched = state.latched | just_latched
    success = ~state.succeeded & latched & (hold_count >= cfg.success_steps) & (shoe_speed < cfg.success_speed)

    first = state.closest_palm < 0.0
    palm_delta = torch.where(first, torch.zeros_like(palm_gap), (state.closest_palm - palm_gap).clamp(min=0.0))
    closest_palm = torch.where(first, palm_gap, torch.minimum(state.closest_palm, palm_gap))
    # revision 3-1: lift progress counts only where a hold could — near the start and below the lift ceiling — so carrying the
    # shoe away earns nothing (phase A r2 lifted and carried it 61 cm with progress paid anywhere); revision 3-2: and only with
    # the thumb closing (phase A r3 scooped the shoe with four fingers while they pushed the thumb open through its backstop);
    # revisions 3-3/3-4: and only with the shoe not sliding fast in the hand — progress_rel_speed, looser than the hold's limit
    # (phase A r4 lifted with the shoe moving 0.11 m/s against the palm; r5 with the hold's 0.05 m/s stopped lifting at all)
    in_hold_zone = (
        (shoe_shift_xy <= cfg.hold_xy_radius_m)
        & (dz_free <= cfg.lift_max_m)
        & (thumb_curl >= cfg.thumb_curl_min_rad)
        & (rel_speed < cfg.progress_rel_speed)
    )
    lift_level = torch.where(
        in_hold_zone, (dz_free - cfg.lift_deadband_m).clamp(0.0, cfg.lift_height_m - cfg.lift_deadband_m), torch.zeros_like(dz_free)
    )
    lift_delta = (lift_level - state.best_lift).clamp(min=0.0)
    best_lift = torch.maximum(state.best_lift, lift_level)
    g = g_factor(q, cfg)
    lifted = (dz_free >= cfg.lift_deadband_m).float()  # revision 3-5: motion penalties only while the shoe is off the table

    terms = {
        "palm_progress": cfg.palm_scale * palm_delta * before_latch,
        "lift_progress": cfg.lift_progress_scale * lift_delta * before_latch,
        "lift_bonus": cfg.lift_bonus * g * just_latched.float(),
        "success_bonus": cfg.success_bonus * g * success.float(),
        "hand_floor": -(cfg.hand_floor_scale * hand_floor_depth.clamp(min=0.0)).clamp(max=cfg.hand_floor_cap),
        "arm_vel": -cfg.arm_vel_scale * arm_speed_sum * lifted,
        "hand_vel": -cfg.hand_vel_scale * hand_speed_sum * lifted,
        "hand_rate": -cfg.hand_rate_scale * hand_command_rate * lifted,
    }
    if tuple(terms) != REWARD_TERMS:
        raise RuntimeError(f"term order drifted: {tuple(terms)}")
    reward = torch.nan_to_num(torch.stack(list(terms.values())).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    new_state = replace(state, closest_palm=closest_palm, best_lift=best_lift, hold_count=hold_count,
                        latched=latched, succeeded=state.succeeded | success)
    lost = latched & ~success & (dz_free < cfg.lift_deadband_m)
    return Stage1Step(reward=reward, terms=terms, state=new_state, held=held, just_latched=just_latched, success=success, lost=lost)


def quality_calibration_document(q_lo: float, q_hi: float, **details) -> dict:
    """The grasp quality calibration file (design §5): ``0 <= q_lo < q_hi <= 1`` and the measurement details, under the
    run-file schema the environment reads it with (``run_files.read_json`` rejects a document without it)."""
    reserved = sorted({"schema", "q_lo", "q_hi"} & set(details))
    if reserved:
        raise ValueError(f"calibration details may not set {reserved}")
    lo, hi = float(q_lo), float(q_hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi <= 1.0):
        raise ValueError(f"grasp quality calibration needs 0 <= q_lo < q_hi <= 1, got q_lo {lo}, q_hi {hi}")
    return {"schema": run_files.SCHEMA_VERSION, **details, "q_lo": lo, "q_hi": hi}


def read_quality_calibration(path) -> tuple[float, float]:
    """(q_lo, q_hi) of a file written from ``quality_calibration_document``."""
    doc = run_files.read_json(path)
    checked = quality_calibration_document(doc["q_lo"], doc["q_hi"])
    return checked["q_lo"], checked["q_hi"]


@dataclass(frozen=True)
class SuccessCapture:
    """Each env's state at its first success since the last clear, kept apart from the same step's reset (auto-loop §5)."""

    joint_pos: torch.Tensor  # (N, J) articulation joint positions
    joint_target: torch.Tensor  # (N, J) targets that hold the state (``holding_targets``)
    shoe_pose: torch.Tensor  # (N, 7) env-local position + wxyz
    palm_pose: torch.Tensor  # (N, 7) env-local position + wxyz
    step: torch.Tensor  # (N,) long, the environment's common step counter at the capture, -1 before
    valid: torch.Tensor  # (N,) bool

    @classmethod
    def empty(cls, num_envs: int, num_joints: int, device: str | torch.device = "cpu") -> "SuccessCapture":
        return cls(
            joint_pos=torch.zeros(num_envs, num_joints, device=device),
            joint_target=torch.zeros(num_envs, num_joints, device=device),
            shoe_pose=torch.zeros(num_envs, 7, device=device),
            palm_pose=torch.zeros(num_envs, 7, device=device),
            step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
            valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )


def capture_rows(
    capture: SuccessCapture,
    mask: torch.Tensor,
    *,
    joint_pos: torch.Tensor,
    joint_target: torch.Tensor,
    shoe_pose: torch.Tensor,
    palm_pose: torch.Tensor,
    step: int,
) -> SuccessCapture:
    """A new capture holding the rows of ``mask`` that hold none yet; a row's later successes are ignored until a clear.
    Tensor ops only, so the step path does not synchronise with the host."""
    take = mask & ~capture.valid
    rows = take[:, None]
    return SuccessCapture(
        joint_pos=torch.where(rows, joint_pos, capture.joint_pos),
        joint_target=torch.where(rows, joint_target, capture.joint_target),
        shoe_pose=torch.where(rows, shoe_pose, capture.shoe_pose),
        palm_pose=torch.where(rows, palm_pose, capture.palm_pose),
        step=torch.where(take, torch.full_like(capture.step, step), capture.step),
        valid=capture.valid | take,
    )


def holding_targets(
    joint_targets: torch.Tensor,
    joint_pos: torch.Tensor,
    arm_ids: Sequence[int] | torch.Tensor,
    hand_ids: Sequence[int] | torch.Tensor,
    hand_targets: torch.Tensor,
) -> torch.Tensor:
    """(N, J) joint targets that hold a state: the arm at its measured position, the hand at its EMA target and every
    other joint at its command."""
    out = joint_targets.clone()
    out[:, arm_ids] = joint_pos[:, arm_ids]
    out[:, hand_ids] = hand_targets
    return out
