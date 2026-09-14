"""Grasp bank for the IKER shoe task (design spec §8). Pure torch: importable without Isaac Sim.

A bank entry is the state right after the synergy hand closed on the shoe resting at the configuration's
start pose. It is kept only if lifting from that state and, separately, from the state restored into a
fresh episode both hold the shoe. Resets restore an entry: joint positions, the **commanded** joint
targets (commanding the measured hand pose would remove the squeeze), and the shoe pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from openarm.agnostic.modules.iker.loop_state import LEARNED_BANK_SOURCE

BANK_SCHEMA = 1
BLOCKED_ERR_RAD = 0.2  # a joint this far from its target (and not at a limit) is pressed against the shoe
LIMIT_MARGIN_RAD = 0.02
OPPOSITE_LIMIT_MARGIN_RAD = 0.05
LIMIT_TOLERANCE_RAD = 0.05
CLOSE_RATE_PER_STEP = 1.0 / 150.0  # full open -> grip in 1.25 s at 120 Hz
MIN_RISE_M = 0.05  # spec §8: the palm lifts 0.10 m; the shoe must follow at least half of it
MAX_HOLD_SLIP_M = 0.01  # spec §8: shoe-to-palm drift over the hold after lifting
FINGER_TRIGGER_BACKBEND_RAD = 0.10  # a joint bent back this far against its closing direction triggers its finger
FINGER_SQUEEZE_RAD = 0.15  # a triggered finger's frozen target sits this far past its trigger position
MAX_BACKBEND_RAD = 0.30  # bank acceptance: no finger bent back further than this after closing
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Pre-grasp distribution measured with probes on configuration 0 (scratchpad notes 2026-09-13).
TILT_DEG_RANGE = (-40.0, -15.0)
HEIGHT_ABOVE_TOP_RANGE = (0.02, 0.035)
NEAR_SIDE_CLEARANCE_RANGE = (0.02, 0.035)
ALONG_LENGTH_RANGE = (-0.04, 0.04)
YAW_DEG_RANGE = (-10.0, 10.0)
THUMB3_RANGE = (0.3, 0.5)


@dataclass(frozen=True)
class PreGrasp:
    tilt_deg: torch.Tensor
    height_above_top: torch.Tensor
    near_side_clearance: torch.Tensor
    along_length: torch.Tensor
    yaw_deg: torch.Tensor
    thumb3: torch.Tensor


def sample_pregrasp(count: int, generator: torch.Generator, device: str | torch.device = "cpu") -> PreGrasp:
    def uniform(bounds):
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(count, generator=generator).to(device)

    return PreGrasp(
        tilt_deg=uniform(TILT_DEG_RANGE),
        height_above_top=uniform(HEIGHT_ABOVE_TOP_RANGE),
        near_side_clearance=uniform(NEAR_SIDE_CLEARANCE_RANGE),
        along_length=uniform(ALONG_LENGTH_RANGE),
        yaw_deg=uniform(YAW_DEG_RANGE),
        thumb3=uniform(THUMB3_RANGE),
    )


SIDE_SIGNS = {"r": 1.0, "l": -1.0}  # the left arm is the right arm reflected through the XZ plane (y -> -y)


def _check_side_sign(side_sign: float) -> None:
    if side_sign not in (1.0, -1.0):
        raise ValueError(f"side_sign must be +1 (right arm) or -1 (left arm), got {side_sign}")


def palm_rotations(tilt_deg: torch.Tensor, yaw_deg: torch.Tensor, side_sign: float = 1.0) -> torch.Tensor:
    """(N, 3, 3) palm orientations: palmar side down, fingers across the shoe toward +y, tilted about world x
    (negative tilt turns the palmar side toward -y) and yawed about world z.

    ``side_sign`` -1 gives the left arm's orientation, the right one reflected through the XZ plane (S R S with
    S = diag(1, -1, 1)): palmar side and fingers are mirrored in y and the matrix stays a proper rotation."""
    _check_side_sign(side_sign)
    down = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=tilt_deg.dtype, device=tilt_deg.device)
    t, y = torch.deg2rad(tilt_deg), torch.deg2rad(yaw_deg)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    rx = torch.stack([ones, zeros, zeros, zeros, t.cos(), -t.sin(), zeros, t.sin(), t.cos()], dim=-1).view(-1, 3, 3)
    rz = torch.stack([y.cos(), -y.sin(), zeros, y.sin(), y.cos(), zeros, zeros, zeros, ones], dim=-1).view(-1, 3, 3)
    rotation = rz @ rx @ down
    if side_sign > 0:
        return rotation
    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=tilt_deg.dtype, device=tilt_deg.device))
    return mirror @ rotation @ mirror


def palm_goal_positions(
    pregrasp: PreGrasp,
    shoe_xy: Sequence[float],
    shoe_yaw_deg: float,
    shoe_half_width: float,
    shoe_top_z: float,
    side_sign: float = 1.0,
) -> torch.Tensor:
    """(N, 3) palm origin goals: above the shoe top, offset toward the shoe's side facing the arm (-y for the right
    arm, +y for the left arm)."""
    _check_side_sign(side_sign)
    yaw = math.radians(shoe_yaw_deg)
    along = pregrasp.along_length
    lateral = -side_sign * (shoe_half_width + pregrasp.near_side_clearance)
    x = shoe_xy[0] + along * math.cos(yaw) - lateral * math.sin(yaw)
    y = shoe_xy[1] + along * math.sin(yaw) + lateral * math.cos(yaw)
    return torch.stack([x, y, shoe_top_z + pregrasp.height_above_top], dim=-1)


def hand_side(joint_names: Sequence[str]) -> str:
    """'r' or 'l', the common prefix of hand joint names shaped ``<side>_hj_<finger>_<k>``; mixed sides are an error."""
    sides = {name.split("_")[0] for name in joint_names}
    if len(sides) != 1 or next(iter(sides)) not in SIDE_SIGNS:
        raise ValueError(f"hand joint names must share one side prefix out of {sorted(SIDE_SIGNS)}, got {sorted(sides)}")
    return next(iter(sides))


def side_sign(joint_names: Sequence[str]) -> float:
    return SIDE_SIGNS[hand_side(joint_names)]


def hand_joint_name(joint_names: Sequence[str], finger: str, k: int) -> str:
    name = f"{hand_side(joint_names)}_hj_{finger}_{k}"
    if name not in joint_names:
        raise ValueError(f"{name!r} is not one of the hand joints {list(joint_names)}")
    return name


def finger_index(joint_names: Sequence[str]) -> torch.Tensor:
    """(J,) long ids into ``FINGERS``, parsed from hand joint names shaped ``<side>_hj_<finger>_<k>``."""
    ids = []
    for name in joint_names:
        parts = name.split("_")
        finger = parts[-2] if len(parts) >= 2 else ""
        if finger not in FINGERS:
            raise ValueError(f"joint name {name!r} does not parse as <side>_hj_<finger>_<k>")
        ids.append(FINGERS.index(finger))
    hand_side(joint_names)
    return torch.tensor(ids, dtype=torch.long)


def worst_backbend(joint_pos: torch.Tensor, start_pose: torch.Tensor, grip_pose: torch.Tensor) -> torch.Tensor:
    """(N,): the max over joints of ``back_bend = -(joint_pos - start_pose) . direction``, positive meaning
    bent against the closing direction. ``joint_pos``/``start_pose`` are (N, J); ``grip_pose`` is (J,)."""
    direction = torch.sign(grip_pose.unsqueeze(0) - start_pose)
    back_bend = -(joint_pos - start_pose) * direction
    return back_bend.max(dim=-1).values


def synergy_step(
    close: torch.Tensor,
    target: torch.Tensor,
    joint_pos: torch.Tensor,
    start_pose: torch.Tensor,
    grip_pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One closing step of the open -> grip synergy with per-joint freeze.

    ``close``/``target``/``joint_pos``/``start_pose`` are (N, J); ``grip_pose``/``lower``/``upper`` are (J,).
    A joint stops closing while it is blocked: farther than ``BLOCKED_ERR_RAD`` from its target and not at
    the limit in its own closing direction (the limit the joint approaches while closing from start to
    grip). A joint pinned at the opposite limit counts as blocked and freezes, rather than being read as
    "at a limit" and let through. Joints whose start and grip poses coincide never move. Returns new
    (close, target).
    """
    direction = torch.sign(grip_pose.unsqueeze(0) - start_pose)
    movable = (grip_pose.unsqueeze(0) - start_pose).abs() > 1e-4
    at_own_limit = torch.where(
        direction > 0,
        joint_pos >= upper - LIMIT_MARGIN_RAD,
        torch.where(direction < 0, joint_pos <= lower + LIMIT_MARGIN_RAD, torch.zeros_like(joint_pos, dtype=torch.bool)),
    )
    blocked = ((target - joint_pos).abs() > BLOCKED_ERR_RAD) & ~at_own_limit
    step = torch.where(movable & ~blocked, torch.full_like(close, CLOSE_RATE_PER_STEP), torch.zeros_like(close))
    new_close = (close + step).clamp(0.0, 1.0)
    new_target = torch.max(torch.min(torch.lerp(start_pose, grip_pose.unsqueeze(0), new_close), upper), lower)
    return new_close, new_target


def hand_state_valid(
    joint_pos: torch.Tensor,
    start_pose: torch.Tensor,
    grip_pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: the hand state is a plausible grasp, not a finger pinned at its opposite limit or bent
    beyond its limits. ``synergy_step`` can only freeze a joint against the shoe or its own closing-direction
    limit; a finger hyperextended past the limit opposite its closing direction under a saturated drive holds
    the shoe in sim but is not the specified grasp, so this rejects it. A joint whose start pose already rests
    within ``OPPOSITE_LIMIT_MARGIN_RAD`` of that same limit is excluded from the opposite-limit check: it was
    already there before closing began, not driven there by a saturated drive. ``joint_pos``/``start_pose``
    are (N, J); ``grip_pose``/``lower``/``upper`` are (J,).
    """
    direction = torch.sign(grip_pose.unsqueeze(0) - start_pose)
    movable = (grip_pose.unsqueeze(0) - start_pose).abs() > 1e-4
    near_lower = joint_pos <= lower + OPPOSITE_LIMIT_MARGIN_RAD
    near_upper = joint_pos >= upper - OPPOSITE_LIMIT_MARGIN_RAD
    start_near_lower = start_pose <= lower + OPPOSITE_LIMIT_MARGIN_RAD
    start_near_upper = start_pose >= upper - OPPOSITE_LIMIT_MARGIN_RAD
    at_opposite_limit = torch.where(
        direction > 0,
        near_lower & ~start_near_lower,
        torch.where(direction < 0, near_upper & ~start_near_upper, torch.zeros_like(joint_pos, dtype=torch.bool)),
    )
    beyond_limit = (joint_pos < lower - LIMIT_TOLERANCE_RAD) | (joint_pos > upper + LIMIT_TOLERANCE_RAD)
    invalid = (movable & at_opposite_limit).any(dim=-1) | beyond_limit.any(dim=-1)
    return ~invalid


@dataclass(frozen=True)
class FingerStopState:
    close: torch.Tensor  # (N, J), passed through to/from synergy_step for untriggered joints
    target: torch.Tensor  # (N, J), the commanded target
    triggered: torch.Tensor  # (N, J) bool, sticky once a finger's joint triggers
    trigger_q: torch.Tensor  # (N, J), joint_pos recorded at each finger's first trigger

    @classmethod
    def start(cls, start_pose: torch.Tensor) -> "FingerStopState":
        return cls(
            close=torch.zeros_like(start_pose),
            target=start_pose,
            triggered=torch.zeros_like(start_pose, dtype=torch.bool),
            trigger_q=start_pose,
        )


def finger_stop_step(
    state: FingerStopState,
    joint_pos: torch.Tensor,
    start_pose: torch.Tensor,
    grip_pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    finger_ids: torch.Tensor,
) -> FingerStopState:
    """One closing step of the per-finger stop-on-contact rule.

    ``synergy_step``'s per-joint freeze lets a finger's ``_2`` keep closing after ``_3`` of the same finger
    is already blocked by the shoe, driving ``_2`` into the shoe and folding ``_3`` back past its own closing
    direction. Here, any movable joint of a finger crossing a back-bend or tracking-error threshold triggers
    the whole finger: every movable joint of that finger freezes at its position when the finger first
    triggered, offset by a fixed squeeze, sticky for the rest of closing (and lifting/holding, since callers
    keep calling this after closing ends). Untriggered joints keep advancing via ``synergy_step``.
    ``joint_pos``/``start_pose`` are (N, J); ``grip_pose``/``lower``/``upper``/``finger_ids`` are (J,). Does
    not mutate ``state``; returns a new one.
    """
    direction = torch.sign(grip_pose.unsqueeze(0) - start_pose)
    movable = (grip_pose.unsqueeze(0) - start_pose).abs() > 1e-4
    back_bend = -(joint_pos - start_pose) * direction
    err = (state.target - joint_pos).abs()
    joint_trigger_now = ((back_bend > FINGER_TRIGGER_BACKBEND_RAD) | (err > BLOCKED_ERR_RAD)) & movable

    finger_trigger_now = torch.zeros_like(joint_trigger_now)
    for finger_id in range(len(FINGERS)):
        mask = (finger_ids == finger_id).unsqueeze(0)
        any_trig = (joint_trigger_now & mask).any(dim=-1, keepdim=True) & mask
        finger_trigger_now = finger_trigger_now | any_trig

    newly = finger_trigger_now & ~state.triggered
    triggered = state.triggered | finger_trigger_now
    trigger_q = torch.where(newly, joint_pos, state.trigger_q)

    close, std_target = synergy_step(state.close, state.target, joint_pos, start_pose, grip_pose, lower, upper)
    frozen_target = torch.clamp(trigger_q + FINGER_SQUEEZE_RAD * direction, lower.unsqueeze(0), upper.unsqueeze(0))
    target = torch.where(triggered, frozen_target, std_target)
    return FingerStopState(close=close, target=target, triggered=triggered, trigger_q=trigger_q)


def grasp_acceptable(
    joint_pos: torch.Tensor,
    start_pose: torch.Tensor,
    grip_pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: a plausible grasp (see ``hand_state_valid``) with no finger bent back past ``MAX_BACKBEND_RAD``."""
    return hand_state_valid(joint_pos, start_pose, grip_pose, lower, upper) & (
        worst_backbend(joint_pos, start_pose, grip_pose) <= MAX_BACKBEND_RAD
    )


def lift_held(shoe_z_start: torch.Tensor, shoe_z_end: torch.Tensor, rel_after_lift: torch.Tensor, rel_after_hold: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the shoe rose with the palm and stayed put in the hand during the hold."""
    rise = shoe_z_end - shoe_z_start
    slip = (rel_after_hold - rel_after_lift).norm(dim=-1)
    return (rise >= MIN_RISE_M) & (slip < MAX_HOLD_SLIP_M)


def bank_document(entries: Mapping[str, torch.Tensor], joint_names: Sequence[str], metadata: Mapping) -> dict:
    """JSON-ready bank. ``entries`` holds (K, ...) tensors: joint_pos, joint_target, shoe_pose, palm_pose."""
    required = ("joint_pos", "joint_target", "shoe_pose", "palm_pose")
    missing = [key for key in required if key not in entries]
    if missing:
        raise ValueError(f"bank entries lack {missing}")
    count = entries["joint_pos"].shape[0]
    for key in required:
        if entries[key].shape[0] != count or not torch.isfinite(entries[key]).all():
            raise ValueError(f"bank field {key!r} must be finite with {count} rows")
    return {
        "schema": BANK_SCHEMA,
        "joint_names": list(joint_names),
        "metadata": dict(metadata),
        "entries": {key: entries[key].detach().cpu().tolist() for key in required},
    }


@dataclass(frozen=True)
class GraspBank:
    joint_pos: torch.Tensor  # (K, J) in the articulation's joint order
    joint_target: torch.Tensor  # (K, J)
    shoe_pose: torch.Tensor  # (K, 7) env-local position + wxyz
    palm_pose: torch.Tensor  # (K, 7)
    metadata: Mapping

    @property
    def size(self) -> int:
        return int(self.joint_pos.shape[0])


def load_bank(doc: Mapping, articulation_joint_names: Sequence[str], expected_metadata: Mapping, device) -> GraspBank:
    """Validate a bank document against the running robot and reorder joints by name."""
    if doc.get("schema") != BANK_SCHEMA:
        raise ValueError(f"grasp bank schema {doc.get('schema')} != {BANK_SCHEMA}")
    mismatched = {k: (doc["metadata"].get(k), v) for k, v in expected_metadata.items() if doc["metadata"].get(k) != v}
    if mismatched:
        raise ValueError(f"grasp bank was made with different settings (bank, running): {mismatched}")
    names = list(doc["joint_names"])
    missing = [n for n in articulation_joint_names if n not in names]
    if missing:
        raise ValueError(f"grasp bank lacks joints {missing}")
    order = torch.tensor([names.index(n) for n in articulation_joint_names], dtype=torch.long)
    entries = doc["entries"]

    def tensor(key: str) -> torch.Tensor:
        return torch.tensor(entries[key], dtype=torch.float32)

    bank = GraspBank(
        joint_pos=tensor("joint_pos")[:, order].to(device),
        joint_target=tensor("joint_target")[:, order].to(device),
        shoe_pose=tensor("shoe_pose").to(device),
        palm_pose=tensor("palm_pose").to(device),
        metadata=dict(doc["metadata"]),
    )
    if bank.size == 0:
        raise ValueError("grasp bank is empty")
    return bank


def gains_metadata(joint_names: Sequence[str], stiffness: torch.Tensor, damping: torch.Tensor) -> dict:
    """Per-joint PD gains rounded for exact comparison between bank generation and training."""
    return {
        "stiffness": {n: round(float(k), 4) for n, k in zip(joint_names, stiffness.tolist())},
        "damping": {n: round(float(d), 4) for n, d in zip(joint_names, damping.tolist())},
    }


BOOT_METADATA_KEYS = (
    "config_index", "physics_dt", "friction", "solver_position_iterations", "solver_velocity_iterations", "gains",
    "robot_usd", "shoe_meta_sha256", "scene_config",
)
JOINT_COLUMNS = ("joint_pos", "joint_target")


def sort_joint_columns(entries: Mapping[str, torch.Tensor], joint_names: Sequence[str]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Bank entries with their joint columns ordered by joint name (auto-loop §5); ``load_bank`` reorders by name."""
    order = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
    moved = {
        key: value[:, torch.tensor(order, device=value.device)] if key in JOINT_COLUMNS else value for key, value in entries.items()
    }
    return moved, [joint_names[i] for i in order]


def learned_bank_metadata(
    boot: Mapping,
    *,
    side_sign: float,
    checkpoint: str,
    checkpoint_sha256: str,
    stage1_reward: Mapping,
    seeds: Sequence[int],
    captured: int,
    verified: int,
) -> dict:
    """Metadata of a bank harvested from a stage-1 checkpoint: the environments' boot comparison keys and its origin."""
    missing = [key for key in BOOT_METADATA_KEYS if key not in boot]
    extra = sorted(set(boot) - set(BOOT_METADATA_KEYS))
    if missing or extra:
        raise ValueError(f"boot metadata keys differ from {BOOT_METADATA_KEYS}: missing {missing}, extra {extra}")
    _check_side_sign(float(side_sign))
    if not 0 <= verified <= captured:
        raise ValueError(f"verified {verified} must lie within 0..captured {captured}")
    return {
        **{key: boot[key] for key in BOOT_METADATA_KEYS},
        "source": LEARNED_BANK_SOURCE,
        "side_sign": float(side_sign),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "stage1_reward": dict(stage1_reward),
        "seeds": [int(seed) for seed in seeds],
        "captured": int(captured),
        "verified": int(verified),
    }
