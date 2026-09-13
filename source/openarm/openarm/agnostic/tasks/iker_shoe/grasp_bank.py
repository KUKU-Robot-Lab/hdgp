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

BANK_SCHEMA = 1
BLOCKED_ERR_RAD = 0.2  # a joint this far from its target (and not at a limit) is pressed against the shoe
LIMIT_MARGIN_RAD = 0.02
CLOSE_RATE_PER_STEP = 1.0 / 150.0  # full open -> grip in 1.25 s at 120 Hz
MIN_RISE_M = 0.05  # spec §8: the palm lifts 0.10 m; the shoe must follow at least half of it
MAX_HOLD_SLIP_M = 0.01  # spec §8: shoe-to-palm drift over the hold after lifting

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


def palm_rotations(tilt_deg: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) palm orientations: palmar side down, fingers across the shoe toward +y, tilted about world x
    (negative tilt turns the palmar side toward -y) and yawed about world z."""
    down = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=tilt_deg.dtype, device=tilt_deg.device)
    t, y = torch.deg2rad(tilt_deg), torch.deg2rad(yaw_deg)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    rx = torch.stack([ones, zeros, zeros, zeros, t.cos(), -t.sin(), zeros, t.sin(), t.cos()], dim=-1).view(-1, 3, 3)
    rz = torch.stack([y.cos(), -y.sin(), zeros, y.sin(), y.cos(), zeros, zeros, zeros, ones], dim=-1).view(-1, 3, 3)
    return rz @ rx @ down


def palm_goal_positions(
    pregrasp: PreGrasp, shoe_xy: Sequence[float], shoe_yaw_deg: float, shoe_half_width: float, shoe_top_z: float
) -> torch.Tensor:
    """(N, 3) palm origin goals: above the shoe top, offset toward the shoe's near (-y) side."""
    yaw = math.radians(shoe_yaw_deg)
    along = pregrasp.along_length
    lateral = -(shoe_half_width + pregrasp.near_side_clearance)
    x = shoe_xy[0] + along * math.cos(yaw) - lateral * math.sin(yaw)
    y = shoe_xy[1] + along * math.sin(yaw) + lateral * math.cos(yaw)
    return torch.stack([x, y, shoe_top_z + pregrasp.height_above_top], dim=-1)


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
    A joint stops closing while it is blocked: farther than ``BLOCKED_ERR_RAD`` from its target and not
    at a joint limit. Joints whose start and grip poses coincide never move. Returns new (close, target).
    """
    movable = (grip_pose.unsqueeze(0) - start_pose).abs() > 1e-4
    free = (joint_pos > lower + LIMIT_MARGIN_RAD) & (joint_pos < upper - LIMIT_MARGIN_RAD)
    blocked = ((target - joint_pos).abs() > BLOCKED_ERR_RAD) & free
    step = torch.where(movable & ~blocked, torch.full_like(close, CLOSE_RATE_PER_STEP), torch.zeros_like(close))
    new_close = (close + step).clamp(0.0, 1.0)
    new_target = torch.max(torch.min(torch.lerp(start_pose, grip_pose.unsqueeze(0), new_close), upper), lower)
    return new_close, new_target


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
