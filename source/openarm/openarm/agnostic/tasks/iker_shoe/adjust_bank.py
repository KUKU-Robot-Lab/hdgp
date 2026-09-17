"""Adjust-start bank: stage-2 states in which the shoe was set down off target (2026-09-17, user decision).

Why. Deterministic probes of the best stage-2 policy (iter_05 epoch 600) end 31-36 % of episodes with the shoe
resting on the rack, released, 3-8 cm off target, and the palm 25-35 cm away. Two generated rewards that paid for
going back to the shoe (iter_06, iter_07) left `reach`/`push` flat for 200 epochs: the policy never visits the
states those terms pay in. So a fraction of episodes start exactly there, harvested from the policy's own rollouts
(physically consistent: real joint state, real resting shoe), and the adjustment gets practised directly.

A state qualifies on the first step the environment's own flags say: not placed, resting, still, grip open at least
`retract_open_min` (the hand has let go), not retracting, and keypoint_dist at most `max_keypoint_dist`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .place_stage import PlaceRewardCfg

KEYS = ("joint_pos", "joint_target", "shoe_pose", "other_pose", "grip_scalar")
_WIDTH = {"shoe_pose": 7, "other_pose": 7, "grip_scalar": 1}


@dataclass(frozen=True)
class AdjustBank:
    joint_pos: torch.Tensor     # (M, J) articulation joint order
    joint_target: torch.Tensor  # (M, J)
    shoe_pose: torch.Tensor     # (M, 7) env-local position + quaternion (w, x, y, z)
    other_pose: torch.Tensor    # (M, 7) env-local
    grip_scalar: torch.Tensor   # (M, 1) the env's filtered grip state in [-1, 1]

    @property
    def size(self) -> int:
        return int(self.joint_pos.shape[0])


def qualifies(placed, resting, still, open_frac, retracting, keypoint_dist, cfg: PlaceRewardCfg, *,
              max_keypoint_dist: float) -> torch.Tensor:
    return (~placed & resting & still & (open_frac >= cfg.retract_open_min) & ~retracting
            & (keypoint_dist <= max_keypoint_dist))


def _check(entries: Mapping[str, torch.Tensor], joint_count: int) -> int:
    missing = [k for k in KEYS if k not in entries]
    if missing:
        raise ValueError(f"adjust bank is missing {missing}")
    rows = int(entries["joint_pos"].shape[0])
    if rows == 0:
        raise ValueError("adjust bank is empty")
    for key in KEYS:
        width = _WIDTH.get(key, joint_count)
        if tuple(entries[key].shape) != (rows, width):
            raise ValueError(f"adjust bank {key} has shape {tuple(entries[key].shape)}, expected {(rows, width)}")
    return rows


def save_bank(path, entries: Mapping[str, torch.Tensor], joint_names: Sequence[str], metadata: Mapping) -> Path:
    _check(entries, len(joint_names))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"joint_names": list(joint_names), "metadata": dict(metadata)}
    doc.update({k: entries[k].detach().cpu().float() for k in KEYS})
    torch.save(doc, path)
    return path


def load_bank(path, joint_names: Sequence[str], device) -> AdjustBank:
    doc = torch.load(Path(path), map_location="cpu")
    if list(doc.get("joint_names", [])) != list(joint_names):
        raise ValueError(f"{path}: joint names differ from the articulation's joint order")
    _check(doc, len(joint_names))
    return AdjustBank(**{k: doc[k].to(device) for k in KEYS})


def start_mask(count: int, frac: float, device, generator: torch.Generator | None = None) -> torch.Tensor:
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"adjust start fraction must be in [0, 1], got {frac}")
    return torch.rand(count, generator=generator).to(device) < frac
