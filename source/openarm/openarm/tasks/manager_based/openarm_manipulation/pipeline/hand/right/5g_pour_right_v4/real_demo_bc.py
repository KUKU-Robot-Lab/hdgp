# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0.

"""External real-demo behavior cloning support for 5g_pour_right_v4.

The teleoperation HDF5 files are deploy-compatible on the actor side
(`obs/actor_obs`, 91D) but carry an 18D action from the bimanual recording
pipeline.  This module keeps that dataset separate from the in-sim success
trajectory buffer and exposes `(obs, action, mask)` batches matching the
existing LSTM BC loss contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import torch
from torch import Tensor


REAL_DEMO_OBS_DIM = 91
REAL_DEMO_RAW_ACTION_DIM = 18
REAL_DEMO_ACTION_DIM = 11
DEFAULT_REAL_DEMO_PATHS = tuple(
    Path(f"/home/user/rl_ws/datasets/pour_v1_a{i}.hdf5") for i in range(11, 21)
)


@dataclass(frozen=True)
class RealDemoEpisode:
    """One real teleoperation episode after validation and action conversion."""

    path: Path
    obs: Tensor
    raw_actions: Tensor
    actions: Tensor
    timestamps_ns: Tensor
    right_eef_pose: Tensor
    right_target_eef_pose: Tensor
    source_cup_pose: Tensor
    target_cup_pose: Tensor
    pour_mask: Tensor


def _unscale(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return 2.0 * (value - lower) / (upper - lower).clamp(min=1.0e-8) - 1.0


def _scale(action: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return 0.5 * (action + 1.0) * (upper - lower) + lower


def _quat_xyzw_from_matrix(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions in xyzw order."""
    m = matrix
    qw = torch.sqrt(torch.clamp(1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], min=0.0)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=0.0)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=0.0)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=0.0)) / 2.0
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return torch.nn.functional.normalize(quat, dim=-1)


def pose_from_matrix(matrix: Tensor) -> Tensor:
    """Return `pos + xyzw quat` from homogeneous transform matrices."""
    return torch.cat([matrix[..., :3, 3], _quat_xyzw_from_matrix(matrix[..., :3, :3])], dim=-1)


def _quat_conj_xyzw(quat: Tensor) -> Tensor:
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _quat_mul_xyzw(lhs: Tensor, rhs: Tensor) -> Tensor:
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )


def _axis_angle_from_quat_xyzw(quat: Tensor) -> Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    quat = torch.where(quat[..., 3:4] < 0.0, -quat, quat)
    xyz = quat[..., :3]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 3:4].clamp(min=1.0e-8))
    axis = torch.where(sin_half > 1.0e-8, xyz / sin_half.clamp(min=1.0e-8), torch.zeros_like(xyz))
    return axis * angle


def _quat_from_axis_angle_xyzw(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = torch.where(angle > 1.0e-8, rotvec / angle.clamp(min=1.0e-8), torch.zeros_like(rotvec))
    half = 0.5 * angle
    quat = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)
    return torch.nn.functional.normalize(quat, dim=-1)


def target_pose_to_pose_action(
    target_pose: Tensor,
    base_pose: Tensor,
    delta_mins: Tensor,
    delta_maxs: Tensor,
) -> Tensor:
    """Convert a target palm pose into the v4 normalized 6D delta action.

    This helper mirrors v4's delta convention before cup-local tilt remapping:
    position is relative to the per-episode base pose and rotation is a
    world-frame axis-angle delta left-multiplied onto the base quaternion.
    """
    target_pose = target_pose.to(dtype=torch.float32)
    base_pose = base_pose.to(device=target_pose.device, dtype=target_pose.dtype)
    delta_mins = delta_mins.to(device=target_pose.device, dtype=target_pose.dtype)
    delta_maxs = delta_maxs.to(device=target_pose.device, dtype=target_pose.dtype)

    pos_delta = target_pose[..., :3] - base_pose[..., :3]
    delta_quat = _quat_mul_xyzw(target_pose[..., 3:7], _quat_conj_xyzw(base_pose[..., 3:7]))
    rot_delta = _axis_angle_from_quat_xyzw(delta_quat)
    delta = torch.cat([pos_delta, rot_delta], dim=-1)
    return _unscale(delta, delta_mins, delta_maxs).clamp(-1.0, 1.0)


def pose_action_to_target(
    action: Tensor,
    base_pose: Tensor,
    delta_mins: Tensor,
    delta_maxs: Tensor,
) -> Tensor:
    """Invert :func:`target_pose_to_pose_action` for tests and diagnostics."""
    action = action.to(dtype=torch.float32)
    base_pose = base_pose.to(device=action.device, dtype=action.dtype)
    delta_mins = delta_mins.to(device=action.device, dtype=action.dtype)
    delta_maxs = delta_maxs.to(device=action.device, dtype=action.dtype)
    delta = _scale(action, delta_mins, delta_maxs)
    pos = base_pose[..., :3] + delta[..., :3]
    quat = _quat_mul_xyzw(_quat_from_axis_angle_xyzw(delta[..., 3:6]), base_pose[..., 3:7])
    return torch.cat([pos, torch.nn.functional.normalize(quat, dim=-1)], dim=-1)


def _default_delta_bounds(device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    rad = 2.09439510239
    mins = torch.tensor([-0.5, -0.5, -0.5, -rad, -rad, -rad], device=device)
    maxs = torch.tensor([0.5, 0.5, 0.5, rad, rad, rad], device=device)
    return mins, maxs


def _pour_phase_mask(
    source_cup_pose: Tensor,
    right_eef_pose: Tensor,
    timestamps_ns: Tensor,
    threshold_deg: float = 80.0,
    fallback_final_seconds: float = 4.0,
) -> Tensor:
    # Cup local +Z axis in world coordinates. Pour phase starts when it is
    # tilted at least threshold_deg away from world +Z.
    cup_up = source_cup_pose[:, :3, 2]
    cos_threshold = torch.cos(torch.tensor(threshold_deg * torch.pi / 180.0, device=cup_up.device))
    cup_mask = cup_up[:, 2] < cos_threshold
    if torch.any(cup_mask):
        return cup_mask

    # Some real-demo files keep object_pose/source_cup static while the right
    # EEF pose carries the actual pouring rotation.  In those files, local +X
    # points downward during the pour segment.
    eef_pour_axis_z = right_eef_pose[:, :3, 0][:, 2]
    eef_mask = eef_pour_axis_z < -0.7
    if torch.any(eef_mask):
        return eef_mask

    # Last-resort fallback for short successful teleop recordings where the
    # useful segment is the final pouring motion.
    elapsed_s = (timestamps_ns - timestamps_ns[0]).to(torch.float32) / 1.0e9
    start_s = torch.clamp(elapsed_s[-1] - fallback_final_seconds, min=0.0)
    return elapsed_s >= start_s


def _validate_episode(path: Path, obs: Tensor, raw_actions: Tensor, timestamps_ns: Tensor) -> None:
    if obs.ndim != 2 or obs.shape[1] != REAL_DEMO_OBS_DIM:
        raise ValueError(f"{path}: expected actor_obs (*, 91), got {tuple(obs.shape)}")
    if raw_actions.ndim != 2 or raw_actions.shape[1] != REAL_DEMO_RAW_ACTION_DIM:
        raise ValueError(f"{path}: expected actions (*, 18), got {tuple(raw_actions.shape)}")
    if obs.shape[0] != raw_actions.shape[0] or obs.shape[0] != timestamps_ns.shape[0]:
        raise ValueError(f"{path}: obs/action/timestamp length mismatch")
    if obs.shape[0] < 2:
        raise ValueError(f"{path}: episode too short")
    for name, value in (("obs", obs), ("actions", raw_actions)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")


def load_real_demo_episodes(
    paths: Sequence[str | Path] = DEFAULT_REAL_DEMO_PATHS,
    *,
    demo_stride: int = 2,
    device: str | torch.device = "cpu",
) -> list[RealDemoEpisode]:
    """Load a11-a20 style real demos and convert them to 91D/11D BC episodes."""
    if demo_stride < 1:
        raise ValueError("demo_stride must be >= 1")

    episodes: list[RealDemoEpisode] = []
    for item in paths:
        path = Path(item)
        with h5py.File(path, "r") as h5:
            demo = h5["data"]["demo_0"]
            obs = torch.as_tensor(demo["obs"]["actor_obs"][::demo_stride], dtype=torch.float32, device=device)
            raw_actions = torch.as_tensor(demo["actions"][::demo_stride], dtype=torch.float32, device=device)
            timestamps = torch.as_tensor(demo["timestamps_ns"][::demo_stride], dtype=torch.long, device=device)
            right_eef = torch.as_tensor(
                demo["obs"]["datagen_info"]["eef_pose"]["right"][::demo_stride],
                dtype=torch.float32,
                device=device,
            )
            right_target = torch.as_tensor(
                demo["obs"]["datagen_info"]["target_eef_pose"]["right"][::demo_stride],
                dtype=torch.float32,
                device=device,
            )
            source_cup = torch.as_tensor(
                demo["obs"]["datagen_info"]["object_pose"]["source_cup"][::demo_stride],
                dtype=torch.float32,
                device=device,
            )
            target_cup = torch.as_tensor(
                demo["obs"]["datagen_info"]["object_pose"]["target_cup"][::demo_stride],
                dtype=torch.float32,
                device=device,
            )

        _validate_episode(path, obs, raw_actions, timestamps)
        right_pose = pose_from_matrix(right_eef)
        target_pose = pose_from_matrix(right_target)
        delta_mins, delta_maxs = _default_delta_bounds(device)
        # v4 palm actions are episode-base-relative deltas, not step-relative
        # teleop deltas.  The first right EEF pose is the closest HDF5 proxy for
        # the warmstarted pregrasp palm pose.
        base_pose = right_pose[:1].expand_as(target_pose)
        palm_action = target_pose_to_pose_action(target_pose, base_pose, delta_mins, delta_maxs)
        actions = torch.cat([palm_action, raw_actions[:, 6:11].clamp(-1.0, 1.0)], dim=-1)
        episodes.append(
            RealDemoEpisode(
                path=path,
                obs=obs,
                raw_actions=raw_actions,
                actions=actions,
                timestamps_ns=timestamps,
                right_eef_pose=right_eef,
                right_target_eef_pose=right_target,
                source_cup_pose=source_cup,
                target_cup_pose=target_cup,
                pour_mask=_pour_phase_mask(source_cup, right_eef, timestamps),
            )
        )
    return episodes


class RealDemoBCBuffer:
    """Variable-length LSTM sequence sampler for external real demos."""

    def __init__(
        self,
        episodes: Sequence[RealDemoEpisode],
        *,
        device: str | torch.device,
        pour_sample_ratio: float = 0.6,
    ) -> None:
        if not episodes:
            raise ValueError("RealDemoBCBuffer requires at least one episode")
        self.device = torch.device(device)
        self.episodes = [
            RealDemoEpisode(
                path=e.path,
                obs=e.obs.to(self.device),
                raw_actions=e.raw_actions.to(self.device),
                actions=e.actions.to(self.device),
                timestamps_ns=e.timestamps_ns.to(self.device),
                right_eef_pose=e.right_eef_pose.to(self.device),
                right_target_eef_pose=e.right_target_eef_pose.to(self.device),
                source_cup_pose=e.source_cup_pose.to(self.device),
                target_cup_pose=e.target_cup_pose.to(self.device),
                pour_mask=e.pour_mask.to(self.device),
            )
            for e in episodes
        ]
        self.pour_sample_ratio = float(max(0.0, min(1.0, pour_sample_ratio)))
        self.last_pour_phase_ratio = 0.0

    def __len__(self) -> int:
        return len(self.episodes)

    def is_warm(self, min_count: int = 1) -> bool:
        return len(self.episodes) >= min_count

    def _sample_start(self, episode: RealDemoEpisode, seq_len: int, prefer_pour: bool) -> tuple[int, bool]:
        length = int(episode.obs.shape[0])
        max_start = max(length - 1, 0)
        if prefer_pour and torch.any(episode.pour_mask):
            candidates = torch.nonzero(episode.pour_mask, as_tuple=False).flatten()
            pick = int(candidates[torch.randint(candidates.numel(), (1,), device=self.device)].item())
            return max(0, min(pick - seq_len // 2, max_start)), True
        return int(torch.randint(max_start + 1, (1,), device=self.device).item()), False

    def sample(self, batch_size: int, seq_len: int) -> dict | None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")

        obs_batch = torch.zeros(batch_size, seq_len, REAL_DEMO_OBS_DIM, device=self.device)
        act_batch = torch.zeros(batch_size, seq_len, REAL_DEMO_ACTION_DIM, device=self.device)
        mask_batch = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        pour_samples = 0

        for i in range(batch_size):
            episode = self.episodes[int(torch.randint(len(self.episodes), (1,), device=self.device).item())]
            prefer_pour = bool(torch.rand((), device=self.device).item() < self.pour_sample_ratio)
            start, used_pour = self._sample_start(episode, seq_len, prefer_pour)
            stop = min(start + seq_len, int(episode.obs.shape[0]))
            count = stop - start
            if count <= 0:
                continue
            obs_batch[i, :count] = episode.obs[start:stop]
            act_batch[i, :count] = episode.actions[start:stop]
            mask_batch[i, :count] = True
            pour_samples += int(used_pour)

        self.last_pour_phase_ratio = pour_samples / float(batch_size)
        return {"obs": obs_batch, "actions": act_batch, "mask": mask_batch}
