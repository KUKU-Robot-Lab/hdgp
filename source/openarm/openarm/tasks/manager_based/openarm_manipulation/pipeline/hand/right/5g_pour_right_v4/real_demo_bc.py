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
SIM_OBS_DIM = 110  # v4 actor obs 차원


def _remap_hdf5_obs_to_sim_layout(obs: Tensor) -> Tensor:
    """HDF5 real-demo obs(91D)를 sim actor_obs(110D) 레이아웃으로 재배치.

    HDF5 레이아웃 (실측):
      [0:7]   arm_joint_pos
      [7:27]  hand_joint_pos   ← sim에서는 14:34 위치
      [27:34] arm_joint_vel    ← sim에서는 7:14 위치
      [34:54] hand_joint_vel
      [54:68] cup geometry (right_cup_pos_rel_palm 3 + quat 4 + left_cup 7)
      [68:73] tip_force_norm   ← sim에서는 90:95 위치
      [73:79] near-zero field (미사용)
      [79:90] prev_actions(11D) ← sim에서는 95:106 위치 (last_actions)
      [90:91] unknown(1D)

    Sim 레이아웃 (110D):
      [0:7]   arm_joint_pos
      [7:14]  arm_joint_vel
      [14:34] hand_joint_pos
      [34:54] hand_joint_vel
      [54:68] cup geometry
      [68:77] pour geometry (pour_point_to_opening 3 + source_pour_axis 3 + source_up_axis 3) — HDF5 없음 → 0
      [77:85] transport_summary — HDF5 없음 → 0
      [85:90] binary_contact   — HDF5 없음 → 0
      [90:95] tip_force_norm
      [95:106] last_actions
      [106]   bead_in_source_fraction → 1.0 (에피소드 시작 시 가득 참)
      [107:110] bead_in_target/cross/spill → 0
    """
    T = obs.shape[0]
    out = torch.zeros(T, SIM_OBS_DIM, dtype=obs.dtype, device=obs.device)
    out[:, 0:7]   = obs[:, 0:7]    # arm_joint_pos
    out[:, 7:14]  = obs[:, 27:34]  # arm_joint_vel  (HDF5 27:34 → sim 7:14)
    out[:, 14:34] = obs[:, 7:27]   # hand_joint_pos (HDF5 7:27  → sim 14:34)
    out[:, 34:54] = obs[:, 34:54]  # hand_joint_vel
    out[:, 54:68] = obs[:, 54:68]  # cup geometry
    # [68:90]: pour_geometry/transport/contact — HDF5 미포함, 0으로 유지
    out[:, 90:95] = obs[:, 68:73]  # tip_force_norm (HDF5 68:73 → sim 90:95)
    out[:, 95:106] = obs[:, 79:90] # last_actions   (HDF5 79:90 → sim 95:106)
    out[:, 106]   = 1.0            # bead_in_source_fraction (소스컵 가득 참)
    # [107:110]: bead_in_target/cross/spill = 0 (기본값)
    return out
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


# ---------------------------------------------------------------------------
# Cup-local rotation basis helpers  (Phase-1: rotation label re-mapping)
# ---------------------------------------------------------------------------
# Constants mirroring pour_right_preset.py — single source of truth lives there.
_SOURCE_CUP_POUR_POINT_POS_B: list[float] = [0.0, 0.0, 0.100]
_SOURCE_CUP_POUR_AXIS_B: list[float]      = [1.0, 0.0, 0.0]
_SOURCE_CUP_UP_AXIS_B: list[float]        = [0.0, 0.0, 1.0]
_TARGET_CUP_OPENING_POS_B: list[float]    = [0.0, 0.0, 0.100]
_SOURCE_INNER_RADIUS: float               = 0.041


def _safe_normalize_bc(v: Tensor, fallback: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Safe normalize matching env's PourRightEnv._safe_normalize."""
    norm = torch.norm(v, dim=-1, keepdim=True)
    fallback_norm = torch.norm(fallback, dim=-1, keepdim=True).clamp(min=eps)
    return torch.where(norm > eps, v / norm.clamp(min=eps), fallback / fallback_norm)


def _quat_apply_xyzw(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v by xyzw quaternion q.  Handles arbitrary batch dims."""
    qxyz = q[..., :3]
    qw   = q[..., 3:4]
    return v + 2.0 * qw * torch.cross(qxyz, v, dim=-1) + 2.0 * torch.cross(qxyz, torch.cross(qxyz, v, dim=-1), dim=-1)


def _build_cup_local_basis(
    source_cup_mat: Tensor,
    target_cup_mat: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the cup-local rotation basis used by the env's _build_cup_local_tilt_rotvec.

    Args:
        source_cup_mat: (T, 4, 4) homogeneous transforms of the source cup.
        target_cup_mat: (T, 4, 4) homogeneous transforms of the target cup.

    Returns:
        (spin_axis, tilt_toward_axis, tilt_ortho_axis), each (T, 3) unit vectors
        forming an orthonormal basis in world space at each timestep.

    The mapping matches _pre_physics_step exactly:
        rotvec_world = spin*spin_axis + tilt_toward*tilt_toward_axis + tilt_ortho*tilt_ortho_axis
    """
    T   = source_cup_mat.shape[0]
    dev = source_cup_mat.device
    dt  = source_cup_mat.dtype

    source_cup_pos  = source_cup_mat[:, :3, 3]
    source_cup_quat = _quat_xyzw_from_matrix(source_cup_mat[:, :3, :3])
    target_cup_pos  = target_cup_mat[:, :3, 3]
    target_cup_quat = _quat_xyzw_from_matrix(target_cup_mat[:, :3, :3])

    pour_pt_b = torch.tensor(_SOURCE_CUP_POUR_POINT_POS_B, dtype=dt, device=dev).expand(T, -1)
    pour_ax_b = torch.tensor(_SOURCE_CUP_POUR_AXIS_B,      dtype=dt, device=dev).expand(T, -1)
    up_ax_b   = torch.tensor(_SOURCE_CUP_UP_AXIS_B,        dtype=dt, device=dev).expand(T, -1)
    tgt_op_b  = torch.tensor(_TARGET_CUP_OPENING_POS_B,    dtype=dt, device=dev).expand(T, -1)

    cup_up_axis_w   = _quat_apply_xyzw(source_cup_quat, up_ax_b)
    cup_pour_axis_w = _quat_apply_xyzw(source_cup_quat, pour_ax_b)

    # Dynamic source pour point (mirrors _compute_dynamic_source_pour_point_w)
    mouth_center_w = source_cup_pos + _quat_apply_xyzw(source_cup_quat, pour_pt_b)
    gravity_down   = cup_up_axis_w.new_tensor([0.0, 0.0, -1.0]).expand_as(cup_up_axis_w)
    downhill = gravity_down - (gravity_down * cup_up_axis_w).sum(-1, keepdim=True) * cup_up_axis_w
    downhill = _safe_normalize_bc(downhill, cup_pour_axis_w)
    source_pour_point_w = mouth_center_w + _SOURCE_INNER_RADIUS * downhill

    # Target cup opening
    target_opening_w = target_cup_pos + _quat_apply_xyzw(target_cup_quat, tgt_op_b)

    # Tilt basis (mirrors _build_cup_local_tilt_rotvec)
    mouth_delta     = target_opening_w - source_pour_point_w
    pour_axis_plane = cup_pour_axis_w - (cup_pour_axis_w * cup_up_axis_w).sum(-1, keepdim=True) * cup_up_axis_w
    target_dir      = mouth_delta - (mouth_delta * cup_up_axis_w).sum(-1, keepdim=True) * cup_up_axis_w
    target_dir      = _safe_normalize_bc(target_dir, pour_axis_plane)

    tilt_toward_axis = _safe_normalize_bc(
        torch.cross(target_dir, cup_up_axis_w, dim=-1),
        torch.cross(pour_axis_plane, cup_up_axis_w, dim=-1),
    )
    tilt_ortho_axis = _safe_normalize_bc(
        torch.cross(cup_up_axis_w, tilt_toward_axis, dim=-1),
        pour_axis_plane,
    )
    spin_axis = _safe_normalize_bc(
        cup_up_axis_w,
        cup_up_axis_w.new_tensor([0.0, 0.0, 1.0]).expand_as(cup_up_axis_w),
    )
    return spin_axis, tilt_toward_axis, tilt_ortho_axis


def target_pose_to_pose_action_cup_local(
    target_pose: Tensor,
    base_pose: Tensor,
    source_cup_mat: Tensor,
    target_cup_mat: Tensor,
    delta_mins: Tensor,
    delta_maxs: Tensor,
) -> Tensor:
    """Convert target palm pose to v4 6D action using cup-local rotation semantics.

    Rotation is decomposed into [spin, tilt_toward_target, tilt_ortho] by projecting
    the world-frame rotation delta onto the cup-local orthonormal basis produced by
    _build_cup_local_basis.  This matches _pre_physics_step's interpretation of the
    rotation portion of the action vector, eliminating the label/env mismatch that
    previously made BC labels meaningless for RL fine-tuning.
    """
    target_pose = target_pose.to(dtype=torch.float32)
    base_pose   = base_pose.to(device=target_pose.device, dtype=target_pose.dtype)
    delta_mins  = delta_mins.to(device=target_pose.device, dtype=target_pose.dtype)
    delta_maxs  = delta_maxs.to(device=target_pose.device, dtype=target_pose.dtype)

    pos_delta = target_pose[..., :3] - base_pose[..., :3]

    # World-frame rotation delta
    delta_quat = _quat_mul_xyzw(target_pose[..., 3:7], _quat_conj_xyzw(base_pose[..., 3:7]))
    rotvec_w   = _axis_angle_from_quat_xyzw(delta_quat)

    # Project onto cup-local orthonormal basis
    spin_ax, tilt_toward_ax, tilt_ortho_ax = _build_cup_local_basis(source_cup_mat, target_cup_mat)
    spin        = (rotvec_w * spin_ax).sum(-1, keepdim=True)
    tilt_toward = (rotvec_w * tilt_toward_ax).sum(-1, keepdim=True)
    tilt_ortho  = (rotvec_w * tilt_ortho_ax).sum(-1, keepdim=True)

    delta = torch.cat([pos_delta, spin, tilt_toward, tilt_ortho], dim=-1)
    return _unscale(delta, delta_mins, delta_maxs).clamp(-1.0, 1.0)


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
        obs = _remap_hdf5_obs_to_sim_layout(obs)  # HDF5 91D → sim 110D 레이아웃으로 재배치
        right_pose = pose_from_matrix(right_eef)
        target_pose = pose_from_matrix(right_target)
        delta_mins, delta_maxs = _default_delta_bounds(device)
        # v4 palm actions are episode-base-relative deltas, not step-relative
        # teleop deltas.  The first right EEF pose is the closest HDF5 proxy for
        # the warmstarted pregrasp palm pose.
        base_pose = right_pose[:1].expand_as(target_pose)
        # Phase-1: use cup-local rotation basis so the rotation label matches
        # _pre_physics_step's interpretation of delta[:, 3:6].
        palm_action = target_pose_to_pose_action_cup_local(
            target_pose, base_pose, source_cup, target_cup, delta_mins, delta_maxs,
        )
        # freeze_grasp_hand_during_episode=True means the env always overrides
        # finger actions with 1.0 (full grasp).  Supervising BC on raw teleop
        # finger values wastes gradient on a signal the env discards.
        finger_action = torch.ones(target_pose.shape[0], 5, dtype=torch.float32, device=device)
        actions = torch.cat([palm_action, finger_action], dim=-1)
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

        obs_batch = torch.zeros(batch_size, seq_len, SIM_OBS_DIM, device=self.device)
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
