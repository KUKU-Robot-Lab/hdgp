# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HDF5 실제 데모 → v5 (obs 110D, action 11D) 변환 및 BC 배치 샘플러.

HDF5 파일(pour_v1_a11~a20.hdf5)에서 관절/포즈/힘 데이터를 직접 읽어
v5 환경의 actor obs 레이아웃(110D)과 action 컨벤션(11D)으로 변환한다.

Obs 110D 레이아웃 (pour_right_constants.py 기준):
  [0:7]   arm_joint_pos
  [7:14]  arm_joint_vel
  [14:34] hand_joint_pos
  [34:54] hand_joint_vel
  [54:57] right_cup_pos_rel_palm  (source_cup_pos - palm_pos)
  [57:61] right_cup_quat          (xyzw)
  [61:64] left_cup_pos_rel_palm   (target_cup_pos - palm_pos)
  [64:68] left_cup_quat           (xyzw)
  [68:71] pour_point_to_opening
  [71:74] source_pour_axis_w
  [74:77] source_up_axis_w
  [77:85] transport_summary (8D)
  [85:90] fingertip_contact_binary
  [90:95] tip_force_norm
  [95:106] last_actions (11D, shifted by 1)
  [106]   bead_in_source_fraction  (= 1.0 at demo start)
  [107:110] bead fractions         (= 0.0)

Action 11D:
  [0:3]  normalized pos delta from episode-base palm pose
  [3:6]  cup-local rotation: spin, tilt_toward, tilt_ortho (normalized)
  [6:11] per-finger lerp = 1.0 (grasp 유지)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import torch
from torch import Tensor

from .pour_right_constants import (
    NUM_OBSERVATIONS,
    NUM_ACTIONS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
)

# ---------------------------------------------------------------------------
# Cup geometry constants (pour_right_preset.py 와 동일)
# ---------------------------------------------------------------------------
_POUR_POINT_B    = [0.0, 0.0, 0.100]
_POUR_AXIS_B     = [1.0, 0.0, 0.0]
_UP_AXIS_B       = [0.0, 0.0, 1.0]
_TARGET_OPEN_B   = [0.0, 0.0, 0.100]
_SOURCE_INNER_R  = 0.041

# ---------------------------------------------------------------------------
# Action delta 범위 (pour_right_env_cfg.py 기준)
# ---------------------------------------------------------------------------
_DELTA_XYZ = 0.5
_DELTA_RAD  = math.radians(90.0)   # env palm_delta_rot_deg=90° 와 통일

# ---------------------------------------------------------------------------
# Pour phase 검출: cup 기울기 임계값 (source_up_axis z < cos(threshold))
# ---------------------------------------------------------------------------
_POUR_PHASE_TILT_DEG = 70.0

# ---------------------------------------------------------------------------
# HDF5 경로 기본값
# ---------------------------------------------------------------------------
DEFAULT_DEMO_PATHS = tuple(
    Path(f"/home/user/rl_ws/datasets/pour_v1_a{i}.hdf5") for i in range(11, 21)
)


# ---------------------------------------------------------------------------
# 수학 헬퍼 (순수 torch, 외부 의존 없음)
# ---------------------------------------------------------------------------

def _quat_xyzw_from_matrix(R: Tensor) -> Tensor:
    """(..., 3, 3) → (..., 4) xyzw."""
    qw = torch.sqrt(torch.clamp(1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], min=0.0)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], min=0.0)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], min=0.0)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], min=0.0)) / 2.0
    qx = torch.copysign(qx, R[..., 2, 1] - R[..., 1, 2])
    qy = torch.copysign(qy, R[..., 0, 2] - R[..., 2, 0])
    qz = torch.copysign(qz, R[..., 1, 0] - R[..., 0, 1])
    return torch.nn.functional.normalize(torch.stack([qx, qy, qz, qw], dim=-1), dim=-1)


def _quat_apply(q_xyzw: Tensor, v: Tensor) -> Tensor:
    """xyzw quaternion으로 벡터 회전. (..., 4), (..., 3) → (..., 3)."""
    xyz = q_xyzw[..., :3]
    w   = q_xyzw[..., 3:4]
    return v + 2.0 * w * torch.cross(xyz, v, dim=-1) + 2.0 * torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1)


def _safe_normalize(v: Tensor, fallback: Tensor, eps: float = 1e-6) -> Tensor:
    norm = v.norm(dim=-1, keepdim=True)
    fb_norm = fallback.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.where(norm > eps, v / norm.clamp(min=eps), fallback / fb_norm)


def _quat_conj(q: Tensor) -> Tensor:
    """xyzw → conjugate."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul(l: Tensor, r: Tensor) -> Tensor:
    """xyzw quaternion 곱."""
    lx, ly, lz, lw = l.unbind(dim=-1)
    rx, ry, rz, rw = r.unbind(dim=-1)
    return torch.stack([
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
        lw*rw - lx*rx - ly*ry - lz*rz,
    ], dim=-1)


def _axis_angle_from_quat(q: Tensor) -> Tensor:
    """xyzw → axis-angle (rotvec)."""
    q = torch.nn.functional.normalize(q, dim=-1)
    q = torch.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., 3:4].clamp(min=1e-8))
    axis  = torch.where(sin_half > 1e-8, xyz / sin_half.clamp(min=1e-8), torch.zeros_like(xyz))
    return axis * angle


def _unscale(val: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    return 2.0 * (val - lo) / (hi - lo).clamp(min=1e-8) - 1.0


# ---------------------------------------------------------------------------
# Cup geometry helpers (env의 _compute_intermediate_values 미러)
# ---------------------------------------------------------------------------

def _cup_geometry(
    source_mat: Tensor,  # (T, 4, 4)
    target_mat: Tensor,  # (T, 4, 4)
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """cup 포즈에서 pour geometry 계산.

    Returns:
        pour_point_to_opening (T,3), source_pour_axis (T,3), source_up_axis (T,3),
        pour_point_w (T,3), target_opening_w (T,3), transport_summary (T,8)
    """
    T = source_mat.shape[0]

    def _const(lst: list[float]) -> Tensor:
        return torch.tensor(lst, dtype=dtype, device=device).expand(T, -1)

    pour_pt_b  = _const(_POUR_POINT_B)
    pour_ax_b  = _const(_POUR_AXIS_B)
    up_ax_b    = _const(_UP_AXIS_B)
    tgt_open_b = _const(_TARGET_OPEN_B)

    src_pos  = source_mat[:, :3, 3]
    src_quat = _quat_xyzw_from_matrix(source_mat[:, :3, :3])
    tgt_pos  = target_mat[:, :3, 3]
    tgt_quat = _quat_xyzw_from_matrix(target_mat[:, :3, :3])

    source_pour_axis = _quat_apply(src_quat, pour_ax_b)
    source_up_axis   = _quat_apply(src_quat, up_ax_b)

    # Dynamic pour point (env _compute_dynamic_source_pour_point_w 미러)
    mouth_center = src_pos + _quat_apply(src_quat, pour_pt_b)
    gravity_down = source_up_axis.new_tensor([0.0, 0.0, -1.0]).expand_as(source_up_axis)
    downhill = gravity_down - (gravity_down * source_up_axis).sum(-1, keepdim=True) * source_up_axis
    downhill = _safe_normalize(downhill, source_pour_axis)
    source_pour_point = mouth_center + _SOURCE_INNER_R * downhill

    target_opening = tgt_pos + _quat_apply(tgt_quat, tgt_open_b)
    mouth_delta = target_opening - source_pour_point

    # transport_summary 8D (env _get_observations 의 transport_summary와 동일 순서)
    mouth_dist    = mouth_delta.norm(dim=-1)
    mouth_xy_dist = mouth_delta[:, :2].norm(dim=-1)
    cup_xy_dist   = (src_pos[:, :2] - target_opening[:, :2]).norm(dim=-1)
    mouth_z_clear = source_pour_point[:, 2] - target_opening[:, 2]
    up_dot_world  = source_up_axis[:, 2].clamp(-1.0, 1.0)

    mouth_dir_xy  = mouth_delta[:, :2] / mouth_delta[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tilt_dir_xy   = source_up_axis[:, :2] / source_up_axis[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_tilt_cos  = (tilt_dir_xy * mouth_dir_xy).sum(-1).clamp(-1.0, 1.0)

    mouth_dir_3d  = mouth_delta / mouth_dist.unsqueeze(-1).clamp(min=1e-6)
    pour_head_xy  = source_up_axis[:, :2]
    pour_head_norm = pour_head_xy.norm(dim=-1, keepdim=True)
    eff_head_xy = torch.where(pour_head_norm > 1e-4, pour_head_xy / pour_head_norm.clamp(min=1e-6), torch.zeros_like(pour_head_xy))
    eff_pour_head = torch.cat([eff_head_xy, torch.zeros(T, 1, dtype=dtype, device=device)], dim=-1)
    mouth_align_cos = (eff_pour_head * mouth_dir_3d).sum(-1).clamp(-1.0, 1.0)

    # g_ready = g_align_xy × g_clear (env_cfg 기본값 사용)
    _GATE_XY_SCALE    = 5.0
    _GATE_CLEAR_SCALE = 80.0
    _CLEAR_MIN        = 0.015
    _MOUTH_Z_MAX      = 0.15
    g_align_xy = torch.exp(-_GATE_XY_SCALE * mouth_xy_dist)
    g_clear = (
        torch.sigmoid(_GATE_CLEAR_SCALE * (mouth_z_clear - _CLEAR_MIN))
        * torch.sigmoid(_GATE_CLEAR_SCALE * (_MOUTH_Z_MAX - mouth_z_clear))
    )
    g_ready = g_align_xy * g_clear

    transport_summary = torch.stack([
        mouth_dist, mouth_xy_dist, cup_xy_dist, mouth_z_clear,
        up_dot_world, dir_tilt_cos, mouth_align_cos, g_ready,
    ], dim=-1)

    return mouth_delta, source_pour_axis, source_up_axis, source_pour_point, target_opening, transport_summary


def _build_cup_local_basis(
    source_mat: Tensor,  # (T, 4, 4)
    target_mat: Tensor,  # (T, 4, 4)
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """env의 _build_cup_local_tilt_rotvec 와 동일한 cup-local 직교 기저 계산.

    Returns:
        spin_axis (T,3), tilt_toward_axis (T,3), tilt_ortho_axis (T,3)
    """
    T = source_mat.shape[0]

    def _const(lst: list[float]) -> Tensor:
        return torch.tensor(lst, dtype=dtype, device=device).expand(T, -1)

    pour_pt_b  = _const(_POUR_POINT_B)
    pour_ax_b  = _const(_POUR_AXIS_B)
    up_ax_b    = _const(_UP_AXIS_B)
    tgt_open_b = _const(_TARGET_OPEN_B)

    src_pos  = source_mat[:, :3, 3]
    src_quat = _quat_xyzw_from_matrix(source_mat[:, :3, :3])
    tgt_pos  = target_mat[:, :3, 3]
    tgt_quat = _quat_xyzw_from_matrix(target_mat[:, :3, :3])

    cup_up_w   = _quat_apply(src_quat, up_ax_b)
    cup_pour_w = _quat_apply(src_quat, pour_ax_b)

    mouth_center = src_pos + _quat_apply(src_quat, pour_pt_b)
    gravity_down = cup_up_w.new_tensor([0.0, 0.0, -1.0]).expand_as(cup_up_w)
    downhill = gravity_down - (gravity_down * cup_up_w).sum(-1, keepdim=True) * cup_up_w
    downhill = _safe_normalize(downhill, cup_pour_w)
    source_pour_pt = mouth_center + _SOURCE_INNER_R * downhill

    target_opening = tgt_pos + _quat_apply(tgt_quat, tgt_open_b)

    mouth_delta   = target_opening - source_pour_pt
    pour_ax_plane = cup_pour_w - (cup_pour_w * cup_up_w).sum(-1, keepdim=True) * cup_up_w
    target_dir    = mouth_delta - (mouth_delta * cup_up_w).sum(-1, keepdim=True) * cup_up_w
    target_dir    = _safe_normalize(target_dir, pour_ax_plane)

    tilt_toward = _safe_normalize(
        torch.cross(target_dir, cup_up_w, dim=-1),
        torch.cross(pour_ax_plane, cup_up_w, dim=-1),
    )
    tilt_ortho = _safe_normalize(
        torch.cross(cup_up_w, tilt_toward, dim=-1),
        pour_ax_plane,
    )
    spin_axis = _safe_normalize(
        cup_up_w,
        cup_up_w.new_tensor([0.0, 0.0, 1.0]).expand_as(cup_up_w),
    )
    return spin_axis, tilt_toward, tilt_ortho


# ---------------------------------------------------------------------------
# 에피소드 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoEpisode:
    """변환 완료된 1개 demo 에피소드."""
    obs: Tensor      # (T, 110)
    actions: Tensor  # (T, 11)
    pour_mask: Tensor  # (T,) bool


# ---------------------------------------------------------------------------
# HDF5 로딩 + 변환
# ---------------------------------------------------------------------------

def _load_episode(
    path: Path,
    stride: int,
    device: torch.device,
) -> DemoEpisode:
    with h5py.File(path, "r") as h5:
        demo = h5["data"]["demo_0"]
        s = slice(None, None, stride)

        arm_pos   = torch.as_tensor(demo["obs"]["right_arm_joint_pos"][s],  dtype=torch.float32, device=device)
        hand_pos  = torch.as_tensor(demo["obs"]["right_hand_joint_pos"][s], dtype=torch.float32, device=device)
        joint_vel = torch.as_tensor(demo["obs"]["right_joint_vel"][s],      dtype=torch.float32, device=device)
        tip_force = torch.as_tensor(demo["obs"]["tip_force_norm"][s],       dtype=torch.float32, device=device)
        eef_mat   = torch.as_tensor(demo["obs"]["datagen_info"]["eef_pose"]["right"][s],        dtype=torch.float32, device=device)
        tgt_mat   = torch.as_tensor(demo["obs"]["datagen_info"]["target_eef_pose"]["right"][s], dtype=torch.float32, device=device)
        src_cup   = torch.as_tensor(demo["obs"]["datagen_info"]["object_pose"]["source_cup"][s],dtype=torch.float32, device=device)
        tgt_cup   = torch.as_tensor(demo["obs"]["datagen_info"]["object_pose"]["target_cup"][s],dtype=torch.float32, device=device)
        pour_start= torch.as_tensor(demo["obs"]["datagen_info"]["subtask_start_signals"]["pour_start"][s], dtype=torch.bool, device=device)

    T  = arm_pos.shape[0]
    dt = torch.float32

    # ---- Obs 조립 ----
    arm_vel  = joint_vel[:, :7]   # (T, 7)
    hand_vel = joint_vel[:, 7:27] # (T, 20)

    palm_pos = eef_mat[:, :3, 3]           # (T, 3)
    src_pos  = src_cup[:, :3, 3]           # (T, 3)
    tgt_pos  = tgt_cup[:, :3, 3]           # (T, 3)
    src_quat = _quat_xyzw_from_matrix(src_cup[:, :3, :3])  # (T, 4) xyzw
    tgt_quat = _quat_xyzw_from_matrix(tgt_cup[:, :3, :3])  # (T, 4) xyzw

    right_cup_rel_palm = src_pos - palm_pos   # (T, 3)
    left_cup_rel_palm  = tgt_pos - palm_pos   # (T, 3)

    mouth_delta, pour_axis, up_axis, _, _, transport = _cup_geometry(src_cup, tgt_cup, device, dt)

    binary_contact = (tip_force > CONTACT_FORCE_THRESHOLD / CONTACT_FORCE_MAX).to(dt)
    tip_norm       = (tip_force / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

    # last_actions: action[t-1], action[-1] = zeros
    bead_src = torch.ones(T, 1, device=device, dtype=dt)   # 소스 컵 가득 참
    bead_zero = torch.zeros(T, 3, device=device, dtype=dt)  # target/cross/spill = 0

    # ---- Action 계산 ----
    # base pose = 에피소드 첫 frame의 eef pose (pregrasp 근사)
    base_pos  = eef_mat[0, :3, 3].unsqueeze(0).expand(T, -1)         # (T, 3)
    base_quat = _quat_xyzw_from_matrix(eef_mat[0, :3, :3]).unsqueeze(0).expand(T, -1)  # (T, 4)

    tgt_pos_act  = tgt_mat[:, :3, 3]
    tgt_quat_act = _quat_xyzw_from_matrix(tgt_mat[:, :3, :3])

    pos_delta  = tgt_pos_act - base_pos                              # (T, 3)
    dq         = _quat_mul(tgt_quat_act, _quat_conj(base_quat))     # (T, 4)
    rotvec_w   = _axis_angle_from_quat(dq)                          # (T, 3)

    spin_ax, tilt_toward_ax, tilt_ortho_ax = _build_cup_local_basis(src_cup, tgt_cup, device, dt)
    spin_comp   = (rotvec_w * spin_ax).sum(-1, keepdim=True)
    tilt_toward_comp = (rotvec_w * tilt_toward_ax).sum(-1, keepdim=True)
    tilt_ortho_comp  = (rotvec_w * tilt_ortho_ax).sum(-1, keepdim=True)

    lo = torch.tensor(
        [-_DELTA_XYZ, -_DELTA_XYZ, -_DELTA_XYZ, -_DELTA_RAD, -_DELTA_RAD, -_DELTA_RAD],
        dtype=dt, device=device,
    )
    hi = torch.tensor(
        [_DELTA_XYZ, _DELTA_XYZ, _DELTA_XYZ, _DELTA_RAD, _DELTA_RAD, _DELTA_RAD],
        dtype=dt, device=device,
    )
    delta_6d = torch.cat([pos_delta, spin_comp, tilt_toward_comp, tilt_ortho_comp], dim=-1)
    palm_act = _unscale(delta_6d, lo, hi).clamp(-1.0, 1.0)
    finger_act = torch.ones(T, 5, dtype=dt, device=device)
    actions = torch.cat([palm_act, finger_act], dim=-1)  # (T, 11)

    # last_actions = shift 1 (t=0: zeros)
    last_actions = torch.zeros(T, 11, dtype=dt, device=device)
    last_actions[1:] = actions[:-1]

    obs = torch.cat([
        arm_pos,               # 7
        arm_vel,               # 7
        hand_pos,              # 20
        hand_vel,              # 20
        right_cup_rel_palm,    # 3
        src_quat,              # 4
        left_cup_rel_palm,     # 3
        tgt_quat,              # 4
        mouth_delta,           # 3
        pour_axis,             # 3
        up_axis,               # 3
        transport,             # 8
        binary_contact,        # 5
        tip_norm,              # 5
        last_actions,          # 11
        bead_src,              # 1
        bead_zero,             # 3
    ], dim=-1)  # 110D

    if obs.shape[1] != NUM_OBSERVATIONS:
        raise RuntimeError(f"{path}: obs dim {obs.shape[1]} != {NUM_OBSERVATIONS}")
    if actions.shape[1] != NUM_ACTIONS:
        raise RuntimeError(f"{path}: action dim {actions.shape[1]} != {NUM_ACTIONS}")

    # pour_mask: subtask 신호가 없으면 cup tilt 기준으로 fallback
    if pour_start.any():
        pour_mask = pour_start
    else:
        cos_thr = math.cos(math.radians(_POUR_PHASE_TILT_DEG))
        cup_up_z = _quat_apply(src_quat, torch.tensor(_UP_AXIS_B, dtype=dt, device=device).expand(T, -1))[:, 2]
        pour_mask = cup_up_z < cos_thr

    return DemoEpisode(obs=obs, actions=actions, pour_mask=pour_mask)


# ---------------------------------------------------------------------------
# DemoBCBuffer
# ---------------------------------------------------------------------------

class DemoBCBuffer:
    """HDF5 실제 데모 기반 BC 배치 샘플러."""

    def __init__(
        self,
        paths: Sequence[str | Path] = DEFAULT_DEMO_PATHS,
        stride: int = 2,
        device: str | torch.device = "cpu",
        pour_ratio: float = 0.6,
    ) -> None:
        self.device = torch.device(device)
        self.pour_ratio = float(pour_ratio)

        self.episodes: list[DemoEpisode] = []
        errors: list[str] = []
        for p in paths:
            try:
                ep = _load_episode(Path(p), stride=max(1, stride), device=self.device)
                self.episodes.append(ep)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{p}: {exc}")

        if errors:
            print(f"[DemoBCBuffer] {len(errors)} file(s) failed to load:", flush=True)
            for e in errors:
                print(f"  {e}", flush=True)

        if not self.episodes:
            raise RuntimeError("DemoBCBuffer: no episodes loaded successfully.")

        print(f"[DemoBCBuffer] loaded {len(self.episodes)} episodes.", flush=True)
        self._last_pour_ratio: float = 0.0

    def __len__(self) -> int:
        return len(self.episodes)

    def is_warm(self, min_count: int = 1) -> bool:
        return len(self.episodes) >= min_count

    def _sample_start(self, ep: DemoEpisode, seq_len: int, prefer_pour: bool) -> int:
        T = int(ep.obs.shape[0])
        max_start = max(T - 1, 0)
        if prefer_pour and ep.pour_mask.any():
            cands = ep.pour_mask.nonzero(as_tuple=False).flatten()
            pick  = int(cands[torch.randint(cands.numel(), (1,), device=self.device)].item())
            return max(0, min(pick - seq_len // 2, max_start))
        return int(torch.randint(max_start + 1, (1,), device=self.device).item())

    def sample(self, batch_size: int, seq_len: int) -> dict:
        obs_out = torch.zeros(batch_size, seq_len, NUM_OBSERVATIONS, device=self.device)
        act_out = torch.zeros(batch_size, seq_len, NUM_ACTIONS, device=self.device)
        mask    = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        pour_samples = 0

        for i in range(batch_size):
            ep = self.episodes[int(torch.randint(len(self.episodes), (1,), device=self.device).item())]
            prefer_pour = torch.rand((), device=self.device).item() < self.pour_ratio
            start = self._sample_start(ep, seq_len, prefer_pour)
            stop  = min(start + seq_len, int(ep.obs.shape[0]))
            cnt   = stop - start
            if cnt <= 0:
                continue
            obs_out[i, :cnt] = ep.obs[start:stop]
            act_out[i, :cnt] = ep.actions[start:stop]
            mask[i, :cnt]    = True
            if prefer_pour and ep.pour_mask[start:stop].any():
                pour_samples += 1

        self._last_pour_ratio = pour_samples / float(batch_size)
        return {"obs": obs_out, "actions": act_out, "mask": mask}
