"""DemoBCDatasetV2 — 52D 양팔 FK obs (datagen_info cup 미사용).

Isaac Sim 의존 없음. 순수 HDF5 → torch 변환.

Obs 52D:
  [0:7]   right_arm_joint_pos
  [7:14]  right_arm_joint_vel
  [14:19] right_hand_summary (finger grasp progress)
  [19:26] left_arm_joint_pos
  [26:33] left_arm_joint_vel  (zeros — HDF5에 없음)
  [33:36] right_palm_pos      (from eef_pose/right FK)
  [36:40] right_palm_quat     (xyzw, from eef_pose/right FK)
  [40:43] left_palm_pos       (from eef_pose/left FK = target cup 추정)
  [43:46] pour_vec            (left_palm_pos - right_palm_pos)
  [46:52] last_palm_actions   (action[t-1][:6])

Action 11D (기존과 동일):
  [0:3]  pos delta (normalized)
  [3:6]  cup-local rotation (spin, tilt_toward, tilt_ortho, normalized)
  [6:11] per-finger lerp = 1.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 상수 (pour_right_preset.py에서 복사, Isaac 의존 없음)
# ---------------------------------------------------------------------------
_HAND_APPROACH_POSE = [
    0.0, -1.57, -0.5, 0.0,
    0.0,  0.0,   0.0, 0.0,
    0.0,  0.0,   0.0, 0.0,
    0.0,  0.0,   0.0, 0.0,
    0.0,  0.0,   0.0, 0.0,
]
_HAND_GRASP_POSE = [
    0.0, -1.57, 1.5, 1.5,
    0.0,  1.6,  1.5, 1.5,
    0.0,  1.6,  1.5, 1.5,
    0.0,  1.6,  1.5, 1.5,
    0.0,  0.0,  1.5, 1.5,
]

_POUR_POINT_B    = [0.0, 0.0, 0.100]
_POUR_AXIS_B     = [1.0, 0.0, 0.0]
_UP_AXIS_B       = [0.0, 0.0, 1.0]
_TARGET_OPEN_B   = [0.0, 0.0, 0.100]
_SOURCE_INNER_R  = 0.041

_DELTA_XYZ = 0.5
_DELTA_RAD  = math.radians(120.0)
_POUR_PHASE_TILT_DEG = 70.0

NUM_FINGER_ACTION = 5

# ---------------------------------------------------------------------------
# 수학 헬퍼 (순수 numpy, Isaac 의존 없음)
# ---------------------------------------------------------------------------

def _quat_xyzw_from_matrix_np(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) → (..., 4) xyzw"""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = np.sqrt(np.clip(1.0 + trace, 0.0, None)) / 2.0
    qx = np.sqrt(np.clip(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0.0, None)) / 2.0
    qy = np.sqrt(np.clip(1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 0.0, None)) / 2.0
    qz = np.sqrt(np.clip(1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], 0.0, None)) / 2.0
    qx = np.copysign(qx, R[..., 2, 1] - R[..., 1, 2])
    qy = np.copysign(qy, R[..., 0, 2] - R[..., 2, 0])
    qz = np.copysign(qz, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([qx, qy, qz, qw], axis=-1)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _quat_apply_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q_xyzw[..., :3]
    w   = q_xyzw[..., 3:4]
    return v + 2.0 * w * np.cross(xyz, v) + 2.0 * np.cross(xyz, np.cross(xyz, v))


def _safe_normalize_np(v: np.ndarray, fallback: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    fb_norm = np.linalg.norm(fallback, axis=-1, keepdims=True).clip(eps)
    return np.where(norm > eps, v / norm.clip(eps), fallback / fb_norm)


def _build_cup_local_basis_np(
    src_pos: np.ndarray,  # (T, 3)
    src_quat: np.ndarray, # (T, 4) xyzw
    tgt_pos: np.ndarray,  # (T, 3)
    tgt_quat: np.ndarray, # (T, 4) xyzw
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cup-local (spin, tilt_toward, tilt_ortho) 직교 기저."""
    T = src_pos.shape[0]
    pour_ax_b  = np.broadcast_to(_POUR_AXIS_B, (T, 3))
    up_ax_b    = np.broadcast_to(_UP_AXIS_B, (T, 3))
    pour_pt_b  = np.broadcast_to(_POUR_POINT_B, (T, 3))
    tgt_open_b = np.broadcast_to(_TARGET_OPEN_B, (T, 3))

    cup_up_w   = _quat_apply_np(src_quat, up_ax_b)
    cup_pour_w = _quat_apply_np(src_quat, pour_ax_b)

    mouth_center = src_pos + _quat_apply_np(src_quat, pour_pt_b)
    gravity_down = np.broadcast_to([0.0, 0.0, -1.0], (T, 3)).copy()
    downhill = gravity_down - (gravity_down * cup_up_w).sum(-1, keepdims=True) * cup_up_w
    downhill = _safe_normalize_np(downhill, cup_pour_w)
    source_pour_pt = mouth_center + _SOURCE_INNER_R * downhill

    target_opening = tgt_pos + _quat_apply_np(tgt_quat, tgt_open_b)

    mouth_delta   = target_opening - source_pour_pt
    pour_ax_plane = cup_pour_w - (cup_pour_w * cup_up_w).sum(-1, keepdims=True) * cup_up_w
    target_dir    = mouth_delta - (mouth_delta * cup_up_w).sum(-1, keepdims=True) * cup_up_w
    target_dir    = _safe_normalize_np(target_dir, pour_ax_plane)

    tilt_toward = _safe_normalize_np(
        np.cross(target_dir, cup_up_w),
        np.cross(pour_ax_plane, cup_up_w),
    )
    tilt_ortho = _safe_normalize_np(
        np.cross(cup_up_w, tilt_toward),
        pour_ax_plane,
    )
    spin_axis = _safe_normalize_np(
        cup_up_w,
        np.broadcast_to([0.0, 0.0, 1.0], (T, 3)).copy(),
    )
    return spin_axis, tilt_toward, tilt_ortho


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def _quat_mul_np(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = l[..., 0], l[..., 1], l[..., 2], l[..., 3]
    rx, ry, rz, rw = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
        lw*rw - lx*rx - ly*ry - lz*rz,
    ], axis=-1)


def _axis_angle_from_quat_np(q: np.ndarray) -> np.ndarray:
    """xyzw → axis-angle (rotvec)"""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / norm.clip(1e-8)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., 3:4].clip(1e-8))
    axis  = np.where(sin_half > 1e-8, xyz / sin_half.clip(1e-8), np.zeros_like(xyz))
    return axis * angle


def _unscale_np(val: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (val - lo) / max(hi - lo, 1e-8) - 1.0


def _finger_grasp_progress_np(hand_pos: np.ndarray) -> np.ndarray:
    """(T, 20) → (T, 5) per-finger grasp progress ∈ [0, 1]"""
    open_pose  = np.array(_HAND_APPROACH_POSE, dtype=np.float32)  # (20,)
    grasp_pose = np.array(_HAND_GRASP_POSE, dtype=np.float32)     # (20,)
    delta = grasp_pose - open_pose
    valid = np.abs(delta) > 1e-6
    denom = np.where(valid, delta, np.ones_like(delta))
    progress_20 = ((hand_pos - open_pose[None]) / denom[None]).clip(0.0, 1.0)  # (T, 20)
    progress_20 = progress_20 * valid[None].astype(np.float32)
    valid_counts = valid.reshape(NUM_FINGER_ACTION, 4).sum(axis=-1).clip(1).astype(np.float32)  # (5,)
    return progress_20.reshape(-1, NUM_FINGER_ACTION, 4).sum(axis=-1) / valid_counts[None]  # (T, 5)


# ---------------------------------------------------------------------------
# 에피소드 데이터
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoEpisodeFk:
    obs: np.ndarray      # (T, 52)
    actions: np.ndarray  # (T, 11)
    pour_mask: np.ndarray  # (T,) bool


# ---------------------------------------------------------------------------
# HDF5 로딩
# ---------------------------------------------------------------------------

def _load_episode_fk(
    path: Path,
    stride: int = 1,
) -> DemoEpisodeFk:
    """HDF5 → 52D FK obs + 11D action.

    datagen_info cup pos 사용 안 함. EEF FK만 사용.
    cup-local basis 계산에는 datagen_info cup pos 사용 (action 일관성 유지).
    """
    with h5py.File(path, "r") as h5:
        demo = h5["data"]["demo_0"]
        s = slice(0, None, stride)

        arm_pos   = np.array(demo["obs"]["right_arm_joint_pos"][s], dtype=np.float32)     # (T, 7)
        arm_vel   = np.array(demo["obs"]["right_joint_vel"][s], dtype=np.float32)[:, :7]  # (T, 7)
        hand_pos  = np.array(demo["obs"]["right_hand_joint_pos"][s], dtype=np.float32)    # (T, 20)
        left_arm  = np.array(demo["obs"]["left_arm_joint_pos"][s], dtype=np.float32)      # (T, 7)
        r_eef_mat = np.array(demo["obs"]["datagen_info"]["eef_pose"]["right"][s], dtype=np.float32)  # (T, 4, 4)
        l_eef_mat = np.array(demo["obs"]["datagen_info"]["eef_pose"]["left"][s], dtype=np.float32)   # (T, 4, 4)
        tgt_mat   = np.array(demo["obs"]["datagen_info"]["target_eef_pose"]["right"][s], dtype=np.float32)  # (T, 4, 4)
        # cup pos: action 계산용 (cup-local basis)
        src_cup   = np.array(demo["obs"]["datagen_info"]["object_pose"]["source_cup"][s], dtype=np.float32)  # (T, 4, 4)
        tgt_cup   = np.array(demo["obs"]["datagen_info"]["object_pose"]["target_cup"][s], dtype=np.float32)  # (T, 4, 4)
        pour_start_raw = demo["obs"]["datagen_info"]["subtask_start_signals"]["pour_start"][s]
        pour_start = np.array(pour_start_raw, dtype=bool)

    T = arm_pos.shape[0]

    # ---- FK obs 조립 ----
    r_palm_pos  = r_eef_mat[:, :3, 3]                                   # (T, 3)
    r_palm_quat = _quat_xyzw_from_matrix_np(r_eef_mat[:, :3, :3])       # (T, 4) xyzw
    l_palm_pos  = l_eef_mat[:, :3, 3]                                   # (T, 3)
    pour_vec    = l_palm_pos - r_palm_pos                                # (T, 3)

    left_vel    = np.zeros((T, 7), dtype=np.float32)
    finger_prog = _finger_grasp_progress_np(hand_pos)                    # (T, 5)

    # last_palm_actions = shift-1 (t=0: zeros)
    last_palm_actions = np.zeros((T, 6), dtype=np.float32)

    obs = np.concatenate([
        arm_pos,           # 7
        arm_vel,           # 7
        finger_prog,       # 5
        left_arm,          # 7
        left_vel,          # 7
        r_palm_pos,        # 3
        r_palm_quat,       # 4
        l_palm_pos,        # 3
        pour_vec,          # 3
        last_palm_actions, # 6
    ], axis=-1)  # (T, 52)

    assert obs.shape == (T, 52), f"{path}: obs shape {obs.shape}"

    # ---- Action 계산 (demo_bc_buffer._load_episode() 방식과 동일) ----
    base_pos  = r_eef_mat[0, :3, 3][None].repeat(T, axis=0)             # (T, 3)
    base_quat = _quat_xyzw_from_matrix_np(r_eef_mat[0:1, :3, :3]).repeat(T, axis=0)  # (T, 4)

    tgt_pos_act  = tgt_mat[:, :3, 3]                                     # (T, 3)
    tgt_quat_act = _quat_xyzw_from_matrix_np(tgt_mat[:, :3, :3])        # (T, 4)

    pos_delta = tgt_pos_act - base_pos                                   # (T, 3)

    dq = _quat_mul_np(tgt_quat_act, _quat_conj_np(base_quat))
    rotvec_w = _axis_angle_from_quat_np(dq)                             # (T, 3)

    src_pos_cup  = src_cup[:, :3, 3]
    src_quat_cup = _quat_xyzw_from_matrix_np(src_cup[:, :3, :3])
    tgt_pos_cup  = tgt_cup[:, :3, 3]
    tgt_quat_cup = _quat_xyzw_from_matrix_np(tgt_cup[:, :3, :3])

    spin_ax, tilt_toward_ax, tilt_ortho_ax = _build_cup_local_basis_np(
        src_pos_cup, src_quat_cup, tgt_pos_cup, tgt_quat_cup,
    )

    spin_comp        = (rotvec_w * spin_ax).sum(-1, keepdims=True)        # (T, 1)
    tilt_toward_comp = (rotvec_w * tilt_toward_ax).sum(-1, keepdims=True) # (T, 1)
    tilt_ortho_comp  = (rotvec_w * tilt_ortho_ax).sum(-1, keepdims=True)  # (T, 1)

    delta_6d = np.concatenate([pos_delta, spin_comp, tilt_toward_comp, tilt_ortho_comp], axis=-1)

    lo = np.array([-_DELTA_XYZ, -_DELTA_XYZ, -_DELTA_XYZ, -_DELTA_RAD, -_DELTA_RAD, -_DELTA_RAD], dtype=np.float32)
    hi = np.array([ _DELTA_XYZ,  _DELTA_XYZ,  _DELTA_XYZ,  _DELTA_RAD,  _DELTA_RAD,  _DELTA_RAD], dtype=np.float32)
    palm_act = (2.0 * (delta_6d - lo) / (hi - lo).clip(1e-8) - 1.0).clip(-1.0, 1.0)  # (T, 6)

    finger_act = np.ones((T, 5), dtype=np.float32)
    actions = np.concatenate([palm_act, finger_act], axis=-1)  # (T, 11)

    # obs의 last_palm_actions를 shift-1로 채움
    obs[1:, 46:52] = palm_act[:-1]

    # pour_mask
    if pour_start.any():
        pour_mask = pour_start
    else:
        cos_thr = math.cos(math.radians(_POUR_PHASE_TILT_DEG))
        cup_up_z = _quat_apply_np(src_quat_cup, np.broadcast_to(_UP_AXIS_B, (T, 3)).copy())[:, 2]
        pour_mask = cup_up_z < cos_thr

    return DemoEpisodeFk(obs=obs, actions=actions, pour_mask=pour_mask)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DemoBCDatasetV2(Dataset):
    """52D FK obs + 11D action BC dataset.

    각 sample = (obs, action_chunk, mask)
    - obs: (52,) float32 — t 시점의 obs
    - action_chunk: (chunk_size, 11) float32 — t~t+chunk 행동 시퀀스
    - mask: (chunk_size,) bool — 유효 timestep (에피소드 끝 근처 패딩)
    """

    def __init__(
        self,
        demo_paths: Sequence[Path],
        stride: int = 1,
        chunk_size: int = 16,
        pour_only_ratio: float = 0.0,
    ) -> None:
        self.chunk_size = chunk_size
        self._obs: List[np.ndarray]     = []
        self._actions: List[np.ndarray] = []
        self._masks: List[np.ndarray]   = []

        for path in demo_paths:
            if not Path(path).exists():
                print(f"[DemoBCDatasetV2] skip missing: {path}")
                continue
            ep = _load_episode_fk(path, stride=stride)
            T = ep.obs.shape[0]

            for t in range(T):
                end = min(t + chunk_size, T)
                length = end - t
                chunk = np.zeros((chunk_size, 11), dtype=np.float32)
                chunk[:length] = ep.actions[t:end]
                mask = np.zeros(chunk_size, dtype=bool)
                mask[:length] = True

                self._obs.append(ep.obs[t])
                self._actions.append(chunk)
                self._masks.append(mask)

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx: int):
        return (
            torch.as_tensor(self._obs[idx]),
            torch.as_tensor(self._actions[idx]),
            torch.as_tensor(self._masks[idx]),
        )
