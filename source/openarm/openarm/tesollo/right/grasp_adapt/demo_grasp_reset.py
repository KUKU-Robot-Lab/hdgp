"""Demo-derived reset references for 5g_grasp_adapt."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch


_REQUIRED_KEYS = (
    "obs/right_arm_joint_pos",
    "obs/right_hand_joint_pos",
    "obs/datagen_info/eef_pose/right",
)


@dataclass(frozen=True)
class DemoGraspResetBank:
    """Compact demo bank for reset start poses and one fixed lift target."""

    start_arm_joint_pos: torch.Tensor
    start_hand_joint_pos: torch.Tensor
    start_palm_pose: torch.Tensor
    start_palm_pose_euler_zyx: torch.Tensor
    lift_arm_joint_pos: torch.Tensor
    lift_palm_pose: torch.Tensor
    lift_palm_pose_euler_zyx: torch.Tensor
    lift_medoid_arm_joint_pos: torch.Tensor
    start_frames: torch.Tensor
    lift_frames: torch.Tensor
    source_paths: tuple[str, ...]

    @property
    def num_demos(self) -> int:
        return int(self.start_arm_joint_pos.shape[0])

    @classmethod
    def from_hdf5_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        device: str | torch.device = "cpu",
    ) -> "DemoGraspResetBank":
        resolved_paths = _resolve_demo_paths(paths)
        if not resolved_paths:
            raise ValueError("demo_grasp_pose_paths is empty; provide at least one HDF5 path.")

        chunks = [_load_path(path) for path in resolved_paths]
        merged: dict[str, np.ndarray] = {}
        for key in (
            "start_arm",
            "start_hand",
            "start_palm",
            "start_palm_euler",
            "lift_arm",
            "lift_palm",
            "lift_palm_euler",
            "start_frames",
            "lift_frames",
        ):
            merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

        for key, value in merged.items():
            _require_finite(key, value)

        lift_positions = merged["lift_palm"][:, :3]
        mean_pos = lift_positions.mean(axis=0, keepdims=True)
        medoid_idx = int(np.argmin(np.linalg.norm(lift_positions - mean_pos, axis=1)))

        return cls(
            start_arm_joint_pos=torch.as_tensor(merged["start_arm"], dtype=torch.float32, device=device),
            start_hand_joint_pos=torch.as_tensor(merged["start_hand"], dtype=torch.float32, device=device),
            start_palm_pose=torch.as_tensor(merged["start_palm"], dtype=torch.float32, device=device),
            start_palm_pose_euler_zyx=torch.as_tensor(
                merged["start_palm_euler"], dtype=torch.float32, device=device
            ),
            lift_arm_joint_pos=torch.as_tensor(merged["lift_arm"], dtype=torch.float32, device=device),
            lift_palm_pose=torch.as_tensor(merged["lift_palm"], dtype=torch.float32, device=device),
            lift_palm_pose_euler_zyx=torch.as_tensor(
                merged["lift_palm_euler"], dtype=torch.float32, device=device
            ),
            lift_medoid_arm_joint_pos=torch.as_tensor(
                merged["lift_arm"][medoid_idx], dtype=torch.float32, device=device
            ),
            start_frames=torch.as_tensor(merged["start_frames"], dtype=torch.long, device=device),
            lift_frames=torch.as_tensor(merged["lift_frames"], dtype=torch.long, device=device),
            source_paths=tuple(str(path) for path in resolved_paths),
        )


def compute_demo_cup_spawn_local(
    start_palm_pose_euler_zyx: torch.Tensor,
    pregrasp_offset_xy: torch.Tensor,
    object_spawn_z: float,
    noise_xy: torch.Tensor,
) -> torch.Tensor:
    """Return local cup positions for demo reset: palm_xy - offset_xy + noise_xy."""
    obj_pos = torch.empty(
        start_palm_pose_euler_zyx.shape[0],
        3,
        dtype=start_palm_pose_euler_zyx.dtype,
        device=start_palm_pose_euler_zyx.device,
    )
    obj_pos[:, :2] = start_palm_pose_euler_zyx[:, :2] - pregrasp_offset_xy.unsqueeze(0) + noise_xy
    obj_pos[:, 2] = float(object_spawn_z)
    return obj_pos


def _resolve_demo_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    requested = tuple(Path(path) for path in paths)
    search_dirs = []
    for env_name in ("POUR_V1_DATASET_DIR", "DEMO_POSE_DATASET_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            search_dirs.append(Path(env_value))
    search_dirs.extend(
        [
            Path("/home/oem/rl_ws/datasets"),
            Path("/home/user/rl_ws/datasets"),
            Path("/home/user/rl_ws/teleopration_openarm_tesollo/datasets"),
        ]
    )

    resolved = []
    missing = []
    for path in requested:
        if path.exists():
            resolved.append(path)
            continue
        replacement = next((base / path.name for base in search_dirs if (base / path.name).exists()), None)
        if replacement is not None:
            resolved.append(replacement)
        else:
            missing.append(path)

    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        candidates = ", ".join(str(base) for base in search_dirs)
        raise FileNotFoundError(
            f"demo grasp reset HDF5 file(s) do not exist: {missing_text}. "
            f"Searched fallback dataset dirs: {candidates}."
        )
    return tuple(resolved)


def _load_path(path: Path) -> dict[str, np.ndarray]:
    chunks: list[dict[str, np.ndarray]] = []
    with h5py.File(path, "r") as h5:
        if "data" not in h5:
            raise KeyError(f"{path}: missing top-level 'data' group")
        for demo_name in sorted(h5["data"].keys()):
            demo = h5["data"][demo_name]
            missing = [key for key in _REQUIRED_KEYS if key not in demo]
            if missing:
                raise KeyError(f"{path}:{demo_name}: missing required key(s): {', '.join(missing)}")

            n = int(demo["obs/right_arm_joint_pos"].shape[0])
            if n <= 0:
                raise ValueError(f"{path}:{demo_name}: demo is empty")
            start_frame = 0
            lift_frame = _select_lift_frame(demo, n=n)

            start_arm = np.asarray(demo["obs/right_arm_joint_pos"][start_frame], dtype=np.float32)[None, :]
            start_hand = np.asarray(demo["obs/right_hand_joint_pos"][start_frame], dtype=np.float32)[None, :]
            lift_arm = np.asarray(demo["obs/right_arm_joint_pos"][lift_frame], dtype=np.float32)[None, :]

            eef = np.asarray(demo["obs/datagen_info/eef_pose/right"], dtype=np.float32)
            start_palm = _pose_matrix_to_pos_quat(eef[start_frame : start_frame + 1])
            lift_palm = _pose_matrix_to_pos_quat(eef[lift_frame : lift_frame + 1])
            start_euler = _pose_matrix_to_pos_euler_zyx(eef[start_frame : start_frame + 1])
            lift_euler = _pose_matrix_to_pos_euler_zyx(eef[lift_frame : lift_frame + 1])

            chunks.append(
                {
                    "start_arm": start_arm,
                    "start_hand": start_hand,
                    "start_palm": start_palm,
                    "start_palm_euler": start_euler,
                    "lift_arm": lift_arm,
                    "lift_palm": lift_palm,
                    "lift_palm_euler": lift_euler,
                    "start_frames": np.asarray([start_frame], dtype=np.int64),
                    "lift_frames": np.asarray([lift_frame], dtype=np.int64),
                }
            )

    if not chunks:
        raise ValueError(f"{path}: no demo groups found")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


def _select_lift_frame(demo: h5py.Group, *, n: int) -> int:
    key = "obs/datagen_info/subtask_start_signals/pour_start"
    if key in demo:
        starts = np.flatnonzero(np.asarray(demo[key], dtype=bool))
        if starts.size:
            return int(starts[0])
    return n - 1


def _pose_matrix_to_pos_quat(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected pose matrices with shape (N, 4, 4), got {mats.shape}")
    out = np.zeros((mats.shape[0], 7), dtype=np.float32)
    out[:, :3] = mats[:, :3, 3]
    out[:, 3:7] = _matrix_to_quat_xyzw(mats[:, :3, :3])
    return out


def _pose_matrix_to_pos_euler_zyx(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected pose matrices with shape (N, 4, 4), got {mats.shape}")
    out = np.zeros((mats.shape[0], 6), dtype=np.float32)
    out[:, :3] = mats[:, :3, 3]
    out[:, 3:6] = _matrix_to_euler_zyx(mats[:, :3, :3])
    return out


def _matrix_to_euler_zyx(rot: np.ndarray) -> np.ndarray:
    euler = np.empty((rot.shape[0], 3), dtype=np.float32)
    for i, r in enumerate(rot):
        sy = float(np.clip(-r[2, 0], -1.0, 1.0))
        ey = np.arcsin(sy)
        cy = np.cos(ey)
        if abs(cy) > 1e-6:
            ez = np.arctan2(r[1, 0], r[0, 0])
            ex = np.arctan2(r[2, 1], r[2, 2])
        else:
            ez = np.arctan2(-r[0, 1], r[1, 1])
            ex = 0.0
        euler[i] = np.asarray([ez, ey, ex], dtype=np.float32)
    return euler


def _matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    q = np.empty((rot.shape[0], 4), dtype=np.float32)
    for i, r in enumerate(rot):
        trace = float(np.trace(r))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (r[2, 1] - r[1, 2]) / s
            qy = (r[0, 2] - r[2, 0]) / s
            qz = (r[1, 0] - r[0, 1]) / s
        else:
            axis = int(np.argmax(np.diag(r)))
            if axis == 0:
                s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                qw = (r[2, 1] - r[1, 2]) / s
                qx = 0.25 * s
                qy = (r[0, 1] + r[1, 0]) / s
                qz = (r[0, 2] + r[2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
                qw = (r[0, 2] - r[2, 0]) / s
                qx = (r[0, 1] + r[1, 0]) / s
                qy = 0.25 * s
                qz = (r[1, 2] + r[2, 1]) / s
            else:
                s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
                qw = (r[1, 0] - r[0, 1]) / s
                qx = (r[0, 2] + r[2, 0]) / s
                qy = (r[1, 2] + r[2, 1]) / s
                qz = 0.25 * s
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float32)
        q[i] = quat / max(float(np.linalg.norm(quat)), 1e-8)
    return q


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"demo grasp reset '{name}' contains NaN or Inf")
