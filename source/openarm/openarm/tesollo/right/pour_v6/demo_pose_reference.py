"""Demo pose reference bank for 5g_pour_right_v5 reward shaping.

This module intentionally only extracts compact pose statistics from HDF5
demonstrations. It does not expose actions or behavior-cloning targets.
"""

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
    "obs/right_hand_reference_joint_pos",
    "obs/left_arm_joint_pos",
    "obs/left_gripper_joint_pos",
    "obs/datagen_info/eef_pose/left",
    "obs/datagen_info/eef_pose/right",
    "obs/datagen_info/target_eef_pose/right",
    "timestamps_ns",
)


@dataclass(frozen=True)
class DemoPoseReferenceBank:
    """Compact tensor bank used by v5 rewards."""

    arm_joint_pos: torch.Tensor
    hand_joint_pos: torch.Tensor
    hand_reference_joint_pos: torch.Tensor
    left_joint_pos: torch.Tensor
    palm_pose: torch.Tensor
    target_palm_pose: torch.Tensor
    target_cup_pose: torch.Tensor
    arm_joint_mean: torch.Tensor
    arm_joint_std: torch.Tensor
    palm_pos_mean: torch.Tensor
    palm_pos_std: torch.Tensor
    palm_quat_mean: torch.Tensor
    thumb_joint_mean: torch.Tensor
    thumb_joint_std: torch.Tensor
    arm_vel_l2_p95: torch.Tensor
    arm_jerk_l2_p95: torch.Tensor
    demo_start_indices: torch.Tensor
    demo_start_match_linf: torch.Tensor
    demo_start_mode: str
    source_paths: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return int(self.arm_joint_pos.shape[1])

    @property
    def num_demos(self) -> int:
        return int(self.arm_joint_pos.shape[0])

    @classmethod
    def from_hdf5_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        phase: str = "all",
        device: str | torch.device = "cpu",
        episode_steps: int = 1200,
        demo_start_fraction: float = 0.46,
        demo_pose_start_mode: str = "fraction",
        warm_state_paths: Iterable[str | Path] | None = None,
    ) -> "DemoPoseReferenceBank":
        resolved_paths = _resolve_demo_paths(paths)
        arrays = [_load_path(path, phase=phase) for path in resolved_paths]
        if not resolved_paths:
            raise ValueError("demo_pose_paths is empty; provide at least one HDF5 path.")

        merged: dict[str, np.ndarray] = {}
        for key in ("arm", "hand", "hand_ref", "left", "palm", "target_palm", "target_cup"):
            merged[key] = np.concatenate([arr[key] for arr in arrays], axis=0)
            _require_finite(key, merged[key])
        if merged["arm"].shape[0] == 0:
            raise ValueError("demo reference bank is empty after phase selection.")

        # stats는 리샘플링 전 원본 데이터로 계산 (분포 통계는 전체 기준)
        arm_raw  = torch.as_tensor(merged["arm"],  dtype=torch.float32)
        palm_raw = torch.as_tensor(merged["palm"], dtype=torch.float32)
        hand_ref_raw = torch.as_tensor(merged["hand_ref"], dtype=torch.float32)

        arm_std      = arm_raw.std(dim=0, unbiased=False).clamp(min=0.05)
        palm_pos_std = palm_raw[:, :3].std(dim=0, unbiased=False).clamp(min=0.01)
        thumb_ref    = hand_ref_raw[:, :4]
        # Teleop thumb references are nearly constant in the pour segment. A small
        # statistical std would turn the grip term into a hard supervised target,
        # so use a broad floor and keep it as shaping.
        thumb_std = thumb_ref.std(dim=0, unbiased=False).clamp(min=0.5)

        quat = palm_raw[:, 3:7]
        quat = torch.where((quat[:, 3:4] < 0.0), -quat, quat)
        quat_mean = torch.nn.functional.normalize(quat.mean(dim=0), dim=0)

        vel_l2 = np.concatenate([arr["arm_vel_l2"] for arr in arrays], axis=0)
        jerk_l2 = np.concatenate([arr["arm_jerk_l2"] for arr in arrays], axis=0)
        arm_vel_p95  = float(np.percentile(vel_l2,  95)) if vel_l2.size  else 1.0
        arm_jerk_p95 = float(np.percentile(jerk_l2, 95)) if jerk_l2.size else 1.0

        # 각 demo를 독립적으로 demo_start_fraction 지점부터 끝까지 잘라낸 뒤
        # episode_steps 길이로 선형 리샘플링한다.
        # warmstart 에피소드 시작 상태(컵 들린 직후)와 demo의 해당 시점을 정렬.
        # phase tag를 쓰지 않고 단순 시간 비율로 구분.
        keys_to_resample = ("arm", "hand", "hand_ref", "left", "palm", "target_palm", "target_cup")
        resampled: dict[str, list[np.ndarray]] = {key: [] for key in keys_to_resample}
        segment_lengths: list[int] = []
        start_indices, start_match_linf = _resolve_demo_start_indices(
            arrays,
            demo_start_fraction=demo_start_fraction,
            demo_pose_start_mode=demo_pose_start_mode,
            warm_state_paths=warm_state_paths,
        )
        for arr, start_i in zip(arrays, start_indices, strict=True):
            raw_len = int(arr["arm"].shape[0])
            if raw_len < 2:
                raise ValueError("demo trajectory must contain at least 2 frames after phase selection.")
            segment_lengths.append(raw_len - start_i)
            for key in keys_to_resample:
                resampled[key].append(_resample_array(arr[key][start_i:], episode_steps))

        stacked = {key: np.stack(values, axis=0) for key, values in resampled.items()}
        for key in keys_to_resample:
            _require_finite(key, stacked[key])

        arm        = torch.as_tensor(stacked["arm"],        dtype=torch.float32, device=device)
        hand       = torch.as_tensor(stacked["hand"],       dtype=torch.float32, device=device)
        hand_ref   = torch.as_tensor(stacked["hand_ref"],   dtype=torch.float32, device=device)
        left       = torch.as_tensor(stacked["left"],       dtype=torch.float32, device=device)
        palm       = torch.as_tensor(stacked["palm"],       dtype=torch.float32, device=device)
        target_palm = torch.as_tensor(stacked["target_palm"], dtype=torch.float32, device=device)
        target_cup = torch.as_tensor(stacked["target_cup"], dtype=torch.float32, device=device)

        print(
            f"[DemoPoseReferenceBank] resampled: demos={len(arrays)}, "
            f"raw_total={merged['arm'].shape[0]} frames, "
            f"start_mode={demo_pose_start_mode}, "
            f"start_fraction={demo_start_fraction:.2f}, "
            f"start_i_range={min(start_indices)}..{max(start_indices)}, "
            f"seg_len_range={min(segment_lengths)}..{max(segment_lengths)}, "
            f"match_linf_range={min(start_match_linf):.4f}..{max(start_match_linf):.4f}, "
            f"episode_steps={episode_steps}",
            flush=True,
        )

        return cls(
            arm_joint_pos=arm,
            hand_joint_pos=hand,
            hand_reference_joint_pos=hand_ref,
            left_joint_pos=left,
            palm_pose=palm,
            target_palm_pose=target_palm,
            target_cup_pose=target_cup,
            arm_joint_mean=arm_raw.mean(dim=0).to(device),
            arm_joint_std=arm_std.to(device),
            palm_pos_mean=palm_raw[:, :3].mean(dim=0).to(device),
            palm_pos_std=palm_pos_std.to(device),
            palm_quat_mean=quat_mean.to(device),
            thumb_joint_mean=thumb_ref.mean(dim=0).to(device),
            thumb_joint_std=thumb_std.to(device),
            arm_vel_l2_p95=torch.tensor(max(arm_vel_p95,  1e-3), dtype=torch.float32, device=device),
            arm_jerk_l2_p95=torch.tensor(max(arm_jerk_p95, 1e-3), dtype=torch.float32, device=device),
            demo_start_indices=torch.as_tensor(start_indices, dtype=torch.long, device=device),
            demo_start_match_linf=torch.as_tensor(start_match_linf, dtype=torch.float32, device=device),
            demo_start_mode=demo_pose_start_mode,
            source_paths=tuple(str(path) for path in resolved_paths),
        )


def _resolve_demo_start_indices(
    arrays: list[dict[str, np.ndarray]],
    *,
    demo_start_fraction: float,
    demo_pose_start_mode: str,
    warm_state_paths: Iterable[str | Path] | None,
) -> tuple[list[int], list[float]]:
    mode = str(demo_pose_start_mode).strip().lower()
    if mode in ("fraction", "time_fraction"):
        start_indices: list[int] = []
        for arr in arrays:
            raw_len = int(arr["arm"].shape[0])
            start_i = int(round(raw_len * demo_start_fraction))
            start_indices.append(max(0, min(start_i, raw_len - 2)))
        return start_indices, [0.0 for _ in start_indices]

    if mode != "warm_state_match":
        raise ValueError(
            "demo_pose_start_mode must be 'fraction' or 'warm_state_match', "
            f"got {demo_pose_start_mode!r}"
        )

    warm_arm_by_demo = _load_warm_arm_median_by_demo(warm_state_paths, num_demos=len(arrays))
    start_indices = []
    match_linf = []
    for demo_id, arr in enumerate(arrays):
        raw_len = int(arr["arm"].shape[0])
        arm = arr["arm"]
        target = warm_arm_by_demo[demo_id]
        err = np.max(np.abs(arm - target[None, :]), axis=1)
        matched_i = int(np.argmin(err))
        start_i = max(0, min(matched_i, raw_len - 2))
        start_indices.append(start_i)
        match_linf.append(float(err[start_i]))
    return start_indices, match_linf


def _load_warm_arm_median_by_demo(
    paths: Iterable[str | Path] | None,
    *,
    num_demos: int,
) -> list[np.ndarray]:
    if paths is None:
        raise ValueError("warm_state_paths is required when demo_pose_start_mode='warm_state_match'")

    resolved = _resolve_warm_state_paths(paths)
    if not resolved:
        raise ValueError("warm_state_paths is empty when demo_pose_start_mode='warm_state_match'")

    arm_chunks: list[np.ndarray] = []
    demo_idx_chunks: list[np.ndarray] = []
    for path in resolved:
        with h5py.File(path, "r") as h5:
            if "warm_states" not in h5:
                raise KeyError(f"{path}: missing top-level 'warm_states' group")
            grp = h5["warm_states"]
            for key in ("arm_joint_pos", "demo_file_idx"):
                if key not in grp:
                    raise KeyError(f"{path}: missing warm_states/{key}")
            arm = np.asarray(grp["arm_joint_pos"], dtype=np.float32)
            demo_idx = np.asarray(grp["demo_file_idx"], dtype=np.int64)
            if arm.ndim != 2 or arm.shape[1] != 7:
                raise ValueError(f"{path}: warm_states/arm_joint_pos must have shape (N, 7), got {arm.shape}")
            if demo_idx.shape != (arm.shape[0],):
                raise ValueError(
                    f"{path}: warm_states/demo_file_idx shape {demo_idx.shape} "
                    f"does not match arm rows {arm.shape[0]}"
                )
            arm_chunks.append(arm)
            demo_idx_chunks.append(demo_idx)

    warm_arm = np.concatenate(arm_chunks, axis=0)
    demo_idx = np.concatenate(demo_idx_chunks, axis=0)
    medians = []
    for demo_id in range(num_demos):
        valid = demo_idx >= 0
        mask = valid & ((demo_idx % num_demos) == demo_id)
        if not np.any(mask):
            raise ValueError(
                "warm-state cache does not contain a valid demo_file_idx "
                f"for demo_id={demo_id}"
            )
        medians.append(np.median(warm_arm[mask], axis=0).astype(np.float32))
    return medians


def _resolve_warm_state_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    requested = tuple(Path(path) for path in paths)
    search_dirs = []
    for env_name in ("POUR_V1_DATASET_DIR", "DEMO_POSE_DATASET_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            search_dirs.append(Path(env_value))
    search_dirs.extend(
        [
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
        candidates = ", ".join(str(base) for base in search_dirs)
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "warm-state HDF5 file(s) do not exist: "
            f"{missing_text}. Searched fallback dataset dirs: {candidates}."
        )
    return tuple(resolved)


def _resolve_demo_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    requested = tuple(Path(path) for path in paths)
    if not requested:
        return ()

    search_dirs = []
    for env_name in ("POUR_V1_DATASET_DIR", "DEMO_POSE_DATASET_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            search_dirs.append(Path(env_value))
    search_dirs.extend(
        [
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
        candidates = ", ".join(str(base) for base in search_dirs)
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "demo reference HDF5 file(s) do not exist: "
            f"{missing_text}. Searched fallback dataset dirs: {candidates}. "
            "Set POUR_V1_DATASET_DIR or DEMO_POSE_DATASET_DIR to the directory containing pour_v1_a11.hdf5 ... pour_v1_a20.hdf5."
        )
    return tuple(resolved)


def _load_path(path: Path, *, phase: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"demo reference HDF5 does not exist: {path}")

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
            selector = _select_phase_indices(demo, phase=phase, n=n)
            if selector.size == 0:
                raise ValueError(f"{path}:{demo_name}: selected '{phase}' segment is empty")

            arm = np.asarray(demo["obs/right_arm_joint_pos"][selector], dtype=np.float32)
            hand = np.asarray(demo["obs/right_hand_joint_pos"][selector], dtype=np.float32)
            hand_ref = np.asarray(demo["obs/right_hand_reference_joint_pos"][selector], dtype=np.float32)
            left_arm = np.asarray(demo["obs/left_arm_joint_pos"][selector], dtype=np.float32)
            left_gripper = np.asarray(demo["obs/left_gripper_joint_pos"][selector], dtype=np.float32)
            left = np.concatenate([left_arm, left_gripper], axis=-1)
            palm = _pose_matrix_to_pos_quat(np.asarray(demo["obs/datagen_info/eef_pose/right"][selector]))
            target_palm = _pose_matrix_to_pos_quat(
                np.asarray(demo["obs/datagen_info/target_eef_pose/right"][selector])
            )
            target_cup = _left_eef_matrix_to_cup_pose(
                np.asarray(demo["obs/datagen_info/eef_pose/left"][selector])
            )

            timestamps = np.asarray(demo["timestamps_ns"][selector], dtype=np.float64)
            arm_vel_l2, arm_jerk_l2 = _motion_stats(arm, timestamps)
            chunks.append(
                {
                    "arm": arm,
                    "hand": hand,
                    "hand_ref": hand_ref,
                    "left": left,
                    "palm": palm,
                    "target_palm": target_palm,
                    "target_cup": target_cup,
                    "arm_vel_l2": arm_vel_l2,
                    "arm_jerk_l2": arm_jerk_l2,
                }
            )

    if not chunks:
        raise ValueError(f"{path}: no demo groups found")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


def _resample_array(raw: np.ndarray, episode_steps: int) -> np.ndarray:
    if episode_steps <= 0:
        raise ValueError(f"episode_steps must be positive, got {episode_steps}")
    if raw.shape[0] < 2:
        raise ValueError("cannot resample a trajectory with fewer than 2 frames")

    src = np.linspace(0.0, raw.shape[0] - 1, episode_steps)
    lo = np.floor(src).astype(np.int64).clip(0, raw.shape[0] - 2)
    hi = lo + 1
    w = (src - lo)[:, None]
    return (raw[lo] * (1.0 - w) + raw[hi] * w).astype(np.float32)


def _select_phase_indices(demo: h5py.Group, *, phase: str, n: int) -> np.ndarray:
    if phase != "pour":
        return np.arange(n, dtype=np.int64)

    start_key = "obs/datagen_info/subtask_start_signals/pour_start"
    done_keys = (
        "obs/datagen_info/subtask_term_signals/pour_done",
        "obs/datagen_info/subtask_start_signals/pour_done",
    )
    if start_key in demo:
        starts = np.flatnonzero(np.asarray(demo[start_key], dtype=bool))
        if starts.size:
            start = int(starts[0])
            end = n
            for done_key in done_keys:
                if done_key in demo:
                    dones = np.flatnonzero(np.asarray(demo[done_key], dtype=bool) & (np.arange(n) >= start))
                    if dones.size:
                        end = int(dones[0]) + 1
                        break
            if end - start >= min(30, n):
                return np.arange(start, end, dtype=np.int64)

    if "timestamps_ns" in demo and n > 1:
        ts = np.asarray(demo["timestamps_ns"], dtype=np.int64)
        end_ns = int(ts[-1])
        start_ns = end_ns - 4_000_000_000
        idx = np.flatnonzero(ts >= start_ns)
        if idx.size:
            return idx.astype(np.int64)

    tilt_idx = _tilt_fallback_indices(demo, n)
    if 0 < tilt_idx.size < int(0.8 * n):
        return tilt_idx

    start = max(0, int(round(n * 0.75)))
    return np.arange(start, n, dtype=np.int64)


def _tilt_fallback_indices(demo: h5py.Group, n: int) -> np.ndarray:
    key = "obs/datagen_info/eef_pose/right"
    if key not in demo:
        return np.empty((0,), dtype=np.int64)
    mats = np.asarray(demo[key], dtype=np.float32)
    if mats.shape[0] != n or mats.shape[-2:] != (4, 4):
        return np.empty((0,), dtype=np.int64)
    z_dot = mats[:, 2, 2]
    idx = np.flatnonzero(z_dot < 0.75)
    return idx.astype(np.int64)


def _pose_matrix_to_pos_quat(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected pose matrices with shape (N, 4, 4), got {mats.shape}")
    out = np.zeros((mats.shape[0], 7), dtype=np.float32)
    out[:, :3] = mats[:, :3, 3]
    out[:, 3:7] = _matrix_to_quat_xyzw(mats[:, :3, :3])
    return out


def _left_eef_matrix_to_cup_pose(mats: np.ndarray, local_z: float = 0.05) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected left EEF matrices with shape (N, 4, 4), got {mats.shape}")
    rot = mats[:, :3, :3]
    pos = mats[:, :3, 3] + rot @ np.asarray([0.0, 0.0, local_z], dtype=np.float32)
    c, s = np.cos(np.pi / 2.0), np.sin(np.pi / 2.0)
    r_y90 = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    cup_rot = rot @ r_y90
    out = np.zeros((mats.shape[0], 7), dtype=np.float32)
    out[:, :3] = pos
    out[:, 3:7] = _matrix_to_quat_xyzw(cup_rot)
    return out


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


def _motion_stats(arm: np.ndarray, timestamps_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if arm.shape[0] < 2:
        return np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)
    dt = np.diff(timestamps_ns.astype(np.float64)) * 1e-9
    dt = np.clip(dt, 1.0 / 240.0, 1.0)
    vel = np.diff(arm, axis=0) / dt[:, None]
    if vel.shape[0] < 2:
        jerk = np.zeros_like(vel[:1])
    else:
        acc = np.diff(vel, axis=0) / dt[1:, None]
        jerk = np.diff(acc, axis=0) / dt[2:, None] if acc.shape[0] > 1 else acc
    return (
        np.linalg.norm(vel, axis=-1).astype(np.float32),
        np.linalg.norm(jerk, axis=-1).astype(np.float32),
    )


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"demo reference '{name}' contains NaN or Inf")
