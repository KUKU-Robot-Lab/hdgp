# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pour warmstart bank: grasp 디스크 캐시 로더.

5g_grasp_right_v7_2 의 ``GraspWarmStateCache`` 가 저장한 HDF5 를 로드해
pour env 의 warmstart 초기 상태(_warmstart_* 버퍼)로 채우기 좋은 형태로
노출한다. 로드 시 grasp 저장 당시의 spawn/workspace 메타데이터를 pour
설정과 대조해 정합성을 조기에 검증한다 (silent fail 금지).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

_GROUP = "warm_states"
_DATASETS = (
    "arm_joint_pos",
    "hand_joint_pos",
    "palm_pose_quat_xyzw",
    "palm_pose_euler_zyx",
    "cup_pos_local",
    "cup_quat_wxyz",
    "num_contacts",
)
# 구 HDF5 에는 없을 수 있으므로 선택적으로 로드
_OPTIONAL_DATASETS = ("demo_file_idx", "per_finger_contact", "stable_contact_steps")
# grasp 저장 spawn z 와 pour cfg object_spawn_z 허용 오차 (geometry critical)
_SPAWN_Z_TOL = 1e-4


@dataclass(frozen=True)
class PourWarmStateBank:
    """grasp 성공 상태 묶음 (pour warmstart 소스).

    palm_pose_quat_xyzw 는 pour 가 ``_warmstart_palm_pose`` (N,7) 로 그대로
    소비하는 표현이다.
    """

    arm_joint_pos: torch.Tensor        # (N, 7)
    hand_joint_pos: torch.Tensor       # (N, 20)
    palm_pose_quat_xyzw: torch.Tensor  # (N, 7) = pos3 + quat_xyzw4
    palm_pose_euler_zyx: torch.Tensor  # (N, 6) = pos3 + ezyx3
    cup_pos_local: torch.Tensor        # (N, 3)
    cup_quat_wxyz: torch.Tensor        # (N, 4)
    num_contacts: torch.Tensor         # (N,)
    # 이 warmstart 를 생성한 demo 파일 인덱스. 구 HDF5 에는 없으므로 None 가능.
    demo_file_idx: torch.Tensor | None  # (N,) int64, -1=미태깅
    per_finger_contact: torch.Tensor | None  # (N, 5) bool, 구 HDF5 에는 없음
    stable_contact_steps: torch.Tensor | None  # (N,) int64, 구 HDF5 에는 없음
    source_meta: dict[str, Any]
    source_paths: tuple[str, ...]

    @property
    def num_states(self) -> int:
        return int(self.arm_joint_pos.shape[0])

    def __len__(self) -> int:
        return self.num_states

    @classmethod
    def from_hdf5_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        device: str | torch.device = "cpu",
        expected_object_spawn_z: float | None = None,
        expected_palm_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> "PourWarmStateBank":
        """HDF5 경로들을 로드/병합. spawn z 불일치는 hard fail, workspace 는 warn."""
        resolved = _resolve_paths(paths)
        if not resolved:
            raise ValueError("warm_state_paths is empty; provide at least one HDF5 path.")

        chunks = [_load_path(path) for path in resolved]
        merged: dict[str, np.ndarray] = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in _DATASETS
        }
        for key, value in merged.items():
            if not np.isfinite(value).all():
                raise ValueError(f"warm-state '{key}' contains NaN or Inf")
        # optional: 모든 chunk 에 있을 때만 병합 (구 HDF5 혼용 시 skip)
        for key in _OPTIONAL_DATASETS:
            if all(key in chunk for chunk in chunks):
                merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

        # 메타데이터는 첫 파일 기준 (collect 는 단일 grasp cfg 산출이므로 동질)
        meta = chunks[0]["__meta__"]

        if expected_object_spawn_z is not None and "object_spawn_z" in meta:
            cached_z = float(meta["object_spawn_z"])
            if abs(cached_z - float(expected_object_spawn_z)) > _SPAWN_Z_TOL:
                raise ValueError(
                    "warm-state cache object_spawn_z mismatch: "
                    f"cache={cached_z} vs pour cfg={expected_object_spawn_z}. "
                    "Re-collect the grasp warm-state cache with matching spawn z."
                )

        if expected_palm_bounds is not None:
            _warn_on_workspace_mismatch(meta, expected_palm_bounds, resolved)

        # demo_file_idx: 구 HDF5 에 없으면 None (하위 호환)
        raw_demo_idx = merged.get("demo_file_idx")
        demo_file_idx_t: torch.Tensor | None = None
        if raw_demo_idx is not None:
            demo_file_idx_t = torch.as_tensor(raw_demo_idx, dtype=torch.long, device=device)
        raw_per_finger = merged.get("per_finger_contact")
        per_finger_contact_t: torch.Tensor | None = None
        if raw_per_finger is not None:
            per_finger_contact_t = torch.as_tensor(raw_per_finger, dtype=torch.bool, device=device)
        raw_stable_steps = merged.get("stable_contact_steps")
        stable_contact_steps_t: torch.Tensor | None = None
        if raw_stable_steps is not None:
            stable_contact_steps_t = torch.as_tensor(raw_stable_steps, dtype=torch.long, device=device)

        return cls(
            arm_joint_pos=_to_t(merged["arm_joint_pos"], device),
            hand_joint_pos=_to_t(merged["hand_joint_pos"], device),
            palm_pose_quat_xyzw=_to_t(merged["palm_pose_quat_xyzw"], device),
            palm_pose_euler_zyx=_to_t(merged["palm_pose_euler_zyx"], device),
            cup_pos_local=_to_t(merged["cup_pos_local"], device),
            cup_quat_wxyz=_to_t(merged["cup_quat_wxyz"], device),
            num_contacts=_to_t(merged["num_contacts"], device),
            demo_file_idx=demo_file_idx_t,
            per_finger_contact=per_finger_contact_t,
            stable_contact_steps=stable_contact_steps_t,
            source_meta=dict(meta),
            source_paths=tuple(str(p) for p in resolved),
        )


def _to_t(arr: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _resolve_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    requested = tuple(Path(p) for p in paths)
    search_dirs: list[Path] = []
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

    resolved: list[Path] = []
    missing: list[Path] = []
    for path in requested:
        if path.is_file():
            resolved.append(path)
            continue
        replacement = next(
            (base / path.name for base in search_dirs if (base / path.name).is_file()),
            None,
        )
        if replacement is not None:
            resolved.append(replacement)
        else:
            missing.append(path)

    if missing:
        missing_text = ", ".join(str(p) for p in missing)
        candidates = ", ".join(str(b) for b in search_dirs)
        raise FileNotFoundError(
            f"warm-state HDF5 file(s) do not exist: {missing_text}. "
            f"Searched fallback dirs: {candidates}. "
            "Run scripts/tools/collect_grasp_warm_states.py first."
        )
    return tuple(resolved)


def _load_path(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if _GROUP not in h5:
            raise KeyError(f"{path}: missing '{_GROUP}' group (wrong schema?)")
        grp = h5[_GROUP]
        missing = [key for key in _DATASETS if key not in grp]
        if missing:
            raise KeyError(f"{path}: missing dataset(s): {', '.join(missing)}")

        n = int(grp["arm_joint_pos"].shape[0])
        if n <= 0:
            raise ValueError(f"{path}: warm-state cache is empty")

        out: dict[str, np.ndarray] = {
            key: np.asarray(grp[key], dtype=np.float32) for key in _DATASETS
        }
        for key in _OPTIONAL_DATASETS:
            if key in grp:
                out[key] = np.asarray(grp[key])
        meta = {}
        for key, value in h5.attrs.items():
            key = str(key)
            if not key.startswith("meta/"):
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            meta[key.split("meta/", 1)[1]] = value
        out["__meta__"] = meta  # type: ignore[assignment]
        return out


def _warn_on_workspace_mismatch(
    meta: dict[str, Any],
    expected_palm_bounds: tuple[float, float, float, float, float, float],
    resolved: tuple[Path, ...],
) -> None:
    """grasp 저장 palm workspace 가 pour workspace 를 벗어나면 경고만 출력.

    pour reset 은 palm pos 를 자체 workspace 로 클램프하므로 hard fail 은
    아니지만, 큰 불일치는 palm target ↔ 실제 자세 괴리를 유발할 수 있다.
    """
    keys = ("palm_min_x", "palm_min_y", "palm_min_z", "palm_max_x", "palm_max_y", "palm_max_z")
    if not all(k in meta for k in keys):
        return
    cached = tuple(float(meta[k]) for k in keys)
    p_min_x, p_min_y, p_min_z, p_max_x, p_max_y, p_max_z = expected_palm_bounds
    expected = (p_min_x, p_min_y, p_min_z, p_max_x, p_max_y, p_max_z)
    deltas = [abs(c - e) for c, e in zip(cached, expected)]
    if max(deltas) > 1e-3:
        print(
            "[PourWarmStateBank][WARN] grasp/pour palm workspace differ "
            f"(cache={cached} pour={expected}). "
            "pour reset will clamp palm pos to its own workspace; "
            "large gaps may decouple palm target from actual arm pose. "
            f"source={', '.join(str(p) for p in resolved)}",
            flush=True,
        )
