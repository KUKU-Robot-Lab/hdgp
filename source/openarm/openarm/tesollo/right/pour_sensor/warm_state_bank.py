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
from typing import Iterable

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
    source_meta: dict[str, float]
    source_paths: tuple[str, ...]
    # ★수집 당시 각 상태가 **어느 물체**에서 나왔는가 (다물체 뱅크 전용, (N,)).
    #   `MultiAssetSpawnerCfg(random_choice=False)` 는 env_i 의 물체를 `i % N` 로 고정한다.
    #   그래서 복원은 **같은 물체에서 수집한 상태**에서만 골라야 한다 — 안 그러면 큰 컵
    #   자세(s130 컵-손 61.9mm)를 작은 컵 env(s085 45.8mm)에 넣게 되고 파지가 성립하지
    #   않는다. 구 뱅크에는 없으므로 None 을 허용한다(호출부가 fallback 을 고른다).
    object_spec_idx: torch.Tensor | None = None
    # ★손 관절 **지령** (N, 20). `hand_joint_pos` 는 컵을 누르며 멈춘 **측정**이다.
    #   PD 에서 힘을 만드는 것은 위치 오차이므로, 파지를 유지시키려면 hold 목표로
    #   **이쪽**을 써야 한다. 측정을 목표로 주면 오차가 0 이 되어 파지력이 사라진다
    #   (09.01 실측: 손가락이 최대 80° 벌어지고 컵이 사이에 끼워지기만 함).
    #   구 뱅크에는 없으므로 None 을 허용하고, **소비 시점에** fail-loud 한다.
    hand_joint_pos_target: torch.Tensor | None = None

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
        expected_robot_usd: str | None = None,
    ) -> "PourWarmStateBank":
        """HDF5 경로들을 로드/병합. spawn z 불일치는 hard fail, workspace 는 warn.

        expected_robot_usd: 기대하는 로봇 자산 이름(부분 일치, 예 "openarm_tesollo_sensor_rl").
        캐시에 `robot_usd` 출처가 기록돼 있고 다르면 hard fail, 기록이 없으면 warn.
        """
        resolved = _resolve_paths(paths)
        if not resolved:
            raise ValueError("warm_state_paths is empty; provide at least one HDF5 path.")

        chunks = [_load_path(path) for path in resolved]
        _warn_on_robot_usd_mismatch(
            tuple(str(chunk.get("__robot_usd__", "")) for chunk in chunks),
            expected_robot_usd,
            resolved,
        )
        merged: dict[str, np.ndarray] = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in _DATASETS
        }
        for key, value in merged.items():
            if not np.isfinite(value).all():
                raise ValueError(f"warm-state '{key}' contains NaN or Inf")

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

        return cls(
            arm_joint_pos=_to_t(merged["arm_joint_pos"], device),
            hand_joint_pos=_to_t(merged["hand_joint_pos"], device),
            palm_pose_quat_xyzw=_to_t(merged["palm_pose_quat_xyzw"], device),
            palm_pose_euler_zyx=_to_t(merged["palm_pose_euler_zyx"], device),
            cup_pos_local=_to_t(merged["cup_pos_local"], device),
            cup_quat_wxyz=_to_t(merged["cup_quat_wxyz"], device),
            num_contacts=_to_t(merged["num_contacts"], device),
            source_meta={k: float(v) for k, v in meta.items()},
            source_paths=tuple(str(p) for p in resolved),
            object_spec_idx=_merged_spec_idx(chunks, device),
            hand_joint_pos_target=_merged_optional(chunks, "__hand_target__", device),
        )


def _merged_optional(chunks: list[dict], key: str, device) -> "torch.Tensor | None":
    """선택 데이터셋은 **전 파일에 있어야** 이어붙인다. 하나라도 없으면 None.

    일부만 있는 채로 섞으면 "어떤 상태는 지령이 있고 어떤 상태는 없는" 뱅크가 되어,
    파지가 되는 env 와 안 되는 env 가 뒤섞인다 — 가장 찾기 어려운 종류의 버그다.
    """
    parts = [c.get(key) for c in chunks]
    if any(p is None for p in parts):
        return None
    return _to_t(np.concatenate(parts, axis=0), device)


def _merged_spec_idx(chunks: list[dict], device) -> "torch.Tensor | None":
    """물체 인덱스는 **전 파일에 있어야** 이어붙인다. 하나라도 없으면 None 이다 —
    일부만 있는 채로 섞으면 어느 상태가 어느 컵인지 모르는 뱅크가 된다."""
    parts = [c.get("__object_spec_idx__") for c in chunks]
    if any(p is None for p in parts):
        return None
    return _to_t(np.concatenate(parts, axis=0), device).long()


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
            "Run scripts/warm_states/collect_grasp_warm_states.py first."
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
        # 선택 데이터셋 — 없으면 None. 필수로 두면 구 뱅크가 전부 막힌다.
        out["__object_spec_idx__"] = (  # type: ignore[assignment]
            np.asarray(grp["object_spec_idx"], dtype=np.float32)
            if "object_spec_idx" in grp else None
        )
        out["__hand_target__"] = (  # type: ignore[assignment]
            np.asarray(grp["hand_joint_pos_target"], dtype=np.float32)
            if "hand_joint_pos_target" in grp else None
        )
        # 수치 메타만 로드. grasp_v1/v7_2 collector가 기록하는 문자열 메타
        # (cup_z_mode="actual_lifted", export_mode=...)는 로더 계약상 불필요하므로 건너뛴다.
        # (float() 강제 변환 시 문자열에서 ValueError → warm-state 로드 전체 실패 방지)
        meta = {}
        for k, v in h5.attrs.items():
            ks = str(k)
            if not ks.startswith("meta/"):
                continue
            try:
                meta[ks.split("meta/", 1)[1]] = float(v)
            except (TypeError, ValueError):
                continue
        # ★자산 출처(robot_usd) — both/pour_v1 가드 이식(2026-08-18).
        #   2026-08-17 사고: 로봇 자산이 DG-5F → DG-5FS 로 교체됐는데 warm state 텐서 차원이
        #   같아서(arm7+hand20) 구 캐시가 **에러 없이** 로드됐다. 손 기하가 달라 palm/컵 상대
        #   자세가 어긋난 초기상태로 학습이 돌 수 있었다. 문자열 속성은 수치 메타 루프에서
        #   버려지므로 여기서 별도로 보존한다.
        _usd = h5.attrs.get("robot_usd", h5.attrs.get("meta/robot_usd"))
        out["__robot_usd__"] = (  # type: ignore[assignment]
            str(_usd) if _usd is not None else ""
        )
        out["__meta__"] = meta  # type: ignore[assignment]
        return out


def _warn_on_robot_usd_mismatch(
    robot_usds: tuple[str, ...], expected_robot_usd: str | None, resolved: tuple[Path, ...]
) -> None:
    """캐시가 어느 로봇 자산에서 나왔는지 확인한다.

    출처 기록이 없으면(구 캐시) 경고만 한다 — hard fail 로 두면 기존 캐시가 전부 막힌다.
    기록이 있고 기대값과 다르면 **hard fail**: 이 불일치는 조용히 잘못된 학습으로 이어진다.
    """
    tagged = [u for u in robot_usds if u]
    if not tagged:
        print(
            "[PourWarmStateBank][WARN] warm 캐시에 robot_usd 출처 기록이 없다 "
            f"({[p.name for p in resolved]}). 이 캐시가 현재 로봇 자산(sensor_rl/DG-5F)에서 "
            "나온 것인지 자동 확인할 수 없다 — 수집 자산을 직접 확인할 것.",
            flush=True,
        )
        return
    if expected_robot_usd is None:
        return
    bad = sorted({u for u in tagged if expected_robot_usd not in u})
    if bad:
        raise ValueError(
            f"warm 캐시 robot_usd 불일치: 기대='{expected_robot_usd}', 캐시={bad}. "
            "다른 로봇 자산에서 수집한 캐시다 — 손 기하가 달라 초기 파지가 어긋난다. 재수집 필요."
        )


def _warn_on_workspace_mismatch(
    meta: dict[str, float],
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
