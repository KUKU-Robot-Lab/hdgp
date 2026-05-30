#!/usr/bin/env python3
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

"""Demo별 균형 warm-state HDF5 생성기.

collect_grasp_warm_states_v5.py 가 출력한 raw HDF5 를 읽어:
  1. num_contacts >= min_contacts 로 필터
  2. demo_file_idx 기준으로 demo 별 그룹 분류
  3. demo 당 최대 per_demo 개, 최대 demo_count 개 demo 를 랜덤 샘플
  4. 결과를 새 HDF5 로 저장 (메타데이터 attrs 유지)

사용 예:
    python3 scripts/tools/balance_grasp_warm_states_by_demo.py \\
        --src /home/user/rl_ws/datasets/grasp_warm_v7_2_contact4_raw.hdf5 \\
        --out /home/user/rl_ws/datasets/grasp_warm_v7_2_contact4_balanced_400x10.hdf5 \\
        --per_demo 400 \\
        --demo_count 10 \\
        --min_contacts 4 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_GROUP = "warm_states"
_REQUIRED = (
    "arm_joint_pos",
    "hand_joint_pos",
    "palm_pose_quat_xyzw",
    "palm_pose_euler_zyx",
    "cup_pos_local",
    "cup_quat_wxyz",
    "num_contacts",
)
_OPTIONAL = ("demo_file_idx", "per_finger_contact", "stable_contact_steps")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="raw HDF5 경로 (collect 출력)")
    p.add_argument("--out", required=True, help="balanced HDF5 출력 경로")
    p.add_argument(
        "--per_demo",
        type=int,
        default=400,
        help="demo 당 최대 샘플 수",
    )
    p.add_argument(
        "--demo_count",
        type=int,
        default=10,
        help="사용할 최대 demo 수 (demo_file_idx 기준)",
    )
    p.add_argument(
        "--min_contacts",
        type=int,
        default=4,
        help="유지할 최소 접촉 손가락 수",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _validate(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if not src.is_file():
        raise FileNotFoundError(f"src not found: {src}")
    if args.per_demo <= 0:
        raise ValueError(f"--per_demo must be positive, got {args.per_demo}")
    if args.demo_count <= 0:
        raise ValueError(f"--demo_count must be positive, got {args.demo_count}")
    if args.min_contacts < 1:
        raise ValueError(f"--min_contacts must be >= 1, got {args.min_contacts}")


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as h5:
        if _GROUP not in h5:
            raise KeyError(f"missing '{_GROUP}' group in {path}")
        grp = h5[_GROUP]
        missing = [k for k in _REQUIRED if k not in grp]
        if missing:
            raise KeyError(f"missing datasets: {', '.join(missing)}")
        data: dict[str, np.ndarray] = {k: np.asarray(grp[k]) for k in _REQUIRED}
        for k in _OPTIONAL:
            if k in grp:
                data[k] = np.asarray(grp[k])
        meta = {}
        for key, value in h5.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            meta[str(key)] = value
    return data, meta


def _select_indices(
    data: dict[str, np.ndarray],
    *,
    min_contacts: int,
    per_demo: int,
    demo_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = data["arm_joint_pos"].shape[0]
    num_contacts = data["num_contacts"].flatten().astype(np.int32)

    # 1. contact 필터
    contact_ok = num_contacts >= min_contacts
    contact_ids = np.where(contact_ok)[0]
    print(
        f"[balance] contact filter: {int(contact_ok.sum())}/{n} "
        f"(num_contacts >= {min_contacts})",
        flush=True,
    )
    if contact_ids.size == 0:
        raise ValueError(
            f"No states pass num_contacts >= {min_contacts}. "
            "Lower --min_contacts or re-collect with lower --warm_min_contacts."
        )

    # 2. demo_file_idx 기반 그룹 샘플링
    if "demo_file_idx" not in data:
        # demo 태깅 없는 구 HDF5: contact 필터 후 균일 샘플
        n_out = min(per_demo * demo_count, contact_ids.size)
        chosen = rng.choice(contact_ids, size=n_out, replace=False)
        print(
            f"[balance] no demo_file_idx; uniform sample {n_out}/{contact_ids.size}",
            flush=True,
        )
        return np.sort(chosen)

    demo_idx_all = data["demo_file_idx"].flatten().astype(np.int64)
    demo_idx_filtered = demo_idx_all[contact_ids]

    unique_demos = np.unique(demo_idx_filtered)
    # -1 (미태깅)은 별도 처리
    tagged_demos = unique_demos[unique_demos >= 0]
    untagged_mask = demo_idx_filtered == -1
    untagged_ids = contact_ids[untagged_mask]

    print(
        f"[balance] demos available: {len(tagged_demos)} tagged + "
        f"{untagged_ids.size} untagged states",
        flush=True,
    )

    # demo 선택: 랜덤으로 최대 demo_count
    if len(tagged_demos) > demo_count:
        selected_demos = rng.choice(tagged_demos, size=demo_count, replace=False)
    else:
        selected_demos = tagged_demos
    print(
        f"[balance] selected {len(selected_demos)} demos (max {demo_count}), "
        f"per_demo={per_demo}",
        flush=True,
    )

    chosen: list[np.ndarray] = []
    for demo_id in sorted(selected_demos):
        mask = demo_idx_filtered == demo_id
        ids_for_demo = contact_ids[mask]
        if ids_for_demo.size > per_demo:
            sampled = rng.choice(ids_for_demo, size=per_demo, replace=False)
        else:
            sampled = ids_for_demo
        chosen.append(sampled)
        print(
            f"[balance]   demo {demo_id}: {ids_for_demo.size} → sampled {len(sampled)}",
            flush=True,
        )

    # 미태깅 상태: 남은 슬롯에 추가 (선택)
    total_tagged = sum(len(c) for c in chosen)
    remaining_slots = per_demo * demo_count - total_tagged
    if untagged_ids.size > 0 and remaining_slots > 0:
        n_untagged = min(untagged_ids.size, remaining_slots)
        chosen.append(rng.choice(untagged_ids, size=n_untagged, replace=False))
        print(
            f"[balance]   untagged: {untagged_ids.size} → sampled {n_untagged}",
            flush=True,
        )

    result = np.sort(np.concatenate(chosen))
    print(f"[balance] total selected: {len(result)}", flush=True)
    return result


def _write(
    path: Path,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        # 메타 attrs 복사
        for key, value in meta.items():
            h5.attrs[key] = value
        grp = h5.create_group(_GROUP)
        for key in _REQUIRED:
            grp.create_dataset(key, data=data[key][indices], compression="gzip")
        for key in _OPTIONAL:
            if key in data:
                grp.create_dataset(key, data=data[key][indices], compression="gzip")
    tmp.replace(path)
    print(f"[balance] written: {path} ({len(indices)} states)", flush=True)


def main() -> int:
    args = _parse_args()
    _validate(args)

    rng = np.random.default_rng(args.seed)
    src = Path(args.src)
    out = Path(args.out)

    print(f"[balance] loading {src} ...", flush=True)
    data, meta = _load(src)
    n_raw = data["arm_joint_pos"].shape[0]
    print(f"[balance] raw states: {n_raw}", flush=True)

    indices = _select_indices(
        data,
        min_contacts=args.min_contacts,
        per_demo=args.per_demo,
        demo_count=args.demo_count,
        rng=rng,
    )
    _write(out, data, indices, meta)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[balance] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
