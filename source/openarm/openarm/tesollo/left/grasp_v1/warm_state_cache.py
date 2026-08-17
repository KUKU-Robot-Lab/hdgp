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

"""Grasp success warm-state cache.

5g_grasp_left_v1 가 grasp 성공(종료) 시점의 robot/cup 상태를 디스크
(HDF5)에 명시 저장하기 위한 모듈. pour 계열 env 가 이 캐시를 로드해
warmstart 초기 상태로 재사용한다(pour 가 grasp 정책을 재실행하지 않음).

palm pose 는 pour 가 직접 소비하는 7D (pos3 + quat_xyzw4) 와, 추적/검증용
6D euler_zyx (pos3 + ezyx3) 를 함께 저장한다. euler→quat 변환은 pour env
의 ``_quat_xyzw_from_euler_zyx`` 와 **동일 알고리즘**(isaaclab quat 연산)을
사용해 포맷 불일치를 원천 차단한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

SCHEMA_VERSION = 2
WARM_STATE_GROUP = "warm_states"


def compute_arm_joint_match(
    actual_arm: torch.Tensor,
    target_arm: torch.Tensor,
    *,
    tol: float,
    previous_hold_steps: torch.Tensor,
    required_hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return arm target match mask and updated consecutive hold counts."""
    arm_close = (actual_arm - target_arm).abs().amax(dim=1) <= float(tol)
    next_hold = torch.where(
        arm_close,
        previous_hold_steps + 1,
        torch.zeros_like(previous_hold_steps),
    )
    matched = next_hold >= max(int(required_hold_steps), 1)
    return matched, next_hold


compute_demo_frame0_match = compute_arm_joint_match


def euler_zyx_to_quat_xyzw(euler_zyx: torch.Tensor) -> torch.Tensor:
    """ZYX Euler → quaternion (x, y, z, w).

    pour env ``_quat_xyzw_from_euler_zyx`` 와 비트 단위로 동일해야 한다
    (grasp 저장 quat == pour 가 직접 쓰는 quat 보장).
    """
    batch = euler_zyx.shape[0]
    unit_x = euler_zyx.new_tensor([1.0, 0.0, 0.0]).expand(batch, -1)
    unit_y = euler_zyx.new_tensor([0.0, 1.0, 0.0]).expand(batch, -1)
    unit_z = euler_zyx.new_tensor([0.0, 0.0, 1.0]).expand(batch, -1)
    qz = quat_from_angle_axis(euler_zyx[:, 0], unit_z)
    qy = quat_from_angle_axis(euler_zyx[:, 1], unit_y)
    qx = quat_from_angle_axis(euler_zyx[:, 2], unit_x)
    quat_wxyz = quat_mul(quat_mul(qz, qy), qx)
    return quat_wxyz[:, [1, 2, 3, 0]]


class GraspWarmStateCache:
    """Grasp 성공 종료 상태의 append-only 버퍼 + HDF5 영속화.

    capacity 만큼 채워지면 더 이상 append 하지 않는다 (overflow 무시).
    저장 시 ``source_meta`` 를 HDF5 attrs 로 함께 기록해, pour 로드 시
    spawn/workspace 정합성을 검증할 수 있게 한다.
    """

    def __init__(self, capacity: int, device, source_meta: dict[str, Any]) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._cap = int(capacity)
        self._device = device
        self.source_meta = dict(source_meta)
        self._count = 0
        self.arm = torch.zeros(self._cap, 7, device=device)
        self.hand = torch.zeros(self._cap, 20, device=device)
        # pour 가 직접 소비하는 7D: [x, y, z, qx, qy, qz, qw]
        self.palm7 = torch.zeros(self._cap, 7, device=device)
        # 추적/검증용 6D: [x, y, z, ez, ey, ex]
        self.palm_euler = torch.zeros(self._cap, 6, device=device)
        self.cup_pos_local = torch.zeros(self._cap, 3, device=device)
        self.cup_quat_wxyz = torch.zeros(self._cap, 4, device=device)
        self.num_contacts = torch.zeros(self._cap, device=device)
        self.per_finger_contact = torch.zeros(self._cap, 5, dtype=torch.bool, device=device)
        self.stable_contact_steps = torch.zeros(self._cap, dtype=torch.long, device=device)
        # 이 warmstart 를 생성한 demo 파일 인덱스 (0-based; -1 = 미태깅)
        self.demo_file_idx = torch.full((self._cap,), -1, dtype=torch.long, device=device)
        # 이 warmstart 의 물체 스펙 인덱스 (_ACTIVE_OBJECT_SPECS 순서; -1 = 미태깅)
        #   pour warm 로더가 컵 스케일 매칭/필터에 사용한다.
        self.object_spec_idx = torch.full((self._cap,), -1, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._cap

    def append(
        self,
        *,
        arm: torch.Tensor,
        hand: torch.Tensor,
        palm_euler: torch.Tensor,
        cup_pos_local: torch.Tensor,
        cup_quat_wxyz: torch.Tensor,
        num_contacts: torch.Tensor,
        per_finger_contact: torch.Tensor,
        stable_contact_steps: torch.Tensor,
        demo_file_idx: torch.Tensor | None = None,
        object_spec_idx: torch.Tensor | None = None,
    ) -> int:
        """성공 env 배치를 추가. 실제 저장한 개수(capacity 한정)를 반환."""
        if self.is_full:
            return 0
        batch = int(arm.shape[0])
        if batch == 0:
            return 0
        n = min(batch, self._cap - self._count)
        s, e = self._count, self._count + n
        self.arm[s:e] = arm[:n]
        self.hand[s:e] = hand[:n]
        pe = palm_euler[:n]
        self.palm_euler[s:e] = pe
        self.palm7[s:e, :3] = pe[:, :3]
        self.palm7[s:e, 3:7] = euler_zyx_to_quat_xyzw(pe[:, 3:6])
        self.cup_pos_local[s:e] = cup_pos_local[:n]
        self.cup_quat_wxyz[s:e] = cup_quat_wxyz[:n]
        self.num_contacts[s:e] = num_contacts[:n].float()
        self.per_finger_contact[s:e] = per_finger_contact[:n].bool()
        self.stable_contact_steps[s:e] = stable_contact_steps[:n].long()
        if demo_file_idx is not None:
            self.demo_file_idx[s:e] = demo_file_idx[:n].long()
        if object_spec_idx is not None:
            self.object_spec_idx[s:e] = object_spec_idx[:n].long()
        self._count = e
        return n

    def save_hdf5(self, path: str | Path) -> None:
        """현재까지 수집한 상태를 HDF5 로 원자적 저장 (tmp → replace)."""
        if self._count == 0:
            raise RuntimeError("GraspWarmStateCache is empty; nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        c = self._count
        tmp = path.with_suffix(path.suffix + ".tmp")
        with h5py.File(tmp, "w") as h5:
            h5.attrs["schema_version"] = SCHEMA_VERSION
            h5.attrs["count"] = c
            for key, value in self.source_meta.items():
                attr_key = f"meta/{key}"
                if isinstance(value, (str, bytes)):
                    h5.attrs[attr_key] = value
                else:
                    try:
                        h5.attrs[attr_key] = float(value)
                    except (TypeError, ValueError):
                        h5.attrs[attr_key] = str(value)
            grp = h5.create_group(WARM_STATE_GROUP)
            grp.create_dataset("arm_joint_pos", data=self.arm[:c].cpu().numpy())
            grp.create_dataset("hand_joint_pos", data=self.hand[:c].cpu().numpy())
            grp.create_dataset("palm_pose_quat_xyzw", data=self.palm7[:c].cpu().numpy())
            grp.create_dataset("palm_pose_euler_zyx", data=self.palm_euler[:c].cpu().numpy())
            grp.create_dataset("cup_pos_local", data=self.cup_pos_local[:c].cpu().numpy())
            grp.create_dataset("cup_quat_wxyz", data=self.cup_quat_wxyz[:c].cpu().numpy())
            grp.create_dataset("num_contacts", data=self.num_contacts[:c].cpu().numpy())
            grp.create_dataset(
                "per_finger_contact",
                data=self.per_finger_contact[:c].to(torch.uint8).cpu().numpy(),
            )
            grp.create_dataset(
                "stable_contact_steps",
                data=self.stable_contact_steps[:c].cpu().numpy(),
            )
            grp.create_dataset("demo_file_idx", data=self.demo_file_idx[:c].cpu().numpy())
            grp.create_dataset("object_spec_idx", data=self.object_spec_idx[:c].cpu().numpy())
        tmp.replace(path)
