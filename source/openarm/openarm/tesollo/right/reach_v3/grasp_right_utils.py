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

"""유틸리티: 5g_grasp_right_v1"""

import torch


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 정규화 액션을 [lower, upper] 범위로 스케일."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(t, max_t), min_t)


def compute_joint7_lift_wait_target(
    actual_arm: torch.Tensor,
    *,
    joint7_delta: float,
    joint7_min: float,
    joint7_max: float,
) -> torch.Tensor:
    """Keep the grasp arm pose and move only right joint7 into lift-wait."""
    target = actual_arm.clone()
    target[:, 6] = (target[:, 6] + float(joint7_delta)).clamp(
        min=float(joint7_min),
        max=float(joint7_max),
    )
    return target


def compute_lift_readiness(
    num_contacts: torch.Tensor,
    is_grasp_phase: torch.Tensor,
    previous_hold_count: torch.Tensor,
    previous_latched: torch.Tensor,
    min_contacts: int,
    hold_steps: int,
    num_envelope_fingers: torch.Tensor | None = None,
    min_envelope_fingers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """접촉(손끝) + 인벨롭(tip&mid) + hold 로 lift 진입 판정. '감싸 잡으면 리프트' 래치.

    step-기반(LIFT_START_STEP) 트리거를 대체. 한번 래치되면 유지된다.
    인벨롭 게이트: num_envelope_fingers(손끝&중간마디 동시 접촉 손가락 수)가
    min_envelope_fingers 이상이어야 latch 진입 → 손끝만으로 일찍 리프트하는 것을 차단.
    """
    lift_contact_now = num_contacts >= int(min_contacts)
    if num_envelope_fingers is not None and int(min_envelope_fingers) > 0:
        lift_contact_now = lift_contact_now & (
            num_envelope_fingers >= int(min_envelope_fingers)
        )
    next_hold_count = torch.where(
        lift_contact_now & is_grasp_phase,
        previous_hold_count + 1,
        torch.where(
            previous_latched,
            previous_hold_count,
            torch.zeros_like(previous_hold_count),
        ),
    )
    lift_contact_ready_now = next_hold_count >= int(hold_steps)
    next_latched = previous_latched | lift_contact_ready_now
    return next_hold_count, lift_contact_ready_now, next_latched


def to_torch(x, dtype=torch.float, device: str = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
