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

"""유틸리티: 5g_pour_right_v1"""

import torch


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 정규화 액션을 [lower, upper] 범위로 스케일."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def pour_alignment_score(
    pour_point: torch.Tensor,      # (N, 3) env-local 또는 world 동일 프레임
    target_opening: torch.Tensor,  # (N, 3) 동일 프레임
    z_margin: float,               # 목표 z 오프셋 (두 컵 충돌 방지: 입구 위 z_margin)
    scale: float,                  # 거리 민감도
) -> torch.Tensor:
    """pour_point ↔ (target_opening + [0,0,z_margin]) 의 3D 정렬 score.

    xy 정렬(빗나감=spill)과 z 높이(고공 살포/충돌)를 단일 거리로 동시 처리한다.
    완벽 조준 시 1, 멀수록 0으로 단조 감소(항상 양수 gradient).
    """
    delta = pour_point - target_opening
    delta_z = delta[:, 2] - z_margin
    dist = torch.sqrt(
        delta[:, 0] * delta[:, 0]
        + delta[:, 1] * delta[:, 1]
        + delta_z * delta_z
        + 1e-12
    )
    return torch.exp(-scale * dist)


def to_torch(x, dtype=torch.float, device: str = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
