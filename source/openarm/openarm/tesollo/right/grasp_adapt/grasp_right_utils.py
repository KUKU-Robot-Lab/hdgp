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


def to_torch(x, dtype=torch.float, device: str = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def compute_precision_grasp_mask(
    tip_contact_bool: torch.Tensor, min_opposing: int = 2
) -> torch.Tensor:
    """엄지(idx0) 접촉 AND 대향(idx1~4) min_opposing개 이상 접촉.

    envelope(감싸쥐기) 강요하던 5/5 hard gate를 대체하는 fingertip precision 파지
    판정. 엄지 대향 파지가 안정 파지의 핵심이므로 엄지 접촉을 필수로 한다.

    Args:
        tip_contact_bool: (N,5) bool — 손끝 접촉 여부(idx0=엄지, idx1~4=검지/중지/약지/소지).
        min_opposing: 엄지 대향으로 요구되는 최소 손가락 수(기본 2).
    Returns:
        (N,) bool — 안정 precision 파지 여부.
    """
    thumb = tip_contact_bool[:, 0]
    opposing = tip_contact_bool[:, 1:].sum(dim=1)
    return thumb & (opposing >= min_opposing)


def compute_precision_grasp_frac(tip_contact_bool: torch.Tensor) -> torch.Tensor:
    """precision 파지 품질 [0,1] (엄지 없으면 0). graded 게이팅용.

    엄지 + 대향 손가락 수를 최대 3접촉(엄지+2)으로 정규화한다.

    Args:
        tip_contact_bool: (N,5) bool — 손끝 접촉 여부(idx0=엄지).
    Returns:
        (N,) float — [0,1] precision 파지 품질.
    """
    thumb = tip_contact_bool[:, 0].float()
    opposing = tip_contact_bool[:, 1:].sum(dim=1).float()
    frac = (1.0 + opposing) / 3.0
    return (thumb * frac).clamp(0.0, 1.0)
