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


def compute_radial_compression(
    contact_pos: torch.Tensor,
    contact_force: torch.Tensor,
    cup_center: torch.Tensor,
    cup_axis: torch.Tensor,
    contact_mask: torch.Tensor,
) -> torch.Tensor:
    """접촉력의 컵 중심축 방향 inward(radial 압축) 성분 합.

    종이컵 좌굴은 radial 압축이 임계를 넘을 때 발생 → 형상파괴 억제 신호.
    감싸기/palm 지지 자체는 무방(형상만 안 부수면 됨), 과도한 radial 압박만 damage.

    Args:
        contact_pos: (N,K,3) 접촉점 world 좌표.
        contact_force: (N,K,3) 접촉력 world 벡터.
        cup_center: (N,3) 컵 중심 world.
        cup_axis: (N,3) 컵 up축(정규화 가정).
        contact_mask: (N,K) 유효 접촉 {0,1}.
    Returns:
        (N,) radial inward 성분 합(≥0).
    """
    rel = contact_pos - cup_center.unsqueeze(1)                      # (N,K,3)
    axis = cup_axis.unsqueeze(1)                                     # (N,1,3)
    axial = (rel * axis).sum(dim=-1, keepdim=True) * axis            # 축방향 성분
    radial_vec = rel - axial                                        # 축에 수직(바깥 방향)
    radial_out = radial_vec / radial_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # 힘의 inward 성분(바깥 방향의 반대)만 양수로
    inward = (-(contact_force * radial_out).sum(dim=-1)).clamp(min=0.0)  # (N,K)
    return (inward * contact_mask).sum(dim=-1)                       # (N,)


def compute_damage_dose(
    prev_dose: torch.Tensor,
    radial_compression: torch.Tensor,
    f_safe: float,
    dt: float,
    q: float,
) -> torch.Tensor:
    """누적 형상파괴 dose: D_{t+1} = D_t + dt·relu((radial-f_safe)/f_safe)^q (설계 §6).

    순간 압박이 아니라 '얼마나 오래·세게 눌러 형상을 파괴했나'를 누적한다.
    성공조건(dose < 임계)에 사용. reset 시 0으로 초기화(호출부).

    Args:
        prev_dose: (N,) 이전 누적 dose.
        radial_compression: (N,) 현 radial 압축 [N].
        f_safe: 형상파괴 시작 임계 [N].
        dt: 시간 스텝(누적 스케일).
        q: 초과분 지수.
    Returns:
        (N,) 갱신된 누적 dose(≥ prev).
    """
    over = (radial_compression - f_safe).clamp(min=0.0) / max(f_safe, 1e-6)
    return prev_dose + dt * over.pow(q)
