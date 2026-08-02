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


@torch.jit.script
def compute_panel_deformation_deg(panel_joint_pos: torch.Tensor) -> torch.Tensor:
    """Phase 2: segmented-shell deformable cup의 형상파괴 신호 = 최대 |패널 힌지각|(deg).

    각 패널은 base 링에 접선 힌지로 붙어 안쪽으로 눌리면 각이 발생(=변형), 스프링이 복원.
    파지 squeeze는 안쪽 굽힘만 유발하므로 |각| = 안쪽 변형량. **max**(최악 국소 crush)를
    쓰면 기존 힘-proxy `radial_compression`(단일 scalar severity)과 1:1 대체 가능 →
    r_damage/dose/buckle 하류 코드 무변경으로 실제 기하 변형에 연동된다.

    Args:
        panel_joint_pos: (N, num_panels) 패널 힌지각 [rad].
    Returns:
        (N,) 최대 패널 변형 [deg].
    """
    return panel_joint_pos.abs().max(dim=-1).values * (180.0 / 3.141592653589793)


def compute_mass_shift_trigger(
    lift_latched: torch.Tensor,
    height_delta: torch.Tensor,
    height_threshold: float,
    hold_counter: torch.Tensor,
    delay_steps: int,
    already_done: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase 3 '물 추가' 발동 판정 (설계 §5).

    lift latch + 컵 높이 ≥ 임계를 delay_steps 이상 연속 유지하면 1회 발동.
    높이 미달 시 hold_counter 초기화(연속 유지만 인정).

    Args:
        lift_latched: (N,) lift latch 여부.
        height_delta: (N,) 현 리프트 높이 [m].
        height_threshold: 발동 높이 임계 [m].
        hold_counter: (N,) 높이 유지 누적 step (int).
        delay_steps: 발동에 필요한 유지 step 수.
        already_done: (N,) 이미 이번 에피소드에 발동했는지.
    Returns:
        (trigger_mask (N,bool), new_hold_counter (N,)).
    """
    at_height = lift_latched & (height_delta >= height_threshold)
    new_hold = torch.where(at_height, hold_counter + 1, torch.zeros_like(hold_counter))
    trigger = at_height & (~already_done) & (new_hold >= int(delay_steps))
    return trigger, new_hold
