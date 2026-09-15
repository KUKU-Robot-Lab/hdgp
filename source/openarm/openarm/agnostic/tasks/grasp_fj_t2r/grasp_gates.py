"""보상 게이트용 에피소드 단계 래치 — 순수 torch (Isaac 불요).

★09.15 사용자: "기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트하면서 조임(이번엔 제외)".
  여섯 라운드 동안 생성 보상이 단계를 건너뛴 자세(엄지만 대기·기울이며 테이블 누르기·손가락 등으로 기대기)에 점수를 줬다.
  env 가 단계 완료를 **에피소드 래치**로 만들어 ctx(`approach_done`·`envelope_done`)로 넘기고, 생성기는 그 순서대로 보상한다.
★로그 퍼널(`stage_funnel.py`)과 정의가 다르다 — 퍼널은 관찰용(접근 = 간극 ≤ 5 cm), 게이트는 보상 순서용이라 더 엄격하다.
★수치는 이 파일 한 곳. `t2r/context.py` 주석이 같은 값을 영어로 적는다(`tests/test_grasp_gates.py` 가 대조).
"""

from __future__ import annotations

import torch

#: 접근 완료 — 손바닥 중심(palm_ee)에서 컵 파지 띠 표면까지 [m]
APPROACH_GAP_M = 0.02
#: 접근 완료 — 손바닥 법선과 "손바닥 → 컵 축" 방향의 cos 하한(≈ 45°)
APPROACH_FACING_MIN = 0.7
#: 접근 완료 — 움직이는 손 관절의 정규화 각(0 = 하한, 1 = 상한)이 기본 자세에서 벗어난 최대치
APPROACH_POSE_TOL = 0.15
#: 인벨롭 완료 — 컵에 닿은 손가락 수(엄지 포함, 손가락마다 마디 하나라도) 하한
ENVELOPE_MIN_DIGITS = 4
#: 인벨롭 완료 — 조건이 연속으로 유지돼야 하는 스텝 수
ENVELOPE_HOLD_STEPS = 5
#: "닿았다"로 셀 컵 접촉력 [N]
TOUCH_N = 0.1


def palm_facing(palm_pos: torch.Tensor, palm_normal: torch.Tensor, cup_pos: torch.Tensor,
                cup_axis: torch.Tensor) -> torch.Tensor:
    """손바닥 법선 · (손바닥 → 컵 축) 단위벡터 (N,). +1 = 손바닥이 컵 축을 똑바로 향한다."""
    v = palm_pos - cup_pos
    radial = v - (v * cup_axis).sum(-1, keepdim=True) * cup_axis
    u = radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return -(palm_normal * u).sum(-1)


def pose_deviation(hand_q_norm: torch.Tensor, default_q_norm: torch.Tensor,
                   movable: torch.Tensor) -> torch.Tensor:
    """움직이는 관절만 본 |정규화 각 − 기본 자세| 최대치 (N,). 잠긴 관절(폭 ≤ 0.05 rad)은 정규화가 노이즈를 키워 뺀다."""
    dev = (hand_q_norm - default_q_norm.unsqueeze(0)).abs()
    return torch.where(movable.unsqueeze(0), dev, torch.zeros_like(dev)).amax(dim=-1)


def update_gates(approach: torch.Tensor, envelope_count: torch.Tensor, envelope: torch.Tensor, *,
                 gap: torch.Tensor, facing: torch.Tensor, pose_dev: torch.Tensor,
                 palm_touch: torch.Tensor, finger_touch: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """이 스텝 상태로 래치를 갱신한 **새** 텐서 (approach (N,) bool, envelope_count (N,) long, envelope (N,) bool).

    finger_touch (N,5) bool — 손가락 0 = 엄지. 인벨롭은 이번 스텝까지 켜진 접근 래치 뒤에만 센다.
    """
    approach_new = approach | ((gap <= APPROACH_GAP_M) & (facing >= APPROACH_FACING_MIN)
                               & (pose_dev <= APPROACH_POSE_TOL))
    holding = (approach_new & palm_touch & finger_touch[:, 0]
               & (finger_touch.sum(dim=1) >= ENVELOPE_MIN_DIGITS))
    count_new = torch.where(holding, envelope_count + 1, torch.zeros_like(envelope_count))
    envelope_new = envelope | (count_new >= ENVELOPE_HOLD_STEPS)
    return approach_new, count_new, envelope_new
