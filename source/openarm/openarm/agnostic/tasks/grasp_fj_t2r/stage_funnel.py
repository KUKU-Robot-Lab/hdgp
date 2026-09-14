"""에피소드 단계 퍼널(접근 → 파지 → 인벨롭 → 리프트 → 성공) — t2r 루프 틱의 판정·관찰 재료.

★09.14 사용자: "컵에 접근, 파지, 리프트가 잘 되는지 틱을 확인하면서 t2r feedback reward design 구조 활성화".
보상·관측에는 쓰지 않는다(sim2real 무관 — 로그 전용).

왜 에피소드 단위인가: 스텝 평균은 접근 중인 스텝이 섞여 "못 감"과 "지나침"을 뭉갠다. 에피소드마다 한 번이라도
그 단계에 닿았는지를 래치로 잡고, 에피소드가 끝날 때 이벤트 EMA 로 적는다(env `_reset_idx`).
순수 torch — Isaac 없이 테스트한다.
"""

from __future__ import annotations

import torch

#: 단계 이름 = 로그 `stage/<이름>_ep` · 퍼널 순서. 루프 도구(`t2r_fj_round.STAGE_NAMES`)가 같은 순서를 쓴다.
STAGES = ("reach", "grasp", "envelope", "lift", "success")
#: 접근 = 손바닥 원점이 컵 파지 띠(반경 R · 축 방향 ±H 원통 옆면)에서 이 거리 안 [m]
REACH_GAP_M = 0.05
#: 파지 = 컵에 닿은(마디 하나라도) 손가락 수가 이 이상
GRASP_MIN_FINGERS = 3


def palm_band_gap(palm_pos: torch.Tensor, cup_pos: torch.Tensor, cup_axis: torch.Tensor,
                  cup_radius: torch.Tensor, cup_half_height: torch.Tensor) -> torch.Tensor:
    """손바닥 원점 → 컵 파지 띠까지 거리 (N,) [m]. 띠 옆면 안쪽이면 0."""
    v = palm_pos - cup_pos
    h = (v * cup_axis).sum(-1)
    r = (v - h.unsqueeze(-1) * cup_axis).norm(dim=-1)
    return torch.sqrt(torch.relu(r - cup_radius) ** 2 + torch.relu(h.abs() - cup_half_height) ** 2)


def step_flags(gap: torch.Tensor, link_touch: torch.Tensor, palm_touch: torch.Tensor,
               lifted: torch.Tensor, num_successes: torch.Tensor) -> torch.Tensor:
    """이 스텝에 각 단계에 있나 (N, len(STAGES)) bool.

    gap (N,) · link_touch (N,F,L) bool · palm_touch (N,) bool · lifted (N,) bool · num_successes (N,).
    단계끼리 포함 관계를 강제하지 않는다(파지 없이 밀어 올려도 lift 는 선다) — 틱에서 그 차이가 보여야 한다.
    """
    fingers = link_touch.any(dim=2)
    return torch.stack([gap <= REACH_GAP_M,
                        fingers.sum(dim=1) >= GRASP_MIN_FINGERS,
                        palm_touch & fingers.all(dim=1),
                        lifted,
                        num_successes > 0], dim=1)
