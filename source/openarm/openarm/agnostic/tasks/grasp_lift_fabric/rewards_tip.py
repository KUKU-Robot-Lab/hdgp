"""grasp_lift_fabric 진단 유틸 — `envelope_fraction_graded` 만 존치.

구 사다리 보상(compute_tip_rewards)은 08.26 재설계에서 삭제됐다(계층 보상으로
대체·상시화). 이 함수는 감쌈 진단 로깅(`task/envelope_frac_raw`)이 소비한다.
"""

from __future__ import annotations

import torch


def envelope_fraction_graded(
    mid_force: torch.Tensor,   # (N, E) envelope 손가락별 중간마디(_3) 접촉력
    dist_force: torch.Tensor,  # (N, E) envelope 손가락별 원위마디(_4) 접촉력
    threshold: float,
) -> torch.Tensor:
    """감싼 **마디** 비율: mean_f 0.5·(1[mid] + 1[dist]) ∈ [0,1].

    공유 `envelope_fraction` 은 손가락별 OR 이라 **_3 만 닿아도 그 손가락을 1 로 센다**
    (사용자 지적). 받치기와 감쌈이 같은 점수가 되고, 실측으로 같은 정책이 느슨(OR)
    0.50 · 엄격(전 마디 AND) 0.069 로 7 배 벌어진다.

    AND 로 가지 않는 이유: 손가락마다 닿는 마디가 다르다(grasp_v1 실측 — 정책에 따라
    엄지가 tip 0.907/_4 0.249 이거나 tip 0.183/_4 0.808). AND 는 유효한 파지도 0 으로
    세고 0.069 대역이라 초기 gradient 가 없다. 부분 점수는 0 → 0.5 → 1.0 사다리를
    만들어 받치기와 감쌈을 구분하면서 gradient 를 남긴다.

    rim-hook 반증(구 v2 공식 `0.5·(mid평균+dist평균)` 이 0.56 을 준 건)은 **접촉 깊이**를
    쟀기 때문이다. 이 함수는 이진 접촉의 평균이라 깊이로 올릴 수 없다:
    검지만 두 마디 = 0.2 · 두 손가락 = 0.4 · 5 지 전 마디 = 1.0.
    팁 접촉은 여기 없다 — 핀치(팁만)로는 이 항이 0 이다.
    """
    graded = 0.5 * ((mid_force > threshold).float() + (dist_force > threshold).float())
    return graded.mean(dim=-1)

