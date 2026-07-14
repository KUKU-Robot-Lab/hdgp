"""per-finger 손 제어(grasp_v1 방식) 계약.

배경 — 왜 PCA 를 버리는가:
  현행 시너지 경로는 PCA basis(Allegro 리타겟)로 20관절을 커플링한다.
  finger_action_utils.py:61 이 스스로 적어둔 한계가 있다 —
  "PC1 하나가 20관절을 커플링 → 2지 최소해가 action 공간에서 표현 불가".

  LEFT 정책의 성공 파지(89119 샘플, 평균 리프트 18.2cm)를 추출해 보니
      thumb  +0.011 +1.561 +0.280 -1.142
      index  +0.001 +0.541 +0.834 +1.567
      pinky  -0.002 -0.000 +0.677 +1.451     ← pinky_2 만 전혀 안 굽는다
  손가락마다 다른 굽힘이 필요한데, PCA 로는 표현할 수 없어 정책이 5축을 전부
  극단값(-1/+1)으로 밀어 억지 근사하고 있었다: [-1.000, +0.979, -0.997, -0.999, +0.975].

  per-finger 는 손가락별 독립 계수라 형상 적응(envelope)이 가능하다.
  물체에 닿거나 한계에 포화된 관절은 멈추고, 나머지만 계속 감긴다.
"""

from __future__ import annotations

import torch

from openarm.tesollo.left.grasp_v2.finger_action_utils import (
    compute_per_finger_progress_targets,
    compute_synergy_progress_targets,
)


def test_per_finger_is_independent_across_fingers():
    """한 손가락의 action 을 바꿔도 다른 손가락의 목표는 그대로여야 한다.

    PCA 가 못 하는 바로 그것이다 — 커플링 없음.
    """
    a = torch.zeros(1, 5)
    p0 = compute_per_finger_progress_targets(a)

    a_idx = a.clone()
    a_idx[0, 1] = 1.0                       # index 만 완전 폐쇄
    p1 = compute_per_finger_progress_targets(a_idx)

    assert torch.allclose(p1[0, 4:8], torch.ones(4)), "index 4관절이 전부 닫혀야 한다"
    # 나머지 손가락은 미동도 없어야 한다
    untouched = torch.cat([p1[0, 0:4], p1[0, 8:20]])
    assert torch.allclose(untouched, torch.cat([p0[0, 0:4], p0[0, 8:20]]))


def test_action_maps_to_progress_zero_to_one():
    """action -1 → 완전 개방(0), +1 → 완전 폐쇄(1). 손가락 내 4관절은 같은 값."""
    p_open = compute_per_finger_progress_targets(torch.full((1, 5), -1.0))
    p_shut = compute_per_finger_progress_targets(torch.full((1, 5), +1.0))

    assert torch.allclose(p_open, torch.zeros(1, 20))
    assert torch.allclose(p_shut, torch.ones(1, 20))

    mid = compute_per_finger_progress_targets(torch.zeros(1, 5))
    assert torch.allclose(mid, torch.full((1, 20), 0.5))


def test_pinky_only_grasp_is_representable():
    """'pinky 만 빼고 닫기' — LEFT 성공 자세가 요구하는 형태. PCA 로는 불가능했다."""
    a = torch.full((1, 5), 1.0)
    a[0, 4] = -1.0                          # pinky 만 개방
    p = compute_per_finger_progress_targets(a)

    assert torch.allclose(p[0, 0:16], torch.ones(16)), "thumb~ring 은 닫힌다"
    assert torch.allclose(p[0, 16:20], torch.zeros(4)), "pinky 만 열려 있다"


def test_synergy_cannot_represent_pinky_only_grasp():
    """대조군 — 같은 요구를 PCA 로는 낼 수 없음을 명시적으로 남긴다.

    현행 basis 는 pinky_1 열이 5축 전부 0.0 이라(Allegro 는 4지라 대응 관절이 없다)
    어떤 계수를 줘도 pinky 를 독립적으로 다룰 수 없다.
    """
    from openarm.tesollo.left.grasp_v2.tesollo_hand_synergy import HAND_SYNERGY_BASIS

    basis = torch.tensor(HAND_SYNERGY_BASIS)
    pinky_1_col = basis[:, 16]              # finger-major: pinky 는 16..19, _1 은 16
    assert torch.allclose(pinky_1_col, torch.zeros(5)), (
        "현행 basis 의 pinky_1 열은 전부 0 — Allegro(4지) 리타겟의 흔적"
    )


def test_progress_is_clamped_to_unit_interval():
    """action 이 범위를 벗어나도 progress 는 [0,1] 안에 있어야 한다."""
    p = compute_per_finger_progress_targets(torch.tensor([[-5.0, 5.0, 0.0, -1.0, 1.0]]))
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert torch.allclose(p[0, 0:4], torch.zeros(4))
    assert torch.allclose(p[0, 4:8], torch.ones(4))


def test_shape_and_dtype_match_synergy_path():
    """env 가 두 경로를 교체 가능하게 쓰므로 출력 규격이 같아야 한다."""
    from openarm.tesollo.left.grasp_v2.tesollo_hand_synergy import (
        HAND_SYNERGY_ANCHOR,
        HAND_SYNERGY_BASIS,
        HAND_SYNERGY_COEFF_MAXS,
        HAND_SYNERGY_COEFF_MINS,
    )
    from openarm.tesollo.left.grasp_v2.grasp_left_preset import (
        HAND_APPROACH_POSE,
        HAND_FULL_GRIP_POSE,
    )

    a = torch.rand(7, 5) * 2.0 - 1.0
    p_pf = compute_per_finger_progress_targets(a)
    p_syn = compute_synergy_progress_targets(
        a,
        torch.tensor(HAND_SYNERGY_BASIS),
        torch.tensor(HAND_SYNERGY_ANCHOR),
        torch.tensor(HAND_SYNERGY_COEFF_MINS),
        torch.tensor(HAND_SYNERGY_COEFF_MAXS),
        torch.tensor(HAND_APPROACH_POSE),
        torch.tensor(HAND_FULL_GRIP_POSE),
    )
    assert p_pf.shape == p_syn.shape == (7, 20)
    assert p_pf.dtype == p_syn.dtype
