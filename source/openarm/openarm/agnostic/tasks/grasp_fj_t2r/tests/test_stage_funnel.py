"""에피소드 단계 퍼널 정의 — 순수 torch (Isaac 불요).

잠그는 것: 접근 = 손바닥↔컵 파지 띠 간극 ≤ 5 cm · 파지 = 닿은 손가락 ≥ 3 · 인벨롭 = 손바닥 + 5손가락 동시 ·
리프트 = env 래치 · 성공 = 이번 에피소드 성공 ≥ 1. 단계끼리 포함 관계는 강제하지 않는다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

import torch

from openarm.agnostic.tasks.grasp_fj_t2r import stage_funnel as SF


def test_palm_band_gap_is_the_distance_to_the_graspable_band_surface():
    n = 4
    cup = torch.zeros(n, 3)
    axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3)
    R, H = torch.full((n,), 0.05), torch.full((n,), 0.05)
    palm = torch.tensor([[0.08, 0.0, 0.0], [0.05, 0.0, 0.0], [0.38, 0.0, 0.0], [0.05, 0.0, 0.09]])
    gap = SF.palm_band_gap(palm, cup, axis, R, H)
    assert torch.allclose(gap, torch.tensor([0.03, 0.0, 0.33, 0.04]), atol=1e-6)


def test_step_flags_follow_the_funnel_definitions():
    n = 4
    gap = torch.tensor([0.02, 0.20, 0.0, 0.0])
    touch = torch.zeros(n, 5, 3, dtype=torch.bool)
    touch[1, :3, 2] = True           # 손끝 셋만 — 손바닥은 멀다 → 접근 없이 파지
    touch[2, :, 0] = True            # 다섯 손가락 전부
    palm = torch.tensor([False, False, True, False])
    lifted = torch.tensor([False, False, True, True])
    n_succ = torch.tensor([0.0, 0.0, 2.0, 0.0])
    f = SF.step_flags(gap, touch, palm, lifted, n_succ)
    assert f.shape == (n, len(SF.STAGES)) and f.dtype == torch.bool
    col = {s: f[:, k].tolist() for k, s in enumerate(SF.STAGES)}
    assert col["reach"] == [True, False, True, True]
    assert col["grasp"] == [False, True, True, False]
    assert col["envelope"] == [False, False, True, False]
    assert col["lift"] == [False, False, True, True]      # 4번: 파지 없이 들림 — 포함 관계를 강제하지 않는다
    assert col["success"] == [False, False, True, False]
