"""Unit tests for bead-count curriculum (DexPour-style difficulty progression).

설계: 물리 비드는 startup에 max개(=schedule 마지막) 고정 생성됨. 커리큘럼은
"활성 개수 N"을 success 기반으로 1→5→8→10→20→30 으로 올린다. 비활성 비드는
hide(z=-10) 되고, 모든 bead fraction은 활성 N으로 정규화(=앞 N개 슬라이스)된다.
"""

import pytest

from openarm.tesollo.right.pour_v5.bead_curriculum import BeadCountCurriculum


def test_starts_at_first_stage():
    c = BeadCountCurriculum(schedule=(1, 5, 8, 10, 20, 30))
    assert c.current_count == 1
    assert c.max_count == 30
    assert not c.is_final


def test_advance_on_sustained_success():
    c = BeadCountCurriculum(schedule=(1, 5, 8), success_threshold=0.5, min_updates_per_stage=3)
    # min_updates 미충족 동안은 성공해도 advance 안 함
    assert c.update(0.9) is False  # update 1
    assert c.update(0.9) is False  # update 2
    assert c.current_count == 1
    advanced = c.update(0.9)        # update 3 → 임계+최소 충족
    assert advanced is True
    assert c.current_count == 5


def test_no_advance_below_threshold():
    c = BeadCountCurriculum(schedule=(1, 5), success_threshold=0.5, min_updates_per_stage=1)
    for _ in range(10):
        assert c.update(0.3) is False
    assert c.current_count == 1


def test_no_advance_before_min_updates():
    c = BeadCountCurriculum(schedule=(1, 5), success_threshold=0.5, min_updates_per_stage=5)
    for _ in range(4):
        assert c.update(1.0) is False
    assert c.current_count == 1
    assert c.update(1.0) is True
    assert c.current_count == 5


def test_caps_at_final_stage():
    c = BeadCountCurriculum(schedule=(1, 5), success_threshold=0.5, min_updates_per_stage=1)
    assert c.update(1.0) is True   # → 5 (final)
    assert c.is_final
    # final에서는 더 advance 안 함
    assert c.update(1.0) is False
    assert c.current_count == 5


def test_min_updates_reset_between_stages():
    c = BeadCountCurriculum(schedule=(1, 5, 8), success_threshold=0.5, min_updates_per_stage=2)
    assert c.update(1.0) is False  # stage0 update1
    assert c.update(1.0) is True   # stage0 update2 → advance to 5
    # stage1 카운터 리셋: 다시 min_updates 충족해야 함
    assert c.update(1.0) is False  # stage1 update1
    assert c.update(1.0) is True   # stage1 update2 → advance to 8
    assert c.current_count == 8


def test_active_slice_count_never_exceeds_max():
    c = BeadCountCurriculum(schedule=(1, 5, 8, 10, 20, 30))
    for _ in range(50):
        c.update(1.0)
    assert c.current_count == 30
    assert c.current_count <= c.max_count


def test_state_dict_roundtrip():
    """체크포인트 재개 시 stage 보존."""
    c = BeadCountCurriculum(schedule=(1, 5, 8), success_threshold=0.5, min_updates_per_stage=1)
    c.update(1.0)  # → stage1 (5)
    state = c.state_dict()
    c2 = BeadCountCurriculum(schedule=(1, 5, 8), success_threshold=0.5, min_updates_per_stage=1)
    c2.load_state_dict(state)
    assert c2.current_count == 5
