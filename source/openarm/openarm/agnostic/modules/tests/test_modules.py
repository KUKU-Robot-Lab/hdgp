"""adr / agents / object_bank 단위 테스트 (Isaac 불필요).

physics_dr 은 isaaclab 을 import 하므로 여기서 다루지 않는다 —
그 계약은 env 부팅 시 fail-loud 로 확인한다.
"""

from __future__ import annotations

import pytest

from openarm.agnostic.modules import adr as A
from openarm.agnostic.modules import agents as AG
from openarm.agnostic.modules import object_bank as OB
from openarm.agnostic.modules import robots as R


# =============================================================================
# TaskADR
# =============================================================================
CFG = {"spawn": {"xy_range": (0.02, 0.10)}}


def test_disabled_adr_is_identical_to_fixed_initial():
    """스위치를 끄면 커리큘럼 없는 고정 세팅과 **정확히 동치**여야 한다."""
    a = A.TaskADR(CFG, enabled=False, increment_interval=0, trigger_threshold=0.0)
    for _ in range(10_000):
        assert a.maybe_increment(1.0) is False
    assert a.get_param("spawn", "xy_range") == pytest.approx(0.02)
    assert a.progress == 0.0


def test_increment_requires_both_interval_and_threshold():
    a = A.TaskADR(CFG, num_increments=4, increment_interval=3, trigger_threshold=0.4)
    # 지표는 충분하지만 interval 미달 → 증분 없음
    assert [a.maybe_increment(0.9) for _ in range(3)] == [False, False, False]
    assert a.maybe_increment(0.9) is True          # interval 충족
    # interval 은 충족돼도 지표가 낮으면 증분 없음
    for _ in range(10):
        assert a.maybe_increment(0.1) is False
    assert a.increment_counter == 1


def test_linear_interpolation_and_cap():
    a = A.TaskADR(CFG, num_increments=4, increment_interval=0, trigger_threshold=0.0)
    assert a.get_param("spawn", "xy_range") == pytest.approx(0.02)
    for _ in range(4):
        a.maybe_increment(1.0)
    assert a.increment_counter == 4
    assert a.progress == pytest.approx(1.0)
    assert a.get_param("spawn", "xy_range") == pytest.approx(0.10)
    # 만렙 이후로는 더 오르지 않는다
    for _ in range(100):
        assert a.maybe_increment(1.0) is False
    assert a.get_param("spawn", "xy_range") == pytest.approx(0.10)


def test_set_increment_clamps():
    a = A.TaskADR(CFG, num_increments=5)
    a.set_increment(99)
    assert a.increment_counter == 5
    a.set_increment(-3)
    assert a.increment_counter == 0


def test_unknown_param_raises_instead_of_silent_default():
    a = A.TaskADR(CFG)
    with pytest.raises(KeyError):
        a.get_param("spawn", "typo")
    with pytest.raises(KeyError):
        a.get_param("nope", "xy_range")


def test_log_dict_keys():
    a = A.TaskADR(CFG)
    assert set(a.log_dict()) == {"adr/increment", "adr/progress"}


# =============================================================================
# agents
# =============================================================================
def test_agent_cfg_defaults_to_mlp():
    p = R.get("bis_right")
    assert AG.resolve_agent_cfg(p) == AG.MLP_CFG
    assert AG.resolve_agent_cfg(p, use_lstm=True) == AG.LSTM_CFG


def test_profile_override_wins():
    import dataclasses

    p = dataclasses.replace(R.get("bis_right"), agent_cfg_name="custom.yaml")
    assert AG.resolve_agent_cfg(p) == "custom.yaml"
    assert AG.resolve_agent_cfg(p, use_lstm=True) == "custom.yaml"


# =============================================================================
# object_bank
# =============================================================================
def test_banks_resolve_and_files_exist():
    for name in OB.BANKS:
        bank = OB.get(name)
        assert len(bank) > 0, f"{name} 이 비었다"
        assert not bank.missing_files()


def test_single_cup_does_not_need_multi_asset():
    """Phase A 는 replicate_physics 를 켠 채로 돌 수 있어야 한다(성능)."""
    b = OB.get("single_cup")
    assert len(b) == 1
    assert b.needs_multi_asset is False
    assert b.requires_replicate_physics_off is False


def test_cup_family_matches_grasp_v1_ids():
    """순서가 env_id % 8 배정과 onehot 인덱스를 동시에 정한다 — 바꾸면 안 된다."""
    assert OB.get("cup_family").ids == (
        "cup_big_s085", "cup_big_s100", "cup_big_s115", "cup_big_s130",
        "shaker_closed", "cup_big_s090", "cup_big_s105", "cup_big_s120",
    )


def test_assign_is_deterministic_env_id_modulo():
    b = OB.get("cup_family")
    assert b.assign_indices(20) == [i % 8 for i in range(20)]


def test_expected_size_guard_catches_asset_drift():
    """glob 뱅크에 자산이 하나 추가되면 onehot 차원이 조용히 바뀐다 — 막는다."""
    n = len(OB.get("visdex"))
    OB.get("visdex", expected_size=n)              # 일치하면 통과
    with pytest.raises(RuntimeError):
        OB.get("visdex", expected_size=n + 1)


def test_spawn_order_guard():
    multi = OB.get("cup_family")
    single = OB.get("single_cup")
    with pytest.raises(RuntimeError):
        OB.assert_spawned_after_clone(multi, cloned=False)
    OB.assert_spawned_after_clone(multi, cloned=True)      # 예외 없음
    OB.assert_spawned_after_clone(single, cloned=False)    # 단일은 무관


def test_unknown_bank_raises():
    with pytest.raises(KeyError):
        OB.get("nope")


def test_base_mass_is_uniform_across_specs():
    """USD 기본질량이 제각각이면 같은 scale DR 에도 절대 질량이 벌어진다."""
    for name in ("single_cup", "cup_family"):
        masses = {s.mass for s in OB.get(name).specs}
        assert masses == {OB.BASE_OBJECT_MASS}, f"{name}: {masses}"
