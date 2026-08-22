"""BimanualPair 합성 계약.

프로필 데이터(modules/robots)가 바뀌어도 양팔 합성이 조용히 틀어지지 않게 pin 한다.
Isaac 불요 — bimanual/robots 는 순수 데이터다.
"""
import re

import pytest

from openarm.agnostic.modules import robots as _rb
from openarm.agnostic.tasks.pour_fabric import bimanual as _bm


def test_pairs_registered_and_skips_are_loud():
    assert "bis" in _bm.PAIRS
    # rh56 좌팔은 fabric 이 없다 — 조용히 빠지지 않고 사유가 남아야 한다.
    assert "rh56" in _bm.SKIPPED
    assert "fabric" in _bm.SKIPPED["rh56"] or "Fabrics" in _bm.SKIPPED["rh56"]


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_pair_same_asset_opposite_side(short):
    p = _bm.PAIRS[short]
    assert p.source.asset is p.receiver.asset
    assert {p.source.side, p.receiver.side} == {"r", "l"}


def test_pair_validation_rejects_same_side():
    p = _bm.get_pair("bis")
    with pytest.raises(ValueError):
        _bm.BimanualPair(source=p.source, receiver=p.source)


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_init_covers_both_sides_plus_head(short):
    pair = _bm.PAIRS[short]
    init = pair.init_joint_pos
    for prof in (pair.source, pair.receiver):
        pat = re.compile(f"^({prof.arm_joint_regex}|{prof.hand_joint_regex})$")
        n = sum(1 for k in init if pat.match(k))
        assert n == prof.num_arm_joints + prof.num_hand_joints, (prof.name, n)


def test_init_uses_active_home_not_tuck():
    """양팔 init 은 각 프로필의 **자기 쪽 활성 홈**이어야 한다.

    프로필의 init_joint_pos 에는 반대팔 tuck 이 섞여 있다 — 합성이 receiver 팔에
    source 프로필의 tuck 값을 넣으면(=버그) 두 값이 일치해 버린다. 실제로
    tuck 과 홈은 다른 자세이므로 불일치를 확인한다.
    """
    pair = _bm.get_pair("bis")
    init = pair.init_joint_pos
    rcv = pair.receiver
    pat = re.compile(f"^{rcv.arm_joint_regex}$")
    rcv_arm = {k: v for k, v in init.items() if pat.match(k)}
    own_home = {k: v for k, v in rcv.init_joint_pos.items() if pat.match(k)}
    tuck_from_src = {k: v for k, v in pair.source.init_joint_pos.items() if pat.match(k)}
    assert rcv_arm == own_home
    assert rcv_arm != tuck_from_src, "receiver 팔이 tuck 자세다 — 합성 버그"


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_actuator_specs_cover_every_init_joint(short):
    """actuator 커버리지 누락 관절은 조용히 무구동 자유회전한다(adf0b24 교훈)."""
    pair = _bm.PAIRS[short]
    specs = pair.actuator_specs
    assert not any(g.startswith("idle_") for g in specs), "idle 게인이 섞였다"
    pats = [re.compile(f"^{e}$") for spec in specs.values()
            for e in spec["joint_names_expr"]]
    uncovered = [j for j in pair.init_joint_pos
                 if not any(p.match(j) for p in pats)]
    assert not uncovered, f"actuator 미커버 관절: {uncovered}"


def test_head_actuator_once():
    specs = _bm.get_pair("bis").actuator_specs
    assert sum(1 for g in specs if g == "head") == 1


def test_pairs_only_from_registry_profiles():
    for pair in _bm.PAIRS.values():
        assert pair.source is _rb.PROFILES[pair.source.name]
        assert pair.receiver is _rb.PROFILES[pair.receiver.name]
