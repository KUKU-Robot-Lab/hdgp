"""BimanualPair 합성 계약.

프로필 데이터(modules/robot_profiles)가 바뀌어도 양팔 합성이 조용히 틀어지지 않게 pin 한다.
Isaac 불요 — bimanual/robot_profiles 는 순수 데이터다.
"""
import re

import pytest

from openarm.agnostic.modules import robot_profiles as _rp
from openarm.agnostic.tasks.pour_fabric import bimanual as _bm


def test_pairs_registered_and_skips_are_loud():
    assert "short" in _bm.PAIRS
    assert _bm.DEFAULT_PAIR in _bm.PAIRS
    # 한쪽 프로필뿐인 자산(dg5f-m 우측만 등)은 조용히 빠지지 않고 사유가 남아야 한다.
    assert _bm.SKIPPED, "한쪽뿐인 자산이 하나도 없다 — 레지스트리가 바뀌었나"
    for short, why in _bm.SKIPPED.items():
        assert why, short


def test_gym_slot_names_have_no_hyphen():
    """train.py run_naming 정규식 `open-\\w+` — 하이픈이 들어가면 로그가 오분류된다."""
    for short in _bm.PAIRS:
        assert re.fullmatch(r"\w+", short), short


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_pair_same_asset_opposite_side(short):
    p = _bm.PAIRS[short]
    assert p.source.usd_relpath == p.receiver.usd_relpath
    assert {_bm.side_of(p.source), _bm.side_of(p.receiver)} == {"r", "l"}
    assert _bm.side_of(p.source) == "r", "source=우(붓기) 규약"


def test_pair_validation_rejects_same_side():
    p = _bm.get_pair("short")
    with pytest.raises(ValueError):
        _bm.BimanualPair(short="x", source=p.source, receiver=p.source)


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_init_covers_both_sides_plus_head(short):
    pair = _bm.PAIRS[short]
    init = pair.init_joint_pos
    for prof in (pair.source, pair.receiver):
        pat = re.compile(f"^({prof.arm_joint_regex}|{prof.hand_joint_regex})$")
        n = sum(1 for k in init if pat.match(k))
        assert n == prof.num_arm_joints + prof.num_hand_joints, (prof.name, n)
    assert "head_j_pan" in init and "head_j_tilt" in init


def test_init_uses_active_home_not_idle():
    """양팔 init 은 각 프로필의 **자기 쪽 활성 홈**이어야 한다.

    프로필의 init_joint_pos 에는 반대팔 유휴 자세가 섞여 있다 — 합성이 receiver 팔에
    source 프로필의 유휴 값을 넣으면(=버그) 두 값이 일치해 버린다. 좌 short 프로필의
    활성 좌팔 홈은 우팔 홈의 정확한 부호 미러이고, 우 프로필의 유휴 좌팔은 다른 자세다.
    """
    pair = _bm.get_pair("short")
    init = pair.init_joint_pos
    rcv = pair.receiver
    pat = re.compile(f"^{rcv.arm_joint_regex}$")
    rcv_arm = {k: v for k, v in init.items() if pat.match(k)}
    own_home = {k: v for k, v in rcv.init_joint_pos.items() if pat.match(k)}
    idle_from_src = {k: v for k, v in pair.source.init_joint_pos.items() if pat.match(k)}
    assert rcv_arm == own_home
    assert rcv_arm != idle_from_src, "receiver 팔이 source 의 유휴 자세다 — 합성 버그"


def test_receiver_home_is_mirror_of_source_home():
    pair = _bm.get_pair("short")
    init = pair.init_joint_pos
    sign = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
    for i, s in enumerate(sign, start=1):
        assert init[f"l_aj_{i}"] == pytest.approx(s * init[f"r_aj_{i}"])
    # 엄지 대향은 좌우 부호가 반대(자산 한계 r [-2.7,0] / l [0,2.7])
    assert init["r_hj_thumb_2"] == pytest.approx(-init["l_hj_thumb_2"])


@pytest.mark.parametrize("short", sorted(_bm.PAIRS))
def test_actuator_specs_cover_every_init_joint(short):
    pair = _bm.PAIRS[short]
    specs = pair.actuator_specs
    pats = [re.compile(f"^{e}$") for s in specs.values() for e in s["joint_names_expr"]]
    for jn in pair.init_joint_pos:
        assert any(p.match(jn) for p in pats), f"{jn} 에 actuator 가 없다"


def test_active_side_gains_come_from_own_profile():
    """src_* 그룹은 우측 관절만, rcv_* 그룹은 좌측 관절만 몰아야 한다(유휴 게인 혼입 금지)."""
    pair = _bm.get_pair("short")
    for name, spec in pair.actuator_specs.items():
        exprs = spec["joint_names_expr"]
        if name.startswith("src_"):
            assert all(e.startswith("r_") for e in exprs), (name, exprs)
        elif name.startswith("rcv_"):
            assert all(e.startswith("l_") for e in exprs), (name, exprs)
        else:
            assert name == "head", name


def test_head_actuator_once():
    pair = _bm.get_pair("short")
    heads = [k for k in pair.actuator_specs if "head" in k]
    assert heads == ["head"]


def test_pairs_only_from_registry_profiles():
    names = {p.name for p in _rp.PROFILES.values()}
    for pair in _bm.PAIRS.values():
        assert pair.source.name in names and pair.receiver.name in names
