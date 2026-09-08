"""오른팔 전용 USD 오버레이 계약 — Isaac 없이 pxr 만으로 돈다.

`replicate_physics=False`(다물체 필수)는 env 마다 articulation 을 통째로 PhysX 가 파싱한다.
우리는 오른팔만 학습하는데 왼팔 39링크·39관절이 env 마다 파싱됐다. 16,384 env 실측에서
PhysX 가 요구한 접촉이 24,777,720(env 당 1,512)이었고 그게 기동 실패의 직접 원인이다.

오버레이는 base USD 를 subLayer 로 물고 왼팔 subtree 를 `active=false` 로 끈다.
메시를 다시 만들지 않는다. 2,048 env 실측: 처리량 24,155 → 39,448 fps(+63%),
부팅+3epoch 101 → 59초, 그리고 **물리 지표는 비트 단위로 동일**했다.
"""

import pathlib

import pytest

pxr = pytest.importorskip("pxr", reason="pxr(USD) 없음")
from pxr import Usd, UsdPhysics  # noqa: E402

_ROBOT = pathlib.Path(__file__).resolve().parents[6] / "assets" / "robot"

#: 자산 → (오른쪽 링크 수, 오버레이 후 가동관절, base 왼쪽 링크 수, base 가동관절, 프로필 쌍)
#: ★자산이 늘면 여기 한 줄만 추가한다 — 오버레이는 `scripts/tools/make_right_only_overlay.py`
#:   가 manifest 에서 생성하므로 이름 목록을 손으로 적지 않는다.
_CASES = {
    "openarm_dg5f-m_bi_rl": dict(right_links=39, right_mov=29, left_links=39, base_mov=56,
                                 profiles=("tesollo_right", "tesollo_right_only"), init_keys=29),
    "openarm_rh56f1_bi_rl": dict(right_links=35, right_mov=21, left_links=35, base_mov=40,
                                 profiles=("rh56f1_right", "rh56f1_right_only"), init_keys=21),
}


def _base(asset):
    return _ROBOT / asset / f"{asset}.usd"


def _right(asset):
    return _ROBOT / asset / f"{asset}_right.usda"


def _stage(p):
    if not p.is_file():
        pytest.skip(f"자산 없음: {p}")
    return Usd.Stage.Open(str(p))


def _parts(path):
    st = _stage(path)
    root = st.GetDefaultPrim()
    links = [p.GetName() for p in root.GetChildren()
             if p.GetName().startswith(("l_al_", "l_hl_", "r_al_", "r_hl_"))]
    joints = list(st.GetPrimAtPath(root.GetPath().AppendChild("joints")).GetChildren())
    mov = [j for j in joints if "Revolute" in str(j.GetTypeName()) or "Prismatic" in str(j.GetTypeName())]
    return st, links, joints, mov


@pytest.mark.parametrize("asset", sorted(_CASES))
def test_overlay_removes_every_left_link_and_joint(asset):
    exp = _CASES[asset]
    _, links, _, mov = _parts(_right(asset))
    assert [n for n in links if n.startswith(("l_al_", "l_hl_"))] == []
    assert len([n for n in links if n.startswith(("r_al_", "r_hl_"))]) == exp["right_links"], (
        "오른쪽이 줄면 안 된다")
    assert len(mov) == exp["right_mov"], f"{asset}: 가동관절 {exp['right_mov']} 여야 한다, got {len(mov)}"


@pytest.mark.parametrize("asset", sorted(_CASES))
def test_overlay_leaves_no_dangling_joint_reference(asset):
    """★링크만 끄고 관절을 남기면 PhysX 가 없는 body 를 참조해 죽는다."""
    st, _, joints, _ = _parts(_right(asset))
    bad = []
    for j in joints:
        api = UsdPhysics.Joint(j)
        for rel in (api.GetBody0Rel(), api.GetBody1Rel()):
            for t in rel.GetTargets():
                p = st.GetPrimAtPath(t)
                if not p.IsValid() or not p.IsActive():
                    bad.append((j.GetName(), str(t)))
    assert bad == [], f"끊긴 관절 참조: {bad[:5]}"


@pytest.mark.parametrize("asset", sorted(_CASES))
def test_base_asset_is_untouched(asset):
    """오버레이는 base 를 수정하지 않는다 — 양팔 트랙(grasp_s2r 등)이 그대로 써야 한다."""
    exp = _CASES[asset]
    _, links, _, mov = _parts(_base(asset))
    assert len([n for n in links if n.startswith(("l_al_", "l_hl_"))]) == exp["left_links"]
    assert len(mov) == exp["base_mov"]


@pytest.mark.parametrize("asset", sorted(_CASES))
def test_overlay_sublayers_the_base_rather_than_copying(asset):
    """복사본이면 base 캘리브 갱신이 조용히 안 실린다 — subLayer 여야 한다."""
    st = _stage(_right(asset))
    subs = [str(x) for x in st.GetRootLayer().subLayerPaths]
    assert any(_base(asset).name in x for x in subs), subs


@pytest.mark.parametrize("asset", sorted(_CASES))
def test_profile_drops_left_entries_from_cfg_dicts(asset):
    """자산에서만 빼면 IsaacLab 이 `Not all regular expressions are matched!` 로 죽는다(실측)."""
    from openarm.agnostic.tasks.grasp_kp.robot_profiles import PROFILES
    exp = _CASES[asset]
    full, right = PROFILES[exp["profiles"][0]], PROFILES[exp["profiles"][1]]
    assert [k for k in right.init_joint_pos if k.startswith("l_")] == []
    assert len(right.init_joint_pos) == exp["init_keys"]
    assert [n for n, spec in right.actuator_specs.items()
            if any(str(e).startswith("l_") for e in spec["joint_names_expr"])] == []
    for spec in right.actuator_specs.values():
        assert not any(str(e).startswith("l_") for e in spec.get("joint_names_expr", []))
    # 오른쪽 계약은 원본과 동일해야 한다 — 게인·정규식·palm 박스를 건드리지 않았다.
    for f in ("arm_joint_regex", "hand_joint_regex", "num_arm_joints", "num_hand_joints"):
        assert getattr(right, f) == getattr(full, f), f
    assert right.actuator_specs.keys() <= full.actuator_specs.keys()
    for k in right.actuator_specs:
        assert right.actuator_specs[k] == full.actuator_specs[k], k
