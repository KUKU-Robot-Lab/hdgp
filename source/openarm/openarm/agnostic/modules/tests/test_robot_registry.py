"""로봇 레지스트리 계약 테스트 — **URDF 원본과 직접 대조**한다.

Isaac 앱 없이 돈다. 이 테스트가 통과하면 "이름은 맞다"가 보장된다.
물리·IK 가 맞는지는 여기서 알 수 없다 — 그건 probe 의 몫이고,
`RobotProfile.probe_verified` 로 구분한다(선언 ≠ 검증).

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/modules/tests/ -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openarm.agnostic.modules import robots as R

_HDGP = Path(__file__).resolve().parents[6]          # rl_ws/hdgp
_REPO = _HDGP.parent                                  # rl_ws/
_URDF_DIR = _REPO / "urdf" / "generated" / "rl"
_ASSET_DIR = _HDGP / "assets"
_FABRIC_URDF = (_HDGP / "source" / "FABRICS" / "src" / "fabrics_sim"
                / "models" / "robots" / "urdf")

_ACTUATED = ("revolute", "prismatic", "continuous")


def _limits(body: str) -> tuple | None:
    """<limit lower/upper> — 속성 순서가 파일마다 달라 개별로 잡는다."""
    lo = re.search(r'<limit[^>]*\blower="([^"]*)"', body)
    hi = re.search(r'<limit[^>]*\bupper="([^"]*)"', body)
    if lo is None or hi is None:
        return None
    return float(lo.group(1)), float(hi.group(1))


def _parse_urdf(asset: R.RobotAsset) -> tuple[set[str], set[str], dict]:
    """(구동 관절, 링크, 관절한계) — 정규식 파싱(속성 순서가 파일마다 다르다)."""
    text = (_URDF_DIR / asset.urdf_relpath).read_text()
    joints, limits = set(), {}
    for m in re.finditer(
        r'<joint name="([^"]+)"[^>]*type="([^"]+)"[^>]*>(.*?)</joint>', text, re.S
    ):
        if m.group(2) not in _ACTUATED:
            continue
        joints.add(m.group(1))
        lim = _limits(m.group(3))
        if lim is not None:
            limits[m.group(1)] = lim
    links = {m.group(1) for m in re.finditer(r'<link name="([^"]+)"', text)}
    return joints, links, limits


# URDF 원본(`rl_ws/urdf/generated/rl/`)은 **로컬 전용**이다 — 학습 서버에는 USD 만 있다.
# 이름 계약 검사는 URDF 가 원본이므로, 없으면 조용히 통과시키지 말고 명시적으로 skip 한다.
_HAS_URDF = _URDF_DIR.is_dir()
_SKIP_NO_URDF = pytest.mark.skipif(
    not _HAS_URDF,
    reason=f"URDF 원본 없음({_URDF_DIR}) — 이름 계약 검사는 URDF 가 있는 곳에서만 유효",
)


@pytest.fixture(scope="module")
def urdf_cache() -> dict:
    return {a.name: _parse_urdf(a) for a in R.ASSETS.values()}


ALL_PROFILES = list(R.PROFILES.values())
IDS = [p.name for p in ALL_PROFILES]


# --------------------------------------------------------------------------
# 자산
# --------------------------------------------------------------------------
def test_usd_assets_exist():
    """USD 는 학습 서버에도 있어야 한다."""
    for a in R.ASSETS.values():
        assert (_ASSET_DIR / a.usd_relpath).is_file(), f"USD 없음: {a.usd_relpath}"


@_SKIP_NO_URDF
def test_urdf_sources_exist():
    for a in R.ASSETS.values():
        assert (_URDF_DIR / a.urdf_relpath).is_file(), f"URDF 없음: {a.urdf_relpath}"


def test_asset_tags_and_shorts_unique():
    tags = [a.tag for a in R.ASSETS.values()]
    shorts = [a.short for a in R.ASSETS.values()]
    assert len(set(tags)) == len(tags), f"자산 태그 중복: {tags}"
    assert len(set(shorts)) == len(shorts), f"short 중복: {shorts}"


def test_asset_tags_match_run_naming():
    """run_naming.ASSET_TAGS 와 어휘가 어긋나면 자산 게이트가 조용히 통과한다."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "run_naming", _HDGP / "scripts" / "tools" / "run_naming.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # dataclass 해석이 sys.modules 를 참조하므로 exec 전에 등록해야 한다.
    sys.modules["run_naming"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("run_naming", None)
    for a in R.ASSETS.values():
        assert a.tag in mod.ASSET_TAGS, (
            f"'{a.tag}'({a.name}) 가 run_naming.ASSET_TAGS 에 없다 — 자산 게이트 무력화"
        )
        assert mod.ASSET_TAGS[a.tag] == a.name, (
            f"태그 {a.tag}: 레지스트리 {a.name} vs run_naming {mod.ASSET_TAGS[a.tag]}"
        )


# --------------------------------------------------------------------------
# 프로필 × URDF
# --------------------------------------------------------------------------
@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_joint_regex_resolves_to_declared_count(p, urdf_cache):
    joints, _, _ = urdf_cache[p.asset.name]
    arm = {j for j in joints if re.fullmatch(p.arm_joint_regex, j)}
    hand = {j for j in joints if re.fullmatch(p.hand_joint_regex, j)}
    assert len(arm) == p.num_arm_joints, f"{p.name} 팔: {sorted(arm)}"
    assert len(hand) == p.num_hand_joints, f"{p.name} 손: {sorted(hand)}"
    assert not (arm & hand), f"{p.name} 팔/손 regex 가 겹친다: {sorted(arm & hand)}"


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_all_referenced_bodies_exist(p, urdf_cache):
    _, links, _ = urdf_cache[p.asset.name]
    missing = [b for b in (p.palm_body, *p.fingertip_bodies, *p.all_sensor_bodies)
               if b not in links]
    assert not missing, f"{p.name} 에 없는 링크: {missing}"


@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_contact_groups_are_opposing_and_sensor_backed(p):
    a, b = set(p.contact_group_a), set(p.contact_group_b)
    assert a and b, f"{p.name} 대향 그룹이 비었다"
    assert not (a & b), f"{p.name} 대향 그룹이 겹친다: {a & b}"
    assert (a | b) <= set(p.fingers), f"{p.name} 그룹이 손가락 목록 밖: {(a | b) - set(p.fingers)}"
    for f in a | b:
        assert p.sensor_bodies(f), f"{p.name}/{f} 에 센서 body 가 하나도 없다"


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_actuator_coverage_is_total(p, urdf_cache):
    """커버리지 누락 관절은 조용히 무구동 자유회전한다(adf0b24 교훈)."""
    joints, _, _ = urdf_cache[p.asset.name]
    covered: set[str] = set()
    for spec in p.actuator_specs.values():
        for expr in spec["joint_names_expr"]:
            covered |= {j for j in joints if re.fullmatch(expr, j)}
    assert covered == joints, (
        f"{p.name} actuator 미커버: {sorted(joints - covered)} / "
        f"자산에 없는 관절 매칭: {sorted(covered - joints)}"
    )


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_init_joint_pos_covers_every_actuated_joint(p, urdf_cache):
    joints, _, _ = urdf_cache[p.asset.name]
    declared = set(p.init_joint_pos)
    assert declared == joints, (
        f"{p.name} init 누락: {sorted(joints - declared)} / "
        f"잉여: {sorted(declared - joints)}"
    )


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_init_joint_pos_is_within_urdf_limits(p, urdf_cache):
    """★init 값이 관절한계 밖이면 Articulation 검증이 부팅을 막는다.

    실제로 막혔다: DG-5F-S 대향 관절 thumb_2 는 **부호가 좌우 반대**다
    (r=[-2.670, 0] / l=[0, +2.670]). 우측 값을 좌측에 복사하면 한계 밖이 된다.
    이 저장소에서 반복된 "엄지 부호" 버그라 오프라인 게이트로 못 박는다.
    """
    _, _, limits = urdf_cache[p.asset.name]
    bad = []
    for j, v in p.init_joint_pos.items():
        if j not in limits:
            continue
        lo, hi = limits[j]
        if not (lo - 1e-6 <= v <= hi + 1e-6):
            bad.append(f"{j}={v:+.3f} not in [{lo:+.3f}, {hi:+.3f}]")
    assert not bad, f"{p.name} init 값이 관절한계 밖: {bad}"


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_fabric_joint_order_is_a_permutation_of_controlled_joints(p, urdf_cache):
    """★articulation 은 depth-major, fabric URDF 는 finger-major 다.

    이 순서를 프로필이 명시하지 않으면 손 관절이 통째로 어긋나 fabric 이 엉뚱한
    손 자세로 충돌구 FK 를 계산한다(probe 실측: 없는 자기충돌을 피하려 팔이 밀림).
    """
    if p.fabric_class is None:
        return
    joints, _, _ = urdf_cache[p.asset.name]
    order = list(p.fabric_joint_order)
    assert order, f"{p.name}: fabric_class 를 선언했으면 fabric_joint_order 도 필요하다"
    assert len(order) == len(set(order)), f"{p.name}: fabric_joint_order 에 중복"
    missing = [j for j in order if j not in joints]
    assert not missing, f"{p.name}: 자산에 없는 관절 {missing}"

    arm = {j for j in joints if re.fullmatch(p.arm_joint_regex, j)}
    assert order[: p.num_arm_joints] == sorted(arm), (
        f"{p.name}: fabric 팔 순서가 오름차순이 아니다 {order[: p.num_arm_joints]}"
    )


@_SKIP_NO_URDF
@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_fabric_hand_order_is_finger_major(p, urdf_cache):
    """손 부분이 손가락별로 뭉쳐 있어야 한다(finger-major). 실측 fabric URDF 규약."""
    if p.fabric_class is None or len(p.fabric_joint_order) <= p.num_arm_joints:
        return
    hand = list(p.fabric_joint_order[p.num_arm_joints:])
    # 각 손가락 이름이 연속 구간을 이루는지
    seen_start = {}
    for k, j in enumerate(hand):
        for f in p.fingers:
            if f"_{f}_" in j:
                seen_start.setdefault(f, []).append(k)
    for f, ks in seen_start.items():
        assert ks == list(range(ks[0], ks[0] + len(ks))), (
            f"{p.name}/{f}: fabric 손 순서가 흩어져 있다 {ks} — finger-major 위반"
        )


@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_fabric_asset_exists(p):
    """fabric_class 를 선언했으면 그 URDF 가 실제로 있어야 한다."""
    if p.fabric_class is None:
        assert p.fabric_robot_dir is None
        return
    urdf = _FABRIC_URDF / p.fabric_robot_dir / f"{p.fabric_robot_dir}.urdf"
    assert urdf.is_file(), f"{p.name}: fabric URDF 없음 {urdf}"


@pytest.mark.parametrize("p", ALL_PROFILES, ids=IDS)
def test_fabric_urdf_has_full_arm(p):
    if p.fabric_class is None:
        return
    text = (_FABRIC_URDF / p.fabric_robot_dir / f"{p.fabric_robot_dir}.urdf").read_text()
    names = re.findall(r'<joint name="([^"]+)"', text)
    assert len(names) >= p.num_arm_joints, (
        f"{p.name}: fabric URDF 관절이 {len(names)}개뿐 (팔 {p.num_arm_joints} 필요)"
    )


def test_left_fabric_urdf_is_not_a_copy_of_right():
    """좌측 fabric URDF 는 관절 **이름이 우측과 같고 기하만 미러**인 규약이다.

    따라서 이름으로는 좌/우를 구분할 수 없다. 대신 두 파일이 실제로 다른지 본다 —
    같으면 좌팔이 우팔 기구학으로 IK 를 푸는 조용한 버그다.
    """
    pairs = [(p, R.PROFILES[p.name.replace("_left", "_right")])
             for p in ALL_PROFILES
             if p.side == "l" and p.name.replace("_left", "_right") in R.PROFILES]
    assert pairs, "좌/우 짝이 하나도 없다 — 프로필 네이밍 규약이 깨졌다"
    for left, right in pairs:
        if left.fabric_class is None or right.fabric_class is None:
            continue
        assert left.fabric_robot_dir != right.fabric_robot_dir, (
            f"{left.name} 이 우측과 같은 fabric URDF 를 가리킨다: {left.fabric_robot_dir}"
        )
        lt = (_FABRIC_URDF / left.fabric_robot_dir
              / f"{left.fabric_robot_dir}.urdf").read_bytes()
        rt = (_FABRIC_URDF / right.fabric_robot_dir
              / f"{right.fabric_robot_dir}.urdf").read_bytes()
        assert lt != rt, f"{left.name}: 좌/우 fabric URDF 내용이 동일하다(미러 미적용)"


# --------------------------------------------------------------------------
# 레지스트리 자체
# --------------------------------------------------------------------------
def test_profile_names_unique_and_keyed():
    for name, p in R.PROFILES.items():
        assert name == p.name


def test_gym_id_slots_are_unique():
    """`open-<short>_<side>_<task>` 가 겹치면 두 로봇이 같은 로그 폴더를 쓴다."""
    slots = [(p.asset.short, p.side) for p in ALL_PROFILES]
    assert len(set(slots)) == len(slots), f"gym id 슬롯 충돌: {slots}"


def test_default_profile_registered():
    assert R.DEFAULT_PROFILE in R.PROFILES
    assert R.get(R.DEFAULT_PROFILE).fabric_class is not None


def test_wrap_absent_profiles_are_documented():
    """감쌀 마디가 없는 손(2지 그리퍼)은 envelope_frac := grip_frac 규약을 탄다."""
    for p in ALL_PROFILES:
        if not p.has_wrap_sensors:
            assert p.num_hand_joints <= 2, (
                f"{p.name}: 손 관절이 {p.num_hand_joints}개인데 wrap 센서가 없다 — 선언 누락 의심"
            )


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        R.get("nope")
