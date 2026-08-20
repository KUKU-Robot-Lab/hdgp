"""lift 레시피 이식 계약 고정 (Isaac 불필요).

env_cfg 는 isaaclab 을 끌어와 Isaac 앱 없이는 import 가 안 된다. 그래서 소스를 ast 로 읽어
**어떤 값을 어디에 넣었는지**를 검사한다(저장소 관례).

여기서 잡는 것은 전부 "정적으로는 멀쩡한데 학습을 통째로 망치는" 종류다:
  · minimal_height 를 절대 z 로 안 올려서 컵이 놓인 채로도 리프트 보상이 상시 1
  · 컵 스폰 z 를 bbox 반높이로 역산해 컵이 테이블에 파묻힘
  · 그리퍼 지령이 mimic 관절까지 가서 제약과 싸움
  · lift 레시피의 학습 조건(scale 0.5, default offset)을 무심코 바꿈
"""

import ast
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from openarm import OPENARM_ROOT_DIR
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

_HDGP = Path(OPENARM_ROOT_DIR).resolve().parents[2]
_ROBOT_URDF = _HDGP / "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
_CUP_USD = _HDGP / "assets/cup" / P.CUP_USD_NAME
_TABLE_USD = _HDGP / "assets/scene_objects/table.usd"
_CFG_SRC = Path(__file__).resolve().parents[1] / "grasp_left_env_cfg.py"
# 상속 원본. 커리큘럼 onset 처럼 "레퍼런스가 정하는 값"은 여기서 읽어야
# 레퍼런스가 바뀌었을 때 계약이 조용히 거짓이 되지 않는다.
_ISAACLAB_LIFT_ENV_CFG = (
    _HDGP.parent
    / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift"
    / "lift_env_cfg.py"
)


def _cfg_source() -> str:
    return _CFG_SRC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 자산이 실제로 존재하고 이름이 맞는가
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_ROBOT_URDF, _CUP_USD, _TABLE_USD])
def test_referenced_assets_exist(path):
    assert path.is_file(), f"자산 없음: {path}"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_every_preset_joint_exists_in_robot_urdf():
    """없는 이름을 넣으면 Isaac Lab 이 resolve_matching_names_values 에서 예외를 던진다."""
    root = ET.parse(_ROBOT_URDF).getroot()
    joints = {j.get("name") for j in root.iter("joint")}
    named = (
        list(P.LEFT_ARM_JOINT_NAMES)
        + list(P.GRIPPER_JOINT_NAMES)
        + list(P.RIGHT_REST_JOINT_POS)
    )
    missing = [n for n in named if n not in joints]
    assert not missing, f"URDF 에 없는 조인트: {missing}"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_gripper_base_is_a_link_but_tcp_is_not_used_as_one():
    """`l_hl_gripper_tcp` 는 physics USD 에서 강체로 병합돼 사라진다 → base + 오프셋을 쓴다."""
    root = ET.parse(_ROBOT_URDF).getroot()
    links = {ln.get("name") for ln in root.iter("link")}
    assert P.GRIPPER_BASE_BODY in links
    assert "l_hl_gripper_tcp" not in P.GRIPPER_BASE_BODY
    assert P.TCP_OFFSET_IN_BASE_Z > 0.0


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_gripper_stroke_matches_urdf_limit():
    root = ET.parse(_ROBOT_URDF).getroot()
    j = next(x for x in root.iter("joint") if x.get("name") == P.GRIPPER_DRIVE_JOINT)
    lim = j.find("limit")
    assert lim is not None
    assert j.get("type") == "prismatic"
    assert math.isclose(float(lim.get("lower", "nan")), P.GRIPPER_CLOSED_POS, abs_tol=1e-6)
    assert math.isclose(float(lim.get("upper", "nan")), P.GRIPPER_OPEN_POS, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 씬 기하 — 절대 z 함정
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _TABLE_USD.is_file(), reason="테이블 USD 없음")
def test_table_surface_matches_usd_extent():
    """★상면 = init pos z + Cube extent 반높이.

    이 값을 두 번 틀렸다:
      · 0.2082 — right/grasp_sensor 가 컵 반높이로 역산한 중간값. 상면이 아니다.
      · 0.2004 — USD **BBoxCache** 로 읽은 값. extent 는 이미 xformOp:scale 반영값인데
                 BBoxCache 가 scale 을 또 곱한다. extent 를 직접 읽어야 한다.
    이 테스트는 그 "직접 읽기"를 고정한다.
    """
    pxr_usd = pytest.importorskip("pxr.Usd", reason="pxr 없음")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    stage = pxr_usd.Stage.Open(str(_TABLE_USD))
    halves = []
    for prim in stage.Traverse():
        ext = UsdGeom.Boundable(prim).GetExtentAttr() if prim.IsA(UsdGeom.Boundable) else None
        if ext and ext.HasAuthoredValue():
            _lo, hi = ext.Get()
            halves.append((hi[0], hi[1], hi[2]))
    assert halves, "table.usd 에서 authored extent 를 못 찾았다"
    hx, _, hz = max(halves, key=lambda h: h[0] * h[1])

    assert math.isclose(P.TABLE_HALF_X, hx, abs_tol=1e-6)
    assert math.isclose(P.TABLE_SURFACE_Z, P.TABLE_POS[2] + hz, abs_tol=1e-6)
    # 과거에 틀렸던 두 값이 다시 들어오지 않도록
    assert not math.isclose(P.TABLE_SURFACE_Z, 0.2082, abs_tol=1e-4)
    assert not math.isclose(P.TABLE_SURFACE_Z, 0.2004, abs_tol=1e-4)


def test_lift_gate_matches_the_reference_open_at_rest_closed_when_tipped():
    """★★두 번 틀린 자리. `mdp.object_is_lifted` 는 물체 **root 원점**의 절대 world z 를 본다.

    1차 오해 — 기준선을 테이블 상면으로 잡았다(0.255). shaker 원점은 바닥에서 92 mm 위라
      놓인 원점(0.30709)보다 낮았고, lifting 이 상시 1(14.63/15.0)이 됐다(test1-r2).
    2차 오해 — 그래서 "놓인 원점 + 4 cm = 0.34709" 로 올렸다. **이것도 틀렸다.**
      레퍼런스 openarm lift 는 큐브를 world z **0.055** 에 놓고 `minimal_height`=**0.04**
      를 쓴다 — 놓인 상태에서 **게이트가 이미 열려 있다**. 즉 `object_is_lifted` 는 상수라
      gradient 가 없고, 실제 신호는 **목표까지의 거리 기울기**뿐이며 목표 z 가 공중이라
      들어올리기가 저절로 유도된다. 넘어야 할 문턱이 없다.
      내 0.34709 는 레퍼런스에 없는 **절벽**이었고, 태스크공간 1차가 483 epoch 동안
      `lifting_object` 0.00 이었던 이유다(reaching 0.3~0.57 = 팔은 컵 곁에 있었다).

    ⚠ 그 절벽을 **이 테스트가 옳다고 고정하고 있었다**. 같은 오해를 공유한 테스트는
      버그를 막지 못한다. 그래서 이제 레퍼런스와의 **관계**를 고정한다.

    공짜 보상은 게이트를 여는 것으로 막지 않는다 — `_held` 의 **TCP 근접(8 cm)** 게이트가
    막는다(던지기 방지용으로 이미 있다). 그리고 임계가 누운 컵 원점보다 위라 쓰러지면
    게이트가 닫힌다 = "세워서 들고 있을 때만 목표 보상".
    """
    assert P.MINIMAL_LIFT_HEIGHT < P.CUP_SPAWN_Z, (
        "게이트가 놓인 상태에서 닫혀 있다 — 레퍼런스에 없는 절벽이 생긴다"
    )
    assert P.MINIMAL_LIFT_HEIGHT > P.CUP_TIPPED_ORIGIN_Z, (
        "누운 컵에서도 게이트가 열린다 — 쓰러뜨린 채 목표로 밀 수 있다"
    )
    # 레퍼런스와 같은 여유(0.055 − 0.04 = 0.015)에서 파생되어야 한다.
    assert math.isclose(
        P.MINIMAL_LIFT_HEIGHT, P.CUP_SPAWN_Z - P.LIFT_GATE_BELOW_REST, abs_tol=1e-9
    )
    # 공짜 보상 차단은 TCP 근접 게이트가 맡는다 — 없어지면 test3 의 던지기가 돌아온다.
    src = _cfg_source()
    assert "GRASP_MAX_EE_DISTANCE" in src
    assert src.count("P.MINIMAL_LIFT_HEIGHT") >= 3


def test_drop_threshold_catches_a_tipped_cup_not_just_a_fallen_one():
    """★종료 임계는 낙하뿐 아니라 **쓰러짐**도 잡아야 한다 — 전제가 한 번 뒤집힌 항목.

    처음에는 "넘어짐은 lift 레시피가 보지 않으니 종료시키지 않는다"로 뒀다. 렌더 관찰이
    그 판단을 뒤집었다: shaker 는 가늘고 길어 잘 쓰러지는데 2지 그리퍼로는 다시 세울 수
    없어, 쓰러진 뒤의 에피소드가 통째로 낭비된다. 게다가 원점 z 는 기울기에 둔감해서
    (완전히 누워도 0.25199) 레퍼런스식 임계 0.165 로는 영원히 안 잡힌다.
    """
    assert P.OBJECT_DROP_HEIGHT < P.CUP_SPAWN_Z, "정상 상태를 종료시키면 안 된다"
    assert P.OBJECT_DROP_HEIGHT > P.CUP_TIPPED_ORIGIN_Z, "누운 컵을 잡지 못한다"
    assert P.OBJECT_DROP_HEIGHT > P.CUP_BOTTOM_TO_ORIGIN, "테이블 밖 낙하도 잡아야 한다"
    # 옛 레퍼런스식(상면 −0.05)으로 되돌아가지 않도록
    assert not math.isclose(P.OBJECT_DROP_HEIGHT, P.TABLE_SURFACE_Z - 0.05, abs_tol=1e-6)


@pytest.mark.skipif(not _CUP_USD.is_file(), reason="컵 USD 없음")
def test_reference_object_body_name_is_overridden():
    """★레퍼런스는 큐브 prim 이름 `"Object"` 를 SceneEntityCfg 에 박아 둔다.

    우리 shaker 의 강체는 `baseLink` 라 그대로 두면 매니저가 이름을 resolve 하는 순간
    죽는다("Object: [] / Available strings: ['baseLink']").

    ⚠ 이 결함은 **로컬에서 드러나지 않는다** — `_process_term_cfg_at_play` 는
      `sim.is_playing()` 일 때만 도는데, 프로브 경로에서는 그 타이밍을 안 타서 resolve 가
      스킵되고 정상 동작한다. 서버 학습 기동에서만 터졌다. 그래서 정적으로 못을 박는다.
    """
    # ★`importorskip` 보다 먼저 `from pxr import ...` 를 쓰면 pxr 없는 머신(서버)에서
    #   skip 이 아니라 **에러**가 난다. 로컬만 보고 넘어가지 말 것.
    pxr_usd = pytest.importorskip("pxr.Usd", reason="pxr 없음")
    stage = pxr_usd.Stage.Open(str(_CUP_USD))
    prims = {p.GetName() for p in stage.Traverse()}
    assert P.CUP_BODY_NAME in prims, f"컵 USD 에 {P.CUP_BODY_NAME} 이 없다: {sorted(prims)}"
    assert "Object" not in prims

    src = _cfg_source()
    assert 'reset_object_position.params["asset_cfg"]' in src, (
        "레퍼런스의 body_names=\"Object\" 를 덮어쓰지 않았다"
    )


def test_settling_at_the_goal_is_rewarded():
    """★"목표지점에 이동시켜 **가만히 정지**시켜야 한다" — 레퍼런스에 없는 요구.

    `mdp.object_goal_distance` 는 거리만 보므로 목표 근처에서 컵이 흔들려도 만점이다
    (test8: goal-tracking 은 상한의 68% 인데 정밀 항은 16%). 속도 항이 있어야 한다.

    선속도·각속도를 **둘 다** 봐야 한다 — 각속도를 빼면 제자리에서 빙빙 도는 상태가
    만점이 된다.
    """
    src = _cfg_source()
    assert "object_settled_at_goal" in src
    assert "settled_at_goal" in src
    assert "lin_vel_std" in src and "ang_vel_std" in src, "각속도 항이 빠졌다"
    assert P.SETTLE_LIN_VEL_STD > 0 and P.SETTLE_ANG_VEL_STD > 0
    assert 0.0 < P.SETTLE_REWARD_WEIGHT <= 15.0, (
        "정지 보너스가 lifting(15) 을 넘으면 파지보다 정지가 우선이 된다"
    )
    # ★★임계는 **실측 규모에 맞춰야** 신호가 산다. 처음에 0.10 m/s·1.00 rad/s 로 잡았다가
    #   보상이 학습 내내 정확히 0 이었다(test10). 실측은 0.444 m/s·3.43 rad/s 였고 그
    #   값에 옛 임계를 넣으면 품질이 0.0003·0.0021 이라 곱하면 신호가 사라진다.
    measured_lin, measured_ang = 0.444, 3.432        # test8 정책, 쥐고 있을 때
    assert 1.0 - math.tanh(measured_lin / P.SETTLE_LIN_VEL_STD) > 0.05, (
        "선속도 임계가 실측 대비 너무 빡빡해 보상 신호가 죽는다"
    )
    assert 1.0 - math.tanh(measured_ang / P.SETTLE_ANG_VEL_STD) > 0.05, (
        "각속도 임계가 실측 대비 너무 빡빡해 보상 신호가 죽는다"
    )
    # 그래도 "정지"를 요구할 만큼은 조여야 한다 — 실측값에서 품질이 이미 높으면 무의미
    assert 1.0 - math.tanh(measured_lin / P.SETTLE_LIN_VEL_STD) < 0.5


def test_grasp_pose_is_a_bonus_never_a_gate():
    """★★자세는 **연속 보너스로만** 건다 — 게이트로 넣었다가 학습을 통째로 죽였다.

    컵 자세를 40° AND 게이트로 걸자(test6/test7) 파지 중 필연적인 흔들림이 전부 차단돼
    양의 보상이 **완전히 0** 이 됐고, 남은 것이 페널티뿐이라 **에피소드를 빨리 끝내는 것이
    최적**이 됐다:
        lifting 6.14 → 0.0000 / 에피소드 길이 130 → 13 / 총보상 +34.9 → −0.46
    test6(옛 홈)·test7(새 홈)이 똑같이 붕괴해 원인이 홈이 아니라 게이트임이 갈렸다.
    """
    src = _cfg_source()
    # 판정 게이트는 lifted & near 까지만 (test5 에서 검증된 구성)
    assert "min_upright_cos" not in src, "컵 자세를 게이트에 넣지 말 것 — 학습이 죽는다"
    # ★결과적으로 제어해야 할 것은 **컵을 똑바로 드는 것**이다(실기에서도 컵 로컬 z 는
    #   파악 가능하다). TCP축⊥컵축 항은 |sin| 이라 81.8° 에서 이미 0.99 여서 개선 압력이
    #   없었고 test8 81.2° → test12 81.8° 로 제자리였다 → 자세 보너스에서 뺐다.
    rewards_src = (Path(_CFG_SRC).parent / "grasp_left_rewards.py").read_text(encoding="utf-8")
    pose_fn = rewards_src.split("def held_with_good_pose")[1]
    assert "perpendicular_quality(" not in pose_fn, "직교 항은 개선 압력이 없어 뺐다"
    # upright 는 cos 를 그대로 쓰면 12.8° 에서 0.975 라 압력이 없다 → 재척도해 가파르게
    assert "upright_zero_at_cos" in pose_fn
    assert 0.0 < P.CUP_UPRIGHT_ZERO_AT_COS < 1.0
    q_at_measured = (math.cos(math.radians(12.8)) - P.CUP_UPRIGHT_ZERO_AT_COS) / (
        1.0 - P.CUP_UPRIGHT_ZERO_AT_COS
    )
    assert 0.3 < q_at_measured < 0.95, (
        "현재 자세(12.8°)에서 품질이 1 에 붙으면 개선 압력이 없고, 0 이면 신호가 죽는다"
    )
    assert "held_with_good_pose" in src
    assert "grasp_pose" in src
    assert 0.0 < P.GRASP_POSE_REWARD_WEIGHT <= 15.0 / 3.0, (
        "자세 보너스가 lifting(15) 대비 1/3 을 넘으면 자세만 맞추는 국소최적이 생긴다"
    )


def test_goal_is_a_specific_point_not_a_wide_range():
    """★이송 목표는 **우리가 정하는 특정 점**이다(실기에서도 옮길 자리는 우리가 지정한다).

    넓은 랜덤 범위는 정밀 도달과 정지를 동시에 어렵게 만든다 — test12 에서 goal_fine 이
    상한의 8%, settle 이 7.8% 에 머문 이유 중 하나다.
    """
    span_x = P.GOAL_POS_X[1] - P.GOAL_POS_X[0]
    span_y = P.GOAL_POS_Y[1] - P.GOAL_POS_Y[0]
    span_z = P.GOAL_POS_Z[1] - P.GOAL_POS_Z[0]
    for span in (span_x, span_y, span_z):
        assert span <= 0.06, f"목표 범위가 넓다({span:.3f} m) — 특정 점이어야 한다"
    # 스폰 자리에 그대로 두는 것이 목표가 되면 "들어서 옮기기"가 성립하지 않는다
    dz = P.GOAL_POINT[2] - P.CUP_SPAWN_Z
    assert dz > 0.05, "목표가 스폰 높이와 가까우면 이송을 요구하지 못한다"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_home_pose_keeps_action_range_slack_from_joint_limits():
    """★★액션 범위가 ±0.5 rad 인데 홈이 한계에 붙어 있으면 **탐색이 잘린다**.

    옛 홈(Fabrics 시절 IK)은 여유가 0.116 rad 뿐이었다 — l_aj_6 −0.6695(한계 ±0.7854).
    렌더에서 "j7 이 꺾여 보인다"고 관찰된 것도, 정책이 한쪽으로 못 움직인 것도 같은 원인.
    """
    root = ET.parse(_ROBOT_URDF).getroot()
    limits = {}
    for j in root.iter("joint"):
        name = j.get("name") or ""
        lim = j.find("limit")
        if name.startswith("l_aj_") and lim is not None:
            limits[name] = (float(lim.get("lower", "0")), float(lim.get("upper", "0")))

    action_half_range = 0.5          # JointPositionActionCfg(scale=0.5) 의 액션 ±1
    tight = []
    for name, value in P.LEFT_ARM_HOME_JOINT_POS.items():
        lo, hi = limits[name]
        slack = min(value - lo, hi - value)
        if slack < action_half_range:
            tight.append(f"{name}={value:+.4f} 여유 {slack:.3f}")
    assert not tight, f"한계 여유가 {action_half_range} rad 미만인 관절: {tight}"


def test_left_arm_velocity_limit_matches_the_reference():
    """★★레퍼런스 `OPENARM_UNI_CFG` 는 팔에 velocity 2.175/2.175/2.61 을 명시한다.

    이걸 빠뜨리면 URDF 기본값(5.4~20.9 rad/s)이 쓰이는데, damping 이 4 뿐이라 팔이
    과속으로 오버슈트하며 진동한다("시작할 때 진자처럼 흔들린다"는 렌더 관찰).
    20.9 rad/s 는 한 스텝(0.02 s)에 0.42 rad — 액션 범위(±0.5 rad)를 한 스텝에 소화한다.
    """
    src = _cfg_source()
    assert "velocity_limit_sim=P.ARM_VELOCITY_LIMIT" in src
    assert set(P.ARM_VELOCITY_LIMIT.values()) == {2.175, 2.61}
    assert set(P.ARM_EFFORT_LIMIT.values()) == {40.0, 27.0, 7.0}
    # URDF 기본값으로 되돌아가지 않도록
    assert max(P.ARM_VELOCITY_LIMIT.values()) < 5.0


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_tcp_offset_lands_inside_the_fingers():
    """TCP 가 손가락 범위 밖이면 정책이 조준하는 점과 실제 파지점이 어긋난다.

    URDF 기준: 손가락 링크 원점은 base 에서 z=0.015, 손가락 메시는 링크 기준
    z∈[-0.0145, +0.0804] → base 기준 z∈[+0.0005, +0.0954]. TCP 는 +0.08 로 그 안이다.
    """
    root = ET.parse(_ROBOT_URDF).getroot()
    tcp_joint = next(
        j for j in root.iter("joint") if j.get("name") == "l_hj_gripper_tcp"
    )
    origin = tcp_joint.find("origin")
    assert origin is not None
    xyz = [float(v) for v in (origin.get("xyz") or "").split()]
    assert math.isclose(xyz[2], P.TCP_OFFSET_IN_BASE_Z, abs_tol=1e-6)
    assert xyz[0] == 0.0 and xyz[1] == 0.0, "TCP 가 그리퍼 중심선에서 벗어났다"
    # 손가락 메시 범위(base 기준, 실측) 안이어야 한다
    finger_span = (0.0005, 0.0954)
    assert finger_span[0] < P.TCP_OFFSET_IN_BASE_Z < finger_span[1]


def test_idle_joints_get_an_explicit_pd_target():
    """★★`init_state.joint_pos` 는 관절의 **상태**만 정하고 PD 목표는 정하지 않는다.

    액션 대상이 아닌 관절(오른팔 7 + 오른손 20 + 헤드 2)은 아무도 목표를 써 주지 않아
    목표가 0 인 채 남고, 팔이 "차렷"으로 내려가 바닥에 닿는다(렌더 관찰 → 프로브 확인:
    목표 미지정 25.4° vs 지정 2.1°). 리셋마다 목표를 명시하는 이벤트가 반드시 있어야 한다.
    """
    src = _cfg_source()
    assert "hold_idle_joints" in src, "유휴 관절 목표 고정 이벤트가 없다 — 오른팔이 내려앉는다"
    assert "hold_joints_at_target" in src
    assert 'mode="reset"' in src
    # 오른팔·오른손·헤드가 모두 포함돼야 한다
    assert "P.RIGHT_REST_JOINT_POS" in src
    assert "head_j_pan" in src and "head_j_tilt" in src


def test_idle_right_arm_has_enough_effort_to_hold_its_pose():
    """★URDF 팔 effort 는 j1/j2=40, j3/j4=27, **j5~j7=7 N·m** 뿐이다.

    이걸 안 올리면 stiffness 400 이 무의미하게 포화해 20 관절 손(약 1.4 kg)을 단 오른팔이
    중력에 처지고 손끝이 테이블에 얹힌다(실측 관절 오차 최대 49.9°). 학습에 쓰이지 않는
    배경 팔이므로 sim 에서만 강하게 잡아 둔다.
    """
    src = _cfg_source()
    idle = src[src.index('"idle_right_arm"'):]
    idle = idle[: idle.index("),")]
    assert "effort_limit_sim" in idle, "유휴 오른팔에 effort_limit_sim 이 없다 — 처진다"


def test_cup_spawn_z_puts_bottom_on_table():
    """컵 원점은 bbox 반높이가 아니라 **메시 bottom** 에서 역산해야 판에 정확히 앉는다."""
    assert math.isclose(
        P.CUP_SPAWN_Z, P.TABLE_SURFACE_Z + P.CUP_BOTTOM_TO_ORIGIN, abs_tol=1e-9
    )


def test_cup_spawn_box_sits_on_the_table():
    near = P.TABLE_POS[0] - P.TABLE_HALF_X
    far = P.TABLE_POS[0] + P.TABLE_HALF_X
    base_r = 0.0295          # shaker 바닥 원판 반경 (shaker_closed_rl.usd bottom_plug)
    assert near <= P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE - base_r
    assert P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE + base_r <= far


def test_spawn_box_clears_the_home_pose_arm():
    """★홈은 컵을 감싼 **파지 자세**라, 그 자리에 컵을 스폰하면 팔 메시가 컵을 관통해
    PhysX 가 컵을 수백 mm 날려버린다(zero-action 실측 최대 886 mm, tilt 85°).

    probe_lift_left_gripper_smoke.py 의 1e 스윕이 잰 경계가 `SPAWN_X_SAFE_MIN`이고,
    스폰 박스 **전체**가 그 바깥이어야 한다. 중심만 확인하면 랜덤화 하한에서 터진다.
    """
    assert P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE >= P.SPAWN_X_SAFE_MIN - 1e-9
    # 목표 커맨드는 **들어올린 뒤** 옮길 지점이라 스폰 경계에 묶이지 않는다(컵이 공중이면
    # 팔이 점유한 공간과 충돌하지 않는다). 다만 팔 도달 범위 안이어야 한다.
    assert P.GOAL_POS_X[0] > P.TABLE_POS[0] - P.TABLE_HALF_X


def test_spawn_center_is_no_longer_the_left_grasp_v1_position():
    """반전 기록: 처음엔 `tesollo/left/grasp_v1`(x 0.30, y 0.20)을 따랐다.

    그 자리는 lift 홈 자세에서 **팔이 점유한 공간**이라 쓸 수 없다. 홈을 교체한 뒤에는
    관통 영역이 "x 가 낮고 y 가 높은" 코너로 바뀌어 y 도 함께 내렸다.
    ★스폰 박스는 홈에 종속이다 — 홈을 바꿀 때마다 스윕을 다시 돌려야 한다.
    """
    assert P.CUP_SPAWN_X_CENTER > 0.30
    assert P.CUP_SPAWN_Y_CENTER <= 0.20


def test_goal_z_range_sits_above_the_lift_threshold():
    """목표도 컵 **원점** 좌표다. 하한이 리프트 임계보다 낮으면 게이트는 닫혀 있는데 목표는
    이미 발밑에 있는 꼴이 되어 "먼저 들어라 → 옮겨라" 순서가 무너진다."""
    lo, hi = P.GOAL_POS_Z
    assert lo >= P.MINIMAL_LIFT_HEIGHT
    assert hi > lo


# ---------------------------------------------------------------------------
# lift 레시피의 학습 조건을 깨지 않았는가
# ---------------------------------------------------------------------------
def test_arm_action_keeps_lift_recipe_settings():
    """scale 0.5 + use_default_offset 이 "액션 0 = 파지 준비 자세"를 만든다.

    이게 lift 가 단순한 제어로도 학습되는 핵심 이유라 바꾸면 안 된다.
    """
    src = _cfg_source()
    assert "JointPositionActionCfg" in src
    assert "scale=0.5" in src
    assert "use_default_offset=True" in src
    assert "DifferentialInverseKinematics" not in src, "IK 경로는 RL 학습에 쓰지 않는다"
    # Fabrics 는 폐기됐다. 도크스트링에는 그 경위가 적혀 있으므로 **임포트**만 금지한다.
    imported = {
        n.module or ""
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.ImportFrom)
    } | {
        a.name
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Import)
        for a in n.names
    }
    assert not [m for m in imported if "fabric" in m.lower()], "Fabrics 임포트 금지"


def test_gripper_action_is_binary_and_targets_only_the_drive_joint():
    """mimic 관절(gripper_2)까지 지령하면 PhysX mimic 제약과 싸운다."""
    src = _cfg_source()
    assert "BinaryJointPositionActionCfg" in src
    assert "P.GRIPPER_DRIVE_JOINT" in src
    assert P.GRIPPER_DRIVE_JOINT == "l_hj_gripper_1"
    # 액추에이터 커버리지는 두 관절 모두 (없으면 무구동 자유이동)
    assert "l_hj_gripper_[1-2]" in src


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_action_dim_is_eight():
    """7 관절 + 이진 그리퍼 1. cfg 의 정규식을 URDF 조인트에 실제로 전개해서 센다.

    `JointPositionAction` 은 정규식이 하나도 안 풀리면 예외로 죽고, 너무 많이 풀리면
    조용히 차원이 늘어 체크포인트가 안 맞는다. 둘 다 여기서 잡는다.
    """
    root = ET.parse(_ROBOT_URDF).getroot()
    movable = [
        j.get("name")
        for j in root.iter("joint")
        if j.get("type") in ("revolute", "prismatic", "continuous")
    ]
    src = _cfg_source()
    arm_pattern = re.search(r'joint_names=\["(l_aj_\[[^"]+\])"\]', src)
    assert arm_pattern, "팔 액션의 joint_names 정규식을 못 찾았다"
    arm_hits = [n for n in movable if n and re.fullmatch(arm_pattern.group(1), n)]
    assert len(arm_hits) == 7, f"팔 액션 차원 {len(arm_hits)} (기대 7): {arm_hits}"

    # 그리퍼는 이진 스칼라 1차원 — 구동 관절 하나에만 지령한다.
    assert P.GRIPPER_DRIVE_JOINT in movable
    assert len(arm_hits) + 1 == 8


def test_env_cfg_inherits_isaaclab_lift():
    """보상·관측·커맨드·커리큘럼을 직접 짜지 않고 물려받는다 — 그래야 레시피가 보존된다."""
    tree = ast.parse(_cfg_source())
    bases = [
        b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "GraspLeftGripperEnvCfg"
        for b in cls.bases
    ]
    assert "LiftEnvCfg" in bases
    src = _cfg_source()
    assert "isaaclab_tasks.manager_based.manipulation.lift" in src
    # ★물려받은 6 개 term 의 weight 는 재정의하지 않는다. 신설은 jaw 수평 보너스 하나뿐이고
    #   그 weight 는 preset 상수로만 온다(리터럴 금지 — 값이 코드에 흩어지지 않게).
    inherited = (
        "reaching_object", "lifting_object", "object_goal_tracking",
        "object_goal_tracking_fine_grained", "action_rate", "joint_vel",
    )
    for name in inherited:
        assert f"self.rewards.{name} = " not in src, f"{name} 을 재정의하지 말 것"
    # 신설 term 은 보너스 둘(grasp_pose, settled_at_goal)뿐이다.
    # 판정 게이트를 늘리는 term 은 금지 — test6/test7 에서 학습을 죽였다.
    assert src.count("RewTerm(") <= 2, "신설 term 은 보너스 둘뿐이다"


def test_smoothing_is_the_reference_curriculum_not_an_extra_term():
    """★평활화의 주체는 **레퍼런스 커리큘럼**이지 새 항이 아니다.

    새 항을 얹기 전에 기존 커리큘럼이 무엇을 하고 있는지부터 분리해야 한다. test13 에서
    `action_rate` 는 −5.3 → −2.3 으로 좋아졌지만, 엔트로피로 σ 를 역산해 분해하면 그 개선의
    대부분이 **탐색 노이즈(σ 1.67 → 0.93) 축소**였고 정책 평균의 평활도는 평탄했다
    (⟨Δa²⟩ 평균부 12.3 → 9.9). 결정론 프로브도 같은 말을 한다 — |Δa| 1.515 → 1.713.

    즉 `action_rate` 곡선은 평활화 진척의 증거가 못 된다. 자세한 내용은 이 태스크의
    CLAUDE.md "함정: action_rate 곡선을 평활화 진척으로 읽지 말 것" 절에 있다.
    그래서 jerk 항도 배선하지 않는다 — 같은 액션공간 통계라 같은 노이즈 오염을 받는다.
    한 런에 한 가설만 바꾼다.
    """
    src = _cfg_source()
    assert "self.rewards.action_jerk" not in src, "jerk 항은 의도적으로 배선하지 않는다"
    assert "self.curriculum.action_jerk" not in src
    # 레퍼런스 커리큘럼 둘은 반드시 살아 있어야 한다(상속받으므로 재정의가 없어야 정상).
    for name in ("action_rate", "joint_vel"):
        assert f"self.rewards.{name} = " not in src, f"{name} 은 레퍼런스 것을 그대로 쓴다"


def test_penalty_curriculum_fires_after_lifting_and_leaves_room_to_act():
    """★평활화 페널티는 **lifting 이 자리를 잡은 뒤에** 켜져야 하고, 켠 뒤에도 충분히 돌아야 한다.

    test15 는 레퍼런스 onset(10000 step = epoch 417)에서 붕괴했다. 액션 변화율 상한 때문에
    초기 탐색이 느려져 그 시점 lifting 이 1.19 뿐이었고, 페널티가 1000 배가 되자 총보상이
    음수(+6.80 → −10.99)가 됐다. 그러자 최적해가 "에피소드를 빨리 끝내기" 즉 **컵을 쳐서
    떨어뜨리기**가 됐다(epoch 500 에 drop 100% · ep_len 13).

    두 방향 모두 계약이 필요하다:
      · 너무 이르면 → 위처럼 부호가 뒤집혀 자살 전략으로 붕괴한다.
      · 너무 늦으면 → 켜고 나서 평활화할 시간이 없다.
    """
    import pathlib as _pl

    yaml_path = (
        _pl.Path(__file__).resolve().parents[1] / "config" / "agents" / "rl_games_ppo_cfg.yaml"
    )
    text = yaml_path.read_text()
    max_epochs = int(re.search(r"max_epochs:\s*(\d+)", text).group(1))
    horizon = int(re.search(r"horizon_length:\s*(\d+)", text).group(1))
    onset_epoch = P.ACTION_PENALTY_CURRICULUM_STEPS / horizon

    # 레퍼런스 onset(epoch 417)보다 확실히 뒤여야 한다 — 거기서 실제로 붕괴했다.
    assert onset_epoch > 1000, f"onset epoch {onset_epoch:.0f} 는 lifting 학습 전이다"
    # 켠 뒤에 평활화가 일할 epoch 이 남아야 한다.
    assert max_epochs - onset_epoch >= 2000, (
        f"onset(epoch {onset_epoch:.0f}) 이후 남는 epoch 이 {max_epochs - onset_epoch:.0f} 뿐"
    )
    # env_cfg 가 실제로 이 상수로 덮어써야 한다(프리셋만 바꾸고 배선을 잊는 것을 막는다).
    src = _cfg_source()
    assert 'params["num_steps"] = P.ACTION_PENALTY_CURRICULUM_STEPS' in src


def test_lift_gate_requires_holding_the_cup():
    """★★리프트 판정이 z 만 보면 **컵을 쳐 날리는 것**이 최적 전략이 된다.

    test3(1500 epoch) 실측: 리프트 판정 비율 85.9% 동안 **TCP–컵 거리 평균 3044 mm**,
    `reaching_object` 0.019 → 0.018 평탄, `object_dropping` 종료 99.8%.
    컵이 134 g·높이 175 mm 라 위로 치면 에피소드(1.8 초) 내내 공중에 떠 있고, 그동안
    lifting(15) + goal-tracking(16) 을 모두 받는다. 큐브 레퍼런스에는 없는 문제다.

    → 판정을 `z > 임계 AND TCP 가 컵 곁` 으로 바꾼다. weight 는 건드리지 않는다.
    goal-tracking 두 개도 내부에서 z 게이트를 직접 계산하므로 **함께** 교체해야 한다.
    """
    src = _cfg_source()
    assert "object_is_held_and_lifted" in src
    assert "object_goal_distance_when_held" in src
    # 세 term 전부에 근접 임계가 들어갔는가 (하나라도 빠지면 그쪽으로 hack 이 되살아난다)
    assert src.count("P.GRASP_MAX_EE_DISTANCE") >= 2
    assert 0.0 < P.GRASP_MAX_EE_DISTANCE < 0.13, (
        "임계가 홈 자세의 TCP–컵 거리(약 130 mm)만큼 크면 게이트가 무의미하다"
    )


def test_each_asset_cfg_assignment_builds_a_fresh_instance():
    """★`SceneEntityCfg` 는 매니저가 `resolve()` 로 제자리 변경하는 **가변** 객체다.

    한 인스턴스를 여러 term 에 공유하면 첫 term 이 joint_ids 를 채워 넣고, 두 번째 term 에서
    "joint_names 와 joint_ids 가 불일치" 로 env 생성이 죽는다(Isaac 실측으로 확인).
    그래서 `params["asset_cfg"] = ...` 우변은 항상 **호출(새 인스턴스)** 이어야 한다.
    """
    tree = ast.parse(_cfg_source())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.slice, ast.Constant)
                and tgt.slice.value == "asset_cfg"
                and not isinstance(node.value, ast.Call)
            ):
                offenders.append(ast.unparse(node))
    assert not offenders, f"공유 인스턴스를 대입한 곳: {offenders}"


def test_idle_right_arm_is_not_a_mirror_of_the_left_home():
    """왼팔 홈이 그리퍼 파지 전용 자세라 그 미러는 오른팔에 의미가 없다(렌더로 확인됨)."""
    mirror_sign = (-1.0, -1.0, -1.0, +1.0, -1.0, -1.0, -1.0)
    mirrored = [
        s * P.LEFT_ARM_HOME_JOINT_POS[n]
        for n, s in zip(P.LEFT_ARM_JOINT_NAMES, mirror_sign)
    ]
    actual = [P.RIGHT_ARM_REST_JOINT_POS[f"r_aj_{i}"] for i in range(1, 8)]
    assert not all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(mirrored, actual))


def test_arm_action_command_cannot_outrun_the_joint():
    """★지령은 관절이 낼 수 있는 속도를 넘으면 안 된다.

    test13 결정론 정책은 관절당 0.324 rad/스텝(= 16 rad/s)을 지령했다. 관절 속도 한계
    2.175~2.61 rad/s 의 **7 배**라 팔은 그냥 포화했다(실측 관절속도 2.02 rad/s ≈ 한계).
    보상으로는 못 고친다 — `action_rate_l2` 는 액션공간 통계라 탐색 노이즈에 오염되고,
    옵티마이저는 σ 만 줄인다(CLAUDE.md 함정 절).

    그래서 액션 자체를 제한한다. 상한은 **관절 속도 한계 그 자체**여야 한다 — 임의의
    숫자를 박아두면 근거가 사라지고, 액추에이터 한계를 바꿨을 때 조용히 어긋난다.
    실제 제한 동작은 `scripts/probes/probe_action_rate_limit.py` 가 잰다(7 관절 PASS).
    """
    src = _cfg_source()
    assert "RateLimitedJointPositionActionCfg" in src, "팔 액션에 변화율 상한이 있어야 한다"
    assert "rate_limit=P.ARM_TARGET_RATE_LIMIT" in src
    # 레시피의 핵심 성질은 그대로여야 한다.
    assert "scale=0.5" in src and "use_default_offset=True" in src
    # 상한이 관절 속도 한계와 **같은 표**여야 한다. 리터럴을 따로 두면 어긋난다.
    assert P.ARM_TARGET_RATE_LIMIT == P.ARM_VELOCITY_LIMIT, (
        "목표 변화율 상한은 관절 속도 한계에서 파생되어야 한다"
    )
    for expr, rate in P.ARM_TARGET_RATE_LIMIT.items():
        assert rate > 0.0, f"{expr} 상한이 0 이하"


def test_ik_variant_controls_the_same_point_the_reward_measures():
    """★태스크공간 변형은 **보상이 보는 점**을 제어해야 한다.

    보상의 EE 프레임은 `l_hl_gripper_base` + z 오프셋(TCP_OFFSET_IN_BASE_Z)이다.
    IK 액션의 `body_offset` 이 이와 다르면 "제어하는 점"과 "평가받는 점"이 어긋나
    거리 보상이 영영 0 에 안 붙는데, 어디서도 에러가 나지 않는다.

    또한 스케일이 레퍼런스 Franka 값(0.5)이면 안 된다 — dt 0.02 에서 25 m/s 지령이라
    관절공간에서 진단한 포화(지령이 관절 한계의 7 배)를 그대로 재현한다.
    """
    src = (
        Path(__file__).resolve().parents[1] / "grasp_left_ik_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "use_relative_mode=True" in src and 'ik_method="dls"' in src
    assert "P.TCP_OFFSET_IN_BASE_Z" in src, "보상 EE 프레임과 같은 오프셋을 써야 한다"
    assert "scale=P.IK_ACTION_SCALE" in src
    # 위치·회전 상한이 팔의 실제 능력 근처여야 한다(관절 한계 2.175~2.61 rad/s).
    pos = P.IK_ACTION_SCALE[:3]
    rot = P.IK_ACTION_SCALE[3:]
    dt = 0.02
    assert all(v / dt <= 2.0 for v in pos), "TCP 선속도 지령 상한이 2 m/s 를 넘는다"
    assert all(v / dt <= max(P.ARM_VELOCITY_LIMIT.values()) for v in rot), (
        "TCP 각속도 지령 상한이 관절 속도 한계를 넘는다"
    )
    # IK 해는 관절 한계로 clamp 돼야 한다 — 레퍼런스 액션은 그걸 안 한다.
    act_src = (
        Path(__file__).resolve().parents[1] / "grasp_left_actions.py"
    ).read_text(encoding="utf-8")
    assert "soft_joint_pos_limits" in act_src


def test_joint_space_task_is_left_untouched_by_the_ik_variant():
    """★IK 변형은 **별도 등록**이다. 관절공간 태스크의 액션/관측 차원은 그대로여야 한다.

    hdgp 규칙상 obs/action 차원은 명시 요청 없이 바꾸지 않는다. 비교가 성립하려면
    두 태스크가 나란히 살아 있어야 하기도 한다.
    """
    reg = (
        Path(__file__).resolve().parents[1] / "config" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert 'id="open-grip_l_grasp_sensor"' in reg
    assert 'id="open-grip_l_grasp_sensor_ik"' in reg
    cfg = _cfg_source()
    # 관절공간판의 팔 액션은 여전히 관절 위치(변화율 상한 포함)여야 한다.
    assert "RateLimitedJointPositionActionCfg" in cfg
    assert "DifferentialInverseKinematics" not in cfg
