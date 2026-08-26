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
def test_work_surface_matches_env_usd_mesh_points():
    """★작업면 z 는 `env.usd` 의 `top_plate` 메시 점에서 직접 나와야 한다.

    env.usd 는 xformOp 이 하나도 없어 **메시 좌표가 곧 Env 프레임 값**이고, Env 원점은
    로봇 base link 원점이다(사용자 지정). 그래서 top_plate 의 max z 가 작업면이다.

    ⚠ 과거 이 값을 두 번 틀렸다: right/grasp_sensor 의 0.2082 는 컵 bbox 반높이로 역산한
      중간값이고, USD BBoxCache 로 읽으면 extent 에 scale 이 **또** 곱해져 0.2004 가 된다.
      메시 점을 직접 읽으면 두 함정과 무관하다.
    """
    usda = _HDGP / "assets/env/usd/env.usda"
    if not usda.is_file():
        pytest.skip("env.usda 없음")
    txt = usda.read_text(encoding="utf-8")

    def aabb(mesh_name: str):
        m = re.search(
            r'def Mesh "%s"(.*?)point3f\[\] points = \[(.*?)\]\s*\n' % mesh_name, txt, re.S
        )
        assert m, f"{mesh_name} 메시를 못 찾았다"
        vals = re.findall(r"\(([-0-9.eE]+),\s*([-0-9.eE]+),\s*([-0-9.eE]+)\)", m.group(2))
        cols = list(zip(*[[float(c) for c in v] for v in vals]))
        return [(min(c), max(c)) for c in cols]

    # 변환이 없어야 좌표를 그대로 믿을 수 있다 — 나중에 누가 넣으면 이 계약이 깨져야 한다.
    assert "xformOp" not in txt, "env.usd 에 변환이 생겼다 — 메시 좌표를 그대로 못 쓴다"
    assert "metersPerUnit = 1" in txt, "단위가 m 이 아니다(0.01 이면 자산이 100 배 작아진다)"

    top = aabb("top_plate")
    assert math.isclose(P.TABLE_SURFACE_Z, top[2][1], abs_tol=1e-6)
    assert math.isclose(P.WORK_SURFACE_X[0], top[0][0], abs_tol=1e-6)
    assert math.isclose(P.WORK_SURFACE_X[1], top[0][1], abs_tol=1e-6)
    # 로봇이 서는 면과 바닥판
    assert math.isclose(P.ROBOT_MOUNT_Z, aabb("platform")[2][1], abs_tol=1e-6)
    assert math.isclose(P.ENV_FLOOR_Z, aabb("base_plate")[2][1], abs_tol=1e-6)
    # 과거에 틀렸던 두 값이 다시 들어오지 않도록
    assert not math.isclose(P.TABLE_SURFACE_Z, 0.2082, abs_tol=1e-4)
    assert not math.isclose(P.TABLE_SURFACE_Z, 0.2004, abs_tol=1e-4)
    # env 는 로봇 원점에 그대로 붙는다 — 오프셋이 생기면 모든 높이 상수가 어긋난다.
    assert P.ENV_POS == (0.0, 0.0, 0.0)
    src = _cfg_source()
    assert "P.ENV_USD_REL" in src and "rigid_props" not in src.split("P.ENV_USD_REL")[1][:200]


def test_lift_gate_is_measured_from_the_resting_cup_origin():
    """★리프트 임계는 **놓인 컵의 원점 + 4 cm** 여야 한다.

    이 태스크에서 두 번 태운 자리다.
      · 상면(0.200)만 더한 0.24 는 놓인 컵 원점(0.29209)보다 **낮아** 게이트가 상시 열린다
        (test1-r2: lifting 14.63/15 인데 reaching 은 0.007 로 떨어졌다 = 가만히 있는 것이 최적).
      · IK test3 은 `minimal_height 0.27709` 로 같은 함정에 빠져 총보상 149 를 냈지만
        컵을 3.6 mm 만 올렸다.
    08.22 연속 램프에서 하드 게이트로 되돌렸다 — 램프의 근거는 위 IK 런이었는데 그 원인은
    게이트 모양이 아니라 임계값이었고, 제대로 준 관절공간 런(test13/16)은 lift 0.83~0.84 로
    실제로 들어 올렸다.
    """
    assert math.isclose(P.MINIMAL_LIFT_HEIGHT, P.CUP_SPAWN_Z + 0.04, abs_tol=1e-9), (
        "리프트 임계가 '놓인 컵 원점 + 4 cm' 에서 파생되지 않았다"
    )
    assert P.MINIMAL_LIFT_HEIGHT > P.CUP_SPAWN_Z, "임계가 놓인 높이보다 낮다 = 공짜 게이트"
    # 목표 z 하한도 임계 위여야 "먼저 들어라 → 옮겨라" 순서가 유지된다.
    assert P.GOAL_POS_Z[0] > P.MINIMAL_LIFT_HEIGHT

    rsrc = (
        Path(__file__).resolve().parents[1] / "grasp_left_rewards.py"
    ).read_text(encoding="utf-8")
    # ★★08.23 이 단언을 **뒤집었다.** 원래는 "이진 게이트여야 한다" 를 고정하고 있었는데,
    #   Fabrics 트랙 세 런(누적 6,747 epoch, 약 17 억 스텝)에서 lifting 이 정확히 0.0000 이었다.
    #   컵은 +17.2 mm 까지 올라가는데 40 mm 문턱까지 신호가 없어 거기서 멈춘다.
    #   관절공간 test17 이 문턱을 넘은 것은 지령 포화(한계의 7 배)로 컵을 튕겨 올린 우연이었고,
    #   Fabrics 는 그 거친 움직임을 없애려고 넣은 것이라 그 메커니즘이 사라졌다.
    #   → 높이는 **연속 램프**, 근접·자세는 게이트로 남는다.
    assert "obj_pos_w[:, 2] > minimal_height" not in rsrc, "이진 게이트가 되살아났다"
    # ★★fab_test39: 램프의 **기준점**을 원점 z → 컵 **최저점**(`lift_height`)으로 옮겼다.
    #   원점 z 는 기울기에 민감해 컵을 바닥 림으로 피벗시키면 최대 4.61 mm 가 **실제로 오른다**.
    #   그래서 옛 설계는 램프 0 점을 그 위(놓인 높이 +6 mm)에 둬 방어했는데, 그러면 첫 6 mm 가
    #   사구간이라 "접촉했는데 아직 못 든" 상태에 gradient 가 없다.
    #   t38(4000 ep 완주) 실측이 정확히 그 함정이었다 — 컵 최대 상승 +2.9 mm(= 사구간 안),
    #   1 cm 이상 올린 스텝 0.0%. 그 판이 한 것은 리프트가 아니라 **기울이기**였다.
    #   최저점은 기울여도 정확히 0 이므로 사구간 없이 첫 1 mm 부터 신호가 산다.
    assert "lift_height(env, object_cfg) / (minimal_height - P.CUP_SPAWN_Z)" in rsrc, (
        "높이 항이 최저점 기준 연속 램프가 아니다"
    )
    assert "P.CUP_BASE_RADIUS * sin_tilt" in rsrc, (
        "`lift_height` 가 기울기를 보정하지 않는다 — 기울이기가 가짜 리프트로 계산된다"
    )
    # ★08.23 램프에 enclose 를 곱한다 — 순수 램프는 "쳐 올리기" 를 부분 보상해 정책을
    #   주먹으로 고착시켰다(fab_test6: enclose 0.019 · drop% 0.733). 근거는 test_fab_contract.
    assert "lifted * held * (near & upright).float()" in rsrc, (
        "근접·자세 게이트 또는 enclose 인자가 빠졌다"
    )
    # ★공짜 차단: 램프 0 점은 놓인 높이보다 위여야 하고, 컵을 바닥 모서리로 기울여 얻는
    #   최대 상승(CUP_TIP_RISE_MAX)보다도 위여야 한다. 아니면 "흔들기" 가 보상을 받는다.
    assert P.LIFT_RAMP_ZERO_Z > P.CUP_SPAWN_Z + P.CUP_TIP_RISE_MAX, (
        "램프 0 점이 기울임 상한 아래다 — 컵을 흔들기만 해도 보상이 생긴다"
    )
    assert P.LIFT_RAMP_ZERO_Z < P.MINIMAL_LIFT_HEIGHT, "램프 0 점이 상단보다 높다"
    assert "lift_span" not in rsrc, "램프 파라미터가 남아 있다"

    src = _cfg_source()
    assert src.count("P.MINIMAL_LIFT_HEIGHT") >= 3, "게이트를 곱하는 항이 셋 미만이다"
    # 공짜 보상 차단의 나머지 절반은 TCP 근접 게이트다 — 없어지면 test3 던지기가 돌아온다.
    assert "GRASP_MAX_EE_DISTANCE" in src


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
    # ⚠ 08.23 기준 실측을 **갱신했다.** 예전엔 test8 값(0.444 m/s·3.43 rad/s)을 박아
    #   뒀는데, 그 사이 Fabrics 트랙이 0.193 m/s·1.473 rad/s 로 2.3 배 좋아졌다.
    #   옛 값에 맞춘 임계(0.40/3.00)는 현재 영역에서 품질 0.55 로 포화해 "더 멈춰라"는
    #   압력이 사라진다(사용자 관찰: "가만히 있질 못함"). 실측이 바뀌면 임계도 따라간다.
    measured_lin, measured_ang = 0.193, 1.473        # fab_test7 best, 쥐고 있을 때
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


def test_goal_is_the_user_specified_region_not_wider():
    """★이송 목표는 **사용자가 지정한 박스**. 08.25 현재 x±8 **y±9** z±7 cm (워크스페이스 스캔 실측 전역).

    이력: 처음엔 넓은 범위(test12: goal_fine 8%·settle 7.8% 정체) → 점 ±2 cm 로 좁혀
    test17 이 이송까지 성공 → pour 용 목표-조건부 이송을 위해 x±5 y±7 z±5 로 확대 →
    **08.25 사용자 지시(ADR "모드을"=보수적)로 y 를 ±11 로, z 를 ±7 로 확대(도달성 실측이 ±15 를 기각)**.
    ⚠ 옛 정체(test12)는 **전 축이 넓었을 때** 나온 것이다. 이번엔 x 를 그대로 두고 y 만
      넓혔다 — 작업면 Y 는 90cm 로 X(40cm)의 2.25배이고 x 는 테이블 앞모서리까지 10mm
      여유뿐이라 애초에 못 넓힌다. y 확대가 정체를 되살리는지는 fab_test17 이 판정한다.
    """
    assert P.GOAL_JITTER == (0.08, 0.09, 0.07), "사용자 지정 목표 영역이 바뀌었다"
    for jit, (lo, hi), c in zip(
        P.GOAL_JITTER, (P.GOAL_POS_X, P.GOAL_POS_Y, P.GOAL_POS_Z), P.GOAL_POINT
    ):
        assert math.isclose(hi - lo, 2 * jit, abs_tol=1e-9)
        assert math.isclose(0.5 * (lo + hi), c, abs_tol=1e-9)
    # 스폰 자리에 그대로 두는 것이 목표가 되면 "들어서 옮기기"가 성립하지 않는다
    dz = P.GOAL_POS_Z[0] - P.CUP_SPAWN_Z
    assert dz > 0.05, "목표 하한이 스폰 높이와 가까우면 이송을 요구하지 못한다"


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

    # ★★fab_test42 기록된 예외. 사용자가 홈을 책상 위로 올리면서 `l_aj_1 = +0.9` 를 지정했고,
    #   상한 1.3963 대비 여유가 **0.496** 으로 기준에 **3.7 mrad** 못 미친다.
    #   허용하는 근거 두 가지:
    #     ⑴ 지금 학습하는 fab 태스크는 **관절 델타 액션을 쓰지 않는다**(fabric 팜 6D).
    #        이 기준은 관절공간 태스크(t16 계보)의 것이다.
    #     ⑵ 잘리는 양이 액션 반범위의 **0.7%** 다.
    #   ⚠ 예외를 **값과 함께** 못 박는다 — 홈이 더 움직이면 이 테스트가 다시 터져서
    #     재검토를 강제한다(조용한 완화가 아니다).
    ALLOWED = {"l_aj_1": 0.49}       # 관절: 허용 최소 여유

    tight = []
    for name, value in P.LEFT_ARM_HOME_JOINT_POS.items():
        lo, hi = limits[name]
        slack = min(value - lo, hi - value)
        floor = ALLOWED.get(name, action_half_range)
        if slack < floor:
            tight.append(f"{name}={value:+.4f} 여유 {slack:.3f} (하한 {floor})")
    assert not tight, f"한계 여유 부족: {tight}"


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


def _urdf_parent_joint(root, link_name):
    for j in root.iter("joint"):
        ch = j.find("child")
        if ch is not None and ch.get("link") == link_name:
            return j
    return None


_MANIFEST = _ROBOT_URDF.parent / "openarm_tesollo_sensor_rl_manifest.yaml"


@pytest.mark.skipif(not _MANIFEST.is_file(), reason="자산 매니페스트 없음")
def test_tcp_anchor_exists_as_a_rigid_body_in_the_built_asset():
    """★TCP 앵커는 **빌드된 USD 의 강체**여야 한다 — URDF 에 링크가 있는 것만으론 부족하다.

    08.21 08:02 빌드(ffe4239)는 고정조인트를 병합해 `l_hl_gripper_base` 가 강체에서
    사라졌고 태스크가 `body_names.index()` 에서 죽었다. 그때 계약 37개가 **전부 통과**했다 —
    URDF 링크 존재만 봤기 때문이다. 13:49 빌드(81dfcf0)가 병합을 껐지만, 병합 여부는
    **빌드 도구의 정책**이라 URDF 로는 예측할 수 없다. 그래서 산출물의 자기기록인
    매니페스트 `link_order` 를 근거로 삼는다(실측: 57개 = USD 강체 57개와 일치).
    """
    text = _MANIFEST.read_text()
    block = text[text.index("link_order:"):]
    links = re.findall(r"^  - (\S+)$", block, re.M)
    assert len(links) > 10, "link_order 를 못 읽었다"
    assert P.GRIPPER_BASE_BODY in links, (
        f"{P.GRIPPER_BASE_BODY} 가 빌드 산출물의 강체 목록에 없다 — 고정조인트로 병합됐을 수 있다. "
        f"살아남는 링크로 앵커를 옮기고 TCP_OFFSET_IN_BASE_Z 에 병합된 변환을 합산할 것."
    )
    for body in P.GRIPPER_FINGER_BODIES:
        assert body in links, f"{body} 가 강체 목록에 없다"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_tcp_offset_lands_inside_the_fingers():
    """TCP 가 손가락 범위 밖이면 정책이 조준하는 점과 실제 파지점이 어긋난다.

    앵커(`GRIPPER_BASE_BODY`)에서 TCP 까지의 오프셋은 **URDF 고정조인트 체인의 합**이어야
    한다. 현재 체인: l_al_7 --(l_hj_gripper_mount 0.1001)--> l_hl_gripper_base
    --(l_hj_gripper_tcp 0.08)--> l_hl_gripper_tcp = 0.1801.
    손가락 메시는 gripper_base 기준 z∈[+0.0005, +0.0954] 이고 TCP 0.08 은 그 안이다.
    """
    root = ET.parse(_ROBOT_URDF).getroot()
    # 앵커에서 TCP 링크까지 거슬러 올라가며 고정 변환을 합산한다.
    total_z, link = 0.0, "l_hl_gripper_tcp"
    chain = []
    while link != P.GRIPPER_BASE_BODY:
        j = _urdf_parent_joint(root, link)
        assert j is not None, f"{link} 위로 체인이 끊겼다"
        assert j.get("type") == "fixed", (
            f"{j.get('name')} 이 fixed 가 아니다 — 오프셋 상수로 표현할 수 없다"
        )
        o = j.find("origin")
        xyz = [float(v) for v in ((o.get("xyz") if o is not None else "") or "0 0 0").split()]
        assert xyz[0] == 0.0 and xyz[1] == 0.0, f"{j.get('name')} 이 중심선에서 벗어났다"
        rpy = [float(v) for v in ((o.get("rpy") if o is not None else "") or "0 0 0").split()]
        assert all(v == 0.0 for v in rpy), f"{j.get('name')} 에 회전이 있어 z 합산이 무효다"
        total_z += xyz[2]
        chain.append(j.get("name"))
        link = j.find("parent").get("link")
    assert math.isclose(total_z, P.TCP_OFFSET_IN_BASE_Z, abs_tol=1e-6), (
        f"체인 {chain} 합계 {total_z} != TCP_OFFSET_IN_BASE_Z {P.TCP_OFFSET_IN_BASE_Z}"
    )
    # 손가락 메시 범위는 gripper_base 기준이므로 그쪽 오프셋으로 검사한다.
    finger_span = (0.0005, 0.0954)
    tcp_from_gripper_base = 0.08
    assert finger_span[0] < tcp_from_gripper_base < finger_span[1]


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
    near, far = P.WORK_SURFACE_X
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
    assert P.GOAL_POS_X[0] > P.WORK_SURFACE_X[0]
    assert P.GOAL_POS_X[1] < P.WORK_SURFACE_X[1], "목표가 판 앞쪽 밖이다"


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
    # 램프가 완전히 서는 높이(놓인 높이 + span)보다 목표가 위여야 한다.
    assert lo >= P.LIFT_RAMP_ZERO_Z + P.LIFT_RAMP_SPAN
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


def test_gripper_action_commands_both_jaws_not_just_the_drive_joint():
    """★두 조 모두에 지령해야 한다 — USD 에 mimic 이 없다.

    예전 계약은 정반대였다("mimic 이 따라오니 gripper_1 만 지령"). 08.22 자산 재빌드
    (urdf 6d065f7) 후 USD 의 `l_hj_gripper_2` 에는 `PhysicsDriveAPI` 만 있고 mimic API 가
    없다. 액션 대상이 아닌 관절은 PD 목표가 0 이므로 두 번째 조가 닫힌 채 고정되고,
    open 지령에도 조 간격이 **56 mm**(정상 100 mm)에 그쳐 컵(58~88 mm)을 물지 못한다.
    URDF 의 `<mimic>` 태그만 보고 "시뮬에도 있겠지"라고 넘기면 조용히 재발한다.
    """
    src = _cfg_source()
    # ⚠ 08.24 게이트 버전으로 교체됐다. 부분문자열이라 옛 단언이 그대로 통과해
    #   화석이 될 뻔했다 — 명시적으로 게이트 버전을 요구한다.
    assert "BinaryJointPositionActionCfg" in src, (
        "그리퍼가 이진 액션이 아니다"
    )
    assert "P.GRIPPER_JOINT_NAMES" in src, "두 조 모두에 지령해야 한다"
    assert set(P.GRIPPER_JOINT_NAMES) == {"l_hj_gripper_1", "l_hj_gripper_2"}
    # 액추에이터 커버리지도 두 관절 모두 (없으면 무구동 자유이동)
    assert "l_hj_gripper_[1-2]" in src


def test_self_collision_is_enabled():
    """self-collision 을 켠 채 학습한다 (08.22).

    자산이 self-collision-safe 로 재빌드됐고(콜라이더 전부 convexDecomposition,
    감사 WARN 쌍 전부 filtered_pairs), 이 태스크 홈에서 랜덤 액션 롤아웃을 돌려도
    팔 링크 자기충돌력이 0.00 N 이다(`probe_selfcollision_gate.py`).
    ⚠ 폐기된 ABORTED 홈에서는 `l_al_5↔l_al_7` 이 5.4 kN 으로 유령접촉한다 —
      **홈이 다르면 자기충돌 결론이 이식되지 않는다.**
    """
    src = _cfg_source()
    assert "enabled_self_collisions=True" in src
    assert "disable_gravity=False" in src, "중력도 켠 채 학습한다"


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
    # ★물려받은 term 의 **weight** 는 재정의하지 않는다. 레시피가 보존되는 이유가 그 비율이다.
    #   ⚠ 08.22 이 계약을 **좁혔다**. 원래는 `reaching_object` 의 재정의 자체를 금지했는데,
    #     그 금지가 실제 버그를 고정하고 있었다: 레퍼런스 도달 보상은 컵 **원점**을 겨냥하고,
    #     우리 shaker 는 원점(상면 +92 mm)이 그리퍼 통과 대역(+10~85 mm) **밖**이라
    #     보상이 들어갈 수 없는 높이를 가리켰다(G3 실측 진입 오차 100.2 mm).
    #     → 금지 대상을 "재정의"에서 **"weight/std 변경"**으로 바꾼다. 목표점 교정은 허용하되
    #       레퍼런스 비율(1.1 / std 0.1)은 그대로여야 한다.
    #     같은 오해를 공유한 테스트는 버그를 막지 못한다 — 이 파일에서 세 번째다.
    frozen_weight = (
        "lifting_object", "object_goal_tracking",
        "object_goal_tracking_fine_grained", "action_rate", "joint_vel",
    )
    for name in frozen_weight:
        assert f"self.rewards.{name} = " not in src, f"{name} 을 재정의하지 말 것"
    if "self.rewards.reaching_object = " in src:
        blk = src[src.index("self.rewards.reaching_object = "):]
        blk = blk[: blk.index(")\n\n")]
    # ★★fab_test32: 접근 보상을 agnostic 트랙 이식본(`approach_opposed`, weight 2.0)으로
    #   교체했다. 구 계약은 `1 − tanh(d/std)` 전제였고 그 함수는 더 이상 쓰지 않는다.
        assert "weight=P.APPROACH_WEIGHT" in blk and '"sharpness"' in blk, (
            "접근 보상이 이식본 규약이 아니다 — weight 는 APPROACH_WEIGHT, "
            "커널은 exp(−sharpness·(d_palm+d_side))"
        )
    # 판정 게이트를 늘리는 term 은 여전히 금지 — test6/test7 에서 학습을 죽였다.
    # 신설: grasp_pose · settled_at_goal · cup_between_jaws · grip_closure_when_enclosed
    #      · gate_rate(진단 weight 0.001) · dwell_at_goal(fab_test13 — 순회 국소최적을
    #        가르는 보너스, 게이트 아님) + 도달 목표점 교정 1
    assert src.count("RewTerm(") <= 7, "신설 term 이 예상보다 많다"


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


def test_ik_joint_solution_is_rate_limited_too():
    """★TCP 변위 상한(scale)만 묶으면 부족하다 — IK 가 관절로 푸는 단계가 안 묶인다.

    test4(램프판, epoch 1100) 실측: 적용된 관절 목표 변화 **2.17 rad/s** 로 관절 속도
    한계(2.175)에 정확히 포화, 방향 반전 **49.3%**, jaw 수평 이탈 **32.4°**,
    그리퍼 개도 최소 **16.9 mm**(컵 지름 58 mm 를 감쌌다면 30.2 mm 에서 막혀야 한다)
    = 떨면서 접근해 **한 번도 컵을 물지 못했다**.

    관절공간 판에는 같은 상한을 넣어 두고 IK 판에는 빠뜨렸던 것이다. 같은 표에서
    파생되어야 한다 — 액추에이터 한계를 바꿨을 때 한쪽만 어긋나면 안 된다.
    """
    src = (
        Path(__file__).resolve().parents[1] / "grasp_left_ik_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "rate_limit=P.ARM_TARGET_RATE_LIMIT" in src, "IK 액션에 변화율 상한이 없다"
    act_src = (
        Path(__file__).resolve().parents[1] / "grasp_left_actions.py"
    ).read_text(encoding="utf-8")
    # 관절공간판과 IK판 **양쪽** 모두 상한을 적용해야 한다.
    assert act_src.count("_max_step_delta") >= 4
    # ★기준은 **직전 목표**여야 한다. 현재 관절 기준으로 묶으면 PD 오차가 묶이고, 그건
    #   곧 토크 천장이다(400 N·m/rad × 0.0261 rad = 10.44 N·m < effort 40/27). test5 가
    #   그래서 364 epoch 동안 lift/goal/pose/settle/drop 전부 0.000 이었다 — 컵까지는
    #   가는데(reaching 0.94, TCP–컵 15 mm) 들지 못했다.
    assert "joint_pos_des - self._prev_target" in act_src
    assert "joint_pos_des - joint_pos, min=" not in act_src, (
        "현재 관절 기준 클램프가 되살아났다 — 토크가 갇힌다"
    )
    # apply_actions 는 물리 스텝마다 불리므로 상한도 물리 스텝 기준이어야 한다.
    assert "physics_dt" in act_src


def test_relative_ik_seeds_from_previous_target_and_caps_windup_by_effort():
    """★★relative IK 의 해는 `씨앗 + J⁺·Δ` 다. 씨앗을 **현재 관절**로 주면 Δ=0 일 때
    목표가 처지는 팔을 그대로 따라가 복원력이 사라진다.

    스크립트 지령 실측(정책 아님, 4 초):
        씨앗 = 현재 관절 : 지령 0 에 TCP **−111.5 mm** · +z 절반에도 **−11.4 mm**
                          · −z 최대인데 **+25.8 mm**(모순)
        씨앗 = 직전 목표 : 지령 0 에 **−8.8 mm** · +z 절반 **+176.5 mm** · −z 최대 −25.9 mm
    앞의 것으로는 정책이 제자리를 지키는 데만 +z 권한의 절반 이상을 쓴다. test3·test4·
    test5 가 전부 그 위에서 돌았다.

    그리고 windup 상한은 **effort/강성**에서 나와야 한다. 속도 한계로 잡으면
    (v·dt = 0.0261 rad) 토크가 400×0.0261 = 10.44 N·m 로 잘려 effort 한계(40/27) 아래가
    되고, test5 가 364 epoch 동안 lift·goal·pose·settle·drop 전부 0.000 이었다.
    """
    act_src = (
        Path(__file__).resolve().parents[1] / "grasp_left_actions.py"
    ).read_text(encoding="utf-8")
    assert "jacobian, self._prev_target" in act_src, "IK 씨앗이 직전 목표가 아니다"
    assert "self._max_tracking_error" in act_src

    # 상한이 effort/강성 에서 파생돼야 한다 — 리터럴이면 액추에이터를 바꿨을 때 어긋난다.
    for expr, val in P.ARM_IK_MAX_TRACKING_ERROR.items():
        assert math.isclose(val * P.ARM_IK_STIFFNESS, val * P.ARM_IK_STIFFNESS)
        assert 0.0 < val < 0.5
    # j5~7 은 effort 7 N·m 라 가장 작아야 한다.
    assert P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[5-7]"] < P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[1-2]"]
    # 속도 한계로 잡던 값(v·dt≈0.026)으로 되돌아가면 j1-2 상한이 그 근처로 내려온다.
    assert P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[1-2]"] > 0.05, "토크가 갇힌다"


def test_lift_reward_is_gated_by_contact_not_by_height():
    """★★리프트 보상의 게이트는 **접촉**이어야 한다 — 높이가 아니라.

    논문 Fig.3 이 `μ·r_lift` 로 쓰고 본문이 못 박는다: *"Once the cup reaches a certain
    height threshold, the lift reward **ceases to accumulate**"* → 높이는 **여는 하한이
    아니라 끊는 상한**이다. 구 `_held` 는 높이가 하한이라 t22~t40 열아홉 판 내내 0 이었다.

    ★fab_test41 이후 fab 태스크의 리프트 항은 `stage_lift` 다. `object_is_held_and_lifted`
      는 **관절공간 태스크(t16 계보 positive control)** 가 계속 쓰는 항이고, 그쪽 씬에는
      접촉 센서가 없으므로 `sensor_names` 가 비면 구 `_held` 로 되돌아간다(그게 옳다).
    """
    rsrc = (
        Path(__file__).resolve().parents[1] / "grasp_left_rewards.py"
    ).read_text(encoding="utf-8")

    # fab 태스크: stage_lift 가 접촉 게이트여야 한다
    body = rsrc.split("def stage_lift(")[1].split("\ndef ")[0]
    assert "s.mu * s.U_tol * s.H" in body, "fab 리프트가 `μ × 자세 × 높이진척` 이 아니다"
    assert "s.nu" not in body, "fab 리프트가 아직 높이 게이트(ν) 뒤에 있다"

    # 높이는 컵 **최저점** 기준이어야 한다 — 원점 z 는 기울여서 4.61 mm 를 위조한다
    ssrc = (
        Path(__file__).resolve().parents[1] / "grasp_left_stages.py"
    ).read_text(encoding="utf-8")
    assert "s.lift_h = rewards.lift_height(env)" in ssrc, "높이가 최저점 기준이 아니다"
    assert "s.H = (s.lift_h / P.STAGE_LIFT_REF_M).clamp(0.0, 1.0)" in ssrc, (
        "리프트 진척이 목표에서 포화하지 않는다"
    )

    # 관절공간 태스크: 센서가 없으면 구 거동으로 안전 복귀해야 한다
    old = rsrc.split("def object_is_held_and_lifted(")[1].split("\ndef ")[0]
    assert "if not sensor_names:" in old, (
        "센서 없는 태스크(관절공간)에서 KeyError 로 죽는다 — fab_test42 스모크에서 실제로 터졌다"
    )

def test_grasp_pose_shapes_the_bite_before_the_lift():
    """★★`grasp_pose` 는 리프트 **전에** 물기 자세를 만들어야 한다.

    이 항은 올바른 물기 자세를 만드는 유일한 gradient 인데 `_held`(= 리프트 성립) 뒤에
    갇혀 있었다. t38 4000 ep 완주 실측이 그 대가다:
        grasp_pose 0.00001 · TCP z↔컵 z **49.8°**(올바름 90°) · jaw 수평이탈 **37.6°**
        · lateral **62.4 mm**(`grasp_ok` 문턱 30 mm) · 액션 게이트 개방률 **< 0.5%**
    물기가 비스듬해 게이트가 안 열리고 → 못 들고 → 물기를 고칠 신호가 안 온다.

    ⚠ 게이트를 **그냥 제거하면 안 된다.** `jaw_level_quality` 는 로봇 자세만 보고
      `upright` 는 컵이 서 있기만 해도 1.0 이라, 무게이트면 아무 데서나 그리퍼를 수평으로
      들면 만점이다(reward-audit Check 2). 컵 의존 게이트가 반드시 남아야 한다.
    """
    rsrc = (
        Path(__file__).resolve().parents[1] / "grasp_left_rewards.py"
    ).read_text(encoding="utf-8")
    body = rsrc.split("def held_with_good_pose")[1].split("\ndef ")[0]
    assert "_held(" not in body, "`grasp_pose` 가 아직 리프트 게이트 뒤에 있다"
    assert "gate = grasp_quality(" in body, (
        "`grasp_pose` 의 게이트가 컵 의존(`grasp_quality`)이 아니다 — 무게이트면 해킹면이 생긴다"
    )


def test_cup_tipping_is_penalty_plus_truncation():
    """★fab_test43: 전도는 **벌점 + `truncated`** 다 (사용자 결정).

    구 계약은 "전도는 `terminated` 여야 한다"였다. 그 계약 아래 t42 를 1218 epoch
    돌린 실측이 계약을 뒤집었다 — 종료가 살아 있으면 정책은 **컵 근처에 가지 않는
    것**을 배운다(컵 앞 리턴 0.72 vs 물러섬 2.95 = 4.1배). λ 가 내내 0.0002 였다.

    지금 계약 두 가지:
      · 기울기는 `rewards.tip` 벌점이 연속으로 문다(8° 마진, 60°에서 1.5).
      · 60° 를 넘으면 **truncated** 로 리셋한다 — 쓰러진 컵은 다시 못 세우므로
        남은 스텝이 낭비 표본이다. `value_bootstrap` 이 γ·V(s) 를 얹어 자살 이득은 없다.
    """
    fab_src = (
        Path(__file__).resolve().parents[1] / "grasp_left_fab_env_cfg.py"
    ).read_text(encoding="utf-8")
    block = fab_src.split("self.terminations.object_tipped = DoneTerm(")[1].split(")")[0]
    assert "time_out=True" in block, "전도가 terminated 다 — 벌점 체계에서 자살 경로가 된다"
    assert "max_tilt_deg" in block, "전도 임계가 없다"
    assert "self.rewards.tip = RewTerm(" in fab_src, "전도 벌점 항이 없다"
    assert P.STAGE_TIP_WEIGHT < 0.0, "전도 항이 벌점이 아니다"
    assert P.STAGE_TIP_MARGIN_DEG >= 8.0, (
        f"전도 벌점 마진 {P.STAGE_TIP_MARGIN_DEG}° 가 성공 파지 흔들림(4.1°)에 너무 가깝다"
    )
    assert P.STAGE_TIP_MARGIN_DEG < P.OBJECT_TIP_MAX_DEG, (
        "벌점이 절단 임계보다 늦게 시작하면 연속 신호가 없다"
    )


def test_lift_and_lateral_diagnostics_are_logged():
    """★t38 은 이 둘을 TB 에서 못 봐 4000 epoch 내내 사후 프로브로만 확인할 수 있었다.

    · `diag_lift_height` — 컵 최저점 상승. `lifting_object` 가 0 일 때 "안 들었다"인지
      "게이트가 막았다"인지를 가른다.
    · `diag_jaw_lateral` — `grasp_ok` 의 1차 조건이자 D2 가 겨냥하는 값(t38 62.4 mm).
    """
    fab_src = (
        Path(__file__).resolve().parents[1] / "grasp_left_fab_env_cfg.py"
    ).read_text(encoding="utf-8")
    for name in ("diag_lift_height", "diag_jaw_lateral"):
        assert f"self.rewards.{name} = RewTerm(" in fab_src, f"{name} 진단항이 없다"
