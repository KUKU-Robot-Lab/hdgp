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


def test_minimal_lift_height_is_measured_from_the_resting_cup_origin():
    """★★가장 위험한 함정. `mdp.object_is_lifted` 는 물체 **root 원점**의 절대 z 를 본다.

    기준선은 테이블 상면이 아니라 **컵이 놓여 있을 때의 원점 z**(= CUP_SPAWN_Z)다.
    레퍼런스가 상면 기준으로 맞는 건 큐브 원점이 기하 중심이라서일 뿐이고, shaker 는
    원점이 바닥에서 92 mm 위라 상면만 더하면 임계가 **놓인 상태보다 낮아진다**.

    실제로 이 실수로 한 번 학습을 태웠다(test1-r2): 임계 0.255 < 놓인 원점 0.30709 라
    lifting 보상이 상시 1(14.63/15.0), goal 게이트도 늘 열려 "가만히 있기"가 최적이 됐고
    reaching_object 가 0.024 → 0.007 로 떨어졌다.
    """
    assert P.MINIMAL_LIFT_HEIGHT > P.CUP_SPAWN_Z, (
        "임계가 놓인 컵의 원점보다 낮다 — lifting 보상이 상시 1 이 된다"
    )
    assert math.isclose(
        P.MINIMAL_LIFT_HEIGHT, P.CUP_SPAWN_Z + P.LIFT_HEIGHT_ABOVE_TABLE, abs_tol=1e-9
    )
    # 옛 계산식(상면 기준)으로 되돌아가지 않도록 못을 박는다
    assert not math.isclose(
        P.MINIMAL_LIFT_HEIGHT, P.TABLE_SURFACE_Z + P.LIFT_HEIGHT_ABOVE_TABLE, abs_tol=1e-6
    )
    # cfg 가 실제로 이 값을 세 군데 전부에 넣는지
    src = _cfg_source()
    assert src.count("P.MINIMAL_LIFT_HEIGHT") >= 3


def test_drop_threshold_is_below_the_resting_cup_but_catches_a_fallen_one():
    """낙하 종료는 컵이 **테이블 밖으로** 떨어졌을 때만 발화해야 한다.

    놓인 원점(0.30709)보다 낮고, 바닥에 선 컵의 원점(= bottom→원점 0.09209)보다는 높아야
    한다. 테이블 위에서 넘어진 컵(원점 ≈ 상면 + 반경)은 잡지 않는다 — lift 레시피는
    넘어짐을 보지 않으므로 여기서 종료시키면 학습 신호만 끊긴다.
    """
    assert P.OBJECT_DROP_HEIGHT < P.CUP_SPAWN_Z
    assert P.OBJECT_DROP_HEIGHT > P.CUP_BOTTOM_TO_ORIGIN
    tipped_on_table = P.TABLE_SURFACE_Z + 0.044      # 옆으로 누운 컵의 원점 근사(반경)
    assert P.OBJECT_DROP_HEIGHT < tipped_on_table


def test_object_drop_threshold_is_below_table():
    assert P.OBJECT_DROP_HEIGHT < P.TABLE_SURFACE_Z


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
    assert P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE >= P.SPAWN_X_SAFE_MIN
    # 목표도 같은 이유로 팔이 점유한 공간 밖이어야 한다.
    assert P.GOAL_POS_X[0] >= P.SPAWN_X_SAFE_MIN


def test_spawn_center_is_no_longer_the_left_grasp_v1_position():
    """반전 기록: 처음엔 `tesollo/left/grasp_v1`(x 0.30, y 0.20)을 따랐다.

    그 자리는 lift 홈 자세에서 **팔이 점유한 공간**이라 쓸 수 없다. x 만 앞으로 옮겼고
    y 는 그대로 두었다(스윕에서 y 는 관통에 무관했다).
    """
    assert P.CUP_SPAWN_X_CENTER > 0.30
    assert math.isclose(P.CUP_SPAWN_Y_CENTER, 0.20, abs_tol=1e-9)


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
    # reward weight 를 새로 정의하지 않는다
    assert "RewTerm(" not in src, "보상 term 을 재정의하지 말 것 — 물려받은 weight 를 쓴다"


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
