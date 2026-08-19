"""기하·자산 계약 고정 (Isaac 불필요).

프로브가 실측으로 정한 값들이 코드에서 조용히 어긋나는 것을 막는다. 특히:
  · 프리셋 조인트 이름이 **실제 USD/URDF 에 존재**하는가
    (없는 이름을 init_state 에 넣으면 Isaac Lab resolve_matching_names_values 가 예외)
  · 파지 높이가 그리퍼 통과 가능 대역 안인가
  · fabric URDF 의 cspace 가 팔 7 DOF 인가 (손이 fixed 로 굳혀졌는가)
"""

import ast
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openarm import OPENARM_ROOT_DIR
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

_HDGP = Path(OPENARM_ROOT_DIR).resolve().parents[2]
_ROBOT_URDF = _HDGP / "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
_FABRIC_URDF = (
    _HDGP / "source/FABRICS/src/fabrics_sim/models/robots/urdf"
    / P.FABRIC_ROBOT_DIR / f"{P.FABRIC_ROBOT_DIR}.urdf"
)
_CFG_SRC = Path(__file__).resolve().parents[1] / "grasp_left_env_cfg.py"


def _cfg_literals() -> dict:
    tree = ast.parse(_CFG_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspLeftGripperEnvCfg")
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        try:
            out[node.target.id] = ast.literal_eval(node.value)
        except ValueError:
            pass
    return out


def _urdf_names(path: Path) -> tuple[set[str], set[str]]:
    root = ET.parse(path).getroot()
    joints = {j.get("name") for j in root.iter("joint")}
    links = {l.get("name") for l in root.iter("link")}
    return joints, links


# ---------------------------------------------------------------------------
# 조인트·링크 이름이 실제 자산에 존재하는가
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_every_preset_joint_exists_in_robot_urdf():
    joints, _ = _urdf_names(_ROBOT_URDF)
    named = (
        list(P.LEFT_ARM_JOINT_NAMES)
        + list(P.LEFT_GRIPPER_JOINT_NAMES)
        + list(P.RIGHT_REST_JOINT_POS)
    )
    missing = [n for n in named if n not in joints]
    assert not missing, f"URDF 에 없는 조인트: {missing}"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_gripper_bodies_exist_but_tcp_is_not_a_body():
    """`l_hl_gripper_tcp` 는 URDF 에는 있지만 physics USD 에서 강체로 병합돼 사라진다.

    그래서 preset 은 TCP 를 body 로 조회하지 않고 gripper_base + z 오프셋으로 계산한다.
    """
    _, links = _urdf_names(_ROBOT_URDF)
    assert P.GRIPPER_BASE_BODY in links
    for body in P.GRIPPER_FINGER_BODIES:
        assert body in links
    assert "l_hl_gripper_tcp" not in P.GRIPPER_FINGER_BODIES
    assert P.GRIPPER_BASE_BODY != "l_hl_gripper_tcp"


@pytest.mark.skipif(not _ROBOT_URDF.is_file(), reason="로봇 URDF 없음")
def test_gripper_stroke_matches_urdf_limit():
    root = ET.parse(_ROBOT_URDF).getroot()
    j = next(x for x in root.iter("joint") if x.get("name") == "l_hj_gripper_1")
    lim = j.find("limit")
    assert j.get("type") == "prismatic"
    assert math.isclose(float(lim.get("lower")), P.GRIPPER_CLOSED_POS, abs_tol=1e-6)
    assert math.isclose(float(lim.get("upper")), P.GRIPPER_OPEN_POS, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 좌우 대칭
# ---------------------------------------------------------------------------
def test_idle_right_arm_is_sign_mirror_of_left_home():
    """유휴 팔이 파지 팔 홈의 부호 미러여야 장면이 좌우 대칭이고 양팔 pour 와 이어진다."""
    for left_name, s in zip(P.LEFT_ARM_JOINT_NAMES, P._ARM_MIRROR_SIGN):
        right_name = left_name.replace("l_aj_", "r_aj_")
        assert math.isclose(
            P.RIGHT_ARM_REST_JOINT_POS[right_name],
            s * P.LEFT_ARM_HOME_JOINT_POS[left_name],
            abs_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# 파지 기하 (프로브 실측 고정)
# ---------------------------------------------------------------------------
def test_grasp_height_inside_gripper_feasible_band():
    """shaker 는 계단형 원뿔이라 이 대역 밖에서는 그리퍼 개구(84.5mm)로 컵이 안 들어간다."""
    lo, hi = P.GRASP_HEIGHT_BAND
    assert lo <= P.GRASP_HEIGHT_ABOVE_TABLE <= hi


def _cfg_field_source(name: str) -> str:
    """cfg 필드 기본값의 소스 표현. 리터럴이 아니라 **무엇을 참조하는지** 를 본다."""
    tree = ast.parse(_CFG_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspLeftGripperEnvCfg")
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.unparse(node.value)
    raise KeyError(f"cfg 필드 없음: {name}")


@pytest.mark.parametrize(
    "field,preset_name",
    [
        ("grasp_height_above_table", "GRASP_HEIGHT_ABOVE_TABLE"),
        ("table_surface_z", "TABLE_SURFACE_Z"),
        ("object_spawn_z", "CUP_SPAWN_Z"),
        ("object_spawn_x_center", "CUP_SPAWN_X_CENTER"),
        ("object_spawn_y_center", "CUP_SPAWN_Y_CENTER"),
    ],
)
def test_cfg_reuses_preset_constants_instead_of_literals(field, preset_name):
    """cfg 가 프리셋을 **참조**해야 프로브가 값을 갱신했을 때 자동으로 따라간다.

    리터럴로 베껴 넣으면 프리셋만 고쳤을 때 조용히 어긋난다 —
    저장소에 반복된 "리터럴 하드코딩" 실패 패턴이다.
    """
    assert _cfg_field_source(field) == preset_name


def test_cup_spawn_z_puts_bottom_on_table():
    """컵 원점은 bbox 반높이가 아니라 **메시 bottom** 에서 역산해야 테이블에 정확히 앉는다."""
    assert math.isclose(P.CUP_SPAWN_Z, P.TABLE_SURFACE_Z + P.CUP_BOTTOM_TO_ORIGIN, abs_tol=1e-9)


def test_grasp_axes_are_orthonormal_and_jaw_is_horizontal():
    width, jaw, approach = P.grasp_axes()
    for v in (width, jaw, approach):
        assert math.isclose(sum(c * c for c in v), 1.0, abs_tol=1e-9)
    assert abs(sum(a * b for a, b in zip(jaw, approach))) < 1e-9
    # jaw 가 수평이어야 두 접촉점이 컵 단면 지름 양끝에 놓인다 (force-closure 최소조건)
    assert abs(jaw[2]) < 1e-9


def test_grasp_euler_reproduces_grasp_axes():
    """GRASP_PALM_EULER_ZYX_DEG 는 (θ, φ) 에서 유도된 값이다 — 둘이 어긋나면 안 된다."""
    ez, ey, ex = (math.radians(v) for v in P.GRASP_PALM_EULER_ZYX_DEG)

    def matmul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    cz, sz = math.cos(ez), math.sin(ez)
    cy, sy = math.cos(ey), math.sin(ey)
    cx, sx = math.cos(ex), math.sin(ex)
    R = matmul(matmul([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
                      [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
               [[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    width, jaw, approach = P.grasp_axes()
    for col, expected in enumerate((width, jaw, approach)):
        for row in range(3):
            assert math.isclose(R[row][col], expected[row], abs_tol=1e-9)


def test_palm_rotation_bounds_avoid_euler_gimbal():
    """euler_zyx 는 ey=±90° 에서 퇴화한다 — 클램프 범위가 거기 닿으면 안 된다.

    기준자세 ey 가 75° 라 액션(±20°)만으로도 90° 를 넘길 수 있어 별도 상한을 둔다.
    퇴화 구간에서는 같은 회전이 여러 euler 로 갈려 클램프가 의미를 잃는다.
    """
    lo = P.palm_pose_mins()
    hi = P.palm_pose_maxs()
    ey_lo, ey_hi = math.degrees(lo[4]), math.degrees(hi[4])
    assert -89.0 < ey_lo < 89.0
    assert -89.0 < ey_hi < 89.0
    assert ey_lo < P.GRASP_PALM_EULER_ZYX_DEG[1] < ey_hi


def test_fabric_default_config_matches_preset_home():
    """Fabrics cspace attractor 가 당기는 자세 = 이 태스크의 홈이어야 한다.

    두 값이 어긋나면 attractor 가 파지 자세를 방해한다 — 그게 첫 Isaac 실패의 원인이었다
    (홈이 파지 자세군 밖이라 jaw 가 28.5° 기울었다). 파일이 둘로 나뉘어 있어 조용히 어긋난다.
    """
    src = (_HDGP / "source/FABRICS/src/fabrics_sim/fabrics/openarm_tesollo_pose_fabric.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    values = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", None) == "_GRIPPER_LEFT_DEFAULT_CONFIG"):
            values = ast.literal_eval(node.value)
    assert values is not None, "_GRIPPER_LEFT_DEFAULT_CONFIG 를 못 찾았다"
    home = [P.LEFT_ARM_HOME_JOINT_POS[n] for n in P.LEFT_ARM_JOINT_NAMES]
    assert len(values) == len(home)
    for got, want in zip(values, home):
        assert math.isclose(got, want, abs_tol=1e-6), f"{values} != {home}"


def test_spawn_box_x_is_not_the_naive_mirror_of_right_task():
    """우측 x=0.30 을 그대로 미러하면 낮은 파지점에 팔이 못 미친다(실측 잔차 11~20mm)."""
    assert P.CUP_SPAWN_X_CENTER < 0.30


# ---------------------------------------------------------------------------
# fabric 자산
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _FABRIC_URDF.is_file(), reason="fabric URDF 미생성")
def test_fabric_cspace_is_arm_only():
    """손 20관절이 fixed 로 굳혀져야 Fabrics cspace 가 팔 7 DOF 가 된다."""
    root = ET.parse(_FABRIC_URDF).getroot()
    revolute = [j.get("name") for j in root.iter("joint") if j.get("type") == "revolute"]
    assert len(revolute) == 7, f"revolute {len(revolute)}개: {revolute}"


@pytest.mark.skipif(not _FABRIC_URDF.is_file(), reason="fabric URDF 미생성")
def test_fabric_keeps_frame_names_required_by_params_yaml():
    """fabric_params 의 프레임 리스트가 이름 하드코딩이라 우측과 동일해야 한다."""
    _, links = _urdf_names(_FABRIC_URDF)
    required = {"palm_link", "palm_x", "palm_x_neg", "palm_y", "palm_y_neg",
                "palm_z", "palm_z_neg", "palm_link_sphere2"}
    required |= {f"rl_dg_{i}_tip" for i in range(1, 6)}
    missing = required - links
    assert not missing, f"fabric URDF 프레임 누락: {sorted(missing)}"


def test_cfg_points_at_the_left_gripper_fabric():
    cfg = _cfg_literals()
    assert cfg["fabric_robot_dir"] == P.FABRIC_ROBOT_DIR
    assert "sensor_left_gripper" in cfg["fabric_robot_dir"]
