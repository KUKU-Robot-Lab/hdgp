"""RobotProfile 계약 — Isaac 앱 없이 검증 가능한 것 전부.

robot-agnostic 태스크의 합격 조건: 태스크 코드에 조인트/바디 이름이 없고,
로봇 추가 = 프로필 1개 추가. 이 테스트가 그 계약을 고정한다.
"""

import re
from pathlib import Path

import pytest

from openarm.agnostic.tasks.grasp_lift.robot_profiles import PROFILES

_TASK_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profile_fields_complete(name):
    p = PROFILES[name]
    assert p.usd_relpath.endswith(".usd")
    assert p.num_arm_joints > 0
    assert p.num_hand_joints > 0
    re.compile(p.arm_joint_regex)
    re.compile(p.hand_joint_regex)
    assert p.palm_body
    assert p.fingertip_bodies
    assert p.finger_sensor_bodies
    assert p.init_joint_pos
    assert p.actuator_specs


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_contact_groups_are_opposing(name):
    """대향 게이트가 성립하려면 A/B 그룹이 비어있지 않고 서로 겹치지 않아야 한다."""
    p = PROFILES[name]
    fingers = set(p.finger_sensor_bodies)
    a, b = set(p.contact_group_a), set(p.contact_group_b)
    assert a and b
    assert a.isdisjoint(b), f"{name}: 대향 그룹 겹침 {a & b}"
    assert (a | b) <= fingers, f"{name}: 센서 없는 손가락이 그룹에 있음 {(a | b) - fingers}"


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_usd_asset_exists(name):
    p = PROFILES[name]
    hdgp_root = _TASK_DIR.parents[5]
    assert (hdgp_root / "assets" / p.usd_relpath).exists(), p.usd_relpath


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_arm_hand_regex_disjoint_on_init_pose(name):
    """arm/hand regex 가 같은 관절을 이중으로 잡으면 액션이 충돌한다."""
    p = PROFILES[name]
    arm_re, hand_re = re.compile(p.arm_joint_regex), re.compile(p.hand_joint_regex)
    both = [j for j in p.init_joint_pos if arm_re.fullmatch(j) and hand_re.fullmatch(j)]
    assert not both, f"{name}: arm∩hand regex 겹침 {both}"


def test_task_code_has_no_robot_names():
    """robot-agnostic 핵심 계약: env/reward 코드에 로봇 조인트/바디 이름 하드코딩 금지."""
    banned = re.compile(r'"(r|l)_(aj|hj|hl)_|\'(r|l)_(aj|hj|hl)_')
    for fname in ("grasp_lift_env.py", "rewards.py", "grasp_lift_env_cfg.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        m = banned.search(code)
        assert m is None, f"{fname} 에 로봇 이름 하드코딩: {m.group(0)}"


def test_env_reward_import_free_of_profiles_constants():
    """env 는 PROFILES 조회 외에 특정 프로필 상수를 import 하지 않는다."""
    src = (_TASK_DIR / "grasp_lift_env.py").read_text(encoding="utf-8")
    assert "TESOLLO_RIGHT" not in src and "GRIPPER_LEFT" not in src


# ---------------------------------------------------------------------------
# Fabrics 전환(2026-08-22) 계약
# ---------------------------------------------------------------------------
_FABRIC_URDF_DIR = (
    _TASK_DIR.parents[4] / "FABRICS" / "src" / "fabrics_sim" / "models" / "robots" / "urdf"
)


@pytest.mark.parametrize("name", list(PROFILES))
def test_fabric_asset_exists(name):
    """fabric_class 가 있으면 URDF 자산이 실제로 있어야 한다(조용한 폴백 금지)."""
    p = PROFILES[name]
    if p.fabric_class is None:
        assert p.fabric_robot_dir is None, f"{name}: class 없는데 robot_dir 만 있다"
        return
    assert p.fabric_robot_dir, f"{name}: fabric_class 만 있고 robot_dir 이 없다"
    urdf = _FABRIC_URDF_DIR / p.fabric_robot_dir / f"{p.fabric_robot_dir}.urdf"
    assert urdf.is_file(), f"{name}: fabric URDF 없음 {urdf}"


@pytest.mark.parametrize("name", list(PROFILES))
def test_fabric_joint_order(name):
    """★articulation(depth-major) ↔ fabric URDF(finger-major) 순서 계약.

    틀리면 fabric 이 엉뚱한 손 자세로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다.
    """
    p = PROFILES[name]
    if p.fabric_class is None:
        return
    order = p.fabric_joint_order
    assert len(order) == p.num_arm_joints + p.num_hand_joints, f"{name}: 길이 불일치"
    assert len(set(order)) == len(order), f"{name}: 중복 관절"
    arm_re, hand_re = re.compile(p.arm_joint_regex), re.compile(p.hand_joint_regex)
    for j in order[: p.num_arm_joints]:
        assert arm_re.fullmatch(j), f"{name}: 앞 {p.num_arm_joints}개는 팔이어야 한다 ({j})"
    for j in order[p.num_arm_joints:]:
        assert hand_re.fullmatch(j), f"{name}: 뒤는 손이어야 한다 ({j})"
    missing = [j for j in order if j not in p.init_joint_pos]
    assert not missing, f"{name}: init_joint_pos 에 없는 관절 {missing}"


@pytest.mark.parametrize("name", list(PROFILES))
def test_palm_box_sane(name):
    """palm 워크스페이스 박스는 리프트 여유를 덮어야 한다."""
    p = PROFILES[name]
    if p.fabric_class is None:
        return
    lo, hi = p.palm_box_min, p.palm_box_max
    assert all(a < b for a, b in zip(lo, hi)), f"{name}: palm_box min>=max"
    assert hi[2] - lo[2] >= 0.15, f"{name}: z 여유가 goal_height_offset(0.15) 보다 작다"


def test_no_diff_ik_left_behind():
    """diff-IK 재도입 방지 — 복원력 0 이 전환의 이유였다."""
    src = (_TASK_DIR / "grasp_lift_env.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    for banned in ("DifferentialIKController", "get_jacobians"):
        assert banned not in code, f"diff-IK 잔재: {banned}"


def test_truncation_reset_paired_with_value_bootstrap():
    """전도/낙하 truncation 리셋 ↔ value_bootstrap 짝 계약.

    bootstrap 없는 truncation 은 termination 과 같아져(미래 보상 절벽) 접근 회피
    학습(agn_test2)이 재발한다. env 가 truncation 리셋을 쓰는 한 yaml 의
    value_bootstrap 은 True 여야 한다(main + central_value 둘 다).
    """
    import yaml
    env_src = (_TASK_DIR / "grasp_lift_env.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in env_src.split("\n") if not l.lstrip().startswith("#"))
    assert "respawn" not in code, "컵 단독 리스폰 재도입 금지 — truncation 리셋이 대체"
    assert "tilt_reset_deg" in code, "전도 truncation 리셋이 사라짐"
    for name in ("rl_games_ppo_lstm_cfg.yaml", "rl_games_ppo_cfg.yaml"):
        cfg = yaml.safe_load((_TASK_DIR / "config" / "agents" / name).read_text())
        conf = cfg["params"]["config"]
        assert conf["value_bootstrap"] is True, f"{name}: value_bootstrap 이 False"
        assert conf["central_value_config"]["value_bootstrap"] is True, (
            f"{name}: central_value value_bootstrap 이 False")


def test_no_palm_leash_left_behind():
    """palm leash 재도입 방지 — 정책이 팔 목표에 대해 전권을 가져야 한다.

    leash 는 목표를 실측±5cm 로 되클램프해 **걸린 축의 액션을 통째로 버렸다**
    (lstm_test2: leash_active_frac 0.43~0.90). 목표 상한은 워크스페이스 박스만
    담당한다. 와인드업은 `fabric/palm_err_{mean,p95,max}` 로 감시한다.
    """
    for fname in ("grasp_lift_env.py", "grasp_lift_env_cfg.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        assert "leash" not in code, f"{fname}: leash 재도입"


def test_no_fabric_literals_in_task_code():
    """fabric 자산 이름도 프로필 경유 — 태스크 코드에 리터럴 금지."""
    for fname in ("grasp_lift_env.py", "grasp_lift_env_cfg.py", "rewards.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        for lit in ("openarm_tesollo_sensor_right", "open_tesollo_boxes"):
            assert lit not in code, f"{fname}: fabric 리터럴 {lit}"


def test_spawn_z_is_single_source():
    """스폰 높이 이중 패딩 구조적 차단 — 프로필에 object_spawn_z 필드가 있으면 안 된다."""
    p = PROFILES["tesollo_right"]
    assert not hasattr(p, "object_spawn_z"), "프로필에 object_spawn_z 가 되살아났다"
