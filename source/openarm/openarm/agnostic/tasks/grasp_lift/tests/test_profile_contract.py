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
