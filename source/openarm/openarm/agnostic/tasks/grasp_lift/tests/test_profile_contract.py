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


def test_envelope_contract():
    """인벨롭 그립 성립 계약 (08.22 보상 재설계, 사용자 최우선 목표).

    ① 프로필마다 envelope_fingers 정의 + tesollo 는 pinky 제외(굴곡축 부재 —
      분모에 넣으면 상한 0.8, 메모리 tesollo-pinky-joint-kinematics)
    ② 그룹-min reaching(rim-hook 의 원인) 재도입 금지
    ③ 성공 판정은 3조건(goal·envelope·tilt) — 코드 문자열 계약
    """
    for p in PROFILES.values():
        assert p.envelope_fingers, f"{p.name}: envelope_fingers 미정의"
        for f in p.envelope_fingers:
            assert f in p.finger_sensor_bodies, f"{p.name}: envelope 손가락 {f} 센서 없음"
        assert set(p.envelope_fingers) & set(p.contact_group_b), (
            f"{p.name}: contact_group_b ∩ envelope_fingers 공집합")
    tes = PROFILES["tesollo_right"]
    assert "pinky" not in tes.envelope_fingers, "tesollo pinky 는 envelope 분모에서 제외"

    rew_src = (_TASK_DIR / "rewards.py").read_text(encoding="utf-8")
    rew_code = "\n".join(l for l in rew_src.split("\n") if not l.lstrip().startswith("#"))
    assert "group_reaching" not in rew_code, "그룹-min reaching 재도입 금지 (rim-hook 원인)"
    assert "approach_reward" in rew_code and "envelope_fraction" in rew_code

    env_src = (_TASK_DIR / "grasp_lift_env.py").read_text(encoding="utf-8")
    env_code = "\n".join(l for l in env_src.split("\n") if not l.lstrip().startswith("#"))
    for needle in ("success_envelope_min", "success_tilt_max_deg"):
        assert needle in env_code, f"성공 판정 3조건 누락: {needle}"


def test_envelope_discriminates_rim_hook():
    """P-B (합성 접촉 패턴): 보상이 rim-hook 과 인벨롭을 실제로 구분하는가.

    물리 재현 probe 는 rim-hook 을 정확히 재현하지 못한다(컵을 손에 밀착시키고
    손가락을 굽히면 그 자체가 2~3지 인벨롭이 된다) — 판별은 접촉 패턴 수준에서
    결정적이므로 합성 텐서로 검증한다. env 손가락 4개(thumb,index,middle,ring):
      rim-hook 실측 패턴 = 검지 마디 걸침 + 엄지 팁만  → env_frac 0.25
      2지 깊은 걸침       = 검지+엄지 마디             → env_frac 0.50
      인벨롭              = 3~4지 마디 감쌈            → env_frac 0.75~1.0
    성공 임계 0.6 은 앞의 둘을 배제하고 뒤의 둘만 통과시켜야 한다.
    """
    import torch
    from openarm.agnostic.tasks.grasp_lift.rewards import envelope_fraction
    thr = 1.0
    F = 5.0  # 접촉력 [N]

    def frac(mid_fingers, dist_fingers):
        mid = torch.zeros(1, 4)
        dist = torch.zeros(1, 4)
        for i in mid_fingers:
            mid[0, i] = F
        for i in dist_fingers:
            dist[0, i] = F
        return float(envelope_fraction(mid, dist, thr))

    # 손가락 인덱스: 0=thumb, 1=index, 2=middle, 3=ring
    rim_real = frac(mid_fingers=[1], dist_fingers=[1])          # 검지만 마디, 엄지 팁만
    rim_deep2 = frac(mid_fingers=[0, 1], dist_fingers=[0, 1])   # 2지 깊은 걸침
    env3 = frac(mid_fingers=[0, 1, 2], dist_fingers=[1, 2])     # 3지 감쌈
    env4 = frac(mid_fingers=[0, 1, 2, 3], dist_fingers=[0, 1, 2, 3])
    assert rim_real == 0.25 and rim_deep2 == 0.5
    assert env3 == 0.75 and env4 == 1.0

    # cfg 모듈은 isaaclab 앱 없이 import 불가 — 소스에서 임계값을 파싱
    import re
    cfg_src = (_TASK_DIR / "grasp_lift_env_cfg.py").read_text(encoding="utf-8")
    m = re.search(r"success_envelope_min:\s*float\s*=\s*([0-9.]+)", cfg_src)
    assert m, "success_envelope_min 정의 부재"
    th = float(m.group(1))
    assert rim_real < th and rim_deep2 < th, "성공 임계가 rim-hook 을 통과시킨다"
    assert env3 >= th and env4 >= th, "성공 임계가 정상 인벨롭을 배제한다"


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
