"""RobotProfile 계약 — Isaac 앱 없이 검증 가능한 것 전부.

robot-agnostic 태스크의 합격 조건: 태스크 코드에 조인트/바디 이름이 없고,
로봇 추가 = 프로필 1개 추가. 이 테스트가 그 계약을 고정한다.
"""

import re
from pathlib import Path

import pytest

from openarm.agnostic.tasks.grasp_sensor.robot_profiles import PROFILES

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
    for fname in ("grasp_sensor_env.py", "rewards.py", "grasp_sensor_env_cfg.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        m = banned.search(code)
        assert m is None, f"{fname} 에 로봇 이름 하드코딩: {m.group(0)}"


def test_env_reward_import_free_of_profiles_constants():
    """env 는 PROFILES 조회 외에 특정 프로필 상수를 import 하지 않는다."""
    src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
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
    src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
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
    # ★08.24 규약 반전(사용자 지시): tesollo pinky 를 분모에 **포함**한다.
    #   구 규약("굴곡축 부재 → 제외, 상한 0.8")은 관절공간 액션 시절 근거였다.
    #   tip_cyl(손가락별 손끝 IK + 원통 (r,z))에서는 probe 실측으로 pinky 도 손바닥면
    #   100% 감쌈했다 — 액션 구조가 바뀌면 도달성 결론도 바뀐다. 5지 인벨롭이 목표.
    tes = PROFILES["tesollo_right"]
    assert "pinky" in tes.envelope_fingers, "tesollo 5지 인벨롭 — pinky 포함 규약"

    rew_src = (_TASK_DIR / "rewards.py").read_text(encoding="utf-8")
    rew_code = "\n".join(l for l in rew_src.split("\n") if not l.lstrip().startswith("#"))
    assert "group_reaching" not in rew_code, "그룹-min reaching 재도입 금지 (rim-hook 원인)"
    assert "approach_reward" in rew_code and "envelope_fraction" in rew_code

    env_src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
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
    from openarm.agnostic.tasks.grasp_sensor.rewards import envelope_fraction
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
    cfg_src = (_TASK_DIR / "grasp_sensor_env_cfg.py").read_text(encoding="utf-8")
    m = re.search(r"success_envelope_min:\s*float\s*=\s*([0-9.]+)", cfg_src)
    assert m, "success_envelope_min 정의 부재"
    th = float(m.group(1))
    assert rim_real < th and rim_deep2 < th, "성공 임계가 rim-hook 을 통과시킨다"
    assert env3 >= th and env4 >= th, "성공 임계가 정상 인벨롭을 배제한다"


def test_goal_family_requires_envelope():
    """goal 계열(tracking/success/lift)은 감쌈 없이는 수확 불가여야 한다.

    ★lstm_test1 반증: 이진 대향 접촉 게이트는 **팁 접촉만으로 성립**해 press-grip
    (엄지 미접촉·네 손가락으로 컵을 손바닥에 압착)으로 goal 계열 13.5 를 전부
    수확했다. ∂(goal)/∂(env_frac)=0 이라 감쌈이 순수 비용이 되어, 정책이 ep500 에
    스스로 배운 감쌈(0.335)을 lift 가 오르면서 되팔았다(ep1000 0.215).
    → 유효 게이트 = 이진 접촉 × clamp(env_frac/target). 연속이라 절벽은 아니다.
    """
    import torch
    from openarm.agnostic.tasks.grasp_sensor.rewards import envelope_gate
    g = torch.tensor([True, True, True, True])
    env = torch.tensor([0.0, 0.25, 0.6, 1.0])
    eff = envelope_gate(g, env, 0.6)
    assert float(eff[0]) == 0.0, "감쌈 0 인데 goal 계열이 열린다"
    assert 0.0 < float(eff[1]) < 1.0, "부분 감쌈이 부분 보상을 줘야 한다(절벽 금지)"
    assert float(eff[2]) == 1.0 and float(eff[3]) == 1.0, "임계 이상에서 포화해야 한다"
    # 접촉 자체가 없으면 감쌈 값과 무관하게 0
    assert float(envelope_gate(torch.tensor([False]), torch.tensor([1.0]), 0.6)[0]) == 0.0

    src = (_TASK_DIR / "rewards.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    # 정의부가 아니라 **호출부**를 본다(정의부는 인자명이 gate 라 오탐).
    # 중첩 괄호(float(cfg...)) 때문에 괄호 매칭 대신 "호출 직후 g_eff 등장"으로 검사.
    for term in ("tracking_reward", "success_reward", "lift_reward"):
        assert re.search(r"\*\s*" + term + r"\(.{0,140}?g_eff", code, re.S), (
            f"{term} 호출이 유효 게이트(g_eff)를 안 쓴다")
    assert "contact_weight) * g.float()" in code, (
        "contact 항은 순수 이진 게이트로 남아야 한다 — 접촉 사다리 한 칸")


def test_failure_is_terminated_not_truncated():
    """실패(전도/낙하/이탈)는 terminated, truncated 는 시간 만기 전용.

    ★lstm_test1 ep1600 붕괴의 근본 원인. 실패를 truncated 로 내보내면 IsaacLab 이
    extras["time_outs"] 로 싣고 rl_games 가 value_bootstrap 으로
    `shaped_rewards += gamma·V(s_t)` 를 더한다 → **컵을 쓰러뜨릴 때마다 보너스**.
    실측: 실제 보상 3307→103 인데 shaped_rewards 는 72.8→79.4 로 상승(익스플로잇).
    ★08.25: `value_bootstrap` 은 **False**(DEXTRAH Kuka 원본값)로 바뀌었다. 위 사고의
    근본 원인은 "실패가 truncated 로 나간 것"이지 bootstrap 자체가 아니므로, 실패를
    terminated 로 내보내는 아래 불변조건만 지켜지면 재발하지 않는다. 다만 시간 만기
    부트스트랩을 잃어 시간제한 편향이 생긴다 — 값 추정이 이상하면 여기부터 되돌린다.
    """
    import yaml
    env_src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in env_src.split("\n") if not l.lstrip().startswith("#"))
    assert "respawn" not in code, "컵 단독 리스폰 재도입 금지 — 전체 리셋이 대체"
    assert "tilt_reset_deg" in code, "전도 리셋이 사라짐"
    assert re.search(r"truncated\s*=\s*timeout\s*$", code, re.M), (
        "truncated 는 timeout 단독이어야 한다 — 실패를 실으면 bootstrap 이 "
        "실패에 γ·V(s) 보너스를 지급한다(ep1600 붕괴 재발)")
    assert re.search(r"terminated\s*=.*fell.*tipped.*out_xy", code), (
        "실패(fell/tipped/out_xy)는 terminated 로 나가야 한다")
    for name in ("rl_games_ppo_lstm_cfg.yaml", "rl_games_ppo_cfg.yaml"):
        cfg = yaml.safe_load((_TASK_DIR / "config" / "agents" / name).read_text())
        conf = cfg["params"]["config"]
        # ★Kuka 원본 정합(08.25): 최상위는 False, central_value 는 **키 자체가 없다**
        #   (Kuka central_value_config 가 지정하지 않아 rl_games 기본값을 쓴다).
        assert conf["value_bootstrap"] is False, (
            f"{name}: value_bootstrap 이 Kuka 원본값(False) 이 아니다")
        assert "value_bootstrap" not in conf["central_value_config"], (
            f"{name}: central_value 에 value_bootstrap 이 남아 있다 — Kuka 는 미지정")


def test_no_palm_leash_left_behind():
    """palm leash 재도입 방지 — 정책이 팔 목표에 대해 전권을 가져야 한다.

    leash 는 목표를 실측±5cm 로 되클램프해 **걸린 축의 액션을 통째로 버렸다**
    (lstm_test2: leash_active_frac 0.43~0.90). 목표 상한은 워크스페이스 박스만
    담당한다. 와인드업은 `fabric/palm_err_{mean,p95,max}` 로 감시한다.
    """
    for fname in ("grasp_sensor_env.py", "grasp_sensor_env_cfg.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        assert "leash" not in code, f"{fname}: leash 재도입"


def test_no_fabric_literals_in_task_code():
    """fabric 자산 이름도 프로필 경유 — 태스크 코드에 리터럴 금지."""
    for fname in ("grasp_sensor_env.py", "grasp_sensor_env_cfg.py", "rewards.py"):
        src = (_TASK_DIR / fname).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        for lit in ("openarm_tesollo_sensor_right", "open_tesollo_boxes"):
            assert lit not in code, f"{fname}: fabric 리터럴 {lit}"


def test_spawn_z_is_single_source():
    """스폰 높이 이중 패딩 구조적 차단 — 프로필에 object_spawn_z 필드가 있으면 안 된다."""
    p = PROFILES["tesollo_right"]
    assert not hasattr(p, "object_spawn_z"), "프로필에 object_spawn_z 가 되살아났다"


# ---------------------------------------------------------------------------
# gym 등록 — Fabrics 자산이 없는 프로필은 등록하지 않는다
#
# 이 태스크는 Fabrics 로만 돈다(`_setup_fabrics` 가 fabric_class=None 이면 RuntimeError).
# 그런데 등록은 되고 있어서 `open-sens_l_grasp_sensor` 4종이 "존재하지만 띄우면 죽는"
# 상태였다. 형제 태스크(grasp_lift_fabric)는 이미 SKIPPED 규약으로 사유를 남긴다.
# ---------------------------------------------------------------------------
_CONFIG_SRC = (
    __import__("pathlib").Path(__file__).resolve().parents[1] / "config/__init__.py"
).read_text(encoding="utf-8")


def test_registration_skips_profiles_without_fabrics_and_records_why():
    """정적 검사 — Isaac 없이도 돈다."""
    assert "SKIPPED" in _CONFIG_SRC, "등록 실패 사유를 남기는 규약이 없다"
    assert "if _profile.fabric_class is None:" in _CONFIG_SRC
    assert "continue" in _CONFIG_SRC


def test_registration_records_the_ids_it_actually_created():
    assert "REGISTERED.append(" in _CONFIG_SRC


def test_gripper_left_profile_is_the_one_without_fabrics():
    """SKIPPED 대상이 실제로 어느 프로필인지는 레지스트리가 정한다."""
    assert PROFILES["gripper_left"].fabric_class is None
    assert PROFILES["tesollo_right"].fabric_class is not None


# ---- Isaac 이 있을 때만: 실제 등록 결과 ----------------------------------------
def _config_module():
    # ★`importorskip("isaaclab")` 만으로는 부족하다 — isaaclab 패키지 자체는 임포트되지만
    #   cfg 모듈이 끌어오는 isaaclab.utils 는 **앱 기동 후에만** 존재해서 더 깊은 곳에서
    #   ImportError 가 난다(= skip 이 아니라 FAIL). 실제 임포트를 감싸야 한다.
    pytest.importorskip("isaaclab", reason="gym 등록 확인은 Isaac 환경에서만 가능")
    try:
        from openarm.agnostic.tasks.grasp_sensor import config as _cfg
    except ImportError as e:
        pytest.skip(f"Isaac 앱 없이는 cfg 임포트 불가: {e}")
    return _cfg


def test_tesollo_right_registration_is_unchanged():
    """★학습 중인 구성이다 — id 4종이 그대로여야 한다."""
    cfg = _config_module()
    for suffix in ("", "-play", "-lstm", "-play-lstm"):
        assert f"open-sens_r_grasp_sensor{suffix}" in cfg.REGISTERED


def test_gripper_left_is_skipped_not_registered():
    cfg = _config_module()
    assert not any(i.startswith("open-sens_l_") for i in cfg.REGISTERED), cfg.REGISTERED
    assert "sens_l" in cfg.SKIPPED
    assert "fabric" in cfg.SKIPPED["sens_l"].lower()


def test_registration_reads_the_profile_name_from_dataclass_fields():
    """★실측 결함(349b179 → 77bcb9a): `_cls.profile_name` 은 존재하지 않는다.

    @configclass 가 클래스 속성을 dataclass 필드로 옮기면서 클래스에서 제거하고,
    기본값도 default_factory 쪽으로 보낸다. 그걸 모르고 클래스 속성으로 읽었더니
    `isaaclab_tasks` import 자체가 죽어 **어떤 태스크도 못 띄우는** 상태가 됐다.
    그 결함은 이 파일을 import 해야만 드러나는데 import 에는 isaaclab 이 필요해
    로컬 테스트가 통째로 놓쳤다 — 그래서 소스 단언으로도 잠근다.
    """
    assert "__dataclass_fields__" in _CONFIG_SRC
    assert "default_factory" in _CONFIG_SRC, "default 만 보면 MISSING 이 나온다"
    # ★주석에는 그 표현이 **일부러** 남아 있다(왜 안 되는지 설명한다). 코드만 본다.
    code = "\n".join(
        line for line in _CONFIG_SRC.splitlines() if not line.lstrip().startswith("#")
    )
    assert "_cls.profile_name" not in code, (
        "@configclass 이후에는 클래스 속성으로 남지 않는다"
    )
    assert "_profile_name_of(_cls)" in code


def test_envelope_counts_only_palmar_contact():
    """감쌈은 **손바닥 접촉만** 인정해야 한다 — 손등으로 밀어도 세면 지표가 허수가 된다.

    ★08.23 lstm_test3 ep5000 실측(64env, 접촉한 스텝만 판정): 손바닥 법선을 URDF 의
    cross(굴곡축, 장축)으로 유도해 dot(법선, 컵중심−마디위치) 부호를 봤더니
        thumb_3 +30.6mm(손등비 0.00) · index_3 +45.0mm(0.00) · middle_4 −19.7mm(**1.00**)
    중지는 접촉 시간 100% 를 손등으로 접촉했다. 원인은 middle_2 가 77% 시간 하한 0 에
    붙어 밑동이 안 펴진 채 _3/_4 만 최대로 말려 갈고리가 되는 것.
    크기만 세는 구 판정은 env_frac 0.746 을 보고했지만 손바닥만 세면 ≈0.55 로 성공
    임계 0.6 미달 — 지표가 실패를 성공으로 통과시키고 있었다.
    """
    cfg_src = (_TASK_DIR / "grasp_sensor_env_cfg.py").read_text(encoding="utf-8")
    assert re.search(r"require_palmar_contact:\s*bool\s*=\s*True", cfg_src), (
        "손바닥 접촉 판정이 기본 off — 손등 파지가 다시 감쌈으로 계상된다")

    env_src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
    assert "def _palmar_mask" in env_src, "손바닥/손등 판정 함수가 없다"
    # 게이팅이 실제로 감쌈 입력에 걸리는가 — 정의만 있고 안 쓰면 무의미하다.
    assert re.search(r"mid_f\s*=\s*mid_raw\s*\*", env_src), "판정이 mid 힘에 적용되지 않는다"
    assert re.search(r"dist_f\s*=\s*dist_raw\s*\*", env_src), "판정이 dist 힘에 적용되지 않는다"
    # ★구 판정을 나란히 남겨야 손등 비중을 상시 관측할 수 있다(회귀 조기 발견).
    assert "task/envelope_frac_raw" in env_src, "구 판정 비교선이 없다"
    # 대향 게이트와 obs 는 건드리지 않는다 — 실기 센서도 방향을 모른다(s2r 규약).
    assert re.search(r"thumb_force=contact\[", env_src), (
        "대향 게이트가 손바닥 판정에 오염됐다 — obs/게이트는 크기 그대로여야 한다")


def test_every_registered_profile_defines_palmar_axes():
    """손바닥 법선은 프로필이 정의한다 — 기본축 가정은 판정을 조용히 뒤집는다.

    등록되는(= Fabrics 자산이 있는) 프로필은 전 손가락에 축이 있어야 하고,
    등록되지 않는 프로필은 비워 두되 env 가 fail-loud 로 막아야 한다.
    """
    env_src = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
    assert "palmar_axis_local 미정의" in env_src, "미정의 프로필을 조용히 통과시킨다"

    for name, p in PROFILES.items():
        if p.fabric_class is None:      # 등록 SKIP 대상 — 실측 전까지 비워 두는 게 맞다
            continue
        missing = [f for f in p.finger_sensor_bodies if f not in p.palmar_axis_local]
        assert not missing, f"[{name}] palmar_axis_local 누락: {missing}"
        for f, ax in p.palmar_axis_local.items():
            assert len(ax) == 3 and any(abs(v) > 0.0 for v in ax), (
                f"[{name}] {f} 손바닥축이 영벡터")


def test_lift_progress_peaks_at_target_and_decays_beyond():
    """리프트 진척은 목표에서 정점, 초과분은 감소해야 한다(08.24 R1).

    구판 clamp(h/목표)는 목표 위가 무벌점 포화 → lstm_test4 가 h=0.35(목표의 2.3배)
    까지 표류하며 gd 0.32 의 tanh 저감도 구간으로 들어갔다. 하강 기울기는 상승의
    절반 — 관측 최대 표류(0.35)에서도 gradient 가 살아 있어야 되돌아온다.
    """
    import torch
    from openarm.agnostic.tasks.grasp_sensor.rewards import lift_progress
    h = torch.tensor([0.0, 0.075, 0.15, 0.225, 0.30, 0.45, 0.60])
    v = lift_progress(h, 0.15)
    assert float(v[0]) == 0.0 and abs(float(v[1]) - 0.5) < 1e-6, "상승부 선형이 깨졌다"
    assert abs(float(v[2]) - 1.0) < 1e-6, "목표에서 정점이어야 한다"
    assert float(v[3]) < 1.0 and float(v[4]) < float(v[3]), "초과분이 감소하지 않는다"
    assert float(v[4]) > 0.0, "관측 표류 대역(0.30)에서 gradient 가 죽었다"
    assert float(v[5]) <= 1e-6 and float(v[6]) == 0.0, "2×목표에서 0 이어야 한다"
    # 하강 기울기 = 상승의 절반
    up_slope = (float(v[2]) - float(v[1])) / 0.075
    down_slope = (float(v[3]) - float(v[2])) / 0.075
    assert abs(abs(down_slope) - up_slope / 2.0) < 1e-4, "하강 기울기가 상승의 절반이 아니다"

