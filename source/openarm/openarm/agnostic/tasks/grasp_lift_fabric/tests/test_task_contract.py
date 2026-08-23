"""grasp_lift_fabric 태스크 계약 — Isaac 없이 검증 가능한 것만.

물리·IK 는 여기서 알 수 없다. probe 의 몫이다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/tests/ -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robots as _rb

_TASK_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# agnosticism — 태스크 소스에 로봇 이름이 있으면 실패
# =============================================================================
_ROBOT_LITERAL = re.compile(r"\b[rl]_(aj|hj|hl)_")


@pytest.mark.parametrize(
    "fname",
    ["grasp_lift_fabric_env.py", "grasp_lift_fabric_env_cfg.py"],
)
def test_no_robot_specific_literals_in_task_code(fname):
    """로봇 종속 정보는 RobotProfile 에만 있어야 한다."""
    text = (_TASK_DIR / fname).read_text()
    hits = [
        ln for ln in text.splitlines()
        if _ROBOT_LITERAL.search(ln) and not ln.lstrip().startswith("#")
    ]
    assert not hits, f"{fname} 에 로봇 조인트/바디 리터럴: {hits}"


def test_env_does_not_import_specific_profiles():
    text = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for banned in ("BIS_RIGHT", "TESOLLO_BI_S", "RH56_RIGHT", "SENS_RIGHT"):
        assert banned not in text, f"env 가 특정 프로필 {banned} 를 직접 참조한다"


def test_fabric_state_is_not_in_observation():
    """★fabric_q/qd 는 실기에 없는 내부 상태다 — grasp_v1 s2r 실패의 직접 원인."""
    text = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    obs_src = text[text.index("def _get_observations"):text.index("def _get_rewards")]
    for banned in ("fabric_q", "fabric_qd", "fabric_qdd"):
        assert banned not in obs_src, f"_get_observations 에 {banned} 가 들어갔다"


def _identifiers(path: Path) -> set[str]:
    """실제 코드 식별자만 — 주석·docstring 은 제외한다(설명문에 금칙어가 나온다)."""
    import ast

    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)          # cfg 키 문자열도 검사 대상
    return out


def test_rewards_do_not_use_a_lift_latch():
    """래치로 항을 이분하면 grasp gradient 가 죽는다(grasp_v1 실측 0.036 평탄).

    ★08.23 부터 보상은 자매 트랙 것을 쓰므로 검사 대상도 그쪽 파일이다.
    """
    ids = _identifiers(_TASK_DIR.parent / "grasp_sensor" / "rewards.py")
    for banned in ("lift_latched", "pre_lift_gate", "lift_gate"):
        assert banned not in ids, f"보상이 래치 게이트 {banned} 를 쓴다"


def test_env_does_not_use_a_lift_latch():
    ids = _identifiers(_TASK_DIR / "grasp_lift_fabric_env.py")
    for banned in ("lift_latched", "lift_start_step_buf", "lift_palm_pose_buf"):
        assert banned not in ids, f"env 가 래치 상태 {banned} 를 쓴다"


def test_sim_state_is_not_mutated_during_reward():
    """보상 계산 중 sim 을 건드리지 않는다(리스폰은 _pre_physics_step 으로)."""
    text = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = text[text.index("def _get_rewards"):text.index("def _get_dones")]
    assert "write_root_state_to_sim" not in body
    assert "set_joint_position_target" not in body


# =============================================================================
# cfg 차원
# =============================================================================
def _dims(profile, bank, onehot: bool):
    """★고정(외전) 관절은 액션에서 빠지지만 obs 의 joint_* 에는 남는다.

    이 공식이 실제 cfg 파생과 어긋나면 테스트가 자기 공식만 검증하게 된다 —
    `test_formula_matches_real_cfg` 가 그걸 막는다.
    """
    j = profile.num_arm_joints + profile.num_hand_joints
    f = len(profile.fingers)
    n_free = profile.num_hand_joints - len(profile.frozen_hand_joints)
    action = 6 + n_free
    # +6 = palm 지령(slew 상태). 73c2adc 에서 obs 에 들어갔는데 이 공식이 함께
    # 안 고쳐져 115/121 을 pin 한 채 표류했다(Isaac 밖에선 importorskip 으로 숨음).
    obs = 3 * j + f + 7 + 3 + action + 6 + (bank.onehot_dim if onehot else 0)
    return action, obs, obs + 6


@pytest.mark.parametrize("pname", sorted(_rb.PROFILES))
def test_dimension_formula_matches_profile(pname):
    p = _rb.get(pname)
    bank = _ob.get("single_cup")
    action, obs, state = _dims(p, bank, onehot=False)
    n_free = p.num_hand_joints - len(p.frozen_hand_joints)
    assert action == 6 + n_free, f"{pname}: 고정 {len(p.frozen_hand_joints)} 반영 안 됨"
    assert obs > action
    assert state == obs + 6


def test_bis_right_reference_dimensions():
    """_1 관절 전체 + thumb_2 + pinky_2 = 7 고정 → 액션 6+13=19, obs 121, critic 127."""
    action, obs, state = _dims(_rb.get("bis_right"), _ob.get("single_cup"), onehot=False)
    assert (action, obs, state) == (19, 121, 127), (action, obs, state)


def test_formula_matches_real_cfg():
    """★테스트가 자기 공식만 검증하지 않도록 실제 cfg 파생과 대조한다."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    a, o, st = _dims(_rb.get(cfg.profile_name), _ob.get(cfg.object_bank), cfg.enable_object_onehot)
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (a, o, st), (
        f"cfg {(cfg.action_space, cfg.observation_space, cfg.state_space)} vs 공식 {(a, o, st)}"
    )


def test_onehot_changes_obs_dim():
    """onehot 을 켜면 obs 차원이 바뀐다 = 체크포인트 비호환(문서화된 사실)."""
    p = _rb.get("bis_right")
    bank = _ob.get("cup_family")
    _, off, _ = _dims(p, bank, onehot=False)
    _, on, _ = _dims(p, bank, onehot=True)
    assert on - off == len(bank)


# =============================================================================
# 보상
# =============================================================================
N = 8


def _ones(v=0.0):
    return torch.full((N,), float(v))


# =============================================================================
# 보상 — ★08.23 부터 자매 트랙 grasp_sensor 의 함수를 **그대로 import** 한다.
#
# 수식 계약(감쌈 판정·goal 계열의 감쌈 선행·직립의 리프트 곱)은 그쪽
# tests/test_profile_contract.py 가 지킨다. 여기서 같은 수식을 다시 검사하면
# 두 벌이 갈라졌을 때 **양쪽 다 통과하면서 거동만 달라진다** — 그 실패 방식이
# 정확히 이번에 고친 결함(lift 분모 0.10 vs 0.15)이었다.
# 그래서 이 파일이 지키는 것은 **"갈라지지 않았는가"** 하나다.
# =============================================================================
_SENSOR_DIR = _TASK_DIR.parent / "grasp_sensor"


def test_reward_is_imported_from_sibling_track_not_copied():
    """보상 함수를 복사하지 않고 import 하는가."""
    assert not (_TASK_DIR / "rewards.py").exists(), (
        "rewards.py 가 되살아났다 — 복사본은 다시 갈라진다. "
        "grasp_sensor.rewards 를 import 하라")
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "from ..grasp_sensor.rewards import" in src, (
        "env 가 자매 트랙 보상을 import 하지 않는다")
    assert "compute_grasp_sensor_rewards" in src


@pytest.mark.parametrize("key", [
    # 보상 수식이 읽는 계수 — 두 트랙이 **같은 값**이어야 비교가 성립한다.
    "approach_weight", "approach_sharpness", "grasp_z_offset", "side_radius",
    "envelope_weight", "contact_weight", "contact_force_threshold",
    "tracking_weight", "tracking_std", "success_weight", "success_std",
    "lift_weight", "upright_weight", "upright_exponent",
    "tilt_penalty_weight", "tilt_free_deg",
    "action_l2_weight", "action_rate_l2_weight",
    "goal_height_offset", "success_envelope_min", "success_tilt_max_deg",
    "success_pos_tolerance",
])
def test_reward_coefficients_match_sibling_track(key):
    """계수가 자매 트랙과 동기인가 — 함수만 같고 값이 다르면 비교가 무의미하다."""
    import re

    def declared(path):
        out = {}
        for m in re.finditer(r"^    ([a-z0-9_]+):\s*(?:float|int)\s*=\s*([-\d.]+)",
                             path.read_text(), re.M):
            out[m.group(1)] = float(m.group(2))
        return out

    ours = declared(_TASK_DIR / "grasp_lift_fabric_env_cfg.py")
    theirs = declared(_SENSOR_DIR / "grasp_sensor_env_cfg.py")
    assert key in ours, f"{key} 가 우리 cfg 에 없다 — 보상이 하드코딩 기본값으로 돈다"
    assert key in theirs, f"{key} 가 자매 트랙 cfg 에 없다 — 목록이 낡았다"
    assert ours[key] == theirs[key], (
        f"{key} 가 갈라졌다: 우리 {ours[key]} vs grasp_sensor {theirs[key]}")


def test_lift_and_upright_saturate_at_goal_height():
    """★lift/upright 의 분모는 goal 높이여야 한다.

    분모가 goal 보다 작으면 "거기까지만 올리면 만점"이 되어 그 위가 평지가 된다.
    실측: 분모 0.10 · goal 0.15 이던 우팔이 18cm 까지 들었다가 6cm 로 되돌아왔다
    (되돌려도 lift 손해가 없으므로 되돌리는 것이 이득이었다).
    자매 트랙 보상은 둘 다 `goal_height_offset` 을 분모로 쓴다 — 그 규약을 고정한다.
    """
    src = (_SENSOR_DIR / "rewards.py").read_text()
    body = src[src.index("def compute_grasp_sensor_rewards"):]
    for term in ('"lift"', '"upright"'):
        seg = body[body.index(term):body.index(term) + 400]
        assert "goal_height_offset" in seg, (
            f"{term} 의 리프트 분모가 goal_height_offset 이 아니다")
    assert "lift_success_height" not in src, (
        "goal 과 어긋나는 별도 분모가 되살아났다")


def test_envelope_has_no_gate_floor():
    """★감쌈 곱수에 하한을 두면 감쌈 없이도 이송 보상이 흐른다.

    실측: 하한 0.3 이던 좌팔이 감쌈 0.21 로 이송(goal 0.58)만 학습했다 —
    사용자 우선순위 ①(인벨롭 그립)이 무너진 직접 원인.
    """
    src = (_SENSOR_DIR / "rewards.py").read_text()
    seg = src[src.index("def envelope_gate"):src.index("def tracking_reward")]
    assert "clamp(0.0, 1.0)" in seg
    for banned in ("floor", "mul_floor", "+ 0.3"):
        assert banned not in seg, f"감쌈 곱수에 하한 {banned} 가 들어갔다"


def test_success_requires_envelope_and_upright():
    """성공 판정이 goal 거리 하나가 아니라 3조건 AND 인가.

    거리만 보면 "감쌈 0.21 로 이송만 한" 상태가 성공으로 집계된다(좌팔 0.58).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _get_rewards"):src.index("def _get_dones")]
    assert "success_envelope_min" in body, "성공 판정이 감쌈을 안 본다"
    assert "success_tilt_max_deg" in body, "성공 판정이 직립을 안 본다"
    assert "_pass_pos & _pass_env & _pass_tilt" in body


def test_envelope_fingers_excludes_pinky_for_tesollo():
    """감쌈 분모에서 pinky 제외 — 굴곡축이 없어 상한이 0.8 로 깎인다."""
    rb = pytest.importorskip("openarm.agnostic.modules.robots",
                             reason="패키지 임포트 불가")
    for name in ("bis_right", "bis_left"):
        p = next(x for x in vars(rb).values()
                 if getattr(x, "name", None) == name)
        assert p.envelope_fingers, f"{name} envelope_fingers 미정의"
        assert "pinky" not in p.envelope_fingers, f"{name} 분모에 pinky 가 있다"
        assert [f for f in p.contact_group_b if f in p.envelope_fingers]


def test_strict_envelope_is_still_measured():
    """엄격 감쌈(전 마디 동시접촉)을 대조 지표로 계속 재는가.

    보상은 느슨한 규약(마디 하나라도)을 쓴다. 같은 정책을 두 판정으로 재면
    0.503 vs 0.069 로 7배가 벌어졌다 — 느슨한 쪽만 오르면 "받치기"다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "task/envelope_strict" in src, "엄격 감쌈 대조 지표가 사라졌다"

# =============================================================================
# cfg ↔ 보상 키 정합
# =============================================================================
def test_every_reward_cfg_key_is_declared():
    """★`_cfg(cfg, "x", 기본값)` 패턴은 cfg 에 없어도 **조용히 기본값으로 동작**한다.

    그러면 그 값이 env.yaml dump 에 안 나와 "무엇으로 학습했는가"의 기록이 비고,
    튜닝도 불가능하다. 실제로 4개가 그렇게 빠져 있었다.
    """
    import re

    reward_keys = set(re.findall(
        r'cfg\.([a-z0-9_]+)\)', (_SENSOR_DIR / "rewards.py").read_text()))
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    declared = set(re.findall(r"^    ([a-z0-9_]+):", cfg_src, re.M))
    missing = sorted(reward_keys - declared)
    assert not missing, f"보상이 읽는데 cfg 에 선언되지 않음(하드코딩 기본값으로 동작): {missing}"


def test_no_dead_reward_cfg_fields():
    """cfg 에만 있고 아무도 안 읽는 보상 필드 = 죽은 설정(오해를 부른다)."""
    import re

    src = (_SENSOR_DIR / "rewards.py").read_text() + (
        _TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    declared = set(re.findall(r"^    ([a-z0-9_]+(?:_weight|_credit|_tau|_std|_deg)):", cfg_src, re.M))
    dead = sorted(k for k in declared if k not in src)
    assert not dead, f"죽은 보상 설정: {dead}"


# =============================================================================
# hydra CLI 오버라이드 반영 (모듈 설정으로 학습 조합을 바꾸는 통로)
#
# cfg 모듈은 isaaclab 을 import 하므로 Isaac 환경에서만 돈다.
# 조용히 통과시키지 않고 명시적으로 skip 한다.
# =============================================================================
# ★모듈 레벨 importorskip 은 쓰지 않는다 — 파일 전체(30개)가 조용히 skip 된다.
#   실제로 그렇게 만들었다가 되돌렸다. 필요한 테스트 안에서만 skip 한다.
def _cfg_module():
    return pytest.importorskip(
        "openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg",
        reason="Isaac(pxr) 없음 — cfg 파생 검사는 `isaaclab.sh -p -m pytest` 에서만 유효")


def test_resolve_cfg_is_idempotent_and_applies_switches():
    """★hydra 는 `env_cfg.from_dict(...)` 로 필드만 덮어쓰고 __post_init__ 을 다시
    돌리지 않는다. 파생값을 재계산하지 않으면 "cup_family 라고 적혀 있는데 컵 하나만
    스폰" 같은 조용히 틀린 조합이 된다.
    """
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    base_obs, base_act = cfg.observation_space, cfg.action_space
    assert cfg.scene.replicate_physics is True          # single_cup 은 복제 가능

    # 멱등: 같은 값으로 다시 풀어도 결과 동일
    C.resolve_cfg(cfg)
    assert (cfg.observation_space, cfg.action_space) == (base_obs, base_act)

    # 스위치를 hydra 처럼 필드만 바꾼 뒤 재해석
    cfg.object_bank = "cup_family"
    cfg.enable_object_onehot = True
    C.resolve_cfg(cfg)
    assert cfg.scene.replicate_physics is False, "MultiAsset 인데 physics 복제가 켜져 있다"
    assert cfg.observation_space == base_obs + 8, "onehot 차원이 반영되지 않았다"
    assert cfg.state_space == cfg.observation_space + 6


def test_profile_override_changes_dimensions():
    """로봇 교체가 차원까지 따라와야 한다(손 자유도가 다르다)."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    cfg.profile_name = "rh56_right"          # 손 12관절, 엄지 외전 1개 고정
    C.resolve_cfg(cfg)
    assert cfg.action_space == 6 + 11
    cfg.profile_name = "sens_left"           # 2지 그리퍼, 손 1관절, 고정 없음
    C.resolve_cfg(cfg)
    assert cfg.action_space == 6 + 1


def test_fabricless_profile_fails_loud_on_resolve():
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    cfg.profile_name = "rh56_left"           # fabric 자산 없음
    with pytest.raises(RuntimeError, match="Fabrics"):
        C.resolve_cfg(cfg)


# =============================================================================
# goal 랜덤화 (이송 학습, 08.22)
# =============================================================================
def test_goal_radius_starts_at_zero():
    """★ADR goal 축의 initial 은 **0** 이어야 한다 — 반경 0 = 구 고정 goal 과 동치라는
    보장이 있어야 초기 학습 난이도가 바뀌지 않는다(reward-audit Check 4 의 전제).
    바꾸려면 이 테스트를 의도적으로 고쳐야 한다.
    """
    C = _cfg_module()
    real = C.GraspLiftFabricEnvCfg()
    assert real.goal_xy_radius_initial == 0.0
    assert real.goal_z_radius_initial == 0.0
    # final 은 이송을 실제로 배울 만큼 양수여야 한다 (0 이면 축이 죽은 코드)
    assert real.goal_xy_radius_final > 0.0
    assert real.goal_z_radius_final >= 0.0
    # 클램프 마진은 박스 절반보다 작아야 한다 (아니면 클램프가 goal 을 한 점으로 붕괴시킨다)
    assert 0.0 < real.goal_box_margin < 0.05


# =============================================================================
# 정적 미정의 이름 (08.22 — TEST1 첫 기동이 epoch 10 에서 죽은 원인)
# =============================================================================
def test_no_undefined_names_static():
    """★런타임 도달이 늦은 코드(console_log_interval=600 스텝마다 발화하는 METRICS
    라인)는 smoke(150스텝)로는 실행되지 않는다 — `thr` 잔존 참조가 그렇게 검증을
    통과해 epoch 10 에서 NameError 로 학습을 죽였다. F821 정적 검사로 전 라인을 덮는다.
    """
    import os
    import shutil
    import subprocess
    if shutil.which("ruff") is None:
        pytest.skip("ruff 없음")
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(["ruff", "check", "--select", "F821", pkg],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"미정의 이름 존재:\n{r.stdout}"


# =============================================================================
# 손 제어: Fabrics 손끝 IK (08.23)
#
# 손을 Fabrics 밖에 두면 ①fabric 이 아는 손 자세가 홈에 고정되고(body_repulsion 이
# 실제보다 큰 손으로 회피) ②손가락↔손가락 쌍을 넣을 수 없어 PhysX self-collision 을
# 못 끈다. 그 구조를 되돌리지 않게 고정한다.
# =============================================================================
def test_tip_ik_action_is_fingertip_positions():
    """tip IK 모드의 손 액션은 관절이 아니라 손끝 5점 × xyz 다."""
    cfg_mod = _cfg_module()
    c = cfg_mod.GraspLiftFabricEnvCfg()
    profile = _rb.get(c.profile_name)
    if not c.use_tip_fabric:
        pytest.skip("use_tip_fabric=False 로 되돌려져 있다(의도된 대조군이면 정상)")
    assert c.action_space == 6 + 3 * len(profile.fingertip_bodies)


def test_tip_ik_gives_fabric_the_whole_hand():
    """손 20-DOF 를 전부 fabric 이 준다 — 일부만 얼리면 fabric 이 아는 자세와 어긋난다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _apply_action"):src.index("def _apply_gravity_compensation")]
    assert "self.hand_ids" in body, "tip 모드에서 손 전체에 목표를 주지 않는다"
    assert "self.fabric_q[:, self.profile.num_arm_joints:]" in body, (
        "손 목표가 fabric 산출물이 아니다")


def test_tip_workspace_is_measured_not_hardcoded():
    """손끝 도달 박스는 부팅 시 실측한다 — 자산이 바뀌면 값도 따라가야 한다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "_measure_tip_workspace" in src
    body = src[src.index("def _measure_tip_workspace"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_fingertip_taskmap" in body, "FK 로 재지 않는다"
    # 손끝 좌표 리터럴이 박혀 있으면 실측이 아니다
    assert not re.search(r"0\.0\d{2}\s*,\s*0\.0\d{2}\s*,\s*0\.0\d{2}", body), (
        "손끝 좌표가 하드코딩돼 있다")


def test_tip_target_uses_actual_palm_not_commanded():
    """손끝 목표의 palm 기준은 **지령**이 아니라 fabric 이 도달한 palm 이다.

    지령을 쓰면 추종오차만큼 손끝 목표가 컵에서 어긋나고, 그 오차는 파지 순간에 가장 크다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("if self._tip_ik:"):src.index("def _step_fabric")]
    seg = body[:body.index("set_features")]
    assert "_palm_frame(self.fabric_q" in seg, "palm 기준이 fabric 실제 자세가 아니다"
    assert "self.palm_cmd" not in seg.split("_palm_frame")[0][-400:], (
        "손끝 목표를 지령 palm 기준으로 만들고 있다")
