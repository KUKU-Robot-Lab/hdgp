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


def test_fabric_state_requires_deployed_fabric():
    """★★08.25 뒤집힘. 구 계약은 "fabric_q 는 실기에 없는 내부 상태라 금지"였다.

    그 계약은 **배포 시 fabric 을 돌리지 않는다**는 가정 위에 있었다. KUKA DEXTRAH 는
    반대 가정을 쓴다 — fabric 은 시뮬레이터가 아니라 **정책과 함께 배포되는 컨트롤러
    계층**이고, 실기 루프에서 같은 fabrics_sim 이 돈다. 그래서 원본 policy obs 가
    fabric_q/qd/qdd 를 그대로 받는다. 우리도 그 가정을 채택했으므로 금지를 푼다.

    다만 **가정이 바뀐 것이지 위험이 사라진 것이 아니다** — 배포 스택에 fabric 이
    없으면 이 관측은 실기에서 채울 수 없다. 그래서 태스크가 Fabrics 전용임을
    (fabric_class 필수) 함께 못박아 둔다. grasp_v1 s2r 실패는 fabric 없이 배포한
    정책이 fabric 상태를 기대한 사례였다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    assert "fabric_class is None" in src, (
        "fabric 상태를 관측에 넣으면서 Fabrics 필수 검사가 없다 — "
        "fabric 없는 프로필로 부팅되면 실기에 못 올리는 정책이 조용히 학습된다")


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
def _dims(profile, bank, onehot: bool, tip: bool = False):
    """★고정(외전) 관절은 액션에서 빠지지만 obs 의 joint_* 에는 남는다.

    ★tip 모드(hand_control="tip")는 손 액션이 관절이 아니라 손끝 5 점 × xyz 라
    frozen_hand_joints 와 무관하다 — fabric 이 손 20-DOF 를 전부 소유한다.

    이 공식이 실제 cfg 파생과 어긋나면 테스트가 자기 공식만 검증하게 된다 —
    `test_formula_matches_real_cfg` 가 그걸 막는다.
    """
    j = profile.num_arm_joints + profile.num_hand_joints
    f = len(profile.fingers)
    n_free = profile.num_hand_joints - len(profile.frozen_hand_joints)
    action = 6 + (3 * len(profile.fingertip_bodies) if tip else n_free)
    # +6 = palm 지령(slew 상태). 73c2adc 에서 obs 에 들어갔는데 이 공식이 함께
    # 안 고쳐져 115/121 을 pin 한 채 표류했다(Isaac 밖에선 importorskip 으로 숨음).
    # ★★08.25 정리분: joint pos/vel(2j) + object_pos(3) + goal(3) + action
    #   + TCP 위치(3)·자세(6) + 손끝(3T) + 물체크기(3) + fabric_q(j).
    #   critic 은 + object_rot(4) + contact(f) + 물체 6D 속도 + fabric qd/qdd(2j).
    _T = len(profile.fingertip_bodies)
    obs = (2 * j + 3 + 3 + action + 3 + 6 + 3 * _T + 3 + j
           + (bank.onehot_dim if onehot else 0))
    return action, obs, obs + 4 + f + 6 + 2 * j


@pytest.mark.parametrize("pname", sorted(_rb.PROFILES))
def test_dimension_formula_matches_profile(pname):
    p = _rb.get(pname)
    bank = _ob.get("single_cup")
    action, obs, state = _dims(p, bank, onehot=False)
    n_free = p.num_hand_joints - len(p.frozen_hand_joints)
    assert action == 6 + n_free, f"{pname}: 고정 {len(p.frozen_hand_joints)} 반영 안 됨"
    assert obs > action
    assert state == obs + 4 + len(p.fingers) + 6 + 2 * (p.num_arm_joints + p.num_hand_joints)


def test_bis_right_reference_dimensions():
    """관절 모드: _1 전체 + thumb_2 = 6 고정 → 6+14=20, obs 134, critic 203.

    ★08.25 pinky_2 를 고정에서 뺐다(굴곡축이라 얼리면 밑동이 안 접힌다) — 19→20.
    """
    action, obs, state = _dims(_rb.get("bis_right"), _ob.get("single_cup"), onehot=False)
    assert (action, obs, state) == (20, 134, 203), (action, obs, state)


def test_bis_right_reference_dimensions_tip():
    """tip 모드: 손끝 5×xyz → 6+15=21, obs 135, critic 204. 08.24 채택 배선."""
    action, obs, state = _dims(_rb.get("bis_right"), _ob.get("single_cup"),
                               onehot=False, tip=True)
    assert (action, obs, state) == (21, 135, 204), (action, obs, state)


def test_formula_matches_real_cfg():
    """★테스트가 자기 공식만 검증하지 않도록 실제 cfg 파생과 대조한다."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    _tip = (cfg.hand_control == "tip") or cfg.use_tip_fabric
    a, o, st = _dims(_rb.get(cfg.profile_name), _ob.get(cfg.object_bank),
                     cfg.enable_object_onehot, tip=_tip)
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
    # ★08.24 축소: 가중치 동기는 폐기됐다 — 두 트랙 모두 자기 손 제어에 맞는 보상으로
    #   갈라섰다(자매 tip_cyl stage_* / 우리 rewards_tip 단계형 1·2·3·5·8·12).
    #   남는 동기는 **판정·물리 상수**뿐: 이들이 갈라지면 성공률 비교가 무의미해진다.
    "contact_force_threshold",
    "goal_height_offset", "success_envelope_min", "success_tilt_max_deg",
    "success_pos_tolerance",
])
def test_reward_coefficients_match_sibling_track(key):
    """판정·물리 상수가 자매 트랙과 동기인가(가중치는 08.24 의도적 분기 — 위 주석)."""
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


def test_envelope_fingers_includes_pinky_for_tesollo():
    """감쌈 분모는 **5 지**다(08.24 복귀).

    ★★08.25 "pinky 는 굴곡축이 없다"는 구 결론과 "멀쩡하다"는 08.24 번복이 **둘 다**
    반쪽이었다. palm 좌표계 축 실측이 정확한 답을 준다:
      · index/middle/ring 은 _2·_3·_4 가 굴곡축(+y) — 밑동 포함 3 개.
      · pinky 는 _3·_4 만 굴곡축이고 _1 은 +z(회전) · _2 는 +x 다.
      · 단 pinky_1 은 굴곡 자유도를 **재분배**한다: q1=60° 에서 _2 의 굴곡성분이
        0.00 → 0.87 이 되어 밑동이 접힌다.
    즉 pinky 는 q1 을 열어 두면 5 지 감쌈이 가능하다. q1=0 에 얼렸던 것이 학습 실측
    접촉률 0.001(다른 4 지 0.50~0.86)의 원인이었다 — 분모 문제가 아니라 배선 문제.
    "모든 물체는 5 지로 감쌀 수 있다"(사용자)가 이 트랙의 목표다.
    """
    rb = pytest.importorskip("openarm.agnostic.modules.robots",
                             reason="패키지 임포트 불가")
    for name in ("bis_right", "bis_left"):
        p = next(x for x in vars(rb).values()
                 if getattr(x, "name", None) == name)
        assert p.envelope_fingers, f"{name} envelope_fingers 미정의"
        assert "pinky" in p.envelope_fingers, f"{name} 분모에 pinky 가 없다"
        assert set(p.envelope_fingers) == set(p.contact_group_a) | set(p.contact_group_b), (
            f"{name}: 감쌈 분모가 전 손가락이 아니다 {p.envelope_fingers}")


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

    src = ((_SENSOR_DIR / "rewards.py").read_text()
           + (_TASK_DIR / "rewards_tip.py").read_text()
           + (_TASK_DIR / "grasp_lift_fabric_env.py").read_text())
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
    if not ((c.hand_control == "tip") or c.use_tip_fabric):
        pytest.skip("tip 모드가 아니다(의도된 대조군이면 정상)")
    assert c.action_space == 6 + 3 * len(profile.fingertip_bodies)


def test_tip_ik_gives_fabric_the_whole_hand():
    """손 20-DOF 를 전부 fabric 이 준다 — 일부만 얼리면 fabric 이 아는 자세와 어긋난다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _apply_action"):src.index("def _apply_gravity_compensation")]
    assert "self.hand_ids" in body, "tip 모드에서 손 전체에 목표를 주지 않는다"
    # ★리터럴이 아니라 **의미**로 본다: 손 목표가 fabric_q 를 손 인덱스로 자른 것인가.
    #   (08.25 num_arm_joints 를 지역변수 n_arm 으로 뽑으면서 리터럴 검사가 깨졌다)
    assert "self.fabric_q[:, n_arm:][:, self._hand_from_fab]" in body, (
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


def test_left_fabric_forwards_every_parent_init_arg():
    """좌팔 fabric 서브클래스가 상위 __init__ 인자를 **빠짐없이** super 로 넘기는가.

    ★같은 부류의 버그가 두 번 났다: `hand_mode` 미전달로 좌팔 부팅이 TypeError 로 죽었고,
      `use_hand_repulsion` 은 시그니처에만 있고 전달되지 않아 좌팔에서 **조용히 꺼져** 있었다
      (cfg 에는 true 인데 fabric 은 기본값으로 동작 — 로그가 없어야 알아챈다).
      조용히 틀리는 쪽이 더 위험해서 정적으로 막는다.
    """
    import ast

    src = Path("source/FABRICS/src/fabrics_sim/fabrics/openarm_tesollo_pose_fabric.py")
    if not src.exists():
        pytest.skip("fabrics 소스 없음")
    tree = ast.parse(src.read_text())
    cls = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    parent = cls.get("OpenArmTeoslloPoseFabric")
    child = cls.get("OpenArmTeoslloLeftPoseFabric")
    assert parent and child

    def init_kwargs(c):
        fn = next(f for f in c.body
                  if isinstance(f, ast.FunctionDef) and f.name == "__init__")
        return fn, {a.arg for a in fn.args.args if a.arg != "self"} | {
            a.arg for a in fn.args.kwonlyargs}

    _, p_args = init_kwargs(parent)
    c_fn, c_args = init_kwargs(child)
    call = next(n for n in ast.walk(c_fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "__init__")
    forwarded = {k.arg for k in call.keywords if k.arg} | {
        a.id for a in call.args if isinstance(a, ast.Name)}
    # 자식이 받아들이면서 부모도 받는 인자는 전부 넘겨야 한다
    missing = sorted((c_args & p_args) - forwarded)
    assert not missing, f"좌팔이 super 로 안 넘기는 인자: {missing}"


# =============================================================================
# 08.24 tip 배선 — 이번에 고친 결함 3 건을 고정한다.
# =============================================================================
def test_tip_per_finger_is_passed_to_fabric():
    """`tip_per_finger` 가 fabric 생성자에 **전달**되는지.

    fabric 층에는 08.23 부터 구현돼 있었는데 env 가 넘기지 않아 저장소 전체에서
    이 플래그를 쓰는 곳이 fabric 파일 자신뿐이었다 — 켤 수단이 없는 기능이었다.
    좌팔 `use_hand_repulsion` 이 super 로 안 넘어가 조용히 꺼져 있던 것과 같은 부류라
    같은 방식(호출 인자 존재 검사)으로 막는다.
    """
    import ast
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw = {k.arg for k in node.keywords if k.arg}
        if "use_tip_fabric" in kw:                     # fabric 생성 호출
            found = True
            assert "tip_per_finger" in kw, (
                "fabric 생성에 tip_per_finger 를 넘기지 않는다 — 손가락별 taskmap 이 "
                "조용히 꺼진다(fabric 기본값 False)."
            )
    assert found, "fabric 생성 호출(use_tip_fabric=)을 찾지 못했다"


def test_fabrics_dt_matches_physics_rate():
    """fabric 시간이 실시간보다 빨리 흐르면 계획 궤적을 실기가 못 따라간다.

    정책 스텝 = decimation × sim.dt. 그 안에서 fabric 은 fabric_decimation 번
    fabrics_dt 로 적분하므로 둘이 같아야 한다. 08.24 이전에는 1/60 × 2 = 1/30 로
    **2 배** 흘렀다(자매 트랙 grasp_sensor 는 1/120 로 맞춰져 있었다).
    """
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    policy_dt = cfg.decimation * cfg.sim.dt
    fabric_dt = cfg.fabric_decimation * cfg.fabrics_dt
    assert abs(policy_dt - fabric_dt) < 1e-9, (
        f"정책 스텝 {policy_dt:.6f}s vs fabric {fabric_dt:.6f}s — "
        f"fabric 이 {fabric_dt/policy_dt:.1f}배 속도로 계획한다"
    )


def test_finger_crossing_has_some_guard():
    """손가락 교차를 막는 장치가 **하나는** 있어야 한다.

    셋 중 하나면 된다: ①PhysX self-collision ②Fabrics hand_repulsion
    ③외전 관절 고정(frozen_hand_joints). 단 ③은 tip 모드에서 적용되지 않으므로
    (fabric 이 손 20-DOF 를 전부 소유) 그 모드에서는 ①이나 ②가 필요하다.
    """
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    tip = (cfg.hand_control == "tip") or cfg.use_tip_fabric
    guards = []
    if cfg.enable_self_collisions:
        guards.append("PhysX self-collision")
    if cfg.use_hand_repulsion:
        guards.append("Fabrics hand_repulsion")
    if not tip and _rb.get(cfg.profile_name).frozen_hand_joints:
        guards.append("frozen_hand_joints")
    assert guards, (
        "손가락 교차를 막는 장치가 하나도 없다 — "
        "enable_self_collisions / use_hand_repulsion 중 하나는 켜야 한다"
    )


def test_envelope_is_graded_by_phalanx_not_or():
    """감쌈 판정은 손가락별 OR 이 아니라 **마디 부분 점수**다.

    OR 이면 `_3` 만 닿아도 그 손가락이 1 로 세어져 받치기와 감쌈이 같은 점수가 된다
    (사용자 지적). 실측으로 같은 정책이 느슨(OR) 0.50 · 엄격(전 마디 AND) 0.069 로
    7 배 벌어진다. AND 는 손가락마다 닿는 마디가 달라(grasp_v1 실측) 유효한 파지도
    0 으로 세므로, 0 → 0.5 → 1.0 사다리를 쓴다.
    """
    torch = pytest.importorskip("torch")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rt", _TASK_DIR / "rewards_tip.py")
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    # 상대 import 를 피하려고 함수 본문만 떼어 실행한다(테스트가 패키지 로드에 안 묶인다).
    body = src[src.index("def envelope_fraction_graded"):src.index("def compute_tip_rewards")]
    ns: dict = {"torch": torch}
    exec(compile(body, "rewards_tip:graded", "exec"), ns)
    f = ns["envelope_fraction_graded"]

    def call(mid, dist):
        return float(f(torch.tensor([mid], dtype=torch.float),
                       torch.tensor([dist], dtype=torch.float), 1.0))

    assert call([2] * 5, [2] * 5) == pytest.approx(1.0), "5 지 전 마디 = 1.0"
    assert call([2] * 5, [0] * 5) == pytest.approx(0.5), (
        "★_3 만 닿은 5 지가 1.0 으로 세어진다 — OR 판정으로 되돌아갔다")
    assert call([0] * 5, [2] * 5) == pytest.approx(0.5), "_4 만도 절반"
    assert call([2, 2, 0, 0, 0], [2, 2, 0, 0, 0]) == pytest.approx(0.4), "2 지 rim-hook"
    assert call([0] * 5, [0] * 5) == pytest.approx(0.0), "무접촉 = 0"


def test_staged_weights_are_monotone():
    """단계 가중치 1<2<3<5<8<12 — 다음 단계가 항상 커야 앞 단계 국소해가 없다."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    seq = [cfg.approach_weight, cfg.grip_weight, cfg.envelope_weight,
           cfg.lift_weight, cfg.tracking_weight, cfg.success_weight]
    assert seq == sorted(seq) and len(set(seq)) == len(seq), f"단계 역전: {seq}"
    assert seq == [1.0, 2.0, 3.0, 5.0, 8.0, 12.0], f"확정 구조와 다름: {seq}"


def test_grip_term_turns_off_after_touch_and_has_floor():
    """파지(grip) 항 계약 2건.

    ①닿은 손가락은 off — 안 끄면 접촉 후에도 계속 당겨 압입이 된다(자매 실측
      팁 압입 28~46N 의 동역학). ②floor(팁 반경 9mm) 아래는 보상 증가 없음 —
      접촉 감지가 실패해도 압입이 무한보상이 되지 않는다.
    """
    torch = pytest.importorskip("torch")
    import types
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    assert "open_f = (~finger_touch).float()" in src, "touch-off 가 사라졌다"
    assert "grip_dist_floor" in src, "거리 floor 가 사라졌다"
    # 수치: 같은 거리라도 touch=True 면 기여 0
    cfg = types.SimpleNamespace(grip_sharpness=20.0, grip_dist_floor=0.009)
    d = torch.full((1, 5), 0.03)
    k = torch.exp(-cfg.grip_sharpness * (d - cfg.grip_dist_floor).clamp(min=0.0))
    touched = torch.tensor([[True, True, True, True, True]])
    assert float(((~touched).float() * k).mean()) == 0.0
    # floor: 9mm 이하 어디서든 커널이 동일(=압입 이득 0)
    k1 = torch.exp(-cfg.grip_sharpness * (torch.tensor(0.009) - cfg.grip_dist_floor).clamp(min=0.0))
    k2 = torch.exp(-cfg.grip_sharpness * (torch.tensor(0.001) - cfg.grip_dist_floor).clamp(min=0.0))
    assert float(k1) == float(k2) == 1.0


def test_palmar_filter_applies_to_envelope_only():
    """손바닥면 필터는 감쌈 **입력만** 거른다 — 대향 게이트·obs 는 불변(s2r 규약)."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "def _palmar_mask" in src
    assert "mid_use = mid_f * _pal" in src and "dist_use = dist_f * _pal" in src, (
        "필터가 감쌈 입력에 적용되지 않는다")
    assert "env_mid_force=mid_use" in src and "env_dist_force=dist_use" in src
    # 게이트는 필터 안 거친 원값을 쓴다 (mid_use/dist_use 는 게이트에 안 들어간다)
    tip_call = src[src.index("compute_tip_rewards("):]
    tip_call = tip_call[:tip_call.index("cfg=self.cfg")]
    assert "group_a_force=contact[" in tip_call, "대향 게이트가 필터에 오염됐다"
    assert "group_a_force=mid_use" not in tip_call and "group_b_force=dist_use" not in tip_call
    assert "task/envelope_frac_raw" in src, "필터 전 비교선 로깅이 없다"


def test_registered_tesollo_profiles_define_palmar_axes():
    """등록 tesollo 프로필은 palmar 축 필수(영벡터 금지) — 미정의는 env fail-loud."""
    rb = pytest.importorskip("openarm.agnostic.modules.robots",
                             reason="패키지 임포트 불가")
    for name in ("bis_right", "bis_left"):
        p = next(x for x in vars(rb).values() if getattr(x, "name", None) == name)
        missing = [f for f in p.fingers if f not in p.palmar_axis_local]
        assert not missing, f"{name} palmar_axis_local 미정의: {missing}"
        for f, ax in p.palmar_axis_local.items():
            assert any(abs(v) > 1e-6 for v in ax), f"{name}/{f} 영벡터"
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "palmar_axis_local 미정의" in env_src, "부팅 fail-loud 가 없다"


def test_envelope_gate_saturation_decoupled_from_success_threshold():
    """success 램프 분모(0.85) > 판정 임계(0.6) — 같으면 3지에서 4·5지 유인 소멸."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    assert cfg.envelope_gate_saturation > cfg.success_envelope_min
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    assert "envelope_gate_saturation" in src, "rewards_tip 이 saturation 을 안 읽는다"


def test_approach_is_palm_only_in_tip_mode():
    """tip 모드 approach 는 **팔 전용**(파지중심 기준)이다(08.24 사용자 지시).

    max over palm+손끝 커널은 손끝을 물체 중심으로 당기는데 grasp_radial 은 같은
    손끝을 표면 반경으로 당긴다 — 한 손끝에 반대 방향 gradient 두 개. palm 전용이면
    팔 액션 ← approach, 손가락 액션 ← 자세 3 항으로 역할이 갈린다.
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    body = src[src.index("def compute_tip_rewards"):]
    seg = body[body.index("approach = "):body.index("approach = ") + 200]
    # ★기준점은 palm **원점이 아니라 파지중심**이다. 원점 기준은 최적점 d=0 이
    #   "손바닥으로 컵을 관통"이라 목표가 물리적으로 틀리다(유효 파지 ~150mm 뒤).
    assert "d_grasp" in seg, "approach 가 파지중심 거리를 쓰지 않는다"
    pre_seg = body[:body.index("approach = ")]
    assert "grasp_center_pos" in pre_seg, "d_grasp 가 파지중심에서 오지 않는다"
    # max-over-points 커널로의 회귀를 막는다
    pre = body[:body.index("approach = ")]
    assert ".max(dim=-1)" not in pre.split("# ①")[-1], (
        "approach 계산에 max-over-points 가 되살아났다")


def test_pinky_flexion_is_not_frozen_for_tesollo():
    """★pinky 의 **굴곡** 관절을 얼리지 않는가.

    pinky 만 _1=회전(+z) / _2=굴곡(+x, q1 에 따라 +y 로 전환)으로 뒤바뀌어 있다.
    "_1 은 전부 외전이니 얼린다"는 규칙을 그대로 적용해 _1·_2 를 둘 다 얼렸더니
    밑동이 아예 안 접혀 학습 실측 접촉률이 0.001 이었다(다른 4 지 0.50~0.86).
    계약: _2 는 자유여야 하고, _1 은 홈에서 한계각(±60°)에 고정돼 _2 를 굴곡축으로
    돌려놔야 한다.
    """
    for name, sign in (("bis_right", +1.0), ("bis_left", -1.0)):
        p = _rb.get(name)
        side = "r" if sign > 0 else "l"
        frozen = set(p.frozen_hand_joints)
        assert f"{side}_hj_pinky_2" not in frozen, (
            f"{name}: pinky_2(굴곡)가 고정돼 밑동이 안 접힌다")
        assert f"{side}_hj_pinky_1" in frozen, (
            f"{name}: pinky_1 은 고정이어야 한다(굴곡 배분을 정하는 자세 파라미터)")
        q1 = p.init_joint_pos[f"{side}_hj_pinky_1"]
        assert abs(abs(q1) - 1.047) < 1e-3, f"{name}: pinky_1 홈이 60° 가 아니다({q1})"
        assert q1 * sign > 0, (
            f"{name}: pinky_1 부호가 좌우 규약과 반대다({q1}) — 한계는 우 [0,+60]·좌 [-60,0]")


def test_lift_and_tracking_are_gated_by_envelope_ramp():
    """★리프트·이송이 대향 접촉만으로 열리지 않는가.

    실측(pdg_l 231 iter): lift 3.169 + tracking 3.430 = 6.60 vs grip 0.167 +
    envelope 0.578 = 0.745 → 8.9 배. 2~3 지 핀치로도 gate 가 0.83 이라 "대충 잡고
    들기"가 압도적 이득이었고, grip 이 0.350 → 0.167 로 반토막 나며 후퇴했다.
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    seg = src[src.index("env_gate = "):src.index("# ⑥ 성공")]
    assert "gf_env = gf * env_gate" in seg, "리프트·이송 게이트에 감쌈 램프가 없다"
    assert "* gf_env" in seg[seg.index("lift = "):], "lift 가 감쌈 램프를 안 쓴다"
    assert "* gf_env" in seg[seg.index("tracking = "):], "tracking 이 감쌈 램프를 안 쓴다"


def test_lift_envelope_ramp_has_no_floor():
    """★감쌈 램프에 하한을 두지 않는가 — 자매 트랙이 하한 0.3 으로 반증한 패턴.

    하한이 있으면 감쌈 0 에서도 리프트·이송이 흐르므로 지금 막으려는 구멍이
    그대로 남는다(자매: 하한 0.3 → 감쌈 0.21 고착, 이송만 학습).
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    seg = src[src.index("env_gate = "):src.index("# ⑥ 성공")]
    code = "\n".join(l for l in seg.splitlines() if not l.strip().startswith("#"))
    for banned in ("floor", "lift_envelope_floor"):
        assert banned not in code, f"감쌈 램프에 하한 {banned} 가 들어갔다"


def test_approach_target_is_midpoint_of_opposing_groups():
    """★approach 기준점이 대향 두 그룹의 **중점**인가.

    손끝 5 점 평균은 어느 손가락에도 없는 허공의 점이다 — thumb 이 대향해 손바닥에
    붙어 있어(홈 z=+30) 4 지(z=+190)와 평균하면 중간 빈 공간이 나온다. FK 실측으로
    진짜 파지점과 35~60mm 괴리였고, 실제로 양팔 독립 런이 똑같이 79mm 에서 멈추고
    후퇴했다(정책이 완벽해도 그 밑으로 못 내려가는 하한).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _get_rewards"):src.index("def _get_dones")]
    assert "_gc_w = 0.5 * (tips[:, self._grp_a].mean(dim=1)" in body, (
        "approach 기준점이 대향 그룹 중점이 아니다")
    assert "grasp_center_pos=_gc_w," in body, (
        "tips 는 이미 env-local 이다 — _local() 을 다시 걸면 원점을 두 번 뺀다")
    ws = src[src.index("def _measure_tip_workspace"):src.index("def _step_fabric")]
    assert "rel[0].mean(dim=0)" not in ws, "파지중심 상수가 아직 5 점 평균이다"


def test_fabric_velocity_is_fed_forward_to_pd():
    """★fabric 계획 **속도**를 PD 속도 목표로 내리는가 — 0 을 넣으면 안 된다.

    속도목표 0 : kp·err = kd·v + τ_마찰 → err ≈ (kd/kp)·v = 0.2·v [rad]  (kp=400,kd=80)
    속도목표 qd: kp·err = τ_마찰만      → err ≈ τ_f/kp
    즉 0 을 넣으면 PD 감쇠항이 움직임을 되밀어 **속도에 정비례하는** 추종오차가 된다.
    DEXTRAH 원본은 dof_vel_targets = clone(fabric_qd) 를 내리고 velocity_target_factor
    를 ADR 로 1.0 → 0.0 으로 줄여 간다 — 우리는 그 최종 단계에서 시작하고 있었다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _apply_action"):src.index("def _apply_gravity_compensation")]
    assert "torch.zeros_like(arm_target)" not in body, (
        "팔 속도 목표에 0 을 넣고 있다 — fabric_qd 를 버리는 배선")
    assert "self.fabric_qd[:, :n_arm]" in body, "팔 속도 피드포워드가 없다"
    assert "velocity_target_factor" in body, "피드포워드 계수가 cfg 로 노출되지 않았다"


def test_velocity_target_factor_is_declared_and_unity_by_default():
    """계수 기본값이 1.0(완전 피드포워드)인가 — ADR 로 낮추는 건 그 다음이다."""
    # ★isaaclab 없는 환경에서도 돌도록 소스 텍스트로 검사한다(다른 계약과 같은 방식).
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"velocity_target_factor:\s*float\s*=\s*([0-9.]+)", src)
    assert m, "velocity_target_factor 미선언"
    assert abs(float(m.group(1)) - 1.0) < 1e-9, (
        f"기본값이 1.0 이 아니다({m.group(1)}) — 0 이면 구 결함 재현")


# ===================================================================
# KUKA 고정(08.25) — 원본 dextrah_kuka_allegro 정합 계약
# ===================================================================

def test_fabrics_dt_is_kuka_two_times_wallclock():
    """★fabric 시간이 벽시계의 2 배속인가 — 원본 규약.

    원본: sim_dt 1/120 · decimation 2(정책 1/60s) · fabrics_dt 1/60 · fabric_decimation 2
          → 정책 스텝당 fabric 1/30s = 벽시계 2 배속. 계획이 앞서야 PD 가 따라잡는다.
    08.24 에 이를 "2 배로 앞서간다"며 1/120 으로 낮췄다가 A(정책목표→계획) 오차가
    perr 의 99% 를 차지했다 — 그 되돌림을 고정한다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"fabrics_dt:\s*float\s*=\s*1\.0\s*/\s*([0-9.]+)", src)
    assert m and abs(float(m.group(1)) - 60.0) < 1e-9, (
        f"fabrics_dt 가 1/60 이 아니다({m.group(1) if m else '미발견'})")
    d = re.search(r"fabric_decimation:\s*int\s*=\s*(\d+)", src)
    assert d and int(d.group(1)) == 2, "fabric_decimation 이 2 가 아니다"


def test_palm_slew_is_disabled_like_kuka():
    """★원본에 rate limit 이 없다(전 파일 grep 0건) — fabric 자체가 rate limiter 다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    for k in ("palm_slew_pos", "palm_slew_rot_deg"):
        m = re.search(rf"{k}:\s*float\s*=\s*([0-9.]+)", src)
        assert m and float(m.group(1)) == 0.0, f"{k} 가 0 이 아니다 — slew 재도입"


def test_actions_use_kuka_absolute_box_mapping():
    """★팔·손끝 모두 원본 compute_absolute_action 과 같은 절대 박스 매핑인가."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "def _kuka_absolute(" in src, "절대 박스 매핑 헬퍼가 없다"
    assert "0.5 * (hi - lo) * a + 0.5 * (hi + lo)" in src, "scale() 식이 원본과 다르다"
    body = src[src.index("def _pre_physics_step"):src.index("def _step_fabric")]
    assert body.count("_kuka_absolute(") == 2, (
        "팔·손끝 중 한쪽이 구 매핑(홈 기준 구간별 선형)을 쓴다")
    cfg = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    assert re.search(r'tip_action_mode:\s*str\s*=\s*"absolute"', cfg), (
        "tip_action_mode 가 absolute 가 아니다 — 원본에 누산 모드가 없다")


def test_obs_noise_starts_at_zero_like_kuka_adr():
    """★관측 노이즈는 ADR 축이고 **시작이 0** 이어야 한다.

    원본은 object_pos_noise (0→0.03) 처럼 전부 0 에서 시작한다. 처음부터 걸면
    "ADR 끝점에서 시작"이 되어 과제 성립을 방해한다(velocity_target_factor 0 ·
    fabric_damping 20 에서 이미 겪은 실수와 같은 부류).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"obs_noise_scale:\s*float\s*=\s*([0-9.]+)", src)
    assert m and float(m.group(1)) == 0.0, "obs_noise_scale 이 0 에서 시작하지 않는다"


def test_fabric_damping_starts_at_adr_initial():
    """★cspace 감쇠는 원본 ADR 10 → 20 의 **시작값** 10 이어야 한다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"fabrics_damping_gain:\s*float\s*=\s*([0-9.]+)", src)
    assert m and abs(float(m.group(1)) - 10.0) < 1e-9, (
        f"fabrics_damping_gain 이 10 이 아니다({m.group(1) if m else '미발견'}) — ADR 끝점 고정")


def test_agent_network_is_kuka_lstm():
    """★actor LSTM 1024(before_mlp) + critic LSTM 2048(after mlp) — 원본 teacher 구조."""
    import yaml
    y = yaml.safe_load((_TASK_DIR / "config/agents/rl_games_ppo_cfg.yaml").read_text())
    net = y["params"]["network"]
    assert net["mlp"]["units"] == [512, 512], net["mlp"]["units"]
    rnn = net.get("rnn")
    assert rnn and rnn["name"] == "lstm" and rnn["units"] == 1024, rnn
    assert rnn["before_mlp"] is True and rnn["layer_norm"] is True
    cv = y["params"]["config"]["central_value_config"]["network"]
    assert cv["mlp"]["units"] == [1024, 512], cv["mlp"]["units"]
    assert cv["rnn"]["units"] == 2048 and cv["rnn"]["before_mlp"] is False, cv["rnn"]
    c = y["params"]["config"]
    assert c["horizon_length"] == 16 and c["seq_length"] == 16, "BPTT 창이 원본과 다르다"
    assert c["zero_rnn_on_done"] is True, "에피소드 경계 은닉상태 리셋이 꺼져 있다"


def test_hand_damping_ratio_matches_kuka():
    """★손 kd/kp 가 원본 allegro 비율(0.033)인가 — 구 0.40 은 12 배 과감쇠였다.

    손은 fabric 이 아니라 PD 직결이라 속도 목표가 없다 — 감쇠항이 그대로 지연이 된다.
    """
    src = (_TASK_DIR.parent.parent / "modules/robots.py").read_text()
    m = re.search(r"_HAND_GAINS = dict\(stiffness=([0-9.]+), damping=([0-9.]+)", src)
    assert m, "_HAND_GAINS 미발견"
    kp, kd = float(m.group(1)), float(m.group(2))
    assert abs(kd / kp - 0.033) < 0.01, f"손 kd/kp = {kd/kp:.3f} (원본 0.033)"


# =============================================================================
# KUKA 정합 — 물리 재질·강체 속성·관측 프레임 (08.25)
# =============================================================================
def test_sim_friction_matches_kuka():
    """★★IsaacLab 기본 물리 재질은 static/dynamic 0.5 인데 원본은 **둘 다 1.0** 이다.

    파지 태스크에서 마찰 2 배는 미끄러짐 한계 하중 2 배 — "쥘 수 있는가"가 바뀐다.
    명시하지 않으면 조용히 기본 0.5 로 돈다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"physics_material=RigidBodyMaterialCfg\((.*?)\)", src, re.S)
    assert m, "sim.physics_material 미지정 — IsaacLab 기본 0.5/0.5 로 돈다"
    body = m.group(1)
    for key in ("static_friction=1.0", "dynamic_friction=1.0"):
        assert key in body, f"{key} 가 없다: {body!r}"


def test_robot_rigid_props_match_kuka():
    """★원본이 명시하는 강체 속성 — 미지정이면 **USD 값**을 쓰므로 자산마다 갈린다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    for key in ("retain_accelerations=True", "linear_damping=0.0",
                "angular_damping=0.0", "max_linear_velocity=1000.0",
                "max_angular_velocity=1000.0", "sleep_threshold=0.005",
                "stabilization_threshold=0.0005"):
        assert key in src, f"강체 속성 {key} 가 KUKA 와 다르다"
    assert 'JointDrivePropertiesCfg(drive_type="force")' in src, (
        "관절 구동 타입이 'force' 로 고정돼 있지 않다 — acceleration 이면 관성이 무시된다")


def test_obs_uses_base_frame_like_kuka():
    """★★원본 teacher 관측은 전부 env-local(로봇 베이스) 기준이다.

    palm 상대 좌표로 주면 palm 자신의 pose 가 관측에서 사라진다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    blk = src[src.index("def _get_observations"):src.index("def _get_rewards")]
    assert "subtract_frame_transforms" not in blk, "물체 pose 가 아직 palm 프레임이다"
    assert "self.scene.env_origins" in blk, "손 위치가 env-local 이 아니다"


def test_tcp_is_palm_ee_with_orientation():
    """★★손 TCP = `palm_ee` 이고 **자세까지** 관측해야 한다.

    사용자 확인: palm_ee 의 +x 축이 손바닥 법선이다. URDF 상 palm_ee 의 rpy 는 0 이라
    회전이 palm 과 같고, 오프셋 (0.028,0,0.04) 방향은 법선과 다르다 — 즉 **위치만
    넣으면 정렬(align)을 볼 수 없다**. 회전행렬 열을 넣어야 접근 축이 관측된다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "self._tcp_idx" in src, "TCP 인덱스가 없다"
    assert "palm_ee_body" in src, "palm_ee 를 프로필에서 읽지 않는다"
    blk = src[src.index("def _get_observations"):src.index("def _get_rewards")]
    assert "tcp_pos" in blk and "tcp_axes" in blk, "TCP 위치·자세가 obs 에 없다"
    assert "matrix_from_quat" in blk, "자세를 회전행렬로 넣지 않는다(quat 은 부호 이중성)"


@pytest.mark.parametrize("pname", ["bis_right", "bis_left"])
def test_bis_profiles_define_palm_ee(pname):
    """★bis 자산에는 palm_ee 프레임이 있다 — 없으면 정렬 관측이 palm 원점으로 퇴화한다."""
    p = _rb.get(pname)
    assert p.palm_ee_body, f"{pname}: palm_ee_body 미정의"
    assert p.palm_ee_body.endswith("_palm_ee"), p.palm_ee_body


def test_fabric_q_in_policy_and_velocities_only_in_critic():
    """★policy 는 `fabric_q`(계획 상태)를 받고, qd/qdd 는 **critic 전용**이다.

    원본은 셋 다 policy 에 넣지만 `observation_annealing` 계수가 (0,0) 이라 qd/qdd 가
    **항상 0** 이다(같은 계수가 원본의 joint_vel·hand_vel 까지 죽인다). 상수 0 을 LSTM 에
    통과시킬 이유가 없어 policy 에서 빼고 critic 에만 실값으로 준다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    blk = src[src.index("def _get_observations"):src.index("def _get_rewards")]
    # ★주석에도 fabric_qd 가 나오므로 **parts 리스트 본문**만 본다.
    parts = blk[blk.index("parts = ["):blk.index("obs = torch.cat(")]
    assert "fabric_q" in parts, "policy obs 에 fabric_q 가 없다"
    assert "fabric_qd" not in parts and "fabric_qdd" not in parts, (
        f"policy obs 에 fabric_qd/qdd 가 남아 있다 — 항상 0 인 상수다: {parts}")
    st = blk[blk.index("state = torch.cat("):]
    assert "self.fabric_qd" in st and "self.fabric_qdd" in st, st[:300]


def test_critic_sees_unannealed_fabric_velocity():
    """★원본 critic 은 annealing 이 걸리지 않은 fabric_qd/qdd 를 본다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    blk = src[src.index("def _get_observations"):src.index("def _get_rewards")]
    st = blk[blk.index("state = torch.cat("):]
    assert "self.fabric_qd" in st and "self.fabric_qdd" in st, st[:400]


def test_obs_annealing_starts_at_zero_like_kuka():
    """★원본 ADR 범위가 (0., 0.) — 시작도 끝도 0 이다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"obs_annealing_coefficient:\s*float\s*=\s*([0-9.]+)", src)
    assert m and float(m.group(1)) == 0.0, "observation_annealing 시작값이 0 이 아니다"


# =============================================================================
# KUKA ADR 축 전수 연결 (08.25) — 시작값만 맞추면 커리큘럼이 아니라 고정값이다
# =============================================================================
_KUKA_ADR_GROUPS = ("spawn", "goal", "pd_targets", "fabric_damping",
                    "robot_spawn", "object_state_noise", "object_wrench")


@pytest.mark.parametrize("group", _KUKA_ADR_GROUPS)
def test_kuka_adr_group_is_wired(group):
    """★★원본 ADR 13 그룹 중 보상 가중치 축을 뺀 전부가 TaskADR 에 연결돼야 한다.

    `fabrics_damping_gain=10` 처럼 **시작값만** 원본과 맞추고 끝으로 가는 경로가 없으면
    커리큘럼이 죽은 채 고정값으로 도는 것이다 — 08.25 대조에서 6 그룹이 그 상태였다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    blk = src[src.index("self.adr = _adr.TaskADR("):]
    blk = blk[:blk.index("enabled=")]
    assert f'"{group}"' in blk, f"ADR 그룹 '{group}' 이 연결되지 않았다"


def test_adr_params_are_read_not_frozen():
    """★ADR 값은 **매 스텝 읽어야** 한다 — 부팅 시 한 번 읽으면 증분이 반영되지 않는다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for group, name in (("pd_targets", "velocity_target_factor"),
                        ("fabric_damping", "gain"),
                        ("object_state_noise", "scale"),
                        ("robot_spawn", "joint_pos_noise"),
                        ("object_wrench", "max_linear_accel")):
        assert f'get_param("{group}", "{name}")' in src, (
            f"{group}.{name} 을 ADR 에서 읽지 않는다 — cfg 값에 고정돼 있다")


def test_object_wrench_exists_and_is_gated():
    """★원본 `apply_object_wrench` — 1 초마다, 손이 가까울 때만, 가속도로 준다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "_apply_object_wrench" in src, "외란 렌치가 구현되지 않았다"
    blk = src[src.index("def _apply_object_wrench"):]
    blk = blk[:blk.index("\n    def ", 10)]
    assert "wrench_trigger_every" in blk, "주기 게이트가 없다"
    assert "wrench_hand_distance_threshold" in blk, "손↔물체 거리 게이트가 없다"
    assert "mass" in blk, "가속도가 아니라 힘으로 주면 질량 DR 과 곱해진다"
    assert "_step_fabric()" in src and "_apply_object_wrench()" in src


def test_reset_restores_velocity_target():
    """★원본은 리셋에서 위치·속도 목표를 **둘 다** 실제 상태로 되돌린다."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    blk = src[src.index("def _reset_idx"):]
    _end = blk.find("\n    def ", 10)
    blk = blk if _end < 0 else blk[:_end]
    assert "set_joint_velocity_target" in blk, (
        "리셋에서 속도 목표를 안 되돌린다 — 직전 에피소드 목표가 샌다")
    assert "fabric_qd[env_ids] = qd0" in blk, "fabric 속도가 리셋 상태와 동기화되지 않는다"


def test_object_rigid_props_match_kuka():
    """★원본 물체 강체 속성 — 로봇과 `stabilization_threshold` 가 **다르다**(0.0025 vs 0.0005)."""
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    blk = src[src.index("def build_object_cfg"):src.index("# ---------------------------------------------------------------------------\n# 환경 픽스처")]
    for key in ("enable_gyroscopic_forces=True", "sleep_threshold=0.005",
                "stabilization_threshold=0.0025", "max_angular_velocity=1000.0",
                "max_linear_velocity=1000.0", "disable_gravity=False"):
        assert key in blk, f"물체 강체 속성 {key} 가 KUKA 와 다르다"


def test_gravity_and_self_collision_match_kuka():
    """★로봇 중력 OFF · 자기충돌 ON — 원본 KUKA_ALLEGRO_CFG 와 같은 조합.

    Fabrics 는 중력보상을 하지 않으므로 중력을 켜면 PD 정상상태 오차가 그대로 처짐이
    된다. 원본도 같은 이유로 `disable_gravity=True` 다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"enable_gravity:\s*bool\s*=\s*(\w+)", src)
    assert m and m.group(1) == "False", "로봇 중력이 KUKA(OFF)와 다르다"
    m = re.search(r"enable_self_collisions:\s*bool\s*=\s*(\w+)", src)
    assert m and m.group(1) == "True", "자기충돌이 KUKA(ON)와 다르다"


def test_left_fabric_forwards_parent_args():
    """★★좌팔 서브클래스가 상위 fabric 의 제어 인자를 **전부** 받아 전달하는가.

    같은 사고가 두 번 났다 — 08.23 `hand_mode`, 08.25 `use_body_repulsion_pairs`.
    둘 다 상위 클래스에만 인자를 추가하고 좌팔 서브클래스를 안 고쳐서, 우팔은 멀쩡히
    도는데 **좌팔만 TypeError 로 죽었다**(부팅 직후라 로그에 params 만 남는다).

    허용 누락 3 개는 좌팔이 **자체 값을 하드코딩해 넘기는 것**이라 시그니처에 없는 게 맞다.
    """
    import ast

    src = (_TASK_DIR.parents[4] / "FABRICS/src/fabrics_sim/fabrics"
           / "openarm_tesollo_pose_fabric.py")
    if not src.is_file():                      # FABRICS 가 없는 체크아웃
        pytest.skip("FABRICS 소스 없음")
    tree = ast.parse(src.read_text())
    sigs = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "__init__":
                sigs[node.name] = [a.arg for a in m.args.args if a.arg != "self"]
    base = set(sigs.get("OpenArmTeoslloPoseFabric", []))
    assert base, "상위 fabric 클래스를 못 찾았다"
    left = set(sigs.get("OpenArmTeoslloLeftPoseFabric", []))
    assert left, "좌팔 fabric 클래스를 못 찾았다"
    # 좌팔이 자체 값을 넘기는 것들 — 시그니처에 없는 게 정상
    OWN = {"default_config_override", "default_palm_euler_zyx", "fabric_params_filename"}
    missing = base - left - OWN
    assert not missing, (
        f"좌팔 fabric 이 상위 인자 {sorted(missing)} 를 안 받는다 — "
        "우팔만 돌고 좌팔은 부팅 즉시 TypeError 로 죽는다")


# =============================================================================
# 접근 정렬(align) — 08.25 사용자 지시: palm_ee +x ⊥ cup +z (측면 파지)
# =============================================================================
def test_align_term_is_gated_by_distance():
    """★★정렬 항에 거리 커널이 곱해져야 한다.

    안 곱하면 **컵에서 멀리 떨어져 자세만 맞추는 것이 만점**이 된다
    (reward-audit Check 2). approach 와 같은 커널을 써서 "가까이 가되 바른 자세로"
    를 하나의 표면으로 만든다.
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    m = re.search(r"align\s*=\s*\(1\.0 - align_cos\)\s*\*\s*(\w+)", src)
    assert m, "align 항이 없거나 형태가 다르다"
    assert m.group(1) == "approach", (
        f"align 에 거리 커널이 아니라 '{m.group(1)}' 이 곱해졌다 — "
        "원거리 자세맞추기가 만점이 된다")


def test_align_does_not_multiply_approach():
    """★approach 에 곱수로 넣으면 유일하게 작동 중인 초기 신호가 깎인다.

    별도 항이어야 한다. floor 도 없어야 한다(08.22 envelope_mul_floor 0.3 이
    감쌈 없이 이송 보상 30% 를 유출시킨 선례).
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    m = re.search(r'"approach":\s*float\(cfg\.approach_weight\)\s*\*\s*(\w+)', src)
    assert m and m.group(1) == "approach", "approach 항이 정렬로 오염됐다"
    assert '"align":' in src, "align 이 별도 항으로 등록되지 않았다"


def test_align_uses_world_z_not_object_rotation():
    """★★컵 축을 **추정하지 않는다** — 컵은 중력 반대로 서 있어야 하므로 world +z 다.

    이 선택의 핵심 이득은 s2r: 회전 추정이 불필요하고, 정책이 같은 값을
    obs(TCP 자세 x축의 z 성분)로 직접 본다 — 보상이 관측 불가능한 양의 함수가 되지 않는다.
    """
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    assert "tcp_normal_z" in src, "TCP 법선의 z 성분을 받지 않는다"
    blk = src[src.index("align_cos ="):src.index("align_cos =") + 300]
    assert "object" not in blk.split("\n")[0], (
        "정렬 판정에 물체 회전이 섞였다 — world +z 규약과 어긋난다")


def test_stage_weights_stay_monotonic_with_align():
    """★단계 단조: approach + align < grip < envelope < lift < tracking < success.

    앞 단계 합이 다음 단계를 넘으면 거기 머무는 국소해가 생긴다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()

    def _w(name):
        m = re.search(rf"{name}:\s*float\s*=\s*([0-9.]+)", src)
        assert m, f"{name} 미발견"
        return float(m.group(1))

    ap, al = _w("approach_weight"), _w("align_weight")
    grip, env = _w("grip_weight"), _w("envelope_weight")
    lift, trk, suc = _w("lift_weight"), _w("tracking_weight"), _w("success_weight")
    assert ap + al < grip, f"approach({ap})+align({al}) 이 grip({grip}) 이상이다"
    assert grip < env < lift < trk < suc, (grip, env, lift, trk, suc)
    # reward-audit: 정렬 고착 위험은 목표 대비 배율로 본다(과거 weight_align=10.0 실패)
    assert al / suc < 0.1, f"align 이 success 대비 {al/suc:.2f} — 정렬 고착 위험"


def test_paper_agent_cfg_has_mlp_critic():
    """★★`-paper` 태스크는 DEXTRAH **논문 E.6** 구성이다 — critic 이 MLP 여야 한다.

    논문: "크리틱은 모든 특권 정보에 접근할 수 있으므로 시간적 의존성을 포착할 필요가
    없으며, 이에 따라 MLP 네트워크를 사용한다." 반면 레포 yaml 은 critic 에도 LSTM
    2048 을 둔다(논문 이후 코드가 바뀐 것으로 보인다). 둘 다 돌릴 수 있어야 하므로
    기본 태스크와 `-paper` 를 나눠 등록하고, 이 계약이 둘이 섞이지 않게 막는다.
    """
    import yaml

    base = yaml.safe_load(
        (_TASK_DIR / "config/agents/rl_games_ppo_cfg.yaml").read_text())
    paper = yaml.safe_load(
        (_TASK_DIR / "config/agents/rl_games_ppo_paper_cfg.yaml").read_text())

    def _cv_rnn(d):
        return d["params"]["config"]["central_value_config"]["network"].get("rnn")

    assert _cv_rnn(base) is not None, "기본 yaml 은 레포 구성(critic LSTM)이어야 한다"
    assert _cv_rnn(paper) is None, "-paper yaml 의 critic 에 LSTM 이 남아 있다"
    # actor 는 둘 다 같아야 한다 — 한 번에 하나씩만 바꾼다
    for d in (base, paper):
        rnn = d["params"]["network"].get("rnn")
        assert rnn and rnn["units"] == 1024 and rnn["before_mlp"] is True, rnn
        assert rnn["concat_input"] and rnn["concat_output"], "skip connection 이 없다"
