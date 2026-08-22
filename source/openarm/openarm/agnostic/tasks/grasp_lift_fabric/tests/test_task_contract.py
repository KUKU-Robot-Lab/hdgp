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
from openarm.agnostic.tasks.grasp_lift_fabric import rewards as RW

_TASK_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# agnosticism — 태스크 소스에 로봇 이름이 있으면 실패
# =============================================================================
_ROBOT_LITERAL = re.compile(r"\b[rl]_(aj|hj|hl)_")


@pytest.mark.parametrize(
    "fname",
    ["grasp_lift_fabric_env.py", "grasp_lift_fabric_env_cfg.py", "rewards.py"],
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
    """래치로 항을 이분하면 grasp gradient 가 죽는다(grasp_v1 실측 0.036 평탄)."""
    ids = _identifiers(_TASK_DIR / "rewards.py")
    for banned in ("lift_latched", "pre_lift_gate", "lift_gate"):
        assert banned not in ids, f"rewards.py 가 래치 게이트 {banned} 를 쓴다"


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


class _Cfg:
    contact_force_threshold = 1.0
    participation_force_threshold = 0.1
    approach_weight = 1.0
    approach_sharpness = 4.0
    grasp_quality_weight = 8.0
    grasp_envelope_credit = 0.7
    grasp_grip_credit = 0.2
    grasp_persist_credit = 0.1
    persistence_ref_steps = 20
    enclosure_radius = 0.03
    enclosure_thumb_weight = 0.6
    upright_max_deg = 60.0
    lift_weight = 2.0
    envelope_reference_frac = 0.8
    envelope_mul_floor = 0.3
    upright_weight = 2.0
    upright_lift_ref = 0.05
    lift_overshoot_start = 0.20
    lift_overshoot_scale = 0.06
    envelope_force_threshold = 0.5
    lift_success_height = 0.10
    success_weight = 10.0
    success_pos_std = 0.05
    action_rate_weight = -0.3        # 실제 cfg 와 동기(test_stub_matches_real_cfg 가 강제)


def _call(**over):
    kw = dict(
        palm_to_object=_ones(0.0), tip_side_dist=_ones(0.0),
        envelope_frac=_ones(1.0), grip_frac=_ones(1.0), persistence=_ones(1.0),
        object_height_delta=_ones(0.10), object_to_goal=_ones(0.0),
        object_xy_displacement=_ones(0.0), object_tilt_deg=_ones(0.0),
        group_a_force=torch.full((N, 1), 5.0), group_b_force=torch.full((N, 4), 5.0),
        actions=torch.zeros(N, 26), prev_actions=torch.zeros(N, 26), cfg=_Cfg(),
    )
    kw.update(over)
    return RW.compute_rewards(**kw)


def test_five_terms_exactly():
    """08.22 TEST2: upright 독립항 신설 → 6항 (우선순위 ②를 곱수가 아닌 항으로)."""
    _, terms, _, _ = _call()
    assert set(terms) == {
        "approach", "grasp_quality", "upright", "lift", "success", "action_rate",
    }


def test_gate_requires_both_opposing_groups():
    _, _, gate, _ = _call(group_b_force=torch.zeros(N, 4))
    assert not gate.any(), "그룹 B 접촉 없이 게이트가 열렸다"
    _, _, gate, _ = _call(group_a_force=torch.zeros(N, 1))
    assert not gate.any(), "그룹 A 접촉 없이 게이트가 열렸다"


def test_grasp_quality_survives_after_lift():
    """★래치가 없으므로 리프트 후에도 감쌈 보상이 살아 있어야 한다."""
    _, low, _, _ = _call(object_height_delta=_ones(0.0))
    _, high, _, _ = _call(object_height_delta=_ones(0.10))
    assert torch.allclose(low["grasp_quality"], high["grasp_quality"])
    assert (high["grasp_quality"] > 0).all()


def test_grasp_quality_drops_when_wrap_is_lost():
    """만렙 후 감쌈 침식에 반대 gradient 가 있어야 한다."""
    _, full, _, _ = _call(envelope_frac=_ones(1.0))
    _, eroded, _, _ = _call(envelope_frac=_ones(0.2))
    assert (eroded["grasp_quality"] < full["grasp_quality"]).all()


def test_no_term_is_zeroed_by_tilt_alone():
    """★grasp_v2 의 '인자 하나가 0 이면 항 전체가 0' 붕괴를 막았는지."""
    _, flat, _, _ = _call(object_tilt_deg=_ones(0.0))
    _, tipped, _, _ = _call(object_tilt_deg=_ones(90.0))
    for k in ("lift", "success"):
        assert (tipped[k] > 0).all(), f"{k} 가 자세만으로 0 이 됐다"
        assert (tipped[k] < flat[k]).all(), f"{k} 가 자세를 반영하지 않는다"


def test_persistence_raises_grasp_quality():
    """08.22 B: 대향 접촉을 연속 유지하면 grasp_quality 가 더 크다(잡았다-놓기 억제)."""
    _, held, _, _ = _call(persistence=_ones(1.0))
    _, blink, _, _ = _call(persistence=_ones(0.0))
    assert (held["grasp_quality"] > blink["grasp_quality"]).all()
    # credit 합 = 1.0 (persistence 재도입 시 재분배 누락 재발 방지 — 08.22 의 0.8 사고)
    c = _Cfg()
    assert abs(c.grasp_envelope_credit + c.grasp_grip_credit + c.grasp_persist_credit - 1.0) < 1e-9


def test_progress_beats_standing_still():
    """reward-audit Check 1 — 08.22 재정의.

    구 기준은 "정체(approach+grasp) : 추가(lift+success) ≥ 1:3" 이었다. 그 전제는
    '파지는 통과점이고 목표는 드는 것'이었는데, 사용자 우선순위가 **①인벨롭 그립**으로
    바뀌면서 전제가 뒤집혔다 — 완벽한 감쌈은 '정체'가 아니라 **1단계 달성**이다.
    (구 기준을 유지하면 grasp 가중을 못 올려 우선순위를 보상에 실을 수 없다.)

    그래서 검사 대상을 비율에서 **단계 진행 유인**으로 바꾼다: 각 단계를 밟을 때마다
    총보상이 유의미하게 늘어야 하고, 마지막 단계가 가장 커야 한다.
    """
    def total(**kw):
        return float(_call(**kw)[0][0])
    # ① 감쌈 없이 접근만 → ② 감쌈 완성 → ③ 들기 → ④ 이송 성공
    s1 = total(envelope_frac=_ones(0.1), grip_frac=_ones(0.2),
               object_height_delta=_ones(0.0), object_to_goal=_ones(1.0))
    s2 = total(object_height_delta=_ones(0.0), object_to_goal=_ones(1.0))
    s3 = total(object_height_delta=_ones(0.15), object_to_goal=_ones(1.0))
    s4 = total(object_height_delta=_ones(0.15), object_to_goal=_ones(0.0))
    assert s2 > s1, f"감쌈을 완성해도 보상이 안 는다 ({s1:.2f} → {s2:.2f})"
    assert s3 > s2, f"감싸고 **들어도** 보상이 안 는다 — 정체 국소최적 ({s2:.2f} → {s3:.2f})"
    assert s4 > s3, f"이송 성공이 보상되지 않는다 ({s3:.2f} → {s4:.2f})"
    # 최종 목표(이송)가 가장 큰 증분이어야 한다
    assert (s4 - s3) > (s3 - s2), "이송 증분이 리프트 증분보다 작다"


def test_no_gate_on_approach():
    """접촉 전 유일한 gradient 라 게이트가 걸리면 탐색이 죽는다."""
    _, r, gate, _ = _call(group_a_force=torch.zeros(N, 1),
                          group_b_force=torch.zeros(N, 4))
    assert not gate.any()
    assert (r["approach"] > 0).all()


def test_action_rate_has_live_gradient_at_realistic_jitter():
    """★실측 지터에서 gradient 가 살아 있어야 한다.

    구 설정 `-0.005 * sum((Δa)²).clamp(max=1.0)` 은 실측 sum 12.0 에서 **12배 포화**해
    페널티가 상수였다 — 지터가 12→1 로 줄 때까지 gradient 가 0.
    (fab_test2 iter15: 팔 |Δa| 0.572 / 손 0.709)
    """
    a = torch.zeros(N, 26)
    lo = _call(actions=a, prev_actions=a)[1]["action_rate"]

    def jitter(scale):
        d = torch.full((N, 26), scale)
        return _call(actions=d, prev_actions=torch.zeros(N, 26))[1]["action_rate"]

    r_small, r_mid, r_big = jitter(0.2), jitter(0.5), jitter(0.9)
    assert (lo == 0).all(), "지터 0 이면 페널티도 0 이어야"
    # 단조 감소(더 큰 지터 = 더 큰 페널티)이고, 어느 구간도 평탄하지 않아야
    assert (r_big < r_mid).all() and (r_mid < r_small).all() and (r_small < lo).all()
    gap_lo = float((r_small - r_mid)[0])
    gap_hi = float((r_mid - r_big)[0])
    assert gap_lo > 1e-4 and gap_hi > 1e-4, f"평탄 구간 존재: {gap_lo}, {gap_hi}"


def test_action_rate_is_dimension_invariant():
    """★robot-agnostic: 같은 지터면 액션 차원이 달라도 같은 압력이어야 한다.

    `sum` 을 쓰면 26D=12.0 / 18D=8.3 / 7D=3.2 로 같은 weight 가 로봇마다
    다른 압력이 된다.
    """
    vals = []
    for dim in (7, 18, 26):
        d = torch.full((N, dim), 0.5)
        z = torch.zeros(N, dim)
        vals.append(float(_call(actions=d, prev_actions=z)[1]["action_rate"][0]))
    assert max(vals) - min(vals) < 1e-6, f"차원별 페널티가 다르다: {vals}"


def test_action_rate_magnitude_is_meaningful_but_not_dominant():
    """approach(최대 1.0) 대비 5~30% 여야 한다. 너무 작으면 죽고 크면 얼어붙는다."""
    d = torch.full((N, 26), 0.68)          # mean((Δa)²) ≈ 0.46 = 실측값
    pen = abs(float(_call(actions=d, prev_actions=torch.zeros(N, 26))[1]["action_rate"][0]))
    assert 0.05 <= pen <= 0.30, f"페널티 {pen:.3f} 가 범위 밖"


def test_stub_matches_real_cfg_defaults():
    """★테스트 스텁이 실제 cfg 에서 표류하면 테스트가 옛 값을 검증한다.

    실제로 그랬다: action_rate_weight 를 cfg 에서 -0.3 으로 바꿨는데 스텁은 -0.005 라
    "페널티가 의미 있는 크기인가" 테스트가 옛 값을 재고 있었다.
    """
    C = _cfg_module()
    real = C.GraspLiftFabricEnvCfg()
    stub = _Cfg()
    drift = []
    for k in dir(stub):
        if k.startswith("_"):
            continue
        if hasattr(real, k) and getattr(real, k) != getattr(stub, k):
            drift.append(f"{k}: 스텁 {getattr(stub, k)} vs cfg {getattr(real, k)}")
    assert not drift, "스텁이 실제 cfg 와 다르다: " + " / ".join(drift)


def test_total_is_nan_safe():
    total, _, _, _ = _call(object_to_goal=torch.full((N,), float("nan")))
    assert torch.isfinite(total).all()


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
        r'_cfg\(cfg, "([a-z_]+)"', (_TASK_DIR / "rewards.py").read_text()))
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    declared = set(re.findall(r"^    ([a-z_]+):", cfg_src, re.M))
    missing = sorted(reward_keys - declared)
    assert not missing, f"보상이 읽는데 cfg 에 선언되지 않음(하드코딩 기본값으로 동작): {missing}"


def test_no_dead_reward_cfg_fields():
    """cfg 에만 있고 아무도 안 읽는 보상 필드 = 죽은 설정(오해를 부른다)."""
    import re

    src = (_TASK_DIR / "rewards.py").read_text() + (
        _TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    declared = set(re.findall(r"^    ([a-z_]+(?:_weight|_credit|_tau|_std|_deg)):", cfg_src, re.M))
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
# 08.22 우선순위 재설계: ①인벨롭 그립 ②똑바로 ③이송
# =============================================================================
def test_envelope_completion_beats_lazy_lift():
    """★핵심 계약: '감쌈 완성'의 이득이 '감쌈 없이 들기'보다 커야 한다.

    구 설계는 반대였다 — envelope 0.50→1.00 이득 +0.59 vs 대충 들기 lift 0.79.
    그래서 정책이 2.7 지 파지로 수렴했다(우팔 ep950 실측).
    """
    lazy = _call(envelope_frac=_ones(0.2), grip_frac=_ones(0.4),
                 object_height_delta=_ones(0.15))[0][0]
    full = _call(envelope_frac=_ones(1.0), grip_frac=_ones(1.0),
                 object_height_delta=_ones(0.15))[0][0]
    assert full > lazy * 1.5, f"감쌈 완성 {full:.2f} 가 대충 들기 {lazy:.2f} 를 압도하지 못한다"


def test_upright_requires_lift():
    """★audit Check 2: 들지 않고 감싸기만 해도 자세 만점이면 hacking 경로가 된다."""
    on_table = _call(object_height_delta=_ones(0.0), object_tilt_deg=_ones(0.0))[1]["upright"][0]
    lifted = _call(object_height_delta=_ones(0.15), object_tilt_deg=_ones(0.0))[1]["upright"][0]
    assert float(on_table) == 0.0, "테이블 위 컵에서 upright 가 0 이 아니다"
    assert float(lifted) > 0.0


def test_lift_penalizes_overshoot():
    """★평지 해소: dz 0.10~∞ 가 전부 만점이라 정책이 0.27 까지 표류했다."""
    at_goal = _call(object_height_delta=_ones(0.15))[1]["lift"][0]
    over = _call(object_height_delta=_ones(0.30))[1]["lift"][0]
    assert over < at_goal * 0.5, f"과지남(0.30m) 이 goal(0.15m) 대비 충분히 깎이지 않는다"


def test_upright_gradient_alive_at_observed_tilt():
    """★실측 tilt 31.5° 에서 자세 개선 gradient 가 살아 있어야 한다.

    구 설계(upright_max_deg=30)는 30° 초과가 전부 up_mul 0.5 고정이라
    롤아웃의 66% 가 '자세를 고쳐도 보상이 안 늘어나는' 평지에 있었다.
    """
    worse = _call(object_tilt_deg=_ones(35.0), object_height_delta=_ones(0.15))[0][0]
    better = _call(object_tilt_deg=_ones(15.0), object_height_delta=_ones(0.15))[0][0]
    assert better > worse, "31° 대역에서 자세 개선이 보상으로 이어지지 않는다"
