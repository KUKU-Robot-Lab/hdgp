"""grasp_lift_fabric 태스크 계약 — Isaac 없이 검증 가능한 것만.

물리·IK 는 여기서 알 수 없다. probe 의 몫이다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/tests/ -q
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

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
def test_no_dead_reward_cfg_fields():
    """★cfg 전 필드가 어딘가에서 소비돼야 한다 — 접미사 정규식 스캔(구판)은 죽은
    필드 40+ 를 하나도 못 잡았다(08.26 인벤토리). 전 필드 × 전 소비자 스캔으로 강화.
    framework 소유 필드(IsaacLab 기반 클래스가 읽음)만 allowlist.
    """
    _FRAMEWORK = {"episode_length_s", "decimation", "action_space",
                  "observation_space", "state_space", "seed"}
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    consumers = cfg_src
    for f in ((_TASK_DIR / "grasp_lift_fabric_env.py"),
              (_TASK_DIR / "rewards_tip.py"),
              (_TASK_DIR.parent / "grasp_sensor" / "rewards_tip_cyl.py"),
              (_TASK_DIR.parent / "grasp_sensor" / "rewards.py"),
              (_TASK_DIR.parent.parent / "modules" / "adr.py")):
        consumers += f.read_text()
    fields = set(re.findall(r"^\s{4}([a-z_][a-z_0-9]*)\s*:\s*[a-zA-Z]", cfg_src, re.M))
    dead = []
    for k in sorted(fields - _FRAMEWORK):
        uses = (len(re.findall(rf"\b{k}\b", consumers))
                - len(re.findall(rf"^\s{{4}}{k}\s*:", cfg_src, re.M)))
        if uses == 0:
            dead.append(k)
    assert not dead, f"죽은 cfg 필드: {dead}"

# cfg 모듈은 isaaclab 을 import 하므로 Isaac 환경에서만 돈다.
# ★모듈 레벨 importorskip 금지 — 파일 전체가 조용히 skip 된다(실제 사고 이력).
def _cfg_module():
    return pytest.importorskip(
        "openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg",
        reason="Isaac(pxr) 없음 — cfg 파생 검사는 isaaclab 환경에서만 유효")


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
    body = src[src.index("def envelope_fraction_graded"):]
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


def test_tip_action_box_covers_reachable_poses():
    """액션 박스 계약 — **지시 가능한 자세가 실제로 존재해야 한다**.

    08.25 span_frac=0.3 은 FK 표본 32,768 중 5 손끝이 동시에 박스에 드는 자세가
    **0 개(0.000%)** 였다. 엄지 z[-15,32] 와 4 지 z[130,198] 이 98mm 갈라져 대향
    파지를 지시할 수 없었고, 그래서 좌·우 전 런에서 엄지 접촉력이 정확히 0.00N,
    대향 게이트 0, lift/tracking/success 영구 0 이었다 — 4 런을 그렇게 소모했다.

    ★부팅 검증이 **조용히 통과하지 않도록** 하한과 예외를 함께 못박는다. 상수만
      되돌려 놓으면 같은 함정이 그대로 재발한다.
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    assert float(C.tip_action_span_frac) >= 0.85, (
        f"span_frac={C.tip_action_span_frac} — 실측상 0.85 미만은 대향 자세가 1% 미만이다")
    assert 0.0 < float(C.tip_box_min_coverage) <= 0.5, "포함률 하한이 무의미하다"
    assert float(C.opposition_reach_target) >= 0.035, "대향 판정이 컵 지름보다 빡빡하다"

    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "액션 박스 포함률" in src, "포함률 진단 로그가 사라졌다"
    assert "tip_box_min_coverage" in src, "포함률 하한 검사가 사라졌다"
    # fail-loud 여야 한다 — 경고만 찍고 넘어가면 4 런을 또 태운다.
    i = src.index("tip_box_min_coverage")
    assert "raise RuntimeError" in src[i:i + 900], (
        "포함률 미달이 예외가 아니라 경고로만 처리된다 = 조용히 넘어간다")


def test_lr_schedule_is_adaptive_in_all_agent_configs():
    """LR 스케줄 계약 — 모든 agent yaml 에서 **adaptive** 여야 한다.

    `linear` 는 `max_epochs=20000` 기준이라 e7,000 에서 23% 만 감쌘다(2.55e-4 →
    1.96e-4). 그 사이 KL 은 전 구간의 98% 에서 임계(0.013)를 넘고 최근500 평균이
    **0.038 — 임계의 3배**였다. σ 가 좁아진 정책(entropy 22.4, σ≈0.72)이 큰 스텝을
    밟아 좋은 해에서 이탈했고, kuka5 좌·우가 모두 정점 후 퇴행했다
    (좌 11.60→5.3 · 우 4.40→0.4).

    ★자매 트랙이 **같은 증상으로 같은 수정**을 이미 했다(커밋 8db1449
      "lr_schedule linear → adaptive (KL 폭주 붕괴)"). 그 뒤 hier_test1 은 KL
      0.015~0.017 로 임계에 붙고 LR 이 8.9e-5 → 2.6e-5 로 3.4배 감쇠하며
      e14,416 을 안정적으로 돌았다. DexPour Table I 도 Adaptive 다.

    ★actor 와 central_value **둘 다** 검사한다 — critic 만 linear 로 남으면 같은
      결함이 value 쪽에 그대로 남는다.
    """
    import re
    cfg_dir = _TASK_DIR / "config" / "agents"
    yamls = sorted(cfg_dir.glob("rl_games_ppo*.yaml"))
    assert yamls, "agent yaml 을 못 찾았다"
    for y in yamls:
        src = y.read_text()
        found = re.findall(r"^\s*lr_schedule:\s*(\S+)", src, re.M)
        assert found, f"{y.name}: lr_schedule 이 없다 — rl_games 기본값에 맡기면 조용히 바뀐다"
        for v in found:
            assert v == "adaptive", f"{y.name}: lr_schedule={v} — linear 는 KL 폭주를 방치한다"


def test_stage_rewards_gates_are_cumulative_binary():
    """계층 게이트 계약 — λ⊇μ⊇ν⊇ρ **이진 누적 곱**(DexPour 식 3~6).

    상위 게이트가 0 이면 하위도 반드시 0 이어야 한다. 자매 실측에서 이 구조로
    goal_reached 0.335 에 도달했고, 우리 kuka6 은 두 팔이 정확히 한 칸씩 다른
    칸에서 끊겼다(우 μ 2.19<3 · 좌 ν 7mm<50mm).
    """
    torch = pytest.importorskip("torch")
    src = (_TASK_DIR.parent / "grasp_sensor" / "rewards_tip_cyl.py").read_text()   # 08.26 수식 단일 소스 = 자매 파일
    for pat in ("lam = ", "mu = lam *", "nu = mu *", "rho = nu *"):
        assert pat in src, f"누적 곱 사슬이 끊겼다: {pat!r}"
    # contact 는 게이트 **밖** — λ=1·μ=0 사각지대에서 보상이 0 이 되지 않게 한다.
    i = src.index("contact = touch_f")
    assert "mu" not in src[i:i + 40], "contact 가 게이트 안으로 들어갔다"


def test_stage_weights_are_monotonic():
    """단계 가중 단조 — approach < grasp < lift < transfer < stay.

    게이트가 이진이라 열린 칸의 지급 = 가중 × 진척이고, 가중이 단조면 실지급도
    단조다. 인자곱 구조는 매번 <1 이라 역전됐다(자매 실측: grasp 1.469 > lift 0.757).
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    seq = [C.stage_approach_weight, C.stage_grasp_weight, C.stage_lift_weight,
           C.stage_transfer_weight, C.stage_stay_weight]
    assert seq == sorted(seq) and len(set(seq)) == len(seq), f"단계 역전: {seq}"
    assert seq == [2.0, 3.0, 5.0, 7.0, 10.0], f"자매 값과 다름: {seq}"


def test_upright_gate_is_stricter_than_transfer_tolerance():
    """계층 역전 금지 — 직립(U_up)이 이송 관용(U_tol)보다 **엄격**해야 한다.

    느슨하면 stay 가 lift 보다 쉬워져 계층이 뒤집힌다(자매가 부팅 검사로 막은 것).
    사용자 규격: "이송 중 20° 내는 괜찮음. 목표 5cm 내로 오면 똑바로 세워 정지."
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    tol_lo, tol_hi = C.stage_tilt_tolerance_deg      # (30, 20) — 20° 까지 1.0
    up_lo, up_hi = C.stage_upright_gate_deg          # (15, 5)  — 5° 까지 1.0
    assert up_hi < tol_hi, f"직립 {up_hi}° 가 이송 관용 {tol_hi}° 보다 느슨하다"
    assert up_lo < tol_lo, f"직립 전이 시작 {up_lo}° 가 이송 {tol_lo}° 보다 늦다"


def test_success_has_speed_factor():
    """성공에 **속도 인자**가 있어야 한다.

    없으면 목표를 **스쳐 지나가도** 성공으로 센다 — 우리 `pass_pos` 가 0.00~0.31 을
    진동한 원인이고, 자매가 `s_v` 를 신설해 막은 구멍이다.
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    src = (_TASK_DIR.parent / "grasp_sensor" / "rewards_tip_cyl.py").read_text()
    assert "s_v = smoothstep(obj_speed" in src, "속도 인자가 없다"
    assert "s_v" in src[src.index("succ_soft ="):src.index("succ_soft =") + 120]
    lo, hi = C.stage_succ_speed_band
    assert hi < lo, f"속도 밴드가 내려가는 전이가 아니다: {(lo, hi)}"


def test_grasp_center_is_palm_anchored():
    """파지중심은 **palm 부착**이어야 한다 (08.26 사용자 결정).

    손끝 평균을 쓰면 **손을 오므리는 것만으로 d_gc 가 줄어**, 손바닥을 붙이지 않는
    핀치가 접근 보상을 다 먹는다. 목표가 인벨롭 그립이므로 파지중심은 손바닥에
    붙어 있어야 한다.

    ★d_gc 가 35~60mm 에서 포화하는 것은 자매도 같다(그쪽 실측 83mm). 그래서 λ 임계가
      120mm 다 — 포화값보다 **크게** 잡아야 게이트가 열린다.
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "_gc_palm = palm_pos + torch.einsum(" in src, "palm 부착 파지중심이 없다"
    assert "_grasp_center_local" in src
    # λ 임계는 palm 부착 포화값(자매 83mm · 우리 35~60mm)보다 커야 한다.
    assert float(C.stage_gate_approach_m) >= 0.09, (
        f"λ 임계 {C.stage_gate_approach_m}m 가 palm 부착 d_gc 포화값에 너무 가깝다")


def test_stage_constants_match_sibling():
    """게이트·품질 상수가 자매와 **같은 값**이어야 한다.

    다르면 자매 결과(goal_reached 0.335)를 근거로 쓸 수 없다. 우리 kuka6 실측이
    이 임계들의 경계에 정확히 놓여 재유도가 불필요하다는 것이 이식의 전제다.
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    assert (C.stage_gate_approach_m, C.stage_gate_contact_n,
            C.stage_gate_lift_m, C.stage_gate_transfer_m) == (0.12, 3.0, 0.05, 0.08)
    assert (C.stage_graspq_touch, C.stage_graspq_deep,
            C.stage_graspq_persist, C.stage_graspq_thumb_floor) == (0.25, 0.55, 0.20, 0.30)
    assert C.stage_lift_height_ref == 0.15 and C.stage_disp_limit == 0.06


def test_finger_freeze_releases_on_palm_normal_distance():
    """★★접근 중 손은 **편 채 고정**되고, palm_ee 법선거리로 풀린다(08.27 사양).

    "천천히 side-to-side 로 접근(핸드 고정) → palm 이 닿을 정도가 되면(palm_ee x
     거리) 고정을 풀고 액션으로 말리게 함."

    계약 4중:
      ①판정이 **부호 있는 법선거리** — 3D norm 은 "옆에 나란히 선 것"과 "정면에서
        다가온 것"을 못 가른다. μ(접촉 수)로 풀면 손이 고정돼 접촉이 없고 접촉이
        없으니 μ 가 안 열리는 **순환 의존**이 되므로 접촉 기반도 금지.
      ②고정 목표는 **홈 관절**(`_hand_home_free`) — 손 경로가 08.27 부터 관절
        직결이라 손끝 좌표로 고정할 수단이 없다.
      ③히스테리시스 > 0 — 경계에서 손이 떨리면 파지가 성립하지 못한다.
      ④리셋에서 래치 해제 — 안 하면 이전 에피소드의 "풀림"이 새 에피소드로 샌다.

    ★이 계약은 Isaac 게이트 밖에 둔다. 구판은 게이트 뒤라 **구현이 통째로 없는데도
      한 번도 실행되지 않아 "통과"로 보였다**(로컬·서버 모두 skip).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    # ★적용부 기준으로 찾는다 — 부팅 로그 블록도 같은 플래그를 읽어 첫 매치가
    #   그쪽으로 밀린다(실제로 그래서 계약이 헛짚었다).
    i = src.index("_rel_m = float(self.cfg.finger_release_dist_m)")
    blk = src[max(0, i - 700):i + 700]
    assert "finger_release_dist_m" in blk, "해제 임계를 안 읽는다"
    assert "_nrm).sum(-1)" in blk, "부호 있는 법선거리 판정이 아니다"
    assert "stage_gate_contact_n" not in blk, (
        "접촉 수로 손을 푼다 — 손이 고정돼 접촉이 없고 μ 도 안 열리는 순환이 된다")
    assert "self._hand_home_free)" in blk, "고정 목표가 홈 관절이 아니다"
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"finger_release_hysteresis_m:\s*float\s*=\s*([0-9.]+)", cfg_src)
    assert m and float(m.group(1)) > 0.0, "히스테리시스가 없다"
    assert "self._fingers_free[env_ids] = False" in src, "리셋에서 래치를 안 푼다"
    assert 'task/pose/palm/normal_dist_m' in src, "법선거리 지표가 로깅되지 않는다"


def test_action_cmd_markers_render_in_gui_and_video():
    """액션 cmd 마커는 **GUI 와 영상 녹화 둘 다**에서 보여야 한다(08.26 사용자 요청).

    구 계약은 GUI 전용이라 play --video(headless+카메라)에서 마커가 빠졌다 —
    사용자가 영상으로 거동을 확인하는데 정책이 뭘 지시하는지 안 보였다.
    렌더 대상이 전혀 없는 순수 headless 학습에서는 여전히 만들지 않는다
    (cameras_enabled carb 설정으로 판별). palm **지령** 마커(흰색)도 함께 —
    손끝만 보면 팔이 어디로 가라는 지시인지 모른다.
    """
    C = _cfg_module().GraspLiftFabricEnvCfg()
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("enable_tip_markers")
    blk = src[i - 200:i + 900]
    assert "has_gui() or _cams" in blk, "카메라 녹화 경로에서 마커가 안 켜진다"
    assert "cameras_enabled" in blk, "카메라 판별이 carb 설정이 아니다"
    assert '"palm_cmd"' in blk, "palm 지령 마커가 없다"
    j = src.index("def _update_tip_markers")
    assert "self.palm_cmd[0, :3]" in src[j:j + 1200], (
        "visualize 가 palm 지령을 안 그린다")
    assert float(C.tip_marker_radius) > 0.0


def test_envelope_gate_saturation_decoupled_from_success_threshold():
    """success 램프 분모(0.85) > 판정 임계(0.6) — 같으면 3지에서 4·5지 유인 소멸."""
    C = _cfg_module()
    cfg = C.GraspLiftFabricEnvCfg()
    assert cfg.envelope_gate_saturation > cfg.success_envelope_min
    src = (_TASK_DIR / "rewards_tip.py").read_text()
    assert "envelope_gate_saturation" in src, "rewards_tip 이 saturation 을 안 읽는다"


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


def test_fabric_velocity_is_fed_forward_to_pd():
    """★fabric 계획 **속도**를 PD 속도 목표로 내리는가 — 0 을 넣으면 안 된다.

    속도목표 0 : kp·err = kd·v + τ_마찰 → err ≈ (kd/kp)·v = 0.2·v [rad]  (kp=400,kd=80)
    속도목표 qd: kp·err = τ_마찰만      → err ≈ τ_f/kp
    즉 0 을 넣으면 PD 감쇠항이 움직임을 되밀어 **속도에 정비례하는** 추종오차가 된다.
    DEXTRAH 원본은 dof_vel_targets = clone(fabric_qd) 를 내리고 velocity_target_factor
    를 ADR 로 1.0 → 0.0 으로 줄여 간다 — 우리는 그 최종 단계에서 시작하고 있었다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    body = src[src.index("def _apply_action"):src.index("def ", src.index("def _apply_action") + 10)]
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


def test_action_command_rate_limits_match_user_spec():
    """★★지령 속도 상한은 **액션 인터페이스**가 진다(08.26 사용자 결정).

    "fabric 은 보상을 돕는 실행 계층이지, 지령 속도 제한을 fabric 이나 보상이
    해결할 몫이 아니다." — palm 0.1 m/step · 손끝(미세 조작) 0.01 m/step.
    구 계약(test_palm_slew_is_disabled_like_kuka)은 KUKA 원본 충실을 근거로
    slew=0 을 강제했는데, 그 논리는 지령 텔레포트를 fabric 필터에 맡기는 것이라
    설계 결정으로 뒤집혔다. 절대 매핑 위의 slew 라 목표 인플레 문제는 없다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    for k, want in (("palm_slew_pos", 0.10), ("palm_slew_rot_deg", 15.0)):
        m = re.search(rf"^\s*{k}\s*:\s*float\s*=\s*([0-9.]+)", src, re.M)
        assert m and float(m.group(1)) == want, (
            f"{k} = {m.group(1) if m else '없음'} — 사용자 지정 {want} 이어야 한다")
    # 손끝 slew 는 tip IK 폐기(풀 관절 전환)와 함께 제거 — 관절 절대 매핑은 지령이
    # 액션의 순수 함수라 별도 리미터가 없다(팔만 slew).
    assert "tip_slew" not in src, "tip slew 잔재가 cfg 에 남아 있다"


def test_actions_use_kuka_absolute_box_mapping():
    """★팔·손끝 모두 원본 compute_absolute_action 과 같은 절대 박스 매핑인가."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "def _kuka_absolute(" in src, "절대 박스 매핑 헬퍼가 없다"
    assert "0.5 * (hi - lo) * a + 0.5 * (hi + lo)" in src, "scale() 식이 원본과 다르다"
    body = src[src.index("def _pre_physics_step"):src.index("def _step_fabric")]
    assert body.count("_kuka_absolute(") == 1, "팔이 절대 박스 매핑이 아니다"
    # 손(풀 관절)은 대칭 관절 매핑: a=-1 완전 개방 · a=+1 완전 폐합.
    assert "0.5 * (self.actions[:, 6:] + 1.0)" in body, "손 관절 대칭 매핑이 아니다"


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


# 원본 KUKA ADR 그룹(보상 가중치 축 제외) — 전부 TaskADR 에 연결돼야 한다.
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


# =============================================================================
# 자세 항 — 08.26 hierarchical4 실측으로 드러난 두 결함의 재발 방지
# =============================================================================
# ★cfg 대신 **가벼운 대역**을 쓴다 — 실제 cfg 는 isaaclab 을 끌어와 Isaac 없는
#   환경에서 통째로 skip 된다. 자세 항은 계약의 핵심이라 어디서든 돌아야 한다.
#   값이 실제 cfg 와 같은지는 `test_stage_constants_match_sibling` 이 따로 지킨다.
_STAGE_CFG = SimpleNamespace(
    stage_align_floor=0.25, stage_approach_sharpness=8.0,
    stage_approach_tilt_margin_deg=8.0, stage_approach_tilt_penalty=0.08,
    stage_approach_weight=2.0, stage_approach_xy_margin=0.025,
    stage_approach_xy_penalty=8.0, stage_contact_weight=2.0,
    stage_disp_limit=0.06, stage_gate_approach_m=0.12, stage_gate_contact_n=3.0,
    stage_gate_lift_m=0.05, stage_gate_transfer_m=0.08,
    stage_graspq_deep=0.55, stage_graspq_persist=0.20,
    stage_graspq_thumb_floor=0.30, stage_graspq_touch=0.25,
    stage_grasp_weight=3.0, stage_lift_height_ref=0.15, stage_lift_weight=5.0,
    stage_orient_floor=0.15, stage_perp_exponent=2.0, stage_roll_exponent=4.0,
    stage_approach_z_band=(0.15, 0.05), stage_approach_z_frac=0.5,
    stage_stay_speed_ref=0.05, stage_stay_weight=10.0, stage_success_weight=6.0,
    stage_succ_goal_band_m=(0.09, 0.05), stage_succ_graspq_band=(0.35, 0.70),
    stage_succ_height_band=(0.04, 0.12), stage_succ_speed_band=(0.10, 0.03),
    stage_succ_tilt_band_deg=(18.0, 6.0), stage_thumb_force_ref=0.5,
    stage_tilt_tolerance_deg=(30.0, 20.0), stage_tracking_std=0.1,
    stage_transfer_weight=7.0, stage_upright_gate_deg=(15.0, 5.0),
    action_l2_weight=0.0, action_rate_l2_weight=0.0,
)


def _stage_inputs(**over):
    """`compute_stage_rewards` 최소 입력 한 벌. 자세 항만 보므로 나머지는 무해값."""
    N = 1
    z = torch.zeros(N, 3)
    base = dict(
        palm_pos=z.clone(),
        grasp_center_pos=z.clone(),
        object_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        goal_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        persist_frac=torch.zeros(N),
        tip_c=torch.zeros(N, 5, dtype=torch.bool),
        wrap_c=torch.zeros(N, 4, dtype=torch.bool),
        deep_c=torch.zeros(N, 4, dtype=torch.bool),
        oppose=torch.zeros(N, dtype=torch.bool),
        touch_c=torch.zeros(N, 5, dtype=torch.bool),
        thumb_force=torch.zeros(N),
        corridor_ok=torch.ones(N),
        syn_close_mean=torch.zeros(N),
        height_delta=torch.zeros(N),
        tilt_deg=torch.zeros(N),
        xy_disp=torch.zeros(N),
        palm_x=torch.tensor([[1.0, 0.0, 0.0]]),
        palm_y=torch.tensor([[0.0, 0.0, 1.0]]),
        ref_up=torch.tensor([[0.0, 0.0, 1.0]]),
        obj_up=torch.tensor([[0.0, 0.0, 1.0]]),
        obj_speed=torch.zeros(N),
        actions=torch.zeros(N, 21),
        prev_actions=torch.zeros(N, 21),
        cfg=_STAGE_CFG,
    )
    base.update(over)
    return base


def _stage_terms(**over):
    from openarm.agnostic.tasks.grasp_lift_fabric.rewards_stage import (
        compute_stage_rewards,
    )
    return compute_stage_rewards(**_stage_inputs(**over))[1]


def test_align_is_compatible_with_zero_grasp_distance():
    """★★align=1 과 d_gc=0 은 **동시에 달성 가능**해야 한다 — 양립성 계약.

    08.26 하루에 두 번 뒤집힌 항이다. 법선(palm_x) 방위 판은 파지중심 오프셋
    (법선에서 52.8°)과 기하 충돌해 d_gc 바닥을 121mm 로 만들었다 — 좌팔이 정확히
    거기서 정체했고, λ 가 열리는 순간 접촉이 0.55→0.00 붕괴했다(컵이 파지중심
    옆 90mm = 손가락 박스 밖). 포위중심 프로브: **오므려도** 중점 방위각 50~68°
    (50% 굴곡 49.8°) — 컵이 법선의 50° 옆에 앉는 것이 이 손의 기하다.

    그래서 "무엇을 재는가" 대신 **불변식**을 검사한다: 컵이 파지중심에 정확히
    앉은 배치에서 align 이 1 이어야 한다. 법선 판은 여기서 cos(52.8°)≈0.60 로
    떨어진다 — 어떤 재작성이 와도 이 함정은 다시 못 들어온다.
    """
    # ★배치는 반드시 **side-to-side 자세**로 짠다: palm_y=연직, palm_x/z=수평.
    #   원위 성분(92mm)을 연직에 두면(홈 자세) 오프셋의 수평 투영이 법선과 거의
    #   겹쳐(70,7) 법선 판도 0.995 로 통과해 버린다 — 처음에 그렇게 짰다가
    #   RED 가 0.004 차이로만 걸리는 걸 보고 고쳤다. 실제 파지 자세에서는 원위가
    #   **수평**이라 두 판이 52.8° 로 갈린다.
    palm = torch.zeros(1, 3)
    px = torch.tensor([[1.0, 0.0, 0.0]])          # 법선(수평)
    py = torch.tensor([[0.0, 0.0, 1.0]])          # 연직 — side-to-side
    pz = torch.tensor([[0.0, -1.0, 0.0]])         # 원위(수평) = x × y
    off = 0.070 * px + 0.007 * py + 0.092 * pz    # 실측 (70,7,92) 을 world 로
    for k in (1.0, 2.5):                    # 컵이 파지중심 위(=d_gc=0)와 그 연장선 밖
        t = _stage_terms(
            palm_pos=palm, grasp_center_pos=off, object_pos=k * off,
            goal_pos=k * off, palm_x=px, palm_y=py,
        )
        assert float(t["_align"]) > 0.999, (
            f"컵이 파지중심 연장선(k={k})에 있는데 align={float(t['_align']):.3f} — "
            "d_gc=0 과 양립하지 않는 기준을 보고 있다(법선 판이면 0.60 이 나온다)")
    # 반례: 컵이 오프셋 방위에서 **수평면 안에서** 90° 옆이면 align ≈ 0.
    _ox, _oy = float(off[0, 0]), float(off[0, 1])
    side = _stage_terms(
        palm_pos=palm, grasp_center_pos=off,
        object_pos=torch.tensor([[-_oy, _ox, float(off[0, 2])]]),
        palm_x=px, palm_y=py)
    assert abs(float(side["_align"])) < 0.05, (
        f"컵이 90° 옆인데 align={float(side['_align']):.3f}")


def test_pose_and_object_position_are_logged_with_reward():
    """자세·물체 위치를 보상과 **같은 스텝**에 남긴다(08.26 사용자 요청).

    보상 숫자만으로는 그 값이 어떤 자세에서 나왔는지 못 본다 — orient_q 0.95 가
    손날 자세였다. 특히 `normal_yaw_err_deg`(법선이 컵에서 몇 도 어긋났나)가 핵심이다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for tag in ("task/pose/", "palm/normal_yaw_err_deg", "palm/normal_pitch_deg",
                "palm/roll_deg", "palm/pitch_deg", "palm/yaw_deg",
                "obj/x", "palm/to_obj_mm"):
        assert tag in src, f"자세 로깅 누락: {tag}"


def test_stage_cfg_standin_matches_real_cfg():
    """대역 cfg(`_STAGE_CFG`) 가 실제 cfg 와 어긋나면 자세 계약이 **거짓 통과**한다.

    Isaac 없이 돌리려고 둔 대역이라, 실제 cfg 가 바뀌면 조용히 낡는다. 소스에서
    직접 읽어 대조한다(실제로 밴드 두 개를 서로 바꿔 적어 이 검사가 잡았다).
    """
    import ast
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()

    def _norm(x):
        return tuple(float(i) for i in x) if isinstance(x, (list, tuple)) else float(x)

    bad = []
    for k, v in vars(_STAGE_CFG).items():
        if not k.startswith("stage_"):
            continue
        m = re.search(rf"^\s*{k}\s*:[^=\n]*=\s*(\(.*?\)|[^\s#]+)", src, re.M)
        if m is None:
            bad.append((k, "cfg 에 없음"))
            continue
        real = ast.literal_eval(m.group(1).rstrip(","))
        if _norm(real) != _norm(v):
            bad.append((k, f"대역 {_norm(v)} vs cfg {_norm(real)}"))
    assert not bad, f"대역 cfg 가 낡았다: {bad}"


def test_zx_tilt_is_logged_directly_not_derived():
    """ZX 기울기는 **직접** 로깅해야 한다 — 평균 orient_q 에서 역산하면 안 된다.

    orient_q 는 비선형 곱이라 그 평균에서 되돌린 각도는 per-env 평균이 아니다
    (Jensen). 꼬리를 보려면 p95 도 필요하다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "task/pose/palm/zx_tilt_deg" in src, "ZX 기울기 로깅 없음"
    assert "zx_tilt_p95_deg" in src, "ZX 기울기 p95 없음 — 평균만으로는 꼬리가 숨는다"


def test_reward_formula_is_single_source():
    """★★수식은 자매 파일 **하나**다(08.26 사용자 확정: "리워드 이름부터 수식 모두
    동일, 로봇 대상만 다름"). 복제는 이름이 같아도 드리프트를 못 막는다 — 실제로
    같은 날 자매가 진화(코리더 래치)한 것을 복제본은 모르고 지나쳤다. import 가 막는다.
    """
    from openarm.agnostic.tasks.grasp_lift_fabric import rewards_stage as rs
    from openarm.agnostic.tasks.grasp_sensor import rewards_tip_cyl as sib
    assert rs.compute_stage_rewards is sib.compute_tip_cyl_rewards, (
        "compute_stage_rewards 가 자매 함수의 재수출이 아니다 — 수식이 갈라졌다")
    src = (_TASK_DIR / "rewards_stage.py").read_text()
    assert "d_gc" not in src.replace("test", ""), (
        "rewards_stage.py 에 수식 본문이 남아 있다 — 재수출만 있어야 한다")


def test_mirror_sign_is_input_adaptation_not_formula_change():
    """★roll_q 미러 사망(h4 좌팔: cos(palm_y,up)=−1 → orient_q≡floor)의 방지책은
    **수식이 아니라 입력**이다 — env 가 부팅 실측 부호(_palm_y_sign)를 곱해 넘긴다.
    수식을 abs 로 바꾸면 자매와 갈라진다(실제로 하루 그랬다가 되돌렸다).
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "_palm_y_sign * _R_tcp[:, :, 1]" in env_src, "palm_y 미러 부호 미적용"
    assert "_palm_y_sign" in env_src and "sign(_R0" in env_src.replace(
        "torch.sign(_R0", "sign(_R0"), "부호가 부팅 실측이 아니다"


def test_corridor_latch_and_persist_are_wired_and_reset():
    """자매 08.26 승인분(코리더 래치·deep persist)이 배선되고 **리셋에서 지워지는가**.
    래치가 에피소드를 넘으면 이전 위반이 새 에피소드의 ν 를 몰수한다.
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for pat in ("self._corridor_latch |=", "corridor_ok=(~self._corridor_latch)",
                "self._persist_buf = torch.where", "persist_frac=_persist"):
        assert pat in env_src, f"배선 누락: {pat}"
    i_reset = env_src.index("self._persist_buf[env_ids] = 0")
    tail = env_src[i_reset:i_reset + 600]
    assert "self._corridor_latch[env_ids] = False" in tail, "래치 리셋 누락"
    assert "self._persist_buf[env_ids] = 0" in tail, "persist 리셋 누락"


def test_stage_new_constants_match_sibling_source():
    """신규 동기화 상수(코리더·persist·접촉임계·gc오버라이드)가 자매 cfg 와 같은 값."""
    import ast
    sib = (_TASK_DIR.parent / "grasp_sensor" / "grasp_sensor_env_cfg.py").read_text()
    ours = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    for k in ("stage_contact_threshold", "stage_contact_persistence_steps",
              "stage_corridor_xy_m", "stage_corridor_tilt_deg",
              "stage_gc_local_override", "stage_stay_hold_steps",
              "stage_approach_z_band", "stage_approach_z_frac",
              "stage_stay_pos_tol_m", "stage_stay_tilt_deg",
              "stage_success_envelope_min"):
        vs = []
        m_sib = re.search(rf"^\s*{k}\s*:[^=\n]*=\s*(\(.*?\)|[^\s#]+)", sib, re.M)
        if m_sib is None:
            # ★자매 세션이 그 필드를 제거/개편 중일 수 있다(불가침 — 실제로 08.26
            #   stage_gc_* 가 자매 WIP 로 사라졌다). 자매에 없는 필드는 대조 생략 —
            #   우리 쪽 존재만 확인한다.
            assert re.search(rf"^\s*{k}\s*:", ours, re.M), f"우리 cfg 에 {k} 없음"
            continue
        m_our = re.search(rf"^\s*{k}\s*:[^=\n]*=\s*(\(.*?\)|[^\s#]+)", ours, re.M)
        assert m_our, f"우리 cfg 에 {k} 없음"
        vs = [ast.literal_eval(m_sib.group(1).rstrip(",")),
              ast.literal_eval(m_our.group(1).rstrip(","))]
        assert vs[0] == vs[1], f"{k}: 자매 {vs[0]} vs 우리 {vs[1]}"


def test_stage_hit_semantics_match_sibling():
    """task/stage/* 는 자매처럼 **에피소드 누적 hit** 를 리셋 때 기록해야 한다.

    순간 게이트 평균과 의미가 다르다(자매 hier_test2 의 stage/approach 1.0 은 hit).
    같은 태그가 두 트랙에서 다른 것을 재면 비교 분석 전체가 오독된다 — 실제로
    h2 의 λ 0.93(순간)과 자매의 1.0(hit)을 같은 자리에서 비교하고 있었다.
    ⑤ 정지는 hold_steps **연속 유지**여야 한다(스침을 정지로 세지 않는다).
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "self._stage_hit |= torch.stack" in env_src, "hit 누적 없음"
    i_reset = env_src.index("self._corridor_latch[env_ids] = False")
    tail = env_src[i_reset:i_reset + 900]
    assert 'task/stage/' in tail and "_stage_hit[env_ids]" in tail, "리셋 기록 없음"
    assert "self._stay_run >= int(self.cfg.stage_stay_hold_steps)" in env_src
    assert "task/gate_now/" in env_src, "순간 게이트 로깅(이름 분리)이 없다"


def test_adr_increments_on_lift_success_not_strict_goal():
    """ADR 승급은 **리프트 성공**으로 — 엄격 goal 로 걸면 성공률 0 에서 난이도가
    영영 안 오르고(자매 실측 difficulty 0.0000 고착), 코리더도 느슨한 시작값에
    묶인다. 엄격 판정은 task/goal_reached 로 계속 로깅한다."""
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "maybe_increment(self._lift_success_now" in env_src, (
        "ADR 이 엄격 goal_reached 로 승급한다")


def test_approach_kernel_is_z_first():
    """★★Z-우선 접근(08.26 사용자 지시) — "Z 를 먼저 맞추지 못하면 side-to-side
    접근이 안 된다." 구 3D 커널은 위 27cm 와 옆 27cm 를 같게 쳐서, 머리 위로 들어간
    정책이 내려가려면 수평으로 멀어져야만 하는 **로컬 최소**를 만들었다(h1 좌팔이
    컵 직상방 수평 24mm·Δz 165mm 에서 400 에폭 정체 — XY 먼저 뽑은 시드 운).

    계약: 같은 3D 거리에서 ①옆(같은 높이) ≫ ②머리 위 여야 하고, 머리 위에서는
    수평 커널이 닫혀(z_ok≈0) 접근 보상이 수직 성분에서만 나와야 한다.
    """
    def _appr(dz, dxy):
        # 파지중심을 원점에 두고 컵을 (dxy, 0, dz) 에 — 접근 항만 읽는다.
        t = _stage_terms(
            grasp_center_pos=torch.zeros(1, 3),
            object_pos=torch.tensor([[dxy, 0.0, dz]]),
            goal_pos=torch.tensor([[dxy, 0.0, dz]]),
            palm_pos=torch.tensor([[-0.05, 0.0, 0.0]]),
        )
        return float(t["approach"]), float(t["_z_ok"])
    side, z_side = _appr(0.01, 0.17)     # 같은 높이, 옆 17cm
    over, z_over = _appr(0.17, 0.01)     # 머리 위 17cm, 수평 1cm
    assert z_side > 0.95, f"높이 맞춤이 z_ok 를 안 연다: {z_side:.3f}"
    assert z_over < 0.05, f"머리 위인데 z_ok 가 열려 있다: {z_over:.3f}"
    assert side > 2.0 * over, (
        f"Z-우선 비대칭이 없다: 옆 {side:.3f} vs 머리위 {over:.3f} — "
        "구 3D 커널이면 둘이 같다(호버 로컬최소 재발)")


def test_close_bridge_is_wired_with_real_closure_for_tip_ik():
    """close_bridge(자매 05b6a3f)가 **실제 폐쇄도**로 배선됐는가.

    자매의 비-synergy 경로는 0 을 넘겨 항이 죽는다 — 우리가 0 을 넘기면 가중 0.5 를
    줘도 게이트 개방 후 gradient 공백(실측 P(n지≥3) 0.6%)이 그대로다. tip-IK 는
    fabric 손 관절 폐쇄도(홈→닫힘한계 정규화)로 같은 의미를 만든다.
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    # v2(자매 3ac85a9): 엄지/4지 **분리** 폐쇄도 — min 합성이 뒤처진 그룹에
    # gradient 를 주려면 두 입력이 실제 관절 폐쇄도여야 한다.
    for key in ("syn_close_thumb=(", "syn_close_fingers=("):
        assert key in env_src, f"{key} 미배선"
        i = env_src.index(key)
        blk = env_src[i:i + 400]
        assert "_fab_hand_home" in blk and "_close_den" in blk, (
            f"{key} 가 상수 0 이다 — close_bridge 가 죽는다")
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"stage_close_bridge_weight\s*:\s*float\s*=\s*([0-9.]+)", cfg_src)
    # 0.5→0.25 (08.26 seed-robust): 허공 오므림 대비 접촉의 상대가치 4배 교정.
    assert m and float(m.group(1)) == 0.25, "close_bridge 가중이 0.25 가 아니다"


def test_all_three_bridges_are_active_and_correctly_shaped():
    """★브리지 3종(08.26 사용자 지시) — 리미터가 "우연한 탐색"을 없앤 자리마다
    gradient 다리가 있어야 한다: 접근 뒤(close v2 min(엄지,4지)) · 손가락별
    (tip_bridge — 폐쇄도 스칼라로는 어느 손가락을 어디로 보낼지 못 가른다) ·
    파지 뒤 리프트 첫 mm(lift_bridge, 자매 실측 h 2mm 정체).
    """
    import ast as _ast
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    for k, want in (("stage_close_bridge_weight", 0.25),
                    ("stage_lift_bridge_weight", 1.0),
                    ("stage_tip_bridge_weight", 0.5)):
        m = re.search(rf"{k}\s*:\s*float\s*=\s*([0-9.]+)", cfg_src)
        assert m and float(m.group(1)) == want, f"{k} != {want}"
    # 수식(공유): tip_bridge 는 λ 게이트 + per-finger 커널 mean
    sib = (_TASK_DIR.parent / "grasp_sensor" / "rewards_tip_cyl.py").read_text()
    i = sib.index('"tip_bridge"')
    blk = sib[i:i + 500]
    # 08.26 지령 기준으로 전환 — 지령 거리 우선, 실-손끝은 하위호환으로만 남는다.
    assert "lam" in blk and "tip_cmd_surf_dist" in blk and "mean(dim=-1)" in blk
    # env: v2 분리 폐쇄도 + 지령-표면 거리 배선(상수 0 이면 항이 죽는다)
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for pat in ("syn_close_thumb=(", "syn_close_fingers=(",
                "tip_cmd_surf_dist=self._tip_cmd_surface_dist",
                "_close_thumb_m", "_close_fingers_m"):
        assert pat in env_src, f"배선 누락: {pat}"
    # lift_bridge 는 μ 게이트 — λ 로 걸면 잡기 전에 낚아채기를 유도한다
    j = sib.index('"lift_bridge"')
    assert "mu" in sib[j:j + 260], "lift_bridge 가 μ 게이트가 아니다"


def test_spawn_sits_inside_measured_side_grasp_region():
    """★스폰은 **제약 IK 실측 밴드에서 역산**한 허용창 안이어야 한다(08.26).

    파지 시 palm = 컵 − 오프셋(side-to-side world 분해: 원위 64mm→x, 법선 57mm→
    바깥 |y|). 제약 IK 실측 palm 밴드: x 0.06~0.26 · |y| 0.06~0.34 (컵 높이,
    법선방위 ±y±30° — 사용자 명세). ADR 만렙 반경 전 범위에서 palm 이 밴드 안:
        cup_x − 0.064 ± r ∈ [0.06, 0.26]  ∧  cup_|y| + 0.057 ± r ∈ [0.06, 0.34]
    구 (0.24,0.26) 은 |y| 만렙에서 0.377 > 0.34 로 이탈 — 동역학 지도(철회됨)의
    산물이었다. 이 검산 자체가 수치 결정의 근거다(감으로 정하지 않는다).
    """
    import ast as _ast
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"object_spawn_center_override[^=]*=\s*(\([^)]*\))", src)
    assert m, "스폰 오버라이드가 없다"
    cx, cy = _ast.literal_eval(m.group(1))
    r = float(re.search(r"spawn_range_final:\s*float\s*=\s*([0-9.]+)", src).group(1))
    OFF_X, OFF_Y = 0.064, 0.057
    BAND_X, BAND_Y = (0.06, 0.26), (0.06, 0.34)
    px_lo, px_hi = cx - OFF_X - r, cx - OFF_X + r
    py_lo, py_hi = abs(cy) + OFF_Y - r, abs(cy) + OFF_Y + r
    assert BAND_X[0] - 1e-9 <= px_lo and px_hi <= BAND_X[1] + 1e-9, (
        f"palm_x 창 [{px_lo:.3f},{px_hi:.3f}] 가 밴드 {BAND_X} 밖")
    assert BAND_Y[0] - 1e-9 <= py_lo and py_hi <= BAND_Y[1] + 1e-9, (
        f"palm_|y| 창 [{py_lo:.3f},{py_hi:.3f}] 가 밴드 {BAND_Y} 밖")
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = env_src.index("spawn[:, 0] = _scx")
    blk = env_src[max(0, i - 900):i]
    assert "object_spawn_center_override" in blk, "스폰이 오버라이드를 안 읽는다"
    assert "abs(float(_ovr_sp[1]))" in blk, (
        "y 부호가 미러 자동이 아니다 — 좌팔 스폰이 반대쪽에 떨어진다")


def test_tip_bridge_rewards_command_not_actual_tip():
    """★★tip_bridge 는 **지령(cmd)** 을 평가해야 한다(08.26 사용자 지시).

    실측(probe_tip_cmd_placement, h4 e800): 지령↔실제 간극이 손가락별 18~79mm.
    실 손끝으로 보상하면 그 간극만큼 정책 액션과 보상의 인과가 끊긴다 —
    900 에폭 동안 4지 접촉이 정확히 0.00 이었던 이유다. 지령을 평가하면
    정책이 낸 값이 곧 평가 대상이 된다.
    목표는 컵 **표면**(측면 띠) — 지령 실측이 표면 65mm 를 두고 index 173mm,
    pinky h−94mm 로 흩어져 있었다.
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "tip_cmd_surf_dist=self._tip_cmd_surface_dist" in env_src, (
        "tip_bridge 가 지령 기준이 아니다")
    assert "tip_obj_dist=(" not in env_src, "구 실-손끝 기준이 아직 배선돼 있다"
    i = env_src.index("def _tip_cmd_surface_dist")
    blk = env_src[i:i + 1200]
    for pat in ("self._tip_cmd_local", "root_quat_w", "radial", "relu(h.abs()"):
        assert pat in blk, f"표면 거리 계산 누락: {pat}"
    # 형상 가정은 env 에만 — 공유 수식은 거리만 받는다
    sib = (_TASK_DIR.parent / "grasp_sensor" / "rewards_tip_cyl.py").read_text()
    j = sib.index('"tip_bridge"')
    assert "radius" not in sib[j:j + 400], "공유 수식에 형상 상수가 들어갔다"


def test_palm_box_z_override_only_relaxes():
    """palm 박스 z바닥 오버라이드는 **완화 전용**이어야 한다.

    프로필 박스(0.34)는 자매 공유라 트랙 전용으로만 낮춘다. 실측: palm_ee z
    최소가 정확히 0.340 = 박스 바닥에 붙어 dz 가 61mm(=0.340−0.278)에서
    안 줄었다. 조이는 방향이면 프로필을 조용히 덮어써 도달역을 잃는다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"palm_box_z_min_override[^=]*=\s*([0-9.]+)", src)
    assert m, "오버라이드가 없다"
    z = float(m.group(1))
    assert z < 0.34, f"완화가 아니다: {z}"
    assert z > 0.20, f"테이블 상면(0.20) 아래다: {z}"
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = env_src.index("palm_box_z_min_override")
    assert "보다" in env_src[i:i + 500] and "RuntimeError" in env_src[i:i + 500], (
        "조이는 방향을 fail-loud 로 막지 않는다")


def test_hand_control_is_full_joint_with_frozen_override():
    """★★손 제어 = **풀 관절**(fabric direct) + 고정 관절 오버라이드(08.26 사용자).

    tip IK 폐기 근거(probe_tip_cmd_placement, h5 e600): 지령↔실제 간극 최대 111mm
    (middle 지령 r 42mm vs 실제 150mm) — 5지 15D 독립 지시는 조합이 기구학적으로
    성립하지 않는다. 저장소가 같은 결론에 도달한 이력이 있다(fabrics-fingertip-control).
    고정 규칙: **_1 전부 + 소지 _2** (+ thumb_2 는 대향 자세용으로 기존 유지).
    프로필 상수는 자매 공유 → 트랙 전용 오버라이드로만 덮는다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    assert re.search(r'hand_control:\s*str\s*=\s*"fabric"', src), "풀 관절이 아니다"
    assert "use_tip_fabric" not in src, "tip IK cfg 잔재"
    m = re.search(r"frozen_hand_joints_override[^=]*=\s*\((.*?)\n    \)", src, re.S)
    assert m, "고정 오버라이드가 없다"
    names = re.findall(r'"\{side\}_hj_([a-z]+)_(\d)"', m.group(1))
    got = {(f, j) for f, j in names}
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        assert (f, "1") in got, f"{f}_1 이 고정 목록에 없다"
    assert ("pinky", "2") in got, "소지 _2 가 고정 목록에 없다(사용자 지시)"
    assert ("thumb", "2") in got, "thumb_2(대향 자세) 고정이 풀렸다"
    # env: 오버라이드를 실제로 읽고 {side} 를 치환하는가
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = env_src.index("frozen_hand_joints_override")
    assert 'replace("{side}"' in env_src[i:i + 700], "{side} 치환이 없다"


def test_tip_bridge_command_source_exists_in_full_joint_mode():
    """풀 관절 모드에서도 tip_bridge 의 **지령 손끝**이 정의돼야 한다.

    tip 모드의 `_tip_cmd_local` 은 손끝 지령 그 자체였다. 풀 관절 모드에는 그 값이
    없으므로 **지령 관절의 FK** 로 만든다 — 없으면 tip_bridge 가 죽거나(0) 낡은
    값을 재사용해 조용히 틀린다.
    """
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = env_src.index("self._fabric_hand_cmd = _full[:, self._fab_from_hand]")
    blk = env_src[i:i + 2200]
    assert "_fingertip_taskmap" in blk and "_tip_cmd_local" in blk, (
        "풀 관절 모드에 지령 손끝 FK 가 없다")
    # obs 에 손끝 지령 없음(풀 관절 지령은 액션의 순수 함수 — 히든 상태 없음)
    i2 = env_src.index("parts = [joint_pos")
    assert "_tip_cmd" not in env_src[i2:i2 + 400], "obs 에 tip 지령 잔재"


# =============================================================================
# 08.26 재설계 계약 — sim2real obs · seed-robust · 다물체 표면 · 커리큘럼
# =============================================================================
def test_policy_obs_has_no_sim_only_terms():
    """★policy obs 에 sim 전용 값 금지 — 전부 실기에서 취득 가능해야 한다.

    금지 목록(배포 선례 grasp_v1/grasp_sensor): 물체 회전·6D 속도·scale 참값·
    접촉력 스칼라(마디)·fabric qd/qdd 원값. 전부 critic 전용.
    tip F/T 는 **tip-local** 만 허용 — world 프레임은 배포 변환 오차가 조용히
    obs 를 죽인다("손 obs zeros" 동형 사고).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("parts = [joint_pos")
    j = src.index("obs = torch.cat(parts", i)
    parts = src[i:j]
    for banned in ("_object_scale", "object_rot", "root_lin_vel", "root_ang_vel",
                   "contact.clamp", "self.fabric_qd", "self.fabric_qdd"):
        assert banned not in parts, f"policy obs 에 sim 전용 항: {banned}"
    for required in ("tip_ft", "hand_err"):
        assert required in parts, f"sim2real obs 누락: {required}"
    # 정규화 상수는 cfg(실기 공유 계약)에서 온다 — 하드코딩 금지.
    blk = src[src.index("tip_ft = torch.einsum"):i]
    assert "contact_force_max" in blk and "joint_pos_err_max" in blk


def test_contact_marginal_gain_beats_airborne_closing():
    """★seed-robust 핵심 부등식 — 손가락 1개 추가의 한계 이득 ≥ close_bridge 상한.

    h6 실측(좌우=2-seed): 우팔은 허공 오므림으로 close_bridge 를 벌며 접촉을
    회피했고(1300ep 배회), 좌팔만 우연히 접촉 복리에 진입했다. 이 부등식이
    깨지면 "안 닿고 오므리기"가 합리적 전략이 된다 — seed 운으로 갈리는 지형.
    """
    C = _STAGE_CFG
    marginal_per_finger = C.stage_contact_weight / 5.0
    assert marginal_per_finger >= 0.25 - 1e-9, (
        f"손가락 1개 한계 이득 {marginal_per_finger:.2f} < close_bridge 상한 0.25")


def test_grasp_surface_is_per_object_not_scalar():
    """★다물체 표면 — tip_bridge 목표면은 물체별 실측(fail-loud)이어야 한다.

    스칼라 하드코딩은 cup 스케일 0.85~1.30 에서 최대 ±30% 어긋났다. ObjectSpec 이
    None 이면 부팅 사망 — 조용히 틀린 목표면 금지.
    """
    from openarm.agnostic.modules import object_bank as ob
    bank = ob.CUP_FAMILY
    for sp in bank.specs:
        assert sp.grasp_radius_m > 0 and sp.grasp_halfheight_m > 0
    import pytest as _pt
    with _pt.raises(RuntimeError):
        _ = ob.ObjectSpec(id="x", usd_path="x").grasp_radius_m
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = env_src.index("def _tip_cmd_surface_dist")
    blk = env_src[i:i + 1400]
    assert "self._grasp_radius[:, None]" in blk, "표면 반경이 per-object 텐서가 아니다"
    cfg_src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    assert "stage_tip_target_radius_m" not in cfg_src, "스칼라 표면 상수 잔재"
    assert re.search(r'object_bank:\s*str\s*=\s*"cup_family"', cfg_src), (
        "기본 뱅크가 cup_family 가 아니다(다양한 컵 목표)")


def test_no_reverse_curriculum_cup_spawns_on_table():
    """★08.27 역순 커리큘럼 폐기 — 컵은 늘 테이블 스폰 위치에 나온다(사용자 사양
    "천천히 컵으로 side-to-side 접근"). 팔 IK 텔레포트도, 컵 근접 스폰도 없다.

    이력: ①팔을 IK 로 컵 옆에 텔레포트 → 단일 해가 나쁜 시드를 뽑은 팔을 영구
    고착시켰다(h7 우팔 2172ep 동결·ADR 승급 0). ②컵을 손 앞에 놓는 방식으로
    바꿨으나 사양이 "멀리서 천천히 접근"이라 이번 판에서 폐기.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    for banned in ("_solve_pregrasp_ik", "_cup_near_xy", "_home_grasp_center_xy",
                   "reset_near"):
        assert banned not in src, f"역순 커리큘럼 잔재: {banned}"
    j = src.index("spawn[:, 0] = _scx")
    assert "+ offs[:, 0]" in src[j:j + 120], "스폰 노이즈가 빠졌다"


def test_hand_is_outside_fabric():
    """★★손은 fabric 밖이다(08.27 사용자 지시 · 자매 grasp_sensor 배선).

    근거는 h7 실측이다: |fabric_q_hand − 정책 지령| 이 우 0.956rad(55°)·좌 0.645
    인데 |실측 − fabric_q| 는 0.12~0.25rad 였다 — PD 는 따라가고 fabric 이 지령을
    깎았다. 계약 3중: ①PD 목표가 fabric 산출물이 아니라 정책 목표 ②속도 목표는
    지령 램프 도함수 ③fabric 손 **상태**는 매 스텝 지령으로 동기화(안 하면 fabric 이
    다른 손으로 충돌 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다).
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("def _apply_action")
    blk = src[i:i + 2000]
    assert "self.hand_targets, joint_ids=self.hand_ids" in blk, (
        "손 PD 목표가 정책 목표가 아니다")
    assert "fabric_qd[:, n_arm:]" not in blk, "손 속도 목표가 아직 fabric 산출물이다"
    assert "hand_velocity_ff_scale" in blk and "_hand_vel" in blk, (
        "손 속도 피드포워드(지령 램프 도함수) 누락 — 감쇠항이 닫는 동작을 되민다")
    assert "self.fabric_q[:, self.profile.num_arm_joints:] = self._fabric_hand_cmd" in src, (
        "fabric 손 상태 동기화 누락 — 팔이 없는 자기충돌을 피해 밀린다")


def test_pinky_home_is_straight():
    """★pinky_1 홈 = 0(08.27 사용자 지시). 프로필은 60° 로 고정하는데 그 근거
    ("q1=60° 라야 pinky_2 가 굴곡축")는 **pinky_2 도 고정된 지금** 성립하지 않는다.
    남는 효과는 소지가 영구히 벌어져 가짜 접촉을 만드는 것뿐이었다
    (h7 우팔 pinky touch 0.543·wrap 0.290 = 우팔 접촉의 절반).
    ★robots.py 는 자매 공유라 건드리지 않고 트랙 cfg 로만 덮는다."""
    cfg = _cfg_module().GraspLiftFabricEnvCfg()
    ovr = getattr(cfg, "hand_home_override", None)
    assert ovr, "hand_home_override 없음"
    names = {n for n, _ in ovr}
    assert "{side}_hj_pinky_1" in names, "pinky_1 홈 오버라이드 없음"
    val = dict(ovr)["{side}_hj_pinky_1"]
    assert float(val) == 0.0, f"pinky_1 홈이 0 이 아니다({val})"
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("hand_home_override")
    assert src.index("self._setup_fabrics()") > i, (
        "홈 오버라이드가 _setup_fabrics 뒤다 — fabric cspace rest 가 옛 홈을 쥔다")


def test_markers_are_called_in_fabric_path():
    """액션 cmd 마커가 **fabric(현행) 경로에서 호출**되는가 — tip 분기 전용이라
    fabric 모드에서 한 번도 안 그려지던 실버그(인벤토리 발견)의 재발 방지."""
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("self._fabric_hand_cmd = _full[:, self._fab_from_hand]")
    blk = src[i:i + 2600]
    assert "_update_tip_markers" in blk, "마커 호출이 fabric 경로에 없다"


# =============================================================================
# 08.27 배선 결함 3건 수리 계약 — h7/h8 "손가락이 난리" 의 직접 원인이었다.
# =============================================================================
def test_hand_action_maps_home_to_open_not_joint_limit():
    """★★`a=−1` 은 **홈(펴짐)**, `a=+1` 은 **굴곡 한계**여야 한다.

    구 매핑은 `a∈[−1,1] → [관절 lo, hi]` 선형이었는데 `_3`/`_4` 한계가 좌우 모두
    대칭 ±90° 라 홈(0)이 한계 **중앙**이다 — 즉 액션 범위의 **절반이 손등 쪽
    역굴곡**을 지시했다(사용자 사양 "반대로 회전하면 안 됨" 정면 위반).
    이 매핑에서는 역굴곡이 액션 공간 밖이라 클램프·벌점이 필요 없다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    i = src.index("u_hand = 0.5 * (self.actions[:, 6:] + 1.0)")
    blk = src[i:i + 400]
    assert "self._hand_home_free" in blk and "self._flex_limit" in blk, (
        "손 액션이 홈→굴곡한계 매핑이 아니다")
    assert "self._hand_lo + u_hand" not in src, (
        "구 [lo,hi] 매핑이 남아 있다 — 액션 절반이 역굴곡을 지시한다")


def test_flexion_sign_is_measured_at_boot_and_fails_loud():
    """★굴곡 부호는 **부팅 FK 실측**이어야 한다(하드코딩·좌우 상수 금지).

    엄지 `_3`/`_4` 는 우 `+q`·좌 `−q` 가 굴곡인데(URDF origin rpy 가 좌우 뒤집힘,
    axis 는 둘 다 (0,0,1) 이라 **한계에는 안 드러난다**) 액션 매핑에 미러가 없어
    좌손 엄지는 `a=+1` 이 완전 개방이었다. 저장소 관례(`_palm_y_sign`)와 같이
    부팅에서 재고, 판별이 모호하면 fail-loud 한다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert "def _measure_flex_signs" in src, "굴곡 부호 실측 메서드 없음"
    blk = src[src.index("def _measure_flex_signs"):][:4200]
    assert "_fingertip_taskmap" in blk, "FK 실측이 아니다"
    assert "contact_group_a" in blk and "contact_group_b" in blk, (
        "대향 그룹 기준이 아니다 — palm +x 성분만 보면 엄지에서 오판한다")
    assert "RuntimeError" in blk, "판별 불가 시 fail-loud 가 없다"
    # 폐쇄도 분모도 같은 굴곡 한계를 써야 좌우 부호가 통일된다
    assert "_den = self._fab_flex_limit - self._fab_hand_home" in src, (
        "close 분모가 hi 고정이다 — 좌팔 엄지 폐쇄도 부호가 뒤집힌다")


def test_unusable_fingers_is_declared_not_just_commented():
    """★소지는 감쌈/접촉 분모에서 빠져야 한다 — 필드가 **실제로 선언**돼 있는가.

    이 필드가 없어 `getattr(..., ())` 가 늘 빈 집합이었고, `_1`·`_2` 가 둘 다
    고정이라 사실상 강체인 pinky 가 μ(touch_n≥3) 를 대신 열고 있었다
    (h7 우팔 pinky touch 0.543 · wrap 0.290 = 접촉의 절반).

    ★이 검사는 Isaac 없이도 돌아야 한다 — 이 저장소의 계약 20여 건이 Isaac 게이트
    뒤에서 조용히 skip 되고 있었고, 손가락 동결 계약이 그중 하나라 **구현이 없는데도
    "통과"로 보였다**. cfg 소스를 직접 읽어 게이트 밖에 둔다.
    """
    src = (_TASK_DIR / "grasp_lift_fabric_env_cfg.py").read_text()
    m = re.search(r"hand_unusable_fingers[^=]*=\s*(\([^)]*\))", src)
    assert m, "hand_unusable_fingers 가 선언되지 않았다"
    assert "pinky" in m.group(1), f"소지가 제외 목록에 없다({m.group(1)})"
    env_src = (_TASK_DIR / "grasp_lift_fabric_env.py").read_text()
    assert 'getattr(self.cfg, "hand_unusable_fingers"' in env_src, (
        "env 가 이 필드를 읽지 않는다")
