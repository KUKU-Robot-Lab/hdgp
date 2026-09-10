"""grasp_fj 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST 검사).

Track B 는 A(`grasp_kp`)의 팔 액션 어댑터만 바꾼다. 여기서 잠그는 것은 그 경계다:
fabric 런타임 0 · 팔은 위치 목표만 · 증분+EMA 식 · 훅 집합 · 차원(22/131/155) · 등록·yaml.
헬퍼는 A 의 계약 테스트에서 가져온다(같은 소스 검사 규약, 사본 금지).

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj/tests -q
"""

from __future__ import annotations

import ast
import re
import textwrap
import types
from pathlib import Path

import pytest

from openarm.agnostic.modules.source_contract import (
    _class_methods,
    _code,
    _fn_block,
    _ordered,
)

_HERE = Path(__file__).resolve().parent.parent
_KP = _HERE.parent / "grasp_kp"
_ENV = (_HERE / "grasp_fj_env.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")
_LSTM = (_HERE / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")
_MLP = (_HERE / "config" / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")
# ★★09.10 포크 이후 두 갈래다.
#   `_PARENT_*` = fj 가 **실제로 상속해 도는** 포크본. fj 의 계약은 여기를 봐야 한다.
#   `_KP_*`     = 원본 track A. "A 는 안 건드린다" 류 불변만 여기를 본다.
#   섞으면 fj 가 안 쓰는 파일을 검사하는 무증상 no-op 계약이 된다.
_PARENT_ENV = (_HERE / "fj_kp_env.py").read_text(encoding="utf-8")
_PARENT_CFG = (_HERE / "fj_kp_cfg.py").read_text(encoding="utf-8")
_KP_ENV = (_KP / "grasp_kp_env.py").read_text(encoding="utf-8")
_KP_CFG = (_KP / "grasp_kp_env_cfg.py").read_text(encoding="utf-8")
_SAPG = (_HERE / "config" / "agents" / "rl_games_ppo_lstm_sapg_cfg.yaml").read_text(encoding="utf-8")


def _calls(src: str, name: str) -> list[str]:
    """`self.robot.<name>(...)` 호출의 괄호 안 본문 전부(괄호 균형)."""
    out = []
    for m in re.finditer(rf"self\.robot\.{re.escape(name)}\(", src):
        i, depth, j = m.end() - 1, 0, m.end() - 1
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    out.append(src[i + 1:j])
                    break
            j += 1
    return out


def _class_body(src: str, name: str) -> str:
    """클래스 `name` 의 본문 소스(하위 클래스 제외). 방화벽 검사 = "이 리터럴이 어느 클래스에 있나"."""
    tree = ast.parse(src)
    node = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)
    assert node is not None, f"클래스 {name} 부재"
    return "\n".join(src.split("\n")[node.lineno - 1:node.end_lineno])


def _is_noop(src: str, name: str) -> bool:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            return all(isinstance(n, ast.Pass) or (isinstance(n, ast.Return) and (
                n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None)))
                for n in body)
    raise AssertionError(f"함수 {name} 부재")


# ---------------------------------------------------------------- 등록
def test_four_task_ids_registered():
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_fj.grasp_fj_env:GraspFJEnv"' in _REG
    for suffix in ('""', '"-play"', '"-lstm"', '"-play-lstm"'):
        assert suffix in _REG, f"태스크 id 접미사 {suffix} 미등록"
    assert 'f"open-{_tag}_grasp_fj{_suffix}"' in _REG
    assert re.search(r'"sens_r":\s*GraspFJTesolloRightEnvCfg', _REG)
    assert "grasp_s2r" not in _code(_REG) and "grasp_kp" not in _code(_REG)


def test_registration_has_no_fabric_gate():
    """Track B 는 Fabrics 자산이 필요 없다 — `fabric_class` 게이트가 있으면 안 된다."""
    assert "fabric_class" not in _code(_REG)
    assert "SKIPPED" in _REG and "REGISTERED" in _REG


def test_robot_profiles_is_a_shared_reexport():
    """★09.10 단일 출처는 유지하되 출처가 s2r 이 아니라 중립 `modules/` 다.

    사용자 확정 "s2r 하고 fj 는 공유 금지" 는 벤더 값을 두 벌로 만들라는 뜻이 아니다
    ("모든 값은 벤더 기준, 단일 출처"). 그래서 실체를 `modules/` 로 옮기고 fj 는
    거기서 읽는다 — fj 에는 `grasp_s2r` 참조가 하나도 남지 않는다.
    """
    from openarm.agnostic.modules import robot_profiles as shared
    from openarm.agnostic.tasks.grasp_fj import robot_profiles as fj
    assert fj.PROFILES is shared.PROFILES and "tesollo_right" in fj.PROFILES


# ---------------------------------------------------------------- 훅 집합·fabric 0
def test_env_overrides_exactly_the_adapter_hook_set():
    """팔 어댑터 훅만 덮는다 — 보상·관측·종료·리셋 본체를 덮으면 A/B 대조가 깨진다."""
    names = _class_methods(_ENV, "GraspFJEnv")
    required = {"_setup_fabrics", "_init_home_palm", "_step_fabric", "_post_command",
                "_arm_command", "_apply_action", "_cmd_state", "_log_fabric_metrics", "_reset_idx"}
    helpers = {"_build_joint_index", "_build_syn_to_fab_idx", "_apply_arm_target_box",
               "_hand_targets",                      # ★09.08 손 20관절 full-joint(선형 + 관절 EMA)
               "_action_obs",                        # ★09.08 관측 액션 블록: 손 20 = 정규화 관절 목표(SimToolReal prev_action_targets)
               "_progress_reward",                   # ★09.08 B 전용 보상 이음매(모듈 포크)
               "_restart_goal_clock",                # ★09.08 목표당 스텝 예산
               "_build_hand_action_range",           # ★09.08 관절별 액션한계(soft limit ∩ 프로필 override)
               "_hand_mask",                         # 정규식 해석(관절명은 프로필 소유)
               "_hand_curl",                         # ★09.09 감쌈 보상 입력 — **실측** _2/_3 정규화 굴곡.
                                                      #   보상 이음매(_progress_reward)가 쓴다. 지령이 아니라
                                                      #   실측이어야 "시키기만 하고 끝"이 안 된다.
               "_log_joint_limit_violation",         # ★09.10 관절별 한계 이탈 로깅 — **진단 전용**
                                                      #   09.08~09.10 내내 이탈이 문제였는데 TB 에 지표가 없어
                                                      #   판정을 매번 체크포인트 재생으로만 할 수 있었다.
                                                      #   비율(viol/frac)과 크기(viol/max_rad)를 나눠 남긴다 —
                                                      #   `ctrl/hand_joint_err_max` 는 전 env 최대값이라
                                                      #   "얼마나 자주"를 못 말한다.
               "_seg_masks"}                          # ★09.09 마디별 진단 마스크 캐시 — **진단 전용**
                                                      #   순수 인덱스 헬퍼이고 보상·관측·종료 어디에도 안 쓴다.
                                                      #   실측 폐쇄도(task/syn_close_actual_seg*)를 마디별로 남기려고
                                                      #   추가했다: `task/syn_close` 가 지령이라 "안 닫는다"와
                                                      #   "못 닫는다"가 3200 epoch 동안 구분되지 않았다.
    forbidden = {"__init__", "_setup_scene", "_init_task_state", "_pre_physics_step", "_hand_command",
                 "_get_observations", "_get_rewards", "_get_dones", "_synergy_targets",
                 "_setup_synergy", "_palm_anchor", "_apply_gravity_compensation", "_apply_wrench"}
    assert required <= names, required - names
    assert not (forbidden & names), forbidden & names
    assert names <= required | helpers, names - (required | helpers)


def test_no_fabric_runtime_in_track_b():
    code = _code(_ENV) + "\n" + _code(_CFG)
    for banned in ("fabrics_sim", "WorldMeshesModel", "DisplacementIntegrator", "integrator",
                   "set_features", "_fingertip_taskmap", "get_palm_pose", "initialize_warp",
                   "_build_fabric_index", "_build_fabric_world", "_fabric_hand_cmd", "_fabric_damping"):
        assert banned not in code, banned
    assert "self.fabric = None" in _fn_block(_ENV, "_setup_fabrics")
    assert _is_noop(_ENV, "_step_fabric") and _is_noop(_ENV, "_post_command")


def test_no_contact_sensor_consumers_anywhere():
    code = _code(_ENV) + "\n" + _code(_CFG)
    for banned in ("ContactSensor", "force_matrix_w", "net_forces_w", "_tip_force_local",
                   "_contact_forces", "_log_diagnostics", "_palmar_mask", "_hold_count",
                   "_wrap_at_latch", "_disp_at_latch", "compute_grasp_s2r_rewards"):
        assert banned not in code, banned


def test_no_robot_joint_literals():
    pat = re.compile(r"\b[rl]_(aj|hj|hl)_")
    for name, src in (("env", _ENV), ("cfg", _CFG), ("reg", _REG)):
        assert pat.search(_code(src)) is None, name
    assert re.search(r"(stiffness|damping)\s*=\s*[0-9]", _code(_ENV) + _code(_CFG)) is None


# ---------------------------------------------------------------- 팔 경로
def test_arm_gets_position_target_only():
    """DESIGN §1 B: 팔은 위치 목표만 — 속도 목표는 손(`_syn_ids`)에만, 팔(`arm_ids`)엔 없다."""
    code = _code(_ENV)
    block = _fn_block(_ENV, "_apply_action")
    assert "self.robot.set_joint_position_target(self._arm_q_target, joint_ids=self.arm_ids)" in block
    assert "self._apply_gravity_compensation()" in block
    vel = _calls(code, "set_joint_velocity_target")
    assert len(vel) == 1, f"속도 목표 호출 {len(vel)}개(손 1개여야 한다)"
    assert "joint_ids=self._syn_ids" in vel[0] and "arm_ids" not in vel[0]
    assert code.count("joint_ids=self.arm_ids") == 1, "팔 관절에 위치 목표 외의 지령이 있다"
    assert "hand_velocity_ff_scale" in vel[0], "손 경로는 mixin 그대로여야 한다(A/B 대조)"


def test_arm_command_is_delta_ema_clamped_on_previous_target():
    block = _fn_block(_ENV, "_arm_command")
    _ordered(block, [
        "float(c.k_arm) * self.actions[:, :n_arm]",
        "q_free = self._arm_q_target + step",
        "q_raw = q_free.clamp(self._arm_lo, self._arm_hi)",
        "alpha = float(c.arm_ema)",
        "self._arm_q_target = (alpha * q_raw + (1.0 - alpha) * self._arm_q_target).clamp(",
    ])
    assert "_palm_anchor" not in block and "palm_targets" not in block


def test_cmd_state_is_previous_arm_target():
    assert "return self._arm_q_target" in _fn_block(_ENV, "_cmd_state")


def test_reset_seeds_target_from_home_after_super():
    _ordered(_fn_block(_ENV, "_reset_idx"), [
        "super()._reset_idx(env_ids)",
        "self._arm_q_target[env_ids] = self._default_q[env_ids][:, self._arm_ids_t]",
    ])


def test_init_home_palm_zero_offset_and_box_check():
    block = _fn_block(_ENV, "_init_home_palm")
    for token in ("write_joint_state_to_sim(q0", "self._palm_pose_6d()[0]",
                  "self._fab_to_env = torch.zeros(3", "self._palm_lo", "raise RuntimeError"):
        assert token in block, token


def test_setup_fabrics_keeps_parent_buffers_and_logs_boot_line():
    block = _fn_block(_ENV, "_setup_fabrics")
    _ordered(block, ["self._setup_synergy()", "self.fabric = None", "self._fab_t =",
                     "self._syn_to_fab_idx =", "self.fabric_q =", "self.fabric_qd =",
                     "self.fabric_qdd =", "self._palm_lo =", "self._palm_hi =",
                     "self.palm_targets =", "self._home_palm =", "self._arm_q_target ="])
    assert "[grasp_fj] fabric OFF" in block and "k_arm" in block and "arm_ema" in block


def test_log_metrics_are_ctrl_not_fabric():
    block = _fn_block(_ENV, "_log_fabric_metrics")
    assert '"ctrl/joint_err_max"' in block and '"ctrl/joint_err_mean"' in block
    assert "fabric/" not in _code(_ENV)


# ---------------------------------------------------------------- Track A 의 B 훅 (A 가 잠그지 않는 계약)
def test_track_a_exposes_the_hooks_b_relies_on():
    """A 의 손 슬라이스·cmd_state 폭 검사가 `_arm_action_dim` 훅에서 와야 B(7)가 부팅한다."""
    assert "self._hand_action_offset = int(self.cfg._arm_action_dim(self.profile))" in \
        _fn_block(_PARENT_ENV, "_init_task_state")
    assert "self.actions[:, self._hand_action_offset:]" in _fn_block(_PARENT_ENV, "_hand_command")
    guard = _fn_block(_PARENT_ENV, "_assert_kp_contract")
    assert "c._arm_action_dim(self.profile)" in guard and "!= 6" not in guard
    # ★09.10 Phase C — `_log_fabric_metrics`·`_cmd_state` 의 부모 본문은 이 포크에서
    #   사문이라 지웠다(fj 가 super 없이 덮는다). 그래서 불변식을 **fj 소유 위치**에서 본다.
    #   부모에 남겨두고 검사하면 fj 가 쓰지도 않는 코드를 지키는 no-op 계약이 된다.
    assert "self.fabric = None" in _fn_block(_ENV, "_setup_fabrics"), "B 는 fabric 을 만들지 않는다"
    assert "return self._arm_q_target" in _fn_block(_ENV, "_cmd_state"), "B 의 cmd_state 는 관절 목표다"
    for banned in ("_log_fabric_metrics", "_cmd_state", "_action_obs", "_arm_command"):
        assert f"def {banned}(" not in _PARENT_ENV, f"부모에 {banned} 사문이 되살아났다"


# ---------------------------------------------------------------- cfg·차원
def test_cfg_fields_and_arm_action_dim_hook():
    code = _code(_CFG)
    for token in ("arm_cmd_dim: int = 7", "k_arm: float = 0.025", "arm_ema: float = 0.1",
                  "arm_slew_rad_s: float = 0.15",
                  "class GraspFJEnvCfg(FJKeypointEnvCfg)", 'profile_name: str = "tesollo_right"'):
        assert token in code, token
    assert "return int(profile.num_arm_joints)" in _fn_block(_CFG, "_arm_action_dim")
    assert "_derive_spaces" not in _class_methods(_CFG, "GraspFJEnvCfg"), "차원 공식은 A 단일 출처"
    val = _fn_block(_CFG, "_validate_fj_fields")
    for token in ("num_arm_joints", "k_arm", "arm_ema", '"per_finger"', "raise RuntimeError",
                  "float(self.arm_ema) * float(self.k_arm) / _dt", "arm_slew_rad_s"):
        assert token in val, token


def test_effective_arm_slew_matches_simtoolreal():
    """★09.08 `k_arm` 은 SimToolReal 의 `dof_speed_scale · dt` 를 **상수 하나로 접은 값**이다.

    그들: `arm_raw = prev + dof_speed_scale · dt · a` (action_utils.py:52), dt = 1/60,
          controlFrequencyInv 1 → 정책 스텝 1/60 s ⇒ 계수 = 1.5 × 1/60 = 0.025.
    우리: `arm_raw = prev + k_arm · a`, 정책 스텝 = sim.dt(1/120) × decimation 2 = 1/60 s.
    ⇒ **k_arm ≡ dof_speed_scale × 정책_dt**. 구 0.167 은 dof_speed_scale 10.0 = 그들의 6.7배였다.
    ★이 등식은 정책 dt 가 양쪽 다 1/60 이라서 성립한다 — dt 를 바꾸면 k_arm 을 다시 접어야 한다.
      그래서 실효 slew 뿐 아니라 **환산된 dof_speed_scale 자체**를 여기서도, cfg 에서도 잠근다.
    """
    code = _code(_CFG)
    k = float(re.search(r"k_arm: float = ([0-9.]+)", code).group(1))
    a = float(re.search(r"arm_ema: float = ([0-9.]+)", code).group(1))
    slew = float(re.search(r"arm_slew_rad_s: float = ([0-9.]+)", code).group(1))
    _dt = 2.0 / 120.0                                                      # 정책 dt = sim.dt × decimation
    assert abs(a * k / _dt - slew) <= 0.02 * slew, (a, k, slew)
    assert abs(k / _dt - 1.5) <= 0.02 * 1.5, f"환산 dofSpeedScale {k / _dt} ≠ 1.5"
    val = _fn_block(_CFG, "_validate_fj_fields")
    assert "_implied_speed_scale" in val and "arm_dof_speed_scale" in val, (
        "환산식이 cfg 검증기에도 잠겨 있어야 한다")
    assert "arm_dof_speed_scale: float = 1.5" in code, "선언값이 SimToolReal dofSpeedScale 이어야 한다"
    assert "self.tol_eval = self.tol_floor" in _fn_block(_REG, "__post_init__")
    fin = _fn_block(_CFG, "finalize_after_overrides")
    _ordered(fin, ["super().finalize_after_overrides()", "self._validate_fj_fields("])


def test_derived_dims_are_22_131_155_when_the_hand_stays_synergic():
    """A 의 `_derive_spaces` 공식을 B 의 훅(팔 7 · 손 시너지)으로 실행한다 — 131 / critic +24 = 155.

    ★09.08 주의: **등록되는** tesollo_right leaf 는 `hand_direct=True` 라 실제 계약은
      27/136/160 이다(`test_shipped_tesollo_right_contract_is_27_136_160`). 이 테스트는
      "손을 시너지로 두면" 이라는 **조건부** 공식 검사다 — fj_b9 체크포인트가 사는 차원.
    """
    tree = ast.parse(_PARENT_CFG)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_derive_spaces")
    src = textwrap.dedent("\n".join(_PARENT_CFG.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = {"_KP_DIM": 12, "NUM_KEYPOINTS": 4}
    exec(src, ns)  # noqa: S102 — 소스 자신의 공식
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    cfg = types.SimpleNamespace(hand_layout="coupled3", arm_cmd_dim=7,
                                _arm_action_dim=lambda profile: int(profile.num_arm_joints),
                                _hand_action_dim=lambda profile: 3 * len(profile.finger_sensor_bodies))
    ns["_derive_spaces"](cfg, PROFILES["tesollo_right"])
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (22, 131, 155)


# ---------------------------------------------------------------- 손 20관절 독립 (09.08)
def test_hand_direct_is_off_by_default_so_existing_dims_and_siblings_are_untouched():
    """★기본은 False. 켜는 건 런처의 `env.hand_direct=true` 뿐이다.

    이유 둘: (1) 기존 22/131/155 계약과 fj_b9 체크포인트를 그대로 둔다.
    (2) `grasp_fj_rh`(RH56F1)가 이 cfg 를 상속하므로, 기본이 True 면 그 트랙의 차원이
        조용히 바뀐다 — 남의 진행 중 작업을 잡아가지 않는다.
    """
    assert "hand_direct: bool = False" in _class_body(_CFG, "GraspFJEnvCfg")
    # leaf 는 켠다(사용자 지시: 7+20 DOF). base 가 아니라 leaf 라야 fj_rh 가 안 잡힌다.
    assert "hand_direct: bool = True" in _class_body(_CFG, "GraspFJTesolloRightEnvCfg")


def test_hand_direct_widens_only_the_hand_slice():
    """켜면 손 폭이 관절 수(20)가 된다. 팔 훅은 그대로 7."""
    block = _fn_block(_CFG, "_hand_action_dim")
    assert "int(profile.num_hand_joints)" in block
    assert "super()._hand_action_dim(profile)" in block, "꺼져 있으면 A 공식 그대로"


def test_hand_direct_law_is_the_simtoolreal_linear_map_plus_joint_ema():
    """★09.08 사용자 확정: "SimToolReal 처럼 풀 조인트" — 폐쇄도·램프·close_gate·blocked·open→grip
    보정을 전부 걷어냈다. 남는 것은 `action_utils.py:61-69` 와 같은 꼴 하나다:

        raw = lo + ½(a+1)(hi−lo) → q* = α·raw + (1−α)·q*_{t-1} → clamp(lo, hi)

    `q*_{t-1}` 은 `_syn_target`(A 의 `_hand_command` 가 반환값을 거기 넣는다) 이라 리셋 자세에서
    출발한다(SimToolReal `prev_targets = joint_pos`). 안전장치는 법칙이 아니라 **범위**에 있다 —
    `_build_hand_action_range` 가 soft limit ∩ 프로필 override 로 `_3/_4` 하한 0 을 만든다.
    """
    block = _fn_block(_ENV, "_hand_targets")
    _ordered(block, [
        "if not bool(self.cfg.hand_direct):",
        "return super()._hand_targets(a_hand)",
        "lo, hi = self._act_lo.unsqueeze(0), self._act_hi.unsqueeze(0)",
        "raw = lo + 0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0) * (hi - lo)",
        "alpha = float(self.cfg.hand_ema)",
        "tgt = (alpha * raw + (1.0 - alpha) * self._syn_target).clamp(lo, hi)",
        "return tgt",
    ])
    # 걷어낸 것들이 법칙 안에 되살아나면 안 된다 — 진단(_hand_blocked)은 로그 훅에만 산다.
    for banned in ("_close_gate", "_hand_blocked()", "synergy_close_speed", "_syn_open", "_syn_grip",
                   "lerp", "_shape_mask", "_close_home", "_closure_to_target", "_hand_step"):
        assert banned not in _code(block), banned
    assert "_hand_step" not in _class_methods(_ENV, "GraspFJEnv")
    assert "_closure_to_target" not in _class_methods(_ENV, "GraspFJEnv")


def test_hand_target_state_is_observed_through_the_action_obs_seam():
    """★09.08 리뷰: 손 EMA 상태 q*_{t-1}(20) 이 관측에 없었다 — SimToolReal 은 post-EMA prev_action_targets
    (팔+손)를 주고 raw action 은 안 준다. α=0.1 이면 다음 목표의 90% 가 미관측 상태로 정해지고 a_{t-1}·hand_q
    (약한 PD, 접촉)로 복원되지 않는다. `_get_observations` 는 B 금지 훅이라 A 에 이음매 `_action_obs` 를 두고
    B 가 손 20칸만 정규화 목표로 바꾼다 — 폭 27 불변, A 는 `return self.actions` 그대로.
    """
    blk = _fn_block(_ENV, "_action_obs")
    _ordered(blk, [
        "if not bool(self.cfg.hand_direct):",
        "return self.actions",
        "off = self._hand_action_offset",
        "hand = 2.0 * (self._syn_target - self._act_lo.unsqueeze(0)) / self._act_span.unsqueeze(0) - 1.0",
        "return torch.cat([self.actions[:, :off], hand], dim=1)",
    ])
    # 이음매 자체는 부모가 계속 소유한다(관측 조립은 B 가 안 덮는다).
    assert "_act = self._action_obs()" in _fn_block(_PARENT_ENV, "_get_observations"), "이음매가 사라졌다"
    # ★09.10 Phase C — 부모의 `_action_obs`(= `return self.actions`) 본문은 fj 가 super 없이
    #   덮어 사문이라 지웠다. "산술 불변" 은 이제 A 트랙 원본에서 대조한다.
    assert "return self.actions" in _fn_block(_KP_ENV, "_action_obs"), "A 는 산술 불변이어야 한다"


def test_hand_action_range_is_soft_limit_intersect_profile_override_and_fails_loud():
    """★[lo, hi] = articulation soft limit ∩ 프로필 `hand_action_limit_override`. **좁히기만** 한다.

    SHARPA 는 URDF PIP/DIP 하한이 0 이라 원시 한계 매핑이 안전했지만 테솔로 `_3/_4` 는 ±1.571 대칭이다
    — a=−1 이 손등 −90° 를 지령하고, 접촉 항 0개인 이 보상에서는 손등 갈고리 파지가 정상 파지와 같은
    점수를 받는다(08.23 실측 exploit). 하한 0 이 유일한 방어선이므로 부팅에서 세 가지를 죽인다.
    """
    rng = _fn_block(_ENV, "_build_hand_action_range")
    _ordered(rng, [
        "lo, hi = self._syn_lo.clone(), self._syn_hi.clone()",
        "if not bool(self.cfg.hand_direct):",
        "self.profile.hand_action_limit_override",
        "m = self._hand_mask(regex)",
        "torch.maximum(lo, torch.full_like(lo, float(olo)))",      # 교집합: 하한은 올리기만
        "torch.minimum(hi, torch.full_like(hi, float(ohi)))",      # 교집합: 상한은 내리기만
        "폭 0 액션 칸",
        "self.robot.data.default_joint_pos[0, self._syn_ids]",
        "self._hand_reset_q = q0.clamp(lo, hi)",                 # 리셋 자세는 거부가 아니라 clamp 시드
        "hand_reset_clamp_max_rad",
        "self._act_lo, self._act_hi, self._act_span = lo, hi, span",
        "self._syn_movable = torch.ones_like(self._syn_movable)",
    ])
    # SimToolReal 은 **하드**(URDF) 한계에 매핑한다 — soft factor 1.0 전제를 부팅에서 대조한다.
    assert "self.robot.data.joint_pos_limits[0, self._syn_ids, :]" in rng and "soft_joint_pos_limit_factor" in rng
    # IsaacLab find_joints 는 fullmatch + 미매칭 ValueError — 우리 메시지로 감싼다(죽은 분기 금지).
    mask = _fn_block(_ENV, "_hand_mask")
    assert "except ValueError as e:" in mask and "아무것도 못 잡았다" in mask and "if not ids" not in mask
    assert rng.count("raise RuntimeError") >= 2, "폭 0·리셋 자세 과이탈 둘 다 부팅에서 죽어야 한다"
    # B 리셋은 clamp 된 손 자세를 관절 상태와 EMA 시드 둘 다에 심는다(A 산술 불변 — 프로필 init 은 그대로).
    rs = _fn_block(_ENV, "_reset_idx")
    _ordered(rs, ["super()._reset_idx(env_ids)", "_qh[:, self._syn_ids] = self._hand_reset_q.unsqueeze(0)",
                  "self.robot.write_joint_state_to_sim(_qh", "self._syn_target[env_ids] = self._hand_reset_q.unsqueeze(0)"])
    assert "hand_open_pose" not in _code(rng) and "hand_grip_pose" not in _code(rng), \
        "full-joint 범위는 시너지 자세와 무관하다(soft limit ∩ override 만)"
    # 프로필(관절명 소유자)에 테솔로 override 가 실재하고, **인벨롭 부호 규약**을 그대로 박는다
    # (09.10 사용자 확정 — Isaac Sim lula 실측 + URDF FK 대조).
    #   우손 _1 전부 ±0.01(0 유지) · _2 pinky 만 규칙(0~+) · _3/_4 10개 전부 하한 0.
    #   ⚠좌손은 `_3/_4` 가 thumb −, `_2` 가 pinky − 로 **부호가 다르다**.
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    ov = dict(PROFILES["tesollo_right"].hand_action_limit_override)
    names = PROFILES["tesollo_right"].hand_joint_names
    floor0 = [n for n in names if any(re.fullmatch(rx, n) and lo == 0.0 for rx, (lo, _) in ov.items())]
    assert sorted(floor0) == sorted([n for n in names if n.endswith(("_3", "_4"))] + ["r_hj_pinky_2"]), floor0
    assert sum(n.endswith(("_3", "_4")) for n in names) == 10
    zeroed = [n for n in names if any(re.fullmatch(rx, n) and (lo, hi) == (-0.01, 0.01)
                                      for rx, (lo, hi) in ov.items())]
    assert sorted(zeroed) == sorted(n for n in names if n.endswith("_1")), zeroed
    # ★대향(`thumb_2`)은 URDF 전폭이어야 한다 — 규칙에 걸리면 감쌈 범위를 깎는 회귀다.
    for free in ("r_hj_thumb_2",):
        assert not any(re.fullmatch(rx, free) for rx in ov), f"{free} 은 매뉴얼 전폭이어야 한다"
    # 소지 `_2` 는 **감싸는 관절**이다(우손 0~+1.571) — 구 코드가 외전으로 오인해 ±0.16 으로
    # 잠가 뒀다. 그 값으로 되돌아가면 소지가 감쌈에 참여하지 못한다(09.10 FK 실측: 회귀 금지).
    got = [v for rx, v in ov.items() if re.fullmatch(rx, "r_hj_pinky_2")]
    assert got == [(0.0, None)], f"r_hj_pinky_2: {got}"


def test_hand_direct_dims_are_27_136_160():
    """A 의 `_derive_spaces` 공식을 B 의 두 훅(팔 7 · 손 20)으로 실행한다."""
    tree = ast.parse(_PARENT_CFG)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_derive_spaces")
    src = textwrap.dedent("\n".join(_PARENT_CFG.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = {"_KP_DIM": 12, "NUM_KEYPOINTS": 4}
    exec(src, ns)  # noqa: S102
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    cfg = types.SimpleNamespace(hand_layout="coupled3", arm_cmd_dim=7,
                                _arm_action_dim=lambda profile: int(profile.num_arm_joints),
                                _hand_action_dim=lambda profile: int(profile.num_hand_joints))
    ns["_derive_spaces"](cfg, PROFILES["tesollo_right"])
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (27, 136, 160)


# ---------------------------------------------------------------- PPO yaml (DESIGN §6)
def test_lstm_yaml_bootstrap_gamma_and_name():
    assert "value_bootstrap: True" in _LSTM and "value_bootstrap: False" not in _LSTM
    assert "gamma: 0.99\n" in _LSTM and "gamma: 0.998" not in _LSTM
    assert _LSTM.count("units: [1024, 1024, 512, 512]") == 2
    assert "name: agn_grasp_fj-lstm" in _LSTM and "agn_grasp_kp" not in _LSTM
    assert "mixed_precision: True" in _LSTM
    # ★09.07 SimToolReal 정렬 — network 하이퍼파라미터를 원본과 동일하게 맞춘다
    #   (isaacgymenvs/cfg/train/SimToolReal{PPO,LSTMAsymmetricPPO}.yaml).
    #   ★bound_loss_type 은 원본이 키를 지정하지 않아 rl_games 기본값 'bound' 가 쓰인다.
    #     구 'regularization'(미국식 z)은 어느 분기에도 안 걸려 bounds loss 가 꺼져 있었다.
    # ★검사는 **키 값**만 본다 — 주석이 구값 이름을 설명하므로 단어 검색은 오탐이다.
    assert "bound_loss_type: bound" in _LSTM
    assert "bound_loss_type: regularization" not in _LSTM
    assert "bounds_loss_coef: 0.0001" in _LSTM
    assert "entropy_coef: 0.0\n" in _LSTM and "entropy_coef: 0.002" not in _LSTM
    assert "e_clip: 0.1" in _LSTM and "e_clip: 0.2" not in _LSTM
    assert _LSTM.count("mini_epochs: 2") == 2, "actor·central_value 둘 다 2"
    assert "concat_input: False" in _LSTM and "concat_input: True" not in _LSTM
    assert "concat_output: False" in _LSTM and "concat_output: True" not in _LSTM
    # ★minibatch 는 바꾸지 않는다 — 4,096env×16/16,384 = 4 개로 원본(24,576×16/98,304)과
    #   에폭당 미니배치 수가 이미 같다. 환경 수에 묶인 값이라 숫자를 그대로 옮기면 안 된다.
    assert "minibatch_size: 16384" in _LSTM


def test_mlp_yaml_bootstrap_and_name():
    assert "value_bootstrap: True" in _MLP and "gamma: 0.99\n" in _MLP
    assert "name: agn_grasp_fj\n" in _MLP and "agn_grasp_kp" not in _MLP


# ---------------------------------------------------------------- 리프트 후 안정 파지 (09.07 B-v)
def test_arm_cmd_rate_is_rms_action_delta_over_two():
    """B 는 증분+EMA 라 과지령이 없다(스텝당 목표 변화 = α·k·a) — 진동은 a 의 **반전**이다.

    수준 |a| 를 벌하면 이송(전속 1 rad/s)까지 세금이므로 1차 차분 RMS/2 ∈ [0,1] 을 쓴다
    (1.0 = 매 스텝 ±1 반전). 리셋 직후는 직전 액션이 0 이라 차분이 |a| 로 튀므로 0.
    """
    block = _fn_block(_ENV, "_arm_command")
    _ordered(block, [
        "_da = self.actions[:, :n_arm] - self._prev_arm_action",
        "self._cmd_rate = torch.where(",
        "self.episode_length_buf == 0",
        "0.5 * _da.pow(2).mean(dim=1).sqrt()",
        "self._prev_arm_action = self.actions[:, :n_arm].clone()",
    ])


def test_prev_arm_action_is_allocated_and_zeroed_on_reset():
    assert "self._prev_arm_action = torch.zeros(" in _fn_block(_ENV, "_setup_fabrics")
    _ordered(_fn_block(_ENV, "_reset_idx"), [
        "super()._reset_idx(env_ids)",
        "self._prev_arm_action[env_ids] = 0.0",
    ])


def test_b_cmd_rate_scale_is_one_because_the_measure_is_bounded():
    """A 의 0.1 은 작동점 ≈10×(비유계)에 맞춘 값이고, B 의 측도는 [0,1] 이라 1.0 이 같은 자릿수다."""
    assert "rw_cmd_rate_scale: float = 1.0" in _code(_CFG)


def test_log_metrics_include_lifted_action_rate():
    assert '"ctrl/arm_action_rate_lifted"' in _fn_block(_ENV, "_log_fabric_metrics")


# ---------------------------------------------------------------- SAPG yaml = b1 하이퍼 + SAPG 덮개 (09.07 B-iv)
def test_sapg_yaml_is_b1_hyperparameters_plus_sapg_overlay():
    """b1→b2 에서 9개 키가 한꺼번에 바뀌었고 b2~b6 는 전부 e100~150 에 같은 서명으로 무너졌다.
    b1 만 안 무너졌다. 그래서 학습 하이퍼는 b1 그대로, SAPG 는 그 위의 덮개로만 얹는다.
    `bound_loss_type: regularization` 은 rl_games 두 분기 어디에도 안 걸려 bounds loss OFF — b1 의
    **실효** 상태를 그대로 재현하는 것이지 오타를 못 본 게 아니다.
    """
    for token in ("concat_input: True", "concat_output: True", "mixed_precision: False",
                  "e_clip: 0.2", "clip_observations: 5.0",
                  "bound_loss_type: regularization", "bounds_loss_coef: 0.005",
                  "fixed_sigma: coef_cond", "use_others_experience: lf", "off_policy_ratio: 1.0",
                  "expl_type: mixed_expl_learn_param", "expl_reward_type: entropy",
                  "expl_coef_block_size: 2048", "score_to_win: 1000000",
                  "value_bootstrap: True", "zero_rnn_on_done: True", "entropy_coef: 0.0\n"):
        assert token in _SAPG, token
    # 사다리 상단 = b1 의 균일 0.002: linspace(0.5,0,4)×0.004 = [0.002, 0.00133, 0.00067, 0]
    assert "expl_reward_coef_scale: 0.004" in _SAPG
    assert _SAPG.count("learning_rate: 3e-4") == 1 and _SAPG.count("learning_rate: 1e-4") == 1, "actor 3e-4 · critic 1e-4 (b1)"
    assert _SAPG.count("mini_epochs: 4") == 2, "actor·central_value 둘 다 4 (b1)"
    # 8,192×16 = 131,072 (+lf 증강 32,768) / 32,768 → 5 개. b4~b6 가 실제로 돈 값(CLI 덮어쓰기)을 yaml 에 고정한다.
    assert _SAPG.count("minibatch_size: 32768") == 2
    for bad in ("concat_input: False", "concat_output: False", "mixed_precision: True", "e_clip: 0.1",
                "bound_loss_type: bound", "mini_epochs: 2", "minibatch_size: 65536",
                "clip_observations: 10.0", "expl_reward_coef_scale: 0.002", "bounds_loss_coef: 0.0001"):
        assert bad not in _SAPG, bad


def test_shipped_tesollo_right_contract_is_27_136_160():
    """★등록부가 인스턴스화하는 것은 leaf 하나뿐이고, leaf 는 `hand_direct=True` 다.

    따라서 **출하되는** 계약은 27/136/160 이다. 위 22/131/155 테스트는 조건부 공식 검사이고
    이쪽이 실제 계약이다 — 둘이 갈리면 "차원은 통과하는데 체크포인트가 안 맞는" 상태가 된다.
    fj_b9(22/131/155) 는 이 판과 **호환되지 않는다**. FRESH 로 돌린다.
    """
    assert "hand_direct: bool = True" in _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    assert re.search(r'"sens_r":\s*GraspFJTesolloRightEnvCfg', _REG), "등록부는 leaf 만 쓴다"


# ------------------------------------------------- 09.08 SimToolReal 과제 의미 정합
def test_hand_ema_is_a_base_mechanism_and_the_validator_kills_bad_pairs():
    """★관절 목표 EMA(SimToolReal handMovingAverage 0.1)는 `arm_ema` 와 같은 급의 base 메커니즘이다.

    검증기는 (0, 1] 밖을 죽이고, hand_direct 에서 속도 피드포워드를 금지한다 — 램프가 없어
    `_syn_vel` 이 최대 ≈19 rad/s 까지 뛰기 때문이다(SimToolReal 은 위치 목표만 준다).
    """
    assert "hand_ema: float = 0.1" in _class_body(_CFG, "GraspFJEnvCfg")
    val = _fn_block(_CFG, "_validate_fj_fields")
    assert "0.0 < float(self.hand_ema) <= 1.0" in val
    assert "float(self.hand_velocity_ff_scale) != 0.0" in val and "raise RuntimeError" in val
    # 걷어낸 필드가 cfg 어디에도 되살아나면 안 된다.
    for gone in ("synergy_close_ema", "hand_range_mode", "hand_shape_span_rad", "hand_wide_span_rad"):
        assert gone not in _code(_CFG), gone
    assert "synergy_close_speed" not in _class_body(_CFG, "GraspFJTesolloRightEnvCfg"), \
        "full-joint 손에 램프 값은 의미가 없다 — leaf 가 갖고 있으면 누군가 읽는 줄 안다"


def test_goal_clock_restart_is_off_by_default_and_lives_in_the_only_legal_hook():
    """★목표당 스텝 예산(SimToolReal env.py:2437-2439). 기본은 끔(−1).

    time-out 술어는 불변 트랙(`grasp_s2r_env`)이 소유하고 `_get_dones` 는 B 의 계약 금지 훅이다.
    `_get_rewards`(성공 확정) → `_log_step` → `_log_fabric_metrics` 순서라, 성공이 확정된
    **같은 스텝 안**에서 시계를 되돌릴 수 있는 유일한 허용 지점이 로그 훅의 첫 줄이다.
    """
    assert "goal_clock_restart_step: int = -1" in _class_body(_CFG, "GraspFJEnvCfg")
    blk = _fn_block(_ENV, "_restart_goal_clock")
    assert "self.episode_length_buf.masked_fill_(self._success_now" in blk
    log = _fn_block(_ENV, "_log_fabric_metrics")
    assert "self._restart_goal_clock()" in log
    body = [ln.strip() for ln in log.split("\n") if ln.strip() and not ln.strip().startswith(('"""', "#"))]
    assert body[1] == "self._restart_goal_clock()", f"로그 훅 첫 줄이어야 한다: {body[:3]}"
    assert "_get_dones" not in _class_methods(_ENV, "GraspFJEnv"), "종료 술어는 여전히 안 덮는다"


def test_goal_clock_restart_value_dodges_the_reset_diagnostic():
    """★왜 0 이 아니라 2 인가 — `episode_length_buf` 가 0/1 일 때 반응하는 소비자가 넷 있다.

    · 액션·관측 지연 flush(`grasp_kp_env`) — 목표마다 지연 큐가 비워진다(DR 축이 무력화).
    · **물체** 지연 flush(10스텝) — 목표마다 물체 참값을 훔쳐본다. 보상과 직결이라 제일 나쁘다.
    · `_fresh <= 1` 리셋 진단(`grasp_s2r_env`) — 영구 오염되면 `reset/arm_q_dev_max` 가
      `replicate_physics` 오리셋(27.85 rad)을 잡는 fail-loud 가드 기능을 잃는다.
    2 면 넷 다 피하고 비용은 목표당 600 → 597 스텝(0.5%)이다. 회귀 울타리.
    """
    assert "goal_clock_restart_step: int = 2" in _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    s2r = (_HERE.parent / "grasp_s2r" / "grasp_s2r_env.py").read_text(encoding="utf-8")
    assert "_fresh" in s2r and "<= 1" in s2r, "회피 대상인 리셋 진단 술어가 실재해야 한다"


def test_tolerance_success_predicate_and_goal_z_are_locked_as_a_triple():
    """★kp_a8: 연속 판정을 공차 없이 베끼면 리프트가 죽는다. 그리고 공차를 올리면
    첫 목표 z 하한도 같이 올려야 한다 — `goal_bonus` 는 lifted 게이트가 없어서,
    near_goal 이 리프트 래치보다 낮은 곳에서 켜지면 물체를 안 들고도 보너스가 나간다.

    현행 A: 0.16 − 0.06 = 0.10 = 래치.  B: 0.2125 − 0.1125 = 0.10 = 래치. 같은 불변식이다.
    """
    leaf = _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    for token in ("tol_start: float = 0.1125", "goal_force_consecutive: bool = True",
                  "goal_first_z_range: tuple[float, float] = (0.2125, 0.28)",
                  "tol_success_threshold: float = 2.0"):
        assert token in leaf, token
    # ★게이트는 cap 에 대한 **비율**이다 — goal_max 5 에 상류값 3.0 을 그대로 두면 60%(상류 6% 의 10배).
    _thr = float(re.search(r"tol_success_threshold: float = ([0-9.]+)", leaf).group(1))
    _gm = int(re.search(r"goal_max: int = (\d+)", leaf).group(1))
    assert 0.05 <= _thr / _gm <= 0.5, f"게이트 비율 {_thr / _gm:.2f} — 상류 6%~ 절반 사이여야 한다"
    z0 = float(re.search(r"goal_first_z_range: tuple\[float, float\] = \(([0-9.]+),", leaf).group(1))
    tol = float(re.search(r"tol_start: float = ([0-9.]+)", leaf).group(1))
    latch = float(re.search(r"rw_lift_latch_height: float = ([0-9.]+)", _PARENT_CFG).group(1))
    assert z0 - tol >= latch - 1e-9, f"{z0} − {tol} = {z0 - tol} < 래치 {latch}"
    val = _fn_block(_CFG, "_validate_fj_fields")
    assert "goal_first_z_range" in val and "rw_lift_latch_height" in val, "짝이 검증기에도 잠겨야 한다"
    assert "tol_start ≥ 0.10" in val


def test_task_semantics_live_on_the_leaf_so_grasp_fj_rh_is_not_captured():
    """★포획 방화벽의 기계 검사.

    형제 `grasp_fj_rh`(RH56F1, 진행 중)가 `GraspFJEnvCfg` 를 상속한다. 과제 의미 값이 base 에
    있으면 남의 트랙의 성공 술어·공차·예산·외란이 조용히 바뀐다. 등록부가 인스턴스화하는 것은
    leaf 하나뿐이므로, 이 값들은 leaf 에만 있어야 한다.
    ★`k_arm`/`arm_slew_rad_s` 만 base 인 이유: fj_rh 가 이미 동일 값으로 고정·assert 한다.
    """
    base = _class_body(_CFG, "GraspFJEnvCfg")
    leaf = _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    for field in ("tol_start", "tol_success_threshold", "goal_force_consecutive",
                  "goal_first_z_range", "wrench_force_scale", "wrench_torque_scale",
                  "hand_velocity_ff_scale"):
        assert re.search(rf"^\s+{field}:", leaf, re.M), f"{field} 가 leaf 에 없다"
        assert not re.search(rf"^\s+{field}:", base, re.M), f"{field} 가 base 에 있다 — fj_rh 포획"
    # 반대로 이 둘은 base 라야 한다(fj_rh 가 같은 값을 이미 고정했으므로 no-op).
    for field in ("k_arm", "arm_slew_rad_s", "hand_ema", "goal_clock_restart_step"):
        assert re.search(rf"^\s+{field}:", base, re.M), f"{field} 는 base 메커니즘이어야 한다"


def test_goal_sequence_is_dwell_only_for_grasp_lift():
    """★09.08 D1-a(사용자 확정): 과제 목적이 grasp-lift 만이라 목표열은 "제자리 유지" 다.

    Δ 0 → 두 번째 목표부터 같은 자리, 연속 10회 판정과 합쳐 "들고 정지" = 성공. goal_max 5 는
    커리큘럼 게이트(3.0)보다 커야 한다 — 같거나 작으면 tol 이 영원히 안 조여진다(검증기가 죽인다).
    회전 델타는 A 값(0°) 상속. A 자체(0.08 / 50)는 안 건드린다.
    """
    leaf = _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    assert "goal_delta_distance: float = 0.0" in leaf and "goal_max: int = 5" in leaf
    assert "goal_delta_rotation_deg" not in leaf
    assert "goal_delta_distance: float = 0.08" in _KP_CFG and "goal_max: int = 50" in _KP_CFG
    gm = int(re.search(r"goal_max: int = (\d+)", leaf).group(1))
    thr = float(re.search(r"tol_success_threshold: float = ([0-9.]+)", leaf).group(1))
    assert gm > thr, f"goal_max {gm} ≤ 게이트 {thr}: 커리큘럼 교착"
    val = _fn_block(_CFG, "_validate_fj_fields")
    assert "int(self.goal_max) <= float(self.tol_success_threshold)" in val


def test_drop_sticky_and_start_distance_guard_live_in_the_log_hook():
    """★09.08 계측 2종. `done/fell` 은 판정선 0.15 < 상판 0.205 라 죽어 있어 낙하를 sticky 로 센다.
    시작 거리는 IK 상수 7개에 얹힌 값이라 첫 리셋 뒤 한 번 대역 검사로 부팅을 죽인다.
    """
    log = _fn_block(_ENV, "_log_fabric_metrics")
    _ordered(log, [
        "self._drop_sticky |= self._latched & (_dz < 0.03)",
        '"ctrl/drop_sticky_frac"',
        "_fresh = (self.episode_length_buf <= 1).float()",
        "_start = (_pd * _fresh).sum() / _nf_t.clamp(min=1.0)",      # 마스크 곱·합 — GPU 텐서 유지
        '"ctrl/start_palm_dist"',
        "self.common_step_counter <= 4",                               # host 판단은 부팅 직후 몇 스텝만
        "_nf = int(_nf_t)",
        "raise RuntimeError",
    ])
    # ★09.10 기준은 **손바닥 중심**이다. 손끝 평균은 같은 팔 자세에서도 손가락 굽힘에 따라
    #   98.8→67.3→82.5 mm 로 요동쳐(FK 실측) 팔 위치를 못 잰다. 손바닥은 손가락 관절의
    #   상류라 불변(150.4 mm). 손끝으로 되돌리면 이 테스트가 막는다.
    assert "self.palm_idx" in log, "손바닥 body 로 재야 한다"
    assert "_tip_ids_t" not in log, "손끝 평균으로 되돌아가면 안 된다(자세에 흔들린다)"
    # ★스텝당 host 동기화 금지(코드베이스 불변식) — 마스크 인덱싱·무조건 int() 가 되살아나면 안 된다.
    assert "int(_fresh.sum())" not in log and "_pd[_fresh]" not in log
    assert '"ctrl/prev_ep_successes_mean"' in log, "커리큘럼 게이트 입력이 로깅돼야 한다"
    _ordered(_fn_block(_ENV, "_reset_idx"), ["super()._reset_idx(env_ids)", "self._drop_sticky[env_ids] = False"])
    assert "self._drop_sticky = torch.zeros(" in _fn_block(_ENV, "_setup_fabrics")


def test_disturbance_is_explicit_in_the_leaf_and_off_by_decision():
    """외란은 **leaf 에서 명시**되어야 하고, 지금은 사용자 확정(09.10)으로 **꺼져 있다**.

    ★왜 "값" 이 아니라 "명시" 를 잠그나 — 09.10 사고. base 를 0.0 으로 껐는데 이 leaf 가
      같은 이름을 2.7 로 재선언하고 있어 상속이 닿지 않았고, 두 런이 extF 켜진 채 돌았다.
      그러니 계약은 (a) leaf 가 두 필드를 **반드시** 자기 값으로 적을 것, (b) 그 값이
      끄기 결정과 일치할 것 — 이 둘이다. 켤 때 쓸 배율 근거는 별도로 대조한다.
      (Kuka 어깨 300 N·m vs OpenArm 40 N·m = 7.5배 → 20.0/7.5 = 2.67 ≈ 2.7 · 2.0/7.5 = 0.27)
    """
    leaf = _class_body(_CFG, "GraspFJTesolloRightEnvCfg")
    m_f = re.search(r"wrench_force_scale: float = ([\d.]+)", leaf)
    m_t = re.search(r"wrench_torque_scale: float = ([\d.]+)", leaf)
    assert m_f and m_t, "leaf 가 외란 두 필드를 명시하지 않으면 base 값이 조용히 산다"
    f, t = float(m_f.group(1)), float(m_t.group(1))
    assert (f, t) == (0.0, 0.0), f"사용자 확정 09.10 extF off — leaf 가 ({f}, {t})"
    # 켜는 경우의 배율 근거는 주석으로 남아 있어야 한다(되살릴 때 값을 다시 짓지 않도록).
    assert "7.5" in leaf, "외란 배율(어깨 토크비 7.5) 근거가 leaf 주석에서 사라졌다"
    assert "wrench_force_scale: float = 20.0" in _KP_CFG, "A 는 안 건드린다"


def test_reward_module_is_forked_for_track_b():
    """★09.08 사용자 확정: A(fabric/palm)와 B(관절 직접)는 제어 방식이 달라 보상 모듈을 안 나눈다.

    `_get_rewards` 는 B 의 계약 금지 훅이므로 통째로 덮지 않는다 — A 가 만든 이음매
    `_progress_reward` 하나만 덮는다. 갈리는 항은 `goal_bonus` 뿐(성공 순간 1회 전액).
    """
    _pr = _fn_block(_ENV, "_progress_reward")
    assert "compute_fj_reward(" in _pr and "**kw)" in _pr
    # ★09.09 이음매가 `hand_curl` 을 만들어 넘긴다. 부모 `_get_rewards` 는 계약 금지 훅이라
    #   인자를 못 늘리는데 이 이음매는 self 를 갖는다 — 그래서 여기가 유일한 지점이다.
    assert "hand_curl=self._hand_curl()" in _pr, "감쌈 보상 입력이 이음매에서 안 온다"
    # ★로깅 이름표를 인스턴스에 두지 않는다 — 부모 `_init_task_state` 가 super() 뒤에 덮어써서
    #   B 에서 아예 안 먹었다(09.08 감사에서 실행 재현). A 의 `_log_step` 이 terms 를 직접 순회한다.
    assert "_rw_terms" not in _ENV, "인스턴스 이름표는 부모가 덮어쓴다"
    assert "_get_rewards" not in _class_methods(_ENV, "GraspFJEnv")
    fj = (_HERE / "fj_reward.py").read_text(encoding="utf-8")
    assert "cfg.goal_bonus * is_success.float()" in fj, "B 는 성공 순간 1회 전액"
    assert "is_success: torch.Tensor," in fj, "선택 인자면 안 넘겼을 때 조용히 0 이 된다"
    # 1회성 보너스는 **연속 판정과 짝**이다 — 누적에서는 성공 순간이 목표 이탈 뒤에 올 수 있다.
    assert "goal_force_consecutive: bool = True" in _class_body(_CFG, "GraspFJTesolloRightEnvCfg")


def test_hand_health_metrics_exist_for_the_faster_closure_risk():
    """★폐쇄가 빨라지면 자유공간에서 손 PD 가 목표를 못 따라가 `blocked` 임계(1.0 rad)를
    오발할 수 있다 — 폐쇄가 중간에 얼어붙고 겉보기는 "손이 안 닫힌다"와 구분되지 않는다.
    `task/hand_blocked_frac` 은 `_log_diagnostics` 에 사는데 그 토큰은 이 트랙 계약 금지다.
    """
    log = _fn_block(_ENV, "_log_fabric_metrics")
    for key in ("ctrl/hand_joint_err_max", "ctrl/hand_blocked_frac", "ctrl/hand_target_step"):
        assert f'"{key}"' in log, key
    assert "_log_diagnostics" not in _ENV, "계약 금지 토큰"


# ------------------------------------------------- 검증기를 **실행**한다 (isaaclab 없이)
def _run_validator(**over):
    """`_validate_fj_fields` 소스를 leaf 기본값 + 덮어쓴 값으로 **실제 실행**한다.

    왜 필요한가: 위 테스트들은 전부 소스 문자열 검사라 검증기 안의 오타·NameError 를 못 잡는다.
    이 트랙은 "부팅에서 시끄럽게 죽는 것"이 안전장치의 전부이므로, 그 코드가 도는지 자체를 본다.
    """
    tree = ast.parse(_CFG)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_validate_fj_fields")
    src = textwrap.dedent("\n".join(_CFG.split("\n")[node.lineno - 1:node.end_lineno]))
    ns: dict = {}
    exec(src, ns)  # noqa: S102 — 소스 자신의 검증기
    base = dict(
        arm_cmd_dim=7, k_arm=0.025, arm_ema=0.1, arm_slew_rad_s=0.15, arm_target_box_rad=0.0,
        arm_dof_speed_scale=1.5,
        arm_reset_offset_rad=(),
        hand_layout="coupled3", hand_direct=True, hand_ema=0.1, hand_velocity_ff_scale=0.0,
        hand_reset_clamp_max_rad=0.6,
        episode_length_s=10.0, goal_clock_restart_step=2,
        goal_force_consecutive=True, tol_start=0.1125, goal_max=5, tol_success_threshold=2.0,
        goal_delta_distance=0.0,
        goal_first_z_range=(0.2125, 0.28), rw_lift_latch_height=0.10,
        sim=types.SimpleNamespace(dt=1.0 / 120.0), decimation=2,
        _supports_per_finger_hand=lambda: False,
    )
    base.update(over)
    cfg = types.SimpleNamespace(**base)
    ns["_validate_fj_fields"](cfg, types.SimpleNamespace(
        num_arm_joints=7, arm_reset_joint_pos=(), arm_joint_regex="r_aj_[1-7]",
        init_joint_pos={}, name="stub"))


def test_validator_accepts_the_shipped_leaf_values():
    """출하 값이 검증기를 통과한다 — 안 그러면 부팅이 안 된다."""
    _run_validator()
    # 팔 속도 대조군(fj_c1x): 둘을 **같이** 줘야 통과한다 — 하나만 주면 짝이 깨져 부팅이 죽는다.
    _run_validator(k_arm=0.167, arm_slew_rad_s=1.0, arm_dof_speed_scale=10.0)


def test_validator_kills_every_broken_pair():
    """★잘못된 hydra 오버라이드가 조용한 하이브리드로 도는 대신 부팅에서 죽는다."""
    cases = {
        "hand_ema 상한": dict(hand_ema=1.5),
        "hand_ema 0(목표 불변)": dict(hand_ema=0.0),
        "hand_direct + 속도 피드포워드": dict(hand_velocity_ff_scale=1.0),
        "slew 짝 이탈": dict(k_arm=0.167),                                 # slew 0.15 인 채로
        "dofSpeedScale 선언 누락": dict(k_arm=0.167, arm_slew_rad_s=1.0),   # 선언은 1.5 인 채로
        "연속+낮은 공차": dict(tol_start=0.06, goal_first_z_range=(0.16, 0.24)),
        "z짝 위반": dict(goal_first_z_range=(0.16, 0.24)),                # tol 0.1125 인 채로
        "시계 범위": dict(goal_clock_restart_step=600),
        "커리큘럼 교착(goal_max ≤ 게이트)": dict(goal_max=2),
        "시계 0/1(지연 flush·리셋 진단 오염)": dict(goal_clock_restart_step=1),
        "음수 목표 델타": dict(goal_delta_distance=-0.01),
    }
    for name, over in cases.items():
        try:
            _run_validator(**over)
        except RuntimeError as e:
            assert "grasp_fj cfg" in str(e), (name, e)
        else:
            raise AssertionError(f"'{name}' 을 검증기가 안 잡았다: {over}")


def test_validator_kills_a_dt_change_that_unfolds_k_arm():
    """★`k_arm ≡ dof_speed_scale × 정책_dt` — decimation 을 바꾸면 k_arm 도 환산해야 한다.

    슬루 짝 검사만으로는 못 잡는다. 현실적인 실수는 "decimation 을 바꾸고 **선언 slew 는 성실히
    고쳤는데** k_arm 이 자기 dt 를 품고 있다는 걸 잊는" 것이다:
      decimation 4 (dt 1/30) · k_arm 0.025 그대로 · slew 0.075  → α·k/dt 짝은 **맞는다**.
      그런데 k/dt = 0.75 ≠ 1.5 — 팔이 원본의 절반 속도가 된 채로 조용히 돈다.
    올바른 환산은 k_arm 0.05 · slew 0.15 이고, 그건 통과해야 한다.
    """
    _run_validator(decimation=4, k_arm=0.05, arm_slew_rad_s=0.15)      # 제대로 환산 → 통과
    with pytest.raises(RuntimeError, match=r"dofSpeedScale"):
        _run_validator(decimation=4, k_arm=0.025, arm_slew_rad_s=0.075)   # 짝은 맞고 환산만 틀림


def test_arm_reset_offset_is_fixed_not_cup_relative():
    """★08.18 에 컵 **참값** pregrasp 텔레포트를 버렸다 — 실기에서 컵 위치는 지각 결과라
    재현이 불가능했기 때문이다. 시작 거리를 맞추더라도 그 결정을 되돌리면 안 된다:
    오프셋은 물체 상태를 읽지 않는 **고정 상수**여야 한다.

    SimToolReal 도 같은 구조다 — 고정 팔 리셋 자세(`desired_kuka_pos`)가 물체의 스폰 분포
    **평균**에 대해 가까울 뿐, 매 리셋마다 물체를 보고 자세를 계산하지 않는다.
    """
    # ★09.10 시작 자세는 **프로필이 소유하는 절대 관절값**(`arm_reset_joint_pos`)이다.
    #   구 태스크 cfg 델타(`arm_reset_offset_rad`)는 자산 간 이식이 불가능해 폐기했다 —
    #   dg5f-m-short 가 그래서 홈에 앉아 시작 거리 243mm 로 돌았다(09.10).
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    assert "arm_reset_offset_rad" not in _code(_CFG), "폐기된 델타 필드가 되살아났다"
    assert len(PROFILES["tesollo_right"].arm_reset_joint_pos) == 7, "시작 자세는 프로필이 들고 있다"
    blk = _fn_block(_ENV, "_reset_idx")
    assert "self._arm_reset_q" in blk
    for banned in ("object", "obj_pos", "goal_pos", "root_pos_w", "spawn"):
        assert banned not in blk.split("_arm_reset_q")[1][:900], (
            f"리셋 자세 계산이 물체 상태('{banned}')를 읽는다 — 컵 상대 텔레포트 부활")
    # 지령 목표와 실측을 같이 옮겨야 한다(안 그러면 첫 스텝이 거대한 추종 오차로 시작한다).
    off = blk.split("self._arm_reset_q is not None")[1]
    _ordered(off, ["self.robot.write_joint_state_to_sim(_q", "self._arm_q_target[env_ids] = _q",
                   "self._prev_arm_q_target[env_ids] = self._arm_q_target[env_ids]"])
    assert "torch.clamp(" in off and "self._arm_lo[env_ids]" in off, "관절 한계를 넘으면 안 된다"


def test_registry_leaves_and_arm_specific_values_do_not_leak_to_the_short_arm():
    """★등록부는 leaf **두 개**를 쓴다: `sens_r`(tesollo_right)와 `short_r`(tesollo_right_short).
    그리고 Short 는 Right 를 **상속**한다 — 09.08 에 내가 "leaf 하나뿐"이라고 잘못 전제했다.

    과제 의미 값(공차·성공 술어·예산·외란·손 범위)은 팔 기하와 무관하므로 상속이 맞다.
    그러나 **팔 리셋 오프셋은 홈 관절값 기준 델타**라 홈이 다른 팔에 얹으면 TCP 가 어긋난다
    (Short 는 docstring 이 "팔 홈 관절값이 다르다"고 명시한다). 그래서 Short 는 반드시 끈다.
    """
    assert "GraspFJTesolloRightShortEnvCfg" in _REG, "등록부가 short 판을 쓴다"
    short = _class_body(_CFG, "GraspFJTesolloRightShortEnvCfg")
    assert "class GraspFJTesolloRightShortEnvCfg(GraspFJTesolloRightEnvCfg)" in _code(_CFG)
    # ★09.10 시작 자세는 프로필이 소유한다 — cfg 상속 문제가 아니라 **자산별 값**의 문제다.
    #   short 는 팔 홈 관절값이 달라(손이 47.8mm 짧아 더 뻗는다) 부모 값을 쓸 수 없다.
    #   IK 로 풀기 전까지 비어 있어야 하고, 그 상태로 학습하면 시작 거리 가드가 죽인다.
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    _p, _s = PROFILES["tesollo_right"], PROFILES["tesollo_right_short"]
    assert _s.arm_reset_joint_pos != _p.arm_reset_joint_pos, (
        "short 가 부모의 시작 자세를 그대로 쓰면 안 된다 — 홈 관절값이 다르다")
    assert "arm_reset_offset_rad" not in short, "폐기된 델타 필드가 되살아났다"
    # 반대로 팔 기하와 무관한 값들은 상속해야 한다(Short 가 덮지 않아야 한다).
    for field in ("tol_start", "goal_force_consecutive", "wrench_force_scale", "hand_direct"):
        assert not re.search(rf"^\s+{field}:", short, re.M), f"{field} 는 상속해야 한다"
