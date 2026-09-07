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

from openarm.agnostic.tasks.grasp_kp.tests.test_grasp_kp_contract import (
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
    from openarm.agnostic.tasks.grasp_fj import robot_profiles as fj
    from openarm.agnostic.tasks.grasp_s2r import robot_profiles as s2r
    assert fj.PROFILES is s2r.PROFILES and "tesollo_right" in fj.PROFILES


# ---------------------------------------------------------------- 훅 집합·fabric 0
def test_env_overrides_exactly_the_adapter_hook_set():
    """팔 어댑터 훅만 덮는다 — 보상·관측·종료·리셋 본체를 덮으면 A/B 대조가 깨진다."""
    names = _class_methods(_ENV, "GraspFJEnv")
    required = {"_setup_fabrics", "_init_home_palm", "_step_fabric", "_post_command",
                "_arm_command", "_apply_action", "_cmd_state", "_log_fabric_metrics", "_reset_idx"}
    helpers = {"_build_joint_index", "_build_syn_to_fab_idx"}
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
        _fn_block(_KP_ENV, "_init_task_state")
    assert "self.actions[:, self._hand_action_offset:]" in _fn_block(_KP_ENV, "_hand_command")
    guard = _fn_block(_KP_ENV, "_assert_kp_contract")
    assert "c._arm_action_dim(self.profile)" in guard and "!= 6" not in guard
    assert 'getattr(self, "fabric", None) is None' in _fn_block(_KP_ENV, "_log_fabric_metrics")
    assert "return self.palm_targets - self._palm_anchor()" in _fn_block(_KP_ENV, "_cmd_state")


# ---------------------------------------------------------------- cfg·차원
def test_cfg_fields_and_arm_action_dim_hook():
    code = _code(_CFG)
    for token in ("arm_cmd_dim: int = 7", "k_arm: float = 0.167", "arm_ema: float = 0.1",
                  "arm_slew_rad_s: float = 1.0",
                  "class GraspFJEnvCfg(GraspKPEnvCfg)", 'profile_name: str = "tesollo_right"'):
        assert token in code, token
    assert "return int(profile.num_arm_joints)" in _fn_block(_CFG, "_arm_action_dim")
    assert "_derive_spaces" not in _class_methods(_CFG, "GraspFJEnvCfg"), "차원 공식은 A 단일 출처"
    val = _fn_block(_CFG, "_validate_fj_fields")
    for token in ("num_arm_joints", "k_arm", "arm_ema", '"per_finger"', "raise RuntimeError",
                  "float(self.arm_ema) * float(self.k_arm) / _dt", "arm_slew_rad_s"):
        assert token in val, token


def test_effective_arm_slew_is_one_rad_per_s():
    """EMA 가 누적 목표에 걸려 스텝당 변화 = α·k_arm·a — 구 0.0167 은 실효 0.1 rad/s 로 설계와 10배 어긋났다."""
    code = _code(_CFG)
    k = float(re.search(r"k_arm: float = ([0-9.]+)", code).group(1))
    a = float(re.search(r"arm_ema: float = ([0-9.]+)", code).group(1))
    slew = float(re.search(r"arm_slew_rad_s: float = ([0-9.]+)", code).group(1))
    assert abs(a * k / (2.0 / 120.0) - slew) <= 0.02 * slew, (a, k, slew)   # 정책 dt = 1/120·2
    assert "self.tol_eval = self.tol_floor" in _fn_block(_REG, "__post_init__")
    fin = _fn_block(_CFG, "finalize_after_overrides")
    _ordered(fin, ["super().finalize_after_overrides()", "self._validate_fj_fields("])


def test_derived_dims_tesollo_right_are_22_131_155():
    """A 의 `_derive_spaces` 공식을 B 의 훅(7)으로 실행한다 — DESIGN §4 B: 131 / critic +24 = 155."""
    tree = ast.parse(_KP_CFG)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_derive_spaces")
    src = textwrap.dedent("\n".join(_KP_CFG.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = {"_KP_DIM": 12, "NUM_KEYPOINTS": 4}
    exec(src, ns)  # noqa: S102 — 소스 자신의 공식
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    cfg = types.SimpleNamespace(hand_layout="coupled3", arm_cmd_dim=7,
                                _arm_action_dim=lambda profile: int(profile.num_arm_joints))
    ns["_derive_spaces"](cfg, PROFILES["tesollo_right"])
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (22, 131, 155)


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
