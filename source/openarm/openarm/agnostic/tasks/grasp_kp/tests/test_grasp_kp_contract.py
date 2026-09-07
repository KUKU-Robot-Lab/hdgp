"""grasp_kp 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST 검사).

cfg/env 는 isaaclab 을 끌어와 Isaac 앱 없이 import 가 안 되므로, 값이 아니라 **소스의
계약**을 잠근다. 각 테스트에 왜 이 계약이 생겼는지(어떤 사고를 막는지)를 적어 둔다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_kp/tests -q
"""

from __future__ import annotations

import ast
import re
import textwrap
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ENV = (_HERE / "grasp_kp_env.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_kp_env_cfg.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")
_LSTM = (_HERE / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")
_MLP = (_HERE / "config" / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """주석·docstring 을 뺀 실행 코드만 — 설명문에 적힌 이름이 계약을 통과시키면 안 된다."""
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            doc_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in doc_lines:
            continue
        s = line.split("#", 1)[0]
        if s.strip():
            out.append(s)
    return "\n".join(out)


def _fn_block(src: str, name: str) -> str:
    """`def name(` 함수 본문(주석 제거)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.split("\n")[node.lineno - 1:node.end_lineno]
            return _code(textwrap.dedent("\n".join(lines)))
    raise AssertionError(f"함수 {name} 부재")


def _call_block(src: str, name: str) -> str:
    """`name = torch.cat(` 다중행 호출의 괄호 안 본문 — 괄호 균형으로 끝을 찾는다."""
    m = re.search(rf"\n\s*{re.escape(name)} = torch\.cat\(", src)
    assert m, f"{name} = torch.cat( 부재"
    i = src.index("(", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise AssertionError(f"{name} 호출의 괄호가 안 닫힌다")


def _ordered(block: str, tokens: list[str]) -> None:
    idx = [block.find(t) for t in tokens]
    missing = [t for t, i in zip(tokens, idx) if i < 0]
    assert not missing, f"누락 {missing}"
    assert idx == sorted(idx), f"순서 어긋남 {list(zip(tokens, idx))}"


def _class_methods(src: str, cls: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"클래스 {cls} 부재")


# ---------------------------------------------------------------- 등록
def test_four_task_ids_registered():
    """id 4종(train/play × mlp/lstm). `-play` 가 없으면 play.py·warm 수집이 죽는다."""
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_kp.grasp_kp_env:GraspKPEnv"' in _REG
    for suffix in ('""', '"-play"', '"-lstm"', '"-play-lstm"'):
        assert suffix in _REG, f"태스크 id 접미사 {suffix} 미등록"
    assert 'f"open-{_tag}_grasp_kp{_suffix}"' in _REG
    assert re.search(r'"sens_r":\s*GraspKPTesolloRightEnvCfg', _REG)
    assert "grasp_s2r" not in _code(_REG), "등록부가 grasp_s2r 를 가리킨다"


def test_registration_keeps_fabric_gate():
    """Track A 는 Fabrics 로만 돈다 — 자산 없는 프로필은 record-loud 로 건너뛴다."""
    assert "SKIPPED" in _REG and "REGISTERED" in _REG
    assert "fabric_class is None" in _code(_REG)


def test_robot_profiles_is_a_shared_reexport():
    """프로필 사본 금지 — grasp_s2r 캘리브 갱신이 조용히 안 실리는 사고 차단."""
    from openarm.agnostic.tasks.grasp_kp import robot_profiles as kp
    from openarm.agnostic.tasks.grasp_s2r import robot_profiles as s2r
    assert kp.PROFILES is s2r.PROFILES and "tesollo_right" in kp.PROFILES


# ---------------------------------------------------------------- 접촉 센서 0
def test_no_contact_sensor_consumers_anywhere():
    """사용자 확정(09.06): 접촉 센서는 생성·obs·state·reward·termination 어디에도 없다."""
    code = _code(_ENV) + "\n" + _code(_CFG)
    for banned in ("ContactSensor", "force_matrix_w", "net_forces_w", "_tip_force_local",
                   "_contact_forces", "_log_diagnostics", "_palmar_mask", "_contact_azimuth_spread",
                   "contact_force_threshold", "contact_force_max", "_hold_count", "_wrap_at_latch",
                   "_disp_at_latch", "compute_grasp_s2r_rewards"):
        assert banned not in code, banned


def test_no_robot_joint_literals():
    """로봇 이름은 프로필에서만 — env/cfg/등록부에 관절·바디 리터럴 금지."""
    pat = re.compile(r"\b[rl]_(aj|hj|hl)_")
    for name, src in (("env", _ENV), ("cfg", _CFG), ("reg", _REG)):
        assert pat.search(_code(src)) is None, name


def test_cfg_overrides_contact_dependent_defaults():
    """DESIGN §8 덮어쓰기 5개 + 관절속도 노이즈 상향 — env 부팅 가드와 짝이다."""
    code = _code(_CFG)
    for token in ('respawn_on_fail: bool = False', 'synergy_hold_mode: str = "blocked"',
                  'synergy_contact_freeze: bool = False', 'obs_object_noise_coherent: bool = True',
                  'enable_adr: bool = False', 'obs_noise_qvel: float = 0.1',
                  'adr_obs_noise_qvel_max: float = 0.1'):
        assert token in code, token
    guard = _fn_block(_ENV, "_assert_kp_contract")
    for token in ('"blocked"', "synergy_contact_freeze", "respawn_on_fail", "enable_adr", "raise RuntimeError"):
        assert token in guard, token


def test_every_obs_noise_override_has_adr_max_companion():
    """부모 `_assert_adr_monotonic` 은 ADR OFF 여도 base ≤ max 를 요구한다 — 09.06 두 트랙 부팅 사망의 원인.

    `obs_noise_*` 를 덮어쓰면 `adr_obs_noise_*_max` 도 같은 파일에서 base 이상으로 덮어써야 한다.
    """
    code = _code(_CFG)
    bases = dict(re.findall(r"^\s*obs_noise_(\w+): float = ([0-9.]+)", code, flags=re.M))
    assert bases, "obs_noise_* 오버라이드가 하나도 없다(테스트 전제 붕괴)"
    for name, base in bases.items():
        m = re.search(rf"^\s*adr_obs_noise_{name}_max: float = ([0-9.]+)", code, flags=re.M)
        assert m, f"obs_noise_{name} 오버라이드에 adr_obs_noise_{name}_max 짝이 없다"
        assert float(m.group(1)) >= float(base), (name, m.group(1), base)


def test_goal_box_reach_assert_exists_and_runs_at_boot():
    """목표 박스 ⊄ 팔 지령 범위(앵커±델타 ∩ 클램프 박스)면 목표열이 조용히 멈춘다(09.06 리뷰) — 부팅 가드."""
    init = _fn_block(_ENV, "_init_task_state")
    _ordered(init, ["self._apply_palm_floor_override()", "self._goal_cfg = c.goal_seq_cfg()",
                    "self._assert_goal_box_in_arm_reach()"])
    block = _fn_block(_ENV, "_assert_goal_box_in_arm_reach")
    # ★09.07 A-i: 증분 매핑이라 "이동량 ⊂ 델타"가 아니라 "걸음 수 ≤ 에피소드 예산"을 본다.
    for token in ("_palm_step_gain", "_TRAVERSE_BUDGET_FRAC", "self.max_episode_length",
                  "_box_lo", "_box_hi", "_anchor_off", "box_min", "box_max",
                  "spawn_range", "_obj_origin_off", "tol_floor", "raise RuntimeError"):
        assert token in block, token
    assert "_delta_lo" not in block and "_delta_hi" not in block, \
        "증분 매핑에서 델타 도달성 검사는 의미가 없다 — 되살아나면 구식 전제가 섞인 것"
    assert 'getattr(self, "fabric", None) is None' in block, "Track B(관절공간)는 건너뛰어야 한다"
    code = _code(_CFG)
    assert "palm_delta_xyz: tuple[float, float, float] = (0.10, 0.10, 0.35)" in code
    assert "goal_box_xy_halfwidth: float = 0.08" in code


def test_tol_eval_fixes_curriculum_for_play():
    """커리큘럼 상태는 체크포인트에 없다 — play 는 고정 tol(tol_eval>0)로 성공수를 비교 가능하게 잰다."""
    assert "tol_eval: float = 0.0" in _code(_CFG)
    init = _fn_block(_ENV, "_init_task_state")
    assert "float(c.tol_eval) > 0.0" in init and "start=float(c.tol_eval), floor=float(c.tol_eval)" in init
    assert "self.tol_eval = self.tol_floor" in _fn_block(_REG, "__post_init__")


# ---------------------------------------------------------------- 차원 공식 = 조립
def test_obs_formula_tokens_in_derive_spaces():
    block = _fn_block(_CFG, "_derive_spaces")
    for token in ("2 * n_arm", "2 * n_hand", "3 * num_tips", "int(self.arm_cmd_dim)",
                  "_KP_DIM + _KP_DIM", "self.action_space",
                  "+ 6 + 6 + 1 + num_tips + 1 + 1 + 1 + 1 + 1 + 1"):
        assert token in block, token
    assert "_KP_DIM = 3 * NUM_KEYPOINTS" in _code(_CFG)


def test_derived_dims_tesollo_right_are_21_129_153():
    """공식을 실제로 실행한다(Isaac 불필요) — DESIGN §4 A: 129 / critic +24 = 153."""
    tree = ast.parse(_CFG)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_derive_spaces")
    src = textwrap.dedent("\n".join(_CFG.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = {"_KP_DIM": 12, "NUM_KEYPOINTS": 4}
    exec(src, ns)  # noqa: S102 — 소스 자신의 공식
    from openarm.agnostic.tasks.grasp_kp.robot_profiles import PROFILES
    cfg = types.SimpleNamespace(hand_layout="coupled3", arm_cmd_dim=6,
                                _arm_action_dim=lambda profile: 6)
    ns["_derive_spaces"](cfg, PROFILES["tesollo_right"])
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (21, 129, 153)


def test_actor_obs_assembly_matches_design_order():
    """DESIGN §4 actor 순서 — `_derive_spaces` 공식과 같은 순서로 cat 해야 한다."""
    _ordered(_call_block(_code(_ENV), "_noisy"), [
        '"arm_q"', '"arm_qd"', '"hand_q"', '"hand_qd"', '"palm_pos"', '"palm_ax"',
        '"tips_rel_palm"', '"cmd_state"', '"n_kp_rel_palm"', '"n_kp_rel_goal"',
        "self.actions",
    ])
    _ordered(_call_block(_code(_ENV), "clean"), [
        '"arm_q"', '"arm_qd"', '"hand_q"', '"hand_qd"', '"palm_pos"', '"palm_ax"',
        '"tips_rel_palm"', '"cmd_state"', '"kp_rel_palm"', '"kp_rel_goal"',
        "self.actions",
    ])
    # 물체 쿼터니언 금지(09.06 리뷰): 키포인트 밖 정보는 yaw·부호뿐 — 축대칭 실기 yaw 는 임의라 분포 밖 채널.
    obs = _fn_block(_ENV, "_get_observations") + _fn_block(_ENV, "_object_blocks")
    assert "n_quat" not in obs and 'ob["quat"]' not in obs and "quat=" not in obs


def test_critic_state_assembly_matches_design_order():
    """critic = clean + 물체 속도(6) + palm 속도(6) + d*_kp + d*_ft + lifted + progress + successes + reward + dz + d_kp."""
    _ordered(_fn_block(_ENV, "_privileged_blocks"), [
        "root_lin_vel_w", "root_ang_vel_w", "body_lin_vel_w", "body_ang_vel_w",
        "closest_kp", "closest_ft", "_latched", "episode_length_buf", "successes",
        "_last_reward", '"dz"', '"kp_dist"',
    ])
    assert "state = torch.cat([clean] + self._privileged_blocks(ob)" in _code(_ENV)


def test_obs_shape_guard_raises_with_both_numbers():
    block = _fn_block(_ENV, "_check_obs_shapes_once")
    assert "observation_space" in block and "state_space" in block and "raise RuntimeError" in block
    assert "self._check_obs_shapes_once(_noisy, state)" in _fn_block(_ENV, "_get_observations")


# ---------------------------------------------------------------- 보상·목표·래치
def test_reward_is_progress_only_via_shared_module():
    block = _fn_block(_ENV, "_get_rewards")
    assert "compute_progress_reward(" in block and "update_near_goal(" in block
    assert "PROGRESS_REWARD_TERMS" in _code(_ENV)
    assert 'self._latched = out["lifted"]' in block, "래치가 높이 래치(lifted)로 재정의되지 않았다"
    assert "_tol.update(self._trk.prev_episode_successes)" in block


def test_goal_advance_uses_module_sampler_from_previous_goal():
    """다음 목표는 **이전 목표** 기준 델타(SimToolReal) — 물체 위치 기준이면 목표가 물체를 쫓는다."""
    block = _fn_block(_ENV, "_advance_goals")
    assert "sample_delta_goal(self.goal_pos, self.goal_quat" in block
    assert "clear_goal(" in block and "successes += " in block


def test_stage_ladder_is_lift_then_goals():
    assert '("lifted", "goal1", "goal2", "goal3")' in _code(_ENV)


# ---------------------------------------------------------------- 액션 경로
def test_pre_physics_step_is_delay_then_arm_hand_post_wrench():
    _ordered(_fn_block(_ENV, "_pre_physics_step"), [
        "self.episode_length_buf == 0", "_act_delay.push(", "self._arm_command()",
        "self._hand_command()", "self._post_command()", "self._apply_wrench()",
    ])


def test_arm_command_has_both_mappings_behind_one_switch():
    """★09.07 A-iv: 절대·증분 두 매핑이 `palm_cmd_incremental` 하나로 갈린다.

    증분(A-i/A-ii/A-iii)은 포화를 없애는 데는 성공했지만(rate_sat 0.98 → 0.000,
    step_raw 0.17 → 0.0067 m, arm_qd_p99 2.30 → 1.30) **과제를 세 번 다 못 배웠다** —
    kp_a3/a4/a5 모두 lifted 0.0000. 같은 보상·박스의 kp_a2(절대)는 e476 에 0.658 이다.
    그래서 기본을 절대로 되돌리되 증분 경로는 지우지 않는다(제어 품질 목표는 여전히 유효).
    """
    block = _fn_block(_ENV, "_arm_command")
    assert "if bool(self.cfg.palm_cmd_incremental):" in block, "매핑 스위치가 없다"
    # 증분 가지 — 게인·복원·클램프된 적분기 저장
    for token in ("self._palm_step_gain * self.actions[:, :6]",
                  "_pull = self._palm_pull * (self._palm_anchor() - _prev6)",
                  "_prev6 + _pull + step"):
        assert token in block, "증분 가지가 사라졌다: " + token
    # 절대 가지 — 앵커 + 델타
    for token in ("self._palm_anchor() + delta", "self._delta_hi - self._delta_lo"):
        assert token in block, "절대 가지가 없다: " + token
    # 리미터는 두 가지 공통의 안전망이다
    for token in ("palm_cmd_rate_limit_m", "palm_cmd_rate_limit_rot_deg",
                  "self._update_cmd_markers()"):
        assert token in block, token
    # 적분기는 **클램프된** 값을 저장해야 한다(원값 저장 = 박스 밖 와인드업).
    assert "self._prev_palm_cmd = self.palm_targets[:, :3].clone()" in block


def test_absolute_mapping_is_the_default():
    """기본이 증분으로 되돌아가면 검증된 유일한 리프트 경로(kp_a2)를 잃는다."""
    assert "palm_cmd_incremental: bool = False" in _code(_CFG), \
        "기본이 절대가 아니다 — a3/a4/a5 가 전부 lifted 0.0000 이었다"


def test_increment_only_state_is_skipped_under_absolute_mapping():
    """절대 매핑에는 유지할 적분기 상태도, 걸어갈 시간 제약도 없다 — 둘 다 스위치로 꺼져야 한다."""
    seed = _fn_block(_ENV, "_seed_palm_integrator")
    assert "if not bool(self.cfg.palm_cmd_incremental):" in seed and "return" in seed, \
        "절대 매핑에서 앵커 씨딩이 안 꺼진다 — 부모의 홈 리셋을 덮어쓴다"
    reach = _fn_block(_ENV, "_assert_goal_box_in_arm_reach")
    assert "if not bool(c.palm_cmd_incremental):" in reach, \
        "절대 매핑에서 traverse 예산 검사가 안 꺼진다 — 없는 제약으로 부팅이 죽는다"


def test_palm_step_gain_is_derived_from_the_rate_limiter():
    """게인을 상수로 적으면 리미터를 바꿀 때 조용히 어긋난다 — 반드시 파생시킨다."""
    block = _fn_block(_ENV, "_setup_palm_step_gain")
    for token in ("palm_cmd_rate_limit_m", "palm_cmd_rate_limit_rot_deg",
                  "math.sqrt(3.0)", "raise RuntimeError"):
        assert token in block, token
    # √3 로 나눠야 정육면체 최악(대각) 노름이 리미터와 같아진다. ★거기서 **여유**를 더 빼야
    #   float32 오차로 리미터가 nm 단위로 걸려 rate_sat 진단이 오염되는 것을 막는다(스모크 0.0625).
    assert "_STEP_GAIN_MARGIN" in block and "_lm * _k" in block and "_lr * _k" in block


def test_anchor_pull_shares_the_rate_budget_with_the_action():
    """★A-ii: 복원항과 액션이 리미터 예산을 **나눠 써야** A-i 가 없앤 포화가 안 돌아온다.

    안 나누면 pull 과 step 이 같은 방향일 때 합이 리미터를 넘어 다시 잘리기 시작한다.
    """
    gain = _fn_block(_ENV, "_setup_palm_step_gain")
    assert "palm_cmd_leak_reserve" in gain
    assert "(1.0 - _res) * _STEP_GAIN_MARGIN / math.sqrt(3.0)" in gain, "액션 게인이 남은 예산 기준이 아니다"
    assert "_palm_pull_cap" in gain

    arm = _fn_block(_ENV, "_arm_command")
    for token in ("_pull = self._palm_pull * (self._palm_anchor() - _prev6)",
                  "_clamp_norm(_pull[:, :3]", "_clamp_norm(_pull[:, 3:6]",
                  "_prev6 + _pull + step"):
        assert token in arm, token
    # 방향 보존 — 축별 클램프는 대각 지령의 방향을 왜곡한다(09.07 진단 이력).
    assert "norm(dim=-1, keepdim=True)" in _fn_block(_ENV, "_clamp_norm")


def test_pull_and_step_budget_cannot_exceed_the_limiter():
    """수치로도 확인 — 최악(대각·최대 이탈)에서 pull+step 노름이 리미터 이하인가."""
    import math
    lim, res, margin = 0.02, 0.2, 0.999
    k = (1.0 - res) * margin / math.sqrt(3.0)
    worst_step = (lim * k) * math.sqrt(3.0)      # a=(±1,±1,±1)
    worst_pull = lim * res                        # 노름 상한
    assert worst_step + worst_pull <= lim + 1e-12, (worst_step, worst_pull, lim)


def test_reset_seeds_the_increment_integrator_from_the_anchor():
    """A-i 의 유일한 방어선 — 안 하면 `a=0` 이 홈 유지가 되어 Track B 의 긴 무보상 이동을 물려받는다."""
    seed = _fn_block(_ENV, "_seed_palm_integrator")
    for token in ("self._palm_anchor()[env_ids]", "self.palm_targets[env_ids] = _anc",
                  "self._prev_palm_cmd[env_ids] = _anc[:, :3]",
                  "self._prev_palm_cmd_rot[env_ids] = _anc[:, 3:6]",
                  "self._palm_cmd_primed[env_ids] = True"):
        assert token in seed, token
    # 부모가 홈으로 되돌린 **뒤에** 씌워야 한다.
    _ordered(_fn_block(_ENV, "_reset_idx"),
             ["super()._reset_idx(env_ids)", "self._seed_palm_integrator(env_ids)"])
    # 부팅에서도 씨딩해야 첫 `_arm_command` 의 이전 지령이 0(→박스 구석)이 되지 않는다.
    _ordered(_fn_block(_ENV, "_init_task_state"),
             ["self._setup_palm_step_gain()", "self._seed_palm_integrator(slice(None))"])


def test_post_command_syncs_fabric_hand_and_integrates_once():
    code = _code(_ENV)
    assert "_syn_to_fab(self._syn_target)" in _fn_block(_ENV, "_post_command")
    assert code.count("self._step_fabric()") == 1
    assert "integrator.step(" not in code, "적분은 부모 `_step_fabric` 한 곳"


def test_env_overrides_exactly_the_design_hook_set():
    """DESIGN §8 훅만 덮어쓴다 — `_apply_action`/`_step_fabric` 등을 덮으면 Track A 가 아니다."""
    names = _class_methods(_ENV, "GraspKPEnv")
    required = {"_setup_scene", "_init_task_state", "_pre_physics_step", "_arm_command",
                "_hand_command", "_post_command", "_get_observations", "_get_rewards",
                "_get_dones", "_reset_idx"}
    forbidden = {"__init__", "_apply_action", "_step_fabric", "_setup_fabrics", "_init_home_palm",
                 "_synergy_targets", "_setup_synergy", "_palm_anchor", "_apply_gravity_compensation"}
    assert required <= names, required - names
    assert not (forbidden & names), forbidden & names


# ---------------------------------------------------------------- 외란·지연
def test_wrench_applied_every_step_in_world_frame_on_lifted():
    block = _fn_block(_ENV, "_apply_wrench")
    assert "self._wrench.step(self._obj_mass, self._latched)" in block
    assert "set_external_force_and_torque(forces, torques, is_global=True)" in block
    assert "WrenchDR(" in _fn_block(_ENV, "_init_task_state")


def test_three_delay_queues_obs_action_object():
    init = _fn_block(_ENV, "_init_task_state")
    assert init.count("DelayQueue(") == 3
    assert "_OBJ_POSE_DIM" in init and "_OBJ_POSE_DIM = 7" in _code(_ENV)
    obs = _fn_block(_ENV, "_object_blocks")
    assert "_obj_delay.push(" in obs and "noisy_pose(" in obs
    assert "_obs_delay.push(torch.nan_to_num(_noisy), flush)" in _fn_block(_ENV, "_get_observations")


# ---------------------------------------------------------------- 종료·리셋·박스
def test_dones_add_floor_termination_and_goal_truncation():
    block = _fn_block(_ENV, "_get_dones")
    assert "super()._get_dones()" in block
    assert "hand_floor_terminate_depth" in block and "goal_max" in block
    assert "self._trk.successes >= int(self.cfg.goal_max)" in block


def test_reset_order_super_goal_trackers_queues_wrench_latch():
    _ordered(_fn_block(_ENV, "_reset_idx"), [
        "super()._reset_idx(env_ids)", "sample_first_goal(", "_trk.full_reset(env_ids)",
        "_obs_delay.reset(env_ids)", "_act_delay.reset(env_ids)", "_obj_delay.reset(env_ids)",
        "_wrench.reset(env_ids)", "self._latched[env_ids] = False",
    ])


def test_palm_floor_override_raises_both_lower_bounds():
    """09.06 실측 "a=0 에서 손이 상판 49 mm 관통" — 지령 박스와 최종 클램프 둘 다 올려야 한다."""
    block = _fn_block(_ENV, "_apply_palm_floor_override")
    assert "palm_box_min_z_override" in block
    assert "self._palm_lo[2] = max(" in block and "self._box_lo[2] = max(" in block
    assert "palm_box_min_z_override: float = 0.27" in _code(_CFG)


def test_goal_box_derived_from_profile_spawn_center_in_finalize():
    fin = _fn_block(_CFG, "finalize_after_overrides")
    assert "super().finalize_after_overrides()" in fin and "_derive_goal_box(" in fin
    box = _fn_block(_CFG, "_derive_goal_box")
    for token in ("object_spawn_center", "goal_box_xy_halfwidth", "goal_box_z_range",
                  "object_origin_offset_z", "table_surface_z"):
        assert token in box, token


def test_cfg_defaults_match_design_and_reward_audit():
    code = _code(_CFG)
    for token in ("goal_first_z_range: tuple[float, float] = (0.16, 0.24)",
                  "goal_box_z_range: tuple[float, float] = (0.10, 0.30)",
                  "tol_start: float = 0.06", "tol_floor: float = 0.015",
                  "rw_lift_latch_height: float = 0.10", "rw_hand_floor_z: float = 0.215",
                  "obs_delay_steps: int = 3", "action_delay_steps: int = 3",
                  "object_delay_steps: int = 10", "goal_max: int = 50",
                  "keypoint_scale: float = 1.5", "keypoint_fixed_height: float = 0.12"):
        assert token in code, token


# ---------------------------------------------------------------- 리프트 후 안정 파지 (09.07 A-v)
def test_cmd_rate_is_normalised_raw_command_step_over_both_limiters():
    """벌하는 양은 리미터 **전** 원지령 변화(리미터 상한으로 정규화, 위치·회전 평균).

    리미터 **후** 값은 포화 구간(a2/a6 rate_sat 0.96)에서 상수라 μ 에 기울기가 없다 —
    그걸 벌하면 정책은 이미 리미터 안에 들어온 뒤에야 신호를 받는다. 상한(clamp)도 두지 않는다.
    리셋 첫 스텝(primed False)은 0 — 리셋 점프를 벌점으로 세지 않는다.
    """
    block = _fn_block(_ENV, "_arm_command")
    _ordered(block, [
        "_step3 = self.palm_targets[:, :3] - self._prev_palm_cmd",
        "_dr = self.palm_targets[:, 3:6] - self._prev_palm_cmd_rot",
        "self._palm_cmd_step_raw_rot = torch.where(",
        "self._cmd_rate = torch.where(",
        "0.5 * (self._palm_cmd_step_raw / max(_lim, 1e-9)",
        "+ self._palm_cmd_step_raw_rot / max(_lim_r, 1e-9)",
    ])
    assert "0.5 * (" in block, "위치·회전 두 채널의 평균이어야 한다(합이면 작동점이 2배로 커진다)"
    _seg = block.split("self._cmd_rate = torch.where(")[1].split("torch.zeros_like(self._cmd_rate))")[0]
    assert ".clamp(" not in _seg, "측도에 상한을 두면 작동점(≈10×)에서 항이 상수가 된다"


def test_reward_consumes_cmd_rate_and_cfg_locks_gate_and_scale():
    assert "cmd_rate=self._cmd_rate" in _fn_block(_ENV, "_get_rewards")
    code = _code(_CFG)
    assert "rw_cmd_rate_scale: float = 0.1" in code
    assert "cmd_rate_scale=float(self.rw_cmd_rate_scale)" in _fn_block(_CFG, "progress_reward_cfg")
    # ★성공은 **연속** 10 스텝 — 누적이면 공차 안팎을 오가며(흔들리며) 성공을 세어 준다.
    #   SimToolReal 논문 런처도 forceConsecutiveNearGoalSteps=True 로 강제했다.
    assert "goal_force_consecutive: bool = True" in code


def test_cfg_refuses_cmd_rate_penalty_without_both_limiters():
    """정규화 분모가 리미터다 — 리미터가 0(꺼짐)이면 벌점이 정의되지 않으니 cfg 에서 죽인다."""
    block = _fn_block(_CFG, "_validate_kp_fields")
    for token in ("rw_cmd_rate_scale", "palm_cmd_rate_limit_m", "palm_cmd_rate_limit_rot_deg"):
        assert token in block, token


def test_post_lift_stillness_is_logged_lifted_only():
    """리프트 후 정지 여부는 전체 평균이 아니라 lifted env 만 따로 봐야 한다(스텝 평균은 뭉갠다)."""
    assert '"task/cmd_rate_lifted"' in _fn_block(_ENV, "_log_step")
    probe = _fn_block(_ENV, "_log_probe_metrics")
    assert '"diag/arm_qd_p99_lifted"' in probe and '"diag/obj_speed_lifted"' in probe
    helper = _fn_block(_ENV, "_lifted_mean")
    assert "self._latched" in helper and ".clamp(min=1.0)" in helper, "빈 마스크에서 nan 이 나오면 안 된다"
    assert ".nonzero(" not in probe and "bool(" not in probe, "per-step GPU 동기화 금지(util killer)"


# ---------------------------------------------------------------- PPO yaml (DESIGN §6)
def test_lstm_yaml_bootstrap_gamma_and_architecture():
    assert "value_bootstrap: True" in _LSTM and "value_bootstrap: False" not in _LSTM
    assert "gamma: 0.99\n" in _LSTM and "gamma: 0.998" not in _LSTM
    assert _LSTM.count("units: [1024, 1024, 512, 512]") == 2, "actor·critic mlp 4층"
    assert "learning_rate: 1e-4" in _LSTM
    assert _LSTM.count("kl_threshold: 0.016") == 2 and "kl_threshold: 0.013" not in _LSTM
    assert "name: agn_grasp_kp-lstm" in _LSTM
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


def test_mlp_yaml_bootstrap_and_gamma():
    assert "value_bootstrap: True" in _MLP and "gamma: 0.99\n" in _MLP
    assert "name: agn_grasp_kp\n" in _MLP
