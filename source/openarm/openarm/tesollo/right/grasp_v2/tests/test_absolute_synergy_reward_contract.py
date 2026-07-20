from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


def _load_finger_action_utils():
    path = ROOT / "finger_action_utils.py"
    spec = importlib.util.spec_from_file_location("grasp_v2_finger_action_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(filename: str, name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synergy_action_couples_all_five_fingers() -> None:
    # 시너지(eigengrasp) 계약: PC1(a₁) 하나로 다섯 손가락 curl 진행도가 모두
    # 상승 — per-finger 독립 제어의 "2지 최소해"가 action 공간에서 표현 불가.
    utils = _load_finger_action_utils()
    syn = _load_module("tesollo_hand_synergy.py", "grasp_v2_tesollo_hand_synergy")

    basis = torch.tensor(syn.HAND_SYNERGY_BASIS, dtype=torch.float32)
    anchor = torch.tensor(syn.HAND_SYNERGY_ANCHOR, dtype=torch.float32)
    mins = torch.tensor(syn.HAND_SYNERGY_COEFF_MINS, dtype=torch.float32)
    maxs = torch.tensor(syn.HAND_SYNERGY_COEFF_MAXS, dtype=torch.float32)
    # preset 의 open(APPROACH)/FULL_GRIP — 좌우 미러 수치라 리터럴 하드코드 금지
    preset = _load_module("grasp_right_preset.py", "grasp_v2_preset_for_synergy_test")
    open_pose = anchor.clone()
    grip = torch.tensor(preset.HAND_FULL_GRIP_POSE, dtype=torch.float32)

    # a₁ = +1 (전지 감김 축 최대), 나머지 최소
    action = torch.tensor([[1.0, -1.0, -1.0, -1.0, -1.0]])
    p = utils.compute_synergy_progress_targets(
        action, basis, anchor, mins, maxs, open_pose, grip,
    )
    # 각 손가락 대표 curl 관절 진행도 (finger-major [thumb,index,middle,ring,pinky]×4)
    reps = {"thumb_4": 3, "index_2": 5, "middle_2": 9, "ring_2": 13, "pinky_3": 18}
    for name, idx in reps.items():
        assert p[0, idx] > 0.25, f"{name} 진행도 {p[0, idx]:.3f} — 시너지 커플링 실패"

    # 열림 보장 (lstm_test7 in_success 0 버그의 계약): action=-1 → 완전 열림,
    # action=+1 → 5지 전부 감김. 계수 범위가 축별 "열림(0)↔감김(c_grip)" 정렬일 것.
    p_open = utils.compute_synergy_progress_targets(
        -torch.ones(1, 5), basis, anchor, mins, maxs, open_pose, grip,
    )
    assert p_open.max() < 0.05, f"열림 실패: max 진행도 {p_open.max():.3f}"
    p_close = utils.compute_synergy_progress_targets(
        torch.ones(1, 5), basis, anchor, mins, maxs, open_pose, grip,
    )
    for name, idx in reps.items():
        assert p_close[0, idx] > 0.8, f"{name} 감김 실패 {p_close[0, idx]:.3f}"

    # 직교정규 basis 검증
    eye_err = (basis @ basis.T - torch.eye(5)).abs().max()
    assert eye_err < 1e-4


def test_finger_close_is_contact_gated_adaptive() -> None:
    # ① 접촉-게이트 적응 폐쇄: 손가락이 중간마디(_3) 접촉까지 점진 폐쇄 후 동결.
    # 고정 포즈 lerp(compute_lift_finger_targets) 제어를 대체.
    env = _text("grasp_right_env.py")
    preset = _text("grasp_right_preset.py")

    assert "HAND_FULL_GRIP_POSE" in preset
    assert "finger_close_buf" in env
    assert "middle_binary_contact_buf" in env
    assert "self.hand_full_grip_pose" in env
    assert "finger_close_speed" in env
    # 고정 lerp 제어 제거 확인
    assert "compute_lift_finger_targets(" not in env
    assert "self.lift_finger_pos_buf" not in env


def test_finger_close_is_per_joint_contact_gated() -> None:
    # 관절별 적응 폐쇄: PIP@middle, DIP@distal|tip 독립 동결, MCP 무게이트 full close.
    # 손가락당 1-DOF(repeat_interleave 후 lerp) 제어를 대체.
    env = _text("grasp_right_env.py")

    # 관절별 (N,20) 버퍼
    assert "self.finger_close_buf = torch.zeros(self.num_envs, NUM_HAND_DOF" in env
    # 3개 접촉 밴드 모두 게이트 입력으로 사용
    assert "self.binary_contact_buf.float()" in env
    assert "self.distal_binary_contact_buf.float()" in env
    assert "self.middle_binary_contact_buf.float()" in env
    # 관절별 게이트 스택 → (N,20)
    assert "gate20 = torch.stack" in env
    # 시너지(eigengrasp) 경로: per-finger 독립 명령(cmd.repeat_interleave) 제거,
    # PCA 계수 → 관절별 진행도 목표를 rate-limit 추종
    assert "compute_synergy_progress_targets(" in env
    assert "cmd.repeat_interleave(4" not in env
    # 래칫(전진만): 양방향 추종은 h2o max-거리 보상과 결합해 "손 펴기" 국소최적
    # (test8: f2~f4 -0.95 수렴, in_success 0). clamp 하한 0.0 필수.
    assert ".clamp(0.0, _step)" in env
    assert ".clamp(-_step, _step)" not in env
    # 1-DOF lerp(close_buf를 repeat_interleave 후 lerp) 제거 확인
    assert "self.finger_close_buf.repeat_interleave(4" not in env


def test_lift_latch_gate_enabled_via_compute_lift_readiness() -> None:
    # [08-21] grasp_v1 staged reward 이식: 접촉 latch(compute_lift_readiness)가
    # step 스크립트(LIFT_START_STEP) 대신 리프트/안정화 보상 게이트를 연다.
    # 인벨롭 hard 게이트(lift_start_min_envelope_fingers)는 여전히 0(비활성) —
    # envelope 은 grasp reward credit(grasp_envelope_credit, soft gradient)으로 유도.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    assert "lift_start_min_envelope_fingers: int = 0" in cfg
    assert "grasp_envelope_credit: float = 0.5" in cfg
    assert "compute_lift_readiness(" in reward_body
    assert "self.lift_latched_buf" in reward_body


def test_reward_is_grasp_v1_staged() -> None:
    # [08-21] grasp_v1 staged reward(compute_grasp_reward_terms) 이식 — DEXTRAH
    # 4항(거리 기반 carry-to-goal)·force_closure 는 걷어냈다.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "approach_weight", "grasp_weight", "lift_reward_weight",
        "stabilize_weight", "success_bonus_weight",
        "post_lift_contact_loss_weight", "stability_reward_weight",
        "grasp_envelope_credit", "enclosure_thumb_weight",
    ):
        assert name in cfg

    for term in (
        "compute_grasp_reward_terms(",
        "envelope_frac", "grip_frac", "fingertip_side_dist",
        "cup_height_delta", "cup_xy_displacement", "cup_tilt_deg",
        "success_now",
    ):
        assert term in reward_body

    # DEXTRAH 4항/force_closure 구조 미사용 확인
    for removed in (
        "hand_to_object_reward", "object_to_goal_reward",
        "object_to_goal_err", "force_closure_reward", "fc_gate",
    ):
        assert removed not in reward_body, removed
    for removed_cfg in (
        "hand_to_object_weight", "object_to_goal_weight", "lift_weight",
        "force_closure_weight", "grasp_contact_weight", "object_goal_tol",
    ):
        assert removed_cfg not in cfg, removed_cfg


def test_goal_is_fixed_dextrah() -> None:
    # DEXTRAH 정렬 계약: goal = cfg 고정 절대점 (settle/reset 갱신 없음).
    # object_init_pos 스냅샷은 object_height 로깅 baseline 으로만 유지.
    env = _text("grasp_right_env.py")

    assert "self.episode_length_buf == int(self.cfg.settle_steps)" in env
    assert "self.object_init_pos[snap] = self.object_pos[snap]" in env
    assert "self.object_goal[snap" not in env
    assert "cfg.object_goal_pos" in env


def test_obs_is_dextrah_teacher_structure() -> None:
    # [08-21] Tesollo-native obs 계약 (일반화 개편 — 물체 identity/scale/rotation 은
    # critic 만 privileged 로 받고, actor(policy) 는 pos 만 본다):
    # policy = dof pos/vel noisy + hand pos/vel noisy(fabric FK) + object pos noisy
    #          + goal + actions + fabric q/qd/qdd  (208, 물체 수 무관)
    # critic = clean + hand_forces + measured torque + object_vel + onehot + scale (277 + N_obj)
    env = _text("grasp_right_env.py")
    obs_body = env.split("def _get_observations", 1)[1].split("def _get_rewards", 1)[0]
    actor_body, critic_body = obs_body.split("==== critic obs", 1)

    for term in (
        "self.robot_dof_pos_noisy",
        "self.robot_dof_vel_noisy",
        "self.hand_pos_noisy",
        "self.hand_vel_noisy",
        "self.object_pos_noisy",
        "self.object_goal",
        "self.fabric_q",
        "self.fabric_qd",
        "self.fabric_qdd",
    ):
        assert term in actor_body, term
    # 물체 identity/scale/rotation 은 actor 에서 빠져야 한다 — 미학습 신규 물체 일반화.
    for term in ("self.object_rot_noisy", "self.multi_object_idx_onehot", "self.object_scale"):
        assert term not in actor_body, f"actor obs 에 privileged 항목 '{term}' 이 남아있다"

    for term in (
        "self.multi_object_idx_onehot",
        "self.object_scale",
        "self.object_rot",
        "get_link_incoming_joint_force",
        "get_dof_projected_joint_forces",
    ):
        assert term in critic_body, term
    # 구 106D 구성 제거 확인
    assert "cup_to_fingertip" not in obs_body
    assert "object_feature" not in obs_body

    cfg = _text("grasp_right_env_cfg.py")
    assert "observation_space: int = NUM_OBS_BASE" in cfg
    assert "NUM_CRITIC_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)" in cfg


def test_adr_curriculum_is_dextrah() -> None:
    # DEXTRAH ADR 커리큘럼 계약: wrench 0→10, spawn 0→최대·회전, 관측 노이즈 점진,
    # in_success 트리거. [08-21] "reward_weights" ADR 그룹은 DEXTRAH 4항/
    # force_closure 전용이라 그 둘 제거와 함께 걷어냈다(grasp_v1 staged reward는
    # 고정 cfg 값 사용, ADR 안 함 — grasp_v1 자신도 그렇다).
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for group in (
        '"object_wrench"', '"object_spawn"', '"object_state_noise"',
        '"robot_state_noise"', '"fabric_damping"',
        '"observation_annealing"', '"robot_spawn"', '"pd_targets"',
    ):
        assert group in cfg, group
    assert '"reward_weights"' not in cfg
    assert "adr_trigger_threshold: float = 0.4" in cfg
    # 트리거 = in_success 순간 평균 (DEXTRAH success_for_adr) — 이제 grasp_v1식
    # in-place lift+hold success_now 를 본다(carry-to-goal 아님).
    assert "maybe_increment(self.in_success_region.float().mean())" in env
    # wrench/spawn 가 ADR 파라미터를 사용
    assert 'self._adr("robot_state_noise", "robot_joint_pos_noise")' in env
    assert '"object_spawn", "xy_range"' in env
    assert '"object_wrench", "max_linear_accel"' in env
    # DEXTRAH 완전 정렬 5건 (07.09):
    # (1) physics DR: EventCfg + adr_physics_cfg (mass 0.5~3× 등) ADR 확장
    assert "class EventCfg" in cfg
    assert "adr_physics_cfg" in cfg
    assert '"object_scale_mass"' in cfg
    # (2) wrench 게이트 = hand↔object 거리 (in_success 게이트 아님)
    assert "hand_to_object_dist_threshold" in cfg
    assert "self.hand_to_object_err <= float(self.cfg.hand_to_object_dist_threshold)" in env
    # (3) robot_spawn 리셋 노이즈 커리큘럼
    assert 'self._adr("robot_spawn", "joint_pos_noise")' in env
    # (4) 케이던스 = 5×에피소드(3000 steps, DEXTRAH min_steps_for_dr_change)
    assert "adr_increment_interval: int  = 3000" in cfg
    # (5) pd velocity feedforward factor (1→0)
    assert '"pd_targets", "velocity_target_factor"' in env


def test_palm_target_rate_limit() -> None:
    # palm 목표 rate limit 계약 (07.11): 정책의 목표 순간이동(bang-bang)이
    # 접근 밀침·리프트 후 스윙을 만듦 — 스텝당 변화량을 기구적으로 제한.
    #
    # palm action 은 절대 pose 다(DEXTRAH 원본 구조, 07.13). delta 방식은 물체까지
    # 20~30cm 를 rate limit 으로 수백 스텝 적분해야 해서 정책이 "가만히 있기"로
    # 수렴했다(curl_fix run: hand_to_object ep200 0.216 → ep400 0.017).
    # rate limit 은 절대 pose 에서도 그대로 유지된다 — 목표는 1스텝에 찍되
    # 실제 이동은 부드럽게.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "palm_rate_xyz_per_step:     float = 0.04" in cfg
    assert "palm_rate_rot_deg_per_step: float = 8.0" in cfg
    assert "scale(palm_action, self.palm_mins_env, self.palm_maxs_env)" in env, \
        "palm action 이 절대 pose 가 아님 (delta 로 되돌아갔다)"
    assert "palm_delta_xyz" not in cfg, "delta cfg 가 되살아났다"
    assert "self.palm_rate_limits" in env
    assert "(palm_pose - self.palm_pose_targets).clamp(" in env
    # settle 동안 팔 동결: 낙하 중 물체를 팔로 쳐내는 것 방지 (finger 게이트 공유)
    assert "in_settle, self.pregrasp_palm_pose_buf, palm_pose" in env


def test_single_phase_no_scripted_lift() -> None:
    # DEXTRAH 단일 phase 계약: scripted joint7 lift/latch/freeze 없이
    # 정책이 전 구간 Fabrics arm을 연속 제어한다. [08-21] compute_lift_readiness는
    # 이제 reward 게이트로 쓰이지만(별도 계약, test_lift_latch_gate_enabled_*),
    # arm 제어(_apply_action)에는 여전히 없다 — reward 판정과 arm phase 는 분리.
    env = _text("grasp_right_env.py")
    apply_body = env.split("def _apply_action", 1)[1].split("def ", 1)[0]

    assert "lift_progress" not in apply_body
    assert "arm_target = self.fabric_q[:, :NUM_ARM_DOF]" in apply_body
    assert "compute_lift_readiness" not in apply_body
    assert "is_lift_phase" not in env
