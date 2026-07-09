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


def test_five_finger_actions_drive_absolute_twenty_joint_synergy() -> None:
    module = _load_finger_action_utils()
    open_pose = torch.arange(20, dtype=torch.float32)
    closed_pose = open_pose + 4.0
    lower = torch.full((20,), -100.0)
    upper = torch.full((20,), 100.0)
    actions = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])

    target = module.compute_absolute_finger_targets(
        finger_action=actions,
        open_pose=open_pose,
        closed_pose=closed_pose,
        lower_limits=lower,
        upper_limits=upper,
    )

    expected_blend = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]]).repeat_interleave(4, dim=1)
    assert torch.allclose(target, open_pose.unsqueeze(0) + 4.0 * expected_blend)


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
    assert "cmd.repeat_interleave(4" in env
    # 1-DOF lerp(close_buf를 repeat_interleave 후 lerp) 제거 확인
    assert "self.finger_close_buf.repeat_interleave(4" not in env


def test_lift_latch_gate_disabled_envelope_via_reward() -> None:
    # 인벨롭 latch hard 게이트는 비활성(=0): success를 죽이면서 envelope은 못 만듦.
    # envelope은 grasp/lift 보상 credit(soft gradient)으로 유도한다.
    cfg = _text("grasp_right_env_cfg.py")
    utils = _text("grasp_right_utils.py")

    assert "lift_start_min_envelope_fingers: int = 0" in cfg
    # 게이트 배선은 보존(재활성 가능), compute_lift_readiness가 지원만
    assert "min_envelope_fingers" in utils


def test_reward_is_dextrah_four_terms() -> None:
    # DEXTRAH compute_rewards 이식 계약: 4항(거리 기반) + in_success_region.
    # 접촉 synergy 항(compute_grasp_reward_terms)은 사용하지 않는다.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "object_goal_pos",
        "object_goal_tol",
        "hand_to_object_weight",
        "hand_to_object_sharpness",
        "object_to_goal_weight",
        "object_to_goal_sharpness",
        "lift_weight",
        "lift_sharpness",
        "finger_curl_reg_weight",
    ):
        assert name in cfg

    for term in (
        "hand_to_object_reward",
        "object_to_goal_reward",
        "finger_curl_reg",
        "lift_reward",
        "in_success_region",
    ):
        assert term in reward_body

    # hand_to_object: palm+5손끝 MAX 거리 (OpenArm 포팅 규약)
    assert ".max(dim=-1).values" in reward_body
    # 구 접촉 synergy 코어 미사용
    assert "compute_grasp_reward_terms(" not in reward_body


def test_goal_is_fixed_dextrah() -> None:
    # DEXTRAH 정렬 계약: goal = cfg 고정 절대점 (settle/reset 갱신 없음).
    # object_init_pos 스냅샷은 object_height 로깅 baseline 으로만 유지.
    env = _text("grasp_right_env.py")

    assert "self.episode_length_buf == int(self.cfg.settle_steps)" in env
    assert "self.object_init_pos[snap] = self.object_pos[snap]" in env
    assert "self.object_goal[snap" not in env
    assert "cfg.object_goal_pos" in env


def test_obs_is_dextrah_teacher_structure() -> None:
    # DEXTRAH teacher obs 계약 (distillation 대비 원본 동일 구조):
    # policy = dof pos/vel noisy + hand pos/vel noisy(fabric FK) + object pose noisy
    #          + goal + onehot + scale + actions + fabric q/qd/qdd  (193 + N_obj)
    # critic = clean + hand_forces + measured torque + object_vel  (247 + N_obj)
    env = _text("grasp_right_env.py")
    obs_body = env.split("def _get_observations", 1)[1].split("def _get_rewards", 1)[0]

    for term in (
        "self.robot_dof_pos_noisy",
        "self.robot_dof_vel_noisy",
        "self.hand_pos_noisy",
        "self.hand_vel_noisy",
        "self.object_pos_noisy",
        "self.object_rot_noisy",
        "self.object_goal",
        "self.multi_object_idx_onehot",
        "self.object_scale",
        "self.fabric_q",
        "self.fabric_qd",
        "self.fabric_qdd",
        "get_link_incoming_joint_force",
        "get_dof_projected_joint_forces",
    ):
        assert term in obs_body, term
    # 구 106D 구성 제거 확인
    assert "cup_to_fingertip" not in obs_body
    assert "object_feature" not in obs_body

    cfg = _text("grasp_right_env_cfg.py")
    assert "NUM_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)" in cfg
    assert "NUM_CRITIC_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)" in cfg


def test_adr_curriculum_is_dextrah() -> None:
    # DEXTRAH ADR 커리큘럼 계약: wrench 0→10, spawn 0→최대·회전, reward 스케줄
    # (lift 5→0, sharpness 15→20, curl_reg), 관측 노이즈 점진, in_success 트리거.
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for group in (
        '"object_wrench"', '"object_spawn"', '"object_state_noise"',
        '"robot_state_noise"', '"reward_weights"', '"fabric_damping"',
        '"observation_annealing"',
    ):
        assert group in cfg, group
    assert '"lift_weight":              (5.0, 0.0)' in cfg
    assert "adr_trigger_threshold: float = 0.4" in cfg
    # 트리거 = in_success 순간 평균 (DEXTRAH success_for_adr)
    assert "maybe_increment(self.in_success_region.float().mean())" in env
    # wrench/spawn/reward 가 ADR 파라미터를 사용
    assert 'self._adr("robot_state_noise", "robot_joint_pos_noise")' in env
    assert '"object_spawn", "xy_range"' in env
    assert '"object_wrench", "max_linear_accel"' in env
    assert '"reward_weights", "lift_weight"' in env


def test_single_phase_no_scripted_lift() -> None:
    # DEXTRAH 단일 phase 계약: scripted joint7 lift/latch/freeze 없이
    # 정책이 전 구간 Fabrics arm을 연속 제어한다.
    env = _text("grasp_right_env.py")
    apply_body = env.split("def _apply_action", 1)[1].split("def ", 1)[0]

    assert "lift_progress" not in apply_body
    assert "arm_target = self.fabric_q[:, :NUM_ARM_DOF]" in apply_body
    assert "compute_lift_readiness" not in env
    assert "is_lift_phase" not in env
