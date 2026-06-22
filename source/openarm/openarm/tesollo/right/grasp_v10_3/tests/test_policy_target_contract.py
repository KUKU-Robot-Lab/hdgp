from pathlib import Path

import torch

from openarm.tesollo.right.grasp_v10_3.finger_action_utils import (
    compute_preset_residual_finger_targets,
)
from openarm.tesollo.right.grasp_v10_3 import grasp_right_constants as constants
from openarm.tesollo.right.grasp_v10_3.grasp_right_preset import HAND_GRASP_POSE


_ROOT = Path(__file__).resolve().parents[1]


def test_action_contract_is_restored_27d_absolute_target() -> None:
    assert constants.NUM_ACTIONS == 27
    assert constants.PALM_POS_ACTION_SLICE == slice(0, 3)
    assert getattr(constants, "PALM_QUAT_ACTION_SLICE", None) == slice(3, 7)
    assert constants.FINGER_ACTION_SLICE == slice(7, 27)


def test_finger_action_is_small_residual_around_preset() -> None:
    preset = torch.tensor([0.0, 1.0, 3.0])
    lower = torch.tensor([-1.0, 0.0, 2.0])
    upper = torch.tensor([1.0, 2.0, 4.0])
    action = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    target = compute_preset_residual_finger_targets(
        preset,
        action,
        lower,
        upper,
        residual_scale=0.2,
    )
    assert torch.all(target >= lower.unsqueeze(0))
    assert torch.all(target <= upper.unsqueeze(0))
    assert torch.allclose(target[0], preset)
    assert torch.allclose(target[1], torch.tensor([0.2, 0.8, 3.1]))


def test_hand_residual_mask_keeps_fixed_joints_at_preset() -> None:
    preset = torch.tensor(HAND_GRASP_POSE, dtype=torch.float32)
    lower = torch.full_like(preset, -4.0)
    upper = torch.full_like(preset, 4.0)
    action = torch.ones(2, preset.numel(), dtype=torch.float32)
    action[1] = -1.0
    mask = torch.ones_like(preset)
    mask[[0, 4, 8, 12, 16, 17]] = 0.0

    target = compute_preset_residual_finger_targets(
        preset,
        action,
        lower,
        upper,
        residual_scale=0.15,
        residual_mask=mask,
    )

    fixed = torch.tensor([0, 4, 8, 12, 16, 17])
    assert torch.allclose(target[:, fixed], preset[fixed].unsqueeze(0).expand(2, -1))
    assert torch.allclose(target[0, mask.bool()], preset[mask.bool()] + 0.15)
    assert torch.allclose(target[1, mask.bool()], preset[mask.bool()] - 0.15)


def test_live_fabrics_uses_anchored_close_grasp_and_lift_targets() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")

    assert '"quaternion"' in env
    assert "self.palm_pose_targets = torch.zeros(self.num_envs, 7" in env
    assert "compute_lift_stabilize_palm_targets" not in env
    assert "LIFT_START_STEP" not in env
    assert "self.grasp_anchor_palm_pose_buf = torch.zeros(self.num_envs, 7" in env
    assert "self.lift_palm_start_pose_buf = torch.zeros(self.num_envs, 7" in env
    assert "self.lift_finger_pos_buf = torch.zeros(self.num_envs, NUM_HAND_DOF" in env
    assert "self.is_close_grasp_phase" in env
    assert "compose_incremental_palm_pose(" not in env
    assert "palm_quat_action = self.actions[:, PALM_QUAT_ACTION_SLICE]" in env
    assert "compute_preset_residual_finger_targets" in env
    assert "compute_lift_finger_targets" in env
    assert "approach_min_steps: int = 10" in cfg
    assert "approach_palm_local_z_min: float = -0.02" in cfg
    assert "approach_palm_local_z_max: float = 0.08" in cfg
    assert "grasp_palm_delta_scale: float = 0.25" in cfg
    assert "palm_local_workspace_radius: float = 0.1" in cfg
    assert "palm_target_max_delta: float = 0.01" in cfg
    assert "lift_palm_delta_xyz: float = 0.03" in cfg
    assert "transport" not in env


def test_palm_action_is_absolute_target_with_per_step_rate_limit() -> None:
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    assert "palm_local_workspace_radius: float = 0.1" in cfg
    assert "palm_target_max_delta: float = 0.01" in cfg
    assert "palm_pos_action = self.actions[:, PALM_POS_ACTION_SLICE]" in env
    assert "palm_quat_action = self.actions[:, PALM_QUAT_ACTION_SLICE]" in env
    assert "approach_pos_raw = (" in env
    assert "lift_pos_raw = (" in env


def test_palm_contact_sensor_uses_palm_link_net_force_without_actor_obs_growth() -> None:
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    constants_text = (_ROOT / "grasp_right_constants.py").read_text(encoding="utf-8")

    assert "palm_sensor_cfg" in cfg
    palm_cfg = cfg.split("palm_sensor_cfg", 1)[1]
    assert 'prim_path="/World/envs/env_.*/Robot/rl_dg_palm"' in palm_cfg
    assert "filter_prim_paths_expr" not in palm_cfg.split("history_length=1", 1)[0]
    assert "self._palm_sensor.data.net_forces_w[:, 0, :]" in env
    assert "quat_apply_inverse" in env
    assert "palm_force = torch.relu(-palm_force_local[:, 0])" in env
    assert "NUM_OBSERVATIONS = 134" in constants_text
    assert constants.NUM_OBSERVATIONS_NO_MASS == 133
    assert constants.NUM_OBSERVATIONS == 134
    assert constants.NUM_CRITIC_EXTRAS == 37
    assert constants.NUM_CRITIC_OBSERVATIONS == 170


def test_reward_gate_success_contract_uses_tip5_body_band_and_stationary_stability() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")

    assert "stage0_lift_start_min_contacts: int = 5" in cfg
    assert "grasp_body_local_z_min: float = -0.04" in cfg
    assert "grasp_body_local_z_max: float = 0.05" in cfg
    assert "prelift_max_cup_height_delta: float = 0.01" in cfg
    assert "prelift_cup_lin_vel_threshold: float = 0.04" in cfg
    assert "compute_tesollo_prelift_lift_readiness(" in env
    latch_block = env.split("compute_tesollo_prelift_lift_readiness(", 1)[1].split("if just_latched.any():", 1)[0]
    assert "tip_local_z_mean=tip_local_z_mean" in latch_block
    assert "cup_height_delta=cup_height_delta" in latch_block
    assert "cup_lin_vel_norm=prelift_cup_lin_vel" in latch_block
    assert "is_close_grasp_phase=close_grasp_mask" in latch_block

    assert "compute_grasp_reward_terms(" in env
    assert "compute_grasp_v2_stability(" in env
    assert "compute_stationary_grasp_success(" in env
    assert "full_tip_contact=full_tip_contact" in env
    assert "lift_success_candidate = in_or_past_lift & lifted & full_tip_contact & upright_success" in env
    assert "self._lift_success_hold_count = torch.where(" in env
    assert "self.is_stabilize_phase.copy_(self._lift_success_latched_buf)" in env
    assert "transport" not in env
    assert "previous_success_hold_count=self._success_hold_count" in env
    assert "finger_depth =" not in env
    assert "middle_contact_ready" not in env
    assert 'self.extras["contact/middle_count"]' not in env


def test_final_success_is_gated_by_stationary_stabilize() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    done_block = env.split("stationary_success = compute_stationary_grasp_success(", 1)[1]
    assert "stabilize_started=self.is_stabilize_phase" in done_block
    assert "previous_success_hold_count=self._success_hold_count" in done_block


def test_tesollo_debug_logs_are_namespaced_and_cover_rim_hook_diagnostics() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    assert 'self.extras["debug/tesollo/task/palm_local_z"]' in env
    assert 'self.extras["debug/tesollo/task/tip_local_z_mean"]' in env
    assert 'self.extras["debug/tesollo/task/rim_contact_proxy"]' in env
    assert 'self.extras["debug/tesollo/task/prelift_cup_height_delta"]' in env
    assert 'self.extras["debug/tesollo/task/prelift_cup_lin_vel"]' in env
    assert 'self.extras["debug/rh56f1/' not in env


def test_incremental_fabrics_logs_command_and_tracking_diagnostics() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    assert '"debug/tesollo/control/raw_palm_action_norm"' in env
    assert '"debug/tesollo/control/ema_palm_action_norm"' in env
    assert '"debug/tesollo/control/palm_target_position_error"' in env


def test_state_latched_fast_episode_and_default_training_no_actor_mass() -> None:
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")
    constants = (_ROOT / "grasp_right_constants.py").read_text(encoding="utf-8")
    task_cfg = (_ROOT / "config" / "__init__.py").read_text(encoding="utf-8")

    assert "episode_length_s: float = 10.0" in cfg
    assert "success_hold_steps: int = 30" in cfg
    assert "grasp_ready_hold_steps: int = 20" in cfg
    assert "full_grip_hold_steps: int = 30" in cfg
    assert "transport" not in cfg
    assert "EPISODE_STEPS           = 600" in constants
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass"' in task_cfg


def test_critic_mass_is_privileged_not_actor_clean_base() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    clean_block = env.split("actor_obs_clean = torch.cat([", 1)[1].split("], dim=-1)   # 133D", 1)[0]
    critic_block = env.split("critic_obs = torch.cat([", 1)[1].split("], dim=-1)   # 170D", 1)[0]

    assert "self._bead_mass_normalized" not in clean_block
    assert "self._bead_mass_normalized.unsqueeze(-1),  # 1" in critic_block
    assert "actor_obs_clean,        # 133" in critic_block


def test_phase_step_ratio_is_observation_only_not_reward_or_latch_gate() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    assert "phase_step_ratio = (" in env
    assert "phase_step_ratio,       # 1" in env
    reward_block = env.split("def _get_rewards", 1)[1].split("def _get_dones", 1)[0]
    latch_block = reward_block.split("compute_tesollo_prelift_lift_readiness(", 1)[1].split("if just_latched.any():", 1)[0]

    assert "phase_step_ratio" not in reward_block
    assert "time_ratio" not in reward_block
    assert "early_gate" not in reward_block
    assert "episode_length_buf" not in latch_block
    assert "EPISODE_STEPS" not in reward_block
    assert "GRASP_PHASE_STEPS" not in env


def test_state_based_reward_gates_and_upright_quality() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")

    assert "stabilize_upright_reward_scale_deg: float = 10.0" in cfg
    assert "stability_reward_weight: float = 1.0" in cfg
    assert "stabilize_spawn_xy_scale: float = 0.03" in cfg
    assert "upright_quality = torch.exp(" in env
    assert "compute_grasp_reward_terms(" in env
    assert "log_grasp_v2_common_scalars(" in env
    assert 'reward_terms["grasp"]' in env
    assert 'reward_terms["lift"]' in env
    assert 'reward_terms["stabilize"]' in env
    assert 'reward_terms["stability"]' in env
    assert 'reward_terms["success_bonus"]' in env
    assert "transport" not in env
    assert '"task/stable_rate"' in env
    assert "r_time_penalty" not in env
    assert "time_penalty_weight" not in cfg
    assert '"phase/approach"' in env
    assert '"phase/grasp"' in env


def test_approach_reward_uses_cup_origin_not_z_offset_target() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")
    reward_block = env.split("def _get_rewards", 1)[1].split("num_tip_contacts =", 1)[0]

    assert "grasp_center = self.object_pos" in reward_block
    assert "cup_grasp_z_offset" not in reward_block
    assert "cup_grasp_z_offset" not in cfg
