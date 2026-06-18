from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def test_v11_registers_train_and_play_ids() -> None:
    t = (_ROOT / "config" / "__init__.py").read_text(encoding="utf-8")

    assert 'id="open-rh56f1_r_grasp_v1"' in t
    assert 'id="open-rh56f1_r_grasp_v1-lstm"' in t
    assert 'id="open-rh56f1_r_grasp_v1-play"' in t
    assert 'id="open-rh56f1_r_grasp_v1-play-lstm"' in t
    assert "openarm.rh56f1.right.grasp_v1.grasp_right_env:GraspRightEnv" in t
    assert ".grasp_right_env:GraspRightEnv" in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass_PLAY"' in t


def test_v11_keeps_actor_and_critic_observation_shapes() -> None:
    constants = _text("grasp_right_constants.py")
    cfg = _text("grasp_right_env_cfg.py")

    assert "NUM_OBSERVATIONS = 99" in constants
    assert "NUM_OBSERVATIONS_WITH_MASS = 100" in constants
    assert "NUM_OBSERVATIONS_NO_MASS = NUM_OBSERVATIONS" in constants
    assert "NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 117" in constants
    assert "observation_space: int = NUM_OBSERVATIONS" in cfg
    assert "state_space:       int = NUM_CRITIC_OBSERVATIONS" in cfg
    assert "observation_space: int = NUM_OBSERVATIONS_NO_MASS" in cfg
    assert "actor_observe_bead_mass: bool = False" in cfg


def test_v11_declares_four_phase_episode_and_transport_params() -> None:
    constants = _text("grasp_right_constants.py")
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for name in (
        "GRASP_PHASE_STEPS",
        "LIFT_PHASE_STEPS",
        "STABILIZE_PHASE_STEPS",
        "TRANSPORT_PHASE_STEPS",
        "TRANSPORT_START_STEP",
    ):
        assert name in constants
    assert "TRANSPORT_PHASE_STEPS = 120" in constants
    for name in (
        "transport_goal_dist_threshold",
        "transport_goal_x_range",
        "transport_goal_y_range",
        "transport_goal_z_range",
        "transport_success_hold_steps",
        "lift_target_z_delta",
        "enable_grasp_phase_full_grip_blend",
        "grasp_phase_full_grip_contact_threshold",
        "grasp_phase_full_grip_progress_threshold",
        "stage0_lift_start_min_contacts",
        "stage0_lift_start_hold_steps",
        "lift_contact_hold_steps",
        "full_grip_hold_steps",
        "lift_min_force_ratio",
        "enable_phase_curriculum",
        "phase_curriculum_initial_stage",
        "phase_curriculum_lift_success_threshold",
        "phase_curriculum_stabilize_success_threshold",
        "terminate_on_lift_failure",
    ):
        assert name in cfg
    assert "enable_demo_grasp_reset: bool = False" in cfg
    assert "compute_grasp_v2_success" in env
    assert "compute_grasp_v2_stability" in env
    assert "compute_slip_proxy" in env
    assert "transport_palm_target_pose_buf" in env


def test_v11_lift_uses_policy_delta_and_caps_demo_target_near_ten_cm() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "lift_target_z_delta: float = LIFT_Z_DELTA" in cfg
    assert "lift_elapsed_steps = torch.where(" in env
    assert "self._lift_started_buf," in env
    assert "self.episode_length_buf - self._lift_start_step_buf" in env
    assert "ema_action_alpha: float = 0.7" in cfg
    assert "lift_palm_delta_xyz: float = 0.03" in cfg
    assert "self._ema_palm_action = torch.zeros(self.num_envs, 6, device=self.device)" in env
    assert "self._ema_palm_action[env_ids] = 0.0" in env
    assert "fabric_palm_action = self._ema_palm_action" in env
    assert "lift_policy_delta = scale(fabric_palm_action, self.lift_delta_mins, self.lift_delta_maxs)" in env
    assert "lift_palm_pose = self.lift_palm_start_pose_buf + lift_policy_delta" in env
    assert "demo_lift_target[:, 2] = torch.minimum(" in env
    assert "pregrasp_palm_pose[:, 2] + float(self.cfg.lift_target_z_delta)" in env


def test_v11_actor_and_critic_observe_cup_to_goal() -> None:
    constants = _text("grasp_right_constants.py")
    env = _text("grasp_right_env.py")

    assert "cup_to_goal:              3" in constants
    assert "cup_to_goal = self.object_goal - cup_pos_noisy" in env
    assert "cup_to_goal_clean = self.object_goal - cup_pos_clean" in env
    assert "cup_to_goal," in env
    assert "cup_to_goal_clean," in env


def test_v11_actor_observes_cup_orientation_and_critic_observes_angular_velocity() -> None:
    constants = _text("grasp_right_constants.py")
    env = _text("grasp_right_env.py")

    assert "cup_ang_vel:" in constants
    assert "cup_rot (quat):           4" in constants
    assert "cup_ang_vel  = self.cup.data.root_ang_vel_w" in env
    assert "cup_rot      = self.object_rot" in env
    assert "cup_ang_vel," in env
    assert "cup_rot,                # 4" in env


def test_v11_samples_transport_goal_per_reset_env() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "transport_goal_x_range: tuple[float, float] = (0.25, 0.40)" in cfg
    assert "transport_goal_y_range: tuple[float, float] = (-0.20, 0.00)" in cfg
    assert "transport_goal_z_range: tuple[float, float] = (0.30, 0.55)" in cfg
    assert "def _sample_transport_goals" in env
    assert "self.object_goal[env_ids] = self._sample_transport_goals(n)" in env
    # 정책구동 transport: scripted lerp 제거, start 앵커 + action·radius reach 사용
    assert "transport_palm_pose = torch.lerp(" not in env
    assert "self.cfg.transport_palm_workspace_radius" in env
    assert "goal_delta[:, 2] = 0.0" not in env


def test_v11_disables_physical_beads_for_small_cup_grasp() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for name in (
        "physical_beads_enabled",
        "bead_count_min",
        "bead_count_max",
        "dynamic_bead_spawn_enabled",
        "dynamic_bead_spawn_step",
        "bead_initial_count_min",
        "bead_initial_count_max",
        "dynamic_bead_add_count_min",
        "dynamic_bead_add_count_max",
    ):
        assert name in cfg
    assert "bead_count_min: int = 0" in cfg
    assert "bead_count_max: int = 0" in cfg
    assert "physical_beads_enabled: bool = False" in cfg
    assert "dynamic_bead_spawn_enabled: bool = False" in cfg
    assert "bead_initial_count_max: int = 0" in cfg
    assert "if not self.cfg.physical_beads_enabled:" in env
    assert "bead_count = torch.zeros(n, dtype=torch.long, device=self.device)" in env
    assert "target_bead_count = torch.zeros(n, dtype=torch.long, device=self.device)" in env
    assert "if self.cfg.physical_beads_enabled and active.any():" in env
    assert "self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.num_beads" in env
    assert "self._bead_mass_normalized * self.cfg.num_beads * self.cfg.bead_single_mass" in env
    assert "if self.cfg.physical_beads_enabled and self.cfg.dynamic_bead_spawn_enabled:" in env
    assert "dynamic_bead_delay = max(int(self.cfg.dynamic_bead_spawn_step) - STABILIZE_START_STEP, 0)" in env
    assert "(stabilize_elapsed == dynamic_bead_delay)" in env
    assert "& self._lift_success_latched_buf" in env
    assert "self._spawn_dynamic_beads(dynamic_bead_mask)" in env
    assert "self._bead_count_current" in env
    assert "self._bead_count_target" in env
    assert "quat_apply(cup_quat_w[activate], local_offset)" in env
    assert '"stat_bead_count_initial"' not in env
    assert '"stat_bead_count_current"' not in env
    assert '"stat_dynamic_bead_added"' not in env
    assert "bin_" not in env
    assert 'self.extras["cup/friction"]' not in env


def test_v11_keeps_bead_asset_hidden_when_physical_beads_disabled() -> None:
    cfg = _text("grasp_right_env_cfg.py")

    assert "_DEFAULT_BEAD_MASS = 0.010" in cfg
    assert "_DEFAULT_BEAD_SCALE = 0.35" in cfg
    assert "scale=(_DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE)" in cfg
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=_DEFAULT_BEAD_MASS)" in cfg
    assert "linear_damping=0.5" in cfg
    assert "angular_damping=0.5" in cfg
    assert "static_friction=0.1" in cfg
    assert "dynamic_friction=0.08" in cfg
    assert "restitution=0.1" in cfg


def test_v11_actively_levels_cup_after_lift_with_movable_arm() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "post_lift_arm_stiffness_scale" not in cfg
    assert "post_lift_arm_damping_scale" not in cfg
    assert "_apply_post_lift_arm_compliance" not in env
    assert "stabilize_upright_orientation_enabled: bool = True" in cfg
    assert "stabilize_upright_orientation_gain: float = 1.5" in cfg
    assert "stabilize_upright_orientation_max_deg: float = 25.0" in cfg
    assert "stabilize_upright_orientation_blend_steps: int = STABILIZE_PHASE_STEPS // 2" in cfg
    assert "transport_xyz_hold_enabled: bool = True" in cfg
    assert "transport_xyz_hold_gain: float = 4.0" in cfg
    assert "transport_xyz_hold_max_delta: float = 0.12" in cfg
    assert "stabilize_spawn_xy_hold_enabled: bool = True" in cfg
    assert "def _apply_upright_palm_orientation_correction" in env
    assert "def _apply_transport_xyz_palm_correction" in env
    assert "def _transport_xyz_cfg" in env
    assert "xy_error = self.object_goal[:, :2] - self.object_pos[:, :2]" in env
    assert '"transport_xyz_hold_gain"' in env
    assert '"stabilize_spawn_xy_hold_gain"' in env
    assert "self._transport_xyz_palm_correction_buf[phase_mask] = xy_correction[phase_mask]" in env
    assert 'self.extras["debug/rh56f1/task/transport_xyz_palm_correction"]' in env
    assert "lift_palm_pose = self._apply_transport_xyz_palm_correction(" not in env
    # 정책구동 transport: scripted xy hold 보정 호출 제거 (메서드 정의는 보존)
    assert "transport_palm_pose = self._apply_transport_xyz_palm_correction(" not in env
    assert "cup_z_world = quat_apply(self.object_rot, z_local)" in env
    assert "cup_z_world[:, 1]" in env
    assert "-cup_z_world[:, 0]" in env
    assert "lift_palm_pose = self._apply_upright_palm_orientation_correction(" in env
    assert "transport_palm_pose = self._apply_upright_palm_orientation_correction(" in env
    assert "self.robot.set_joint_position_target(self.fabric_q[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices)" in env


def test_v11_declares_pour_warm_state_export_contract() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for name in (
        "enable_warm_state_export",
        "warm_state_export_path",
        "warm_state_target_count",
        "warm_state_success_source",
    ):
        assert name in cfg
    for name in (
        "bead_state_local",
        "bead_count_current",
        "object_goal_local",
        "meta/schema_version",
        "meta/bead_single_mass",
        "meta/bead_scale",
        "def _maybe_export_warm_states",
        "def _write_warm_state_export_file",
    ):
        assert name in env
    assert "bead_scale: float = _DEFAULT_BEAD_SCALE" in cfg
    assert 'attrs["meta/bead_scale"] = float(self.cfg.bead_scale)' in env


def test_v11_phase_curriculum_starts_lift_from_readiness_and_gates_late_phases() -> None:
    env = _text("grasp_right_env.py")

    assert "self._phase_curriculum_stage = min(max(int(cfg.phase_curriculum_initial_stage), 0), 2)" in env
    assert "def _maybe_update_phase_curriculum" in env
    assert "self._episode_curriculum_stage_buf >= 1" in env
    assert "self._episode_curriculum_stage_buf >= 2" in env
    assert "transport_disabled = self._episode_curriculum_stage_buf[env_ids] < 2" not in env
    assert "self.object_goal[env_ids_tensor[transport_disabled]] = obj_pos_local[transport_disabled]" not in env
    assert "just_entering_lift = self._lift_contact_ready_latched_buf & (~self._lift_started_buf)" in env
    assert "self._lift_start_step_buf[just_entering_lift] = self.episode_length_buf[just_entering_lift]" in env
    assert "just_entering_stabilize = (" in env
    stabilize_gate = env.split("just_entering_stabilize = (", 1)[1].split(
        "if just_entering_stabilize.any():", 1
    )[0]
    assert "& self._transport_arrived_latched_buf" in stabilize_gate
    assert "& self._lift_success_latched_buf" not in stabilize_gate
    assert "_full_grip_ready" not in stabilize_gate
    assert "just_entering_transport = (" in env
    transport_gate = env.split("just_entering_transport = (", 1)[1].split(
        "if just_entering_transport.any():", 1
    )[0]
    assert "& self._transport_ready_latched_buf" in transport_gate
    assert "& self._lift_success_latched_buf" not in transport_gate
    assert "& self._full_grip_ready_buf" not in env
    assert "goal_delta[:, 2] = 0.0" not in env
    assert "curriculum_lift_horizon" in env
    assert "curriculum_stabilize_horizon" in env
    assert "grasp_timeout_failed = (" in env
    assert "(self.episode_length_buf >= LIFT_START_STEP)" in env
    assert "& (~self._lift_contact_ready_latched_buf)" in env
    assert "lift_failed = (" in env
    assert '"task/lift_success_rate"' in env
    assert '"task/stabilize_success_rate"' in env
    assert '"task/lift_started_rate"' in env
    assert 'self.extras["debug/rh56f1/task/grasp_timeout_fail_rate"]' in env


def test_v11_transport_and_success_do_not_start_directly_from_lift_success() -> None:
    env = _text("grasp_right_env.py")
    cfg = _text("grasp_right_env_cfg.py")

    assert "lift_to_transport_hold_steps: int = 15" in cfg
    assert "transport_to_stabilize_hold_steps: int = 1" in cfg
    assert "self._transport_ready_hold_count = torch.zeros" in env
    assert "self._transport_arrived_hold_count = torch.zeros" in env
    assert "self._transport_ready_latched_buf |= transport_ready_now" in env
    assert "self._transport_arrived_latched_buf |= transport_arrived_now" in env
    assert "transport_control_mask = is_transport | is_stabilize" in env
    assert "transport_control_mask.unsqueeze(1)" in env
    assert "transport_started=in_stabilize" in env
    assert "transport_started=in_transport" not in env


def test_v11_grip_first_curriculum_uses_split_readiness_gates() -> None:
    env = _text("grasp_right_env.py")

    assert "lift_contact_phase = close_grasp_mask" in env
    assert "compute_grasp_phase_finger_targets(" in env
    assert "compute_late_grasp_full_grip_mask(" in env
    assert "full_grip_pose=self.hand_full_grip_pose" in env
    assert "late_grasp_mask=late_grasp_full_grip_mask" in env
    assert "compute_lift_readiness(" in env
    assert "min_contacts=self.cfg.stage0_lift_start_min_contacts" in env
    assert "hold_steps=self.cfg.stage0_lift_start_hold_steps" in env
    assert "full_grip_ready_now = (" in env
    assert "& has_5_contact_bool" in env
    assert "& lifted_for_full_grip" in env
    assert "& upright_success_for_grip" in env
    assert "& no_slip_gate.bool()" not in env
    assert "force_delta_ratio_abs_for_ready <= self.cfg.stabilize_force_delta_threshold" not in env
    assert "self._full_grip_ready_latched_buf |= full_grip_ready_now" in env
    assert "compute_grasp_v2_success(" in env
    assert "lift_success_candidate = in_or_past_lift & lifted & lift_grasped & upright_success" in env
    assert "self._lift_success_hold_count = torch.where(" in env
    assert "lift_success_now = self._lift_success_hold_count >= int(self.cfg.full_grip_hold_steps)" in env


def test_v11_tracks_pre_lift_full_contact_rate() -> None:
    env = _text("grasp_right_env.py")

    assert '"task/five_tip_contact_rate"' in env
    assert '"task/prelift_five_tip_contact_rate"' in env
    assert '"task/lift_five_tip_contact_rate"' in env
    assert "full_tip_middle_contact & self.is_grasp_phase" in env
    assert 'self.extras["debug/rh56f1/task/prelift_force_ratio"]' in env


def test_v11_phase_rewards_match_tip_lift_and_stabilize_contract() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for name in (
        "approach_weight",
        "approach_sharpness",
        "approach_xy_penalty_weight",
        "approach_tilt_penalty_weight",
        "grasp_weight",
        "lift_reward_weight",
        "stabilize_weight",
        "transport_xyz_scale",
        "transport_xyz_reward_weight",
        "transport_height_target_delta",
        "transport_height_quality_power",
        "transport_upright_quality_power",
        "transport_success_hold_steps",
        "stabilize_spawn_xy_scale",
        "success_bonus_weight",
        "post_lift_contact_loss_weight",
        "action_smooth_weight",
        "stability_reward_weight",
        "stability_cup_lin_vel_threshold",
        "stability_cup_ang_vel_threshold",
        "stability_contact_delta_threshold",
        "stability_action_delta_threshold",
        "stabilize_upright_max_deg: float = 5.0",
        "stabilize_upright_reward_scale_deg",
        "stage0_lift_start_min_contacts: int = 4",
        "grasp_phase_full_grip_contact_threshold: int = 4",
        "grasp_phase_full_grip_progress_threshold: float = 0.65",
    ):
        assert name in cfg

    for removed_name in (
        "full_grasp_bonus_weight",
        "tip_approach_bonus_weight",
        "grasp_quality_lift_weight",
        "grasp_all_tip_bonus_weight",
        "lift_tip_contact_reward_weight",
        "lift_tip_force_reward_weight",
        "stabilize_tip_contact_reward_weight",
        "stabilize_hold_reward_weight",
        "force_balance_weight",
        "force_balance_sharpness",
    ):
        assert removed_name not in cfg

    assert "compute_grasp_reward_terms(" in env
    assert "self.cfg.stage0_lift_start_hold_steps" in env
    assert "lift_latched=self._lift_started_buf" in env
    assert "full_tip_contact=five_tip_contact" in env
    assert "compute_action_delta_norm(self.actions, self.prev_actions)" in env
    assert '"reward/approach"' in env
    assert '"reward/grasp"' in env
    assert '"reward/lift"' in env
    assert '"reward/post_lift_contact_loss"' in env
    assert '"reward/stabilize"' in env
    assert '"reward/transport_track"' in env
    assert '"reward/transport_progress"' in env
    assert '"reward/stability"' in env
    assert '"task/transport_height_quality"' in env
    assert '"task/transport_posture_quality"' in env
    assert '"reward/success_bonus"' in env
    for removed_term in (
        "r_align_upright",
        "r_grasp_contact_dense",
        "r_grasp_five_tip_hold",
        "r_grasp_five_tip_contact",
        "r_stabilize_upright",
        "r1c_full_grasp",
        "r1b_force_balance",
        "r2_tip_bonus",
        "r5_quality_lift",
        "r_grasp_all_tip_bonus",
        "r_lift_tip_contact",
        "r_lift_tip_force",
        "r_stabilize_tip_contact",
        "r_stabilize_hold",
        'reward/full_grasp_bonus',
        'reward/force_balance',
        'reward/tip_approach_bonus',
        'reward/grasp_quality_lift',
        'reward/grasp_all_tip_bonus',
        'reward/lift_tip_contact',
        'reward/lift_tip_force',
        'reward/stabilize_tip_contact',
        'reward/stabilize_hold',
        'task/force_balance_err',
    ):
        assert removed_term not in env
    assert "compute_grasp_v2_success(" in env
    assert "stabilize_upright_max_deg: float = 5.0" in cfg
    assert "stable=stability.stable" in env


def test_v11_logs_only_curated_cup_task_reward_groups() -> None:
    env = _text("grasp_right_env.py")

    for name in (
        "phase/approach",
        "phase/grasp",
        "phase/lift",
        "phase/stabilize",
            "contact/palm_force",
            "contact/palm_at_lift_start",
            "object_stat/obj_z",
            "cup/height_delta",
            "cup/tilt_deg",
            "cup/lift_tilt_deg",
            "task/stable_rate",
            "task/cup_lin_vel",
            "task/cup_ang_vel",
            "task/action_delta_norm",
            "task/lift_success_now",
            "task/stabilize_success_now",
            "task/transport_success_now",
            "reward/total",
        ):
        if name == "object_stat/obj_z":
            assert f'self.extras["{name}"]' in env
        else:
            assert f'"{name}"' in env

    assert 'self.extras.clear()' in env
    for removed_name in (
        "sensor/palm/force_x",
        "sensor/palm/force_norm",
        "sensor/contact_count",
        "cup/pos_x",
        "cup/quat_w",
        "cup/lin_vel_norm",
        "cup/ang_vel_norm",
    ):
        assert f'self.extras["{removed_name}"]' not in env
    assert 'self.extras[f"sensor/tip/{tip_name}/force_x"]' not in env
    assert 'self.extras[f"joint/{joint_name}/pos"]' not in env
    assert '"stat_' not in env
    assert '"bin_' not in env


def test_v11_rl_games_config_uses_v11_name() -> None:
    t = (_ROOT / "config" / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")

    assert "name: inspire_r_grasp_v1" in t
    assert "load_checkpoint: False" in t


def test_v11_lstm_rl_games_config_uses_no_actor_mass_recurrent_name() -> None:
    t = (_ROOT / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")

    assert "name: inspire_r_grasp_v1-lstm" in t
    assert "Actor 99D MLP [512, 512] -> LSTM 1024 / critic 117D MLP [512, 512, 256, 128]" in t
    assert "name: lstm" in t
    assert "before_mlp: False" in t
    assert "units: [512, 512, 256, 128]" in t
    assert "entropy_coef: 0.0005" in t
    assert "horizon_length: 32" in t
    assert "minibatch_size: 16384" in t
    assert "seq_length: 16" in t
    central_value = t.split("central_value_config:", 1)[1]
    assert "rnn:" not in central_value
    assert "load_checkpoint: False" in t
