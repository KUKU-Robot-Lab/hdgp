from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def test_v11_registers_train_and_play_ids() -> None:
    t = (_ROOT / "config" / "__init__.py").read_text(encoding="utf-8")

    assert 'id="5g_grasp_right-v11"' in t
    assert 'id="5g_grasp_right-v11-lstm"' in t
    assert 'id="5g_grasp_right-play-v11"' in t
    assert 'id="5g_grasp_right-play-v11-lstm"' in t
    assert ".pipeline.hand.right.5g_grasp_right_v11" in t
    assert ".grasp_right_env:GraspRightEnv" in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY"' in t
    assert 'env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass_PLAY"' in t


def test_v11_keeps_actor_and_critic_observation_shapes() -> None:
    constants = _text("grasp_right_constants.py")
    cfg = _text("grasp_right_env_cfg.py")

    assert "NUM_OBSERVATIONS = 144" in constants
    assert "NUM_OBSERVATIONS_WITH_MASS = 145" in constants
    assert "NUM_OBSERVATIONS_NO_MASS = NUM_OBSERVATIONS" in constants
    assert "NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 174" in constants
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
    for name in (
        "transport_goal_dist_threshold",
        "transport_goal_x_range",
        "transport_goal_y_range",
        "transport_goal_z_range",
        "transport_success_hold_steps",
        "lift_target_z_delta",
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
    assert "enable_demo_grasp_reset: bool = True" in cfg
    assert "compute_transport_success_mask" in env
    assert "compute_slip_proxy" in env
    assert "transport_palm_target_pose_buf" in env


def test_v11_lift_target_is_capped_near_ten_cm_for_demo_and_non_demo() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "lift_target_z_delta: float = LIFT_Z_DELTA" in cfg
    assert "lift_elapsed_steps = torch.where(" in env
    assert "self._lift_started_buf," in env
    assert "self.episode_length_buf - self._lift_start_step_buf" in env
    assert "+ float(self.cfg.lift_target_z_delta) * lift_progress.squeeze(1)" in env
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


def test_v11_actor_observes_cup_orientation_and_angular_velocity() -> None:
    constants = _text("grasp_right_constants.py")
    env = _text("grasp_right_env.py")

    assert "cup_ang_vel:              3" in constants
    assert "cup_rot (quat):           4" in constants
    assert "cup_ang_vel  = self.cup.data.root_ang_vel_w" in env
    assert "cup_rot      = self.object_rot" in env
    assert "cup_ang_vel,            # 3" in env
    assert "cup_rot,                # 4" in env


def test_v11_samples_transport_goal_per_reset_env() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "transport_goal_x_range: tuple[float, float] = (0.22, 0.42)" in cfg
    assert "transport_goal_y_range: tuple[float, float] = (-0.02, 0.18)" in cfg
    assert "transport_goal_z_range: tuple[float, float] = (0.42, 0.58)" in cfg
    assert "def _sample_transport_goals" in env
    assert "self.object_goal[env_ids] = self._sample_transport_goals(n)" in env
    assert "goal_delta = self.object_goal[just_entering_transport] - current_object" in env


def test_v11_samples_static_bead_bins_and_keeps_dynamic_spawn_disabled() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    for name in (
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
    assert "bead_count_max: int = 30" in cfg
    assert "dynamic_bead_spawn_enabled: bool = False" in cfg
    assert "bead_initial_count_max: int = 0" in cfg
    assert "_bead_lvl = torch.randint(min_level, max_level + 1, (n,), device=self.device)" in env
    assert "bead_count = _bead_lvl * 10  # {0, 10, 20, 30}" in env
    assert "target_bead_count = bead_count" in env
    assert "self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.num_beads" in env
    assert "self._bead_mass_normalized * self.cfg.num_beads * self.cfg.bead_single_mass" in env
    assert "dynamic_bead_mask = (" in env
    assert "& is_stabilize" in env
    assert "dynamic_bead_delay = max(int(self.cfg.dynamic_bead_spawn_step) - STABILIZE_START_STEP, 0)" in env
    assert "(stabilize_elapsed == dynamic_bead_delay)" in env
    assert "& self._lift_success_latched_buf" in env
    assert "self._spawn_dynamic_beads(dynamic_bead_mask)" in env
    assert "self._bead_count_current" in env
    assert "self._bead_count_target" in env
    assert "quat_apply(cup_quat_w[activate], local_offset)" in env
    assert 'self.extras["stat_dynamic_bead_added"]' in env
    assert 'self.extras["stat_bead_count_initial"]' in env
    assert 'self.extras["stat_bead_count_current"]' in env
    assert 'self.extras["stat_cup_friction"]' in env
    for tag in ("0b", "10b", "20b", "30b"):
        assert f'"{tag}"' in env
    assert 'self.extras[f"bin_{_tag}_f_ratio"] = force_ratio[_mask].mean()' in env
    assert 'self.extras[f"bin_{_tag}_sr"]' in env
    assert 'self.extras[f"bin_{_tag}_contacts"] = self.num_contacts_buf[_mask].float().mean()' in env
    assert "_zero = torch.zeros((), device=self.device)" in env
    assert 'self.extras[f"bin_{_tag}_f_ratio"] = _zero' in env
    assert 'self.extras[f"bin_{_tag}_contacts"] = _zero' in env


def test_v11_bead_spawn_uses_pour_material_contract() -> None:
    cfg = _text("grasp_right_env_cfg.py")

    assert "_DEFAULT_BEAD_MASS = 0.010" in cfg
    assert "scale=(0.5, 0.5, 0.5)" in cfg
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=_DEFAULT_BEAD_MASS)" in cfg
    assert "linear_damping=0.5" in cfg
    assert "angular_damping=0.5" in cfg
    assert "static_friction=0.1" in cfg
    assert "dynamic_friction=0.08" in cfg
    assert "restitution=0.1" in cfg


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
        "def _maybe_export_warm_states",
        "def _write_warm_state_export_file",
    ):
        assert name in env


def test_v11_phase_curriculum_starts_lift_by_time_and_gates_late_phases() -> None:
    env = _text("grasp_right_env.py")

    assert "self._phase_curriculum_stage = min(max(int(cfg.phase_curriculum_initial_stage), 0), 2)" in env
    assert "def _maybe_update_phase_curriculum" in env
    assert "self._episode_curriculum_stage_buf >= 1" in env
    assert "self._episode_curriculum_stage_buf >= 2" in env
    assert "transport_disabled = self._episode_curriculum_stage_buf[env_ids] < 2" in env
    assert "self.object_goal[env_ids_tensor[transport_disabled]] = obj_pos_local[transport_disabled]" in env
    assert "time_lift_ready = self.episode_length_buf >= LIFT_START_STEP" in env
    assert "just_entering_lift = time_lift_ready & (~self._lift_started_buf)" in env
    assert "self._lift_start_step_buf[just_entering_lift] = self.episode_length_buf[just_entering_lift]" in env
    assert "just_entering_stabilize = (" in env
    assert "& self._full_grip_ready_latched_buf" in env
    assert "just_entering_transport = (" in env
    assert "& self._stabilize_success_latched_buf" in env
    assert "& self._full_grip_ready_buf" in env
    assert "curriculum_lift_horizon" in env
    assert "curriculum_stabilize_horizon" in env
    assert "lift_failed = (" in env
    assert 'self.extras["stat_curriculum_stage"]' in env
    assert 'self.extras["stat_lift_success_rate"]' in env
    assert 'self.extras["stat_stabilize_success_rate"]' in env
    assert 'self.extras["stat_lift_contact_ready_rate"]' in env
    assert 'self.extras["stat_lift_started_rate"]' in env
    assert 'self.extras["stat_full_grip_ready_rate"]' in env


def test_v11_grip_first_curriculum_uses_split_readiness_gates() -> None:
    env = _text("grasp_right_env.py")

    assert "lift_contact_now = self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS" in env
    assert "lift_contact_phase = self.is_grasp_phase | self.is_lift_phase" in env
    assert "self._lift_contact_hold_count >= int(self.cfg.lift_contact_hold_steps)" in env
    assert "self._lift_contact_ready_latched_buf |= lift_contact_ready_now" in env
    assert "full_grip_ready_now = (" in env
    assert "& has_5_contact_bool" in env
    assert "& middle_envelope_gate.bool()" in env
    assert "& no_slip_gate.bool()" in env
    assert "& upright_success_for_grip" in env
    assert "force_delta_ratio_abs_for_ready <= self.cfg.stabilize_force_delta_threshold" in env
    assert ") & full_grip_ready_now" in env
    assert "lift_success_now = in_or_past_lift & lifted & lift_grasped & upright_success" in env


def test_v11_tracks_pre_lift_full_contact_rate() -> None:
    env = _text("grasp_right_env.py")

    assert 'self.extras["stat_pre_lift_full_contact_rate"]' in env
    assert "pre_lift_full_contact" in env


def test_v11_rl_games_config_uses_v11_name() -> None:
    t = (_ROOT / "config" / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")

    assert "name: 5g_grasp_right-v11" in t
    assert "load_checkpoint: False" in t


def test_v11_lstm_rl_games_config_uses_no_actor_mass_recurrent_name() -> None:
    t = (_ROOT / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")

    assert "name: 5g_grasp_right-v11-lstm" in t
    assert "Actor 144D MLP [512, 512] -> LSTM 1024 / critic 174D MLP [512, 512, 256, 128]" in t
    assert "name: lstm" in t
    assert "before_mlp: False" in t
    assert "units: [512, 512, 256, 128]" in t
    central_value = t.split("central_value_config:", 1)[1]
    assert "rnn:" not in central_value
    assert "load_checkpoint: False" in t
