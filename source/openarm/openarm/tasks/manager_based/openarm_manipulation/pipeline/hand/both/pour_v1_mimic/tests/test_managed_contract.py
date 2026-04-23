"""Static contract tests for the Managed (ManagerBased) Mimic implementation.

These tests verify file-level structure without importing Isaac Lab, so they
run in any Python environment and catch regressions before sim launch.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# __init__.py
# ---------------------------------------------------------------------------


def test_init_registers_managed_env_for_eval_id() -> None:
    t = _text("__init__.py")
    assert 'id="Pour-Mimic-V1-v0"' in t
    assert "pour_mimic_managed_env:PourMimicManagedEnv" in t
    assert "PourMimicManagedEnvCfg" in t


def test_init_registers_managed_env_for_mimic_id() -> None:
    t = _text("__init__.py")
    assert 'id="Pour-Mimic-V1-Mimic-v0"' in t
    assert "PourMimicManagedMimicEnvCfg" in t


# ---------------------------------------------------------------------------
# pour_mimic_managed_env.py
# ---------------------------------------------------------------------------


def test_managed_env_inherits_manager_based_rl_mimic_env() -> None:
    t = _text("pour_mimic_managed_env.py")
    assert "class PourMimicManagedEnv(ManagerBasedRLMimicEnv)" in t


def test_managed_env_exposes_all_required_mimic_methods() -> None:
    t = _text("pour_mimic_managed_env.py")
    required = (
        "def get_robot_eef_pose(",
        "def target_eef_pose_to_action(",
        "def action_to_target_eef_pose(",
        "def actions_to_gripper_actions(",
        "def get_object_poses(",
        "def get_subtask_term_signals(",
    )
    for method in required:
        assert method in t, f"missing: {method}"


def test_managed_env_has_start_signal_method() -> None:
    t = _text("pour_mimic_managed_env.py")
    assert "def get_subtask_start_signals(" in t


def test_managed_env_tracks_source_cup_init_z() -> None:
    t = _text("pour_mimic_managed_env.py")
    assert "_source_cup_init_z" in t
    assert "_reset_idx" in t


def test_managed_env_all_four_subtask_signals_present() -> None:
    t = _text("pour_mimic_managed_env.py")
    for signal in ("grasp_done", "lift_done", "align_done", "pour_done"):
        assert signal in t, f"missing subtask signal: {signal}"


# ---------------------------------------------------------------------------
# pour_mimic_managed_env_cfg.py
# ---------------------------------------------------------------------------


def test_managed_env_cfg_inherits_manager_based_rl_env_cfg() -> None:
    t = _text("pour_mimic_managed_env_cfg.py")
    assert "class PourMimicManagedEnvCfg(ManagerBasedRLEnvCfg)" in t


def test_mimic_env_cfg_inherits_both_bases() -> None:
    t = _text("pour_mimic_managed_env_cfg.py")
    assert "class PourMimicManagedMimicEnvCfg(PourMimicManagedEnvCfg, MimicEnvCfg)" in t


def test_managed_env_cfg_has_subtask_thresholds() -> None:
    t = _text("pour_mimic_managed_env_cfg.py")
    for threshold in (
        "grasp_force_threshold",
        "lift_threshold_z",
        "align_threshold_xy",
        "pour_threshold_tilt_deg",
    ):
        assert threshold in t, f"missing threshold: {threshold}"


def test_mimic_env_cfg_has_right_and_left_subtask_configs() -> None:
    t = _text("pour_mimic_managed_env_cfg.py")
    assert 'subtask_configs["right"]' in t
    assert 'subtask_configs["left"]' in t
    assert "generation_num_trials" in t


def test_mimic_env_cfg_covers_all_four_right_subtasks() -> None:
    t = _text("pour_mimic_managed_env_cfg.py")
    for signal in ("grasp_done", "lift_done", "align_done", "pour_done"):
        assert signal in t, f"missing subtask term signal: {signal}"


# ---------------------------------------------------------------------------
# pour_mimic_obs_cfg.py
# ---------------------------------------------------------------------------


def test_obs_cfg_policy_group_has_91d_components() -> None:
    t = _text("pour_mimic_obs_cfg.py")
    # 27D right joint pos/vel
    assert "right_joint_pos" in t
    assert "right_joint_vel" in t
    # 7D left arm
    assert "left_joint_pos" in t
    assert "left_joint_vel" in t
    # 5D tip force
    assert "tip_force_norm" in t
    # 18D prev actions
    assert "prev_actions" in t


def test_obs_cfg_subtask_terms_group_not_concatenated() -> None:
    t = _text("pour_mimic_obs_cfg.py")
    assert "concatenate_terms = False" in t
    for signal in ("grasp_done", "lift_done", "align_done", "pour_done"):
        assert f"{signal}_signal" in t or f"{signal} =" in t


def test_obs_cfg_tip_force_reads_five_sensors() -> None:
    t = _text("pour_mimic_obs_cfg.py")
    for i in range(1, 6):
        assert f"tip{i}_sensor" in t


# ---------------------------------------------------------------------------
# pour_mimic_scene_cfg.py
# ---------------------------------------------------------------------------


def test_scene_cfg_uses_sensor_usd() -> None:
    t = _text("pour_mimic_scene_cfg.py")
    assert "openarm_tesollo_sensor.usd" in t
    assert "activate_contact_sensors=True" in t


def test_scene_cfg_has_five_fingertip_sensors() -> None:
    t = _text("pour_mimic_scene_cfg.py")
    for i in range(1, 6):
        assert f"tip{i}_sensor" in t
        assert f"rl_dg_{i}_tip" in t


def test_scene_cfg_has_source_and_target_cups() -> None:
    t = _text("pour_mimic_scene_cfg.py")
    assert "source_cup" in t
    assert "target_cup" in t
    assert "kinematic_enabled=True" in t


# ---------------------------------------------------------------------------
# fabrics_action_term.py
# ---------------------------------------------------------------------------


def test_fabrics_action_term_action_dim_is_11() -> None:
    t = _text("fabrics_action_term.py")
    assert "action_dim" in t
    assert "11" in t


def test_fabrics_action_term_has_process_and_apply() -> None:
    t = _text("fabrics_action_term.py")
    assert "def process_actions(" in t
    assert "def apply_actions(" in t
    assert "def reset(" in t


# ---------------------------------------------------------------------------
# left_arm_action_term.py
# ---------------------------------------------------------------------------


def test_left_arm_action_term_action_dim_is_7() -> None:
    t = _text("left_arm_action_term.py")
    assert "_NUM_LEFT_ARM = 7" in t


def test_left_arm_action_term_slices_indices_11_to_18() -> None:
    t = _text("left_arm_action_term.py")
    assert "11:18" in t
