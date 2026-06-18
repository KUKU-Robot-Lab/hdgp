from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def test_reward_cfg_uses_shared_grasp_core_terms() -> None:
    cfg = _text("grasp_right_env_cfg.py")

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
        "transport_xyz_success_threshold",
        "stabilize_spawn_xy_scale",
        "success_bonus_weight",
        "post_lift_contact_loss_weight",
        "action_smooth_weight",
        "palm_action_delta_reward_scale",
        "finger_action_delta_reward_scale",
        "enclosure_weight",
        "enclosure_sharpness",
        "cup_radius_approx",
        "enclosure_thumb_weight",
        "stabilize_upright_reward_scale_deg",
        "stage0_lift_start_min_contacts: int = 4",
        "grasp_phase_full_grip_contact_threshold: int = 4",
        "grasp_phase_full_grip_progress_threshold: float = 0.65",
    ):
        assert name in cfg

    for removed_name in (
        "full_grasp_bonus_weight",
        "force_balance_weight",
        "force_balance_sharpness",
        "tip_approach_bonus_weight",
        "grasp_quality_lift_weight",
        "grasp_quality_lift_sharpness",
    ):
        assert removed_name not in cfg

    assert "r_margin_weight" not in cfg
    assert "r_slip_weight" not in cfg
    assert "r_height_weight" not in cfg


def test_reward_impl_uses_shared_core_term_shape() -> None:
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for term in (
        "compute_grasp_reward_terms(",
        'reward_terms["approach"]',
        'reward_terms["grasp"]',
        'reward_terms["lift"]',
        'reward_terms["post_lift_contact_loss"]',
        'reward_terms["stabilize"]',
        'reward_terms["transport_track"]',
        'reward_terms["transport_progress"]',
        'reward_terms["success_bonus"]',
        'reward_terms["action_smooth"]',
        'reward_terms["stability"]',
    ):
        assert term in reward_body

    for removed_term in (
        "r1c_full_grasp",
        "r1b_force_balance",
        "r2_tip_bonus",
        "r5_quality_lift",
    ):
        assert removed_term not in reward_body

    for log_name in (
        "reward/approach",
        "reward/grasp",
        "reward/lift",
        "reward/post_lift_contact_loss",
        "reward/stabilize",
        "reward/transport_track",
        "reward/transport_progress",
        "reward/success_bonus",
        "reward/action_smooth",
        "reward/stability",
        "reward/total",
    ):
        assert f'"{log_name}"' in reward_body

    for removed_log in (
        "reward/full_grasp_bonus",
        "reward/force_balance",
        "reward/tip_approach_bonus",
        "reward/grasp_quality_lift",
    ):
        assert f'self.extras["{removed_log}"]' not in reward_body

    assert "r_margin" not in reward_body
    assert "r_slip" not in reward_body
    assert "required_support" not in reward_body
    assert "friction_support" not in reward_body
