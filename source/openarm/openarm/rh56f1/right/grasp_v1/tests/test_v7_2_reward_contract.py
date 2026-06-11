from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def test_reward_cfg_uses_v7_2_dense_enclosure_terms() -> None:
    cfg = _text("grasp_right_env_cfg.py")

    for name in (
        "palm_approach_weight",
        "palm_approach_sharpness",
        "enclosure_weight",
        "enclosure_sharpness",
        "cup_radius_approx",
        "enclosure_thumb_weight",
        "align_upright_reward_weight",
        "lift_reward_weight",
        "grasp_five_tip_contact_reward_weight",
        "grasp_five_tip_hold_reward_weight",
        "grasp_contact_persistence_reward_steps",
        "stabilize_upright_reward_weight",
        "stabilize_upright_reward_scale_deg",
        "action_smoothness_palm_weight",
        "action_smoothness_finger_weight",
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


def test_reward_impl_matches_v7_2_term_shape() -> None:
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for term in (
        "r0_palm_approach",
        "r1_enclosure",
        "r_align_upright",
        "r3_lift",
        "r_grasp_contact_dense",
        "r_grasp_five_tip_hold",
        "r_grasp_five_tip_contact",
        "r_stabilize_upright",
        "r4_smooth",
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
        "reward/palm_approach",
        "reward/enclosure",
        "reward/align_upright",
        "reward/lift",
        "reward/grasp_contact_dense",
        "reward/grasp_five_tip_hold",
        "reward/grasp_five_tip_contact",
        "reward/stabilize_upright",
        "reward/action_smoothness",
        "reward/total",
    ):
        assert f'self.extras["{log_name}"]' in reward_body

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
