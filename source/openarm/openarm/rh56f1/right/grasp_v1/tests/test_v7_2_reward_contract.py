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
        "force_balance_weight",
        "force_balance_sharpness",
        "full_grasp_bonus_weight",
        "thumb_force_ratio_min",
        "tip_approach_bonus_weight",
        "lift_reward_weight",
        "action_smoothness_palm_weight",
        "action_smoothness_finger_weight",
        "grasp_quality_lift_weight",
        "grasp_quality_lift_sharpness",
    ):
        assert name in cfg

    assert "r_margin_weight" not in cfg
    assert "r_slip_weight" not in cfg
    assert "r_height_weight" not in cfg


def test_reward_impl_matches_v7_2_term_shape() -> None:
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for term in (
        "r0_palm_approach",
        "r1_enclosure",
        "r1b_force_balance",
        "r1c_full_grasp",
        "r2_tip_bonus",
        "r3_lift",
        "r4_smooth",
        "r5_quality_lift",
    ):
        assert term in reward_body

    for log_name in (
        "palm_approach_reward",
        "enclosure_reward",
        "force_balance_reward",
        "full_grasp_bonus",
        "tip_approach_bonus",
        "lift_reward",
        "action_smoothness",
        "grasp_quality_lift",
        "rew_total",
    ):
        assert f'self.extras["{log_name}"]' in reward_body

    assert "r_margin" not in reward_body
    assert "r_slip" not in reward_body
    assert "required_support" not in reward_body
    assert "friction_support" not in reward_body
