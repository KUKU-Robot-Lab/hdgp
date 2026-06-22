from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


def _load_finger_action_utils():
    path = ROOT / "finger_action_utils.py"
    spec = importlib.util.spec_from_file_location("grasp_v7_2_finger_action_utils", path)
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


def test_lift_uses_grasp_to_full_grip_absolute_target() -> None:
    env = _text("grasp_right_env.py")
    preset = _text("grasp_right_preset.py")

    assert "HAND_FULL_GRIP_POSE" in preset
    assert "compute_lift_finger_targets(" in env
    assert "grasp_pose=self.hand_grasp_pose" in env
    assert "full_grip_pose=self.hand_full_grip_pose" in env
    assert "self.lift_finger_pos_buf" not in env


def test_reward_uses_rh56f1_shared_core_terms() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")
    reward_body = env.split("def _get_rewards", 1)[1].split("return total", 1)[0]

    for name in (
        "approach_weight",
        "grasp_weight",
        "lift_reward_weight",
        "stabilize_weight",
        "success_bonus_weight",
        "post_lift_contact_loss_weight",
        "action_smooth_weight",
        "stability_reward_weight",
    ):
        assert name in cfg

    for term in (
        "compute_grasp_reward_terms(",
        'reward_terms["approach"]',
        'reward_terms["grasp"]',
        'reward_terms["lift"]',
        'reward_terms["stabilize"]',
        'reward_terms["success_bonus"]',
        'reward_terms["post_lift_contact_loss"]',
        'reward_terms["action_smooth"]',
        'reward_terms["stability"]',
    ):
        assert term in reward_body

    for removed in (
        "r1b_force_balance",
        "r1c_full_grasp",
        "r2_tip_bonus",
        "r5_quality_lift",
        "prelift_rim_lift_penalty",
    ):
        assert removed not in reward_body
