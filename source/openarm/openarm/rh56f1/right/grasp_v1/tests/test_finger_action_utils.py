from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "finger_action_utils.py"
SPEC = importlib.util.spec_from_file_location("finger_action_utils", MODULE_PATH)
finger_action_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = finger_action_utils
SPEC.loader.exec_module(finger_action_utils)

compute_grasp_finger_targets = finger_action_utils.compute_grasp_finger_targets
compute_lift_finger_targets = finger_action_utils.compute_lift_finger_targets


def test_grasp_finger_targets_use_absolute_synergy_interpolation() -> None:
    action = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    approach = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    grasp = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
    lower = torch.zeros(3, dtype=torch.float32)
    upper = torch.full((3,), 10.0, dtype=torch.float32)

    target = compute_grasp_finger_targets(
        finger_action=action,
        approach_pose=approach,
        grasp_pose=grasp,
        lower_limits=lower,
        upper_limits=upper,
    )

    assert target.squeeze(0).tolist() == pytest.approx([1.0, 4.0, 7.0])


def test_lift_finger_targets_clamp_to_joint_limits() -> None:
    action = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    grasp = torch.tensor([0.2, 0.3], dtype=torch.float32)
    full_grip = torch.tensor([1.5, 2.0], dtype=torch.float32)
    lower = torch.zeros(2, dtype=torch.float32)
    upper = torch.tensor([1.0, 1.2], dtype=torch.float32)

    target = compute_lift_finger_targets(
        finger_action=action,
        grasp_pose=grasp,
        full_grip_pose=full_grip,
        lower_limits=lower,
        upper_limits=upper,
    )

    assert target.squeeze(0).tolist() == pytest.approx([1.0, 1.2])
