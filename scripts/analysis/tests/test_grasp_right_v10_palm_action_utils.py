from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


UTILS_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "palm_action_utils.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("palm_action_utils", UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_raise_phase_ignores_arm_action_and_stops_at_five_centimeters():
    module = load_module()

    lift_start = torch.tensor([[0.30, -0.10, 0.40, 0.10, 0.20, 0.30]], dtype=torch.float32)
    grasp_target = torch.zeros_like(lift_start)
    stabilize_delta = torch.ones_like(lift_start)
    palm_mins = torch.full((6,), -10.0, dtype=torch.float32)
    palm_maxs = torch.full((6,), 10.0, dtype=torch.float32)

    half_raise = module.compute_lift_stabilize_palm_targets(
        episode_length_buf=torch.tensor([540]),
        grasp_palm_pose=grasp_target,
        lift_start_pose=lift_start,
        stabilize_delta=stabilize_delta,
        palm_mins=palm_mins,
        palm_maxs=palm_maxs,
        lift_start_step=480,
        stabilize_start_step=600,
        lift_raise_steps=120,
        lift_raise_z_delta=0.05,
    )
    end_raise = module.compute_lift_stabilize_palm_targets(
        episode_length_buf=torch.tensor([600]),
        grasp_palm_pose=grasp_target,
        lift_start_pose=lift_start,
        stabilize_delta=stabilize_delta,
        palm_mins=palm_mins,
        palm_maxs=palm_maxs,
        lift_start_step=480,
        stabilize_start_step=600,
        lift_raise_steps=120,
        lift_raise_z_delta=0.05,
    )

    assert torch.allclose(half_raise[:, :2], lift_start[:, :2])
    assert torch.allclose(half_raise[:, 3:], lift_start[:, 3:])
    assert torch.allclose(half_raise[:, 2], torch.tensor([0.425], dtype=torch.float32))
    assert torch.allclose(end_raise[:, 2], torch.tensor([0.45], dtype=torch.float32))


def test_stabilize_phase_holds_raise_height_and_applies_bounded_arm_action():
    module = load_module()

    lift_start = torch.tensor([[0.30, -0.10, 0.40, 0.10, 0.20, 0.30]], dtype=torch.float32)
    grasp_target = torch.zeros_like(lift_start)
    stabilize_delta = torch.tensor([[0.01, -0.01, 0.01, 0.05, -0.05, 0.10]], dtype=torch.float32)
    palm_mins = torch.full((6,), -10.0, dtype=torch.float32)
    palm_maxs = torch.full((6,), 10.0, dtype=torch.float32)

    target = module.compute_lift_stabilize_palm_targets(
        episode_length_buf=torch.tensor([720]),
        grasp_palm_pose=grasp_target,
        lift_start_pose=lift_start,
        stabilize_delta=stabilize_delta,
        palm_mins=palm_mins,
        palm_maxs=palm_maxs,
        lift_start_step=480,
        stabilize_start_step=600,
        lift_raise_steps=120,
        lift_raise_z_delta=0.05,
    )

    assert torch.allclose(
        target,
        torch.tensor([[0.31, -0.11, 0.45, 0.15, 0.15, 0.40]], dtype=torch.float32),
    )
