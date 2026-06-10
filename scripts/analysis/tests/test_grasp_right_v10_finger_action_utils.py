from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


UTILS_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "finger_action_utils.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("finger_action_utils", UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grasp_and_lift_targets_use_reference_plus_bounded_delta():
    module = load_module()

    action = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)
    current_pos = torch.tensor([[0.10, 0.20, 0.30]], dtype=torch.float32)
    lift_ref = torch.tensor([[0.40, 0.50, 0.60]], dtype=torch.float32)
    lower = torch.tensor([0.00, 0.00, 0.00], dtype=torch.float32)
    upper = torch.tensor([1.00, 1.00, 1.00], dtype=torch.float32)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    grasp_target = module.compute_grasp_finger_targets(
        current_pos=current_pos,
        finger_action=action,
        lower_limits=lower,
        upper_limits=upper,
        delta_scale=0.1,
        delta_mask=mask,
    )
    lift_target = module.compute_lift_finger_targets(
        lift_reference_pos=lift_ref,
        finger_action=action,
        lower_limits=lower,
        upper_limits=upper,
        delta_scale=0.05,
        delta_mask=mask,
    )

    assert torch.allclose(grasp_target, torch.tensor([[0.20, 0.20, 0.35]]))
    assert torch.allclose(lift_target, torch.tensor([[0.45, 0.50, 0.625]]))


def test_resolve_grasp_delta_scale_prefers_adr_override():
    module = load_module()

    assert module.resolve_grasp_delta_scale(default_scale=0.08, adr_delta_scale=None) == 0.08
    assert module.resolve_grasp_delta_scale(default_scale=0.08, adr_delta_scale=0.12) == 0.12


def test_thumb_downward_constraint_scales_negative_thumb_curl_and_clamps_floor():
    module = load_module()

    action = torch.tensor([[-1.0, -1.0, 0.5, 0.0]], dtype=torch.float32)
    current_pos = torch.tensor([[0.0, -1.60, 0.10, 0.90]], dtype=torch.float32)
    lower = torch.tensor([-0.1, -3.14, -0.1, -0.1], dtype=torch.float32)
    upper = torch.tensor([0.1, 0.0, 0.8, 1.6], dtype=torch.float32)
    mask = torch.ones(4, dtype=torch.float32)
    thumb_anchor = torch.tensor([0.0, -1.55, 0.13, 0.99], dtype=torch.float32)

    grasp_target = module.compute_grasp_finger_targets(
        current_pos=current_pos,
        finger_action=action,
        lower_limits=lower,
        upper_limits=upper,
        delta_scale=0.2,
        delta_mask=mask,
        thumb_curl_index=1,
        thumb_downward_action_scale=0.25,
        thumb_anchor_pose=thumb_anchor,
        thumb_curl_max_downward_delta=0.05,
    )

    assert torch.allclose(
        grasp_target,
        torch.tensor([[-0.1, -1.60, 0.20, 0.90]], dtype=torch.float32),
    )


def test_lift_targets_clamp_thumb_joints_while_other_fingers_keep_full_delta():
    module = load_module()

    action = torch.ones((1, 20), dtype=torch.float32)
    action[:, :4] = torch.tensor([[-1.0, 1.0, -1.0, 1.0]], dtype=torch.float32)
    lift_ref = torch.full((1, 20), 0.50, dtype=torch.float32)
    lower = torch.zeros(20, dtype=torch.float32)
    upper = torch.ones(20, dtype=torch.float32)
    mask = torch.ones(20, dtype=torch.float32)
    mask[[0, 4, 8, 12, 16, 17]] = 0.0

    target = module.compute_lift_finger_targets(
        lift_reference_pos=lift_ref,
        finger_action=action,
        lower_limits=lower,
        upper_limits=upper,
        delta_scale=0.08,
        delta_mask=mask,
        thumb_lift_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        thumb_lift_max_delta=0.02,
    )

    assert torch.all(target[:, :4] <= lift_ref[:, :4] + 0.02)
    assert torch.all(target[:, :4] >= lift_ref[:, :4] - 0.02)
    assert torch.allclose(
        target[:, :8],
        torch.tensor([[0.50, 0.52, 0.48, 0.52, 0.50, 0.58, 0.58, 0.58]], dtype=torch.float32),
    )
