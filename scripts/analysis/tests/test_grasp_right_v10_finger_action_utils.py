from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


UTILS_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/"
    "openarm_manipulation/pipeline/hand/right/5g_grasp_right_v10/finger_action_utils.py"
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
