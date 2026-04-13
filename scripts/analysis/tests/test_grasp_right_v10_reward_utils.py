from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


UTILS_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/"
    "openarm_manipulation/pipeline/hand/right/5g_grasp_right_v10/grasp_reward_utils.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("grasp_reward_utils", UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bounded_force_smooth_penalty_saturates_large_outliers():
    module = load_module()

    force_delta_norm = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float32)
    penalty = module.compute_bounded_force_smooth_penalty(
        force_delta_norm=force_delta_norm,
        weight=1.5,
        penalty_cap=2.0,
    )

    assert penalty[0].item() == 0.0
    assert penalty[1].item() < 0.0
    assert penalty[2].item() > -3.0
    assert penalty[2].item() < penalty[1].item()


def test_thumb_pose_anchor_reward_prefers_reference_envelope_shape():
    module = load_module()

    reference = torch.tensor([0.0, -1.55, 0.13, 0.99], dtype=torch.float32)
    close_pose = torch.tensor([[0.0, -1.58, 0.16, 1.02]], dtype=torch.float32)
    far_pose = torch.tensor([[0.0, -2.20, 0.45, 1.35]], dtype=torch.float32)

    close_reward, close_error = module.compute_thumb_pose_anchor_reward(
        thumb_joint_pos=close_pose,
        thumb_reference_pose=reference,
        weight=1.2,
        sharpness=8.0,
    )
    far_reward, far_error = module.compute_thumb_pose_anchor_reward(
        thumb_joint_pos=far_pose,
        thumb_reference_pose=reference,
        weight=1.2,
        sharpness=8.0,
    )

    assert close_reward.item() > far_reward.item()
    assert close_error.item() < far_error.item()


def test_thumb_downward_slide_penalty_grows_when_thumb_tip_drops_below_corridor():
    module = load_module()

    grasp_center = torch.tensor([[0.40, -0.15, 0.36]], dtype=torch.float32)
    safe_thumb_tip = torch.tensor([[0.41, -0.11, 0.37]], dtype=torch.float32)
    slipped_thumb_tip = torch.tensor([[0.41, -0.11, 0.32]], dtype=torch.float32)

    safe_penalty, safe_delta = module.compute_thumb_downward_slide_penalty(
        thumb_tip_pos=safe_thumb_tip,
        grasp_center=grasp_center,
        z_margin=0.01,
        weight=2.0,
    )
    slipped_penalty, slipped_delta = module.compute_thumb_downward_slide_penalty(
        thumb_tip_pos=slipped_thumb_tip,
        grasp_center=grasp_center,
        z_margin=0.01,
        weight=2.0,
    )

    assert safe_penalty.item() == 0.0
    assert slipped_penalty.item() < safe_penalty.item()
    assert slipped_delta.item() > safe_delta.item()


def test_grasp_shape_consistency_reward_tracks_deviation_from_reference_pose():
    module = load_module()

    current_pose = torch.tensor(
        [[0.0, -1.55, 0.13, 0.99, 0.02, 0.66, 0.63, 0.84]],
        dtype=torch.float32,
    )
    distorted_pose = torch.tensor(
        [[0.0, -2.30, 0.45, 1.40, 0.02, 1.40, 1.10, 1.10]],
        dtype=torch.float32,
    )
    reference_pose = torch.tensor(
        [0.0, -1.55, 0.13, 0.99, 0.02, 0.66, 0.63, 0.84],
        dtype=torch.float32,
    )
    lower = torch.tensor([-0.1, -3.14, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0], dtype=torch.float32)
    upper = torch.tensor([0.1, 0.0, 0.8, 1.6, 0.1, 2.0, 1.5, 1.5], dtype=torch.float32)
    active_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    close_reward, close_error = module.compute_grasp_shape_consistency_reward(
        hand_joint_pos=current_pose,
        reference_pose=reference_pose,
        lower_limits=lower,
        upper_limits=upper,
        active_mask=active_mask,
        weight=1.0,
        sharpness=6.0,
    )
    far_reward, far_error = module.compute_grasp_shape_consistency_reward(
        hand_joint_pos=distorted_pose,
        reference_pose=reference_pose,
        lower_limits=lower,
        upper_limits=upper,
        active_mask=active_mask,
        weight=1.0,
        sharpness=6.0,
    )

    assert close_reward.item() > far_reward.item()
    assert close_error.item() < far_error.item()
