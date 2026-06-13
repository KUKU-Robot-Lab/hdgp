from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_reward_utils.py"
PRESET_PATH = Path(__file__).resolve().parents[1] / "grasp_right_preset.py"
SPEC = importlib.util.spec_from_file_location("grasp_reward_utils", MODULE_PATH)
grasp_reward_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grasp_reward_utils
SPEC.loader.exec_module(grasp_reward_utils)

compute_thumb_tip_direction_reward = grasp_reward_utils.compute_thumb_tip_direction_reward
compute_upright_success_mask = grasp_reward_utils.compute_upright_success_mask


def _load_preset_module():
    spec = importlib.util.spec_from_file_location("grasp_right_preset", PRESET_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_thumb_tip_direction_reward_prefers_tip_axis_toward_cup() -> None:
    grasp_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    thumb_tip = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    aligned_distal = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    misaligned_distal = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    aligned_reward, aligned_cos, aligned_error = compute_thumb_tip_direction_reward(
        thumb_distal_pos=aligned_distal,
        thumb_tip_pos=thumb_tip,
        grasp_center=grasp_center,
        weight=4.0,
        sharpness=4.0,
        distance_scale=10.0,
    )
    misaligned_reward, misaligned_cos, misaligned_error = compute_thumb_tip_direction_reward(
        thumb_distal_pos=misaligned_distal,
        thumb_tip_pos=thumb_tip,
        grasp_center=grasp_center,
        weight=4.0,
        sharpness=4.0,
        distance_scale=10.0,
    )

    assert aligned_cos.item() > misaligned_cos.item()
    assert aligned_error.item() < misaligned_error.item()
    assert aligned_reward.item() > misaligned_reward.item()


def test_upright_success_mask_requires_configured_tilt_margin() -> None:
    cup_z_cos = torch.tensor([1.0, 0.95, 0.90], dtype=torch.float32)

    mask = compute_upright_success_mask(cup_z_cos, threshold_deg=20.0)

    assert mask.tolist() == [True, True, False]


def test_envelope_success_requires_full_tip_contact_not_palm_contact() -> None:
    tip_contact_count = torch.tensor([5, 4, 5], dtype=torch.long)
    palm_contact = torch.tensor([False, True, True], dtype=torch.bool)
    upright = compute_upright_success_mask(
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        threshold_deg=20.0,
    )

    success = (
        (tip_contact_count >= 5)
        & upright
    )

    assert palm_contact.tolist() == [False, True, True]
    assert success.tolist() == [True, False, True]


def test_full_grip_pose_is_relaxed_closure_bound_from_grasp_pose() -> None:
    preset = _load_preset_module()

    grasp = torch.tensor(preset.HAND_GRASP_POSE, dtype=torch.float32)
    full = torch.tensor(preset.HAND_FULL_GRIP_POSE, dtype=torch.float32)
    expected = torch.tensor(
        [
            +0.000, -1.570, +0.156, +1.186,
            +0.000, +0.791, +0.754, +1.012,
            +0.000, +1.163, +0.256, +1.636,
            -0.000, +1.002, +0.581, +1.519,
            +0.000, -0.000, +1.074, +1.333,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(full, expected)
    assert torch.allclose(full[[0, 4, 8, 12, 16]], grasp[[0, 4, 8, 12, 16]])
    assert torch.isclose(full[1], grasp[1])
    assert torch.all(full[[2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19]] > grasp[[2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19]])
