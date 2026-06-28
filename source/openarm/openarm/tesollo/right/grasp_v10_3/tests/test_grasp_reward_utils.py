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
compute_tesollo_prelift_lift_readiness = grasp_reward_utils.compute_tesollo_prelift_lift_readiness
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


def test_prelift_lift_readiness_rejects_rim_height_contact() -> None:
    hold_count, ready_now, latched, gates = compute_tesollo_prelift_lift_readiness(
        num_contacts=torch.tensor([5], dtype=torch.long),
        is_close_grasp_phase=torch.tensor([True]),
        tip_local_z_mean=torch.tensor([0.08]),
        cup_height_delta=torch.tensor([0.0]),
        cup_lin_vel_norm=torch.tensor([0.0]),
        previous_hold_count=torch.tensor([7], dtype=torch.long),
        previous_latched=torch.tensor([False]),
        min_contacts=5,
        hold_steps=8,
        body_local_z_min=-0.04,
        body_local_z_max=0.05,
        max_cup_height_delta=0.01,
        cup_lin_vel_threshold=0.04,
    )

    assert hold_count.tolist() == [0]
    assert ready_now.tolist() == [False]
    assert latched.tolist() == [False]
    assert gates["body_band"].tolist() == [False]
    assert gates["rim_contact_proxy"].tolist() == [1.0]


def test_prelift_lift_readiness_rejects_premature_lift_or_motion() -> None:
    hold_count, ready_now, latched, gates = compute_tesollo_prelift_lift_readiness(
        num_contacts=torch.tensor([5, 5], dtype=torch.long),
        is_close_grasp_phase=torch.tensor([True, True]),
        tip_local_z_mean=torch.tensor([0.0, 0.0]),
        cup_height_delta=torch.tensor([0.02, 0.0]),
        cup_lin_vel_norm=torch.tensor([0.0, 0.08]),
        previous_hold_count=torch.tensor([7, 7], dtype=torch.long),
        previous_latched=torch.tensor([False, False]),
        min_contacts=5,
        hold_steps=8,
        body_local_z_min=-0.04,
        body_local_z_max=0.05,
        max_cup_height_delta=0.01,
        cup_lin_vel_threshold=0.04,
    )

    assert hold_count.tolist() == [0, 0]
    assert ready_now.tolist() == [False, False]
    assert latched.tolist() == [False, False]
    assert gates["prelift_height_ok"].tolist() == [0.0, 1.0]
    assert gates["prelift_velocity_ok"].tolist() == [1.0, 0.0]


def test_prelift_lift_readiness_latches_on_body_contact_and_stable_cup() -> None:
    hold_count, ready_now, latched, gates = compute_tesollo_prelift_lift_readiness(
        num_contacts=torch.tensor([5], dtype=torch.long),
        is_close_grasp_phase=torch.tensor([True]),
        tip_local_z_mean=torch.tensor([0.0]),
        cup_height_delta=torch.tensor([0.005]),
        cup_lin_vel_norm=torch.tensor([0.02]),
        previous_hold_count=torch.tensor([7], dtype=torch.long),
        previous_latched=torch.tensor([False]),
        min_contacts=5,
        hold_steps=8,
        body_local_z_min=-0.04,
        body_local_z_max=0.05,
        max_cup_height_delta=0.01,
        cup_lin_vel_threshold=0.04,
    )

    assert hold_count.tolist() == [8]
    assert ready_now.tolist() == [True]
    assert latched.tolist() == [True]
    assert gates["full_contact"].tolist() == [1.0]
    assert gates["body_band"].tolist() == [1.0]
    assert gates["prelift_height_ok"].tolist() == [1.0]
    assert gates["prelift_velocity_ok"].tolist() == [1.0]
    assert gates["rim_contact_proxy"].tolist() == [0.0]


def test_full_grip_pose_is_full_curl_closure_from_grasp_pose() -> None:
    # Phase J: full_grip = v7-2 "풀그립"(곡 관절 1.5/1.6). 감싸는 4지를 완전히 말아두고
    # 물리 접촉이 컵에서 멈추게 한다. 엄지는 opposition 보존.
    preset = _load_preset_module()

    grasp = torch.tensor(preset.HAND_GRASP_POSE, dtype=torch.float32)
    full = torch.tensor(preset.HAND_FULL_GRIP_POSE, dtype=torch.float32)
    expected = torch.tensor(
        [
            +0.000, -1.570, +0.156, +1.186,
            +0.000, +1.600, +1.500, +1.500,
            +0.000, +1.600, +1.500, +1.500,
            +0.000, +1.600, +1.500, +1.500,
            +0.000, -0.000, +1.500, +1.500,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(full, expected)
    assert torch.allclose(full[[0, 4, 8, 12, 16]], grasp[[0, 4, 8, 12, 16]])
    assert torch.isclose(full[1], grasp[1])
    assert torch.all(full[[2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19]] > grasp[[2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19]])
