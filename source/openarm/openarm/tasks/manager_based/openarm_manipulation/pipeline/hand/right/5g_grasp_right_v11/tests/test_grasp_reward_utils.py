from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_reward_utils.py"
PRESET_PATH = Path(__file__).resolve().parents[1] / "grasp_right_preset.py"
SPEC = importlib.util.spec_from_file_location("grasp_reward_utils", MODULE_PATH)
grasp_reward_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grasp_reward_utils
SPEC.loader.exec_module(grasp_reward_utils)

compute_thumb_tip_direction_reward = grasp_reward_utils.compute_thumb_tip_direction_reward
compute_middle_contact_gate = grasp_reward_utils.compute_middle_contact_gate
compute_upright_success_mask = grasp_reward_utils.compute_upright_success_mask
compute_lift_height_reward = grasp_reward_utils.compute_lift_height_reward
compute_gated_lift_height_reward = grasp_reward_utils.compute_gated_lift_height_reward
compute_grip_ready_gate = grasp_reward_utils.compute_grip_ready_gate
compute_full_contact_shaping_reward = grasp_reward_utils.compute_full_contact_shaping_reward
compute_slip_proxy = grasp_reward_utils.compute_slip_proxy
compute_ring_pinky_separation_penalty = grasp_reward_utils.compute_ring_pinky_separation_penalty
compute_transport_success_mask = grasp_reward_utils.compute_transport_success_mask


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


def test_middle_contact_gate_requires_four_middle_contacts() -> None:
    middle_binary = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    gate = compute_middle_contact_gate(middle_binary, min_middle_contacts=4)

    assert gate.tolist() == [False, True, True]


def test_envelope_success_rejects_tip_only_contact() -> None:
    tip_contact_count = torch.tensor([5, 5], dtype=torch.long)
    middle_binary = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    upright = compute_upright_success_mask(
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        threshold_deg=20.0,
    )

    success = (
        (tip_contact_count >= 5)
        & compute_middle_contact_gate(middle_binary, 4)
        & upright
    )

    assert success.tolist() == [False, True]


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


def test_ring_pinky_separation_penalty_only_when_too_close() -> None:
    ring_tip = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=torch.float32)
    pinky_tip = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]], dtype=torch.float32)
    ring_middle = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.03, 0.0]], dtype=torch.float32)
    pinky_middle = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.03, 0.0]], dtype=torch.float32)

    penalty, min_dist = compute_ring_pinky_separation_penalty(
        ring_tip,
        pinky_tip,
        ring_middle,
        pinky_middle,
        min_distance=0.012,
        weight=4.0,
    )

    assert min_dist.tolist() == [0.0, pytest.approx(0.03)]
    assert penalty[0].item() < 0.0
    assert penalty[1].item() == pytest.approx(0.0)


def test_lift_height_reward_is_capped_after_configured_height() -> None:
    height = torch.tensor([0.02, 0.12, 0.20], dtype=torch.float32)
    upright = torch.ones(3, dtype=torch.float32)

    reward, capped_height = compute_lift_height_reward(
        cup_height_delta=height,
        cup_z_cos=upright,
        lift_height_cap=0.12,
        weight=20.0,
    )

    assert capped_height.tolist() == pytest.approx([0.02, 0.12, 0.12])
    assert reward[1].item() == pytest.approx(reward[2].item())


def test_gated_lift_height_reward_requires_grip_ready_and_no_slip() -> None:
    height = torch.tensor([0.08, 0.08, 0.08], dtype=torch.float32)
    upright = torch.ones(3, dtype=torch.float32)
    grip_ready = torch.tensor([False, True, True])
    no_slip = torch.tensor([True, False, True])

    reward, _ = compute_gated_lift_height_reward(
        cup_height_delta=height,
        cup_z_cos=upright,
        lift_height_cap=0.12,
        weight=20.0,
        grip_ready_gate=grip_ready,
        no_slip_gate=no_slip,
    )

    assert reward.tolist() == pytest.approx([0.0, 0.0, 1.6])


def test_grip_ready_gate_requires_contacts_hold_force_slip_and_tilt() -> None:
    tip_binary = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    middle_binary = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    gate = compute_grip_ready_gate(
        tip_binary_contact=tip_binary,
        middle_binary_contact=middle_binary,
        contact_persistence_steps=torch.tensor([30, 30, 10, 30]),
        force_ratio=torch.tensor([2.0, 2.0, 2.0, 2.0]),
        cup_xy_slip=torch.tensor([0.005, 0.005, 0.005, 0.005]),
        cup_tilt_deg=torch.tensor([2.0, 2.0, 2.0, 2.0]),
        min_tip_contacts=5,
        min_middle_contacts=4,
        hold_steps=30,
        min_force_ratio=1.8,
        slip_threshold=0.01,
        tilt_threshold_deg=8.0,
    )

    assert gate.tolist() == [True, False, False, False]


def test_full_contact_reward_prefers_worst_finger_coverage_over_mean_only() -> None:
    mean_only_tip = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    mean_only_distal = mean_only_tip.clone()
    mean_only_middle = mean_only_tip.clone()
    balanced = torch.full((1, 5), 0.5, dtype=torch.float32)

    mean_only_reward, mean_only_score, mean_only_worst, _ = compute_full_contact_shaping_reward(
        mean_only_tip,
        mean_only_distal,
        mean_only_middle,
        weight=12.0,
    )
    balanced_reward, balanced_score, balanced_worst, _ = compute_full_contact_shaping_reward(
        balanced,
        balanced,
        balanced,
        weight=12.0,
    )

    assert balanced_worst.item() > mean_only_worst.item()
    assert balanced_score.item() > mean_only_score.item()
    assert balanced_reward.item() > mean_only_reward.item()


def test_slip_proxy_increases_with_velocity_tilt_and_contact_churn() -> None:
    baseline = compute_slip_proxy(
        cup_xy_velocity=torch.tensor([0.01]),
        cup_tilt_delta_deg=torch.tensor([1.0]),
        contact_delta_abs=torch.tensor([0.0]),
        middle_contact_delta_abs=torch.tensor([0.0]),
        xy_velocity_scale=0.04,
        tilt_delta_scale=8.0,
        contact_delta_scale=1.0,
        middle_contact_delta_scale=1.0,
        contact_delta_weight=0.5,
        middle_contact_delta_weight=0.5,
        tilt_delta_weight=0.5,
    )
    slipped = compute_slip_proxy(
        cup_xy_velocity=torch.tensor([0.08]),
        cup_tilt_delta_deg=torch.tensor([4.0]),
        contact_delta_abs=torch.tensor([2.0]),
        middle_contact_delta_abs=torch.tensor([1.0]),
        xy_velocity_scale=0.04,
        tilt_delta_scale=8.0,
        contact_delta_scale=1.0,
        middle_contact_delta_scale=1.0,
        contact_delta_weight=0.5,
        middle_contact_delta_weight=0.5,
        tilt_delta_weight=0.5,
    )

    assert slipped.item() > baseline.item()


def test_transport_success_requires_goal_upright_contacts_and_no_slip() -> None:
    goal_dist = torch.tensor([0.02, 0.08, 0.02, 0.02, 0.02], dtype=torch.float32)
    upright = torch.tensor([True, True, False, True, True])
    contacts = torch.tensor([True, True, True, False, True])
    middle = torch.tensor([True, True, True, True, False])
    no_slip = torch.tensor([True, True, True, True, True])

    success = compute_transport_success_mask(
        goal_dist=goal_dist,
        upright_success=upright,
        contact_grasped=contacts,
        middle_grasped=middle,
        no_slip=no_slip,
        goal_dist_threshold=0.04,
    )

    assert success.tolist() == [True, False, False, False, False]
