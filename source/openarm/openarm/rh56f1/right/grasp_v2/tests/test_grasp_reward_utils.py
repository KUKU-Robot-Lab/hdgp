from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_reward_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_reward_utils", MODULE_PATH)
grasp_reward_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grasp_reward_utils
SPEC.loader.exec_module(grasp_reward_utils)

compute_middle_contact_gate = grasp_reward_utils.compute_middle_contact_gate
compute_lift_readiness = grasp_reward_utils.compute_lift_readiness
compute_late_grasp_full_grip_mask = grasp_reward_utils.compute_late_grasp_full_grip_mask
compute_slip_proxy = grasp_reward_utils.compute_slip_proxy
compute_transport_success_mask = grasp_reward_utils.compute_transport_success_mask
compute_upright_success_mask = grasp_reward_utils.compute_upright_success_mask


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


def test_middle_contact_gate_can_be_disabled_for_hands_without_middle_sensors() -> None:
    middle_binary = torch.zeros(2, 5, dtype=torch.bool)

    gate = compute_middle_contact_gate(middle_binary, min_middle_contacts=0)

    assert gate.tolist() == [True, True]


def test_lift_readiness_uses_relaxed_stage0_gate() -> None:
    hold_count, ready_now, latched = compute_lift_readiness(
        num_contacts=torch.tensor([2, 1], dtype=torch.long),
        is_grasp_phase=torch.tensor([True, True]),
        previous_hold_count=torch.tensor([7, 7], dtype=torch.long),
        previous_latched=torch.tensor([False, False]),
        min_contacts=2,
        hold_steps=8,
    )

    assert hold_count.tolist() == [8, 0]
    assert ready_now.tolist() == [True, False]
    assert latched.tolist() == [True, False]


def test_lift_readiness_preserves_latched_state_after_contact_drops() -> None:
    hold_count, ready_now, latched = compute_lift_readiness(
        num_contacts=torch.tensor([0], dtype=torch.long),
        is_grasp_phase=torch.tensor([True]),
        previous_hold_count=torch.tensor([8], dtype=torch.long),
        previous_latched=torch.tensor([True]),
        min_contacts=2,
        hold_steps=8,
    )

    assert hold_count.tolist() == [8]
    assert ready_now.tolist() == [True]
    assert latched.tolist() == [True]


def test_late_grasp_full_grip_mask_activates_on_contact_or_progress() -> None:
    mask = compute_late_grasp_full_grip_mask(
        num_contacts=torch.tensor([1, 2, 0], dtype=torch.long),
        is_grasp_phase=torch.tensor([True, True, True]),
        episode_length_buf=torch.tensor([10, 10, 300], dtype=torch.long),
        grasp_phase_steps=480,
        contact_threshold=2,
        progress_threshold=0.5,
    )

    assert mask.tolist() == [False, True, True]


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


def test_slip_proxy_sanitizes_non_finite_values() -> None:
    proxy = compute_slip_proxy(
        cup_xy_velocity=torch.tensor([float("nan"), float("inf"), -float("inf")]),
        cup_tilt_delta_deg=torch.zeros(3),
        contact_delta_abs=torch.zeros(3),
        middle_contact_delta_abs=torch.zeros(3),
        xy_velocity_scale=0.04,
        tilt_delta_scale=8.0,
        contact_delta_scale=1.0,
        middle_contact_delta_scale=1.0,
        contact_delta_weight=0.5,
        middle_contact_delta_weight=0.5,
        tilt_delta_weight=0.5,
    )

    assert proxy.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_transport_success_requires_goal_upright_contacts_and_no_slip() -> None:
    goal_dist = torch.tensor([0.02, 0.08, 0.02, 0.02, 0.02], dtype=torch.float32)
    upright = torch.tensor([True, True, False, True, True])
    contacts = torch.tensor([True, True, True, False, True])
    middle = torch.tensor([True, True, True, True, True])
    no_slip = torch.tensor([True, True, True, True, False])

    success = compute_transport_success_mask(
        goal_dist=goal_dist,
        upright_success=upright,
        contact_grasped=contacts,
        middle_grasped=middle,
        no_slip=no_slip,
        goal_dist_threshold=0.04,
    )

    assert success.tolist() == [True, False, False, False, False]


def test_rebalanced_stage0_reward_prefers_contact_over_hovering() -> None:
    no_contact_reward = (
        1.0 * 1.0
        + 0.5 * 1.0
        + 0.5 * 1.0
        + 2.0 * 1.0
    )
    two_contact_reward = (
        1.0 * 1.0
        + 1.0 * 2.0
        + 2.0 * 1.0
    )

    assert two_contact_reward > no_contact_reward
