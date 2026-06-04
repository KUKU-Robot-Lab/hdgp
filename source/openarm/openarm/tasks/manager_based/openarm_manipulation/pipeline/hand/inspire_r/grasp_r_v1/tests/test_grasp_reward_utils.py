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
