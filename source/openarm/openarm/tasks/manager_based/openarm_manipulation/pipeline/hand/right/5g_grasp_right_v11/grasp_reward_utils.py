from __future__ import annotations

import math

import torch


def compute_upright_success_mask(
    cup_z_cos: torch.Tensor,
    threshold_deg: float,
) -> torch.Tensor:
    threshold_cos = math.cos(math.radians(float(threshold_deg)))
    return cup_z_cos >= threshold_cos


def compute_middle_contact_gate(
    middle_binary_contact: torch.Tensor,
    min_middle_contacts: int,
) -> torch.Tensor:
    return middle_binary_contact.sum(dim=-1) >= int(min_middle_contacts)


def compute_slip_proxy(
    cup_xy_velocity: torch.Tensor,
    cup_tilt_delta_deg: torch.Tensor,
    contact_delta_abs: torch.Tensor,
    middle_contact_delta_abs: torch.Tensor,
    xy_velocity_scale: float,
    tilt_delta_scale: float,
    contact_delta_scale: float,
    middle_contact_delta_scale: float,
    contact_delta_weight: float,
    middle_contact_delta_weight: float,
    tilt_delta_weight: float,
) -> torch.Tensor:
    xy_term = cup_xy_velocity / max(float(xy_velocity_scale), 1e-6)
    tilt_term = cup_tilt_delta_deg / max(float(tilt_delta_scale), 1e-6)
    contact_term = contact_delta_abs / max(float(contact_delta_scale), 1e-6)
    middle_contact_term = middle_contact_delta_abs / max(float(middle_contact_delta_scale), 1e-6)
    proxy = (
        xy_term
        + float(tilt_delta_weight) * tilt_term
        + float(contact_delta_weight) * contact_term
        + float(middle_contact_delta_weight) * middle_contact_term
    )
    return torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)


def compute_transport_success_mask(
    goal_dist: torch.Tensor,
    upright_success: torch.Tensor,
    contact_grasped: torch.Tensor,
    middle_grasped: torch.Tensor,
    no_slip: torch.Tensor,
    goal_dist_threshold: float,
) -> torch.Tensor:
    at_goal = goal_dist <= float(goal_dist_threshold)
    return at_goal & upright_success & contact_grasped & middle_grasped & no_slip
