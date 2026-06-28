from __future__ import annotations

import math

import torch


def compute_bounded_force_smooth_penalty(
    force_delta_norm: torch.Tensor,
    weight: float,
    penalty_cap: float,
) -> torch.Tensor:
    squared = force_delta_norm.pow(2)
    cap = max(float(penalty_cap), 1e-6)
    bounded = cap * (1.0 - torch.exp(-squared / cap))
    return -float(weight) * bounded


def compute_upright_success_mask(
    cup_z_cos: torch.Tensor,
    threshold_deg: float,
) -> torch.Tensor:
    threshold_cos = math.cos(math.radians(float(threshold_deg)))
    return cup_z_cos >= threshold_cos


def compute_lift_readiness(
    num_contacts: torch.Tensor,
    is_grasp_phase: torch.Tensor,
    previous_hold_count: torch.Tensor,
    previous_latched: torch.Tensor,
    min_contacts: int,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """방향B(v1 정렬): 접촉수 + hold만으로 lift 진입 판정 (body_band/height/vel 게이트 제거)."""
    lift_contact_now = num_contacts >= int(min_contacts)
    next_hold_count = torch.where(
        lift_contact_now & is_grasp_phase,
        previous_hold_count + 1,
        torch.where(
            previous_latched,
            previous_hold_count,
            torch.zeros_like(previous_hold_count),
        ),
    )
    lift_contact_ready_now = next_hold_count >= int(hold_steps)
    next_latched = previous_latched | lift_contact_ready_now
    return next_hold_count, lift_contact_ready_now, next_latched


def compute_tesollo_prelift_lift_readiness(
    *,
    num_contacts: torch.Tensor,
    is_close_grasp_phase: torch.Tensor,
    tip_local_z_mean: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_lin_vel_norm: torch.Tensor,
    previous_hold_count: torch.Tensor,
    previous_latched: torch.Tensor,
    min_contacts: int,
    hold_steps: int,
    body_local_z_min: float,
    body_local_z_max: float,
    max_cup_height_delta: float,
    cup_lin_vel_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Gate Tesollo lift entry on body-height contact, not rim-hook lifting."""

    full_contact = num_contacts >= int(min_contacts)
    body_band = (
        (tip_local_z_mean >= float(body_local_z_min))
        & (tip_local_z_mean <= float(body_local_z_max))
    )
    prelift_height_ok = cup_height_delta <= float(max_cup_height_delta)
    prelift_velocity_ok = cup_lin_vel_norm <= float(cup_lin_vel_threshold)
    ready_candidate = (
        is_close_grasp_phase.bool()
        & full_contact
        & body_band
        & prelift_height_ok
        & prelift_velocity_ok
    )
    next_hold_count = torch.where(
        ready_candidate,
        previous_hold_count + 1,
        torch.where(
            previous_latched,
            previous_hold_count,
            torch.zeros_like(previous_hold_count),
        ),
    )
    ready_now = next_hold_count >= int(hold_steps)
    next_latched = previous_latched | ready_now
    gates = {
        "full_contact": full_contact.float(),
        "body_band": body_band.float(),
        "prelift_height_ok": prelift_height_ok.float(),
        "prelift_velocity_ok": prelift_velocity_ok.float(),
        "rim_contact_proxy": (tip_local_z_mean > float(body_local_z_max)).float(),
        "ready_candidate": ready_candidate.float(),
    }
    return next_hold_count, ready_now, next_latched, gates


def compute_middle_contact_gate(
    middle_binary_contact: torch.Tensor,
    min_middle_contacts: int,
) -> torch.Tensor:
    return middle_binary_contact.sum(dim=-1) >= int(min_middle_contacts)


def compute_thumb_pose_anchor_reward(
    thumb_joint_pos: torch.Tensor,
    thumb_reference_pose: torch.Tensor,
    weight: float,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = thumb_reference_pose.unsqueeze(0)
    error = (thumb_joint_pos - reference).pow(2).mean(dim=-1).sqrt()
    reward = float(weight) * torch.exp(-float(sharpness) * error)
    return reward, error


def compute_thumb_downward_slide_penalty(
    thumb_tip_pos: torch.Tensor,
    grasp_center: torch.Tensor,
    z_margin: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    downward_delta = torch.relu((grasp_center[:, 2] - thumb_tip_pos[:, 2]) - float(z_margin))
    penalty = -float(weight) * downward_delta
    return penalty, downward_delta


def compute_thumb_tip_direction_reward(
    thumb_distal_pos: torch.Tensor,
    thumb_tip_pos: torch.Tensor,
    grasp_center: torch.Tensor,
    weight: float,
    sharpness: float,
    distance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    thumb_axis = thumb_tip_pos - thumb_distal_pos
    target_axis = grasp_center - thumb_tip_pos
    thumb_axis = thumb_axis / thumb_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    target_axis = target_axis / target_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    alignment = (thumb_axis * target_axis).sum(dim=-1).clamp(-1.0, 1.0)
    alignment_01 = 0.5 * (alignment + 1.0)
    direction_reward = torch.exp(-float(sharpness) * (1.0 - alignment_01))

    scale = max(float(distance_scale), 1e-6)
    tip_dist = (thumb_tip_pos - grasp_center).norm(dim=-1)
    distance_gate = torch.exp(-tip_dist / scale)

    reward = float(weight) * direction_reward * distance_gate
    direction_error = 1.0 - alignment_01
    return reward, alignment, direction_error


def compute_grasp_shape_consistency_reward(
    hand_joint_pos: torch.Tensor,
    reference_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    active_mask: torch.Tensor,
    weight: float,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_range = (upper_limits - lower_limits).clamp(min=1e-6)
    normalized_delta = (hand_joint_pos - reference_pose.unsqueeze(0)) / joint_range.unsqueeze(0)
    masked_delta = normalized_delta * active_mask.unsqueeze(0)
    denom = active_mask.sum().clamp(min=1.0)
    error = masked_delta.pow(2).sum(dim=-1).div(denom).sqrt()
    reward = float(weight) * torch.exp(-float(sharpness) * error)
    return reward, error
