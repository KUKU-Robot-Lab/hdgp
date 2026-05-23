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


def compute_middle_contact_gate(
    middle_binary_contact: torch.Tensor,
    min_middle_contacts: int,
) -> torch.Tensor:
    return middle_binary_contact.sum(dim=-1) >= int(min_middle_contacts)


def compute_lift_height_reward(
    cup_height_delta: torch.Tensor,
    cup_z_cos: torch.Tensor,
    lift_height_cap: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    capped_height = cup_height_delta.clamp(min=0.0, max=float(lift_height_cap))
    uprightness = cup_z_cos.clamp(min=0.0, max=1.0)
    reward = float(weight) * capped_height * uprightness
    return reward, capped_height


def compute_gated_lift_height_reward(
    cup_height_delta: torch.Tensor,
    cup_z_cos: torch.Tensor,
    lift_height_cap: float,
    weight: float,
    grip_ready_gate: torch.Tensor,
    no_slip_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward, capped_height = compute_lift_height_reward(
        cup_height_delta=cup_height_delta,
        cup_z_cos=cup_z_cos,
        lift_height_cap=lift_height_cap,
        weight=weight,
    )
    reward = reward * grip_ready_gate.float() * no_slip_gate.float()
    return reward, capped_height


def compute_grip_ready_gate(
    tip_binary_contact: torch.Tensor,
    middle_binary_contact: torch.Tensor,
    contact_persistence_steps: torch.Tensor,
    force_ratio: torch.Tensor,
    cup_xy_slip: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    min_tip_contacts: int,
    min_middle_contacts: int,
    hold_steps: int,
    min_force_ratio: float,
    slip_threshold: float,
    tilt_threshold_deg: float,
) -> torch.Tensor:
    tip_ready = tip_binary_contact.sum(dim=-1) >= int(min_tip_contacts)
    middle_ready = middle_binary_contact.sum(dim=-1) >= int(min_middle_contacts)
    persistence_ready = contact_persistence_steps >= int(hold_steps)
    force_ready = force_ratio >= float(min_force_ratio)
    slip_ready = cup_xy_slip <= float(slip_threshold)
    tilt_ready = cup_tilt_deg <= float(tilt_threshold_deg)
    return tip_ready & middle_ready & persistence_ready & force_ready & slip_ready & tilt_ready


def compute_full_contact_shaping_reward(
    tip_force_norm: torch.Tensor,
    distal_force_norm: torch.Tensor,
    middle_force_norm: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    link_depth = torch.stack(
        [
            tip_force_norm.clamp(0.0, 1.0),
            distal_force_norm.clamp(0.0, 1.0),
            middle_force_norm.clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    per_finger_depth = link_depth.mean(dim=-1)
    mean_depth = per_finger_depth.mean(dim=-1)
    worst_finger_depth = per_finger_depth.min(dim=-1).values
    full_contact_score = 0.5 * mean_depth + 0.5 * worst_finger_depth
    reward = float(weight) * full_contact_score
    return reward, full_contact_score, worst_finger_depth, mean_depth


def compute_worst_finger_envelope_reward(
    finger_link_pos: torch.Tensor,
    grasp_center: torch.Tensor,
    cup_radius: float,
    weight: float,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shell_error = (
        (finger_link_pos - grasp_center.unsqueeze(1).unsqueeze(2)).norm(dim=-1)
        - float(cup_radius)
    ).abs()
    per_finger_error = shell_error.min(dim=-1).values
    worst_finger_error = per_finger_error.max(dim=-1).values
    reward = float(weight) * torch.exp(-float(sharpness) * worst_finger_error)
    return reward, worst_finger_error, per_finger_error


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


def compute_slip_penalty(
    slip_proxy: torch.Tensor,
    gate: torch.Tensor,
    threshold: float,
    weight: float,
) -> torch.Tensor:
    return -float(weight) * gate.float() * torch.relu(slip_proxy - float(threshold))


def compute_ring_pinky_separation_penalty(
    ring_tip_pos: torch.Tensor,
    pinky_tip_pos: torch.Tensor,
    ring_middle_pos: torch.Tensor,
    pinky_middle_pos: torch.Tensor,
    min_distance: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tip_dist = (ring_tip_pos - pinky_tip_pos).norm(dim=-1)
    middle_dist = (ring_middle_pos - pinky_middle_pos).norm(dim=-1)
    min_dist = torch.minimum(tip_dist, middle_dist)
    penalty = -float(weight) * torch.relu(float(min_distance) - min_dist)
    return penalty, min_dist


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
