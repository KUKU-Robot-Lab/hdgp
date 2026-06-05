from __future__ import annotations

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
