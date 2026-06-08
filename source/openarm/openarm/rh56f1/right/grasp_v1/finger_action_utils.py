from __future__ import annotations

import torch


def _compute_synergy_target(
    finger_action: torch.Tensor,
    open_pose: torch.Tensor,
    closed_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    action = finger_action.clamp(-1.0, 1.0)
    blend = 0.5 * (action + 1.0)
    target = torch.lerp(
        open_pose.unsqueeze(0).expand_as(action),
        closed_pose.unsqueeze(0).expand_as(action),
        blend,
    )
    return target.clamp(
        lower_limits.unsqueeze(0),
        upper_limits.unsqueeze(0),
    )


def compute_grasp_finger_targets(
    finger_action: torch.Tensor,
    approach_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    return _compute_synergy_target(
        finger_action=finger_action,
        open_pose=approach_pose,
        closed_pose=grasp_pose,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def compute_lift_finger_targets(
    finger_action: torch.Tensor,
    grasp_pose: torch.Tensor,
    full_grip_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    return _compute_synergy_target(
        finger_action=finger_action,
        open_pose=grasp_pose,
        closed_pose=full_grip_pose,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )
