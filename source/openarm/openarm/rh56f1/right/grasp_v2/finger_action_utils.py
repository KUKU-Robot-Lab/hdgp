from __future__ import annotations

import torch


def compute_absolute_finger_targets(
    finger_action: torch.Tensor,
    open_pose: torch.Tensor,
    closed_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    """Map five per-finger actions to bounded absolute targets for 20 joints."""
    blend = 0.5 * (finger_action.clamp(-1.0, 1.0) + 1.0)
    blend_20 = blend.repeat_interleave(4, dim=1)
    target = torch.lerp(
        open_pose.unsqueeze(0).expand_as(blend_20),
        closed_pose.unsqueeze(0).expand_as(blend_20),
        blend_20,
    )
    return target.clamp(lower_limits.unsqueeze(0), upper_limits.unsqueeze(0))


def compute_grasp_finger_targets(
    finger_action: torch.Tensor,
    approach_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    return compute_absolute_finger_targets(
        finger_action, approach_pose, grasp_pose, lower_limits, upper_limits
    )


def compute_lift_finger_targets(
    finger_action: torch.Tensor,
    grasp_pose: torch.Tensor,
    full_grip_pose: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    return compute_absolute_finger_targets(
        finger_action, grasp_pose, full_grip_pose, lower_limits, upper_limits
    )
