from __future__ import annotations

import torch


def resolve_grasp_delta_scale(default_scale: float, adr_delta_scale: float | None) -> float:
    if adr_delta_scale is None:
        return float(default_scale)
    return float(adr_delta_scale)


def _compute_reference_plus_delta_target(
    reference_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float,
    delta_mask: torch.Tensor,
) -> torch.Tensor:
    delta = finger_action * float(delta_scale) * delta_mask.unsqueeze(0)
    return (reference_pos + delta).clamp(
        lower_limits.unsqueeze(0),
        upper_limits.unsqueeze(0),
    )


def compute_grasp_finger_targets(
    current_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float,
    delta_mask: torch.Tensor,
) -> torch.Tensor:
    return _compute_reference_plus_delta_target(
        reference_pos=current_pos,
        finger_action=finger_action,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        delta_scale=delta_scale,
        delta_mask=delta_mask,
    )


def compute_lift_finger_targets(
    lift_reference_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float,
    delta_mask: torch.Tensor,
) -> torch.Tensor:
    return _compute_reference_plus_delta_target(
        reference_pos=lift_reference_pos,
        finger_action=finger_action,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        delta_scale=delta_scale,
        delta_mask=delta_mask,
    )
