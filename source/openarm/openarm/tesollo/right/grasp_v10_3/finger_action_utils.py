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
    thumb_curl_index: int | None = None,
    thumb_downward_action_scale: float = 1.0,
    thumb_anchor_pose: torch.Tensor | None = None,
    thumb_curl_max_downward_delta: float | None = None,
) -> torch.Tensor:
    constrained_action = finger_action.clone()
    if thumb_curl_index is not None:
        thumb_curl_action = constrained_action[:, thumb_curl_index]
        constrained_action[:, thumb_curl_index] = torch.where(
            thumb_curl_action < 0.0,
            thumb_curl_action * float(thumb_downward_action_scale),
            thumb_curl_action,
        )

    delta = constrained_action * float(delta_scale) * delta_mask.unsqueeze(0)
    target = (reference_pos + delta).clamp(
        lower_limits.unsqueeze(0),
        upper_limits.unsqueeze(0),
    )
    if (
        thumb_curl_index is not None
        and thumb_anchor_pose is not None
        and thumb_curl_max_downward_delta is not None
    ):
        lower_floor = float(thumb_anchor_pose[thumb_curl_index]) - float(thumb_curl_max_downward_delta)
        target[:, thumb_curl_index] = torch.maximum(
            target[:, thumb_curl_index],
            torch.full_like(target[:, thumb_curl_index], lower_floor),
        )
    return target


def _clamp_indices_to_reference_delta(
    target: torch.Tensor,
    reference_pos: torch.Tensor,
    indices: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    selected_ref = reference_pos[:, indices]
    selected_target = target[:, indices].clamp(
        selected_ref - float(max_delta),
        selected_ref + float(max_delta),
    )
    target[:, indices] = selected_target
    return target


def compute_grasp_finger_targets(
    current_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float,
    delta_mask: torch.Tensor,
    thumb_curl_index: int | None = None,
    thumb_downward_action_scale: float = 1.0,
    thumb_anchor_pose: torch.Tensor | None = None,
    thumb_curl_max_downward_delta: float | None = None,
) -> torch.Tensor:
    return _compute_reference_plus_delta_target(
        reference_pos=current_pos,
        finger_action=finger_action,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        delta_scale=delta_scale,
        delta_mask=delta_mask,
        thumb_curl_index=thumb_curl_index,
        thumb_downward_action_scale=thumb_downward_action_scale,
        thumb_anchor_pose=thumb_anchor_pose,
        thumb_curl_max_downward_delta=thumb_curl_max_downward_delta,
    )


def compute_lift_finger_targets(
    lift_reference_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float,
    delta_mask: torch.Tensor,
    thumb_curl_index: int | None = None,
    thumb_downward_action_scale: float = 1.0,
    thumb_anchor_pose: torch.Tensor | None = None,
    thumb_curl_max_downward_delta: float | None = None,
    thumb_lift_indices: torch.Tensor | None = None,
    thumb_lift_max_delta: float | None = None,
) -> torch.Tensor:
    target = _compute_reference_plus_delta_target(
        reference_pos=lift_reference_pos,
        finger_action=finger_action,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        delta_scale=delta_scale,
        delta_mask=delta_mask,
        thumb_curl_index=thumb_curl_index,
        thumb_downward_action_scale=thumb_downward_action_scale,
        thumb_anchor_pose=thumb_anchor_pose,
        thumb_curl_max_downward_delta=thumb_curl_max_downward_delta,
    )
    if thumb_lift_indices is not None and thumb_lift_max_delta is not None:
        target = _clamp_indices_to_reference_delta(
            target=target,
            reference_pos=lift_reference_pos,
            indices=thumb_lift_indices.to(device=target.device, dtype=torch.long),
            max_delta=thumb_lift_max_delta,
        )
    return target
