from __future__ import annotations

import torch

# Phase 0 정비: grasp_v1/grasp_v10_3에서 copy된 미사용 lift-retarget 함수군
# (compute_grasp_finger_targets / compute_lift_finger_targets / _compute_reference_plus_delta_target
#  / _clamp_indices_to_reference_delta / resolve_grasp_delta_scale) 제거. env가 쓰는 활성 함수만 유지.


def compute_preset_residual_finger_targets(
    preset_pos: torch.Tensor,
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    residual_scale: float,
    residual_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map normalized policy action to a small residual around a preset pose."""
    residual = finger_action.clamp(-1.0, 1.0) * float(residual_scale)
    if residual_mask is not None:
        residual = residual * residual_mask.unsqueeze(0)
    target = preset_pos.unsqueeze(0) + residual
    return target.clamp(lower_limits.unsqueeze(0), upper_limits.unsqueeze(0))
