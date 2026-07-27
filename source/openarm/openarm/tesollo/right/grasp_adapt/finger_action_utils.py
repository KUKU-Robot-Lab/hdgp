from __future__ import annotations

import torch

# Phase 0 정비: grasp_v1/grasp_v10_3에서 copy된 미사용 lift-retarget 함수군
# (compute_grasp_finger_targets / compute_lift_finger_targets / _compute_reference_plus_delta_target
#  / _clamp_indices_to_reference_delta / resolve_grasp_delta_scale) 제거. env가 쓰는 활성 함수만 유지.


def compute_full_range_finger_targets(
    finger_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    active_mask: torch.Tensor,
    fixed_pose: torch.Tensor,
) -> torch.Tensor:
    """active 관절은 action[-1,1]→[lower,upper] full-range 절대 제어, 나머지(abduction)는 고정.

    감싸기 preset(HAND_GRASP_POSE) 갇힘을 풀고 정책이 손끝 파지 형상까지 자유 탐색하게 한다.

    Args:
        finger_action: (N,D) 정규화 액션 [-1,1].
        lower_limits: (D,) 관절 하한.
        upper_limits: (D,) 관절 상한.
        active_mask: (D,) {0,1} — 1=full-range 제어, 0=abduction 고정.
        fixed_pose: (D,) 비활성(abduction) 관절 고정값.
    Returns:
        (N,D) 손 관절 목표.
    """
    a = finger_action.clamp(-1.0, 1.0)
    full = 0.5 * (a + 1.0) * (upper_limits - lower_limits).unsqueeze(0) + lower_limits.unsqueeze(0)
    return torch.where(active_mask.unsqueeze(0).bool(), full, fixed_pose.unsqueeze(0))


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
