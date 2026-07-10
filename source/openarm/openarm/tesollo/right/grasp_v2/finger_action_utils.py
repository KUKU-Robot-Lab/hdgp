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


def compute_synergy_progress_targets(
    finger_action: torch.Tensor,
    basis: torch.Tensor,
    anchor: torch.Tensor,
    coeff_mins: torch.Tensor,
    coeff_maxs: torch.Tensor,
    open_pose: torch.Tensor,
    grip_pose: torch.Tensor,
) -> torch.Tensor:
    """시너지(eigengrasp) action → 관절별 폐쇄 진행도 목표 p* (N,20)∈[0,1].

    DEXTRAH PCA action space 대응: action (N,5)∈[-1,1] → 계수(mins~maxs 선형)
    → q* = anchor + coeffs·basis → 관절별 open↔grip 축 진행도로 변환.
    per-finger 독립 제어와 달리 PC1 하나가 20관절을 커플링(전지 감김) —
    2지 최소해가 action 공간에서 표현 불가.

    open_pose == anchor (HAND_APPROACH_POSE) 전제. grip==open 인 퇴화 관절
    (외전 _1 등)은 진행도 0 고정(현행과 동일).
    """
    a01 = 0.5 * (finger_action.clamp(-1.0, 1.0) + 1.0)          # (N,5) ∈ [0,1]
    coeffs = coeff_mins.unsqueeze(0) + a01 * (
        coeff_maxs - coeff_mins
    ).unsqueeze(0)                                               # (N,5)
    q_star = anchor.unsqueeze(0) + coeffs @ basis                # (N,20)

    denom = grip_pose - open_pose                                # (20,)
    safe = denom.abs() > 1e-6
    progress = torch.zeros_like(q_star)
    progress[:, safe] = (
        (q_star[:, safe] - open_pose[safe].unsqueeze(0)) / denom[safe].unsqueeze(0)
    ).clamp(0.0, 1.0)
    return progress
