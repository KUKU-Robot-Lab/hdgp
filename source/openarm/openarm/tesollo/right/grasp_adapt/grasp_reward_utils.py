from __future__ import annotations

import math

import torch

# Phase 0 정비: grasp_v10_3/grasp_v11에서 copy된 미사용 reward 헬퍼 6종
# (compute_bounded_force_smooth_penalty / compute_middle_contact_gate /
#  compute_thumb_pose_anchor_reward / compute_thumb_downward_slide_penalty /
#  compute_thumb_tip_direction_reward / compute_grasp_shape_consistency_reward) 제거.
# env가 실제로 import하는 함수만 유지.


def compute_upright_success_mask(
    cup_z_cos: torch.Tensor,
    threshold_deg: float,
) -> torch.Tensor:
    threshold_cos = math.cos(math.radians(float(threshold_deg)))
    return cup_z_cos >= threshold_cos
