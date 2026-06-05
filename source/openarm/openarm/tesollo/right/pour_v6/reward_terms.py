from __future__ import annotations

import math

import torch


def compute_v3_tilt_reward(
    *,
    source_up_dot_world: torch.Tensor,
    mouth_xy_distance: torch.Tensor,
    pour_align_gate_scale: float = 8.0,
    pour_tilt_target_deg: float = 120.0,
    pour_tilt_sharpness: float = 2.0,
) -> dict[str, torch.Tensor]:
    """v3 tilt reward: pour_aligned_gate × exp(-sharpness × |up_dot - target_cos|).

    pour_aligned_gate = exp(-scale × mouth_xy): pour point가 target 위에 있을 때만 tilt 보상.
    """
    target_cos = math.cos(math.radians(pour_tilt_target_deg))
    pour_aligned_gate = torch.exp(-pour_align_gate_scale * mouth_xy_distance)
    r_tilt = pour_aligned_gate * torch.exp(
        -pour_tilt_sharpness * torch.abs(source_up_dot_world - target_cos)
    )
    return {"r_tilt": r_tilt, "pour_aligned_gate": pour_aligned_gate}


def compute_v3_pour_dist_reward(
    *,
    rho: torch.Tensor,
    pour_warmup: torch.Tensor,
    tilt_amount: torch.Tensor,
    z_gate: torch.Tensor,
    mouth_xy_distance: torch.Tensor,
    pour_dist_exp_scale: float = 8.0,
    weight_pour_dist: float = 12.0,
    pour_point_tilt_threshold_deg: float = 15.0,
) -> torch.Tensor:
    """v3 pour distance reward: ρ × pour_warmup × initial_tilt_gate × z_gate × exp(...).

    initial_tilt_gate: 15° 이상 tilt 후에만 활성 → r_tilt와 충돌 제거.
    """
    tilt_threshold_norm = (1.0 - math.cos(math.radians(pour_point_tilt_threshold_deg))) / 2.0
    initial_tilt_gate = (tilt_amount / max(tilt_threshold_norm, 1e-6)).clamp(0.0, 1.0)
    return (
        rho
        * pour_warmup
        * z_gate
        * initial_tilt_gate
        * weight_pour_dist
        * torch.exp(-pour_dist_exp_scale * mouth_xy_distance)
    )
