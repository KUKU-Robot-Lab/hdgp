from __future__ import annotations

import torch


def compute_simple_pour_reward(
    *,
    mouth_xy_distance: torch.Tensor,
    bead_in_target_fraction: torch.Tensor,
    spill_ratio: torch.Tensor,
    rho: torch.Tensor,
    xy_weight: float = 8.0,
    xy_sharpness: float = 80.0,
    capture_weight: float = 300.0,
    spill_weight: float = 80.0,
    spill_capture_coupling: float = 2.0,
    all_beads_bonus_weight: float = 200.0,
    all_beads_eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Simple v5 pouring reward: match pour point XY, then maximize capture under spill."""

    target = bead_in_target_fraction.clamp(0.0, 1.0)
    spill = spill_ratio.clamp(0.0, 1.0)
    gate = rho.clamp(0.0, 1.0)

    r_pour_xy = xy_weight * torch.exp(-xy_sharpness * mouth_xy_distance.pow(2))
    capture = capture_weight * target.pow(3)
    spill_penalty = spill_weight * spill.sqrt()
    capture_gate = (1.0 - spill).clamp(0.0, 1.0).pow(spill_capture_coupling)
    r_capture_spill = capture * capture_gate - spill_penalty
    all_beads_bonus = all_beads_bonus_weight * (target >= 1.0 - all_beads_eps).float()

    total = gate * (r_pour_xy + r_capture_spill + all_beads_bonus)
    return {
        "r_pour_xy": gate * r_pour_xy,
        "r_capture_spill": gate * r_capture_spill,
        "all_beads_bonus": gate * all_beads_bonus,
        "total": total,
    }
