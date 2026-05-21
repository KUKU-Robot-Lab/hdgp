from __future__ import annotations

import torch


def select_warmstart_success(
    lifted: torch.Tensor,
    grasped: torch.Tensor,
    up_z: torch.Tensor,
    j7: torch.Tensor,
    *,
    j7_min: float = 0.20,
    j7_max: float = 1.50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return warmstart cache candidates and diagnostic masks.

    Cup uprightness is intentionally diagnostic-only here: the v7_2 grasp
    checkpoint can keep a lifted grasp while reporting up_z below the old 0.90
    gate, so using it as a hard filter starves the cache.
    """
    j7_in_range = (j7 >= j7_min) & (j7 <= j7_max)
    diagnostics = {
        "j7_in_range": j7_in_range,
        "upright_0_7": up_z > 0.70,
        "upright_0_9": up_z > 0.90,
    }
    return lifted & grasped & j7_in_range, diagnostics
