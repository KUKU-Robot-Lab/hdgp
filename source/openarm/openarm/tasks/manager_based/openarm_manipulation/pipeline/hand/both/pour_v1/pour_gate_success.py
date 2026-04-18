from __future__ import annotations

import torch


def compute_gate_terms(
    *,
    cfg,
    mouth_xy_distance: torch.Tensor,
    mouth_z_clearance: torch.Tensor,
    directional_tilt_cos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    g_align_xy = torch.exp(-cfg.reward_gate_xy_scale * mouth_xy_distance)
    g_clear = torch.sigmoid(
        cfg.reward_gate_clear_scale * (mouth_z_clearance - cfg.reward_clearance_min)
    )
    g_tilt = torch.sigmoid(
        cfg.reward_gate_tilt_scale * (directional_tilt_cos - cfg.reward_tilt_cos_min)
    )
    g_ready = g_align_xy * g_clear
    g_pour = g_ready * g_tilt
    return {
        "g_align_xy": g_align_xy,
        "g_clear": g_clear,
        "g_tilt": g_tilt,
        "g_ready": g_ready,
        "g_pour": g_pour,
    }


def compute_success_terms(
    *,
    cfg,
    bead_in_target_fraction: torch.Tensor,
    spill_ratio: torch.Tensor,
    g_pour: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    source_empty_steps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    success_now = (
        (bead_in_target_fraction >= cfg.assist_success_fill_ratio)
        & (spill_ratio <= cfg.assist_success_spill_max)
        & (g_pour > 0.05)
    )
    is_last_step = episode_length_buf >= (max_episode_length - 1)
    is_source_ending = source_empty_steps >= cfg.source_empty_hold_steps
    episode_ending = is_last_step | is_source_ending
    return {
        "success_now": success_now,
        "is_last_step": is_last_step,
        "is_source_ending": is_source_ending,
        "episode_ending": episode_ending,
    }
