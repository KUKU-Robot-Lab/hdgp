from __future__ import annotations

import torch

from .pour_gate_success import compute_success_terms


def compute_assist_reward_terms(
    *,
    cfg,
    mouth_xy_distance: torch.Tensor,
    mouth_z_clearance: torch.Tensor,
    directional_tilt_cos: torch.Tensor,
    mouth_alignment_cos: torch.Tensor,
    g_align_xy: torch.Tensor,
    g_ready: torch.Tensor,
    g_pour: torch.Tensor,
    bead_cross_fraction: torch.Tensor,
    bead_in_target_fraction: torch.Tensor,
    spill_ratio: torch.Tensor,
    left_action_delta: torch.Tensor,
    left_joint_vel_cost: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    source_empty_steps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    approach_xy = torch.exp(-cfg.reward_gate_xy_scale * mouth_xy_distance)
    clearance_score = torch.sigmoid(
        cfg.reward_gate_clear_scale
        * (mouth_z_clearance - cfg.reward_clearance_min)
    )
    tilt_score = torch.clamp(
        (directional_tilt_cos - cfg.reward_tilt_cos_min)
        / max(1.0 - cfg.reward_tilt_cos_min, 1e-6),
        min=0.0,
        max=1.0,
    )
    align_score = 0.5 * (mouth_alignment_cos + 1.0)

    r_approach = cfg.assist_reward_approach_xy * approach_xy
    r_clearance = cfg.assist_reward_clearance * g_align_xy * clearance_score
    r_ready = cfg.assist_reward_ready * g_ready
    r_prepour = g_align_xy * (
        cfg.assist_reward_tilt * tilt_score
        + cfg.assist_reward_align * align_score
    )
    r_pour = g_pour * (
        cfg.assist_reward_cross * bead_cross_fraction
        + cfg.assist_reward_capture * bead_in_target_fraction
    )

    success_terms = compute_success_terms(
        cfg=cfg,
        bead_in_target_fraction=bead_in_target_fraction,
        spill_ratio=spill_ratio,
        g_pour=g_pour,
        episode_length_buf=episode_length_buf,
        max_episode_length=max_episode_length,
        source_empty_steps=source_empty_steps,
    )
    success_now = success_terms["success_now"].float()
    r_success = cfg.assist_reward_success * success_now
    r_terminal_capture = (
        cfg.assist_reward_terminal_capture
        * bead_in_target_fraction
        * success_terms["episode_ending"].float()
    )

    premature_tilt_cost = (1.0 - g_ready) * tilt_score
    total = (
        r_approach
        + r_clearance
        + r_ready
        + r_prepour
        + r_pour
        + r_success
        + r_terminal_capture
        - cfg.assist_reward_spill * spill_ratio
        - cfg.assist_reward_premature_tilt * premature_tilt_cost
        - cfg.assist_reward_left_action_rate * left_action_delta
        - cfg.assist_reward_left_joint_vel * left_joint_vel_cost
    )

    return {
        "approach_xy": approach_xy,
        "clearance_score": clearance_score,
        "tilt_score": tilt_score,
        "align_score": align_score,
        "r_approach": r_approach,
        "r_clearance": r_clearance,
        "r_ready": r_ready,
        "r_prepour": r_prepour,
        "r_pour": r_pour,
        "success_now": success_now,
        "is_last_step": success_terms["is_last_step"],
        "is_source_ending": success_terms["is_source_ending"],
        "episode_ending": success_terms["episode_ending"],
        "r_success": r_success,
        "r_terminal_capture": r_terminal_capture,
        "premature_tilt_cost": premature_tilt_cost,
        "total": total,
    }
