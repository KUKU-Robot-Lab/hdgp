from __future__ import annotations

import torch


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_grasp_reward_terms(
    *,
    num_tip_contacts: torch.Tensor,
    tip_contact_frac: torch.Tensor,
    full_tip_contact: torch.Tensor,
    palm_to_cup_dist: torch.Tensor,
    fingertip_side_dist: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_xy_displacement: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    upright_quality: torch.Tensor,
    lift_latched: torch.Tensor,
    action_delta_norm: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Shared 5-tip grasp reward used by Tesollo and RH56F1 grasp tasks."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    full_tip_bool = full_tip_contact.bool()

    lifted_bool = cup_height_delta >= _cfg_float(cfg, "lift_success_height", 0.04)
    lifted_gate = lifted_bool.float()
    upright_success = cup_tilt_deg <= _cfg_float(cfg, "success_upright_max_deg", 20.0)
    success_now = lift_latched & lifted_bool & full_tip_bool & upright_success

    xy_margin = _cfg_float(cfg, "grasp_xy_threshold", 0.0)
    tilt_margin = _cfg_float(cfg, "grasp_upright_threshold_deg", 0.0)
    approach = pre_lift_gate * (
        _cfg_float(cfg, "approach_weight", 0.0)
        * torch.exp(
            -_cfg_float(cfg, "approach_sharpness", 1.0)
            * (palm_to_cup_dist + fingertip_side_dist)
        )
        - _cfg_float(cfg, "approach_xy_penalty_weight", 0.0)
        * torch.relu(cup_xy_displacement - xy_margin)
        - _cfg_float(cfg, "approach_tilt_penalty_weight", 0.0)
        * torch.relu(cup_tilt_deg - tilt_margin)
    )
    grasp = _cfg_float(cfg, "grasp_weight", 0.0) * pre_lift_gate * (
        0.4 * tip_contact_frac + 0.6 * full_tip
    )
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * full_tip
        * cup_height_delta
        * upright_quality
    )
    stability_xy_scale = _cfg_float(
        cfg,
        "stabilize_xy_scale",
        max(_cfg_float(cfg, "grasp_xy_threshold", 0.025), 1e-6),
    )
    stability_quality = torch.exp(-cup_xy_displacement / max(stability_xy_scale, 1e-6))
    action_quality = torch.exp(
        -_cfg_float(cfg, "stabilize_action_sharpness", 1.0) * action_delta_norm
    )
    stabilize = (
        _cfg_float(cfg, "stabilize_weight", 0.0)
        * lift_gate
        * lifted_gate
        * full_tip
        * upright_quality
        * stability_quality
        * action_quality
    )
    success_bonus = _cfg_float(cfg, "success_bonus_weight", 0.0) * success_now.float()
    action_smooth = _cfg_float(cfg, "action_smooth_weight", 0.0) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "stabilize": stabilize,
        "success_bonus": success_bonus,
        "action_smooth": action_smooth,
    }
    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "full_tip_contact": full_tip,
        "upright_success": upright_success.float(),
        "success_now": success_now.float(),
        "stability_quality": stability_quality,
        "action_quality": action_quality,
    }
    total = torch.nan_to_num(
        sum(terms.values()),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return total, terms, gates
