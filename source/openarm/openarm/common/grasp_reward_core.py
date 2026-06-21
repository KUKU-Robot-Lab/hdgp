from __future__ import annotations

import torch

from openarm.common.grasp_v2_contract import GRASP_V2_REWARD_TERMS


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_grasp_reward_terms(
    *,
    num_tip_contacts: torch.Tensor,
    tip_contact_frac: torch.Tensor,
    full_tip_contact: torch.Tensor,
    contact_persistence_frac: torch.Tensor,
    palm_to_cup_dist: torch.Tensor,
    fingertip_side_dist: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_xy_displacement: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    upright_quality: torch.Tensor,
    lift_latched: torch.Tensor,
    action_delta_norm: torch.Tensor,
    stabilize_reward_gate: torch.Tensor | None = None,
    success_now: torch.Tensor | None = None,
    stable: torch.Tensor | None = None,
    stability_quality: torch.Tensor | None = None,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Shared grasp v2 reward used by Tesollo and RH56F1 grasp tasks."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    contact_persistence_frac = contact_persistence_frac.clamp(0.0, 1.0)

    lifted_bool = cup_height_delta >= _cfg_float(cfg, "lift_success_height", 0.04)
    lifted_gate = lifted_bool.float()
    reward_upright_success = cup_tilt_deg <= _cfg_float(cfg, "success_upright_max_deg", 20.0)
    final_upright_success = cup_tilt_deg <= _cfg_float(cfg, "stabilize_upright_max_deg", 5.0)

    stabilize_gate = (
        lift_gate if stabilize_reward_gate is None else stabilize_reward_gate.float()
    )

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
        0.25 * tip_contact_frac + 0.35 * full_tip + 0.40 * contact_persistence_frac
    )
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * full_tip
        * cup_height_delta
        * upright_quality
    )
    action_quality = torch.exp(
        -_cfg_float(cfg, "stabilize_action_sharpness", 1.0) * action_delta_norm
    )
    if stable is None:
        stable_bool = torch.ones_like(cup_height_delta, dtype=torch.bool)
    else:
        stable_bool = stable.bool()
    stable_gate = stable_bool.float()
    stability_quality_f = (
        stable_gate if stability_quality is None else stability_quality
    ).clamp(min=0.0, max=1.0)
    post_lift_contact_loss = (
        _cfg_float(cfg, "post_lift_contact_loss_weight", 0.0)
        * lift_gate
        * lifted_gate
        * torch.relu(1.0 - tip_contact_frac)
    )
    stabilize = (
        _cfg_float(cfg, "stabilize_weight", 0.0)
        * lift_gate
        * lifted_gate
        * full_tip
        * upright_quality
        * action_quality
    )
    stability = (
        _cfg_float(cfg, "stability_reward_weight", 0.0)
        * stabilize_gate
        * lifted_gate
        * full_tip
        * upright_quality
        * stability_quality_f
    )
    success_now_bool = (
        torch.zeros_like(lift_latched, dtype=torch.bool)
        if success_now is None
        else success_now.bool()
    )
    success_bonus = _cfg_float(cfg, "success_bonus_weight", 0.0) * success_now_bool.float()
    action_smooth = _cfg_float(cfg, "action_smooth_weight", 0.0) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "stabilize": stabilize,
        "success_bonus": success_bonus,
        "post_lift_contact_loss": post_lift_contact_loss,
        "action_smooth": action_smooth,
        "stability": stability,
    }
    missing_terms = set(GRASP_V2_REWARD_TERMS) - set(terms)
    if missing_terms:
        raise RuntimeError(f"missing grasp v2 reward terms: {sorted(missing_terms)}")
    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "full_tip_contact": full_tip,
        "contact_persistence": contact_persistence_frac,
        "upright_success": reward_upright_success.float(),
        "final_upright_success": final_upright_success.float(),
        "success_now": success_now_bool.float(),
        "stability_quality": stability_quality_f,
        "stable": stable_gate,
        "action_quality": action_quality,
    }
    total = torch.nan_to_num(
        sum(terms.values()),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return total, terms, gates
