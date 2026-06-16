from __future__ import annotations

import torch


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
    transport_xyz_dist: torch.Tensor | None = None,
    cup_tilt_deg: torch.Tensor,
    upright_quality: torch.Tensor,
    lift_latched: torch.Tensor,
    action_delta_norm: torch.Tensor,
    transport_reward_gate: torch.Tensor | None = None,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Shared 5-tip grasp reward used by Tesollo and RH56F1 grasp tasks."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    full_tip_bool = full_tip_contact.bool()
    contact_persistence_frac = contact_persistence_frac.clamp(0.0, 1.0)

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
        0.25 * tip_contact_frac + 0.35 * full_tip + 0.40 * contact_persistence_frac
    )
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * full_tip
        * cup_height_delta
        * upright_quality
    )
    transport_xyz_scale = _cfg_float(
        cfg,
        "transport_xyz_scale",
        _cfg_float(
            cfg,
            "stabilize_spawn_xy_scale",
            _cfg_float(
                cfg,
                "stabilize_xy_scale",
                max(_cfg_float(cfg, "grasp_xy_threshold", 0.025), 1e-6),
            ),
        ),
    )
    if transport_xyz_dist is None:
        transport_xyz_dist = cup_xy_displacement
    if transport_reward_gate is None:
        transport_reward_gate_f = torch.ones_like(cup_height_delta)
        # transport 개념이 없는 호출자(grasp_adapt 등)는 stabilize 보상을 그대로 유지.
        stabilize_phase_gate = torch.ones_like(cup_height_delta)
    else:
        transport_reward_gate_f = transport_reward_gate.float()
        # transport phase에서는 stabilize(정지+자세) 보상을 끈다 → 이동 유도.
        stabilize_phase_gate = 1.0 - transport_reward_gate_f
    transport_xyz_quality = torch.exp(-transport_xyz_dist / max(transport_xyz_scale, 1e-6))
    transport_height_target = max(
        _cfg_float(
            cfg,
            "transport_height_target_delta",
            _cfg_float(cfg, "lift_success_height", 0.04),
        ),
        1e-6,
    )
    transport_height_quality = (
        cup_height_delta.clamp(min=0.0) / transport_height_target
    ).clamp(max=1.0)
    transport_height_quality = transport_height_quality.pow(
        _cfg_float(cfg, "transport_height_quality_power", 1.0)
    )
    transport_posture_quality = upright_quality.clamp(min=0.0, max=1.0).pow(
        _cfg_float(cfg, "transport_upright_quality_power", 1.0)
    )
    action_quality = torch.exp(
        -_cfg_float(cfg, "stabilize_action_sharpness", 1.0) * action_delta_norm
    )
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
        * transport_height_quality
        * action_quality
        * stabilize_phase_gate
    )
    transport_xyz = (
        _cfg_float(
            cfg,
            "transport_xyz_reward_weight",
            _cfg_float(cfg, "stabilize_spawn_xy_reward_weight", 0.0),
        )
        * lift_gate
        * transport_reward_gate_f
        * lifted_gate
        * full_tip
        * transport_xyz_quality
        * transport_height_quality
        * transport_posture_quality
    )
    success_bonus = _cfg_float(cfg, "success_bonus_weight", 0.0) * success_now.float()
    action_smooth = _cfg_float(cfg, "action_smooth_weight", 0.0) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "post_lift_contact_loss": post_lift_contact_loss,
        "stabilize": stabilize,
        "transport_xyz": transport_xyz,
        "success_bonus": success_bonus,
        "action_smooth": action_smooth,
    }
    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "full_tip_contact": full_tip,
        "contact_persistence": contact_persistence_frac,
        "upright_success": upright_success.float(),
        "success_now": success_now.float(),
        "transport_xyz_quality": transport_xyz_quality,
        "transport_height_quality": transport_height_quality,
        "transport_posture_quality": transport_posture_quality,
        "transport_reward_gate": transport_reward_gate_f,
        "spawn_xy_quality": transport_xyz_quality,
        "stability_quality": transport_xyz_quality,
        "action_quality": action_quality,
    }
    total = torch.nan_to_num(
        sum(terms.values()),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return total, terms, gates
