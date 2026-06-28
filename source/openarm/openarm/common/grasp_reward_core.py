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
    # graded contact factor: hard full_tip(>=5) 곱셈 게이트는 TESOLLO에서 검지가
    # 구조적으로 미접촉이라 영원히 0 → lift/stabilize/stability 전부 닫힘. 부분 접촉에도
    # gradient를 주도록 tip_contact_frac(=num_tip/5)로 graded 게이팅한다.
    # success/termination 판정은 별도로 full_tip>=5를 유지(env에서 처리).
    graded_contact = tip_contact_frac.clamp(0.0, 1.0)

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
    lift_height_quality = (
        cup_height_delta
        / max(_cfg_float(cfg, "lift_success_height", 0.04), 1e-6)
    ).clamp(min=0.0, max=1.0)
    # Phase K2: lift height bonus는 4cm 위까지 gradient를 살려야 평형이 4cm 이상에 형성된다.
    # lift_height_quality(4cm clamp)를 그대로 쓰면 4cm 위 gradient=0이라 평형이 ~3.1cm에
    # 고착(test2 확인). bonus 전용 quality를 lift_height_bonus_clamp(기본 1.0=4cm,
    # v10-3은 1.5=6cm)까지 열어 컵이 4cm를 안정적으로 넘게 한다.
    lift_height_bonus_quality = (
        cup_height_delta
        / max(_cfg_float(cfg, "lift_success_height", 0.04), 1e-6)
    ).clamp(min=0.0, max=_cfg_float(cfg, "lift_height_bonus_clamp", 1.0))
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * graded_contact
        * lift_height_quality
        * upright_quality
        # Phase K: contact-독립 height 보상. 풀그립(Phase J)으로 graded_contact가 0.85까지
        # 치솟아 lift 보상(=contact×height)이 2.9cm에서 이미 포화 → 정책이 더 들 유인을
        # 잃고 local optimum 고착. height 자체에 contact와 무관한 보상을 더해 4cm 이상까지
        # 들도록 유도. v1은 weight 0(기본)이라 영향 없음.
        + _cfg_float(cfg, "lift_height_bonus_weight", 0.0)
        * lift_gate
        * lift_height_bonus_quality
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
        * graded_contact
        * upright_quality
        * action_quality
    )
    stability = (
        _cfg_float(cfg, "stability_reward_weight", 0.0)
        * stabilize_gate
        * lifted_gate
        * graded_contact
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
