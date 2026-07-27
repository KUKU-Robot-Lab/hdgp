"""Friction-aware no-slip minimal-force grip objective (grasp_adapt Phase 1).

적응형 파지의 목적함수 코어. lift/stabilize 보상이 아니라 **"안 미끄러지는 최소 힘"**을
직접 목적으로 둔다. 세 항의 평형점에서 mass·friction 적응이 emergent 한다.

  r_secure    : 컵 속도(cup_speed)가 낮을수록(=미끄러지지 않음) 보상. slip 결과를 직접
                측정하므로 μ를 몰라도 friction-aware. hold 구간에서만 활성.
  r_efficient : 법선력 비율(ΣF_n / cup_weight)에 비례한 비용 → 힘을 최소로 압박.
                질량 정규화라 같은 힘이라도 무거운 컵에서 비용이 작다.
  r_drop      : lift latch 후 떨어뜨리거나(lifted 해제) 들린 채 접촉을 잃으면 페널티.

  cup_weight 는 reward 계산 전용 privileged 값이다 (actor obs는 질량 hidden — tactile 추론).
"""

from __future__ import annotations

import torch


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_adaptive_grip_terms(
    *,
    grip_normal_force: torch.Tensor,
    cup_weight: torch.Tensor,
    cup_speed: torch.Tensor,
    tip_contact_frac: torch.Tensor,
    lift_gate: torch.Tensor,
    lifted_gate: torch.Tensor,
    full_tip_contact: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Friction-aware no-slip minimal-force grip reward.

    Args:
        grip_normal_force: (N,) 접촉 tip 법선력 합 ΣF_n [N].
        cup_weight: (N,) 컵 무게 mass·g [N] (privileged, ratio 정규화용).
        cup_speed: (N,) 컵 선속도 크기 [m/s] — slip proxy.
        tip_contact_frac: (N,) 접촉 손가락 비율 [0,1].
        lift_gate: (N,) lift latched {0,1}.
        lifted_gate: (N,) 컵이 lift_success_height 이상 {0,1}.
        full_tip_contact: (N,) 안정 파지 게이트 {0,1}. grasp_adapt Phase 1부터는
            precision_grasp_mask(엄지+대향2지)를 넘긴다(하위호환: 다른 태스크는 5접촉 bool).
        cfg: secure_weight, secure_slip_sharpness, force_efficiency_weight,
             drop_penalty_weight, force_ratio_cost_cap 를 갖는 설정 객체.

    Returns:
        (total, terms) — terms 는 r_secure/r_efficient/r_drop/secure_quality/
        force_ratio/hold_gate/total 진단 dict.
    """
    w_secure = _cfg_float(cfg, "secure_weight", 8.0)
    k_v = _cfg_float(cfg, "secure_slip_sharpness", 30.0)
    w_force = _cfg_float(cfg, "force_efficiency_weight", 1.0)
    w_drop = _cfg_float(cfg, "drop_penalty_weight", 8.0)
    cost_cap = _cfg_float(cfg, "force_ratio_cost_cap", 6.0)

    lift_g = lift_gate.float()
    lifted_g = lifted_gate.float()
    full_tip = full_tip_contact.float()

    hold_gate = lift_g * lifted_g * full_tip

    # r_secure: 컵이 정지할수록 1 (미끄러지지 않음)
    secure_quality = torch.exp(-k_v * cup_speed.clamp(min=0.0)).clamp(0.0, 1.0)
    r_secure = w_secure * hold_gate * secure_quality

    # r_efficient: 질량 정규화 힘 비용 (최소 힘 압박)
    force_ratio = grip_normal_force / cup_weight.clamp(min=1e-6)
    force_ratio_cost = force_ratio.clamp(min=0.0, max=cost_cap)
    r_efficient = -w_force * hold_gate * force_ratio_cost

    # r_drop: 떨어뜨림(들었다 lifted 해제) + 들린 채 접촉 손실
    fell = lift_g * (1.0 - lifted_g)
    contact_loss = lift_g * lifted_g * torch.relu(1.0 - tip_contact_frac)
    r_drop = -w_drop * (fell + contact_loss)

    total = r_secure + r_efficient + r_drop

    terms = {
        "r_secure": r_secure,
        "r_efficient": r_efficient,
        "r_drop": r_drop,
        "secure_quality": secure_quality,
        "force_ratio": force_ratio,
        "hold_gate": hold_gate,
        "total": total,
    }
    return total, terms
