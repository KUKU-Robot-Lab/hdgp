"""Unit tests for the friction-aware no-slip minimal-force grip objective.

설계 의도 (grasp_adapt Phase 1):
  적응형 파지의 목적함수 = "안 미끄러지는(secure) 최소 힘".
  - r_secure   : 컵이 정지(=미끄러지지 않음)할수록 보상. slip 결과를 직접 측정 → friction-agnostic.
  - r_efficient: 법선력(ΣF_n / cup_weight)에 비례한 비용 → 힘을 최소로 압박.
  - r_drop     : 들고 있다 떨어뜨리거나 접촉을 잃으면 페널티.
  세 항의 평형 = "안 미끄러지는 최소 힘" → mass·friction 적응이 emergent.

핵심 불변식: 단순 slip 모델(force_ratio < ratio_min 이면 미끄러짐) 하에서
  보상을 최대화하는 최적 힘이 cup_weight에 선형 비례한다 (= mass-adaptive force).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openarm.common.grasp_adaptive_core import compute_adaptive_grip_terms


@dataclass
class AdaptiveCfg:
    secure_weight: float = 8.0
    secure_slip_sharpness: float = 30.0       # k_v: cup_speed → secure_quality 감쇠
    force_efficiency_weight: float = 1.0       # w_force: ratio당 비용
    drop_penalty_weight: float = 8.0
    force_ratio_cost_cap: float = 6.0          # ratio 비용 상한 (blowup 방지)


def _full_hold(n: int):
    """5접촉·lift latched·들림 상태(=hold_gate 1)의 기본 입력 묶음."""
    ones = torch.ones(n)
    return dict(
        tip_contact_frac=ones.clone(),
        lift_gate=ones.clone(),
        lifted_gate=ones.clone(),
        full_tip_contact=ones.clone(),
    )


def _call(cfg=None, n=1, **overrides):
    cfg = cfg or AdaptiveCfg()
    base = _full_hold(n)
    base.update(
        grip_normal_force=torch.full((n,), 4.0),
        cup_weight=torch.full((n,), 2.0),
        cup_speed=torch.zeros(n),
    )
    base.update(overrides)
    return compute_adaptive_grip_terms(cfg=cfg, **base)


# ---------------------------------------------------------------------------
# r_secure: slip(=cup_speed) 직접 측정
# ---------------------------------------------------------------------------
def test_secure_max_when_cup_still_and_holding():
    total, terms = _call(cup_speed=torch.zeros(1))
    # 정지 컵 → secure_quality == 1 → r_secure == secure_weight
    assert torch.allclose(terms["secure_quality"], torch.ones(1), atol=1e-6)
    assert torch.allclose(terms["r_secure"], torch.tensor([8.0]), atol=1e-5)


def test_secure_monotonic_decreasing_in_cup_speed():
    speeds = torch.tensor([0.0, 0.05, 0.1, 0.3])
    _, terms = _call(n=4, cup_speed=speeds, **{
        k: torch.ones(4) for k in ("tip_contact_frac", "lift_gate", "lifted_gate", "full_tip_contact")
    }, grip_normal_force=torch.full((4,), 4.0), cup_weight=torch.full((4,), 2.0))
    sq = terms["secure_quality"]
    assert torch.all(sq[1:] < sq[:-1])          # 단조 감소
    assert torch.all((sq >= 0.0) & (sq <= 1.0))  # [0,1] 범위


def test_quality_terms_zero_when_not_holding():
    # 들리지 않음(lifted_gate=0) → hold_gate=0 → secure/efficient 0
    total, terms = _call(lifted_gate=torch.zeros(1))
    assert torch.allclose(terms["r_secure"], torch.zeros(1), atol=1e-7)
    assert torch.allclose(terms["r_efficient"], torch.zeros(1), atol=1e-7)


# ---------------------------------------------------------------------------
# r_efficient: 힘을 최소로 (질량 정규화 비용)
# ---------------------------------------------------------------------------
def test_efficient_more_negative_as_force_increases():
    forces = torch.tensor([2.0, 4.0, 8.0])
    _, terms = _call(n=3, grip_normal_force=forces, cup_weight=torch.full((3,), 2.0), **{
        k: torch.ones(3) for k in ("tip_contact_frac", "lift_gate", "lifted_gate", "full_tip_contact")
    }, cup_speed=torch.zeros(3))
    eff = terms["r_efficient"]
    assert torch.all(eff[1:] < eff[:-1])  # 힘↑ → 비용↑(더 음수)


def test_efficient_cost_normalized_by_weight():
    # 같은 힘이라도 무거운 컵에서 비용이 더 작아야 한다 (ratio = F/weight).
    _, light = _call(grip_normal_force=torch.tensor([4.0]), cup_weight=torch.tensor([2.0]))
    _, heavy = _call(grip_normal_force=torch.tensor([4.0]), cup_weight=torch.tensor([4.0]))
    assert heavy["r_efficient"] > light["r_efficient"]  # 덜 음수


def test_efficient_cost_capped():
    # 극단적 ratio에서도 비용이 cap을 넘지 않는다.
    cfg = AdaptiveCfg()
    _, terms = _call(cfg=cfg, grip_normal_force=torch.tensor([1000.0]), cup_weight=torch.tensor([1.0]))
    min_eff = -cfg.force_efficiency_weight * cfg.force_ratio_cost_cap
    assert terms["r_efficient"].item() >= min_eff - 1e-5


# ---------------------------------------------------------------------------
# r_drop: 떨어뜨림 / 접촉 손실 페널티
# ---------------------------------------------------------------------------
def test_drop_penalty_when_fell_after_lift():
    # lift latched(1) 인데 lifted_gate=0 → 떨어뜨림
    _, terms = _call(lift_gate=torch.ones(1), lifted_gate=torch.zeros(1))
    assert terms["r_drop"].item() < 0.0
    assert torch.allclose(terms["r_drop"], torch.tensor([-8.0]), atol=1e-5)


def test_no_drop_when_secure_hold():
    _, terms = _call()  # full hold, 정지
    assert torch.allclose(terms["r_drop"], torch.zeros(1), atol=1e-7)


def test_drop_penalty_on_contact_loss_while_lifted():
    # 들린 상태(lift&lifted)지만 일부 접촉 상실(full_tip=0, frac<1)
    _, terms = _call(
        lift_gate=torch.ones(1), lifted_gate=torch.ones(1),
        full_tip_contact=torch.zeros(1), tip_contact_frac=torch.tensor([0.4]),
    )
    assert terms["r_drop"].item() < 0.0


# ---------------------------------------------------------------------------
# 핵심 적응 불변식: 보상 최대화 힘이 질량에 선형 비례
# ---------------------------------------------------------------------------
def test_optimal_force_scales_linearly_with_mass():
    """단순 slip 모델 하에서 reward argmax 힘이 cup_weight에 비례하는지 검증.

    slip 모델: force_ratio >= ratio_min 이면 secure(speed 0), 아니면
               cup_speed = (ratio_min - ratio) * slip_scale 로 미끄러짐.
    설계가 옳다면 두 질량 모두 최적 ratio == ratio_min → 최적 force ∝ weight.
    """
    cfg = AdaptiveCfg(secure_weight=8.0, secure_slip_sharpness=30.0,
                      force_efficiency_weight=1.0, force_ratio_cost_cap=6.0)
    ratio_min = 1.5
    slip_scale = 0.5

    def best_force(weight: float) -> float:
        forces = torch.linspace(0.1, 6.0 * weight, 400)
        ratio = forces / weight
        cup_speed = torch.clamp(ratio_min - ratio, min=0.0) * slip_scale
        n = forces.shape[0]
        total, _ = compute_adaptive_grip_terms(
            cfg=cfg,
            grip_normal_force=forces,
            cup_weight=torch.full((n,), weight),
            cup_speed=cup_speed,
            tip_contact_frac=torch.ones(n),
            lift_gate=torch.ones(n),
            lifted_gate=torch.ones(n),
            full_tip_contact=torch.ones(n),
        )
        return forces[int(torch.argmax(total))].item()

    f_light = best_force(2.0)
    f_heavy = best_force(4.0)

    # 최적 ratio ≈ ratio_min (양 질량 공통)
    assert abs(f_light / 2.0 - ratio_min) < 0.25
    assert abs(f_heavy / 4.0 - ratio_min) < 0.25
    # 최적 force가 질량에 비례 (heavy ≈ 2 × light)
    assert abs(f_heavy - 2.0 * f_light) < 0.5 * f_light


def test_returns_total_equals_sum_of_terms():
    total, terms = _call()
    expected = terms["r_secure"] + terms["r_efficient"] + terms["r_drop"]
    assert torch.allclose(total, expected, atol=1e-6)
