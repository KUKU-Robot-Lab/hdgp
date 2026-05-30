"""
TDD: pour_right_v3 pour-point pivot reward gates

검증 대상:
  1. initial_tilt_gate  — 15° 이상 tilt 시 r_pour_dist 활성
  2. pour_aligned_gate  — pour_point 정렬도에 비례한 r_tilt 증폭
  3. 두 gate의 stage 순서 — pour_dist가 r_tilt를 막지 않음

수식 (env.py에 추가될 코드):
  _tilt_amount = (1.0 - source_up_dot_world).clamp(0, 2)
  _tilt_threshold = 1.0 - cos(radians(threshold_deg))
  initial_tilt_gate = (_tilt_amount / _tilt_threshold).clamp(0, 1)

  pour_aligned_gate = exp(-pour_align_gate_scale * mouth_xy_dist)
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# 수식 복사 (env.py에 들어갈 것과 동일)
# ---------------------------------------------------------------------------

def compute_initial_tilt_gate(
    source_up_dot_world: torch.Tensor,
    threshold_deg: float = 15.0,
) -> torch.Tensor:
    tilt_amount = ((1.0 - source_up_dot_world) / 2.0).clamp(0.0, 1.0)
    _tilt_threshold_norm = (1.0 - math.cos(math.radians(threshold_deg))) / 2.0
    return (tilt_amount / max(_tilt_threshold_norm, 1e-6)).clamp(0.0, 1.0)


def compute_pour_aligned_gate(
    mouth_xy_dist: torch.Tensor,
    scale: float = 8.0,
) -> torch.Tensor:
    return torch.exp(-scale * mouth_xy_dist)


# ---------------------------------------------------------------------------
# initial_tilt_gate 테스트
# ---------------------------------------------------------------------------

def test_initial_tilt_gate_zero_at_upright():
    """0° 수직 상태에서 gate = 0 → r_pour_dist 완전 비활성."""
    sud = torch.tensor([1.0])  # cos(0°) = 1.0
    gate = compute_initial_tilt_gate(sud, threshold_deg=15.0)
    assert gate.item() == 0.0, f"expected 0.0, got {gate.item()}"


def test_initial_tilt_gate_one_at_threshold():
    """threshold(15°) 정확히 도달 시 gate = 1.0."""
    sud = torch.tensor([math.cos(math.radians(15.0))])
    gate = compute_initial_tilt_gate(sud, threshold_deg=15.0)
    assert abs(gate.item() - 1.0) < 1e-5, f"expected 1.0, got {gate.item()}"


def test_initial_tilt_gate_half_at_midpoint():
    """threshold 절반(7.5°) 에서 gate ≈ 0.5."""
    half_tilt = (1.0 - math.cos(math.radians(15.0))) / 2.0
    sud = torch.tensor([1.0 - half_tilt])
    gate = compute_initial_tilt_gate(sud, threshold_deg=15.0)
    assert abs(gate.item() - 0.5) < 1e-4, f"expected ~0.5, got {gate.item()}"


def test_initial_tilt_gate_saturates_at_high_tilt():
    """90°, 120° 등 큰 tilt에서 gate = 1.0 (포화)."""
    for deg in [45.0, 90.0, 120.0]:
        sud = torch.tensor([math.cos(math.radians(deg))])
        gate = compute_initial_tilt_gate(sud, threshold_deg=15.0)
        assert gate.item() == 1.0, f"at {deg}°: expected 1.0, got {gate.item()}"


def test_initial_tilt_gate_monotone_increasing():
    """tilt 증가 → gate 단조 증가."""
    degs = [0, 5, 10, 15, 30, 60, 90, 120]
    gates = [
        compute_initial_tilt_gate(torch.tensor([math.cos(math.radians(d))]), 15.0).item()
        for d in degs
    ]
    for i in range(len(gates) - 1):
        assert gates[i] <= gates[i + 1], (
            f"non-monotone at {degs[i]}°→{degs[i+1]}°: {gates[i]:.4f}>{gates[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# pour_aligned_gate 테스트
# ---------------------------------------------------------------------------

def test_pour_aligned_gate_one_when_perfectly_aligned():
    """mouth_xy_dist = 0 → gate = 1.0."""
    gate = compute_pour_aligned_gate(torch.tensor([0.0]), scale=8.0)
    assert abs(gate.item() - 1.0) < 1e-6


def test_pour_aligned_gate_decreases_with_distance():
    """거리 증가 → gate 단조 감소."""
    dists = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    gates = [compute_pour_aligned_gate(torch.tensor([d]), 8.0).item() for d in dists]
    for i in range(len(gates) - 1):
        assert gates[i] > gates[i + 1], (
            f"non-decreasing at {dists[i]}→{dists[i+1]}: {gates[i]:.4f}<={gates[i+1]:.4f}"
        )


def test_pour_aligned_gate_range():
    """gate 값은 항상 (0, 1] 범위."""
    dists = torch.linspace(0.0, 0.5, 50)
    gates = compute_pour_aligned_gate(dists, scale=8.0)
    assert gates.min().item() > 0.0
    assert gates.max().item() <= 1.0 + 1e-6


def test_pour_aligned_gate_expected_values():
    """scale=8 기준 특정 거리에서의 기댓값 확인."""
    cases = [
        (0.00, 1.000),
        (0.05, math.exp(-8 * 0.05)),   # ≈ 0.670
        (0.10, math.exp(-8 * 0.10)),   # ≈ 0.449
        (0.15, math.exp(-8 * 0.15)),   # ≈ 0.301
    ]
    for dist, expected in cases:
        got = compute_pour_aligned_gate(torch.tensor([dist]), 8.0).item()
        assert abs(got - expected) < 1e-5, f"dist={dist}: expected {expected:.4f}, got {got:.4f}"


# ---------------------------------------------------------------------------
# Stage 순서 테스트 — 핵심 설계 검증
# ---------------------------------------------------------------------------

def test_r_tilt_stronger_when_pour_point_aligned():
    """pour_point가 정렬됐을 때 r_tilt gradient가 더 강해야 한다."""
    weight_tilt = 40.0
    sharpness = 2.0
    target_cos = math.cos(math.radians(120.0))
    sud_90 = torch.tensor([0.0])  # 90° tilt

    r_tilt_base = torch.exp(-sharpness * torch.abs(sud_90 - target_cos))

    # 정렬됨 (mouth_xy=0.02)
    aligned_gate = compute_pour_aligned_gate(torch.tensor([0.02]), 8.0)
    r_tilt_aligned = weight_tilt * aligned_gate * r_tilt_base

    # 미정렬 (mouth_xy=0.15)
    misaligned_gate = compute_pour_aligned_gate(torch.tensor([0.15]), 8.0)
    r_tilt_misaligned = weight_tilt * misaligned_gate * r_tilt_base

    assert r_tilt_aligned.item() > r_tilt_misaligned.item(), (
        f"aligned r_tilt ({r_tilt_aligned.item():.3f}) should > "
        f"misaligned ({r_tilt_misaligned.item():.3f})"
    )


def test_r_pour_dist_inactive_at_zero_tilt():
    """0° tilt에서 r_pour_dist = 0 → r_tilt와 충돌 없음."""
    sud_upright = torch.tensor([1.0])
    initial_gate = compute_initial_tilt_gate(sud_upright, threshold_deg=15.0)

    weight_pour_dist = 12.0
    mouth_xy = torch.tensor([0.05])
    pour_dist_scale = 8.0
    r_pour_dist = initial_gate * weight_pour_dist * torch.exp(-pour_dist_scale * mouth_xy)

    assert r_pour_dist.item() == 0.0, (
        f"r_pour_dist should be 0 at upright, got {r_pour_dist.item()}"
    )


def test_no_conflict_between_r_pour_dist_and_r_tilt_at_low_tilt():
    """15° 이하에서 r_pour_dist 비활성 → r_tilt gradient가 단독으로 작용."""
    for deg in [0.0, 5.0, 10.0]:
        sud = torch.tensor([math.cos(math.radians(deg))])
        initial_gate = compute_initial_tilt_gate(sud, threshold_deg=15.0)
        assert initial_gate.item() < 1.0, f"at {deg}°: r_pour_dist should be weak (gate={initial_gate.item():.3f})"


def test_sequential_stage_activation():
    """단계별 활성화 순서 확인:
    0° → initial_tilt_gate=0, pour_aligned_gate 낮음
    15° → initial_tilt_gate=1, pour_aligned_gate가 r_tilt 결정
    90° → 둘 다 활성, r_tilt gradient 최대
    """
    scale = 8.0
    # 접근 완료 후 mouth_xy ≈ 0.06m 가정
    mouth_xy_after_approach = torch.tensor([0.06])

    stages = [
        (0.0,  "stage2_tilt_start"),
        (15.0, "stage3_pour_point_form"),
        (90.0, "stage4_full_tilt"),
    ]
    prev_combined = None
    for deg, name in stages:
        sud = torch.tensor([math.cos(math.radians(deg))])
        ig = compute_initial_tilt_gate(sud, threshold_deg=15.0)
        ag = compute_pour_aligned_gate(mouth_xy_after_approach, scale)
        # combined: r_pour_dist 활성도 × r_tilt 증폭
        combined = ig.item() * ag.item()
        if prev_combined is not None:
            assert combined >= prev_combined - 1e-6, (
                f"{name}: combined ({combined:.3f}) < prev ({prev_combined:.3f})"
            )
        prev_combined = combined
