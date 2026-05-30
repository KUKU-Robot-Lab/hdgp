from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "reward_terms.py"
SPEC = importlib.util.spec_from_file_location("reward_terms", MODULE_PATH)
reward_terms = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reward_terms
SPEC.loader.exec_module(reward_terms)

compute_simple_pour_reward = reward_terms.compute_simple_pour_reward
compute_v3_tilt_reward = reward_terms.compute_v3_tilt_reward
compute_v3_pour_dist_reward = reward_terms.compute_v3_pour_dist_reward


def test_pour_xy_reward_increases_as_mouth_xy_error_shrinks() -> None:
    far = torch.tensor([0.20], dtype=torch.float32)
    near = torch.tensor([0.04], dtype=torch.float32)
    zero = torch.zeros(1, dtype=torch.float32)
    one = torch.ones(1, dtype=torch.float32)

    far_terms = compute_simple_pour_reward(
        mouth_xy_distance=far,
        bead_in_target_fraction=zero,
        spill_ratio=zero,
        rho=one,
    )
    near_terms = compute_simple_pour_reward(
        mouth_xy_distance=near,
        bead_in_target_fraction=zero,
        spill_ratio=zero,
        rho=one,
    )

    assert near_terms["r_pour_xy"].item() > far_terms["r_pour_xy"].item()
    assert near_terms["total"].item() > far_terms["total"].item()


def test_capture_reward_is_convex_enough_to_prefer_full_pour_over_stopping_early() -> None:
    mouth = torch.full((1,), 0.04, dtype=torch.float32)
    zero = torch.zeros(1, dtype=torch.float32)
    one = torch.ones(1, dtype=torch.float32)

    quarter = compute_simple_pour_reward(
        mouth_xy_distance=mouth,
        bead_in_target_fraction=torch.tensor([0.25], dtype=torch.float32),
        spill_ratio=zero,
        rho=one,
    )
    full = compute_simple_pour_reward(
        mouth_xy_distance=mouth,
        bead_in_target_fraction=one,
        spill_ratio=zero,
        rho=one,
    )

    assert full["r_capture_spill"].item() > 8.0 * quarter["r_capture_spill"].item()
    assert full["all_beads_bonus"].item() > 0.0


def test_spill_penalty_reduces_capture_spill_reward() -> None:
    mouth = torch.full((1,), 0.04, dtype=torch.float32)
    target = torch.full((1,), 0.75, dtype=torch.float32)
    one = torch.ones(1, dtype=torch.float32)

    clean = compute_simple_pour_reward(
        mouth_xy_distance=mouth,
        bead_in_target_fraction=target,
        spill_ratio=torch.zeros(1, dtype=torch.float32),
        rho=one,
    )
    spilled = compute_simple_pour_reward(
        mouth_xy_distance=mouth,
        bead_in_target_fraction=target,
        spill_ratio=torch.full((1,), 0.25, dtype=torch.float32),
        rho=one,
    )

    assert spilled["r_capture_spill"].item() < clean["r_capture_spill"].item()
    assert spilled["total"].item() < clean["total"].item()


def test_rho_gates_pour_rewards_outside_binary_region() -> None:
    mouth = torch.full((1,), 0.04, dtype=torch.float32)
    target = torch.full((1,), 1.0, dtype=torch.float32)
    spill = torch.zeros(1, dtype=torch.float32)

    gated = compute_simple_pour_reward(
        mouth_xy_distance=mouth,
        bead_in_target_fraction=target,
        spill_ratio=spill,
        rho=torch.zeros(1, dtype=torch.float32),
    )

    assert gated["total"].item() == 0.0


# ── v3 reward design tests ────────────────────────────────────────────────────

def test_v3_pour_aligned_gate_suppresses_tilt_when_mouth_far() -> None:
    """pour_aligned_gate = exp(-scale * mouth_xy): 멀면 r_tilt 억제."""
    import math
    target_cos = math.cos(math.radians(120.0))
    up_dot = torch.tensor([target_cos], dtype=torch.float32)  # 완벽한 tilt

    aligned = compute_v3_tilt_reward(
        source_up_dot_world=up_dot,
        mouth_xy_distance=torch.zeros(1, dtype=torch.float32),
        pour_align_gate_scale=8.0,
        pour_tilt_target_deg=120.0,
        pour_tilt_sharpness=2.0,
    )
    misaligned = compute_v3_tilt_reward(
        source_up_dot_world=up_dot,
        mouth_xy_distance=torch.tensor([0.20], dtype=torch.float32),
        pour_align_gate_scale=8.0,
        pour_tilt_target_deg=120.0,
        pour_tilt_sharpness=2.0,
    )
    # exp(-8*0.20) ≈ 0.20 → misaligned tilt should be << aligned
    assert aligned["r_tilt"].item() > misaligned["r_tilt"].item() * 3.0


def test_v3_tilt_reward_max_at_target_angle() -> None:
    """tilt가 120° 정확히 맞을 때 pour_aligned_gate 안에서 최대."""
    import math
    target_cos = math.cos(math.radians(120.0))
    wrong_cos  = math.cos(math.radians(60.0))
    mouth = torch.zeros(1, dtype=torch.float32)

    at_target = compute_v3_tilt_reward(
        source_up_dot_world=torch.tensor([target_cos]),
        mouth_xy_distance=mouth,
        pour_align_gate_scale=8.0,
        pour_tilt_target_deg=120.0,
        pour_tilt_sharpness=2.0,
    )
    off_target = compute_v3_tilt_reward(
        source_up_dot_world=torch.tensor([wrong_cos]),
        mouth_xy_distance=mouth,
        pour_align_gate_scale=8.0,
        pour_tilt_target_deg=120.0,
        pour_tilt_sharpness=2.0,
    )
    assert at_target["r_tilt"].item() > off_target["r_tilt"].item()


def test_v3_initial_tilt_gate_zero_when_upright() -> None:
    """tilt_amount=0 (직립) → initial_tilt_gate=0 → r_pour_dist=0."""
    r = compute_v3_pour_dist_reward(
        rho=torch.ones(1),
        pour_warmup=torch.tensor(1.0),
        tilt_amount=torch.zeros(1),
        z_gate=torch.ones(1),
        mouth_xy_distance=torch.tensor([0.05]),
        pour_dist_exp_scale=8.0,
        weight_pour_dist=12.0,
        pour_point_tilt_threshold_deg=15.0,
    )
    assert r.item() == 0.0


def test_v3_initial_tilt_gate_active_after_threshold() -> None:
    """tilt_amount > threshold_norm → r_pour_dist > 0."""
    import math
    # threshold_norm = (1 - cos(15°)) / 2 ≈ 0.0171
    threshold_norm = (1.0 - math.cos(math.radians(15.0))) / 2.0
    r = compute_v3_pour_dist_reward(
        rho=torch.ones(1),
        pour_warmup=torch.tensor(1.0),
        tilt_amount=torch.tensor([threshold_norm * 2.0]),  # 2× threshold → gate=1
        z_gate=torch.ones(1),
        mouth_xy_distance=torch.zeros(1),
        pour_dist_exp_scale=8.0,
        weight_pour_dist=12.0,
        pour_point_tilt_threshold_deg=15.0,
    )
    assert r.item() > 0.0


def test_v3_pour_dist_gated_by_rho() -> None:
    """rho=0 → r_pour_dist=0."""
    import math
    threshold_norm = (1.0 - math.cos(math.radians(15.0))) / 2.0
    r = compute_v3_pour_dist_reward(
        rho=torch.zeros(1),
        pour_warmup=torch.tensor(1.0),
        tilt_amount=torch.tensor([threshold_norm * 2.0]),
        z_gate=torch.ones(1),
        mouth_xy_distance=torch.zeros(1),
        pour_dist_exp_scale=8.0,
        weight_pour_dist=12.0,
        pour_point_tilt_threshold_deg=15.0,
    )
    assert r.item() == 0.0
