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
