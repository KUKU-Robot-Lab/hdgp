from __future__ import annotations

from importlib import import_module

import torch


compute_assist_reward_terms = import_module(
    "openarm.tesollo.both.pour_v1.pour_reward"
).compute_assist_reward_terms


class _Cfg:
    reward_gate_xy_scale = 5.0
    reward_gate_clear_scale = 80.0
    reward_clearance_min = 0.015
    reward_tilt_cos_min = 0.15
    assist_reward_approach_xy = 1.5
    assist_reward_clearance = 0.75
    assist_reward_ready = 0.5
    assist_reward_tilt = 0.75
    assist_reward_align = 0.5
    assist_reward_cross = 2.0
    assist_reward_capture = 8.0
    assist_reward_success = 5.0
    assist_reward_terminal_capture = 10.0
    assist_reward_spill = 3.0
    assist_reward_premature_tilt = 0.75
    assist_reward_left_action_rate = 0.02
    assist_reward_left_joint_vel = 0.002
    assist_success_fill_ratio = 0.30
    assist_success_spill_max = 0.15
    source_empty_hold_steps = 5


def test_compute_assist_reward_terms_matches_design_formula():
    cfg = _Cfg()
    inputs = dict(
        mouth_xy_distance=torch.tensor([0.1], dtype=torch.float32),
        mouth_z_clearance=torch.tensor([0.1], dtype=torch.float32),
        directional_tilt_cos=torch.tensor([0.4], dtype=torch.float32),
        mouth_alignment_cos=torch.tensor([0.2], dtype=torch.float32),
        g_align_xy=torch.tensor([0.6], dtype=torch.float32),
        g_ready=torch.tensor([0.4], dtype=torch.float32),
        g_pour=torch.tensor([0.3], dtype=torch.float32),
        bead_cross_fraction=torch.tensor([0.5], dtype=torch.float32),
        bead_in_target_fraction=torch.tensor([0.4], dtype=torch.float32),
        spill_ratio=torch.tensor([0.1], dtype=torch.float32),
        left_action_delta=torch.tensor([2.0], dtype=torch.float32),
        left_joint_vel_cost=torch.tensor([3.0], dtype=torch.float32),
        episode_length_buf=torch.tensor([9], dtype=torch.long),
        max_episode_length=10,
        source_empty_steps=torch.tensor([0], dtype=torch.long),
    )

    terms = compute_assist_reward_terms(cfg=cfg, **inputs)

    assert torch.allclose(terms["approach_xy"], torch.exp(torch.tensor([-0.5])), atol=1e-6)
    assert torch.allclose(terms["clearance_score"], torch.tensor([0.99888748]), atol=1e-6)
    assert torch.allclose(terms["tilt_score"], torch.tensor([0.29411766]), atol=1e-6)
    assert torch.allclose(terms["align_score"], torch.tensor([0.6]), atol=1e-6)
    assert torch.allclose(terms["r_approach"], torch.tensor([0.90979594]), atol=1e-6)
    assert torch.allclose(terms["r_clearance"], torch.tensor([0.44949937]), atol=1e-6)
    assert torch.allclose(terms["r_ready"], torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(terms["r_prepour"], torch.tensor([0.31235296]), atol=1e-6)
    assert torch.allclose(terms["r_pour"], torch.tensor([1.26]), atol=1e-6)
    assert torch.allclose(terms["success_now"], torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(terms["r_success"], torch.tensor([5.0]), atol=1e-6)
    assert torch.allclose(terms["r_terminal_capture"], torch.tensor([4.0]), atol=1e-6)
    assert torch.allclose(terms["premature_tilt_cost"], torch.tensor([0.1764706]), atol=1e-6)
    assert torch.allclose(terms["total"], torch.tensor([11.6533]), atol=1e-4)
