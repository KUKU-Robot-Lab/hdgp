from __future__ import annotations

from importlib import import_module

import torch


_MODULE = import_module(
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.pour_v1.pour_gate_success"
)
compute_gate_terms = _MODULE.compute_gate_terms
compute_success_terms = _MODULE.compute_success_terms


class _Cfg:
    reward_gate_xy_scale = 5.0
    reward_gate_clear_scale = 80.0
    reward_gate_tilt_scale = 15.0
    reward_clearance_min = 0.015
    reward_tilt_cos_min = 0.15
    assist_success_fill_ratio = 0.30
    assist_success_spill_max = 0.15
    source_empty_hold_steps = 5


def test_compute_gate_terms_matches_design_equations():
    cfg = _Cfg()

    terms = compute_gate_terms(
        cfg=cfg,
        mouth_xy_distance=torch.tensor([0.1], dtype=torch.float32),
        mouth_z_clearance=torch.tensor([0.1], dtype=torch.float32),
        directional_tilt_cos=torch.tensor([0.4], dtype=torch.float32),
    )

    assert torch.allclose(terms["g_align_xy"], torch.exp(torch.tensor([-0.5])), atol=1e-6)
    assert torch.allclose(terms["g_clear"], torch.tensor([0.99888748]), atol=1e-6)
    assert torch.allclose(terms["g_tilt"], torch.tensor([0.97702265]), atol=1e-6)
    assert torch.allclose(terms["g_ready"], terms["g_align_xy"] * terms["g_clear"], atol=1e-6)
    assert torch.allclose(terms["g_pour"], terms["g_ready"] * terms["g_tilt"], atol=1e-6)


def test_compute_success_terms_uses_fill_spill_gate_and_episode_end():
    cfg = _Cfg()

    terms = compute_success_terms(
        cfg=cfg,
        bead_in_target_fraction=torch.tensor([0.4, 0.2], dtype=torch.float32),
        spill_ratio=torch.tensor([0.1, 0.1], dtype=torch.float32),
        g_pour=torch.tensor([0.2, 0.2], dtype=torch.float32),
        episode_length_buf=torch.tensor([9, 2], dtype=torch.long),
        max_episode_length=10,
        source_empty_steps=torch.tensor([0, 5], dtype=torch.long),
    )

    assert torch.equal(terms["success_now"], torch.tensor([True, False]))
    assert torch.equal(terms["is_last_step"], torch.tensor([True, False]))
    assert torch.equal(terms["is_source_ending"], torch.tensor([False, True]))
    assert torch.equal(terms["episode_ending"], torch.tensor([True, True]))
