from __future__ import annotations

from importlib import import_module

import torch


compute_pour_observation_metrics = import_module(
    "openarm.tesollo.both.pour_v1.pour_observation"
).compute_pour_observation_metrics


class _Cfg:
    reward_gate_xy_scale = 5.0
    reward_gate_clear_scale = 80.0
    reward_gate_tilt_scale = 15.0
    reward_clearance_min = 0.015
    reward_tilt_cos_min = 0.15


def test_compute_pour_observation_metrics_matches_expected_geometry():
    cup_pos_w = torch.tensor([[0.4, 0.0, 0.3]], dtype=torch.float32)
    source_pour_point_w = torch.tensor([[0.4, 0.0, 0.4]], dtype=torch.float32)
    target_opening_w = torch.tensor([[0.5, 0.0, 0.3]], dtype=torch.float32)
    source_up_axis_w = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    metrics = compute_pour_observation_metrics(
        cup_pos_w=cup_pos_w,
        source_pour_point_w=source_pour_point_w,
        target_opening_w=target_opening_w,
        source_up_axis_w=source_up_axis_w,
        cfg=_Cfg(),
    )

    assert torch.allclose(metrics["mouth_delta"], torch.tensor([[0.1, 0.0, -0.1]]), atol=1e-6)
    assert torch.allclose(metrics["mouth_distance"], torch.tensor([0.14142136]), atol=1e-6)
    assert torch.allclose(metrics["mouth_xy_distance"], torch.tensor([0.1]), atol=1e-6)
    assert torch.allclose(metrics["mouth_z_clearance"], torch.tensor([0.1]), atol=1e-6)
    assert torch.allclose(metrics["source_up_dot_world"], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(metrics["directional_tilt_cos"], torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(metrics["mouth_alignment_cos"], torch.tensor([0.70710677]), atol=1e-6)
    assert torch.allclose(metrics["cup_center_xy_dist"], torch.tensor([0.1]), atol=1e-6)
    assert torch.allclose(metrics["g_align_xy"], torch.exp(torch.tensor([-0.5])), atol=1e-6)
    assert torch.allclose(metrics["g_clear"], torch.tensor([0.99888748]), atol=1e-6)
    assert torch.allclose(metrics["g_tilt"], torch.tensor([0.99999714]), atol=1e-6)
    assert torch.allclose(metrics["g_ready"], metrics["g_align_xy"] * metrics["g_clear"], atol=1e-6)
    assert torch.allclose(metrics["g_pour"], metrics["g_ready"] * metrics["g_tilt"], atol=1e-6)
