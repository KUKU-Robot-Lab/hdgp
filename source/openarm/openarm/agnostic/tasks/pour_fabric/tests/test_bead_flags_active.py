"""bead_flags 활성 마스크 — 파킹된 비드는 어느 비율(in_source/in_target/spill/crossed)·무게중심에도 안 들어간다.

isaaclab(quat_apply_inverse) 이 필요해 서버에서 돈다.
"""
import pytest
import torch

pytest.importorskip("isaaclab")

from openarm.agnostic.tasks.pour_fabric.bead_flags import BeadGeometry, compute_bead_flags  # noqa: E402

_GEOM = BeadGeometry(inner_radius=0.041, inside_z_min=-0.062, inside_z_max=0.100, mouth_z=0.100)
_IDQ = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _scene():
    """env 1개: 소스 컵 원점 (0,0,0), 리시버 (0.3,0,0). 비드 6개 = 소스 안 3 + 리시버 안 1 + 파킹 2."""
    src = torch.zeros(1, 3)
    tgt = torch.tensor([[0.3, 0.0, 0.0]])
    beads = torch.tensor([[
        [0.0, 0.0, -0.05], [0.01, 0.0, -0.03], [0.0, 0.01, -0.01],     # 소스 안
        [0.3, 0.0, -0.05],                                             # 리시버 안
        [-1.0, -0.6, -2.0], [-1.0, -0.5, -2.0],                        # 파킹(멀리 아래 → 마스크 없으면 spill 로 센다)
    ]])
    return beads, src, tgt


def _flags(active):
    beads, src, tgt = _scene()
    return compute_bead_flags(
        bead_pos_w=beads, source_pos_w=src, source_quat_w=_IDQ, target_pos_w=tgt, target_quat_w=_IDQ,
        geom_source=_GEOM, geom_target=_GEOM,
        prev_target_local_z=torch.full((1, 6), -1e6), crossed_mask=torch.zeros(1, 6, dtype=torch.bool),
        active_mask=active)


def test_without_mask_parked_beads_count_as_spilled():
    f = _flags(None)
    assert abs(float(f.spill_frac[0]) - 2 / 6) < 1e-6
    assert abs(float(f.in_source_frac[0]) - 3 / 6) < 1e-6


def test_mask_excludes_parked_from_every_fraction_and_centroid():
    active = torch.tensor([[True, True, True, True, False, False]])
    f = _flags(active)
    assert abs(float(f.in_source_frac[0]) - 3 / 4) < 1e-6
    assert abs(float(f.in_target_frac[0]) - 1 / 4) < 1e-6
    assert float(f.spill_frac[0]) == 0.0
    assert float(f.crossed_frac[0]) == 0.0
    c = f.centroid_w[0]
    assert float(c[2]) > -0.1                           # 파킹 z=-2 가 섞였으면 크게 음수
    assert f.source_local_z.shape == (1, 6)             # 채움 정도 계산용


def test_all_inactive_gives_zero_not_nan():
    f = _flags(torch.zeros(1, 6, dtype=torch.bool))
    for t in (f.in_source_frac, f.in_target_frac, f.spill_frac, f.crossed_frac):
        assert torch.isfinite(t).all() and float(t[0]) == 0.0
