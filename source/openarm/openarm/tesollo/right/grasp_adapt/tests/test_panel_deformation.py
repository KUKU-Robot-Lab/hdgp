"""Phase 2 패널 변형 신호 계약: 최대 |힌지각|(deg), 부호 무관, 미변형 0."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_utils_panel", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_panel_deformation_deg = _mod.compute_panel_deformation_deg


def test_zero_when_intact():
    q = torch.zeros(3, 12)
    out = compute_panel_deformation_deg(q)
    assert torch.allclose(out, torch.zeros(3))


def test_max_over_panels():
    # 한 패널만 크게 굽음 → max가 그 각을 deg로 반환
    q = torch.zeros(1, 12)
    q[0, 4] = math.radians(30.0)
    out = compute_panel_deformation_deg(q)
    assert abs(float(out[0]) - 30.0) < 1e-3


def test_sign_agnostic():
    # 안쪽(+)이든 바깥(-)이든 |각|으로 측정 (파지 squeeze는 안쪽만이지만 부호 무관)
    q_pos = torch.zeros(1, 12); q_pos[0, 0] = math.radians(20.0)
    q_neg = torch.zeros(1, 12); q_neg[0, 0] = -math.radians(20.0)
    assert abs(float(compute_panel_deformation_deg(q_pos)[0])
               - float(compute_panel_deformation_deg(q_neg)[0])) < 1e-4


def test_batch_independent():
    q = torch.zeros(2, 12)
    q[0, 3] = math.radians(10.0)
    q[1, 7] = math.radians(40.0)
    out = compute_panel_deformation_deg(q)
    assert abs(float(out[0]) - 10.0) < 1e-3
    assert abs(float(out[1]) - 40.0) < 1e-3


def test_feeds_buckle_threshold():
    # f_buckle=35deg 계약: 40deg 패널 → buckle 신호(deform > 35)
    q = torch.zeros(1, 12); q[0, 0] = math.radians(40.0)
    deform = compute_panel_deformation_deg(q)
    assert bool((deform > 35.0).item())
    q2 = torch.zeros(1, 12); q2[0, 0] = math.radians(8.0)
    deform2 = compute_panel_deformation_deg(q2)
    assert not bool((deform2 > 10.0).item())  # f_safe=10deg 미만
