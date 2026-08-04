"""residual finger 제어 계약: active 관절은 center±scale(한계 clamp), abduction은 고정."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "finger_action_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_finger", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_residual_finger_targets = _mod.compute_residual_finger_targets

LOWER = torch.tensor([0.0, 0.0])
UPPER = torch.tensor([2.0, 2.0])
CENTER = torch.tensor([1.0, 1.0])


def test_residual_around_center():
    mask = torch.tensor([1.0, 1.0])
    fixed = torch.tensor([0.5, 0.5])
    # a=0 → center 그대로
    out0 = compute_residual_finger_targets(
        torch.tensor([[0.0, 0.0]]), CENTER, 0.3, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out0[0], CENTER)
    # a=+1 → center+scale, a=-1 → center-scale
    out_p = compute_residual_finger_targets(
        torch.tensor([[1.0, 1.0]]), CENTER, 0.3, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out_p[0], torch.tensor([1.3, 1.3]))
    out_m = compute_residual_finger_targets(
        torch.tensor([[-1.0, -1.0]]), CENTER, 0.3, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out_m[0], torch.tensor([0.7, 0.7]))


def test_residual_clamped_to_limits():
    mask = torch.tensor([1.0, 1.0])
    fixed = torch.tensor([0.5, 0.5])
    # scale이 커도 관절 한계로 clamp (center=1, scale=5 → [−4,6] → [0,2])
    out = compute_residual_finger_targets(
        torch.tensor([[1.0, -1.0]]), CENTER, 5.0, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]))


def test_inactive_fixed():
    mask = torch.tensor([1.0, 0.0])
    fixed = torch.tensor([0.3, 0.7])
    out = compute_residual_finger_targets(
        torch.tensor([[1.0, 1.0]]), CENTER, 0.3, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out[0, 0], torch.tensor(1.3))   # active → center+scale
    assert torch.allclose(out[0, 1], torch.tensor(0.7))   # inactive → fixed


def test_action_clamped():
    mask = torch.tensor([1.0, 1.0])
    fixed = torch.tensor([0.5, 0.5])
    # |a|>1 은 clamp(-1,1) 후 residual (a=5→+scale, a=-5→-scale)
    out = compute_residual_finger_targets(
        torch.tensor([[5.0, -5.0]]), CENTER, 0.3, LOWER, UPPER, mask, fixed
    )
    assert torch.allclose(out[0], torch.tensor([1.3, 0.7]))
