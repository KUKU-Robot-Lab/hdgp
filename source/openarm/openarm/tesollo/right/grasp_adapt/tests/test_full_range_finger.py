"""full-range finger 제어 계약: active 관절은 action[-1,1]→[lower,upper], abduction은 고정."""
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

compute_full_range_finger_targets = _mod.compute_full_range_finger_targets

LOWER = torch.tensor([0.0, 0.0])
UPPER = torch.tensor([2.0, 2.0])


def test_active_full_range():
    mask = torch.tensor([1.0, 1.0])
    fixed = torch.tensor([0.5, 0.5])
    out_min = compute_full_range_finger_targets(torch.tensor([[-1.0, -1.0]]), LOWER, UPPER, mask, fixed)
    assert torch.allclose(out_min[0], torch.tensor([0.0, 0.0]))
    out_max = compute_full_range_finger_targets(torch.tensor([[1.0, 1.0]]), LOWER, UPPER, mask, fixed)
    assert torch.allclose(out_max[0], torch.tensor([2.0, 2.0]))
    out_mid = compute_full_range_finger_targets(torch.tensor([[0.0, 0.0]]), LOWER, UPPER, mask, fixed)
    assert torch.allclose(out_mid[0], torch.tensor([1.0, 1.0]))


def test_inactive_fixed():
    # 2번째 관절(abduction) 고정
    mask = torch.tensor([1.0, 0.0])
    fixed = torch.tensor([0.3, 0.7])
    out = compute_full_range_finger_targets(torch.tensor([[1.0, 1.0]]), LOWER, UPPER, mask, fixed)
    assert torch.allclose(out[0, 0], torch.tensor(2.0))   # active → upper
    assert torch.allclose(out[0, 1], torch.tensor(0.7))   # inactive → fixed


def test_action_clamped():
    mask = torch.tensor([1.0, 1.0])
    fixed = torch.tensor([0.5, 0.5])
    out = compute_full_range_finger_targets(torch.tensor([[5.0, -5.0]]), LOWER, UPPER, mask, fixed)
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]))  # clamp(-1,1) 후 매핑
