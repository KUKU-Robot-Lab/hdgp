"""precision mask/frac 순수 로직 정합 검증(gate 배선 회귀 방지).

grasp_right_utils.py를 파일 경로에서 직접 로드(상위 __init__ isaaclab 우회).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_adapt_utils_wiring", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_precision_grasp_mask = _mod.compute_precision_grasp_mask
compute_precision_grasp_frac = _mod.compute_precision_grasp_frac


def test_frac_zero_without_thumb():
    tip = torch.tensor([[False, True, True, True, True]])
    assert float(compute_precision_grasp_frac(tip)[0]) == 0.0


def test_frac_full_at_three_contacts():
    tip = torch.tensor([[True, True, True, False, False]])
    assert abs(float(compute_precision_grasp_frac(tip)[0]) - 1.0) < 1e-6


def test_mask_and_frac_agree_on_boundary():
    # 엄지+대향2지: mask True, frac==1.0
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(compute_precision_grasp_mask(tip)[0]) is True
    assert float(compute_precision_grasp_frac(tip)[0]) >= 1.0 - 1e-6
    # 엄지+대향1지: mask False
    tip1 = torch.tensor([[True, True, False, False, False]])
    assert bool(compute_precision_grasp_mask(tip1)[0]) is False


def test_min_opposing_three_is_stricter():
    # 엄지+대향2지는 min_opposing=3에서 False
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(compute_precision_grasp_mask(tip, min_opposing=3)[0]) is False
    # 엄지+대향3지는 True
    tip3 = torch.tensor([[True, True, True, True, False]])
    assert bool(compute_precision_grasp_mask(tip3, min_opposing=3)[0]) is True


def test_batch_mixed():
    tip = torch.tensor([
        [True, True, True, False, False],   # 엄지+2 → True
        [False, True, True, True, False],   # 엄지 없음 → False
        [True, False, True, False, False],  # 엄지+1 → False
        [True, True, True, True, True],     # 엄지+4 → True
    ])
    out = compute_precision_grasp_mask(tip)
    assert out.tolist() == [True, False, False, True]
