from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "warmstart_logic.py"
SPEC = importlib.util.spec_from_file_location("warmstart_logic", MODULE_PATH)
warmstart_logic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = warmstart_logic
SPEC.loader.exec_module(warmstart_logic)

select_warmstart_success = warmstart_logic.select_warmstart_success


def test_warmstart_success_does_not_hard_gate_uprightness() -> None:
    lifted = torch.tensor([True, True, True])
    grasped = torch.tensor([True, True, True])
    up_z = torch.tensor([0.55, 0.75, 0.95])
    j7 = torch.tensor([0.5, 0.5, 0.5])

    success, diagnostics = select_warmstart_success(lifted, grasped, up_z, j7)

    assert success.tolist() == [True, True, True]
    assert diagnostics["upright_0_7"].tolist() == [False, True, True]
    assert diagnostics["upright_0_9"].tolist() == [False, False, True]


def test_warmstart_success_still_requires_lift_grasp_and_j7_range() -> None:
    lifted = torch.tensor([False, True, True, True])
    grasped = torch.tensor([True, False, True, True])
    up_z = torch.tensor([0.95, 0.95, 0.55, 0.95])
    j7 = torch.tensor([0.5, 0.5, 1.7, 0.5])

    success, diagnostics = select_warmstart_success(lifted, grasped, up_z, j7)

    assert success.tolist() == [False, False, False, True]
    assert diagnostics["j7_in_range"].tolist() == [True, True, False, True]
