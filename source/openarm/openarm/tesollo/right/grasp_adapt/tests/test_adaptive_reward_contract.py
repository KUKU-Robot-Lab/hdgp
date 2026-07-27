"""adaptive grip objective 회귀 방지: hold_gate=0이면 secure/efficient=0.

grasp_adaptive_core.py를 파일 경로에서 직접 로드(상위 __init__ isaaclab 우회).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# .../grasp_adapt/tests/ → parents[4] == .../openarm(패키지 루트), common은 그 아래.
MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "common" / "grasp_adaptive_core.py"
)
SPEC = importlib.util.spec_from_file_location("grasp_adaptive_core", MODULE_PATH)
_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = _mod
SPEC.loader.exec_module(_mod)

compute_adaptive_grip_terms = _mod.compute_adaptive_grip_terms


class _Cfg:
    secure_weight = 10.0
    secure_slip_sharpness = 30.0
    force_efficiency_weight = 2.0
    drop_penalty_weight = 8.0
    force_ratio_cost_cap = 6.0


def test_hold_gate_zero_zeros_secure_and_efficient():
    n = 4
    _, terms = compute_adaptive_grip_terms(
        grip_normal_force=torch.ones(n) * 5.0,
        cup_weight=torch.ones(n) * 2.0,
        cup_speed=torch.zeros(n),
        tip_contact_frac=torch.ones(n),
        lift_gate=torch.zeros(n),      # lift 안 됨 → hold_gate=0
        lifted_gate=torch.zeros(n),
        full_tip_contact=torch.zeros(n),
        cfg=_Cfg(),
    )
    assert torch.allclose(terms["r_secure"], torch.zeros(n))
    assert torch.allclose(terms["r_efficient"], torch.zeros(n))


def test_no_slip_high_secure():
    n = 4
    _, terms = compute_adaptive_grip_terms(
        grip_normal_force=torch.zeros(n),
        cup_weight=torch.ones(n) * 2.0,
        cup_speed=torch.zeros(n),      # 정지 → secure_quality≈1
        tip_contact_frac=torch.ones(n),
        lift_gate=torch.ones(n),
        lifted_gate=torch.ones(n),
        full_tip_contact=torch.ones(n),
        cfg=_Cfg(),
    )
    assert torch.all(terms["secure_quality"] > 0.99)
