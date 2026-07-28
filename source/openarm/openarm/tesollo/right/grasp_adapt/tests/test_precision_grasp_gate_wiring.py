"""precision mask + damage dose 순수 로직 계약(gate/damage 회귀 방지).

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
compute_damage_dose = _mod.compute_damage_dose
compute_mass_shift_trigger = _mod.compute_mass_shift_trigger


def test_mask_boundary():
    # 엄지+대향2지: True / 엄지+대향1지: False
    assert bool(compute_precision_grasp_mask(torch.tensor([[True, True, True, False, False]]))[0]) is True
    assert bool(compute_precision_grasp_mask(torch.tensor([[True, True, False, False, False]]))[0]) is False


def test_min_opposing_three_is_stricter():
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(compute_precision_grasp_mask(tip, min_opposing=3)[0]) is False
    tip3 = torch.tensor([[True, True, True, True, False]])
    assert bool(compute_precision_grasp_mask(tip3, min_opposing=3)[0]) is True


def test_batch_mixed():
    tip = torch.tensor([
        [True, True, True, False, False],   # 엄지+2 → True
        [False, True, True, True, False],   # 엄지 없음 → False
        [True, False, True, False, False],  # 엄지+1 → False
        [True, True, True, True, True],     # 엄지+4 → True
    ])
    assert compute_precision_grasp_mask(tip).tolist() == [True, False, False, True]


def test_dose_zero_below_f_safe():
    # radial < f_safe → 형상파괴 누적 없음
    out = compute_damage_dose(torch.zeros(1), torch.tensor([2.0]), f_safe=3.0, dt=0.1, q=2.0)
    assert float(out[0]) == 0.0


def test_dose_accumulates_above_f_safe():
    # radial=6, f_safe=3 → over=(6-3)/3=1 → +dt·1^2 = 0.1, 누적
    out = compute_damage_dose(torch.zeros(1), torch.tensor([6.0]), f_safe=3.0, dt=0.1, q=2.0)
    assert abs(float(out[0]) - 0.1) < 1e-6
    out2 = compute_damage_dose(out, torch.tensor([6.0]), f_safe=3.0, dt=0.1, q=2.0)
    assert abs(float(out2[0]) - 0.2) < 1e-6


def test_dose_nonlinear_q():
    # radial=9, f_safe=3 → over=2 → 2^2=4 → dt·4 = 0.4 (q=2 비선형)
    out = compute_damage_dose(torch.zeros(1), torch.tensor([9.0]), f_safe=3.0, dt=0.1, q=2.0)
    assert abs(float(out[0]) - 0.4) < 1e-6


def _b(vals):
    return torch.tensor(vals, dtype=torch.bool)


def test_mass_shift_no_trigger_below_height():
    # 높이 미달 → 발동 X, hold 0 유지
    trig, hold = compute_mass_shift_trigger(
        _b([True]), torch.tensor([0.05]), 0.08,
        torch.tensor([10], dtype=torch.long), 15, _b([False]),
    )
    assert bool(trig[0]) is False and int(hold[0]) == 0


def test_mass_shift_holds_then_fires():
    # 높이 충족 유지: delay 도달 시 발동
    hold = torch.tensor([14], dtype=torch.long)
    trig, hold = compute_mass_shift_trigger(
        _b([True]), torch.tensor([0.10]), 0.08, hold, 15, _b([False]),
    )
    assert int(hold[0]) == 15 and bool(trig[0]) is True


def test_mass_shift_not_yet_delay():
    # 높이 충족이나 hold < delay → 발동 X, hold++
    trig, hold = compute_mass_shift_trigger(
        _b([True]), torch.tensor([0.10]), 0.08,
        torch.tensor([5], dtype=torch.long), 15, _b([False]),
    )
    assert int(hold[0]) == 6 and bool(trig[0]) is False


def test_mass_shift_lift_not_latched_resets_hold():
    # lift latch 아님 → 높이 무관 hold 0, 발동 X
    trig, hold = compute_mass_shift_trigger(
        _b([False]), torch.tensor([0.20]), 0.08,
        torch.tensor([14], dtype=torch.long), 15, _b([False]),
    )
    assert int(hold[0]) == 0 and bool(trig[0]) is False


def test_mass_shift_once_only():
    # 이미 발동(already_done) → 조건 충족해도 재발동 X
    trig, _ = compute_mass_shift_trigger(
        _b([True]), torch.tensor([0.20]), 0.08,
        torch.tensor([100], dtype=torch.long), 15, _b([True]),
    )
    assert bool(trig[0]) is False


def test_mass_shift_batch():
    lift = _b([True, True, False, True])
    h = torch.tensor([0.10, 0.05, 0.20, 0.10])
    hold = torch.tensor([14, 3, 14, 14], dtype=torch.long)
    done = _b([False, False, False, True])
    trig, newhold = compute_mass_shift_trigger(lift, h, 0.08, hold, 15, done)
    # env0: 충족+delay → True; env1: 높이미달 → False,hold0; env2: latch X → False,hold0; env3: done → False
    assert trig.tolist() == [True, False, False, False]
    assert newhold.tolist() == [15, 0, 0, 15]
