"""Unit tests for the deep-tilt start-state bank (sparse-reward bootstrap).

Pure-torch logic only (no Isaac Sim), so it runs under plain ``python3``/pytest
with torch installed (CPU is fine).

Bootstrap intent (analysis.md lstm_test1 §6): the policy reaches a deep-tilt,
over-target, beads-in-source state but never discovers the final push that pours
beads into the target, so the 200-weight capture reward never fires. This bank
captures those *real* near-pour frames and restores a fraction (f_boot) of resets
to them, so the sparse reward becomes reachable. f_boot anneals to 0 so the
learned value transfers back to upright starts.

The pure logic under test:
  1. ``capture_mask`` — which envs' current state qualifies as a productive seed
     (deep enough tilt, beads still in source, pour-point over the target).
  2. ``f_boot_ratio`` — monotone anneal of the bootstrap fraction.
  3. ``DeepTiltStateBank`` — a ring buffer that stores full-state snapshots and
     samples start indices; oldest entries are overwritten once full.

RH56F1 adaptation: num_hand=6 (6 actuated drives instead of tesollo's 20).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "deep_tilt_bank.py"
SPEC = importlib.util.spec_from_file_location("deep_tilt_bank", MODULE_PATH)
dtb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dtb
SPEC.loader.exec_module(dtb)

capture_mask = dtb.capture_mask
f_boot_ratio = dtb.f_boot_ratio
DeepTiltStateBank = dtb.DeepTiltStateBank


# ---------------------------------------------------------------------------
# capture_mask
# ---------------------------------------------------------------------------
def _cm(tilt, src, mouth, drift=5.0, contacts=4.0, **overrides):
    """One-env capture_mask helper with clean-grasp defaults."""
    kw = dict(tilt_min=0.40, src_min=0.80, mouth_max=0.08, drift_max=12.0, contacts_min=3.0)
    kw.update(overrides)
    return capture_mask(
        torch.tensor([tilt]),
        torch.tensor([src]),
        torch.tensor([mouth]),
        torch.tensor([drift]),
        torch.tensor([contacts]),
        **kw,
    )


def test_capture_mask_accepts_clean_deep_tilt_over_target_with_beads():
    # deep tilt, beads in source, over target, grasp intact (low drift, full contacts)
    assert bool(_cm(0.45, 0.95, 0.03, drift=5.0, contacts=4.0)[0]) is True


def test_capture_mask_rejects_shallow_tilt():
    assert bool(_cm(0.30, 0.95, 0.03)[0]) is False


def test_capture_mask_rejects_already_poured_seed():
    # beads no longer in source → free-pour seed (hacking) → reject
    assert bool(_cm(0.45, 0.40, 0.03)[0]) is False


def test_capture_mask_rejects_not_over_target():
    assert bool(_cm(0.45, 0.95, 0.20)[0]) is False


def test_capture_mask_rejects_grasp_slip():
    # grasp slipping (high cup_rel_drift) → polluted seed → reject (lstm_test3 self-poisoning fix)
    assert bool(_cm(0.45, 0.95, 0.03, drift=25.0)[0]) is False


def test_capture_mask_rejects_few_contacts():
    # grasp losing fingers (low contact count) → reject
    assert bool(_cm(0.45, 0.95, 0.03, contacts=2.0)[0]) is False


def test_capture_mask_elementwise_batch():
    tilt = torch.tensor([0.45, 0.30, 0.45, 0.45, 0.45])
    src = torch.tensor([0.95, 0.95, 0.40, 0.95, 0.95])
    mouth = torch.tensor([0.03, 0.03, 0.03, 0.20, 0.03])
    drift = torch.tensor([5.0, 5.0, 5.0, 5.0, 25.0])
    contacts = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0])
    mask = capture_mask(
        tilt, src, mouth, drift, contacts,
        tilt_min=0.40, src_min=0.80, mouth_max=0.08, drift_max=12.0, contacts_min=3.0,
    )
    assert mask.tolist() == [True, False, False, False, False]


# ---------------------------------------------------------------------------
# f_boot_ratio
# ---------------------------------------------------------------------------
def test_f_boot_ratio_endpoints_and_midpoint():
    assert f_boot_ratio(0.0, f_start=0.5, f_end=0.0) == 0.5
    assert f_boot_ratio(1.0, f_start=0.5, f_end=0.0) == 0.0
    assert abs(f_boot_ratio(0.5, f_start=0.5, f_end=0.0) - 0.25) < 1e-9


def test_f_boot_ratio_is_monotone_decreasing():
    ps = [i / 10.0 for i in range(11)]
    vals = [f_boot_ratio(p, f_start=0.5, f_end=0.0) for p in ps]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_f_boot_ratio_clamps_out_of_range_progress():
    assert f_boot_ratio(-0.5, f_start=0.5, f_end=0.0) == 0.5
    assert f_boot_ratio(2.0, f_start=0.5, f_end=0.0) == 0.0


# ---------------------------------------------------------------------------
# DeepTiltStateBank ring buffer
# RH56F1: num_hand=6 (6 actuated hand drives, vs tesollo's 20)
# ---------------------------------------------------------------------------
def _make_bank(capacity=4, num_arm=7, num_hand=6, num_beads=3):
    return DeepTiltStateBank(
        capacity=capacity, num_arm=num_arm, num_hand=num_hand,
        num_beads=num_beads, device=torch.device("cpu"),
    )


def _rows(m, *, num_arm=7, num_hand=6, num_beads=3, fill=1.0):
    # order mirrors the env restore tuple: arm, hand, palm_pose(7), cup_pose(7), bead_state
    return (
        torch.full((m, num_arm), fill),
        torch.full((m, num_hand), fill),
        torch.full((m, 7), fill),
        torch.full((m, 7), fill),
        torch.full((m, num_beads, 13), fill),
    )


def test_bank_starts_empty():
    bank = _make_bank()
    assert bank.count == 0
    assert bank.is_ready(1) is False


def test_bank_store_increments_count():
    bank = _make_bank(capacity=4)
    bank.store(*_rows(2))
    assert bank.count == 2
    assert bank.is_ready(2) is True
    assert bank.is_ready(3) is False


def test_bank_store_empty_is_noop():
    bank = _make_bank()
    bank.store(*_rows(0))
    assert bank.count == 0


def test_bank_count_saturates_at_capacity():
    bank = _make_bank(capacity=4)
    bank.store(*_rows(3))
    bank.store(*_rows(3))
    assert bank.count == 4  # saturates, does not exceed capacity


def test_bank_ring_overwrites_oldest():
    bank = _make_bank(capacity=3)
    # fill with id 1,2,3 in arm[:,0]
    for v in (1.0, 2.0, 3.0):
        bank.store(*_rows(1, fill=v))
    # store a 4th (id 4) → overwrites the oldest slot (index 0, which held id 1)
    bank.store(*_rows(1, fill=4.0))
    stored = set(bank.arm[:, 0].tolist())
    assert stored == {2.0, 3.0, 4.0}  # id 1 evicted


def test_bank_sample_returns_valid_indices():
    bank = _make_bank(capacity=8)
    bank.store(*_rows(5))
    idx = bank.sample(16)
    assert idx.shape == (16,)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < bank.count  # never samples uninitialized slots


def test_bank_roundtrip_preserves_stored_tensors():
    bank = _make_bank(capacity=4, num_beads=3)
    arm = torch.randn(2, 7)
    hand = torch.randn(2, 6)   # RH56F1: 6 drives
    palm = torch.randn(2, 7)
    cup = torch.randn(2, 7)
    bead = torch.randn(2, 3, 13)
    bank.store(arm, hand, palm, cup, bead)
    assert torch.allclose(bank.arm[:2], arm)
    assert torch.allclose(bank.hand[:2], hand)
    assert torch.allclose(bank.palm_pose[:2], palm)
    assert torch.allclose(bank.cup_pose[:2], cup)
    assert torch.allclose(bank.bead_state[:2], bead)
