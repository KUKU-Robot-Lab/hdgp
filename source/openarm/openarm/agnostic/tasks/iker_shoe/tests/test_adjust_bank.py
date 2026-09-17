import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import adjust_bank as ab
from openarm.agnostic.tasks.iker_shoe import place_stage as ps

NAMES = ["j0", "j1", "j2"]


def _entries(n: int) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": torch.randn(n, len(NAMES)),
        "joint_target": torch.randn(n, len(NAMES)),
        "shoe_pose": torch.randn(n, 7),
        "other_pose": torch.randn(n, 7),
        "grip_scalar": torch.rand(n, 1),
    }


def test_qualifies_only_a_stranded_open_resting_still_unplaced_shoe():
    cfg = ps.PlaceRewardCfg()
    t = torch.tensor
    placed = t([False, True, False, False, False, False, False])
    resting = t([True, True, False, True, True, True, True])
    still = t([True, True, True, False, True, True, True])
    open_frac = t([0.95, 0.95, 0.95, 0.95, 0.50, 0.95, 0.95])
    retracting = t([False, False, False, False, False, True, False])
    kp = t([0.05, 0.02, 0.05, 0.05, 0.05, 0.05, 0.50])
    got = ab.qualifies(placed, resting, still, open_frac, retracting, kp, cfg, max_keypoint_dist=0.10)
    assert got.tolist() == [True, False, False, False, False, False, False]


def test_save_then_load_round_trips(tmp_path):
    entries = _entries(5)
    path = ab.save_bank(tmp_path / "bank.pt", entries, NAMES, {"source": "test"})
    bank = ab.load_bank(path, NAMES, "cpu")
    assert bank.size == 5
    assert torch.equal(bank.joint_pos, entries["joint_pos"])
    assert torch.equal(bank.other_pose, entries["other_pose"])


def test_load_rejects_other_joint_order(tmp_path):
    path = ab.save_bank(tmp_path / "bank.pt", _entries(2), NAMES, {})
    with pytest.raises(ValueError, match="joint"):
        ab.load_bank(path, ["j1", "j0", "j2"], "cpu")


def test_save_rejects_empty_and_mismatched_rows(tmp_path):
    with pytest.raises(ValueError):
        ab.save_bank(tmp_path / "a.pt", _entries(0), NAMES, {})
    bad = _entries(3)
    bad["shoe_pose"] = bad["shoe_pose"][:2]
    with pytest.raises(ValueError):
        ab.save_bank(tmp_path / "b.pt", bad, NAMES, {})


def test_start_mask_fraction_bounds():
    g = torch.Generator().manual_seed(0)
    assert not ab.start_mask(1000, 0.0, "cpu", g).any()
    assert ab.start_mask(1000, 1.0, "cpu", g).all()
    frac = ab.start_mask(20000, 0.3, "cpu", g).float().mean().item()
    assert 0.28 < frac < 0.32
    with pytest.raises(ValueError):
        ab.start_mask(10, 1.5, "cpu", g)
