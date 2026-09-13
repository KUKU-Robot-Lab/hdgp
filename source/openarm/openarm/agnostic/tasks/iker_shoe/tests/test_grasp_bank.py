"""grasp_bank — synergy closing, pre-grasp geometry, lift test, bank file (no Isaac)."""

import math

import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb

JOINTS = ["a", "b", "c"]
START = torch.tensor([[0.0, 0.0, 0.5]])
GRIP = torch.tensor([1.0, 1.0, 0.5])  # joint c does not move between start and grip
LOWER = torch.tensor([-1.0, -1.0, -1.0])
UPPER = torch.tensor([2.0, 0.6, 2.0])


def test_synergy_closes_free_joints_and_skips_fixed_ones():
    close, target = gb.synergy_step(torch.zeros(1, 3), START.clone(), START.clone(), START, GRIP, LOWER, UPPER)
    assert torch.allclose(close, torch.tensor([[gb.CLOSE_RATE_PER_STEP, gb.CLOSE_RATE_PER_STEP, 0.0]]))
    assert torch.allclose(target[0, :2], torch.full((2,), gb.CLOSE_RATE_PER_STEP))
    assert target[0, 2] == pytest.approx(0.5)


def test_blocked_joint_freezes_but_joint_at_its_limit_does_not():
    close = torch.tensor([[0.5, 0.7, 0.5]])
    target = torch.tensor([[0.5, 0.6, 0.5]])
    joint_pos = torch.tensor([[0.1, 0.6, 0.5]])  # a lags its target by 0.4 rad; b sits on its upper limit
    new_close, new_target = gb.synergy_step(close, target, joint_pos, START, GRIP, LOWER, UPPER)
    assert new_close[0, 0] == pytest.approx(0.5)
    assert new_close[0, 1] == pytest.approx(0.7 + gb.CLOSE_RATE_PER_STEP)
    assert new_target[0, 1] == pytest.approx(0.6)  # clamped to the joint limit


def test_zero_tilt_palm_faces_down_with_fingers_toward_plus_y():
    rot = gb.palm_rotations(torch.tensor([0.0]), torch.tensor([0.0]))[0]
    assert torch.allclose(rot[:, 0], torch.tensor([0.0, 0.0, -1.0]))  # palmar side
    assert torch.allclose(rot[:, 2], torch.tensor([0.0, 1.0, 0.0]))  # fingers
    tilted = gb.palm_rotations(torch.tensor([-20.0]), torch.tensor([0.0]))[0]
    assert torch.allclose(tilted[:, 0], torch.tensor([0.0, -math.sin(math.radians(20)), -math.cos(math.radians(20))]), atol=1e-6)


def test_palm_goal_sits_beside_the_near_side_above_the_top():
    pre = gb.PreGrasp(*(torch.tensor([v]) for v in (-20.0, 0.03, 0.02, 0.01, 0.0, 0.4)))
    goal = gb.palm_goal_positions(pre, (0.25, 0.14), 90.0, 0.048, 0.309)[0]
    # yawed 90 deg: along-length is +y and the near side is +x
    assert torch.allclose(goal, torch.tensor([0.25 + 0.068, 0.14 + 0.01, 0.339]), atol=1e-6)


def test_samples_stay_in_their_ranges():
    pre = gb.sample_pregrasp(500, torch.Generator().manual_seed(0))
    for value, bounds in ((pre.tilt_deg, gb.TILT_DEG_RANGE), (pre.thumb3, gb.THUMB3_RANGE), (pre.along_length, gb.ALONG_LENGTH_RANGE)):
        assert bounds[0] <= float(value.min()) and float(value.max()) <= bounds[1]


def test_lift_held_needs_rise_and_little_slip():
    z0 = torch.tensor([0.25, 0.25, 0.25])
    z1 = torch.tensor([0.34, 0.26, 0.34])
    rel_lift = torch.zeros(3, 3)
    rel_hold = torch.tensor([[0.0, 0.0, 0.005], [0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    assert gb.lift_held(z0, z1, rel_lift, rel_hold).tolist() == [True, False, False]


def _bank_doc(**metadata):
    entries = {
        "joint_pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "joint_target": torch.tensor([[1.5, 2.5, 3.5]]),
        "shoe_pose": torch.tensor([[0.3, 0.1, 0.26, 1.0, 0.0, 0.0, 0.0]]),
        "palm_pose": torch.tensor([[0.3, 0.05, 0.34, 1.0, 0.0, 0.0, 0.0]]),
    }
    return gb.bank_document(entries, JOINTS, {"physics_dt": 1 / 120, **metadata})


def test_bank_round_trip_reorders_joints_by_name():
    bank = gb.load_bank(_bank_doc(), ["c", "a", "b"], {"physics_dt": 1 / 120}, "cpu")
    assert bank.size == 1
    assert bank.joint_pos.tolist() == [[3.0, 1.0, 2.0]] and bank.joint_target.tolist() == [[3.5, 1.5, 2.5]]


def test_bank_refuses_different_settings_or_missing_joints():
    with pytest.raises(ValueError, match="different settings"):
        gb.load_bank(_bank_doc(), JOINTS, {"physics_dt": 1 / 60}, "cpu")
    with pytest.raises(ValueError, match="lacks joints"):
        gb.load_bank(_bank_doc(), JOINTS + ["d"], {}, "cpu")


def test_bank_document_rejects_non_finite_rows():
    with pytest.raises(ValueError, match="joint_pos"):
        gb.bank_document(
            {"joint_pos": torch.tensor([[float("nan")]]), "joint_target": torch.zeros(1, 1), "shoe_pose": torch.zeros(1, 7), "palm_pose": torch.zeros(1, 7)},
            ["a"],
            {},
        )


def test_gains_metadata_rounds_for_exact_comparison():
    meta = gb.gains_metadata(["a", "b"], torch.tensor([400.00001, 10.0]), torch.tensor([80.0, 0.123456]))
    assert meta == {"stiffness": {"a": 400.0, "b": 10.0}, "damping": {"a": 80.0, "b": 0.1235}}
