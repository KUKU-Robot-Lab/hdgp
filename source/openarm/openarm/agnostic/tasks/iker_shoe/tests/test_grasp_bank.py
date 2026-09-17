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


def test_joint_pushed_to_its_opposite_limit_freezes():
    close = torch.tensor([[0.5, 0.5, 0.5]])
    target = torch.tensor([[0.5, 0.5, 0.5]])
    # a closes upward (START -> GRIP is 0 -> 1); pinning it at LOWER is the opposite limit, not its own.
    joint_pos = torch.tensor([[-1.0, 0.0, 0.5]])
    new_close, _ = gb.synergy_step(close, target, joint_pos, START, GRIP, LOWER, UPPER)
    assert new_close[0, 0] == pytest.approx(0.5)  # frozen despite the large error, not exempted


def test_hand_state_valid_rejects_a_finger_at_its_opposite_limit():
    # joint a pinned at LOWER, opposite its upward closing direction.
    joint_pos = torch.tensor([[-1.0, 0.5, 0.5]])
    valid = gb.hand_state_valid(joint_pos, START, GRIP, LOWER, UPPER)
    assert valid.tolist() == [False]


def test_hand_state_valid_rejects_joints_beyond_limits_and_accepts_normal_closing():
    start = START.expand(2, 3)
    joint_pos = torch.tensor(
        [
            [0.5, 0.7, 0.5],  # b exceeds UPPER (0.6) by more than LIMIT_TOLERANCE_RAD
            [0.5, 0.3, 0.5],  # ordinary mid-closing position, within limits and not at any limit
        ]
    )
    valid = gb.hand_state_valid(joint_pos, start, GRIP, LOWER, UPPER)
    assert valid.tolist() == [False, True]


def test_hand_state_valid_accepts_a_joint_resting_at_its_opposite_limit_from_start():
    start = torch.tensor(
        [
            [-1.0, 0.0, 0.5],  # row0: a already starts at LOWER (its opposite limit for upward closing)
            [0.0, 0.0, 0.5],  # row1: ordinary mid-range start
        ]
    )
    joint_pos = torch.tensor(
        [
            [-1.0, 0.3, 0.5],  # row0: a still sits at LOWER -- it was already there, not driven there
            [0.5, -1.0, 0.5],  # row1: b started mid-range and reached LOWER -- driven there, still invalid
        ]
    )
    valid = gb.hand_state_valid(joint_pos, start, GRIP, LOWER, UPPER)
    assert valid.tolist() == [True, False]


FINGER_JOINTS = ["r_hj_index_2", "r_hj_index_3", "r_hj_middle_2", "r_hj_middle_3"]
FSTART = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
FGRIP = torch.tensor([1.0, 1.0, 1.0, 1.0])
FLOWER = torch.tensor([-1.0, -1.0, -1.0, -1.0])
FUPPER = torch.tensor([2.0, 2.0, 2.0, 2.0])


def test_finger_index_groups_joints_by_finger_and_rejects_unknown_names():
    names = FINGER_JOINTS + ["r_hj_thumb_1", "r_hj_ring_1", "r_hj_pinky_1"]
    ids = gb.finger_index(names)
    expected = [gb.FINGERS.index(f) for f in ("index", "index", "middle", "middle", "thumb", "ring", "pinky")]
    assert ids.tolist() == expected
    with pytest.raises(ValueError, match="joint_x"):
        gb.finger_index(["joint_x"])


def test_worst_backbend_counts_motion_against_the_closing_direction():
    joint_pos = torch.tensor([[-0.2, 0.05, 0.1, 0.05]])  # index_2 back-bent 0.2; others closing normally
    assert gb.worst_backbend(joint_pos, FSTART, FGRIP).tolist() == pytest.approx([0.2])


def test_finger_stop_freezes_the_whole_finger_when_one_joint_bends_back():
    finger_ids = gb.finger_index(FINGER_JOINTS)
    state = gb.FingerStopState.start(FSTART)
    joint_pos = torch.tensor([[0.05, -0.15, 0.05, 0.05]])  # index_3 back-bends 0.15 > FINGER_TRIGGER_BACKBEND_RAD
    state = gb.finger_stop_step(state, joint_pos, FSTART, FGRIP, FLOWER, FUPPER, finger_ids)
    assert state.triggered.tolist() == [[True, True, False, False]]
    expected_index = torch.tensor([0.05, -0.15]) + gb.FINGER_SQUEEZE_RAD
    assert torch.allclose(state.target[0, :2], expected_index)
    assert state.target[0, 2] == pytest.approx(gb.CLOSE_RATE_PER_STEP)
    assert state.target[0, 3] == pytest.approx(gb.CLOSE_RATE_PER_STEP)

    # Second step, the back-bend condition clears -- the index finger stays frozen at the first-trigger values.
    joint_pos2 = torch.tensor([[0.2, 0.0, gb.CLOSE_RATE_PER_STEP, gb.CLOSE_RATE_PER_STEP]])
    state2 = gb.finger_stop_step(state, joint_pos2, FSTART, FGRIP, FLOWER, FUPPER, finger_ids)
    assert torch.allclose(state2.target[0, :2], expected_index)
    # The untriggered middle joints keep advancing by CLOSE_RATE_PER_STEP each step.
    assert state2.target[0, 2] == pytest.approx(2 * gb.CLOSE_RATE_PER_STEP)
    assert state2.target[0, 3] == pytest.approx(2 * gb.CLOSE_RATE_PER_STEP)


def test_finger_stop_triggers_on_tracking_error():
    finger_ids = gb.finger_index(FINGER_JOINTS)
    # index_2's previous commanded target ran far ahead of the (blocked) actual joint position.
    state = gb.FingerStopState(
        close=torch.zeros_like(FSTART),
        target=torch.tensor([[0.5, 0.0, 0.0, 0.0]]),
        triggered=torch.zeros_like(FSTART, dtype=torch.bool),
        trigger_q=FSTART.clone(),
    )
    joint_pos = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # err = |0.5 - 0.0| = 0.5 > BLOCKED_ERR_RAD; no back-bend
    new_state = gb.finger_stop_step(state, joint_pos, FSTART, FGRIP, FLOWER, FUPPER, finger_ids)
    assert new_state.triggered[0, :2].tolist() == [True, True]
    assert new_state.trigger_q[0, 0] == pytest.approx(0.0)


def test_grasp_acceptable_rejects_backbend_above_the_limit():
    joint_pos = torch.tensor(
        [
            [-0.35, 0.05, 0.1, 0.05],  # index_2 back-bent 0.35 > MAX_BACKBEND_RAD
            [-0.1, 0.05, 0.1, 0.05],  # ordinary mid-closing position, within the limit
        ]
    )
    start = FSTART.expand(2, 4)
    ok = gb.grasp_acceptable(joint_pos, start, FGRIP, FLOWER, FUPPER)
    assert ok.tolist() == [False, True]


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


def test_left_arm_pregrasp_is_the_right_one_mirrored_in_y():
    tilt, yaw = torch.tensor([-20.0, -35.0]), torch.tensor([5.0, -8.0])
    right = gb.palm_rotations(tilt, yaw)
    left = gb.palm_rotations(tilt, yaw, side_sign=-1.0)
    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    assert torch.allclose(left[:, :, 0], (mirror @ right[:, :, 0:1]).squeeze(-1), atol=1e-6)  # palmar side
    assert torch.allclose(left[:, :, 2], (mirror @ right[:, :, 2:3]).squeeze(-1), atol=1e-6)  # fingers
    assert torch.allclose(torch.linalg.det(left), torch.ones(2), atol=1e-6)
    flat = gb.palm_rotations(torch.tensor([0.0]), torch.tensor([0.0]), side_sign=-1.0)[0]
    assert torch.allclose(flat[:, 0], torch.tensor([0.0, 0.0, -1.0]))  # palmar side still down
    assert torch.allclose(flat[:, 2], torch.tensor([0.0, -1.0, 0.0]))  # fingers toward -y, across the shoe from the left
    pre = gb.PreGrasp(*(torch.tensor([v]) for v in (-20.0, 0.03, 0.02, 0.01, 0.0, 0.4)))
    goal = gb.palm_goal_positions(pre, (0.25, 0.14), 0.0, 0.048, 0.309, side_sign=-1.0)[0]
    assert torch.allclose(goal, torch.tensor([0.25 + 0.01, 0.14 + 0.068, 0.339]), atol=1e-6)
    with pytest.raises(ValueError, match="side_sign"):
        gb.palm_rotations(tilt, yaw, side_sign=0.5)


def test_hand_side_reads_the_prefix_and_rejects_mixed_hands():
    left = ["l_hj_thumb_3", "l_hj_index_2"]
    assert gb.hand_side(left) == "l" and gb.side_sign(left) == -1.0
    assert gb.side_sign(FINGER_JOINTS) == 1.0
    assert gb.hand_joint_name(left, "thumb", 3) == "l_hj_thumb_3"
    assert gb.finger_index(left).tolist() == [gb.FINGERS.index("thumb"), gb.FINGERS.index("index")]
    with pytest.raises(ValueError, match="one side prefix"):
        gb.finger_index(["l_hj_thumb_3", "r_hj_index_2"])
    with pytest.raises(ValueError, match="not one of the hand joints"):
        gb.hand_joint_name(left, "pinky", 4)


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


BOOT = {key: index for index, key in enumerate(gb.LEARNED_BOOT_KEYS)}


def test_sorted_joint_columns_load_back_in_articulation_order():
    names = ["r_aj_1", "l_hj_thumb_1", "head_pan", "l_aj_1"]
    entries = {
        "joint_pos": torch.tensor([[1.0, 2.0, 3.0, 4.0]]), "joint_target": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        "shoe_pose": torch.zeros(1, 7), "palm_pose": torch.ones(1, 7),
    }
    moved, sorted_names = gb.sort_joint_columns(entries, names)
    assert sorted_names == ["head_pan", "l_aj_1", "l_hj_thumb_1", "r_aj_1"]
    assert moved["joint_pos"].tolist() == [[3.0, 4.0, 2.0, 1.0]] and moved["palm_pose"] is entries["palm_pose"]
    bank = gb.load_bank(gb.bank_document(moved, sorted_names, {"k": 1}), names, {"k": 1}, "cpu")
    assert bank.joint_pos.tolist() == [[1.0, 2.0, 3.0, 4.0]] and bank.joint_target.tolist() == [[5.0, 6.0, 7.0, 8.0]]


def test_learned_bank_metadata_records_whether_the_lift_verification_ran():
    """연결 측정용 뱅크(--skip-verify)는 들어올림 재검증을 건너뛴다 — 뱅크 자체에 그 사실이 남아야 한다."""
    common = dict(side_sign=-1.0, checkpoint="/b/ep350.pth", checkpoint_sha256="ab", stage1_reward={}, seeds=[0])
    assert gb.learned_bank_metadata(BOOT, captured=10, verified=4, **common)["lift_verified"] is True
    skipped = gb.learned_bank_metadata(BOOT, captured=10, verified=10, lift_verified=False, **common)
    assert skipped["lift_verified"] is False


def test_learned_bank_metadata_keeps_the_boot_keys_and_records_the_origin():
    meta = gb.learned_bank_metadata(BOOT, side_sign=-1.0, checkpoint="/b/ep350.pth", checkpoint_sha256="ab",
                                    stage1_reward={"g_min": 0.5}, seeds=(0, 1), captured=90, verified=70)
    assert {key: meta[key] for key in gb.LEARNED_BOOT_KEYS} == BOOT
    assert (meta["source"], meta["side_sign"], meta["seeds"], meta["captured"], meta["verified"]) == ("learned_grasp", -1.0, [0, 1], 90, 70)
    without_gains = {key: value for key, value in BOOT.items() if key != "gains"}
    with pytest.raises(ValueError, match="missing"):
        gb.learned_bank_metadata(without_gains, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=1)
    with pytest.raises(ValueError, match="verified"):
        gb.learned_bank_metadata(BOOT, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=2)
    nine_keys = {key: BOOT[key] for key in gb.BOOT_METADATA_KEYS}  # a boot without the hand backstop (spec §16)
    with pytest.raises(ValueError, match="hand_backstop"):
        gb.learned_bank_metadata(nine_keys, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=1)
    assert gb.LEARNED_BOOT_KEYS == gb.BOOT_METADATA_KEYS + ("hand_backstop",)
