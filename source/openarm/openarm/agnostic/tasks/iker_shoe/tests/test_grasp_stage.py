"""Stage-1 grasp policy pure functions (design 2026-09-14 §4-§6, §10)."""

import math

import pytest
import torch

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb, grasp_stage as gs, layout

NAMES = ("l_hj_thumb_3", "l_hj_index_3", "l_hj_index_4", "l_hj_index_1")
CFG = gs.Stage1RewardCfg(g_min=0.5, q_lo=0.2, q_hi=0.6)


def _step(state, **overrides):
    n = state.closest_palm.shape[0]
    inputs = dict(
        palm_gap=torch.full((n,), 0.10),
        dz_free=torch.zeros(n),
        palm_shoe_dist=torch.full((n,), 0.30),
        rel_speed=torch.zeros(n),
        shoe_speed=torch.zeros(n),
        q=torch.zeros(n),
        hand_floor_depth=torch.zeros(n),
        arm_speed_sum=torch.zeros(n),
        hand_speed_sum=torch.zeros(n),
        shoe_shift_xy=torch.zeros(n),
        thumb_curl=torch.full((n,), 0.5),
    )
    inputs.update(overrides)
    return gs.stage1_step(state, CFG, **inputs)


def _held(n):
    return dict(dz_free=torch.full((n,), 0.06), palm_shoe_dist=torch.full((n,), 0.12), rel_speed=torch.zeros(n))


def test_hand_ema_alpha_matches_the_60hz_time_constant():
    assert gs.HAND_EMA_ALPHA == pytest.approx(1.0 - 0.9**6)
    assert (1.0 - gs.HAND_EMA_ALPHA) == pytest.approx(0.9**6)


def test_hand_action_limits_narrow_only_and_reject_unmatched_or_empty():
    lo, hi = torch.full((4,), -1.571), torch.full((4,), 1.571)
    out_lo, out_hi = gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_[34]$": (0.0, None), r"l_hj_thumb_[34]$": (None, 0.0)})
    assert out_lo.tolist() == pytest.approx([-1.571, 0.0, 0.0, -1.571])
    assert out_hi.tolist() == pytest.approx([0.0, 1.571, 1.571, 1.571])
    widened, _ = gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_1$": (-3.0, None)})
    assert widened[3] == pytest.approx(-1.571)
    with pytest.raises(ValueError, match="matches no joint"):
        gs.hand_action_limits(NAMES, lo, hi, {r"r_hj_index_[34]$": (0.0, None)})
    with pytest.raises(ValueError, match="zero width"):
        gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_1$": (0.5, 0.5)})


def test_frozen_hand_override_pins_joint_roles_at_the_open_pose():
    names = ("l_hj_thumb_2", "l_hj_index_2", "l_hj_pinky_2")
    override = gs.frozen_hand_override(names, (1.57, 0.3, 0.0), ("thumb_2", "pinky_2"), halfwidth=0.01)
    lo, hi = gs.hand_action_limits(names, torch.tensor([0.0, 0.0, -1.571]), torch.tensor([2.705, 2.0, 0.0]), override)
    assert lo.tolist() == pytest.approx([1.56, 0.0, -0.01])
    assert hi.tolist() == pytest.approx([1.58, 2.0, 0.0])
    with pytest.raises(ValueError, match="matches 0"):
        gs.frozen_hand_override(names, (1.57, 0.3, 0.0), ("ring_2",))
    with pytest.raises(ValueError, match="open pose"):
        gs.frozen_hand_override(names, (1.57, 0.3), ("thumb_2",))


def test_stage1_hand_limits_keep_the_profile_narrowing_on_frozen_joints():
    # A merged override dict let the frozen entry replace a profile entry with the same pattern (pinky_2 hi 0.0 -> 0.01).
    names = ("l_hj_thumb_2", "l_hj_pinky_2")
    profile = {r"l_hj_pinky_2$": (None, 0.0)}
    lo, hi = gs.stage1_hand_limits(names, torch.tensor([0.0, -1.571]), torch.tensor([2.705, 0.5]), profile, (1.57, 0.0),
                                   ("thumb_2", "pinky_2"))
    assert lo.tolist() == pytest.approx([1.56, -0.01])
    assert hi.tolist() == pytest.approx([1.58, 0.0])


def test_backstop_limits_move_the_opening_side_limit_to_the_open_pose():
    names = ("l_hj_thumb_3", "l_hj_index_3")
    lo, hi = torch.tensor([-1.5708, -1.5708]), torch.tensor([1.5708, 1.5708])
    open_pose, grip_pose = (0.0, 0.0), (-1.8, 1.8)
    thumb = gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("thumb_3",))
    assert list(thumb) == ["l_hj_thumb_3"] and thumb["l_hj_thumb_3"] == pytest.approx((-1.5708, 0.0))
    index = gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("index_3",))
    assert index["l_hj_index_3"] == pytest.approx((0.0, 1.5708))
    with pytest.raises(ValueError, match="matches 0"):
        gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("pinky_3",))
    with pytest.raises(ValueError, match="closing direction"):
        gs.backstop_limits(names, lo, hi, open_pose, (0.0, 1.8), ("thumb_3",))
    with pytest.raises(ValueError, match="outside"):
        gs.backstop_limits(names, lo, hi, (2.0, 0.0), grip_pose, ("thumb_3",))


def test_the_profile_thumb_closes_toward_its_side_sign_and_is_backstopped_on_the_other_side():
    # the left hand is the right one mirrored: thumb_3 closes in the side-sign direction (robot_profiles, FK-verified 2026-09-13)
    prof = robot_profiles.PROFILES[layout.PROFILE_NAME]
    names = prof.hand_joint_names
    i = gs.role_joint_index(names, "thumb_3")
    sign = gb.side_sign(names)
    open_q, grip_q = torch.tensor(prof.hand_open_pose[i]), torch.tensor(prof.hand_grip_pose[i])
    assert gs.closing_travel(open_q + 0.1 * sign, open_q, grip_q).item() == pytest.approx(0.1)
    hard = torch.full((len(names),), 1.5708)
    lo, hi = gs.backstop_limits(names, -hard, hard, prof.hand_open_pose, prof.hand_grip_pose, ("thumb_3",))[names[i]]
    assert (hi if sign < 0 else lo) == pytest.approx(prof.hand_open_pose[i])
    assert (lo if sign < 0 else hi) == pytest.approx(sign * 1.5708)


def test_hand_targets_map_linearly_filter_and_clamp():
    lo, hi = torch.tensor([0.0, -1.0]), torch.tensor([2.0, 1.0])
    previous = torch.tensor([[1.0, 0.0]])
    full_close = gs.hand_targets(torch.tensor([[1.0, -1.0]]), lo, hi, previous, alpha=1.0)
    assert torch.allclose(full_close, torch.tensor([[2.0, -1.0]]))
    filtered = gs.hand_targets(torch.tensor([[1.0, 1.0]]), lo, hi, previous, alpha=0.5)
    assert torch.allclose(filtered, torch.tensor([[1.5, 0.5]]))
    beyond = gs.hand_targets(torch.tensor([[5.0, -5.0]]), lo, hi, torch.tensor([[3.0, -3.0]]), alpha=0.1)
    assert torch.allclose(beyond, torch.tensor([[2.0, -1.0]]))
    with pytest.raises(ValueError):
        gs.hand_targets(torch.zeros(1, 2), lo, hi, previous, alpha=0.0)


def test_normalized_targets_span_minus_one_to_one():
    lo, hi = torch.tensor([0.0, -1.0]), torch.tensor([2.0, 1.0])
    assert torch.allclose(gs.normalized_targets(torch.tensor([[0.0, 1.0]]), lo, hi), torch.tensor([[-1.0, 1.0]]))
    assert torch.allclose(gs.normalized_targets(torch.tensor([[1.0, 0.0]]), lo, hi), torch.tensor([[0.0, 0.0]]))


def test_surface_subsample_is_deterministic_and_keeps_both_ends():
    points = [[float(i), 0.0, 0.0] for i in range(1000)]
    first, second = gs.surface_subsample(points, 256), gs.surface_subsample(points, 256)
    assert first.shape == (256, 3) and torch.equal(first, second)
    assert first[0, 0] == 0.0 and first[-1, 0] == 999.0
    with pytest.raises(ValueError):
        gs.surface_subsample(points[:10], 256)


def _quality(link_pos, palm_normal=(0.0, 0.0, -1.0)):
    surface = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    return gs.grasp_quality(
        link_pos=link_pos,
        finger_sizes=(1, 1),
        surface=surface,
        palm_pos=torch.tensor([[0.05, 0.0, 0.1]]),
        palm_normal=torch.tensor([palm_normal]),
        shoe_center=torch.tensor([[0.05, 0.0, 0.0]]),
    )


def test_grasp_quality_is_a_soft_min_and_the_weakest_finger_dominates():
    touching, _, _ = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]))
    assert touching.item() == pytest.approx(1.0)
    one_far, w_f, _ = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.2]]]))
    assert w_f[0, 1].item() == pytest.approx(math.exp(-0.2 / gs.KERNEL_TAU_M), abs=1e-6)
    assert w_f.min().item() - 1e-6 <= one_far.item() <= w_f.mean().item() + 1e-6
    assert one_far.item() < 0.2


def test_grasp_quality_is_zero_when_the_palm_faces_away():
    away, w_f, palm_cos = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]), palm_normal=(0.0, 0.0, 1.0))
    assert palm_cos.item() < 0.0 and away.item() == 0.0 and w_f.min().item() == pytest.approx(1.0)


def test_g_factor_runs_from_g_min_to_one_between_q_lo_and_q_hi():
    g = gs.g_factor(torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0]), CFG)
    assert g.tolist() == pytest.approx([0.5, 0.5, 0.75, 1.0, 1.0])


def test_free_lift_height_is_zero_while_the_hull_touches_the_widened_rack_footprint():
    start = torch.tensor([0.20, 0.20, 0.20])
    lifted = torch.tensor([[[0.30, 0.10, 0.28], [0.30, 0.20, 0.30]]])
    beside = torch.tensor([[[0.30, 0.045, 0.28], [0.30, 0.20, 0.30]]])
    touching = torch.tensor([[[0.30, 0.035, 0.28], [0.30, 0.20, 0.30]]])
    surfaces = torch.cat([lifted, beside, touching])
    dz = gs.free_lift_height(surfaces, start, rack_x=(0.11, 0.43), rack_y=(-0.33, 0.03))
    assert dz.tolist() == pytest.approx([0.08, 0.08, 0.0])


def test_idle_policy_earns_nothing():
    state = gs.Stage1State.start(3)
    total = torch.zeros(3)
    for _ in range(120):
        step = _step(state)
        total, state = total + step.reward, step.state
    assert torch.equal(total, torch.zeros(3))


def test_progress_terms_pay_only_new_bests_and_never_for_retreat():
    state = gs.Stage1State.start(1)
    rewards = []
    for gap, dz in ((0.10, 0.0), (0.08, 0.02), (0.09, 0.015), (0.06, 0.03)):
        step = _step(state, palm_gap=torch.tensor([gap]), dz_free=torch.tensor([dz]))
        rewards.append((step.terms["palm_progress"].item(), step.terms["lift_progress"].item()))
        state = step.state
    assert rewards[0] == pytest.approx((0.0, 0.0))
    assert rewards[1] == pytest.approx((50 * 0.02, 2000 * 0.01))  # 2 cm rise, 1 cm of it past the dead band
    assert rewards[2] == pytest.approx((0.0, 0.0))
    assert rewards[3] == pytest.approx((50 * 0.02, 2000 * 0.01))


def test_a_reset_bounce_inside_the_dead_band_pays_nothing():
    state = gs.Stage1State.start(1)
    total = 0.0
    for dz in (0.0, 0.004, 0.009, 0.002, 0.0):
        step = _step(state, dz_free=torch.tensor([dz]))
        total, state = total + step.reward.item(), step.state
    assert total == 0.0
    lifted = _step(state, dz_free=torch.tensor([0.05]))
    assert lifted.terms["lift_progress"].item() == pytest.approx(2000 * (0.05 - CFG.lift_deadband_m))


def test_lift_progress_is_paid_only_inside_the_hold_zone():
    state = gs.Stage1State.start(1)
    far = _step(state, dz_free=torch.tensor([0.03]), shoe_shift_xy=torch.tensor([0.2]))
    assert far.terms["lift_progress"].item() == 0.0
    too_high = _step(far.state, dz_free=torch.tensor([0.2]))
    assert too_high.terms["lift_progress"].item() == 0.0
    back = _step(too_high.state, dz_free=torch.tensor([0.03]))
    assert back.terms["lift_progress"].item() == pytest.approx(2000 * (0.03 - CFG.lift_deadband_m))
    assert back.state.best_lift.item() == pytest.approx(0.03 - CFG.lift_deadband_m)


def test_lift_progress_needs_the_thumb_closing():
    state = gs.Stage1State.start(1)
    open_thumb = _step(state, dz_free=torch.tensor([0.03]), thumb_curl=torch.tensor([0.0]))
    assert open_thumb.terms["lift_progress"].item() == 0.0
    bent_back = _step(open_thumb.state, dz_free=torch.tensor([0.03]), thumb_curl=torch.tensor([-0.4]))
    assert bent_back.terms["lift_progress"].item() == 0.0
    closing = _step(bent_back.state, dz_free=torch.tensor([0.03]), thumb_curl=torch.tensor([0.06]))
    assert closing.terms["lift_progress"].item() == pytest.approx(2000 * (0.03 - CFG.lift_deadband_m))


def test_lift_progress_needs_the_shoe_not_sliding_fast_in_the_hand():
    state = gs.Stage1State.start(1)
    sliding = _step(state, dz_free=torch.tensor([0.03]), rel_speed=torch.tensor([0.3]))
    assert sliding.terms["lift_progress"].item() == 0.0
    at_the_limit = _step(sliding.state, dz_free=torch.tensor([0.03]), rel_speed=torch.tensor([CFG.progress_rel_speed]))
    assert at_the_limit.terms["lift_progress"].item() == 0.0
    # slower than the progress limit but faster than the hold's limit: progress is paid, a hold would not count
    moving = _step(at_the_limit.state, dz_free=torch.tensor([0.03]), rel_speed=torch.tensor([0.1]))
    assert moving.terms["lift_progress"].item() == pytest.approx(2000 * (0.03 - CFG.lift_deadband_m))


def test_reward_cfg_progress_rel_speed_is_looser_than_the_hold():
    assert gs.Stage1RewardCfg().progress_rel_speed == 0.20
    with pytest.raises(ValueError, match="progress_rel_speed"):
        gs.Stage1RewardCfg(progress_rel_speed=0.01)
    with pytest.raises(ValueError, match="progress_rel_speed"):
        gs.Stage1RewardCfg(hold_rel_speed=0.0)


def test_lift_bonus_needs_three_consecutive_held_steps_and_pays_once():
    state = gs.Stage1State.start(1)
    paid = []
    for held in (True, True, False, True, True, True, True):
        step = _step(state, **(_held(1) if held else {}), q=torch.tensor([0.6]))
        paid.append(step.terms["lift_bonus"].item())
        state = step.state
    assert paid == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0])


def test_a_fling_without_the_palm_near_or_moving_with_the_shoe_is_not_held():
    state = gs.Stage1State.start(2)
    for _ in range(5):
        step = _step(state, dz_free=torch.full((2,), 0.08), palm_shoe_dist=torch.tensor([0.30, 0.10]),
                     rel_speed=torch.tensor([0.0, 0.5]))
        assert not step.held.any() and step.terms["lift_bonus"].sum() == 0.0
        state = step.state


def test_success_needs_twenty_consecutive_held_steps_resets_on_a_break_and_pays_once():
    state = gs.Stage1State.start(1)
    success_steps = []
    pattern = [True] * 10 + [False] + [True] * 25
    for index, held in enumerate(pattern):
        step = _step(state, **(_held(1) if held else {}), q=torch.tensor([0.0]))
        if step.success.item():
            success_steps.append(index)
            assert step.terms["success_bonus"].item() == pytest.approx(500.0)
        state = step.state
    assert success_steps == [30]


def test_success_waits_for_the_shoe_to_be_slow():
    state = gs.Stage1State.start(1)
    for _ in range(25):
        step = _step(state, **_held(1), shoe_speed=torch.tensor([0.2]))
        assert not step.success.any()
        state = step.state


def test_a_hold_counts_only_near_the_start_below_the_lift_ceiling_with_the_thumb_closing():
    state = gs.Stage1State.start(5)
    inputs = {
        **_held(5),
        "dz_free": torch.tensor([0.14, 0.16, 0.06, 0.06, 0.06]),
        "shoe_shift_xy": torch.tensor([0.09, 0.0, 0.11, 0.0, 0.0]),
        "thumb_curl": torch.tensor([0.06, 0.5, 0.5, 0.04, -0.3]),
    }
    step = _step(state, **inputs)
    assert step.held.tolist() == [True, False, False, False, False]


def test_a_hold_far_from_the_start_never_latches_or_succeeds():
    state = gs.Stage1State.start(1)
    for _ in range(25):
        step = _step(state, **_held(1), shoe_shift_xy=torch.tensor([0.5]))
        assert not step.held.any() and not step.success.any() and step.terms["lift_bonus"].item() == 0.0
        state = step.state


def test_reward_cfg_revision_3_defaults_and_rejections():
    cfg = gs.Stage1RewardCfg()
    assert (cfg.lift_max_m, cfg.hold_xy_radius_m, cfg.thumb_curl_min_rad) == (0.15, 0.10, 0.05)
    with pytest.raises(ValueError, match="lift_max_m"):
        gs.Stage1RewardCfg(lift_max_m=0.05)
    with pytest.raises(ValueError, match="hold_xy_radius_m"):
        gs.Stage1RewardCfg(hold_xy_radius_m=0.0)
    assert gs.Stage1RewardCfg(thumb_curl_min_rad=-1.0).thumb_curl_min_rad == -1.0  # negative switches the thumb condition off


def test_closing_travel_is_signed_by_the_closing_direction():
    open_pose, grip_pose = torch.tensor([0.0, 0.0]), torch.tensor([-1.8, 1.9])
    q = torch.tensor([[-0.3, 0.3], [0.95, -0.1]])
    assert torch.allclose(gs.closing_travel(q, open_pose, grip_pose), torch.tensor([[0.3, 0.3], [-0.95, -0.1]]))
    with pytest.raises(ValueError, match="closing direction"):
        gs.closing_travel(q, open_pose, torch.tensor([0.0, 1.0]))


def test_role_joint_index_needs_exactly_one_match():
    names = ("l_hj_thumb_3", "l_hj_index_3", "l_hj_thumb_4")
    assert gs.role_joint_index(names, "thumb_3") == 0
    with pytest.raises(ValueError, match="matches 0"):
        gs.role_joint_index(names, "pinky_3")
    with pytest.raises(ValueError, match="matches 2"):
        gs.role_joint_index(("l_hj_thumb_3", "r_hj_thumb_3"), "thumb_3")


def test_progress_stops_after_the_latch():
    state = gs.Stage1State.start(1)
    for _ in range(3):
        state = _step(state, **_held(1)).state
    assert state.latched.item()
    after = _step(state, palm_gap=torch.tensor([0.0]), dz_free=torch.tensor([0.06]), palm_shoe_dist=torch.tensor([0.12]))
    assert after.terms["palm_progress"].item() == 0.0 and after.terms["lift_progress"].item() == 0.0


def test_hand_floor_penalty_is_proportional_and_capped():
    step = _step(gs.Stage1State.start(3), hand_floor_depth=torch.tensor([0.0, 0.1, 2.0]))
    assert step.terms["hand_floor"].tolist() == pytest.approx([0.0, -1.0, -5.0])


def test_reset_rows_restores_only_the_given_envs():
    state = gs.Stage1State.start(2)
    for _ in range(3):
        state = _step(state, **_held(2)).state
    reset = state.reset_rows(torch.tensor([1]))
    assert reset.latched.tolist() == [True, False]
    assert reset.hold_count.tolist() == [3, 0]
    assert reset.closest_palm.tolist() == pytest.approx([0.10, -1.0])
    assert state.latched.tolist() == [True, True]


def test_reward_cfg_defaults_to_phase_a_with_the_grasp_factor_off():
    cfg = gs.Stage1RewardCfg()
    assert cfg.g_min == 1.0
    assert gs.g_factor(torch.tensor([0.0, 0.5, 1.0]), cfg).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_reward_cfg_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(q_lo=0.5, q_hi=0.5)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(g_min=0.0)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(latch_steps=25, success_steps=20)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(lift_deadband_m=0.05, lift_height_m=0.05)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(success_bonus=-1.0)
    overridden = gs.Stage1RewardCfg()
    overridden.q_hi = overridden.q_lo  # a hydra override assigns with setattr and skips __post_init__
    with pytest.raises(ValueError, match="q_lo"):
        overridden.validate()


def test_quality_calibration_document_round_trips_and_a_file_without_schema_is_rejected(tmp_path):
    doc = gs.quality_calibration_document(0.12, 0.34, checkpoint="/a/ep250.pth", latch_events=80)
    assert doc["schema"] == 1 and doc["checkpoint"] == "/a/ep250.pth" and (doc["q_lo"], doc["q_hi"]) == (0.12, 0.34)
    path = run_files.write_json(tmp_path / "grasp_quality_calibration.json", doc)
    assert gs.read_quality_calibration(path) == (0.12, 0.34)
    # what measure_grasp_quality.py wrote before 2026-09-14: the environment could not read it
    run_files.write_json(tmp_path / "old.json", {"checkpoint": "/a/ep250.pth", "q_lo": 0.12, "q_hi": 0.34})
    with pytest.raises(ValueError, match="schema"):
        gs.read_quality_calibration(tmp_path / "old.json")
    for lo, hi in ((0.3, 0.3), (-0.1, 0.2), (0.1, 1.2), (math.nan, 0.2)):
        with pytest.raises(ValueError, match="q_lo < q_hi"):
            gs.quality_calibration_document(lo, hi)
    with pytest.raises(ValueError, match="may not set"):
        gs.quality_calibration_document(0.1, 0.2, schema=2)


def test_capture_rows_keep_each_envs_first_success_until_cleared():
    capture = gs.SuccessCapture.empty(3, 4)
    first = dict(joint_pos=torch.ones(3, 4), joint_target=2 * torch.ones(3, 4), shoe_pose=3 * torch.ones(3, 7), palm_pose=4 * torch.ones(3, 7))
    capture = gs.capture_rows(capture, torch.tensor([True, False, True]), step=5, **first)
    later = {key: 10 * value for key, value in first.items()}
    capture = gs.capture_rows(capture, torch.tensor([True, True, False]), step=9, **later)
    assert capture.valid.tolist() == [True, True, True]
    assert capture.step.tolist() == [5, 9, 5]
    assert capture.joint_pos[:, 0].tolist() == [1.0, 10.0, 1.0]
    assert capture.joint_target[:, 0].tolist() == [2.0, 20.0, 2.0]
    assert capture.shoe_pose[:, 6].tolist() == [3.0, 30.0, 3.0] and capture.palm_pose[:, 0].tolist() == [4.0, 40.0, 4.0]
    cleared = gs.SuccessCapture.empty(3, 4)
    assert not cleared.valid.any() and cleared.step.tolist() == [-1, -1, -1]


def test_holding_targets_pin_the_arm_at_its_position_and_the_hand_at_its_ema_target():
    targets = torch.zeros(2, 6)
    joint_pos = torch.arange(12, dtype=torch.float32).view(2, 6)
    hand = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    out = gs.holding_targets(targets, joint_pos, [0, 1], torch.tensor([3, 4]), hand)
    assert out.tolist() == [[0.0, 1.0, 0.0, 7.0, 8.0, 0.0], [6.0, 7.0, 0.0, 9.0, 10.0, 0.0]]
    assert float(targets.abs().sum()) == 0.0


def test_the_normalized_ema_target_is_a_fixed_point_of_the_hand_law():
    lo, hi = torch.tensor([-0.5, 0.0]), torch.tensor([0.5, 1.2])
    previous = torch.tensor([[0.1, 0.9], [-0.5, 1.2]])
    held = gs.hand_targets(gs.normalized_targets(previous, lo, hi), lo, hi, previous)
    assert torch.allclose(held, previous, atol=1e-6)
