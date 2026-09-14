"""Stage-1 grasp policy pure functions (design 2026-09-14 §4-§6, §10)."""

import math

import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs

NAMES = ("l_hj_thumb_3", "l_hj_index_3", "l_hj_index_4", "l_hj_index_1")
CFG = gs.Stage1RewardCfg(q_lo=0.2, q_hi=0.6)


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


def test_reward_cfg_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(q_lo=0.5, q_hi=0.5)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(g_min=0.0)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(latch_steps=25, success_steps=20)
