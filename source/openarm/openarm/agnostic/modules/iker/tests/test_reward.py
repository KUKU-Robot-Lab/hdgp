"""reward — legacy IKER reward port (no Isaac)."""

import pytest
import torch

from openarm.agnostic.modules.iker.reward import (
    IkerRewardCfg,
    RewardOutput,
    compute_reward_and_termination,
    reward_cfg_for_start_support,
    transform_keypoints,
)

OFFSETS = torch.tensor([[0.0, 0.0, 0.12], [0.0, 0.0, -0.12], [0.0, 0.075, 0.0], [0.0, -0.075, 0.0]])
IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _keypoints(n: int, base: torch.Tensor) -> torch.Tensor:
    return base.reshape(1, 4, 3).repeat(n, 1, 1)


def _inputs(n: int = 2, **overrides) -> dict:
    target = _keypoints(n, OFFSETS + torch.tensor([0.6, 0.1, 1.2]))
    init = _keypoints(n, OFFSETS + torch.tensor([0.3, 0.0, 1.2]))
    inputs = dict(
        actions=torch.zeros(n, 6),
        eef_pos=torch.tensor([[0.3, 0.0, 1.3]]).repeat(n, 1),
        object_pos=torch.tensor([[0.3, 0.0, 1.2]]).repeat(n, 1),
        current_keypoints=init.clone(),
        init_keypoints=init,
        target_keypoints=target,
        progress=torch.zeros(n, dtype=torch.long),
        success_count=torch.zeros(n),
        failure_count=torch.zeros(n),
        force_penalty=torch.zeros(n),
    )
    inputs.update(overrides)
    return inputs


def _legacy_compute_xarm_reward(reset_buf, progress_buf, success_buf, fail_buf, actions, states, max_episode_length):
    """Transcription of shoe_place.py:722-778 without TorchScript; `force` and `eef_acc` are passed as zeros."""
    d = torch.norm(states["obj_pos"] - states["eef_pos"], dim=-1)
    d = torch.clamp(d, min=0.05, max=0.5)
    dist_reward = 1 - torch.tanh(5.0 * d)
    action_penalty = torch.sum(actions**2, dim=-1)
    under_penalty = (states["eef_pos"][:, 2] < 1.0) * 1
    target_dir_vec = states["target"] - states["init_pos"]
    dist_dir_vec = states["track_pos"] - states["init_pos"]
    cos_theta = torch.sum(target_dir_vec * dist_dir_vec, dim=-1) / (
        torch.norm(target_dir_vec, dim=-1) * torch.norm(dist_dir_vec, dim=-1) + 0.1
    )
    dir_reward = torch.clamp(cos_theta, min=0, max=1.0)
    dir_reward = torch.sum(dir_reward, dim=-1) / states["track_pos"].shape[1]
    target_loc = states["target"]
    d_target = torch.norm(states["track_pos"] - target_loc, dim=-1) / torch.norm(target_dir_vec, dim=-1)
    align_reward = torch.clamp(0.5 * (1 - d_target**2) + 0.5 * (1 - torch.tanh(2.0 * d_target)), min=0.0, max=1.0)
    align_reward = torch.sum(align_reward, dim=-1) / states["track_pos"].shape[1]
    success_buf = success_buf + (torch.sum(d_target, dim=-1) / states["track_pos"].shape[1] <= 0.10) * 1.0
    fall_penalty = (states["obj_pos"][:, 2] < 0.9) * 1
    fail_buf = fail_buf + fall_penalty
    force_penalty = 0.5 * torch.clamp(torch.norm(states["force"][:, 8], dim=-1) * 0.1, min=0.0, max=1.0)
    force_penalty += 0.5 * torch.clamp(10 * torch.norm(states["eef_acc"][:, :3], dim=-1), min=0, max=1.0)
    success = success_buf > 20
    fail = fail_buf > 20
    bonus = torch.max(10 * (max_episode_length - progress_buf), 300 * torch.ones_like(progress_buf))
    rewards = (
        0.2 * dist_reward + 4 * align_reward + bonus * success + 2.0 * dir_reward - fall_penalty
        - 0.1 * under_penalty - 750 * fail - 0.20 * action_penalty - 3.0 * force_penalty
    )
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.logical_or(reset_buf, success)
    reset_buf = torch.logical_or(reset_buf, fail)
    return rewards, reset_buf, success_buf, fail_buf


def test_matches_the_legacy_function_on_random_states():
    torch.manual_seed(0)
    n = 256
    init = torch.rand(n, 4, 3) + torch.tensor([0.0, 0.0, 1.0])
    target = init + 0.3 * torch.randn(n, 4, 3)
    near = (torch.arange(n) % 2 == 0)[:, None, None]
    current = torch.where(near, target + 0.01 * torch.randn(n, 4, 3), init + 0.2 * torch.randn(n, 4, 3))
    eef = torch.rand(n, 3) + torch.tensor([0.0, 0.0, 0.5])
    obj = torch.rand(n, 3) + torch.tensor([0.0, 0.0, 0.5])
    actions = torch.randn(n, 6)
    progress = torch.randint(0, 200, (n,))
    success = torch.randint(0, 25, (n,)).float()
    fail = torch.randint(0, 25, (n,)).float()
    states = {
        "obj_pos": obj, "eef_pos": eef, "target": target, "init_pos": init, "track_pos": current,
        "force": torch.zeros(n, 9, 3), "eef_acc": torch.zeros(n, 6),
    }
    legacy = _legacy_compute_xarm_reward(torch.zeros(n, dtype=torch.long), progress, success, fail, actions, states, 200.0)
    ours = compute_reward_and_termination(
        actions=actions, eef_pos=eef, object_pos=obj, current_keypoints=current, init_keypoints=init,
        target_keypoints=target, progress=progress, success_count=success, failure_count=fail,
    )
    assert torch.allclose(ours.reward, legacy[0].float(), atol=1e-4)
    assert torch.equal(ours.terminated, legacy[1].bool())
    assert torch.equal(ours.success_count, legacy[2].float())
    assert torch.equal(ours.failure_count, legacy[3].float())
    assert 0 < int(ours.success_count.gt(success).sum()) < n  # both branches of the success counter ran


def test_start_support_moves_the_legacy_height_offsets():
    cfg = reward_cfg_for_start_support(0.205)
    assert cfg.under_height == pytest.approx(0.268)
    assert cfg.fall_height == pytest.approx(0.168)
    assert IkerRewardCfg().under_height == 1.0 and IkerRewardCfg().fall_height == 0.9


def test_transform_keypoints_rotates_offsets():
    half = 0.5**0.5
    out = transform_keypoints(torch.zeros(1, 3), torch.tensor([[half, 0.0, 0.0, half]]), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    with pytest.raises(ValueError, match="offsets"):
        transform_keypoints(torch.zeros(1, 3), IDENTITY, torch.zeros(4, 2))


def test_output_shapes_and_types():
    out = compute_reward_and_termination(**_inputs(3))
    assert isinstance(out, RewardOutput)
    assert out.reward.shape == (3,) and out.terminated.dtype == torch.bool


def test_zero_action_avoids_the_action_penalty():
    r0 = compute_reward_and_termination(**_inputs(1)).reward
    r1 = compute_reward_and_termination(**_inputs(1, actions=torch.ones(1, 6))).reward
    assert torch.isclose(r0 - r1, torch.tensor([0.20 * 6.0]))


def test_sustained_success_terminates_and_pays_the_bonus():
    inputs = _inputs(1, success_count=torch.tensor([20.0]))
    inputs["current_keypoints"] = inputs["target_keypoints"].clone()
    out = compute_reward_and_termination(**inputs)
    assert out.success_count.item() == 21.0 and out.terminated.item() is True
    assert out.reward.item() > 1000.0


def test_success_counter_is_cumulative_not_consecutive():
    inputs = _inputs(1, success_count=torch.tensor([7.0]))  # the object is away from the target this step
    assert compute_reward_and_termination(**inputs).success_count.item() == 7.0


def test_sustained_failure_terminates_with_the_penalty():
    inputs = _inputs(1, object_pos=torch.tensor([[0.3, 0.0, 0.5]]), failure_count=torch.tensor([20.0]))
    out = compute_reward_and_termination(**inputs)
    assert out.terminated.item() is True and out.reward.item() < -700.0


def test_timeout_terminates_only_on_the_last_step():
    assert compute_reward_and_termination(**_inputs(1, progress=torch.tensor([198]))).terminated.item() is False
    assert compute_reward_and_termination(**_inputs(1, progress=torch.tensor([199]))).terminated.item() is True


def test_bonus_floor_is_300_late_in_the_episode():
    late = _inputs(1, progress=torch.tensor([190]), success_count=torch.tensor([20.0]))
    late["current_keypoints"] = late["target_keypoints"].clone()
    base = _inputs(1, progress=torch.tensor([190]))
    base["current_keypoints"] = base["target_keypoints"].clone()
    diff = compute_reward_and_termination(**late).reward - compute_reward_and_termination(**base).reward
    assert torch.isclose(diff, torch.tensor([300.0]))


def test_force_penalty_is_off_by_default_and_scalable():
    off = compute_reward_and_termination(**_inputs(1, force_penalty=torch.ones(1)))
    on = compute_reward_and_termination(**_inputs(1, force_penalty=torch.ones(1)), cfg=IkerRewardCfg(force_penalty_scale=3.0))
    assert torch.isclose(off.reward - on.reward, torch.tensor([3.0]))


def test_non_finite_keypoints_raise():
    inputs = _inputs(1)
    inputs["current_keypoints"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="current_keypoints"):
        compute_reward_and_termination(**inputs)


def test_cfg_fields_are_assignable_for_the_hydra_round_trip():
    cfg = IkerRewardCfg()
    cfg.max_episode_length = 300
    assert cfg.max_episode_length == 300
