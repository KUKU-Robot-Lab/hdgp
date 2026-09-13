"""Fixed IKER reward and termination: the legacy ``compute_xarm_reward`` in pure PyTorch (design spec §7).

Source, read 2026-09-13: ``repo/skill_gen/IKER/isaacgymenvs/tasks/shoe_place.py:722-778``.
The equations and constants are unchanged; the interface around them differs as follows.

| | legacy | this module |
|---|---|---|
| end effector | xArm ``eef_pos`` | the caller's palm body position |
| heights | absolute 1.0 (under) and 0.9 (fall) in the lifted legacy world | ``under_height`` / ``fall_height`` in the caller's frame; ``reward_cfg_for_start_support`` keeps the legacy offsets from the start support (legacy slab 0.937) |
| force penalty | ``3.0 * (contact force + eef acceleration)`` | ``force_penalty_scale`` 0.0 until a signal exists |
| orientation term | computed, never added to the reward | not computed |
| counters | cumulative within an episode, zeroed on reset (``shoe_place.py:632-633``) | returned; the environment zeroes them on reset |
| termination | timeout or success (count > 20) or fail (count > 20) | same; the environment reports the timeout as truncation |

Quaternions entering this module are ``wxyz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NUM_KEYPOINTS = 4
LEGACY_START_SUPPORT_Z = 0.937
LEGACY_UNDER_HEIGHT = 1.0
LEGACY_FALL_HEIGHT = 0.9


@dataclass
class IkerRewardCfg:
    """Legacy constants with the implicit numbers named.

    Not frozen: Isaac Lab's hydra round trip assigns every nested config field with ``setattr``
    (``isaaclab/utils/dict.py``) and a frozen dataclass raises ``FrozenInstanceError`` there.
    """

    max_episode_length: int = 200
    under_height: float = LEGACY_UNDER_HEIGHT
    fall_height: float = LEGACY_FALL_HEIGHT
    success_tolerance: float = 0.10
    sustain_steps: int = 20
    dist_scale: float = 0.2
    align_scale: float = 4.0
    dir_scale: float = 2.0
    under_scale: float = 0.1
    fail_scale: float = 750.0
    action_scale: float = 0.20
    force_penalty_scale: float = 0.0
    bonus_per_remaining_step: float = 10.0
    bonus_floor: float = 300.0


@dataclass(frozen=True)
class RewardOutput:
    reward: torch.Tensor
    terminated: torch.Tensor
    success_count: torch.Tensor
    failure_count: torch.Tensor


def reward_cfg_for_start_support(start_support_z: float, **overrides) -> IkerRewardCfg:
    """Legacy height thresholds moved to a scene whose moving object starts on ``start_support_z``."""
    return IkerRewardCfg(
        under_height=start_support_z + (LEGACY_UNDER_HEIGHT - LEGACY_START_SUPPORT_Z),
        fall_height=start_support_z - (LEGACY_START_SUPPORT_Z - LEGACY_FALL_HEIGHT),
        **overrides,
    )


def transform_keypoints(position: torch.Tensor, quaternion_wxyz: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Map body-frame keypoint ``offsets`` (K, 3) to the world frame -> (N, K, 3)."""
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"offsets must have shape (K, 3), got {tuple(offsets.shape)}")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position must have shape (N, 3), got {tuple(position.shape)}")
    if quaternion_wxyz.shape != (position.shape[0], 4):
        raise ValueError(f"quaternion_wxyz must have shape ({position.shape[0]}, 4), got {tuple(quaternion_wxyz.shape)}")
    n, k = position.shape[0], offsets.shape[0]
    quat = quaternion_wxyz[:, None, :].expand(n, k, 4)
    vec = offsets.to(position)[None].expand(n, k, 3)
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return position[:, None, :] + vec + w * t + torch.cross(xyz, t, dim=-1)


def compute_reward_and_termination(
    actions: torch.Tensor,
    eef_pos: torch.Tensor,
    object_pos: torch.Tensor,
    current_keypoints: torch.Tensor,
    init_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    progress: torch.Tensor,
    success_count: torch.Tensor,
    failure_count: torch.Tensor,
    force_penalty: torch.Tensor | None = None,
    cfg: IkerRewardCfg | None = None,
) -> RewardOutput:
    """Legacy ``compute_xarm_reward`` with the same equations. All positions share one frame."""
    cfg = cfg if cfg is not None else IkerRewardCfg()
    n = actions.shape[0]
    for name, value in (
        ("eef_pos", eef_pos),
        ("object_pos", object_pos),
        ("current_keypoints", current_keypoints),
        ("init_keypoints", init_keypoints),
        ("target_keypoints", target_keypoints),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name, value in (
        ("current_keypoints", current_keypoints),
        ("init_keypoints", init_keypoints),
        ("target_keypoints", target_keypoints),
    ):
        if tuple(value.shape) != (n, NUM_KEYPOINTS, 3):
            raise ValueError(f"{name} must have shape {(n, NUM_KEYPOINTS, 3)}, got {tuple(value.shape)}")

    d = torch.norm(object_pos - eef_pos, dim=-1).clamp(min=0.05, max=0.5)
    dist_reward = 1.0 - torch.tanh(5.0 * d)
    action_penalty = torch.sum(actions**2, dim=-1)
    under_penalty = (eef_pos[:, 2] < cfg.under_height).float()

    target_dir = target_keypoints - init_keypoints
    moved_dir = current_keypoints - init_keypoints
    target_norm = torch.norm(target_dir, dim=-1)
    cos_theta = torch.sum(target_dir * moved_dir, dim=-1) / (target_norm * torch.norm(moved_dir, dim=-1) + 0.1)
    dir_reward = cos_theta.clamp(0.0, 1.0).mean(dim=-1)

    d_target = torch.norm(current_keypoints - target_keypoints, dim=-1) / target_norm
    align_reward = (0.5 * (1.0 - d_target**2) + 0.5 * (1.0 - torch.tanh(2.0 * d_target))).clamp(0.0, 1.0).mean(dim=-1)

    success_count = success_count + (d_target.mean(dim=-1) <= cfg.success_tolerance).float()
    fall_penalty = (object_pos[:, 2] < cfg.fall_height).float()
    failure_count = failure_count + fall_penalty
    if force_penalty is None:
        force_penalty = torch.zeros_like(dist_reward)

    success = success_count > cfg.sustain_steps
    fail = failure_count > cfg.sustain_steps
    remaining = (cfg.max_episode_length - progress).to(dist_reward)
    bonus = torch.maximum(cfg.bonus_per_remaining_step * remaining, torch.full_like(remaining, cfg.bonus_floor))

    reward = (
        cfg.dist_scale * dist_reward
        + cfg.align_scale * align_reward
        + bonus * success.float()
        + cfg.dir_scale * dir_reward
        - fall_penalty
        - cfg.under_scale * under_penalty
        - cfg.fail_scale * fail.float()
        - cfg.action_scale * action_penalty
        - cfg.force_penalty_scale * force_penalty
    )
    terminated = (progress >= cfg.max_episode_length - 1) | success | fail
    return RewardOutput(reward=reward, terminated=terminated, success_count=success_count, failure_count=failure_count)
