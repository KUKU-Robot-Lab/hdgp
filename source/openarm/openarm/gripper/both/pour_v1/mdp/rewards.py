# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for bimanual pouring task.

Based on leapiper_r_sens_env.py reward design:
- Phase-based rewards (approach vs pour) using rho indicator
- Distance rewards (XY, Z separately) with tanh-based penalties
- Orientation rewards (cup upright, cup tilted towards target)
- Bead transfer rewards

Key insight from leapiper:
- rho = 1 when cups are aligned (xy < threshold, z in range)
- Different reward weights for approach (rho=0) vs pour (rho=1) phases
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Helper: Phase Indicator (rho)
# =============================================================================

def _compute_phase_rho(
    env: ManagerBasedRLEnv,
    source_cup_name: str = "cup",
    target_cup_name: str = "cup2",
    xy_threshold: float = 0.055,
    z_min: float = 0.0,
    z_max: float = 0.17,
) -> torch.Tensor:
    """Compute phase indicator: 1 if cups aligned for pouring, 0 otherwise."""
    source_cup = env.scene[source_cup_name]
    target_cup = env.scene[target_cup_name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_xy = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)
    d_z = source_pos[:, 2] - target_pos[:, 2]

    rho = torch.where(
        (d_xy <= xy_threshold) & (d_z > z_min) & (d_z < z_max),
        torch.ones_like(d_xy),
        torch.zeros_like(d_xy),
    )
    return rho


# =============================================================================
# Distance Rewards (XY)
# =============================================================================

def cups_xy_distance_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    target_xy_min: float = 0.0,
    target_xy_max: float = 0.03,
) -> torch.Tensor:
    """Reward for XY alignment between cups using tanh-based shaping.

    Based on leapiper: in-range gets max reward, outside gets tanh+linear penalty.
    """
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_xy = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)

    in_range = (d_xy >= target_xy_min) & (d_xy <= target_xy_max)

    # Penalties for being outside range
    distance_close = torch.relu(-d_xy + target_xy_min) * 25
    distance_far = torch.relu(d_xy - target_xy_max) * 25

    penalty_tan_close = torch.tanh(distance_close * 8.5) * 1.3
    penalty_linear_close = distance_close * 4.7
    penalty_tan_far = torch.tanh(distance_far * 8.5) * 1.3
    penalty_linear_far = distance_far * 4.7

    reward = torch.where(
        in_range,
        torch.ones_like(d_xy),
        torch.ones_like(d_xy) - penalty_tan_close - penalty_linear_close - penalty_tan_far - penalty_linear_far
    )
    return reward.clamp(min=0.0)


def cups_xy_pour_distance_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    target_xy: float = 0.01,
) -> torch.Tensor:
    """Tighter XY alignment reward for pour phase."""
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_xy = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)

    in_range = d_xy <= target_xy

    distance_far = torch.relu(d_xy - target_xy) * 10
    penalty_tan_far = torch.tanh(distance_far * 20.5) * 0.3
    penalty_linear_far = distance_far * 0.7

    reward = torch.where(
        in_range,
        torch.ones_like(d_xy),
        torch.ones_like(d_xy) - penalty_tan_far - penalty_linear_far
    )
    return reward.clamp(min=0.0)


# =============================================================================
# Distance Rewards (Z)
# =============================================================================

def cups_z_distance_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    target_z_min: float = 0.14,
    target_z_max: float = 0.17,
) -> torch.Tensor:
    """Reward for Z height difference (source above target) for approach phase."""
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_z = source_pos[:, 2] - target_pos[:, 2]

    in_range = (d_z >= target_z_min) & (d_z <= target_z_max)

    distance_close = torch.relu(-d_z + target_z_min) * 25
    distance_far = torch.relu(d_z - target_z_max) * 25

    penalty_tan_close = torch.tanh(distance_close * 20.5) * 0.3
    penalty_linear_close = distance_close * 2.7
    penalty_tan_far = torch.tanh(distance_far * 20.5) * 0.3
    penalty_linear_far = distance_far * 2.7

    reward = torch.where(
        in_range,
        torch.ones_like(d_z),
        torch.ones_like(d_z) - penalty_tan_close - penalty_linear_close - penalty_tan_far - penalty_linear_far
    )
    return reward.clamp(min=0.0)


def cups_z_pour_distance_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    target_z_min: float = 0.07,
    target_z_max: float = 0.09,
) -> torch.Tensor:
    """Tighter Z distance reward for pour phase (closer height)."""
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_z = source_pos[:, 2] - target_pos[:, 2]

    in_range = (d_z >= target_z_min) & (d_z <= target_z_max)

    distance_close = torch.relu(-d_z + target_z_min) * 45
    distance_far = torch.relu(d_z - target_z_max) * 45

    penalty_tan_close = torch.tanh(distance_close * 20.5) * 0.3
    penalty_linear_close = distance_close * 0.7
    penalty_tan_far = torch.tanh(distance_far * 20.5) * 0.3
    penalty_linear_far = distance_far * 2.7

    reward = torch.where(
        in_range,
        torch.ones_like(d_z),
        torch.ones_like(d_z) - penalty_tan_close - penalty_linear_close - penalty_tan_far - penalty_linear_far
    )
    return reward.clamp(min=0.0)


# =============================================================================
# Orientation Rewards
# =============================================================================

def cup_upright_reward(
    env: ManagerBasedRLEnv,
    cup_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for keeping cup upright (drop_reward in leapiper).

    Uses tanh-based shaping: high reward when cup z-axis aligned with world up.
    """
    cup = env.scene[cup_cfg.name]
    cup_quat = cup.data.root_quat_w

    # Local Z axis
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(cup_quat.shape[0], 3)
    z_world = quat_apply(cup_quat, z_local)

    # Dot product with world up (just z component)
    dot = z_world[:, 2].clamp(-1.0, 1.0)

    # Angle from vertical
    theta_rad = torch.acos(dot)

    # Reward: high when upright, decreases with tilt
    # Based on leapiper: 0.9 - tanh((theta - 0.15) * 3.5)
    reward = 0.9 - torch.tanh((theta_rad - 0.15) * 3.5)

    return reward


def cup_pour_orientation_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for tilting source cup towards target (pour_reward in leapiper).

    Cup's Z-axis should align with direction vector from source to target.
    """
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w
    source_quat = source_cup.data.root_quat_w

    # Direction vector from source to target (normalized)
    target_vector = target_pos - source_pos
    target_vector_norm = torch.nn.functional.normalize(target_vector, dim=-1)

    # Source cup's Z-axis in world frame
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(source_quat.shape[0], 3)
    cup_z_world = quat_apply(source_quat, z_local)
    cup_z_norm = torch.nn.functional.normalize(cup_z_world, dim=-1)

    # Dot product
    dot = torch.sum(cup_z_norm * target_vector_norm, dim=-1).clamp(-1.0, 1.0)

    # Angle
    theta_rad = torch.acos(dot)

    # Reward components (from leapiper)
    pour_tanh_reward = (1.0 - torch.tanh(theta_rad * 0.65)) * 0.4
    pour_cos_reward = (1.0 + dot) / 2.0 * 0.6

    return pour_tanh_reward + pour_cos_reward


# =============================================================================
# Combined Phase-based Rewards
# =============================================================================

def phase_distance_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    approach_xy_min: float = 0.0,
    approach_xy_max: float = 0.03,
    approach_z_min: float = 0.14,
    approach_z_max: float = 0.17,
    pour_z_min: float = 0.07,
    pour_z_max: float = 0.09,
) -> torch.Tensor:
    """Combined distance reward that changes based on phase.

    Approach (rho=0): Focus on xy alignment and z height
    Pour (rho=1): Tighter z distance for pouring
    """
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    source_pos = source_cup.data.root_pos_w
    target_pos = target_cup.data.root_pos_w

    d_xy = torch.norm(source_pos[:, :2] - target_pos[:, :2], dim=-1)
    d_z = source_pos[:, 2] - target_pos[:, 2]

    # Phase indicator
    rho = _compute_phase_rho(env, source_cup_cfg.name, target_cup_cfg.name)

    # XY reward (same for both phases but tighter for pour)
    xy_in_range = (d_xy >= approach_xy_min) & (d_xy <= approach_xy_max)
    xy_distance_far = torch.relu(d_xy - approach_xy_max) * 25
    xy_penalty = torch.tanh(xy_distance_far * 8.5) * 1.3 + xy_distance_far * 4.7
    xy_reward = torch.where(xy_in_range, torch.ones_like(d_xy), (1.0 - xy_penalty).clamp(min=0.0))

    # Z reward for approach
    z_approach_in_range = (d_z >= approach_z_min) & (d_z <= approach_z_max)
    z_approach_close = torch.relu(-d_z + approach_z_min) * 25
    z_approach_far = torch.relu(d_z - approach_z_max) * 25
    z_approach_penalty = (torch.tanh(z_approach_close * 20.5) * 0.3 + z_approach_close * 2.7 +
                          torch.tanh(z_approach_far * 20.5) * 0.3 + z_approach_far * 2.7)
    z_approach_reward = torch.where(z_approach_in_range, torch.ones_like(d_z), (1.0 - z_approach_penalty).clamp(min=0.0))

    # Z reward for pour (tighter)
    z_pour_in_range = (d_z >= pour_z_min) & (d_z <= pour_z_max)
    z_pour_close = torch.relu(-d_z + pour_z_min) * 45
    z_pour_far = torch.relu(d_z - pour_z_max) * 45
    z_pour_penalty = (torch.tanh(z_pour_close * 20.5) * 0.3 + z_pour_close * 0.7 +
                      torch.tanh(z_pour_far * 20.5) * 0.3 + z_pour_far * 2.7)
    z_pour_reward = torch.where(z_pour_in_range, torch.ones_like(d_z), (1.0 - z_pour_penalty).clamp(min=0.0))

    # Combine based on phase
    z_reward = z_approach_reward * (1 - rho) + z_pour_reward * rho

    return xy_reward + z_reward


def phase_orientation_reward(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Combined orientation reward that changes based on phase.

    Approach (rho=0): Keep source cup upright (drop_reward)
    Pour (rho=1): Tilt source cup towards target (pour_reward)
    """
    rho = _compute_phase_rho(env, source_cup_cfg.name, target_cup_cfg.name)

    # Upright reward (for approach)
    upright_reward = cup_upright_reward(env, source_cup_cfg)

    # Pour orientation reward
    pour_reward = cup_pour_orientation_reward(env, source_cup_cfg, target_cup_cfg)

    # Combine based on phase
    return upright_reward * (1 - rho) + pour_reward * rho


# =============================================================================
# Bead Transfer Rewards
# =============================================================================

def bead_height_reward(
    env: ManagerBasedRLEnv,
    bead_cfg: SceneEntityCfg,
    min_height: float = 0.05,
) -> torch.Tensor:
    """Reward for keeping bead above minimum height."""
    bead = env.scene[bead_cfg.name]
    bead_pos = bead.data.root_pos_w
    return (bead_pos[:, 2] > min_height).float()


def bead_in_target_cup_reward(
    env: ManagerBasedRLEnv,
    bead_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    xy_radius: float = 0.05,
    z_offset_min: float = -0.02,
    z_offset_max: float = 0.1,
) -> torch.Tensor:
    """Reward for bead being inside the target cup."""
    bead = env.scene[bead_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    bead_pos = bead.data.root_pos_w
    cup_pos = target_cup.data.root_pos_w

    xy_dist = torch.norm(bead_pos[:, :2] - cup_pos[:, :2], dim=-1)
    z_offset = bead_pos[:, 2] - cup_pos[:, 2]

    in_xy = xy_dist < xy_radius
    in_z = (z_offset > z_offset_min) & (z_offset < z_offset_max)
    in_cup = in_xy & in_z

    return in_cup.float()


def bead_approaching_target_reward(
    env: ManagerBasedRLEnv,
    bead_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Smooth reward for bead getting closer to target cup."""
    bead = env.scene[bead_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]

    bead_pos = bead.data.root_pos_w
    cup_pos = target_cup.data.root_pos_w

    distance = torch.norm(bead_pos - cup_pos, dim=-1)
    return torch.exp(-distance**2 / std**2)


# =============================================================================
# Phase Bonus
# =============================================================================

def phase_reached_bonus(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg,
    target_cup_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bonus reward when pour phase is reached (rho=1)."""
    rho = _compute_phase_rho(env, source_cup_cfg.name, target_cup_cfg.name)
    return rho
