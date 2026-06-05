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

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cup_tipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    """Terminate if the cup tilts beyond a threshold angle from vertical.

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration for the cup.
        max_tilt_deg: Maximum allowed tilt angle in degrees.

    Returns:
        Boolean tensor indicating termination for each environment.
    """
    cup = env.scene[asset_cfg.name]
    cup_quat_w = cup.data.root_quat_w

    # Local Z axis
    z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(cup_quat_w.shape[0], 3)
    # Transform to world
    z_axis_world = quat_apply(cup_quat_w, z_axis_local)
    # World up
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(z_axis_world)
    # Dot product
    dot = torch.sum(z_axis_world * world_up, dim=-1)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    return dot < cos_thresh


def bead_spill(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    min_height_offset: float = -0.04,
    xy_radius: float = 0.08,
) -> torch.Tensor:
    """Terminate if bead falls below target cup or too far from it.

    Args:
        env: The RL environment.
        bead_name: Name of the bead asset.
        target_name: Name of the target cup asset.
        min_height_offset: Minimum height offset relative to target cup.
        xy_radius: Maximum XY distance from target cup center.

    Returns:
        Boolean tensor indicating termination for each environment.
    """
    bead = env.scene[bead_name]
    target = env.scene[target_name]

    bead_pos = bead.data.root_pos_w
    target_pos = target.data.root_pos_w

    # Check if bead is below the target cup
    height_diff = bead_pos[:, 2] - target_pos[:, 2]
    below_target = height_diff < min_height_offset

    # Check if bead is too far from target in XY
    xy_dist = torch.norm(bead_pos[:, :2] - target_pos[:, :2], dim=-1)
    too_far = xy_dist > xy_radius

    # Terminate if bead is below AND too far (spilled)
    return below_target & too_far


def bead_on_ground(
    env: ManagerBasedRLEnv,
    bead_name: str,
    ground_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if bead falls to the ground.

    Args:
        env: The RL environment.
        bead_name: Name of the bead asset.
        ground_height: Height threshold for ground.

    Returns:
        Boolean tensor indicating termination for each environment.
    """
    bead = env.scene[bead_name]
    bead_pos = bead.data.root_pos_w
    return bead_pos[:, 2] < ground_height
