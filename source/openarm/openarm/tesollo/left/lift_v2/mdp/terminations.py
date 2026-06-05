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
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cup_tipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    """Terminate if the cup tilts beyond the allowed angle from world-up."""
    cup = env.scene[asset_cfg.name]
    cup_quat_w = cup.data.root_quat_w
    z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(cup.data.root_pos_w)
    z_axis_world = quat_apply(cup_quat_w, z_axis_local)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(cup.data.root_pos_w)
    dot = torch.sum(z_axis_world * world_up, dim=-1)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    return dot < cos_thresh


def cup_xy_displacement_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_xy_displacement: float = 0.10,
) -> torch.Tensor:
    """Terminate if cup XY displacement from episode-start position exceeds threshold."""
    cup = env.scene[asset_cfg.name]
    current_xy = cup.data.root_pos_w[:, :2]

    cache_attr = "_cup_initial_xy_w_left_term_v2"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, current_xy.clone())
    initial_xy = getattr(env, cache_attr)

    ep_len = env.episode_length_buf.squeeze(-1) if env.episode_length_buf.dim() > 1 else env.episode_length_buf
    reset_mask = (ep_len <= 1)
    initial_xy[reset_mask] = current_xy[reset_mask]
    setattr(env, cache_attr, initial_xy)

    displacement_xy = torch.norm(current_xy - initial_xy, dim=1)
    return displacement_xy > max_xy_displacement
