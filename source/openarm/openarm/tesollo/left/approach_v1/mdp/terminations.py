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

"""Termination functions for the 5g_approach_left_v1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
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
