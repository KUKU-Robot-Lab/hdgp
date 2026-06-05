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

from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def reset_root_state_from_command(
    env,
    env_ids: Sequence[int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset asset root pose to the current command pose (in robot base frame)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    cmd_pos_b = command[env_ids, :3]
    cmd_quat_b = command[env_ids, 3:7]

    cmd_pos_w, cmd_quat_w = combine_frame_transforms(
        robot.data.root_pos_w[env_ids],
        robot.data.root_quat_w[env_ids],
        cmd_pos_b,
        cmd_quat_b,
    )

    zeros = torch.zeros((len(env_ids), 6), device=env.device, dtype=cmd_pos_w.dtype)
    root_state = torch.cat([cmd_pos_w, cmd_quat_w, zeros], dim=-1)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
