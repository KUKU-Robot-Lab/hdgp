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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import WorldPoseCommandCfg


class WorldPoseCommand(CommandTerm):
    """Command generator that samples target poses in world frame."""

    cfg: WorldPoseCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: WorldPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command in base frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        # command in world frame (for sampling)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.pose_command_w[:, 3] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets in world frame
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # convert world command to base frame
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[env_ids, :3],
            self.pose_command_w[env_ids, 3:],
        )
        self.pose_command_b[env_ids, :3] = cmd_pos_b
        self.pose_command_b[env_ids, 3:] = cmd_quat_b

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
