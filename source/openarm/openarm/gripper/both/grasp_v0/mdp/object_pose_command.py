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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import ObjectPoseCommandCfg


class ObjectPoseCommand(CommandTerm):
    """Command generator that tracks an object's pose with configurable offsets."""

    cfg: ObjectPoseCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: ObjectPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.target_object: RigidObject = env.scene[cfg.asset_cfg.name]

        # command in base frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Object pose is state-driven, so just refresh on resample.
        self._update_command(env_ids)

    def _update_command(self, env_ids: Sequence[int] | None = None):
        # compute command for all envs, then slice
        obj_pos_w = self.target_object.data.root_pos_w
        obj_quat_w = self.target_object.data.root_quat_w

        # Determine if the object is lifted
        is_lifted = obj_pos_w[:, 2] > self.cfg.lift_threshold_z

        # Create offset tensors
        pre_grasp_offset = torch.tensor(self.cfg.pre_grasp_offset, device=self.device, dtype=torch.float32)
        hold_offset = torch.tensor(self.cfg.hold_offset, device=self.device, dtype=torch.float32)
        selected_offset = torch.where(is_lifted.unsqueeze(1), hold_offset, pre_grasp_offset)

        # apply the selected offset to the object's position
        cmd_pos_w = obj_pos_w + selected_offset
        cmd_quat_w = obj_quat_w

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            cmd_pos_w,
            cmd_quat_w,
        )
        if env_ids is None:
            self.pose_command_b[:, :3] = cmd_pos_b
            self.pose_command_b[:, 3:] = cmd_quat_b
        else:
            self.pose_command_b[env_ids, :3] = cmd_pos_b[env_ids]
            self.pose_command_b[env_ids, 3:] = cmd_quat_b[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
