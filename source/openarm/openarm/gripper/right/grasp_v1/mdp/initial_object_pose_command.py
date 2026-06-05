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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import InitialObjectPoseCommandCfg


class InitialObjectPoseCommand(CommandTerm):
    """Command generator that targets the object's initial pose with a fixed offset."""

    cfg: "InitialObjectPoseCommandCfg"
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: "InitialObjectPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.target_object: RigidObject = env.scene[cfg.asset_cfg.name]

        # command in base frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        # cached initial object pose in world frame
        self._init_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._init_quat_w[:, 0] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # cache initial object pose at reset
        self._init_pos_w[env_ids] = self.target_object.data.root_pos_w[env_ids]
        self._init_quat_w[env_ids] = self.target_object.data.root_quat_w[env_ids]
        # refresh command for these envs
        self._update_command(env_ids)

    def _update_command(self, env_ids: Sequence[int] | None = None):
        offset = torch.tensor(self.cfg.goal_offset, device=self.device, dtype=torch.float32)
        cmd_pos_w = self._init_pos_w + offset
        cmd_quat_w = self._init_quat_w

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
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                prim_path = f"/Visuals/Command/{self.cfg.asset_cfg.name}_init_goal"
                cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
                cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
                self.goal_pose_visualizer = VisualizationMarkers(cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        cmd_pos_w, cmd_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        self.goal_pose_visualizer.visualize(cmd_pos_w, cmd_quat_w)
