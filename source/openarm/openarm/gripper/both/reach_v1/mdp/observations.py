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

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    command_name: str | None = None,
) -> torch.Tensor:
    """Object observations in env frame with relative vectors to both end effectors."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_quat_b = command[:, 3:7]
        robot = env.scene["robot"]
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
        )
        object_pos = des_pos_w - env.scene.env_origins
        object_quat = des_quat_w
        object_lin_vel = torch.zeros_like(object_pos)
        object_ang_vel = torch.zeros_like(object_pos)
    else:
        object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
        object_quat = env.scene["object"].data.root_quat_w
        object_lin_vel = env.scene["object"].data.root_lin_vel_w
        object_ang_vel = env.scene["object"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    # mask cross-hand info to avoid symmetric policies
    right_eef_to_object = torch.zeros_like(right_eef_to_object)
    # add a left-hand token without changing dimension
    right_eef_to_object[:, 0] = 1.0

    return torch.cat(
        (
            object_pos,
            object_quat,
            object_lin_vel,
            object_ang_vel,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object2_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    command_name: str | None = None,
) -> torch.Tensor:
    """Second object observation for compatibility with grasp_2g."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_quat_b = command[:, 3:7]
        robot = env.scene["robot"]
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
        )
        object_pos = des_pos_w - env.scene.env_origins
        object_quat = des_quat_w
        object_lin_vel = torch.zeros_like(object_pos)
        object_ang_vel = torch.zeros_like(object_pos)
    else:
        object_pos = env.scene["object2"].data.root_pos_w - env.scene.env_origins
        object_quat = env.scene["object2"].data.root_quat_w
        object_lin_vel = env.scene["object2"].data.root_lin_vel_w
        object_ang_vel = env.scene["object2"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    # mask cross-hand info to avoid symmetric policies
    left_eef_to_object = torch.zeros_like(left_eef_to_object)
    # add a right-hand token without changing dimension
    left_eef_to_object[:, 0] = -1.0

    return torch.cat(
        (
            object_pos,
            object_quat,
            object_lin_vel,
            object_ang_vel,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )
