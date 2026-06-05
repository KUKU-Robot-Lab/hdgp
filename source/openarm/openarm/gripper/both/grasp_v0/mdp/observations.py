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

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Object observations in env frame with relative vectors to both end effectors."""

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

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


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_quat = body_quat_w[:, eef_idx]
    return eef_quat


def object2_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Object observations in env frame with relative vectors to both end effectors."""

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

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
