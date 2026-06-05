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

"""Termination terms specific to the bimanual approach task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_reached_and_stopped(
    env: ManagerBasedRLEnv,
    left_command_name: str = "left_ee_pose",
    right_command_name: str = "right_ee_pose",
    pos_threshold: float = 0.01,
    ori_threshold: float = 0.05,
    joint_vel_threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
    right_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
) -> torch.Tensor:
    """Terminate when both end-effectors are at the goal pose and the robot is nearly stationary."""
    robot: Articulation = env.scene[robot_cfg.name]

    # resolve joint ids if needed
    joint_ids = robot_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)

    # desired poses for left and right in world frame
    left_cmd = env.command_manager.get_command(left_command_name)
    right_cmd = env.command_manager.get_command(right_command_name)

    left_des_pos_b = left_cmd[:, :3]
    right_des_pos_b = right_cmd[:, :3]
    left_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_des_pos_b)
    right_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_des_pos_b)

    left_des_quat_w = quat_mul(robot.data.root_quat_w, left_cmd[:, 3:7])
    right_des_quat_w = quat_mul(robot.data.root_quat_w, right_cmd[:, 3:7])

    left_body_id = left_ee_cfg.body_ids[0]
    right_body_id = right_ee_cfg.body_ids[0]
    left_pos_err = torch.norm(robot.data.body_pos_w[:, left_body_id] - left_des_pos_w, dim=1)
    right_pos_err = torch.norm(robot.data.body_pos_w[:, right_body_id] - right_des_pos_w, dim=1)

    left_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, left_body_id], left_des_quat_w)
    right_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, right_body_id], right_des_quat_w)

    pos_ok = torch.logical_and(left_pos_err < pos_threshold, right_pos_err < pos_threshold)
    ori_ok = torch.logical_and(left_ori_err < ori_threshold, right_ori_err < ori_threshold)

    joint_vel = robot.data.joint_vel[:, joint_ids]
    vel_ok = torch.max(torch.abs(joint_vel), dim=1)[0] < joint_vel_threshold

    return pos_ok & ori_ok & vel_ok
