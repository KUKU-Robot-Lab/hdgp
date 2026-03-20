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

"""Observation functions for bi_pouring_v1."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from ..bi_pouring_env import BiPouringEnv


def arm_joint_pos(env: BiPouringEnv) -> torch.Tensor:
    """오른팔 관절 위치 (7D)."""
    return env.robot.data.joint_pos[:, env._right_arm_joint_ids]


def arm_joint_vel(env: BiPouringEnv) -> torch.Tensor:
    """오른팔 관절 속도 (7D)."""
    return env.robot.data.joint_vel[:, env._right_arm_joint_ids]


def cup_relative_pose(env: BiPouringEnv) -> torch.Tensor:
    """source cup 기준으로 target cup 상대 포즈 (7D: pos3 + quat4)."""
    right_cup_pos_w = env.right_source_cup.data.root_pos_w
    right_cup_quat_w = env.right_source_cup.data.root_quat_w
    left_cup_pos_w = env.left_target_cup.data.root_pos_w
    left_cup_quat_w = env.left_target_cup.data.root_quat_w
    rel_pos, rel_quat = subtract_frame_transforms(
        right_cup_pos_w, right_cup_quat_w, left_cup_pos_w, left_cup_quat_w
    )
    return torch.cat([rel_pos, rel_quat], dim=-1)


def pour_point_to_opening(env: BiPouringEnv) -> torch.Tensor:
    """source cup pour point → target cup opening 벡터 (3D, source cup local frame)."""
    right_cup_pos_w = env.right_source_cup.data.root_pos_w
    right_cup_quat_w = env.right_source_cup.data.root_quat_w
    left_cup_pos_w = env.left_target_cup.data.root_pos_w
    left_cup_quat_w = env.left_target_cup.data.root_quat_w

    target_opening_w = left_cup_pos_w + quat_apply(
        left_cup_quat_w,
        env._target_cup_opening_pos_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    target_opening_in_source, _ = subtract_frame_transforms(
        right_cup_pos_w, right_cup_quat_w, target_opening_w
    )
    return target_opening_in_source - env._source_cup_pour_point_pos_b.unsqueeze(0)


def source_cup_velocity_summary(env: BiPouringEnv) -> torch.Tensor:
    """source cup 선속도/각속도 크기 (2D)."""
    lin_speed = torch.norm(env.right_source_cup.data.root_lin_vel_w, dim=-1, keepdim=True)
    ang_speed = torch.norm(env.right_source_cup.data.root_ang_vel_w, dim=-1, keepdim=True)
    return torch.cat([lin_speed, ang_speed], dim=-1)


def tilt_alignment_summary(env: BiPouringEnv) -> torch.Tensor:
    """틸팅/정렬 요약 (3D): [source_up·world_up, pour_axis·target_dir, source_up·target_up]."""
    right_cup_pos_w = env.right_source_cup.data.root_pos_w
    right_cup_quat_w = env.right_source_cup.data.root_quat_w
    left_cup_pos_w = env.left_target_cup.data.root_pos_w
    left_cup_quat_w = env.left_target_cup.data.root_quat_w

    source_up_w = quat_apply(
        right_cup_quat_w,
        env._source_cup_up_axis_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    source_pour_axis_w = quat_apply(
        right_cup_quat_w,
        env._source_cup_pour_axis_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    target_up_w = quat_apply(
        left_cup_quat_w,
        env._target_cup_up_axis_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    source_pour_point_w = right_cup_pos_w + quat_apply(
        right_cup_quat_w,
        env._source_cup_pour_point_pos_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    target_opening_w = left_cup_pos_w + quat_apply(
        left_cup_quat_w,
        env._target_cup_opening_pos_b.unsqueeze(0).expand(env.num_envs, -1),
    )
    to_target_vec = target_opening_w - source_pour_point_w
    to_target_dir = to_target_vec / torch.clamp(
        torch.norm(to_target_vec, dim=-1, keepdim=True), min=1e-6
    )
    world_up = env._world_up_axis.expand(env.num_envs, -1)

    return torch.stack(
        [
            torch.sum(source_up_w * world_up, dim=-1),
            torch.sum(source_pour_axis_w * to_target_dir, dim=-1),
            torch.sum(source_up_w * target_up_w, dim=-1),
        ],
        dim=-1,
    )
