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

if TYPE_CHECKING:
    from ..bi_pouring_env import BiPouringEnv


def right_arm_joint_pos(env: BiPouringEnv) -> torch.Tensor:
    return env.robot.data.joint_pos[:, env._right_arm_joint_ids]


def right_arm_joint_vel(env: BiPouringEnv) -> torch.Tensor:
    return env.robot.data.joint_vel[:, env._right_arm_joint_ids]


def right_palm_env_pos(env: BiPouringEnv) -> torch.Tensor:
    return env._right_palm_pos_w - env.scene.env_origins


def source_to_target_mouth_vec(env: BiPouringEnv) -> torch.Tensor:
    return env._target_opening_w - env._source_pour_point_w


def source_cup_quat(env: BiPouringEnv) -> torch.Tensor:
    return env.right_source_cup.data.root_quat_w


def target_cup_quat(env: BiPouringEnv) -> torch.Tensor:
    return env.left_target_cup.data.root_quat_w


def transport_summary(env: BiPouringEnv) -> torch.Tensor:
    return torch.stack(
        [
            env._mouth_distance,
            env._mouth_xy_distance,
            env._mouth_z_clearance,
            env._source_up_dot_world,
            env._directional_tilt_cos,
        ],
        dim=-1,
    )
