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
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint positions relative to their default values."""
    robot = env.scene["robot"]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities relative to zero (scaled)."""
    robot = env.scene["robot"]
    return robot.data.joint_vel


def _joint_indices_by_names(env: ManagerBasedRLEnv, names: list[str]) -> list[int]:
    cache = getattr(env, "_joint_indices_by_names", None)
    if cache is None:
        cache = {}
        setattr(env, "_joint_indices_by_names", cache)
    key = tuple(names)
    if key in cache:
        return cache[key]
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(name) for name in names]
    cache[key] = indices
    return indices


def left_joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left joint positions relative to default values."""
    robot = env.scene["robot"]
    names = [
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_left_finger_joint1",
        "openarm_left_finger_joint2",
    ]
    idx = _joint_indices_by_names(env, names)
    return robot.data.joint_pos[:, idx] - robot.data.default_joint_pos[:, idx]


def right_joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right joint positions relative to default values."""
    robot = env.scene["robot"]
    names = [
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
        "openarm_right_finger_joint1",
        "openarm_right_finger_joint2",
    ]
    idx = _joint_indices_by_names(env, names)
    return robot.data.joint_pos[:, idx] - robot.data.default_joint_pos[:, idx]


def left_joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left joint velocities relative to zero (scaled)."""
    robot = env.scene["robot"]
    names = [
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_left_finger_joint1",
        "openarm_left_finger_joint2",
    ]
    idx = _joint_indices_by_names(env, names)
    return robot.data.joint_vel[:, idx]


def right_joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right joint velocities relative to zero (scaled)."""
    robot = env.scene["robot"]
    names = [
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
        "openarm_right_finger_joint1",
        "openarm_right_finger_joint2",
    ]
    idx = _joint_indices_by_names(env, names)
    return robot.data.joint_vel[:, idx]

def constant_value(env: ManagerBasedRLEnv, value: float, size: int = 1) -> torch.Tensor:
    """Return a constant observation of given size."""
    return torch.full((env.num_envs, size), float(value), device=env.device)

def body_pose(env: ManagerBasedRLEnv, body_name: str) -> torch.Tensor:
    """Body pose (position + quaternion) in world frame relative to env origin."""
    robot = env.scene["robot"]
    body_idx = robot.data.body_names.index(body_name)
    body_pos = robot.data.body_pos_w[:, body_idx] - env.scene.env_origins
    body_quat = robot.data.body_quat_w[:, body_idx]
    return torch.cat([body_pos, body_quat], dim=-1)

def root_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Root pose of an asset (position + quaternion) in env frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    root_quat = asset.data.root_quat_w
    return torch.cat([root_pos, root_quat], dim=-1)

def target_pos_in_tcp_frame(
    env: ManagerBasedRLEnv,
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Target position (with offset) in the TCP frame.

    The offset is in the target object's local frame.
    """
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.expand(target.data.root_quat_w.shape[0], -1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    target_pos_tcp, _ = subtract_frame_transforms(
        tcp_pos_w, tcp_quat_w, target_pos_w
    )
    return target_pos_tcp

def gripper_state(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    """Gripper joint positions (for sensing finger state)."""
    robot = env.scene["robot"]
    joint_indices = [robot.data.joint_names.index(name) for name in joint_names]
    return robot.data.joint_pos[:, joint_indices]


def tcp_to_cup_distance(
    env: ManagerBasedRLEnv,
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Scalar distance from TCP to target object center."""
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_pos_w

    dist = torch.norm(tcp_pos_w - target_pos_w, p=2, dim=-1, keepdim=True)
    return dist


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Object (cup) observations in env frame with relative vectors to both end effectors."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["cup"].data.root_quat_w
    object_lin_vel = env.scene["cup"].data.root_lin_vel_w
    object_ang_vel = env.scene["cup"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

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


def object2_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Object2 (cup2) observations in env frame with relative vectors to both end effectors."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["cup2"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["cup2"].data.root_quat_w
    object_lin_vel = env.scene["cup2"].data.root_lin_vel_w
    object_ang_vel = env.scene["cup2"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

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
