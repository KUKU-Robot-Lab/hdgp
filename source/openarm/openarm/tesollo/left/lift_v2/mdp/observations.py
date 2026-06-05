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

"""Observation functions for force-based grasp in lift_left_v2."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


__all__ = [
    "contact_flags_multi",
    "normal_force_magnitude_multi",
    "slip_velocity",
]


def _contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Collect contact force magnitudes from one multi-body sensor.

    Returns:
        Force magnitudes (num_envs, num_bodies)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if force_magnitudes.dim() == 1:
        force_magnitudes = force_magnitudes.unsqueeze(-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    return force_magnitudes


def contact_flags_multi(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Get binary contact flags for a multi-body contact sensor.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor entity config
        threshold: Force threshold for contact detection (N)

    Returns:
        Binary contact flags (num_envs, num_bodies)
    """
    force_magnitudes = _contact_force_magnitudes(env, sensor_cfg)
    return (force_magnitudes > threshold).float()


def normal_force_magnitude_multi(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get force magnitudes for a multi-body contact sensor.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor entity config

    Returns:
        Force magnitudes (num_envs, num_bodies)
    """
    return _contact_force_magnitudes(env, sensor_cfg)


def slip_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Get slip velocity (relative tangential velocity) per contact link.

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_names for contact links
        object_cfg: Optional object config for relative velocity

    Returns:
        Slip velocity magnitudes (num_envs, num_bodies)
    """
    robot = env.scene[robot_cfg.name]

    # Get contact link velocities
    link_vel = robot.data.body_lin_vel_w

    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    if object_cfg is not None:
        # Compute relative velocity to object
        obj = env.scene[object_cfg.name]
        obj_vel = obj.data.root_lin_vel_w  # (num_envs, 3)
        obj_vel = obj_vel.unsqueeze(1)  # (num_envs, 1, 3)
        relative_vel = link_vel - obj_vel
    else:
        relative_vel = link_vel

    # Return velocity magnitude (slip proxy)
    slip_mag = torch.norm(relative_vel, dim=-1)

    return slip_mag
