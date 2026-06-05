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

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .observations import get_eef_pos, get_selected_object_pose, get_selected_object_lin_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward for lifting the selected object above a minimal height."""
    object_pos_w, _ = get_selected_object_pose(env, object_cfg)
    return torch.where(object_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Distance between the selected object and the specified end-effector link."""
    object_pos_w, _ = get_selected_object_pose(env, object_cfg)
    object_pos = object_pos_w - env.scene.env_origins
    eef_pos = get_eef_pos(env, eef_link_name)
    return torch.norm(object_pos - eef_pos, dim=1)


def object_eef_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward the end-effector being close to the selected object using tanh-kernel."""
    distance = object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def object_contact_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_eef_link_name: str,
    right_eef_link_name: str,
    max_dist: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_grasp"),
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    """Reward contact when the object is near the end effectors."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    if body_name_pattern is not None:
        key = f"_contact_body_ids_{body_name_pattern}"
        if not hasattr(env, key):
            body_ids = [
                index
                for index, name in enumerate(contact_sensor.body_names)
                if re.fullmatch(body_name_pattern, name)
            ]
            if not body_ids:
                body_ids = list(range(contact_sensor.num_bodies))
            setattr(env, key, torch.tensor(body_ids, device=net_forces.device, dtype=torch.long))
        body_ids = getattr(env, key)
        net_forces = net_forces[:, :, body_ids, :]
    contact_mag = torch.norm(net_forces, dim=-1)
    contact_any = torch.max(contact_mag, dim=1)[0].max(dim=1)[0] > threshold

    object_pos_w, _ = get_selected_object_pose(env, object_cfg)
    object_pos = object_pos_w - env.scene.env_origins
    left_eef_pos = get_eef_pos(env, left_eef_link_name)
    right_eef_pos = get_eef_pos(env, right_eef_link_name)
    left_dist = torch.norm(object_pos - left_eef_pos, dim=1)
    right_dist = torch.norm(object_pos - right_eef_pos, dim=1)
    near = torch.minimum(left_dist, right_dist) < max_dist

    return (contact_any & near).to(torch.float)


def object_grasp_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    left_eef_link_name: str,
    right_eef_link_name: str,
    max_dist: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_grasp"),
    body_name_pattern: str | None = None,
) -> torch.Tensor:
    """Reward successful grasp: contact near the object and object lifted."""
    contact = object_contact_reward(
        env,
        threshold=threshold,
        left_eef_link_name=left_eef_link_name,
        right_eef_link_name=right_eef_link_name,
        max_dist=max_dist,
        object_cfg=object_cfg,
        sensor_cfg=sensor_cfg,
        body_name_pattern=body_name_pattern,
    )
    object_pos_w, _ = get_selected_object_pose(env, object_cfg)
    lifted = object_pos_w[:, 2] > minimal_height
    return (contact.bool() & lifted).to(torch.float)


def object_stability_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Reward stable grasps by favoring low object velocity when lifted."""
    object_pos_w, _ = get_selected_object_pose(env, object_cfg)
    lifted = object_pos_w[:, 2] > minimal_height
    lin_vel = get_selected_object_lin_vel(env, object_cfg)
    speed = torch.norm(lin_vel, dim=1)
    reward = torch.exp(-speed / std)
    return reward * lifted.to(torch.float)


# =============================================================================
# Bimanual Grasp Rewards (for two-object manipulation)
# =============================================================================

def bimanual_lift_reward(
    env: ManagerBasedRLEnv,
    lift_threshold: float = 0.1,
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    """Reward for lifting both objects above threshold height.

    Success = both cups lifted above initial_height + lift_threshold
    """
    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]

    left_height = left_obj.data.root_pos_w[:, 2]
    right_height = right_obj.data.root_pos_w[:, 2]

    # Get initial heights from default state
    left_init_height = left_obj.data.default_root_state[:, 2].mean()
    right_init_height = right_obj.data.default_root_state[:, 2].mean()

    left_lifted = left_height > (left_init_height + lift_threshold)
    right_lifted = right_height > (right_init_height + lift_threshold)

    # Both must be lifted for reward
    both_lifted = left_lifted & right_lifted

    return both_lifted.float()


def continuous_hold_reward(
    env: ManagerBasedRLEnv,
    lift_threshold: float = 0.1,
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    """Reward for continuously holding both objects above threshold.

    This encourages the policy to maintain grasp after lifting.
    """
    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]

    left_height = left_obj.data.root_pos_w[:, 2]
    right_height = right_obj.data.root_pos_w[:, 2]

    left_init_height = left_obj.data.default_root_state[:, 2].mean()
    right_init_height = right_obj.data.default_root_state[:, 2].mean()

    left_held = left_height > (left_init_height + lift_threshold)
    right_held = right_height > (right_init_height + lift_threshold)

    # Reward for each object being held
    reward = left_held.float() * 0.5 + right_held.float() * 0.5

    return reward


def drop_penalty(
    env: ManagerBasedRLEnv,
    drop_threshold: float = 0.05,
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    """Penalty for dropping objects after having lifted them.

    Tracks max height reached and penalizes if current height drops significantly.
    """
    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]

    left_height = left_obj.data.root_pos_w[:, 2]
    right_height = right_obj.data.root_pos_w[:, 2]

    # Initialize max height tracking
    if not hasattr(env, "_max_left_height"):
        env._max_left_height = left_height.clone()
        env._max_right_height = right_height.clone()

    # Update max heights
    env._max_left_height = torch.maximum(env._max_left_height, left_height)
    env._max_right_height = torch.maximum(env._max_right_height, right_height)

    # Check for drops
    left_drop = env._max_left_height - left_height
    right_drop = env._max_right_height - right_height

    left_dropped = left_drop > drop_threshold
    right_dropped = right_drop > drop_threshold

    # Penalty if either object dropped
    dropped = left_dropped | right_dropped

    return dropped.float()


def reset_height_tracking(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> None:
    """Reset height tracking for specified environments.

    Call this in the reset event.
    """
    if not hasattr(env, "_max_left_height"):
        return

    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]

    env._max_left_height[env_ids] = left_obj.data.root_pos_w[env_ids, 2]
    env._max_right_height[env_ids] = right_obj.data.root_pos_w[env_ids, 2]


def bimanual_grasp_success(
    env: ManagerBasedRLEnv,
    lift_threshold: float = 0.1,
    contact_threshold: float = 1.0,
    left_eef_link_name: str = "openarm_left_eef_link",
    right_eef_link_name: str = "openarm_right_eef_link",
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_grasp"),
) -> torch.Tensor:
    """Reward for successful bimanual grasp.

    Success criteria:
    1. Both objects lifted above threshold
    2. Both hands in contact with their respective objects
    """
    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]

    left_height = left_obj.data.root_pos_w[:, 2]
    right_height = right_obj.data.root_pos_w[:, 2]

    left_init_height = left_obj.data.default_root_state[:, 2].mean()
    right_init_height = right_obj.data.default_root_state[:, 2].mean()

    left_lifted = left_height > (left_init_height + lift_threshold)
    right_lifted = right_height > (right_init_height + lift_threshold)

    both_lifted = left_lifted & right_lifted

    return both_lifted.float()


def bimanual_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    left_eef_link_name: str = "openarm_left_eef_link",
    right_eef_link_name: str = "openarm_right_eef_link",
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> torch.Tensor:
    """Reward for both end-effectors approaching their target objects."""
    left_obj = env.scene[left_object_cfg.name]
    right_obj = env.scene[right_object_cfg.name]
    robot = env.scene["robot"]

    # Get object positions (local frame)
    left_obj_pos = left_obj.data.root_pos_w - env.scene.env_origins
    right_obj_pos = right_obj.data.root_pos_w - env.scene.env_origins

    # Get EEF positions
    left_eef_idx = robot.data.body_names.index(left_eef_link_name)
    right_eef_idx = robot.data.body_names.index(right_eef_link_name)

    left_eef_pos = robot.data.body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = robot.data.body_pos_w[:, right_eef_idx] - env.scene.env_origins

    # Calculate distances
    left_dist = torch.norm(left_obj_pos - left_eef_pos, dim=1)
    right_dist = torch.norm(right_obj_pos - right_eef_pos, dim=1)

    # Tanh reward
    left_reward = 1 - torch.tanh(left_dist / std)
    right_reward = 1 - torch.tanh(right_dist / std)

    return (left_reward + right_reward) / 2
