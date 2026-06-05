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

import os
from typing import Sequence, TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Global cache for terminal states
_TERMINAL_STATES_CACHE: dict = {}


def load_terminal_states(path: str, device: str) -> dict | None:
    """Load and cache terminal states from file."""
    if path not in _TERMINAL_STATES_CACHE:
        if os.path.exists(path):
            print(f"[events] Loading terminal states from: {path}")
            data = torch.load(path, map_location=device, weights_only=False)
            # Move all tensors to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            _TERMINAL_STATES_CACHE[path] = data
            print(f"[events] Loaded {len(data.get('joint_pos', []))} terminal states")
        else:
            print(f"[events] Warning: Terminal states file not found: {path}")
            _TERMINAL_STATES_CACHE[path] = None
    return _TERMINAL_STATES_CACHE.get(path)


def reset_robot_from_terminal_states(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    terminal_states_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joints from pre-saved terminal states.

    This function samples random terminal states from a saved file
    and uses them to initialize the robot's joint positions/velocities.
    """
    # Load terminal states (cached after first load)
    terminal_states = load_terminal_states(terminal_states_path, str(env.device))

    if terminal_states is None or len(terminal_states.get("joint_pos", [])) == 0:
        print(f"[events] No terminal states available, using default reset")
        return

    asset = env.scene[asset_cfg.name]
    num_states = len(terminal_states["joint_pos"])

    # Sample random indices for each environment being reset
    random_indices = torch.randint(0, num_states, (len(env_ids),), device=env.device)

    # Get sampled joint positions and velocities
    sampled_joint_pos = terminal_states["joint_pos"][random_indices]
    sampled_joint_vel = terminal_states["joint_vel"][random_indices]

    # Handle dimension mismatch if terminal states have different joint count
    current_joint_dim = asset.data.joint_pos.shape[1]
    saved_joint_dim = sampled_joint_pos.shape[1]

    if saved_joint_dim != current_joint_dim:
        print(f"[events] Warning: Joint dimension mismatch. "
              f"Saved: {saved_joint_dim}, Current: {current_joint_dim}. Using minimum.")
        min_dim = min(saved_joint_dim, current_joint_dim)
        sampled_joint_pos = sampled_joint_pos[:, :min_dim]
        sampled_joint_vel = sampled_joint_vel[:, :min_dim]

        # Pad with default values if needed
        if saved_joint_dim < current_joint_dim:
            default_pos = asset.data.default_joint_pos[env_ids]
            default_vel = asset.data.default_joint_vel[env_ids]
            sampled_joint_pos = torch.cat([
                sampled_joint_pos,
                default_pos[:, saved_joint_dim:]
            ], dim=1)
            sampled_joint_vel = torch.cat([
                sampled_joint_vel,
                default_vel[:, saved_joint_dim:]
            ], dim=1)

    # Set joint positions and velocities
    asset.write_joint_state_to_sim(sampled_joint_pos, sampled_joint_vel, env_ids=env_ids)


def reset_object_from_terminal_states(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    terminal_states_path: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_key: str = "object_pos",
    quat_key: str = "object_quat",
) -> None:
    """Reset object pose from pre-saved terminal states."""
    terminal_states = load_terminal_states(terminal_states_path, str(env.device))

    if terminal_states is None:
        return

    if pos_key not in terminal_states or len(terminal_states[pos_key]) == 0:
        return

    obj = env.scene[object_cfg.name]
    num_states = len(terminal_states[pos_key])

    # Sample random indices
    random_indices = torch.randint(0, num_states, (len(env_ids),), device=env.device)

    # Get sampled positions and quaternions
    sampled_pos = terminal_states[pos_key][random_indices]
    sampled_quat = terminal_states[quat_key][random_indices]

    # Convert local position to world position
    world_pos = sampled_pos + env.scene.env_origins[env_ids]

    # Set object pose
    obj.write_root_pose_to_sim(
        torch.cat([world_pos, sampled_quat], dim=-1),
        env_ids=env_ids
    )
    # Zero velocity
    obj.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device),
        env_ids=env_ids
    )


def reset_bimanual_objects_symmetric(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset left/right objects symmetrically in the robot root frame."""
    left_obj: RigidObject = env.scene[left_cfg.name]
    right_obj: RigidObject = env.scene[right_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    left_states = left_obj.data.default_root_state[env_ids].clone()
    right_states = right_obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)

    pos_offset_left = rand_samples[:, 0:3]
    pos_offset_right = pos_offset_left.clone()
    pos_offset_right[:, 1] *= -1.0

    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]

    left_pos_w = robot_pos_w + quat_apply(robot_quat_w, pos_offset_left)
    right_pos_w = robot_pos_w + quat_apply(robot_quat_w, pos_offset_right)

    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    left_quat_w = quat_mul(left_states[:, 3:7], orient_delta)
    right_quat_w = quat_mul(right_states[:, 3:7], orient_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)
    left_vel = left_states[:, 7:13] + rand_samples
    right_vel = right_states[:, 7:13] + rand_samples

    left_obj.write_root_pose_to_sim(torch.cat([left_pos_w, left_quat_w], dim=-1), env_ids=env_ids)
    right_obj.write_root_pose_to_sim(torch.cat([right_pos_w, right_quat_w], dim=-1), env_ids=env_ids)
    left_obj.write_root_velocity_to_sim(left_vel, env_ids=env_ids)
    right_obj.write_root_velocity_to_sim(right_vel, env_ids=env_ids)


def reset_bimanual_objects_symmetric_world(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> None:
    """Reset left/right objects symmetrically in world frame (mirror across Y)."""
    left_obj: RigidObject = env.scene[left_cfg.name]
    right_obj: RigidObject = env.scene[right_cfg.name]

    left_states = left_obj.data.default_root_state[env_ids].clone()
    right_states = right_obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)

    pos_left = rand_samples[:, 0:3]
    pos_right = pos_left.clone()
    pos_right[:, 1] *= -1.0

    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    left_quat = quat_mul(left_states[:, 3:7], orient_delta)
    right_quat = quat_mul(right_states[:, 3:7], orient_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)
    left_vel = left_states[:, 7:13] + rand_samples
    right_vel = right_states[:, 7:13] + rand_samples

    left_obj.write_root_pose_to_sim(torch.cat([pos_left, left_quat], dim=-1), env_ids=env_ids)
    right_obj.write_root_pose_to_sim(torch.cat([pos_right, right_quat], dim=-1), env_ids=env_ids)
    left_obj.write_root_velocity_to_sim(left_vel, env_ids=env_ids)
    right_obj.write_root_velocity_to_sim(right_vel, env_ids=env_ids)


def reset_root_state_uniform_robot_frame(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset asset root using a pose sampled in the robot root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    pos_offset = rand_samples[:, 0:3]
    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]
    positions = robot_pos_w + quat_apply(robot_quat_w, pos_offset)

    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
