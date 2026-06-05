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
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def load_unidex_pc_feat(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pc_feat_paths: list[str] | tuple[str, ...],
    object_names: list[str] | None = None,
):
    """Load pc_feat vectors for UniDexGrasp assets and cache on the env."""

    if hasattr(env, "unidex_pc_feat"):
        return

    feats = []
    for feat_path in pc_feat_paths:
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"pc_feat file not found: {feat_path}")
        feat = np.load(feat_path).astype(np.float32).reshape(-1)
        feats.append(torch.tensor(feat, device=env.device))

    env.unidex_pc_feat = torch.stack(feats, dim=0)
    env.unidex_pc_feat_dim = env.unidex_pc_feat.shape[1]
    env.unidex_object_names = object_names or [os.path.basename(path) for path in pc_feat_paths]
    env.unidex_object_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


def load_unidex_grasp_prior(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    posedata_path: str,
    object_code: str,
    scale_values: list[float] | tuple[float, ...],
):
    """Load UniDex grasp priors for object pose initialization."""
    if hasattr(env, "unidex_pose_prior"):
        return

    if not posedata_path or not os.path.exists(posedata_path):
        print(f"[WARN] UniDex grasp prior file not found: {posedata_path}")
        env.unidex_pose_prior = None
        return

    data = np.load(posedata_path, allow_pickle=True).item()
    if object_code not in data:
        print(f"[WARN] UniDex grasp prior missing object code: {object_code}")
        env.unidex_pose_prior = None
        return

    obj_data = data[object_code]
    scale_key_map = {}
    for key in obj_data.keys():
        try:
            scale_key_map[float(key)] = key
        except (TypeError, ValueError):
            continue

    pose_prior = []
    for scale_val in scale_values:
        key = scale_key_map.get(scale_val)
        if key is None:
            pose_prior.append(None)
            continue
        scale_entry = obj_data.get(key, {})
        euler_xy = scale_entry.get("object_euler_xy", None)
        init_z = scale_entry.get("object_init_z", None)
        if euler_xy is None or init_z is None:
            pose_prior.append(None)
            continue
        euler_xy = torch.tensor(np.asarray(euler_xy, dtype=np.float32), device=env.device)
        init_z = torch.tensor(np.asarray(init_z, dtype=np.float32).reshape(-1), device=env.device)
        pose_prior.append({"euler_xy": euler_xy, "init_z": init_z})

    env.unidex_pose_prior = pose_prior


def reset_unidex_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    parking_pos: tuple[float, float, float] = (0.0, 0.0, -2.0),
):
    """Reset a selected object and park the rest for each environment."""

    objects: RigidObjectCollection = env.scene[object_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=objects.device)

    num_envs = len(env_ids)
    num_objects = objects.num_objects

    if not hasattr(env, "unidex_object_ids"):
        env.unidex_object_ids = torch.zeros(env.scene.num_envs, device=objects.device, dtype=torch.long)

    # sample object indices per env
    obj_ids = torch.randint(0, num_objects, (num_envs,), device=objects.device)
    env.unidex_object_ids[env_ids] = obj_ids

    # start from default state
    object_state = objects.data.default_object_state[env_ids].clone()

    # park all objects away from the workspace
    parking_pos_tensor = torch.tensor(parking_pos, device=objects.device)
    object_state[..., 0:3] = env.scene.env_origins[env_ids].unsqueeze(1) + parking_pos_tensor
    object_state[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=objects.device)
    object_state[..., 7:13] = 0.0

    # sample a pose for the selected objects
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=objects.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=objects.device)

    if hasattr(env, "unidex_pose_prior") and env.unidex_pose_prior:
        for i in range(num_envs):
            obj_id = int(obj_ids[i].item())
            prior = env.unidex_pose_prior[obj_id] if obj_id < len(env.unidex_pose_prior) else None
            if prior is None:
                continue
            num_samples = prior["euler_xy"].shape[0]
            if num_samples == 0:
                continue
            sample_idx = torch.randint(0, num_samples, (1,), device=objects.device)
            rand_samples[i, 2] = prior["init_z"][sample_idx]
            rand_samples[i, 3] = prior["euler_xy"][sample_idx, 0]
            rand_samples[i, 4] = prior["euler_xy"][sample_idx, 1]

    default_selected = objects.data.default_object_state[env_ids, obj_ids].clone()
    positions = default_selected[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientation_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_selected[:, 3:7], orientation_delta)

    env_idx = torch.arange(num_envs, device=objects.device)
    object_state[env_idx, obj_ids, 0:3] = positions
    object_state[env_idx, obj_ids, 3:7] = orientations
    object_state[env_idx, obj_ids, 7:13] = default_selected[:, 7:13]

    objects.write_object_state_to_sim(object_state, env_ids=env_ids)


# =============================================================================
# Bimanual Symmetric Reset Functions (from grasp_2g)
# =============================================================================

def reset_bimanual_objects_symmetric(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset left/right objects symmetrically in the robot root frame.

    This function ensures both objects are placed symmetrically across the Y-axis,
    which helps the policy learn symmetric bimanual manipulation.
    """
    left_obj: RigidObject = env.scene[left_cfg.name]
    right_obj: RigidObject = env.scene[right_cfg.name]
    robot = env.scene[robot_cfg.name]

    left_states = left_obj.data.default_root_state[env_ids].clone()
    right_states = right_obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)

    # Left object position offset
    pos_offset_left = rand_samples[:, 0:3]
    # Right object: mirror across Y-axis
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


# =============================================================================
# Roll-out Reset Functions (for curriculum learning)
# =============================================================================

def load_reach_terminal_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    terminal_states_path: str,
) -> None:
    """Load pre-saved reach policy terminal states for curriculum learning.

    This should be called in 'startup' mode to cache the terminal states.
    """
    if hasattr(env, "_reach_terminal_states"):
        return

    if not os.path.exists(terminal_states_path):
        print(f"[WARN] Reach terminal states file not found: {terminal_states_path}")
        env._reach_terminal_states = None
        return

    print(f"[INFO] Loading reach terminal states from: {terminal_states_path}")
    env._reach_terminal_states = torch.load(terminal_states_path, map_location=env.device)
    print(f"[INFO] Loaded {len(env._reach_terminal_states['joint_pos'])} terminal states")


def reset_from_reach_terminal_states(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    use_rollout: bool = True,
    fallback_to_default: bool = True,
) -> None:
    """Reset robot and objects from pre-saved reach policy terminal states.

    This enables curriculum learning where grasp training starts from
    successful reach positions.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        robot_cfg: Robot configuration.
        left_object_cfg: Left object configuration.
        right_object_cfg: Right object configuration.
        use_rollout: Whether to use rollout states (if False, uses default reset).
        fallback_to_default: If True, falls back to default reset when states unavailable.
    """
    if not use_rollout:
        return

    if not hasattr(env, "_reach_terminal_states") or env._reach_terminal_states is None:
        if fallback_to_default:
            return  # Will use default reset
        raise RuntimeError("Reach terminal states not loaded. Call load_reach_terminal_states first.")

    terminal_states = env._reach_terminal_states
    num_states = len(terminal_states["joint_pos"])
    num_envs = len(env_ids)

    # Randomly select terminal states for each environment
    state_indices = torch.randint(0, num_states, (num_envs,), device=env.device)

    # Get robot and objects
    robot = env.scene[robot_cfg.name]

    # Set robot joint positions
    joint_pos = terminal_states["joint_pos"][state_indices]
    joint_vel = terminal_states.get("joint_vel", torch.zeros_like(joint_pos))
    if isinstance(joint_vel, torch.Tensor):
        joint_vel = joint_vel[state_indices]
    else:
        joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Set object positions if available
    if left_object_cfg.name in env.scene.keys():
        left_obj: RigidObject = env.scene[left_object_cfg.name]
        if "object_pos" in terminal_states:
            obj_pos = terminal_states["object_pos"][state_indices]
            obj_quat = terminal_states.get("object_quat", None)
            if obj_quat is not None:
                obj_quat = obj_quat[state_indices]
            else:
                obj_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).expand(num_envs, -1)

            left_obj.write_root_pose_to_sim(
                torch.cat([obj_pos, obj_quat], dim=-1),
                env_ids=env_ids
            )
            left_obj.write_root_velocity_to_sim(
                torch.zeros(num_envs, 6, device=env.device),
                env_ids=env_ids
            )

    if right_object_cfg.name in env.scene.keys():
        right_obj: RigidObject = env.scene[right_object_cfg.name]
        if "object2_pos" in terminal_states:
            obj2_pos = terminal_states["object2_pos"][state_indices]
            obj2_quat = terminal_states.get("object2_quat", None)
            if obj2_quat is not None:
                obj2_quat = obj2_quat[state_indices]
            else:
                obj2_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).expand(num_envs, -1)

            right_obj.write_root_pose_to_sim(
                torch.cat([obj2_pos, obj2_quat], dim=-1),
                env_ids=env_ids
            )
            right_obj.write_root_velocity_to_sim(
                torch.zeros(num_envs, 6, device=env.device),
                env_ids=env_ids
            )


def reset_gripper_open(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_gripper_joints: list[str] | None = None,
    right_gripper_joints: list[str] | None = None,
    open_position: float = 0.04,
) -> None:
    """Reset grippers to open position after roll-out reset.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        robot_cfg: Robot configuration.
        left_gripper_joints: Left gripper joint names.
        right_gripper_joints: Right gripper joint names.
        open_position: Open position for gripper joints.
    """
    robot = env.scene[robot_cfg.name]

    # Default gripper joint patterns
    if left_gripper_joints is None:
        left_gripper_joints = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
    if right_gripper_joints is None:
        right_gripper_joints = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

    all_gripper_joints = left_gripper_joints + right_gripper_joints

    # Find joint indices
    joint_indices = []
    for joint_name in all_gripper_joints:
        if joint_name in robot.data.joint_names:
            joint_indices.append(robot.data.joint_names.index(joint_name))

    if not joint_indices:
        return

    joint_indices = torch.tensor(joint_indices, device=env.device)

    # Set gripper positions to open
    current_joint_pos = robot.data.joint_pos[env_ids].clone()
    current_joint_pos[:, joint_indices] = open_position

    current_joint_vel = robot.data.joint_vel[env_ids].clone()
    current_joint_vel[:, joint_indices] = 0.0

    robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel, env_ids=env_ids)
