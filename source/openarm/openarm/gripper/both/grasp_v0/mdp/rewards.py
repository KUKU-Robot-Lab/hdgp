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

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp import joint_vel_l2
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance between the object and the specified end-effector link."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.norm(object_pos - eef_pos, dim=1)


def _object_eef_orientation_error(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Orientation error between the object and the specified end-effector link."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]
    return quat_error_magnitude(eef_quat, object_quat)


def _object_eef_any_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Maximum absolute alignment between any EE axis and any object axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    x_axis = x_axis.repeat(env.num_envs, 1)
    y_axis = y_axis.repeat(env.num_envs, 1)
    z_axis = z_axis.repeat(env.num_envs, 1)

    eef_axes = [
        quat_apply(eef_quat, x_axis),
        quat_apply(eef_quat, y_axis),
        quat_apply(eef_quat, z_axis),
    ]
    obj_axes = [
        quat_apply(object_quat, x_axis),
        quat_apply(object_quat, y_axis),
        quat_apply(object_quat, z_axis),
    ]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)

    return max_align


def _hand_closure_amount(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Compute normalized closure amount for the hand associated with the given link."""
    # identify which hand is being used to access the appropriate action term
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        # fallback: no closure info if the link does not correspond to a hand
        return torch.zeros(env.num_envs, device=env.device)

    # processed_actions has shape (num_envs, num_gripper_joints); take mean across joints
    hand_action = hand_term.processed_actions  # current commanded positions

    # Determine default (open) finger position.
    if isinstance(hand_term._offset, torch.Tensor):
        default_pos = hand_term._offset.mean(dim=1)
    else:
        default_pos = torch.full((env.num_envs,), float(hand_term._offset), device=env.device)

    # Compute closure amount: positive when fingers are closing relative to default.
    mean_action = hand_action.mean(dim=1)
    return torch.clamp(
        (default_pos - mean_action) / (torch.abs(default_pos) + 1e-6), min=0.0, max=1.0
    )


def eef_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the end-effector being close to the object using tanh-kernel."""
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance_with_ee(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    reach_std: float = 0.1,
) -> torch.Tensor:
    """Reward goal tracking only when the designated EE stays near the object."""
    goal_reward = object_goal_distance(
        env,
        std=std,
        minimal_height=minimal_height,
        command_name=command_name,
        object_cfg=object_cfg,
    )
    ee_reward = object_ee_distance(env, std=reach_std, object_cfg=object_cfg, ee_frame_cfg=ee_frame_cfg)
    return goal_reward * ee_reward


def eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the end-effector aligning with any object axis (loose tanh-kernel)."""
    max_align = _object_eef_any_axis_alignment(env, eef_link_name, object_cfg)
    error = 1.0 - max_align
    return 1 - torch.tanh(error / std)


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Continuous reward encouraging the end effector to close onto the object.

    This function provides a smoother shaping signal than the previous binary
    reward. It combines two components:
    - Proximity component: encourages the end effector to approach the object.
    - Closure component: encourages the gripper fingers to close relative to
      their nominal open position.
    """
    # distance between end-effector and object
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    # Sharper shaping: only rewards closing when sufficiently close.
    reach_radius = 0.05
    dist_scale = 0.03
    close_center = 0.6
    close_scale = 0.2
    dist_score = torch.sigmoid((reach_radius - eef_dist) / dist_scale)
    close_score = torch.sigmoid((closure_amount - close_center) / close_scale)

    return dist_score * close_score


def object_is_lifted_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    eef_link_name: str,
    reach_radius: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting only when the hand is close and sufficiently closed."""
    object: RigidObject = env.scene[object_cfg.name]
    lifted = object.data.root_pos_w[:, 2] > minimal_height

    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    gated = (eef_dist < reach_radius) & (closure_amount > close_threshold)
    return torch.where(lifted & gated, 1.0, 0.0)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting the object above a minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_held_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    eef_link_name: str,
    reach_radius: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for holding the object above a height while grasp gate is satisfied."""
    object: RigidObject = env.scene[object_cfg.name]

    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    gated = (eef_dist < reach_radius) & (closure_amount > close_threshold)
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    should_hold = is_lifted & gated

    attr_name = f"hold_counter_{object_cfg.name}_{eef_link_name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))

    hold_counter = getattr(env, attr_name)
    hold_counter = torch.where(
        should_hold,
        hold_counter + env.step_dt,
        torch.zeros_like(hold_counter),
    )
    setattr(env, attr_name, hold_counter)

    return torch.where(hold_counter > hold_duration, 1.0, 0.0)


def object_is_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for holding the object above a minimal height for a certain duration."""
    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "hold_counter"):
        env.hold_counter = torch.zeros(env.num_envs, device=env.device)

    is_lifted = object.data.root_pos_w[:, 2] > minimal_height

    env.hold_counter = torch.where(
        is_lifted,
        env.hold_counter + env.step_dt,
        torch.zeros_like(env.hold_counter),
    )

    return torch.where(env.hold_counter > hold_duration, 1.0, 0.0)


def object_is_held_per_object(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for holding each object above a minimal height for a duration."""
    object: RigidObject = env.scene[object_cfg.name]

    attr_name = f"hold_counter_{object_cfg.name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))

    hold_counter = getattr(env, attr_name)
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    hold_counter = torch.where(
        is_lifted,
        hold_counter + env.step_dt,
        torch.zeros_like(hold_counter),
    )
    setattr(env, attr_name, hold_counter)

    return torch.where(hold_counter > hold_duration, 1.0, 0.0)


def _phase_weight(phase: torch.Tensor, weights: list[float], device: torch.device) -> torch.Tensor:
    """Select per-phase weights for each env."""
    weights_tensor = torch.tensor(weights, device=device)
    return weights_tensor[phase]


def _reach_success(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    reach_distance: float,
    align_threshold: float | None,
) -> torch.Tensor:
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    if align_threshold is None or align_threshold <= 0.0:
        return dist < reach_distance
    align = _object_eef_any_axis_alignment(env, eef_link_name, object_cfg)
    return (dist < reach_distance) & (align > align_threshold)


def _grasp_success(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    grasp_distance: float,
    close_threshold: float,
) -> torch.Tensor:
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    close = _hand_closure_amount(env, eef_link_name)
    return (dist < grasp_distance) & (close > close_threshold)


def object_lift_progress(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Linear lift progress reward from table to target height."""
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2]
    progress = height / lift_height
    return torch.clamp(progress, min=0.0, max=1.0)


def _update_grasp2g_phase(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    lift_height: float,
    reach_distance: float,
    align_threshold: float,
    grasp_distance: float,
    close_threshold: float,
    hold_duration: float,
) -> torch.Tensor:
    """Update phase based on reach -> grasp -> lift conditions."""
    if "left" in eef_link_name:
        phase_attr = "grasp2g_phase_left"
    elif "right" in eef_link_name:
        phase_attr = "grasp2g_phase_right"
    else:
        phase_attr = "grasp2g_phase"

    if not hasattr(env, phase_attr):
        setattr(env, phase_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))

    phase = getattr(env, phase_attr)
    if hasattr(env, "reset_buf"):
        phase = torch.where(env.reset_buf, torch.zeros_like(phase), phase)

    reach_ok = _reach_success(env, eef_link_name, object_cfg, reach_distance, align_threshold)
    grasp_ok = _grasp_success(env, eef_link_name, object_cfg, grasp_distance, close_threshold)
    obj: RigidObject = env.scene[object_cfg.name]
    lift_ok = obj.data.root_pos_w[:, 2] > lift_height

    phase = torch.where((phase == 0) & reach_ok, torch.tensor(1, device=env.device), phase)
    phase = torch.where((phase == 1) & grasp_ok, torch.tensor(2, device=env.device), phase)
    phase = torch.where((phase == 2) & lift_ok, torch.tensor(3, device=env.device), phase)
    setattr(env, phase_attr, phase)
    return phase


def phase_eef_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = eef_to_object_distance(env, std, eef_link_name, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = eef_to_object_orientation(env, std, eef_link_name, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = grasp_reward(env, eef_link_name, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_object_goal_distance_with_ee(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    reach_std: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated goal tracking with EE proximity."""
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_goal_distance_with_ee(
        env,
        std=std,
        minimal_height=minimal_height,
        command_name=command_name,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
        reach_std=reach_std,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def gripper_open_reward(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Reward keeping the gripper open (low closure amount)."""
    closure_amount = _hand_closure_amount(env, eef_link_name)
    return 1.0 - closure_amount


def reach_preclose_reward(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Small shaping reward to encourage pre-closing during reach."""
    closure_amount = _hand_closure_amount(env, eef_link_name)
    reward = 0.2 * closure_amount - 0.05
    return torch.clamp(reward, min=-0.05, max=0.15)


def phase_gripper_open_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = gripper_open_reward(env, eef_link_name)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_reach_preclose_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = reach_preclose_reward(env, eef_link_name)
    return reward * _phase_weight(phase, phase_weights, env.device)


def closed_far_reach_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_threshold: float,
    std: float,
) -> torch.Tensor:
    """Encourage moving closer if the gripper is already closed."""
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)
    dist_reward = 1 - torch.tanh(eef_dist / std)
    return torch.where(closure_amount > close_threshold, dist_reward, 0.0)


def phase_closed_far_reach_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_threshold: float,
    std: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = closed_far_reach_reward(env, eef_link_name, object_cfg, close_threshold, std)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_lift_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_lift_progress(env, lift_height, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_hold_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_is_held_per_object(env, lift_height, hold_duration, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def grasp2g_phase_value(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    phase_params: dict,
) -> torch.Tensor:
    """Return the per-hand phase as a float for logging."""
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    return phase.to(torch.float32)


def phase_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["align_threshold"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = joint_vel_l2(env, asset_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def joints_near_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Terminate when specified joints are near zero (abs < threshold)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return (torch.abs(q) < threshold).all(dim=1)
