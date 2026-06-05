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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Distance between the object and the specified end-effector link."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    # Optional per-task offset (e.g., cup root -> grasp center)
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        object_pos = object_pos + torch.tensor(offset, device=object_pos.device)
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.norm(object_pos - eef_pos, dim=1)

def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)

def _object_eef_any_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        return torch.zeros(env.num_envs, device=env.device)

    joint_ids = hand_term._joint_ids
    joint_pos = hand_term._asset.data.joint_pos[:, joint_ids]
    joint_limits = hand_term._asset.data.joint_pos_limits[:, joint_ids]
    min_lim = joint_limits[..., 0]
    max_lim = joint_limits[..., 1]

    if hasattr(hand_term, "_open_command") and hasattr(hand_term, "_close_command"):
        open_lim = hand_term._open_command.unsqueeze(0).expand_as(joint_pos)
        close_lim = hand_term._close_command.unsqueeze(0).expand_as(joint_pos)
    else:
        # Determine which limit corresponds to open based on configured offset.
        if isinstance(hand_term._offset, torch.Tensor):
            open_pos = hand_term._offset
        else:
            open_pos = torch.full_like(min_lim, float(hand_term._offset))

        # For each joint, pick the limit closest to open_pos as the "open" limit.
        dist_to_min = torch.abs(open_pos - min_lim)
        dist_to_max = torch.abs(open_pos - max_lim)
        open_lim = torch.where(dist_to_min <= dist_to_max, min_lim, max_lim)
        close_lim = torch.where(dist_to_min <= dist_to_max, max_lim, min_lim)

    denom = close_lim - open_lim
    denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
    closure = (joint_pos - open_lim) / denom
    closure = torch.clamp(closure, min=0.0, max=1.0)
    return closure.mean(dim=1)


def _hand_closure_debug_stats(
    env: ManagerBasedRLEnv, eef_link_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closure amount along with joint/action/limit stats for debug logging."""
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        zeros = torch.zeros(env.num_envs, device=env.device)
        return zeros, zeros, zeros, zeros, zeros

    joint_ids = hand_term._joint_ids
    joint_pos = hand_term._asset.data.joint_pos[:, joint_ids]
    joint_limits = hand_term._asset.data.joint_pos_limits[:, joint_ids]
    min_lim = joint_limits[..., 0]
    max_lim = joint_limits[..., 1]
    if hasattr(hand_term, "_open_command") and hasattr(hand_term, "_close_command"):
        open_lim = hand_term._open_command.unsqueeze(0).expand_as(joint_pos)
        close_lim = hand_term._close_command.unsqueeze(0).expand_as(joint_pos)
    else:
        if isinstance(hand_term._offset, torch.Tensor):
            open_pos = hand_term._offset
        else:
            open_pos = torch.full_like(min_lim, float(hand_term._offset))

        dist_to_min = torch.abs(open_pos - min_lim)
        dist_to_max = torch.abs(open_pos - max_lim)
        open_lim = torch.where(dist_to_min <= dist_to_max, min_lim, max_lim)
        close_lim = torch.where(dist_to_min <= dist_to_max, max_lim, min_lim)

    denom = close_lim - open_lim
    denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
    closure = torch.clamp((joint_pos - open_lim) / denom, min=0.0, max=1.0).mean(dim=1)

    hand_action = hand_term.processed_actions
    mean_action = hand_action.mean(dim=1)
    mean_joint = joint_pos.mean(dim=1)
    open_mean = open_lim.mean(dim=1)
    close_mean = close_lim.mean(dim=1)
    return closure, mean_action, mean_joint, open_mean, close_mean


def _maybe_log_grasp_debug(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    dist: torch.Tensor,
    close: torch.Tensor,
    mean_action: torch.Tensor,
    mean_joint: torch.Tensor,
    open_mean: torch.Tensor,
    close_mean: torch.Tensor,
) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return

    if "left" in eef_link_name:
        enabled = getattr(cfg, "debug_grasp_left", False)
        interval = int(getattr(cfg, "debug_grasp_left_interval", 200))
        last_key = "_debug_grasp_left_last_step"
        side = "left"
    elif "right" in eef_link_name:
        enabled = getattr(cfg, "debug_grasp_right", False)
        interval = int(getattr(cfg, "debug_grasp_right_interval", 200))
        last_key = "_debug_grasp_right_last_step"
        side = "right"
    else:
        return

    if not enabled:
        return
    step_count = getattr(env, "common_step_counter", None)
    if step_count is None:
        return
    last_step = getattr(env, last_key, None)
    if last_step == int(step_count):
        return
    if int(step_count) % max(interval, 1) != 0:
        return

    setattr(env, last_key, int(step_count))
    close_mean_val = float(close.mean().item())
    dist_mean = float(dist.mean().item())
    action_mean = float(mean_action.mean().item())
    joint_mean = float(mean_joint.mean().item())
    open_val = float(open_mean.mean().item())
    close_val = float(close_mean.mean().item())
    delta_mean = float((open_mean - mean_action).mean().item())
    print(
        "[GRASP_DEBUG] "
        f"step={int(step_count)} side={side} "
        f"close_mean={close_mean_val:.3f} dist_mean={dist_mean:.3f} "
        f"action_mean={action_mean:.4f} joint_mean={joint_mean:.4f} "
        f"open_mean={open_val:.4f} close_mean={close_val:.4f} "
        f"delta_mean={delta_mean:.4f}"
    )


def _maybe_log_reach_xy_z_debug(env: ManagerBasedRLEnv) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    if not getattr(cfg, "debug_reach_reward", False):
        return
    step_count = getattr(env, "common_step_counter", None)
    if step_count is None:
        return
    last_step = getattr(env, "_debug_reach_xyz_last_step", None)
    if last_step == int(step_count):
        return
    interval = int(getattr(cfg, "debug_reach_reward_interval", 200))
    if int(step_count) % max(interval, 1) != 0:
        return
    setattr(env, "_debug_reach_xyz_last_step", int(step_count))

    offset = getattr(cfg, "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        offset = (0.0, 0.0, 0.0)
    offset_t = torch.tensor(offset, device=env.device)

    std_xy = float(getattr(cfg, "grasp2g_reach_std_xy", 0.15))
    std_z = float(getattr(cfg, "grasp2g_reach_std_z", 0.1))
    z_weight = float(getattr(cfg, "grasp2g_reach_z_weight", 2.0))

    body_pos_w = env.scene["robot"].data.body_pos_w
    body_names = env.scene["robot"].data.body_names

    def _log(side: str, cup_name: str, hand_name: str):
        try:
            eef_idx = body_names.index(hand_name)
        except ValueError:
            return
        obj = env.scene[cup_name].data.root_pos_w + offset_t
        ee = body_pos_w[:, eef_idx]
        diff = ee - obj
        dist_xy = torch.linalg.norm(diff[:, :2], dim=1)
        dist_z = torch.abs(diff[:, 2])
        weighted = dist_xy / std_xy + z_weight * (dist_z / std_z)
        reward = 1 - torch.tanh(weighted)
        dist_xy_m = dist_xy.mean().item()
        dist_z_m = dist_z.mean().item()
        w_m = weighted.mean().item()
        r_m = reward.mean().item()
        print(
            f"[REACH_XYZ] step={int(step_count)} side={side} "
            f"dist_xy={dist_xy_m:.4f} dist_z={dist_z_m:.4f} "
            f"weighted={w_m:.4f} reward={r_m:.4f}"
        )

    _log("left", "cup", "openarm_left_hand")
    _log("right", "cup2", "openarm_right_hand")


def _maybe_visualize_grasp_targets(env: ManagerBasedRLEnv) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    if not getattr(cfg, "debug_grasp_target_vis", False):
        return

    step_count = getattr(env, "common_step_counter", None)
    if step_count is None:
        return
    last_step = getattr(env, "_debug_grasp_target_last_step", None)
    interval = int(getattr(cfg, "debug_grasp_target_vis_interval", 10))
    if last_step == int(step_count):
        return
    if int(step_count) % max(interval, 1) != 0:
        return
    setattr(env, "_debug_grasp_target_last_step", int(step_count))

    if not hasattr(env, "_debug_grasp_target_left"):
        left_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/GraspTargetLeft")
        left_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        env._debug_grasp_target_left = VisualizationMarkers(left_cfg)
        env._debug_grasp_target_left.set_visibility(True)
    if not hasattr(env, "_debug_grasp_target_right"):
        right_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/GraspTargetRight")
        right_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        env._debug_grasp_target_right = VisualizationMarkers(right_cfg)
        env._debug_grasp_target_right.set_visibility(True)

    offset = getattr(cfg, "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        offset = (0.0, 0.0, 0.0)
    offset_t = torch.tensor(offset, device=env.device)

    left_obj = env.scene["cup"].data.root_pos_w + offset_t
    right_obj = env.scene["cup2"].data.root_pos_w + offset_t
    quat = torch.zeros((env.num_envs, 4), device=env.device)
    quat[:, 0] = 1.0
    env._debug_grasp_target_left.visualize(left_obj, quat)
    env._debug_grasp_target_right.visualize(right_obj, quat)


def eef_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the end-effector being close to the object using tanh-kernel."""
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    # Optional per-task offset (e.g., cup root -> grasp center)
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        cube_pos_w = cube_pos_w + torch.tensor(offset, device=cube_pos_w.device)
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(dist / std)


def object_ee_distance_error(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """Dense reach error (distance) without kernel scaling."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        obj_pos = obj_pos + torch.tensor(offset, device=obj_pos.device)
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    return torch.norm(obj_pos - ee_w, dim=1)


def phase_object_ee_distance_error(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated dense reach error (distance)."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_ee_distance_error(env, object_cfg, eef_link_name)
    return reward * _phase_weight(phase, phase_weights, env.device)


def object_ee_distance_xyz_weighted(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """Reach reward with separate XY and Z errors (heavier Z weighting)."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    # Optional per-task offset (e.g., cup root -> grasp center)
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        obj_pos = obj_pos + torch.tensor(offset, device=obj_pos.device)
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos = env.scene["robot"].data.body_pos_w[:, eef_idx]

    diff = ee_pos - obj_pos
    dist_xy = torch.linalg.norm(diff[:, :2], dim=1)
    dist_z = torch.abs(diff[:, 2])
    weighted = dist_xy / std_xy + z_weight * (dist_z / std_z)
    return 1 - torch.tanh(weighted)


def object_ee_distance_xy_only(
    env: ManagerBasedRLEnv,
    std_xy: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """Reach reward for XY plane only (ignore Z axis)."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        obj_pos = obj_pos + torch.tensor(offset, device=obj_pos.device)
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos = env.scene["robot"].data.body_pos_w[:, eef_idx]

    diff = ee_pos - obj_pos
    dist_xy = torch.linalg.norm(diff[:, :2], dim=1)
    return 1 - torch.tanh(dist_xy / std_xy)


def object_ee_distance_xy_then_z(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    xy_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """XY-first curriculum reach reward.

    Only activates Z reward when XY distance is below xy_threshold.
    This encourages the policy to first learn XY positioning before Z.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        obj_pos = obj_pos + torch.tensor(offset, device=obj_pos.device)
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos = env.scene["robot"].data.body_pos_w[:, eef_idx]

    diff = ee_pos - obj_pos
    dist_xy = torch.linalg.norm(diff[:, :2], dim=1)
    dist_z = torch.abs(diff[:, 2])

    # XY reward (always active)
    reward_xy = 1 - torch.tanh(dist_xy / std_xy)

    # Z reward (only when XY is close enough)
    reward_z = 1 - torch.tanh(dist_z / std_z)
    z_gate = (dist_xy < xy_threshold).float()

    # Combined: XY always + Z only when XY is good
    return reward_xy + z_weight * reward_z * z_gate


def object_ee_reach_sparse(
    env: ManagerBasedRLEnv,
    reach_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
) -> torch.Tensor:
    """Sparse reach reward: 1 if within reach_distance, else 0."""
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    return torch.where(dist < reach_distance, 1.0, 0.0)


def phase_object_ee_reach_sparse(
    env: ManagerBasedRLEnv,
    reach_distance: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated sparse reach reward."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_ee_reach_sparse(env, reach_distance, object_cfg, eef_link_name)
    return reward * _phase_weight(phase, phase_weights, env.device)


def gripper_hold_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    close_threshold: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
    distance_threshold: float,
    hold_decay: float,
) -> torch.Tensor:
    """Reward holding the gripper closed near the object for a minimum duration."""
    closure = _hand_closure_amount(env, eef_link_name)
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    should_hold = (closure > close_threshold) & (dist < distance_threshold)
    attr_name = f"_gripper_hold_counter_{eef_link_name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))
    counter = getattr(env, attr_name)
    counter = torch.where(should_hold, counter + env.step_dt, torch.zeros_like(counter))
    setattr(env, attr_name, counter)
    sustained = torch.clamp(counter - hold_duration, min=0.0)
    if hold_decay > 0.0:
        reward = torch.exp(-sustained / hold_decay)
    else:
        reward = torch.where(counter > hold_duration, 1.0, 0.0)
    return reward


def phase_gripper_hold_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    close_threshold: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
    hold_decay: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated gripper hold reward."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = gripper_hold_reward(
        env,
        eef_link_name,
        close_threshold,
        hold_duration,
        object_cfg,
        phase_params["grasp_distance"],
        hold_decay,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_object_ee_distance_xyz_weighted(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated reach reward with heavier Z weighting."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_ee_distance_xyz_weighted(
        env,
        std_xy=std_xy,
        std_z=std_z,
        z_weight=z_weight,
        object_cfg=object_cfg,
        eef_link_name=eef_link_name,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_object_ee_distance_xy_then_z(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    xy_threshold: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated XY-first curriculum reach reward."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_ee_distance_xy_then_z(
        env,
        std_xy=std_xy,
        std_z=std_z,
        z_weight=z_weight,
        xy_threshold=xy_threshold,
        object_cfg=object_cfg,
        eef_link_name=eef_link_name,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated reach reward using tanh distance (reach_env_cfg 방식)."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_ee_distance_tanh(
        env,
        std=std,
        object_cfg=object_cfg,
        eef_link_name=eef_link_name,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "openarm_left_hand",
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
    ee_reward = object_ee_distance(env, std=reach_std, object_cfg=object_cfg, eef_link_name=eef_link_name)
    return goal_reward * ee_reward


def eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the end-effector aligning with any object axis (loose tanh-kernel)."""
    max_align = _object_eef_any_axis_alignment(env, eef_link_name, object_cfg)
    error = 1.0 - max_align
    return 1 - torch.tanh(error / std)


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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


def grasp_closure_band_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_min: float,
    close_max: float,
    distance_threshold: float,
) -> torch.Tensor:
    """Reward keeping the gripper partially closed near the object."""
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure = _hand_closure_amount(env, eef_link_name)
    center = 0.5 * (close_min + close_max)
    half = max(1e-6, 0.5 * (close_max - close_min))
    band = 1.0 - torch.clamp(torch.abs(closure - center) / half, min=0.0, max=1.0)
    gate = (dist < distance_threshold).to(dtype=band.dtype)
    return band * gate


def phase_grasp_band_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_min: float,
    close_max: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated grasp reward using closure band."""
    phase = _update_grasp2g_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = grasp_closure_band_reward(
        env,
        eef_link_name,
        object_cfg,
        close_min,
        close_max,
        phase_params["grasp_distance"],
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def object_is_lifted_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    eef_link_name: str,
    reach_radius: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
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
    return (dist < grasp_distance) | (close > close_threshold)

def bimanual_reach_min_reward(
    env: ManagerBasedRLEnv,
    std: float,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Reward based on min(reach_L, reach_R) with phase gating."""
    phase_left = _update_grasp2g_phase(
        env,
        left_eef_link_name,
        left_object_cfg,
        left_phase_params["lift_height"],
        left_phase_params["reach_distance"],
        left_phase_params.get("align_threshold", 0.0),
        left_phase_params["grasp_distance"],
        left_phase_params["close_threshold"],
        left_phase_params["hold_duration"],
    )
    phase_right = _update_grasp2g_phase(
        env,
        right_eef_link_name,
        right_object_cfg,
        right_phase_params["lift_height"],
        right_phase_params["reach_distance"],
        right_phase_params.get("align_threshold", 0.0),
        right_phase_params["grasp_distance"],
        right_phase_params["close_threshold"],
        right_phase_params["hold_duration"],
    )

    dist_left = _object_eef_distance(env, left_eef_link_name, left_object_cfg)
    dist_right = _object_eef_distance(env, right_eef_link_name, right_object_cfg)
    reach_left = 1 - torch.tanh(dist_left / std)
    reach_right = 1 - torch.tanh(dist_right / std)
    reach_min = torch.minimum(reach_left, reach_right)

    w_left = _phase_weight(phase_left, phase_weights, env.device)
    w_right = _phase_weight(phase_right, phase_weights, env.device)
    return reach_min * torch.minimum(w_left, w_right)


def bimanual_phase_lag_penalty(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Penalty for phase mismatch between left and right."""
    phase_left = _update_grasp2g_phase(
        env,
        left_eef_link_name,
        left_object_cfg,
        left_phase_params["lift_height"],
        left_phase_params["reach_distance"],
        left_phase_params.get("align_threshold", 0.0),
        left_phase_params["grasp_distance"],
        left_phase_params["close_threshold"],
        left_phase_params["hold_duration"],
    )
    phase_right = _update_grasp2g_phase(
        env,
        right_eef_link_name,
        right_object_cfg,
        right_phase_params["lift_height"],
        right_phase_params["reach_distance"],
        right_phase_params.get("align_threshold", 0.0),
        right_phase_params["grasp_distance"],
        right_phase_params["close_threshold"],
        right_phase_params["hold_duration"],
    )
    return torch.abs(phase_left - phase_right).float()


def bimanual_grasp_and_reward(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Binary reward for grasp_L AND grasp_R with phase gating."""
    phase_left = _update_grasp2g_phase(
        env,
        left_eef_link_name,
        left_object_cfg,
        left_phase_params["lift_height"],
        left_phase_params["reach_distance"],
        left_phase_params.get("align_threshold", 0.0),
        left_phase_params["grasp_distance"],
        left_phase_params["close_threshold"],
        left_phase_params["hold_duration"],
    )
    phase_right = _update_grasp2g_phase(
        env,
        right_eef_link_name,
        right_object_cfg,
        right_phase_params["lift_height"],
        right_phase_params["reach_distance"],
        right_phase_params.get("align_threshold", 0.0),
        right_phase_params["grasp_distance"],
        right_phase_params["close_threshold"],
        right_phase_params["hold_duration"],
    )

    dist_left = _object_eef_distance(env, left_eef_link_name, left_object_cfg)
    dist_right = _object_eef_distance(env, right_eef_link_name, right_object_cfg)
    close_left = _hand_closure_amount(env, left_eef_link_name)
    close_right = _hand_closure_amount(env, right_eef_link_name)
    grasp_left = (dist_left < left_phase_params["grasp_distance"]) & (close_left > left_phase_params["close_threshold"])
    grasp_right = (dist_right < right_phase_params["grasp_distance"]) & (close_right > right_phase_params["close_threshold"])
    success = grasp_left & grasp_right

    w_left = _phase_weight(phase_left, phase_weights, env.device)
    w_right = _phase_weight(phase_right, phase_weights, env.device)
    return success.float() * torch.minimum(w_left, w_right)


def bimanual_grasp_xor_penalty(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Penalty when exactly one hand grasps."""
    phase_left = _update_grasp2g_phase(
        env,
        left_eef_link_name,
        left_object_cfg,
        left_phase_params["lift_height"],
        left_phase_params["reach_distance"],
        left_phase_params.get("align_threshold", 0.0),
        left_phase_params["grasp_distance"],
        left_phase_params["close_threshold"],
        left_phase_params["hold_duration"],
    )
    phase_right = _update_grasp2g_phase(
        env,
        right_eef_link_name,
        right_object_cfg,
        right_phase_params["lift_height"],
        right_phase_params["reach_distance"],
        right_phase_params.get("align_threshold", 0.0),
        right_phase_params["grasp_distance"],
        right_phase_params["close_threshold"],
        right_phase_params["hold_duration"],
    )

    dist_left = _object_eef_distance(env, left_eef_link_name, left_object_cfg)
    dist_right = _object_eef_distance(env, right_eef_link_name, right_object_cfg)
    close_left = _hand_closure_amount(env, left_eef_link_name)
    close_right = _hand_closure_amount(env, right_eef_link_name)
    grasp_left = (dist_left < left_phase_params["grasp_distance"]) & (close_left > left_phase_params["close_threshold"])
    grasp_right = (dist_right < right_phase_params["grasp_distance"]) & (close_right > right_phase_params["close_threshold"])
    xor = grasp_left ^ grasp_right

    w_left = _phase_weight(phase_left, phase_weights, env.device)
    w_right = _phase_weight(phase_right, phase_weights, env.device)
    return xor.float() * torch.minimum(w_left, w_right)


def object_lift_progress(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Linear lift progress reward from table to target height."""
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2]
    progress = height / lift_height
    return torch.clamp(progress, min=0.0, max=1.0)


def object_lift_delta_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward only positive lift progress (delta)."""
    obj: RigidObject = env.scene[object_cfg.name]
    attr_name = f"_prev_lift_height_{object_cfg.name}"
    if not hasattr(env, attr_name):
        prev = obj.data.root_pos_w[:, 2].clone()
    else:
        prev = getattr(env, attr_name)
    if hasattr(env, "reset_buf"):
        prev = torch.where(env.reset_buf, obj.data.root_pos_w[:, 2], prev)
    height = obj.data.root_pos_w[:, 2]
    delta = torch.clamp(height - prev, min=0.0)
    setattr(env, attr_name, height)
    return torch.clamp(delta / max(lift_height, 1e-6), min=0.0, max=1.0)


def _object_root_displacement_from_init(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute displacement from per-episode initial root position."""
    obj: RigidObject = env.scene[object_cfg.name]
    attr_name = f"_init_root_pos_{object_cfg.name}"

    if not hasattr(env, attr_name):
        init_pos = obj.data.root_pos_w.clone()
    else:
        init_pos = getattr(env, attr_name)

    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.unsqueeze(1)
        init_pos = torch.where(reset_mask, obj.data.root_pos_w, init_pos)

    setattr(env, attr_name, init_pos)
    return torch.linalg.norm(obj.data.root_pos_w - init_pos, dim=1)


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
    """Update phase based on reach -> grasp -> lift conditions.

    N-step 연속 조건: env.cfg의 phase_stability_*_steps 파라미터로
    각 phase 전환에 필요한 연속 충족 step 수를 설정 가능.
    역전환(demotion): env.cfg.phase_demotion_enabled로 토글 (기본 비활성).
    """
    # Phase 정의:
    # 0: reach (EEF가 물체 근처, 필요시 정렬 조건 포함)
    # 1: grasp (EEF가 충분히 가깝거나 그리퍼 닫힘 조건 만족)
    # 2: lift (물체 높이가 lift_height 이상)
    # 3: hold/goal (lift 이후; phase 가중 보상에서 사용)
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

    # --- N-step 연속 카운터 초기화 ---
    reach_count_attr = f"{phase_attr}_reach_count"
    grasp_count_attr = f"{phase_attr}_grasp_count"
    lift_count_attr = f"{phase_attr}_lift_count"

    for attr in (reach_count_attr, grasp_count_attr, lift_count_attr):
        if not hasattr(env, attr):
            setattr(env, attr, torch.zeros(env.num_envs, device=env.device))

    if hasattr(env, "reset_buf"):
        for attr in (reach_count_attr, grasp_count_attr, lift_count_attr):
            counter = getattr(env, attr)
            setattr(env, attr, torch.where(env.reset_buf, torch.zeros_like(counter), counter))

    # cfg에서 N-step 설정 읽기 (없으면 1 = 기존 동작과 동일)
    cfg = getattr(env, "cfg", None)
    n_reach = float(getattr(cfg, "phase_stability_reach_steps", 1))
    n_grasp = float(getattr(cfg, "phase_stability_grasp_steps", 1))
    n_lift = float(getattr(cfg, "phase_stability_lift_steps", 1))

    _maybe_visualize_grasp_targets(env)
    _maybe_log_reach_xy_z_debug(env)
    # Phase 0 -> 1: reach 게이트 (거리 + 선택적 정렬).
    reach_ok = _reach_success(env, eef_link_name, object_cfg, reach_distance, align_threshold)
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    close, mean_action, mean_joint, open_mean, close_mean = _hand_closure_debug_stats(env, eef_link_name)
    _maybe_log_grasp_debug(
        env,
        eef_link_name,
        dist,
        close,
        mean_action,
        mean_joint,
        open_mean,
        close_mean,
    )
    # Phase 1 -> 2: grasp 게이트 (거리 AND 닫힘).
    grasp_ok = (dist < grasp_distance) & (close > close_threshold)
    obj: RigidObject = env.scene[object_cfg.name]
    # Phase 2 -> 3: lift 게이트 (물체 높이 임계값 통과).
    lift_ok = obj.data.root_pos_w[:, 2] > lift_height

    # --- N-step 연속 카운터 업데이트 ---
    # Reach 카운터: 조건 충족 시 +1, 미충족 시 0으로 리셋
    reach_count = getattr(env, reach_count_attr)
    reach_count = torch.where(reach_ok & (phase == 0), reach_count + 1, torch.zeros_like(reach_count))
    setattr(env, reach_count_attr, reach_count)
    stable_reach = reach_count >= n_reach

    # Grasp 카운터
    grasp_count = getattr(env, grasp_count_attr)
    grasp_count = torch.where(grasp_ok & (phase == 1), grasp_count + 1, torch.zeros_like(grasp_count))
    setattr(env, grasp_count_attr, grasp_count)
    stable_grasp = grasp_count >= n_grasp

    # Lift 카운터
    lift_count = getattr(env, lift_count_attr)
    lift_count = torch.where(lift_ok & (phase == 2), lift_count + 1, torch.zeros_like(lift_count))
    setattr(env, lift_count_attr, lift_count)
    stable_lift = lift_count >= n_lift

    # Phase 전환 (N-step 연속 조건 적용)
    phase = torch.where((phase == 0) & stable_reach, torch.tensor(1, device=env.device), phase)
    phase = torch.where((phase == 1) & stable_grasp, torch.tensor(2, device=env.device), phase)
    phase = torch.where((phase == 2) & stable_lift, torch.tensor(3, device=env.device), phase)

    # --- 역전환 (demotion) --- 기본 비활성
    demotion_enabled = getattr(cfg, "phase_demotion_enabled", False)
    if demotion_enabled:
        margin = getattr(cfg, "phase_demotion_margin", 1.5)
        # Phase 1에서 reach 거리가 margin배 이상 벌어지면 Phase 0으로 복귀
        demote_reach = (phase == 1) & (dist > reach_distance * margin)
        # Phase 2에서 grasp 조건 미충족 시 Phase 1로 복귀
        demote_grasp = (phase == 2) & (~grasp_ok)
        phase = torch.where(demote_reach, torch.tensor(0, device=env.device), phase)
        phase = torch.where(demote_grasp, torch.tensor(1, device=env.device), phase)

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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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
    eef_link_name: str,
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
        phase_params.get("align_threshold", 0.0),
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
        eef_link_name=eef_link_name,
        reach_std=reach_std,
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_object_root_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for moving the object away from its per-episode initial position."""
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    displacement = _object_root_displacement_from_init(env, object_cfg) * scale
    return displacement * _phase_weight(phase, phase_weights, env.device)


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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_lift_progress(env, lift_height, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_lift_delta_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated lift reward based on positive height change only."""
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = object_lift_delta_reward(env, lift_height, object_cfg)
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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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
        phase_params.get("align_threshold", 0.0),
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


def hand_x_align_object_z_reward(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward aligning hand +X axis with command/object +Z axis.

    Returns a [0, 1] reward using (1 + cos(theta)) / 2.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    x_axis = x_axis.repeat(curr_quat_w.shape[0], 1)
    z_axis = z_axis.repeat(curr_quat_w.shape[0], 1)

    hand_x = quat_apply(curr_quat_w, x_axis)
    obj_z = quat_apply(des_quat_w, z_axis)
    cos_sim = torch.sum(hand_x * obj_z, dim=1)
    return 0.5 * (1.0 + cos_sim)


def hand_x_align_object_z_penalty_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    gate_std: float,
) -> torch.Tensor:
    """Penalty for misalignment, gated by distance to the object."""
    align = hand_x_align_object_z_reward(env, command_name, asset_cfg)
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    gate = torch.exp(-dist / max(gate_std, 1e-6))
    return (1.0 - align) * gate


def phase_hand_x_align_object_z_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
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
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = hand_x_align_object_z_reward(env, command_name, asset_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


## ──────────────────────────────────────────────
## Diagnostic (weight=0) reward terms for tensorboard logging
## ──────────────────────────────────────────────

def hand_closure_diagnostic(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
) -> torch.Tensor:
    """Diagnostic: mean gripper closure [0=open, 1=closed]. Use with weight=0."""
    return _hand_closure_amount(env, eef_link_name)


def eef_distance_diagnostic(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: EEF-to-object distance in meters. Use with weight=0."""
    return _object_eef_distance(env, eef_link_name, object_cfg)


def arm_action_norm_diagnostic(
    env: ManagerBasedRLEnv,
    action_name: str,
) -> torch.Tensor:
    """Diagnostic: L2 norm of arm action. Use with weight=0."""
    action_term = env.action_manager.get_term(action_name)
    return torch.norm(action_term.processed_actions, dim=-1)


def hand_action_norm_diagnostic(
    env: ManagerBasedRLEnv,
    action_name: str,
) -> torch.Tensor:
    """Diagnostic: L2 norm of hand action. Use with weight=0."""
    action_term = env.action_manager.get_term(action_name)
    return torch.norm(action_term.processed_actions, dim=-1)


def eef_dist_xy_diagnostic(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: XY-plane distance from EEF to object (horizontal approach). Use with weight=0."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        object_pos = object_pos + torch.tensor(offset, device=object_pos.device)
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    diff = object_pos - eef_pos
    return torch.norm(diff[:, :2], dim=1)  # XY only


def eef_dist_z_diagnostic(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: Z-axis distance from EEF to object (vertical approach). Use with weight=0."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        object_pos = object_pos + torch.tensor(offset, device=object_pos.device)
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.abs(object_pos[:, 2] - eef_pos[:, 2])


def object_height_diagnostic(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: object Z-position (absolute height). Use with weight=0."""
    return env.scene[object_cfg.name].data.root_pos_w[:, 2]


def object_displacement_diagnostic(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: object displacement from episode start position. Use with weight=0."""
    return _object_root_displacement_from_init(env, object_cfg)


def eef_dist_delta_diagnostic(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Diagnostic: change in EEF distance between steps (negative=approaching). Use with weight=0."""
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    attr_name = f"_prev_eef_dist_{eef_link_name}_{object_cfg.name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, dist.clone())
    prev = getattr(env, attr_name)
    delta = dist - prev
    # reset on episode boundary
    if hasattr(env, "reset_buf"):
        delta = torch.where(env.reset_buf, torch.zeros_like(delta), delta)
    setattr(env, attr_name, dist.clone())
    return delta


def phase_hand_x_align_object_z_penalty_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    gate_std: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    phase = _update_grasp2g_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params.get("align_threshold", 0.0),
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
        phase_params["hold_duration"],
    )
    reward = hand_x_align_object_z_penalty_gated(
        env, command_name, asset_cfg, eef_link_name, object_cfg, gate_std
    )
    return reward * _phase_weight(phase, phase_weights, env.device)


# ============================================================
# STAGED REWARD FUNCTIONS (for Role-Separated Curriculum v2)
# ============================================================

def _get_curriculum_stage(env: ManagerBasedRLEnv) -> int:
    """Get current curriculum stage from env.cfg."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return 2  # Default to BIMANUAL
    return int(getattr(cfg, "curriculum_stage", 2))


def _is_stage_active(env: ManagerBasedRLEnv, active_stages: list[int]) -> bool:
    """Check if current curriculum stage is in active_stages list."""
    current_stage = _get_curriculum_stage(env)
    return current_stage in active_stages


def _stage_mask(env: ManagerBasedRLEnv, active_stages: list[int]) -> torch.Tensor:
    """Return 1.0 if current stage is active, 0.0 otherwise (per-env tensor)."""
    if _is_stage_active(env, active_stages):
        return torch.ones(env.num_envs, device=env.device)
    return torch.zeros(env.num_envs, device=env.device)


def staged_phase_object_ee_distance_error(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase object EE distance error."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_object_ee_distance_error(env, eef_link_name, object_cfg, phase_weights, phase_params)


def staged_phase_object_ee_distance_xyz_weighted(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase object EE distance XYZ weighted."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_object_ee_distance_xyz_weighted(
        env, std_xy, std_z, z_weight, eef_link_name, object_cfg, phase_weights, phase_params
    )


def staged_phase_object_ee_distance_xy_then_z(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    z_weight: float,
    xy_threshold: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated XY-first curriculum reach reward.

    XY reward is always active. Z reward only activates when XY distance < xy_threshold.
    This encourages the policy to first learn horizontal positioning before vertical.
    """
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_object_ee_distance_xy_then_z(
        env, std_xy, std_z, z_weight, xy_threshold, eef_link_name, object_cfg, phase_weights, phase_params
    )


def staged_phase_object_root_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
    scale: float = 1.0,
) -> torch.Tensor:
    """Stage-gated phase object root displacement penalty."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_object_root_displacement_penalty(env, object_cfg, phase_weights, phase_params, scale)


def staged_phase_hand_x_align_object_z_penalty_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    gate_std: float,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase hand X align object Z penalty."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_hand_x_align_object_z_penalty_gated(
        env, command_name, asset_cfg, eef_link_name, object_cfg, gate_std, phase_weights, phase_params
    )


def staged_phase_grasp_band_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_min: float,
    close_max: float,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase grasp band reward."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_grasp_band_reward(env, eef_link_name, object_cfg, close_min, close_max, phase_weights, phase_params)


def staged_phase_gripper_hold_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    close_threshold: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
    hold_decay: float,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase gripper hold reward."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_gripper_hold_reward(
        env, eef_link_name, close_threshold, hold_duration, object_cfg, hold_decay, phase_weights, phase_params
    )


def staged_phase_lift_delta_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase lift delta reward."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_lift_delta_reward(env, lift_height, object_cfg, phase_weights, phase_params)


def staged_phase_object_goal_distance_with_ee(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    reach_std: float,
    active_stages: list[int],
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Stage-gated phase object goal distance with EE."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return phase_object_goal_distance_with_ee(
        env, std, minimal_height, command_name, object_cfg, eef_link_name, reach_std, phase_weights, phase_params
    )


def staged_bimanual_reach_min_reward(
    env: ManagerBasedRLEnv,
    std: float,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Stage-gated bimanual reach min reward."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return bimanual_reach_min_reward(
        env, std, left_eef_link_name, right_eef_link_name,
        left_object_cfg, right_object_cfg, phase_weights,
        left_phase_params, right_phase_params
    )


def staged_bimanual_phase_lag_penalty(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    active_stages: list[int],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Stage-gated bimanual phase lag penalty."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return bimanual_phase_lag_penalty(
        env, left_eef_link_name, right_eef_link_name,
        left_object_cfg, right_object_cfg,
        left_phase_params, right_phase_params
    )


def staged_bimanual_grasp_and_reward(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Stage-gated bimanual grasp AND reward."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return bimanual_grasp_and_reward(
        env, left_eef_link_name, right_eef_link_name,
        left_object_cfg, right_object_cfg, phase_weights,
        left_phase_params, right_phase_params
    )


def staged_bimanual_grasp_xor_penalty(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    left_object_cfg: SceneEntityCfg,
    right_object_cfg: SceneEntityCfg,
    active_stages: list[int],
    phase_weights: list[float],
    left_phase_params: dict,
    right_phase_params: dict,
) -> torch.Tensor:
    """Stage-gated bimanual grasp XOR penalty."""
    if not _is_stage_active(env, active_stages):
        return torch.zeros(env.num_envs, device=env.device)
    return bimanual_grasp_xor_penalty(
        env, left_eef_link_name, right_eef_link_name,
        left_object_cfg, right_object_cfg, phase_weights,
        left_phase_params, right_phase_params
    )


# ============================================================
# CURRICULUM STAGE OBSERVATION AND DIAGNOSTIC
# ============================================================

def curriculum_stage_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current curriculum stage as observation (normalized to [-1, 1])."""
    stage = _get_curriculum_stage(env)
    # Normalize: 0 -> -1, 1 -> 0, 2 -> 1
    normalized = (stage - 1.0)
    return torch.full((env.num_envs, 1), normalized, device=env.device)


def curriculum_stage_diagnostic(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Diagnostic: current curriculum stage (0, 1, or 2). Use with weight=0."""
    stage = _get_curriculum_stage(env)
    return torch.full((env.num_envs,), float(stage), device=env.device)


# ============================================================
# CURRICULUM STAGE ADVANCEMENT
# ============================================================

def advance_curriculum_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    left_phase_threshold: float,
    right_phase_threshold: float,
    success_rate_threshold: float,
    min_steps_per_stage: int,
) -> float:
    """Curriculum term to advance stage based on phase progression.

    Args:
        env: The RL environment.
        env_ids: Environment indices (required by Isaac Lab curriculum manager).
        left_phase_threshold: Left arm phase threshold for Stage 0 -> 1.
        right_phase_threshold: Right arm phase threshold for Stage 1 -> 2.
        success_rate_threshold: Required success rate to advance.
        min_steps_per_stage: Minimum steps before stage advancement.

    Returns:
        Current curriculum stage as float (scalar).

    Stage 0 -> 1: When left arm phase >= left_phase_threshold
    Stage 1 -> 2: When right arm phase >= right_phase_threshold
    """
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return 0.0

    current_stage = getattr(cfg, "curriculum_stage", 0)

    # Track steps in current stage
    stage_step_attr = "_curriculum_stage_steps"
    if not hasattr(env, stage_step_attr):
        setattr(env, stage_step_attr, 0)

    stage_steps = getattr(env, stage_step_attr)
    setattr(env, stage_step_attr, stage_steps + 1)

    if stage_steps < min_steps_per_stage:
        return float(current_stage)

    # Get phase values
    left_phase = getattr(env, "grasp2g_phase_left", torch.zeros(env.num_envs, device=env.device))
    right_phase = getattr(env, "grasp2g_phase_right", torch.zeros(env.num_envs, device=env.device))

    if current_stage == 0:
        # Stage 0 -> 1: Check left arm success rate
        left_success = (left_phase >= left_phase_threshold).float().mean().item()
        if left_success >= success_rate_threshold:
            cfg.curriculum_stage = 1
            setattr(env, stage_step_attr, 0)
            print(f"[CURRICULUM] Advanced to Stage 1 (RIGHT_ONLY) after {stage_steps} steps. Left success rate: {left_success:.2%}")
            return 1.0

    elif current_stage == 1:
        # Stage 1 -> 2: Check right arm success rate
        right_success = (right_phase >= right_phase_threshold).float().mean().item()
        if right_success >= success_rate_threshold:
            cfg.curriculum_stage = 2
            setattr(env, stage_step_attr, 0)
            print(f"[CURRICULUM] Advanced to Stage 2 (BIMANUAL) after {stage_steps} steps. Right success rate: {right_success:.2%}")
            return 2.0

    return float(current_stage)
