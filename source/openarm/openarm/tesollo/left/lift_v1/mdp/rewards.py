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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w)
    return obj_pos_b


def _get_episode_initial_object_xy(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    cache_attr: str,
) -> torch.Tensor:
    """Track per-episode initial object XY in world frame."""
    current_xy = obj.data.root_pos_w[:, :2]
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, current_xy.clone())
    initial_xy = getattr(env, cache_attr)
    reset_mask = (env.episode_length_buf == 0).squeeze(-1)
    initial_xy[reset_mask] = current_xy[reset_mask]
    setattr(env, cache_attr, initial_xy)
    return initial_xy


def _compute_grasp_target_pos_w(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    ee_pos_w: torch.Tensor,
    use_dynamic_z: bool,
    dynamic_z_state_attr: str | None = None,
) -> torch.Tensor:
    """Compute grasp target from grasp2g offset in world frame."""
    cfg = getattr(env, "cfg", None)
    obj_pos_w = obj.data.root_pos_w.clone()
    base_offset = getattr(cfg, "grasp2g_target_offset", (0.0, 0.0, 0.08))

    if not (isinstance(base_offset, (list, tuple)) and len(base_offset) == 3):
        return obj_pos_w

    offset_xy_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_xy_local[:, 0] = base_offset[0]
    offset_xy_local[:, 1] = base_offset[1]
    offset_xy_w = quat_apply(obj.data.root_quat_w, offset_xy_local)
    target_xy_w = obj_pos_w[:, :2] + offset_xy_w[:, :2]

    z_value: torch.Tensor | float = base_offset[2]
    if use_dynamic_z:
        # Use XY distance to the offset target (not cup center) for z transition.
        xy_dist = torch.norm(ee_pos_w[:, :2] - target_xy_w, dim=1)
        z_high = float(getattr(cfg, "reach_dynamic_z_high", 0.2))
        x_hi = float(getattr(cfg, "reach_dynamic_xy_hi", 0.06))
        x_lo = float(getattr(cfg, "reach_dynamic_xy_lo", 0.015))
        x_gate = float(getattr(cfg, "reach_dynamic_xy_gate", x_hi))
        x_gate = min(x_gate, x_hi)
        x_gate = max(x_gate, x_lo + 1e-6)
        x_hi = max(x_hi, x_lo + 1e-6)

        # Keep high-Z approach until XY is close enough, then start descending.
        u = torch.clamp((x_gate - xy_dist) / (x_gate - x_lo), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        z_value_raw = z_high * (1.0 - u) + float(base_offset[2]) * u

        # Limit per-step Z descent speed to avoid sudden dives toward the cup.
        descent_rate = float(getattr(cfg, "reach_dynamic_z_descent_rate", 0.0))
        if descent_rate > 0.0 and dynamic_z_state_attr is not None:
            if not hasattr(env, dynamic_z_state_attr):
                setattr(env, dynamic_z_state_attr, torch.full_like(z_value_raw, z_high))
            z_prev = getattr(env, dynamic_z_state_attr)
            reset_mask = (env.episode_length_buf == 0).squeeze(-1)
            z_prev[reset_mask] = z_high

            z_floor = z_prev - descent_rate
            z_value = torch.maximum(z_value_raw, z_floor)
            setattr(env, dynamic_z_state_attr, z_value)
        else:
            z_value = z_value_raw

    offset_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_local[:, 0] = base_offset[0]
    offset_local[:, 1] = base_offset[1]
    if isinstance(z_value, torch.Tensor):
        offset_local[:, 2] = z_value
    else:
        offset_local[:, 2] = float(z_value)
    offset_w = quat_apply(obj.data.root_quat_w, offset_local)
    return obj_pos_w + offset_w


# =============================================================================
# DexPour-style Binary Triggers (λ, μ, ν, ρ)
# =============================================================================

def _get_fingertip_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all 5 fingertip positions in world frame.

    Returns: (num_envs, 5, 3) tensor of fingertip positions
    """
    robot = env.scene["robot"]

    # Fingertip info: (body_name, offset_axis, offset_value)
    tip_info = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363),  # thumb
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),   # index
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),   # middle
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),   # ring
        ("tesollo_left_ll_dg_5_4", "z", 0.0363),   # pinky
    ]

    tip_positions = []
    for body_name, offset_axis, offset_val in tip_info:
        if body_name in robot.data.body_names:
            body_idx = robot.data.body_names.index(body_name)
            link_pos = robot.data.body_pos_w[:, body_idx]
            link_quat = robot.data.body_quat_w[:, body_idx]

            if offset_axis == "y":
                local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
            else:  # "z"
                local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

            world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
            tip_pos = link_pos + world_offset
            tip_positions.append(tip_pos)

    return torch.stack(tip_positions, dim=1)  # (num_envs, 5, 3)


def _get_fingertip_world_position(
    env: ManagerBasedRLEnv,
    body_name: str,
    offset_axis: str,
    offset_val: float,
) -> torch.Tensor | None:
    """Return fingertip world position using link pose + local offset."""
    robot = env.scene["robot"]
    if body_name not in robot.data.body_names:
        return None

    body_idx = robot.data.body_names.index(body_name)
    link_pos = robot.data.body_pos_w[:, body_idx]
    link_quat = robot.data.body_quat_w[:, body_idx]

    local_offset = torch.zeros(3, device=env.device, dtype=link_pos.dtype)
    if offset_axis == "x":
        local_offset[0] = offset_val
    elif offset_axis == "y":
        local_offset[1] = offset_val
    else:
        local_offset[2] = offset_val

    world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
    return link_pos + world_offset


def _finger_surface_contact_gate(
    env: ManagerBasedRLEnv,
    body_name: str,
    offset_axis: str,
    offset_val: float,
    object_cfg: SceneEntityCfg,
    cup_radius: float = 0.045,
    radial_std: float = 0.015,
    cup_height: float = 0.109,
) -> torch.Tensor:
    """Continuous contact gate in [0, 1] based on radial proximity and valid Z range."""
    tip_pos = _get_fingertip_world_position(env, body_name, offset_axis, offset_val)
    if tip_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]
    cup_z = obj.data.root_pos_w[:, 2]
    tip_xy = tip_pos[:, :2]
    tip_z = tip_pos[:, 2]

    radial_dist = torch.norm(tip_xy - cup_xy, dim=1)
    radial_error = torch.abs(radial_dist - cup_radius)
    radial_gate = 1.0 - torch.tanh(radial_error / max(radial_std, 1e-6))

    z_gate = ((tip_z >= cup_z) & (tip_z <= cup_z + cup_height)).float()
    return radial_gate * z_gate


def _thumb_opposition_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward thumb opposing the average direction of the other fingers in XY."""
    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]

    thumb_tip = _get_fingertip_world_position(env, "tesollo_left_ll_dg_1_4", "y", -0.0363)
    if thumb_tip is None:
        return torch.zeros(env.num_envs, device=env.device)

    others = []
    for body_name, axis, offset in [
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]:
        tip = _get_fingertip_world_position(env, body_name, axis, offset)
        if tip is not None:
            others.append(tip[:, :2] - cup_xy)

    if not others:
        return torch.zeros(env.num_envs, device=env.device)

    thumb_vec = thumb_tip[:, :2] - cup_xy
    others_mean = torch.stack(others, dim=0).mean(dim=0)
    thumb_unit = thumb_vec / (torch.norm(thumb_vec, dim=1, keepdim=True) + 1e-6)
    others_unit = others_mean / (torch.norm(others_mean, dim=1, keepdim=True) + 1e-6)
    dot = torch.sum(thumb_unit * others_unit, dim=1)
    return torch.clamp(-dot, min=0.0, max=1.0)


def _maybe_log_grasp_quality(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    lambda_trigger: torch.Tensor,
    grasp_gate: torch.Tensor,
    thumb_contact: torch.Tensor,
    pinky_contact: torch.Tensor,
    thumb_opposition: torch.Tensor,
) -> None:
    """Periodic debug log for diagnosing grasp quality behavior."""
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_grasp_quality", False):
        return

    step_count = int(getattr(env, "common_step_counter", -1))
    interval = int(getattr(cfg, "debug_grasp_quality_interval", 50))
    if step_count < 0 or (interval > 1 and step_count % interval != 0):
        return

    last_step = int(getattr(env, "_debug_grasp_quality_last_step", -2))
    if last_step == step_count:
        return
    env._debug_grasp_quality_last_step = step_count

    obj: RigidObject = env.scene[object_cfg.name]
    current_xy = obj.data.root_pos_w[:, :2]
    initial_xy = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")
    displacement = torch.norm(current_xy - initial_xy, dim=1)
    contacts = _count_finger_contacts(env, object_cfg)

    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    ee_dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    print(
        f"[Step {step_count}] grasp_quality | "
        f"env0: λ={lambda_trigger[0].item():.2f}, g={grasp_gate[0].item():.2f}, "
        f"contacts={contacts[0].item():.1f}, thumb_c={thumb_contact[0].item():.2f}, "
        f"pinky_c={pinky_contact[0].item():.2f}, opp={thumb_opposition[0].item():.2f}, "
        f"disp={displacement[0].item():.4f}, ee_dist={ee_dist[0].item():.4f} | "
        f"mean: λ={lambda_trigger.mean().item():.2f}, g={grasp_gate.mean().item():.2f}, "
        f"contacts={contacts.mean().item():.2f}, thumb_c={thumb_contact.mean().item():.2f}, "
        f"pinky_c={pinky_contact.mean().item():.2f}, opp={thumb_opposition.mean().item():.2f}, "
        f"disp={displacement.mean().item():.4f}, ee_dist={ee_dist.mean().item():.4f}"
    )


def _count_finger_contacts(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    cup_radius: float = 0.045,
    contact_threshold: float = 0.02,
) -> torch.Tensor:
    """Count number of fingertips in contact with cup surface.

    Contact = fingertip-to-cup-surface distance < contact_threshold

    Returns: (num_envs,) tensor of contact counts (0-5)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos = obj.data.root_pos_w[:, :3]  # (num_envs, 3)

    tip_positions = _get_fingertip_positions(env)  # (num_envs, 5, 3)

    # Distance from each tip to cup center (XY plane for cylindrical cup)
    tip_xy = tip_positions[:, :, :2]  # (num_envs, 5, 2)
    cup_xy = cup_pos[:, :2].unsqueeze(1)  # (num_envs, 1, 2)

    dist_to_center_xy = torch.norm(tip_xy - cup_xy, dim=2)  # (num_envs, 5)
    dist_to_surface = dist_to_center_xy - cup_radius  # distance to cup surface

    # Also check Z height is within cup (roughly)
    tip_z = tip_positions[:, :, 2]  # (num_envs, 5)
    cup_z = cup_pos[:, 2].unsqueeze(1)  # (num_envs, 1)
    cup_height = 0.109  # cup height at scale (1,1,1.2)
    z_valid = (tip_z >= cup_z) & (tip_z <= cup_z + cup_height)

    # Contact: surface distance < threshold AND valid Z
    contacts = (dist_to_surface < contact_threshold) & z_valid
    num_contacts = contacts.float().sum(dim=1)  # (num_envs,)

    return num_contacts


def _approach_trigger(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
    d_approach: float = 0.05,
) -> torch.Tensor:
    """λ: Approach trigger - EE is close to grasp target.

    Returns: (num_envs,) binary tensor (0 or 1)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    return (dist < d_approach).float()


def _grasp_trigger(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
    min_contacts: int = 3,
    cup_radius: float = 0.045,
    contact_threshold: float = 0.02,
) -> torch.Tensor:
    """μ: Grasp trigger - enough fingers in contact AND approach complete.

    μ = λ × (num_contacts >= min_contacts)

    Returns: (num_envs,) binary tensor (0 or 1)
    """
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    num_contacts = _count_finger_contacts(env, object_cfg, cup_radius, contact_threshold)

    contact_satisfied = (num_contacts >= min_contacts).float()

    return lambda_trigger * contact_satisfied


def _get_episode_initial_object_z(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    cache_attr: str,
) -> torch.Tensor:
    """Track per-episode initial object Z in world frame."""
    current_z = obj.data.root_pos_w[:, 2]
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, current_z.clone())
    initial_z = getattr(env, cache_attr)

    ep_len = env.episode_length_buf.squeeze(-1) if env.episode_length_buf.dim() > 1 else env.episode_length_buf
    reset_mask = (ep_len <= 1)
    initial_z[reset_mask] = current_z[reset_mask]
    setattr(env, cache_attr, initial_z)
    return initial_z


def _lift_trigger(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
    h_lift: float = 0.04,
) -> torch.Tensor:
    """ν: Lift trigger - cup lifted above initial height + threshold AND grasp complete.

    ν = μ × (cup_height >= initial_cup_z + h_lift)

    Returns: (num_envs,) binary tensor (0 or 1)
    """
    mu_trigger = _grasp_trigger(env, object_cfg, eef_link_name)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_height = obj.data.root_pos_w[:, 2]
    initial_z = _get_episode_initial_object_z(env, obj, "_cup_initial_z_w_left")

    lift_satisfied = (cup_height >= initial_z + h_lift).float()

    return mu_trigger * lift_satisfied


def _track_trigger(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
    command_name: str = "object_pose",
    d_track: float = 0.15,
) -> torch.Tensor:
    """ρ: Track trigger - cup close to target AND lift complete.

    ρ = ν × (dist_to_goal < d_track)

    Returns: (num_envs,) binary tensor (0 or 1)
    """
    nu_trigger = _lift_trigger(env, object_cfg, eef_link_name)

    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    dist_to_goal = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    track_satisfied = (dist_to_goal < d_track).float()

    return nu_trigger * track_satisfied


# Debug function for binary triggers
def _debug_triggers(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
) -> None:
    """Print trigger states for debugging (env 0 only, every 50 steps)."""
    cfg = getattr(env, "cfg", None)
    debug_enabled = getattr(cfg, "debug_triggers", True)
    step_count = int(getattr(env, "common_step_counter", -1))

    if not debug_enabled or step_count % 50 != 0:
        return

    lambda_t = _approach_trigger(env, object_cfg, eef_link_name)
    num_contacts = _count_finger_contacts(env, object_cfg)
    mu_t = _grasp_trigger(env, object_cfg, eef_link_name)
    nu_t = _lift_trigger(env, object_cfg, eef_link_name)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_z = obj.data.root_pos_w[0, 2].item()

    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    ee_dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    print(f"[Step {step_count}] λ={lambda_t[0].item():.0f} | "
          f"Contacts={num_contacts[0].item():.0f}/4 | "
          f"μ={mu_t[0].item():.0f} | "
          f"CupZ={cup_z:.3f}m | "
          f"ν={nu_t[0].item():.0f} | "
          f"ee_dist={ee_dist[0].item():.4f}m (mean={ee_dist.mean().item():.4f})")


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.

    Uses dynamic z offset: starts high (0.15) to approach from above,
    then lowers to grasp position (0.08) as xy alignment improves.
    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # Get EE position first for dynamic offset calculation
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env,
        obj,
        ee_pos_w,
        use_dynamic_z=True,
        dynamic_z_state_attr="_reach_dynamic_z_prev_left",
    )
    _maybe_visualize_approach_target_all(env, target_pos_w, obj.data.root_quat_w, marker_attr="_debug_approach_target_left")

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    # Suppress reaching reward when cup is pushed in XY during approach.
    cfg = getattr(env, "cfg", None)
    disp_free = float(getattr(cfg, "reach_displacement_free_threshold", 0.005))
    disp_scale = float(getattr(cfg, "reach_displacement_suppress_scale", 0.01))
    current_xy = obj.data.root_pos_w[:, :2]
    initial_xy = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")
    displacement_xy = torch.norm(current_xy - initial_xy, dim=1)
    displacement_excess = torch.clamp(displacement_xy - disp_free, min=0.0)
    if disp_scale > 0.0:
        reach_reward = reach_reward * torch.exp(-displacement_excess / disp_scale)

    # DexPour: reaching reward stays active at all phases (no (1-λ) gating).
    # Debug triggers
    _debug_triggers(env, object_cfg, eef_link_name)

    return reach_reward


def object_ee_distance_fine(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Fine-grained reaching reward toward the static grasp target (use_dynamic_z=False).

    Provides gradient to guide the EE from the dynamic target position
    all the way down to the actual grasp position (Z=offset[2]).
    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env, obj, ee_pos_w, use_dynamic_z=False,
    )

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    # DexPour: reaching reward stays active at all phases (no (1-λ) gating).
    return reach_reward


def ee_descent_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.04,
    target_z_offset: float = 0.04,
    xy_proximity_std: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward EE descending to grasp height after approach is complete.

    After approach is complete (λ=1), guide EE to descend further
    to target_z_offset (default 0.04) for proper grasping.
    DexPour: Active when λ=1 (approach complete).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    ee_z = ee_pos_w[:, 2]

    # Target: cup_z + target_z_offset (lower than reaching target of 0.08)
    cup_z = obj.data.root_pos_w[:, 2]
    target_z = cup_z + target_z_offset

    z_error = torch.abs(ee_z - target_z)
    descent_reward = 1.0 - torch.tanh(z_error / std)

    # XY proximity: reward decays if EE is far from cup in XY
    cup_pos_xy = obj.data.root_pos_w[:, :2]
    xy_dist = torch.norm(ee_pos_w[:, :2] - cup_pos_xy, dim=1)
    xy_proximity = torch.exp(-xy_dist / xy_proximity_std)

    # DexPour: Active when λ=1 AND μ=0 (approach complete, grasp NOT yet achieved)
    # When μ=1 (grasp achieved), descent force disabled → arm free to lift
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    mu_trigger = _grasp_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * (1.0 - mu_trigger) * descent_reward * xy_proximity


def _maybe_visualize_approach_target_all(
    env: ManagerBasedRLEnv,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    marker_attr: str,
) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_approach_target_vis", True):
        return

    interval = int(getattr(cfg, "debug_approach_target_vis_interval", 10))
    step_count = int(getattr(env, "common_step_counter", 0))
    if interval > 1 and (step_count % interval) != 0:
        return

    if not hasattr(env, marker_attr):
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/ApproachTargetLeft")
        marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
        marker = VisualizationMarkers(marker_cfg)
        marker.set_visibility(True)
        setattr(env, marker_attr, marker)

    marker = getattr(env, marker_attr)
    marker.visualize(target_pos_w, target_quat_w)


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

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    eef_axes = [quat_apply(eef_quat, x_axis), quat_apply(eef_quat, y_axis), quat_apply(eef_quat, z_axis)]
    obj_axes = [quat_apply(object_quat, x_axis), quat_apply(object_quat, y_axis), quat_apply(object_quat, z_axis)]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)
    return max_align


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


def eef_z_perpendicular_object_z(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str = "ll_dg_ee",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward 90-degree alignment between EE +Z axis and object +Z axis.

    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat.dtype).repeat(env.num_envs, 1)
    ee_z = quat_apply(eef_quat, z_axis)
    obj_z = quat_apply(object_quat, z_axis)

    cos_theta = torch.sum(ee_z * obj_z, dim=1).clamp(-1.0, 1.0)
    error = torch.abs(cos_theta)
    orientation_reward = 1 - torch.tanh(error / std)

    # DexPour: orientation reward stays active at all phases (no (1-λ) gating).
    return orientation_reward


def _is_reaching_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    reach_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if EE has reached the grasp position (z_low offset near object).

    Returns a boolean mask (float 0/1) per environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return (dist < reach_threshold).float()


def _is_reaching_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate progression after reaching is maintained for multiple consecutive steps.

    Once reaching is complete (counter >= hold_steps), it stays complete for the rest
    of the episode. This allows EE to descend further for grasping without losing
    the reaching completion status.
    """
    cfg = getattr(env, "cfg", None)
    reach_threshold = float(getattr(cfg, "reach_switch_threshold", 0.01))
    hold_steps = int(getattr(cfg, "reach_switch_hold_steps", 10))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=reach_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_reach_hold_counter_left"):
        env._reach_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    if not hasattr(env, "_reach_latched_left"):
        env._reach_latched_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    counter = env._reach_hold_counter_left
    latched = env._reach_latched_left

    # Update once per sim step even if multiple reward terms query this gate.
    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_reach_hold_counter_left_last_step"):
        env._reach_hold_counter_left_last_step = -2
    if env._reach_hold_counter_left_last_step != step_count:
        # Reset counter and latched state at episode boundary.
        # Use episode_length_buf == 1 to detect first step of new episode
        ep_len = env.episode_length_buf.squeeze(-1) if env.episode_length_buf.dim() > 1 else env.episode_length_buf
        reset_mask = (ep_len <= 1)

        # Also reset in first few global steps to ensure clean start
        if step_count < 10:
            counter[:] = 0
            latched[:] = False
        else:
            counter[reset_mask] = 0
            latched[reset_mask] = False

        # Debug: print EE distance at episode start for env 0
        if ep_len[0].item() == 1:
            obj_dbg = env.scene[object_cfg.name]
            eef_idx_dbg = env.scene["robot"].data.body_names.index(eef_link_name)
            ee_pos_dbg = env.scene["robot"].data.body_pos_w[:, eef_idx_dbg]
            target_dbg = _compute_grasp_target_pos_w(env, obj_dbg, ee_pos_dbg, use_dynamic_z=False)
            dist_dbg = torch.norm(target_dbg - ee_pos_dbg, dim=1)
            print(f"[Step {step_count}] EPISODE START: dist={dist_dbg[0].item():.4f}m (threshold={reach_threshold:.3f}m)")

        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._reach_hold_counter_left = counter

        # Once counter reaches hold_steps, latch it (stays True until episode reset)
        newly_latched = (counter >= hold_steps) & (~latched)

        # Debug: print when latched becomes True for env 0
        if newly_latched[0].item():
            obj = env.scene[object_cfg.name]
            eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
            ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
            target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
            dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
            print(f"[Step {step_count}] *** LATCHED! *** dist={dist[0].item():.4f}m, EpLen={ep_len[0].item():.0f}")

        latched = latched | newly_latched
        env._reach_latched_left = latched

        env._reach_hold_counter_left_last_step = step_count

        # Debug output (every 50 steps, env 0 only) - inside step check to print once per step
        debug_enabled = getattr(cfg, "debug_reaching", True)
        if debug_enabled and step_count % 50 == 0:
            obj = env.scene[object_cfg.name]
            eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
            ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
            target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
            dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

            print(f"[Step {step_count}] EE-Target dist: {dist[0].item():.4f}m | "
                  f"Threshold: {reach_threshold:.3f}m | "
                  f"Reached: {reached_now[0].item():.0f} | "
                  f"Counter: {counter[0].item()}/{hold_steps} | "
                  f"Reach Active: {latched[0].item():.0f} | "
                  f"EpLen: {ep_len[0].item():.0f}")

    # Return latched state (once complete, stays complete until episode reset)
    result = latched.to(dtype=reached_now.dtype)

    return result


def _reaching_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous [0, 1] gate based on EE distance to static grasp target."""
    cfg = getattr(env, "cfg", None)
    near = float(getattr(cfg, "reach_soft_gate_near", 0.02))
    far = float(getattr(cfg, "reach_soft_gate_far", 0.08))
    far = max(far, near + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    gate = torch.clamp((far - dist) / (far - near), 0.0, 1.0)
    return gate


def _reaching_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Combine soft and stable gates for robust phase transition."""
    stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    soft = _reaching_soft_gate(env, object_cfg, eef_link_name)
    soft_relaxed = torch.clamp(soft * 1.2, 0.0, 1.0)
    return torch.maximum(stable, soft_relaxed)


def _is_grasp_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate finger-closing progression after very-close reaching is maintained."""
    cfg = getattr(env, "cfg", None)
    grasp_threshold = float(getattr(cfg, "grasp_switch_threshold", 0.025))
    hold_steps = int(getattr(cfg, "grasp_switch_hold_steps", 2))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=grasp_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_grasp_hold_counter_left"):
        env._grasp_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._grasp_hold_counter_left

    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_grasp_hold_counter_left_last_step"):
        env._grasp_hold_counter_left_last_step = -2
    if env._grasp_hold_counter_left_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0
        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._grasp_hold_counter_left = counter
        env._grasp_hold_counter_left_last_step = step_count

    return (counter >= hold_steps).to(dtype=reached_now.dtype)


def _grasp_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous [0, 1] gate for grasp activation (closer band than reaching gate)."""
    cfg = getattr(env, "cfg", None)
    near = float(getattr(cfg, "grasp_soft_gate_near", 0.012))
    far = float(getattr(cfg, "grasp_soft_gate_far", 0.03))
    far = max(far, near + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    gate = torch.clamp((far - dist) / (far - near), 0.0, 1.0)
    return gate


def _grasp_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Combine soft and stable gates for robust finger-closing transition."""
    stable = _is_grasp_stably_complete(env, object_cfg, eef_link_name)
    soft = _grasp_soft_gate(env, object_cfg, eef_link_name)
    soft_relaxed = torch.clamp(soft * 1.2, 0.0, 1.0)
    return torch.maximum(stable, soft_relaxed)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Binary reward if object is lifted above initial height + minimal_height.

    DexPour: Active when μ=1 (grasp complete with contacts).
    Returns 1.0 when cup lifted by minimal_height from initial position AND grasp is complete.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_height = obj.data.root_pos_w[:, 2]
    initial_z = _get_episode_initial_object_z(env, obj, "_cup_initial_z_w_left")

    mu_trigger = _grasp_trigger(env, object_cfg, eef_link_name)
    return mu_trigger * (cup_height > initial_z + minimal_height).float()


def cup_lift_progress_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Continuous reward for cup lifting once grasp is complete (μ=1).

    tanh kernel: maximum gradient at delta=0 → immediately rewards any upward
    cup movement once μ=1. Does not require cup to already be lifted (unlike
    object_is_lifted which is binary at 4cm).
    DexPour: Active when μ=1 (grasp complete).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_z = obj.data.root_pos_w[:, 2]
    initial_z = _get_episode_initial_object_z(env, obj, "_cup_initial_z_w_left")

    lift_delta = torch.clamp(cup_z - initial_z, min=0.0)
    lift_reward = torch.tanh(lift_delta / std)

    mu_trigger = _grasp_trigger(env, object_cfg, eef_link_name)
    return mu_trigger * lift_reward


def _is_lifting_sustained(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_seconds: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Check if object has been lifted above minimal_height for hold_seconds.

    Returns 1.0 if lifting has been sustained, 0.0 otherwise.
    Resets at episode boundary.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    is_lifted_now = (obj.data.root_pos_w[:, 2] > minimal_height).to(dtype=torch.int64)

    # Calculate hold_steps from hold_seconds
    cfg = getattr(env, "cfg", None)
    dt = float(getattr(cfg, "sim", {}).get("dt", 0.01) if isinstance(getattr(cfg, "sim", None), dict) else 0.01)
    decimation = int(getattr(cfg, "decimation", 4))
    step_time = dt * decimation  # time per step
    hold_steps = max(1, int(hold_seconds / step_time))

    if not hasattr(env, "_lift_hold_counter_left"):
        env._lift_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    if not hasattr(env, "_lift_latched_left"):
        env._lift_latched_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    counter = env._lift_hold_counter_left
    latched = env._lift_latched_left

    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_lift_hold_counter_left_last_step"):
        env._lift_hold_counter_left_last_step = -2
    if env._lift_hold_counter_left_last_step != step_count:
        # Reset at episode boundary
        ep_len = env.episode_length_buf.squeeze(-1) if env.episode_length_buf.dim() > 1 else env.episode_length_buf
        reset_mask = (ep_len <= 1)

        # Also reset in first few global steps to ensure clean start
        if step_count < 10:
            counter[:] = 0
            latched[:] = False
        else:
            counter[reset_mask] = 0
            latched[reset_mask] = False

        counter = torch.where(is_lifted_now > 0, counter + 1, torch.zeros_like(counter))
        env._lift_hold_counter_left = counter

        # Once sustained, latch it
        newly_latched = (counter >= hold_steps) & (~latched)
        latched = latched | newly_latched
        env._lift_latched_left = latched

        env._lift_hold_counter_left_last_step = step_count

    return latched.float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    hold_seconds: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward tracking the goal pose using tanh-kernel.

    DexPour: Active when ν=1 (lift complete - cup height >= threshold AND grasp complete).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)

    # DexPour: Active when ν=1 (lift complete)
    nu_trigger = _lift_trigger(env, object_cfg, eef_link_name, h_lift=minimal_height)
    return nu_trigger * (1 - torch.tanh(distance / std))


def object_goal_settle_reward(
    env: ManagerBasedRLEnv,
    goal_std: float,
    lin_vel_std: float,
    ang_vel_std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward stable settling near goal by preferring low object velocities.

    Notes:
    - No observation-space change required (uses simulator state only).
    - Active after lift trigger (nu) to avoid over-constraining early approach.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    near_goal = 1.0 - torch.tanh(distance / max(goal_std, 1e-6))

    lin_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    ang_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    lin_stability = torch.exp(-lin_speed / max(lin_vel_std, 1e-6))
    ang_stability = torch.exp(-ang_speed / max(ang_vel_std, 1e-6))
    settle = lin_stability * ang_stability

    nu_trigger = _lift_trigger(env, object_cfg, eef_link_name, h_lift=minimal_height)
    return nu_trigger * near_goal * settle


def object_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    threshold: float = 0.02,
) -> torch.Tensor:
    """Penalize object movement from initial position (XY only).

    Uses a non-linear penalty to strongly discourage cup pushing.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cfg = getattr(env, "cfg", None)
    scale = float(getattr(cfg, "displacement_penalty_scale", 0.02))
    power = float(getattr(cfg, "displacement_penalty_power", 2.0))
    gate_mix = float(getattr(cfg, "displacement_penalty_gate_mix", 0.5))
    gate_mix = max(0.0, min(1.0, gate_mix))

    current_pos = obj.data.root_pos_w[:, :2]
    initial_pos = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")

    displacement = torch.norm(current_pos - initial_pos, dim=1)
    excess = torch.clamp(displacement - threshold, min=0.0)
    penalty = torch.pow(excess / max(scale, 1e-6), power)

    # Keep some baseline penalty early, and strengthen as grasp phase progresses.
    grasp_gate = _grasp_progress_gate(env, object_cfg, eef_link_name="ll_dg_ee")
    penalty = penalty * ((1.0 - gate_mix) + gate_mix * grasp_gate)
    # Disable displacement penalty once grasp is established (μ=1).
    mu_trigger = _grasp_trigger(env, object_cfg, eef_link_name="ll_dg_ee")
    penalty = penalty * (1.0 - mu_trigger)

    return penalty


def finger_normal_range_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize left thumb+pinky joints going outside their normal curl range.

    Returns total violation amount (positive). Use negative weight in config.
    Joints outside the normal range (e.g. bending backward) accumulate violation in radians.
    """
    robot = env.scene["robot"]

    # Left hand normal ranges (from user-confirmed curl directions)
    # 1_1 (thumb spread) excluded - full range is acceptable
    _RANGES = {
        "lj_dg_1_2": (0.0, 1.571),      # positive = curl
        "lj_dg_1_3": (-1.571, 0.0),      # negative = curl
        "lj_dg_1_4": (-1.571, 0.0),      # negative = curl
        "lj_dg_5_1": (-0.1, 0.1),        # should stay near 0
        "lj_dg_5_2": (-0.611, 0.05),     # 0.0 ideal, slight positive tolerance
        "lj_dg_5_3": (0.0, 1.571),       # positive = curl
        "lj_dg_5_4": (0.0, 1.571),       # positive = curl
    }

    total_violation = torch.zeros(env.num_envs, device=env.device)

    for joint_name, (lo, hi) in _RANGES.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_violation += torch.clamp(lo - pos, min=0.0) + torch.clamp(pos - hi, min=0.0)

    return total_violation


def thumb_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward thumb (finger 1) staying near initial open pose during reaching.

    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    robot = env.scene["robot"]

    # Thumb target = open pose
    _TARGETS = {
        "lj_dg_1_2": 1.571,    # max open
        "lj_dg_1_3": 0.0,
        "lj_dg_1_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    # DexPour: Active when λ=0 (before approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def pinky_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward pinky (finger 5) staying near initial open pose during reaching.

    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    robot = env.scene["robot"]

    # Pinky target = open pose
    _TARGETS = {
        "lj_dg_5_3": 0.0,
        "lj_dg_5_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    # DexPour: Active when λ=0 (before approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def finger_contact_reward(
    env: ManagerBasedRLEnv,
    cup_radius: float = 0.045,
    contact_threshold: float = 0.02,
    cup_height: float = 0.109,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward for fingertips in contact with cup surface.

    Uses XY surface distance (cylindrical) + Z range check,
    consistent with _count_finger_contacts / _grasp_trigger.

    Contact = XY distance to surface < contact_threshold AND tip Z within cup height.
    Returns: contact_count / 5 gated by λ.

    DexPour: Active when λ=1 (approach complete).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]  # (num_envs, 2)
    cup_z = obj.data.root_pos_w[:, 2]    # (num_envs,)

    _TIPS = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363),
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]

    contact_count = torch.zeros(env.num_envs, device=env.device)
    tip_count = 0

    for body_name, axis, offset in _TIPS:
        tip = _get_fingertip_world_position(env, body_name, axis, offset)
        if tip is None:
            continue
        tip_count += 1

        # XY surface distance (cylindrical)
        radial_dist = torch.norm(tip[:, :2] - cup_xy, dim=1)
        surface_dist = radial_dist - cup_radius

        # Z range check
        z_valid = (tip[:, 2] >= cup_z) & (tip[:, 2] <= cup_z + cup_height)

        # Contact: near surface AND valid Z
        in_contact = (surface_dist.abs() < contact_threshold) & z_valid
        contact_count += in_contact.float()

    if tip_count == 0:
        return torch.zeros(env.num_envs, device=env.device)

    contact_ratio = contact_count / float(tip_count)

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * contact_ratio


def thumb_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward thumb tip approaching cup center in XY (task-space).

    Instead of rewarding fixed joint targets (which curl the thumb in a
    fixed direction regardless of cup position), this rewards the thumb
    tip getting closer to the cup center.  The policy discovers the right
    joint configuration through the task-space gradient.

    DexPour: Active when λ=1 (approach complete).
    """
    thumb_tip = _get_fingertip_world_position(
        env, "tesollo_left_ll_dg_1_4", "y", -0.0363
    )
    if thumb_tip is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]
    tip_xy = thumb_tip[:, :2]

    # Surface approach: thumb tip → cup surface (not center)
    cup_radius = 0.045  # actual cup outer radius at scale (1,1,1.2)
    radial_dist = torch.norm(tip_xy - cup_xy, dim=1)
    surface_dist = torch.abs(radial_dist - cup_radius)
    approach_reward = 1.0 - torch.tanh(surface_dist / std)

    # Penetration penalty: penalize thumb being inside the cup
    penetration = torch.clamp(cup_radius - radial_dist, min=0.0)
    approach_reward = approach_reward - torch.tanh(penetration / 0.01)

    # Z gate: only reward approach when thumb is at or below finger 2 Z
    finger2_tip = _get_fingertip_world_position(
        env, "tesollo_left_ll_dg_2_4", "z", 0.0255
    )
    if finger2_tip is not None:
        z_above = torch.clamp(thumb_tip[:, 2] - finger2_tip[:, 2], min=0.0)
        z_gate = 1.0 - torch.tanh(z_above / 0.03)
    else:
        z_gate = torch.ones(env.num_envs, device=env.device)

    reward = approach_reward * z_gate

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)

    # Logging (contact/opposition still tracked for diagnostics)
    thumb_contact = _finger_surface_contact_gate(
        env, "tesollo_left_ll_dg_1_4", "y", -0.0363, object_cfg=object_cfg
    )
    grasp_gate = _grasp_progress_gate(env, object_cfg, eef_link_name)
    pinky_contact = _finger_surface_contact_gate(
        env, "tesollo_left_ll_dg_5_4", "z", 0.0363, object_cfg=object_cfg
    )
    thumb_opposition = _thumb_opposition_reward(env, object_cfg)
    _maybe_log_grasp_quality(
        env=env,
        object_cfg=object_cfg,
        eef_link_name=eef_link_name,
        lambda_trigger=lambda_trigger,
        grasp_gate=grasp_gate,
        thumb_contact=thumb_contact,
        pinky_contact=pinky_contact,
        thumb_opposition=thumb_opposition,
    )
    return lambda_trigger * reward


def pinky_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward pinky tip approaching cup center in XY (task-space).

    Same principle as thumb_grasp_reward: let the policy discover
    the right joint configuration via task-space gradient.

    DexPour: Active when λ=1 (approach complete).
    """
    pinky_tip = _get_fingertip_world_position(
        env, "tesollo_left_ll_dg_5_4", "z", 0.0363
    )
    if pinky_tip is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]
    tip_xy = pinky_tip[:, :2]

    # Surface approach: pinky tip → cup surface (not center)
    cup_radius = 0.045  # actual cup outer radius at scale (1,1,1.2)
    radial_dist = torch.norm(tip_xy - cup_xy, dim=1)
    surface_dist = torch.abs(radial_dist - cup_radius)
    approach_reward = 1.0 - torch.tanh(surface_dist / std)

    # Penetration penalty: penalize pinky being inside the cup
    penetration = torch.clamp(cup_radius - radial_dist, min=0.0)
    approach_reward = approach_reward - torch.tanh(penetration / 0.01)

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * approach_reward


def synergy_grip_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "left_hand_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    cup_radius: float = 0.045,
    proximity_std: float = 0.01,
) -> torch.Tensor:
    """Reward synergy gripper closing fully after approach.

    grip_strength in [-1, +1]: -1 = open, +1 = closed.
    Rewards grip_strength → +1 when λ=1 and synergy tips (fingers 2,3,4)
    are close to the cup surface in XY.

    DexPour: Active when λ=1 (approach complete).
    """
    action_term = env.action_manager.get_term(action_name)
    grip_strength = action_term.raw_actions[:, 0]  # (num_envs,)

    close_reward = torch.clamp((grip_strength + 1.0) / 2.0, min=0.0, max=1.0)

    obj: RigidObject = env.scene[object_cfg.name]
    cup_xy = obj.data.root_pos_w[:, :2]
    synergy_tips = [
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),
    ]

    total_surface_error = torch.zeros(env.num_envs, device=env.device)
    valid_tip_count = 0
    for body_name, axis, offset in synergy_tips:
        tip = _get_fingertip_world_position(env, body_name, axis, offset)
        if tip is None:
            continue
        valid_tip_count += 1
        radial_dist = torch.norm(tip[:, :2] - cup_xy, dim=1)
        total_surface_error += torch.abs(radial_dist - cup_radius)

    if valid_tip_count == 0:
        surface_proximity_gate = torch.zeros(env.num_envs, device=env.device)
    else:
        mean_surface_error = total_surface_error / float(valid_tip_count)
        surface_proximity_gate = 1.0 - torch.tanh(mean_surface_error / proximity_std)

    # DexPour: Only active when λ=1 (approach complete), with surface gate
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * surface_proximity_gate * close_reward


def synergy_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward synergy fingers (2,3,4) staying near initial open pose during reaching.

    DexPour: Active when λ=0 (before approach complete), deactivates when λ=1.
    """
    robot = env.scene["robot"]

    # Synergy fingers target = open pose (all joints near 0)
    _TARGETS = {
        "lj_dg_2_2": 0.0, "lj_dg_2_3": 0.0, "lj_dg_2_4": 0.0,
        "lj_dg_3_2": 0.0, "lj_dg_3_3": 0.0, "lj_dg_3_4": 0.0,
        "lj_dg_4_2": 0.0, "lj_dg_4_3": 0.0, "lj_dg_4_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    # DexPour: Active when λ=0 (before approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def finger_tip_to_cup_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward fingertips approaching cup center (XY plane).

    Encourages fingers to wrap around the cup by rewarding tips that
    get closer to the cup's XY position.
    DexPour: Active when λ=1 (approach complete).
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]  # (num_envs, 2)

    # Fingertip info: (body_name, local_offset_axis, offset_value)
    # Offset is applied in the link's local frame to get actual tip position
    _TIP_INFO = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363),  # thumb: Y offset
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),   # index: Z offset
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),   # middle: Z offset
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),   # ring: Z offset
        ("tesollo_left_ll_dg_5_4", "z", 0.0363),   # pinky: Z offset
    ]

    total_reward = torch.zeros(env.num_envs, device=env.device)
    num_tips = 0

    for body_name, offset_axis, offset_val in _TIP_INFO:
        if body_name in robot.data.body_names:
            body_idx = robot.data.body_names.index(body_name)
            link_pos = robot.data.body_pos_w[:, body_idx]  # (num_envs, 3)
            link_quat = robot.data.body_quat_w[:, body_idx]  # (num_envs, 4)

            # Create local offset vector
            if offset_axis == "x":
                local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
            elif offset_axis == "y":
                local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
            else:  # "z"
                local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

            # Transform offset to world frame and add to link position
            world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
            tip_pos = link_pos + world_offset
            tip_pos_xy = tip_pos[:, :2]  # (num_envs, 2)

            dist_xy = torch.norm(tip_pos_xy - cup_pos_xy, dim=1)
            tip_reward = 1.0 - torch.tanh(dist_xy / std)
            total_reward += tip_reward
            num_tips += 1

    if num_tips > 0:
        total_reward = total_reward / num_tips  # Average over tips

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)

    # Visualize fingertip positions
    _maybe_visualize_fingertips(env, robot, _TIP_INFO, obj)

    return lambda_trigger * total_reward


def finger_wrap_cylinder_reward(
    env: ManagerBasedRLEnv,
    target_radius: float = 0.045,
    radial_std: float = 0.015,
    opposition_weight: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward cylindrical wrap grasp around the cup.

    Components:
    - Radial ring reward: fingertips should lie near the cup radius in XY.
    - Opposition reward: thumb should oppose the mean direction of other fingers.

    DexPour: Active when λ=1 (approach complete).
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    # Fingertip info: (finger_name, body_name, local_offset_axis, offset_value)
    tip_info = [
        ("thumb", "tesollo_left_ll_dg_1_4", "y", -0.0363),
        ("index", "tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("middle", "tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("ring", "tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("pinky", "tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]

    tip_xy_by_name: dict[str, torch.Tensor] = {}
    radial_reward_sum = torch.zeros(env.num_envs, device=env.device)
    tip_count = 0

    for finger_name, body_name, offset_axis, offset_val in tip_info:
        if body_name not in robot.data.body_names:
            continue

        body_idx = robot.data.body_names.index(body_name)
        link_pos = robot.data.body_pos_w[:, body_idx]
        link_quat = robot.data.body_quat_w[:, body_idx]

        if offset_axis == "x":
            local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
        elif offset_axis == "y":
            local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
        else:
            local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

        world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
        tip_pos = link_pos + world_offset
        tip_xy = tip_pos[:, :2]
        tip_xy_by_name[finger_name] = tip_xy

        radial_dist = torch.norm(tip_xy - cup_pos_xy, dim=1)
        radial_error = torch.abs(radial_dist - target_radius)
        radial_reward = 1.0 - torch.tanh(radial_error / radial_std)
        radial_reward_sum += radial_reward
        tip_count += 1

    if tip_count == 0:
        return torch.zeros(env.num_envs, device=env.device)

    radial_reward_mean = radial_reward_sum / float(tip_count)

    opposition_reward = torch.zeros(env.num_envs, device=env.device)
    if "thumb" in tip_xy_by_name:
        other_vectors = []
        for key in ("index", "middle", "ring", "pinky"):
            if key in tip_xy_by_name:
                other_vectors.append(tip_xy_by_name[key] - cup_pos_xy)

        if other_vectors:
            thumb_vec = tip_xy_by_name["thumb"] - cup_pos_xy
            others_mean_vec = torch.stack(other_vectors, dim=0).mean(dim=0)

            thumb_unit = thumb_vec / (torch.norm(thumb_vec, dim=1, keepdim=True) + 1e-6)
            others_unit = others_mean_vec / (torch.norm(others_mean_vec, dim=1, keepdim=True) + 1e-6)
            dot = torch.sum(thumb_unit * others_unit, dim=1)
            opposition_reward = torch.clamp(-dot, min=0.0, max=1.0)

    opposition_weight = float(max(0.0, min(1.0, opposition_weight)))
    reward = (1.0 - opposition_weight) * radial_reward_mean + opposition_weight * opposition_reward

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    _maybe_visualize_fingertips(env, robot, [(x[1], x[2], x[3]) for x in tip_info], obj)
    return lambda_trigger * reward


def finger_wrap_coverage_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward angular coverage of fingertips around cup center in XY.

    Uses pairwise angular separation among available fingertips.
    Higher reward when fingers are not collapsed into a narrow sector.
    DexPour: Active when λ=1 (approach complete).
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    # (finger_name, body_name, local_offset_axis, offset_value)
    tip_info = [
        ("thumb", "tesollo_left_ll_dg_1_4", "y", -0.0363),
        ("index", "tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("middle", "tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("ring", "tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("pinky", "tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]

    unit_vecs = []
    for _, body_name, offset_axis, offset_val in tip_info:
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        link_pos = robot.data.body_pos_w[:, body_idx]
        link_quat = robot.data.body_quat_w[:, body_idx]

        if offset_axis == "x":
            local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
        elif offset_axis == "y":
            local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
        else:
            local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

        world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
        tip_xy = (link_pos + world_offset)[:, :2]
        vec = tip_xy - cup_pos_xy
        unit = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-6)
        unit_vecs.append(unit)

    if len(unit_vecs) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    # Average pairwise angular-distance surrogate: (1 - cos(theta))/2 in [0,1]
    pair_scores = torch.zeros(env.num_envs, device=env.device)
    pair_count = 0
    for i in range(len(unit_vecs)):
        for j in range(i + 1, len(unit_vecs)):
            cos_ij = torch.sum(unit_vecs[i] * unit_vecs[j], dim=1).clamp(-1.0, 1.0)
            pair_scores += (1.0 - cos_ij) * 0.5
            pair_count += 1

    coverage_reward = pair_scores / float(max(1, pair_count))

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * coverage_reward


def finger_tip_orientation_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward fingertip normals pointing toward cup center (XY plane).

    - Thumb (finger 1): local Y axis is the normal direction
    - Fingers 2-5: local X axis is the normal direction

    Rewards alignment between finger normal and direction to cup center.
    DexPour: Active when λ=1 (approach complete).
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]  # (num_envs, 2)

    # Fingertip info: (body_name, offset_axis, offset_val, normal_axis)
    # offset: to get actual tip position
    # normal_axis: the local axis that points outward from fingertip
    _TIP_INFO = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363, "y"),  # thumb: Y offset, Y normal
        ("tesollo_left_ll_dg_2_4", "z", 0.0255, "x"),   # index: Z offset, X normal
        ("tesollo_left_ll_dg_3_4", "z", 0.0255, "x"),   # middle: Z offset, X normal
        ("tesollo_left_ll_dg_4_4", "z", 0.0255, "x"),   # ring: Z offset, X normal
        ("tesollo_left_ll_dg_5_4", "z", 0.0363, "x"),   # pinky: Z offset, X normal
    ]

    total_reward = torch.zeros(env.num_envs, device=env.device)
    num_tips = 0

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    for body_name, offset_axis, offset_val, normal_axis in _TIP_INFO:
        if body_name in robot.data.body_names:
            body_idx = robot.data.body_names.index(body_name)
            link_pos = robot.data.body_pos_w[:, body_idx]  # (num_envs, 3)
            link_quat = robot.data.body_quat_w[:, body_idx]  # (num_envs, 4)

            # Create local offset vector for tip position
            if offset_axis == "x":
                local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
            elif offset_axis == "y":
                local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
            else:  # "z"
                local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

            # Calculate actual tip position
            world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
            tip_pos = link_pos + world_offset
            tip_pos_xy = tip_pos[:, :2]  # (num_envs, 2)

            # Get the normal direction in world frame
            if normal_axis == "x":
                local_normal = x_axis.unsqueeze(0).repeat(env.num_envs, 1)
            elif normal_axis == "y":
                local_normal = y_axis.unsqueeze(0).repeat(env.num_envs, 1)
            else:  # "z"
                local_normal = z_axis.unsqueeze(0).repeat(env.num_envs, 1)

            # Transform local normal to world frame
            normal_world = quat_apply(link_quat, local_normal)  # (num_envs, 3)
            normal_xy = normal_world[:, :2]  # (num_envs, 2)
            normal_xy = normal_xy / (torch.norm(normal_xy, dim=1, keepdim=True) + 1e-6)

            # Direction from tip to cup center (XY)
            dir_to_cup = cup_pos_xy - tip_pos_xy  # (num_envs, 2)
            dir_to_cup = dir_to_cup / (torch.norm(dir_to_cup, dim=1, keepdim=True) + 1e-6)

            # Dot product: 1.0 when aligned, -1.0 when opposite
            alignment = torch.sum(normal_xy * dir_to_cup, dim=1)  # (num_envs,)

            # Reward when pointing toward cup
            tip_reward = torch.clamp(alignment, min=0.0)

            total_reward += tip_reward
            num_tips += 1

    if num_tips > 0:
        total_reward = total_reward / num_tips  # Average over tips

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)

    return lambda_trigger * total_reward


def _maybe_visualize_fingertips(
    env: ManagerBasedRLEnv,
    robot,
    tip_info: list[tuple],
    obj: RigidObject,
) -> None:
    """Visualize lines from fingertips to cup center (XY projection)."""
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_fingertip_vis", False):
        return

    step_count = int(getattr(env, "common_step_counter", 0))
    interval = int(getattr(cfg, "debug_fingertip_vis_interval", 10))
    if interval > 1 and (step_count % interval) != 0:
        return

    # Initialize debug draw and carb types
    if not hasattr(env, "_debug_draw"):
        try:
            from isaacsim.util.debug_draw import _debug_draw
            import carb
            env._debug_draw = _debug_draw.acquire_debug_draw_interface()
            env._carb = carb
            print("[DEBUG] debug_draw interface acquired successfully")
        except ImportError:
            try:
                from omni.isaac.debug_draw import _debug_draw
                import carb
                env._debug_draw = _debug_draw.acquire_debug_draw_interface()
                env._carb = carb
                print("[DEBUG] debug_draw interface acquired (omni.isaac)")
            except Exception as e:
                print(f"[DEBUG] Failed to acquire debug_draw: {e}")
                env._debug_draw = None
                return

    if env._debug_draw is None:
        return

    carb = env._carb

    # Clear previous drawings
    env._debug_draw.clear_lines()

    # Get cup center position (env 0) - use tip Z height for XY plane visualization
    cup_pos = obj.data.root_pos_w[0].cpu().numpy()  # (3,)

    # Colors for each finger (RGBA)
    colors = [
        carb.ColorRgba(1.0, 0.0, 0.0, 1.0),  # thumb - red
        carb.ColorRgba(0.0, 1.0, 0.0, 1.0),  # index - green
        carb.ColorRgba(0.0, 0.0, 1.0, 1.0),  # middle - blue
        carb.ColorRgba(1.0, 1.0, 0.0, 1.0),  # ring - yellow
        carb.ColorRgba(1.0, 0.0, 1.0, 1.0),  # pinky - magenta
    ]

    # Collect all points for batch drawing
    starts = []
    ends = []
    line_colors = []
    sizes = []

    # Draw lines from each fingertip to cup center (XY plane - same Z as tip)
    for i, tip_data in enumerate(tip_info):
        body_name, offset_axis, offset_val = tip_data[0], tip_data[1], tip_data[2]
        if body_name in robot.data.body_names:
            body_idx = robot.data.body_names.index(body_name)
            link_pos = robot.data.body_pos_w[0, body_idx]  # (3,)
            link_quat = robot.data.body_quat_w[0, body_idx]  # (4,)

            # Create local offset vector
            if offset_axis == "x":
                local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
            elif offset_axis == "y":
                local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
            else:  # "z"
                local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

            # Transform offset to world frame and get actual tip position
            world_offset = quat_apply(link_quat.unsqueeze(0), local_offset.unsqueeze(0)).squeeze(0)
            tip_pos = (link_pos + world_offset).cpu().numpy()

            # Use tip's Z for both start and end to show XY plane distance
            tip_z = float(tip_pos[2])
            starts.append(carb.Float3(float(tip_pos[0]), float(tip_pos[1]), tip_z))
            ends.append(carb.Float3(float(cup_pos[0]), float(cup_pos[1]), tip_z))
            line_colors.append(colors[i % len(colors)])
            sizes.append(5.0)

    if starts:
        env._debug_draw.draw_lines(starts, ends, line_colors, sizes)


def thumb_tip_z_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    xy_proximity_std: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    cup_height: float = 0.09,
) -> torch.Tensor:
    """Reward thumb tip Z being at or below index finger (finger 2) tip Z.

    One-sided: full reward when thumb_z <= finger2_z, decays when above.
    DexPour: Active when λ=1 (approach complete).
    """
    thumb_tip = _get_fingertip_world_position(
        env, "tesollo_left_ll_dg_1_4", "y", -0.0363
    )
    finger2_tip = _get_fingertip_world_position(
        env, "tesollo_left_ll_dg_2_4", "z", 0.0255
    )
    if thumb_tip is None or finger2_tip is None:
        return torch.zeros(env.num_envs, device=env.device)

    thumb_z = thumb_tip[:, 2]
    finger2_z = finger2_tip[:, 2]

    # One-sided: penalize only when thumb is ABOVE finger 2
    z_above = torch.clamp(thumb_z - finger2_z, min=0.0)
    z_reward = 1.0 - torch.tanh(z_above / std)

    # XY proximity: reward decays if EE is far from cup in XY
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    xy_dist = torch.norm(ee_pos_w[:, :2] - cup_pos_xy, dim=1)
    xy_proximity = torch.exp(-xy_dist / xy_proximity_std)

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * z_reward * xy_proximity


def synergy_tip_z_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.06,
    cup_height: float = 0.09,
    xy_proximity_std: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward index finger (finger 2) tip Z position approaching cup top height.

    Uses finger 2 as representative for synergy fingers (2,3,4).
    Guides the fingertip to descend to the cup's top (grasp) height.
    DexPour: Active when λ=1 (approach complete).
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_top_z = obj.data.root_pos_w[:, 2] + cup_height  # target: cup top
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    # Index finger (2) tip: body + local offset
    body_name = "tesollo_left_ll_dg_2_4"
    offset_axis, offset_val = "z", 0.0255

    if body_name not in robot.data.body_names:
        return torch.zeros(env.num_envs, device=env.device)

    body_idx = robot.data.body_names.index(body_name)
    link_pos = robot.data.body_pos_w[:, body_idx]
    link_quat = robot.data.body_quat_w[:, body_idx]

    local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)
    world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
    tip_pos = link_pos + world_offset
    tip_z = tip_pos[:, 2]

    z_error = torch.abs(tip_z - cup_top_z)
    z_reward = 1.0 - torch.tanh(z_error / std)

    # XY proximity: reward decays if EE is far from cup in XY
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    xy_dist = torch.norm(ee_pos_w[:, :2] - cup_pos_xy, dim=1)
    xy_proximity = torch.exp(-xy_dist / xy_proximity_std)

    # DexPour: Active when λ=1 (approach complete)
    lambda_trigger = _approach_trigger(env, object_cfg, eef_link_name)
    return lambda_trigger * z_reward * xy_proximity
