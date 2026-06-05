from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _ensure_selected_object_ids(env: ManagerBasedEnv, num_objects: int) -> torch.Tensor:
    if not hasattr(env, "selected_object_ids"):
        env.selected_object_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.selected_object_ids = torch.clamp(env.selected_object_ids, 0, max(0, num_objects - 1))
    return env.selected_object_ids


def reset_object_collection(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    parking_pos: tuple[float, float, float] = (0.0, 0.0, -2.0),
) -> None:
    """Randomly select one object per env, place it on table, and park others."""
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=objects.device)
    if len(env_ids) == 0:
        return

    num_envs = len(env_ids)
    num_objects = objects.num_objects
    selected_ids = _ensure_selected_object_ids(env, num_objects)
    sampled_obj_ids = torch.randint(0, num_objects, (num_envs,), device=objects.device)
    selected_ids[env_ids] = sampled_obj_ids

    # Start from default object states.
    object_state = objects.data.default_object_state[env_ids].clone()

    # Park all objects below table.
    parking = torch.tensor(parking_pos, device=objects.device)
    object_state[..., 0:3] = env.scene.env_origins[env_ids].unsqueeze(1) + parking
    object_state[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=objects.device)
    object_state[..., 7:13] = 0.0

    # Random pose for selected object.
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=objects.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=objects.device)

    default_sel = objects.data.default_object_state[env_ids, sampled_obj_ids].clone()
    pos = default_sel[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    quat_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    quat = math_utils.quat_mul(default_sel[:, 3:7], quat_delta)

    env_local = torch.arange(num_envs, device=objects.device)
    object_state[env_local, sampled_obj_ids, 0:3] = pos
    object_state[env_local, sampled_obj_ids, 3:7] = quat
    object_state[env_local, sampled_obj_ids, 7:13] = default_sel[:, 7:13]

    objects.write_object_state_to_sim(object_state, env_ids=env_ids)


def _get_selected_object_pose(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> tuple[torch.Tensor, torch.Tensor]:
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    selected_ids = _ensure_selected_object_ids(env, objects.num_objects)
    env_ids = torch.arange(env.num_envs, device=objects.device)
    pos_w = objects.data.object_link_pos_w[env_ids, selected_ids]
    quat_w = objects.data.object_link_quat_w[env_ids, selected_ids]
    return pos_w, quat_w


def _get_selected_object_lin_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    selected_ids = _ensure_selected_object_ids(env, objects.num_objects)
    env_ids = torch.arange(env.num_envs, device=objects.device)
    return objects.data.object_link_lin_vel_w[env_ids, selected_ids]


def selected_object_obs_left(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Object pose + relative vector from left EE in env frame."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    left_eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins

    obj_pos_w, obj_quat_w = _get_selected_object_pose(env, object_cfg)
    obj_pos = obj_pos_w - env.scene.env_origins
    rel = obj_pos - left_eef_pos
    return torch.cat((obj_pos, obj_quat_w, rel), dim=1)


def selected_object_id_onehot(env: ManagerBasedRLEnv, num_classes: int | None = None) -> torch.Tensor:
    """One-hot selected object ID for object-conditioned policy."""
    num_objects = int(num_classes) if num_classes is not None else int(getattr(env, "object_bank_size", 1))
    if num_objects <= 1:
        return torch.ones((env.num_envs, 1), device=env.device)
    selected_ids = _ensure_selected_object_ids(env, num_objects)
    onehot = torch.zeros((env.num_envs, num_objects), device=env.device)
    onehot.scatter_(1, selected_ids.unsqueeze(1), 1.0)
    return onehot


def selected_object_eef_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    left_eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(dist / std)


def selected_object_contact_reward_left(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_eef_link_name: str,
    max_dist: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_grasp"),
    body_name_pattern: str | None = ".*lj_dg.*",
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    if body_name_pattern is not None:
        key = f"_contact_body_ids_{body_name_pattern}"
        if not hasattr(env, key):
            body_ids = [i for i, n in enumerate(contact_sensor.body_names) if re.fullmatch(body_name_pattern, n)]
            if not body_ids:
                body_ids = list(range(contact_sensor.num_bodies))
            setattr(env, key, torch.tensor(body_ids, device=net_forces.device, dtype=torch.long))
        body_ids = getattr(env, key)
        net_forces = net_forces[:, :, body_ids, :]
    contact_mag = torch.norm(net_forces, dim=-1)
    contact_any = torch.max(contact_mag, dim=1)[0].max(dim=1)[0] > threshold

    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    near = torch.norm(obj_pos_w - ee_pos_w, dim=1) < max_dist
    return (contact_any & near).to(torch.float)


def selected_object_grasp_success_left(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    left_eef_link_name: str,
    max_dist: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_grasp"),
    body_name_pattern: str | None = ".*lj_dg.*",
) -> torch.Tensor:
    contact = selected_object_contact_reward_left(
        env=env,
        threshold=threshold,
        left_eef_link_name=left_eef_link_name,
        max_dist=max_dist,
        object_cfg=object_cfg,
        sensor_cfg=sensor_cfg,
        body_name_pattern=body_name_pattern,
    )
    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    lifted = obj_pos_w[:, 2] > minimal_height
    return (contact.bool() & lifted).to(torch.float)


def selected_object_stability_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    lifted = obj_pos_w[:, 2] > minimal_height
    speed = torch.norm(_get_selected_object_lin_vel(env, object_cfg), dim=1)
    return torch.exp(-speed / std) * lifted.to(torch.float)


def selected_object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    return (obj_pos_w[:, 2] > minimal_height).to(torch.float)


def selected_object_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    obj_pos_w, _ = _get_selected_object_pose(env, object_cfg)
    return obj_pos_w[:, 2] < minimum_height
