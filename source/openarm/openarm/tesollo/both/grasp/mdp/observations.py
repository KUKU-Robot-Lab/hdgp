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

from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_object_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "unidex_object_ids"):
        env.unidex_object_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return env.unidex_object_ids


def get_selected_object_pose(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("objects")
) -> tuple[torch.Tensor, torch.Tensor]:
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    obj_ids = _get_object_ids(env)
    env_ids = torch.arange(env.num_envs, device=objects.device)
    pos_w = objects.data.object_link_pos_w[env_ids, obj_ids]
    quat_w = objects.data.object_link_quat_w[env_ids, obj_ids]
    return pos_w, quat_w


def get_selected_object_lin_vel(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("objects")
) -> torch.Tensor:
    objects: RigidObjectCollection = env.scene[object_cfg.name]
    obj_ids = _get_object_ids(env)
    env_ids = torch.arange(env.num_envs, device=objects.device)
    return objects.data.object_link_lin_vel_w[env_ids, obj_ids]


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Object observations in env frame with relative vectors to both end effectors."""

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos_w, object_quat_w = get_selected_object_pose(env, object_cfg)
    object_pos = object_pos_w - env.scene.env_origins

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat_w,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def object_pc_feat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-object pc feature for the selected object."""

    obj_ids = _get_object_ids(env)
    if not hasattr(env, "unidex_pc_feat"):
        feat_dim = getattr(env, "unidex_pc_feat_dim", 64)
        return torch.zeros((env.num_envs, feat_dim), device=env.device)
    pc_feat = env.unidex_pc_feat
    if pc_feat.device != env.device:
        pc_feat = pc_feat.to(env.device)
    return pc_feat[obj_ids]


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_quat = body_quat_w[:, eef_idx]
    return eef_quat
