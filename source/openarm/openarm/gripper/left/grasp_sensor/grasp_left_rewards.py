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

"""리프트 판정에 **쥐고 있는가**를 더한 보상.

레퍼런스(`lift/mdp/rewards.py`)의 리프트 판정은 물체 원점 z 하나만 본다. 큐브에는 그것으로
충분하다 — 쳐도 잘 안 뜨고, 뜨더라도 곧 떨어진다. 우리 shaker 는 134 g 에 높이 175 mm 라
사정이 다르다.

★test3(1500 epoch)이 실증한 것: 정책이 컵을 **위로 힘껏 쳐 날리고**, 컵이 공중에 있는
  1.8 초 동안 리프트 보상(weight 15)과 goal-tracking(16)을 계속 받았다.
      리프트 판정 비율 85.9% / 그 동안 **TCP–컵 거리 평균 3044 mm**
      reaching_object 0.019 → 0.018 (평탄, 그리퍼는 컵에 가지 않는다)
      object_dropping 종료 99.8% (결국 떨어져서 끝난다)
  z 만 보는 판정에서 이것은 완벽하게 합리적인 전략이고, 보상을 아무리 재조정해도
  "던지기"가 "집기"보다 쉬운 한 사라지지 않는다.

그래서 판정을 `z > 임계` 에서 `z > 임계 **그리고** TCP 가 컵 곁에 있다` 로 바꾼다.
쳐서 날린 컵은 즉시 TCP 에서 멀어지므로 보상이 끊긴다.

레퍼런스 시그니처를 그대로 유지하고 `max_ee_distance` 만 더했다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _held(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    max_ee_distance: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """물체가 임계 위로 올라갔고 **동시에** TCP 가 곁에 있는가. (num_envs,) float 0/1."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    lifted = obj_pos_w[:, 2] > minimal_height
    near = torch.norm(obj_pos_w - ee_pos_w, dim=1) < max_ee_distance
    return (lifted & near).float()


def object_is_held_and_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    max_ee_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_is_lifted` 에 근접 조건을 더한 것."""
    return _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg)


def object_goal_distance_when_held(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    max_ee_distance: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_goal_distance` 의 게이트를 근접 조건까지 요구하도록 바꾼 것.

    레퍼런스와 마찬가지로 목표 위치는 **로봇 베이스 기준** 명령을 world 로 변환해 쓴다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    gate = _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg)
    return gate * (1 - torch.tanh(distance / std))
