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


def _cup_upright_cos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """컵 로컬 +z 의 world z 성분. 1 = 완전히 세워짐, 0 = 옆으로 누움."""
    obj: RigidObject = env.scene[object_cfg.name]
    w, x, y, z = obj.data.root_quat_w.unbind(-1)
    return 1.0 - 2.0 * (x * x + y * y)


def jaw_level_quality(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, body_name: str
) -> torch.Tensor:
    """jaw 축이 수평인 정도. 1 = 완전 수평, 0 = 수직.

    jaw 축은 두 손가락을 잇는 방향, 즉 `gripper_base` 프레임의 **y 축**이다
    (URDF: 손가락이 base 의 ±y 로 벌어진다). 그 world z 성분의 크기가 곧 기울기의 sin 이다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    w, x, y, z = robot.data.body_quat_w[:, body_idx, :].unbind(-1)
    jaw_axis_z = 2.0 * (y * z + w * x)              # 회전행렬 R[2,1]
    return (1.0 - jaw_axis_z.abs()).clamp(min=0.0)


def _held(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    max_ee_distance: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    min_upright_cos: float = -1.0,
) -> torch.Tensor:
    """물체를 **제대로** 들고 있는가. (num_envs,) float 0/1.

    · 임계 높이 위로 올라갔고
    · TCP 가 곁에 있으며(그리퍼가 아닌 부위로 떠받치는 것 차단)
    · 컵이 세워져 있다(`min_upright_cos`)

    ★컵 자세 조건이 필요한 이유: 근접 조건만으로는 컵을 **47° 기울인 채** 손가락 끝으로
      걸어 올리는 파지가 학습된다(test4 실측: 컵 기울기 47.1°, 그리퍼 개도 5.6 mm 로
      몸통을 물지 못한 상태). 사용자 요구는 "수평으로 제대로 잡기" 다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    lifted = obj_pos_w[:, 2] > minimal_height
    near = torch.norm(obj_pos_w - ee_pos_w, dim=1) < max_ee_distance
    upright = _cup_upright_cos(env, object_cfg) > min_upright_cos
    return (lifted & near & upright).float()


def held_with_good_pose(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    max_ee_distance: float,
    body_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """든 상태에서 **자세가 좋을수록** 커지는 보너스 (0~1).

    품질 = (컵이 세워진 정도) × (jaw 가 수평인 정도). 둘 다 0~1 연속.

    ★★자세를 **게이트로 넣으면 안 된다** — 한 번 실패한 설계다. 컵 자세를 40° AND 게이트로
      걸었더니(test6/test7) 파지 중 필연적인 흔들림이 전부 차단돼 양의 보상이 **완전히 0**이
      됐고, 남은 것이 페널티뿐이라 **에피소드를 빨리 끝내는 것이 최적**이 됐다:
          lifting 6.14 → 0.0000 / 에피소드 길이 130 → 13 / 총보상 +34.9 → −0.46
      학습이 시작조차 못 한다. 자세는 반드시 연속 보너스로만 유도한다.
    """
    gate = _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg)
    upright = _cup_upright_cos(env, object_cfg).clamp(min=0.0)
    return gate * upright * jaw_level_quality(env, robot_cfg, body_name)


def object_is_held_and_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    max_ee_distance: float,
    min_upright_cos: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_is_lifted` 에 근접·컵 자세 조건을 더한 것."""
    return _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg, min_upright_cos)


def object_goal_distance_when_held(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    max_ee_distance: float,
    command_name: str,
    min_upright_cos: float = -1.0,
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
    gate = _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg, min_upright_cos)
    return gate * (1 - torch.tanh(distance / std))


def object_settled_at_goal(
    env: "ManagerBasedRLEnv",
    std: float,
    lin_vel_std: float,
    ang_vel_std: float,
    minimal_height: float,
    max_ee_distance: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """컵을 목표까지 옮겨 **가만히 정지**시켰는가 (0~1).

    ★레퍼런스 lift 의 `object_goal_distance` 는 **거리만** 본다. 목표 근처에서 컵이 계속
      흔들려도 만점이라, "옮겨서 세워 둔다"는 요구를 표현하지 못한다. 실제로 test8 은
      goal-tracking 이 상한의 68% 까지 갔는데 정밀 항(`goal_fine`)은 16% 에 머물렀다.

    품질 = (게이트) × (목표 근접) × (정지 정도). 셋 다 연속이라 gradient 가 이어진다.
      · 목표 근접을 곱하므로 "든 채 제자리에 가만히 있기"는 0 이다(목표에서 멀면 근접이 0).
      · 정지 정도는 컵의 선속도·각속도 둘 다 본다 — 각속도를 빼면 제자리에서 빙빙 도는
        상태가 만점이 된다.

    ⚠ 자세와 마찬가지로 **게이트가 아니라 보너스**다. 판정 게이트에 조건을 더 얹으면
      양의 보상이 0 이 되고 조기 종료가 최적이 된다(test6/test7 에서 실증).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    near_goal = 1.0 - torch.tanh(distance / std)

    lin = torch.norm(obj.data.root_lin_vel_w, dim=1)
    ang = torch.norm(obj.data.root_ang_vel_w, dim=1)
    still = (1.0 - torch.tanh(lin / lin_vel_std)) * (1.0 - torch.tanh(ang / ang_vel_std))

    gate = _held(env, minimal_height, max_ee_distance, object_cfg, ee_frame_cfg)
    return gate * near_goal * still
