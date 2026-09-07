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

"""관측 항 — **절대 태스크공간 액션**에 필요한 피드백.

★★fab_test67 실측이 만든 항목이다. 액션은 절대 palm 6D 지령인데 obs 에는
`last_action`(raw) 밖에 없었다. 그래서 정책은 두 가지를 못 봤다:

1. **자기 palm 이 지금 어디 있는가.** 관절각으로부터 7-DOF FK 를 스스로 배워야 했다.
2. **지령이 지금 어디 있는가.** `PALM_CMD_RATE_LIMIT`(0.02 m/step) 리미터는 적분기다 —
   raw 액션과 실제 지령은 다르고, 그 차이는 액션 이력 전체를 적분해야 알 수 있다.
   메모리 없는 MLP 에는 **원리적으로 관측 불가능한 숨은 상태**였다.

정규화 규약: 위치·회전 모두 **액션 박스와 같은 스케일**로 낸다. 그래서 지령 채널과
실측 채널의 차이가 곧 추종오차이고, 정책이 뺄셈 한 번으로 얻는다. 값은 대략 [-1, 1] 이라
`clip_observations` 100 에 걸리지 않는다.

sim2real: 세 항 모두 실기에서 얻는다 — palm pose 는 엔코더 FK, 지령은 실기에서도 같은
fabric 을 돌리므로 우리 쪽 상태다. 특권 정보가 아니다([[설계 불변식 2]]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, wrap_to_pi

from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_BOX_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _boxes(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """액션 박스의 (위치중심, 위치반폭, 회전중심). 액션 항과 **같은 상수**에서 파생한다.

    ★캐시하는 이유: obs 는 매 스텝 호출된다. `torch.tensor([...], device=cuda)` 는
    호스트→디바이스 복사라 hot path 에 놓으면 동기화 지점이 된다.
    """
    hit = _BOX_CACHE.get(device)
    if hit is None:
        lo = torch.tensor(
            [P.PALM_BOX_X[0], P.PALM_BOX_Y[0], P.PALM_BOX_Z[0]], device=device
        )
        hi = torch.tensor(
            [P.PALM_BOX_X[1], P.PALM_BOX_Y[1], P.PALM_BOX_Z[1]], device=device
        )
        e = torch.tensor(P.PALM_EULER_ZYX_CENTER, device=device, dtype=torch.float32)
        hit = (0.5 * (lo + hi), 0.5 * (hi - lo), e)
        _BOX_CACHE[device] = hit
    return hit


def palm_command(env: ManagerBasedRLEnv, action_term: str = "arm_action") -> torch.Tensor:
    """리미터를 통과한 **실제 지령**을 액션 스케일로. (num_envs, 6)

    raw 액션(`last_action`)이 아니라 `processed_actions` 다 — 둘의 차이가 리미터
    적분기이고, 그게 이 항의 존재 이유다.
    """
    term = env.action_manager.get_term(action_term)
    cmd = term.processed_actions
    center, half, e_center = _boxes(cmd.device)
    pos = (cmd[:, :3] - center) / half
    rot = wrap_to_pi(cmd[:, 3:6] - e_center) / P.PALM_MAX_POSE_ANGLE
    return torch.cat([pos, rot], dim=-1)


def tcp_position_in_root(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """TCP 위치를 로봇 root 프레임에서, **지령과 같은 스케일**로. (num_envs, 3)

    `object_position_in_robot_root_frame` 과 같은 프레임이라 정책이 뺄셈만으로
    TCP→컵 접근 벡터를 얻는다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee.data.target_pos_w[:, 0, :],
    )
    center, half, _ = _boxes(pos_b.device)
    return (pos_b - center) / half


def palm_rot6d_in_root(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[P.GRIPPER_BASE_BODY]),
) -> torch.Tensor:
    """그리퍼 base 자세를 **6D 회전표현**으로. (num_envs, 6)

    ★euler 로 내지 않는 이유: `PALM_EULER_ZYX_CENTER` 의 roll 이 3.095 rad 로 π 코앞이라
    euler 관측은 ±π 경계에서 6.28 rad 를 널뛴다. 회전행렬 앞 두 열은 그런 불연속이
    없고, fabric palm 링크와 USD body 의 프레임 규약이 달라도 안전하다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if isinstance(robot_cfg.body_ids, slice):
        raise ValueError(
            "robot_cfg 가 resolve 되지 않았다 — SceneEntityCfg 는 ObsTerm(params=...) 로 "
            "넘겨야 매니저가 body_ids 를 채운다. 기본 인자로 두면 slice(None) 이 온다."
        )
    body_id = robot_cfg.body_ids[0]
    _, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id, :],
        robot.data.body_quat_w[:, body_id, :],
    )
    return matrix_from_quat(quat_b)[:, :, :2].reshape(-1, 6)
