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

"""v2 관측 — v1 항을 그대로 쓰고 **조건부 관계를 직접 주는 항**만 더한다.

전 항이 실기 배포 가능하다: 관절은 엔코더, TCP·palm 자세는 FK, 컵 위치는 `/cup_pose`
(FoundationPose), 목표는 우리가 내리는 명령이다. 참고 문서 6.1 도 sim2real 초기
검증에는 vision-only 보다 pose estimator 를 권한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

# v1 관측 함수 재사용 — 차원·규약이 이미 검증돼 있다.
from ..grasp_sensor.grasp_left_observations import (  # noqa: F401
    palm_rot6d_in_root,
    tcp_position_in_root,
)
from . import v2_stages as S

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_noisy(env: "ManagerBasedRLEnv",
                          robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                          jaw_cfg: SceneEntityCfg | None = None,
                          step_noise: float = 0.0) -> torch.Tensor:
    """컵 위치(로봇 베이스 기준) + **에피소드 bias** + 스텝 잡음 (라운드 9 obs DR).

    bias 는 `dr_obs_bias` 리셋 이벤트가 env 버퍼(`_v2_cup_obs_bias`)에 샘플해 두고,
    여기서 더한다. 실기 `/cup_pose` 캘리브 오차(41 mm)의 성질이 "한 판 안에서 고정"
    이라 Unoise(스텝 독립)로는 표현이 안 되기 때문이다.
    ⚠ 보상·판정은 ground truth 를 그대로 쓴다 — 노이즈는 **정책의 눈**에만 낀다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    from isaaclab.utils.math import subtract_frame_transforms
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3])
    bias = getattr(env, "_v2_cup_obs_bias", None)
    if bias is not None:
        pos_b = pos_b + bias
    if step_noise > 0.0:
        # ★★08.31 라운드 15 (사용자 지적) — 잡음은 **파지 전에만** 얹는다.
        #   실기에서 컵 좌표의 출처가 국면마다 다르다:
        #     · 파지 전 — `/cup_pose` 인식 → 캘리브 bias + 인식 잡음
        #     · 파지 후 — **TCP FK + 파지 오프셋** → 엔코더 정밀도, 스텝 잡음 없음
        #   파지 후에도 매 스텝 흔들면 **절대 위치 지령이 그대로 떨고** fabric 이
        #   충실히 따라간다. 학습 내내 그러면 정책이 고이득 반응성을 배운다
        #   (추론 때만 잡음을 꺼도 안 사라진다 — ablation 으로 확인).
        noise = (torch.rand_like(pos_b) * 2.0 - 1.0) * step_noise
        if jaw_cfg is not None:
            from . import v2_stages as _S
            held = _S.stage_close(env, jaw_cfg, object_cfg) > 0.5
            noise = noise * (~held).unsqueeze(-1).to(pos_b.dtype)
        pos_b = pos_b + noise
    return pos_b


def goal_minus_cup(env: "ManagerBasedRLEnv",
                   command_name: str = "object_pose",
                   robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """**목표 − 컵** 상대벡터 (N, 3), 로봇 베이스 기준.

    ★★왜 필요한가. t79 best 결정론 프로브에서 목표→지령 기울기가 0 이 아니라 **음수**
      였다(x −0.087 · y −0.118 · z +0.121, 1.0 이 정상). 음수는 "약한 신호를 못 따라간다"가
      아니라 **관계를 표현하지 못한다**(잡음 적합)는 뜻이다. 축 포화를 99.1% → 0.7% 로
      없애도 기울기는 안 생겼으므로 표현력 쪽 문제로 좁혀진다.

      정책은 지금 이 관계를 두 절대 위치의 차로 **합성**해야 한다. 그런데
      `normalize_input` 은 축별 running mean/std 라, 궤적 전체를 도는 컵 위치(약 300 mm
      범위)와 목표 명령(±57 mm)이 서로 다른 이득으로 정규화된다. 차분을 직접 주면 그
      합성 학습이 통째로 불필요해진다.

    ⚠ t68/t69 함정: **지령과 실측을 둘 다** 주면 학습이 죽었다(lift 0.00, 두 시드).
      이 항은 중복이 아니라 파생이지만 확인된 바 없으므로 seed 2 개로 검증한다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return S.goal_pos_w(env, command_name, robot_cfg) - obj.data.root_pos_w


def cup_upright(env: "ManagerBasedRLEnv",
                object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """컵 로컬 +z 의 world z 성분 (N, 1). `R_upright` 와 짝을 이룬다.

    보상에 직립을 요구하면서 관측에 주지 않으면 정책이 그 항을 제어할 수 없다.
    실기에서는 `/cup_pose` 의 자세 성분으로 얻는다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return matrix_from_quat(obj.data.root_quat_w)[:, 2, 2].unsqueeze(-1)
