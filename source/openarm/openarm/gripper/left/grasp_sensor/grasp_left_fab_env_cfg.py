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

"""Fabrics 팔 액션 변형 — `open-grip_l_grasp_sensor_fab`.

관절공간판(`GraspLeftGripperEnvCfg`, test17 검증)에서 **팔 액션 하나만** Fabrics 절대
palm 6D 로 바꾼다. 보상·씬·커맨드·커리큘럼·물리 플래그는 전부 부모 것 그대로다 —
그래야 test17 과 제어기만 다른 직접 비교가 성립한다(IK 변형 때와 같은 패턴).

왜 바꾸나: test17 은 이송까지 성공했지만 목표에서 못 멈춘다(잔류 0.17 m/s). 원인은
정책 raw 지령의 상시 포화이고, Fabrics 는 그 진동을 2차 적분으로 흡수한다.
근거 전문은 `grasp_left_fabric_action.py` docstring.

액션 차원 8 → **7** (팔 6D + 그리퍼 1). obs 의 `last_action` 은 자동 적응(36→35D).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from . import grasp_left_fabric_action as fab
from . import grasp_left_preset as P
from .grasp_left_env_cfg import GraspLeftGripperEnvCfg


@configclass
class GraspLeftGripperFabEnvCfg(GraspLeftGripperEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이송 — Fabrics 팔 제어."""

    def __post_init__(self):
        super().__post_init__()
        # 팔 액션만 교체. 그리퍼(BinaryJointPositionAction 양조 지령)는 부모 그대로.
        self.actions.arm_action = fab.FabricPalmActionCfg()

        # ★태스크공간 추종에는 단단한 PD 가 필요하다(레퍼런스 HIGH_PD 패턴, IK 변형과 동일).
        #   80/4 로는 중력 처짐이 fabric 목표를 삼킨다 — G2 실측: 관절오차 j4=45 mrad
        #   (τ = 80×0.045 ≈ 3.6 N·m = 중력토크), TCP 44.8 mm. 400/80 이면 처짐이 1/5.
        #   ⚠ 레퍼런스 HIGH_PD 는 disable_gravity=True 도 켜지만 우리는 **중력을 켠 채** 간다
        #     — 실기에는 중력이 있다. 남는 소량의 처짐은 정책이 절대 목표를 보정해 흡수한다
        #     (관절공간 test17 이 같은 방식으로 성공했다).
        self.scene.robot.actuators["left_arm"].stiffness = P.ARM_IK_STIFFNESS
        self.scene.robot.actuators["left_arm"].damping = P.ARM_IK_DAMPING


@configclass
class GraspLeftGripperFabEnvCfg_PLAY(GraspLeftGripperFabEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
