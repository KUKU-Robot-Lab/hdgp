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

import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import grasp_left_fabric_action as fab
from . import grasp_left_observations as obs
from . import grasp_left_preset as P
from .grasp_left_env_cfg import GraspLeftGripperEnvCfg


@configclass
class GraspLeftGripperFabEnvCfg(GraspLeftGripperEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이송 — Fabrics 팔 제어."""

    def __post_init__(self):
        super().__post_init__()
        # 팔 액션만 교체. 그리퍼(BinaryJointPositionAction 양조 지령)는 부모 그대로.
        self.actions.arm_action = fab.FabricPalmActionCfg()
        # ★★fab_test61(사용자 지시): GUI 학습 중 **정책이 실제로 내는 지령**을 본다.
        #   env 0 만 그린다(마커 콜백에서 슬라이스). 큰 프레임 = palm 6D 지령,
        #   작은 프레임 = 이송 목표. 레퍼런스 `object_pose` 마커는 전 env 에 TCP/goal 을
        #   그려 화면을 덮으므로 끈다 — 지령과 실제의 어긋남이 보이지 않게 된다.
        self.actions.arm_action.debug_vis = True
        self.commands.object_pose.debug_vis = False

        # ★태스크공간 추종에는 단단한 PD 가 필요하다(레퍼런스 HIGH_PD 패턴, IK 변형과 동일).
        #   80/4 로는 중력 처짐이 fabric 목표를 삼킨다 — G2 실측: 관절오차 j4=45 mrad
        #   (τ = 80×0.045 ≈ 3.6 N·m = 중력토크), TCP 44.8 mm. 400/80 이면 처짐이 1/5.
        #   ⚠ 레퍼런스 HIGH_PD 는 disable_gravity=True 도 켜지만 우리는 **중력을 켠 채** 간다
        #     — 실기에는 중력이 있다. 남는 소량의 처짐은 정책이 절대 목표를 보정해 흡수한다
        #     (관절공간 test17 이 같은 방식으로 성공했다).
        self.scene.robot.actuators["left_arm"].stiffness = P.ARM_IK_STIFFNESS
        self.scene.robot.actuators["left_arm"].damping = P.ARM_IK_DAMPING

        # ★★fab_test68: 절대 태스크공간 액션에 **피드백**을 준다(fab_test67 실측 근거).
        #   t67 은 리프트 후 y 액션의 99.7% · z 의 99.9% 가 ±1 밖에 있었다(mu y=3.11,
        #   z=2.04). 목표에 가려면 필요한 액션은 y=-0.15 · z=+0.68 로 박스 **안쪽**인데
        #   정책은 반대 모서리를 상시로 때렸고, 그래서 컵이 목표보다 126 mm 위·212 mm
        #   옆에서 멈췄다. 원인 둘 중 하나가 이것이다 — 정책이 절대 좌표를 지시하면서
        #   자기 palm 이 어디 있는지도, 리미터가 지령을 어디까지 옮겼는지도 못 봤다.
        #   (다른 하나는 `mu_activation` — yaml 에서 tanh 로 막았다.)
        #   ⚠ obs 차원이 +15 된다 → **fresh 학습 전용**. 이전 체크포인트와 호환되지 않는다.
        #   ★★fab_test70/71 (귀속 실험): 어느 항이 학습을 죽였는지 가른다.
        #     t68(51D·tanh) 과 t69(51D·linear) 가 **동일하게** lift 0.00 으로 죽었고,
        #     두 실패에 공통이면서 t67 성공에 없는 것은 obs 추가 하나뿐이다.
        #     유력 용의자는 `palm_cmd` — 리미터 적분기, 즉 **팔이 실제로 추종하는 상태**라
        #     `a_t = palm_cmd_{t-1}` 항등사상이 곧 **완전 정지**라는 안정 고정점이 된다.
        #     실패 형태가 정확히 그것이었다(에피소드 길이 245~248 만렙 = 컵을 건드린 적 없음).
        #     `last_action` 은 raw 라 복사해도 리미터가 계속 적분해 팔이 움직인다 — 다르다.
        #   ★★귀속 완료(t70 vs t71, 각 ep479/476). **pose 가 이겼다** — 기본값이다.
        #     정점 기준  t70(45D) lift 11.89 · goal 6.47 · fine 0.57 · best 132.2
        #                t71(42D) lift 12.39 · goal 4.93 · fine 0.28 · best 119.0
        #     리프트는 t71 이 높은데 goal tracking 은 t70 이 **두 배**다. 리프트는
        #     "높이 들면 끝"이라 축별 조건부 표현이 필요 없고, 이송만 그걸 요구한다.
        #     t71 은 mu 가 x 2.5 · y -4.04 · z 2.05 로 표류해 세 축이 다 포화(87~96%)했고
        #     fine 이 400 epoch 내내 0.22 에서 평평했다 = t67 병리의 재현.
        #   HDGP_OBS_SET: pose(기본·tcp_pos+palm_rot) | cmd(palm_cmd 만) | all(셋 다)
        #     ⚠ all(51D)은 t68·t69 가 lift 0.00 으로 죽은 조합이다. 지령과 실측은
        #       서로 공선(리프트 중 지령↔TCP 50 mm)이라 둘을 다 주면 안 된다.
        _obs_set = os.environ.get("HDGP_OBS_SET", "pose")
        if _obs_set not in ("all", "pose", "cmd"):
            raise ValueError(f"HDGP_OBS_SET 은 all|pose|cmd — 받은 값: {_obs_set!r}")
        if _obs_set in ("all", "cmd"):
            self.observations.policy.palm_cmd = ObsTerm(func=obs.palm_command)
        if _obs_set in ("all", "pose"):
            self.observations.policy.tcp_pos = ObsTerm(func=obs.tcp_position_in_root)
        #   ⚠ `SceneEntityCfg` 는 **params 에 넣어야** 매니저가 resolve 한다. 기본 인자로
        #     두면 `body_ids` 가 slice(None) 인 채로 들어와 인덱싱에서 죽는다(실측).
        if _obs_set in ("all", "pose"):
            self.observations.policy.palm_rot = ObsTerm(
                func=obs.palm_rot6d_in_root,
                params={"robot_cfg": SceneEntityCfg("robot",
                                                    body_names=[P.GRIPPER_BASE_BODY])},
            )


@configclass
class GraspLeftGripperFabEnvCfg_PLAY(GraspLeftGripperFabEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
