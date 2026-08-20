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

"""태스크공간(diff-IK) 변형 — 팔 액션만 바꾸고 나머지는 관절공간판과 **완전히 동일**하다.

왜 만드나. 이 태스크의 요구는 6-DOF 자세 구속이다 — "컵을 **똑바로** 들고", "jaw 가
**수평**", "TCP z ⊥ 컵 z = 90°", "목표점에서 **정지**". 그런데 레퍼런스 lift 의 보상에는
회전 항이 하나도 없다. 관절공간 액션으로 이걸 시키면 정책이 7 관절 비선형 사상을 통해
자세 다양체를 **부수적으로** 학습해야 한다. 실측이 그 부담을 보여준다(동일 cfg, 결정론):

    지표                 test12   test13
    TCP z ↔ 컵 z         82.6°    73.9°   ← 목표 90°, 오히려 멀어졌다
    컵 기울기            43.8°    28.5°
    jaw 수평 이탈        20.5°    23.3°   ← 나빠졌다
    목표 10 cm 내 컵속도 0.682    0.794 m/s

보상 셰이핑 세 번(test8/12/13)으로 이 정도다. 태스크공간에서는 이 양들이 액션의 직접적
함수라 "가만히 있기" = 0 지령 한 점이고, 관절공간에서는 7 개 값이 동시에 고정돼야 한다.
탐색 노이즈도 마찬가지다 — σ≈1 이면 관절당 ±0.5 rad 이고 레버암을 타고 TCP 에서 수십 cm
로 증폭된다. σ 가 꺼질 때까지(약 epoch 1300) 정책은 "정지"를 샘플링조차 못 한다.

★이건 폐기한 Fabrics 경로가 아니다.
  Fabrics = 로봇당 수제 자산 4 종 + 정상상태 droop 22 mm + 오픈루프. diff-IK(dls) =
  PhysX Jacobian 만 쓰고 사전 자산 0, agnostic 트랙에서 같은 로봇에 **추종 2.8 mm** 실측.
  그리고 레퍼런스 자체가 같은 lift 태스크의 IK 변형을 배포한다
  (`lift/config/franka/ik_rel_env_cfg.py` — 팔 액션만 갈아끼우는 22 줄짜리 서브클래스).

바뀌는 것은 셋뿐이다.
  1. 팔 액션: 관절 7D → **TCP 상대 pose 6D**. 총 액션 8 → 7, 관측 36 → 35.
  2. 팔 PD: 80/4 → **400/80**. 레퍼런스가 IK 경로에서 명시적으로 요구한다.
  3. 변화율 상한이 관절이 아니라 **스케일**로 들어간다(`IK_ACTION_SCALE`).
보상·씬·목표·커리큘럼·에피소드는 전부 관절공간판과 같다 — 그래야 비교가 성립한다.
"""

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from . import grasp_left_actions as actions
from . import grasp_left_preset as P
from .grasp_left_env_cfg import GraspLeftGripperEnvCfg, GraspLeftGripperEnvCfg_PLAY


def _apply_ik_arm(cfg: GraspLeftGripperEnvCfg) -> None:
    """팔 액션과 팔 PD 만 태스크공간용으로 교체한다."""
    # ★IK 추종용 PD. 레퍼런스 HIGH_PD 는 `disable_gravity=True` 도 켜지만 우리는 켜지
    #   않는다 — 실기에는 중력이 있고, 끄면 sim2real 이 무효가 된다.
    arm = cfg.scene.robot.actuators["left_arm"]
    arm.stiffness = P.ARM_IK_STIFFNESS
    arm.damping = P.ARM_IK_DAMPING

    cfg.actions.arm_action = actions.JointLimitedDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=["l_aj_[1-7]"],
        body_name=P.GRIPPER_BASE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=P.IK_ACTION_SCALE,
        # ★TCP 변위 상한(scale)만으로는 부족하다 — IK 가 그걸 관절로 푸는 단계가
        #   안 묶여 자코비안 조건이 나쁜 자세에서 관절 속도 한계까지 포화한다
        #   (test4 실측 2.17 rad/s = 한계, 방향 반전 49.3%). 관절공간 판과 같은 표를 쓴다.
        rate_limit=P.ARM_TARGET_RATE_LIMIT,
        max_tracking_error=P.ARM_IK_MAX_TRACKING_ERROR,
        # TCP 는 실제 링크가 아니라 gripper_base 에서 z 로 띄운 프레임이다(보상의 EE
        # 프레임과 같은 정의를 써야 "보상이 보는 점"과 "제어하는 점"이 일치한다).
        body_offset=actions.JointLimitedDifferentialIKActionCfg.OffsetCfg(
            pos=(0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z)
        ),
    )


@configclass
class GraspLeftGripperIKEnvCfg(GraspLeftGripperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_arm(self)


@configclass
class GraspLeftGripperIKEnvCfg_PLAY(GraspLeftGripperEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_ik_arm(self)
