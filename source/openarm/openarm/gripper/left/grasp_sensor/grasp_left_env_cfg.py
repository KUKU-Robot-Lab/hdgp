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

"""환경 설정: gripper/left/grasp_sensor — 왼팔 2지 그리퍼로 shaker 집어 옮기기.

IsaacLab `Isaac-Lift-Cube-OpenArm-v0` 레시피를 그대로 물려받고 **로봇·물체·씬만** 바꾼다.
그래서 이 파일이 짧다 — 보상·관측·커맨드·이벤트·커리큘럼은 손대지 않는다.

왜 이 방식인가
--------------
처음에는 우측 다지 손 태스크(Direct RL + Fabrics + 정확 6D TCP 포즈 attractor)를 이식했다가
막혔다. 2지 그리퍼는 jaw 가 수평이어야 파지가 성립해 팔에 특정 6-DOF 자세를 강제하는데,
이 팔은 손목 j6 가 ±45°·손목 3축 effort 가 7 N·m 뿐이라 낼 수 있는 자세가 얇은 곡선이다.
거기에 "정확한 포즈를 내라"는 가장 빡빡한 제어를 얹은 셈이었다(실측 자세 오차 28°, j5 한계 고착).

lift 레시피는 정반대다:
  · 팔 = 관절 위치 델타(JointPositionAction) → 정책이 내는 모든 액션이 항상 유효한 타깃
  · 그리퍼 = 이진 스칼라(BinaryJointPositionAction) → 파지력·개도를 학습할 필요 없음
  · 보상에 **회전 항이 하나도 없다** → "자세 도달성" 문제가 발생할 지점이 없다

★바꾸지 말 것: scale=0.5, use_default_offset=True, BinaryJointPositionAction,
  커리큘럼 2개, decimation=2 / episode_length_s=5.0, reward weight 조합.
  이것들이 이 레시피가 학습되는 이유다.
"""

from __future__ import annotations

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import (
    LiftEnvCfg,
)

from openarm import OPENARM_ROOT_DIR

from . import grasp_left_preset as P

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspLeftGripperEnvCfg(LiftEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이동."""

    def __post_init__(self):
        super().__post_init__()

        # ── 로봇 ────────────────────────────────────────────────────
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_os.path.join(
                    _ASSETS_DIR,
                    "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd",
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # ★중력 켠 채 학습한다. 우측 태스크와 IK 경로는 disable_gravity 를 쓰지만
                    #   그건 포즈 추종을 위한 타협이라 실기 이식성을 해친다.
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    # ★액션 0 = 이 자세다(use_default_offset). 파지 준비 자세여야 한다.
                    **P.LEFT_ARM_HOME_JOINT_POS,
                    P.GRIPPER_JOINT_NAMES[0]: P.GRIPPER_OPEN_POS,
                    P.GRIPPER_JOINT_NAMES[1]: P.GRIPPER_OPEN_POS,
                    **P.RIGHT_REST_JOINT_POS,
                },
            ),
            actuators={
                # 팔: IsaacLab OpenArm 값(80/4). ★400/80 은 IK 추종용이라 쓰지 않는다.
                "left_arm": ImplicitActuatorCfg(
                    joint_names_expr=["l_aj_[1-7]"], stiffness=80.0, damping=4.0,
                ),
                # 그리퍼: 두 관절 모두 커버리지를 준다(없으면 무구동 자유이동).
                # 지령은 gripper_1 에만 간다 — mimic 과 싸우지 않게.
                "left_gripper": ImplicitActuatorCfg(
                    joint_names_expr=["l_hj_gripper_[1-2]"],
                    velocity_limit_sim=0.2,
                    effort_limit_sim=333.33,
                    stiffness=2e3,
                    damping=1e2,
                ),
                # 유휴 오른팔·오른손: rest 유지만.
                "idle_right_arm": ImplicitActuatorCfg(
                    joint_names_expr=["r_aj_[1-7]"], stiffness=400.0, damping=80.0,
                ),
                "idle_right_hand": ImplicitActuatorCfg(
                    joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                    stiffness=5.0, damping=2.0, effort_limit_sim=1.5,
                ),
                "head_camera": ImplicitActuatorCfg(
                    joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        # ── 액션 ────────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["l_aj_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )
        # ⚠ gripper_2 는 USD PhysX mimic 이라 지령 대상에서 뺀다. BinaryJointPositionAction 은
        #   joint_names 가 하나라도 안 풀리면 ValueError 로 즉사하므로 정규식이 아니라 정확한
        #   이름을 쓴다.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[P.GRIPPER_DRIVE_JOINT],
            open_command_expr={P.GRIPPER_DRIVE_JOINT: P.GRIPPER_OPEN_POS},
            close_command_expr={P.GRIPPER_DRIVE_JOINT: P.GRIPPER_CLOSED_POS},
        )

        # ── 씬: 테이블 (로컬 자산) ──────────────────────────────────
        # 레퍼런스는 클라우드 Nucleus 의 SeattleLabTable 을 쓰는데 이 머신에 캐시가 없다.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=list(P.TABLE_POS), rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True,
                ),
            ),
        )
        # 바닥면: 레퍼런스는 테이블 상면이 z=0 이라 -1.05 에 뒀다. 우리 상면은 0.215 이고
        # 로봇 베이스가 바닥에 서 있으므로 z=0 이 맞다.
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)

        # ── 씬: 물체 (shaker) ──────────────────────────────────────
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_Y_CENTER, P.CUP_SPAWN_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=_os.path.join(_ASSETS_DIR, "cup", P.CUP_USD_NAME),
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=P.CUP_MASS),
            ),
        )

        # ── EE 프레임 (보상 계산용, 액션과 무관) ────────────────────
        # `l_hl_gripper_tcp` 는 physics USD 에 강체로 없다 → base + z 오프셋으로 TCP 를 만든다
        # (Franka 가 panda_hand + offset 0.1034 를 쓰는 것과 같은 패턴).
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{P.GRIPPER_BASE_BODY}",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z)),
                ),
            ],
        )

        # ── 물체 스폰 랜덤화 ────────────────────────────────────────
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_X_RANGE),
            "y": (-P.CUP_SPAWN_Y_RANGE, P.CUP_SPAWN_Y_RANGE),
            "z": (0.0, 0.0),
        }

        # ── 목표 커맨드 ────────────────────────────────────────────
        self.commands.object_pose.body_name = P.GRIPPER_BASE_BODY
        self.commands.object_pose.ranges.pos_x = P.GOAL_POS_X
        self.commands.object_pose.ranges.pos_y = P.GOAL_POS_Y
        self.commands.object_pose.ranges.pos_z = P.GOAL_POS_Z

        # ── 관측: 왼팔 관절만 ──────────────────────────────────────
        # 기본값은 로봇 전체(오른팔 27관절 포함)라, 이 팔이 못 건드리는 값이 관측을 채운다.
        # ★term 마다 **새 인스턴스**를 만든다. SceneEntityCfg 는 매니저가 resolve() 로
        #   제자리 변경(joint_ids 를 채워 넣음)하는 가변 객체다. 하나를 공유하면 두 번째
        #   term 에서 "joint_names 와 joint_ids 가 불일치" 로 env 생성이 죽는다(실측).
        def _left_joints() -> SceneEntityCfg:
            return SceneEntityCfg(
                "robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"]
            )

        self.observations.policy.joint_pos.params["asset_cfg"] = _left_joints()
        self.observations.policy.joint_vel.params["asset_cfg"] = _left_joints()
        self.rewards.joint_vel.params["asset_cfg"] = _left_joints()

        # ── 리프트 임계: 절대 world z ──────────────────────────────
        # ★레퍼런스는 테이블 상면이 z=0 이라 0.04 를 쓴다. 우리 상면은 0.215 이므로
        #   그만큼 올려야 "4cm 들어올림"이 된다. 안 그러면 컵이 놓인 채로도 lifting 보상이
        #   상시 1 이 되어 학습이 통째로 망가진다.
        self.rewards.lifting_object.params["minimal_height"] = P.MINIMAL_LIFT_HEIGHT
        self.rewards.object_goal_tracking.params["minimal_height"] = P.MINIMAL_LIFT_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = (
            P.MINIMAL_LIFT_HEIGHT
        )
        self.terminations.object_dropping.params["minimum_height"] = P.OBJECT_DROP_HEIGHT


@configclass
class GraspLeftGripperEnvCfg_PLAY(GraspLeftGripperEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
