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
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
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

from . import grasp_left_events as events
from . import grasp_left_preset as P
from . import grasp_left_rewards as rewards

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
                # 팔: IsaacLab OpenArm 값. ★400/80 은 IK 추종용이라 쓰지 않는다.
                # ★★`velocity_limit_sim` 을 반드시 함께 준다. 레퍼런스 `OPENARM_UNI_CFG` 가
                #   명시하는데 처음 이식할 때 빠뜨렸고, 그러면 USD/URDF 기본값(5.4~20.9 rad/s,
                #   레퍼런스의 2.5~9.6 배)이 쓰인다. damping 이 4 뿐이라 팔이 과속으로 오버슈트
                #   하며 흔들리고("시작할 때 진자처럼 흔들린다"는 렌더 관찰), TCP 로 컵을
                #   정조준할 수 없게 된다. 20.9 rad/s 면 한 스텝(0.02 s)에 0.42 rad — 액션
                #   범위(±0.5 rad)를 한 스텝에 소화해 버린다. 2.175 면 0.0435 rad 로 부드럽다.
                "left_arm": ImplicitActuatorCfg(
                    joint_names_expr=["l_aj_[1-7]"],
                    velocity_limit_sim=P.ARM_VELOCITY_LIMIT,
                    effort_limit_sim=P.ARM_EFFORT_LIMIT,
                    stiffness=80.0,
                    damping=4.0,
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
                # 유휴 오른팔·오른손: rest 자세 유지만 하면 된다.
                # ★★`effort_limit_sim` 을 반드시 올린다. URDF 의 팔 effort 는
                #   j1/j2=40, j3/j4=27, **j5~j7=7 N·m** 뿐이라 stiffness 400 이 무의미하게
                #   포화하고, 20 관절 손(약 1.4 kg)을 단 오른팔이 중력에 그대로 처진다.
                #   실측(프로브 1a): 관절 오차 최대 49.9°·평균 27°, 손끝이 테이블 상면
                #   바로 위(0.223)까지 내려와 **테이블에 얹힌다**. 렌더에서 사용자가 지적한
                #   "오른팔이 바닥에 닿아 있다"가 이것이다.
                #   이 팔은 학습에 쓰이지 않는 배경이고 실기로 배포되지도 않으므로,
                #   sim 에서 자세만 고정되면 된다.
                "idle_right_arm": ImplicitActuatorCfg(
                    joint_names_expr=["r_aj_[1-7]"],
                    stiffness=400.0, damping=80.0, effort_limit_sim=1000.0,
                ),
                # 유휴 오른손도 같은 이유로 올린다. effort 1.5 는 실기 정합값이지만
                # 그건 **파지를 학습하는 손**에 필요한 것이고, 여기 오른손은 배경이다.
                "idle_right_hand": ImplicitActuatorCfg(
                    joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                    stiffness=20.0, damping=4.0, effort_limit_sim=50.0,
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
        # ★asset_cfg 를 반드시 교체한다. 레퍼런스는 큐브 prim 이름 `"Object"` 를 박아 두는데
        #   우리 shaker 의 강체는 `baseLink` 라, 매니저가 이름을 resolve 하는 순간 죽는다.
        #   이 이벤트는 root state 만 쓰므로 body_names 자체가 불필요하다.
        #   ⚠ 로컬에서는 sim 이 playing 이 아닌 타이밍이라 resolve 가 스킵돼 통과하고,
        #     서버 학습 기동에서만 터졌다. 아래 계약 테스트로 고정해 둔다.
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_X_RANGE),
            "y": (-P.CUP_SPAWN_Y_RANGE, P.CUP_SPAWN_Y_RANGE),
            "z": (0.0, 0.0),
        }

        # ── 학습 영상: env 하나만 정면에서 ──────────────────────────
        # 기본 뷰어는 여러 env 가 한 화면에 잡혀 파지 자세를 판별할 수 없다.
        # `origin_type="env"` + `env_index=0` 으로 env 0 에 고정하고, 로봇 정면에서
        # 컵·그리퍼를 바라본다(파지 시 jaw 가 수평인지 보이는 각도).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = P.VIEWER_EYE
        self.viewer.lookat = P.VIEWER_LOOKAT
        self.viewer.resolution = (1280, 720)

        # ── 유휴 관절 자세 고정 ────────────────────────────────────
        # ★★없으면 오른팔이 **차렷으로 내려가 바닥에 닿는다**. init_state 는 관절의 상태만
        #   정하고 PD 목표는 정하지 않는데, 액션 대상이 아닌 관절은 아무도 목표를 써 주지
        #   않아 0 인 채로 남기 때문이다. 자세한 경위는 grasp_left_events 참조.
        self.events.hold_idle_joints = EventTermCfg(
            func=events.hold_joints_at_target,
            mode="reset",
            params={
                "joint_targets": {
                    **P.RIGHT_REST_JOINT_POS,
                    "head_j_pan": 0.0,
                    "head_j_tilt": 0.0,
                },
            },
        )

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

        # ── 리프트 판정에 "쥐고 있는가"를 AND ────────────────────────
        # ★★weight 는 그대로 두고 **판정 함수만** 바꾼다. z 만 보는 레퍼런스 판정으로는
        #   컵을 위로 쳐 날리는 것이 최적 전략이 되기 때문이다(test3 실증: 리프트 판정
        #   85.9% 동안 TCP–컵 평균 3044 mm). 자세한 근거는 grasp_left_rewards 참조.
        #   ⚠ goal-tracking 두 개도 내부에서 z 게이트를 직접 계산하므로 함께 교체해야 한다 —
        #     하나라도 남기면 그쪽으로 같은 hack 이 되살아난다.
        self.rewards.lifting_object.func = rewards.object_is_held_and_lifted
        self.rewards.lifting_object.params["max_ee_distance"] = P.GRASP_MAX_EE_DISTANCE
        self.rewards.lifting_object.params["min_upright_cos"] = P.CUP_UPRIGHT_MIN_COS
        for _term in (
            self.rewards.object_goal_tracking,
            self.rewards.object_goal_tracking_fine_grained,
        ):
            _term.func = rewards.object_goal_distance_when_held
            _term.params["max_ee_distance"] = P.GRASP_MAX_EE_DISTANCE
            _term.params["min_upright_cos"] = P.CUP_UPRIGHT_MIN_COS

        # ── jaw 수평 보너스 (신설) ──────────────────────────────────
        # ★게이트가 아니라 **연속 보너스**다. jaw 수평을 AND 로 넣으면 겨우 붙기 시작한
        #   파지가 한꺼번에 무너진다(reward-audit Check 4). 별도 term 이라 TFEvents 에
        #   자동 로깅돼 "제대로 잡는지"를 학습 중에 관측할 수 있다는 이점도 있다.
        self.rewards.jaw_level = RewTerm(
            func=rewards.held_with_level_jaw,
            weight=P.JAW_LEVEL_REWARD_WEIGHT,
            params={
                "minimal_height": P.MINIMAL_LIFT_HEIGHT,
                "max_ee_distance": P.GRASP_MAX_EE_DISTANCE,
                "min_upright_cos": P.CUP_UPRIGHT_MIN_COS,
                "body_name": P.GRIPPER_BASE_BODY,
            },
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
