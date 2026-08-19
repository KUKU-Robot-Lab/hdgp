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

"""환경 설정: gripper/left/grasp_sensor — 왼팔 2지 그리퍼 단일 컵 파지.

- 로봇: openarm_tesollo_sensor_rl (왼팔 7 DOF + 2지 그리퍼, 오른팔은 rest 고정)
- Action 7D: TCP 6D delta (Fabrics IK) + 그리퍼 1D
- Observation: actor 48D / critic 62D (asymmetric)
- 물체: **shaker 단일 종** — MultiAsset 아님, onehot 없음
- Episode: 10s = 600 step @60Hz (접근·파지 480 + 리프트 120)

right/grasp_sensor 와의 차이는 전부 "손이 다르다"에서 나온다. 상세는 각 필드 주석 참조.
"""

from dataclasses import field

import os as _os

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from openarm import OPENARM_ROOT_DIR

from .grasp_left_constants import (
    NUM_ACTIONS,
    NUM_CRITIC_OBSERVATIONS,
    NUM_OBSERVATIONS,
)
from .grasp_left_preset import (
    CUP_USD_NAME,
    TABLE_POS,
    CUP_SPAWN_X_CENTER,
    CUP_SPAWN_X_RANGE,
    CUP_SPAWN_Y_CENTER,
    CUP_SPAWN_Y_RANGE,
    CUP_SPAWN_Z,
    GRASP_HEIGHT_ABOVE_TABLE,
    GRIPPER_OPEN_POS,
    LEFT_ARM_HOME_JOINT_POS,
    RIGHT_REST_JOINT_POS,
    TABLE_SURFACE_Z,
    GRIPPER_FINGER_BODIES,
)

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")

# 컵 질량 [kg]. right/grasp_sensor 전 물체 공통값과 동일해야 warm 상태의 force-ratio 가 맞는다.
_CUP_MASS: float = 0.134


@configclass
class EventCfg:
    """컵 physics DR (매 reset per-env).

    ★`randomize_rigid_body_material` 은 **절대 물성값**을 샘플링한다(배율이 아니다).
      right/grasp_sensor 는 이 값들을 "초기=중립(1.0배)"으로 주석해 뒀는데, 그렇게 읽으면
      restitution 1.0 = **완전 탄성 반발**을 넣게 된다. 실제로 이 태스크에서 컵이 팔에
      닿지도 않은 채 매 스텝 미끄러져 200스텝에 44mm 드리프트했다(Isaac 실측).
      마찰 1.0 은 계수로도 타당하지만 restitution 은 0 이어야 컵이 테이블에 가만히 앉는다.
    """

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            # ⚠ 절대값. 1.0 은 완전 탄성이라 컵이 계속 튄다 — 0 으로 둔다.
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class GraspLeftGripperEnvCfg(DirectRLEnvCfg):
    """왼팔 2지 그리퍼 컵 파지 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터 (right/grasp_sensor 와 동일 — 물리 120Hz / 정책 60Hz)
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 48
    action_space: int = NUM_ACTIONS                    # 7
    state_space: int = NUM_CRITIC_OBSERVATIONS         # 62

    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics
    # -----------------------------------------------------------------------
    # ★손 20관절을 fixed 로 동결한 좌팔 전용 URDF → cspace = 팔 7 DOF.
    #   그리퍼 개폐는 Fabrics 가 아니라 RL 액션이 직접 관절 목표를 준다.
    #   생성: scripts/assets_tools/generate_sensor_left_gripper_fabric_urdf.py
    fabric_robot_dir: str = "openarm_tesollo_sensor_left_gripper"
    use_hand_fabric: bool = False
    fabrics_max_objects_per_env: int = 8
    fabrics_damping_gain: float = 20.0
    pregrasp_fabric_steps: int = 60
    reset_fabric_chunk_size: int = 128

    # -----------------------------------------------------------------------
    # 리셋 / 액션 기준점
    # -----------------------------------------------------------------------
    # 물리 리셋은 **고정 홈**(preset LEFT_ARM_HOME_JOINT_POS)이다. 우측과 달리 홈을 IK 로
    # 풀지 않는다 — 이미 측정된 관절값이라 그대로 쓴다(IK 왕복은 오차만 더한다).
    # 액션 기준점은 홈이 아니라 **컵-정준 pregrasp** (right/grasp_sensor A0 규약):
    #   action=0 이면 Fabrics 가 홈에서 pregrasp 까지 스스로 접근하고,
    #   정책은 마지막 진입·정렬·폐쇄만 학습한다.
    reset_from_fixed_home: bool = True
    pregrasp_pos_noise: tuple = (0.01, 0.01, 0.005)   # m, pregrasp 기준점 관측 노이즈 대역

    # TCP delta action 범위. 축 균일 — pregrasp 가 이미 컵 옆이므로 delta 는
    # 미세조정·잔차 보정 용도다(우측 A0 와 동일 판단).
    palm_delta_xyz: tuple = (0.15, 0.15, 0.15)   # ±m
    palm_delta_rot_deg: float = 20.0             # ±° per euler axis

    # -----------------------------------------------------------------------
    # 컵 스폰
    # -----------------------------------------------------------------------
    # ★x 중심 0.25 는 우측(0.30)의 미러가 아니다 — preset 주석 참조(그리퍼가 컵을 낮은
    #   높이에서만 잡을 수 있어 파지점이 우측보다 낮고, x=0.30 에서는 팔이 못 미친다).
    object_spawn_x_center: float = CUP_SPAWN_X_CENTER
    object_spawn_y_center: float = CUP_SPAWN_Y_CENTER
    object_spawn_z: float = CUP_SPAWN_Z
    object_spawn_x_range: float = CUP_SPAWN_X_RANGE
    object_spawn_y_range: float = CUP_SPAWN_Y_RANGE
    table_surface_z: float = TABLE_SURFACE_Z
    grasp_height_above_table: float = GRASP_HEIGHT_ABOVE_TABLE

    # -----------------------------------------------------------------------
    # 파지 / 성공 판정
    # -----------------------------------------------------------------------
    # ⚠ 게이트를 완화하지 말 것. 1지 접촉 래치를 허용하면 부실 파지 국소최적이 생긴다.
    lift_start_min_contacts: int = 2      # 래치 진입: 두 핑거 모두 컵에 접촉
    success_min_contacts: int = 2
    grasp_ready_hold_steps: int = 8       # 접촉을 연속 유지해야 래치
    lift_success_height: float = 0.04     # m, 성공 판정 상승 높이
    lift_height_ref: float = 0.10         # m, lift 보상 정규화 기준(성공 임계와 분리)
    lift_height_delta: float = 0.10       # m, 래치 후 TCP z 램프 총량
    success_upright_max_deg: float = 20.0
    stabilize_upright_max_deg: float = 5.0
    stability_action_delta_threshold: float = 0.2
    cup_xy_disp_limit: float = 0.08       # m, 밀림 soft 감쇠 분모

    # -----------------------------------------------------------------------
    # Reward weights (right/grasp_sensor 계승 — reward-audit ACCEPT 2026-08-19)
    # -----------------------------------------------------------------------
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 25.0
    grasp_xy_threshold: float = 0.025
    approach_tilt_penalty_weight: float = 0.08
    grasp_upright_threshold_deg: float = 8.0

    grasp_weight: float = 12.0
    # 2지 고유 배분. 합이 1.0 을 넘지 않게 유지할 것 — grasp 최대치가 곧 12.0 이다.
    grasp_opposition_credit: float = 0.25   # 두 접촉점이 컵 단면 지름 양끝인가
    grasp_squeeze_credit: float = 0.20      # 실제로 힘이 걸렸는가(접촉 게이트 적용됨)

    lift_reward_weight: float = 30.0
    stabilize_weight: float = 10.0
    stabilize_action_sharpness: float = 5.0
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02

    # -----------------------------------------------------------------------
    # 접촉 센서
    # -----------------------------------------------------------------------
    # ⚠ `l_hl_gripper_tcp` 는 physics USD 에 강체로 존재하지 않는다 → 센서 대상 불가.
    #    두 핑거 링크만 건다.
    gripper_finger_contact_links: tuple = GRIPPER_FINGER_BODIES
    # 컵만 필터(Cup-only). 무필터면 핑거가 테이블·자기 몸에 닿아도 grip 으로 잡혀
    # 거짓 성공이 생긴다(우측에서 실증된 버그).
    # ⚠ 루트 prim("/Cup")은 필터로 쓸 수 없다 — PhysX GPU 파이프라인이 지원하지 않아
    #   env 마다 "GPU contact filter for collider ... is not supported" 경고만 쏟는다(실측).
    #   강체가 실제로 붙어 있는 **baseLink** 를 직접 건다(visdex 표준 구조).
    object_contact_filter: tuple = (
        "/World/envs/env_.*/Cup/baseLink",
    )

    # -----------------------------------------------------------------------
    # 물리 시뮬레이션
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    events: EventCfg = field(default_factory=EventCfg)

    # 단일 물체이므로 physics 복제가 가능하다(우측은 MultiAsset 이라 False 강제).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -----------------------------------------------------------------------
    # 테이블
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        # ★기본 위치(x 0.5725)면 판이 x>=0.310 이라 이 팔의 파지 영역이 판 밖이다.
        #   preset 에서 당겨온 위치를 쓴다 — tests 가 스폰 박스가 판 위인지 검사한다.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(TABLE_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    # -----------------------------------------------------------------------
    # 컵 — **shaker 단일 종, scale 1.0**
    # -----------------------------------------------------------------------
    # ★스케일을 줄이지 않는다. shaker 는 계단형 원뿔이라 몸통이 58/68/78/88 mm 로 단계적으로
    #   굵어지고, 그리퍼 최대 개구 84.5 mm 로 **테이블 위 10~85 mm 구간을 통과**한다
    #   (채택 h=65 mm 에서 통과지름 68 mm, 편측 여유 8.2 mm — probe_gripper_opening.py).
    #   bbox 지름 88 mm 만 보고 "개구보다 크다"고 판단하면 잡을 수 있는 컵을 버린다.
    # ★질량은 USD 기본값(shaker 0.263 kg)이 아니라 여기서 고정한다 — 자산마다 다른 기본질량을
    #   그대로 쓰면 ADR 질량 배율이 자산별로 다른 절대 구간으로 튄다(우측에서 실증된 문제).
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[CUP_SPAWN_X_CENTER, CUP_SPAWN_Y_CENTER, CUP_SPAWN_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup", CUP_USD_NAME),
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_CUP_MASS),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(
                _ASSETS_DIR, "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                **LEFT_ARM_HOME_JOINT_POS,
                "l_hj_gripper_1": GRIPPER_OPEN_POS,
                "l_hj_gripper_2": GRIPPER_OPEN_POS,
                **RIGHT_REST_JOINT_POS,
            },
        ),
        actuators={
            # head pan/tilt: revolute 라 DOF 로 잡힌다 → 커버리지 없으면 자유회전.
            "head_camera": ImplicitActuatorCfg(
                joint_names_expr=["head_j_(pan|tilt)"],
                stiffness=400.0,
                damping=80.0,
            ),
            # 왼팔(파지 팔): 우측 팔의 07.29 캘리브 friction 을 좌우 대칭으로 그대로 적용.
            # group 경계(l_aj_[1-3] / 4 / [5-7])는 assets/robot/CALIBRATION.md 규약.
            "left_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-3]"], stiffness=400.0, damping=80.0, friction=0.213,
            ),
            "left_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_4"], stiffness=400.0, damping=80.0, friction=0.493,
            ),
            "left_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[5-7]"], stiffness=400.0, damping=80.0, friction=0.151,
            ),
            # 그리퍼(파지 자유도). 우측 태스크의 400/80 은 **유휴 hold 용** 값이라 그대로 쓰면 안 된다 —
            # 여기서는 실제로 컵을 쥐어야 하므로 힘 영역을 정한다.
            #   stiffness 2000 N/m × 오차 5 mm ≈ 10 N 파지력, effort_limit 70 N 은 OpenArm hand
            #   급 평행 그리퍼의 상용 최대 파지력 대역.
            # ⚠ 이 값은 잠정치다. scripts/probes/probe_gripper_grip_force.py (P0-3) 로
            #   **포화율 < 20%** 를 확인한 뒤 확정할 것 — 우측 손에서 effort 포화(80.6%)가
            #   파지력 제어를 통째로 무효화한 이력이 있다.
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_gripper_[1-2]"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=70.0,
            ),
            # 유휴 오른팔·오른손: rest 유지만.
            "idle_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"], stiffness=400.0, damping=80.0,
            ),
            "idle_right_hand": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_[1-4]"], stiffness=5.0, damping=2.0,
                effort_limit_sim=1.5,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
