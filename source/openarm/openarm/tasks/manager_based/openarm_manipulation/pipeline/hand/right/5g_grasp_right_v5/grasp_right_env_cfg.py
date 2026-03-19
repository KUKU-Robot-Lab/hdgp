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

"""환경 설정: 5g_grasp_right_v5

v5: FABRICS pregrasp reset + 정책 손가락 grasp formation + Scripted lift checker.
- Action: 5D (per-finger lerp)
- Observation: actor 101D / critic 125D (asymmetric, critic = actor + 24D privileged)
- Episode: Grasp phase (arm 고정, finger 정책 제어) + Lift phase (arm 보간 상승, finger 고정)
- Contact: 물리 ContactSensor 기반 (fingertip 5개 actor, distal+middle critic)
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v5 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    # Episode: 10s = 600 steps @ 60Hz (9s grasp + 1s lift)
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0   # GRASP_PHASE(9s) + LIFT_PHASE(1s)
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS         # 101 (actor)
    action_space:      int = NUM_ACTIONS              # 5
    state_space:       int = NUM_CRITIC_OBSERVATIONS  # 119 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 6

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 30    # 200 → 60 → 30: GPU idle 감소 (30 step 수렴 충분)
    reset_fabric_chunk_size: int = 128   # reset 전용 소형 Fabrics batch 크기 (env_ids chunk 단위)
    pregrasp_offset_x:     float = -0.06   # palm_link가 cup -X 방향 5cm
                                            # palm_ee = palm_link + local_z(0.04) → palm_ee_x = cup_x - 0.01
                                            # 즉 cup_root_x ≈ palm_ee_x + 0.01 (손가락 뿌리에 컵이 위치)
                                            # 이전 -0.19: 컵이 손가락 끝(rl_dg_3_tip/rl_dg_3_1 사이)에 위치 → 밀침 발생
    pregrasp_offset_y:     float = -0.07   # 컵 -Y 방향 7cm (컵 반경 4.5cm + 여유 2.5cm → 손가락 관통 방지)
    pregrasp_offset_z:     float = 0.00    # cup_root + 7cm → palm z ≈ 컵 높이 중간 (pinky 테이블 여유 확보)
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.06   # cup root → 실제 파지 중심 z offset (cup 높이 17.76cm 중간 ≈ 0.09)
    lift_success_height: float = 0.04    # 성공 판정: cup_z > init_z + 4cm

    # -----------------------------------------------------------------------
    # Reward 파라미터
    #
    # [방향 A: Lift-phase conditioned reward]
    # dense reward (contact/enclosure/opposition)에 phase scale 적용:
    #   Grasp phase: dense × grasp_shaping_scale (방향 안내만, 누적 지배 방지)
    #   Lift  phase: dense × 1.0                 (파지 유지 + 리프트 동시 달성 시 풀 보상)
    # 목표: "터치만 540step" local optimum 탈피 → "파지 후 리프트" 전략 학습
    # -----------------------------------------------------------------------
    # 1. contact_reward: 접촉 유지 (num_contacts / 5)
    contact_reward_weight: float = 2.0

    # 2. contact_delta: 접촉 증가 보상 / 감소 패널티
    #    ADR: 2.0 → 0.5 (접촉 획득 유도 후 감소, 이미 학습됨)
    contact_delta_weight: float = 2.0   # ADR 초기값

    # 3. enclosure: fingertip → cup 거리 기반 exp reward
    #    ADR: 4.0 → 8.0 (점점 강화)
    enclosure_weight:    float = 4.0    # ADR 초기값 (2.0 → 4.0, 더 강한 enclosure 압력)
    enclosure_sharpness: float = 15.0  # exp(-sharpness * mean_dist)  (10.0 → 15.0)

    # 4. opposition: thumb + 다른 손가락 동시 접촉
    opposition_weight: float = 2.0     # 5.0 → 2.0 (지배 방지, 67% → ~25% 목표)

    # 5a. grasp_shaping_scale: Grasp phase dense reward 억제 스케일
    #   dense_scale = grasp_shaping_scale + (1 - grasp_shaping_scale) × is_lift_flag
    #   Grasp phase: dense × 0.05  Lift phase: dense × 1.00
    grasp_shaping_scale: float = 0.05

    # 5b. action_reg: ||action||² 패널티 (scale 미적용, 항상 full)
    action_reg_weight: float = -0.005

    # 5c. lift_reward: Lift phase 동안 cup_z - init_z 비례 보상
    #   weight=60: 최대 0.04m 상승 시 2.4/step → Lift phase 120step × 2.4 = 288 total
    lift_reward_weight: float = 60.0   # 20.0 → 60.0

    # 6. terminal rewards
    terminal_success_weight: float = 200.0  # 10.0 → 200.0 (성공이 수지맞는 전략이 되도록)
    terminal_fail_weight:    float = -1.0

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 200
    adr_trigger_threshold: float = 0.02   # 0.1 → 0.02 (success 2% 이상이면 진행)

    adr_custom_cfg: dict = field(default_factory=lambda: {
        "reward_weights": {
            "contact_delta_weight": (2.0, 0.5),   # 접촉 획득 유도 후 감소
            "enclosure_weight":     (4.0, 8.0),   # 점점 강화
        },
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 60.0
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20  # cup_big origin이 cup 중간 → 바닥 기준 z≈0.22, 여유값 0.20

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.40
    object_spawn_y_center: float = -0.15
    object_spawn_z:        float = 0.297  # table_surface(0.215) + cup_z_min_abs(0.0773) + margin(0.005)
                                           # cup_big.usd: origin이 컵 중간, z_min=-0.0773 → 바닥이 원점 아래 7.73cm
    object_spawn_xy_range: float = 0.01   # 초기: 매우 좁게 (±1cm) → 학습 안정화 후 확대

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**22,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    # -----------------------------------------------------------------------
    # 씬 설정
    # -----------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,   # 디버그 기준  (2048: 학습 기준)
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇 설정
    # openarm_tesollo_sensor.usd: URDF openarm_tesollo_sensor.urdf 기준
    #   rl_dg_*_tip 링크 포함 (ContactSensor 전용 rigid body)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_tesollo_sensor/openarm_tesollo_sensor.usd"),
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
                "openarm_right_joint1":  0.5,
                "openarm_right_joint2":  0.1,
                "openarm_right_joint3":  0.4,
                "openarm_right_joint4":  0.60,
                "openarm_right_joint5": -0.2,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                # HAND_APPROACH_POSE: thumb _2=-1.57 (π/2 opposition pre-curl), 나머지=0
                "rj_dg_1_1": 0.0, "rj_dg_1_2":  0.0,  "rj_dg_1_3": 0.0, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.0,  "rj_dg_2_3": 0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.0,  "rj_dg_3_3": 0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.0,  "rj_dg_4_3": 0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0,  "rj_dg_5_3": 0.0, "rj_dg_5_4": 0.0,
                **LEFT_ARM_REST_JOINT_POS,
            },
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # Teosllo hand: stiffness/damping=None → USD 저장값 사용
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=None,
                damping=None,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=None,
                damping=None,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=None,
                damping=None,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=None,
                damping=None,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # rl_dg_*_tip: fixed joint로 rl_dg_*_4에 붙은 전용 sensor rigid body
    # 단일 ContactSensor에 glob으로 5개 body 묶기 → force_matrix_w: (N, 5, 1, 3)
    # -----------------------------------------------------------------------
    _CUP_FILTER = ["/World/envs/env_.*/Cup"]

    # Actor: fingertip 5개 개별 센서 (USD에 ContactSensor 정의 존재)
    # filter_prim_paths_expr → Cup-only 접촉력 (force_matrix_w)
    # 멀티-body 통합 시 PhysX filter 개수 불일치(expected N×5, found N) 문제로 개별 유지
    right_tip_contact_links: tuple = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )
    right_palm_contact_link: str = "rl_dg_palm"

    # Critic privileged: distal 5개 통합 센서 (USD에 ContactSensor 없음)
    # filter 없이 net_forces_w 사용 — sim-only critic obs이므로 허용
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_4",
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # 컵 설정
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big.usd"),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:       list = HAND_BODY_NAMES_USD
    actuated_joint_names:  list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names:  list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
