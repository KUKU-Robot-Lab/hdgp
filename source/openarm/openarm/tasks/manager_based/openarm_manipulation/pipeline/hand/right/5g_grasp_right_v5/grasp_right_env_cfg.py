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
- Action: 8D (5D per-finger lerp + 3D palm xyz residual)
- Observation: 94D (actor = critic)
- Episode: 6s (5s grasp + 1s scripted lift)
- Contact: 물리 ContactSensor 기반 (fingertip 5개)
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
    # Episode: 6s = 360 steps @ 60Hz (5s grasp + 1s lift)
    # -----------------------------------------------------------------------
    episode_length_s: float = 6.0    # GRASP_PHASE(5s) + LIFT_PHASE(1s)
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS         # 94  (actor)
    action_space:      int = NUM_ACTIONS              # 8
    state_space:       int = NUM_CRITIC_OBSERVATIONS  # 118 (critic, privileged)

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
    pregrasp_fabric_steps: int   = 60
    pregrasp_offset_x:     float = 0.0
    pregrasp_offset_y:     float = -0.12   # cup -Y 방향 12cm (palm_link 기준, lift_v1 palm_ee=-6cm → palm_link≈-9cm + 여유 3cm)
    pregrasp_offset_z:     float = 0.05    # cup +Z 방향 5cm
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.056   # cup root → 실제 파지 중심 z offset
    lift_success_height: float = 0.04    # 성공 판정: cup_z > init_z + 4cm

    # -----------------------------------------------------------------------
    # Reward 파라미터
    # -----------------------------------------------------------------------
    # 1. contact_reward: 접촉 유지 (num_contacts / 5)
    contact_reward_weight: float = 2.0

    # 2. contact_delta: 접촉 증가 보상 / 감소 패널티
    #    ADR: 3.0 → 1.0 (초기 contact 유도 후 감소)
    contact_delta_weight: float = 3.0   # ADR 초기값

    # 3. enclosure: fingertip → cup 거리 기반 exp reward
    #    ADR: 2.0 → 3.0 (점점 강화)
    enclosure_weight:    float = 2.0   # ADR 초기값
    enclosure_sharpness: float = 10.0  # exp(-sharpness * mean_dist)

    # 4. opposition: thumb + 다른 손가락 동시 접촉
    opposition_weight: float = 1.0

    # 5. action_reg: ||action||² 패널티
    action_reg_weight: float = -0.005

    # 6. terminal rewards
    terminal_success_weight: float = 10.0
    terminal_fail_weight:    float = -1.0

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 200
    adr_trigger_threshold: float = 0.1   # success ratio 10% 이상

    adr_custom_cfg: dict = field(default_factory=lambda: {
        "reward_weights": {
            "contact_delta_weight": (3.0, 1.0),   # 초기 contact 유도 후 감소
            "enclosure_weight":     (2.0, 3.0),   # 점점 강화
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
    obj_fallen_z:   float = 0.18

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.40
    object_spawn_y_center: float = -0.15
    object_spawn_z:        float = 0.38
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
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.235],
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
                "openarm_right_joint4":  0.8,
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
            # Teosllo hand: abduction/curl/pip/dip 분리 stiffness (v5 고정값)
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=1.9,
                damping=7.5e-4,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=0.84,
                damping=3.3e-4,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=0.43,
                damping=1.7e-4,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=0.13,
                damping=5.1e-5,
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
    # Fingertip ContactSensor 링크명 (URDF: openarm_tesollo_sensor.urdf)
    # rl_dg_*_tip: fixed joint로 rl_dg_*_4에 붙은 전용 sensor rigid body
    # -----------------------------------------------------------------------
    # Actor ContactSensor (real-compatible: Teosllo fingertip FT)
    right_tip_contact_links: tuple = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )
    right_palm_contact_link: str = "rl_dg_palm"

    # Critic privileged ContactSensor (sim-only)
    # distal (rl_dg_*_4): 5개 전 손가락
    right_distal_contact_links: tuple = (
        "rl_dg_1_4",
        "rl_dg_2_4",
        "rl_dg_3_4",
        "rl_dg_4_4",
        "rl_dg_5_4",
    )
    # middle (rl_dg_*_3): thumb, index, middle 3개 (MD Section 10.2)
    right_middle_contact_links: tuple = (
        "rl_dg_1_3",
        "rl_dg_2_3",
        "rl_dg_3_3",
    )

    # -----------------------------------------------------------------------
    # 컵 설정
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.38],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup_bead/cup.usd"),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.2),
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
