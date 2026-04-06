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

"""환경 설정: 5g_grasp_right_v9

v9: v8 기반 + 20D 손가락 직접 제어 + Slip/Force-Efficiency/Force-Smooth reward
- Action: 26D (6D palm pose + 20D per-joint finger delta)
- Observation: actor 128D / critic 164D (asymmetric)
- Reward: R_slip (cup 수평 속도 기반), R_force_efficiency (mg 기준 최소 충분 힘), R_force_smooth (힘 변화 억제)
- 기존 reward weight 조정: lift 20→6, enclosure 4→3, balance 8→6, multi_phalanx 8→6
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
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

# 비드 4단계 이산 질량: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g
# mesh scale=0.5x (크기 절반), mass는 그대로 10g (밀도 8배)
_DEFAULT_BEAD_COUNT = 30


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    """컵 내부 무게 도메인 랜덤화용 bead 설정 (30개, 각 10g, mesh 0.5x)."""
    rigid_objects: dict = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        rigid_objects[f"bead_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Bead_{i:02d}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42, -0.15, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
                scale=(0.5, 0.5, 0.5),          # mesh 절반 크기, mass는 10g 유지
                activate_contact_sensors=False,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=5.0,
                    max_linear_velocity=10.0,
                    max_angular_velocity=20.0,
                ),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v9 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 138 (actor, tip_force 5D→15D)
    action_space:      int = NUM_ACTIONS               # 26
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 174 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 6
    fabrics_damping_gain:       float = 20.0

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01
    obs_noise_joint_vel: float = 0.05
    obs_noise_body_pos:  float = 0.005
    obs_noise_cup_pos:   float = 0.015

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.06
    lift_success_height: float = 0.04
    success_hold_steps: int = 90

    # -----------------------------------------------------------------------
    # Delta palm action
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.01
    palm_delta_rot_deg: float = 20.0

    # -----------------------------------------------------------------------
    # Finger joint delta 제어 (v9 신규)
    # action ∈ [-1,1] → ±finger_delta_scale rad per step
    # -----------------------------------------------------------------------
    finger_delta_scale:      float = 0.08   # grasp phase: ±0.08 rad/step (grasp_pose 근처 미세 조정)
    lift_finger_delta_scale: float = 0.05   # lift phase: ±0.05 rad micro-delta

    # -----------------------------------------------------------------------
    # Reward 파라미터
    # -----------------------------------------------------------------------
    # R0. palm_approach
    palm_approach_weight:    float = 0.5
    palm_approach_sharpness: float = 10.0

    # R1. fingertip_enclosure (v8: 4.0 → v9: 3.0, slip/efficiency 강화로 비중 완화)
    enclosure_weight:       float = 3.0
    enclosure_sharpness:    float = 15.0
    cup_radius_approx:      float = 0.045
    enclosure_thumb_weight: float = 0.6

    # R1b. force_balance (v8: 8.0 → v9: 6.0)
    force_balance_weight:    float = 6.0
    force_balance_sharpness: float = 8.0

    # R1c. multi_phalanx_contact (v8: 8.0 → v9: 6.0)
    multi_phalanx_weight: float = 6.0

    # R2. slip_reward (v9 신규): cup 수평 속도 기반 slip proxy
    # gate: grasp phase AND contact 시 활성
    # R_slip = slip_weight * gate * exp(-slip_sharpness * cup_horiz_vel)
    slip_weight:    float = 8.0
    slip_sharpness: float = 20.0

    # R3. Adaptive Force Reward (v9.3 철학 유지: 단조감소, decay 강화)
    # 설계 철학:
    #   slip          → "force 부족" 커버 (결과 기반)
    #   adaptive_force → "force 과도" 억제 (단조감소)
    #   최적 ratio → slip과 R3의 균형에서 policy가 스스로 학습 (target 없음)
    #
    # R_af = af_weight * is_lift * contact * exp(-decay * ratio)
    #   force_ratio = F_total / mg  (질량 정규화)
    #   decay 0.3 → 0.8: 과도 grip 패널티 강화
    #     ratio=1.0: 0.45, ratio=2.0: 0.20, ratio=4.85: 0.02
    adaptive_force_weight:         float = 10.0
    adaptive_force_decay:          float = 0.8   # v9.3: 0.3 → 0.8 (과도 grip 억제 강화)
    cup_base_mass:                 float = 0.170  # kg (빈 컵 질량)
    bead_single_mass:              float = 0.010  # kg per bead

    # R_ft. fingertip_guide: fingertip → cup 거리 기반 (항상 gradient, seed 분산 방지)
    # sim2real 영향 없음: fingertip_pos는 FK 또는 FT 센서로 실 로봇에서도 획득 가능
    # cup_pos: 관측 노이즈 처리된 값 사용 (σ_cp 적용됨)
    fingertip_guide_weight:    float = 0.5
    fingertip_guide_sharpness: float = 5.0

    # R5. force_smooth (v9 신규): 파지력 변화율 억제 (sim2real 안정성)
    force_smooth_weight: float = 1.5

    # R6. lift_reward (v8: 20.0 → v9: 6.0 → v9.2: 20.0, slip local-min 탈출)
    lift_reward_weight: float = 20.0

    # R8. success_bonus: lift 성공 유지 중 step당 보너스 (slip local-min 탈출)
    success_bonus_weight: float = 20.0

    # R9. full_contact_bonus: 5손가락 전체 접촉 보너스 (sim2real envelope grip 유도)
    # step당 보너스 → 유지할수록 누적 (grasp + lift phase 모두)
    full_contact_bonus_weight: float = 0.5

    # R7. action_smoothness
    action_smoothness_palm_weight:   float = -0.01
    action_smoothness_finger_weight: float = -0.01   # v9.2: v8 수준 복원 (entropy explosion 억제)

    # -----------------------------------------------------------------------
    # ADR — contact curriculum (threshold=0.1, 먼저 진행)
    # -----------------------------------------------------------------------
    # force_balance gate의 최소 others 접촉 수: 1 → 4 (thumb+1 → thumb+4)
    # slip/adaptive/full_contact gate의 최소 총 접촉 수: 2 → 5
    enable_contact_adr:             bool  = True
    contact_adr_num_increments:     int   = 50
    contact_adr_increment_interval: int   = 400
    contact_adr_trigger_threshold:  float = 0.1   # 10% 성공률에서 진행 (early curriculum)

    contact_adr_custom_cfg: dict = field(default_factory=lambda: {
        "contact": {
            # int(round(value)) 로 사용: 2 → 5 (전 손가락)
            "min_contacts": (2.0, 5.0),
        },
    })

    # -----------------------------------------------------------------------
    # ADR — 난이도 (threshold=0.8, contact ADR 이후 진행)
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 400
    adr_trigger_threshold: float = 0.8   # 80% 성공률에서 진행 (contact 학습 후)

    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "object_spawn_xy_range": (0.01, 0.06),
        },
        "noise": {
            "obs_noise_cup_pos": (0.005, 0.025),
        },
        "finger": {
            "delta_scale": (0.05, 0.15),
        },
        "reward": {
            "adaptive_force_weight": (5.0, 10.0),
        }
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 60.0
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.40
    object_spawn_y_center: float = -0.15
    object_spawn_z:        float = 0.297
    object_spawn_xy_range: float = 0.06

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
        num_envs=128,
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
                "rj_dg_1_1": 0.0, "rj_dg_1_2": -1.57, "rj_dg_1_3": -0.5, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.0,  "rj_dg_2_3":  0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.0,  "rj_dg_3_3":  0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.0,  "rj_dg_4_3":  0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0,  "rj_dg_5_3":  0.0, "rj_dg_5_4": 0.0,
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
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=30.0,
                damping=5.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=30.0,
                damping=5.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=30.0,
                damping=5.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=30.0,
                damping=5.0,
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
    # -----------------------------------------------------------------------
    right_tip_contact_links: tuple = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )

    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_4",
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_3",
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
    # Bead 무게 도메인 랜덤화
    # -----------------------------------------------------------------------
    beads_cfg: RigidObjectCollectionCfg = field(default_factory=_make_beads_cfg)
    num_beads: int = _DEFAULT_BEAD_COUNT              # 30
    bead_count_min: int = 0
    bead_count_max: int = 30                           # 이산: {0, 10, 20, 30}개
    bead_spawn_z_offset: float = 0.035

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
