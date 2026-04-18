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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .pour_constants import NUM_ACTIONS
from .pour_utils import ACTOR_OBSERVATION_DIM
from .pour_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_JOINT_NAMES,
    LEFT_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_TARGET_CUP_ATTACH_FRAME_NAME,
    LEFT_TARGET_CUP_ATTACH_POS_B,
    LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B,
    RIGHT_ACTUATED_JOINT_NAMES,
    SOURCE_CUP_POUR_AXIS_B,
    SOURCE_CUP_POUR_POINT_POS_B,
    SOURCE_CUP_UP_AXIS_B,
    TARGET_CUP_OPENING_POS_B,
    TARGET_CUP_UP_AXIS_B,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_DEFAULT_BEAD_COUNT = 20


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.010),  # 10g 구슬 (5g→10g: 관성 향상, 진동 날림 방지)
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,  # 5.0→0.5: 벽 penetration 시 순간이동 방지
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        )
        bead_spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.10,
            dynamic_friction=0.08,
            restitution=0.05,
            friction_combine_mode="min",
            restitution_combine_mode="max",
        )
        rigid_objects[f"bead_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Bead_{i:02d}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42, -0.18, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=bead_spawn_cfg,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class PourRightEnvCfg(DirectRLEnvCfg):
    """5g_pour_right_v3 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    # Episode: 20s = 1200 steps @ 60Hz
    # -----------------------------------------------------------------------
    episode_length_s: float = 20.0
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # actor obs (DexPour-aligned superset):
    #   right arm+hand joint pos/vel (54)
    #   left arm joint pos/vel (14)
    #   fingertip positions (15)
    #   source cup pose/velocity (13)
    #   target opening position (3)
    #   bead centroid position (3)
    #   previous actions (18)
    #   mouth geometry + fill/spill + gates (14)
    # = 134D
    # critic obs is kept symmetric for now.
    # -----------------------------------------------------------------------
    observation_space: int = ACTOR_OBSERVATION_DIM
    action_space:      int = NUM_ACTIONS
    state_space:       int = ACTOR_OBSERVATION_DIM

    num_observations: int = ACTOR_OBSERVATION_DIM
    num_actions:      int = NUM_ACTIONS
    num_states:       int = ACTOR_OBSERVATION_DIM

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0  # 180.0 -> 45.0: 접근/이동 중 기괴한 손목 회전 억제
    fabrics_max_objects_per_env: int  = 6
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 200
    episode_hold_steps:    int   = 60   # 텔레포트 후 contact 재정립 대기 (1s @ 60Hz)
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True    # 13×13 grid IK 사전 계산 → reset 시 lookup (랜덤화와 호환)
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # GUI helper
    enable_visual_markers: bool = False  # 시각화 마커 (붉은/푸른 점) 표시 여부


    # bead / cup geometry (cup_big.usd 기준: bottom=-0.077m, rim=+0.100m, inner_r=0.041m)
    target_inner_radius:  float = 0.041   # 컵 내부 반경
    target_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    target_inside_z_max:  float = 0.100   # 림 높이
    target_mouth_z:       float = 0.100   # 림 높이 (bead crossing 기준)
    source_inner_radius:  float = 0.041   # 컵 내부 반경
    source_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    source_inside_z_max:  float = 0.100   # 림 높이
    bead_count: int = _DEFAULT_BEAD_COUNT

    # -----------------------------------------------------------------------
    # bead count ADR: 1 → 20 단계적 증가 (슬라이딩 윈도우 성공률 기반)
    # bead N개: success_bonus = weight_success_per_bead × N
    #           spill_penalty  = weight_spill_per_bead  × N (per step, spill_ratio에 적용)
    # -----------------------------------------------------------------------
    enable_bead_count_adr: bool = True
    bead_count_stages: tuple = (1, 2, 3, 5, 10, 20)     # 진급 단계
    bead_count_adr_trigger_threshold: float = 0.80       # 80% 성공 시 다음 단계 진급
    bead_count_adr_window_size: int = 500                # [v4 수정] 200→500: 진급 속도 감소 (초반 too-fast progression 방지)
    weight_success_per_bead: float = 20.0                # bead 1개당 성공 보너스

    weight_spill_per_bead: float = 2.0                   # bead 1개당 스필 패널티 (per step)

    # Hidden parking grid: 비활성 bead를 env 외부에 숨기는 env-local 좌표 오프셋
    bead_hidden_base_x: float = -1.20
    bead_hidden_base_y: float = -0.60
    bead_hidden_z: float = 0.02
    bead_hidden_cols: int = 5
    bead_hidden_spacing: float = 0.03

    # -----------------------------------------------------------------------
    # Policy action / pouring target
    # -----------------------------------------------------------------------
    palm_delta_xyz: float = 0.5   
    warmstart_collect_palm_delta_xyz: float = 0.10
    palm_delta_rot_deg: float = 120.0  
    left_arm_delta_joint: float = 0.35
    tilt_action_gate_xy_near: float = 0.04
    tilt_action_gate_xy_far: float = 0.12
    approach_xy_off_near: float = 0.02
    approach_xy_off_far: float = 0.03
    left_cup_world_z_offset: float = -0.08   
    reward_gate_xy_scale: float = 5.0   
    reward_gate_clear_scale: float = 80.0
    reward_gate_tilt_scale: float = 15.0
    reward_clearance_min: float = 0.015
    reward_tilt_cos_min: float = 0.15

    # -----------------------------------------------------------------------
    # Left-arm assist obs / reward
    # right-hand grasp 유지와 source cup handling은 warmstart prior로 간주하고,
    # left arm target cup alignment + capture shaping에 집중한다.
    # -----------------------------------------------------------------------
    assist_reward_approach_xy: float = 1.5
    assist_reward_clearance: float = 0.75
    assist_reward_ready: float = 0.5
    assist_reward_tilt: float = 0.75
    assist_reward_align: float = 0.5
    assist_reward_cross: float = 2.0
    assist_reward_capture: float = 8.0
    assist_reward_success: float = 5.0
    assist_reward_terminal_capture: float = 10.0
    assist_reward_spill: float = 3.0
    assist_reward_premature_tilt: float = 0.75
    assist_reward_left_action_rate: float = 0.02
    assist_reward_left_joint_vel: float = 0.002
    assist_success_fill_ratio: float = 0.30
    assist_success_spill_max: float = 0.15
    source_empty_hold_steps: int = 5

    # -----------------------------------------------------------------------
    # Warmstart quality / success
    # -----------------------------------------------------------------------
    lift_success_height: float = 0.03   
    warmstart_cache_min_lift_height: float = 0.15   # 테이블 기준 최소 15cm (≈ 컵 절반 높이 이상)
    warmstart_cache_min_contacts:    int   = 5       # 최소 5손가락 접촉 (기존 2보다 강화)
    warmstart_stable_hold_steps:     int   = 10      # 연속 10프레임(0.167s) 유지해야 저장

    enable_success_warmstart_reset: bool = False   
    success_warmstart_cache_size: int = 512
    success_cache_reset_ratio: float = 0.0         # v3 0.40 → v4 0.0 (mid-pour 리셋 차단)
    grasp_warmstart_reset_ratio: float = 1.0       # 0.85 → 1.0: 100% warmstart (v7 grasp 완료 상태)
    success_cache_store_min_bead_cross_fraction: float = 0.05
    success_cache_store_min_bead_in_target_fraction: float = 0.02

    # -----------------------------------------------------------------------
    # Trajectory Capture (BC loss용 궤적 수집)
    # -----------------------------------------------------------------------
    enable_trajectory_capture: bool = True
    trajectory_capture_window: int = 200
    trajectory_buffer_capacity: int = 256
    trajectory_success_bead_threshold: float = 0.50   # bead_in_target >= 50%
    trajectory_success_spill_max: float = 0.10        # spill < 10%
    trajectory_min_steps: int = 60            # 최소 궤적 길이 (1s @ 60Hz)

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
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
    object_spawn_xy_range: float = 0.06   # ±6cm 랜덤화 (Fabrics arm 학습으로 보정 가능)

    # -----------------------------------------------------------------------
    # Warmstart reset cache
    # -----------------------------------------------------------------------
    enable_warmstart_reset: bool = True
    warmstart_checkpoint_path: str = (
        "/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7/test1/nn/5g_grasp_right-v7.pth"
    )
    warmstart_cache_size: int = 256
    warmstart_max_rollout_steps: int = 6000
    freeze_grasp_hand_during_episode: bool = False
    bead_spawn_pos_source_cup_b: tuple[float, float, float] = tuple(BEAD_SPAWN_POS_SOURCE_CUP_B)
    bead_spawn_quat_source_cup_wxyz: tuple[float, float, float, float] = tuple(
        BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ
    )
    left_target_cup_attach_frame_name: str = LEFT_TARGET_CUP_ATTACH_FRAME_NAME
    left_target_cup_attach_pos_b: tuple[float, float, float] = tuple(LEFT_TARGET_CUP_ATTACH_POS_B)
    left_target_cup_attach_quat_wxyz_b: tuple[float, float, float, float] = tuple(
        LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B
    )
    source_cup_pour_point_pos_b: tuple[float, float, float] = tuple(SOURCE_CUP_POUR_POINT_POS_B)
    target_cup_opening_pos_b: tuple[float, float, float] = tuple(TARGET_CUP_OPENING_POS_B)
    source_cup_pour_axis_b: tuple[float, float, float] = tuple(SOURCE_CUP_POUR_AXIS_B)
    source_cup_up_axis_b: tuple[float, float, float] = tuple(SOURCE_CUP_UP_AXIS_B)
    target_cup_up_axis_b: tuple[float, float, float] = tuple(TARGET_CUP_UP_AXIS_B)

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_pairs_capacity=4 * 1024 * 1024,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**24,
            gpu_max_rigid_contact_count=2**24,
            gpu_collision_stack_size=2**26,
            gpu_max_num_partitions=64,
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
    # 로봇 설정 (openarm_tesollo_sensor.usd: rl_dg_*_tip ContactSensor 포함)
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
                "openarm_right_joint3":  0.0,
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
                stiffness=2000.0,   # 400→2000: 오른팔 충돌 저항 강화
                damping=200.0,
            ),
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=90.0,
                damping=15.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=100.0,
                damping=18.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=100.0,
                damping=18.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=100.0,
                damping=18.0,
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
    # Actor: fingertip 5개 개별 센서 (Cup-only, real-compatible FT sensor)
    # Critic: distal + middle 통합 센서 (sim-only)
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
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_sdf.usd"),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    left_target_cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/LeftTargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42, 0.18, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_sdf.usd"),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=-0.1,
                rest_offset=-0.1,
            ),
        ),
    )

    beads_cfg: RigidObjectCollectionCfg = _make_beads_cfg()

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_JOINT_NAMES
    left_gripper_joint_names: list = LEFT_GRIPPER_JOINT_NAMES
