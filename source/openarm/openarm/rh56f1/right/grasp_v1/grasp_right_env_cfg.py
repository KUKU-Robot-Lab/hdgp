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

"""환경 설정: 5g_grasp_right_v11

v10: v9 기반 + 버그 수정
- Fix 1: rj_dg_1_1 (thumb abduction) = 0.0 고정 (v9: -0.283 → 엄지 새끼손가락 방향 치우침 수정)
- Fix 2: MIN_CONTACTS_FOR_SUCCESS = 4 (v9: 2, ADR 연동 → 2접촉으로 success 오판정 수정)
- Fix 3: has_5_contact = num_contacts>=5 고정 (v9: has_4_contact와 동일 식 버그 수정)
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

from openarm import OPENARM_ROOT_DIR
from .grasp_right_constants import (
    NUM_OBSERVATIONS,
    NUM_OBSERVATIONS_NO_MASS,
    NUM_ACTIONS,
    NUM_CRITIC_OBSERVATIONS,
    LIFT_PHASE_STEPS,
    LIFT_Z_DELTA,
    STABILIZE_PHASE_STEPS,
    STABILIZE_START_STEP,
)
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_HAND_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_HAND_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
)
from .real2sim_actuator_cfg import get_actuator_params, load_real2sim_calibration

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_REAL2SIM_CALIBRATION = load_real2sim_calibration(
    _os.environ.get("OPENARM_REAL2SIM_ACTUATOR_CALIBRATION", "")
)

# 비드 4단계 이산 질량: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g
# cup_middle용 소형 bead. mass는 그대로 10g으로 유지해 hidden-mass bin을 바꾸지 않는다.
_DEFAULT_BEAD_COUNT = 30
_DEFAULT_BEAD_MASS = 0.010
_DEFAULT_BEAD_SCALE = 0.35


def _actuator_params(group_name: str, default_stiffness: float, default_damping: float) -> dict:
    return get_actuator_params(
        group_name,
        _REAL2SIM_CALIBRATION,
        default_stiffness=default_stiffness,
        default_damping=default_damping,
    )


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    """컵 내부 무게 도메인 랜덤화용 bead 설정 (30개, 각 10g, mesh 0.35x)."""
    rigid_objects: dict = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(_DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE),
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=_DEFAULT_BEAD_MASS),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=0.5,
                angular_damping=0.5,
                max_depenetration_velocity=5.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        )
        bead_spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.1,
            dynamic_friction=0.08,
            restitution=0.1,
            friction_combine_mode="min",
            restitution_combine_mode="max",
        )
        rigid_objects[f"bead_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Bead_{i:02d}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42, -0.15, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=bead_spawn_cfg,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v11 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0   # grasp 7s + lift 2s + stabilize 1s
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 102 (actor, with oracle mass=103)
    action_space:      int = NUM_ACTIONS               # 12 (palm 6 + hand 6)
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 117 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS
    actor_observe_bead_mass: bool = False

    # -----------------------------------------------------------------------
    # Pour warm-state export (play/collector only)
    # -----------------------------------------------------------------------
    enable_warm_state_export: bool = False
    warm_state_export_path: str = "/home/oem/rl_ws/datasets/grasp_warm_v11_pour.hdf5"
    warm_state_target_count: int = 2048
    warm_state_success_source: str = "transport"

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 8   # open_tesollo_boxes_no_table 객체 7개 → ≥7 필요
    fabrics_damping_gain:       float = 20.0
    stabilize_upright_orientation_enabled: bool = True
    stabilize_upright_orientation_gain: float = 1.5
    stabilize_upright_orientation_max_deg: float = 25.0
    stabilize_upright_orientation_blend_steps: int = STABILIZE_PHASE_STEPS // 2
    stabilize_spawn_xy_hold_enabled: bool = True
    stabilize_spawn_xy_hold_gain: float = 1.0
    stabilize_spawn_xy_hold_max_delta: float = 0.05

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # pregrasp_offset_* is the desired palm sensor offset from the cup.
    # The environment converts it to the Fabric palm_link target before IK.
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True
    # RH56F1 pregrasp는 Tesollo 값을 복사하지 않고, 실제 RH56F1/cup 기하 기준으로 둔다.
    # palm sensor는 palm_link 기준 (0.00, 0.03, 0.04) 오프셋이고, cup 반경은 약 0.035m다.
    # reset orientation에서 thumb_1 루트가 palm sensor보다도 +x 방향으로 더 앞으로 나온다.
    # test3의 top3 fingertip shell error가 컵 반경보다 약 2cm 멀어서 y/z 오프셋을 줄인다.
    pregrasp_offset_x:     float = -0.045
    pregrasp_offset_y:     float = -0.055
    pregrasp_offset_z:     float = 0.015
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Demo reset (optional): Tesollo 20-DOF demo 데이터(pour_v1_a*) → RH56F1(6-DOF)엔 무효.
    # RH56F1 포팅: 비활성화하고 절차적 13-DOF 리셋(robot_start_joint_pos) 사용.
    # -----------------------------------------------------------------------
    enable_demo_grasp_reset: bool = False
    demo_grasp_pose_paths: tuple[str, ...] = tuple(
        f"/home/oem/rl_ws/datasets/pour_v1_a{i}.hdf5" for i in range(11, 21)
    )

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01
    obs_noise_joint_vel: float = 0.05
    obs_noise_body_pos:  float = 0.005
    obs_noise_cup_pos:   float = 0.015

    # -----------------------------------------------------------------------
    # Real2Sim actuator randomization
    # -----------------------------------------------------------------------
    real2sim_actuator_randomization_enabled: bool = bool(_REAL2SIM_CALIBRATION)
    real2sim_stiffness_scale_range: tuple[float, float] = (0.8, 1.25)
    real2sim_damping_scale_range: tuple[float, float] = (0.7, 1.5)
    real2sim_friction_scale_range: tuple[float, float] = (0.7, 1.3)

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.06
    lift_success_height: float = 0.04
    lift_target_z_delta: float = LIFT_Z_DELTA
    success_hold_steps: int = 30
    transport_goal_dist_threshold: float = 0.04
    transport_success_hold_steps: int = 90

    # Phase curriculum:
    # 0 = grasp/lift only, 1 = add stabilize, 2 = full transport.
    enable_phase_curriculum: bool = True
    phase_curriculum_initial_stage: int = 1
    phase_curriculum_min_episodes: int = 100
    phase_curriculum_lift_success_threshold: float = 0.70
    phase_curriculum_stabilize_success_threshold: float = 0.70
    terminate_on_lift_failure: bool = True

    # -----------------------------------------------------------------------
    # Delta palm action
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.08
    palm_delta_rot_deg: float = 20.0

    # -----------------------------------------------------------------------
    # Finger action semantics
    # RH56F1 v1은 6D absolute synergy target을 사용한다.
    # grasp:      HAND_APPROACH_POSE(-1) ~ HAND_GRASP_POSE(+1)
    # post-grasp: HAND_GRASP_POSE(-1)    ~ HAND_FULL_GRIP_POSE(+1)
    # 아래 delta_scale 항목은 이전 run/config 호환을 위해 유지되며 현재 env에서는 사용하지 않는다.
    # -----------------------------------------------------------------------
    finger_delta_scale:      float = 0.08
    lift_finger_delta_scale: float = 0.08
    enable_grasp_phase_full_grip_blend: bool = True
    grasp_phase_full_grip_contact_threshold: int = 5
    grasp_phase_full_grip_progress_threshold: float = 1.1

    # -----------------------------------------------------------------------
    # Reward 파라미터: Tesollo grasp_v7_2 스타일 dense enclosure reward
    # -----------------------------------------------------------------------
    palm_approach_weight:    float = 1.0
    palm_approach_sharpness: float = 10.0

    enclosure_weight:       float = 3.0
    enclosure_sharpness:    float = 15.0
    cup_radius_approx:      float = 0.035
    enclosure_thumb_weight: float = 0.6

    align_upright_reward_weight: float = 2.0
    lift_reward_weight: float = 30.0
    grasp_five_tip_contact_reward_weight: float = 6.0
    grasp_five_tip_hold_reward_weight: float = 10.0
    grasp_contact_persistence_reward_steps: int = 30
    stabilize_upright_reward_weight: float = 10.0
    stabilize_upright_reward_scale_deg: float = 5.0

    action_smoothness_palm_weight:   float = -0.02
    action_smoothness_finger_weight: float = -0.01

    # 질량 파라미터 (force-ratio/bin logging용 privileged variable)
    cup_base_mass:  float = 0.170          # kg (빈 컵 질량)
    bead_single_mass: float = _DEFAULT_BEAD_MASS  # kg per bead
    bead_scale: float = _DEFAULT_BEAD_SCALE

    min_middle_contacts_for_success: int = 0

    # Lift-entry grip readiness gate (state tracking용, reward가 아님)
    stage0_lift_start_min_contacts: int = 5
    stage0_lift_start_hold_steps:   int = 30
    lift_contact_hold_steps: int = 30
    full_grip_hold_steps:    int = 30
    lift_min_force_ratio:    float = 1.8

    # Slip proxy (no_slip_gate 계산용, 게임로직용)
    slip_proxy_threshold:                float = 1.0
    slip_proxy_contact_delta_weight:     float = 0.5
    slip_proxy_middle_contact_delta_weight: float = 0.5
    slip_proxy_tilt_delta_weight:        float = 0.5
    slip_proxy_tilt_delta_scale:         float = 8.0

    # Stabilize/transport 판정 임계값 (full_grip_ready gate용)
    stabilize_cup_lin_vel_threshold:  float = 0.04
    stabilize_cup_ang_vel_threshold:  float = 0.50
    stabilize_force_delta_threshold:  float = 0.35
    stabilize_contact_delta_threshold: float = 1.0

    # Legacy delta-control knob (absolute synergy semantics에서는 미사용)
    thumb_curl_downward_action_scale: float = 0.25
    thumb_curl_max_downward_delta:    float = 0.05

    # -----------------------------------------------------------------------
    # ADR — contact curriculum (threshold=0.1, 먼저 진행)
    # -----------------------------------------------------------------------
    # slip/adaptive/full_contact gate의 최소 총 접촉 수: 2 → 5
    enable_contact_adr:             bool  = True
    contact_adr_num_increments:     int   = 50
    contact_adr_increment_interval: int   = 400
    contact_adr_trigger_threshold:  float = 0.1   # 10% 성공률에서 진행 (early curriculum)

    # 6.2: ADR trigger moving-window 크기
    # 최근 N episode 성공률을 ADR trigger에 사용 (0: 기존 cumulative 방식 유지)
    adr_window_size: int = 500

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
        "reward_weights": {
            "enclosure_weight": (10.0, 20.0),
        },
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 35.0
    success_upright_max_deg: float = 20.0
    stabilize_upright_max_deg: float = 5.0
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

    # -----------------------------------------------------------------------
    # Transport goal sampling
    # -----------------------------------------------------------------------
    # min=max on each axis gives a fixed deployment target.
    transport_goal_x_range: tuple[float, float] = (0.22, 0.42)
    transport_goal_y_range: tuple[float, float] = (-0.02, 0.18)
    transport_goal_z_range: tuple[float, float] = (0.42, 0.58)

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.40
    object_spawn_y_center: float = -0.10
    object_spawn_z:        float = 0.2773
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
            gpu_collision_stack_size=2**28,
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
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_bi_rh56f1/openarm_bi_rh56f1.usd"),
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
                # RH56F1 우측 손 drive (approach pose)
                "rh56f1_right_right_thumb_1_joint":  HAND_APPROACH_POSE[0],   # 0.60
                "rh56f1_right_right_thumb_2_joint":  HAND_APPROACH_POSE[1],   # 0.15
                "rh56f1_right_right_index_1_joint":  HAND_APPROACH_POSE[2],   # 0.30
                "rh56f1_right_right_middle_1_joint": HAND_APPROACH_POSE[3],   # 0.30
                "rh56f1_right_right_ring_1_joint":   HAND_APPROACH_POSE[4],   # 0.30
                "rh56f1_right_right_little_1_joint": HAND_APPROACH_POSE[5],   # 0.30
                # mimic 추종 (= drive × multiplier, 결합 init 으로 snap 방지)
                "rh56f1_right_right_thumb_3_joint":  HAND_APPROACH_POSE[1] * 1.1425,            # 0.171
                "rh56f1_right_right_thumb_4_joint":  HAND_APPROACH_POSE[1] * 1.1425 * 0.7508,   # 0.129
                "rh56f1_right_right_index_2_joint":  HAND_APPROACH_POSE[2] * 1.1169,            # 0.335
                "rh56f1_right_right_middle_2_joint": HAND_APPROACH_POSE[3] * 1.1169,
                "rh56f1_right_right_ring_2_joint":   HAND_APPROACH_POSE[4] * 1.1169,
                "rh56f1_right_right_little_2_joint": HAND_APPROACH_POSE[5] * 1.1169,
                **LEFT_ARM_REST_JOINT_POS,
                **LEFT_HAND_REST_JOINT_POS,
            },
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                **_actuator_params("openarm_right_arm", 400.0, 80.0),
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                **_actuator_params("openarm_left_arm", 400.0, 80.0),
            ),
            # RH56F1 우측 손 drive 6 (RL 위치제어)
            "rh56f1_right_drive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rh56f1_right_right_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint"
                ],
                **_actuator_params("rh56f1_right_drive", 30.0, 5.0),
            ),
            # RH56F1 우측 손 mimic 추종 6 (passive — PhysxMimicJoint 가 커플)
            "rh56f1_right_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rh56f1_right_right_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint"
                ],
                stiffness=0.0, damping=0.0,
            ),
            # RH56F1 좌측 손 drive 6 (학습 비사용 → 0 hold)
            "rh56f1_left_drive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rh56f1_left_left_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint"
                ],
                **_actuator_params("rh56f1_left_drive", 30.0, 5.0),
            ),
            # RH56F1 좌측 손 mimic 추종 6 (passive)
            "rh56f1_left_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rh56f1_left_left_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint"
                ],
                stiffness=0.0, damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # -----------------------------------------------------------------------
    # fingertip 힘센서 (실 *_force_sensor → 병합된 말단 링크). 순서: thumb,index,middle,ring,little
    right_tip_contact_links: tuple = (
        "rh56f1_right_right_thumb_4",
        "rh56f1_right_right_index_2",
        "rh56f1_right_right_middle_2",
        "rh56f1_right_right_ring_2",
        "rh56f1_right_right_little_2",
    )

    tip_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rh56f1_right_right_(thumb_4|index_2|middle_2|ring_2|little_2)",
        history_length=1,
        track_air_time=False,
    )

    # palm 힘센서 (실 plam_force_sensor)
    palm_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rh56f1_right_plam_force_sensor",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
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
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_middle.usd"),
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
    # 컵 마찰계수 도메인 랜덤화
    # -----------------------------------------------------------------------
    # μ_static  ~ Uniform[cup_friction_min, cup_friction_max]  (에피소드별 리셋)
    # μ_dynamic = μ_static × 0.9
    # 목적: max-grip이 Pareto-optimal이 되는 것을 방지 → 마찰 변동으로 적응형 파지 학습
    cup_friction_min: float = 0.15
    cup_friction_max: float = 0.60

    # 6.4: friction ablation — 고정값 (>= 0)이면 DR 비활성화, -1이면 랜덤화
    cup_friction_fixed: float = -1.0

    # -----------------------------------------------------------------------
    # Bead 무게 도메인 랜덤화
    # cup_middle에서는 물리 bead가 컵을 관통/튕김시키므로 기본 비활성화한다.
    # -----------------------------------------------------------------------
    beads_cfg: RigidObjectCollectionCfg = field(default_factory=_make_beads_cfg)
    num_beads: int = _DEFAULT_BEAD_COUNT              # 30
    physical_beads_enabled: bool = False
    bead_count_min: int = 0
    bead_count_max: int = 0
    bead_spawn_z_offset: float = 0.035

    # Keep dynamic insertion disabled for hidden-mass static-bin grasp/lift training.
    dynamic_bead_spawn_enabled: bool = False
    dynamic_bead_spawn_step: int = STABILIZE_START_STEP + 30
    bead_initial_count_min: int = 0
    bead_initial_count_max: int = 0
    dynamic_bead_add_count_min: int = 10
    dynamic_bead_add_count_max: int = 20

    # Eval-only: lift 중 oracle/effective mass만 바꿔 force 반응을 측정한다.
    eval_mass_shift_enabled: bool = False
    eval_mass_shift_step: int = LIFT_PHASE_STEPS // 2
    eval_mass_shift_target_bead_count: int = 30

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_HAND_JOINT_NAMES


class GraspRightEnvCfgNoActorMass(GraspRightEnvCfg):
    """Asymmetric teacher config: actor excludes oracle mass, critic keeps it."""

    observation_space: int = NUM_OBSERVATIONS_NO_MASS
    num_observations: int = NUM_OBSERVATIONS_NO_MASS
    actor_observe_bead_mass: bool = False
