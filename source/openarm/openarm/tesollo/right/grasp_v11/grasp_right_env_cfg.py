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
    STABILIZE_START_STEP,
)
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)
from .real2sim_actuator_cfg import get_actuator_params, load_real2sim_calibration

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_REAL2SIM_CALIBRATION = load_real2sim_calibration(
    _os.environ.get("OPENARM_REAL2SIM_ACTUATOR_CALIBRATION", "")
)

# 비드 4단계 이산 질량: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g
# mesh scale=0.5x (크기 절반), mass는 그대로 10g (밀도 8배)
_DEFAULT_BEAD_COUNT = 30
_DEFAULT_BEAD_MASS = 0.010


def _actuator_params(group_name: str, default_stiffness: float, default_damping: float) -> dict:
    return get_actuator_params(
        group_name,
        _REAL2SIM_CALIBRATION,
        default_stiffness=default_stiffness,
        default_damping=default_damping,
    )


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    """컵 내부 무게 도메인 랜덤화용 bead 설정 (30개, 각 10g, mesh 0.5x)."""
    rigid_objects: dict = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(0.5, 0.5, 0.5),
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
    episode_length_s: float = 18.0   # grasp 8s + lift 4s + stabilize 2s + transport 4s
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 144 (no oracle mass, includes cup orientation)
    action_space:      int = NUM_ACTIONS               # 26
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 174 (critic, privileged)

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
    fabrics_max_objects_per_env: int  = 8
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
    # Demo reset (optional): pour_v1_a11~a20 grasp start and pour_start lift target
    # -----------------------------------------------------------------------
    enable_demo_grasp_reset: bool = True
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
    lift_success_height: float = 0.04
    lift_target_z_delta: float = LIFT_Z_DELTA
    success_hold_steps: int = 90
    transport_goal_dist_threshold: float = 0.04
    transport_success_hold_steps: int = 90

    # Phase curriculum:
    # 0 = grasp/lift only, 1 = add stabilize, 2 = full transport.
    enable_phase_curriculum: bool = True
    phase_curriculum_initial_stage: int = 0
    phase_curriculum_min_episodes: int = 100
    phase_curriculum_lift_success_threshold: float = 0.70
    phase_curriculum_stabilize_success_threshold: float = 0.70
    terminate_on_lift_failure: bool = True

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
    lift_finger_delta_scale: float = 0.08   # lift phase: ±0.08 rad micro-delta

    # -----------------------------------------------------------------------
    # Reward 파라미터 (HTML: Mass-Adaptive Enveloping Grip Reward Design)
    # -----------------------------------------------------------------------
    # r_height: exp(-α_h * (z_cup - z*)²)  — lift phase에서만 활성
    r_height_weight:    float = 10.0
    r_height_sharpness: float = 100.0  # z* = lift_target_z_delta (0.10m). 3cm error → 0.91

    # r_ori: exp(-α_R * tilt_rad²)  — 컵 수직 자세 유지
    r_ori_weight:    float = 4.0
    r_ori_sharpness: float = 4.0   # 30° → 0.33, 10° → 0.89

    # r_slip: -w_s * Σᵢ 1_{cᵢ} * v_cup_xy²  — 수평 슬립 억제
    r_slip_weight: float = 5.0   # 10 contacts @ 0.05m/s → 0.125 penalty

    # r_margin: -w_m * [max(0, s·mg - μ·ΣFn)]²  — 마찰 안전마진 (lift phase에서만 활성)
    r_margin_weight:       float = 5.0
    friction_safety_factor: float = 1.2  # grip ≥ 1.2×mg friction support

    # r_contact: w_tip·Σtip + w_phalanx·Σphalanx + w_palm·palm  — enveloping contact 유도
    # HTML: w_palm > w_phalanx >= w_tip
    r_contact_tip_weight:     float = 0.2   # 5 tips max → 1.0
    r_contact_phalanx_weight: float = 0.5   # 10 phalanx max → 5.0
    r_contact_palm_weight:    float = 1.0   # palm 1개 — phalanx보다 높게 설정

    # r_force: -w_f · Σ fn²  — 과도 grip force 억제 (max-grip 방지)
    r_force_weight: float = 0.002  # 5N×15 contacts → ~0.75 penalty

    # r_deltaf: -w_Δf · Σ (fn,t - fn,t-1)²  — 급격한 force 변화 억제
    r_deltaf_weight: float = 0.002

    # 질량 파라미터 (r_margin 계산용 privileged variable)
    cup_base_mass:  float = 0.170          # kg (빈 컵 질량)
    bead_single_mass: float = _DEFAULT_BEAD_MASS  # kg per bead

    min_middle_contacts_for_success: int = 4

    # Lift-entry grip readiness gate (state tracking용, reward가 아님)
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

    # Thumb downward shortcut 억제
    thumb_curl_downward_action_scale: float = 0.25
    thumb_curl_max_downward_delta:    float = 0.05

    # -----------------------------------------------------------------------
    # ADR — contact curriculum (threshold=0.1, 먼저 진행)
    # -----------------------------------------------------------------------
    # force_balance gate의 최소 others 접촉 수: 1 → 4 (thumb+1 → thumb+4)
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
        # adaptive_force_weight는 ADR에서 제거 (v10: Gaussian target 방식으로 변경, 가중치 고정)
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 35.0
    success_upright_max_deg: float = 20.0
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
    object_spawn_x_center: float = 0.27
    object_spawn_y_center: float = -0.10
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
                **_actuator_params("openarm_right_arm", 400.0, 80.0),
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                **_actuator_params("openarm_left_arm", 400.0, 80.0),
            ),
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                **_actuator_params("tesollo_hand_abduction", 30.0, 5.0),
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                **_actuator_params("tesollo_hand_curl", 30.0, 5.0),
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                **_actuator_params("tesollo_hand_pip", 30.0, 5.0),
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                **_actuator_params("tesollo_hand_dip", 30.0, 5.0),
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                **_actuator_params("openarm_left_gripper", 400.0, 80.0),
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

    palm_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_palm",
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
    # -----------------------------------------------------------------------
    beads_cfg: RigidObjectCollectionCfg = field(default_factory=_make_beads_cfg)
    num_beads: int = _DEFAULT_BEAD_COUNT              # 30
    bead_count_min: int = 0
    bead_count_max: int = 30                           # Static mass-adaptive bins: {0, 10, 20, 30}.
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
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES


class GraspRightEnvCfgNoActorMass(GraspRightEnvCfg):
    """Asymmetric teacher config: actor excludes oracle mass, critic keeps it."""

    observation_space: int = NUM_OBSERVATIONS_NO_MASS
    num_observations: int = NUM_OBSERVATIONS_NO_MASS
    actor_observe_bead_mass: bool = False
