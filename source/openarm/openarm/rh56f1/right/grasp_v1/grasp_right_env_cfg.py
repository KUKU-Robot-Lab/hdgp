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

"""환경 설정: open-rh56f1_r_grasp_v1."""

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
    observation_space: int = NUM_OBSERVATIONS          # 96 (actor, with oracle mass=97)
    action_space:      int = NUM_ACTIONS               # 12 (palm 6 + hand 6)
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 114 (critic, privileged)

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
    warm_state_success_source: str = "stabilize"

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
    # Backward-compatible aliases for older launch overrides.
    stabilize_spawn_xy_hold_enabled: bool = True
    stabilize_spawn_xy_hold_gain: float = 2.0
    stabilize_spawn_xy_hold_max_delta: float = 0.10

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
    # envelope 유도: palm standoff 0.6배 축소(컵에 근접 → 근위 마디가 컵에 닿아 wrap 가능).
    # 과근접 시 컵 관통 위험 → render 조기확인 필요.
    pregrasp_offset_x:     float = -0.027
    pregrasp_offset_y:     float = -0.033
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
    lift_success_height: float = 0.04
    lift_target_z_delta: float = LIFT_Z_DELTA
    success_hold_steps: int = 20   # Phase2-b: 30→20 (decay 0.5/steps 30은 hold_count 4.7서 정체 = success_held ~0). 문턱 완화
    success_hold_miss_decay: float = 0.25   # Phase2-b: 0.5→0.25 (순증 +0.16→+0.30/step at success_now 0.44). leaky: miss 시 0 리셋 대신 감쇠. -1=기존 hard reset
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.4   # Phase B: stable 판정 완화 (0.2는 과도하게 빡빡 → success_held=0)

    # Phase curriculum:
    # 0 = grasp/lift only, 1 = add stabilize.
    enable_phase_curriculum: bool = False
    phase_curriculum_initial_stage: int = 1
    phase_curriculum_min_episodes: int = 100
    phase_curriculum_lift_success_threshold: float = 0.70
    phase_curriculum_stabilize_success_threshold: float = 0.70
    terminate_on_lift_failure: bool = True

    # -----------------------------------------------------------------------
    # Delta palm action
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.03
    palm_delta_rot_deg: float = 5.0
    ema_action_alpha: float = 0.7
    approach_min_steps: int = 10
    approach_timeout_steps: int = 90
    approach_palm_radial_min: float = 0.025
    approach_palm_radial_max: float = 0.105
    approach_palm_local_z_min: float = -0.015
    approach_palm_local_z_max: float = 0.095
    approach_max_tip_contacts: int = 2
    approach_upright_max_deg: float = 20.0
    approach_timeout_grasp_reward_scale: float = 0.25
    grasp_palm_delta_scale: float = 1.0
    grasp_palm_inward_offset: float = 0.05   # envelope 유도: grasp 중 palm 을 컵쪽으로 더 당김(0.025→0.05)
    lift_palm_delta_xyz: float = 0.03
    lift_palm_delta_rot_deg: float = 15.0

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
    grasp_phase_full_grip_contact_threshold: int = 4
    grasp_phase_full_grip_progress_threshold: float = 0.65

    # -----------------------------------------------------------------------
    # Shared 5-tip grasp reward parameters
    # -----------------------------------------------------------------------
    grasp_upright_threshold_deg: float = 8.0
    grasp_xy_threshold: float = 0.025
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    stabilize_weight: float = 10.0
    stabilize_spawn_xy_scale: float = 0.03
    stabilize_upright_reward_scale_deg: float = 5.0
    stabilize_action_sharpness: float = 1.5
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02
    palm_action_delta_reward_scale: float = 0.25
    finger_action_delta_reward_scale: float = 1.0

    # Legacy names kept for compatibility with older launch overrides.
    palm_approach_weight:    float = 1.0
    palm_approach_sharpness: float = 10.0

    enclosure_weight:       float = 3.0
    enclosure_sharpness:    float = 15.0
    # palm-seat: 컵을 palm 힘센서에 밀착(enclosing grasp)하도록 palm 접촉을 보상.
    # 수동 조작으로 palm 접촉이 가능함이 확인됨 → 명시적 유인 부여(현재 palm 접촉 reward 없음).
    palm_seat_weight:       float = 4.0
    cup_radius_approx:      float = 0.035
    enclosure_thumb_weight: float = 0.6

    lift_reward_weight: float = 30.0
    grasp_contact_persistence_reward_steps: int = 30
    approach_tip_contact_penalty_weight: float = -4.0
    grasp_palm_anchor_penalty_weight: float = -8.0
    palm_target_motion_penalty_weight: float = -2.0
    stabilize_spawn_xy_reward_weight: float = 40.0

    action_smoothness_palm_weight:   float = -0.10
    action_smoothness_finger_weight: float = -0.01

    # 질량 파라미터 (force-ratio/bin logging용 privileged variable)
    cup_base_mass:  float = 0.170          # kg (빈 컵 질량)
    bead_single_mass: float = _DEFAULT_BEAD_MASS  # kg per bead
    bead_scale: float = _DEFAULT_BEAD_SCALE

    min_middle_contacts_for_success: int = 0

    # Lift-entry grip readiness gate (state tracking용, reward가 아님)
    # Phase A: starting hurdle for the contact_adr 3→4→5 curriculum (was fixed 4).
    stage0_lift_start_min_contacts: int = 3
    stage0_lift_start_hold_steps:   int = 20
    lift_contact_hold_steps: int = 30
    full_grip_hold_steps:    int = 30
    lift_min_force_ratio:    float = 1.8

    # Slip proxy (no_slip_gate 계산용, 게임로직용)
    slip_proxy_threshold:                float = 1.0
    slip_proxy_contact_delta_weight:     float = 0.5
    # 0.5 → 0.0: middle 접촉을 이제 실제로 채우므로, 이번 단계(계측+critic)를 reward-neutral 로
    # 유지하기 위해 slip proxy 의 middle 기여를 0 으로 둔다. envelope 를 reward 로 적극 유도하려면
    # reward-audit 통과 후 재활성화.
    slip_proxy_middle_contact_delta_weight: float = 0.0
    slip_proxy_tilt_delta_weight:        float = 0.5
    slip_proxy_tilt_delta_scale:         float = 8.0

    # Stabilize 판정 임계값 (full_grip_ready gate용)
    stabilize_cup_lin_vel_threshold:  float = 0.04
    stabilize_cup_ang_vel_threshold:  float = 0.50
    stabilize_force_delta_threshold:  float = 0.35
    stabilize_contact_delta_threshold: float = 1.0
    stabilize_spawn_xy_success_threshold: float = 0.01

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
    contact_adr_trigger_threshold:  float = 0.5   # Phase A: lift_started_rate 50%에서 다음 허들로 (트리거 신호를 success_rate→lift_started_rate로 교체)

    # 6.2: ADR trigger moving-window 크기
    # 최근 N episode 성공률을 ADR trigger에 사용 (0: 기존 cumulative 방식 유지)
    adr_window_size: int = 500

    contact_adr_custom_cfg: dict = field(default_factory=lambda: {
        "contact": {
            # int(round(value)) 로 사용. Phase A: 3 → 5 (전 손가락 FULL-GRASPING)
            "min_contacts": (3.0, 5.0),
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
    stabilize_upright_max_deg: float = 12.0   # Phase B: success_now upright 게이트 완화 (5°는 달성 불가 → success_held=0)
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

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
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd"),
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
                "r_aj_1":  0.5,
                "r_aj_2":  0.1,
                "r_aj_3":  0.4,
                "r_aj_4":  0.60,
                "r_aj_5": -0.2,
                "r_aj_6":  0.0,
                "r_aj_7":  0.0,
                # RH56F1 우측 손 drive (approach pose)
                "r_hj_thumb_1":  HAND_APPROACH_POSE[0],   # 0.60
                "r_hj_thumb_2":  HAND_APPROACH_POSE[1],   # 0.15
                "r_hj_index_1":  HAND_APPROACH_POSE[2],   # 0.30
                "r_hj_middle_1": HAND_APPROACH_POSE[3],   # 0.30
                "r_hj_ring_1":   HAND_APPROACH_POSE[4],   # 0.30
                "r_hj_pinky_1": HAND_APPROACH_POSE[5],   # 0.30
                # mimic 추종 (= drive × multiplier, 결합 init 으로 snap 방지)
                "r_hj_thumb_3":  HAND_APPROACH_POSE[1] * 1.1425,            # 0.171
                "r_hj_thumb_4":  HAND_APPROACH_POSE[1] * 1.1425 * 0.7508,   # 0.129
                "r_hj_index_2":  HAND_APPROACH_POSE[2] * 1.1169,            # 0.335
                "r_hj_middle_2": HAND_APPROACH_POSE[3] * 1.1169,
                "r_hj_ring_2":   HAND_APPROACH_POSE[4] * 1.1169,
                "r_hj_pinky_2": HAND_APPROACH_POSE[5] * 1.1169,
                **LEFT_ARM_REST_JOINT_POS,
                **LEFT_HAND_REST_JOINT_POS,
            },
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                **_actuator_params("openarm_right_arm", 400.0, 80.0),
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                **_actuator_params("openarm_left_arm", 400.0, 80.0),
            ),
            # RH56F1 우측 손 굴곡 5 (thumb_2 + 4손가락_1) — 07.02: 30→400 (tesollo pour curl/pip/dip 참조).
            "rh56f1_right_flexion": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"
                ],
                **_actuator_params("rh56f1_right_flexion", 400.0, 60.0),
            ),
            # abduction(thumb_1) — 30→200 (tesollo abduction 참조, 굴곡보다 낮게: 반력교란 회피).
            "rh56f1_right_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_thumb_1"],
                **_actuator_params("rh56f1_right_abduction", 200.0, 35.0),
            ),
            # RH56F1 우측 손 mimic(원위) 6 — 0→400 (tesollo dip 참조). PhysxMimicJoint 미결합 시
            # 원위가 흐물해 컵을 못 감쌈 → 강성 부여.
            "rh56f1_right_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"
                ],
                stiffness=400.0, damping=60.0,
            ),
            # RH56F1 좌측 손 drive 6 (학습 비사용 → 0 hold)
            "rh56f1_left_drive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"
                ],
                **_actuator_params("rh56f1_left_drive", 30.0, 5.0),
            ),
            # RH56F1 좌측 손 mimic 추종 6 (passive)
            "rh56f1_left_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"
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
        "r_hl_thumb_4",
        "r_hl_index_2",
        "r_hl_middle_2",
        "r_hl_ring_2",
        "r_hl_pinky_2",
    )
    # 근위(proximal, finger_1) 마디 접촉 = envelope 그립 signature (tip pinch 와 구분).
    # sim-only ContactSensor(실물엔 없음) → 계측/critic privileged 용. 엄지는 mid 마디 thumb_3.
    right_middle_contact_links: tuple = (
        "r_hl_thumb_3",
        "r_hl_index_1",
        "r_hl_middle_1",
        "r_hl_ring_1",
        "r_hl_pinky_1",
    )

    tip_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_(thumb_4|index_2|middle_2|ring_2|pinky_2)",
        history_length=1,
        track_air_time=False,
    )

    # palm force sensor body = r_hl_palm_sensor (OLD rh56f1_right_plam_force_sensor 대응).
    # 구 r_al_7(팔 손목)는 palm-cup 접촉 신호가 죽어 파지 저하 → 07.01 복구.
    palm_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_palm_sensor",
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
    # 가상질량 도메인 랜덤화: {0,10,20,30}개 × 10g → 컵 실효질량 {170,270,370,470}g.
    # 물리 bead 없이 hidden-mass 로 grip force 하중강건성 학습(actor 는 질량 미관측, critic oracle).
    bead_count_min: int = 0
    bead_count_max: int = 30
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
