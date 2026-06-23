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

"""환경 설정: 5g_grasp_right_v10-3

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
    """5g_grasp_right_v10-3 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0  # grasp/lift/stabilize
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 134 optional mass/debug actor
    action_space:      int = NUM_ACTIONS               # 27
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 170 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS
    actor_observe_bead_mass: bool = True

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    palm_local_workspace_radius: float = 0.1
    palm_target_max_delta: float = 0.01
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0
    approach_min_steps: int = 10
    approach_palm_local_z_min: float = -0.02
    approach_palm_local_z_max: float = 0.08
    # approach orientation: 고정 pregrasp 자세(컵 향함) 주변 bounded euler residual (grasp-v1 방식, ±deg)
    approach_palm_residual_rot_deg: float = 5.0
    grasp_body_local_z_min: float = -0.04
    grasp_body_local_z_max: float = 0.05
    prelift_max_cup_height_delta: float = 0.01
    prelift_cup_lin_vel_threshold: float = 0.04
    prelift_rim_lift_penalty_weight: float = 1.0
    grasp_palm_delta_scale: float = 0.25
    # lift phase palm 상승 한계. 0.03(3cm) < lift_success_height(4cm)이라 컵이 4cm에
    # 물리적으로 도달 불가 → success_held=0 plateau였음(lstm_test11 진단). 0.07로 키워
    # palm이 4cm 위로 들 수 있게 함. per-step은 palm_target_max_delta(0.01)로 여전히 rate-limit.
    lift_palm_delta_xyz: float = 0.07

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
    # Demo reset (pour_v1_a11~a20 grasp start and pour_start lift target)
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
    success_hold_steps: int = 30
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.2

    # -----------------------------------------------------------------------
    # Policy-driven reward/gate 파라미터
    # -----------------------------------------------------------------------
    stage0_lift_start_min_contacts: int = 3   # Phase A: contact_adr 3→4→5 커리큘럼 시작 허들 (기존 고정 5)
    grasp_ready_hold_steps: int = 20
    grasp_contact_persistence_reward_steps: int = 30
    full_grip_hold_steps: int = 30
    grasp_upright_threshold_deg: float = 8.0
    grasp_xy_threshold: float = 0.025
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    stabilize_weight: float = 10.0
    stabilize_spawn_xy_scale: float = 0.03
    stabilize_upright_reward_scale_deg: float = 10.0
    stabilize_ang_vel_sharpness: float = 2.0
    stabilize_lin_vel_sharpness: float = 10.0
    stabilize_action_sharpness: float = 1.5
    stability_reward_weight: float = 1.0
    action_smooth_weight: float = -0.02
    post_lift_contact_loss_weight: float = -8.0
    hand_residual_magnitude_weight: float = 0.0
    hand_residual_scale: float = 0.15
    # close-grasp 전용 residual 확대 — envelope grip(full_grip까지 말아쥐기) 가능하게.
    # lift는 hand_residual_scale(0.15) 유지해 hold 안정성 보존.
    hand_close_residual_scale: float = 0.5

    # Legacy names kept for compatibility with older launch overrides.
    # R0. palm_approach
    palm_approach_weight:    float = 0.5
    palm_approach_sharpness: float = 10.0

    # R1. fingertip_enclosure (v8: 4.0 → v9: 3.0, slip/efficiency 강화로 비중 완화)
    enclosure_weight:       float = 3.0
    enclosure_sharpness:    float = 15.0
    cup_radius_approx:      float = 0.045
    enclosure_thumb_weight: float = 0.6

    # R1b. force_balance (v8: 8.0 → v9: 6.0 → v9.4: 3.5)
    # 축소 이유: 과포화 시 force_balance local-min → multi_phalanx 저하 (test3 붕괴 원인)
    # 역할은 보조 제약 수준으로 유지 (엄지 대립 약화 방지)
    force_balance_weight:    float = 0.0
    force_balance_sharpness: float = 8.0

    # R1c. multi_phalanx_contact (v10.2: 12.0 → 16.0)
    # 증가 이유: tip-only local optimum 탈출, r1d/e와 함께 deep envelope 강화
    multi_phalanx_weight: float = 0.0

    # R1d. middle_phalanx_guide (v10.1 신규)
    # middle phalanx → cup 거리 기반 exp reward (enclosure와 동일 구조, 항상 활성)
    # actor obs middle_to_cup 15D에 직접 대응하는 reward gradient 제공 (위치 단계)
    middle_guide_weight:    float = 0.0
    middle_guide_sharpness: float = 10.0

    # R1e. middle_contact (v10.2 신규)
    # middle_norm 단독 reward — tip contact 여부와 무관 (접촉 단계)
    # finger_depth(tip×middle 곱)와 달리 tip-only 상태에서도 gradient 살아있음
    # tip-only 초반 고착 이후 middle contact 탐색을 독립적으로 유도
    middle_contact_weight: float = 0.0
    middle_contact_envelope_bonus_weight: float = 0.0
    min_middle_contacts_for_success: int = 4

    # Final stationary stabilization upright gate.
    stabilize_upright_max_deg: float = 5.0

    # Grasp phase에서 컵을 세운 채 감싸도록 유도한다.
    grasp_upright_weight: float = 0.0

    # Grasp phase에서 컵을 밀거나 과도하게 파고드는 접근을 억제한다.
    grasp_cup_xy_penalty_weight: float = 4.0
    grasp_cup_xy_penalty_margin: float = 0.01
    grasp_cup_tilt_penalty_weight: float = 0.08
    grasp_cup_tilt_penalty_margin_deg: float = 8.0
    grasp_palm_overshoot_penalty_weight: float = 4.0

    # R2. slip_reward (v9 신규): cup 수평 속도 기반 slip proxy
    # gate: grasp phase AND contact 시 활성
    # R_slip = slip_weight * gate * exp(-slip_sharpness * cup_horiz_vel)
    slip_weight:    float = 0.0
    slip_sharpness: float = 20.0

    # Legacy preload fields kept only for older launch overrides.
    # 설계: -w * has_contact * relu(target_ratio - force_ratio)
    #   → 목표 ratio 미달 시 선형 패널티 (상한이 없으므로 과도 grip은 억제 안 함)
    #   → R3(adaptive_force)의 상한 억제와 쌍으로 동작
    # target_ratio=1.6: 탐색 시작점 (권장 sweep 범위 1.2~2.0)
    preload_penalty_weight:     float = 0.0
    preload_force_target_ratio: float = 1.6
    preload_start_step:         int   = 400

    # R3. Adaptive Force Reward (v10: 단조감소 → Gaussian target 방식으로 변경)
    # 설계 철학:
    #   기존(v9): exp(-decay × ratio) — 단조감소, slip과 상충하여 붕괴 유발
    #   변경(v10): exp(-sharpness × (ratio - target)²) — Gaussian, target ratio에 sweet spot
    #
    #   → policy가 bead_mass_normalized 관측을 활용해 target ratio 유지하는 adaptive grip 학습
    #   → force가 부족해도(ratio<target) 페널티, 과해도(ratio>target) 페널티
    #   → slip_reward와 방향 일치: target ratio(2.5×mg)는 slip 방지에 충분한 수준
    #
    # R_af = weight * is_lift * contact * exp(-sharpness * (ratio - target)²)
    #   target=2.5: ratio=0→0.04, ratio=1→0.33, ratio=2.5→1.0(최대), ratio=4→0.33, ratio=6→0.04
    adaptive_force_weight:     float = 0.0
    af_target_ratio:           float = 2.5   # 최적 grip = 2.5 × mg
    af_sharpness:              float = 0.5   # Gaussian 폭 (클수록 좁은 sweet spot)
    cup_base_mass:             float = 0.170  # kg (빈 컵 질량)
    bead_single_mass:          float = 0.010  # kg per bead

    # R_ft. fingertip_guide: fingertip → cup 거리 기반 (항상 gradient, seed 분산 방지)
    # sim2real 영향 없음: fingertip_pos는 FK 또는 FT 센서로 실 로봇에서도 획득 가능
    # cup_pos: 관측 노이즈 처리된 값 사용 (σ_cp 적용됨)
    fingertip_guide_weight:    float = 0.0
    fingertip_guide_sharpness: float = 5.0

    # R_ft2. thumb_tip_direction: 엄지 distal->tip 축이 컵 중심을 향하도록 유도
    # HAND_GRASP_POSE anchor와 별개로, 접촉 위치는 맞지만 엄지가 돌아가는 local optimum을 억제
    thumb_tip_direction_weight: float = 0.0
    thumb_tip_direction_sharpness: float = 4.0
    thumb_tip_direction_distance_scale: float = 0.08

    # R5. force_smooth (v9 신규): 파지력 변화율 억제 (sim2real 안정성)
    force_smooth_weight: float = 0.0
    force_smooth_penalty_cap: float = 2.0
    # 6.5: lift phase 초반 N step 동안 force_smooth 완화 (0: 비활성)
    # lift 시작 직후 grip force 급변을 허용해 과도 패널티 방지
    force_smooth_lift_warmup_steps: int = 20

    # R6. lift_reward (v8: 20.0 → v9: 6.0 → v9.2: 20.0, slip local-min 탈출)
    lift_reward_weight: float = 30.0

    # R8. success_bonus: lift 성공 유지 중 step당 보너스 (slip local-min 탈출)
    success_bonus_weight: float = 20.0

    # R9. full_contact_bonus: 5손가락 전체 접촉 보너스 (sim2real envelope grip 유도)
    # step당 보너스 → 유지할수록 누적 (grasp + lift phase 모두)
    # v10.1: 5.0 → 8.0, middle_guide와 함께 5-contact envelope 강화
    full_contact_bonus_weight: float = 0.0

    # R10. thumb pose / grasp shape consistency
    # v10.1: thumb_pose_anchor_weight 1.2 → 2.5 (엄지 미끄러짐 방지)
    # v10.2: thumb_pose_anchor_weight 2.5 → 4.0 (test3/4 분석: anchor_error 단조증가 확인)
    # v10.1: thumb_pose_anchor_sharpness 8.0 → 10.0 (error plateau 좁히기)
    # 전체 hand pose imitation은 adaptive closure를 방해하므로 고정 관절 anchor 수준으로만 둔다.
    thumb_pose_anchor_weight: float = 0.0
    thumb_pose_anchor_sharpness: float = 10.0
    thumb_slide_penalty_weight: float = 0.0
    thumb_slide_z_margin: float = 0.01
    grasp_shape_consistency_weight: float = 0.0
    grasp_shape_consistency_sharpness: float = 6.0

    # Lift 후반 hold 안정화: 컵 속도와 action 변화를 낮게 유지
    lift_hold_stability_weight: float = 0.0
    lift_hold_stability_start_step: int = LIFT_PHASE_STEPS - 90
    lift_hold_cup_vel_sharpness: float = 15.0
    lift_hold_cup_ang_vel_sharpness: float = 2.0
    lift_hold_action_sharpness: float = 2.0

    # Thumb downward shortcut 억제
    thumb_curl_downward_action_scale: float = 0.25
    thumb_curl_max_downward_delta: float = 0.05

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
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.27
    object_spawn_y_center: float = -0.10
    object_spawn_z:        float = 0.297
    object_spawn_xy_range: float = 0.06
    # 컵 스폰 위치: palm_ee 프레임 +X(손바닥 정면)로 이만큼 앞에 스폰 (열린 손 정면, 엄지 충돌 회피)
    cup_spawn_palm_front_x: float = 0.05

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
    bead_count_max: int = 30                           # 이산: {0, 10, 20, 30}개
    bead_spawn_z_offset: float = 0.035

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
