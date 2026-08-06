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

"""환경 설정: 5g_grasp_adapt

v10: v9 기반 + 버그 수정
- Fix 1: r_hj_thumb_1 (thumb abduction) = 0.0 고정 (v9: -0.283 → 엄지 새끼손가락 방향 치우침 수정)
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
)
from .grasp_right_preset import (
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
    """5g_grasp_adapt 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 6.0   # grasp/lift/stabilize(height-hold) (state-latched)
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
    max_pose_angle:             float = 45.0
    palm_local_workspace_radius: float = 0.10
    palm_target_max_delta:       float = 0.01
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True
    # offset 확대(palm_far1)는 실패(정책이 파지 위해 palm 재접근) → 원복.
    # palm 접촉 근본은 손가락 감싸기 preset 갇힘 → full-range 손가락 제어로 해소(별도).
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Demo reset (pour_v1_a11~a20 grasp start and pour_start lift target)
    # 로더가 /home/oem → /home/user/rl_ws/datasets 자동 폴백 (v10_3와 동일)
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
    lift_success_height: float = 0.04   # lift off 판정(중간 latch): grasp 안정화 게이트
    lift_target_height:  float = 0.10   # 최종 목표 높이 10cm (height-hold reward·success 기준)
    success_hold_steps: int = 30

    # -----------------------------------------------------------------------
    # Policy-driven reward/gate 파라미터
    # -----------------------------------------------------------------------
    # Phase 1: fingertip precision 파지 판정 — 엄지 대향 최소 손가락 수.
    precision_min_opposing: int = 2   # 엄지 + 대향 2지 이상 = 안정 파지
    # (미사용) lift latch가 precision_grasp_bool(엄지+대향2지)로 대체됨. Phase 1 이전엔
    # 접촉수 게이트였고 contact ADR가 이 값을 5까지 올렸으나 이제 적용점 없음.
    stage0_lift_start_min_contacts: int = 3
    grasp_ready_hold_steps: int = 20
    grasp_contact_persistence_reward_steps: int = 30
    full_grip_hold_steps: int = 30
    grasp_upright_threshold_deg: float = 8.0
    grasp_xy_threshold: float = 0.025
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 4.0   # Fork C: 평가 gate로 강등 (12→4)
    # stabilize = height-hold reward (lift off 후 컵을 세운 채 10cm까지 상승·유지).
    # 코어 stabilize 항은 transport_height_target_delta를 높이 품질 ramp 목표로 사용한다.
    stabilize_weight: float = 2.0   # Fork C: 평가 gate로 강등 (10→2)
    transport_height_target_delta: float = 0.10  # height-hold 목표 = lift_target_height(10cm)
    transport_height_quality_power: float = 1.0
    transport_upright_quality_power: float = 1.0
    stabilize_upright_reward_scale_deg: float = 10.0
    stabilize_action_sharpness: float = 1.5
    action_smooth_weight: float = -0.15   # -0.02→-0.15: 손가락 chatter 억제(||Δaction|| 페널티 강화)

    # 손가락 타겟 저역통과(EMA): smoothed = α·prev + (1-α)·new. full-range 절대제어라 정책
    # 스텝간 변동이 타겟을 크게 흔들어 chatter → EMA로 평활(실물 컨트롤러도 명령 평활, s2r 안전).
    # α↑=부드럽지만 느림. 0=필터off. reset 시 grasp_pose로 초기화.
    finger_target_ema_alpha: float = 0.7
    post_lift_contact_loss_weight: float = -8.0
    # Fragile 형상파괴 억제 (radial 압축 좌굴 = 종이컵 벽을 눌러 찌그러뜨림).
    # 손끝-only가 아니라 "형상 덜 파괴하며 파지"가 목적 — 감싸기/palm 지지는 무방,
    # 과도한 radial 압박만 형상파괴로 벌점한다.
    #   radial_compression = 접촉력(tip+middle)의 컵 중심 inward 성분 합.
    #   r_damage    = -damage_penalty_weight · hold_gate · relu(radial - f_safe)  (순간 과압박)
    #   damage_dose = Σ dt·relu((radial-f_safe)/f_safe)^q  (누적 형상파괴 — 성공조건에 사용)
    #   radial > f_buckle → 좌굴 종료(buckle) + buckle_penalty.
    # f_safe/f_buckle = 종이컵 좌굴 임계 placeholder. ★학습 초기 task/radial_compression
    # 분포 보고 보정(로그 먼저). 07.27 lstm_test4 실측 radial ~4.5N(hold ~2.7N) 기준 현 값.
    damage_penalty_weight: float = 3.0
    f_safe:   float = 3.0    # N, 안전 radial 상한(형상파괴 시작, 초과분 penalty·dose 누적)
    f_buckle: float = 8.0    # N, 좌굴(파손) radial 임계
    buckle_penalty: float = 10.0   # 파손 종료 시 음의 보상
    # 누적 damage dose(형상파괴 총량). 성공조건: dose < damage_dose_success_max.
    damage_dose_q: float = 2.0            # dose 지수 (설계 §6: relu((radial-f_safe)/f_safe)^q)
    damage_dose_success_max: float = 1.0  # 성공 허용 형상파괴 상한(초과 시 부숨 = 실패)
    hand_residual_magnitude_weight: float = -0.005
    hand_residual_scale: float = 0.35  # rad, grasp pose 기준 residual 반경.
    # GRASP→FULL_GRIP 간극(~0.13~0.27rad)을 덮고 개방 여유까지. 한계로 clamp됨.

    # approach term의 fingertip_side_dist 기하 계산용
    cup_radius_approx:      float = 0.045
    enclosure_thumb_weight: float = 0.6

    # Final success upright gate (10cm height-hold 최종 성공 판정).
    stabilize_upright_max_deg: float = 5.0

    # 컵 무게 (reward 전용 privileged): cup_weight = (cup_base_mass + bead·bead_single_mass)·g
    cup_base_mass:             float = 0.170   # kg (빈 컵 질량)
    bead_single_mass:          float = 0.010   # kg per bead
    slip_shear_eps:            float = 0.5     # N, shear severity KPI 분모 오프셋 (진단 전용)

    # ---------------------------------------------------------------------
    # Adaptive grasping 목적함수 (Phase 1: friction-aware no-slip 최소 힘)
    # compute_adaptive_grip_terms 가 지배항. lift/stabilize는 평가 gate로 강등.
    #   r_secure    = secure_weight · hold · exp(-secure_slip_sharpness · cup_slip_speed)
    #   r_efficient = -force_efficiency_weight · hold · clamp(ΣF_n/cup_weight, max=cap)
    #   r_drop      = -drop_penalty_weight · (떨어뜨림 + 들린 채 접촉손실)
    # 평형 = "안 미끄러지는 최소 힘" → mass·friction 적응 emergent.
    # (구 Gaussian adaptive_force·shear no_slip 항 및 미사용 reward cfg는 제거됨 — Phase 3 정리)
    # ---------------------------------------------------------------------
    secure_weight:             float = 10.0   # no-slip(컵 정지) 보상 — 지배항
    secure_slip_sharpness:     float = 30.0   # cup_slip_speed[m/s] → secure_quality 감쇠
    force_efficiency_weight:   float = 2.0    # ΣF_n/cup_weight 당 비용 (최소 힘 압박)
    drop_penalty_weight:       float = 8.0    # 떨어뜨림/접촉손실 페널티
    force_ratio_cost_cap:      float = 6.0    # 효율 비용 ratio 상한 (blowup 방지)

    # lift_reward — Fork C: 평가 bootstrap gate로 강등 (30→6)
    lift_reward_weight: float = 6.0

    # lift_height_bonus — lstm_test10 진단: main lift(lift_height_quality)가 4cm에서
    # clamp 포화 → 4cm 위 gradient 0. efficient가 grip을 깎아 5cm 약파지 hold에 고착
    # (success 0). contact-독립 height 보상을 10cm까지 열어 "형상 유지+10cm까지 리프트"를
    # reward 최대로 만든다. clamp=2.5 = lift_target(0.10)/lift_success(0.04).
    lift_height_bonus_weight: float = 1.5   # 4.0→1.5: lift-지배(reward/lift 12) 완화 → grasp 품질 비중 회복
    lift_height_bonus_clamp:  float = 2.5

    # success_bonus — Fork C: 10cm·upright·5접촉 유지 평가 bonus로 강등 (20→4)
    success_bonus_weight: float = 4.0

    # -----------------------------------------------------------------------
    # ADR — contact curriculum (threshold=0.1, 먼저 진행)
    # -----------------------------------------------------------------------
    # Phase 1: contact curriculum 비활성화. lift/success gate가 precision_grasp_bool
    # (엄지+대향2지)로 바뀌어 min_contacts(2→5) ADR의 적용점이 사라졌다(설계 §9:
    # "contact ADR 2→5 증가 제거"). general difficulty ADR(enable_adr)은 유지.
    enable_contact_adr:             bool  = False
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
            # 8192 검증된 값(num_envs 확장은 ROI 낮아 미채택 — 병목은 접촉 물리+per-step sync).
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
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"),
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
                "r_hj_thumb_1": 0.0, "r_hj_thumb_2": -1.57, "r_hj_thumb_3": -0.5, "r_hj_thumb_4": 0.0,
                "r_hj_index_1": 0.0, "r_hj_index_2":  0.0,  "r_hj_index_3":  0.0, "r_hj_index_4": 0.0,
                "r_hj_middle_1": 0.0, "r_hj_middle_2":  0.0,  "r_hj_middle_3":  0.0, "r_hj_middle_4": 0.0,
                "r_hj_ring_1": 0.0, "r_hj_ring_2":  0.0,  "r_hj_ring_3":  0.0, "r_hj_ring_4": 0.0,
                "r_hj_pinky_1": 0.0, "r_hj_pinky_2":  0.0,  "r_hj_pinky_3":  0.0, "r_hj_pinky_4": 0.0,
                **LEFT_ARM_REST_JOINT_POS,
            },
        ),
        actuators={
            # head pan/tilt 카메라 관절(신규 USD). revolute라 DOF로 잡히므로
            # actuator 커버리지 필수(없으면 크래시/자유회전). 기본각 0 고정 hold.
            # obs/action은 이름 기반이라 head 미포함 유지 → 차원 불변.
            "head_camera": ImplicitActuatorCfg(
                joint_names_expr=["head_j_(pan|tilt)"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                **_actuator_params("openarm_right_arm", 400.0, 80.0),
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                **_actuator_params("openarm_left_arm", 400.0, 80.0),
            ),
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_1"],
                **_actuator_params("tesollo_hand_abduction", 30.0, 5.0),
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_2"],
                **_actuator_params("tesollo_hand_curl", 90.0, 5.0),
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_3"],
                **_actuator_params("tesollo_hand_pip", 90.0, 5.0),
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_4"],
                **_actuator_params("tesollo_hand_dip", 90.0, 5.0),
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_gripper_[1-2]"],
                **_actuator_params("openarm_left_gripper", 400.0, 80.0),
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # -----------------------------------------------------------------------
    right_tip_contact_links: tuple = (
        "r_hl_thumb_tip",
        "r_hl_index_tip",
        "r_hl_middle_tip",
        "r_hl_ring_tip",
        "r_hl_pinky_tip",
    )

    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_4",
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_3",
        history_length=1,
        track_air_time=False,
    )

    palm_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_palm",
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
    # Phase 2: segmented-shell deformable cup (기본 비활성 — rigid 태스크 무영향).
    # True면 env가 cup을 Articulation으로 스폰하고 리셋 시 패널 관절(수동 스프링)을 0으로
    # 되돌린다. 스폰 자산·actuator(스프링 gain)는 deformable 서브클래스가 cup_cfg override로 지정.
    # 패널 관절은 정책 action에 편입되지 않음(컵은 로봇과 별개 articulation) → 순수 수동 스프링.
    # -----------------------------------------------------------------------
    cup_is_articulated: bool = False
    deform_panel_count: int = 12   # deformable cup 패널 수(USD와 일치). contact 필터 구성에 사용.

    # Phase-1 grasp-in-place: 테이블 위 컵에 lift 보상 0 → lift 없이 gentle 파지만 학습(fix_root_link은
    # Isaac Lab이 로봇 articulation을 깨뜨려 미사용). success=precision+안부숨(dose<max).
    # True면 success를 "precision 파지 + 안 부숨(dose<max) 유지"로 재정의(lift/upright 무관).
    # lift-지배 보상이 사라져 파지 품질에 집중 → "안 부수고 잡기" 검증에 적합.
    cup_anchored: bool = False

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
    eval_mass_shift_target_bead_count: int = 30

    # -----------------------------------------------------------------------
    # Phase 3: 동적 mass 이벤트 (물 추가) — lift 후 hidden bead를 컵으로 물리 teleport.
    # eval_mass_shift(oracle 값만 변경)와 달리 실제 하중이 증가한다.
    # 리셋은 가벼운 컵(bead_count_max 낮게)으로, lift 후 target까지 bead를 추가한다.
    # actor는 mass를 관측하지 않으므로(tactile 추론) 무게 증가를 느껴 grip을 조절해야 한다.
    # -----------------------------------------------------------------------
    mass_shift_enabled:              bool  = False   # Phase 3 run에서만 True
    mass_shift_target_bead_count:    int   = 30      # 추가 후 목표 bead 수
    mass_shift_height_threshold:     float = 0.08    # 이 높이(m) 이상 유지 시 발동
    mass_shift_delay_steps:          int   = 15      # 높이 도달 후 유지 step 수

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES


@configclass
class GraspRightEnvCfgNoActorMass(GraspRightEnvCfg):
    """Asymmetric teacher config: actor excludes oracle mass, critic keeps it.

    **@configclass 필수(sim2real 버그 수정)**: 데코레이터 없으면 부모 dataclass __init__이
    인스턴스 속성을 부모 기본값으로 세팅해 아래 override가 무시된다 → actor가 oracle bead
    mass를 관측(obs 134)하며 학습돼 실물 배포 불가(실물엔 oracle mass 없음). @configclass로
    override 적용 → actor tactile-only(obs 133), sim2real 배포 가능.
    ⚠️ obs 134→133 변경이므로 이전 obs-134 체크포인트(test11 등)와 비호환 → 클린 재학습 필요.
    """

    observation_space: int = NUM_OBSERVATIONS_NO_MASS
    num_observations: int = NUM_OBSERVATIONS_NO_MASS
    actor_observe_bead_mass: bool = False


@configclass
class GraspRightEnvCfgMassShift(GraspRightEnvCfgNoActorMass):
    """Phase 3: 동적 mass(물 추가). 리셋은 가벼운 컵(0~10 bead), lift 후 target까지 추가.

    NoActorMass 상속 → actor tactile-only(obs 133, sim2real 정합). 무게 증가를 oracle가
    아니라 tactile로 느껴 grip을 조절해야 한다(adapt의 핵심). @configclass 필수.
    ⚠️ obs 133이라 obs-134 체크포인트(massshift2 등)와 비호환 → 클린 재학습 필요.
    """

    mass_shift_enabled: bool = True
    bead_count_min: int = 0
    bead_count_max: int = 10   # 리셋 가벼움 → shift로 mass_shift_target_bead_count까지 추가


@configclass
class GraspRightEnvCfgDeformable(GraspRightEnvCfgNoActorMass):
    """Phase 2: segmented-shell deformable cup.

    컵을 12 rigid 패널 + 접선 힌지 스프링 articulation으로 교체 → **과파지 = 실제 패널
    안쪽 눌림(형상파괴)** 이 되어 power-grip이 벌점화된다. under-grip(떨굼)도 over-grip(부숨)도
    불가 → 정책이 "딱 필요한 만큼"의 mass 적응 파지를 강제로 학습(power-grip 수렴 해소).

    Gate B: 높은 stiffness(rigid-like, 패널 거의 안 움직임)로 현 base와 동등 파지 회귀 검증.
    Gate D: stiffness 커리큘럼으로 점진 물렁하게(진짜 변형 학습). armature는 경량 패널
    안정화에 필수(Gate A 실증: 없으면 NaN 폭주).

    actor obs 133(tactile-only, NoActorMass 상속 → sim2real 정합).
    ⚠️ 컵 물리 자체가 바뀜(동역학 변화) = rigid 체크포인트 비전이 → **fresh 재학습 필수**.
    obs/action 차원은 불변이라 코드 호환만 유지.
    """

    cup_is_articulated: bool = True
    # lift 복원(2026-08-05): 고정(anchored) 검증서 gentle 파지는 되나 mass 적응 불가 판명
    # (컵이 안 떨어져 촉각 mass 신호 없음 → grip_force 무게 무관 균일 8.5N power-grip).
    # 적응의 유일한 교사 = "힘 부족→미끄럼/낙하→촉각 피드백" 루프 → lift 필요.
    # cup_anchored=False + lift_reward/lift_height_bonus는 base값(6.0/1.5) 상속.
    cup_anchored: bool = False

    # -------------------------------------------------------------------
    # 리워드 teardown (2026-08-04): 핵심만 남기고 파생항 제거 → secure/drop/slip 재설계 기반.
    #   - approach_tilt_penalty / stabilize 제거(0): 지금 단계 불필요.
    #   - r_damage는 grasp 하위로(env: hold_gate → full_tip_contact 게이트) — cfg는 weight 유지.
    #   - secure/drop/slip weight는 미변경(함수 재설계 대기).
    # v2 (2026-08-07): 성공 funnel 진단 — 병목=10cm 미달(평균 7cm, 32%만 도달). tilt 4°(통과)·
    #   precision 0.87(OK). 원인=4cm 위로 끄는 힘 부족(lift_reward는 4cm clamp 포화, dense 견인은
    #   lift_height_bonus 1.5뿐인데 secure 10에 눌림) + success_bonus 0(완주 보상 없음).
    #   → success_bonus 재설정(sparse 완주보상 4.0) + lift_height_bonus 강화(dense 견인 1.5→2.5,
    #   4.0은 lift-지배 이력이라 보수적). 10cm 상승엔 더 firm 파지 필요 → adaptation gradation 시너지.
    # -------------------------------------------------------------------
    approach_tilt_penalty_weight: float = 0.0
    stabilize_weight: float = 0.0
    success_bonus_weight: float = 4.0        # 0→4.0: 10cm+upright+precision+intact 완주 보상 재설정
    lift_height_bonus_weight: float = 2.5    # 1.5→2.5: 4→10cm dense 견인 강화(4.0은 lift-지배 위험)

    # -------------------------------------------------------------------
    # damage 신호 단위 전환: 힘 proxy(N) → 실제 패널 최대 변형(deg).
    # radial_compression = compute_panel_deformation_deg(패널 힌지각) [deg].
    #   r_damage    = -damage_penalty_weight · hold_gate · relu(deform - f_safe)
    #   damage_dose = Σ dt·relu((deform-f_safe)/f_safe)^q  (성공조건: dose < max)
    #   deform > f_buckle → 좌굴(파손) 종료 + buckle_penalty.
    # ★ placeholder — 학습 초기 task/radial_compression(=deg) 분포 보고 보정(로그 먼저).
    #   Gate B(stiffness 5.0)에선 파지 시 ~8deg < f_safe라 damage 미발동(rigid 회귀).
    #   Gate D 커리큘럼서 stiffness 낮추면 변형↑ → damage 활성 → gentle 적응 강제.
    # -------------------------------------------------------------------
    # K와 f_safe 분리(08.03 발견): K=파지가능성, f_safe=damage 임계. K=2.5는 너무 물렁해
    # 파지 자체가 안 됐음(패널이 손끝 피해 후퇴). → K=5.0(파지 가능) 유지 + f_safe 낮춰
    # 과파지만 벌점. K=5.0서 파지력↔변형 비례(gentle ~2° / firm ~6°)라 f_safe 3.5°면
    # gentle 통과·firm 벌점. f_buckle은 backstop(K=5.0선 도달 드묾).
    f_safe:   float = 3.5     # deg, 안전 패널각(초과분 penalty·dose)
    f_buckle: float = 10.0    # deg, 좌굴(파손) 패널각 — backstop
    damage_penalty_weight: float = 0.3   # relu(deform-3.5)~2.5 스케일. 로그 보고 보정.

    cup_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/deformable_cup.usd"),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        # 패널 관절 = 정책 미제어 수동 스프링(target=default 0). Gate B는 rigid-like 고강성.
        actuators={
            "panels": ImplicitActuatorCfg(
                joint_names_expr=["revolute_.*"],
                # K=5.0: 파지 가능성 확보(K=2.5는 물렁해 파지 실패). 적응 압박은 K가 아니라
                # f_safe(3.5°)를 낮춰 만든다(과파지=변형↑=벌점). firm grip ~6°>f_safe → damage 발동.
                stiffness=5.0,
                damping=0.1,
                armature=1.0e-3,    # 경량 패널 NaN 방지(Gate A 필수)
                effort_limit=1.0e6,
            ),
        },
    )
