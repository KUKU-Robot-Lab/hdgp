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
    action_smooth_weight: float = -0.02
    post_lift_contact_loss_weight: float = -8.0
    # Phase 1 envelope penalty 제거(geometry penalty는 secure와 상충해 실패 — radial로 일원화).
    # 필드는 존치하되 0. (Phase 1 lstm_test3: 감싸기 회귀 실증)
    envelope_penalty_weight: float = 0.0
    # Phase 2 radial-압축 fragile damage (하드웨어 제약: 손끝-only를 물리로 유도).
    #   radial_compression = 접촉력(tip+middle)의 컵 중심 inward 성분 합.
    #   r_damage = -damage_penalty_weight · hold_gate · relu(radial - f_safe)
    #   radial > f_buckle → 파손 종료(buckle) + buckle_penalty.
    # f_safe/f_buckle은 종이컵 추정 placeholder(설계 §4: f_safe≈0.6~0.8·F_yield).
    # ★ 학습 초기 task/radial_compression 분포를 보고 보정할 것(로그 먼저).
    damage_penalty_weight: float = 3.0
    # 07.27 lstm_test4 초기 로그로 보정: 실측 radial_compression ~4.5N(hold ~2.7N)이라
    # 기존 f_safe=8/f_buckle=15는 penalty·buckle이 거의 안 걸려 damage 신호가 꺼졌음.
    f_safe:   float = 3.0    # N, 안전 radial 압축 상한(초과분 penalty)
    f_buckle: float = 8.0    # N, 좌굴(파손) radial 압축 임계(강한 감싸기만)
    buckle_penalty: float = 10.0   # 파손 종료 시 음의 보상 크기
    hand_residual_magnitude_weight: float = -0.005
    hand_residual_scale: float = 0.15

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
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES


class GraspRightEnvCfgNoActorMass(GraspRightEnvCfg):
    """Asymmetric teacher config: actor excludes oracle mass, critic keeps it."""

    observation_space: int = NUM_OBSERVATIONS_NO_MASS
    num_observations: int = NUM_OBSERVATIONS_NO_MASS
    actor_observe_bead_mass: bool = False
