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

"""환경 설정: 5g_pour_right_v3

v7: Fabrics 팔 학습(6D palm) + per-finger lerp(5D) + sim2real 가능 obs
- Action: 11D (6D palm pose + 5D per-finger lerp)
- Observation: actor 106D / critic 143D (asymmetric)
- Episode: Grasp phase (Fabrics arm + finger 정책) + Lift phase (scripted arm + frozen hand)
- Contact: fingertip FT sensor (actor, real-compatible) + distal/middle sensors (critic only)
"""

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
from .pour_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .pour_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
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
            static_friction=0.3,
            dynamic_friction=0.2,
            restitution=0.05,
            friction_combine_mode="average",
            restitution_combine_mode="min",
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
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 106 (actor)
    action_space:      int = NUM_ACTIONS               # 11
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 143 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

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

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # actor obs에만 적용; critic obs는 privileged clean state 유지
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01    # joint position σ [rad]
    obs_noise_joint_vel: float = 0.05    # joint velocity σ [rad/s]
    obs_noise_body_pos:  float = 0.005   # FK body position σ [m] (palm, fingertip)
    obs_noise_cup_pos:   float = 0.015   # cup position observation σ [m]

    # GUI helper
    enable_visual_markers: bool = False  # 시각화 마커 (붉은/푸른 점) 표시 여부

    # ADR: noise 스케줄 (low→high) — 성공률이 오르면 강건성 위해 노이즈 증대
    enable_noise_adr: bool = True
    noise_adr_custom_cfg: dict = {
        "noise": {
            "obs_noise_joint_pos": (0.002, 0.01),
            "obs_noise_joint_vel": (0.01, 0.05),
            "obs_noise_body_pos":  (0.001, 0.005),
            "obs_noise_cup_pos":   (0.003, 0.015),
        }
    }
    noise_adr_num_increments: int = 40
    noise_adr_increment_interval: int = 20000
    noise_adr_trigger_threshold: float = 0.3


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

    success_bead_cross_count: int = 1
    success_target_fill_ratio: float = 0.50
    success_spill_max: float = 0.20   # curriculum 성공도 과도한 spill은 허용하지 않음
    # 최종 목표 진척도는 curriculum 성공과 별도로 더 엄격하게 측정한다.
    final_success_target_fill_ratio: float = 0.95
    final_success_spill_max: float = 0.05

    # -----------------------------------------------------------------------
    # Policy action / pouring target
    # -----------------------------------------------------------------------
    palm_delta_xyz: float = 0.5   # 0.3 → 0.5: workspace-target 거리 불일치 해소
    warmstart_collect_palm_delta_xyz: float = 0.10
    palm_delta_rot_deg: float = 120.0  # 45→120: cup 135° tilt 도달 가능하도록 확장
    tilt_action_gate_xy_near: float = 0.04
    tilt_action_gate_xy_far: float = 0.12
    approach_xy_off_near: float = 0.02
    approach_xy_off_far: float = 0.03
    left_cup_world_z_offset: float = -0.12   # -0.08→-0.12: target cup 추가 하강 (높은 Z에서 tilt 시 spill 과다)
    reward_gate_xy_scale: float = 10.0   
    reward_gate_clear_scale: float = 80.0
    reward_gate_tilt_scale: float = 15.0
    reward_clearance_min: float = 0.015
    reward_tilt_cos_min: float = 0.15

    # -----------------------------------------------------------------------
    # Warmstart quality / success
    # -----------------------------------------------------------------------
    lift_success_height: float = 0.08   # 0.03→0.08: 리프트 안정성 강화
    warmstart_cache_min_lift_height: float = 0.15   # 테이블 기준 최소 15cm (≈ 컵 절반 높이 이상)
    warmstart_cache_min_contacts:    int   = 3       # 최소 3손가락 접촉 (기존 2보다 강화)
    warmstart_stable_hold_steps:     int   = 30      # 연속 30프레임(0.5s) 유지해야 저장

    enable_success_warmstart_reset: bool = False   # v3 True → v4 False
    success_warmstart_cache_size: int = 512
    success_cache_reset_ratio: float = 0.0         # v3 0.40 → v4 0.0 (mid-pour 리셋 차단)
    grasp_warmstart_reset_ratio: float = 1.0       # 0.85 → 1.0: 100% warmstart (v7 grasp 완료 상태)
    success_cache_store_min_bead_cross_fraction: float = 0.05
    success_cache_store_min_bead_in_target_fraction: float = 0.02

    # -----------------------------------------------------------------------
    # [v4 신규] Trajectory Capture — 성공 에피소드 궤적 수집 (BC loss용)
    # -----------------------------------------------------------------------
    enable_trajectory_capture: bool = True
    trajectory_capture_window: int = 200      # 마지막 N 스텝 캡처 (200@60Hz ≈ 3.3s)
    trajectory_buffer_capacity: int = 256     # 성공 궤적 최대 저장 수
    trajectory_success_bead_threshold: float = 0.50   # bead_in_target >= 50%
    trajectory_success_spill_max: float = 0.10        # spill < 10%
    trajectory_min_steps: int = 60            # 최소 궤적 길이 (1s @ 60Hz)

    # -----------------------------------------------------------------------
    # [v4 신규] BC (Behavioral Cloning) Aux Loss — PourLstmBCAgent 전용
    # -----------------------------------------------------------------------
    bc_seq_len: int = 16           # LSTM BPTT sequence length
    bc_batch_size: int = 64
    bc_loss_weight_init: float = 1.0
    bc_loss_weight_final: float = 0.1
    bc_loss_warmup_epochs: int = 500
    bc_loss_decay_epochs: int = 3000
    bc_min_buffer_size: int = 20   # 이 이하이면 BC 비활성
    success_reset_hold_steps: int = 20
    success_reset_palm_delta_xyz: float = 0.05
    success_mouth_xy_threshold: float = 0.030
    success_z_clearance_min: float = 0.015
    success_z_clearance_max: float = 0.050
    success_hold_steps: int = 10
    drop_force_hold_steps: int = 10
    # 소스 컵이 비어있는 상태가 N 스텝 연속 지속되면 에피소드 종료
    # 비드 낙하 + 착지에 ~0.3~0.5초 필요 → 60 steps (1.0s @ 60Hz) 여유
    source_empty_hold_steps: int = 60

    weight_grasp_maintain: float = 0.50   # 2.00→0.50: r_hold local optimum 해소
    weight_contact_maintain: float = 0.30  # 1.00→0.30
    weight_force_balance: float = 0.30
    weight_finger_curl: float = 0.50      # 2.00→0.50: r_hold max 5.0→1.3/step
    weight_approach_xy: float = 10.00   # cup center 기반 approach → 더 직접적으로 유도 가능
    weight_approach_z: float = 3.00
    weight_cup_upright: float = 0.80

    weight_transport_progress: float = 24.00  # 6.00→12.00→24.00: success neighborhood 유입 강화
    weight_prepour_dir: float = 1.50
    weight_prepour_align: float = 1.00
    weight_cross: float = 80.00
    weight_capture: float = 160.00
    weight_pour_align: float = 2.00  # pour stage 중 방향 정렬 유지 (0→2.0)
    weight_first_capture_bonus: float = 40.00
    weight_tilt_onset_bonus: float = 10.00   # 80° 부근 첫 진입에 더 강한 bridge reward
    tilt_onset_dot_threshold: float = 0.34   # [v4] 0.17→0.34: source_up_dot < 0.34 (>70° 기울기) 시 트리거 — pour gate(80°)와 충분한 간격 확보
    tilt_onset_dist_threshold: float = 0.08  # success neighborhood 안에서만 onset 보상
    weight_terminal_pour: float = 60.00        # 마지막 step pour pose 유지 보너스
    terminal_pour_tilt_thresh: float = 0.17    # [v4] 0.0→0.17: pour_binary_tilt_thresh와 일치 (>80° tilt)
    weight_terminal_capture: float = 200.0     # [v4 신규] 에피소드 종료 시 bead_in_target_fraction 비례 최종 보너스
    terminal_pour_mouth_xy_thresh: float = 0.03
    terminal_pour_mouth_z_min: float = -0.01
    terminal_pour_mouth_z_max: float = 0.03
    weight_success_overfill: float = 0.0  # 성공 기준 초과 비율 보너스 (0=비활성화)
    weight_success: float = 100.00  # 30→300→100
    weight_spill: float = 3.00     # mouth-to-mouth gate 이후에도 overspill 억제를 위해 penalty 강화
    weight_premature_tilt: float = 1.50
    weight_grasp_loss: float = 0.05      # 0.30→0.05: [test4] cost_grasp_loss=0.73/step (전체 cost 73%) → tilt 억제. DexPour는 contact reward로 대체

    weight_arm_joint_vel: float = 0.002   # arm_qd 제곱합 페널티 (작은 값으로 시작)
    weight_arm_joint_acc: float = 0.0005  # arm 가속도 프록시 페널티
    arm_joint_vel_sq_clip: float = 64.0   # (arm_qd L2 norm)^2 클리핑 상한 (8 rad/s L2 기준)
    arm_vel_tilt_gate_only: bool = False   # True→False
    weight_arm_joint_vel_approach: float = 0.0005  # approach 구간 (tilt 구간 0.002의 1/4)
    weight_arm_joint_jerk: float = 0.0002
    ema_action_alpha: float = 0.7   # 새 action 70% / 이전 EMA 30%
    weight_action_rate_palm: float = 0.02    # palm 6D: arm jerk 억제 강화
    weight_action_rate_finger: float = 0.005  # finger 5D: 채터링 적당히 억제

    enable_spill_adr: bool = False  # bead ADR이 spill 스케일을 담당하므로 비활성화
    spill_adr_custom_cfg: dict = {
        "reward": {
            # start small to allow exploration, ramp to 기존 10.0 페널티
            "spill_weight": (0.5, 8.0),  # 초기 허용도 ↑ (pour 시도 장려)
        }
    }
    spill_adr_num_increments: int = 50
    spill_adr_increment_interval: int = 20000
    spill_adr_trigger_threshold: float = 0.05  # 0.3→0.05: 0% 성공에서 ADR 절대 미진행 문제 해소

    # ADR: success 기준 커리큘럼 (fill_ratio: 낮은 기준→높은 기준)
    # [v4 수정] fill_ratio 시작값 0.30→0.80: 초기 bead_count ADR 진급이 fill_ratio=0.30으로 너무 쉬워
    #          bead 1개 stage에서도 1/1=100%≥0.80 이어야 성공 → 진급 속도 정상화
    # bead 10개 기준: 0.80=8개, 0.95=19개
    enable_success_adr: bool = True
    success_adr_custom_cfg: dict = {
        "success": {
            "fill_ratio": (0.80, 0.99),  # [v4 수정] 0.30→0.80 시작: 초반 과빠른 ADR 진급 방지
        }
    }
    success_adr_num_increments: int = 8
    success_adr_increment_interval: int = 20000
    success_adr_trigger_threshold: float = 0.30  # [v4 수정] 0.15→0.30: 30% 성공률 달성 시 상향 (기준 강화)
    reward_grasp_slip_sharpness: float = 3.0   # grasp_maintain 감쇠율 [5→3: tilt 중 slip 허용]
    contact_maintain_min_others: int = 2       # contact_maintain: others 최소 접촉 수
    force_balance_sharpness: float = 2.0       # force_balance exp 감쇠율 (v8=2.0)
    reward_approach_xy_scale: float = 6.0
    stage_approach_xy_threshold: float = 0.14
    stage_pour_xy_threshold: float = 0.15
    transport_dist_exp_scale: float = 8.0
    transport_tilt_penalty_weight: float = 2.0
    pour_tilt_target_deg: float = 100.0  # 135→100: 물리적으로 달성 가능한 각도 (비드 쏟기 충분)
    pour_tilt_sharpness: float = 2.0    # 6→2: gradient 범위 확대 (45°부터 학습 신호 확보)
    lift_height_cap: float = 0.05 
    pour_binary_mouth_xy_thresh: float = 0.08  # 0.03→0.08: policy plateau(11cm)에서 gate 도달 불가 해소
    pour_binary_mouth_z_min: float = -0.01
    pour_binary_mouth_z_max: float = 0.03
    pour_binary_tilt_thresh: float = 0.17  # [v4] 0.0→0.17: >80° 기울기에서 pour gate 개방 (90° 내리꽂기 전략 억제)

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
            gpu_max_rigid_patch_count=2**23,
            gpu_max_rigid_contact_count=2**23,
            gpu_collision_stack_size=2**26,  # 32MB→64MB: 비드×컵 contact pair overflow 방지
            gpu_max_num_partitions=32,
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
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
