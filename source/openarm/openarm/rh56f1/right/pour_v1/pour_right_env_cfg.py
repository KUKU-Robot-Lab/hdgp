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

"""환경 설정: inspire_r_pour_v1 (RH56F1 6-DOF 손, pour_v6 구조 이식)

Tesollo pour_v6(20-DOF 손, 27D robot)의 학습 구조를 RH56F1 6 actuated DOF(13D robot)로 이식.
- Action: 12D (6D palm pose + 1D nullspace α + 5D per-finger lerp)
- Observation: actor 51D / critic 112D (asymmetric, left_arm=7D)
- Episode: Fabrics arm policy + frozen grasp hand (pour_v6 구조 동일)
- Contact: fingertip FT sensor (actor, real-compatible). RH56F1 distal/middle 센서 없음
  → critic distal = tip 재사용, middle = zeros.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from .pour_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .pour_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    HAND_APPROACH_POSE,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_HAND_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_HAND_REST_JOINT_POS,
    LEFT_TARGET_CUP_POS_ENV_LOCAL,
    LEFT_TARGET_CUP_QUAT_WXYZ,
    RIGHT_ACTUATED_JOINT_NAMES,
    SOURCE_CUP_POUR_AXIS_B,
    SOURCE_CUP_POUR_POINT_POS_B,
    SOURCE_CUP_UP_AXIS_B,
    TARGET_CUP_OPENING_POS_B,
    TARGET_CUP_UP_AXIS_B,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_DEFAULT_BEAD_COUNT = 20
_DEFAULT_DEMO_POSE_DATASET_DIR = _os.path.normpath(_os.path.join(_HDGP_ROOT, "..", "datasets"))


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(0.5, 0.5, 0.5),
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),  # 1g 구슬 (5g→1g: tesollo pour_v5 검증값. deep tilt 쏠림토크↓→grasp 슬립 완화. rh56f1 약한그립엔 5g(150g)이 과부하라 비드소환 즉시 drop)
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=8,   # 16→8: GPU contact stage 연산 부하 감소
                solver_velocity_iteration_count=2,   # 4→2: 동일 이유
                linear_damping=0.0,                  # 0.1→0.0: 인위 공기저항 제거 (컵 벽 자연 흐름)
                angular_damping=0.0,                 # 0.1→0.0: 구름 방해 제거
                max_depenetration_velocity=1.0,      # 5.0→1.0: 침투 보정 폭발 방지 (PhysX crash 주원인)
                max_linear_velocity=10.0,            # 5.0→10.0: 속도 제한 완화 (깊은 tilt 시 비드 흐름)
                max_angular_velocity=100.0,          # 10.0→100.0: 회전 제한 완화 (자연 굴림)
            ),
        )
        # 이 IsaacLab 버전의 UsdFileCfg는 physics_material 생성자 인자를 직접 받지 않는다.
        # spawn_from_usd()는 cfg.physics_material 속성이 있으면 바인딩하므로 생성 후 후첨가한다.
        # 기본 material 마찰(0.5/0.5)보다 낮춰 컵 내부에서 구슬이 더 쉽게 굴러가게 한다.
        bead_spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.1,
            dynamic_friction=0.08,
            restitution=0.3,                         # 0.1→0.3: 반발력 증가 (표면 접착 완화)
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
    """inspire_r_pour_v1 환경 설정 (RH56F1 6-DOF 손, pour_v6 구조 이식).

    Action (12D): [0:6] 6D palm pose → Fabrics IK, [6] 1D nullspace α, [7:12] 5D finger lerp
    Actor obs (51D): arm_pos/vel(7+7) + finger_progress(5) + left_arm_pos/vel(7+7)
                     + pour_point_to_opening(3) + source_pour_axis(3) + source_up_axis(3)
                     + target_up_axis(3) + last_palm_actions(6)
    Critic (112D): base(77) + extra(35) — left_arm=7D (왼손 rest 고정)
    """

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 20.0
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 51 (actor)
    action_space:      int = NUM_ACTIONS               # 12
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 112 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0
    # cspace attractor mass: nullspace 어트랙터 무게. 너무 크면 palm-pose 추종 침범 주의.
    cspace_attractor_mass:      float = 3.0
    # B-full explicit nullspace: 주둥이 위치 고정(J_spout·Δq=0)하며 arm을 demo deep-tilt로 구동.
    palm_position_only: bool = False
    pour_bfull_nullspace: bool = True
    bfull_step:   float = 0.04   # arm→demo 향한 per-step 관절증분 상한 [rad]
    bfull_lambda: float = 0.05   # DLS pseudo-inverse 댐핑(특이점 방지)
    # approach 제어 방식: "rim"(action xy=주둥이 직접) | "palm"(action xy=palm 직접).
    pour_approach_pivot: str = "palm"
    # aim 정밀화: 주둥이를 target 입구 중심으로 당기는 smooth 보상 weight.
    weight_aim_precision: float = 18.0

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 200
    episode_hold_steps:    int   = 120
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
    # actor obs에만 적용; critic obs는 privileged clean state 유지
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01
    obs_noise_joint_vel: float = 0.05
    obs_noise_body_pos:  float = 0.005
    obs_noise_cup_pos:   float = 0.015

    enable_visual_markers: bool = False

    # ADR: noise 스케줄
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

    # bead / cup geometry (.usd 기준: bottom=-0.077m, rim=+0.100m, inner_r=0.041m)
    target_inner_radius:  float = 0.041
    target_inside_z_min:  float = -0.070
    target_inside_z_max:  float = 0.100
    target_mouth_z:       float = 0.100
    source_inner_radius:  float = 0.041
    source_outer_radius:  float = 0.045
    source_inside_z_min:  float = -0.070
    source_inside_z_max:  float = 0.100
    # pour_point xy 방향: tilt_amount < dyn_lo → 정적(target 고정), > dyn_hi → 동적(gravity_perp).
    pour_point_dyn_lo:    float = 0.15    # ≈45°: 이하 정적(이송, wobble 회피)
    pour_point_dyn_hi:    float = 0.30    # ≈67°: 이상 동적(deep tilt 정밀 배출구). smoothstep blend.
    bead_count: int = _DEFAULT_BEAD_COUNT
    success_bead_cross_count: int = 1
    success_target_fill_ratio: float = 0.50
    success_spill_max: float = 0.40

    # -----------------------------------------------------------------------
    # Policy action / pouring target
    # -----------------------------------------------------------------------
    palm_delta_xyz: float = 0.03  # per-step palm target 오프셋 상한 [m] (속도 cap)
    warmstart_collect_palm_delta_xyz: float = 0.15
    warmstart_collect_palm_delta_rot_deg: float = 20.0
    palm_delta_rot_deg: float = 15.0  # incremental palm orientation target [deg/step]
    tilt_action_gate_xy_near: float = 0.06
    tilt_action_gate_xy_far: float = 0.25

    # -----------------------------------------------------------------------
    # Warmstart quality / success
    # -----------------------------------------------------------------------
    warmstart_palm_z_boost: float = 0.0  # 07.02: 0.12는 palm 목표를 hdf5 자세보다 12cm 위로 설정→fabric이 시작 즉시 팔을 들어올려 성공자세 이탈+palm 디커플링. 0으로 정합(hdf5 자세 그대로 시작).
    lift_success_height: float = 0.03
    success_mouth_xy_threshold: float = 0.030
    success_z_clearance_min: float = 0.015
    success_z_clearance_max: float = 0.050
    success_hold_steps: int = 10
    drop_force_hold_steps: int = 10
    # 파지 붕괴 종료: cup_rel_drift 과대 상태 지속 시 terminated.
    grasp_break_drift_deg: float = 45.0   # 정상 deep tilt drift(~30°) 위 마진
    grasp_break_hold_steps: int = 15      # 연속 지속 시 종료 (transient spike 무시)
    source_empty_hold_steps: int = 60

    # =====================================================================
    # Reward weights — [pour_v6 구조 이식] 2-Stage 가산
    #   total = r_hold + r_approach + r_introt + r_tilt
    #           + release_context·aim_gate·r_source_release
    #           + r_target_capture + w_success·r_success
    #           − ready_context·w_spill·sqrt(spill)
    # =====================================================================

    # Stage A — Grasp maintain
    weight_grasp_maintain: float = 0.50
    weight_contact_maintain: float = 0.50
    # per-finger grasp 보상 (DexPour r_contact+r_grasp 통합)
    weight_grasp: float = 3.0
    grasp_full_count: int = 4          # 완전파지 판정 손가락 수 (5중 4)
    grasp_full_bonus: float = 0.5
    weight_force_balance: float = 0.30
    weight_finger_curl: float = 0.50

    # Stage A — Approach: blended rim_center→pour_point xyz corridor
    weight_dist_to_target: float = 8.0
    weight_corridor_escape_after_ready: float = 0.0
    approach_anti_floor: float = 0.4
    dist_to_target_exp_scale: float = 5.0
    cup_transport_saturate_xy: float = 0.17  # 레거시, 미사용
    rim_approach_scale: float = 5.0
    rim_approach_saturate: float = 0.03  # mouth_xy 이 이하: 거리항 max

    # nullspace 잉여 1-DOF action(α) 스케일
    nullspace_action_scale: float = 1.0
    # α offset 축 모드: "true_nullspace"=palm 보존 elbow-swivel, "demo_minus_start"=tilt 슬라이더
    nullspace_offset_mode: str = "true_nullspace"

    # B-trajectory action 모드: "b_trajectory"=β(pour progress)→R(β) 전신협응, "legacy"=3D tilt
    pour_action_mode: str = "b_trajectory"
    beta_action_index: int = 4   # action[4] = β 채널
    # β=1 목표 tilt_amount. (1-cos135°)/2 = 0.854 (135° dump)
    beta_target_tilt_amount: float = 0.854
    beta_tilt_kp:           float = 3.0
    beta_tilt_max_step:     float = 0.06   # tilt_toward 회전 증분 상한 [rad/step]

    # orientation 풀기: ready 단계에서 palm 방향 명령 제거 → cspace가 j5 deep tilt 전담
    pour_orient_release: bool = True
    # 주둥이 z를 target 입구 위 margin으로 구조적 강제
    pour_spout_z_lock: bool = True
    pour_z_margin:     float = 0.03

    # phase별 관절 범위 클램프 (현재 비활성)
    pour_phase_clamp_enable: bool = False
    pour_phase_arm_lo: tuple = (-9.9, -9.9, -9.9, -9.9, -1.571, -0.30, -9.9)
    pour_phase_arm_hi: tuple = ( 9.9,  9.9,  9.9,  9.9,  0.0,    0.35,  9.9)

    # nullspace baseline(α=0 지점): "robot_start"=순수DRL, "demo"=hard prior
    nullspace_baseline: str = "demo"

    # Stage A→B 공간 게이트 (target 입구 corridor + ready latch)
    g_ready_center: float = 0.05
    g_ready_width: float = 0.04
    pour_corridor_xy_margin: float = 0.015
    pour_corridor_z_min: float = -0.02
    pour_corridor_z_max: float = 0.12
    pour_corridor_scale: float = 20.0
    ready_latch_threshold: float = 0.60
    ready_latch_floor: float = 0.50
    release_gate_floor_after_ready: float = 0.40

    # tilt: 0→135° 단일 연속 ramp, always-on
    tilt_pre_amount: float = 0.456   # 로깅 전용(85° 돌파 추적). 보상 미사용
    weight_tilt_pre: float = 8.0     # 미사용. 구 기록 참조용 유지
    weight_tilt: float = 20.0        # tilt 직접 유도
    weight_tilt_delta: float = 100.0  # tilt 증분(delta) 보상 (더 기울이는 순간만)
    tilt_aim_floor: float = 0.35     # r_tilt pre-ready bootstrap floor
    # 연속 근접 게이트: prox_gate = clamp((far - approach_xy_dist)/(far-near), 0, 1)
    tilt_prox_gate_far:  float = 0.25
    tilt_prox_gate_near: float = 0.06

    # Pour 정밀 조정
    weight_pour:        float = 50.0
    weight_transport:   float = 30.0
    weight_pour_bead:   float = 50.0   # r_pour = w·corridor_score·bead_cross_fraction
    capture_delta_weight: float = 30.0  # target_capture_delta 가중
    pour_z_target:      float = 0.03   # 주둥이를 target 입구 위 3cm로 유도
    pour_z_scale:       float = 20.0
    pour_aim_scale:     float = 10.0
    pour_aim_z_max:     float = 0.05

    # 상시 내회전 유도
    weight_introt: float = 5.0
    pour_tilt_target_deg: float = 135.0   # 수평(90°) 너머 dump까지

    # Stage B — pour-point 정렬 (보조)
    weight_align: float = 5.0
    pour_align_scale: float = 15.0
    pour_align_z_margin: float = 0.10

    # Stage B — bead
    weight_bead_in: float = 0.0
    weight_source_release: float = 100.0  # 소스 잔량 감소분만 transient 보상
    weight_target_capture_delta: float = 200.0
    weight_bead_cross: float = 150.0
    weight_source_drain: float = 0.0
    drain_tilt_min: float = 0.05   # aim_gate tilt 임계
    align_gate_scale: float = 15.0
    bead_near_scale: float = 12.0

    # 내회전 게이트
    internal_rot_thresh: float = 0.0
    internal_rot_temp: float = 0.4
    rot_tilt_floor: float = 0.0

    # Outcome
    weight_success: float = 50.0
    weight_spill: float = 0.0

    # EMA palm action smoothing
    ema_action_alpha: float = 0.7

    # -----------------------------------------------------------------------
    # Demo (critic privileged obs 전용 — 정책 reward에 사용하지 않음)
    # -----------------------------------------------------------------------
    enable_demo_critic_obs: bool = True
    demo_pose_dataset_dir: str = _DEFAULT_DEMO_POSE_DATASET_DIR
    demo_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )
    demo_pose_phase: str = "pour"
    demo_nn_lookahead_frames: int = 10
    enable_demo_pose_reward: bool = False  # critic 전용. actor reward 비활성
    weight_demo_arm_pose: float = 20.0
    weight_demo_arm_pose_floor: float = 5.0
    weight_demo_j5: float = 15.0          # j5(틸트 주역) 앵커 시작값, ready 이후만
    weight_demo_j5_floor: float = 3.0
    demo_j5_sharpness: float = 2.0
    demo_pose_near_gate_xy: float = 9999.0
    demo_pose_warmup_steps: int = 1
    demo_graduate_flow_target: float = 0.05   # flow EMA 도달 시 weight→floor
    demo_graduate_ema_alpha: float = 0.001

    # ADR: spill penalty (pour_v6 기준: OFF)
    enable_spill_adr: bool = False
    spill_adr_custom_cfg: dict = {
        "reward": {
            "spill_weight": (1.0, 15.0),
        }
    }
    spill_adr_num_increments: int = 50
    spill_adr_increment_interval: int = 20000
    spill_adr_trigger_threshold: float = 0.10

    # ADR: success 기준 커리큘럼
    enable_success_adr: bool = True
    success_adr_custom_cfg: dict = {
        "success": {
            "fill_ratio": (0.20, 0.50),
        }
    }
    success_adr_num_increments: int = 8
    success_adr_increment_interval: int = 20000
    success_adr_trigger_threshold: float = 0.15

    # ADR: outcome (자세 성공률 80%+ 시 bead 보상 활성)
    enable_outcome_adr: bool = True
    outcome_adr_custom_cfg: dict = {
        "outcome": {
            "weight_pour_bead": (0.0, 50.0),
        }
    }
    outcome_adr_num_increments: int = 8
    outcome_adr_increment_interval: int = 20000
    outcome_adr_trigger_threshold: float = 0.80
    pose_ready_thresh: float = 0.60   # 자세 성공 게이트: corridor_score ≥
    pose_tilt_thresh: float = 0.587   # 자세 성공 게이트: tilt_amount ≥ (100°+)

    reward_grasp_slip_sharpness: float = 3.0
    contact_maintain_min_others: int = 2
    force_balance_sharpness: float = 2.0

    # ρ binary pour gate
    pour_binary_xy_thresh: float = 0.20

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
    object_spawn_x_center: float = 0.27   # demo 데이터와 일치 (0.40→0.27)
    object_spawn_y_center: float = -0.10  # demo 데이터와 일치 (-0.15→-0.10)
    object_spawn_z:        float = 0.2773  # 07.02: grasp_v1(0.2773)·warm-state와 일치 (구 0.297은 grasp와 2cm 불일치로 warmstart 거부)
    object_spawn_xy_range: float = 0.06   # ±6cm 랜덤화 (Fabrics arm 학습으로 보정 가능)

    # -----------------------------------------------------------------------
    # Warmstart reset cache
    # -----------------------------------------------------------------------
    # RH56F1 포팅: 과거 warmstart 캐시(5g_grasp_right_v7_2, grasp_warm_v7_2.hdf5)는
    # Tesollo 20-DOF 산출물 → RH56F1(6-DOF)엔 무효였다. 이제 RH56F1 전용 grasp_v1
    # warm state(data/grasp_warm_rh56f1.hdf5, collect_grasp_v1_warm_states.py --robot rh56f1)를
    # 생성해 warm_state_paths 로 연결한다. 활성화하려면 아래를 True 로 바꾼다
    # (실제 hdf5 생성·검증 후 학습 시점 결정). 파일 없으면 자동 rollout fallback.
    # 07.01: grasp_v1 2048 ep3000 best 체크포인트로 grasp_warm_rh56f1.hdf5(2048개, num_contacts 3.92) 생성·검증 완료 → 활성화.
    enable_warmstart_reset: bool = True
    warmstart_checkpoint_path: str = (
        _os.path.join(_HDGP_ROOT, "log/rl_games/pipeline/right/5g_grasp_right_v7_2/test3/nn/5g_grasp_right-v7-2.pth")
    )
    warmstart_cache_size: int = 256
    warmstart_max_rollout_steps: int = 6000
    # warmstart 초기 상태 소스:
    #   "disk"   : grasp 가 디스크에 저장한 캐시(grasp_warm_rh56f1.hdf5) 로드 (권장).
    #              startup 시 grasp policy rollout 불필요 → 분포/포맷 불일치 제거.
    #   "rollout": (레거시 fallback) startup 에서 grasp 체크포인트를
    #              pour env 안에서 rollout 해 캐시 수집.
    #   "preset" : 캐시 없이 preset/pregrasp 합성 시작 (디버그용).
    # disk 로드 실패(파일 없음/검증 실패) 시 rollout 으로 안전하게 degrade한다.
    # RH56F1 전용 산출물: data/grasp_warm_rh56f1.hdf5
    #   (collect_grasp_v1_warm_states.py --robot rh56f1).
    # deep-tilt 부트스트랩: sparse chicken-and-egg 해소.
    #   학습 중 정책이 만드는 "deep-tilt + target 위 + 비드 source 보유" 실제 프레임을
    #   full-snapshot으로 캡처했다가 일부 reset을 그 상태에서 시작 → 정책이 마지막 push만
    #   학습해 200 캡처 보상을 경험. f_boot anneal로 직립 시작에 전이.
    enable_deep_tilt_boot: bool = True
    deep_tilt_boot_capacity: int = 4096
    deep_tilt_capture_tilt_min: float = 0.40       # tilt_amount=(1-cosθ)/2; 0.40≈78°
    deep_tilt_capture_src_min: float = 0.80        # 비드 source 보유율 하한
    deep_tilt_capture_mouth_max: float = 0.08      # pour-point xy 거리 상한
    deep_tilt_capture_drift_max: float = 12.0      # cup_rel_drift 상한 [deg]
    deep_tilt_capture_contacts_min: float = 3.0    # 최소 접점 수
    deep_tilt_capture_prob: float = 0.05           # qualifying env를 step당 저장할 확률
    deep_tilt_boot_min_count: int = 64             # 이 수 이상 쌓여야 부트스트랩 시작
    deep_tilt_f_boot_start: float = 0.5
    deep_tilt_f_boot_end: float = 0.0
    deep_tilt_anneal_steps: int = 300_000

    warm_state_source: str = "disk"
    warm_state_paths: tuple[str, ...] = (
        _os.path.normpath(_os.path.join(_HDGP_ROOT, "data", "grasp_warm_rh56f1.hdf5")),
    )
    freeze_grasp_hand_during_episode: bool = True
    # 최상위 비드 z=0.063m (림 0.100에서 3.7cm 아래, 리셋 시 기울어진 컵에서 탈출 방지)
    bead_spawn_pos_source_cup_b: tuple[float, float, float] = (0.0, 0.0, 0.015)
    bead_spawn_quat_source_cup_wxyz: tuple[float, float, float, float] = tuple(
        BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ
    )
    # FK 기반 고정 배치 (LEFT_ARM_REST_JOINT_POS에서 hand local_z=0.05)
    left_target_cup_pos_env_local: tuple[float, float, float] = tuple(LEFT_TARGET_CUP_POS_ENV_LOCAL)
    left_target_cup_quat_wxyz: tuple[float, float, float, float] = tuple(LEFT_TARGET_CUP_QUAT_WXYZ)
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
            bounce_threshold_velocity=0.05,  # 0.2→0.05: 낮은 속도 충돌도 바운스 허용 (비드 자연 반발)
            gpu_found_lost_pairs_capacity=4 * 1024 * 1024,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**24,
            gpu_max_rigid_contact_count=2**24,
            gpu_collision_stack_size=2**30,
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
    # 로봇 설정 (openarm_bi_rh56f1.usd: RH56F1 6-DOF underactuated 손)
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
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=2000.0,   # 400→2000: 오른팔 충돌 저항 강화
                damping=200.0,
            ),
            # RH56F1 우측 손 drive 6 (RL 위치제어 — pour 중 grasp pose freeze)
            # 07.02: 굴곡/원위 강성 30→400 (tesollo pour curl/pip/dip=400 참조). 30은 회전 관성토크에
            # 손가락이 밀려 그립 풀림. 굴곡(thumb_2 + 4손가락_1) 400.
            "rh56f1_right_flexion": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"
                ],
                stiffness=400.0,
                damping=60.0,
            ),
            # abduction(thumb_1): tesollo abduction=200 참조. 반력교란 위험이라 굴곡보다 낮게.
            "rh56f1_right_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_thumb_1"],
                stiffness=200.0,
                damping=35.0,
            ),
            # 원위(mimic) 6 — tesollo dip=400 참조. _apply_action 의 drive×mult 타겟으로 컵 envelope wrap.
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
                stiffness=30.0,
                damping=5.0,
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
    # Actor: fingertip 5개 개별 센서 (Cup-only, real-compatible FT sensor)
    # RH56F1: 별도 distal/middle phalanx 센서 없음 → critic distal = tip 재사용, middle = zeros.
    # fingertip = 병합된 말단 링크(thumb_4, *_2). 순서: thumb,index,middle,ring,little
    # -----------------------------------------------------------------------
    right_tip_contact_links: tuple = (
        "r_hl_thumb_4",
        "r_hl_index_2",
        "r_hl_middle_2",
        "r_hl_ring_2",
        "r_hl_pinky_2",
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
    left_arm_joint_names: list = LEFT_ARM_AND_HAND_JOINT_NAMES
