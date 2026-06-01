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
- Observation: actor 60D / critic 140D (asymmetric)
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
    LEFT_TARGET_CUP_POS_ENV_LOCAL,
    LEFT_TARGET_CUP_QUAT_WXYZ,
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
_DEFAULT_DEMO_POSE_DATASET_DIR = _os.path.normpath(_os.path.join(_HDGP_ROOT, "..", "datasets"))


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(0.5, 0.5, 0.5),
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),  # 5g 구슬 (5g→10g: 관성 향상, 진동 날림 방지)
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
    observation_space: int = NUM_OBSERVATIONS          # 60 (actor)
    action_space:      int = NUM_ACTIONS               # 11
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 140 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0  # 180.0 -> 45.0: 접근/이동 중 기괴한 손목 회전 억제
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 200
    episode_hold_steps:    int   = 120  # warmstart prelift 2s 확보 (컵 높이 0.12m 리프트)
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


    # bead / cup geometry (.usd 기준: bottom=-0.077m, rim=+0.100m, inner_r=0.041m)
    target_inner_radius:  float = 0.041   # 컵 내부 반경
    target_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    target_inside_z_max:  float = 0.100   # 림 높이
    target_mouth_z:       float = 0.100   # 림 높이 (bead crossing 기준)
    source_inner_radius:  float = 0.041   # 컵 내부 반경
    source_outer_radius:  float = 0.045   # 컵 외부 반경 (최하단 림 점 계산용)
    source_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    source_inside_z_max:  float = 0.100   # 림 높이
    bead_count: int = _DEFAULT_BEAD_COUNT
    success_bead_cross_count: int = 1
    success_target_fill_ratio: float = 0.50
    success_spill_max: float = 0.40   # tilt 탐색 중 spill 허용 (ADR로 점진 강화)

    # -----------------------------------------------------------------------
    # Policy action / pouring target
    # -----------------------------------------------------------------------
    # pregrasp palm y = cup_y_spawn(-0.10) + pregrasp_offset_y(-0.12) = -0.22m
    #   delta=0.3m → max palm y = -0.22 + 0.30 = +0.08m (workspace y_max=0.22 이전에 delta 소진)
    #   타겟 컵 y ≈ 0.10m (demo 데이터 기준, LEFT_ARM_REST FK 기준)
    #   → 최소 cup-target XY gap = 0.10 - 0.08 = 0.02m (달성 가능)
    #
    #   delta=0.5m + y_max=0.22m
    #   max palm y = min(-0.22+0.50, 0.22) = 0.22m
    #   → cup-target gap ≈ 0.27 - 0.22 = 0.05m → g_align_xy(scale=5) = exp(-5×0.05) = 0.78
    #   → pre-pour reward 완전 활성화 가능
    palm_delta_xyz: float = 0.5   # 0.3 → 0.5: workspace-target 거리 불일치 해소
    # warmstart cache 수집(체크포인트 rollout) 시 사용할 palm delta.
    # v7-2 grasp checkpoint 학습 조건과 반드시 일치해야 한다:
    #   5g_grasp_right_v7_2: palm_delta_xyz=0.15m, palm_delta_rot_deg=20°
    warmstart_collect_palm_delta_xyz: float = 0.15   # v7-2 학습 값과 일치
    warmstart_collect_palm_delta_rot_deg: float = 20.0  # v7-2 학습 값과 일치 (≠ pour 120°)
    palm_delta_rot_deg: float = 120.0  # 45→120: cup 135° tilt 도달 가능하도록 확장
    # 회전(action[3:6])은 타겟컵 근처에서만 충분히 허용.
    # mouth_xy >= far 이면 회전 0, <= near 이면 회전 1, 그 사이는 선형 보간.
    # near < far 여야 선형 보간이 성립하므로 작은 값(가까움) → 1, 큰 값(멀어짐) → 0 순서로 둔다.
    #
    tilt_action_gate_xy_near: float = 0.06
    tilt_action_gate_xy_far: float = 0.25  # 0.32→0.20→0.25: equilibrium 0.16m에서 gate 28%→47%

    # -----------------------------------------------------------------------
    # Warmstart quality / success
    # -----------------------------------------------------------------------
    # warmstart는 테이블 위에서 막 잡힌 자세가 아니라, 테이블 기준 약 3cm 든 자세에서 시작한다.
    # 리셋 시 palm target z를 0.12m 올림 → hold phase(2s) 동안 Fabrics가 팔을 올리면서 컵도 같이 올라감.
    # 120° tilt 시 pour point가 target rim(0.391m) 위에 위치.
    # 계산: warmstart cup_z ≈ 0.327m → boost → 0.447m → pour_point_z ≈ 0.397m > 0.391m ✓
    warmstart_palm_z_boost: float = 0.12
    lift_success_height: float = 0.03
    success_mouth_xy_threshold: float = 0.030
    success_z_clearance_min: float = 0.015
    success_z_clearance_max: float = 0.050
    success_hold_steps: int = 10
    drop_force_hold_steps: int = 10
    # 소스 컵이 비어있는 상태가 N 스텝 연속 지속되면 에피소드 종료
    # 비드 낙하 + 착지에 ~0.3~0.5초 필요 → 60 steps (1.0s @ 60Hz) 여유
    source_empty_hold_steps: int = 60

    # -----------------------------------------------------------------------
    # Reward weights
    # total = r_hold + r_dist + ρ*(r_tilt+r_align+r_bead+r_drain) + r_success
    #         - p_tilt - p_spill - p_action - demo_costs
    #
    # ρ = (cup_center_xy_dist < pour_binary_xy_thresh).float()
    # r_align = 0.5*(1 + directional_tilt_cos)  ← DexPour eq.2
    #
    # Bead reward 설계 (40% trap 방지):
    #   r_bead_progressive = w * fraction^2  : 40%→0.16w, 80%→0.64w, 100%→w (비선형 가속)
    #   r_bead_delta       = w * delta.clamp(0) : bead 유입 즉각 피드백 (LSTM temporal)
    #   spill_cost         = w * sqrt(spill) : 초기 spill 강하게 패널티
    # -----------------------------------------------------------------------

    # Grasp maintain (r_hold) — tilt-phase aware
    weight_grasp_maintain: float = 0.50
    weight_contact_maintain: float = 0.50   # 0.30→0.50 소폭 강화
    weight_force_balance: float = 0.30
    weight_finger_curl: float = 0.50

    # Transport Stage 1a: Cartesian 근접 (cup_center_xy 기반, 거친 approach gradient)
    weight_dist_to_target: float = 5.0
    dist_to_target_exp_scale: float = 5.0
    cup_transport_saturate_xy: float = 0.17  # 이 이하: transport max (saturate)

    # Transport Stage 1b: "좋은 arm+palm 자세" — demo pour palm pose 추종 (a11~a20 평균)
    # palm pose가 맞으면 Fabrics IK가 j0~4를 demo 자세로 자동 수렴시킴 (redundancy 1DOF).
    transport_palm_pos: tuple[float, ...] = (0.2938, -0.0781, 0.5629)
    transport_palm_quat_xyzw: tuple[float, ...] = (-0.4532, 0.5712, 0.2235, 0.6469)
    weight_palm_pose: float = 10.0
    palm_pose_pos_sharpness: float = 8.0
    palm_pose_rot_sharpness: float = 1.0

    # Pour distance: Stage 2 (ρ gate + pour_warmup)
    # pour point(rim 최하단) → target center XY 거리 기반
    # 가까울수록 bead가 target에 들어갈 확률 높아짐
    weight_pour_dist: float = 12.0
    pour_dist_exp_scale: float = 8.0
    # z_window: pour_point Z soft gate (1~5cm 활성 구역, 정책이 최적 위치 탐색)
    # z_lower_ramp: 0→lower_ramp 구간에서 0→1 상승 (하한)
    # z_upper_end:  이 높이에서 완전히 0 (상한)
    # z_upper_ramp: upper_end 기준으로 이 폭만큼 앞에서 하강 시작
    z_window_lower_ramp: float = 0.01   # 0~1cm: 하한 ramp
    z_window_upper_end:  float = 0.08   # 8cm에서 완전 소멸
    z_window_upper_ramp: float = 0.03   # 5~8cm: 상한 ramp (8-3=5cm부터 하강)

    # Pour: Stage 3 (ρ gate — binary, pour_warmup/bead_warmup 적용)
    weight_tilt: float = 40.0             # 120° 타겟 gradient (test5: 8→40, approach local min 탈출)
    weight_align: float = 6.00            # 방향 신호 강화
    # pour-point pivot gates (test6)
    # initial_tilt_gate: pour_dist는 이 각도 이상 tilt 후 활성 → r_tilt와 충돌 제거
    pour_point_tilt_threshold_deg: float = 15.0
    # pour_aligned_gate: pour_point 정렬도 비례로 r_tilt 증폭 → pour_point pivot 행동 유도
    pour_align_gate_scale: float = 8.0
    weight_bead_progressive: float = 200.0   # quadratic fill: fraction^2 → 40% trap 방지
    weight_bead_entry_delta: float = 300.0   # 비드 유입 즉각 피드백 강화
    weight_source_drain: float = 20.0     # pour gate 중 소스 배출 incentive

    # -----------------------------------------------------------------------
    # Curriculum warmup (step 기반 선형 증가)
    # pour_warmup: pour_dist + tilt + align 을 0→max 로 점진 증가
    # bead_warmup: bead progressive/delta + source_drain 을 0→max 로 점진 증가
    # -----------------------------------------------------------------------
    curriculum_pour_warmup_steps: int = 40000   # 0~40k: pour stage 탐색 유도
    curriculum_bead_warmup_start: int = 6400    # 100 에포크 (100 × horizon 64) 이후 활성
    curriculum_bead_warmup_steps: int = 1       # start 즉시 1.0으로 활성

    # Outcome
    weight_success: float = 100.00
    weight_success_overfill: float = 0.0
    weight_spill: float = 40.0            # 5.0→40.0: spill 강하게 패널티 (40% trap 방지)


    # [Phase-1 Step 7] EMA palm action smoothing: Fabrics IK에 smooth 궤적 전달
    ema_action_alpha: float = 0.7   # 새 action 70% / 이전 EMA 30%

    # -----------------------------------------------------------------------
    # Demo-guided pose shaping (pure DRL: no BC loss / no action supervision)
    # -----------------------------------------------------------------------
    enable_demo_pose_reward: bool = True
    demo_pose_dataset_dir: str = _DEFAULT_DEMO_POSE_DATASET_DIR
    demo_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )
    demo_pose_phase: str = "pour"
    weight_demo_arm_pose: float = 20.0
    demo_pose_warmup_steps: int = 1
    # near_gate = exp(-(dist/9999)^2) ≈ 1.0 (항상 열린 상태)
    demo_pose_near_gate_xy: float = 9999.0
    demo_nn_lookahead_frames: int = 10

    # ADR: spill penalty 스케줄 (low→high)
    enable_spill_adr: bool = True
    spill_adr_custom_cfg: dict = {
        "reward": {
            # 초기 1.0→최대 15.0: 초반 spill 허용 폭 확대 (비드 유입 탐색 촉진)
            "spill_weight": (1.0, 15.0),
        }
    }
    spill_adr_num_increments: int = 50
    spill_adr_increment_interval: int = 20000
    spill_adr_trigger_threshold: float = 0.10

    # ADR: success 기준 커리큘럼 (fill_ratio: 낮은 기준→높은 기준)
    # bead 10개 기준: 0.20=2개, 0.30=3개, 0.40=4개, 0.50=5개
    # 해당 기준에서 success_rate >= 15%이면 한 단계 올림 (8단계 × 0.0375 = 0.30 range)
    enable_success_adr: bool = True
    success_adr_custom_cfg: dict = {
        "success": {
            "fill_ratio": (0.20, 0.50),  # 2개→5개 커리큘럼
        }
    }
    success_adr_num_increments: int = 8
    success_adr_increment_interval: int = 20000
    success_adr_trigger_threshold: float = 0.15  # 현재 기준에서 15% 성공률 달성 시 상향

    reward_grasp_slip_sharpness: float = 3.0
    contact_maintain_min_others: int = 2
    force_balance_sharpness: float = 2.0
    pour_tilt_target_deg: float = 120.0
    pour_tilt_sharpness: float = 4.0   # 120° 목표 집중도 (test8: 2→4, 90° local min 탈출)

    # ρ binary pour gate: cup_center_xy_dist < thresh → pour stage 활성
    pour_binary_xy_thresh: float = 0.20
    pour_binary_tilt_thresh: float = 0.50  # gate_pour_binary 진단용 (ρ에는 미사용)

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
    object_spawn_z:        float = 0.297
    object_spawn_xy_range: float = 0.06   # ±6cm 랜덤화 (Fabrics arm 학습으로 보정 가능)

    # -----------------------------------------------------------------------
    # Warmstart reset cache
    # -----------------------------------------------------------------------
    enable_warmstart_reset: bool = True
    warmstart_checkpoint_path: str = (
        _os.path.join(_HDGP_ROOT, "log/rl_games/pipeline/right/5g_grasp_right_v7_2/test3/nn/5g_grasp_right-v7-2.pth")
    )
    warmstart_cache_size: int = 256
    warmstart_max_rollout_steps: int = 6000
    # warmstart 초기 상태 소스:
    #   "disk"   : grasp 가 디스크에 저장한 캐시(grasp_warm_v7_2.hdf5) 로드 (권장).
    #              startup 시 grasp policy rollout 불필요 → 분포/포맷 불일치 제거.
    #   "rollout": (레거시 fallback) 기존처럼 startup 에서 v7-2 체크포인트를
    #              pour env 안에서 rollout 해 캐시 수집.
    #   "preset" : 캐시 없이 preset/pregrasp 합성 시작 (디버그용).
    # disk 로드 실패(파일 없음/검증 실패) 시 rollout 으로 안전하게 degrade한다.
    # 기본 "disk": train.py 가 override 없이도 grasp_warm_v7_2.hdf5 를 로드.
    # 파일이 없으면 자동으로 rollout 으로 fallback 하므로 안전.
    warm_state_source: str = "disk"
    warm_state_paths: tuple[str, ...] = (
        _os.path.normpath(_os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, "grasp_warm_v7_2.hdf5")),
    )
    freeze_grasp_hand_during_episode: bool = False
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
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
