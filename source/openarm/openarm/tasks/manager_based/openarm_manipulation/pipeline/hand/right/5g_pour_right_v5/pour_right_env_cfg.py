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

"""환경 설정: 5g_pour_right_v5

v5: Fabrics 팔 학습(6D palm) + per-finger lerp(5D) + sim2real 가능 obs
- Action: 11D (6D palm pose + 5D per-finger lerp)
- Observation: actor 60D / critic 143D (asymmetric)
- Episode: Grasp phase (Fabrics arm + finger 정책) + Lift phase (scripted arm + frozen hand)
- Contact: fingertip/distal/middle sensors are kept in critic full-state, not actor LSTM input
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
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=0.5,   # 0.0→0.1: 구슬이 컵 안에서 너무 활발히 튀는 현상 완화
                angular_damping=0.5,  # 0.0→0.1: 구슬이 컵 안에서 너무 활발히 튀는 현상 완화
                max_depenetration_velocity=5.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        )
        # 이 IsaacLab 버전의 UsdFileCfg는 physics_material 생성자 인자를 직접 받지 않는다.
        # spawn_from_usd()는 cfg.physics_material 속성이 있으면 바인딩하므로 생성 후 후첨가한다.
        # 기본 material 마찰(0.5/0.5)보다 낮춰 컵 내부에서 구슬이 더 쉽게 굴러가게 한다.
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
                pos=[0.42, -0.18, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=bead_spawn_cfg,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class PourRightEnvCfg(DirectRLEnvCfg):
    """5g_pour_right_v5 환경 설정."""

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
    # Demo reset (match 5g_grasp_right_v7_2/test3 warmstart collection)
    # -----------------------------------------------------------------------
    enable_demo_grasp_reset: bool = True
    demo_grasp_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_HDGP_ROOT, "..", "datasets", f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )

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
    source_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    source_inside_z_max:  float = 0.100   # 림 높이
    bead_count: int = _DEFAULT_BEAD_COUNT
    success_bead_cross_count: int = 1
    success_target_fill_ratio: float = 0.30   # [test2] 0.50→0.30: ADR 통과 허들 낮춤
    success_spill_max: float = 0.35   # [test2] 0.20→0.35: spill 30% > 20% 항상 실패 차단 해제

    # -----------------------------------------------------------------------
    # Policy action / pouring target
    # -----------------------------------------------------------------------
    # test2에서 reset 직후 mouth_xy가 0.30~0.36m인데 delta_xyz=0.10m로는 일부 env가
    # 타겟컵 근처까지 도달 불가하다. transport 여유를 키운다.
    #
    # [test1/3 분석] Workspace-Target 거리 불일치:
    #   pregrasp palm y = cup_y_spawn(-0.10) + pregrasp_offset_y(-0.07) = -0.17m
    #   delta=0.3m → max palm y = -0.17 + 0.30 = +0.13m (workspace y_max=0.22 이전에 delta 소진)
    #   타겟 컵 y ≈ LEFT_ARM_REST FK 기준 ≈ +0.27m
    #   → 최소 cup-target XY gap = 0.27 - 0.13 = 0.14m (delta=0.5 시 달성 가능)
    #
    #   수정: delta=0.5m + y_max=0.22m(preset.py 동시 수정)
    #   max palm y = min(-0.22+0.50, 0.22) = 0.22m
    #   → cup-target gap ≈ 0.27 - 0.22 = 0.05m → g_align_xy(scale=5) = exp(-5×0.05) = 0.78
    #   → pre-pour reward 완전 활성화 가능
    palm_delta_xyz: float = 0.5   # 0.3 → 0.5: workspace-target 거리 불일치 해소
    # warmstart cache 수집(체크포인트 rollout) 시 사용할 palm xyz/rot delta.
    # v7-2 학습값(xyz=0.15m, rot=20°)과 일치시켜야 action scale이 맞음.
    warmstart_collect_palm_delta_xyz: float = 0.15  # 0.10→0.15: v7-2 학습값 일치
    warmstart_collect_palm_delta_rot_deg: float = 20.0  # 별도 관리: v7-2=20°, 본 학습=120°
    palm_delta_rot_deg: float = 120.0  # 45→120: cup 135° tilt 도달 가능하도록 확장
    # 회전(action[3:6])은 타겟컵 근처에서만 충분히 허용.
    # mouth_xy >= far 이면 회전 0, <= near 이면 회전 1, 그 사이는 선형 보간.
    # near < far 여야 선형 보간이 성립하므로 작은 값(가까움) → 1, 큰 값(멀어짐) → 0 순서로 둔다.
    #
    # [test1/3 분석] tilt_gate 과도 허용 → 제자리 wrist spin:
    #   기존 far=0.32m: policy 수렴 위치(0.22m)에서 gate=(0.32-0.22)/(0.32-0.06)=0.38 (38% 허용)
    #   → tilt 시도 시 premature_tilt_cost(3.6/step) >> r_prepour(0.99/step) → 실제 tilt 불가
    #   → 대신 cup-local Z축 spin만 발생 (spin은 source_up_dot_world 변화 없어 penalty 없음)
    #   → 정성 관찰 "제자리 회전" 환경의 원인
    #
    #   수정: far=0.20m → 0.22m에서 gate=(0.20-0.22)/(0.20-0.06)=-0.14 → clamp=0
    #   → 0.20m 이내에 도달하기 전에는 tilt action 완전 차단 → 순수 위치 접근만 학습
    tilt_action_gate_xy_near: float = 0.06
    tilt_action_gate_xy_far: float = 0.25  # 0.32→0.20→0.25: equilibrium 0.16m에서 gate 28%→47%

    # stage gate / pre-pour geometry
    # test1에서 mouth_xy≈0.23m일 때 g_align_xy가 1e-3 이하로 죽어 접근 전 stage 신호가 약했음.
    # xy gate/approach를 넓혀 먼 거리에서도 target 방향 gradient를 유지한다.
    #
    # [test1/3 분석] g_ready Gate 너무 엄격 → Stage C/D 학습 완전 차단:
    #   g_align_xy = exp(-scale × cup_center_xy_dist)
    #   scale=12, dist=0.22m → exp(-12×0.22) = exp(-2.64) = 0.071 (7.1%)
    #   g_ready = g_align_xy × g_clear ≈ 0.071 × g_clear → TB 관찰값 g_ready≈0.11과 일치
    #
    #   g_ready=0.5가 되려면: exp(-12×d)=0.5 → d = ln(2)/12 = 0.058m
    #   그런데 workspace 제한으로 cup-target gap 최소 0.19m → 물리적으로 달성 불가
    #
    #   결과: r_prepour = g_ready(0.11) × 9.0 = 0.99/step
    #         premature_tilt_cost = (1-0.11) × 4.0 = 3.56/step
    #         tilt 시 cost > reward → policy가 tilt 학습 자체를 포기
    #
    #   수정: scale=5 → g_align_xy @ 0.22m = exp(-5×0.22) = 0.33 (4.7배 증가)
    #         g_ready=0.5 달성 거리: ln(2)/5 = 0.14m (workspace 확장 후 달성 가능)
    reward_gate_xy_scale: float = 5.0   # 12.0→5.0: g_align_xy @ 0.22m: 0.07→0.33, 50%선 0.06→0.14m
    reward_gate_clear_scale: float = 80.0
    reward_gate_tilt_scale: float = 15.0
    reward_clearance_min: float = 0.015
    reward_tilt_cos_min: float = 0.15

    # -----------------------------------------------------------------------
    # Warmstart quality / success
    # -----------------------------------------------------------------------
    # warmstart는 테이블 위에서 막 잡힌 자세가 아니라, 테이블 기준 약 3cm 든 자세에서 시작한다.
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
    # reward_v5 =
    #   approach_stage + g_ready * pre_pour_stage + g_pour * pour_stage + outcome
    #   + grasp/contact/force/finger 유지 보상
    #   - spill/premature_tilt/grasp_loss/action_rate/wrist_spin 비용
    # -----------------------------------------------------------------------
    # 유지계는 패널티형 reward로 바꿨으므로 과도한 상수항이 되지 않게 낮추고,
    # transport/capture/success 쪽 가중치를 상대적으로 키운다.
    #
    # [test1/3 분석] r_hold Local Optimum → 안 움직임:
    #   r_hold max = grasp_maintain(2.0) + contact_maintain(1.0) + finger_curl(2.0) = 5.0/step
    #   TB 관찰: r_hold≈4.87/step (포화), cup_center_xy_dist=0.22m 에서 8756 epoch 내내 plateau
    #   원인: 안 움직이면 step당 5.0을 안정적으로 받음. 이동 시 grasp slip → grasp_maintain 감소.
    #         이동으로 얻는 r_approach 증분보다 잃는 r_hold가 크면 이동하지 않는 게 유리.
    #   수정: r_hold max → 0.5+0.3+0.5 = 1.3/step (75% 감소) → 이동 유인 상대적 강화
    weight_grasp_maintain: float = 0.5    # [test2] 0.0→0.5: 엄지 붕괴 방지 grasp slip 억제
    weight_contact_maintain: float = 0.0  # [test4] 0.30→0.0
    weight_force_balance: float = 0.0     # [test4] 0.30→0.0
    weight_finger_curl: float = 0.3       # [test2] 0.0→0.3: 손가락 닫힘 유지 (curl 자세 유지)
    # Pour 보상 (v3 스타일, pour_reward_start_step 이후 warmup으로 점진 활성)
    weight_dist_to_target: float = 5.0      # transport: source→target XY exp reward
    dist_to_target_exp_scale: float = 5.0
    weight_tilt: float = 3.0               # pour stage: tilt 각도 정밀도
    weight_align: float = 3.0              # pour stage: 방향 정렬
    weight_bead_progressive: float = 200.0  # pour stage: bead fraction^2
    weight_bead_entry_delta: float = 50.0   # pour stage: 매 step bead 유입 즉각 피드백
    weight_source_drain: float = 20.0       # pour stage: source cup 비우기
    # Pour 보상 step-based curriculum
    # [0, start)          → demo만 활성 (approach/grasp 학습)
    # [start, start+ramp) → pour 0→1 선형 증가
    # [start+ramp, ∞)     → demo + pour 풀 활성
    pour_reward_start_step: int = 50_000
    pour_reward_warmup_steps: int = 50_000
    # gamma=0.998, ep~500 step → terminal discount ≈ 0.37 → success 현재가치 충분히 크려면 500+ 필요
    # dense r_pour 에피소드 누적 수백 대비 success 30은 noise 수준 → 100으로 강화 (300은 과도했음)
    weight_success: float = 100.00  # 30→300→100
    # 성공 기준을 넘은 뒤 추가로 더 많이 채우면 보너스를 주어 과도기 구간의 탐색을 돕는다.
    # 0이면 비활성.
    weight_success_overfill: float = 0.0
    weight_spill: float = 5.00
    weight_premature_tilt: float = 1.5    # (1 - ρ) × tilt_amount: 원거리 tilt 페널티
    weight_grasp_loss: float = 0.20      # [test4] 0.05→0.20: 유일한 grasp safety net 강화
    # [test2] cup-cup 외경 충돌 방지: 두 컵 중심 거리가 margin 미만이면 페널티
    # cup_external_radius_sum ≈ 0.09m, margin=0.12m → 3cm 안전 여유
    weight_cup_collision: float = 5.0     # cup-cup collision penalty weight
    cup_collision_margin: float = 0.12    # cup-cup XY dist threshold (m)
    # [Phase-1 Step 4] arm joint velocity / acceleration penalty (grasp v9 미존재, pour 신규 추가)
    # arm_qd^2 sum의 clamp 후 패널티 → pouring 직전 arm 흔들림 직접 억제
    weight_arm_joint_vel: float = 0.002   # arm_qd 제곱합 페널티 (작은 값으로 시작)
    weight_arm_joint_acc: float = 0.0005  # arm 가속도 프록시 페널티
    arm_joint_vel_sq_clip: float = 64.0   # (arm_qd L2 norm)^2 클리핑 상한 (8 rad/s L2 기준)
    # [Phase-1 Step 5] arm vel penalty gate
    # [test4] cost_arm_vel=0.001/step (사실상 0) → approach 구간 빠른 이동 억제 없음
    # → arm_vel_tilt_gate_only=False: approach 구간에도 낮은 가중치로 적용
    arm_vel_tilt_gate_only: bool = False   # True→False
    weight_arm_joint_vel_approach: float = 0.0005  # approach 구간 (tilt 구간 0.002의 1/4)
    # [Phase-1 Step 6] arm joint jerk penalty (acc 변화율, 흔들림 급변 억제)
    weight_arm_joint_jerk: float = 0.0002
    # [Phase-1 Step 7] EMA palm action smoothing: Fabrics IK에 smooth 궤적 전달
    # action_rate_penalty는 raw action 기반 유지 (training gradient 보존)
    ema_action_alpha: float = 0.7   # 새 action 70% / 이전 EMA 30%
    # [Phase-2 Step 9] action_rate를 palm(6D) / finger(5D) 분리
    # grasp v9 패턴과 동일 (action_smoothness_palm/finger_weight)
    # 기존 단일 weight_action_rate=0.01 → palm 강화, finger 완화
    weight_action_rate_palm: float = 0.02    # palm 6D: arm jerk 억제 강화
    weight_action_rate_finger: float = 0.005  # finger 5D: 채터링 적당히 억제

    # -----------------------------------------------------------------------
    # Demo-guided pose shaping (pure DRL: no BC loss / no action supervision)
    # -----------------------------------------------------------------------
    enable_demo_pose_reward: bool = True
    demo_pose_dataset_dir: str = _DEFAULT_DEMO_POSE_DATASET_DIR
    demo_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )
    demo_pose_phase: str = "all"
    weight_demo_arm_pose: float = 9.00   # [test4] 6.00→9.00: 공간 접근 gradient 보강
    weight_demo_palm_pose: float = 0.50  # [test2] 3.0→0.5: palm sim2real gap 최소화, arm 집중
    weight_demo_smooth: float = 0.20
    weight_thumb_grip_pose: float = 1.00  # [test2] 0.0→1.0: 엄지 demo 평균 자세 복원 (엄지 붕괴 방지)
    demo_pose_warmup_steps: int = 20000
    demo_pose_near_gate_xy: float = 0.20  # unused: demo_pose_phase="all"은 거리 gate 없이 항상 참조
    demo_nn_lookahead_frames: int = 15    # [test3] 30→15: K=30이 demo 추종 방해 (0.5s→0.25s)

    # ADR: spill penalty 스케줄 (low→high)
    enable_spill_adr: bool = True   # [test3] False→True: spill 점진적 억제 (5.0→8.0 ADR)
    spill_adr_custom_cfg: dict = {
        "reward": {
            # start small to allow exploration, ramp to 기존 10.0 페널티
            "spill_weight": (1.0, 8.0),  # [test3] 0.5→1.0: 초기 spill 기준 상향
        }
    }
    spill_adr_num_increments: int = 50
    spill_adr_increment_interval: int = 20000
    spill_adr_trigger_threshold: float = 0.3

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

    reward_grasp_slip_sharpness: float = 3.0   # grasp_maintain 감쇠율 [5→3: tilt 중 slip 허용]
    contact_maintain_min_others: int = 2       # contact_maintain: others 최소 접촉 수
    force_balance_sharpness: float = 2.0       # force_balance exp 감쇠율 (v8=2.0)
    pour_tilt_target_deg: float = 100.0  # 135→100: 물리적으로 달성 가능한 각도 (비드 쏟기 충분)
    pour_tilt_sharpness: float = 2.0    # 6→2: gradient 범위 확대 (45°부터 학습 신호 확보)
    pour_binary_xy_thresh: float = 0.18   # ρ gate: cup_center_xy_dist < thresh 시 pour 활성

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
    # 궤적 캡처 + 성공 궤적 버퍼 (LSTM BC auxiliary loss 학습용)
    # -----------------------------------------------------------------------
    enable_trajectory_capture: bool = True
    trajectory_capture_window: int = 200       # 에피소드 마지막 N 스텝 캡처
    trajectory_buffer_capacity: int = 256      # 성공 궤적 최대 저장 개수
    trajectory_min_steps: int = 20             # 이 미만이면 저장 skip
    trajectory_success_bead_threshold: float = 0.5   # bead_in_target_fraction 하한
    trajectory_success_spill_max: float = 0.2        # spill_ratio 상한

    # -----------------------------------------------------------------------
    # Warmstart reset cache
    # -----------------------------------------------------------------------
    enable_warmstart_reset: bool = True
    warmstart_checkpoint_path: str = (
        _os.path.join(_HDGP_ROOT, "log/rl_games/pipeline/right/5g_grasp_right_v7_2/test3/nn/5g_grasp_right-v7-2.pth")
    )
    warmstart_cache_size: int = 256
    warmstart_max_rollout_steps: int = 6000
    # warmstart 초기 상태 소스 (5g_pour_right_v3 와 동일):
    #   "disk"   : grasp 가 저장한 grasp_warm_v7_2.hdf5 로드 (기본, 권장).
    #              startup 시 grasp policy rollout 불필요 → 분포/포맷 불일치 제거.
    #   "rollout": (레거시 fallback) startup 에서 v7-2 체크포인트를 pour env
    #              안에서 rollout 해 캐시 수집.
    #   "preset" : 캐시 없이 preset/pregrasp 합성 시작 (디버그용).
    # disk 로드 실패(파일 없음/검증 실패) 시 rollout 으로 안전 degrade.
    warm_state_source: str = "disk"
    warm_state_paths: tuple[str, ...] = (
        _os.path.normpath(_os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, "grasp_warm_v7_2.hdf5")),
    )
    freeze_grasp_hand_during_episode: bool = False
    bead_spawn_pos_source_cup_b: tuple[float, float, float] = tuple(BEAD_SPAWN_POS_SOURCE_CUP_B)
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
            bounce_threshold_velocity=0.01,
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
