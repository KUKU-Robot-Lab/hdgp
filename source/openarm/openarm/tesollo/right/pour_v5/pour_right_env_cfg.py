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

"""환경 설정: 5g_pour_right_v4

v7: Fabrics 팔 학습(6D palm) + frozen grasp + sim2real 가능 obs
- Action: 6D (6D palm pose)
- Observation: actor 55D / critic 144D (asymmetric)
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

from openarm import OPENARM_ROOT_DIR
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),  # 1g 구슬 (5g→1g: deep tilt 시 쏠림 토크 감소 → grasp 슬립 완화)
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
    """5g_pour_right_v4 환경 설정."""

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
    observation_space: int = NUM_OBSERVATIONS          # 55 (actor)
    action_space:      int = NUM_ACTIONS               # 6
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 144 (critic, privileged)

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
    # [lstm_test5] cspace_attractor mass(nullspace 어트랙터 무게). YAML 기본 1.0은 약해서 정책 pour
    #   pose가 elbow를 demo(j4=1.87)에서 0.70으로 무너뜨림. ↑ 하면 demo j1-4 nullspace를 강하게 유지.
    #   주의: 너무 크면 palm-pose 추종(corridor) 침범 → Stage-A 저하. 3부터 시작, 결과 보고 조정.
    cspace_attractor_mass:      float = 3.0  # [B-full] 6→3 복원: cspace(soft)는 j5를 못 끔(검증). j5 깊이는 아래 explicit nullspace가 담당.

    # [B-full explicit nullspace] 주둥이 위치를 정확히 고정(J_spout·Δq=0)하며 arm을 demo deep-tilt로 구동.
    #   J_spout = Fabrics palm 7점 위치 Jacobian의 선형결합(spout=palm_link+R·off). cspace(soft)가 못 한
    #   j5 깊이를 nullspace 투영으로 강제 — 주둥이 task와 경쟁 안 함(orthogonal). ready 단계만.
    pour_bfull_nullspace: bool = True
    bfull_step:   float = 0.04   # arm→demo 향한 per-step 관절증분 상한 [rad]
    bfull_lambda: float = 0.05   # DLS pseudo-inverse 댐핑(특이점 방지)

    # [대조군] approach 제어 방식: "rim"(action xy=주둥이 직접) | "palm"(action xy=palm 직접).
    #   나머지(B-full nullspace·z-lock·orientation release·reward) 전부 공통 → 제어방식만 비교.
    #   v5="rim", v6="palm".
    pour_approach_pivot: str = "rim"

    # [aim 정밀화] 주둥이를 target 입구 중심으로 당기는 smooth 보상(aim_score, radius=0 gradient-everywhere).
    #   진단: corridor flat-top(5.6cm)이 8.7cm서 충족돼 주둥이가 target서 8.7cm 벗어나 plateau→spill 0.3.
    #   corridor(ready-latch 필수)는 불변, 별도 정밀-aim gradient만 추가해 주둥이를 입구로 수렴.
    weight_aim_precision: float = 18.0  # [A 조준강화] 8->18: arm 조준을 r_grasp(3)보다 우위로 (mouth_xy 18cm 정체 해소)

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
    # [pour_point 동적화] xy 방향 정적(target)→동적(gravity_perp 실제배출구) blend 전환 임계 (tilt_amount).
    pour_point_dyn_lo:    float = 0.15    # ≈45°: 이하 정적(이송, wobble 회피)
    pour_point_dyn_hi:    float = 0.30    # ≈67°: 이상 동적(deep tilt 정밀 배출구). 사이 smoothstep blend.
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
    palm_delta_xyz: float = 0.03  # [H9] 0.1→0.03: policy step(60Hz)당 최대 target 오프셋(=속도 cap).
                                  #   0.1은 max 6m/s로 급이동→극단/외회전 귀결. 0.03(max 1.8m/s)+EMA 0.7로
                                  #   demo(0.025m/s)에 가까운 천천한 이동. 도달은 palm_mins/maxs 박스가 누적 보장.
    # warmstart cache 수집(체크포인트 rollout) 시 사용할 palm delta.
    # v7-2 grasp checkpoint 학습 조건과 반드시 일치해야 한다:
    #   5g_grasp_right_v7_2: palm_delta_xyz=0.15m, palm_delta_rot_deg=20°
    warmstart_collect_palm_delta_xyz: float = 0.15   # v7-2 학습 값과 일치
    warmstart_collect_palm_delta_rot_deg: float = 20.0  # v7-2 학습 값과 일치 (≠ pour 120°)
    palm_delta_rot_deg: float = 15.0  # current-palm incremental target. 120° absolute target은 Fabrics tracking을 깨뜨림.
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
    # [lstm_test4] 파지 붕괴 종료: cup_rel_drift 과대(슬립/타겟충돌로 파지 붕괴)가 지속되면 terminated.
    #   약한 접촉이 남아 drop_force_hold가 못 잡는 케이스 → 깨진 grasp로 episode 오염 방지.
    grasp_break_drift_deg: float = 45.0   # 정상 deep tilt drift(~30°) 위 마진 (정상 tilt 안 죽임)
    grasp_break_hold_steps: int = 15      # 연속 지속 시 종료 (transient spike 무시)
    # 소스 컵이 비어있는 상태가 N 스텝 연속 지속되면 에피소드 종료
    # 비드 낙하 + 착지에 ~0.3~0.5초 필요 → 60 steps (1.0s @ 60Hz) 여유
    source_empty_hold_steps: int = 60

    # =====================================================================
    # Reward weights — [전면 재설계] 2-Stage 가산 (정책·목표 분리)
    #   total = r_hold + r_approach + r_introt + r_tilt
    #           + r_align(0, dashboard compatibility)
    #           + release_context·aim_gate·r_source_release
    #           + r_target_capture
    #           + w_success·r_success − ready_context·w_spill·sqrt(spill)  # w_spill=0
    #
    #   corridor_score = 1 inside target inlet corridor, exp falloff only outside
    #   ready_context = max(corridor_score, ready_latched·latch_floor)
    #   tilt_ready_factor = tilt_aim_floor + (1 - tilt_aim_floor)·ready_latched
    #   tilt_progress = (tilt_amount / tilt_target).clamp(0,1)      # tilt 강제(직립 farming 차단)
    #   r_approach = −before_ready·(1−approach_corridor_score) − after_ready·(1−corridor_score)
    #   r_align    = 0  # pour_point 고정 local min 제거
    #   aim_gate   = (dir_cos_c>0) & (tilt_amount>drain_tilt_min)
    #   r_source_release = W·clamp(prev_source_fraction − current_source_fraction, 0)
    #   r_target_capture = W·clamp(current_target_fraction − prev_target_fraction, 0)
    #   demo는 reward에 없음 — critic privileged obs로만 (deadlock 제거).
    # =====================================================================

    # Stage A — Grasp maintain (r_hold), tilt-phase aware
    weight_grasp_maintain: float = 0.50
    weight_contact_maintain: float = 0.50
    # [재설계] per-finger 학습 grasp 보상 (DexPour r_contact+r_grasp 통합): 손가락 action 동기
    weight_grasp: float = 3.0          # [A] 5->3: arm 조준 밀림 방지 (cup_drift 유지 확인)
    grasp_full_count: int = 4          # 완전파지 판정 손가락 수 (5중 4)
    grasp_full_bonus: float = 0.5      # 완전파지 시 추가 (DexPour r_grasp 역할)
    weight_force_balance: float = 0.30
    weight_finger_curl: float = 0.50

    # [H11] Stage A — Approach: blended rim_center→pour_point xyz corridor miss penalty.
    #   기존 cup_center(바닥) 기준 폐기 — "rim 평면을 target rim에 마주대러 간다"(사용자).
    #   before ready: r_approach = -W·(1 - corridor_score(blended_xyz)).
    #   after ready: positive precision reward off; actual pour_point corridor miss remains as penalty.
    #   score=1이면 penalty=0. corridor 정밀조준은 reward farming이 아니라 constraint로 둔다.
    weight_dist_to_target: float = 8.0   # [06.18 복원] approach positive exp 당김 weight (이동 잘됨)
    weight_corridor_escape_after_ready: float = 0.0  # [06.21] tilt-swing 처벌로 approach 음수·진동 → v3식 순수 positive pull 복원(penalty 비활성). farming은 spill/pour_gate로 감시.
    approach_anti_floor: float = 0.4         # [06.18 복원] 직립·원거리 transport gradient 보존 (anti=0서도 0.4)
    dist_to_target_exp_scale: float = 5.0
    cup_transport_saturate_xy: float = 0.17  # (레거시, 미사용 — rim_approach_saturate로 대체)
    rim_approach_scale: float = 5.0          # mouth_xy 거리 exp 민감도
    rim_approach_saturate: float = 0.03      # [H12] mouth_xy(pour_point) 이 이하: 거리항 max. rim 반경(0.041) 안쪽으로 견인 (이전 0.05는 rim 밖에서 포화)

    # [2b] nullspace 잉여 1-DOF action(α) 스케일: null_ref = baseline + scale·α·(demo−start).
    #   1.0 → α=±1이 ±full demo변위. v4 baseline=demo, v5 baseline=robot_start (offset·scale 공통).
    nullspace_action_scale: float = 1.0   # [stage2] 0.0→1.0 복원: n_demo nullspace(palm 보존)로 α가 잉여 1-DOF(elbow-swivel) 조절. tilt 안 망침(Stage1: 기존 offset은 tilt 슬라이더라 α 미사용).
    # [stage2] α offset 축 모드: "true_nullspace"=palm 보존 elbow-swivel(n_demo, J@n≈0),
    #   "demo_minus_start"=기존 tilt 슬라이더. true_nullspace면 α가 tilt 안 망치고 잉여 1-DOF만 조절.
    nullspace_offset_mode: str = "true_nullspace"

    # [B-trajectory] 액션 모드. "b_trajectory": action[4]=β(pour progress)→R(β) 전신협응 구동
    #   (cspace baseline=R(β) + ready 시 j5 하드구동). "legacy": 기존 3D tilt 액션.
    #   설계: 보상은 협응 분포를 못 가르침 → R(β)로 직접 부과(7D탐색→1D β). j5만 하드(위치-safe),
    #   j4·어깨는 R(β) soft bias+위치task, j6 작은밴드. introt(spin=action[3])는 유지.
    pour_action_mode: str = "b_trajectory"
    beta_action_index: int = 4   # action[4](구 tilt-toward) = β 채널

    # [β tilt setpoint] β=action[idx]→[0,1]를 목표 tilt_amount로 해석, rim-pivot이 pour-point를
    #   보존하며 그 목표까지 tilt_toward를 피드백 구동. (구 R(β) cspace 절대바이어스 + j5 override는
    #   IK 후 단일관절 덮어써 pour-point를 깨뜨려 폐기 — 검증: v6 ready=0.89인데 β억제로 frac_110=0.)
    beta_target_tilt_amount: float = 0.854  # β=1 목표. (1-cos135°)/2 = 0.854 (135° dump)
    beta_tilt_kp:           float = 3.0     # 목표-현재 tilt_amount 오차 비례게인
    beta_tilt_max_step:     float = 0.06    # tilt_toward 회전 증분 상한 [rad/step] (급격 회전 방지)

    # [B-light] orientation 풀기: palm 방향 명령 제거(=현재 추종) → orientation task가 cspace j5와
    #   경쟁 안 함 → cspace가 j5를 deep까지 끌어 tilt. β는 cspace j5 타겟을 0→demo로 graded 구동.
    #   위치는 주둥이(pour-point) 고정(approach 중 body offset 동결→예측 hold). v5 deep tilt 천장
    #   원인(IK가 j5 대신 손목 포화)을 "orientation task 제거+cspace j5 직접구동"으로 우회.
    pour_orient_release: bool = True

    # [robust] B-light pour 단계에서 주둥이 z를 정책 학습에 맡기지 않고 target 입구 위 margin으로
    #   구조적 강제. v5 실패모드("주둥이가 target 11cm 아래 → 붓기 기하 불가") 원천 차단.
    #   xy 조준은 정책이 유지. (z-barrier 보상은 hinge pour 충돌로 폐기됐으므로 제어로 강제.)
    pour_spout_z_lock: bool = True
    pour_z_margin:     float = 0.03   # 주둥이를 target 입구 위 3cm로 (bead 진입 높이)

    # [stage3] phase별 차등 관절 범위(하드 클램프). ready-latch(pour 단계)일 때만 fabric_q를 band로 클램프.
    #   approach(미ready)=full range(접근/grasp 자유). pour(ready)=아래 lo/hi band(deep tilt 강제).
    #   FK 검증: j6 클램프(leak 차단)+j5 음수 강제(roll 엔진) 동시필요. j6 단독은 80°뿐, 둘이면 113°.
    #   None 성분(±9.9)=해당 단계서 사실상 무제한. 점진 적용 위해 j5/j6만 우선 band.
    pour_phase_clamp_enable: bool = False  # [β 수정] post-IK 관절 클램프/override 전면 비활성화.
    #   deep tilt를 단일관절 강제가 아니라 β-setpoint→rim-pivot(pour-point 보존)으로 구동.
    pour_phase_arm_lo: tuple = (-9.9, -9.9, -9.9, -9.9, -1.571, -0.30, -9.9)  # j6 [-0.30,0.35] (demo 자연범위, 문서검증)
    pour_phase_arm_hi: tuple = ( 9.9,  9.9,  9.9,  9.9,  0.0,    0.35,  9.9)  # j5 상한 0(b_traj는 하드구동이라 무관)

    # [v6 ablation] nullspace baseline(α=0 지점) 선택 — demo prior 주입의 hard 경로.
    #   "robot_start": 중립(=v5 순수 DRL).  "demo": j1-4 항상 + j5 ready 후 demo 구조(=v4).
    #   enable_demo_pose_reward(soft 경로)와 직교 → 둘 조합이 4셀 ablation 매트릭스.
    nullspace_baseline: str = "demo"  # [stage1 진단] robot_start→demo: j5(deep tilt 주역) roll 복원. FK 검증: demo 비율이 tilt 111° 도달, j4 단독은 max 77°. test5(demo+mass3) tilt 해결 메커니즘 복원.

    # Stage A→B 공간 게이트 (target 입구 corridor + ready latch)
    g_ready_center: float = 0.05   # [test_lstm3 재설계] 0.20→0.05: pour_point(mouth_xy)가 target rim 범위(~5cm) 와야 stageB 개방 (정조준 게이트)
    g_ready_width: float = 0.04    # [test7] 0.02→0.04: (a)로 정조준 완벽(mouth_xy~0.003)→sharp 불필요. 깊은 tilt 과도기 mouth_xy 흔들림에 stageB(tilt/align) 절벽 완화 (bead_in은 이미 g_ready 분리)
    pour_corridor_xy_margin: float = 0.015
    pour_corridor_z_min: float = -0.02
    pour_corridor_z_max: float = 0.12
    pour_corridor_scale: float = 20.0
    ready_latch_threshold: float = 0.60
    ready_latch_floor: float = 0.50
    release_gate_floor_after_ready: float = 0.40

    # [tilt 식 교체/test8] 2단 A/B 폐기 → 0→135° 단일 연속 ramp, always-on(aim_floor 부분종속).
    #   test7 진단: A는 85°(tilt_pre)서 saturate(grad→0), B는 85° 넘어야 시작 → 82-85° dead spot에서
    #   정책 정지(peak 0.43<0.456) → tilt_progress_B 영구 미발현. 끊김 없는 단일 gradient로 교체.
    tilt_pre_amount: float = 0.456   # [test8] 로깅 전용(85° 돌파 추적, tilt_progress_B). 보상 미사용
    weight_tilt_pre: float = 8.0     # [test8] 미사용(r_tilt_A 폐기). 구 기록 참조용 유지

    # tilt 직접 유도 (v6 ALIGN 실패 교훈: tilt를 직접 보상해 직립 회피해 차단)
    weight_tilt: float = 20.0      # [deep_tilt_boot1] tilt 독립(latched_ready 제거) 시 유지보상 축소(35→20): 유지 farming 완화, 증분(delta) 위주.
    weight_tilt_delta: float = 100.0   # [test4] tilt 증분(delta) 보상 가중. 더 기울이는 순간만(relu)→75° 너머 deep tilt 유도. 위치(z/xy) 맞춰진 test3 위에서 deep tilt 점프 유발.
    tilt_aim_floor: float = 0.35   # r_tilt pre-ready bootstrap floor: w·progress·rot_dir·(floor+(1-floor)·prox_gate)
    # [06.21 Phase2] tilt 게이트를 latch(binary)→연속 근접 게이트로 교체(순환 게이트 절단).
    #   prox_gate = clamp((far - approach_xy_dist)/(far-near), 0, 1). approach_xy_dist=rim_center 기준.
    tilt_prox_gate_far:  float = 0.25  # 이 거리 밖: prox_gate=0 (floor만)
    tilt_prox_gate_near: float = 0.06  # 이 거리 안: prox_gate=1 (full tilt)
    # [06.21 Phase3] pour 정밀 조정텀 r_pour = w_pour·tilt_progress·aim_score. aim_score=관대 radius corridor.
    #   tilt_progress 스케일 → 회전 시작(흔들림) 구간 자동 억제, 70°+ 안정 구간만 본격 작동.
    weight_pour:        float = 50.0   # [Phase1 미사용] 구 r_pour=w_pour·progress·aim 곱셈 → 덧셈(transport+align)로 분해. 참조용 유지.
    # [Phase1] DexPour additive 분해: 곱셈 r_pour(progress×aim saddle) → r_transport(aim, tilt무관) + r_align(dir_cos, tilt무관).
    #   진단: deep tilt 시 pour_point가 target에서 이탈(pp_z +14cm)해도 곱셈이 둘 다 높을 때만 보상→saddle.
    #   덧셈이면 위치(transport)와 기울임(tilt)을 독립 보상 → "위치 유지하며 deep tilt"가 합 최대.
    #   aim_score는 z corridor(z_max=0.05) 내장 → r_transport가 pp_z 솟음을 자동 페널티(Phase2 포함).
    weight_transport:   float = 30.0   # [구 r_pour z-only, 재설계 Phase3서 미사용] 보존(롤백 참조).
    weight_pour_bead:   float = 50.0   # [재설계 Phase3] r_pour = w·corridor_score·bead_cross_fraction. 실제 붓기 outcome 보상(z-only 대리 폐기).
    capture_delta_weight: float = 30.0  # [재설계 Phase2b] r_pour에 진입 증분(target_capture_delta) 가중. 자기참조(상태)+증분 혼합 → plateau 해소 + farming 차단.
    pour_z_target:      float = 0.03   # [test3] 주둥이를 target 입구 위 3cm로 유도 (충돌회피 마진 + bead 진입 높이)
    pour_z_scale:       float = 20.0   # [test3] z-clearance 오차 exp 민감도 (3.5cm서 반감)
    pour_aim_scale:     float = 10.0   # [test_aim2] aim corridor 완만화(공유 pour_corridor_scale=20 절벽 → 10). env서 radius=0(flat-top 제거)와 함께 → target서 부드러운 봉우리(gradient 어디서나) → miss 교정 + 학습 stiffness↓. 부드러운 동작인데 reward 출렁이던 ② 원인 제거.
    pour_aim_z_max:     float = 0.05   # [test_aim] aim 전용 z 상한(공유 pour_corridor_z_max=0.12와 분리). spout이 입구 위 5cm 넘으면 감점 → release 높이발 산란 차단. z_min은 pour_corridor_z_min(-0.02) 공유(soft band, tilt 자연 하강 허용).
    #   ready_latched 이후에는 live corridor wobble이 tilt reward를 끄지 않음. 미정조준 ceiling=35×0.35=12.25.
    # [H10] 상시 내회전 유도 — r_tilt(곱)는 tilt 전엔 회전 gradient=0(chicken-and-egg) →
    #   tilt 비종속 항으로 "내회전이 옳다"를 직접 학습. w_tilt보다 작아
    #   "회전만 park" 아닌 "회전 후 tilt"로 견인. lstm_test2 internal_rot_gate 자발하락 대응.
    weight_introt: float = 5.0
    pour_tilt_target_deg: float = 135.0   # 수평(90°) 너머 dump까지 tilt_progress gradient 유지

    # Stage B — direct pour-point 정렬 reward 제거. Corridor는 phase/release/logging context로만 사용.
    weight_align: float = 5.0   # [재설계 Phase3] 20→5 하향: 방향항이 최대 보상(16.8)으로 방향-only farming 유발 → 보조로 축소. cosθ=directional_tilt_cos_c.
    pour_align_scale: float = 15.0  # 레거시 pour_alignment_score config. 현재 reward path 미사용.
                                    #   (6.8 vs 4cm가 score .58 vs .73) mouth_xy가 입구반경(4.1cm) 밖에서 천장 → bead_in=0.
                                    #   sharp화로 마지막 4cm 파고들어 bead_in 개통 유도.
    pour_align_z_margin: float = 0.10  # 레거시 pour_alignment_score config. 현재 reward path 미사용.
                                       #   정책이 독립 제어(rim-pivot, env.py L1028)인데, z 상한 5cm가
                                       #   깊은 tilt(따르기 자세 mouthZ 6~8cm)에 페널티를 줘 ~80° park 유발.
                                       #   10cm까지 z 무관·xy만 강제 → 깊은 tilt gradient 생존,
                                       #   10cm↑ 고공 살포만 차단(흘림 방지). delta_z.clamp(min=0)은 유지.

    # [제거] weight_pour_z / pour_z_margin: z barrier가 hinge pour와 상충하여 삭제 (lstm_test4 주기적 붕괴 원인)

    # Stage B — bead (정밀 조준 종속 — "높은 데서 대충 부어 넣기" 차단)
    weight_bead_in: float = 0.0  # [release-delta probe] 누적 target 상태 reward 제거
    weight_source_release: float = 100.0  # 소스 잔량 감소분만 transient 보상
    weight_target_capture_delta: float = 200.0  # [lstm_test4] test3 400→200 복원: bank 품질 게이트 격리 검증.
    weight_bead_cross: float = 150.0  # 레거시 입구 관통 reward config. 현재 reward path 미사용.
    weight_source_drain: float = 0.0  # [release-delta probe] 누적 source-empty 상태 reward 제거
    drain_tilt_min: float = 0.05   # aim_gate tilt 임계 (직립 조기 drain 차단)
    align_gate_scale: float = 15.0    # 레거시 align gate config. 현재 reward path 미사용.
    bead_near_scale: float = 12.0     # _compute_bead_flags의 _bead_near_score 계산용 (진단 버퍼)

    # [H4] 내회전 게이트 — 붓는 방향(source→target) 대비 손바닥 법선 chirality (2D 외적)
    #   rot_cross = pour_dir × palm_normal. demo(내회전) -0.74~-1.0, 외회전 >-0.2 (완벽 분리).
    #   r_tilt에 곱해 외회전 tilt local min 차단. r_introt는 부트스트랩.
    # [H11] 내회전 판정 = rl_dg_palm +y · world +x < thresh (cos<0=둔각 90~270°, 손바닥 roll).
    #   기존 palm+z(손가락축, H10b)는 roll 무감지(cos≈1 고정) → drift 못 막음. sim 렌더링 확인.
    #   gate = sigmoid((thresh - cos)/temp).
    internal_rot_thresh: float = 0.0   # cos<thresh → 내회전. (경계 cos=0=90°)
    internal_rot_temp: float = 0.4     # [test_aim2] 0.1→0.4 완만화: temp 0.1 가파른 sigmoid가 부드러운 rim_facing 변화→gate 급변→r_tilt(×rot_dir) 출렁(학습 진동 ①). 완만화로 reward stiffness↓.
    # [H5] roll 방향성을 r_tilt에 결합 (별도 r_introt 가산은 자세 압도 부작용 → 폐기).
    #   r_tilt *= rot_tilt_floor + (1-rot_tilt_floor)*internal_rot_gate.
    # [H10] floor 0.3 → 0.0 (lstm_test2 분석): floor=0.3이 외회전(gate≈0) tilt에 30% 보상 →
    #   "외회전으로 살짝 기울여 받기"가 가장 쉬운 tilt local min (rim_facing_cos 0.21→0.42 외회전 심화,
    #   j5 demo −1.16과 정반대 +1.02). 외회전 유도 불필요 → floor=0으로 내회전 시에만 tilt 보상.
    #   rot_dir = internal_rot_gate. bootstrap은 초기 gate 분포(0.13~0.5)에 의존.
    rot_tilt_floor: float = 0.0

    # Outcome
    weight_success: float = 50.0   # [06.21 Phase4] outcome 신호 연결: shaping이 닻 없이 farming하던 문제 해소(이전 0)
    weight_spill: float = 0.0      # [test7] 2→0: lstm_test6 bead_in/spill 동조 붕괴 → spill 페널티 OFF (pour 회피 local min 제거)

    # EMA palm action smoothing: Fabrics IK에 smooth 궤적 전달
    ema_action_alpha: float = 0.7   # 새 action 70% / 이전 EMA 30%

    # -----------------------------------------------------------------------
    # Demo (critic privileged obs 전용 — 정책 reward에 사용하지 않음)
    #   "현재자세 ↔ demo pour 자세" 거리를 critic obs로만 제공 → value 추정 가속,
    #   초기 탐색 감소. 정책은 demo를 못 봄(reward hacking·deadlock 제거).
    # -----------------------------------------------------------------------
    enable_demo_critic_obs: bool = True
    demo_pose_dataset_dir: str = _DEFAULT_DEMO_POSE_DATASET_DIR
    demo_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )
    demo_pose_phase: str = "pour"
    demo_nn_lookahead_frames: int = 10

    # -----------------------------------------------------------------------
    # Demo pose REWARD (v3 이식 — pour_v4)
    #   진단(lstm_test2 붕괴): deep tilt 시 j6/j7 손목이 range 소진 → pour-point z escape.
    #   원인은 "실현 가능한 tilt joint_state를 탐색으로 못 찾음".
    #   a11~a20 pour 구간 joint 분포(검증된 한계 내 deep tilt 자세)로 j1-4 + j5(틸트 주역)를
    #   당겨 초기에 올바른 joint_state를 빠르게 찾게 한다. weight는 flow EMA로 floor까지 감쇠하되
    #   floor를 남겨 후반 escape도 억제(lstm_test2는 iter 1000+ 후반 붕괴).
    # -----------------------------------------------------------------------
    enable_demo_pose_reward: bool = False  # [06.21 재설계] actor demo 보상 OFF(r_demo_arm_pose/j5=0). demo는 critic 전용(enable_demo_critic_obs=True)으로만 초기 탐색 축소.
    weight_demo_arm_pose: float = 20.0        # j1-4 demo 앵커 시작값
    weight_demo_arm_pose_floor: float = 5.0   # 감쇠 하한 (후반 anchor 유지)
    weight_demo_j5: float = 15.0              # j5(틸트 주역) 앵커 시작값, ready 이후만
    weight_demo_j5_floor: float = 3.0
    demo_j5_sharpness: float = 2.0
    demo_pose_near_gate_xy: float = 9999.0    # 사실상 gate off (항상 1)
    demo_pose_warmup_steps: int = 1
    demo_graduate_flow_target: float = 0.05   # flow EMA가 이 값 도달 시 weight→floor
    demo_graduate_ema_alpha: float = 0.001    # flow EMA 갱신 속도

    # ADR: spill penalty 스케줄 (low→high)
    enable_spill_adr: bool = False   # [test6] spill 패널티 OFF와 함께 ADR도 끔 (weight_spill=0 사용, ADR이 덮어쓰지 않게)
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

    # [재설계 outcome ADR] DexPour 커리큘럼: 자세 성공률 80%+ 시 bead 보상(r_pour) weight 0→50 램프.
    #   1단계=자세만(approach/tilt/align), bead 로깅만. 2단계=자세 완성 후 bead_in_target 상태보상 활성 → 정밀 pour.
    enable_outcome_adr: bool = True
    outcome_adr_custom_cfg: dict = {
        "outcome": {
            "weight_pour_bead": (0.0, 50.0),  # 1단계 0(자세만) → 2단계 50(bead 보상)
        }
    }
    outcome_adr_num_increments: int = 8
    outcome_adr_increment_interval: int = 20000
    outcome_adr_trigger_threshold: float = 0.80  # 자세 성공률 80%+ 시 outcome 활성
    pose_ready_thresh: float = 0.60   # 자세 성공 게이트: corridor_score ≥ (위치 준비)
    pose_tilt_thresh: float = 0.587   # 자세 성공 게이트: tilt_amount ≥ (1-cos100°)/2 → rim_antiparallel ≤ -0.174 (100°+)

    reward_grasp_slip_sharpness: float = 3.0
    contact_maintain_min_others: int = 2
    force_balance_sharpness: float = 2.0

    # success 판정용 게이트: cup_center_xy_dist < thresh
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

    # [lstm_test1 §6] deep-tilt 부트스트랩: sparse chicken-and-egg 해소.
    #   학습 중 정책이 만드는 "deep-tilt + target 위 + 비드 source 보유" 실제 프레임을
    #   full-snapshot으로 캡처했다가 일부 reset을 그 상태에서 시작 → 정책이 마지막 push만
    #   학습해 200 캡처 보상을 경험. f_boot anneal로 직립 시작에 전이. (probe로 feasibility 확증)
    enable_deep_tilt_boot: bool = True
    deep_tilt_boot_capacity: int = 4096            # ring buffer 용량 (full-snapshot 수)
    deep_tilt_capture_tilt_min: float = 0.40       # tilt_amount=(1-cosθ)/2; 0.40≈78°
    deep_tilt_capture_src_min: float = 0.80        # 비드 source 보유율 하한 (공짜 pour 방지, audit Check 2)
    deep_tilt_capture_mouth_max: float = 0.08      # pour-point xy 거리 상한 (target 위)
    # [lstm_test4] grasp 품질 게이트: 슬립 중인 상태 캡처 금지 → bank 자기 오염 방지 (test3 붕괴 원인)
    deep_tilt_capture_drift_max: float = 12.0      # cup_rel_drift 상한 [deg] (grasp 일관성)
    deep_tilt_capture_contacts_min: float = 3.0    # 최소 접점 수 (파지 유지)
    deep_tilt_capture_prob: float = 0.05           # qualifying env를 step당 저장할 확률 (중복 억제)
    deep_tilt_boot_min_count: int = 64             # 이 수 이상 쌓여야 부트스트랩 시작
    deep_tilt_f_boot_start: float = 0.5            # 초기 부트스트랩 비율
    deep_tilt_f_boot_end: float = 0.0              # anneal 종착 (직립 전이)
    deep_tilt_anneal_steps: int = 300_000          # progress = common_step_counter / anneal_steps
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
            # [grip fix v2 절충] stiffness 800은 과도 — 손가락이 컵을 너무 꽉 잡아 deep tilt 시 target
            #   충돌 반력이 팔로 전달돼 tilt 경직(t27/t29: frac_110 0.4→0, pose_succ 0.95→0.2, entropy 발산).
            #   약한 손가락의 충돌 완충이 사라진 것. 100(deep tilt OK/slip)↔800(grip OK/tilt막힘) 절충=400.
            #   grip은 원래(100)보다 강화 유지하되 deep tilt 완충 여지 남김.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=200.0,
                damping=35.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=400.0,
                damping=60.0,
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
