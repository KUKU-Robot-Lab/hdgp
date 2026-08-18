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
from openarm.common.bead_assets import make_beads_cfg as _make_shared_beads_cfg
from .pour_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .pour_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_HAND_JOINT_NAMES,
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


def _make_beads_cfg(n: int = _DEFAULT_BEAD_COUNT) -> RigidObjectCollectionCfg:
    """★[both/pour_v1] 비드 정의를 `openarm.common.bead_assets` 로 옮겼다.

    이유: pour_v1 은 warm state 의 `bead_state` 를 **그대로 복원**해 시작한다. 그 상태를
    만든 쪽(grasp_v1 수집)과 복원하는 쪽(여기)의 비드 물성이 다르면 같은 좌표를 복원해도
    동역학이 달라진다. 정의를 한 곳에 두어 양쪽이 물리적으로 동일한 비드를 쓰게 한다.
    (수치는 그대로 이관 — 거동 변화 없음)
    """
    return _make_shared_beads_cfg(_ASSETS_DIR, n)


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
    # [B-full 복귀] 새 구조(soft cspace) 3814ep 완주 검증: j5 미구동(gap 1.6 정체) → soft 불가 확정.
    #   explicit B-full만 palm 고정+j5 강제 가능 → palm_position_only=False(7-point), bfull=True 복귀.
    palm_position_only: bool = False
    pour_bfull_nullspace: bool = True
    bfull_step:   float = 0.04   # arm→demo 향한 per-step 관절증분 상한 [rad]
    bfull_lambda: float = 0.05   # DLS pseudo-inverse 댐핑(특이점 방지)

    # [대조군] approach 제어 방식: "rim"(action xy=주둥이 직접) | "palm"(action xy=palm 직접).
    #   나머지(B-full nullspace·z-lock·orientation release·reward) 전부 공통 → 제어방식만 비교.
    #   v5="rim", v6="palm".
    pour_approach_pivot: str = "palm"

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
    # [단계형 학습 인계] 이전 단계가 도달한 ADR 레벨을 시작값으로 주입(0=처음부터).
    #   체크포인트에는 카운터가 저장되지 않으므로, 이전 런 TB의 `log/adr_inc_<name>` 최종값을 넣는다.
    #   PourADR.set_increment가 [0, num_increments]로 클램프. 기본 0 = 기존 동작 완전 동일.
    #   ⚠ ablation 대조군 런에는 넣지 않는다(커리큘럼 진도가 달라져 비교 무효).
    noise_adr_initial_increment: int = 0


    # bead / cup geometry (.usd 기준: bottom=-0.077m, rim=+0.100m, inner_r=0.041m)
    target_inner_radius:  float = 0.041   # 컵 내부 반경
    target_inside_z_min:  float = -0.070  # bottom(-0.077) + bead_radius(~0.01) 여유
    target_inside_z_max:  float = 0.100   # 림 높이
    target_mouth_z:       float = 0.100   # 림 높이 (bead crossing 기준)
    # [generalization] 받는(target) 컵 크기 sweep. 1.0=학습. env __init__에서 !=1.0이면
    #   물리 자산(left_target_cup_cfg.spawn.scale)+판정기하(inner_radius/inside_z/mouth_z/opening_pos_b)
    #   를 s배로 비례 조정 → 미학습 컵 크기 일반화(작을수록 정밀도 하드). 학습(1.0)은 무영향.
    left_target_cup_scale: float = 1.0
    # [generalization] 받는 컵 opening-only sweep. 1.0=학습. env __init__에서 !=1.0이면
    #   xy만 s배(높이 z 불변) → uniform scale과 달리 opening 직경만 변하는 geometry OOD
    #   (narrow/wide opening). inner_radius만 s배 조정(opening_pos_b는 xy=0이라 무변).
    #   left_target_cup_scale과 곱으로 조합 가능. 학습(1.0)은 무영향.
    left_target_cup_scale_xy: float = 1.0
    # [DR 재학습] 받는 컵 스케일 세트 — 비어있지 않으면 env_id % K 결정론 배정
    #   (MultiAssetSpawnerCfg, replicate_physics=False 자동)으로 per-env 다른 스케일 컵을
    #   스폰하고 판정기하(inner_radius/inside_z/mouth_z/opening_pos)를 per-env 텐서로 조정.
    #   학습 시 receiver geometry DR → cup-scale 일반화창 확장 목적. ()=기존 단일 스케일.
    #   ★[both/pour_v1] 컵 다양화 학습 — source/receiver 둘 다 스케일을 섞는다.
    #     receiver 는 물리 파지로 바뀌었으므로 warm 파지자세의 컵 스케일과 반드시 일치해야
    #     한다 → `left_warm_spec_map` 으로 scale_set[i] ↔ grasp spec idx 를 명시한다.
    left_target_cup_scale_set: tuple[float, ...] = (0.85, 1.0, 1.15, 1.30)
    # scale_set[i] ↔ grasp_v1 warm spec idx (grasp `_ACTIVE_OBJECT_SPECS` 순서: 0=s085,1=s100,2=s115,3=s130)
    left_warm_spec_map: tuple[int, ...] = (0, 1, 2, 3)
    # -------------------------------------------------------------------
    # ★[both/pour_v1] 좌우 컵 겹침 — 해결 경로와 폐기된 시도 (2026-08-18)
    # -------------------------------------------------------------------
    # 문제: grasp_v1 의 리프트는 joint7 만 0.31rad 돌리는 스크립트 동작이라 잡은 컵을
    #   측면으로 스윙시킨다. 컵이 스폰에서 좌우 각각 약 6cm 중앙으로 모여, pour_v1 에서
    #   둘을 합치면 페어의 44.3% 가 물리적으로 관통했다(중심거리 중앙 98mm < 반경합 94mm).
    #   겹친 채 리셋하면 PhysX 가 밀어내며 파지를 뜯는다
    #   (게이트 실측: 우컵 유지 0.188 · 접촉 0.81개 · 64env 중 11 완주 · 비드 0.042).
    #
    # 해결: **grasp 스폰을 ∓0.20 으로 벌린다**(`object_spawn_y_center`). 정책은 바꾸지 않는다.
    #   실제 좌/우 페어(2048×2048=419만 조합) 실측:
    #     ∓0.10 → 간격  80mm · 겹침 44.3%
    #     ∓0.20 → 간격 197mm · 겹침 **0.019%** (pour 사용 spec 0-3 은 0.060%)
    #   남는 꼬리는 아래 `left_right_cup_min_gap_m` 재추첨이 걸러낸다.
    #   실기에서도 컵을 벌려 놓으므로 s2r 과 더 정합한다.
    #
    # ⚠ 폐기된 시도 1 — "정책이 y 를 끌어당기니 스폰을 벌려도 소용없다"
    #   틀렸다. y 분산은 압축되지만(비 0.45) **평균은 따라 이동한다**. 같은 체크포인트로
    #   ∓0.16 · ∓0.20 에서 각각 256/256 을 수집해 확인했다. 재학습도 필요 없었다.
    #
    # ⚠ 폐기된 시도 2 — "j1(베이스 요)로 warm 자세를 회전해 벌린다"
    #   `j1` 은 베이스 요가 아니다. URDF `axis="0 0 1"` 은 조인트 로컬 기준이고 부모 마운트에
    #   rpy=(-pi/2,0,0) 이 걸려 있다. FK 실측: `l_aj_1 +0.15` → Δpalm=[-0.057, 0.000, -0.034]
    #   (수평반경 0.275→0.222, z 0.333→0.299) — **Δy=0.000**. y 주축은 j2 다.
    #   구현은 관절을 j1 으로 돌리고 컵·palm 좌표는 z축 회전으로 옮겨 **손과 컵이 분리**됐다
    #   (접촉 0.81→0.03개 · 완주 0/64 · 비드 0.859→0.181). 좌표만 맞추고 팔이 그 pose 에
    #   도달하는지 검증하지 않은 것이 원인이다. 되살리려면 목표 palm pose 를 IK 로 풀 것.
    # ★[both/pour_v1] 좌우 컵이 겹치는 페어는 **재추첨**한다 (안전장치).
    #   grasp 스폰을 ∓0.20 으로 벌려 겹침을 44.3% → 0.05% 로 낮췄지만(실제 좌/우 페어
    #   8000쌍 실측), 분포 꼬리에 소수가 남는다. 겹친 상태로 리셋하면 PhysX 가 두 컵을
    #   밀어내며 파지를 뜯어낸다(게이트 실측: 접촉 0.81개, 64env 중 11개만 완주).
    #   재추첨 비용은 0.05% 에 대해 무시할 수준이다.
    #   min_gap: 두 컵 반경 합에 더할 여유[m]. 0 이면 접촉 직전까지 허용.
    #   컵 외경 = per-env 내부 반경(스케일 반영) + 이 벽 두께. 실측 nominal 외경 47.2mm
    #   vs 내부 41mm → 6.2mm. 여유를 둬 7mm 로 잡는다(과소평가하면 겹침을 놓친다).
    cup_wall_thickness_m: float = 0.007
    left_right_cup_min_gap_m: float = 0.005
    # 재추첨 최대 시도. 초과분은 경고와 함께 그대로 둔다(무한 루프 금지).
    left_right_cup_redraw_tries: int = 8
    # [DR 재학습] source 컵 스케일 세트 — grasp_v1 cup 스펙과 동일 순서 권장
    #   (예: (0.85, 1.0, 1.15, 1.30) = grasp_v1 spec 0..3). env_id % K 결정론 배정(MultiAsset)
    #   + warm state의 object_spec_idx 를 env 스케일과 매칭해 샘플(파지자세-컵크기 정합).
    #   bead-in-source 판정만 per-env 스케일; 리워드/pour_point 기준은 nominal 유지(크기 무관 설계).
    source_cup_scale_set: tuple[float, ...] = (0.85, 1.0, 1.15, 1.30)
    # scale_set[i] ↔ grasp_v1 warm spec idx 매핑 (기본: cup_big 4종 = 0..3)
    source_warm_spec_map: tuple[int, ...] = (0, 1, 2, 3)
    # [DR 정합] source_cup_scale_set 스폰 시 컵 질량을 스케일 무관하게 이 값(kg)으로 고정.
    #   grasp_v1 _ACTIVE_OBJECT_SPECS(cup_big 4종 mass=0.134)와 정합 — 수집↔소비 force-ratio 일치.
    #   미고정 시 USD density 파생으로 s³ 변동(0.082/0.134/0.204/0.294)해 1.3배 컵에서 2.2배 무거워진다.
    #   None이면 파생값 유지. scale_set 미사용 기본 경로는 이 값을 읽지 않음(기존 물리 완전 보존).
    source_cup_fixed_mass: float | None = 0.134
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
    pour_orient_release: bool = True  # [B-full 복귀] orientation 풀기(B-light) 재활성

    # [robust] B-light pour 단계에서 주둥이 z를 정책 학습에 맡기지 않고 target 입구 위 margin으로
    #   구조적 강제. v5 실패모드("주둥이가 target 11cm 아래 → 붓기 기하 불가") 원천 차단.
    #   xy 조준은 정책이 유지. (z-barrier 보상은 hinge pour 충돌로 폐기됐으므로 제어로 강제.)
    pour_spout_z_lock: bool = True
    # ★[both/pour_v1] z-lock 기준 높이에 대한 1차 저역통과 계수 (EMA: new = a·raw + (1-a)·prev).
    #   구 pour_sensor 는 receiver 컵이 kinematic 이라 `_target_opening_w[:,2]` 가 준정적이었다.
    #   pour_v1 은 그 컵을 왼손이 물리적으로 쥐고 있어 흔들리고, 최악의 경우 떨어진다.
    #   주둥이 높이를 그 값에 **해석적으로 강제**하는 구조이므로, 필터 없이 쓰면 컵 진동이
    #   오른팔 목표에 그대로 주입되어 서로를 흔드는 되먹임이 생길 수 있다.
    #   1.0 = 필터 없음(구 동작과 동일). 0.2 ≈ 5스텝 시상수.
    pour_spout_z_lock_lpf_alpha: float = 0.2
    pour_z_margin:     float = 0.03   # [test30 정렬] v5(test30 grip400)와 동일 0.03. (aim 실험값 0.07 되돌림 — 대조군 무결성)

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
    pour_aim_scale:     float = 10.0   # [test30 정렬] v5(test30 grip400)와 동일 10.0. (aim 실험값 15 되돌림 — 대조군 무결성)
    pour_aim_z_max:     float = 0.05   # [test30 정렬] v5(test30 grip400)와 동일 0.05. (aim 실험값 0.10 되돌림 — 대조군 무결성). z_min은 pour_corridor_z_min(-0.02) 공유.
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
    # [H11] 내회전 판정 = r_hl_palm +y · world +x < thresh (cos<0=둔각 90~270°, 손바닥 roll).
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
    spill_adr_initial_increment: int = 0   # [단계형 인계] noise_adr_initial_increment 주석 참조

    # ADR: success 기준 커리큘럼 (fill_ratio: 낮은 기준→높은 기준)
    # bead 10개 기준: 0.20=2개, 0.30=3개, 0.40=4개, 0.50=5개
    # 해당 기준에서 success_rate >= 15%이면 한 단계 올림 (8단계 × 0.0375 = 0.30 range)
    enable_success_adr: bool = True
    success_adr_custom_cfg: dict = {
        "success": {
            "fill_ratio": (0.20, 0.50),  # 2개→5개 커리큘럼
            # [pour_v1 정렬] r_aim 급경사도(pour_aim_scale) 단계 상승 10→15 → 주둥이를 입구 중심으로 점진 유도.
            "aim_scale": (10.0, 15.0),
        }
    }
    success_adr_num_increments: int = 8
    # [pour_v1 정렬] 20000이면 첫 체크가 ~40M프레임(env-step 20000)에서야 발생 → success_adr(aim/fill) 미발동.
    #   2000으로 낮춰 ~4M프레임마다 1단계 체크 → 8단계 ≈ 32M프레임에 완주.
    success_adr_increment_interval: int = 2000
    success_adr_trigger_threshold: float = 0.15  # 현재 기준에서 15% 성공률 달성 시 상향
    success_adr_initial_increment: int = 0   # [단계형 인계] noise_adr_initial_increment 주석 참조

    # [재설계 outcome ADR] DexPour 커리큘럼: 자세 성공률 80%+ 시 bead 보상(r_pour) weight 0→50 램프.
    #   1단계=자세만(approach/tilt/align), bead 로깅만. 2단계=자세 완성 후 bead_in_target 상태보상 활성 → 정밀 pour.
    enable_outcome_adr: bool = True
    outcome_adr_custom_cfg: dict = {
        "outcome": {
            "weight_pour_bead": (0.0, 50.0),  # 1단계 0(자세만) → 2단계 50(bead 보상)
        }
    }
    outcome_adr_num_increments: int = 4          # [07.02 speed-fix②] 8→4: weight 0→50 램프를 4단계로(활성 후 ~500ep 완주). 물리 fix가 착취 차단하므로 큰 스텝 안전
    outcome_adr_increment_interval: int = 8000   # [07.02 speed-fix②] 20000→8000: 체크 간격 단축(~125ep마다)
    outcome_adr_trigger_threshold: float = 0.80  # 자세 성공률 80%+ 시 outcome 활성 (이제 EMA 윈도우 기준 — speed-fix①)
    # [단계형 인계 ★핵심] 08.16 실측: 컵 스케일 DR 하에서 pose_success EMA가 0.80을 한 번도 못 넘어
    #   weight_pour_bead가 6000+ iter 내내 0 = pour 보상 부재로 학습 자체가 불가능했다(FullDR_s42/s43).
    #   M4처럼 이미 outcome을 획득한 정책에서 재개할 때는 max(=outcome_adr_num_increments)를 주입해
    #   보상 그라디언트를 유지한 채 DR에 적응시킨다.
    outcome_adr_initial_increment: int = 0
    pose_success_ema_alpha: float = 0.1          # [07.02 speed-fix①] outcome_adr trigger용 pose_success를 누적(평생)→EMA(최근 윈도우)로. 초반 실패에 눌린 느린 활성화 해소(ep937→~ep300)
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
    # 기본 "disk": train.py 가 override 없이도 grasp_warm_tesollo.hdf5 를 로드.
    # 파일이 없으면 자동으로 rollout 으로 fallback 하므로 안전.
    # Tesollo grasp_v1 전용 산출물: data/grasp_warm_tesollo.hdf5
    #   (collect_grasp_v1_warm_states.py --robot tesollo). warm-state 는 grasp 성공
    #   초기 pose(arm/hand/cup)라 v6 의 7-D action 차원 변경과 무관하게 유효하다.
    warm_state_source: str = "disk"
    # [both/pour_v1] **오른팔** warm bank (source 컵을 쥔 상태). right/grasp_v1 산출물.
    #   grasp 성공 초기 pose(arm/hand/cup). 파일 없으면 rollout 으로 안전 degrade.
    warm_state_paths: tuple[str, ...] = (
        _os.path.normpath(_os.path.join(_HDGP_ROOT, "data", "grasp_warm_tesollo_right.hdf5")),
    )
    # ★[both/pour_v1] **왼팔** warm bank (receiver 컵을 쥔 상태). left/grasp_v1 산출물.
    #   구 pour_sensor 에는 없던 항목이다 — 왼손이 2-DOF 그리퍼였고 컵은 kinematic-follow 라
    #   왼팔 초기 파지 상태라는 개념 자체가 없었다. pour_v1 은 양손이 실제로 쥐고 시작하므로
    #   좌/우 두 뱅크를 **각각** 로드해 리셋에서 함께 적용한다.
    #   ⚠ 이 파일이 없으면 왼손은 pre-grasp rest 로 남아 컵을 쥐지 못한다 → 리셋에서 경고 후
    #     기존 FK 고정배치로 degrade 한다(학습은 돌지만 양손 파지 실험이 아니게 되므로 반드시 확인).
    left_warm_state_paths: tuple[str, ...] = (
        _os.path.normpath(_os.path.join(_HDGP_ROOT, "data", "grasp_warm_tesollo_left.hdf5")),
    )
    # [grasp_v1 재연결] multiasset 태깅 캐시에서 사용할 물체 스펙 인덱스
    #   (grasp_v1 _ACTIVE_OBJECT_SPECS 순서; 1 = cup_big_s100 = pour source 컵과 동일 스케일).
    #   구캐시(태깅 없음)는 필터가 자동 무시됨. 빈 튜플 = 필터 없음.
    warm_object_spec_filter: tuple[int, ...] = (1,)
    # 왼팔용 spec 필터 — receiver 컵도 cup_big_s100 동일 스케일이라 오른팔과 같은 값을 쓴다.
    left_warm_object_spec_filter: tuple[int, ...] = (1,)
    # [warm 검증필터] True면 warmstart pick을 랜덤 대신 순차(spec 풀 내 라운드로빈)로 하고
    #   env별 pick 인덱스를 self._last_warm_pick 에 기록 — pour 물리에서 warm bank 전수
    #   안정성 검증(validate_warm_states_in_pour.py)용. 학습에서는 False 고정.
    warm_validation_sequential: bool = False
    freeze_grasp_hand_during_episode: bool = True
    # 최상위 비드 z=0.063m (림 0.100에서 3.7cm 아래, 리셋 시 기울어진 컵에서 탈출 방지)
    bead_spawn_pos_source_cup_b: tuple[float, float, float] = (0.0, 0.0, 0.015)
    bead_spawn_quat_source_cup_wxyz: tuple[float, float, float, float] = tuple(
        BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ
    )
    # FK 기반 고정 배치 (LEFT_ARM_REST_JOINT_POS에서 hand local_z=0.05)
    left_target_cup_pos_env_local: tuple[float, float, float] = tuple(LEFT_TARGET_CUP_POS_ENV_LOCAL)
    left_target_cup_quat_wxyz: tuple[float, float, float, float] = tuple(LEFT_TARGET_CUP_QUAT_WXYZ)
    # 왼팔 TCP(DifferentialIK) 제어 파라미터.
    #   ★pour_v1 은 컵 kinematic-follow 를 제거했다(왼손이 물리적으로 쥔다) → 관련
    #     파라미터 `left_cup_follow_local_z` 도 함께 삭제했다.
    #   left_tcp_action_delta_m:  policy step당 왼팔 TCP 위치 최대 이동량[m] (속도 cap; rest 기준 누적)
    #   left_tcp_workspace_range: rest TCP 기준 위치 클램프 박스 half-extent [m] (x,y,z)
    # ★[both/pour_v1] 리셋 시 receiver 를 **내려놓는** 높이[m] (2026-08-18).
    #
    # 왜: pour_v1 은 왼손이 컵을 들고 있어 receiver 가 pour_sensor 대비 7.4cm 높다
    #   (z 0.291 → 0.365). source 도 z 0.367 이라 두 컵이 같은 높이에서 시작하는데,
    #   붓기는 원리상 source 가 위여야 성립한다. 그 결과 `pour`/`bead_in`/`success`
    #   보상이 **한 번도 발화하지 못했고**(A-E1-frozen 2442 epoch · A-E1-learned 319 epoch
    #   모두 0.0000), gradient 가 없으니 정책은 접근을 포기하고 자세 항만 챙겼다
    #   (approach 보상 1.33→1.10, mouth_xy_dist 0.174→0.221 로 **멀어짐**).
    #
    # 어떻게: 왼팔 rest TCP 를 이만큼 낮게 잡는다. `episode_hold_steps`(120) 동안 왼팔
    #   목표는 rest 로 강제되므로 DLS 가 팔을 내리고, 컵은 **물리적으로 쥐여 있어** 따라
    #   내려온다(텔레포트 아님 = s2r 정합). 하강률 cap 은 `left_tcp_action_delta_m`(1cm/step)
    #   이라 8cm 는 약 8 스텝이면 끝난다.
    #
    # ★2026-08-18 **기본값 0.0 (비활성)** — 도입 근거가 반증됐다.
    #   도입 이유였던 "두 컵이 같은 높이라 붓기 불가" 는 틀렸다. `pour_spout_z_lock` 이
    #   주둥이 z 를 `receiver 입구 z + margin` 으로 강제해 **상대 높이를 이미 묶고 있다**.
    #   실측: receiver 를 7.2cm 내리자 우팔도 z-lock 을 따라 같이 내려가(0.374→0.298)
    #   z 여유는 −0.006 → −0.026 으로 오히려 나빠졌다.
    #   게다가 켜면 왼팔이 오른팔 영역으로 내려와 **양손 최소거리 0.0mm · 좌컵 grip
    #   108mm 이탈 · dropped_by_force 50** 을 유발했다(E0-3 FAIL).
    #   기능 자체는 동작하므로(하강 7.2cm 확인) 코드는 남기되 기본은 끈다.
    # ⚠ 켤 때는 `left_tcp_z_down_m` 이 이 값 이상이어야 하고, 양팔 간격을 반드시 재볼 것.
    # ★hold 구간에 palm 목표를 warmstart pose 로 **고정**한다 (2026-08-18).
    #   구 코드는 `palm_action` 만 0 으로 만들고 목표는 live 계산이라, "가만히 있어야 할"
    #   hold 에서 팔이 25cm 밀리고 128 env 중 58 개가 죽었다(step 120 시점 컵 x 0.325→0.119).
    #   주석이 선언한 동작("warmstart pose 강제 유지")을 코드로 실제 구현한 것이다.
    #   False = 구 동작(회귀 확인용).
    hold_freeze_palm_target: bool = True
    # ★hold → 제어 전환을 **램프**로 넘긴다 (2026-08-18).
    #   실측: 전환 스텝(121)에 palm 목표가 **106.8mm** 튄다
    #     hold  (0.2872, -0.1734, 0.4743)  ← warmstart palm + z_boost(12cm)
    #     전환  (0.2607, -0.2139, 0.3979)  ← z-lock 이 주둥이를 입구+3cm 로 강제
    #   z 가 76mm 떨어지는 것이 지배적이다. `palm_ee` offset(48.8mm)의 2배가 넘으므로
    #   좌표 규약 문제가 아니라 **prelift 와 z-lock 이 서로 반대로 당기는** 것이다.
    #   (z_boost 를 0 으로 해도 점프 방향만 +44mm 로 바뀔 뿐 계단은 그대로다 — 실측에서
    #    z_boost 0/0.06/0.12 의 사망이 동일했던 이유.)
    #   그 계단 킥에 사망이 몰린다: hold 중(1~110) 사망 0, 111~160 에 51/128.
    #   N 스텝에 걸쳐 hold 목표 → live 목표로 선형 보간한다. 0 이면 구 동작(계단).
    control_handover_ramp_steps: int = 30
    receiver_lower_m: float = 0.0
    left_tcp_action_delta_m: float = 0.01
    left_tcp_workspace_range: tuple[float, float, float] = (0.08, 0.08, 0.08)
    # 왼팔 TCP z 하강 허용량[m].
    #
    # ★2026-08-18 pour_v1 에서 0.0 → 0.08. **구 제약의 근거가 사라졌다.**
    #   pour_sensor 에서 0 이었던 이유는 receiver 컵이 `kinematic-follow` 라 테이블과
    #   물리충돌이 없어, 하강을 허용하면 컵이 **테이블을 관통**했기 때문이다(s2r 무효).
    #   pour_v1 의 왼컵은 dynamic rigid body 를 왼손이 실제로 쥐고 있어 물리가 관통을 막는다.
    #   포팅 때 값만 따라와 근거 없는 제약으로 남아 있었다.
    #
    #   이게 왜 치명적이었나: pour_v1 은 왼손이 컵을 **들고** 있어 receiver 가
    #   pour_sensor 대비 **7.4cm 높다**(z 0.291 → 0.365). source 컵도 z 0.367 이라
    #   두 컵이 같은 높이에서 시작하고, 붓기는 원리상 source 가 위여야 성립한다.
    #   그런데 왼팔은 **내려갈 수가 없어서** 이 격차를 해소할 수단이 없었다.
    #   실측 결과: A-E1-frozen 이 2442 epoch 동안 완전 평탄
    #   (bead_in_target 0.000 · mouth_xy_dist 0.2255→0.2265 · mouth_z_clearance −0.009).
    #
    #   0.08 = workspace_range 와 동일. 필요한 하강은 약 6.5cm(→ pour_sensor 기하 복원)이고
    #   그 아래는 테이블이 물리적으로 막는다.
    left_tcp_z_down_m: float = 0.08
    # -------------------------------------------------------------------
    # ★[both/pour_v1] 왼컵 낙하 판정 — dynamic 전환으로 새로 생긴 실패 모드
    # -------------------------------------------------------------------
    # 왼컵이 kinematic 이 아니게 되면서 "왼손이 컵을 놓친다"가 실제로 가능해졌다.
    # 이를 종료로 잡지 않으면 컵을 떨어뜨린 뒤에도 에피소드가 계속되어, 정책이
    # 빈손으로 조준 보상만 긁는 경로가 열린다(보상 해킹).
    #
    # 접촉센서 대신 **기하**로 판정한다(왼손엔 센서를 안 붙였다 — obs 차원 불변 유지):
    #   ① 왼손 palm ↔ 컵 중심 거리가 리셋 시점 대비 `left_cup_drop_dist_m` 이상 벌어짐, 또는
    #   ② 컵이 리셋 시점보다 `left_cup_drop_z_m` 이상 낮아짐
    # 두 조건은 OR. hold 구간(episode_hold_steps)에는 물리 안착 중이라 판정하지 않는다.
    left_cup_drop_enable: bool = True
    left_cup_drop_dist_m: float = 0.06
    left_cup_drop_z_m: float = 0.08
    # 낙하 시 페널티(즉시 종료 + 음수 보상). source 컵 낙하 페널티와 같은 크기로 둔다.
    left_cup_drop_penalty: float = 5.0
    # [RA-L ablation] receiver(왼팔) 제어 모드 — M0/M2/M4 공정비교 + EXP-2 necessity.
    #   learned : 정책 action[12:15] 누적 (M4, 기본 — 기존 학습과 완전 동일)
    #   frozen  : 왼팔 TCP를 rest에 고정 (M0, EXP-2 freeze)
    #   scripted: source pour-point를 추종하는 기하 규칙 (M2, §3.1)
    #   실행별 주입: hydra override `env.receiver_control_mode=frozen` 등.
    receiver_control_mode: str = "learned"
    # [EXP-2 necessity, eval-time] learned action 스케일 축소 / 지연(step, 60Hz → 100ms≈6)
    receiver_action_scale: float = 1.0
    receiver_action_delay_steps: int = 0
    # [scripted 전용] pour-point 대비 receiver 목표 XY offset[m] / 하강 clearance[m]
    scripted_receiver_offset_xy: tuple[float, float] = (0.0, 0.0)
    scripted_receiver_clearance: float = 0.05
    # [Ablation #1 — joint-space action] task-space+Fabric 우수성 대조군.
    #   True면 우팔을 palm-pose/Fabric 대신 action[:7](palm6+null1 슬롯)을 관절 delta로 직접 구동
    #   (Fabric 우회). fabric_q[:7]을 덮어써 _apply_action이 그대로 사용. base=False(현행 task-space).
    #   demo nullspace prior는 Fabric-bound라 jointspace에선 무효(action space 기여 격리).
    #   별도 cfg: pour_right_env_cfg_ablation1_actionspace.py / hydra: env.right_arm_jointspace=true
    right_arm_jointspace: bool = False
    jointspace_action_scale: float = 0.03   # 관절 delta 스케일 [rad/step]
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
            # [DR] scale_set(MultiAsset, replicate_physics=False)에서 bead 초기 스폰 웨이브가
            #   filter 적용 후에도 최대 ~46M pairs 요구 (부족 시 접촉 miss → bead 관통 위험) → 64M.
            gpu_found_lost_pairs_capacity=64 * 1024 * 1024,
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
    # 로봇 설정
    # ★[both/pour_v1] openarm_tesollo_sensor_rl → openarm_tesollo_bi_s_rl 전환.
    #   구 자산은 오른손만 Tesollo 20관절이고 왼손은 2-DOF 그리퍼(l_hj_gripper_1/2)라
    #   왼손으로 컵을 파지할 수 없었다. 실기가 DG-5FS 양손이므로 좌우 모두 20관절인
    #   bi_s_rl 로 바꾼다 (좌우 각 arm 7 + hand 20, l_hl_*_tip/r_hl_*_tip 접촉링크 보유).
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_bi_s_rl/openarm_tesollo_bi_s_rl.usd"),
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
            # ═══════════════════════════════════════════════════════════════
            # ★[both/pour_v1] 액추에이터를 **grasp_v1 세팅으로 통일** (사용자 확정)
            # ═══════════════════════════════════════════════════════════════
            # 왜: pour_v1 은 좌/우 grasp_v1 이 만든 warm state 에서 에피소드를 시작한다.
            #   warm state 를 만든 물리와 재생하는 물리가 다르면 리셋 직후 파지가 어긋난다
            #   (left/grasp_v1 주석이 기록한 실패: 수집 게인 ≠ 소비 게인 → 컵 이탈·bead 유실).
            #   따라서 "수집 물리 = 소비 물리" 를 원칙으로 삼는다.
            #
            # 팔: grasp_v1 과 동일하게 proximal/elbow/wrist 3분할.
            #   · stiffness/damping 값은 기존 pour(400/80)과 같아 **거동 변화 없음**.
            #   · 분할하는 이유 두 가지 —
            #     ① `friction` 이 부위별 실측값(07.29 real2sim)이라 한 그룹으로는 못 넣는다.
            #        기존 pour_v1 에는 이 마찰이 아예 없었다(실물 마찰은 실재하므로 보존해야 한다).
            #     ② r2s_autotune 캘리브가 `{side}_arm_proximal/elbow/wrist` 그룹명을 기대한다.
            #        이름이 다르면 주입이 **조용히 fallback** 된다(문서화된 함정).
            #   · pour_v1 은 양팔이 모두 능동이므로 좌우 모두 3분할한다
            #     (grasp_v1 은 유휴 팔만 단일 그룹으로 뒀다).
            #   캘리브 원본: assets/robot/openarm_tesollo_sensor_rl/calibration/right_arm.json
            #     (kp 67.587/66.979/12.019, kd 6.376/5.635/2.154 — 좌우 동일 HW). 되돌릴 때 사용.
            "right_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-3]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.213,
            ),
            "right_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_4"],
                stiffness=400.0,
                damping=80.0,
                friction=0.493,
            ),
            "right_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[5-7]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.151,
            ),
            "left_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-3]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.213,
            ),
            "left_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_4"],
                stiffness=400.0,
                damping=80.0,
                friction=0.493,
            ),
            "left_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[5-7]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.151,
            ),
            # 손: **좌우 모두 5.0 / 2.0** (grasp_v1 의 "파지하는 손" 값).
            #   grasp_v1 은 파지 손만 5/2 로 두고 유휴 손은 400/60 으로 굳혔다. pour_v1 은
            #   양손이 실제로 컵을 쥐므로 둘 다 파지 손 값을 쓴다.
            #
            #   ⚠ 이 변경은 구 pour 튜닝(200~400 / 35~60)을 **의도적으로 폐기**한다.
            #     구 pour 주석의 근거는 "100=deep tilt OK/slip, 800=grip OK/tilt막힘 → 절충 400"
            #     이었으나, grasp_v1 의 S1~S4 실측이 그 전제를 무너뜨렸다:
            #       · URDF effort limit 7.5 N·m 인데 접촉 요구 토크가 143 N·m(19배)
            #       · k=400 에서 접촉 관절 80.6% 상시 포화 → 위치 제어가 아닌 bang-bang 그리퍼
            #         (목표를 더 밀어도 힘이 안 올라감 = 파지력 제어가 원리적으로 불가)
            #       · 실기는 effort 명령 + P게인 1.5 라 이 영역에 도달 불가 → sim2real 불성립
            #     즉 "400" 은 실효 게인이 아니라 포화 상태였고, 그래서 tilt/grip 절충이라는
            #     해석 자체가 포화 위 관측이었다.
            #     S1 포화 이탈선 k≈5, S4 damping 스윕 kd=2.0(포화 0.8% + 최저 채터).
            #     S2 실측: 게인을 80배 낮춰도 이탈가속 내성은 37→27 m/s² 로만 감소했다
            #     (외란 내성은 파지력보다 **기하**가 지배).
            #   ★예상 지표 이동: 손 접촉력↓·토크 포화율 대폭↓, deep tilt 는 완충 여지가 늘어
            #     tilt_frac 유지 또는 개선. 대신 파지 마진이 줄어 `grasp_broken`/`left_cup_dropped`
            #     상승 가능 — Phase 1 게이트에서 이 두 지표를 먼저 확인할 것.
            #   실기 d=0.0 이므로 이 damping 은 기계마찰의 sim 대역품 — r2s 복구 후
            #     armature/joint friction 실측치로 교체할 것.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_1"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_2"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_3"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_4"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_left_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_1"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_left_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_2"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_left_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_3"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_left_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_4"],
                stiffness=5.0,
                damping=2.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # Actor: fingertip 5개 개별 센서 (Cup-only, real-compatible FT sensor)
    # Critic: distal + middle 통합 센서 (sim-only)
    # -----------------------------------------------------------------------
    # ★[both/pour_v1] 왼손(l_hl_*_tip)에는 접촉센서를 **의도적으로 붙이지 않는다.**
    #   왼손은 warm 파지자세 position-hold 만 하고 action/obs 에 들어가지 않으므로 촉각 관측이
    #   불필요하다. 센서를 추가하면 obs 차원이 바뀌어(=재학습 강제) action 15D 유지 결정과 충돌한다.
    #   왼컵 낙하 판정은 접촉이 아니라 **기하**(왼손 body ↔ 컵 거리 + 컵 z 낙하)로 한다
    #   — `left_cup_drop_*` 파라미터 참조.
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            # ★[both/pour_v1] receiver 컵을 **kinematic → dynamic** 으로 바꾼다.
            #   구 pour_sensor: kinematic_enabled=True + disable_gravity=True 로, 컵은 중력도
            #     충돌 반력도 받지 않고 왼손 pose 를 그대로 따라갔다(= 물리 파지가 아님).
            #   pour_v1: 왼손이 실제로 쥐어야 하므로 중력·반력을 받는 보통 강체로 만든다.
            #   solver iteration 은 source 컵과 동일하게 맞춘다(비드 충돌 관통 방지).
            #   질량은 source 컵과 같이 USD density 파생값(≈0.134kg)을 쓴다.
            #   ⚠ 이 전환으로 **왼컵 낙하**라는 새 실패 모드가 생긴다 → `left_cup_drop_*` 종료 조건 참조.
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            # [07.02 reward-hacking fix] 음수 offset(-0.1)은 target cup을 유령(충돌無)으로 만들어
            #   source cup 관통·겹침 이중카운트(bead_in_source+target=1.4)를 허용했다. 정상값으로 복원해
            #   SDF 충돌 활성화: source 몸체 관통 불가 + bead가 림 위로 넘어와야 진입 + target이 실제 담음.
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
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
