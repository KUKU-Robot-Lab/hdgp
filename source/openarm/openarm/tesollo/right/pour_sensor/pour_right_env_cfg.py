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
from isaaclab.envs import mdp as _mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
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


# grasp_s2r `GraspS2REventCfg` 이식 — **warm 뱅크 생산자와 같은 재질·게인 계약**.
# ★이게 없으면 로봇/컵의 재질은 USD 값 그대로다. `sim.physics_material` 은 **자체 재질이
#   없는 프림에만** 적용되므로 대체가 안 된다. 생산자는 매 리셋마다 마찰을 1.0 으로
#   덮어쓴다 — 손가락↔컵 마찰의 진짜 출처가 여기다.
# ★재질 term 은 **절대값**, 관절/질량 term 은 `operation="scale"` 이라 배율이다.
#   공칭값에서는 전부 항등이며, DR 을 켤 때 여기가 확장 지점이다.
_FRICTION = 1.0


@configclass
class PourEventCfg:
    """도메인 랜덤화 — 전 term `mode="reset"`, 공칭 파라미터에서는 항등."""

    robot_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # ★pour 의 소스 컵 씬 엔티티 이름은 `cup` 이다(grasp_s2r 은 `object`).
    cup_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_friction = EventTermCfg(
        func=_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


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
    fabrics_damping_gain:       float = 10.0  # [09.01] 20→10: 생산자(grasp_s2r) 값. 팔 제어 감쇠가 다르면 warm 자세 유지 거동이 달라진다  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소
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

    # Fabrics params 파일 재정의. None = 공유 기본값(`openarm_tesollo_pose_params.yaml`).
    #   ★공유 yaml 은 2026-08-23 `cbff1ce` 에서 충돌구 프레임이 구 `*_sphere2` → 신 `*_sph1`
    #     으로 바뀌었는데, 신 링크는 a2(`openarm_tesollo_bi_s.urdf`) 에만 들어갔다. 이 트랙은
    #     a1(`openarm_tesollo.urdf`) 을 읽으므로 그대로 두면 env 생성이 KeyError 로 죽는다.
    #   a1 체크포인트를 **학습 당시 fabric 으로** 재생하려면
    #     `openarm_tesollo_pose_params_pre0823.yaml` 을 지정한다.
    #   ★b1 자산 정렬(09.01): grasp_s2r 과 같은 `openarm_tesollo_sensor_right` URDF +
    #     그 전용 params 로 맞춘다. 프레임 65개가 그 URDF 에 전수 존재하고 cspace 27 로
    #     팔7+손20 과 일치함을 확인했다. 구 체크포인트 재생은 worktree(07e61fb)로 하므로
    #     이 기본값 변경이 폴백을 깨지 않는다.
    fabric_params_filename: str | None = "openarm_tesollo_sensor_pose_params.yaml"
    # Fabrics URDF 디렉토리 재정의. None = 클래스 기본값(`openarm_tesollo`).
    #   ★b1(grasp_s2r) 자산 정렬: 그 트랙은 `openarm_tesollo_sensor_right` 를 쓴다 —
    #     "FK 게이트 0.0um 로 sensor_rl 에서 재생성한 자산(08.22). 레거시 openarm_tesollo /
    #      openarm_tesollo_sensor 는 같은 DG-5F 손이지만 팔 베이스가 +8mm 어긋나
    #      RL URDF 대비 worst 17.93mm"(robot_profiles.py). warm 뱅크를 b1 에서 받으려면
    #     같은 URDF 여야 한다.
    fabric_robot_dir: str | None = "openarm_tesollo_sensor_right"
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


    # ------------------------------------------------------------------
    # bead / cup geometry — **전부 물체 뱅크 스펙에서 파생**한다. 손으로 적지 말 것.
    # ------------------------------------------------------------------
    # ★여기 있던 상수 8개는 cup_big(scale 1.0) 한 종에 고정돼 있었다. 받는 컵을
    #   shaker 로 바꾸고 붓는 컵을 8종으로 늘리자 전부 어긋났고, 특히 치명적이었던 것:
    #     `target_inside_z_min = -0.070` vs shaker 바닥 -0.0921
    #       → 컵 안에 가라앉은 bead(중심 -0.082)가 "안"에서 빠지고, spill 식
    #         (`~in_source & z < target_inside_z_min`)에 걸려 **성공이 손실로 집계**된다.
    #   그래서 아래 값들은 `finalize_after_overrides()` 가 스펙에서 다시 채운다.
    #   target_* = 받는 컵(`left_target_cup_spec`) 단일 스칼라.
    #   source_* = 붓는 컵이 다종이라 여기 값은 **폴백**이고, env 가 env 별 텐서를 만든다.
    left_target_cup_spec: str = "shaker_closed"   # 받는 컵 = 뱅크 스펙 id (자산+기하 동시 파생)
    target_inner_radius:  float = 0.0432  # 파생: spec.inner_radius_m
    target_inside_z_min:  float = -0.0848 # 파생: spec.inside_z_min
    target_inside_z_max:  float = 0.0829  # 파생: spec.rim_z
    target_mouth_z:       float = 0.0829  # 파생: spec.rim_z (bead crossing 기준면)
    source_inner_radius:  float = 0.0409  # 폴백(cup_big s100). 실사용은 env 의 env 별 텐서
    source_outer_radius:  float = 0.0467  # 폴백(cup_big s100)
    source_inside_z_min:  float = -0.0700 # 폴백(cup_big s100)
    source_inside_z_max:  float = 0.1003  # 폴백(cup_big s100)
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
    # 리셋 시 palm target z 를 이만큼 올린다 — warm 상태가 **아직 안 든 자세**일 때만 필요하다.
    #
    # ★0.12 였던 이유(구 뱅크): 테이블 위에서 막 잡힌 자세(cup_z ≈ 0.327m)로 시작해서,
    #   hold phase 동안 Fabrics 가 팔을 올려 cup_z ≈ 0.447m 를 만들어야 붓기가 가능했다.
    # ★0.0 으로 내린 이유(b1 뱅크, 09.01): 수집기가 **lifted ∧ still ∧ grip** 에서만 담아
    #   warm 상태 자체가 이미 든 자세다 — 실측 cup_z 0.3996~0.4684 (평균 0.4435) 로
    #   구 boost 의 도착점(0.447)에 이미 도달해 있다. 여기서 또 0.12 를 더하면 palm 목표가
    #   workspace z_max(0.68) 를 넘어 **12cm 클램프**되고, 그만큼 palm 목표가 실제 팔
    #   자세에서 떨어진다(= Fabrics 가 리셋 직후 팔을 끌어올린다). 이중 리프트다.
    #   뱅크를 "안 든 자세" 로 되돌리면 이 값도 같이 되돌려야 한다.
    warmstart_palm_z_boost: float = 0.0
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

    # ADR: success 기준 커리큘럼 (fill_ratio: 낮은 기준→높은 기준)
    # bead 10개 기준: 0.20=2개, 0.30=3개, 0.40=4개, 0.50=5개
    # 해당 기준에서 success_rate >= 15%이면 한 단계 올림 (8단계 × 0.0375 = 0.30 range)
    enable_success_adr: bool = True
    success_adr_custom_cfg: dict = {
        "success": {
            "fill_ratio": (0.20, 0.50),  # 2개→5개 커리큘럼
            # [ADR 조준개선] r_aim 급경사도(pour_aim_scale)를 단계 상승 → 주둥이를 입구 중심으로 점진 유도.
            #   10(완만) → 15(급경사). scale 20 진동 이력 → 15 상한. fill_ratio와 동일 성공 trigger.
            "aim_scale": (10.0, 15.0),
        }
    }
    success_adr_num_increments: int = 8
    # maybe_increment는 env-step당 1회 호출됨. 20000이면 첫 체크가 ~40M프레임(env-step 20000)에서야 발생 →
    # 실질 run 안에서 aim/fill ADR 미발동. 2000으로 낮춰 ~4M프레임마다 1단계 체크 → 8단계 ≈ 32M프레임에 완주.
    success_adr_increment_interval: int = 2000
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
    # ★생산자(grasp_s2r `sens_r` 프로필)의 `object_spawn_center=(0.362, -0.16)` 와 일치.
    #   warm 상태는 그 중심에서 스폰된 컵을 잡은 자세다 — 중심이 다르면 fresh 폴백과
    #   warm 복원이 서로 다른 곳을 가리킨다(09.01: 구 값 0.27/-0.10, x 9cm·y 6cm 차이).
    object_spawn_x_center: float = 0.362
    object_spawn_y_center: float = -0.16
    # ★`finalize_after_overrides()` 파생값이다 — 여기 리터럴을 고치지 말 것.
    #   warm 뱅크 로더가 허용오차 1e-4 로 **하드 대조**하므로, 수집 트랙(grasp_s2r)과
    #   **같은 식**으로 뽑아야 자동으로 맞는다:
    #     table_surface_z + max(spec.origin_offset_z) + object_spawn_pad
    #   cup_family 면 s130 의 0.0773×1.30=0.10049 가 최댓값 → 0.30549 (b1 뱅크와 일치).
    object_spawn_z:        float = 0.0
    table_surface_z:       float = 0.2      # 테이블 상면(= table_cfg.init_state.pos[2])
    object_spawn_pad:      float = 0.005
    #   뱅크에서 파생. 다물체면 **최댓값**이다 — 작은 값을 쓰면 큰 컵이 테이블을 뚫는다.
    object_origin_offset_z: float = 0.0773
    # ★물체 뱅크. b1(grasp_s2r)이 `cup_family` 8종으로 학습했고 warm 뱅크도 거기서 나왔다.
    #   상태마다 잡은 컵 크기가 달라(s085 45.8mm ↔ s130 61.9mm) 단일 컵으로 복원하면
    #   손 개구와 컵이 안 맞는다 — 다물체가 선택이 아니라 뱅크의 전제다.
    object_bank: str = "cup_family"
    #   접촉 필터. ★루트 Xform 을 가리키면 `force_matrix_w` 가 **항상 0** 이다
    #   (grasp_s2r 실측 함정 ②) — 뱅크의 rigid_body_name 으로 파생한다.
    object_contact_filter: tuple[str, ...] = ("/World/envs/env_.*/Cup",)
    object_spawn_xy_range: float = 0.06   # ±6cm 랜덤화 (Fabrics arm 학습으로 보정 가능)

    # -----------------------------------------------------------------------
    # Warmstart reset cache
    # -----------------------------------------------------------------------
    enable_warmstart_reset: bool = True
    # ★08.18 기본값 제거: 종전 경로(log/rl_games/pipeline/.../5g_grasp_right-v7-2.pth)는
    #   디스크 어디에도 존재하지 않는 dangling 참조였다(구 rl-USD 이전 산출물).
    #   rollout 소스를 쓰려면 실존하는 체크포인트를 명시적으로 지정하라.
    warmstart_checkpoint_path: str = ""
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
    #   "disk"   : grasp 가 디스크에 저장한 캐시 로드 (권장).
    #              startup 시 grasp policy rollout 불필요 → 분포/포맷 불일치 제거.
    #   "rollout": (레거시) startup 에서 지정 체크포인트를 pour env 안에서 rollout 해
    #              캐시 수집 — warmstart_checkpoint_path 명시 필수(기본값 없음).
    #   "preset" : 캐시 없이 preset/pregrasp 합성 시작 (디버그용).
    # ★08.18 fail-loud: disk 로드 실패 시 rollout 으로 조용히 degrade 하지 않고
    #   즉시 에러다(env._build_warmstart_reset_cache) — s2r 트랙은 뱅크가 계약이다.
    warm_state_source: str = "disk"
    # ★08.18 pour_sensor(a1) 재배선: right/grasp_sensor(openarm_tesollo_sensor_rl)
    #   산출물 전용. 수집 = collect_grasp_v1_warm_states.py --robot tesollo_sensor.
    #   bi_s_rl(DG-5FS) 계열 grasp_warm_tesollo*.hdf5 는 텐서 차원이 같아 조용히
    #   로드되므로 파일명부터 분리했고, 로더의 meta/robot_usd 하드 가드가 2차 방어한다.
    #   ⚠ 스폰 기하 주의: grasp_sensor 는 y_center -0.20 에서 스폰하지만 리프트 스윙
    #   (joint7 0.31rad)이 컵을 몸쪽으로 당겨 warm 컵 y 는 ≈-0.085 로 이동한다
    #   (bimanual 8000쌍 실측, 우팔 기준). object_spawn_*(fresh 폴백)와 target 기하는
    #   뱅크 수집 후 **실측으로** 재튜닝한다 — 추정으로 미리 바꾸지 않는다.
    warm_state_paths: tuple[str, ...] = (
        # ★뱅크를 바꾸면 **손 액추에이터 게인도 같이** 확인할 것 — warm 상태는 그것을
        #   만든 게인 아래서만 재현된다(09.01). d3 = grasp_s2r s2r_d3_liftonly_fresh_v2
        #   ep20000 (md5 485d8abf, k=5 · d=2 · effort 1.5 — 위 손 액추에이터와 일치).
        #   n2048_maxgrip = 에피소드 **최대 접촉** 프레임만 골라 모은 판. pour 은 붓는
        #   동안 손가락을 freeze 하므로 뱅크의 파지 품질이 그대로 붓기 내내의 상한이 된다.
        #   ★09.01 실측(d3 뱅크로 학습한 pour, 컵당 64~75 에피소드):
        #     corr(bead_at_done, 뱅크 ≥4지 비율) = **+0.935** — 컵별 이송률을 파지가
        #     거의 그대로 설명한다. 컵을 놓쳐서가 아니다(grasp_broken 0, 에피소드 중
        #     접촉 낙폭도 약한 컵이 더 작다) — 처음부터 손가락이 덜 닿은 채로 끝까지 간다.
        #   그래서 뱅크 교체가 pour 성능의 직접 지렛대다. e1 = e1_perc 최종
        #   (md5 fc48c5cc, 손 게인 d3 와 동일 k=5/d=2/effort 1.5):
        #     ≥4지 79.3%→94.0% · 5지 42%→55% · 엄지 접촉 46%→58% · 엄지 팁힘 0.26→1.33N
        #     최약 컵 40%(s100)→81%(s090)
        _os.path.normpath(_os.path.join(_HDGP_ROOT, "data", "grasp_warm_s2r_e1_n2048_maxgrip.hdf5")),
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
    events: PourEventCfg = PourEventCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # ★★warm 뱅크 생산자(grasp_s2r)와 **같은 기본 마찰**. 안 적으면 IsaacLab 기본
        #   static/dynamic 0.5 가 조용히 들어가고, 생산자는 1.0 이라 **파지 마찰이 절반**이
        #   된다. 09.01 실측: 제로 액션 240 스텝에서 8종 전부 컵이 손에서 빠져 테이블로
        #   떨어졌다(접촉 0개·드리프트 34~118°). 지령·게인을 다 맞춰도 이것 하나로 무너진다.
        #   같은 함정이 grasp_v1 에서도 났다 — "cfg 에 안 적은 물리는 조용한 기본값"이다.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            # [09.01] 0.05→0.2: 생산자(grasp_s2r) 값으로 복귀. 비드 반발을 위해 낮췄던
            #   값인데, 파지 접촉의 반발 거동까지 같이 바꿔 warm 재현을 흔든다.
            bounce_threshold_velocity=0.2,
            # [09.01] 4M→16M. 다물체(replicate_physics=False)는 env 간 페어를
            #   브로드페이즈 **뒤에** 거르므로 같은 씬이라도 페어 사용량이 크다.
            gpu_found_lost_pairs_capacity=16 * 1024 * 1024,
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
    # 로봇 설정 (openarm_tesollo_sensor_rl.usd: r_hl_*_tip ContactSensor 포함, rl 통일 네이밍)
    # [rl USD 마이그레이션] 구 assets/openarm_tesollo_sensor/ → 신 assets/robot/openarm_tesollo_sensor_rl/
    #   조인트 r_aj_/r_hj_, 링크 r_hl_ (pour_sensor/rh56f1과 동일 최신 네이밍).
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_sensor_rl_hull/openarm_tesollo_sensor_rl.usd"),
            activate_contact_sensors=True,
            # ★★로봇 물리 프로퍼티는 **warm 뱅크 생산자(grasp_s2r)와 동일**해야 한다.
            #   09.01 실효 cfg 대조로 드러난 차이(왼쪽=생산자 d3 / 오른쪽=구 pour):
            #     enabled_self_collisions True/False · depenetration 1000/5 ·
            #     solver pos·vel 8·0 / 16·1 · retain_accelerations True/미설정
            #   self-collision 은 손가락끼리 못 겹치게 하므로 **같은 지령에서 다른 파지
            #   평형**이 나온다. 뱅크는 True 아래서 만들어졌다.
            #   ※구 메모(다물체는 self-collision OFF 필수)는 손 hull USD 이전 이야기다 —
            #     d3 는 같은 hull USD 로 True·replicate_physics=False 조합에서 정상 학습됐다.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ★★여기만 생산자(True)와 다르게 둔다 — 의도적 예외다.
                #   grasp_s2r 은 손을 **벌린 채 시작해 서서히 닫는다**. pour 은 warm 상태로
                #   **이미 닫힌 손에 텔레포트**한다. 그 순간 손가락 hull 이 미세하게 겹치면
                #   self-collision 이 그것을 폭발적으로 밀어내 컵이 튕겨 나간다
                #   (09.01 실측: 팁힘 167 N — 게인 5·effort 1.5 로는 못 내는 값 = 관통 반력).
                #   같은 현상이 grasp_s2r 다물체 기동에서도 났다([[repfalse-selfcollision-explosion]]).
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            # ★★생산자와 같은 드라이브 타입. 미설정이면 USD/IsaacLab 기본(`acceleration`)이
            #   들어가는데, 그 모드는 PhysX 가 게인에 **관성을 곱한다** — 같은
            #   stiffness=5 가 전혀 다른 토크를 낸다. 게인·마찰을 다 맞춰도 이것 하나로
            #   파지가 어긋난다. warm 뱅크는 `force` 아래서 만들어졌다.
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
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
                "r_hj_thumb_1":  0.0, "r_hj_thumb_2":  -1.57, "r_hj_thumb_3":  -0.5, "r_hj_thumb_4":  0.0,
                "r_hj_index_1":  0.0, "r_hj_index_2":   0.0,  "r_hj_index_3":   0.0, "r_hj_index_4":  0.0,
                "r_hj_middle_1": 0.0, "r_hj_middle_2":  0.0,  "r_hj_middle_3":  0.0, "r_hj_middle_4": 0.0,
                "r_hj_ring_1":   0.0, "r_hj_ring_2":    0.0,  "r_hj_ring_3":    0.0, "r_hj_ring_4":   0.0,
                "r_hj_pinky_1":  0.0, "r_hj_pinky_2":   0.0,  "r_hj_pinky_3":   0.0, "r_hj_pinky_4":  0.0,
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
            # ★★우팔은 **warm 뱅크 생산자(grasp_s2r)의 per-joint 계약**을 그대로 쓴다.
            #   09.01 실측: 구 값(전 관절 400/80)은 손목이 생산자보다 j5 4배·j6 8배·
            #   j7 16배 뻣뻣했다. warm 상태는 "그 강성 아래서 컵 무게에 손목이 이만큼
            #   내준" 평형이므로, 더 뻣뻣한 팔에 그 자세를 넣으면 평형이 아니라 **초기
            #   응력**이 되어 컵이 손에서 튕겨 나간다(제로액션 240스텝 드리프트 171°).
            #   ⚠뱅크를 다른 게인으로 수집한 것으로 바꾸면 여기도 같이 바꿔야 한다.
            "openarm_right_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-4]"],
                stiffness=300.0,
                damping=45.0,
                effort_limit_sim=300.0,
            ),
            "openarm_right_wrist_5": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_5"],
                stiffness=100.0,
                damping=20.0,
                effort_limit_sim=300.0,
            ),
            "openarm_right_wrist_6": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_6"],
                stiffness=50.0,
                damping=15.0,
                effort_limit_sim=300.0,
            ),
            "openarm_right_wrist_7": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_7"],
                stiffness=25.0,
                damping=15.0,
                effort_limit_sim=300.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=2000.0,   # 400→2000: 오른팔 충돌 저항 강화
                damping=200.0,
            ),
            # ★★손 게인은 **warm 뱅크 생산자와 같아야** 한다. 09.01 실측:
            #   생산자(grasp_s2r b1/d3) k=5 · d=2 · effort_limit_sim=1.5 N·m,
            #   소비자(여기) k=200~400 — **80배**. warm 상태는 "지령과 측정이 크게 벌어진 채
            #   토크 상한으로 누르는" 상태라, 그 지령을 80배 센 스프링에 그대로 주면
            #   0.69 rad × 400 = 276 N·m 로 컵이 터지고, 측정을 주면 오차 0 → 파지력 0 이다.
            #   [09.01 실측] 생산자 게인(k=5·effort 1.5)을 그대로 채택하면 pour 에서
            #   컵이 즉시 빠진다(제로액션 240스텝: 접촉 0.38개·드리프트 155°). 지령은
            #   정상 로드됨을 확인했으므로(gap 0.18 rad) 원인은 게인 쪽 — pour 는 손을
            #   **전 구간 freeze** 하므로 미끄러짐을 되잡을 수 없고, 물렁한 스프링으로는
            #   버티지 못한다. grasp 는 매 스텝 다시 조여 버틴다. → pour 는 제 게인을 쓴다.
            #   구 값(400)은 t27/t29 에서 800 을 낮춘 절충이었다(딱딱하면 deep tilt 시 target
            #   충돌 반력이 팔로 전달돼 tilt 경직: frac_110 0.4→0, pose_succ 0.95→0.2).
            #   k=5 는 그보다 **더 물러서** 그 실패 방향엔 유리하다.
            #   ⚠뱅크를 다른 게인으로 수집한 것으로 바꾸면 여기도 같이 바꿔야 한다.
            # ★★손도 생산자 계약 그대로 — **그룹 분할까지** 같게 둔다(생산자는 한 그룹).
            #   파지력은 스프링이 아니라 effort 상한 1.5 N·m 가 낸다. warm 상태는
            #   "지령과 측정이 크게 벌어진 채 그 상한으로 누르는" 평형이다.
            "tesollo_hand": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                stiffness=5.0,
                damping=2.0,
                effort_limit_sim=1.5,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_gripper_[1-2]"],
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
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_rl.usd"),
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
            usd_path=_os.path.join(_ASSETS_DIR, "cup/shaker_closed_rl.usd"),
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

    # ------------------------------------------------------------------
    # 파생 구조 (물체 뱅크 · 스폰고) — grasp_s2r `finalize_after_overrides` 이식
    # ------------------------------------------------------------------
    def __post_init__(self):
        self.finalize_after_overrides()

    def finalize_after_overrides(self) -> None:
        """cfg 필드에서 **파생되는 구조**를 다시 만든다. 멱등이어야 한다.

        ★★hydra 는 `__post_init__` **뒤에** `from_dict(...)` 로 오버라이드를 적용하고
          `__post_init__` 를 다시 부르지 않는다. 그래서 `env.object_bank=...` 는 파생
          구조(스폰 cfg·replicate_physics·접촉필터·스폰고)에 반영되지 않는다.
          env `__init__` 이 `super()` **전에** 이걸 다시 부른다 — `replicate_physics` 는
          `InteractiveScene.__init__` 이 소비하므로 `_setup_scene` 은 이미 늦다
          (grasp_s2r 08.29 실측).
        """
        self._apply_object_bank()
        self._apply_target_cup_spec()
        # 스폰 높이는 여기 한 곳에서만 파생한다(이중 패딩 사고 차단).
        self.object_spawn_z = (
            self.table_surface_z + self.object_origin_offset_z + self.object_spawn_pad)
        self.cup_cfg.init_state.pos = [
            self.object_spawn_x_center, self.object_spawn_y_center, self.object_spawn_z,
        ]

    def _apply_target_cup_spec(self) -> None:
        """받는 컵의 **자산과 기하를 같은 스펙에서** 동시에 파생한다. 멱등.

        ★자산 경로만 바꾸고 기하 상수를 안 고치는 드리프트를 구조적으로 막는다.
          09.01 에 실제로 났다: 받는 컵을 cup_big→shaker_closed 로 바꿨는데 입구 z 가
          0.1003 인 채였다(17.4mm 위 허공 조준). 이제 둘 다 스펙에서 나온다.
        """
        from openarm.agnostic.modules import object_bank as _ob

        spec = _ob.spec_by_id(self.left_target_cup_spec)
        if not _os.path.isfile(spec.usd_path):
            raise RuntimeError(f"받는 컵 USD 누락: {spec.usd_path}")
        self.left_target_cup_cfg.spawn.usd_path = spec.usd_path
        self.left_target_cup_cfg.spawn.scale = tuple(spec.scale)
        self.target_inner_radius = spec.inner_radius_m
        self.target_inside_z_min = spec.inside_z_min
        self.target_inside_z_max = spec.rim_z
        self.target_mouth_z = spec.rim_z
        self.target_cup_opening_pos_b = (0.0, 0.0, spec.rim_z)

    def _apply_object_bank(self) -> None:
        """뱅크 크기에 따라 스폰·물리복제·접촉필터를 한 곳에서 조립한다.

        grasp_s2r `_apply_object_bank()` 이식. 그 트랙이 막는 함정 3개를 그대로 막는다:
          ①뱅크>1 인데 `replicate_physics=True` 면 전 env 가 같은 물체를 받는다.
          ②접촉 필터가 루트 Xform 이면 `force_matrix_w` 가 항상 0 이다.
          ③`base_origin_offset_z` 미측정 스펙이 섞이면 안착 높이를 못 구한다(fail-loud).
        """
        from openarm.agnostic.modules import object_bank as _ob

        # ★멱등성 — `__post_init__` 과 env `__init__` 에서 **두 번** 불린다. 원본 단일
        #   스폰을 보존하지 않으면 두 번째 호출이 MultiAsset 을 또 감싼다.
        if getattr(self, "_object_spawn_base", None) is None:
            self._object_spawn_base = self.cup_cfg.spawn
            self._table_usd_base = self.table_cfg.spawn.usd_path

        bank = _ob.get(self.object_bank)
        _missing = bank.missing_files()
        if _missing:
            raise RuntimeError(f"물체 뱅크 '{bank.name}' 의 USD 누락: {list(_missing)}")
        self.object_origin_offset_z = max(sp.origin_offset_z for sp in bank.specs)
        self.object_contact_filter = (f"/World/envs/env_.*/Cup/{bank.rigid_body_name}",)

        if not bank.needs_multi_asset:
            self.cup_cfg.spawn = self._object_spawn_base
            self.table_cfg.spawn.usd_path = self._table_usd_base
            return

        from dataclasses import replace as _dc_replace

        from isaaclab.sim.spawners.wrappers import wrappers_cfg as _wrap

        self.scene.replicate_physics = False
        # ★★다물체는 `replicate_physics=False` 필수이고, 그때 `clone_environments` 의
        #   env 간 충돌 격리가 사라진다. 작업면이 원시 정적 프림이면 전 env 가 한 충돌
        #   그룹에 남아 팔이 물린다(08.29 실측 abnormal 0.849 · joint_err 0.74 rad).
        #   pour 는 테이블이 이미 RigidObject 라 USD 만 kinematic 사본으로 바꾸면 된다.
        _rigid_usd = _os.path.join(_ASSETS_DIR, "env", "usd", "env_rigid.usd")
        if not _os.path.isfile(_rigid_usd):
            raise RuntimeError(
                f"다물체({bank.name})는 kinematic 작업면이 필요하다: {_rigid_usd} 없음 — "
                "`python3 scripts/assets_tools/build_env_rigid_usd.py` 로 먼저 빌드할 것")
        self.table_cfg.spawn.usd_path = _rigid_usd
        # ★★USD 만 바꾸고 **위치를 그대로 두면 작업면이 통째로 어긋난다** — 09.01 실측 사고.
        #   `scene_objects/table.usd` 는 [0.5725, 0.003, 0.2] 에 놓도록 만든 소품이고,
        #   `env/usd/env_rigid.usd` 는 **원점에 놓도록 저작된 씬 전체**다(pxr 실측:
        #   상면 z=+0.200 · x[-0.75,+0.47] · y[-0.45,+0.45], env.usd 와 정점까지 동일).
        #   위치를 안 바꾸면 상면이 z=0.4 로 20cm 뜨고 x 가 57cm 밀려, `table_surface_z=0.2`
        #   기준으로 계산한 스폰고·warm 상태와 전부 어긋난다(생산자 grasp_s2r 은 원점 배치).
        self.table_cfg.init_state.pos = [0.0, 0.0, 0.0]
        self.table_cfg.init_state.rot = [1.0, 0.0, 0.0, 0.0]
        _base = self._object_spawn_base
        self.cup_cfg.spawn = _wrap.MultiAssetSpawnerCfg(
            assets_cfg=[
                _dc_replace(_base, usd_path=sp.usd_path, scale=tuple(sp.scale),
                            mass_props=sim_utils.MassPropertiesCfg(mass=float(sp.mass)))
                for sp in bank.specs
            ],
            random_choice=False,      # env_id % N — `assign_indices` 와 같은 규약
            activate_contact_sensors=True,
        )
