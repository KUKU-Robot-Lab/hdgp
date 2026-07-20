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

"""환경 설정: tesollo grasp_v2 — DEXTRAH 구조 (다물체 파지→goal 운반)

- Action: 16D (6D palm pose Fabrics IK + 5D 시너지 + 5D abduction/opposition)
- Observation: DEXTRAH teacher 구조 — policy 198+N_obj / critic 252+N_obj
- Reward: DEXTRAH 4항 + ADR reward 스케줄 (lift 5→0)
- Goal: 고정 절대점 (object_goal_pos), success = |obj-goal| < tol
- ADR: wrench/spawn/노이즈/reward 커리큘럼 (in_success > 0.4 트리거)
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from openarm.distillation.camera import depth_randomization_cfg, make_cam_matrix
from .grasp_right_constants import (
    NUM_OBS_BASE, NUM_ACTIONS, NUM_CRITIC_OBS_BASE, NUM_STUDENT_OBS,
)
from .grasp_right_preset import (
    CAMERA_CLIPPING_RANGE,
    CAMERA_D_MAX,
    CAMERA_D_MIN,
    CAMERA_CROP_FRAC,
    CAMERA_FOCAL_LENGTH,
    CAMERA_HORIZONTAL_APERTURE,
    CAMERA_IMG_HEIGHT,
    CAMERA_IMG_WIDTH,
    CAMERA_POS,
    CAMERA_ROT,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    SIDE_APPROACH_OBJECT_NAMES,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_TEXTURE_ROOT = _os.path.join(_ASSETS_DIR, "dextrah_textures")

_CAM_MATRIX = make_cam_matrix(
    CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT,
    CAMERA_FOCAL_LENGTH, CAMERA_HORIZONTAL_APERTURE,
)


# ---------------------------------------------------------------------------
# grasp_v2 파지 대상 물체 (다물체): primitives
# ---------------------------------------------------------------------------
# 이름 규칙(STL bbox 측정): 숫자=형태, 5=세로 긴 기둥 / 8=정육면체 / 12=납작 원반.
# large = small 의 2배. 파지 폭(XY): large 5/8/12 = 5/8/12cm, small = 2.5/4/6cm.
#
# 커리큘럼(대표값 고정 방식): cup 크기 상수 cup_radius_approx=0.045(폭 9cm) 는 그대로 두고,
# spawn 물체군만 좁혀 대표값과 맞춘다. cup_big 실측 = 폭 9cm / 높이 17.8cm 세로 원통.
#   - STAGE1: large 계열(폭 5~12cm, 반경 2.5~6cm, 중앙 4.5cm = cup radius). 원통 3종 포함.
#   - small 계열(폭 2.5~6cm)은 radius 대표값보다 확실히 작아 STAGE2 로 확장.
# 물체군 확장은 _ACTIVE_OBJECT_NAMES 를 바꾸면 됨(reward 상수 불변).
_PRIMITIVE_CURRICULUM_STAGE1: tuple[str, ...] = (
    "large_5_cyl", "large_8_cyl", "large_12_cyl",
    "large_5_cuboid", "large_8_cuboid", "large_12_cuboid",
)
_PRIMITIVE_ALL: tuple[str, ...] = _PRIMITIVE_CURRICULUM_STAGE1 + (
    "small_5_cyl", "small_8_cyl", "small_12_cyl",
    "small_5_cuboid", "small_8_cuboid", "small_12_cuboid",
)

# 현재 활성 물체군(초기 학습 = STAGE1).
# visdex 실물 뱅크(접근 B): 디렉토리 스캔. code = "visdex:<name>".
_VISDEX_ROOT = _os.path.join(_ASSETS_DIR, "visdex_objects", "USD")
_VISDEX_NAMES: tuple[str, ...] = tuple(sorted(
    _n for _n in _os.listdir(_VISDEX_ROOT)
    if _os.path.isfile(_os.path.join(_VISDEX_ROOT, _n, f"{_n}.usd"))
)) if _os.path.isdir(_VISDEX_ROOT) else ()

# 구조적으로 못 잡는 작은 물체(반경 소 → 손가락 감쌀 여유 없어 force closure 불가)는
# 학습 물체군에서 제외한다. fc2 per-object 실측: right/left 모두 small 계열 전부 <0.1
# (right 최하위 5종·left 최하위군 = small_5/8/12 × cyl/cuboid). cup 은 별도 side approach
# 로 유지(SIDE_APPROACH_OBJECT_NAMES). onehot 차원이 len(active)만큼이라 재학습 필요.
_EXCLUDED_SMALL_OBJECTS: tuple[str, ...] = (
    "small_5_cyl", "small_8_cyl", "small_12_cyl",
    "small_5_cuboid", "small_8_cuboid", "small_12_cuboid",
)
_ACTIVE_OBJECT_ROOT: str = _VISDEX_ROOT
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = tuple(
    _n for _n in _VISDEX_NAMES if _n not in _EXCLUDED_SMALL_OBJECTS
)


def _primitive_usd_cfg(name: str) -> "sim_utils.UsdFileCfg":
    """단일 물체 USD 의 spawn cfg (cup 과 동일 rigid/articulation 속성)."""
    return sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ACTIVE_OBJECT_ROOT, name, f"{name}.usd"),
        activate_contact_sensors=True,
        scale=(1.0, 1.0, 1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=100.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )


# env 별 물체를 env_id % N 로 결정적 배정(random_choice=False → proto[index % len]).
# → per-object 로깅(object_idx = arange(num_envs) % N)과 균등 배정 보장. replicate_physics=False 필요.
def _grasp_object_spawn_for(names: "tuple[str, ...] | list[str]") -> "sim_utils.MultiAssetSpawnerCfg":
    """주어진 물체 이름 목록으로 MultiAsset 스포너 생성 (distillation 실패물체 제외에 재사용)."""
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[_primitive_usd_cfg(_n) for _n in names],
        random_choice=False,
    )


_GRASP_OBJECT_SPAWN = _grasp_object_spawn_for(_ACTIVE_OBJECT_NAMES)


@configclass
class EventCfg:
    """DEXTRAH physics DR (원본 EventCfg 1:1 — asset 이름만 robot/cup).

    초기 범위는 중립(스케일 1 또는 0) — ADR 증분이 adr_physics_cfg 종점까지
    선형 확장(GraspADR._expand_physics_ranges). mode="reset" 이라 에피소드
    리셋마다 per-env 재샘플.
    """

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v1 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    # Episode: 10s = 600 steps @ 60Hz (8s grasp + 2s lift)
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간 — DEXTRAH teacher 구조 (base + 물체 onehot)
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)          # 193 + N_obj
    action_space:      int = NUM_ACTIONS                                        # 11
    state_space:       int = NUM_CRITIC_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)   # 247 + N_obj

    num_observations: int = NUM_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0   # DEXTRAH README teacher 레시피 45°. palm rpy 공칭(90,0,90)±45 — 손바닥 방향을 action 공간에서 constrain(천장 지향 구조 차단). 90은 README 확인 전 오독이었음
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # 접근 자세 분기 (기본 top-down / cup 만 side)
    # -----------------------------------------------------------------------
    # DEXTRAH 와 동일하게 top-down 이 기본. cup 은 내용물을 흘리면 안 되므로 옆면을
    # 감싸 잡는다(grasp_v1 방식). 물체 이름 기반 고정 분기라 ADR 회전과 무관하다.
    approach_branch_enable:      bool = True
    side_approach_object_names:  tuple[str, ...] = SIDE_APPROACH_OBJECT_NAMES
    # pregrasp 를 물체 크기에 비례시키기 위한 bbox (compute_object_bbox.py 산출물)
    object_bbox_path:            str = _os.path.join(_ASSETS_DIR, "object_bbox.json")

    # -----------------------------------------------------------------------
    # Demo reset (pour_v1_a11~a20 grasp start and lift target)
    # -----------------------------------------------------------------------
    # grasp_v2: cup demo pose 는 다물체에 부적합 → demo-free reset(FABRICS pregrasp cache) 사용.
    enable_demo_grasp_reset: bool = False
    demo_grasp_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_HDGP_ROOT, "..", "datasets", f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # actor obs에만 적용; critic obs는 privileged clean state 유지
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.06
    lift_success_height: float = 0.04

    # -----------------------------------------------------------------------
    # Delta palm action (pregrasp 기준 상대 오프셋)
    # action=0 → pregrasp 위치 유지, action=±1 → pregrasp ± delta
    # -----------------------------------------------------------------------
    # 주의: delta는 속도가 아니라 pregrasp 기준 "도달 범위". goal 기하상 리프트
    # 필요량 ~0.10~0.15m (goal z 0.45 − 안착 물체 z ~0.25, tol 0.10) → 0.03이면
    # goal 도달 불가(in_success 붕괴). 접근 속도 문제는 아래 rate limit이 담당.

    # palm 목표 rate limit (스텝당 최대 변화량) — 접근 밀침·리프트 후 스윙 대책.
    # 정책의 목표 순간이동(±delta 범위 bang-bang)을 기구적으로 제한: finger 래칫·
    # max_pose_angle 과 같은 action 공간 제약 (reward 아님). DEXTRAH는 kuka
    # 대반경 top-down + 큰 관성이 자연 필터지만 우리는 단반경 side/대각 접근이라
    # 명시 제한 필요. 0.01m/step @60Hz = 0.6 m/s, 2°/step = 120°/s.
    # 0.01/2°는 과잉 제한 실증(test11 159M: 접근 실패 → in_success 0, h2o 0.05,
    # bang-bang 왕복이 rate에 갇혀 실효 전진 0). fabric 실효 속도 수준(0.04m=2.4m/s,
    # 8°=480°/s)으로 완화 — 순간이동만 차단.
    palm_rate_xyz_per_step:     float = 0.04
    palm_rate_rot_deg_per_step: float = 8.0

    # -----------------------------------------------------------------------
    # Reward 파라미터 — DEXTRAH 4항 (dextrah_kuka_allegro compute_rewards 이식)
    # goal = DEXTRAH식 고정 절대점 (spawn 중심 xy, z = 안착(~0.24)+0.21).
    # success = |obj-goal| < tol. tol 0.10 이 물체별 안착 높이 편차(수 cm)를 흡수.
    # object_to_goal_sharpness·lift_weight·finger_curl_reg 는 ADR 스케줄이 우선
    # (enable_adr=True 시 adr_custom_cfg.reward_weights 로 대체).
    # -----------------------------------------------------------------------
    object_goal_pos:          tuple = (0.27, -0.10, 0.45)
    object_goal_tol:          float = 0.10
    hand_to_object_weight:    float = 1.0
    hand_to_object_sharpness: float = 10.0
    object_to_goal_weight:    float = 5.0
    object_to_goal_sharpness: float = 15.0   # exp(-s·err) 형태(양수 s). DEXTRAH -15·exp(+s·err)와 동치
    lift_weight:              float = 5.0
    lift_sharpness:           float = 8.5
    finger_curl_reg_weight:   float = 0.0    # ADR 미사용 시 fallback (ADR은 -0.01→-0.005)

    # -----------------------------------------------------------------------
    # 접촉/force-closure reward (07.14 신설) — passive baseline 붕괴 + 파지 유도
    #
    # tesollo lstm_test11 실측: contact/tip 0.006(접근조차 안 함), 총보상의 ~100%가
    # 무행동 수입(lift+o2g+h2o passive). DEXTRAH 4항엔 접촉 신호가 없어 "물체를
    # 건드릴 이유"가 없다. 아래 3요소로 접촉→force closure→리프트를 순차 유도한다.
    #
    # (A) grasp_contact: 손끝이 실제 물체에 닿는가(force_matrix Cup-only 개수 기반).
    #     rh56f1 에서 접촉 유도 확인됐으나 "접촉만 유지·리프트 0" 고착 함정도 확인 →
    #     weight 낮게(0.3) + ADR 로 감쇠(부트스트랩 후 끔) + persist 비중 축소.
    # (B) force_closure(opposition): 엄지 접촉력 방향 vs 4지 평균 접촉력 방향의 −cos.
    #     마주보고 조이면 +. rh56f1 이 접촉해도 못 든 진짜 결측 요소(force closure).
    #     grip_frac(양쪽 실접촉)으로 게이트해 hacking 차단. ← 리프트 주력.
    # (C) lift_reward *= grip_frac: 안 잡으면 lift 보상 0 → passive("기다리기") 제거.
    # -----------------------------------------------------------------------
    grasp_contact_weight:        float = 0.3   # (A) ADR 로 0.3→0.0 감쇠(부트스트랩)
    grasp_contact_persist_steps: int   = 15    # 연속 접촉 N step 이면 persist_frac=1
    force_closure_weight:        float = 4.0   # (B) 리프트 주력. o2g(5.0) 급
    force_closure_force_scale:   float = 3.0   # N. tanh(‖f‖/scale) grip 세기 정규화
    lift_grip_gate_enable:       bool  = True  # (C) lift_reward *= grip_frac
    # palm orientation: DEXTRAH 4항엔 손목 방향 제약이 없어 손바닥이 임의(천장) 방향으로
    # 수렴. palm 법선(로컬 +X → world)이 palm→물체 방향과 정렬되도록 보조 shaping.
    # w·exp(s·(align−1)): align=1(완전 정렬)→w, align=−1(반대)→w·exp(−2s). weight는
    # object_to_goal(5.0)의 0.2배로 통제(reward-audit ACCEPT: local-min·hacking 방지).

    # (구) RH56F1 shared grasp-v2 reward contract — DEXTRAH 전환으로 미사용(호환 보존).
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    lift_reward_weight: float = 30.0
    stabilize_weight: float = 10.0
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02
    grasp_xy_threshold: float = 0.025
    grasp_upright_threshold_deg: float = 8.0
    success_upright_max_deg: float = 20.0
    stabilize_upright_max_deg: float = 5.0
    stabilize_upright_reward_scale_deg: float = 5.0
    stabilize_action_sharpness: float = 1.5
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.2
    stage0_lift_start_min_contacts: int = 2  # lift 진입: grip(tip|mid|distal) 손가락 수. visdex 큰물체 2~3 파지 대응(4→3→2, 엄지+1).
    success_min_grip_fingers: int = 3  # success 그립 손가락 수(grip 기준, 엄지 접촉 AND). 큰 물체 대응(4→3).
    # 파지력 확보: 물체 외란 wrench (DEXTRAH apply_object_wrench 이식).
    # 물체가 가만히 있으면 꽉 잡을 유인이 없음(grip 0.93) → 외란을 줘서 정책이 파지력 학습.
    wrench_enable: bool = True
    wrench_max_accel: float = 10.0      # m/s² (DEXTRAH 수준, force~중력급 1N). 물체 실제 흔들려야 파지력 유인. force = object_mass × accel × 랜덤방향
    wrench_torsional_radius: float = 0.03  # torque = mass × accel × radius × 랜덤방향
    wrench_trigger_every: int = 60      # step(=1초 @60Hz)마다 새 랜덤 wrench
    grasp_ready_hold_steps: int = 8   # 접촉 N개를 연속 hold하면 lift 래치 (잡으면 바로 리프트)
    lift_start_min_envelope_fingers: int = 0  # latch 인벨롭 게이트 제거(0=비활성). envelope은 grasp/lift 보상 credit으로 유도(hard 게이트 대체)
    finger_close_speed: float = 0.05  # ① 접촉-게이트 적응 폐쇄: 손가락 폐쇄 진행 속도/step (중간마디 접촉 시 동결)
    # [07-20 재검증 실험] 접촉 동결(g3/g4): 마지막 True 테스트는 07-08(cd91f22, "grip 0.90 정체"로
    # False 복귀)뿐이었는데, 그때는 synergy(PCA) 제어·구 4항 reward·단일물체 시절이라 지금과
    # 조건이 다르다. 이후 per_finger 전환(52e0fb9)·힘-크기 기반 force_closure(tip+distal+middle
    # 합산, 접촉이 아닌 힘 크기 요구)·148물체·cup 스케일 수정이 모두 들어간 채로는 한 번도
    # 재검증되지 않았다. lstm_test1(3지 파지·중지/약지 tip 0.00·6cm herding)은 이 값 False로
    # 학습됨 — 현재 조합에서 True 단독 격리 실험(reward-audit REVISE, 다른 항 불변). 결과에 따라
    # 원복 또는 유지.
    synergy_freeze_enable: bool = True

    # 손 제어 방식 — "per_finger"(grasp_v1) | "synergy"(DEXTRAH PCA)
    #
    # PCA 는 basis 가 20관절을 커플링해 형상 적응이 불가능하다. finger_action_utils.py:61
    # 이 스스로 적어둔 한계다 — "PC1 하나가 20관절을 커플링 → 2지 최소해가 action 공간에서
    # 표현 불가". 게다가 현행 basis 는 Allegro(4지 16관절) 리타겟이라 pinky_1 열이 5축
    # 전부 0 이고, PC3 는 coeff min(0.34) > max(-3.06) 으로 범위가 역전돼 있다.
    #
    # LEFT 성공 파지 실측(89119 샘플, 평균 리프트 18.2cm)이 문제를 드러낸다:
    #     index  +0.001 +0.541 +0.834 +1.567
    #     pinky  -0.002 -0.000 +0.677 +1.451   ← pinky_2 만 전혀 안 굽는다
    # 손가락마다 다른 굽힘이 필요한데 PCA 로는 낼 수 없어, 정책이 5축을 전부 극단값
    # [-1.000, +0.979, -0.997, -0.999, +0.975] 으로 밀어 억지 근사하고 있었다.
    #
    # per_finger 는 손가락 5개가 독립이라 envelope 이 가능하다 — 목표를 FULL_GRIP 까지
    # 밀면 물체에 닿거나 포화된 관절은 멈추고 나머지만 계속 감긴다(grasp_v1 방식, 98% 실증).
    finger_control_mode: str = "per_finger"

    # contact sensor 필터 대상 — Cup prim 하위의 실제 rigid body.
    # probe_contact_filter 실측: /World/envs/env_0/Cup 은 Xform 이고 RigidBodyAPI 는
    # baseLink 에 있다. 이 경로로 필터를 걸면 force_matrix_w 가 (N,1,1,3) 으로 나온다
    # — "GPU 미지원" 이라던 종전 주석은 오진이었다.
    cup_rigid_body_name: str = "baseLink"

    # contact/grip_near 판정 거리 — 필터 복원 후에도 교차검증용으로 남긴다
    # (센서가 물체 접촉만 세는지 거리로 재확인). 로깅 전용, reward 무관.
    contact_near_dist: float = 0.06

    settle_steps: int = 25  # 다물체 drop-settle: episode 초기 N step 손가락 폐쇄 억제 → 물체 낙하 안착
    # [FP 배포 검증] object pose obs 를 settle 시점에 lock(freeze)하고 이후 고정 사용.
    # 물체가 정적이므로 "FoundationPose 로 폐색 전 pose 를 한 번 찍고 open-loop grasp"를
    # 모사한다. teacher 가 이 고정 pose 로도 파지에 성공하면 vision distillation 없이
    # FP + 직접배포가 가능하다는 근거. 기본 False(학습·기존 경로 무영향).
    eval_pose_hold: bool = False
    # abduction 목표 rate limit (rad/step). finger_close_speed 와 같은 취지 —
    # 자기충돌 검사가 꺼져 있어(enabled_self_collisions=False) 순간이동식 abduction 은
    # 인접 손가락을 관통한다. 0.02 rad/step ≈ 전 범위(1.05 rad) 통과에 ~0.9초.
    abduction_rate_limit: float = 0.02
    # finger_curl_reg = -0.01·‖q-q_grip‖² 는 5개 reward 항 중 유일하게 무계(아래로 무한).
    # 물리 solver 가 발산하면 제곱으로 증폭돼 에피소드 리턴이 -4.9e7 까지 튀고
    # (lstm_test1 iter 14111) rl_games 의 리턴/value 통계를 오염시켜 정책이 붕괴한다.
    # 정상 물리에서는 joint limit 때문에 ‖q-q_grip‖ 이 이 값을 넘을 수 없다 →
    # clamp 는 정상 학습에 영향 0, 발산 시에만 발동.
    finger_curl_dist_max: float = 14.0

    grasp_contact_persistence_reward_steps: int = 20
    enclosure_sharpness: float = 15.0
    cup_radius_approx: float = 0.045
    enclosure_thumb_weight: float = 0.6

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    # DEXTRAH ADR 커리큘럼: in_success_region 순간 평균 > threshold 마다 increment,
    # 각 파라미터가 initial→final 선형 진행. (원본 success_for_adr=0.4)
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    # DEXTRAH min_steps_for_dr_change = 5 × 에피소드 길이 (600 steps @10s) = 3000
    adr_increment_interval: int  = 3000
    adr_trigger_threshold: float = 0.4
    # DEXTRAH starting_adr_increments (env_cfg.py:339) 이식. 0=기존 동작(레벨 0 시작),
    # adr_num_increments=만렙 고정 시작. distillation 은 teacher 작동점(만렙)에 고정해야
    # abduction 이 안 잠긴다 — DISTILL cfg 가 50 으로 오버라이드.
    starting_adr_increments: int = 0

    # DEXTRAH physics DR: EventCfg(reset 이벤트) + ADR 범위 확장 종점 (원본 adr_cfg_dict)
    events: EventCfg = field(default_factory=EventCfg)
    adr_physics_cfg: dict = field(default_factory=lambda: {
        "robot_physics_material": {
            "static_friction_range":  (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range":      (0.8, 1.0),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params":   (0.5, 2.0),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0.0, 5.0),
        },
        "object_physics_material": {
            "static_friction_range":  (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range":      (0.8, 1.0),
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.0),
        },
    })

    # DEXTRAH wrench 게이트: 손이 물체 반경 내면 외란 인가 (in_success 게이트 아님)
    hand_to_object_dist_threshold: float = 0.3   # m

    adr_custom_cfg: dict = field(default_factory=lambda: {
        # 외란 wrench: 0→10 점진 (DEXTRAH. 기존 고정 10은 초기 리프트 학습 방해 가능)
        "object_wrench": {
            "max_linear_accel": (0.0, 10.0),
        },
        # spawn 커리큘럼: 반경 0→최대, 회전 0→full (DEXTRAH object_spawn)
        "object_spawn": {
            "xy_range": (0.0, 0.06),
            "rotation": (0.0, 1.0),
        },
        # abduction 커리큘럼: 0 → 1 로 열린다.
        #   0 = HAND_APPROACH_POSE 값에 고정 (thumb_2 = -90° 등) → 실효 11D = DEXTRAH 원본
        #   1 = 전 범위 자유 (thumb_1/thumb_2/index_1/pinky_1/pinky_2)
        # probe 실증: abduction 중립에서 리프트 +17.6cm (palm 물체 위 4cm, thumb_2 -90°),
        # 벌리면(+1) 리프트 0 — 즉 abduction 을 처음부터 열면 파지를 방해한다.
        # 기본 파지를 배운 뒤(in_success > 0.4 로 ADR 상승) 세밀 제어를 연다.
        "abduction": {
            "range_scale": (0.0, 1.0),
        },
        # 관측 노이즈 점진 (DEXTRAH object/robot_state_noise 원본값)
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03),   # m
            "object_pos_bias":  (0.0, 0.02),   # m
            "object_rot_noise": (0.0, 0.1),    # rad
            "object_rot_bias":  (0.0, 0.08),   # rad
        },
        "robot_state_noise": {
            "robot_joint_pos_noise": (0.0, 0.08),  # rad
            "robot_joint_pos_bias":  (0.0, 0.08),  # rad
            "robot_joint_vel_noise": (0.0, 0.18),  # rad/s
            "robot_joint_vel_bias":  (0.0, 0.08),  # rad/s
        },
        # 리셋 시 로봇 초기상태 노이즈 커리큘럼 (DEXTRAH robot_spawn)
        "robot_spawn": {
            "joint_pos_noise": (0.0, 0.35),   # rad
            "joint_vel_noise": (0.0, 1.0),    # rad/s
        },
        # PD velocity feedforward 1→0 (DEXTRAH pd_targets — 종점 0 은 기존 코드의 zeros 와 동치)
        "pd_targets": {
            "velocity_target_factor": (1.0, 0.0),
        },
        # reward 스케줄 (DEXTRAH): lift shaping 5→0 걷어내고 goal 정밀도(sharpness) 강화
        "reward_weights": {
            "finger_curl_reg":          (-0.003, -0.003),  # tesollo 스케일 정합. DEXTRAH 는 Allegro(16관절,
        #   파지 굽힘 ~1.0rad)에서 ‖q-anchor‖²≈12 → 실효 -0.12/step. tesollo 는 20관절·1.8rad 라
        #   ‖·‖²≈37 → 같은 weight 로 -0.37/step (3배). 그 결과 "손을 굽히는 것 자체가 순손실"이 되어
        #   정책이 손을 펴고 물체에서 도망갔다(curl -0.367→-0.005 와 h2o 급락이 동시 발생).
            "object_to_goal_sharpness": (15.0, 20.0),   # 우리 exp(-s·err) 부호
            "lift_weight":              (5.0, 0.0),
            # grasp_contact 부트스트랩 감쇠: 초기엔 접촉 유도(0.3), 파지 학습 후
            # force_closure 가 주력이 되도록 끔(0.0). "접촉만 유지" 고착 함정 회피.
            "grasp_contact_weight":     (0.3, 0.0),
        },
        # fabric cspace damping 강화 (DEXTRAH 10→20)
        "fabric_damping": {
            "gain": (10.0, 20.0),
        },
        # velocity obs annealing: DEXTRAH teacher는 (0,0)=상시 0 (실로봇 vel 추정 부재 대비)
        "observation_annealing": {
            "coefficient": (0.0, 0.0),
        },
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 60.0
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

    # 로봇 발산 종료: fabric 폭주로 손이 도달불가 위치로 튕기면 물체가 테이블에
    # 남아 컵-기준 종료가 안 걸림 → timeout까지 방치되던 문제. palm↔물체 거리가
    # 정상 workspace(리치 + pregrasp offset)를 크게 넘거나 NaN이면 조기 종료.
    robot_escape_dist: float = 0.80

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.27   # demo 데이터와 일치 (0.40→0.27)
    object_spawn_y_center: float = -0.10  # demo 데이터와 일치 (-0.15→-0.10)
    object_spawn_z:        float = 0.297
    # 활성 물체군(spawn 순서와 일치) — onehot·per-object 로깅용 이름. env_id % N 로 배정.
    # (물체 조건화는 DEXTRAH식 onehot 으로 전환 — 접근 B feature 는 obs 미사용)
    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
    # distillation 전용: 여기 나열한 물체는 스폰·배정에서 제외한다(teacher 완료 후 실패물체 주입).
    # onehot 차원은 active_object_names(153) 를 그대로 유지 → teacher 체크포인트 호환.
    # teacher 학습(distillation=False)은 이 필드를 무시한다. DISTILL cfg 에서 스포너를 kept 로 교체.
    distill_excluded_object_names: tuple[str, ...] = ()
    object_spawn_xy_range: float = 0.06   # ADR 미사용 시 fallback (ADR은 0→0.06 커리큘럼)

    # -----------------------------------------------------------------------
    # Warm-state export (grasp 성공 → 디스크 캐시 → pour warmstart 재사용)
    # -----------------------------------------------------------------------
    # 학습 루프에는 영향 없음 (기본 False). collect 스크립트/play 에서만 True.
    # success 이후 오른손 grasp arm pose 를 유지하고 joint7 만 lift-wait 로 이동한 상태를 저장한다.
    # demo cup/phase 구분을 신뢰하지 않고 실제 sim 손/컵 grasp 결과를 그대로 유지한다.
    # 손가락 접촉은 기본 2개 이상, lift-wait arm match 는 기본 1 step 만 요구한다.
    enable_warm_state_export: bool = False
    warm_state_export_path: str = _os.path.normpath(
        _os.path.join(_HDGP_ROOT, "..", "datasets", "grasp_warm_v1.hdf5")
    )
    warm_state_target_count: int = 2048
    warm_min_contacts: int = 2
    warm_contact_stable_steps: int = 1
    warm_lift_wait_arm_tol: float = 0.035
    warm_lift_wait_hold_steps: int = 1
    lift_wait_joint7_delta: float = 0.31
    warm_cup_upright_min: float = 0.90   # legacy override 호환용; lift-wait export 에서는 미사용
    warm_j7_min: float = 0.20
    warm_j7_max: float = 1.50

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
        # grasp_v2: MultiAsset(env 별 다른 물체) spawn 은 physics 복제 불가.
        replicate_physics=False,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            # x: 0.5725 → 0.4725 (렌더 확인: 테이블이 로봇에서 10cm 멀었다)
            pos=[0.4725, 0.003, 0.2],
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
    # 로봇 설정 (openarm_tesollo_bi_rl.usd: 양팔 tesollo, 통일 네이밍 r_aj/r_hj/r_hl + l_hj tesollo 20관절)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.usd"),
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
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # 손 stiffness/damping: pour-v5/6 검증값 채택. 기존 30/5(물렁)은 엄지 _3/_4가
            # 컵을 감을 때 반력이 엄지 대향(_2=-1.57)을 뒤로 밀어냄(play 렌더 관찰). 단단히 유지.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_1"],
                stiffness=600.0,   # 2000→600: 2000은 _1을 0에 고정했으나 컵 반력을 큰 교정토크로 되받아 파지 교란→붕괴(pour 강성과도 메커니즘이 abduction에도 적용). 600은 roll(-1.06)을 크게 줄이되 반력 교란 완화.
                damping=40.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_2"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_3"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_4"],
                stiffness=400.0,
                damping=60.0,
            ),
            # 왼손 tesollo (bi USD): 학습 미사용, rest 자세 유지용 hold
            "tesollo_left_hand": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_[1-4]"],
                stiffness=400.0,
                damping=60.0,
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

    # distal/middle 도 tip 처럼 Cup-only 필터(force_matrix_w). 무필터(net_forces)면
    # 손가락이 컵이 아닌 다른 손가락/palm 에 self-contact 해도 grip 으로 잡혀,
    # 엄지가 컵을 안 닿고도 success(num_grip>=5)를 거짓 충족하던 버그를 차단한다.
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_4",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_3",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # Distillation (teacher → vision student) — RealSense D435i mono RGB-D
    #
    # distillation=False 가 기본. teacher(PPO) 학습 경로는 아래 설정을 일절 타지 않는다.
    # True 로 켜면: TiledCamera 활성 + obs dict 가 4-key (policy/expert_policy/img/rgb).
    # 이때 "policy" 는 student obs(190, 물체 미관측)로 바뀌고 teacher obs 는
    # "expert_policy" 로 이동한다 — teacher 관측 구조 자체는 변경 없음.
    # -----------------------------------------------------------------------
    distillation: bool = False

    num_student_observations: int = NUM_STUDENT_OBS     # 190 (물체 privileged state 제외)
    num_teacher_observations: int = NUM_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)  # 193 + N_obj

    img_width:  int = CAMERA_IMG_WIDTH
    img_height: int = CAMERA_IMG_HEIGHT
    d_min: float = CAMERA_D_MIN
    d_max: float = CAMERA_D_MAX
    # 중앙 crop 비율(물체 detail 확보). 1.0=crop 없음(기존). 0.5=중앙 절반 crop→원크기 업샘플.
    camera_crop_frac: float = CAMERA_CROP_FRAC

    tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_POS, rot=CAMERA_ROT, convention="ros"
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        width=CAMERA_IMG_WIDTH,
        height=CAMERA_IMG_HEIGHT,
    )

    # 이미지 증강 종류 — student 인코더가 보는 텐서를 증강해야 한다.
    #   "rgb"   : a2c_mono_transformer 등 배포 학생망 전부 (use_depth=False → RGB 입력)
    #   "depth" : 구 a2c_with_aux_depth 처럼 depth 를 직접 입력받는 경우
    # 여기서 틀리면 인코더가 안 보는 텐서에 노이즈를 넣게 되어 조용히 헛돈다.
    img_aug_type: str = "rgb"

    # 시각 도메인 랜덤화 텍스처 (DEXTRAH textures.zip). git 비추적 — server 는 별도 확보.
    # enable_visual_dr=False 면 텍스처 없이도 기동한다 (카메라 배치 프리뷰용).
    # 학습에서는 절대 끄지 말 것 — 외형이 고정되면 student 가 단일 장면에 과적합된다.
    enable_visual_dr: bool = True
    texture_root: str = _TEXTURE_ROOT
    disable_dome_light_randomization: bool = False
    disable_robot_randomization: bool = False

    # depth 도메인 랜덤화 (img_aug_type="depth" 일 때만 적용)
    aug_depth: bool = True
    aux_coeff: float = 1.0        # aux head(object_pos 회귀) 손실 가중

    # normal_noise 커널이 픽셀→광선 역투영에 쓰는 정규화 투영행렬.
    # a = 1/tan(hfov/2), b = a * W/H  (DEXTRAH 원본 규약: 주점 오프셋 없음)
    cam_matrix = _CAM_MATRIX
    depth_randomization_cfg_dict: dict = field(
        default_factory=lambda: depth_randomization_cfg(
            _CAM_MATRIX, CAMERA_D_MIN, CAMERA_D_MAX
        )
    )

    # distillation rollout 은 성공 후 조기 종료 (DEXTRAH success_timeout)
    success_timeout: int = 60

    # -----------------------------------------------------------------------
    # 컵 설정
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        # grasp_v2: 단일 cup → primitives 다물체(MultiAsset, env 별 random_choice).
        # prim 이름 "Cup" 은 유지(ContactSensor 필터/센서 참조 재사용).
        spawn=_GRASP_OBJECT_SPAWN,
    )

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
