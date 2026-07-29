# Copyleft 2025 Enactic, Inc.
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

- Action: 11D (6D palm pose Fabrics IK + 5D per-finger 폐쇄)
- Observation: DEXTRAH teacher 구조 — policy 193+N_obj / critic 247+N_obj
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
from .grasp_left_constants import NUM_OBS_BASE, NUM_ACTIONS, NUM_CRITIC_OBS_BASE, NUM_STUDENT_OBS
from openarm.distillation.camera import depth_randomization_cfg, make_cam_matrix
from .grasp_left_preset import (
    HAND_APPROACH_POSE,
    HAND_BODY_NAMES_USD,
    RIGHT_ARM_AND_GRIPPER_JOINT_NAMES,
    RIGHT_ARM_REST_JOINT_POS,
    LEFT_ACTUATED_JOINT_NAMES,
    SIDE_APPROACH_OBJECT_NAMES,
    CAMERA_IMG_WIDTH,
    CAMERA_IMG_HEIGHT,
    CAMERA_FOCAL_LENGTH,
    CAMERA_HORIZONTAL_APERTURE,
    CAMERA_CLIPPING_RANGE,
    CAMERA_POS,
    CAMERA_ROT,
    CAMERA_D_MIN,
    CAMERA_D_MAX,
    CAMERA_CROP_FRAC,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")

# distillation 시각 자산 (right/grasp_v2 규약 동일).
_TEXTURE_ROOT = _os.path.join(_ASSETS_DIR, "dextrah_textures")
_CAM_MATRIX = make_cam_matrix(
    CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT, CAMERA_FOCAL_LENGTH, CAMERA_HORIZONTAL_APERTURE
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
# <name>/<name>.usd 존재하는 디렉토리만 물체로 인정 (.ruff_cache 등 잡동사니 유입 시
# onehot N이 틀어지고 MultiAsset spawn이 FileNotFoundError로 깨짐 — 07.10 실측)
_VISDEX_NAMES: tuple[str, ...] = tuple(sorted(
    _n for _n in _os.listdir(_VISDEX_ROOT)
    if _os.path.isfile(_os.path.join(_VISDEX_ROOT, _n, f"{_n}.usd"))
)) if _os.path.isdir(_VISDEX_ROOT) else ()

# 활성 물체군: visdex 153종. primitives 로 되돌리려면
#   _ACTIVE_OBJECT_ROOT = primitives/USD, _ACTIVE_OBJECT_NAMES = _PRIMITIVE_CURRICULUM_STAGE1, prefix=primitive.
_ACTIVE_OBJECT_ROOT: str = _VISDEX_ROOT
_EXCLUDED_SMALL_OBJECTS: tuple[str, ...] = (
    "small_5_cyl", "small_8_cyl", "small_12_cyl",
    "small_5_cuboid", "small_8_cuboid", "small_12_cuboid",
)
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = tuple(
    _n for _n in _VISDEX_NAMES if _n not in _EXCLUDED_SMALL_OBJECTS
)


# 물체 스케일 커리큘럼 (07.13, 153종 스캔 근거): scale 1.0에선 RH56F1 envelope 이
# 대부분 물체에 닫힘(scripted 리프트 1/153). 0.75에서 존재 증명(cup +14.6cm) —
# 축소 물체로 성공 표본을 만들고, 성공 후 1.0 복원 run 으로 커리큘럼(grasp_v1
# 컵 축소 전례). run 간 커리큘럼 — 스폰 시점 고정이라 학습 중 변경 불가.
OBJECT_SCALE_CURRICULUM: float = 0.75


def _primitive_usd_cfg(name: str) -> "sim_utils.UsdFileCfg":
    """단일 물체 USD 의 spawn cfg (cup 과 동일 rigid/articulation 속성)."""
    _s = OBJECT_SCALE_CURRICULUM
    return sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ACTIVE_OBJECT_ROOT, name, f"{name}.usd"),
        activate_contact_sensors=True,
        scale=(_s, _s, _s),
        # 질량 고정 (07.13): scale 0.75에서 질량이 scale³=0.42배로 줄어 drop 튕김·
        # 팔 스윕만으로 경계 이탈(zero-action 60step 종료 21% 실측, episode 57step
        # 붕괴 — d14). 균일 0.15kg 로 고정(physics DR 이 위에서 ±랜덤화).
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
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
_GRASP_OBJECT_SPAWN = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[_primitive_usd_cfg(_n) for _n in _ACTIVE_OBJECT_NAMES],
    random_choice=False,
)


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
class GraspLeftEnvCfg(DirectRLEnvCfg):
    """5g_grasp_left_v1 환경 설정."""

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
    use_cuda_graph:   bool  = True

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
    # 07.14: 손 = 6D per-finger 직접 제어(시너지/PCA 폐기). fabric 은 팔 IK 만 담당.
    # RH56F1 은 하드웨어 언더액추(원위 mimic)라 이미 물리적 시너지 — 그 위 소프트웨어
    # PCA(5D)는 이중 압축이라 thumb_1(opposition)을 죽였음. 6 액추에이터 직접 제어로 전환.
    max_pose_angle:             float = 45.0   # DEXTRAH README teacher 레시피 45°. palm rpy 공칭(180,0,90)±45 — 손바닥 방향을 action 공간에서 constrain(천장 지향 구조 차단). tesollo 0fa6fb3 정렬
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    # 07.13: pregrasp offset 이 물체 clearance 로 연속값이 되어(아래) grid IK 캐시
    # 전제(고정 offset)가 깨진다 — 캐시 제거, reset 마다 fabrics rollout (tesollo 9f0e4f7).
    # 07.13 접근 자세 분기 반전(tesollo cd29c62 이식): top-down 이 기본, cup 만 side.
    # E3(전 물체 top-down)는 153종 스캔에서 envelope 거의 닫힘 확정(scale1.0 리프트
    # 1/153), 진단 롤아웃(ep2500)에서도 cup(존재증명 물체)만 100% 성공하고 종료
    # 189건이 fallen/out_x(물체 쳐냄) — 형상별 자세 불일치가 근본 원인.
    # 아래 x/y/z 는 side(cup 전용) 기하: x/aj7 는 그대로, y/z 는 clearance 공식이
    # 부호·상수로 사용(_compute_pregrasp_offset) — E3 이전 lstm_test1 60% 검증값.
    pregrasp_offset_x:     float = -0.07
    pregrasp_offset_y:     float = 0.08   # y-미러
    pregrasp_offset_z:     float = -0.15
    # l_aj_7(손목)을 이만큼 낮춰 palm을 rim→물체 중심 높이로 내림(probe 확정, lstm_test1 검증).
    pregrasp_l_aj7_bias:   float = 0.3
    # settle 종료 시 안착된 물체 위치로 anchor xy 재정렬 (drop-settle 롤링 보정).
    reanchor_after_settle: bool  = True
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # 접근 자세 분기 (top-down 기본 / cup 만 side, tesollo cd29c62 이식)
    # -----------------------------------------------------------------------
    approach_branch_enable:        bool  = True
    side_approach_object_names:    tuple[str, ...] = SIDE_APPROACH_OBJECT_NAMES
    object_bbox_path:              str   = _os.path.join(_ASSETS_DIR, "object_bbox.json")

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
    # palm action: workspace 절대 pose (DEXTRAH 원본 구조, tesollo 1aa9dcc 이식)
    # action[0:6] ∈ [-1,1] → palm_mins_env~palm_maxs_env 박스로 직접 스케일.
    # -----------------------------------------------------------------------
    # 07.13: anchor+delta 방식은 물체까지 20~30cm 를 매 스텝 재적분해야 해서
    # credit assignment 가 무너졌다(d9~d15 "가만히 있기" 수렴의 근본원인 중 하나 —
    # tesollo 동일 병리 실증: curl 기준 수정 후에도 hand_to_object ep200 0.216 →
    # ep400 0.017 급락). DEXTRAH 원본은 절대 pose 라 "물체 위로 가라"가 1스텝 결정.
    # 07.14: palm rate limit 도 제거(DEXTRAH 직접제어) — settle override 와 함께
    # tesollo 계보 scaffolding 이 정책 탐색 자유도를 좁혀 object_height 정체시킨
    # 원인 후보로 격리. DEXTRAH 는 t=0 부터 palm 절대 pose 를 매 스텝 그대로 지령.
    # -----------------------------------------------------------------------
    # Reward 파라미터 — DEXTRAH 4항 (dextrah_kuka_allegro compute_rewards 이식)
    # success = |obj-goal| < tol. tol 0.10 은 DEXTRAH 원본과 동일(object_goal_tol=0.1).
    # object_to_goal_sharpness·lift_weight·finger_curl_reg 는 ADR 스케줄이 우선
    # (enable_adr=True 시 adr_custom_cfg.reward_weights 로 대체).
    #
    # goal 위치 (07.14, DEXTRAH 재확인 — 사용자 질문 "먼 물체도 성공하는데 왜 우린
    # 못하나"): 구 goal(0.27,-0.10,0.45)은 spawn(0.27,-0.10,0.297)과 x/y 가 완전히
    # 같고 z 만 15cm 위라 "가만히 있어도" lift+o2g 로 총 reward 최대치의 11.4%를
    # 받았다(실측 reward/lift 0.87~0.99·object_to_goal 0.22~0.24). DEXTRAH 원본은
    # goal(-0.5,0,0.75)이 spawn(-0.55,0.1,~0.3)과 x/y 도 다르고 z 도 45cm 이상 떨어져
    # 07.15 goal 하향: (0.37,-0.25,0.60)은 passive(방치) 비중을 낮추려 goal 을 멀리·높이
    # 잡은 값이었으나(물체-goal 45cm, z 35cm 상승 요구), 파지 실패 시절 우려였다. force_closure
    # 로 파지·리프트가 확실히 되고(정책이 방치보다 파지-운반이 크게 이득) + lift 가 접촉 게이팅
    # (envelope)이라 안 잡으면 lift=0 → passive lift 차단됨. 로봇 base(0,0,0)·물체 spawn
    # (0.5,0,0.25)이 tesollo 와 완전 동일한데 goal 만 과설정이라 물체를 44.8cm 운반해야 성공
    # (리프트 14.6cm 로 도달 불가) → tesollo 검증값(153종 성공)으로 정합. z 20cm·3D 32cm 로 완화.
    object_goal_pos:          tuple = (0.27, 0.10, 0.45)  # y-미러
    object_goal_tol:          float = 0.10
    hand_to_object_weight:    float = 1.0
    hand_to_object_sharpness: float = 10.0
    object_to_goal_weight:    float = 5.0
    object_to_goal_sharpness: float = 15.0   # exp(-s·err) 형태(양수 s). DEXTRAH -15·exp(+s·err)와 동치
    lift_weight:              float = 5.0
    lift_sharpness:           float = 8.5
    finger_curl_reg_weight:   float = 0.0    # ADR 미사용 시 fallback (ADR은 -0.01→-0.005)
    # finger_curl_reg 는 5개 reward 항 중 유일하게 무계(제곱, 아래로 무한) — tesollo
    # 9f0e4f7 실증: 물리 발산 시 제곱 증폭으로 리턴이 -4.9e7 까지 튀어 rl_games
    # 리턴/value 통계를 오염시켜 정책 붕괴. clamp 는 정상 학습(joint limit 이내)엔
    # 영향 없고 발산 시에만 발동하는 안전판.
    finger_curl_dist_max:     float = 14.0
    # grasp 접촉 보상(07.14, grasp_v1 grasp_quality 구조 이식): hand_to_object(거리)는
    # 손 6점이 물체 근처에만 있으면 만족돼 손가락 굴곡 유인이 없음(3000ep 정체 실증).
    # force_matrix 실접촉(자기 env Cup) 개수에 직접 보상 → 굴곡을 능동 유도. persistence
    # 항이 "닿기만 하고 뗌" 해킹 차단. weight 는 hand_to_object(1.0)보다 작게 시작.
    grasp_contact_weight:        float = 0.5
    grasp_contact_persist_steps: int   = 15   # 연속 접촉 N step 이면 persistence_frac=1
    # force_closure(opposition) 보상(07.15, tesollo 이식, 리프트 주력): 접촉 개수(grasp_contact)
    # 로는 못 잡던 "엄지 vs 4지 마주조임 힘 방향"을 직접 보상. lstm_test2 실측 — 접촉·굴곡·
    # 지속(persist 0.90)은 성공했으나 thumb_2(엄지 굴곡=조임)가 진동(조임 유인 부재) → 리프트
    # 3mm 정체(못 조여 미끄러짐). 접촉반력(force_matrix) 엄지 vs 4지평균 −cos + 양쪽 세기 tanh,
    # 양쪽 실접촉 AND 게이트로 hacking 차단. weight 4.0 = o2g(5.0)급 주력.
    force_closure_weight:        float = 4.0
    force_closure_force_scale:   float = 3.0   # N. tanh(‖f‖/scale) grip 세기 정규화
    # per-object episode_success_rate 로깅 주기(step). 물체별 카운트는 _reset_idx 에서만 변하고
    # TB 는 epoch(=horizon_length 16 step) 경계에만 쓰므로 매 step 148종 CPU 전송은 불필요.
    # DEXTRAH 는 매 step 집계 .mean() 만(동기 0). horizon_length 에 맞춰 16 step마다만 갱신 →
    # hot-path 동기 제거(GPU sawtooth 완화). 집계 스칼라(python int)는 매 step 유지.
    per_object_log_interval:     int   = 16

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
    grasp_upleft_threshold_deg: float = 8.0
    success_upleft_max_deg: float = 20.0
    stabilize_upleft_max_deg: float = 5.0
    stabilize_upleft_reward_scale_deg: float = 5.0
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
    settle_steps: int = 25  # 다물체 drop-settle: episode 초기 N step 손을 approach(열림)로 고정 → 물체 낙하 안착

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
    # distillation: env 를 teacher 작동점(만렙 ADR)에 고정 시작 (0=자연 진행).
    starting_adr_increments: int = 0
    # DEXTRAH min_steps_for_dr_change = 5 × 에피소드 길이 (600 steps @10s) = 3000
    adr_increment_interval: int  = 3000
    adr_trigger_threshold: float = 0.4

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
            "finger_curl_reg":          (-0.01, -0.01),   # README teacher 레시피 고정 오버라이드 (tesollo 0fa6fb3 정렬)
            "object_to_goal_sharpness": (15.0, 20.0),   # 우리 exp(-s·err) 부호
            "lift_weight":              (5.0, 0.0),
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
    # out-of-reach 종료(07.11 조임, DEXTRAH 정렬): 물체가 palm workspace(x 0.20~0.65,
    # y -0.55~0.22)+7cm 밖이면 즉시 종료. 기존 x_max 0.85는 도달불가 물체를 남겨
    # 에피소드가 baseline 파밍으로 낭비됨(렌더 관찰: 쳐낸 물체가 닿지 않는 곳에 방치).
    obj_out_x_min:  float = 0.13
    obj_out_x_max:  float = 0.72
    obj_out_y_min:  float = -0.25  # y-미러(우측 max 반전)
    obj_out_y_max:  float = 0.60   # y-미러(우측 min 반전)
    obj_fallen_z:   float = 0.20

    # 로봇 발산 종료: fabric 폭주로 손이 도달불가 위치로 튕기면 물체가 테이블에
    # 남아 컵-기준 종료가 안 걸림 → timeout까지 방치되던 문제. palm↔물체 거리가
    # 정상 workspace(리치 + pregrasp offset)를 크게 넘거나 NaN이면 조기 종료.
    robot_escape_dist: float = 0.80

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.27   # demo 데이터와 일치 (0.40→0.27)
    object_spawn_y_center: float = 0.10   # y-미러(좌팔 워크스페이스). probe 실증: -0.10이면 물체가 우측에 스폰돼 palm↔obj 0.47m
    # 0.27 축소 시도는 역효과(anchor=spawn+0.08 과 간격 3cm → 물체가 손 안으로
    # 스폰돼 충돌 사출, 즉시 종료 49/153) — 0.297 유지 (probe_fallen 실측).
    object_spawn_z:        float = 0.297
    # 활성 물체군(spawn 순서와 일치) — onehot·per-object 로깅용 이름. env_id % N 로 배정.
    # (물체 조건화는 DEXTRAH식 onehot 으로 전환 — 접근 B feature 는 obs 미사용)
    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
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
    warm_cup_upleft_min: float = 0.90   # legacy override 호환용; lift-wait export 에서는 미사용
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
            # x: 0.5725 → 0.4725 (렌더 확인 07.14: 테이블이 로봇에서 10cm 멀어
            # 접근 시 손가락이 테이블 상면에 걸리고 종료. tesollo 가 이미 한 -0.1
            # 교정을 rh56f1 은 못 받아 옛 값이 남아있었음).
            pos=[0.4725, -0.003, 0.2],  # y-미러
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
    # 로봇 설정 (openarm_bi_rh56f1_rl.usd: 양팔 openarm, 우측 RH56F1 6-drive 손 + 좌측 RH56F1)
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
                # 활성 좌팔 시작자세 = 우팔 (0.5,0.1,0.4,0.60,-0.2,0,0) 의 ARM_SIGN 미러
                "l_aj_1": -0.5,
                "l_aj_2": -0.1,
                "l_aj_3": -0.4,
                "l_aj_4":  0.60,
                "l_aj_5":  0.2,
                "l_aj_6":  0.0,
                "l_aj_7":  0.0,
                # RH56F1 우측 손 drive (approach pose)
                "l_hj_thumb_1":  HAND_APPROACH_POSE[0],
                "l_hj_thumb_2":  HAND_APPROACH_POSE[1],
                "l_hj_index_1":  HAND_APPROACH_POSE[2],
                "l_hj_middle_1": HAND_APPROACH_POSE[3],
                "l_hj_ring_1":   HAND_APPROACH_POSE[4],
                "l_hj_pinky_1":  HAND_APPROACH_POSE[5],
                # mimic 추종 (= drive × multiplier, 결합 init 으로 snap 방지)
                "l_hj_thumb_3":  HAND_APPROACH_POSE[1] * 1.1425,
                "l_hj_thumb_4":  HAND_APPROACH_POSE[1] * 1.1425 * 0.7508,
                "l_hj_index_2":  HAND_APPROACH_POSE[2] * 1.1169,
                "l_hj_middle_2": HAND_APPROACH_POSE[3] * 1.1169,
                "l_hj_ring_2":   HAND_APPROACH_POSE[4] * 1.1169,
                "l_hj_pinky_2":  HAND_APPROACH_POSE[5] * 1.1169,
                **RIGHT_ARM_REST_JOINT_POS,
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
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # RH56F1 우측 손 굴곡 5 (thumb_2 + 4손가락_1) — tesollo pour curl/pip/dip 강성 참조.
            "rh56f1_left_flexion": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"],
                stiffness=400.0,
                damping=60.0,
            ),
            # abduction(thumb_1) — 굴곡보다 낮게(반력 교란 회피).
            "rh56f1_left_abduction": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_thumb_1"],
                stiffness=200.0,
                damping=35.0,
            ),
            # RH56F1 우측 손 mimic(원위) 6 — PhysxMimicJoint 미결합 시 원위가 흐물 → 강성 부여.
            "rh56f1_left_mimic": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
                stiffness=400.0,
                damping=60.0,
            ),
            # RH56F1 좌측 손 drive 6 (학습 비사용 → hold)
            "rh56f1_right_drive": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"],
                stiffness=30.0,
                damping=5.0,
            ),
            # RH56F1 좌측 손 mimic 추종 6 (passive)
            "rh56f1_right_mimic": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정 (env.py _setup_scene 에서 링크명 기반 개별 센서 생성)
    # Actor tip: fingertip 힘센서(실 *_force_sensor → 병합 말단 링크). RH56F1 2-마디.
    # -----------------------------------------------------------------------
    left_tip_contact_links: tuple = (
        "l_hl_thumb_4",
        "l_hl_index_2",
        "l_hl_middle_2",
        "l_hl_ring_2",
        "l_hl_pinky_2",
    )

    # env.py 가 링크명으로 직접 센서를 만들므로 아래 cfg 는 참조/호환용(미인스턴스).
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_hl_(thumb_4|index_2|middle_2|ring_2|pinky_2)",
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_hl_(thumb_2|index_1|middle_1|ring_1|pinky_1)",
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
        # grasp_v2: 단일 cup → primitives 다물체(MultiAsset, env 별 random_choice).
        # prim 이름 "Cup" 은 유지(ContactSensor 필터/센서 참조 재사용).
        spawn=_GRASP_OBJECT_SPAWN,
    )
    # visdex 물체 USD 의 rigid body prim 이름 — ContactSensor force_matrix filter 대상.
    # filter 는 실제 rigid body(/Cup/baseLink)를 가리켜야 GPU contact filter 가 작동한다
    # (Xform 루트 /Cup 은 미지원). tesollo grasp_v2 와 동일 (visdex USD 공유, baseLink).
    cup_rigid_body_name: str = "baseLink"

    # -----------------------------------------------------------------------
    # Distillation/occlusion 측정용 D435i 카메라 (right/grasp_v2 규약 동일, 좌우 공용).
    # enable_camera_probe=False 기본 → teacher 학습 경로는 카메라를 생성하지 않는다.
    # -----------------------------------------------------------------------
    enable_camera_probe: bool = False
    img_width:  int = CAMERA_IMG_WIDTH
    img_height: int = CAMERA_IMG_HEIGHT
    d_min: float = CAMERA_D_MIN
    d_max: float = CAMERA_D_MAX
    camera_crop_frac: float = CAMERA_CROP_FRAC
    tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_POS, rot=CAMERA_ROT, convention="ros"
        ),
        data_types=["rgb", "depth"],   # RGB 입력 + depth aux 재구성
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        width=CAMERA_IMG_WIDTH,
        height=CAMERA_IMG_HEIGHT,
    )

    # -----------------------------------------------------------------------
    # Distillation (RGB 입력 + depth aux 재구성). right/grasp_v2 규약 동일.
    # -----------------------------------------------------------------------
    distillation: bool = False
    num_student_observations: int = NUM_STUDENT_OBS                            # 116
    num_teacher_observations: int = NUM_OBS_BASE + len(_ACTIVE_OBJECT_NAMES)   # 124 + N_obj
    distill_excluded_object_names: tuple[str, ...] = ()
    img_aug_type: str = "rgb"
    enable_visual_dr: bool = True
    texture_root: str = _TEXTURE_ROOT
    disable_dome_light_randomization: bool = False
    disable_robot_randomization: bool = False
    aug_depth: bool = False
    aux_coeff: float = 1.0
    cam_matrix = _CAM_MATRIX
    depth_randomization_cfg_dict: dict = field(
        default_factory=lambda: depth_randomization_cfg(_CAM_MATRIX, CAMERA_D_MIN, CAMERA_D_MAX)
    )
    success_timeout: int = 60

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = LEFT_ACTUATED_JOINT_NAMES
    right_arm_joint_names: list = RIGHT_ARM_AND_GRIPPER_JOINT_NAMES
