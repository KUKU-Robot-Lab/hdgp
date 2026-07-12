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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBS_BASE, NUM_ACTIONS, NUM_CRITIC_OBS_BASE
from .grasp_right_preset import (
    HAND_APPROACH_POSE,
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")

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
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = _VISDEX_NAMES


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
    # 07.13 재구성(사용자 결정 "tesollo는 되는데 여긴 안 됨"): hand = 시너지+래칫.
    # False = 시너지(eigengrasp) p* + 전진-only 래칫 (tesollo 91fb455·d250ae5 이식,
    # tesollo test12 성공 경로). dextrah11 실증: PCA 절대 타겟(양방향, True)은
    # "열림 포화" 국소최적(f1=-0.99, 접촉 파도 3회 소멸·리프트 0) — tesollo test8
    # '손 펴기'와 동일 병리. True로 되돌리면 DEXTRAH PCA fabric 경로.
    use_hand_fabric:            bool  = False
    hand_mode:                  str   = "pca"   # use_hand_fabric=True일 때만 사용 ("pca"|"direct")
    max_pose_angle:             float = 45.0   # DEXTRAH README teacher 레시피 45°. palm rpy 공칭(180,0,90)±45 — 손바닥 방향을 action 공간에서 constrain(천장 지향 구조 차단). tesollo 0fa6fb3 정렬
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True    # 13×13 grid IK 사전 계산 → reset 시 lookup (랜덤화와 호환)
    # E3 top-down (07.12, probe sweep B/D 검증): palm이 물체 상공(스폰+0.08)에서
    # 아래보기로 시작 — 스폰·하강 중 물체 무교란(불도저 해소), scripted 파지-리프트 재현.
    # 측면 세팅(-0.07/-0.08/-0.15, ex=90)은 도주로 열림으로 파지 불성립(analysis.md).
    pregrasp_offset_x:     float = 0.0
    pregrasp_offset_y:     float = -0.01
    pregrasp_offset_z:     float = 0.08
    # r_aj_7(손목) bias: top-down 하강 심도 확보 (0→palm floor 0.339, 0.5→0.324,
    # 0.8은 xy 정렬 붕괴). probe sweep B/D에서 0.4~0.5가 스윗스팟.
    pregrasp_r_aj7_bias:   float = 0.5
    # settle 종료 시 안착된 물체 위치로 anchor xy 재정렬 (drop-settle 롤링 보정).
    reanchor_after_settle: bool  = True
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

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
    # 07.12 DEXTRAH 절대 타겟 정합: delta는 anchor 기준 "도달 박스"(좁히면 goal 도달
    # 불가 — 0.03 실수의 교훈). 속도 제한(slew)은 제거 — 측면 밀침 대응 추가물이었으나
    # E3(top-down)에선 테이블이 밀침을 흡수해 불필요 + 행동→효과 지연이 credit
    # assignment 를 희석(dextrah10 노이즈 발산). 실효 속도는 원본처럼 fabric 감쇠가 제한.
    palm_delta_xyz:     float = 0.35   # anchor±0.35m 도달 박스 (workspace clamp가 최종 한계)
    palm_delta_rot_deg: float = 45.0   # anchor±45° — DEXTRAH max_pose_angle 45 정합 (기존 90)

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
    # palm orientation: DEXTRAH 4항엔 손목 방향 제약이 없어 손바닥이 임의(천장) 방향으로
    # 수렴. palm 법선(로컬 +X → world)이 palm→물체 방향과 정렬되도록 보조 shaping.
    # w·exp(s·(align−1)): align=1(완전 정렬)→w, align=−1(반대)→w·exp(−2s). weight는
    # object_to_goal(5.0)의 0.2배로 통제(reward-audit ACCEPT: local-min·hacking 방지).
    # DEXTRAH 원본 재현: palm_orient 비활성(0.0). 이 항은 DEXTRAH 4항에 없는 우리 추가물로,
    # lstm_test6에서 "비접촉 파밍이 접촉보다 +66 유리"한 회피 local min을 만든 주범(분석 확증).
    # 원래 목적(손바닥 천장 방지)은 max_pose_angle 45° constrain(DEXTRAH README 해법)이 담당.
    palm_orient_weight:       float = 0.0
    palm_orient_sharpness:    float = 3.0

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
    # 래칫 전진 허용 거리 (hand_to_object_err = palm+5tip MAX 거리 기준).
    # E3 실측: anchor 0.137 / 하강 후 ~0.10-0.12 → 0.12 = 유의미하게 내려간 뒤에만 감김.
    finger_close_dist_gate: float = 0.12
    # synergy 접촉 동결(g3/g4): 접촉 시 조임 멈춤 → 파지력 약화(grip 0.90 정체).
    # False=동결 제거 → 손가락이 물체를 계속 조임(물리 collision이 관통/형상적응 담당, DEXTRAH식 파지력).
    # primitives 복귀는 True. 기본 False(파지력 확보).
    synergy_freeze_enable: bool = False
    settle_steps: int = 25  # 다물체 drop-settle: episode 초기 N step 손가락 폐쇄 억제 → 물체 낙하 안착

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

    # contact 로깅 거리 게이트(07.11): net_forces 센서는 물체/테이블 구분 불가 —
    # dextrah6b에서 grip 2.2 전부가 테이블·자기 접촉으로 판명(probe). 손끝이 물체
    # 이 거리 이내일 때만 접촉으로 카운트해 진단 지표를 물체 접촉으로 정화.
    contact_object_dist_gate: float = 0.06   # m

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
                "r_aj_1":  0.5,
                "r_aj_2":  0.1,
                "r_aj_3":  0.4,
                "r_aj_4":  0.60,
                "r_aj_5": -0.2,
                "r_aj_6":  0.0,
                "r_aj_7":  0.0,
                # RH56F1 우측 손 drive (approach pose)
                "r_hj_thumb_1":  HAND_APPROACH_POSE[0],
                "r_hj_thumb_2":  HAND_APPROACH_POSE[1],
                "r_hj_index_1":  HAND_APPROACH_POSE[2],
                "r_hj_middle_1": HAND_APPROACH_POSE[3],
                "r_hj_ring_1":   HAND_APPROACH_POSE[4],
                "r_hj_pinky_1":  HAND_APPROACH_POSE[5],
                # mimic 추종 (= drive × multiplier, 결합 init 으로 snap 방지)
                "r_hj_thumb_3":  HAND_APPROACH_POSE[1] * 1.1425,
                "r_hj_thumb_4":  HAND_APPROACH_POSE[1] * 1.1425 * 0.7508,
                "r_hj_index_2":  HAND_APPROACH_POSE[2] * 1.1169,
                "r_hj_middle_2": HAND_APPROACH_POSE[3] * 1.1169,
                "r_hj_ring_2":   HAND_APPROACH_POSE[4] * 1.1169,
                "r_hj_pinky_2":  HAND_APPROACH_POSE[5] * 1.1169,
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
            # RH56F1 우측 손 굴곡 5 (thumb_2 + 4손가락_1) — tesollo pour curl/pip/dip 강성 참조.
            "rh56f1_right_flexion": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"],
                stiffness=400.0,
                damping=60.0,
            ),
            # abduction(thumb_1) — 굴곡보다 낮게(반력 교란 회피).
            "rh56f1_right_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_thumb_1"],
                stiffness=200.0,
                damping=35.0,
            ),
            # RH56F1 우측 손 mimic(원위) 6 — PhysxMimicJoint 미결합 시 원위가 흐물 → 강성 부여.
            "rh56f1_right_mimic": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
                stiffness=400.0,
                damping=60.0,
            ),
            # RH56F1 좌측 손 drive 6 (학습 비사용 → hold)
            "rh56f1_left_drive": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"],
                stiffness=30.0,
                damping=5.0,
            ),
            # RH56F1 좌측 손 mimic 추종 6 (passive)
            "rh56f1_left_mimic": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
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
    right_tip_contact_links: tuple = (
        "r_hl_thumb_4",
        "r_hl_index_2",
        "r_hl_middle_2",
        "r_hl_ring_2",
        "r_hl_pinky_2",
    )

    # env.py 가 링크명으로 직접 센서를 만들므로 아래 cfg 는 참조/호환용(미인스턴스).
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_(thumb_4|index_2|middle_2|ring_2|pinky_2)",
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_(thumb_2|index_1|middle_1|ring_1|pinky_1)",
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

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
