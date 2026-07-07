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

"""환경 설정: 5g_grasp_right_v1

v7: Fabrics 팔 학습(6D palm) + per-finger lerp(5D) + sim2real 가능 obs
- Action: 11D (6D palm pose + 5D per-finger lerp)
- Observation: actor 106D / critic 143D (asymmetric)
- Episode: Grasp phase (Fabrics arm + finger 정책) + right-grip lift-wait (frozen hand)
- Contact: fingertip FT sensor (actor, real-compatible) + distal/middle sensors (critic only)
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .grasp_right_preset import (
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
_VISDEX_NAMES: tuple[str, ...] = tuple(sorted(
    _n for _n in _os.listdir(_VISDEX_ROOT)
    if _os.path.isdir(_os.path.join(_VISDEX_ROOT, _n))
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
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True    # 13×13 grid IK 사전 계산 → reset 시 lookup (랜덤화와 호환)
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
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
    obs_noise_joint_pos: float = 0.01    # joint position σ [rad]
    obs_noise_joint_vel: float = 0.05    # joint velocity σ [rad/s]
    obs_noise_body_pos:  float = 0.005   # FK body position σ [m] (palm, fingertip)
    obs_noise_cup_pos:   float = 0.015   # cup position observation σ [m]

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    cup_grasp_z_offset:  float = 0.06
    lift_success_height: float = 0.04

    # -----------------------------------------------------------------------
    # Delta palm action (pregrasp 기준 상대 오프셋)
    # action=0 → pregrasp 위치 유지, action=±1 → pregrasp ± delta
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.15   # ±0.15m per axis
    palm_delta_rot_deg: float = 20.0   # ±20° per axis

    # -----------------------------------------------------------------------
    # Reward 파라미터
    # -----------------------------------------------------------------------
    # RH56F1 shared grasp-v2 reward contract.
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
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 200
    adr_trigger_threshold: float = 0.02

    adr_custom_cfg: dict = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 60.0
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
    # 활성 물체군(spawn 순서와 일치) — per-object 로깅용 이름. env_id % N 로 배정.
    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
    # 물체 조건 feature(접근 B) lookup용 code 접두사: primitives="primitive", visdex="visdex".
    object_code_prefix: str = "visdex"
    object_feature_path: str = _os.path.join(
        _ASSETS_DIR, "object_pc_features", "openarm_right_object_code_feat_dim64.pt"
    )
    object_spawn_xy_range: float = 0.06   # ±6cm 랜덤화 (Fabrics arm 학습으로 보정 가능)

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
    # 로봇 설정 (openarm_tesollo_sensor_rl.usd: 통일 네이밍 r_aj/r_hj/r_hl, r_hl_*_tip ContactSensor 포함)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"),
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
