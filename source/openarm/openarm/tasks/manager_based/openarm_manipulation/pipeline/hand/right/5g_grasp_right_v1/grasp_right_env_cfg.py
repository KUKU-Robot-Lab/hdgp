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

OpenArm(7 DOF) + Teosllo(20 DOF) 오른손 단일 컵 파지 태스크.
- 제어 방식: Geometric Fabrics (DEXTRAH 방식)
- 로봇 USD: openarm_modular_dual (Teosllo 공식 물성치 반영)
- 액션: 6D palm pose (손가락은 palm_dist 기반 자동 보간)
- 리워드: KUKA_ALLEGRO 방식 + GraspADR
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
from isaaclab.utils import configclass

import os as _os

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS

# hdgp 루트 및 assets 경로 (portable)
_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v1 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    episode_length_s: float = 10.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간 (DirectRLEnvCfg 필수 필드)
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS  # 145
    action_space: int = NUM_ACTIONS            # 6 (6D palm only, 손가락은 자동 보간)
    state_space: int = NUM_OBSERVATIONS        # critic obs = policy obs (단순화)

    # 내부 참조용 (DEXTRAH 호환)
    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    # Palm pose 워크스페이스 각도 한계 (deg): 회전 중심 ±
    max_pose_angle: float = 45.0
    # Fabrics world에서 env당 보유할 최대 장애물 슬롯 수
    # (활성 장애물 수보다 크게만 잡으면 됨. 과대 설정 시 메모리 낭비)
    fabrics_max_objects_per_env: int = 6

    # -----------------------------------------------------------------------
    # 리워드 파라미터
    # -----------------------------------------------------------------------
    # 1. hand_to_object (palm_center only, side-approach 전용)
    #    DEXTRAH: max over 6 hand points, sharpness=10.0
    #    v1: palm_dist만 사용 (side-approach에서 palm이 주 접근 방향)
    hand_to_object_weight: float = 2.0
    hand_to_object_sharpness: float = 10.0  # DEXTRAH 일치

    # 2. object_to_goal: exp(-α * dist) — 물체를 목표 위치로
    #    DEXTRAH: weight=5.0 고정, sharpness는 ADR로 15→20 증가
    object_to_goal_weight: float = 5.0       # 고정 (ADR 대상 아님, DEXTRAH 동일)
    object_to_goal_sharpness: float = 15.0   # ADR 초기값 (ADR 활성 시 덮어씀)

    # 3. lift: exp(-α * |z_obj - z_goal|) — 수직 들기
    #    DEXTRAH: weight는 ADR로 5→0 감소, sharpness=8.5 고정
    lift_weight: float = 5.0      # ADR 초기값 (ADR 활성 시 덮어씀)
    lift_sharpness: float = 8.5   # DEXTRAH 일치, 고정

    # 4. finger curl (DEXTRAH R_curl 방식)
    #    항상 활성, 음수 weight → interp_hand 이탈 패널티
    #    DEXTRAH: w_curl -0.01→-0.005 (완화), v1: 스케일 조정 -2.0→-1.0
    finger_curl_weight: float = -2.0  # ADR 초기값 (ADR 활성 시 덮어씀)

    # 5. palm orientation reward: palm +X(손바닥 법선)이 컵을 향할수록 보상
    # align ∈ [-1, 1] → weight * align ∈ [-weight, +weight]
    # 위에서 덮기(palm +X = world -Z): align ≈ -1 → 패널티
    # 옆면 접근(palm +X = palm→cup 방향): align ≈ +1 → 보상
    palm_orient_weight: float = 1.0

    # -----------------------------------------------------------------------
    # ADR (Automatic Domain Randomization)
    # DEXTRAH DextrahADR 이식 — event_manager 의존성 없음
    # increment_counter가 올라갈수록 파라미터가 initial → final로 선형 보간
    # 트리거: object_z > object_spawn_z + lift_adr_threshold (컵이 들릴 때)
    # -----------------------------------------------------------------------
    # ADR (DEXTRAH 동일 구조)
    # increment_counter 0→num_increments에 따라 파라미터 선형 보간
    # 트리거: object_z > object_spawn_z + lift_adr_threshold (컵이 들릴 때)
    enable_adr: bool = True
    adr_num_increments: int = 50
    adr_increment_interval: int = 200
    adr_trigger_threshold: float = 0.1   # lift 비율 10% 이상이면 increment
    lift_adr_threshold: float = 0.05     # 5cm 이상 들려야 "lift 성공"으로 집계
    # ADR 파라미터 (initial → final), DEXTRAH 패턴:
    #   lift_weight    : 5→0  (초기에 강하게, 점차 감소 → 정밀 배치로 전환)
    #   goal_sharpness : 15→20 (점점 정밀 요구)
    #   curl_weight    : -2→-1 (패널티 완화)
    adr_custom_cfg: dict = field(default_factory=lambda: {
        "reward_weights": {
            "lift_weight":        (5.0, 1.0),    # 최소 1.0 유지 (복구 gradient 보장)
            "finger_curl_weight": (-2.0, -1.0),  # DEXTRAH: -0.01→-0.005 스케일 조정
        },
        "sharpness": {
            "object_to_goal_sharpness": (15.0, 20.0),  # DEXTRAH: 15→20
        },
    })

    # -----------------------------------------------------------------------
    # 손가락 보간 파라미터 (reward 게이트 아님)
    # -----------------------------------------------------------------------
    # approach_trigger_dist: palm_dist 기반 손가락 보간 t 계산용
    #   t = 1 - clamp(palm_dist / approach_trigger_dist, 0, 1)
    #   palm_dist < this → 손가락이 grasp_pose 쪽으로 닫힘
    #   0.20m: side-approach에서 손바닥이 컵에 닿는 시점 ≈ palm_dist 0.15~0.20m
    # test7 분석: palm_dist 최소 0.132m (컵 표면까지 0.087m 공백)
    # trigger=0.20: t=0.34 (34% 닫힘) → 잡을 수 없음
    # trigger=0.40: t=0.67 (67% 닫힘) → 훨씬 확실한 접촉
    approach_trigger_dist: float = 0.40

    # h2o 타겟 z offset: cup root frame → 실제 파지 중심
    # cup root가 바닥보다 0.015m 아래, 파지 중심은 root에서 0.056m 위
    object_grasp_z_offset: float = 0.056

    # -----------------------------------------------------------------------
    # 컵 기울기 종료 (displacement_penalty 제거: h2o 모순 유발)
    # -----------------------------------------------------------------------
    # cup_tipping: 컵 기울기가 max_tilt_deg 초과 → 에피소드 종료
    # 극단적 밀기(넘어뜨리기) 방지. 접근 중 살짝 밀리는 것은 불가피하므로 허용
    # test7: 60°가 너무 엄격 — 접근 중 컵이 살짝 기울어도 바로 종료
    # 90°로 완화: 손바닥 접근/밀기 중 일부 기울기 허용
    cup_tipping_max_deg: float = 90.0

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.55   # palm 시작 x≈0.46 → 컵을 palm 앞에 배치, +x+y 방향 접근
    object_spawn_y_center: float = -0.15  # palm workspace y∈[-0.50,-0.05] 중앙 근처; -Y 방향 접근
    object_spawn_z: float = 0.30        # 컵 기준: 테이블 top(0.25m) + 0.05m 여유 (컵 높이 0.091m)
    object_spawn_xy_range: float = 0.08  # ±0.08m 균등 분포
    # 물체 활성 플래그
    enable_cup: bool = True
    enable_primitives: bool = False

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            # 과도한 선할당을 줄여 DRAM/VRAM 압박 완화
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**22,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    # -----------------------------------------------------------------------
    # 씬 설정
    # -----------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정 (kinematic=True → 고정, 물리적 충돌만 처리)
    # robot pos=[0,0,-0.25] → table pos=[0.5725, 0.003, -0.25] → table top ≈ z=0.13m
    # cup spawn_z=0.38은 table top(0.13) + 컵 높이(0.25) 기준
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.235],   # DEXTRAH table 기준: top ≈ z=0.25m
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # 고정 테이블 (물리 반응 없음)
                disable_gravity=True,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇 설정
    # openarm_modular_dual.usd:
    #   - OpenArm bimanual + Teosllo right hand
    #   - Teosllo 공식 물성치(stiffness/damping)가 USD에 반영됨
    #   - OpenArm arm: Isaac Lab에서 stiffness/damping 설정 (정밀 제어)
    #   - Teosllo hand: USD 물성치 유지 (ImplicitActuatorCfg에서 override 안 함)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_modular_dual/openarm_modular_dual.usd"),
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
                # 오른팔: FK 탐색 결과 → palm ≈ [0.463, -0.311, 0.431]
                # 컵(y=-0.15)에서 0.16m 뒤(-y)에서 시작, +y 방향으로 직진 파지
                # j7=0.0: Fabrics default_config와 일치 (j7=1.571은 palm을 y=-0.01로 밀어냄)
                "openarm_right_joint1":  0.5,
                "openarm_right_joint2":  0.1,
                "openarm_right_joint3":  0.4,
                "openarm_right_joint4":  0.6,  # j4=0.6: palm z≈0.301m (table+5cm, FK검증)
                "openarm_right_joint5": -0.2,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                # 오른손: fully open (thumb opposition=0, 컵 소환 영역 관통 방지)
                "rj_dg_1_1": 0.0, "rj_dg_1_2":  0.0, "rj_dg_1_3": 0.0, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.0, "rj_dg_2_3": 0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.0, "rj_dg_3_3": 0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.0, "rj_dg_4_3": 0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0, "rj_dg_5_3": 0.0, "rj_dg_5_4": 0.0,
                # 왼팔: 학습 제외, 고정 자세
                "openarm_left_joint1": -0.5,
                "openarm_left_joint2": -0.5,
                "openarm_left_joint3":  0.6,
                "openarm_left_joint4":  0.7,
                "openarm_left_joint5":  0.0,
                "openarm_left_joint6":  0.0,
                "openarm_left_joint7": -1.0,
                # 왼쪽 그리퍼: 0 (openarm_modular_dual.usd 기준 — Teosllo 없음)
                "openarm_left_finger_joint1": 0.0,
                "openarm_left_finger_joint2": 0.0,
            },
        ),
        actuators={
            # OpenArm arm: Isaac Lab에서 stiffness/damping 설정
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # Teosllo right hand: USD 공식 물성치 유지 (stiffness/damping = None → USD 값)
            "tesollo_right_hand": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_.*"],
                stiffness=None,
                damping=None,
            ),
            # 왼쪽 그리퍼: 고정
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # 컵 설정 (단일 컵 파지)
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, -0.15, 0.30],   # 초기 USD 위치 (실제 spawn은 _reset_idx에서 결정)
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup_bead/cup.usd"),
            scale=(1.0, 1.0, 1.2),
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

    # -----------------------------------------------------------------------
    # Primitive 설정 (초기 위치는 scene 밖, reset 시 랜덤 배정)
    # -----------------------------------------------------------------------
    primitive_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Primitive",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -10.0],   # 초기에는 scene 밖 아래에 배치
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "primitives/USD/small_5_cyl/small_5_cyl.usd"),
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

    # -----------------------------------------------------------------------
    # Actuated joint names (오른팔 + 오른손, 27 DOF)
    # -----------------------------------------------------------------------
    actuated_joint_names: list = (
        [f"openarm_right_joint{i}" for i in range(1, 8)] +  # arm 7 DOF
        [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]  # hand 20 DOF
    )

    # 왼팔+왼쪽 그리퍼 joint names (고정, openarm_modular_dual.usd 기준)
    left_arm_joint_names: list = (
        [f"openarm_left_joint{i}" for i in range(1, 8)] +
        ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
    )
