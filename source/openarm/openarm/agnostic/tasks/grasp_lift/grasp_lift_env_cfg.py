"""robot-agnostic grasp-lift 환경 설정.

태스크: 고정 초기 자세 → 접근 → 파지 → 스폰 지점 위 +15cm goal 로 들어 유지.
로봇 종속 정보는 전부 RobotProfile 에서 온다(robot_profiles.py 참조).

레퍼런스: IsaacLab Dexsuite Kuka-Allegro lift —
  · 보상 5항+정규화 2항, 곱셈 게이트는 접촉 1개
  · lift 항 없음: 중력 커리큘럼이 대체. 여기서는 **물체 반중력 보상력**으로 구현
    (PhysX 중력은 글로벌이라 per-env 불가 → 물체에 m·g·(1−난이도) 상향력 인가 =
    per-env 유효 중력. dexsuite 의 글로벌 보간보다 커리큘럼 분해능이 높다)
  · 커리큘럼: 성공 기반 per-env 난이도(0~10) → 중력 · 스폰 범위 보간
"""

from __future__ import annotations

import os as _os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.envs import mdp as _mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from .robot_profiles import PROFILES, RobotProfile

_FRICTION = 0.75


@configclass
class GraspLiftEventCfg:
    """물리 재질 — startup 1회. 값은 절대값이다(배율 아님).

    ★씬 기본(SimulationCfg.physics_material)만으로는 부족하다: 로봇·컵 콜라이더는 각자
    재질을 갖고, PhysX 결합이 average 라 한쪽만 올리면 실효 μ 가 중간값이 된다.
    mode="startup" 인 이유 = 이 term 은 CPU 텐서로 버킷을 만들어 초기화 때만 쓰라고
    상류가 명시한다(events.py:167-170). 고정값이라 reset 마다 다시 걸 이유도 없다.
    """

    robot_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    object_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

_HDGP_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *([".."] * 6)))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


def _build_robot_cfg(profile: RobotProfile) -> ArticulationCfg:
    """프로필 → ArticulationCfg. 조인트 이름은 전부 프로필에서."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, profile.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                # 5.0 → 1.0: 관통 시 밀어내는 속도가 크면 충격량이 폭증한다
                # (fabric 트랙 실측: 전형 13~20N 인데 스파이크 7218N).
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ★True (08.21 자산 재생성으로 해소). 구 자산에서는 홈 자세의 상시 접촉쌍이
                #   반발력을 만들어 유휴 480스텝에 palm 이 180mm 표류했으나, 신규 USD 가
                #   감사로 찾은 6쌍을 PhysicsFilteredPairsAPI 로 필터하고 convexHull(64-vert
                #   외접) → convex_decomposition 으로 되돌려 유령 접촉을 없앴다.
                #   실측: 유휴 480스텝 palm 드리프트 0.00mm·|qd| 0.00,
                #        full-grip 손관절합 24.1(OFF) → 18.8(ON) = 손가락 겹침 차단.
                enabled_self_collisions=True,
                # ★P-9 실측(2048env)으로 16 확정: 32 는 fps 를 25% 깎는데(11.9k→8.9k)
                #   파지 품질이 나아지지 않았다. 오히려 접촉력이 더 높았다
                #   (solver32 Fa 28N/Fb 25N vs solver16 Fa 18N/Fb 8N) = 상호침투를 더
                #   밀어내고 있었다는 뜻. self-collision + convex_decomposition 자산이
                #   이미 관통을 막으므로 32 는 불필요한 비용이다.
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos=dict(profile.init_joint_pos),
        ),
        actuators={
            name: ImplicitActuatorCfg(**spec)
            for name, spec in profile.actuator_specs.items()
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class GraspLiftEnvCfg(DirectRLEnvCfg):
    # ---- 로봇 선택 (서브클래스가 덮어씀) ----------------------------------------
    profile_name: str = "tesollo_right"

    # ---- 시뮬레이션: 물리 120 Hz / 정책 60 Hz ------------------------------------
    episode_length_s: float = 8.0            # 480 스텝
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # ★씬 기본 물리 재질. 지정이 없으면 IsaacLab 기본 μ=0.5 로 돌고, PhysX 결합모드가
        #   **average** 라 컵만 0.75 로 올려도 실효는 (0.75+0.5)/2 = 0.625 다.
        #   → 씬 기본 + 로봇 + 컵 + 테이블 **네 곳 전부** 0.75 여야 한다.
        #   파지 용량 ∝ (μcosα + sinα): 0.565 → 0.814 (×1.44).
        #   restitution 0.0 — grasp_v1 의 (1.0,1.0) 은 "중립 1.0배" 주석과 달리 **절대값**이라
        #   그대로 복사하면 컵이 완전탄성이 된다(randomize_rigid_body_material 은 배율 아님).
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.75, dynamic_friction=0.75, restitution=0.0,
        ),
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True,
    )

    # ---- 공간 (프로필에서 __post_init__ 이 계산) ----------------------------------
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # ---- Fabrics (grasp_sensor 검증값 승계) ----------------------------------------
    # ★정책 스텝(1/60)당 fabrics_dt × fabric_decimation 만큼 fabric 시간이 흐른다.
    #   1/120 × 2 = 1/60 으로 실시간과 일치시킨다(1/60 × 2 는 2배속이 된다).
    fabrics_dt: float = 1.0 / 120.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 20.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # ★palm leash 는 제거됐다(08.22) — 정책이 팔 목표에 대해 전권을 갖는다.
    #   목표의 유일한 상한은 아래 워크스페이스 박스(profile.palm_box_*)다.
    #   근거는 grasp_lift_env.py `_pre_physics_step` 주석 참조.

    # ---- 액션 스케일 (스텝당 delta, 60 Hz) ----------------------------------------
    # 팔: palm 절대 목표 누산 delta — pos 1cm/스텝(최대 0.6 m/s), rot 0.05 rad/스텝.
    arm_pos_scale: float = 0.01
    arm_rot_scale: float = 0.05
    # 손: relative joint position (dexsuite RelativeJointPositionAction scale=0.1 동일)
    hand_joint_scale: float = 0.1

    # ---- 보상 (dexsuite lift 가중치 그대로) ---------------------------------------
    reaching_weight: float = 1.0
    reaching_std: float = 0.1
    contact_weight: float = 0.5
    contact_force_threshold: float = 1.0     # [N] dexsuite 동일
    tracking_weight: float = 2.0
    tracking_std: float = 0.1
    success_weight: float = 10.0
    success_std: float = 0.05
    # 리프트 부분 진척 — gate 곱. goal_height_offset(0.15m)에서 포화.
    # dexsuite 는 이 항을 뺐지만 그 전제(goal=물체 근처 랜덤 pose)가 우리와 다르다.
    lift_weight: float = 1.5
    # 전도 페널티 — 20° 여유대(정상 파지 흔들림 무징계) 초과분 비례, 최대 −0.5.
    # 60° 초과 = 사실상 넘어짐 → **truncation 으로 env 전체 리셋**(08.22, 사용자 지시).
    # 컵 단독 리스폰은 폐기 — 텔레포트 전이가 학습 데이터를 오염시키고, 손 위로
    # 겹쳐 소환되는 결함이 있었다. value_bootstrap(yaml)과 반드시 짝이어야 한다:
    # bootstrap 없는 truncation 은 termination 과 같아져 회피 학습(agn_test2)이 재발한다.
    tilt_penalty_weight: float = -0.5
    tilt_free_deg: float = 20.0
    tilt_reset_deg: float = 60.0
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    # 관절한계 위반 종료 페널티 (diff IK 는 관절한계 무방비 → 종료+페널티로 처리)
    abnormal_penalty: float = -1.0

    # ---- 스폰 높이: 단일 소스 -------------------------------------------------------
    # ★이중 패딩 재발 차단(08.21). 프로필이 완성값(0.282)을 들고 env 가 +5mm 를 또 얹어
    #   컵이 정착고보다 9.7mm 높이 스폰됐다 → 정지 상태 height_delta −9.7mm, lift 보상의
    #   첫 9.7mm 데드존, 실효 목표 159.7mm. 게다가 두 프로필이 0.282/0.297 로 갈렸다.
    #   이제 여기 세 값에서만 파생한다(프로필의 object_spawn_z 필드는 삭제됨).
    table_surface_z: float = 0.200           # env.usd top_plate 상면(점군 실측)
    object_origin_offset_z: float = 0.0773   # cup_big USD 원점 ↔ 바닥
    object_spawn_pad: float = 0.005          # 스폰 침투 반동 방지
    # 위 셋에서 __post_init__ 이 파생시키는 캐시. 직접 쓰지 말 것(단일 소스 유지).
    object_spawn_z: float = 0.0

    # ---- goal / 성공 판정 ---------------------------------------------------------
    goal_height_offset: float = 0.15         # goal = 물체 스폰 위치 + z 0.15
    # dexsuite 규약은 **success 항의 pos_std/2**(dexsuite_env_cfg.py:432). 우리 success_std 가
    # 0.05 이므로 0.025 다. 구 값 0.05 는 참조를 tracking_std/2 로 잘못 적은 것이라 판정선이
    # 보상선보다 5배 헐거웠다 — 정책이 못 하는 난이도로 계속 승급했다.
    success_pos_tolerance: float = 0.025
    # 이전 12,000ep 런과의 연속성 비교 전용(보상·커리큘럼 미사용, 로깅만)
    success_pos_tolerance_loose: float = 0.05

    # ---- 커리큘럼 (per-env 난이도 0~10) --------------------------------------------
    curriculum_max_level: int = 10
    # 물체 반중력 보상력: 유효 중력 = g × max(min_frac, level/max).
    # min_frac 0.15 = level 0 에서도 컵이 테이블에 앉아 마찰로 고정되게(완전 무중력이면
    # 수직항력 0 → 부유 표류. probe 실측: 120스텝에 컵이 100mm 떠다님).
    gravity_min_frac: float = 0.15
    # 스폰 xy 반경: 초기 → 최종 보간.
    # ★final 은 프로필 스폰 중심 기준 박스 전체가 "홈 팔 quiet 영역" 안이어야 한다
    #   (probe_spawn_map/probe_solve_v1_home 실측 — 관통은 링크 원점 거리로 안 보임).
    spawn_range_initial: float = 0.02
    spawn_range_final: float = 0.08

    # ---- 종료 ----------------------------------------------------------------------
    object_out_of_bounds_xy: float = 0.35    # 스폰 중심 기준 |Δxy| 초과 시 종료
    object_min_z: float = 0.15               # 테이블 아래로 떨어짐
    # 팔 관절이 (soft) 한계의 이 비율을 넘으면 abnormal 종료
    arm_joint_limit_frac: float = 0.99

    # ---- 씬 --------------------------------------------------------------------------
    # 실기 환경 USD (테이블 상면 z 0.200, 기둥/받침/바닥판 포함 전부 충돌체).
    # 원점을 로봇 base link 원점(env 원점)에 그대로 붙인다 — 사용자 지정 08.20.
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "env/usd/env.usd"),
        ),
    )
    # 단일 물체(cup_big, 질량 0.134kg = pour 실컵)로 시작. 다물체는 Phase 3.
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, -0.20, 0.297]),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup", "cup_big_rl.usd"),
            activate_contact_sensors=True,
            mass_props=MassPropertiesCfg(mass=0.134),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            # ★물체에도 solver/depenetration 을 명시한다 — 빠지면 PhysX 기본(4회)이라
            #   파지 조임 중 손끝이 컵 벽을 파고든다(사용자 영상 08.20).
            #   값은 grasp_v1·grasp_lift_fabric 과 동일.
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )

    # 물체 rigid body 는 USD 안 baseLink 에 있다(cup_big_rl.usd 규약)
    object_contact_filter: tuple = ("/World/envs/env_.*/Object/baseLink",)

    # 물리 재질 이벤트(로봇·컵). 테이블은 scene 자산이 아니라 정적 프림이라
    # env 가 clone 전에 bind_physics_material 로 직접 건다.
    events: GraspLiftEventCfg = GraspLiftEventCfg()
    surface_friction: float = _FRICTION

    robot_cfg: ArticulationCfg = None  # __post_init__ 에서 프로필로 조립

    def __post_init__(self):
        profile = PROFILES[self.profile_name]
        self.robot_cfg = _build_robot_cfg(profile)
        num_joints = profile.num_arm_joints + profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D delta + 손 관절 delta
        # 액션 = palm 6D + **자유** 손관절(고정 관절은 정책이 안 건드린다)
        self.action_space = 6 + profile.num_hand_joints - profile.num_locked_hand_joints
        # 관측 = q + qd + palm pose(7) + tips(3T) + obj pose(7) + goal(3)
        #        + 손가락별 접촉력 크기(F) + last action
        self.observation_space = (
            2 * num_joints + 7 + 3 * num_tips + 7 + 3 + num_fingers + self.action_space
        )
        # critic = 관측 + 물체 속도(6) + 난이도(1)
        self.state_space = self.observation_space + 7
        # 스폰 높이는 cfg 세 값에서만 파생(단일 소스) — 프로필은 xy 만 준다
        self.object_spawn_z = (
            self.table_surface_z + self.object_origin_offset_z + self.object_spawn_pad)
        self.object_cfg.init_state.pos = [
            profile.object_spawn_center[0], profile.object_spawn_center[1], self.object_spawn_z,
        ]




@configclass
class GraspLiftTesolloRightEnvCfg(GraspLiftEnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspLiftGripperLeftEnvCfg(GraspLiftEnvCfg):
    profile_name: str = "gripper_left"
