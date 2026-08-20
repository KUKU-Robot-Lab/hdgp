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

from .robot_profiles import PROFILES, RobotProfile

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

    # ---- 액션 스케일 (스텝당 delta, 60 Hz) ----------------------------------------
    # 팔: diff IK relative pose — pos 1cm/스텝(최대 0.6 m/s), rot 0.05 rad/스텝.
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
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    # 관절한계 위반 종료 페널티 (diff IK 는 관절한계 무방비 → 종료+페널티로 처리)
    abnormal_penalty: float = -1.0

    # ---- goal / 성공 판정 ---------------------------------------------------------
    goal_height_offset: float = 0.15         # goal = 물체 스폰 위치 + z 0.15
    success_pos_tolerance: float = 0.05      # 난이도 스케줄러 판정 (tracking_std/2, dexsuite)

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

    robot_cfg: ArticulationCfg = None  # __post_init__ 에서 프로필로 조립

    def __post_init__(self):
        profile = PROFILES[self.profile_name]
        self.robot_cfg = _build_robot_cfg(profile)
        num_joints = profile.num_arm_joints + profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D delta + 손 관절 delta
        self.action_space = 6 + profile.num_hand_joints
        # 관측 = q + qd + palm pose(7) + tips(3T) + obj pose(7) + goal(3)
        #        + 손가락별 접촉력 크기(F) + last action
        self.observation_space = (
            2 * num_joints + 7 + 3 * num_tips + 7 + 3 + num_fingers + self.action_space
        )
        # critic = 관측 + 물체 속도(6) + 난이도(1)
        self.state_space = self.observation_space + 7
        # 물체 스폰 중심을 프로필로 정렬
        self.object_cfg.init_state.pos = [
            profile.object_spawn_center[0], profile.object_spawn_center[1], profile.object_spawn_z,
        ]


@configclass
class GraspLiftTesolloRightEnvCfg(GraspLiftEnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspLiftGripperLeftEnvCfg(GraspLiftEnvCfg):
    profile_name: str = "gripper_left"
