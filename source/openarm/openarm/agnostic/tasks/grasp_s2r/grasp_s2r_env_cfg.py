"""grasp_s2r — fabric 제어 기반 파지·리프트·이송 환경 설정.

태스크: 고정 홈 → 제자리 파지 → 리프트 → **수평 이동 포함 목표 지점 이송** → 정지.

계보:
  · 제어 스택(Fabrics 팔 + 관절공간 시너지 손) = `agnostic/tasks/grasp_sensor`
  · 액션 규약·보상 8항 = `tesollo/right/grasp_v1` (grasp→lift→stabilize 98% 이력)
  · 이송 2항(transfer·stay)과 성공 재정의는 이 트랙 신설

grasp_v1 과의 결정적 차이: grasp_v1 은 접촉 래치가 걸리면 팔 지령을 **스크립트**
(z 램프)로 대체했다. 여기서는 그 오버라이드를 이식하지 않는다 — 래치는 보상 단계를
여는 신호로만 쓰고, 팔은 처음부터 끝까지 정책이 fabric 을 통해 제어한다.

로봇 종속 정보는 전부 `robot_profiles.RobotProfile` 에서 온다.
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

# DEXTRAH Kuka EventCfg 값.
_FRICTION = 1.0

_HDGP_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *([".."] * 6)))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspS2REventCfg:
    """도메인 랜덤화 — 전 term `mode="reset"`, 공칭 파라미터에서는 전부 항등.

    ADR 은 이 트랙에서 **끄고 시작**한다(과제 성립 확인이 먼저). 켤 때 여기가 확장
    지점이라 term 은 미리 걸어 둔다 — 값만 범위로 바꾸면 된다.

    ★재질 term 의 값은 **절대값**이고(배율 아님), 관절/질량 term 은 `operation="scale"`
      이라 배율이다. 같은 파일 안에서 의미가 다르니 주의.
    """

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
    object_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
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
    object_scale_mass = EventTermCfg(
        func=_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _build_robot_cfg(profile: RobotProfile,
                     enable_self_collisions: bool) -> ArticulationCfg:
    """프로필 → ArticulationCfg. 조인트 이름은 전부 프로필에서 온다."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, profile.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                # ★접촉력 스파이크가 보이면 되돌릴 1순위(구 값 1.0).
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=enable_self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
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
class GraspS2REnvCfg(DirectRLEnvCfg):
    # ---- 로봇 선택 (서브클래스가 덮어씀) --------------------------------------------
    profile_name: str = "tesollo_right"

    # ---- 시뮬레이션: 물리 120 Hz / 정책 60 Hz ---------------------------------------
    episode_length_s: float = 10.0           # 600 스텝
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )
    # 단일 물체라 replicate_physics=True 가 맞다(False 는 MultiAsset 규약).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True,
    )

    # ---- 공간 (프로필에서 __post_init__ 이 계산) ------------------------------------
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # ---- Fabrics -------------------------------------------------------------------
    # 정책 스텝 1/60 s 당 fabric 시간 1/30 s = 2배속 (DEXTRAH Kuka 원본 배선).
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 10.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # ★팔 PD 속도 피드포워드. 0 을 넣으면 감쇠항 kd·(0−q̇) 이 참조 궤적의 움직임을
    #   반대로 밀어 err ≈ (kd/kp)·q̇ 의 상시 지연이 생긴다(실측 −33% 관절오차).
    fabric_velocity_ff_scale: float = 1.0
    hand_velocity_ff_scale: float = 1.0
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = True
    enable_self_collisions: bool = True

    # ---- 액션: palm 6D **홈 기준 델타** + 손 시너지 ----------------------------------
    # ★★grasp_v1 규약. `palm = 홈 + scale(a, −delta, +delta)` 이므로 **a=0 이 홈**이다.
    #   grasp_sensor 의 절대 매핑(`a=0` = 박스 중심)은 σ=1.0(저장소 공통 sigma_init)과
    #   곱해지면 매 스텝 작업공간 전역에서 목표를 재추첨해 접근이 랜덤워크가 된다
    #   (08.27 실측: 클램프 전 요청량 0.33~0.36 m/step 상시 포화). 델타 규약은 탐색을
    #   홈 주변 유계 오프셋으로 묶어 이 문제를 구조적으로 없앤다.
    # ★y 만 범위가 큰 이유: 홈 y −0.38 → 컵 y −0.20 을 액션으로 덮어야 한다.
    palm_delta_xyz: tuple[float, float, float] = (0.15, 0.35, 0.15)
    palm_delta_rot_deg: float = 20.0
    # ★지령 변화율 상한. grasp_v1 에는 없던 항목이다(0.0 = 완전 v1 재현).
    #   08.27 실측: 0.1 → 0.05 로 내리자 지령↔실제 간격 155 → 101mm(−35%),
    #   abnormal 변화 없음. 델타 규약은 연속 스텝 간 최대 점프가 2×delta 라
    #   이 상한이 그 과도를 흡수한다.
    palm_cmd_rate_limit_m: float = 0.05
    palm_cmd_rate_limit_rot_deg: float = 7.5

    # ---- 손: 관절공간 시너지 ---------------------------------------------------------
    # 액션은 **절대 폐쇄도 목표**[0,1] 이고 아래는 그 목표를 향한 **변화율 상한**이다
    # (속도 명령이 아니다 — 속도로 두면 탐색 노이즈 평균만으로 완전 폐쇄되고 못 되돌린다).
    synergy_close_speed: float = 0.005
    # ★★감쌈을 만드는 메커니즘. 원위·팁이 닿은 손가락의 `_3`/`_4` 만 정지시켜 컵 형상에
    #   드리워지게 한다. 끄면 핀치가 된다(grasp_v1 실증: full_envelope 0.176 → 0.035).
    synergy_contact_freeze: bool = True
    # 엄지 독립 · 4지는 채널별 평균으로 묶는다 → "특정 손가락만 안 닫힘"이 액션 공간에서
    # 표현 불가(3지 국소최적 차단).
    couple_four_fingers: bool = True

    # ---- 파지 기하 ---------------------------------------------------------------
    # ★대향축·반경 상수는 08.27 에 제거됐다. 접근 항이 이제 **손 자신의 대향 중점**과
    #   컵 사이 거리를 쓰므로(env `cage_dist`) 물체 반경이 필요 없다 —
    #   구 수식은 대향축을 접근방향의 90° 회전으로 잡아 좌/우 부호가 임의였고,
    #   그래서 엄지 목표가 실제 엄지의 반대편에 놓여 엄지가 걸렸다(사용자 GUI 관찰).
    object_grasp_z_offset: float = 0.03      # 물체 원점 ↔ 파지 높이

    # ---- 접촉 판정 -------------------------------------------------------------------
    contact_force_threshold: float = 1.0     # N — 접촉으로 셀 최소 힘
    contact_force_max: float = 10.0          # N — obs 정규화 포화점
    joint_pos_err_max: float = 1.2           # rad — obs 정규화

    # ---- 래치 (보상 단계 표시 전용 — 팔 지령을 덮지 않는다) --------------------------
    # ★★grasp_v1 의 `torch.where(is_lift, _lift_palm, palm_pose)` z 램프 오버라이드는
    #   **이식하지 않는다**. 래치는 lift/transfer 보상을 여는 신호일 뿐이고, 팔은
    #   처음부터 끝까지 정책이 fabric 을 통해 제어한다.
    lift_start_min_grip_fingers: int = 3
    grasp_ready_hold_steps: int = 8

    # ---- 목표(goal) — 수평 이동 포함 -------------------------------------------------
    # goal = 물체 **정착 위치** + offset. 스폰점 기준이면 패딩이 이중으로 실린다.
    goal_offset_xyz: tuple[float, float, float] = (0.0, 0.20, 0.15)
    goal_pos_tolerance: float = 0.025        # 성공 반경
    goal_pos_tolerance_loose: float = 0.05   # 연속성 비교 로깅 전용
    stay_hold_steps: int = 60                # 1초 — stay 항이 만점이 되는 유지 시간
    lift_height_ref: float = 0.10            # lift 항 높이 정규화 기준
    lift_success_height: float = 0.04        # "들렸다" 판정
    success_tilt_max_deg: float = 5.0
    stable_lin_vel: float = 0.04
    stable_ang_vel: float = 0.5

    # ---- 보상 가중치 (grasp_v1 8항 + 이송 2항) ---------------------------------------
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    grasp_weight: float = 12.0
    grasp_envelope_credit: float = 0.55
    lift_weight: float = 30.0
    lift_envelope_mix: float = 0.6
    transfer_weight: float = 15.0
    transfer_sharpness: float = 6.0
    stay_weight: float = 8.0
    stabilize_weight: float = 10.0
    stability_weight: float = 1.0
    success_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    wrap_retention_weight: float = -6.0
    action_smooth_weight: float = -0.02
    cup_disp_tolerance: float = 0.025        # 접근 중 허용 밀림
    cup_disp_penalty: float = 25.0
    cup_tilt_free_deg: float = 8.0
    cup_tilt_penalty: float = 0.08
    disp_falloff: float = 0.16               # lift·success 에 곱하는 밀림 감쇠 반경
    upright_sharpness: float = 5.0

    # ---- 종료 조건 -------------------------------------------------------------------
    object_out_x: tuple[float, float] = (0.05, 0.85)
    object_out_y: tuple[float, float] = (-0.60, 0.25)
    object_min_z: float = 0.15
    tilt_reset_deg: float = 60.0
    abnormal_qd: float = 20.0
    abnormal_penalty: float = -1.0

    # ---- 관측 노이즈 (actor 전용 — critic 은 clean) ----------------------------------
    obs_noise_qpos: float = 0.01
    obs_noise_qvel: float = 0.05
    obs_noise_body: float = 0.005
    obs_noise_object: float = 0.015

    # ---- 씬 기하 ---------------------------------------------------------------------
    table_surface_z: float = 0.200           # env.usd top_plate 상면(점군 실측)
    object_origin_offset_z: float = 0.0773   # cup_big USD 원점 ↔ 바닥
    object_spawn_pad: float = 0.005          # 스폰 침투 반동 방지
    object_spawn_z: float = 0.0              # __post_init__ 파생 (단일 소스)
    spawn_range: float = 0.02                # 스폰 xy 균등 반범위 (ADR OFF 고정값)

    # ---- 커리큘럼 --------------------------------------------------------------------
    # ★ADR 은 끄고 시작한다. 과제 성립 후 스폰 범위부터 켠다.
    enable_adr: bool = False

    # ---- 디버그 시각화 (GUI/카메라 렌더일 때만 — headless 학습에 비용 0) --------------
    enable_cmd_markers: bool = True
    cmd_marker_axis_len: float = 0.06
    cmd_marker_radius: float = 0.006
    gui_focus_env0: bool = True
    gui_camera_eye: tuple[float, float, float] = (1.1, -0.9, 0.75)
    gui_camera_target: tuple[float, float, float] = (0.35, -0.2, 0.35)

    # ---- 씬 ---------------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "env/usd/env.usd"),
        ),
    )
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
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
        ),
    )
    # 물체 rigid body 는 USD 안 baseLink 에 있다(cup_big_rl.usd 규약).
    # ★이 이름이 틀리면 `force_matrix_w` 가 **무증상 0** 이 된다.
    object_contact_filter: tuple = ("/World/envs/env_.*/Object/baseLink",)

    events: GraspS2REventCfg = GraspS2REventCfg()
    surface_friction: float = _FRICTION

    robot_cfg: ArticulationCfg = None  # __post_init__ 에서 프로필로 조립

    def __post_init__(self):
        profile = PROFILES[self.profile_name]
        self.robot_cfg = _build_robot_cfg(
            profile, bool(self.enable_self_collisions))
        # 스폰 높이는 여기 한 곳에서만 파생한다(이중 패딩 사고 차단).
        self.object_spawn_z = (
            self.table_surface_z + self.object_origin_offset_z + self.object_spawn_pad)
        self.object_cfg.init_state.pos = [
            profile.object_spawn_center[0], profile.object_spawn_center[1],
            self.object_spawn_z,
        ]

        n_arm = profile.num_arm_joints
        n_hand = profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D 델타 + 손가락 × 시너지 채널(프로필 파생).
        n_ch = len(set(profile.hand_channel_of_joint.values()))
        self.action_space = 6 + n_ch * num_fingers

        # policy obs (grasp_v1 계열 + 목표, **물체 정체성 없음**):
        #   arm q/qd(2·n_arm) + hand q/qd(2·n_hand) + palm_pos(3) + palm_ax(6)
        #   + tips_rel_palm(3·nt) + palm_to_obj(3) + obj_to_tips(3·nt)
        #   + tip_force_local(3·nt) + joint_pos_err(n_hand) + last_action
        #   + goal_rel(3)
        # ★물체 onehot·치수·질량·클래스는 넣지 않는다 — 배포 시 알 수 없는 정보다.
        self.observation_space = (
            2 * n_arm + 2 * n_hand + 3 + 6 + 3 * num_tips + 3 + 3 * num_tips
            + 3 * num_tips + n_hand + self.action_space + 3
        )
        # critic = obs + 물체 선/각속도(6) + quat(4) + height_delta(1)
        #          + distal binary/norm(2·nf) + middle binary/norm(2·nf)
        #          + phase_step_ratio(1) + fingertip_signed_dist(nt) + goal_dist(1)
        self.state_space = (
            self.observation_space + 6 + 4 + 1 + 4 * num_fingers + 1 + num_tips + 1)


@configclass
class GraspS2RTesolloRightEnvCfg(GraspS2REnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspS2RGripperLeftEnvCfg(GraspS2REnvCfg):
    profile_name: str = "gripper_left"
