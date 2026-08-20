"""grasp_lift_fabric 환경 설정.

로봇 종속 정보는 전부 `modules.robots.RobotProfile` 이 공급한다 — 이 파일에도
조인트/바디 이름을 하드코딩하지 않는다. cfg 는 **프로필 이름과 스위치**만 든다.

스위치 기본값은 전부 "가장 단순한 쪽"이다(Phase A):
    object_bank="single_cup" · onehot off · physics_dr off · adr off
켜는 순서와 의미는 플랜 §3-7 참조.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robots as _rb

_ASSETS_DIR = _ob.ASSETS_DIR


# =============================================================================
# 자산 조립 — 프로필 데이터 → IsaacLab cfg
# =============================================================================
def build_robot_cfg(profile: _rb.RobotProfile, self_collisions: bool = True) -> ArticulationCfg:
    """프로필 → ArticulationCfg. actuator 커버리지는 계약 테스트가 보증한다."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS_DIR, profile.asset.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,           # 팔 중력보상은 Fabrics 밖에서 다루지 않는다
                # ★5.0 → 1.0. 관통 시 밀어내는 속도가 크면 충격량이 폭증한다 —
                #   실측 접촉력 전형 13~20N 인데 순간 스파이크가 7218N 까지 찍혔다.
                #   깨끗한 파지가 아니라 반복 충돌이라는 신호다.
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ★★2026-08-20 재검토: 앞선 "Fabrics 가 자기충돌을 담당한다"는 근거는
                #   **팔에만 해당**했다. `use_hand_fabric=False` 라 Fabrics 는 손 관절을
                #   제어하지 않으므로 손가락 상호 관통을 막을 수단이 전혀 없었다.
                #   실측(probe_penetration, 파지 자세): 다른 손가락 링크 간 최소거리
                #   평균 7.7mm(마디 반경 ~10mm) → **100% env 에서 겹침**,
                #   컵도 표면 안쪽 최대 23.9mm 까지 관통.
                #   그 상태의 envelope_frac 은 물리적으로 불가능한 감쌈을 세므로 허수다.
                #   ※ 앞서 자기충돌을 켰을 때 팔이 관절한계를 뚫은 실측(r_aj_5=-5.66)이
                #     있었으나, 그건 **fabric 관절 순서가 어긋난 상태**에서 잰 것이다.
                #     그 버그를 고친 뒤 재측정해 문제없음을 확인하고 켠다.
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict(profile.init_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            name: ImplicitActuatorCfg(**spec)
            for name, spec in profile.actuator_specs.items()
        },
        soft_joint_pos_limit_factor=1.0,
    )


def build_object_cfg(bank: _ob.ObjectBank) -> RigidObjectCfg:
    """물체 spawn cfg. 뱅크가 2종 이상이면 MultiAssetSpawner 를 쓴다.

    ★MultiAsset 은 `clone_environments` **이후**에 RigidObject 를 만들어야 한다
      (modules.object_bank.assert_spawned_after_clone 이 강제). 여기서는 cfg 만 만든다.
    """
    def _one(spec: _ob.ObjectSpec) -> sim_utils.UsdFileCfg:
        return sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            scale=spec.scale,
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,     # 로봇과 동일 근거
                disable_gravity=False,
            ),
        )

    if bank.needs_multi_asset:
        # random_choice=False → proto[env_id % N] 결정론적 배정.
        spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_one(s) for s in bank.specs], random_choice=False,
        )
    else:
        spawn = _one(bank.specs[0])

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.20, 0.30)),
    )


# ---------------------------------------------------------------------------
# 환경 픽스처 — assets/env/usd/env.usd
#
# 자산 doc: "Static RL environment fixture from Fusion env.f3d. Meters, Z-up.
#            Static triangle-mesh colliders (no rigid body)."
#   · metersPerUnit = 1, Z-up → 스케일 함정 없음
#   · rigid body 가 **없다** → RigidObject 가 아니라 **정적 프림**으로 올린다
#     (`physics:approximation = "none"` = 정확 삼각메시, 정적 전용)
#
# 원점 규약(사용자 지시): env.usd 원점 = OpenArm base link 원점 → translation (0,0,0).
#
# 실측 구조 (점군에서 산출):
#     base_plate  바닥            z -0.025 ~ -0.015   x -0.75~0.45  y ±0.45
#     platform    로봇 마운트     z -0.015 ~  0.000   x -0.175~0.035
#     pillar_×3   기둥            z  0.000 ~  0.190   x 0.105~0.165
#     top_plate   **작업면**      z  0.190 ~  0.200   x 0.070~0.470  y ±0.45
#
# ★작업면 상면 z = 0.200 으로, 구 table.usd(스폰 z 0.2 + bbox 0.0004 = 0.2004)와
#   **같다**. 그래서 object_spawn_z 를 바꿀 필요가 없다.
#   스폰 중심 (0.30, -0.20) ± 0.10 → x[0.20,0.40] y[-0.30,-0.10] 로 top_plate 안이다.
# ---------------------------------------------------------------------------
ENV_FIXTURE_SPAWN = sim_utils.UsdFileCfg(
    usd_path=os.path.join(_ASSETS_DIR, "env/usd/env.usd"),
)
ENV_FIXTURE_PRIM = "/World/envs/env_.*/EnvFixture"


# =============================================================================
@configclass
class GraspLiftFabricEnvCfg(DirectRLEnvCfg):
    """차원(observation_space/action_space)은 `__post_init__` 에서 프로필로 확정한다."""

    # ---- 로봇 -----------------------------------------------------------------
    profile_name: str = _rb.DEFAULT_PROFILE

    # ---- 모듈 스위치 (기본은 전부 "가장 단순") -----------------------------------
    object_bank: str = _ob.DEFAULT_BANK
    object_bank_expected_size: int | None = None   # glob 뱅크 크기 고정용
    enable_object_onehot: bool = False             # ★켜면 obs 차원 변화 = 재학습
    enable_physics_dr: bool = False
    enable_adr: bool = False

    # ---- 시뮬레이션 -------------------------------------------------------------
    # 물리 120 Hz / 정책 60 Hz. Fabrics 는 fabrics_dt × fabric_decimation 로 120 Hz.
    episode_length_s: float = 8.0
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2 ** 22,
            gpu_max_rigid_contact_count=2 ** 22,
            gpu_collision_stack_size=2 ** 28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.5)

    # ---- Fabrics ----------------------------------------------------------------
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 20.0       # cspace 속도 감쇠(10→20: grasp 구간 떨림 감소)
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False

    # palm 워크스페이스 박스의 **위치 범위는 프로필**이 준다(로봇 도달범위).
    # 여기엔 자세 범위만 둔다(euler_zyx 중심 = 우팔 [90,0,90]°, 좌팔 부호 반전).
    # ★구 주석("박스를 좁히면 안 된다")은 틀렸다 — 78mm 정상상태 오차 자체가
    #   Fabrics 배선 결함이었고, 실제로는 박스가 너무 커서(도달 62% 불가)
    #   정책이 겨냥을 못 배웠다.
    max_pose_angle_deg: float = 45.0

    # ---- 액션 ---------------------------------------------------------------------
    # 전부 **절대값**이다: palm 6D 는 박스 안, 손은 관절한계 전범위.
    # delta 적분기·래치·시너지·PCA 없음 → 복원할 상태가 fabric_q 하나뿐.
    hand_limit_margin: float = 0.0           # 0 = 전범위(full range)

    # ---- 보상 (플랜 §4) -----------------------------------------------------------
    approach_weight: float = 1.0
    approach_sharpness: float = 4.0
    grasp_quality_weight: float = 3.0
    grasp_envelope_credit: float = 0.5       # envelope / grip / persistence = 0.5/0.3/0.2
    grasp_grip_credit: float = 0.3
    # 자세는 **독립 항이 아니다**. reward-audit REVISE: 컵이 애초에 서 있어 독립 항으로
    # 두면 사실상 접촉 보너스의 중복(공짜 보상)이 된다. lift/success 의 완화 곱수
    # (0.5 + 0.5·upright_quality) 로만 쓰고, 교란은 tilt_penalty 로 처벌한다.
    upright_max_deg: float = 30.0            # 이 각도에서 upright_quality = 0
    lift_weight: float = 2.0
    lift_success_height: float = 0.10        # goal 높이와 같게 — height_quality 의 분모
    success_weight: float = 10.0
    success_pos_std: float = 0.05
    # ★페널티는 **상한을 둔다**. 무한대면 멀리 밀린 뒤 회복 불가라 "컵에 아예 안 다가감"이
    #   국소최적이 된다(agn_test2 의 종료-회피와 같은 실패 축). 상한 -0.20/step.
    push_penalty_weight: float = -2.0
    push_margin: float = 0.025               # 이 이하 이동은 봐준다(스폰 흔들림)
    push_penalty_cap: float = 0.10           # relu 상한 [m] → 실효 페널티 -0.20/step
    tilt_penalty_weight: float = -1.0
    tilt_margin_deg: float = 10.0            # 파지 중 정상 흔들림 면제
    tilt_penalty_cap_deg: float = 30.0
    # ★-0.005 → -0.3, 그리고 식이 sum.clamp(1.0) → mean 으로 바뀌었다.
    #   구 설정은 실측 sum 12.0 에서 clamp 에 포화해 gradient 가 0 이었다.
    #   현재 지터에서 -0.139 (approach 최대 1.0 의 13.9%).
    action_rate_weight: float = -0.3
    contact_force_threshold: float = 1.0     # N

    # ---- 태스크 -------------------------------------------------------------------
    # 안착 높이 바로 위에 놓는다. 정확히 같으면 스폰 침투 반동으로 튕기므로 최소 패딩만.
    # ★크게 주면 물체가 낙하하고, 그 낙하량만큼 lift 보상 기준선이 어긋난다.
    object_spawn_pad: float = 0.002
    # 학습 로그(stdout)에 [METRICS] 한 줄을 남기는 주기(정책 스텝). 0 = 끔.
    # 600 스텝 = 약 10초(60Hz) — 로그가 넘치지 않으면서 추세를 볼 수 있는 간격.
    console_log_interval: int = 600
    goal_height_offset: float = 0.15
    success_pos_tolerance: float = 0.05
    object_min_z: float = 0.15               # 이 아래 = 떨어짐 → 리스폰(종료 아님)
    object_out_of_bounds_xy: float = 0.35    # 로깅 전용
    runaway_joint_vel: float = 20.0

    # ---- 커리큘럼 (축 하나: 스폰 반경) ------------------------------------------------
    spawn_range_initial: float = 0.02
    spawn_range_final: float = 0.10
    adr_num_increments: int = 50
    adr_increment_interval: int = 3000
    adr_trigger_threshold: float = 0.4

    # ★False. 손가락 교차는 **외전 관절 고정**(profile.frozen_hand_joints)으로 막는다.
    #   self-collision ON 도 겹침은 없앴지만(100%→0%) 대가가 컸다(실측):
    #     fps 44k→21k · Fabrics 추종 7.7mm→74.8mm · zero-action 에서 컵 5cm 밀림.
    #   교차 자유도 자체를 없애는 편이 싸고 확실하다. 필요하면 True 로 켤 수 있다.
    enable_self_collisions: bool = False

    # ---- 씬 픽스처 ---------------------------------------------------------------------
    env_fixture_spawn: sim_utils.UsdFileCfg = ENV_FIXTURE_SPAWN
    # env.usd 의 platform 상면이 정확히 z=0 이라 기본 지면(z=0)과 겹친다 → 아래로 내린다.
    ground_plane_z: float = -0.10

    # ---- 접촉 센서 필터 ---------------------------------------------------------------
    # ★★rigid body **prim** 을 가리켜야 한다. 루트 Xform(`/…/Object`)을 주면
    #   PhysX 가 "GPU contact filter for collider … is not supported" 경고를 내고
    #   `force_matrix_w` 가 **항상 0** 이 된다 — 2048 env 전 구간에서 접촉력이
    #   정확히 0.0000 이었던 원인이다(경고는 나오고 있었는데 넘겼다).
    #   자산 실측: cup_big_rl / shaker_closed_rl 모두 RigidBodyAPI 가 `baseLink` 에 있다.
    # resolve_cfg 가 물체 뱅크에서 파생한다(하드코딩 금지 — 자산이 바뀌면 조용히 0 이 된다).
    object_contact_filter: tuple = ()

    # ---- 자산 cfg (원본 __post_init__ 에서 채운다) -------------------------------------
    robot_cfg: ArticulationCfg = None
    object_cfg: RigidObjectCfg = None

    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    def __post_init__(self) -> None:
        resolve_cfg(self)


def resolve_cfg(cfg: "GraspLiftFabricEnvCfg") -> None:
    """스위치 → 자산 cfg · 차원 · 씬 플래그 파생. **멱등**이며 두 번 호출된다.

    ★★왜 함수로 뺐는가: hydra 는 `env_cfg.from_dict(...)` 로 **이미 생성된 cfg 의 필드만**
      덮어쓴다(`isaaclab_tasks/utils/hydra.py`). `__post_init__` 은 다시 돌지 않는다.
      그래서 `env.object_bank=cup_family` 로 오버라이드하면 문자열만 바뀌고
      `object_cfg`(single_cup 스포너) · `observation_space` · `scene.replicate_physics` 는
      옛 값 그대로 남아 **조용히 틀린 조합**으로 학습이 돈다.
      env 가 `super().__init__()` 전에 이 함수를 다시 불러 파생값을 맞춘다.
    """
    profile = _rb.get(cfg.profile_name)
    bank = _ob.get(cfg.object_bank, expected_size=cfg.object_bank_expected_size)

    if profile.fabric_class is None:
        raise RuntimeError(
            f"프로필 '{profile.name}' 은 Fabrics 자산이 없다(fabric_class=None). "
            "이 태스크는 Fabrics 전용이라 조용히 다른 제어기로 폴백하지 않는다. "
            f"사유: {profile.notes}"
        )

    cfg.robot_cfg = build_robot_cfg(profile, self_collisions=cfg.enable_self_collisions)
    cfg.object_cfg = build_object_cfg(bank)
    # ★접촉 필터는 **RigidBodyAPI prim** 을 가리켜야 한다. 루트 Xform 이면 PhysX 가
    #   force_matrix_w 를 항상 0 으로 준다(보상 7항 중 6항이 접촉 게이트라 치명적).
    cfg.object_contact_filter = (
        f"/World/envs/env_.*/Object/{bank.rigid_body_name}",
    )

    # ★MultiAsset(env 별 다른 물체)은 physics 복제가 불가능하다.
    if bank.requires_replicate_physics_off:
        cfg.scene.replicate_physics = False

    j = profile.num_arm_joints + profile.num_hand_joints
    f = len(profile.fingers)
    # ★외전 관절은 정책 제어에서 뺀다(손가락 교차 자유도 제거). obs 의 joint_* 에는
    #   여전히 전 관절이 들어간다 — 실기에서도 읽히는 값이고, 고정값이라 해가 없다.
    n_free_hand = profile.num_hand_joints - len(profile.frozen_hand_joints)
    cfg.action_space = 6 + n_free_hand
    # joint pos/vel/effort(3j) + 접촉력(f) + 물체 pos/quat(palm 프레임, 7)
    # + goal-object(3) + prev_action
    cfg.observation_space = 3 * j + f + 7 + 3 + cfg.action_space
    if cfg.enable_object_onehot:
        cfg.observation_space += bank.onehot_dim
    # critic = policy + 물체 선속도/각속도
    cfg.state_space = cfg.observation_space + 6


@configclass
class GraspLiftFabricEnvCfg_PLAY(GraspLiftFabricEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
