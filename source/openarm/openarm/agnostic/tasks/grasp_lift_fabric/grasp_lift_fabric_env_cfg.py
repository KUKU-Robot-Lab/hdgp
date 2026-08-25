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
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robots as _rb

_ASSETS_DIR = _ob.ASSETS_DIR


# =============================================================================
# 자산 조립 — 프로필 데이터 → IsaacLab cfg
# =============================================================================
def build_robot_cfg(profile: _rb.RobotProfile, self_collisions: bool = True,
                    gravity: bool = False, usd_override: str = "") -> ArticulationCfg:
    """프로필 → ArticulationCfg. actuator 커버리지는 계약 테스트가 보증한다."""
    _rel = usd_override or profile.asset.usd_relpath
    _abs = os.path.join(_ASSETS_DIR, _rel)
    if usd_override and not os.path.isfile(_abs):
        raise FileNotFoundError(
            f"robot_usd_override='{usd_override}' 가 가리키는 USD 가 없다: {_abs}\n"
            "  오타면 조용히 프로필 자산으로 돌아가지 않는다 — 측정이 뒤바뀐다.")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_abs,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # ★★KUKA 고정(08.25) — 원본 KUKA_ALLEGRO_CFG 가 명시하는 값들.
                #   IsaacLab 은 None 이면 **USD 에 적힌 값을 그대로 둔다** — 즉 명시하지
                #   않으면 자산마다 다른 물리로 학습하게 된다. 원본과 같은 값을 박아
                #   자산 신원에 의존하지 않게 한다.
                retain_accelerations=True,       # 서브스텝 간 가속 이월
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                # ★★cfg **필드**(enable_gravity)에서만 바꿀 것. 이 객체를 직접 수정하면
                #   env.__init__ 의 resolve_cfg 가 robot_cfg 를 재생성하며 **조용히 되돌린다**
                #   (08.22 실측: probe 가 False 로 바꿔 로그까지 찍었는데 USD 는 True 였다).
                #   Fabrics 는 중력보상을 하지 않으므로 켜면 관절 PD 의 정상상태 오차가
                #   그대로 처짐이 된다(URDF 계산 palm 14.4mm · z -12.2mm).
                disable_gravity=not gravity,
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
                solver_position_iteration_count=8,      # KUKA 고정
                solver_velocity_iteration_count=0,      # KUKA 고정
                sleep_threshold=0.005,                  # KUKA 고정
                stabilization_threshold=0.0005,         # KUKA 고정
            ),
            # ★★KUKA 고정(08.25). "force" = 관절 구동을 **힘**으로 낸다(관성 반영).
            #   "acceleration" 은 관성을 무시해 게인이 실기와 다른 의미가 된다.
            #   명시하지 않으면 USD 에 적힌 값을 쓰므로 자산마다 갈린다.
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
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
                solver_position_iteration_count=8,      # KUKA 고정
                solver_velocity_iteration_count=0,      # KUKA 고정
                # ★★KUKA 고정(08.25) 물체 강체 속성. 원본은 물체와 로봇의
                #   stabilization_threshold 를 **다르게** 준다(로봇 0.0005 / 물체 0.0025) —
                #   물체는 더 일찍 안정화에 참여시켜 파지 중 미세 떨림을 줄인다.
                kinematic_enabled=False,
                enable_gyroscopic_forces=True,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_angular_velocity=1000.0,            # KUKA 고정 (구 100)
                max_linear_velocity=1000.0,             # KUKA 고정 (구 100)
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
    episode_length_s: float = 10.0           # KUKA 고정
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # ★★KUKA 고정(08.25). IsaacLab 기본은 static 0.5 / dynamic 0.5 인데 원본은
        #   **둘 다 1.0** 이다 — 파지 태스크에서 마찰 2 배는 미끄러짐 한계 하중이
        #   2 배라는 뜻이라 "쥘 수 있는가" 자체를 바꾼다. 명시하지 않아 기본 0.5 로
        #   돌던 것을 원본 값으로 맞춘다.
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,      # KUKA 고정 (IsaacLab 기본 0.5)
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2 ** 22,
            gpu_max_rigid_contact_count=2 ** 22,
            gpu_collision_stack_size=2 ** 28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0)  # KUKA 고정

    # ★★KUKA 고정(08.25). 원본 dextrah_kuka_allegro 는 fabrics_dt = 1/60 이고
    #   fabric_decimation = 2 라, 정책 스텝(벽시계 1/60s)당 fabric 시간이 1/30s —
    #   **의도적으로 벽시계의 2 배속**이다. 계획이 로봇보다 앞서야 PD 가 따라잡을
    #   여유가 생긴다(lookahead).
    #   ★08.24 에 이것을 "계획이 2 배로 앞서간다"며 1/120 으로 고쳤는데, 그게 원본
    #     이탈이었다. fabric 수렴이 절반으로 느려져 A(정책목표→계획) 오차가 커졌다
    #     — 실측 A 가 perr 의 99%(17.85 / 17.99mm)를 차지했다.
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    # ★★fabric 계획 속도를 관절 PD 의 **속도 목표**로 얼마나 먹일지. 1.0 = 완전
    #   피드포워드. 0 이면 PD 감쇠항(kd=80)이 움직임을 되밀어 정상상태 오차가
    #   (kd/kp)·v = 0.2·v [rad] 로 속도에 정비례한다 — 08.25 이전이 그 상태였다.
    #   DEXTRAH 원본은 이 값을 ADR 로 1.0 → 0.0 으로 줄여 간다(실기의 불완전한
    #   속도 목표에 대한 강건화 커리큘럼). 우리는 1.0 에서 시작한다.
    velocity_target_factor: float = 1.0

    # ★★palm attractor 오버라이드 — **공유 yaml 을 건드리지 않는다**
    #   (openarm_tesollo_pose_params.yaml 은 pour 계열까지 함께 쓴다).
    #   Attractor 는 params dict 를 참조로 들고 매 스텝 읽으므로 생성 후 덮으면 된다.
    #   동역학: xdd = -gain·tanh(sharpness·|err|)·n̂ - damping·xd  (forcing policy)
    #   정상상태(목표가 v 로 이동): gain·tanh(sharp·err) = damping·v
    #     → gain ↑ 또는 damping ↓ 이면 추종오차 err 이 준다.
    #   ★gain 은 곧 최대 가속도[m/s²]다 — 실기가 못 내는 값으로 올리면 sim 전용
    #     정책이 된다. 손끝 attractor 스윕에서도 과대 게인은 "여유자유도가 팔로
    #     새는" 방식으로 실패했다(400 꼭짓점, 800 부터 악화).
    #   ★0 = yaml 값 그대로(palm: gain 80 · damping 50 · radius 0.2)라는 sentinel 이다.
    #     `float | None = None` 으로 두면 IsaacLab configclass 가 타입을 NoneType 으로
    #     추론해 hydra 오버라이드가 막힌다(실측: "Expected NoneType, Received float").
    #     저장소의 기존 규약과 같다 — palm_slew_pos 도 "0 = 제한 없음"이다.
    palm_attractor_gain: float = 0.0
    palm_attractor_damping: float = 0.0
    # ★★KUKA 고정(08.25): 원본은 이 값을 ADR 로 **10 → 20** 으로 올려 간다. 20 은
    #   커리큘럼의 **끝점**이라 처음부터 걸면 팔이 상시 과감쇠 상태로 학습한다
    #   (같은 부류: velocity_target_factor 를 0=ADR 끝점에서 시작했던 결함).
    fabrics_damping_gain: float = 10.0
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

    # ---- A: 지령 속도 제한 (slew) ---------------------------------------------------
    # ★실측 근거: 정책(결정론적)이 palm 목표를 **19,676 mm/s** 로 명령하는데 팔은
    #   200mm/s 에서 정착오차 18mm 로 따라간다 — **약 100배**. 추종률이 축별 0.00~0.01 로
    #   팔이 지령 변화의 1% 미만만 따라가고, 정책은 Fabrics 를 저역통과 필터 삼아
    #   **듀티비로** 유효 위치를 만드는 축퇴 해를 학습했다(실기 전이 불가).
    #   → 지령 자체에 속도 상한을 걸어 구조적으로 막는다.
    #   4mm/step = 240mm/s. 박스 300mm 를 75스텝(1.25s)에 횡단 — 480스텝 에피소드에 충분.
    # ★★KUKA 고정(08.25) slew 제거 — 원본에 rate limit 이 **없다**(전 파일 grep 0건).
    #   fabric 의 xdd = -gain·tanh(sharp·|err|)·n̂ 가 이미 가속도 상한이라 그 자체가
    #   rate limiter 다. 그 위에 slew 를 겹치면 이중 제한이고, 잘린 축의 액션 성분은
    #   효과도 gradient 도 0 이 된다(실측 cmd_step 6.26mm ≈ 상한 6.93mm 상시 포화).
    #   도입 근거였던 밀침·스윙은 종료 조건(fallen|out_xy|tipped)이 맡는다 —
    #   원본도 out_of_reach 즉시 종료를 쓴다.
    palm_slew_pos: float = 0.0               # [m/step] 0 = 제한 없음
    palm_slew_rot_deg: float = 0.0           # [deg/step] 0 = 제한 없음

    # ---- D: 방향 대칭 액션 스케일 -----------------------------------------------------
    # ★실측: home 이 박스 중심이 아니라 액션 단위당 이동량이 방향마다 최대 **7.5배**
    #   다르다(y: a=+1 이 0.300m, a=-1 이 0.040m). 정책은 대칭 가우시안을 내는데
    #   물리 효과가 비대칭이라 탐색이 한쪽으로 치우친다.
    #   → 축마다 **하나의 스케일**(양쪽 중 큰 쪽)을 쓰고 박스로 clamp 한다.
    #     a=0 → 홈 은 유지되고, 남는 쪽에 생기는 불감대는 과제와 반대 방향이라 무해하다.
    symmetric_action_scale: bool = True

    # ---- 손 제어: Fabrics 손끝 IK (08.23, 사용자 지시) ------------------------------
    # ★손을 Fabrics 밖(직접 관절 PD)에 두면 세 가지가 동시에 걸린다:
    #   ① fabric 이 아는 손 자세가 **홈에 고정**된다 — cspace attractor 가 붙잡고 있고
    #      env 는 손 구간을 갱신하지 않는다. body_repulsion 이 실제보다 큰(펴진) 손으로
    #      회피를 계산해 팔을 불필요하게 밀어낸다.
    #   ② `body_repulsion` 에 손가락↔손가락 쌍을 넣을 수 없다 — 제어하지 않는 관절의
    #      상호 충돌은 계획으로 막을 수 없다. 그래서 PhysX self-collision 을 못 끈다.
    #   ③ 관절공간 목표라 손끝이 어디로 가는지는 정책이 역기구학을 스스로 배워야 한다.
    # tip attractor 는 PCA(5D)와 달리 손 20-DOF 를 그대로 두므로 감쌈을 제약하지 않는다.
    # 액션은 손끝 5점의 **palm 상대** 위치 15D — 절대 좌표면 팔이 움직일 때마다 손 목표가
    # 함께 끌려가 팔·손 제어가 얽힌다.
    # 손 제어 경로 — 셋 중 하나.
    #   "pd"     : 손은 Fabrics 밖. 정책 액션 = 관절 목표 → 직접 PD (구 배선).
    #   "fabric" : 손 20-DOF 를 Fabrics 가 소유. 정책 액션은 **관절 그대로**(의미 불변),
    #              fabric 이 관절 한계·자기충돌을 함께 풀어 관절 목표를 만든다. ★권장
    #   "tip"    : 손끝 5점 위치를 정책이 지시(작업공간 IK).
    #              ★08.23 기각 → **08.24 복귀**. 기각 사유("15D 가 기구학적으로 불가능")는
    #                진단이 틀렸다. 프로브(팔 고정·손가락별·dt 1/120)로 갈린 실제 원인:
    #                ①액션 박스가 도달 영역에서 뜬다 — span_frac=0.8 에서 박스 균일
    #                  샘플의 **9.7%** 만 5mm 이내 도달(0.4→60.9%, 0.2→93.8%).
    #                  정책이 낼 수 있는 지시의 90% 가 실현 불가였다.
    #                ②게인 400 이 낮다 — τ 1.62s(800→0.98 · 1600→0.78 · 6400 발산).
    #                반대로 "손끝끼리 metric 을 공유해 간섭한다"는 재현되지 않았다:
    #                도달불가 2 개를 섞어도 나머지 3 지 오차가 손가락별 0.69mm vs
    #                단일 15D 0.79mm 로 같다(손끝 i 는 손가락 i 관절에만 의존해서
    #                Jacobian 이 이미 블록 대각이다). 손가락별은 손해가 없어 켜 둔다.
    #              ★관절공간(direct)은 사용자 판정으로 폐기: attractor 가 명령을 27°
    #                왜곡하고(hand/cmd_err_rad 0.46) 외전 잠금이 8~9° 풀렸다.
    hand_control: str = "tip"
    # hand_control="fabric" 의 attractor 게인. None = fabric params 기본(50).
    # ★손이 목표를 못 따라가면(hand/cmd_err 가 크면) 여기를 올린다. 손끝 attractor 는
    #   같은 구조에서 400 이 꼭짓점이었다. Jacobian 이 손 구간으로 마스킹돼 있어
    #   게인을 올려도 palm 은 오염되지 않는다(tip 실험에서 확인한 구조적 보장).
    hand_attractor_gain: float | None = 400.0
    # 손가락↔손가락 Fabrics 반발. ★기본 off — 켜면 fabric 이 **계획 단계에서**
    #   손가락 겹침을 피하려 궤적을 튼다(팔 반발과 파라미터·구 집합이 분리돼 있다).
    #   PhysX self-collision 을 끄려면 이 항이 관통 해를 탐색 공간에서 제거해야 한다.
    use_hand_repulsion: bool = False
    # ★★KUKA식 body_repulsion(08.25). 원본 KUKA 는 `collision_link_prefix_pairs` 가
    #   **13 쌍뿐이고 전부 손↔팔뚝(iiwa7_link_2)** 이다 — 손가락↔손가락 쌍이 없다.
    #   손가락 겹침은 PhysX self-collision 이, 교차 자세 자체는 PCA 5D 가 막는다.
    #   우리는 이 항이 통째로 꺼져 있어(팔 그룹 비활성) 계획 단계 방어가 0 이었다.
    #   True 로 켜면 손↔팔·팔↔몸통 쌍이 살아난다(손가락 쌍은 use_hand_repulsion 소관).
    use_body_repulsion_pairs: bool = False
    # ★자산 USD 를 프로필 기본에서 바꾼다(측정용). "" = 프로필 값 그대로.
    #   예) "openarm_tesollo_bi_s_rl_hull/openarm_tesollo_bi_s_rl.usd" = convexHull 변형.
    #   ★프로필을 고치지 않는다 — 다른 트랙이 같은 프로필을 공유한다.
    robot_usd_override: str = ""
    use_tip_fabric: bool = True
    # 손끝 attractor 를 **손가락별 5 개**로 쪼갠다(각 항이 그 손가락 4 관절만 움직인다).
    # 이득은 측정상 미미하지만(위 주석) 손해도 없고, 도달불가 목표의 오차가 그 손가락
    # 안에 갇히는 것이 구조적으로 명확하다.
    tip_per_finger: bool = True
    # 게인 실측 2 회가 갈렸다 — 채택은 08.24 쪽이다.
    #   08.23(단일 taskmap·팔 미고정): 400→2.94mm 최적, 800→9.82, 1200 이상 발산.
    #   08.24(손가락별·팔 고정·dt 1/120·컵 표면 목표): 400→4.00 · 800→2.34 ·
    #         **1600→1.36(τ0.78s)** · 6400 발산.
    # 08.23 이 든 상한 근거("여유자유도가 팔로 샌다")는 SubchainFrameOriginsTaskMap
    # 도입 후 성립하지 않는다 — Jacobian 의 팔 열이 0 이라 샐 경로가 없다.
    tip_attractor_gain: float = 1600.0
    # 액션 박스 = 부팅 시 실측한 손끝 도달 영역 × 이 비율(홈 기준). 1.0 이면 전 범위.
    # ★★이 값이 tip IK 의 1 순위 손잡이다. 박스 균일 샘플의 5mm 이내 도달 비율(게인
    #   1600·손가락별·2 초):  span 0.8 → **9.7%** · 0.4 → 60.9% · 0.2 → **93.8%**.
    #   손가락 하나의 도달 집합은 3D 볼륨인데 축정렬 박스는 그 바깥까지 덮으므로,
    #   박스를 키우면 정책이 낼 수 있는 지시의 대부분이 실현 불가가 된다(08.23 실패의
    #   직접 원인). 0.3 은 실현율과 도달 범위(컵 표면까지 26mm 필요)의 절충이다.
    tip_action_span_frac: float = 0.3
    # ★★KUKA 고정(08.25) — 원본은 절대 액션(`compute_absolute_action`)뿐이고 누산
    #   모드가 없다. delta 누산은 "fabric 이 급변 목표를 못 따라간다"는 증상에 대한
    #   대응이었는데, 그 원인(속도 피드포워드 0 · fabrics_dt 절반 · slew 포화)이
    #   전부 고쳐졌으므로 원본 규약으로 되돌린다.
    tip_action_mode: str = "absolute"
    tip_delta_scale: float = 0.003           # [m/step] 박스 반폭 21mm 를 7스텝에 횡단
    # 워크스페이스 실측 표본 수(부팅 1회, FK 만).
    tip_workspace_samples: int = 4096
    # 박스 경계 백분위. 1.0 이면 극값(구 동작). 홈이 이 대역 밖이면 그 축은 홈에 붙는다.
    tip_workspace_quantile: float = 0.98

    # ---- 보상 — ★08.23 자매 트랙 grasp_sensor 와 **완전 동일 규약** ---------------
    # 사용자 결정: "리워드 구조 등은 grasp_sensor 쪽으로 모두 바꾸기. EP3000 에서
    # 성공 시작됨." 보상 함수 자체를 그쪽에서 import 하므로 여기 계수도 같은 값을 쓴다.
    # 값이 갈리면 함수만 공유하고 거동은 달라져 비교가 무의미해진다.
    #
    # 구 7항에서 실측으로 드러난 결함 셋 — 이 교체로 전부 사라진다:
    #   · envelope_mul_floor 0.3 → 감쌈 없이도 이송 보상 30% 유출(좌팔 감쌈 0.21/이송 0.58)
    #   · grasp_quality 의 grip/persist 30% 몫 → 감쌈 아닌 것으로 채워짐(우팔 1.3 중 0.43)
    #   · lift_success_height 0.10 ≠ goal 0.15 → dz 10cm 포화, 우팔 6cm 고착
    # ── 08.24 단계형 보상(사용자 확정): 접근1→파지2→감쌈3 ‖ 리프트5→이송8→성공12.
    #    가중치 = 단계 상한. 다음 단계가 항상 커야 앞 단계에 머무는 국소해가 없다.
    #    (관절 모드 else 분기는 아래 값 일부를 옛 의미로 계속 읽는다 — 주석 참조)
    approach_weight: float = 1.0
    approach_sharpness: float = 8.0
    # ★★접근 **정렬**(08.25 사용자 지시): palm_ee +x(손바닥 법선)가 컵 축과
    #   **수직**이어야 측면 파지가 된다. 컵은 중력 반대로 서 있어야 하므로
    #   cup_z ≡ world +z 로 두고 |n_tcp.z| → 0 을 목표로 한다(회전 추정 불필요).
    #   ★가중치는 approach(1.0)+align(0.5)=1.5 < grip(2.0) 이 되게 잡았다 —
    #     단계 단조(1<2<3<5<8<12)가 깨지면 앞 단계에 머무는 국소해가 생긴다.
    #     reward-audit ACCEPT: 최대 기여가 success(12)의 4% 라 정렬 고착 위험 없음.
    align_weight: float = 0.5
    grasp_z_offset: float = 0.0              # 파지중심 = 물체중심 + z 오프셋
    side_radius: float = 0.03                # 대향 파지점 반경 [m] — cup 몸통 17.5~30mm
    # 감쌈을 **순수 항**으로 준다(게이트 곱 없음). 접촉 전에는 어차피 0 이고, 게이트를
    # 곱하면 "게이트를 못 만든 부분 감쌈"이 통째로 안 보인다.
    envelope_weight: float = 3.0
    # ── 파지(grip) 항: wrap 마디를 물체 쪽으로, 닿으면 그 손가락 off ────────────
    grip_weight: float = 2.0
    grip_sharpness: float = 20.0             # exp(-k·d) — 5cm 에서 0.37, 1cm 에서 0.82
    # ★손 기하 상수(팁 반경). 물체 형상 아님 — 형상 비의존 원칙에 어긋나지 않는다.
    #   접촉 감지가 실패해도 이 아래로는 보상이 안 늘어 압입 무한보상을 차단.
    grip_dist_floor: float = 0.009
    contact_weight: float = 0.5              # 이진 대향 접촉 = 사다리 한 칸
    contact_force_threshold: float = 1.0     # [N] 게이트·감쌈 판정 공통 (dexsuite 동일)
    tracking_weight: float = 8.0
    tracking_std: float = 0.1
    success_weight: float = 12.0
    success_std: float = 0.05
    # ★lift·upright 둘 다 분모가 goal_height_offset(0.15) 이다 — 포화점 = 목표 높이.
    #   분모를 goal 보다 작게 두면 "거기까지만 올리면 만점"이 되어 그 위 구간이 평지가
    #   된다. 우팔이 18cm 까지 들었다가 6cm 로 되돌아온 것이 정확히 그 결과였다.
    lift_weight: float = 5.0
    upright_weight: float = 3.0
    upright_exponent: float = 4.0            # cos^4 — 소각 판별력(구 관절 모드 전용)
    # ---- tip 모드 전용(rewards_tip) --------------------------------------------------
    # 리프트를 진척률(clamp)에서 **|z 오차| 대칭 커널**로. 구 진척률은 목표 위가
    # 평지라 과주행에 벌점이 없었다(우팔이 18cm 까지 들었다 6cm 로 되돌아온 축).
    lift_sharpness: float = 8.5
    # success 의 회전 성분 σ [rad]. 위치 성분은 success_std 를 그대로 쓴다.
    success_rot_std: float = 0.35
    # ── 감쌈 판정: 손바닥면 접촉만 인정 ────────────────────────────────────────
    # 자매 실측(lstm_test3 ep5000): middle_4 가 접촉 시간의 100% 를 **손등**으로 접촉,
    # env_frac 계상 0.746 vs 정직한 값 ≈0.55 — 성공 임계 0.6 을 허수 통과.
    # 판정은 힘 벡터가 아니라 기하(마디 palmar 축 · (물체−마디) > 0) — 힘 부호 규약은
    # 미검증이고 뒤집혀도 조용하다. 프로필 palmar_axis_local 미정의면 부팅 fail-loud.
    require_palmar_contact: bool = True
    # ★보상 구조를 손 제어 방식과 **분리**한다. "pd" 대조군도 같은 단계형 보상으로
    #   돌려야 손 제어 방식만의 차이를 본다(보상까지 바뀌면 원인 분리가 안 된다).
    use_staged_reward: bool = True
    # ── success 의 감쌈 램프 포화점(판정 임계와 분리) ──────────────────────────
    # 판정(success_envelope_min 0.6)과 같으면 3 지를 넘는 순간 4·5 지째 유인이 소멸
    # (자매 실측: env 0.65 에 2,500 에폭 고착). gradient 만 0.85 까지 연장, 판정 불변.
    envelope_gate_saturation: float = 0.85
    tilt_penalty_weight: float = -0.5
    tilt_free_deg: float = 20.0              # 이 각까지는 무징계(구 관절 모드 전용)
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    # 성공 판정 3조건 — goal 근접 AND 감쌈 AND 직립.
    success_envelope_min: float = 0.6        # g_eff 포화점이자 성공 하한
    success_tilt_max_deg: float = 20.0
    abnormal_penalty: float = 0.0            # 이 트랙은 관절한계를 Fabrics 가 담당

    # ---- 진단 전용(보상에 안 들어감) -----------------------------------------------
    persistence_ref_steps: int = 20          # 대향 게이트 연속 유지 → persist=1
    # 08.22 엄격 감쌈(전 마디 동시 접촉) 대조 지표. 보상은 grasp_sensor 규약(마디 하나
    # 라도 접촉)을 쓰지만, 같은 정책을 두 판정으로 재면 0.503 vs 0.069 로 7배가 벌어진다.
    # 느슨한 쪽만 오르고 이쪽이 안 오르면 "받치기"를 감쌈으로 세고 있다는 신호다.
    envelope_force_threshold: float = 0.5
    participation_force_threshold: float = 0.1   # N — grip_frac 참여 판정

    # ---- 태스크 -------------------------------------------------------------------
    # 안착 높이 바로 위에 놓는다. 정확히 같으면 스폰 침투 반동으로 튕기므로 최소 패딩만.
    # ★크게 주면 물체가 낙하하고, 그 낙하량만큼 lift 보상 기준선이 어긋난다.
    object_spawn_pad: float = 0.002
    # 학습 로그(stdout)에 [METRICS] 한 줄을 남기는 주기(정책 스텝). 0 = 끔.
    # 600 스텝 = 약 10초(60Hz) — 로그가 넘치지 않으면서 추세를 볼 수 있는 간격.
    console_log_interval: int = 600
    goal_height_offset: float = 0.15
    # dexsuite 규약 = success 항 pos_std 의 절반. success_std 0.05 → 0.025.
    success_pos_tolerance: float = 0.025
    # ---- goal 랜덤화 (이송 학습, 08.22 사용자 디렉션) --------------------------------
    # ★goal 이 스폰의 결정론적 함수(수직 +offset)면 정책은 goal obs 를 무시하고
    #   "제자리 들기"만 배운다 — 배포에서 사용자 지정 위치에 반응하지 않는 정책이 된다.
    #   그래서 goal 오프셋을 스폰과 **독립**으로 샘플링하고 반경을 ADR 축으로 확장한다.
    #   initial 0.0 = 구 고정 goal 과 정확히 동치(ADR off 여도 동치) — 초기 난이도 불변.
    goal_xy_radius_initial: float = 0.0
    goal_xy_radius_final: float = 0.10
    goal_z_radius_initial: float = 0.0
    goal_z_radius_final: float = 0.05        # goal z = 스폰 z + goal_height_offset ± 이 값
    # 샘플된 goal 은 palm 워크스페이스 박스 안쪽으로 클램프한다(마진만큼 축소).
    # 박스 밖 goal 은 정책이 도달 불가 — 액션 포화 학습의 재발 경로라 원천 차단.
    goal_box_margin: float = 0.03
    # ★08.22 리스폰 → **종료** 로 전환(사용자 지시, grasp_v1 선례).
    #   grasp_v1 은 out_x|out_y|fallen|tipped 전부 에피소드 종료로 처리하고 98% 파지까지
    #   갔다(grasp_right_env._get_dones). 쓰러진/떨어진 컵을 에피소드에 방치하면
    #   회복 불가 상태의 전이가 배치를 희석하고 value 추정 오차가 GAE 로 번진다.
    #   ※반대 실측(agn_test2: 종료가 접근 회피를 가르침)이 있으므로 감시 지표를 둔다 —
    #     approach 하락 + episode_lengths 상승이 동시에 나타나면 그 시그니처다.
    object_min_z: float = 0.15               # 이 아래 = 낙하 → **종료**
    object_out_of_bounds_xy: float = 0.35    # 스폰 기준 xy 이탈 → **종료** (구: 로깅 전용)
    tipping_termination_deg: float = 40.0    # 컵 축이 이보다 기울면 → **종료** (grasp_v1 동일)
    runaway_joint_vel: float = 20.0

    # ---- 커리큘럼 (축 하나: 스폰 반경) ------------------------------------------------
    spawn_range_initial: float = 0.02
    spawn_range_final: float = 0.10
    # ★★KUKA 고정(08.25) 관측 노이즈 — 원본 teacher 는 전 항목이 *_noisy 다.
    #   ★단 원본은 **전부 ADR 로 0 에서 시작**한다(object_pos_noise 0→0.03 등).
    #     처음부터 걸면 "ADR 끝점에서 시작"이 되어 과제 성립을 방해한다 —
    #     velocity_target_factor 0 · fabric_damping 20 에서 이미 겪은 실수다.
    #   여기 값은 ADR 축의 **최종값**이고, 시작은 0 이다(adr 축 미연결 시 노이즈 없음).
    obs_noise_object_pos: float = 0.03       # [m]   원본 object_pos_noise 최종값
    obs_noise_object_rot: float = 0.1        # [rad] 원본 object_rot_noise 최종값
    obs_noise_joint_pos: float = 0.01        # [rad]
    obs_noise_joint_vel: float = 0.1         # [rad/s]
    # 현재 적용 계수(0 = 노이즈 없음). ADR 이 이 값을 0 → 1 로 올린다.
    obs_noise_scale: float = 0.0
    # ★★KUKA 고정(08.25) — 원본 ADR 축 `observation_annealing.coefficient`.
    #   fabric_qd / fabric_qdd 를 policy obs 에 넣을 때 곱하는 계수다.
    #   원본의 ADR 범위가 **(0., 0.)** 이라 실제로는 항상 0 — 자리는 유지하되 값은
    #   죽여 둔 상태가 원본의 거동이다. 켜고 싶으면 이 값을 올린다.
    #   ※critic 에는 이 계수와 무관하게 원본 fabric_qd / qdd 가 들어간다(원본 동일).
    obs_annealing_coefficient: float = 0.0

    adr_num_increments: int = 50
    adr_increment_interval: int = 3000
    adr_trigger_threshold: float = 0.4

    # ---- KUKA ADR 축 (08.25 통일) --------------------------------------------------
    # ★★원본은 ADR 13 그룹인데 우리는 spawn/goal 2 개만 연결돼 있었다. **시작값만 맞추고
    #   끝으로 가는 경로가 없으면 커리큘럼이 아니라 그냥 고정값**이다 — 원본이 난이도를
    #   올려 가며 강건성을 얻는 축들이 전부 죽어 있었다.
    #   보상 가중치 축(reward_weights)은 사용자 지시로 이번 통일에서 제외한다.
    #
    # `pd_targets.velocity_target_factor` 1 → 0: 실기의 불완전한 속도 목표에 대한 강건화.
    #   ★끝값 0 에서 시작하면 감쇠항이 속도만큼 지연을 만든다(08.25 규명한 결함).
    adr_velocity_target_factor_final: float = 0.0
    # `fabric_damping.gain` 10 → 20. 시작값은 fabrics_damping_gain(10) 이 준다.
    adr_fabric_damping_final: float = 20.0
    # `robot_spawn`: 리셋 시 관절 위치/속도 노이즈. 원본 (0., 0.35) / (0., 1.0).
    adr_robot_joint_pos_noise_final: float = 0.35    # [rad]
    adr_robot_joint_vel_noise_final: float = 1.0     # [rad/s]
    # `object_spawn.rotation` 0 → 1. 스폰 자세 무작위화(원본은 x·y 축 회전 노이즈).
    adr_object_rotation_final: float = 1.0
    # `object_state_noise` — 관측 노이즈 계수를 0 → 1 로 올리는 축.
    #   실제 폭은 obs_noise_* 필드가 준다(그 값이 원본의 ADR 최종값이다).
    adr_obs_noise_scale_final: float = 1.0
    # `object_wrench.max_linear_accel` 0 → 10 [m/s²]. 1 초마다 물체에 외란 가속도.
    adr_object_wrench_accel_final: float = 10.0
    # 외란 렌치 주입 주기 [정책 스텝]. 원본 `wrench_trigger_every = int(1/(decimation·dt))`.
    wrench_trigger_every: int = 60
    # 토크 외란의 모멘트 팔 [m] — 원본 torsional_radius.
    torsional_radius: float = 0.01
    # 손↔물체 거리가 이보다 멀면 외란을 넣지 않는다(원본 규약: 파지 중일 때만 흔든다).
    wrench_hand_distance_threshold: float = 0.3

    # ★★08.24 True 로 되돌림. 이전 근거("손가락 교차는 외전 관절 고정으로 막는다")는
    #   tip 모드에서 성립하지 않는다 — 그 모드는 frozen_hand_joints 를 적용하지 않고
    #   fabric 이 손 20-DOF 를 전부 소유한다(교차 자유도가 되살아난다).
    #   남은 후보는 Fabrics `use_hand_repulsion` 인데 그것도 끈다: 손가락 구 반경이
    #   9mm 씩이라 두 손가락이 18mm 보다 가까워질 수 없고, 그 벽이 컵 둘레를 따라
    #   붙는 감쌈 자세를 계획 단계에서 금지한다(자매 트랙 lstm_test4 에서 2 지 파지로
    #   붕괴). 둘 다 끄면 관통을 막는 장치가 하나도 없으므로 PhysX 로 되돌린다.
    #   대가는 처리량 절반(실측 44k→21k fps)이고, 파지가 성립한 뒤 다시 볼 문제다.
    #   ※되돌릴 때는 use_hand_repulsion 을 함께 볼 것.
    # ★중력은 **반드시 이 필드로만** 켠다. `robot_cfg.spawn.rigid_props` 를 직접 수정하면
    #   env.__init__ 의 resolve_cfg 가 robot_cfg 를 재생성하며 조용히 되돌린다(08.22 실측).
    #   ON 이면 Fabrics 가 중력보상을 안 하므로 palm 이 처진다(URDF 계산 14.4mm).
    #   ※ 켤 때 함께 재검증할 것: palm 워크스페이스 박스 · 홈 기준점(home_palm).
    enable_gravity: bool = False
    # 중력보상 피드포워드 **계수**. enable_gravity=False 면 무시된다.
    #   1.0 = 완전 보상(중력 OFF 와 물리적으로 동치) · 0.9 = 10% 잔류 처짐 · 0.0 = 보상 없음
    #   실측(보상 0): palm 처짐 57.6mm · 자세 의존 37~72mm · success_pos_std(50mm)와 같은 규모.
    #   실기 컨트롤러도 질량 모델 오차로 완벽하지 않으므로 이 계수가 s2r DR 축이 된다.
    gravity_compensation: float = 1.0

    enable_self_collisions: bool = True

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

    cfg.robot_cfg = build_robot_cfg(profile, self_collisions=cfg.enable_self_collisions,
                                    gravity=cfg.enable_gravity,
                                    usd_override=cfg.robot_usd_override)
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
    # ★tip IK 모드에서는 손 액션이 관절이 아니라 **손끝 5점 × xyz** 다.
    #   frozen_hand_joints 는 이 모드에서 의미가 없다 — fabric 이 손 20-DOF 를 전부
    #   소유하고, 손가락 교차는 자유도 제거가 아니라 body_repulsion 이 막는다.
    _tip = (cfg.hand_control == "tip") or cfg.use_tip_fabric
    n_hand_action = 3 * len(profile.fingertip_bodies) if _tip else n_free_hand
    cfg.action_space = 6 + n_hand_action
    # ★★policy obs (08.25 사용자 확정으로 정리) — 실기에서 얻을 수 있는 것만 남긴다.
    #   joint pos/vel      2j    실기 직접 측정
    #   object_pos          3    ★rot 제외(6-DOF 추정에서 회전이 가장 불안정 — critic 전용)
    #   object_goal         3
    #   prev_action        A
    #   TCP 위치            3    palm_ee (손바닥 앞 · 사용자 확인 +x = 손바닥 법선)
    #   TCP 자세            6    회전행렬 x·z 열 — 정렬(align)은 위치가 아니라 이 축이 준다
    #   손끝 위치          3T
    #   물체 크기           3
    #   fabric_q            j    계획 상태(실제 관절과 다르므로 중복 아님)
    #   ※제거분: joint_eff(27) · contact(5) · cmd_rel(6) · hand_vel(3(T+1)) ·
    #     object_rot(4) · fabric_qd/qdd(2j, 원본에서도 항상 0) → 243 에서 크게 줄었다.
    _T = len(profile.fingertip_bodies)
    cfg.observation_space = (
        2 * j + 3 + 3 + cfg.action_space + 3 + 6 + 3 * _T + 3 + j)
    if cfg.enable_object_onehot:
        cfg.observation_space += bank.onehot_dim
    # critic = policy + 물체 회전 4 + 접촉력 f + 물체 6D 속도 + fabric qd/qdd(원본 실값)
    cfg.state_space = cfg.observation_space + 4 + f + 6 + 2 * j


@configclass
class GraspLiftFabricEnvCfg_PLAY(GraspLiftFabricEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
