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
                # self-collision 은 PhysX 가 켠 채로 진다(2026-08-20 확정) — Fabrics
                # repulsion 은 계획 보조일 뿐 물리 접촉을 대체하지 못한다(관통 실측).
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=8,      # KUKA 고정
                solver_velocity_iteration_count=0,      # KUKA 고정
                sleep_threshold=0.005,                  # KUKA 고정
                stabilization_threshold=0.0005,         # KUKA 고정
            ),
            # "force" 구동(관성 반영) — KUKA 고정. 미명시 시 USD 값이라 자산마다 갈림.
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
    # ★기본 = 컵형 다종(08.26 사용자 확정: 다양한 컵 모양 인벨롭 일반화가 목적).
    object_bank: str = "cup_family"
    object_bank_expected_size: int | None = None   # glob 뱅크 크기 고정용
    enable_object_onehot: bool = False             # ★켜면 obs 차원 변화 = 재학습
    enable_physics_dr: bool = True
    # ★기본 True(08.26) — CLI 로만 켜면 기본 부팅에서 코리더가 영원히 느슨한
    #   시작값에 머문다(조용한 함정, 인벤토리 실측).
    enable_adr: bool = True

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

    # ---- 지령 속도 상한 (slew) ------------------------------------------------------
    # 지령 상한은 액션 인터페이스가 진다(사용자 확정: fabric 은 실행 보조 계층).
    # 절대 매핑 위의 slew 라 목표 인플레 없음. 값 0.1m/15° = 사용자 지정.
    palm_slew_pos: float = 0.10              # [m/step] 0 = 제한 없음(사용자 확정)
    palm_slew_rot_deg: float = 15.0          # [deg/step] 08.26 사용자 확정(0.1m/15°)


    # ---- 손 제어: 풀 관절 (fabric direct) — 단일 경로 -------------------------------
    # 손 20-DOF 를 Fabrics 가 소유, 정책 액션 = 자유 관절 목표(대칭 매핑).
    # tip IK 는 08.26 폐기: 5지 손끝 15D 독립 지시는 기구학적으로 성립하지 않았다
    # (지령↔실제 최대 111mm — probe_tip_cmd_placement 실측, 저장소 2회째 같은 결론).
    # ★★08.27 손은 fabric 밖이다(사용자 지시 · 자매 grasp_sensor 배선과 동일).
    #   h7 실측이 근거: |fabric_q_hand − 정책 지령| 우 0.956rad(55°)·좌 0.645rad 인데
    #   |실측 − fabric_q| 는 0.12~0.25rad 였다 — PD 는 잘 따라가고 **fabric 이 지령을
    #   깎았다**. 정책 액션 → 관절 목표 → PD 로 직결한다. fabric 은 팔만 소유하고,
    #   손 상태는 매 스텝 지령으로 동기화해 충돌 FK 만 맞춘다.
    hand_velocity_ff_scale: float = 1.0
    hand_control: str = "fabric"
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


    # 손끝 목표 시각화 — 정책이 어디를 지시하는지 GUI 로 확인한다(반지름 1cm 구).
    # ★headless 에서는 자동으로 꺼진다(렌더 대상이 없다).
    enable_tip_markers: bool = True
    tip_marker_radius: float = 0.01
    # GUI 시작 시 카메라를 env0 정면에 둔다(사용자 지시). eye/target 은 **env0 로컬**
    # 기준이고 부팅 때 env_origins[0] 를 더해 world 로 올린다.
    # ★기본 뷰는 씬 전체를 잡아 2048 env 가 다 보인다 — 확인용으로는 쓸모가 없고
    #   렌더 비용만 크다. env0 를 정면에서 보게 두면 나머지는 frustum 밖으로 나간다.
    gui_focus_env0: bool = True
    gui_camera_eye: tuple[float, float, float] = (1.1, -0.9, 0.75)
    gui_camera_target: tuple[float, float, float] = (0.35, 0.0, 0.35)


    # 계층 게이트 λ→μ→ν→ρ (DexPour 식 3~6). 이진 누적 곱.
    stage_gate_approach_m: float = 0.12      # λ — palm 부착 파지중심↔컵
    stage_gate_contact_n: float = 3.0        # μ — 동시 접촉 손가락 수
    stage_gate_lift_m: float = 0.05          # ν — 상승 높이
    stage_gate_transfer_m: float = 0.08      # ρ — 목표 거리

    # 파지 품질 Q_g. deep(같은 손가락 두 마디 동시)이 지배적 — 팁 스침으로는 못 만든다.
    stage_graspq_touch: float = 0.25
    stage_graspq_deep: float = 0.55
    stage_graspq_persist: float = 0.20
    stage_graspq_thumb_floor: float = 0.30   # 엄지 없는 전략의 상한
    stage_thumb_force_ref: float = 0.5       # [N] 소프트 대향 스케일(센서 스케일 기준)
    stage_contact_persistence_steps: int = 20

    # 진척 스케일
    stage_lift_height_ref: float = 0.15
    stage_tracking_std: float = 0.1
    stage_stay_speed_ref: float = 0.05       # [m/s] 정지 판정
    stage_disp_limit: float = 0.06           # [m] 밀림 F = 1/(1+(xy/limit)²)

    # 직립 — **단계별로 다르게** 요구한다(자매 e6aecb9, 사용자 영상 규격).
    #   이송은 20° 까지 관용, 정지·성공만 5° 직립. 구 배선은 정확히 뒤집혀 있었다.
    stage_tilt_tolerance_deg: tuple[float, float] = (30.0, 20.0)   # U_tol
    stage_upright_gate_deg: tuple[float, float] = (15.0, 5.0)      # U_up

    # 자세(접근) — align 은 **수평 성분만**, perp/roll 이 피치·롤을 잡는다.
    stage_align_floor: float = 0.25
    stage_perp_exponent: float = 2.0
    # ★★08.26 동일 세팅 — 자매 수식(clamp^4)으로 복귀. smoothstep 밴드 (20,5) 를
    #   하루 썼는데(h1: ZX 3~5° 유지 실증) 사용자 확정("리워드 수식 자매와 동일")으로
    #   되돌린다. 미러 사망은 수식이 아니라 **입력**(palm_y 부호 실측 곱)이 막는다.
    #   지수판의 약한 물림(10° 에서 0.941)이 다시 문제가 되면 자매와 **함께** 바꾼다.
    stage_roll_exponent: float = 4.0         # 0 이면 롤 항 무효화(자매 규약)
    stage_orient_floor: float = 0.15

    # ── 08.26 동일 세팅 — 자매 신규 필드(값 자매와 동일) ─────────────────────────
    # 접촉 판정 단일 임계 — 자매는 참여/감쌈 구분 없이 0.1N 하나를 쓴다.
    stage_contact_threshold: float = 0.1
    # persist 분모 — deep 접촉(마디 2개 동시) 연속 유지 스텝 기준.
    stage_contact_persistence_steps: int = 20
    # 코리더 래치(자매 08.26 승인) — 리프트 중 xy 밀림·기울기가 한도를 넘은 **이력**이
    # 있으면 그 에피소드의 ν 이후(lift·transfer·stay·success)를 몰수한다. 순간 게이트는
    # 낚아챔이 "60스텝 통행료 7%" 로 우회했다(자매 probe: xy 253mm·tilt 49°).
    # (느슨한 시작, 만렙) — 자매는 per-env difficulty 로 보간하는데 우리는 그 축이
    # 없어 ADR 진행률(전역)로 보간한다. 낚아챔(25cm/49°)은 시작값에도 걸린다.
    stage_corridor_xy_m: tuple[float, float] = (0.20, 0.05)
    stage_corridor_tilt_deg: tuple[float, float] = (50.0, 20.0)
    # 파지중심 오버라이드 — 자매의 **자유 컵 실측**(probe_seqclose). 홈 손끝 중점
    # 유도는 손이 펴져 있어 원위로 ~35mm 치우친다(실제 무는 지점은 손바닥 쪽).
    # y 부호는 부팅 홈 유도값의 부호를 따른다(좌우 미러 자동). None = 홈 유도 사용.
    stage_gc_local_override: tuple | None = (0.057, -0.001, 0.064)
    # stay/성공 판정(자매 08.26 규격) — "목표 5cm 안·직립(10°)·정지"를 hold_steps 만큼
    # **연속 유지**해야 ⑤단계 hit. 순간 스침을 정지로 세지 않는다.
    stage_stay_hold_steps: int = 30
    stage_stay_pos_tol_m: float = 0.05
    stage_stay_tilt_deg: float = 10.0
    # 성공 판정의 감쌈 하한 — 공유 계약 상수(success_envelope_min)와 분리된 트랙 전용.
    stage_success_envelope_min: float = 0.75

    # 단계 가중 — 단조 증가(계층 역전 금지). 게이트가 이진이라 실지급도 단조다.
    stage_approach_weight: float = 2.0
    stage_approach_sharpness: float = 8.0
    # ★★08.26 Z-우선 접근(사용자 지시). 수직 커널은 무조건, 수평 커널은 높이가
    #   맞아야(z_ok) 열린다 — 머리 위 호버 로컬최소 제거(좌팔 400ep 정체 실측).
    #   band (0.15, 0.05): |Δz| 5cm 안 = 수평 커널 완전 개방, 15cm 밖 = 0.
    #   z_frac 0.5: 수직/수평 커널 반반 — 높이만 맞추고 안 오는 전략은 상한 절반이라
    #   수평 접근(나머지 절반)이 항상 지배한다.
    stage_approach_z_band: tuple[float, float] = (0.15, 0.05)
    stage_approach_z_frac: float = 0.5
    # close_bridge(자매 05b6a3f) — λ(근접) 상태의 **폐쇄 진행**에 소액. 게이트 개방 후
    # 첫 접촉까지의 gradient 공백(실측: 눈먼 탐색 P(n지≥3) 0.6%)을 다리 놓는다.
    # 접촉하면 contact/grasp 가 덮는다(끄지 않음 — grip-contact-cliff 재발 방지).
    # ★자매 기본값은 0.0(비활성)이고 우리는 사용자 지시로 0.5 — 의도된 값 차이.
    stage_close_bridge_weight: float = 0.25
    # lift_bridge(자매 3ac85a9) — 파지(μ) 상태의 상승 첫 mm~5cm 다리. 리미터가
    # "우연한 상방 요동" 탐색원을 없애 생긴 공백(자매 실측 h 2mm 정체). 상한 1.0.
    stage_lift_bridge_weight: float = 1.0
    # tip_bridge(08.26 사용자 지시) — 손끝 IK 는 **손가락별** gradient 필수:
    # 폐쇄도 스칼라로는 어느 손가락을 어디로 보낼지 못 가른다. λ 상태에서
    # 손끝→물체 거리 커널 평균. 상한 = weight(모든 손끝이 물체 중심 부근일 때).
    stage_tip_bridge_weight: float = 0.5
    stage_tip_bridge_sharpness: float = 8.0
    # ★★08.26 도달 지도 — 컵 스폰을 side-to-side 도달 영역 안으로. 프로필 상수
    #   (_SPAWN_R/L (0.30,∓0.20))는 자매와 공유라 **트랙 전용 오버라이드**로 옮긴다.
    #   실측: 구 스폰 주변엔 성공 셀이 사실상 없고(최근접 62mm 오차), (0.24,∓0.26)
    #   주변은 palm 후보 링(파지 자세 위치들)이 전부 성공권. y 부호는 프로필이 좌우
    #   미러이므로 **|y|** 로 적고 env 가 스폰 중심의 부호를 따른다.
    # ★★08.26 재설정(제약 IK 워크스페이스 기반 역산). 파지 시 palm 은 컵에서
    #   오프셋만큼 물러나 선다: palm_x = cup_x − 0.064 · palm_|y| = cup_|y| + 0.057.
    #   ADR ±0.06 전 범위에서 palm 이 검증 밴드(x 0.06~0.26 · |y| 0.06~0.34) 안이려면
    #   컵 중심 허용창은 x [0.184, 0.264] · |y| [0.06, 0.223] — 중앙부 (0.22, 0.16) 채택.
    #   구 (0.24, 0.26) 은 |y| 가 창 밖(만렙에서 palm_|y| 0.377 > 0.34).
    #   ※(0.24,0.26) 은 동역학 정착 지도의 산물이었고 그 지도는 3중 오염으로 철회됨.
    object_spawn_center_override: tuple[float, float] | None = (0.22, 0.16)
    # ★palm 박스 z바닥 완화(08.26 사용자 인정). 프로필 _BOX_*(0.34)는 자매 공유라
    #   트랙 전용 오버라이드로 낮춘다. 제약 IK 는 z 0.278 도달을 증명했고 실측
    #   palm_ee z 최소가 0.340 = 박스 바닥에 붙어 dz 가 61mm 에서 안 줄었다.
    #   0.30 채택: 컵(0.278)까지 22mm 여유를 주되 테이블 상면(0.20)에서 100mm 띄워
    #   손이 상면을 파고드는 위험을 줄인다(0.26 은 손 하단이 상면 아래로 내려감).
    palm_box_z_min_override: float | None = 0.30

    # ── 역순 커리큘럼 — 컵 옆 pregrasp 시작, ADR reset_near 로 홈까지 후퇴.
    #   play/디버그에서 끌 수 있게 선언 필드로 둔다(getattr 전용이면 hydra
    #   오버라이드가 거부된다 — 실제로 play 가 그렇게 막혔다).
    curriculum_reset_near: bool = True

    # ── sim2real obs 정규화 상수 — **실기 노드와 공유하는 계약**(grasp_v1 동일값).
    #   어긋나면 obs 는 형상만 맞고 값이 틀린다. 실기: F/T 는 손끝 wrench 토픽의
    #   force 3축(tip-local — world 변환 금지), err 는 (보낸 지령 − joint_states).
    contact_force_max: float = 10.0          # [N] 손끝 힘 정규화
    joint_pos_err_max: float = 1.2           # [rad] 손 지령-실측 오차 정규화
    # ★★08.26 사용자 지시 — 고정 관절: **_1 전부 + 소지 _2**.
    #   프로필 상수는 자매 공유라 트랙 전용으로 덮는다. thumb_2 는 "1번 관절" 규칙
    #   밖이지만 대향 자세(rest −90°)를 만드는 관절이라 기존대로 고정을 유지한다.
    #   ★주의(반대 실측 존재): 프로필 주석 08.25 — "pinky 는 _1=회전/_2=굴곡 으로
    #     뒤바뀌어 있어 둘 다 얼리면 밑동 굴곡이 사라진다(접촉률 0.001)". 지시대로
    #     소지 _2 를 얼리면 소지는 사실상 강체가 된다. 소지를 감쌈 분모에서 빼는
    #     hand_unusable_fingers 와 함께 쓰는 것을 전제로 한다.
    # ★★08.27 사용자 지시 — pinky_1 은 굽혀 있으면 안 된다. 프로필(robots.py:296)이
    #   sg*1.047(60°)로 고정하는데, 그 근거는 "q1=60° 라야 pinky_2 가 굴곡축(0.87)이
    #   된다"였다. 그런데 **지금은 pinky_2 도 고정**이라 굴곡축을 만들어 놓고 얼렸다 —
    #   근거가 성립하지 않는다. 남는 효과는 소지가 영구히 벌어진 채 굳는 것뿐이고,
    #   h7 우팔 접촉의 절반(pinky touch 0.543 · wrap 0.290)이 거기서 나왔다:
    #   쥔 게 아니라 벌어진 손가락이 컵에 걸린 **가짜 접촉원**이었다.
    #   ★robots.py 는 자매 트랙과 공유라 손대지 않고 여기서만 덮는다.
    #   ★소지는 감쌈 분모에서 빠져 있어야 지표가 정직하다(hand_unusable_fingers).
    hand_home_override: tuple[tuple[str, float], ...] | None = (
        ("{side}_hj_pinky_1", 0.0),
    )
    frozen_hand_joints_override: tuple[str, ...] | None = (
        "{side}_hj_thumb_1", "{side}_hj_thumb_2",
        "{side}_hj_index_1", "{side}_hj_middle_1",
        "{side}_hj_ring_1", "{side}_hj_pinky_1", "{side}_hj_pinky_2",
    )
    stage_approach_xy_penalty: float = 8.0
    stage_approach_xy_margin: float = 0.025
    stage_approach_tilt_penalty: float = 0.08
    stage_approach_tilt_margin_deg: float = 8.0
    # ★seed-robust(08.26, h6 좌우=2-seed 실측): 좌팔만 e652→745 에 n지 0.21→1.00
    #   급전이, 우팔은 1300ep 배회. 갈림 전 우팔의 close_bridge 가 오히려 높았다
    #   (허공 오므림으로 벌 수 있음) — 접촉의 한계 유인(손가락 1개당 w/5)이
    #   절벽의 원인. 접촉 유인 2배↑ + close_bridge ½ 로 상대가치 4배 교정.
    stage_contact_weight: float = 2.0        # 게이트 밖 shaping — 손가락 1개당 +0.4
    stage_grasp_weight: float = 3.0
    stage_lift_weight: float = 5.0
    stage_transfer_weight: float = 7.0
    stage_stay_weight: float = 10.0
    stage_success_weight: float = 6.0

    # 성공 — 6 인자 연속곱. 전이 하한을 **실측 분포가 걸친 곳**에 둬 실패 반쪽에도
    # gradient 가 남게 한다. ★s_v(속도)는 "목표를 스쳐 지나가도 성공"을 막는다.
    stage_succ_height_band: tuple[float, float] = (0.04, 0.12)
    stage_succ_graspq_band: tuple[float, float] = (0.35, 0.70)
    stage_succ_tilt_band_deg: tuple[float, float] = (18.0, 6.0)
    stage_succ_goal_band_m: tuple[float, float] = (0.09, 0.05)
    stage_succ_speed_band: tuple[float, float] = (0.10, 0.03)


    contact_force_threshold: float = 1.0     # [N] 게이트·감쌈 판정 공통 (dexsuite 동일)
    upright_exponent: float = 4.0            # cos^4 — 소각 판별력(구 관절 모드 전용)
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    # 성공 판정 3조건 — goal 근접 AND 감쌈 AND 직립.
    success_envelope_min: float = 0.6        # g_eff 포화점이자 성공 하한
    success_tilt_max_deg: float = 20.0

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
    # ★08.26 도달 지도(probe_sidegrasp_reach_map) — side-to-side 도달 영역은
    #   x≤0.26·|y|≥0.22 로 좁다. 만렙 ±0.10 이면 컵이 영역을 벗어나 그 에피소드는
    #   기구학적으로 파지 불가(학습 노이즈)다. 0.06 으로 영역 안에 묶는다.
    spawn_range_final: float = 0.06
    # ★★KUKA 고정(08.25) 관측 노이즈 — 원본 teacher 는 전 항목이 *_noisy 다.
    #   ★단 원본은 **전부 ADR 로 0 에서 시작**한다(object_pos_noise 0→0.03 등).
    #     처음부터 걸면 "ADR 끝점에서 시작"이 되어 과제 성립을 방해한다 —
    #     velocity_target_factor 0 · fabric_damping 20 에서 이미 겪은 실수다.
    #   여기 값은 ADR 축의 **최종값**이고, 시작은 0 이다(adr 축 미연결 시 노이즈 없음).
    obs_noise_object_pos: float = 0.03       # [m]   원본 object_pos_noise 최종값
    obs_noise_joint_pos: float = 0.01        # [rad]
    obs_noise_joint_vel: float = 0.1         # [rad/s]
    # 현재 적용 계수(0 = 노이즈 없음). ADR 이 이 값을 0 → 1 로 올린다.
    obs_noise_scale: float = 0.0

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
    _fr_ovr = getattr(cfg, "frozen_hand_joints_override", None)
    _n_frozen = len(_fr_ovr) if _fr_ovr is not None else len(profile.frozen_hand_joints)
    n_free_hand = profile.num_hand_joints - _n_frozen
    cfg.action_space = 6 + n_free_hand      # 팔 6 + 자유 손 관절(풀 관절 단일 경로)
    _T = len(profile.fingertip_bodies)
    cfg.observation_space = (
        2 * j + 3 + 3 + cfg.action_space + 3 + 6 + 3 * _T + j + 6
        + 3 * _T + profile.num_hand_joints)
    if cfg.enable_object_onehot:
        cfg.observation_space += bank.onehot_dim
    # critic = policy + 물체 회전 4 + 접촉력 f + 물체 6D 속도 + fabric qd/qdd(원본 실값)
    # critic = policy + 물체 회전 4 + 접촉력 f + 물체 6D 속도 + fabric qd/qdd 2j
    #          + 물체 scale 3(참값 — pos-only 정책이라 actor 에서 뺀 몫)
    cfg.state_space = cfg.observation_space + 4 + f + 6 + 2 * j + 3


@configclass
class GraspLiftFabricEnvCfg_PLAY(GraspLiftFabricEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
