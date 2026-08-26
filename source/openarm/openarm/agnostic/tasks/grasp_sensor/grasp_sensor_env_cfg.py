"""robot-agnostic grasp-sensor 환경 설정 (v5 형상 비의존 재설계, 08.26).

태스크: 접근 → 인벨롭 파지 → 스폰 지점 위 +15cm goal 로 들어 유지.
로봇 종속 정보는 전부 RobotProfile 에서, **물체 종속 정보는 전부 object_bank 에서** 온다.

★v5 재설계 원칙(사용자 확정):
  ① 물체 형상 상수를 코드에 두지 않는다 — 파지중심·게이트 임계는 손 기하(부팅 FK)
    파생, 스폰·리프트 영점은 object_bank 의 per-spec 오프셋 텐서.
  ② obs 는 sim2real 가능량만(132D 불변 — 물체 치수·질량·클래스 없음).
  ③ 다물체(CUP_FAMILY)가 기본 — 단일 컵 과적합을 스폰 분포에서 차단.
  ④ 접근 거리 역커리큘럼 — 난이도 0 은 물체 옆 프리그래스프에서 시작, 승급이
    시작점을 홈까지 후퇴시킨다(start_pose_frac).

레퍼런스: IsaacLab Dexsuite Kuka-Allegro lift + DexPour 계층 게이트(λ→μ→ν→ρ).
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
from isaaclab.utils import configclass

from isaaclab.envs import mdp as _mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from openarm.agnostic.modules import object_bank as _ob

from .robot_profiles import PROFILES, RobotProfile

# ★08.25 DEXTRAH Kuka EventCfg 값(1.0). 구 0.75 는 우리가 고른 값이다.
_FRICTION = 1.0


@configclass
class GraspSensorEventCfg:
    """도메인 랜덤화 — DEXTRAH Kuka `EventCfg` 이식(08.25). 전 term `mode="reset"`.

    ★씬 기본(SimulationCfg.physics_material)만으로는 부족하다: 로봇·컵 콜라이더는 각자
    재질을 갖고, PhysX 결합이 average 라 한쪽만 올리면 실효 μ 가 중간값이 된다.
    재질 term 의 값은 **절대값**이고(배율 아님), 관절/질량 term 은 `operation="scale"`
    이라 배율이다 — 같은 파일 안에서 의미가 다르니 주의.

    ★Kuka 는 mode="reset" 을 쓴다 — ADR 이 매 리셋마다 범위를 넓히기 때문. 우리는
      아직 ADR 이 없어 값이 고정이지만, 모드를 맞춰 두면 ADR 도입이 상수 교체로 끝난다.
    """

    # ★restitution 1.0 은 `bounce_threshold_velocity=0.2` 와 짝 — 상대속도 0.2 m/s
    #   이하 접촉에는 반발이 적용되지 않으므로 느린 파지 접촉에서는 무해하다(Kuka 규약).
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
    # ★아래 셋은 공칭 파라미터에서 전부 항등(scale 1.0)이거나 0 이라 거동이 바뀌지
    #   않지만, ADR 을 붙일 때 확장 지점이다(Kuka adr_cfg_dict: stiffness/damping
    #   (0.5,2.) · joint_friction (0.,5.) · mass (0.5,3.)).
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

_HDGP_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *([".."] * 6)))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


def _build_robot_cfg(profile: RobotProfile,
                     enable_self_collisions: bool) -> ArticulationCfg:
    """프로필 → ArticulationCfg. 조인트 이름은 전부 프로필에서."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, profile.usd_relpath),
            activate_contact_sensors=True,
            # ★★08.25 DEXTRAH Kuka(`assets/kuka_allegro/kuka_allegro.py`) 값.
            #   ★`max_depenetration_velocity` 1000.0 은 **되돌릴 후보 1순위** —
            #     우리 실측(전형 접촉 13~20N·관통 복원 스파이크 7218N)은 1.0 을
            #     가리켰으나 Kuka 값으로 통일했다. 접촉력 스파이크가 보이면 여기부터.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ★자기충돌 True = Kuka-Allegro 구성. 현재 fabric 반발 쌍은 팔↔팔뚝
                #   16쌍(use_body_repulsion_pairs)이고 손가락↔손가락은 없다 — 손가락
                #   측면 이동은 외전 `_1` 이 open/grip 양쪽 0.0 이라 구조적으로 막혀 있다.
                enabled_self_collisions=enable_self_collisions,
                # ★P-9 실측(2048env): solver 32 는 fps −25% 에 파지 이득 없음. Kuka 8/0.
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


def build_object_cfg(bank: _ob.ObjectBank) -> RigidObjectCfg:
    """물체 spawn cfg. 뱅크가 2종 이상이면 MultiAssetSpawner 를 쓴다.

    ★MultiAsset 은 `clone_environments` **이후**에 RigidObject 를 만들어야 한다
      (modules.object_bank.assert_spawned_after_clone 이 강제). 여기서는 cfg 만 만든다.
    ★rigid_props 는 이 트랙의 검증값(Kuka 물체 규약, depenetration 1000)을 쓴다 —
      grasp_lift_fabric 사본(1.0)과 다른 것은 트랙별 의도적 차이다.
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
        )

    if bank.needs_multi_asset:
        # random_choice=False → proto[env_id % N] 결정론적 배정(assign_indices 와 일치).
        spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_one(s) for s in bank.specs], random_choice=False,
        )
    else:
        spawn = _one(bank.specs[0])

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=spawn,
        # z 는 더미 — 실제 스폰 z 는 env 가 per-spec `_object_rest_z` 텐서로 매 리셋 기입.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.20, 0.30)),
    )


@configclass
class GraspSensorEnvCfg(DirectRLEnvCfg):
    # ---- 로봇 선택 (서브클래스가 덮어씀) ----------------------------------------
    profile_name: str = "tesollo_right"

    # ---- 물체 뱅크 (v5 다물체) ------------------------------------------------------
    # ★형상 정보의 단일 소스 = modules/object_bank.py. 원점 오프셋·질량·rigid body
    #   이름 전부 ObjectSpec 에서 오고, 미측정 오프셋은 fail-loud 다.
    #   `single_cup` 오버라이드 = 보상 변경 단독 검증(V1) 스위치.
    object_bank: str = "cup_family"
    object_bank_expected_size: int | None = 8

    # ---- 시뮬레이션: 물리 120 Hz / 정책 60 Hz ------------------------------------
    episode_length_s: float = 10.0           # 600 스텝 (Kuka 동일)
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # ★씬 기본 물리 재질 — PhysX 결합이 average 라 씬+로봇+컵+테이블 전부 같아야
        #   실효 μ 가 의도값이 된다. Kuka 값 μ=1.0 / restitution 0.0.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            # 용량 계열은 손 27링크 convexDecomposition 부팅 요구 — 물리 거동 무관.
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )
    # ★replicate_physics 는 resolve_cfg 가 뱅크에서 파생한다 — MultiAsset(2종 이상)이면
    #   False 필수(env 별 다른 물체는 physics 복제 불가, object_bank 모듈 함정 ①).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True,
    )

    # ---- 공간 (resolve_cfg 가 프로필에서 계산) ----------------------------------
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # ---- Fabrics (grasp_sensor 검증값 승계) ----------------------------------------
    # Kuka 원본: sim_dt 1/120 · decimation 2 · fabrics_dt 1/60 · fabric_decimation 2.
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    # Kuka ADR `fabric_damping.gain = (10., 20.)` 의 시작값.
    fabrics_damping_gain: float = 10.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # ★★팔 PD 속도 피드포워드(08.25 복구). 0 이면 감쇠항이 참조 궤적을 반대로 밀어
    #   err ≈ 0.2·v 의 추종 지연이 생긴다(lstm_test9 실측 일치). DEXTRAH 원본 = 1.0.
    #   ★이 값이 바뀌면 제어 위에서 잰 상수(폐쇄 속도 등)는 전부 재측정 대상이다.
    fabric_velocity_ff_scale: float = 1.0
    # ★로봇 자기충돌 — Kuka-Allegro 구성(True). 전면 convexHull 이라 decomposition
    #   시절의 2.2~2.8배 비용이 나오지 않는다(그 비용의 본체가 조각 수 — 08.23 해부).
    enable_self_collisions: bool = True
    # ★★손 PD 속도 피드포워드 — 손은 fabric 밖(시너지 램프)이라 램프 자체의 도함수
    #   (Δtarget/정책 dt)를 속도 목표로 준다. Kuka 의 fabric_qd 와 같은 의미의 신호.
    hand_velocity_ff_scale: float = 1.0
    # ★policy obs 측정 속도 계수. Kuka 는 ADR 로 0 을 곱하지만(student 차원 정합용)
    #   당장 student 가 없어 1.0(사용자 결정). student 도입 시 0 으로 = Kuka 배선 복귀.
    obs_measured_velocity_scale: float = 1.0

    # ---- 손 제어 = synergy (v5 단일화) --------------------------------------------
    # ★구 pd/fabric/tip_cyl 경로는 08.26 재설계에서 제거됐다. 근거 요약:
    #   tip IK 계열은 파워그립을 만들 수 없음이 실측 확정(관절각 1.5%·원위접촉 0.00,
    #   IK 잉여 자유도가 "펴진 채 안쪽을 가리키는" 해로 풀림), 관절공간 fabric 은
    #   attractor 가 명령을 보존하지 않아 2지 파지로 붕괴(lstm_test4).
    #   synergy = 관절 목표 직접 보간이라 말아 쥐는 것이 구조적으로 보장된다.
    #   팔은 그대로 fabric 이 몰고, 손만 fabric 밖이다.
    # ---- synergy 그립 상수 (grasp_v1 검증값) --------------------------------------
    # ★액션은 "속도"가 아니라 **절대 폐쇄도 목표**[0,1]이고, 이 값은 변화율 상한이다.
    # ★★08.25 스윕 실측 0.005 채택(150스텝 폐쇄): 느릴수록 감쌈이 좋아지는 단조
    #   경향에서 에피소드 예산과의 절충값. 상한이지 강제 램프가 아니다 — 접촉 동결이
    #   마디별로 먼저 멈추므로 정책은 자유공간 빠르게·접촉 근처 천천히를 배울 수 있다.
    #   ★스윕은 특정 물체 위 측정이라 물체가 크게 바뀌면 재검 대상(동결이 1차 방어).
    synergy_close_speed: float = 0.005
    # ★접촉 시 관절 동결 = 감쌈 생성 메커니즘 — 접촉한 마디가 멈춰 손가락이 물체
    #   형상에 드리워진다. **이것이 형상 적응의 본체다**(보상이 아니라 제어가 맡는다).
    synergy_contact_freeze: bool = True
    # 검지~소지 공통 지령 — 손가락별 국소최적(lstm_test8)을 액션 공간에서 차단.
    couple_four_fingers: bool = True
    # fabric body 반발 쌍(Kuka 패턴: 손 링크 ↔ 팔뚝, 손가락↔손가락 없음).
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = True

    # ---- 접촉 판정 ----------------------------------------------------------------
    # [N] dexsuite 동일 — 자매 트랙(grasp_lift_fabric) 계약 공유. 값 변경 금지.
    contact_force_threshold: float = 1.0
    # ★사다리 접촉 임계 0.1N (grasp_v1 검증: 비영 접촉 하위 5분위 1.86N ≫ 0.1).
    stage_contact_threshold: float = 0.1
    # ★감쌈을 손바닥면 접촉만 인정(08.23 reward-audit ACCEPT) — 손등 접촉이 감쌈으로
    #   세어지는 것을 차단. 프로필에 palmar_axis_local 이 없으면 부팅이 죽는다.
    require_palmar_contact: bool = True
    # ★도달 불가능한 손가락 — 접촉 분모에서 제외. pinky `_1`/`_2` 가 open==grip 이라
    #   lerp 상수(자세표 결함). 자세표를 고치면 () 로 되돌린다.
    hand_unusable_fingers: tuple[str, ...] = ("pinky",)

    # ---- 벌점·종료 ------------------------------------------------------------------
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    abnormal_penalty: float = -1.0
    tilt_reset_deg: float = 60.0             # 초과 = 넘어짐 → truncation 리셋
    object_out_of_bounds_xy: float = 0.35    # 스폰 중심 기준 |Δxy| 초과 시 종료
    object_min_z: float = 0.15               # 테이블 아래로 떨어짐

    # ---- 씬 기하 (물체 무관 — 테이블·스폰 패드) ------------------------------------
    table_surface_z: float = 0.200           # env.usd top_plate 상면(점군 실측)
    object_spawn_pad: float = 0.005          # 스폰 침투 반동 방지
    # ★스폰·정착 z 는 env 가 per-spec `origin_offset_z`(object_bank, scale 반영·미측정
    #   fail-loud)로 `_object_rest_z`(N,) 텐서를 만들어 쓴다 — cfg 스칼라 캐시 없음.
    #   (구 `object_origin_offset_z`/`object_spawn_z` 는 cup_big 전용 상수라 삭제됨.)

    # ---- goal / 성공 판정 ---------------------------------------------------------
    goal_height_offset: float = 0.15         # goal = 물체 정착 위치 + z 0.15
    # dexsuite 규약 success_std/2. ★grasp_lift_fabric 계약 공유 5상수(아래) 값 변경 금지:
    #   contact_force_threshold · goal_height_offset · success_envelope_min ·
    #   success_tilt_max_deg · success_pos_tolerance
    success_pos_tolerance: float = 0.025
    success_pos_tolerance_loose: float = 0.05    # 연속성 비교 로깅 전용
    # ★성공 감쌈 임계(자매 계약). 분모 = 가용 손가락(pinky 제외 3지) → 유효 2지 이상.
    success_envelope_min: float = 0.6
    success_tilt_max_deg: float = 20.0

    # ---- v5 접근 기하 — 손 기하 파생 (물체 상수 0) ---------------------------------
    stage_approach_weight: float = 2.0
    stage_approach_sharpness: float = 8.0
    # ★★Z-우선 접근(08.26): 수직 커널 무조건, 수평 커널은 높이가 맞아야(z_ok) 개방
    #   — 머리 위 호버 로컬최소 제거(grasp_lift_fabric h1 실측).
    stage_approach_z_band: tuple[float, float] = (0.15, 0.05)
    stage_approach_z_frac: float = 0.5
    # ★★v5 수직 데드밴드(08.26): |dz| ≤ z_dead 에서 수직 gradient 정확히 0.
    #   구 exp(−s·|dz|)는 |dz|=0 정확 일치까지 밀어 파지중심 오프셋이 수평이 되는
    #   자세에서 손을 테이블로 보냈다(corridor_v3_s777 실증: 손바닥 테이블 정지 국소
    #   최적). 데드밴드 안 높이는 접촉 사다리·동결이 정한다. z_band 하한과 단일 소스
    #   (부팅 어서션 강제). 공유 수식의 getattr 기본 0.0 = 구식과 항등(자매 하위호환).
    stage_approach_z_dead: float = 0.05
    # ★★v5 λ 게이트 임계 — 손 기하 파생(08.26). 부팅 FK 로
    #   r_cage = 0.5·‖tip_thumb − mean(tip_others)‖ 를 재고 scale 을 곱해
    #   `stage_gate_approach_m` 에 기입한다(구 0.12 는 cup_big 실측 d_gc 튜닝값).
    #   0.0 = 부팅 파생 신호 · >0 = 명시 오버라이드(프로브 전용).
    stage_gate_approach_scale: float = 1.25
    stage_gate_approach_m: float = 0.0
    # ─ 자세(approach 인자) — palm_ee +x = 손바닥 법선 ⊥ 컵축, +y ∥ 컵축 ─────────
    stage_perp_exponent: float = 2.0     # (1 − |cos(palm_x, cup_z)|) ** e
    stage_roll_exponent: float = 4.0     # cos(palm_y, cup_z) ** e — 롤 자유도 잠금
    stage_orient_floor: float = 0.15     # 자세 최악에도 approach 의 15% 는 남긴다
    # 정렬 배수의 바닥 — 0 이면 초기 오정렬에서 approach gradient 소멸(Check1).
    stage_align_floor: float = 0.25

    # ---- 파지 품질 Q_g 배합 (합 1.0, 부팅 어서션) ----------------------------------
    stage_graspq_touch: float = 0.25     # 닿았나
    stage_graspq_deep: float = 0.55      # **두 마디 동시** = 실제 감쌈(팁 스침 불가)
    stage_graspq_persist: float = 0.20   # 유지하는가
    stage_graspq_thumb_floor: float = 0.30   # 엄지 없이 4지만 긁으면 상한 30%
    stage_thumb_force_ref: float = 0.5   # [N] 소프트 대향(접촉임계의 5배·p95 의 1/15)
    # 접촉 지속 정규화 스텝수(grasp_v1).
    stage_contact_persistence_steps: int = 20

    # ---- 사다리 가중 (인자가 깊어질수록 상한 확대) ---------------------------------
    stage_contact_weight: float = 1.0    # 게이트 없음 — λ=1·μ=0 사각지대 방지 shaping
    stage_grasp_weight: float = 3.0
    stage_lift_weight: float = 5.0
    stage_transfer_weight: float = 7.0
    stage_stay_weight: float = 10.0
    stage_success_weight: float = 6.0
    stage_lift_height_ref: float = 0.15  # goal_height_offset 과 정렬(부팅 어서션)
    stage_tracking_std: float = 0.1
    # 직립 요구는 단계별로 다르다(08.26 사용자 규격: 이송 중 20° 관용·정지 시 직립).
    stage_tilt_tolerance_deg: tuple[float, float] = (30.0, 20.0)  # lift/transfer
    stage_upright_gate_deg: tuple[float, float] = (15.0, 5.0)     # stay/success
    # 컵 밀림 감쇠 — 제곱역수(선형은 d≥L 에서 gradient 소실).
    stage_disp_limit: float = 0.06
    # 컵 밀기·기울임 벌점(grasp_v1 이식, 08.25 실측 재조정 25→8).
    stage_approach_xy_penalty: float = 8.0
    stage_approach_xy_margin: float = 0.025
    stage_approach_tilt_penalty: float = 0.08
    stage_approach_tilt_margin_deg: float = 8.0

    # ---- 계층 게이트 λ→μ→ν→ρ (DexPour, IROS 2025 식 3~6) --------------------------
    stage_gate_contact_n: float = 3.0     # 가용 손가락 중 3지
    stage_gate_lift_m: float = 0.05       # 목표 0.15 의 1/3 에서 이송 개방
    stage_gate_transfer_m: float = 0.08   # d_goal 시작 0.15 의 절반

    # ---- 성공 — 연속 곱(이진 AND 폐기). 전이 구간은 실측 분포가 걸친 곳 ------------
    stage_succ_height_band: tuple[float, float] = (0.04, 0.12)
    stage_succ_graspq_band: tuple[float, float] = (0.35, 0.70)
    stage_succ_tilt_band_deg: tuple[float, float] = (18.0, 6.0)
    stage_succ_speed_band: tuple[float, float] = (0.10, 0.03)    # 정지(내려가는 전이)
    stage_succ_goal_band_m: tuple[float, float] = (0.09, 0.05)   # 사용자 규격 5cm
    # stay 단계 판정(로깅) 전용 — success_* 는 자매 계약이라 불변.
    stage_stay_pos_tol_m: float = 0.05
    stage_stay_tilt_deg: float = 10.0
    stage_stay_speed_ref: float = 0.05   # S = exp(−|v_obj|/v_ref)
    stage_stay_hold_steps: int = 30      # 성공률 로깅 — 0.5 초 연속 유지

    # ---- 코리더 래치 (08.26 승인: 래치 + 느슨한 시작, 난이도 연동 조임) -------------
    # 순간 게이트는 통행료로 우회된다(낚아챔 실측 253mm·49°·60스텝 통행료 7%) —
    # 에피소드 중 한 번이라도 코리더를 넘으면 그 에피소드의 ν 이후를 몰수한다.
    stage_corridor_xy_m: tuple[float, float] = (0.20, 0.05)
    stage_corridor_tilt_deg: tuple[float, float] = (50.0, 20.0)

    # ---- palm 지령 rate limit (제어층 대책 — 보상과 역할 분리) ----------------------
    # 절대 매핑의 스텝당 텔레포트(실측 평균 632mm·84°/step)를 지령 변화율로 묶는다.
    # 0.0 = 비활성. 학습 런은 오버라이드로 켠다(v4 검증값: 0.1 m / 15°).
    palm_cmd_rate_limit_m: float = 0.0
    palm_cmd_rate_limit_rot_deg: float = 0.0

    # ---- gradient 다리 (리미터가 "우연한 발견" 탐색원을 제거해 생긴 공백) ------------
    # close_bridge = w·λ·min(엄지, 4지) — 근접 상태의 폐쇄 시작에 소액 지급.
    # lift_bridge  = w·μ·(h/gate_lift).clamp(0,1) — 파지 상태의 첫 mm 부터 지급.
    # 접촉·리프트가 열리면 상위 항이 덮는다(끄지 않음 — grip-contact-cliff 함정).
    stage_close_bridge_weight: float = 0.0
    stage_lift_bridge_weight: float = 0.0
    # tip_bridge — 자매 트랙(손끝 IK) 전용 인자. 이 트랙은 synergy 라 미배선(0 고정).
    stage_tip_bridge_weight: float = 0.0
    stage_tip_bridge_sharpness: float = 8.0

    # ---- 커리큘럼 (per-env 난이도 0~10) --------------------------------------------
    curriculum_max_level: int = 10
    # 스폰 xy 반경: 초기 → 최종 보간. final 박스는 "홈 팔 quiet 영역" 안(probe 실측).
    spawn_range_initial: float = 0.02
    spawn_range_final: float = 0.08
    # ★★v5 접근 거리 역커리큘럼(08.26 사용자 지시). 리셋 팔 관절 =
    #   lerp(q_pregrasp, q_home, start_frac), start_frac 은 난이도 0→만렙으로
    #   (initial, final) 보간. q_pregrasp 는 부팅 fabric 오프라인 롤아웃으로 파생
    #   (하드코딩 없음·FK 검증 fail-loud). (1.0, 1.0) = 축 OFF = 구 리셋과 항등.
    #   ★fabric cspace rest(default_config)는 부팅 1회 상수로 불변 — 리셋 자세만
    #     바뀌고 널스페이스 유인·도달영역은 고정이다(계약 테스트 강제).
    #   ★배포는 만렙 체크포인트만(실기 = 항상 홈 시작 = 만렙 분포).
    start_pose_frac: tuple[float, float] = (0.0, 1.0)

    # ---- 씬 --------------------------------------------------------------------------
    # 실기 환경 USD (테이블 상면 z 0.200, 기둥/받침/바닥판 포함 전부 충돌체).
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "env/usd/env.usd"),
        ),
    )
    # ★물체 cfg·접촉 필터는 resolve_cfg 가 뱅크에서 파생한다(하드코딩 금지 —
    #   rigid body 이름이 틀리면 force_matrix_w 가 무증상 0 이 된다).
    object_cfg: RigidObjectCfg = None
    object_contact_filter: tuple = ()

    # 물리 재질 이벤트(로봇·컵). 테이블은 scene 자산이 아니라 정적 프림이라
    # env 가 clone 전에 bind_physics_material 로 직접 건다.
    events: GraspSensorEventCfg = GraspSensorEventCfg()
    surface_friction: float = _FRICTION

    robot_cfg: ArticulationCfg = None  # resolve_cfg 에서 프로필로 조립

    def __post_init__(self):
        resolve_cfg(self)


def resolve_cfg(cfg: GraspSensorEnvCfg) -> None:
    """cfg 의 파생값을 (재)해석한다 — 멱등.

    ★★hydra 는 이미 생성된 cfg 의 필드만 덮어쓰고 `__post_init__` 을 다시 돌리지
    않는다. `env.object_bank=...` 오버라이드가 spawner·필터·replicate_physics 를
    못 바꾸는 함정을 막기 위해 env `__init__` 이 이 함수를 재호출한다
    (grasp_lift_fabric 검증 패턴).
    """
    profile = PROFILES[cfg.profile_name]
    cfg.robot_cfg = _build_robot_cfg(profile, bool(cfg.enable_self_collisions))

    # ---- 물체 뱅크 파생 ------------------------------------------------------------
    bank = _ob.get(cfg.object_bank, expected_size=cfg.object_bank_expected_size)
    cfg.object_cfg = build_object_cfg(bank)
    cfg.object_cfg.init_state.pos = (
        profile.object_spawn_center[0], profile.object_spawn_center[1], 0.30,
    )
    cfg.object_contact_filter = (
        f"/World/envs/env_.*/Object/{bank.rigid_body_name}",
    )
    if bank.requires_replicate_physics_off:
        cfg.scene.replicate_physics = False

    # ---- 공간 ----------------------------------------------------------------------
    num_joints = profile.num_arm_joints + profile.num_hand_joints
    num_tips = len(profile.fingertip_bodies)
    num_fingers = len(profile.finger_sensor_bodies)
    # 액션 = palm 6D 절대 pose + 손가락 × synergy 채널(프로필 파생).
    _nch = len(set(profile.hand_channel_of_joint.values()))
    cfg.action_space = 6 + _nch * len(profile.finger_sensor_bodies)
    # policy obs = 관절각·속도(2·nj) + palm pos(3) + palm 회전 2열(6) + 손끝(3·nt)
    #              + 물체 pos(3) + goal(3) + last action + fabric q(nj)
    # ★물체 치수·질량·클래스·scale·onehot 은 **넣지 않는다** — 형상 비의존 + 배포
    #   시 알 수 없는 정보(사용자 결정 08.26). 접촉력·obj_quat 은 critic 전용.
    cfg.observation_space = (
        2 * num_joints + 3 + 6 + 3 * num_tips + 3 + 3 + cfg.action_space
        + num_joints
    )
    # critic = 관측 + 접촉력(num_fingers) + 물체 quat(4) + 물체 속도(6)+난이도(1)
    #          + 측정 관절토크(num_joints)
    cfg.state_space = (
        cfg.observation_space + num_fingers + 4 + 7 + num_joints)


@configclass
class GraspSensorTesolloRightEnvCfg(GraspSensorEnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspSensorGripperLeftEnvCfg(GraspSensorEnvCfg):
    profile_name: str = "gripper_left"
