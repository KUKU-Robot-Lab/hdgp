"""pour_fabric_mimic 환경 설정 — 양팔 **잡기→들기→붓기**, 저차원(언더액추·PhysX mimic) 손 전용.

★`pour_fabric/pour_fabric_env_cfg.py` 사본(09.14). 원본 트랙은 손대지 않는다. 다른 점:
  · 쌍 레지스트리 = 이 트랙의 `bimanual`(RH56F1 Fabrics 프로필, 기본 쌍 "rh").
  · 손 액션 폭 = 구동관절 수(per_finger, RH56F1 6) → 액션 24 = (palm 6 + 손 6) × 2.
  · `oppose_grip_delta_rad = 0.0`(tesollo 엄지 대향 전용 노브 — RH56F1 ch1 은 엄지 굴곡이라
    −0.6 을 주면 grip 이 한계 밖으로 잘려 엄지가 안 접힌다, 09.02 실측).
  · 접촉 동결은 finger 스코프만(손가락당 구동관절 1개).
  · 종속관절 한계 부팅 확장 `mimic_dep_limit_margin_rad`(grasp_fj_rh 09.07 실측 근거).
  · `_validate_mimic_fields` 가 조용히 무효가 되는 조합을 cfg 단계에서 죽인다.

(아래는 원본 설명)

★09.13 재작성 이유. 구판은 warm 뱅크(이미 잡은 상태)에서 시작하고 손을 동결한 채 붓기만
  배웠다. 이번 트랙은 text2reward 방식의 **보상 자동생성**이 목적이라 과제 문장("양팔로
  컵을 각각 잡고 비드 있는 컵을 없는 컵으로 옮긴다")을 통째로 정책이 배워야 한다 —
  테이블 위 컵 두 개에서 시작하고 손 20관절도 정책(시너지)이 제어한다.
  보상은 이 파일에 **없다**: `reward_code_path` 의 생성 코드가 `RewardContext` 를 읽어
  계산한다(`modules/t2r`). 비어 있으면 영 보상(부팅/무작위 롤아웃용).

★09.14 sim2real(라운드 3, 사용자 지시): actor 관측에서 손 관절속도 제거(실기 드라이버 velocity
  는 관절속도가 아니다 — 09.07 실측), 컵 pose 지각 지연+코히런트 노이즈, 관절 노이즈,
  물리 DR(컵 질량·관절 게인·컵 마찰) + 들린 컵 외란 — 전부 grasp_s2r/grasp_kp 모듈 재사용,
  ADR(순간 성공률 트리거)로 중립 → 종점 확장. 충돌 신호(컵끼리·손↔타물체)를 ctx 에 노출.

제어 스택 = grasp_s2r 현행(팔 Fabrics ×2 · 손 관절공간 시너지 + 접촉 동결) 을 양팔로.
물리 블록 = 구 pour_fabric(비드 20개×N env 접촉 버퍼) 그대로.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as _mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules import object_bank as _ob
from openarm.common.bead_assets import DEFAULT_BEAD_COUNT, make_beads_cfg

from . import bimanual as _bm

_ASSETS_DIR = _ob.ASSETS_DIR

# =============================================================================
# 컵 자산 — pour 전용 **SDF** 콜라이더(convex hull 은 컵 속을 메워 비드가 안 담긴다).
# 원점 오프셋·내부 기하는 pour_v1 실측(.usd bbox) — bead_flags 판정과 같은 값.
# =============================================================================
# ★09.14 사용자 결정: 컵 대신 **shaker 계열**(`shaker_closed_rl.usd` = 바닥 플러그 + 열린 상단, SDF 콜라이더,
#   object_bank 09.01 pxr 실측: 바닥 −0.0921 · 림 +0.0829 · 내경 r 0.0432 · 외경 r 0.0440, 벽 0.8 mm).
#   RH56F1 은 엄지–4지 간극 최대 83.7 mm·실사용 ~70 mm(object_bank SHAKER_SMALL 주석) → cup_scale 0.65 = 지름 57 mm
#   (Track B shaker_one 과 같은 크기). 인벨롭 파지는 포기, 손끝 파지로 잡아도 된다(사용자 09.14).
POUR_CUP_USD = os.path.join(_ASSETS_DIR, "cup", "shaker_closed_rl.usd")
POUR_CUP_ORIGIN_OFFSET_Z = 0.0921     # 바닥 −0.0921 (scale 1)
POUR_CUP_MASS = 0.134                 # = object_bank.BASE_OBJECT_MASS (스케일 무관 실물값)

SOURCE_CUP_PRIM = "/World/envs/env_.*/SourceCup"
RECEIVER_CUP_PRIM = "/World/envs/env_.*/ReceiverCup"
# ★shaker USD 는 강체(RigidBodyAPI)가 루트가 아니라 `baseLink` 하위 prim 이다(cup_big_sdf 는 루트).
#   접촉 센서 prim 과 손가락 센서의 필터는 **강체 prim** 을 가리켜야 한다 — 어긋나면 PhysX 가 조용히 힘 0 을 준다
#   (관측 #0230: 필터 0개 매칭 = 로그 한 줄 + 전부 0). env 가 부팅에서 매칭 수를 검사한다.
POUR_CUP_BODY_NAME = "baseLink"
SOURCE_CUP_BODY = SOURCE_CUP_PRIM + "/" + POUR_CUP_BODY_NAME
RECEIVER_CUP_BODY = RECEIVER_CUP_PRIM + "/" + POUR_CUP_BODY_NAME
TABLE_PRIM = "/World/envs/env_.*/Table"


# 컵 기하(스케일 1.0 기준) — resolve_cfg 가 `cup_scale` 을 곱해 cfg 필드를 채운다(멱등).
CUP_GEOM_UNIT = {"cup_inner_radius": 0.0432, "cup_inside_z_min": -0.085, "cup_inside_z_max": 0.0829,
                 "cup_mouth_z": 0.0829, "object_origin_offset_z": POUR_CUP_ORIGIN_OFFSET_Z}


def build_cup_cfg(prim_path: str, scale: float = 1.0) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=POUR_CUP_USD,
            scale=(scale, scale, scale),
            activate_contact_sensors=True,
            # ★질량은 스케일과 무관하게 실물 컵 값(0.134 kg)을 유지한다 — s³ 로 줄이면 0.6 배 컵이 29 g 이 되어
            #   손가락이 스치기만 해도 넘어진다(09.14 실측). 크기만 줄이는 것이 사용자 의도다.
            mass_props=sim_utils.MassPropertiesCfg(mass=POUR_CUP_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,       # 7218N 스파이크 근거(구 pour)
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.20, 0.30)),
    )


def build_robot_cfg(pair: _bm.BimanualPair, self_collisions: bool,
                    gravity: bool) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS_DIR, pair.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=not gravity,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict(pair.init_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            name: ImplicitActuatorCfg(**spec)
            for name, spec in pair.actuator_specs.items()
        },
        soft_joint_pos_limit_factor=1.0,
    )


# 작업면 — env_v1.usda (top_plate 상면 z=0.205, 09.05 CAD 정정·실기 줄자 일치).
TABLE_SPAWN = sim_utils.UsdFileCfg(
    usd_path=os.path.join(_ASSETS_DIR, "simulation_setting/env_v1/usd/env_v1.usda"),
)


# =============================================================================
# 물리 DR EventTerm — 초기 범위는 **중립**(항등). ADR 이 종점까지 선형 확장한다
# (`modules/physics_dr.py` 규약; 자산 이름이 두 개라 여기서 따로 정의).
# ★재질 term 은 term 생성 시 버킷을 1회 샘플링해 **런타임 확장이 무증상 no-op** 이다
#   (grasp_s2r `_adr_apply_physics` 주석) → 컵 마찰은 cfg 단계에서 고정 범위로 연다.
# =============================================================================
def _material_term(asset: str, lo: float, hi: float) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_material, mode="reset",
        params={"asset_cfg": SceneEntityCfg(asset, body_names=".*"),
                "static_friction_range": (lo, hi), "dynamic_friction_range": (lo, hi),
                "restitution_range": (1.0, 1.0), "num_buckets": 250})


def _mass_term(asset: str) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_mass, mode="reset",
        params={"asset_cfg": SceneEntityCfg(asset), "mass_distribution_params": (1.0, 1.0),
                "operation": "scale", "distribution": "uniform"})


@configclass
class PourFabricEventCfg:
    robot_material = _material_term("robot", 1.0, 1.0)
    source_cup_material = _material_term("source_cup", 1.0, 1.0)     # resolve_cfg 가 범위 적용
    receiver_cup_material = _material_term("receiver_cup", 1.0, 1.0)
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=_mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (1.0, 1.0),
                "damping_distribution_params": (1.0, 1.0),
                "operation": "scale", "distribution": "uniform"})
    source_cup_scale_mass = _mass_term("source_cup")
    receiver_cup_scale_mass = _mass_term("receiver_cup")


# ADR 종점(grasp_s2r E1 값) — 질량·게인만 ADR 로 확장(재질은 위 이유로 cfg 고정).
PHYSICS_ADR_TERMINAL = {
    "source_cup_scale_mass": {"mass_distribution_params": (0.5, 2.5)},
    "receiver_cup_scale_mass": {"mass_distribution_params": (0.5, 2.5)},
    "robot_joint_stiffness_and_damping": {
        "stiffness_distribution_params": (0.5, 2.0),
        "damping_distribution_params": (0.5, 2.0)},
}


# =============================================================================
@configclass
class PourFabricMimicEnvCfg(DirectRLEnvCfg):
    """차원은 resolve_cfg 가 pair 로 확정한다."""

    pair_name: str = _bm.DEFAULT_PAIR

    # ---- 보상 (text2reward 생성 코드) ----------------------------------------------
    # 빈 문자열 = 영 보상. 학습 런은 반드시 생성·검증을 거친 파일을 가리켜야 한다.
    reward_code_path: str = ""

    # ---- 시뮬레이션 (물리 = 구 pour_fabric, 비드 버퍼 포함) ---------------------------
    episode_length_s: float = 15.0            # 900 스텝 @60Hz
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            # 비드 20개 × N env 접촉 폭증 — pour_v1 검증값(128 env overflow 크래시 끝에 확정).
            gpu_found_lost_aggregate_pairs_capacity=64 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 1024,
            gpu_max_rigid_patch_count=2 ** 24,
            gpu_max_rigid_contact_count=2 ** 24,
            gpu_collision_stack_size=2 ** 30,
            gpu_max_num_partitions=64,
            friction_correlation_distance=0.00625,
        ),
    )
    # ★09.14 4096 env 실측 43 GB·10.7k fps(서버 RTX PRO 6000). 128 은 pour_v1 시절 값.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True)

    # ---- 물리 스위치 --------------------------------------------------------------
    enable_gravity: bool = True
    gravity_compensation: float = 1.0
    # ★RH56F1 은 자기충돌 OFF(형제 grasp_s2r/grasp_fj/grasp_fj_rh 의 D3 기본 09.01). 09.14 부팅 실측:
    #   ON 이면 시작 자세에서 r_hl_index_tip 에 226 N 의 자기접촉 힘이 상시 걸리고(foreign_force 오염),
    #   폐쇄 0.7 부터 index_2·thumb_3/4 가 이웃 손가락에 밀려 되돌아가 mimic 오차 0.9 rad 가 난다.
    enable_self_collisions: bool = False
    surface_friction: float = 1.0             # 테이블 재질(컵-테이블). grasp_s2r 규약으로 바인딩.
    ground_plane_z: float = -0.10

    # ---- 작업면·컵 기하 -------------------------------------------------------------
    # ★물체 = shaker_closed × cup_scale(위 주석). 아래 기하 필드 5개는
    #   resolve_cfg 가 CUP_GEOM_UNIT × cup_scale 로 **덮어쓴다**(hydra 로 개별 기하를 덮지 말 것). 질량은 실물값 유지.
    cup_scale: float = 0.65
    # 비드 소환: xy 는 검증 배치(bead_offsets_in_cup) 그대로(반경 14/18 mm < 내경 28 mm), z 는 바닥 기준으로 다시 쌓는다
    #   (원본 z 는 cup_big 원점 기준이라 shaker 0.65 에서는 림 위로 나간다). 층 간격 14 mm 유지.
    bead_z_from_bottom: float = 0.008
    bead_layer_dz: float = 0.014
    table_surface_z: float = 0.205
    object_origin_offset_z: float = POUR_CUP_ORIGIN_OFFSET_Z
    object_spawn_pad: float = 0.005           # 스폰 침투 반동 방지
    object_spawn_range: float = 0.02          # 스폰 중심 xy 균등 ± [m]
    cup_inner_radius: float = 0.041
    cup_inside_z_min: float = -0.070          # bottom(-0.077) + bead 반경 여유
    cup_inside_z_max: float = 0.100           # 림
    cup_mouth_z: float = 0.100
    bead_count: int = DEFAULT_BEAD_COUNT
    # 리셋 직후 비드 정착 대기 — 이 동안 팔은 시작 자세에 고정되고 액션은 무시된다.
    hold_steps: int = 30

    # ---- Fabrics (= grasp_s2r 현행) -------------------------------------------------
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 10.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    fabric_velocity_ff_scale: float = 1.0
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = True
    fabric_table_obstacle: bool = True
    fabric_table_margin_xy: float = 0.10
    fabric_table_thickness: float = 0.05
    fabric_fk_pos_tol: float = 0.005          # 부팅 게이트: fabric FK vs USD palm (2스텝 처짐 1~2mm 가 바닥 — 실측 0.5~1.7mm)

    # ---- 팔 액션: palm 6D = 앵커(시작 자세 palm 실측) + 델타 ---------------------------
    # 델타 박스 (x,y,z [m] · ez,ey,ex [deg]) — lo/hi 비대칭. a=0 이 앵커(=컵 옆 시작 자세).
    # 소스: 붓기 tilt 축 구간을 깊게 연다.
    # ★회전 슬롯 순서 = `_palm_pose_6d` [yaw(ez), pitch(ey), roll(ex)] = fabric "euler_zyx" 지령.
    #   ★09.13 프로브 실측(접근→파지→리프트 뒤 슬롯별 −1 지령): 슬롯 3(yaw)에 −150° 는 도달
    #     불가(회전오차 110°, 컵 tilt 24°)이고 **슬롯 5(roll)** 가 오차 1.3° 로 추종해 컵 tilt 가
    #     지령대로 나온다(−45° → 44.4°). 붓기 축은 **roll = 슬롯 5** 다 — 구 pour cfg 주석
    #     ("첫 슬롯이 roll")은 틀렸다. 리시버(좌)는 별도 cfg(대칭 ±30°)라 미러 문제 없음.
    # ★RH56F1 홈(Track B 캘리브)은 palm 이 컵 원점보다 13.7 cm 위·14.6 cm 바깥이라(09.14 부팅 실측
    #   palm−cup = (−0.08, −0.146, +0.137)) z 하한을 −0.12 → −0.18 로 연다. 나머지는 원본 값.
    src_palm_delta_lo: tuple = (-0.15, -0.10, -0.18, -45.0, -45.0, -150.0)
    src_palm_delta_hi: tuple = (0.15, 0.40, 0.25, 45.0, 45.0, 45.0)
    rcv_palm_delta_lo: tuple = (-0.15, -0.40, -0.18, -30.0, -30.0, -30.0)
    rcv_palm_delta_hi: tuple = (0.15, 0.10, 0.25, 30.0, 30.0, 30.0)
    # 위치는 프로필 palm 박스로 추가 clamp(회전은 델타 박스만).

    # ---- 손: 관절공간 시너지 (= grasp_s2r 현행 coupled3) --------------------------------
    synergy_close_speed: float = 0.005        # 폐쇄도 변화율 상한 / 정책 스텝
    synergy_contact_freeze: bool = True       # 닿은 손가락의 구동관절 정지 → 감쌈은 하드웨어 결합이 만든다
    synergy_freeze_scope: str = "finger"      # 이 트랙은 finger 만(손가락당 구동관절 1개)
    couple_four_fingers: bool = True          # 엄지 독립, 나머지 4지 채널별 평균
    finger_residual_scale: float = 0.0
    # 대향 관절(엄지 ch1) grip = open + delta — grasp_s2r D3 기본. **소스 팔 부호 기준**,
    # 리시버(좌)는 미러 부호(thumb_2 축 Z → −1)를 env 가 적용한다.
    # ★RH56F1: 0.0 고정(엄지 자세는 프로필 open/grip 표 1.57→1.20 · 0.0→0.24 가 담는다).
    oppose_grip_delta_rad: float = 0.0
    hand_velocity_ff_scale: float = 1.0
    joint_pos_err_max: float = 1.2            # obs 정규화 [rad]
    # ★09.13 부팅 실측: 시작 자세에서 palm↔컵 중심이 약 0.16 m(손끝이 3 cm 앞) — 반경은
    #   그보다 커야 시작 자세에서 닫을 수 있다. 0.22/램프 0.3 → 0.154 m 안쪽은 게이트 1.0.
    # ★RH56F1 시작 자세: palm(0.31,−0.30,0.42) ↔ 컵 원점(0.38,−0.16,0.28) ≈ 0.21 m(FK 계산) —
    #   원본 0.22/램프 0.3 이면 시작에서 게이트 0.18 이라 닫기가 거의 막힌다. 0.30/0.3 → 0.21 m 에서 1.0.
    #   ⚠probe 로 실측 후 조정(`probe_pour_fabric_mimic_boot.py` 부팅 로그의 palm↔컵 거리).
    close_gate_enabled: bool = True
    close_gate_radius: float = 0.30
    close_gate_ramp: float = 0.3

    # ---- 언더액추 백스톱 (grasp_fj_rh 09.07 실측 근거 그대로) ------------------------------
    # 접촉이 구동관절을 하한 밑으로 역구동하면 mimic 요구치가 종속 한계 밖이 되어 두 제약이 동시에
    # 만족 불가 → 솔버가 에너지를 주입(rh_b1: 222 epoch 중 118 에서 mimic 오차 400~916 rad).
    # 종속관절 한계를 부팅에서 이만큼 더 넓힌다(위치 권한은 mimic 제약, 한계는 백스톱). 0 = 자산 그대로.
    mimic_dep_limit_margin_rad: float = 1.5
    # ★09.14 실측(t2r_rh_i00/_m3): 손끝으로 컵을 누르며 드는 순간 mimic 결합이 깨져 종속관절이 154~483 rad 튀고 속도 500~5e4 rad/s
    #   가 난다. 한계 여유 3.0 rad 로도 재현 → 여유는 처방이 아니다. 깨진 env 는 에피소드 끝까지 무의미한 물리로 남으므로
    #   **종속관절 속도가 이 값을 넘으면 runaway 로 종료**한다(사용자 결정 "폭주 오면 리셋"). 건전 기준선 09.07: 자유 폐쇄 ≤7.2,
    #   접촉 킥 수십~수백 rad/s → 100 은 킥 상단·폭주(500+) 사이.
    mimic_runaway_dep_qd: float = 100.0
    # ★09.14 라운드 1(t2r_rh_i00_r1): 오차가 속도 100 rad/s 아래로 천천히 벌어진 폭주 2회(epoch 321-323 39→236 rad,
    #   348-351 7→18 rad)가 위 기준을 빠져나갔다. 정상 epoch 의 전 env 최대 오차는 ≤1.7 rad → 결합 오차 자체도 종료 조건
    #   (사용자 결정 09.14 "2 추가").
    mimic_runaway_err_rad: float = 3.0

    # ---- 접촉 --------------------------------------------------------------------
    contact_force_threshold: float = 1.0      # N — 파지(대향) 게이트·동결 판정
    contact_obs_clip: float = 20.0
    # ---- 손끝 촉각 actor obs (사용자 결정 09.14 "3 추가") ------------------------------------
    # 실기 출처: RH56F1 TouchData1.finger_forces[5] (정전용량 손끝 법선력, 0.01 N 단위, 1024 = 10.24 N).
    # 실기 센서는 무엇에 닿든 재므로 sim 도 컵 필터가 아닌 손끝 링크 전체 접촉력(net). 배포 시 손가락 순서 재배열 필요.
    tactile_obs_clip_n: float = 10.0
    tactile_obs_noise_n: float = 0.1
    # ---- 보상 입력(RewardContext) 손바닥 축 — 계약 "palm_axes 앞 3칸 = 손바닥 법선" (09.15) --------------
    # RH56F1 palm_sensor 링크: 열 0 = 손가락이 늘어선 가로(기저 ±y), 열 1 = 손가락 길이(기저 +z), 열 2 = 손바닥 법선(기저 +x,
    # 엄지 기저가 이쪽). 열 0·1 을 그대로 넣으면 iter_03 orient 가 손 옆날을 컵으로 돌리게 보상했다. 정책 obs 는 무관.
    ctx_palm_normal_col: int = 2
    ctx_palm_second_col: int = 1
    collision_force_threshold: float = 1.0    # N — 컵끼리·손↔타물체 충돌 지표 임계
    # ---- 엄지 입구 걸림 접근 계측(로그 전용, 09.15 사용자 "지표로깅으로 확인 가능하게") --------------
    # task/{src,rcv}_near_rate · _thumb_over_rim_near · _thumb_above_rim_mm_near. iter_03 rim_hook 과 같은 기하.
    thumb_rim_near_m: float = 0.10            # palm↔컵 원점 거리가 이 안이면 "접근한 env"
    thumb_rim_band_m: float = 0.02            # 엄지 끝 축방향 높이 ≥ 입구 − 이 값이면 입구 높이
    thumb_rim_radial_margin_m: float = 0.025  # 엄지 끝 반경 < 벽(내경+4 mm) + 이 값이면 입구 위
    # ---- 파지 포켓 계측(로그 전용, 09.16 사용자 "대향 여부를 영상 없이 지표로") -----------
    # task/{src,rcv}_thumb_oppose_near · _tip_gap_mm_near · _cup_in_pocket_near.
    # 라운드 7 영상: 배치가 검지-엄지-컵 이라 손가락을 굽혀도 잡을 수 없는데 보상의 pinch_geo 는
    # 그런 자세에도 0.19~0.38 을 지불했다(합성 검증 09.16). 그래서 대향을 따로 잰다.
    pocket_near_m: float = 0.12               # palm↔컵 원점 거리가 이 안이면 "접근한 env"
    pocket_axis_margin_m: float = 0.005       # 컵 축↔(엄지,4지) 선분 거리 < 내경 + 이 값이면 컵이 사이

    # ---- 성공 판정 (pour_v1 계승) — 보상과 분리된 **기준 지표** ----------------------
    success_fill_ratio: float = 0.50
    success_spill_max: float = 0.40
    success_xy_thresh: float = 0.20           # 두 컵 중심 xy 거리
    success_hold_steps: int = 10
    # ★09.13 hacking 차단: 소스 컵을 리시버 입구에 끼워 넣으면 소스 안 비드가 리시버 원통 안에 들어와
    #   in_target 로 세어졌다(ep 600 영상: 붓기 없이 성공 0.73). 원점 거리가 이보다 짧으면 성공 무효.
    #   붓는 자세(소스 입구가 리시버 림 위)에서는 원점 거리가 ≥ 12~15 cm 다.
    cups_nested_dist: float = 0.09

    # ---- 종료 --------------------------------------------------------------------
    runaway_joint_vel: float = 20.0
    drop_below_table_m: float = 0.03          # 컵 원점이 (테이블 상면 + 원점오프셋 − 이 값) 아래면 낙하

    console_log_interval: int = 600

    # ---- s2r 관측 (09.14) -----------------------------------------------------------
    # 상시(ADR 무관) 관절·FK 노이즈. 실측 근거는 grasp_s2r cfg `obs_noise_*` 주석:
    #   qpos 실측 ~0.001 rad(엔코더 LSB), qvel 운동 구간 0.05, body(FK) 0.005 m.
    obs_noise_qpos: float = 0.002
    obs_noise_qvel: float = 0.05
    obs_noise_body: float = 0.005
    # 컵 pose 지각(FP++) — 지연 큐 + 코히런트 노이즈. ADR 축(초기 → 종점):
    perception_delay_max_steps: int = 3       # 종점: 0~3 정책스텝(0~50 ms) 균등
    perception_delay_base_steps: int = 0
    obs_noise_object_xyz: float = 0.0         # 초기 [m]
    adr_obs_noise_object_xyz_max: float = 0.015
    obs_noise_object_rot_deg: float = 0.0
    adr_obs_noise_object_rot_max_deg: float = 3.0

    # ---- 물리 DR / ADR (09.14) --------------------------------------------------------
    enable_events: bool = True
    events: PourFabricEventCfg = PourFabricEventCfg()
    cup_friction_range: tuple = (0.7, 1.2)    # 재질은 처음부터 고정 범위(런타임 확장 불가)
    enable_adr: bool = True
    adr_num_increments: int = 30
    adr_increment_interval: int = 3000        # 정책 스텝
    adr_trigger_threshold: float = 0.30       # 순간 성공률(success_now 평균)
    # 들린 컵 외란(질량정규화, grasp_s2r W1 값). ADR 로 0 → 종점 스케일.
    wrench_force_scale_max: float = 5.0       # N/kg
    wrench_torque_scale_max: float = 0.5      # N·m/kg
    wrench_prob_range: tuple = (0.001, 0.1)
    wrench_lift_min_m: float = 0.03           # 이만큼 들렸을 때만 외란

    # ---- 파생 자산 cfg -------------------------------------------------------------
    robot_cfg: ArticulationCfg = None
    source_cup_cfg: RigidObjectCfg = None
    receiver_cup_cfg: RigidObjectCfg = None
    beads_cfg = None
    table_spawn: sim_utils.UsdFileCfg = TABLE_SPAWN
    source_contact_filter: tuple = ()
    receiver_contact_filter: tuple = ()

    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0
    # 파생 폭(로깅·검증용) — resolve_cfg 가 채운다
    num_actions_per_side: int = 0

    def __post_init__(self) -> None:
        resolve_cfg(self)


def _hand_action_width(profile) -> int:
    """per_finger: 구동관절 수 = 액션 슬롯 수 (`hand_finger_channels` 가 1:1 을 준다)."""
    slots = sorted(int(s) for m in profile.hand_finger_channels.values() for s in m.values())
    if slots != list(range(profile.num_hand_joints)):
        raise ValueError(f"[{profile.name}] hand_finger_channels 슬롯 {slots} 이 구동 {profile.num_hand_joints} 과 1:1 이 아니다")
    return profile.num_hand_joints


def _validate_mimic_fields(cfg: "PourFabricMimicEnvCfg", pair) -> None:
    """언더액추 손에서 **조용히 무효가 되는 조합**을 cfg 단계에서 죽인다(grasp_fj_rh 규약)."""
    errs = []
    if float(cfg.oppose_grip_delta_rad) != 0.0:
        errs.append("oppose_grip_delta_rad 는 tesollo 엄지 대향(_2) 전용 — RH56F1 은 0.0")
    if str(cfg.synergy_freeze_scope) != "finger":
        errs.append("synergy_freeze_scope 는 finger 만(손가락당 구동관절 1개)")
    if float(cfg.mimic_dep_limit_margin_rad) < 0.0:
        errs.append("mimic_dep_limit_margin_rad 는 ≥ 0")
    if float(cfg.mimic_runaway_err_rad) <= 0.0:
        errs.append("mimic_runaway_err_rad 는 > 0 (0 이면 모든 env 가 매 스텝 종료)")
    cols = (int(cfg.ctx_palm_normal_col), int(cfg.ctx_palm_second_col))
    if not all(0 <= c <= 2 for c in cols) or cols[0] == cols[1]:
        errs.append(f"ctx_palm_normal_col/ctx_palm_second_col 는 0~2 의 서로 다른 열: {cols}")
    for p in (pair.source, pair.receiver):
        for name, spec in p.actuator_specs.items():
            if name.endswith("_hand_mimic") and (spec.get("stiffness"), spec.get("damping")) != (0.0, 0.0):
                errs.append(f"{p.name}:{name} 게인이 0/0 이 아니다 — PhysX mimic 제약과 싸운다(09.07 발산)")
    if errs:
        raise RuntimeError("[pour_fabric_mimic cfg] " + " · ".join(errs))


def resolve_cfg(cfg: "PourFabricMimicEnvCfg") -> None:
    """스위치 → 자산 cfg · 차원 파생. **멱등** — hydra 오버라이드 후 env 가 재호출한다."""
    pair = _bm.get_pair(cfg.pair_name)

    cfg.robot_cfg = build_robot_cfg(pair, self_collisions=cfg.enable_self_collisions,
                                    gravity=cfg.enable_gravity)
    s = float(cfg.cup_scale)
    for k, v in CUP_GEOM_UNIT.items():
        setattr(cfg, k, v * s)
    cfg.source_cup_cfg = build_cup_cfg(SOURCE_CUP_PRIM, s)
    cfg.receiver_cup_cfg = build_cup_cfg(RECEIVER_CUP_PRIM, s)
    cfg.beads_cfg = make_beads_cfg(_ASSETS_DIR, n=int(cfg.bead_count))
    cfg.source_contact_filter = (SOURCE_CUP_BODY,)
    cfg.receiver_contact_filter = (RECEIVER_CUP_BODY,)

    # 물리 DR: 컵 마찰은 고정 범위(런타임 확장 불가) — term 생성 전인 cfg 단계에서 적용.
    if not bool(cfg.enable_events):
        cfg.events = None
    elif cfg.events is not None:
        lo, hi = float(cfg.cup_friction_range[0]), float(cfg.cup_friction_range[1])
        for term in (cfg.events.source_cup_material, cfg.events.receiver_cup_material):
            term.params["static_friction_range"] = (lo, hi)
            term.params["dynamic_friction_range"] = (lo, hi)

    _validate_mimic_fields(cfg, pair)
    for p in (pair.source, pair.receiver):
        if _hand_action_width(p) != _hand_action_width(pair.source):
            raise ValueError("양팔 손 액션 폭이 다르다 — 같은 손 자산이어야 한다")
    hand_w = _hand_action_width(pair.source)
    cfg.num_actions_per_side = 6 + hand_w
    cfg.action_space = 2 * cfg.num_actions_per_side

    # policy obs (팔마다): arm q/qd(2·A) + hand q(H) + palm_pos 3 + palm_axes 6
    #   + tips_rel_palm 3F + palm_to_cup 3 + cup_to_tips 3F + joint_err H + cup_up 3
    #   ★09.14 hand_qd(H) 는 actor 에서 뺐다(실기 드라이버 velocity ≠ 관절속도) — critic 에만.
    # + 공통: src_cup→rcv_cup 3 + 주둥이→개구 3 + prev_action Dact
    per = 0
    hand_qd_total = 0
    for p in (pair.source, pair.receiver):
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3 + f  # tactile
        hand_qd_total += h
    cfg.observation_space = per + 3 + 3 + cfg.action_space
    # critic = policy(clean) + hand_qd(2H) + 비드 분율 4 + 비드 무게중심(rcv 프레임) 3
    #        + 두 컵 lin/ang vel 12 + 진행도 1 + 손가락 접촉력 2F
    f_src = len(pair.source.finger_sensor_bodies)
    f_rcv = len(pair.receiver.finger_sensor_bodies)
    cfg.state_space = cfg.observation_space + hand_qd_total + 4 + 3 + 12 + 1 + f_src + f_rcv


@configclass
class PourFabricMimicEnvCfg_PLAY(PourFabricMimicEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
