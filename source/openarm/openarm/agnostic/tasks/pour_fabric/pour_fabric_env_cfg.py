"""pour_fabric 환경 설정 — 양팔 **잡기→들기→붓기** (09.13 재작성).

★09.13 재작성 이유. 구판은 warm 뱅크(이미 잡은 상태)에서 시작하고 손을 동결한 채 붓기만
  배웠다. 이번 트랙은 text2reward 방식의 **보상 자동생성**이 목적이라 과제 문장("양팔로
  컵을 각각 잡고 비드 있는 컵을 없는 컵으로 옮긴다")을 통째로 정책이 배워야 한다 —
  테이블 위 컵 두 개에서 시작하고 손 20관절도 정책(시너지)이 제어한다.
  보상은 이 파일에 **없다**: `reward_code_path` 의 생성 코드가 `RewardContext` 를 읽어
  계산한다(`modules/t2r`). 비어 있으면 영 보상(부팅/무작위 롤아웃용).

제어 스택 = grasp_s2r 현행(팔 Fabrics ×2 · 손 관절공간 시너지 + 접촉 동결) 을 양팔로.
물리 블록 = 구 pour_fabric(비드 20개×N env 접촉 버퍼) 그대로.
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
from openarm.common.bead_assets import DEFAULT_BEAD_COUNT, make_beads_cfg

from . import bimanual as _bm

_ASSETS_DIR = _ob.ASSETS_DIR

# =============================================================================
# 컵 자산 — pour 전용 **SDF** 콜라이더(convex hull 은 컵 속을 메워 비드가 안 담긴다).
# 원점 오프셋·내부 기하는 pour_v1 실측(.usd bbox) — bead_flags 판정과 같은 값.
# =============================================================================
POUR_CUP_USD = os.path.join(_ASSETS_DIR, "cup", "cup_big_sdf.usd")
POUR_CUP_ORIGIN_OFFSET_Z = 0.0773     # 바닥 −0.0773
POUR_CUP_MASS = 0.134                 # = object_bank.BASE_OBJECT_MASS

SOURCE_CUP_PRIM = "/World/envs/env_.*/SourceCup"
RECEIVER_CUP_PRIM = "/World/envs/env_.*/ReceiverCup"
TABLE_PRIM = "/World/envs/env_.*/Table"


def build_cup_cfg(prim_path: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=POUR_CUP_USD,
            activate_contact_sensors=True,
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
@configclass
class PourFabricEnvCfg(DirectRLEnvCfg):
    """차원은 resolve_cfg 가 pair 로 확정한다."""

    pair_name: str = _bm.DEFAULT_PAIR

    # ---- 보상 (text2reward 생성 코드) ----------------------------------------------
    # 빈 문자열 = 영 보상. 학습 런은 반드시 생성·검증·audit 을 거친 파일을 가리켜야 한다.
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
    # ★비드가 접촉·메모리를 지배해 128 (pour_v1 과 동일). 서버 98GB 에서 상향 실험 가능.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True)

    # ---- 물리 스위치 --------------------------------------------------------------
    enable_gravity: bool = True
    gravity_compensation: float = 1.0
    enable_self_collisions: bool = True
    surface_friction: float = 1.0             # 테이블 재질(컵-테이블). grasp_s2r 규약으로 바인딩.
    ground_plane_z: float = -0.10

    # ---- 작업면·컵 기하 -------------------------------------------------------------
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
    src_palm_delta_lo: tuple = (-0.15, -0.10, -0.12, -45.0, -45.0, -150.0)
    src_palm_delta_hi: tuple = (0.15, 0.40, 0.25, 45.0, 45.0, 45.0)
    rcv_palm_delta_lo: tuple = (-0.15, -0.40, -0.12, -30.0, -30.0, -30.0)
    rcv_palm_delta_hi: tuple = (0.15, 0.10, 0.25, 30.0, 30.0, 30.0)
    # 위치는 프로필 palm 박스로 추가 clamp(회전은 델타 박스만).

    # ---- 손: 관절공간 시너지 (= grasp_s2r 현행 coupled3) --------------------------------
    synergy_close_speed: float = 0.005        # 폐쇄도 변화율 상한 / 정책 스텝
    synergy_contact_freeze: bool = True       # 닿은 마디의 관절만 정지 → 감쌈
    synergy_freeze_scope: str = "joint"       # "joint" | "finger"
    couple_four_fingers: bool = True          # 엄지 독립, 나머지 4지 채널별 평균
    finger_residual_scale: float = 0.0
    # 대향 관절(엄지 ch1) grip = open + delta — grasp_s2r D3 기본. **소스 팔 부호 기준**,
    # 리시버(좌)는 미러 부호(thumb_2 축 Z → −1)를 env 가 적용한다.
    oppose_grip_delta_rad: float = -0.6
    hand_velocity_ff_scale: float = 1.0
    joint_pos_err_max: float = 1.2            # obs 정규화 [rad]
    # 닫기 게이트: palm 이 자기 컵에 이 반경 안으로 와야 오므릴 수 있다(램프). 파지 성립 후 해제.
    # ★09.13 부팅 실측: 시작 자세에서 palm↔컵 중심이 약 0.16 m(손끝이 3 cm 앞) — 반경은
    #   그보다 커야 시작 자세에서 닫을 수 있다. 0.22/램프 0.3 → 0.154 m 안쪽은 게이트 1.0.
    close_gate_enabled: bool = True
    close_gate_radius: float = 0.22
    close_gate_ramp: float = 0.3

    # ---- 접촉 --------------------------------------------------------------------
    contact_force_threshold: float = 1.0      # N — 파지(대향) 게이트·동결 판정
    contact_obs_clip: float = 20.0

    # ---- 성공 판정 (pour_v1 계승) — 보상과 분리된 **기준 지표** ----------------------
    success_fill_ratio: float = 0.50
    success_spill_max: float = 0.40
    success_xy_thresh: float = 0.20           # 두 컵 중심 xy 거리
    success_hold_steps: int = 10

    # ---- 종료 --------------------------------------------------------------------
    runaway_joint_vel: float = 20.0
    drop_below_table_m: float = 0.03          # 컵 원점이 (테이블 상면 + 원점오프셋 − 이 값) 아래면 낙하

    console_log_interval: int = 600

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
    """손가락 수 × 채널 수 (grasp_s2r `_derive_spaces` coupled3 와 동일 규약)."""
    n_ch = len(set(profile.hand_channel_of_joint.values()))
    return n_ch * len(profile.finger_sensor_bodies)


def resolve_cfg(cfg: "PourFabricEnvCfg") -> None:
    """스위치 → 자산 cfg · 차원 파생. **멱등** — hydra 오버라이드 후 env 가 재호출한다."""
    pair = _bm.get_pair(cfg.pair_name)

    cfg.robot_cfg = build_robot_cfg(pair, self_collisions=cfg.enable_self_collisions,
                                    gravity=cfg.enable_gravity)
    cfg.source_cup_cfg = build_cup_cfg(SOURCE_CUP_PRIM)
    cfg.receiver_cup_cfg = build_cup_cfg(RECEIVER_CUP_PRIM)
    cfg.beads_cfg = make_beads_cfg(_ASSETS_DIR, n=int(cfg.bead_count))
    cfg.source_contact_filter = (SOURCE_CUP_PRIM,)
    cfg.receiver_contact_filter = (RECEIVER_CUP_PRIM,)

    for p in (pair.source, pair.receiver):
        if _hand_action_width(p) != _hand_action_width(pair.source):
            raise ValueError("양팔 손 액션 폭이 다르다 — 같은 손 자산이어야 한다")
    hand_w = _hand_action_width(pair.source)
    cfg.num_actions_per_side = 6 + hand_w
    cfg.action_space = 2 * cfg.num_actions_per_side

    # policy obs (팔마다): arm q/qd(2·A) + hand q/qd(2·H) + palm_pos 3 + palm_axes 6
    #   + tips_rel_palm 3F + palm_to_cup 3 + cup_to_tips 3F + joint_err H + cup_up 3
    # + 공통: src_cup→rcv_cup 3 + 주둥이→개구 3 + prev_action Dact
    per = 0
    for p in (pair.source, pair.receiver):
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + 2 * h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3
    cfg.observation_space = per + 3 + 3 + cfg.action_space
    # critic = policy + 비드 분율 4 + 비드 무게중심(rcv 프레임) 3 + 두 컵 lin/ang vel 12
    #        + 진행도 1 + 손가락 접촉력 2F
    f_src = len(pair.source.finger_sensor_bodies)
    f_rcv = len(pair.receiver.finger_sensor_bodies)
    cfg.state_space = cfg.observation_space + 4 + 3 + 12 + 1 + f_src + f_rcv


@configclass
class PourFabricEnvCfg_PLAY(PourFabricEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
