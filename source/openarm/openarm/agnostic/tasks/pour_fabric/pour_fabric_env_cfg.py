"""pour_fabric 환경 설정.

**환경 세팅 동일성 계약**: 물리 블록(sim dt·solver·max_depenetration·중력·자기충돌·
Fabrics 파라미터·지면·픽스처)은 grasp_lift_fabric 과 **값 동일**해야 한다.
grasp 트랙은 학습 중이라 교차 임포트하지 않고 값을 복사하되 출처를 주석으로 남긴다.
다른 것은 action / reward / 태스크 판정뿐이다.

기본 물리: enable_gravity=True · enable_self_collisions=True —
서버 베이스라인 fab_test10(open-bis_r_grasp_lift_fab)의 실행 플래그와 일치.
(grasp cfg 의 **기본값**은 False/False 이고 CLI 로 켠다 — 여기는 pour 가 그 조건을
 물려받는 것이 정합이므로 기본값 자체를 True 로 둔다.)
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
_HDGP_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))

# =============================================================================
# 컵 자산 — pour 전용. ★grasp 뱅크의 cup_big_rl(convex)이 아니라 **SDF** 콜라이더다.
#   convex hull 은 컵 내부 공동을 메워 비드가 담기지 않는다 — pour_v1 이 그래서
#   cup_big_sdf.usd 를 쓴다(양쪽 컵 동일). 원점 오프셋은 같은 메시라 cup_big 과 동일.
# =============================================================================
POUR_CUP_USD = os.path.join(_ASSETS_DIR, "cup", "cup_big_sdf.usd")
POUR_CUP_ORIGIN_OFFSET_Z = 0.0773     # 실측(pxr bbox): 바닥 -0.0773
POUR_CUP_MASS = 0.134                 # object_bank.BASE_OBJECT_MASS 와 동일

# ★cup_big_sdf 는 RigidBodyAPI 가 루트 prim 에 있다(pour_v1 이 루트 필터로
#   force_matrix_w 를 정상 수신한 실적). baseLink 류 자산과 다르니 혼동 금지.
SOURCE_CUP_PRIM = "/World/envs/env_.*/SourceCup"
RECEIVER_CUP_PRIM = "/World/envs/env_.*/ReceiverCup"


def build_cup_cfg(prim_path: str) -> RigidObjectCfg:
    """pour 컵 하나. 물리 값은 grasp_lift_fabric build_object_cfg 와 동일(출처 주석)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=POUR_CUP_USD,
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=POUR_CUP_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,   # = grasp_lift_fabric
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,       # = grasp_lift_fabric (7218N 스파이크 근거)
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.20, 0.30)),
    )


def build_robot_cfg(pair: _bm.BimanualPair, self_collisions: bool,
                    gravity: bool) -> ArticulationCfg:
    """BimanualPair → ArticulationCfg. 물리 값은 grasp_lift_fabric 과 동일."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS_DIR, pair.asset.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # ★cfg 필드(enable_gravity)로만 바꿀 것 — resolve_cfg 가 재생성한다.
                disable_gravity=not gravity,
                max_depenetration_velocity=1.0,       # = grasp_lift_fabric
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=16,   # = grasp_lift_fabric
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


# 환경 픽스처 — grasp_lift_fabric 과 동일 (env_v1.usda, top_plate 상면 z=0.205 · 09.05)
ENV_FIXTURE_SPAWN = sim_utils.UsdFileCfg(
    usd_path=os.path.join(_ASSETS_DIR, "simulation_setting/env_v1/usd/env_v1.usda"),
)
ENV_FIXTURE_PRIM = "/World/envs/env_.*/EnvFixture"


# =============================================================================
@configclass
class PourFabricEnvCfg(DirectRLEnvCfg):
    """차원은 resolve_cfg 가 pair 로 확정한다."""

    # ---- 로봇 (양팔 쌍) ---------------------------------------------------------
    pair_name: str = _bm.DEFAULT_PAIR

    # ---- warm start -------------------------------------------------------------
    # 빈 문자열 = data/pour_fab_warm_<pair>_{src,rcv}.hdf5 파생.
    warm_bank_source_path: str = ""
    warm_bank_receiver_path: str = ""
    # ★False 는 probe 전용이다: 홈+테이블 위 컵(파지 없음)으로 부팅한다.
    #   학습을 이 상태로 돌리면 grasp 부터 다시 배워야 한다 — env 가 경고를 찍는다.
    require_warm_bank: bool = True
    warm_bank_min_states: int = 64

    # ---- 좌팔(receiver) 제어 -----------------------------------------------------
    # frozen: action[6:9] 무시, warm 자세 유지 (액션 폭은 불변 → 체크포인트 인계 가능)
    receiver_control_mode: str = "frozen"     # "frozen" | "learned"

    # ---- 시뮬레이션 (물리 = grasp_lift_fabric, 용량 = pour_v1) ---------------------
    episode_length_s: float = 20.0            # 1200 스텝 @60Hz (pour_v1)
    decimation: int = 2                       # = grasp_lift_fabric
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,                       # = grasp_lift_fabric
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            # ★버퍼만 pour_v1 값으로 확대(비드 20개 × N env 접촉 폭증) — 물리 거동 아님.
            # pour_v1 검증값(128 env + 비드 20 에서 overflow 크래시 이력 끝에 확정).
            # ★로컬에서 다른 학습이 VRAM 을 점유 중이면 이 1GB stack 할당이 실패한다 —
            #   그건 버퍼 결함이 아니라 자원 경합이다(08.22 실측: 22GB 점유 중 부팅 실패).
            gpu_found_lost_aggregate_pairs_capacity=64 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 1024,
            gpu_max_rigid_patch_count=2 ** 24,
            gpu_max_rigid_contact_count=2 ** 24,
            gpu_collision_stack_size=2 ** 30,
            gpu_max_num_partitions=64,
            friction_correlation_distance=0.00625,
        ),
    )
    # ★2048 이 아니라 128 — 비드 20개/env 가 접촉·메모리를 지배한다(pour_v1 과 동일).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.5)

    # ---- Fabrics (= grasp_lift_fabric) -------------------------------------------
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 20.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # 손 attractor 게인 — hand_mode="direct" 용. None = fabric params 기본(50).
    # grasp_lift_fabric 0aadafd 가 400 채택(추종 이동량 부족 26%→8%). 값 동일 유지.
    hand_attractor_gain: float | None = 400.0

    # ---- 액션 (전부 절대값 + slew — grasp 규약. 앵커만 per-env warm pose) -----------
    # [0:6] source palm 6D · [6:9] receiver palm xyz
    palm_slew_pos: float = 0.004              # = grasp_lift_fabric (240mm/s)
    palm_slew_rot_deg: float = 2.0
    symmetric_action_scale: bool = True
    # source palm 자세 박스: 중심 = (sign·90, 0, sign·90)° + 아래 비대칭 오프셋.
    # ★순서는 env 배선대로 **(ex=roll, ey=pitch, ez=yaw)** — euler_xyz 순.
    #   (구 값은 (ez,ey,ex) 로 착각해 -150 을 yaw 에 놨었다. yaw 는 세계 z 회전이라
    #    컵 tilt 기여 0°인 데다 -150° 추종도 불가(오차 43.6°/108mm) — P3 실측.)
    # P3 probe_pour_tilt_reachability 08.22 실측(roll 에 -150, 홈 앵커·중력·self-coll ON):
    #   roll 스윕 tilt 달성 60/90/110→오차 0.4/0.5/1.5° · 130→123.5° · 150→137.9°(단조)
    #   깊 tilt+z상승 0.3 조합 115° 달성 · yaw ±45/pitch ±45 정밀 추종.
    #   → tilt_target 110° 는 오차 1.5° 로 여유, 최대 달성 137.9° ≥ 135°.
    pose_offset_lo_deg: tuple = (-150.0, -45.0, -45.0)   # (ex, ey, ez)
    pose_offset_hi_deg: tuple = (45.0, 45.0, 45.0)
    # receiver 자세는 warm 측정값 고정(직립 유지) — 액션 없음.

    # ---- 인계 안정화 (pour_v1 실측 근거 이식) --------------------------------------
    # hold: 팔 목표 = 측정 pose 동결 + 종료 억제(비드 정착 대기).
    hold_steps: int = 120
    # ramp: hold 해제 직후 anchor→정책 목표 선형 보간. pour_v1 의 106.8mm 계단 킥
    # (사망 51/128) 근거. slew 가 1차 방어지만 회전 점프는 ramp 가 추가로 막는다.
    handover_ramp_steps: int = 30
    # pour_v1 의 z_boost(hold 중 목표 조작)는 계단 킥의 공범이라 이식하지 않는다(기본 0).

    # ---- 비드 / 컵 판정 기하 (pour_v1 실측 .usd 기준) ------------------------------
    bead_count: int = DEFAULT_BEAD_COUNT      # ★수집·소비 동일 필수(bead_assets 단일 출처)
    cup_inner_radius: float = 0.041
    cup_inside_z_min: float = -0.070          # bottom(-0.077) + bead 반경 여유
    cup_inside_z_max: float = 0.100           # 림
    cup_mouth_z: float = 0.100

    # ---- 성공 판정 (pour_v1 그대로) -----------------------------------------------
    success_fill_ratio: float = 0.50          # ADR 켜면 0.20 → 0.50
    success_spill_max: float = 0.40
    success_xy_thresh: float = 0.20           # 두 컵 중심 xy 거리
    success_hold_steps: int = 10              # 로깅용: 이만큼 유지해야 episode_success

    # ---- 낙하 판정 (pour_v1 left_cup_drop 기하 기준 — 양쪽에 적용) ------------------
    # ★임계는 물리 검증된 적 없다(pour_v1 CLAUDE.md) — P2 probe 에서 재실측 대상.
    drop_dist_m: float = 0.06                 # palm↔컵 거리가 기준 대비 이만큼 벌어짐
    drop_z_m: float = 0.08                    # 컵이 기준 대비 이만큼 하강

    # ---- 보상 (rewards.py 9항) ----------------------------------------------------
    hold_source_weight: float = 2.0
    hold_receiver_weight: float = 1.0
    hold_envelope_credit: float = 0.6
    hold_grip_credit: float = 0.4
    aim_weight: float = 1.5
    aim_sharpness: float = 4.0
    aim_height_offset: float = 0.05           # target 개구 위 조준점 높이
    tilt_weight: float = 2.0
    tilt_target_deg: float = 110.0
    tilt_prox_std: float = 0.10
    pour_capture_weight: float = 25.0
    pour_release_weight: float = 0.0          # ★기본 0 — 바닥 붓기 보상 위험(주석 참조)
    success_weight: float = 10.0
    spill_weight: float = -2.0
    spill_step_cap: float = 0.5
    drop_penalty_weight: float = -5.0
    action_rate_weight: float = -0.3
    # ★임계 3종 분리 — grasp_lift_fabric 정렬(08.23). 하나로 쓰면 스침을 막으려
    #   올린 값이 참여 판정까지 올려 약한 접촉을 누락시킨다.
    contact_force_threshold: float = 1.0      # N — 대향 게이트·감쌈 마디 판정
    participation_force_threshold: float = 0.1   # N — grip_frac 참여 판정
    envelope_force_threshold: float = 0.5     # N — 엄격 감쌈(전 마디 동시), 진단 전용

    # ---- 태스크 -------------------------------------------------------------------
    console_log_interval: int = 600
    runaway_joint_vel: float = 20.0           # = grasp_lift_fabric

    # ---- ADR (축 하나: success fill ratio 0.20 → 0.50) ----------------------------
    enable_adr: bool = False
    adr_fill_initial: float = 0.20
    adr_fill_final: float = 0.50
    adr_num_increments: int = 30
    adr_increment_interval: int = 3000
    adr_trigger_threshold: float = 0.3

    # ---- 물리 스위치 (기본 ON — fab_test10 베이스라인 정합) --------------------------
    enable_gravity: bool = True
    gravity_compensation: float = 1.0
    enable_self_collisions: bool = True

    # ---- 씬 픽스처 (= grasp_lift_fabric) ------------------------------------------
    env_fixture_spawn: sim_utils.UsdFileCfg = ENV_FIXTURE_SPAWN
    ground_plane_z: float = -0.10

    # ---- 접촉 필터 (resolve_cfg 파생 — 컵 루트 prim, cup_big_sdf 는 루트가 rigid body)
    source_contact_filter: tuple = ()
    receiver_contact_filter: tuple = ()

    # ---- 파생 자산 cfg -------------------------------------------------------------
    robot_cfg: ArticulationCfg = None
    source_cup_cfg: RigidObjectCfg = None
    receiver_cup_cfg: RigidObjectCfg = None
    beads_cfg = None

    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    def __post_init__(self) -> None:
        resolve_cfg(self)


# 액션 폭 상수: source palm 6 + receiver palm xyz 3.
# ★receiver_control_mode 와 무관하게 **불변**이다 — frozen 은 [6:9] 를 무시할 뿐.
#   pour_v1 의 "frozen→learned 체크포인트 인계" 규약 계승.
NUM_ACTIONS = 9


def resolve_cfg(cfg: "PourFabricEnvCfg") -> None:
    """스위치 → 자산 cfg · 차원 파생. **멱등** — hydra 오버라이드 후 env 가 재호출한다.

    (근거는 grasp_lift_fabric.resolve_cfg 와 동일: hydra 는 필드만 덮어쓰고
     __post_init__ 을 다시 돌리지 않는다.)
    """
    pair = _bm.get_pair(cfg.pair_name)

    if cfg.receiver_control_mode not in ("frozen", "learned"):
        raise ValueError(
            f"receiver_control_mode='{cfg.receiver_control_mode}' — frozen|learned 만.")

    cfg.robot_cfg = build_robot_cfg(pair, self_collisions=cfg.enable_self_collisions,
                                    gravity=cfg.enable_gravity)
    cfg.source_cup_cfg = build_cup_cfg(SOURCE_CUP_PRIM)
    cfg.receiver_cup_cfg = build_cup_cfg(RECEIVER_CUP_PRIM)
    cfg.beads_cfg = make_beads_cfg(_ASSETS_DIR, n=int(cfg.bead_count))
    # cup_big_sdf: RigidBodyAPI = 루트 prim → 필터도 루트.
    cfg.source_contact_filter = (SOURCE_CUP_PRIM,)
    cfg.receiver_contact_filter = (RECEIVER_CUP_PRIM,)

    if not cfg.warm_bank_source_path:
        cfg.warm_bank_source_path = os.path.join(
            _HDGP_ROOT, "data", f"pour_fab_warm_{pair.name}_src.hdf5")
    if not cfg.warm_bank_receiver_path:
        cfg.warm_bank_receiver_path = os.path.join(
            _HDGP_ROOT, "data", f"pour_fab_warm_{pair.name}_rcv.hdf5")

    j = (pair.source.num_arm_joints + pair.source.num_hand_joints
         + pair.receiver.num_arm_joints + pair.receiver.num_hand_joints)
    f = len(pair.source.fingers) + len(pair.receiver.fingers)
    cfg.action_space = NUM_ACTIONS
    # joint pos/vel/effort(3j) + 접촉력(f)
    # + source컵 pose(src palm 프레임, 7) + receiver컵 pose(rcv palm 프레임, 7)
    # + 주둥이→개구 delta(3) + source컵 up-axis(3)
    # + prev_action(A) + palm 지령 상태(A — slew 지령은 액션의 저역통과라 상태다.
    #   grasp_lift_fabric 의 +6 규약을 액션 폭으로 일반화)
    cfg.observation_space = 3 * j + f + 7 + 7 + 3 + 3 + 2 * cfg.action_space
    # critic = policy + 비드 분율 4(in_src/in_tgt/spill/crossed)
    #        + 비드 무게중심(receiver 프레임, 3) + 두 컵 lin/ang vel(12)
    # ★비드 ground truth 는 **critic 전용** — 실기에 비드 추적이 없다(actor 금지).
    cfg.state_space = cfg.observation_space + 4 + 3 + 12


@configclass
class PourFabricEnvCfg_PLAY(PourFabricEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()
