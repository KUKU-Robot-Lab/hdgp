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
                # ★★09.01 확정: **켜지 않는다.** 실기는 중력보상을 켠 채로 운용하므로
                #   PD 가 보는 것은 중력이 상쇄된 잔차뿐이고, `disable_gravity=True` 가
                #   바로 그 상태다. 여기서 중력을 켜면 **중력을 두 번 세는 것**이 된다.
                #   실기 실측(08.31 우팔): 무보상 처짐 12.76° → 중력보상 후 **2.05°**.
                #
                #   ★09.01 sim probe 실측 (`probe_s2r_gravity_droop.py`, 손목 최대
                #     지령↔실제 오차 [deg], 4 env · 300 스텝 정착):
                #         게인      중력 OFF   중력 ON   중력 기여분
                #         KUKA        1.889     4.791     +2.90
                #         r2s 정합    1.162     5.668     **+4.51**
                #     ⚠OFF 의 1~2° 는 처짐이 아니라 **fabric 정상상태 추종 오차**다
                #       (무부하). 중력 성분은 두 열의 **차분**으로 읽는다.
                #     ⚠실기 12.76°/2.05° 와 직접 비교하면 안 된다 — 저쪽은 정적 유지
                #       측정이고 이쪽은 제어 루프가 능동적으로 버티는 중의 오차다.
                #
                #   판정: 중력을 켜면 실기(보상 ON, 잔차 2.05°)에 **없는** 2.9~4.5° 오차가
                #   더해진다. 정합 게인에서 그 폭이 1.6배로 커지는 것도 확인됐다 —
                #   그 게인 자체가 **중력보상을 전제로 동정된 값**이기 때문이다
                #   (r2s collect 가 `gravity_comp_node.py` 를 필수로 요구하는 이유).
                #   남는 실제 격차는 보상 잔차 2.05° 뿐이고, 그건 중력이 아니라 관절
                #   바이어스로 넣을 문제다(현재는 안 넣는다 — obs qpos 노이즈 0.01 rad
                #   이 실측의 10배라 그 안에 묻힌다).
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
    """★★09.01 기본값 = D3 세팅 (`s2r_d3_liftonly_fresh_v2`, 20,000 iter 완주).

    이 전까지 기본값은 "구 동작 보존"이었고 실험은 전부 CLI 오버라이드로 돌았다.
    D3 가 FRESH 에서 8종 전수 성공을 실증해 그 조합을 기본으로 올린다 —
    `# ★D3 기본` 주석이 붙은 필드가 승격 대상이다.

    **20,000 iter 실측 (최근 25%)**: `task/success` 0.604 · `species/success_min`
    0.607 · `stay_run` 14.46 스텝 · `gate/lifted` 0.805 · `abnormal` 0.0000 ·
    `force_max_postlatch` 17.6 N. 종별 0.774~0.949 (s085 0.774 최저, shaker 0.949 최고).

    **같은 자리 warm(D1, B1 ep_12000 인계) 대조**: success 0.185 · min 0.170 ·
    stay_run 0.300. LSTM 이 인계받은 "들고 버티기"를 답습해 이송을 못 배웠다
    (메모리 `fresh-vs-warmstart-lstm-rule`). → 이 세팅은 **FRESH 로 돌리는 것이 기본**.

    승격의 핵심 두 값은 `grasp_weight` 12→4 와 `lift_height_ref` 0.10→0.06 이다.
    구 값은 pre-lift 구간 수입(grasp 4.69 + enclosure 4.17 = 8.86/step)이 리프트
    보상(임계에서 7.3)을 넘겨 **안 드는 것이 이득인 주차장**을 만들었고, D2 가 거기서
    2,500 iter 동안 `lifted` 0.000 으로 멈췄다.
    """

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
    enable_self_collisions: bool = False  # ★D3 기본 (09.01 승격)

    # ---- 실기 정합 액추에이터 게인 (r2s 파이프라인 산출) ------------------------------
    # ★★재생 무결성 필드다. 실제 게인 선택은 `robot_profiles.py` 가 **import 시점**에
    #   `HDGP_S2R_REAL_GAINS` 환경변수로 하는데, 환경변수는 `params/env.yaml` dump 에
    #   **안 남는다**. 그래서 이 게인으로 학습한 체크포인트를 나중에 환경변수 없이
    #   재생하면 조용히 KUKA 게인으로 돌아간다 — m1_final 이 죽은 것과 같은 계열
    #   (차원은 맞는데 의미가 다른 실패)이다.
    #   이 필드는 **의도를 dump 에 남기고**, `finalize_after_overrides` 가 실제 프로필과
    #   대조해 어긋나면 부팅에서 죽인다. 조용히 틀리는 대신 크게 실패한다.
    #   기본값은 환경변수를 따르므로 기존 워크플로(`HDGP_S2R_REAL_GAINS=1 python train.py`)가
    #   그대로 동작하고, dump 에는 해석된 bool 이 기록된다.
    #   실측 근거: `sim2real/docs/R2S_FRAMEWORK.md` — 오버슈트 재현 오차 0.429 → 0.084.
    #   ⚠이 게인은 **중력보상 전제**다(위 `disable_gravity=True` 주석 참조).
    use_real_gains: bool = _os.environ.get("HDGP_S2R_REAL_GAINS") == "1"

    # ---- 액션: palm 6D **홈 기준 델타** + 손 시너지 ----------------------------------
    # ★★grasp_v1 규약. `palm = 홈 + scale(a, −delta, +delta)` 이므로 **a=0 이 홈**이다.
    #   grasp_sensor 의 절대 매핑(`a=0` = 박스 중심)은 σ=1.0(저장소 공통 sigma_init)과
    #   곱해지면 매 스텝 작업공간 전역에서 목표를 재추첨해 접근이 랜덤워크가 된다
    #   (08.27 실측: 클램프 전 요청량 0.33~0.36 m/step 상시 포화). 델타 규약은 탐색을
    #   홈 주변 유계 오프셋으로 묶어 이 문제를 구조적으로 없앤다.
    # ★y 만 범위가 큰 이유: 홈 y −0.38 → 컵 y −0.20 을 액션으로 덮어야 한다.
    palm_delta_xyz: tuple[float, float, float] = (0.10, 0.10, 0.10)  # ★D3 기본 (09.01 승격)
    palm_delta_rot_deg: float = 40.0  # ★D3 기본 (09.01 승격)
    # ---- 액션 앵커 (`a=0` 이 뜻하는 palm 자세) --------------------------------------
    # ★08.29 Phase 0 실측: 홈(0.280, -0.380, 0.418)은 과제에서 멀다 — 컵 파지고보다
    #   z **+14 cm**, 이송 목표보다 y **-27 cm**. 그래서 `a=0` 이 "컵에서 도망"을 뜻하고
    #   LSTM 출력이 조금만 풀리면 palm 이 홈으로 튕긴다. 실측 `palm_post_latch_y`
    #   -0.399 로 **홈(-0.380)보다 더 뒤**, 목표(-0.110)와는 0.29 m 반대. 정책은
    #   `action_norm_arm` 2.29~2.40(이론최대 √6=2.449의 93~98%)으로 상시 저항 중이었다.
    # "spawn" = `object_spawn_pos`(리셋 시 정착 스냅샷) + 아래 오프셋.
    #   ★에피소드 내 상수라 되먹임이 없다 — 실시간 물체 위치를 쓰면 컵이 밀릴 때
    #     액션 원점이 같이 움직인다. 스폰 범위가 넓어지거나 다종 컵으로 가도 따라간다.
    palm_anchor_mode: str = "spawn"          # "home" | "spawn"(D3 기본)
    # 유도: 파지 palm(0.296,-0.207,0.322) ↔ 목표 palm(0.296,-0.157,0.402) 의 중점
    #       (0.296,-0.182,0.362) 에서 스폰(0.362,-0.16,0.2773)을 뺀 값.
    palm_anchor_offset_xyz: tuple[float, float, float] = (-0.066, -0.022, 0.085)
    # ★★지령 변화율 상한 — `gripper/left/grasp_sensor`(t59 fabric 배선) 값과 동일하게 맞췄다
    #   (08.27 사용자 지시 "fabric·액션 세팅을 좌팔과 동일하게").
    #     좌팔 `PALM_CMD_RATE_LIMIT` 0.02 m/step = **1.0 m/s** (IK 액션 스케일 위치 성분과 동일)
    #     좌팔 `PALM_ROT_RATE_LIMIT` 0.05 rad/step = **2.9°/step** (동 회전 성분과 동일)
    #   구 값(0.05 m/step = 3.0 m/s · 7.5°)은 좌팔의 2.5~2.6배였고, 사용자 GUI 관찰
    #   "정책 명령이 너무 빠르게 변한다"가 그 차이였다.
    #   ※fabric 쪽(dt 1/60 · decimation 2 · damping 10 · vel_ff 1.0)은 이미 좌팔과 동일하다.
    palm_cmd_rate_limit_m: float = 0.02
    palm_cmd_rate_limit_rot_deg: float = 2.9

    # ---- 손: 관절공간 시너지 ---------------------------------------------------------
    # 액션은 **절대 폐쇄도 목표**[0,1] 이고 아래는 그 목표를 향한 **변화율 상한**이다
    # (속도 명령이 아니다 — 속도로 두면 탐색 노이즈 평균만으로 완전 폐쇄되고 못 되돌린다).
    # ---- 손 레이아웃 (08.29 O 라운드) ------------------------------------------------
    # "coupled3"(기본·현행): 손가락×3채널 + couple_four_fingers 평균.
    # "per_finger": 프로필 `hand_finger_channels` 의 손가락별 슬롯(엄지 2·검/중/약 각 1·
    #   소지 1 = 6). 미지정 관절은 고정. ★액션 차원이 바뀌므로 warmstart 전면 무효(FRESH).
    hand_layout: str = "coupled3"
    # 동결 범위: "joint"(기본·현행) = 자기 링크 닿은 관절만. "finger" = 그 손가락의
    #   (중간∨원위) 접촉 시 **그 손가락 굴곡관절 전부** 정지 — 08.29 영상 진단:
    #   관절별 동결은 언 손끝을 매단 채 근위가 계속 감겨 큰 컵을 밀어냈다.
    synergy_freeze_scope: str = "joint"
    synergy_close_speed: float = 0.005
    # ★★감쌈을 만드는 메커니즘. 원위·팁이 닿은 손가락의 `_3`/`_4` 만 정지시켜 컵 형상에
    #   드리워지게 한다. 끄면 핀치가 된다(grasp_v1 실증: full_envelope 0.176 → 0.035).
    synergy_contact_freeze: bool = True
    # 엄지 독립 · 4지는 채널별 평균으로 묶는다 → "특정 손가락만 안 닫힘"이 액션 공간에서
    # 표현 불가(3지 국소최적 차단).
    couple_four_fingers: bool = True
    # ★★공통+잔차 손 (08.30 W 라운드 진단 처방). 커플링은 4지 지령을 **평균으로 대체**
    #   하는데, 그 사이를 잇는 연속 손잡이가 없어 "닫히지만 둔한 손(coupled)" ↔ "손가락은
    #   독립인데 안 닫히는 손(15ch)" 두 극단뿐이었다. W 실측: `syn_close` coupled 0.320
    #   vs 15ch 0.106/0.022/0.042 — σ=1.0 잡음이 15채널에 독립으로 걸리면 4지가 동시에
    #   오므리는 결맞음이 급감한다(커플링은 평균으로 분산을 1/4 로 줄인다).
    #   지령 = 공통 + scale·(개별 − 공통). 0 = 현행 coupled 항등 · 1 = 15ch 와 동일.
    #   ★액션 차원 불변(21)이라 커리큘럼 도중 warm 이 깨지지 않는다.
    finger_residual_scale: float = 0.0
    # ADR 다섯째 축 — 레벨에 따라 잔차를 연다("쥐는 법 먼저, dexterity 나중").
    # base(=finger_residual_scale) 이하면 축이 꺼진 것으로 본다.
    adr_finger_residual_max: float = 0.0

    # ---- 닫기 게이트: 위치가 맞기 전에는 오므리지 않는다 -----------------------------
    # ★★08.27 사용자 GUI 관찰: "순간적으로 가깝기만 해도 바로 잡기를 시작한다. 안 가까운데
    #   손가락을 오므리고 그러고 다시 가까워지는 중이다. 위치가 맞춰지기 전엔 잡기를
    #   시작하면 안 된다." 실측 뒷받침(s2r_a5 iter13): `cage_dist` 0.293 = 케이지 반경의
    #   2.4배인데 `syn_close` 0.574 까지 닫혀 있었다.
    # ★래치는 이걸 못 막는다 — 래치는 lift/transfer **보상**을 여는 신호일 뿐이고,
    #   닫힘은 정책의 손 액션이 직접 만든다. 그래서 닫힘 자체를 게이트한다.
    # ★판정은 "엄지·palm·검지가 이루는 직사각형 포함"(사용자 원안) 대신 **케이지 반경 기반**
    #   으로 간다: 손 기하에서 부팅 실측되는 `r_cage` 하나만 쓰므로 물체가 바뀌어도
    #   성립하고, 꼭짓점 3개를 쓰는 것보다 손 자세 변화에 강하다.
    # ★경계에서 0/1 로 끊지 않고 램프를 둔다 — gradient 를 남겨야 "자연스럽게 찾아내는"
    #   여지가 생긴다. **푸는 방향은 항상 허용**한다(갇히면 빠져나올 길이 없다).
    # ★★케이지 **중심은 palm 강체 오프셋**(홈 실측)이고 거리는 **3D** 다. 08.27 실측:
    #   중심을 실시간 손끝 평균으로 두면 팔 정지 구간에서 corr(syn_close, cage_dist)
    #   = −0.974 — 손을 오므리는 것만으로 게이트가 열려 아무것도 막지 못했다.
    #   xy 투영이던 시절엔 palm·검지가 컵보다 내려간 자세도 통과했다(z 항 부재).
    close_gate_enabled: bool = True
    close_gate_ramp: float = 0.5      # r_cage 의 이 비율 구간에서 0→1 로 램프

    # ---- 파지 기하 ---------------------------------------------------------------
    # ★대향축·반경 상수는 08.27 에 제거됐다. 접근 항이 이제 **손 자신의 대향 중점**과
    #   컵 사이 거리를 쓰므로(env `cage_dist`) 물체 반경이 필요 없다 —
    #   구 수식은 대향축을 접근방향의 90° 회전으로 잡아 좌/우 부호가 임의였고,
    #   그래서 엄지 목표가 실제 엄지의 반대편에 놓여 엄지가 걸렸다(사용자 GUI 관찰).
    object_grasp_z_offset: float = 0.03      # 물체 원점 ↔ 파지 높이
    # ★★거리 계산의 **z 데드밴드**. 08.27 실측: 거리가 3D 노름이라 z 오차 1cm 이 xy 오차
    #   1cm 과 정확히 같은 벌점이었고, z 전용 항도 허용대역도 없었다. 그 결과 palm 이
    #   파지높이(테이블+107mm)가 아니라 컵 원점 근처로 눌려 내려갔다 —
    #   palm_above_table mean **0.088**(파지중심보다 19mm 아래) · min **0.066**
    #   (컵 원점보다 11mm 아래), 사용자 GUI "아예 테이블을 박히고 간다".
    #   밴드 안에서는 z 오차를 0 으로 본다: d = √(dxy² + relu(|dz| − band)²).
    grasp_z_deadband: float = 0.03
    # ★★테이블을 **fabric 장애물**로 등록한다. 08.27 발견: `WorldMeshesModel` 에 world 를
    #   안 넘겨서 `object_indicator == 0` → 반발 커널이 첫 줄에서 early-out 했다.
    #   **fabric 이 테이블을 아예 모르는 상태**로 계획하고 있었다(형제 tesollo 트랙은
    #   전부 world_filename 을 넘긴다 — agnostic 트랙만 빠져 있었다).
    #   params 의 `body_repulsion.collision_sphere_frames` 에 palm·5지 전 마디(소지
    #   14개 포함)·팔 링크 충돌구가 **이미** 등록돼 있어, 테이블만 넣으면 손 전체가
    #   한꺼번에 보호된다 — params 파일은 건드릴 필요가 없다.
    # ★박스는 palm 도달영역(프로필 palm_box)에서 **파생**시킨다. 상수를 따로 적으면
    #   물리 테이블과 조용히 어긋난다(08.25 "안 적은 물리 파라미터는 조용한 기본값").
    fabric_table_obstacle: bool = True
    fabric_table_margin_xy: float = 0.10     # 도달영역 밖으로 넓힐 여유
    fabric_table_thickness: float = 0.05

    # ---- 접촉 판정 -------------------------------------------------------------------
    contact_force_threshold: float = 1.0     # N — 접촉으로 셀 최소 힘
    contact_force_max: float = 10.0          # N — obs 정규화 포화점
    joint_pos_err_max: float = 1.2           # rad — obs 정규화

    # ---- 진단 계측 (보상·게이트에 쓰이지 않는다 — 로깅 전용) -------------------------
    # ★08.27: `wrap_frac`(중간 AND 원위)이 4,553 기록점 내내 정확히 0.000 인데 영상에서는
    #   감쌈이 성립한다. "안 닿았다"와 "닿았는데 못 읽는다"를 가르기 위한 계측이다.
    diag_contact_threshold_lo: float = 0.1   # N — 스치는 접촉까지 잡는 낮은 임계
    # ★08.29 사용자 제약: 실기 팁 센서 정격 **0~50 N**, 그 위는 측정 불가.
    #   하드웨어는 넘길 수 있다 — URDF effort 7.5 N·m / 원위 모멘트암 25.5 mm
    #   ⇒ 실기 최대 294 N, sim(effort_limit_sim 1.5) 58.8 N. 아래 둘은 **초과율
    #   로깅에만** 쓰인다(보상 경로 없음). 힘 밴드 도입 시 임계의 출발점이 된다.
    force_sensor_max_n: float = 50.0         # N — 팁 센서 정격 상한
    force_band_hi_n: float = 30.0            # N — 밴드 감쇠 시작(그 아래는 정확히 무손실)
    # ★밴드 바닥. **1.0 = 감쇠 없음(현행 동작)**. 켤 때도 0 으로 두면 안 된다 —
    #   08.25 `grip-contact-cliff` 에서 "닿으면 보상이 꺼지니 접촉을 회피"가 실측됐다.
    #   세게 쥐는 것이 손해이되 **놓는 것보다는 낫게** 남기는 값이 0.3 이다.
    force_band_floor: float = 0.5  # ★D3 기본 (09.01 승격)
    # 손 PD 토크 포화 판정: err ≥ effort_limit_sim / stiffness = 1.5 / 5.0.
    # ★`blocked_err_thr_rad` 와 값은 같지만 용도가 다르다 — 그쪽은 동결 게이트,
    #   이쪽은 "가동 관절이 천장에 붙어 있는 비율" 진단이다. 따로 둔다.
    hand_torque_sat_err_rad: float = 0.30
    # 손 PD 가 버틸 수 있는 최대 정적 오차 = effort_limit_sim / stiffness = 1.5 / 5.0.
    # 이보다 크면 토크가 천장에 붙어 있다는 뜻이라 "막혔다"로 센다.
    blocked_err_thr_rad: float = 0.30
    blocked_limit_eps_rad: float = 0.05      # 관절 한계에서 이만큼 떨어져야 "외부에 막힘"
    # ★probe 가 자세를 **눈으로** 확인할 때만 켠다(기본 OFF — 학습 거동 불변).
    #   센서는 `clone_environments` 전에 만들어야 초기화되므로 `_setup_scene` 에서
    #   cfg 플래그로 분기한다. 나중에 붙이면 "TiledCamera could not be initialized" 로 죽는다.
    # ---- 손 실험 노브 (전부 기본값 = 현행 거동. hydra 로 런마다 오버라이드) ----------
    # ★코드를 갈래마다 고치면 재현이 깨진다 — 같은 커밋에서 `env.<필드>=값` 으로 가른다.
    # "contact"(현행: 닿으면 멈춤) | "blocked"(막힐 때까지 만다 — 접촉 센서 불필요).
    #   ★08.27 실측: "닿으면 멈춤"이 감쌈 **직전**에 멈추게 만든다. 엄지가 중간마디로
    #     먼저 닿아 `_3` 가 0.28 에서 얼고, 이후 열림 래칫으로 0.00 까지 풀렸다.
    synergy_hold_mode: str = "contact"
    # blocked 모드에서 이보다 작은 **여는** 지령은 무시한다(열림 래칫 차단). 0 = 끔.
    synergy_release_deadband: float = 0.0
    # 대향 손가락(`contact_group_a`)의 ch1 관절 grip = open + 이 값. 0 = 현행(고정).
    #   ★URDF 실측 `r_hj_thumb_2` 가동범위 −3.142~0.0(180°)로 손에서 가장 큰데
    #     프로필이 open=grip=−1.57 로 적어 **엄지 대향각이 학습 대상이 아니었다**.
    oppose_grip_delta_rad: float = -0.6  # ★D3 기본 (09.01 승격)
    # 과굴곡 손가락의 ch2(굴곡) grip 각도 배율. 빈 이름 = 끔.
    #   ★소지는 `_3`·`_4` 가 다른 손가락과 같은 채널로 묶여 **같은 각도**로 말리는데
    #     길이가 짧아 과도하게 감긴다(사용자 GUI 관찰).
    weak_finger: str = ""
    weak_finger_curl_scale: float = 1.0

    debug_camera: bool = False
    debug_camera_pos: tuple[float, float, float] = (0.72, -0.40, 0.52)
    debug_camera_rot: tuple[float, float, float, float] = (0.42, 0.24, 0.44, 0.75)

    # ---- 래치 (보상 단계 표시 전용 — 팔 지령을 덮지 않는다) --------------------------
    # ★★grasp_v1 의 `torch.where(is_lift, _lift_palm, palm_pose)` z 램프 오버라이드는
    #   **이식하지 않는다**. 래치는 lift/transfer 보상을 여는 신호일 뿐이고, 팔은
    #   처음부터 끝까지 정책이 fabric 을 통해 제어한다.
    lift_start_min_grip_fingers: int = 3
    grasp_ready_hold_steps: int = 8
    # ---- 낙하/전도 재소환 (08.30, 기본 OFF = 현행 종료) ------------------------------
    # ★종료 대신 이번 에피소드의 **원래 스폰점**으로 컵을 되돌린다(앵커·목표 불변).
    #   palm 여유 미달이면 보류(자매 v2 검증 규약 — 폴백 텔레포트 금지).
    respawn_on_fail: bool = True  # ★D3 기본 (09.01 승격)
    respawn_clearance_m: float = 0.12  # ★D3 기본 (09.01 승격)
    # ★재소환 1회당 벌점(양수 = 비용). Q3 실측: 벌점 0 이면 접촉력 300N+ 의 거친
    #   탐색이 공짜가 된다 — 종료(−수백 상당)보다 훨씬 작되 0 이 아니게.
    respawn_penalty: float = 2.0  # ★D3 기본 (09.01 승격)
    # ★★보류 예산 (08.30 Q3 실측 처방). 여유 미달로 보류만 하면 넘어진 컵이 팔 옆에
    #   방치돼 `cup_disp`·`tilt` 벌점이 계속 나오고 approach 가 **순벌점**(−0.35)이 된다
    #   — 에피소드가 안 끝나니 그 상태로 600 스텝을 버틴다(Q3: defer 0.93·palm 접촉 1.4%
    #   vs 정상 81.7%). 연속 보류가 이 예산을 넘으면 **종료로 폴백**한다(0 = 무제한).
    respawn_defer_budget: int = 60  # ★D3 기본 (09.01 승격)
    # ★여유를 palm 원점이 아니라 **손 전체(palm+손끝) 최소거리**로 잰다. False(현행)면
    #   손끝이 스폰점에 있어도 통과해 컵이 손가락 안으로 텔레포트된다(08.30 힘 실측).
    respawn_clearance_uses_tips: bool = True  # ★D3 기본 (09.01 승격)
    # ★재소환 위치 — "origin"(기본) = 원래 스폰점 복귀(그 자리가 곧 손자리라 보류 0.93).
    #   "free" = 자매 v2 규약: 스폰 상자 안에서 **손이 없는 자리**를 리젝션 샘플링하고
    #   스폰 기준·목표를 그 자리로 옮긴다(물체만 새로 리셋하는 것과 동등).
    respawn_mode: str = "free"  # ★D3 기본 (09.01 승격)
    respawn_tries: int = 24
    # ★free 모드 샘플링 반범위 [m]. 0 = 스폰 범위와 동일(기본).
    #   ★★스폰 범위(ADR level 0 에서 0.02 → 4×4cm)는 **손보다 작아** 어떤 후보도
    #   여유를 못 채운다 — 08.30 R2a 실측 defer 0.62 · 최선후보 거리 0.056.
    #   자매 v2 도 같은 한계를 주석에 남겼다. 재소환 상자는 스폰 상자와 분리한다.
    respawn_range: float = 0.09  # ★D3 기본 (09.01 승격)
    # ★무접촉 정체 보상 처방 (08.30 W4). enclosure(10)는 접촉 없이도 스텝당 ~7.2 를
    #   내는 공짜 보상이라 FRESH 에서 "가만히 있기"가 국소최적이 된다(M0·O1 총보상
    #   실측 일치). floor<1 이면 enclosure_term 에 (floor + (1−floor)·graded_contact)
    #   를 곱해 무접촉 상한을 10·floor 로 낮춘다. ★H 라운드가 기각한 **완전 곱셈**
    #   (=floor 0, 접근 구간에서 항이 죽음)과 다르다 — 접근 gradient 의 floor 비율을
    #   보존한다. 기본 1.0 = 현행 항등.
    enclosure_contact_floor: float = 0.3  # ★D3 기본 (09.01 승격)
    # ★★`finger_closure` 의 목표 마디. "tip"(기본·현행) | "wrap".
    #   08.31 8종 실측이 이 노브를 만들게 했다 — 팁 접촉은 0.65~0.85 로 이미 채워졌는데
    #   wrap(중간∧원위)은 **전 종 0.000** 이다. `graded_contact = 0.4·팁 + 0.6·wrap`
    #   에서 정책이 싼 0.4 만 먹고 멈춘 것이고, 팁 기준 소등 항은 바로 그 지점에서
    #   꺼져 경사를 못 준다. "wrap" 은 소등 조건을 중간∧원위로, 거리를 중간마디로
    #   옮겨 **손끝을 댄 뒤부터 감아 안기까지** 경사를 잇는다.
    finger_closure_target: str = "wrap"  # ★D3 기본 (09.01 승격)
    # ★★접촉 품질의 정의. "tipwrap"(기본·현행) | "anylink".
    #   "anylink" = 손가락 **어느 마디든**(tip|mid|dist) 닿았는가 + 손바닥 한 표,
    #   분모 5+1. 08.31 사용자 확정: "어떤 부분이든 5손가락(또는 손바닥까지) 닿고
    #   안정적으로 유지만 하면 된다". 구 정의는 wrap(중간∧원위 동시)이 8종 전수
    #   0.000 이라 정책이 팁 0.4 만 먹고 2~3개 접촉에서 멈췄고, 그 상태로는 컵을
    #   기울이는 과제에서 놓친다.
    contact_quality_mode: str = "anylink"  # ★D3 기본 (09.01 승격)
    # ★래치 판정 방식 (08.29). "count"(기본·현행) = 접촉 손가락 수 ≥ min.
    #   "opposition" = (그룹A) AND (그룹B OR palm) — 실측 성공 파지(엄지+palm)에서
    #   count 가 래치를 영원히 못 여는 문제의 처방. O1/O2/N1 적용, M1 은 대조 보존.
    latch_mode: str = "opposition"  # ★D3 기본 (09.01 승격)

    # ---- 목표(goal) — 수평 이동 포함 -------------------------------------------------
    # goal = 물체 **정착 위치** + offset. 스폰점 기준이면 패딩이 이중으로 실린다.
    # ★목표는 컵 스폰 기준 **0.1 m 이내**(사용자 지시, 08.27). 구 (0.0, 0.20, 0.15)
    #   = 0.25 m 는 transfer 의 exp(−6·d) 를 0.22 로 깎아 리프트 후 갈 이유가 없었다.
    #   0.094 m 면 0.57 — 2.6배. y 는 컵 y −0.16 기준 −0.11 로 **음수 유지**(y≥0 회피).
    #   z 0.08 > lift_success_height 0.04 라 목표 도달 전에 "들렸다"가 먼저 성립한다.
    #   ★나중에 커리큘럼으로 늘려나갈 값이다.
    goal_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.12)  # ★D3 기본 (09.01 승격)
    goal_pos_tolerance: float = 0.025        # 성공 반경
    goal_pos_tolerance_loose: float = 0.05   # 연속성 비교 로깅 전용
    stay_hold_steps: int = 60                # 1초 — stay 항이 만점이 되는 유지 시간
    lift_height_ref: float = 0.06            # lift 항 높이 정규화 기준  # ★D3 기본 (09.01 승격)
    lift_success_height: float = 0.04        # "들렸다" 판정
    success_tilt_max_deg: float = 5.0
    stable_lin_vel: float = 0.04
    stable_ang_vel: float = 0.5

    # ---- 성공 판정 절 (08.28 신설 — 기본값 = 현행 동작) -------------------------------
    # 사용자 확정: 과제 목적은 "컵이 목표에 제대로 놓여 멈춰 있는가" 다. 아래 두 절은
    # 산술로 `at_goal ∧ stable` 에 함축되므로 끌 수 있다(근거는 env 쪽 주석).
    # `success_min_grip_fingers` 는 구 코드의 리터럴 `n_grip >= 4` 를 cfg 로 올린 것 —
    # 그 리터럴은 2지 그리퍼 프로필에서 절대 성립 불가였다.
    success_require_lifted: bool = True
    success_require_holding: bool = False  # ★D3 기본 (09.01 승격)
    success_min_grip_fingers: int = 4

    # ---- 감쌈 지표 (08.28 신설 — 기본값 = 현행 `deep_and`) ----------------------------
    # "deep_and"     : per-finger (중간 AND 원위), 분모 = contact_group_b (엄지 제외)
    # "surface_count": 손바닥 + 대향그룹 + 반대그룹의 **표면 참여**, 마디 조합 무관
    # 가중치는 합으로 정규화되므로 손바닥이 닿지 않는 프로필은 palm 을 0 으로 두면 된다.
    envelope_metric: str = "deep_and"
    envelope_palm_weight: float = 0.3
    envelope_group_a_weight: float = 0.3     # 대향(엄지) 그룹
    envelope_group_b_weight: float = 0.4     # 반대(4지) 그룹

    # ---- 포위도 (08.28 신설 — 기본 0.0 = 항 비활성) -----------------------------------
    # ★인벨롭을 **접촉이 아니라 기하**로 잰다. Hu et al. 2020 의 `r_topology`(가중 10)
    #   에 대응하며, 우리가 갖고 있지 않던 절반이다. 접촉 기반 지표는 팁 파지로
    #   포화하므로(G 라운드 실증) 정의를 고치는 것만으로는 자세가 바뀌지 않는다.
    # ★형상 정보를 쓰지 않는다 — 물체 중심 하나뿐이라 컵 종류를 늘려도 성립한다.
    #   hull 대신 방향 분산을 쓰는 이유는 sim2real 이다(링크 위치 = FK 는 전이되지만
    #   접촉점 개수는 contact discretization·마찰·강성에 민감해 전이되지 않는다).
    enclosure_weight: float = 10.0  # ★D3 기본 (09.01 승격)
    enclosure_palm_weight: float = 0.3
    enclosure_group_a_weight: float = 0.3
    enclosure_group_b_weight: float = 0.4
    # ★08.29 I1 실측 비중: enclosure 67.7%(상한의 77% 실현) vs transfer 2.6%(1.9%).
    #   가중치는 상한일 뿐이고 실제 비중은 **실현율**이 정한다 — 조밀 항(매 스텝 기하)은
    #   즉시 포화하고 희소·조건부 항은 0~11%만 실현한다. 그래서 가중 30짜리 lift 가
    #   가중 10짜리 enclosure 의 절반도 못 낸다. 게다가 래치 후에도 포위도는 팔 위치와
    #   무관하게 계속 지급되어 **리프트 이후 보상 지형이 palm 위치에 평평해진다**.
    #   래치 **전은 불변**(감쌈을 만든 힘 보존), 래치 **후만** 이 비율로 줄인다.
    enclosure_post_latch_scale: float = 1.0
    # ★손가락별 최소참여 혼합비. **0.0 = 그룹 평균만(현행 동작)**.
    #   위 식은 그룹 키포인트를 평균하므로 손가락 하나가 빠져도 값이 거의 안 떨어진다 —
    #   `couple_four_fingers` 를 넣게 만든 3지 국소최적의 원인 진단("mean/count 보상엔
    #   손가락별 최소참여 신호 부재")과 같은 결함이다. **커플링을 풀기 전 선행조건**이다.
    enclosure_participation_lambda: float = 0.0

    # ---- 손등 접촉 배제 (Hu et al. `p_collision` 대응 — 기본 꺼짐) --------------------
    # 켜면 손바닥면이 물체를 향하는 접촉만 인정한다. ★`palmar_axis_local` 이 프로필에
    # 없으면 부팅에서 fail-loud — 기본축을 가정하면 판정이 조용히 뒤집힌다.
    require_palmar_contact: bool = False

    # ---- 이진 케이지 게이트 (DexPoint `r_contact` — 0 = 꺼짐) -------------------------
    # `엄지 접촉 ∧ (대향 손가락 ≥ n)`. 접촉 **개수 자체를 보상하지 않는** 것이 핵심이다 —
    # 개수 보상은 "손끝을 몰아 개수만 채우는" 수법에 취약하다(저장소 실패 이력 2건).
    cage_gate_min_opposing: int = 0

    # ---- 보상 가중치 (grasp_v1 8항 + 이송 2항) ---------------------------------------
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0          # 손바닥 **면** 어긋남(y·z)
    # ★★법선(palm_ee_x) 방향은 더 날카롭게 — "손바닥이 물체에 밀착"이 인벨롭의 전제다.
    #   08.27 구조 실측: 홈에서 케이지 중심이 palm 에서 **106mm 앞**이라
    #   (cage−palm = 82.2, 66.4, 3.4 mm), approach 가 cage_dist→0 을 요구하면
    #   palm 은 컵에서 106mm 떨어져야 한다 — "밀착"과 **양립 불가**였다.
    #   실측 타협점: palm_to_cup 0.126 / cage_dist 0.041 → 사용자 GUI 관찰
    #   "palm_ee → 손가락 → 컵 순서로 온다"가 여기서 나온다.
    approach_sharpness_normal: float = 12.0
    # ★밀착 상태에서 **정지**해야 손가락이 말릴 시간이 생긴다. palm 실측 선속도 기준.
    palm_still_gain: float = 10.0
    grasp_weight: float = 4.0  # ★D3 기본 (09.01 승격)
    # ★★감쌈 비중 0.55 → 0.80(폐쇄 0.20). 폐쇄 항은 **큰 상금이 아니라 넛지**여야 한다 —
    #   approach 가 손 모양을 못 보게 고친 뒤로는 건너야 할 계곡이 없어졌고, 08.27 실측
    #   (s2r_b1)에서 폐쇄 상금 5.1/step 이 전체의 93% 를 먹으며 주차장이 됐다.
    #   이제 폐쇄 2.4 < 감쌈 9.6 < lift 30·q 로 상한이 확실히 갈린다.
    grasp_envelope_credit: float = 0.80
    lift_weight: float = 30.0
    lift_envelope_mix: float = 0.6
    transfer_weight: float = 15.0
    transfer_sharpness: float = 6.0
    stay_weight: float = 8.0
    stabilize_weight: float = 1.0  # ★D3 기본 (09.01 승격)
    stability_weight: float = 1.0
    success_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    # ★"멈춤"을 정의하는 항. **0.0 = 꺼짐(현행 동작)**. 08.29 J1 실측:
    #   `hand_joint_err_movable_mean` 0.16 rad · `hand_torque_sat_frac` 0.21 —
    #   가동 관절의 20%가 토크 천장에 붙어 있고 보상에는 힘 항이 하나도 없었다.
    #   sim 이 안전한 이유가 `effort_limit_sim`(1.5) 뿐이고 실기는 7.5 N·m 로 5배다.
    hand_overdrive_weight: float = 0.0
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
    # ★★09.01 실기 rosbag 실측 (sim2real/logs 의 6개 bag · /joint_states):
    #     팔 position 양자화 3.815e-4 rad(=100/2^18) · velocity 양자화 4.9e-3~2.2e-2 rad/s
    #     정지 구간 σ: pos 2e-4 rad · vel 4e-3 rad/s
    #     운동 구간 σ(0.2s 이동평균 하이패스, 6 bag 중앙값): **pos 9e-4 · vel 4.5e-2**
    #     손(dg5f) pos 양자화 1.745e-3 rad(0.1°) · 정지 σ 1e-3 rad
    #   ⇒ `obs_noise_qvel` 0.05 는 운동 구간 실측과 거의 일치한다(우연이지만 맞다).
    #   ⇒ ★`obs_noise_qpos` 0.01 은 **실측의 10배**다(0.01 rad = 0.57° = 26 LSB).
    #     정밀 파지 트랙에서 매 관절에 실기의 10배 잡음을 넣고 있었다는 뜻이라
    #     실측값 0.001 로 낮추는 것이 옳지만, **D3 기본값 정합을 위해 지금은 두고**
    #     E 라운드에서 단일변수로 검증한 뒤 승격한다(노이즈는 줄이는 방향이라
    #     성능은 오르면 올랐지 떨어지지 않는다).
    obs_noise_qpos: float = 0.01
    obs_noise_qvel: float = 0.05
    obs_noise_body: float = 0.005
    obs_noise_object: float = 0.015

    # ---- 지각 모델 (09.01 신설 — 둘 다 기본 False = 현행 항등) ------------------------
    # ★★실기에서는 손이 컵을 감싼 순간 비전이 컵을 잃는다. 그런데 지금 actor obs 는
    #   래치 뒤에도 시뮬 **참값** 물체 pose 를 받는다 — 정책이 실기에 없는 정보에
    #   의존하도록 학습되고 있다. True 면 래치 순간의 palm 상대위치를 스냅샷해
    #   이후엔 현재 palm 자세로 굴린 값을 준다(= "잡았으니 손과 같이 움직인다" 추정).
    #   컵이 손안에서 미끄러져도 정책은 모른다 — 실기와 동일하게 접촉력으로만 안다.
    obs_object_rigid_after_latch: bool = False
    # ★★누수 차단. `goal_rel = goal_pos − obj_pos` 인데 `goal_pos` 는 에피소드 상수라
    #   `obj_pos = goal_pos − goal_rel` 로 **참값이 그대로 복원된다**. 즉 지금까지
    #   `obs_noise_object` 를 아무리 키워도 정책은 깨끗한 물체 위치를 볼 수 있었고,
    #   그 축은 사실상 무효였다. True 면 노이즈를 **한 번만** 뽑아 palm_to_obj ·
    #   obj_to_tips · goal_rel 세 항에 **같은 추정값**을 실어 우회를 막는다.
    obs_object_noise_coherent: bool = False

    # ---- finger_closure (08.29 신설·기본 0 = 항등) -----------------------------------
    # 접촉 전 손가락별 연속 신호 — (1−접촉)·exp(−k·‖tip−파지중심‖) 평균 × close_gate.
    # ★가중 상한 1.0 — 소등 항 크기 < 접촉 사다리 이득 부등식(절벽 회피 재발 방지).
    finger_closure_weight: float = 3.0  # ★D3 기본 (09.01 승격)
    finger_closure_sharpness: float = 8.0

    # ---- 씬 기하 ---------------------------------------------------------------------
    table_surface_z: float = 0.200           # env.usd top_plate 상면(점군 실측)
    object_origin_offset_z: float = 0.0773   # cup_big USD 원점 ↔ 바닥
    object_spawn_pad: float = 0.005          # 스폰 침투 반동 방지
    object_spawn_z: float = 0.0              # __post_init__ 파생 (단일 소스)
    spawn_range: float = 0.02                # 스폰 xy 균등 반범위 (ADR OFF 고정값)

    # ---- 커리큘럼 --------------------------------------------------------------------
    # ★ADR 은 끄고 시작한다. 과제 성립 후 스폰 범위부터 켠다.
    enable_adr: bool = False
    # ★★전역 난이도 스칼라(level 0→1) 하나가 세 축을 선형 스케일한다.
    #   env 별 난이도 금지 — 성공 게이팅을 env 별로 걸면 나쁜 시드가 영구 고착된다
    #   (08.27 h7 데드락 실측). 승급은 단조(하강 없음), 판정 창은 종료 에피소드 수.
    adr_success_threshold: float = 0.7       # 창 성공률이 이 값 이상이면 승급
    adr_eval_episodes: int = 4096            # 판정 창 크기 (종료 에피소드 수)
    adr_step: float = 0.1                    # 승급 당 level 증가
    #   ★최대치는 부팅 검증(`_assert_goal_reachable`)이 **비확장 프로필 박스**로
    #   잰다 — 도달영역(y 폭 실측 ~0.32m)이 물리 한계라 스폰+이송 합이 그 안이어야 한다.
    adr_spawn_range_max: float = 0.05        # 축① level=1 의 스폰 xy 반범위
    # ★★09.01 이송 축 무력화(사용자 확정: "목표 이송 부분은 일단 빼도 됨 — IK 로 풀어도 됨").
    #   ADR 의 성격을 **과제 난이도**에서 **sim2real 랜덤화**로 바꿨다. y/z/x 세 축 전부
    #   base 와 같은 값을 넣어 폭 0 으로 만든다. **필드를 지우지 않는 이유**는 아카이브된
    #   런 118개의 `params/env.yaml` dump 정합 때문이다(`run_cfg_restore` 는 미지 키를
    #   조용히 건너뛰므로 삭제해도 안 죽지만, 남겨 두는 편이 재생 이력에 정직하다).
    adr_goal_y_max: float = 0.0              # = |base y| → 폭 0 (구 0.12 = 이송 축)
    # ---- 목표 3축 샘플링 (08.30 — 단조 상승의 망각·단일 방향 한계 처방) -------------
    # ★★실측 근거: m1_final 성공률 지도(1024 ep/칸)에서 이송 y **0.12 → 0.94~0.98**,
    #   0.085 → 0.84~0.86, **0.05 → 0.000**. 난이도를 단조로 올리기만 하면 정책이
    #   시작 구간을 잊고 "한 방향으로 14cm" 하나만 배운다(스폰 범위는 ±2→±5cm 로
    #   넓혀도 −4.4%p 뿐이라 무해했다). → 매 에피소드 **[base, 현재레벨] 안에서 뽑는다**.
    #   x 는 ±범위(방향 다양성), z 는 [base, max] 구간.
    adr_goal_sample: bool = False            # False = 현행(레벨 값 고정)
    adr_goal_x_max: float = 0.0              # level=1 의 x 반범위(0 = 축 끔)
    # ★★09.01 역방향 버그 수정. 구 0.08 은 base z(=goal_offset_xyz[2]=0.12)보다 **작아**
    #   `_z_eff = 0.12 + lvl·(0.08−0.12)` 가 되어 **승급할수록 목표가 낮아졌다**(level 1
    #   에서 −4cm). 아래 obs 노이즈 주석이 경고하던 바로 그 함정이 z 축에도 있었다.
    #   base 와 같은 값으로 맞춰 폭 0 + 버그 동시 제거. 재발은 `_assert_adr_monotonic`
    #   부팅 가드가 막는다.
    adr_goal_z_max: float = 0.12             # = base z → 폭 0 (구 0.08 = 역방향)
    # ★base(obs_noise_object=0.015)보다 커야 한다 — 작으면 보간이 거꾸로 가서
    #   승급할수록 노이즈가 **줄어든다**(V1 스모크 실측 0.0145→0.0125 로 잡음).
    adr_obs_noise_object_max: float = 0.03   # 축① level=1 의 물체 pose obs 노이즈

    # ---- sim2real 랜덤화 축 (09.01 신설 — 전부 기본 = base = 항등) --------------------
    # ★관절 상태 노이즈. 실측 근거는 `obs_noise_qpos` 블록 참조.
    #   승격 목표: qpos 0.003(실측 최악 2.5e-3 의 1.2배) · qvel 0.12(실측 최악 1.05e-1).
    adr_obs_noise_qpos_max: float = 0.01     # = base → 폭 0
    adr_obs_noise_qvel_max: float = 0.05     # = base → 폭 0
    # ★물체 질량 배율. 승격 목표 (0.5, 2.0) — 붓기 과제로 가면 내용물이 질량으로 온다
    #   (빈 컵 ↔ 물 찬 컵). 런타임 확장 가능(`__call__` 인자로 매 reset 읽는다).
    adr_mass_scale_max: tuple[float, float] = (1.0, 1.0)
    # ★관절 PD 게인 배율. 승격 목표 (0.8, 1.2).
    #   실효성 확인: `grasp_s2r_control.py` 는 `set_joint_position_target` 으로 **목표만**
    #   쓰고 토크는 articulation 의 ImplicitActuator PD 가 만든다 → 게인 DR 이 그대로 실린다.
    #   ⚠하한 0.8 은 손 파지력도 20% 깎는다. 손 PD 는 이미 토크 포화 21% 라
    #     파지가 무너지면 하한을 0.9 로 좁힌다(스모크로 판정).
    adr_joint_gain_scale_max: tuple[float, float] = (1.0, 1.0)
    # ★★마찰은 **ADR 축이 될 수 없다**. `randomize_rigid_body_material` 은
    #   `material_buckets` 를 term 인스턴스 생성 시 **1회만** 샘플링하고(PhysX 재질
    #   64,000개 상한) `__call__` 은 그 고정 버킷에서 뽑기만 한다 — 런타임에 범위를
    #   바꿔도 **무증상 no-op** 이다(자매 `grasp_v2/grasp_adr.py` 가 재질을 확장하지만
    #   실제로는 아무 일도 안 일어난다). 그래서 여기만 **cfg 고정 범위**로 연다.
    #   절대값이지 배율이 아니다. 승격 목표 (0.5, 1.5).
    object_friction_range: tuple[float, float] = (1.0, 1.0)

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
    # ★08.29 진단용. **기본 True = 현행 동작**. `replicate_physics=False` 에서
    #   `randomize_rigid_body_material` 이 shape 개수를 잘못 세는 것이 확인됐다
    #   (grasp_v2 는 같은 항에서 `Expected 1163, got 1162` 로 부팅 실패).
    #   재질이 엉뚱한 shape 에 들어가면 마찰·접촉이 무너진다 — 다물체 폭주의
    #   남은 후보라 끌 수 있게 연다.
    enable_events: bool = True
    surface_friction: float = _FRICTION

    # ---- 다물체 (08.29 신설) ----------------------------------------------------------
    # ★기본 `single_cup` = 현행 동작(단일 USD · replicate_physics=True).
    #   `cup_family` 로 바꾸면 8종이 `env_id % 8` 로 결정론 배정된다.
    #   물체 **정체성은 obs 에 들어가지 않는다** — onehot 도 뱅크 인덱스도 넣지 않는다.
    #   policy obs 는 상대 위치(`palm_to_obj`·`obj_to_tips`·`goal_rel`)뿐이라 다물체로
    #   가도 차원·의미가 불변이다. 원점 오프셋은 **스폰·보상 경로에만** 쓴다(특권 정보).
    object_bank: str = "cup_family"  # ★D3 기본 (09.01 승격)

    robot_cfg: ArticulationCfg = None  # __post_init__ 에서 프로필로 조립

    def _assert_gain_branch(self, profile) -> None:
        """`use_real_gains` 의도와 **실제로 조립된 게인**이 일치하는지 부팅에서 대조.

        ★게인 선택은 `robot_profiles.py` 가 import 시점에 환경변수로 하므로 cfg
          오버라이드로는 못 바꾼다. 그래서 여기서 할 수 있는 일은 "조용히 다른 게인으로
          도는" 것을 막는 것뿐이다 — 어긋나면 **어떻게 고치는지까지** 알려주고 죽인다.

        재생 시나리오: dump 가 `use_real_gains: true` 를 복원했는데 환경변수를 안 켜면
        여기서 걸린다. 이게 없으면 4배 단단한 KUKA 게인으로 조용히 재생돼
        "정책이 이상하다"는 오진으로 이어진다(m1_final 계열의 실패).
        """
        _specs = set(profile.actuator_specs)
        _kuka = "right_arm_proximal" in _specs
        _real = "right_arm_j1" in _specs
        if not (_kuka or _real):
            return                      # 게인 분기가 없는 프로필(gripper_left 등)
        if bool(self.use_real_gains) != _real:
            _want = "실측 정합(r2s)" if self.use_real_gains else "KUKA 기본"
            _got = "실측 정합(r2s)" if _real else "KUKA 기본"
            raise RuntimeError(
                f"[grasp_s2r] 게인 분기 불일치 — cfg 는 {_want} 을 요구하는데 "
                f"프로필은 {_got} 으로 조립됐다.\n"
                "  환경변수로 맞춰라: "
                f"HDGP_S2R_REAL_GAINS={'1' if self.use_real_gains else '0'} "
                "(robot_profiles.py 가 **import 시점**에 읽는다)\n"
                "  ★재생이라면 dump 의 use_real_gains 가 진실이다.")

    def finalize_after_overrides(self) -> None:
        """cfg 필드에서 **파생되는 구조**를 다시 만든다. 멱등이어야 한다.

        ★★hydra 는 `__post_init__` **뒤에** `env_cfg.from_dict(...)` 로 오버라이드를
          적용하고 `__post_init__` 를 다시 부르지 않는다(IsaacLab `hydra_task_config`).
          그래서 `env.object_bank=cup_family` 같은 오버라이드는 파생 구조에 반영되지
          않는다 — 08.29 스모크에서 `replicate_physics` 가 True 로 남아 실측됐다.
          env `__init__` 이 `super()` **전에** 이 메서드를 다시 부른다.
          `replicate_physics` 는 `InteractiveScene.__init__` 이 소비하므로
          `_setup_scene` 에서 고치면 이미 늦다.
        """
        profile = PROFILES[self.profile_name]
        # ★robot_cfg 도 파생 구조다 — `enable_self_collisions` 를 `__post_init__` 에서만
        #   소비하면 CLI 오버라이드가 조용한 no-op 이 된다. 08.29 확정: 다물체
        #   (`replicate_physics=False` per-env 파싱)에서 손 hull 초기 겹침 ×
        #   자기충돌 ON 이 폭주의 근본 원인(sick 0/256 완치 실측) → 다물체 학습은
        #   `env.enable_self_collisions=False` 로 기동해야 하고, 그게 실리려면
        #   재구축이 여기 있어야 한다.
        self.robot_cfg = _build_robot_cfg(
            profile, bool(self.enable_self_collisions))
        self._assert_gain_branch(profile)
        if not bool(self.enable_events):
            self.events = None
        else:
            # ★마찰은 **여기서만** 열 수 있다 — ManagerBase 가 cfg 를 deepcopy 하므로
            #   런타임(`env.cfg.events`) 수정은 무효고, 재질 버킷은 term 인스턴스
            #   생성 시 1회 샘플링이라 `event_manager` 를 고쳐도 no-op 다. 즉
            #   **cfg 단계(= deepcopy 이전)가 유일한 유효 지점**이다.
            #   기본 (1.0, 1.0) 이면 구 동작과 동일한 상수 재질이다.
            _fr = tuple(float(v) for v in self.object_friction_range)
            self.events.object_material.params["static_friction_range"] = _fr
            self.events.object_material.params["dynamic_friction_range"] = _fr
        self._apply_object_bank()
        # 스폰 높이는 여기 한 곳에서만 파생한다(이중 패딩 사고 차단).
        # ★다물체면 이 값은 **뱅크 최댓값**이다 — env 별 실제 높이는 런타임에서 준다
        #   (`_obj_origin_off`). 여기서 작은 값을 쓰면 큰 컵이 테이블을 뚫고 스폰된다.
        self.object_spawn_z = (
            self.table_surface_z + self.object_origin_offset_z + self.object_spawn_pad)
        self.object_cfg.init_state.pos = [
            profile.object_spawn_center[0], profile.object_spawn_center[1],
            self.object_spawn_z,
        ]
        self._derive_spaces(profile)

    def _apply_object_bank(self) -> None:
        """뱅크 크기에 따라 스폰·물리복제·접촉필터를 한 곳에서 조립한다.

        ★함정 3개를 여기서 막는다(전부 재발 이력):
          ①뱅크>1 인데 `replicate_physics=True` 면 전 env 가 같은 물체를 받는다.
          ②접촉 필터가 루트 Xform 을 가리키면 `force_matrix_w` 가 **항상 0** 이다 —
            뱅크의 `rigid_body_name`(전 스펙 동일해야 함)에서 만든다.
          ③`base_origin_offset_z` 미측정 스펙이 섞이면 안착 높이를 못 구한다 →
            `origin_offset_z` 프로퍼티가 fail-loud 한다.
        """
        from openarm.agnostic.modules import object_bank as _ob

        # ★멱등성 — 이 메서드는 `__post_init__` 과 env `__init__` 에서 **두 번** 불린다.
        #   원본 단일 스폰을 보존해 두지 않으면 두 번째 호출이 MultiAsset 을 또 감싸고
        #   `replace(...)` 가 usd_path 없는 cfg 에서 터진다.
        if getattr(self, "_object_spawn_base", None) is None:
            self._object_spawn_base = self.object_cfg.spawn
            self._table_usd_base = self.table_cfg.spawn.usd_path
        bank = _ob.get(self.object_bank)
        _missing = bank.missing_files()
        if _missing:
            raise RuntimeError(
                f"물체 뱅크 '{bank.name}' 의 USD 누락: {list(_missing)}")
        _offs = [s.origin_offset_z for s in bank.specs]      # 미측정이면 여기서 fail-loud
        self.object_origin_offset_z = max(_offs)
        self.object_contact_filter = (
            f"/World/envs/env_.*/Object/{bank.rigid_body_name}",)
        if not bank.needs_multi_asset:
            # ★`replicate_physics` 를 여기서 True 로 되돌리지 않는다 — 명시 오버라이드
            #   (`env.scene.replicate_physics=False`)를 덮어쓰면 분리 실험을 못 한다.
            #   기본값은 cfg 선언(True)이 이미 준다.
            self.object_cfg.spawn = self._object_spawn_base
            self.table_cfg.spawn.usd_path = self._table_usd_base
            return

        from dataclasses import replace

        from isaaclab.sim.spawners.wrappers import wrappers_cfg as _wrap

        self.scene.replicate_physics = False
        # ★★다물체는 `replicate_physics=False` 가 필수이고, 그때는 `clone_environments`
        #   의 `enable_env_ids` env 간 충돌 격리가 사라진다. 작업면이 원시 정적 프림이면
        #   전 env 가 한 충돌 그룹에 남아 팔이 물린다 — 08.29 분리 실측: 단일 컵으로
        #   고정하고 플래그만 뒤집어도 abnormal 0.0000→0.849 · joint_err 0.058→0.74 rad.
        #   그래서 다물체에서는 kinematic RigidBodyAPI 를 저작한 사본을 쓰고 테이블을
        #   **씬 자산**(`RigidObject`)으로 올린다(자매 `tesollo/grasp_v2` 와 같은 규약).
        #   사본 생성: `scripts/assets_tools/build_env_rigid_usd.py`
        _rigid_usd = _os.path.join(_ASSETS_DIR, "env", "usd", "env_rigid.usd")
        if not _os.path.isfile(_rigid_usd):
            raise RuntimeError(
                f"다물체({bank.name})는 kinematic 작업면이 필요하다: {_rigid_usd} 없음 — "
                "`python3 scripts/assets_tools/build_env_rigid_usd.py` 로 먼저 빌드할 것")
        self.table_cfg.spawn.usd_path = _rigid_usd
        _base = self._object_spawn_base
        self.object_cfg.spawn = _wrap.MultiAssetSpawnerCfg(
            assets_cfg=[
                replace(_base, usd_path=s.usd_path, scale=tuple(s.scale),
                        mass_props=MassPropertiesCfg(mass=float(s.mass)))
                for s in bank.specs
            ],
            random_choice=False,          # env_id % N — `assign_indices` 와 같은 규약
            activate_contact_sensors=True,
        )

    def __post_init__(self):
        # robot_cfg·공간 차원 전부 finalize_after_overrides 가 만든다
        # (★hydra 는 __post_init__ 뒤에 덮고 재호출하지 않는다 — CLI 반영 지점).
        self.finalize_after_overrides()

    def _derive_spaces(self, profile) -> None:
        """액션/관측 차원 파생 — `hand_layout` 을 소비하므로 finalize 에서 불려야
        `env.hand_layout=per_finger` CLI 가 실린다(O1 부팅 fail-loud 실측 08.29)."""
        n_arm = profile.num_arm_joints
        n_hand = profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D 델타 + 손 슬롯(레이아웃 파생).
        if str(self.hand_layout) == "per_finger":
            _slots = [s for m in profile.hand_finger_channels.values()
                      for s in m.values()]
            if not _slots:
                raise RuntimeError(
                    f"[{profile.name}] hand_layout=per_finger 인데 "
                    "hand_finger_channels 가 비어 있다")
            if sorted(set(_slots)) != list(range(max(_slots) + 1)):
                raise RuntimeError(
                    f"[{profile.name}] 액션 슬롯이 연속이 아니다: {sorted(set(_slots))}")
            self.action_space = 6 + max(_slots) + 1
        else:
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
