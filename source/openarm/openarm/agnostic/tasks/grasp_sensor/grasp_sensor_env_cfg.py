"""robot-agnostic grasp-sensor 환경 설정.

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

from isaaclab.envs import mdp as _mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

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

    ★구 주석은 mode="startup" 을 "고정값이라 reset 마다 걸 이유가 없다"로 정당화했다.
      Kuka 는 **reset** 을 쓰는데, ADR 이 매 리셋마다 범위를 넓히기 때문이다. 우리는
      아직 ADR 이 없어 값이 고정이지만, 모드를 맞춰 두면 ADR 도입이 상수 교체로 끝난다.
    """

    # ★★08.25 DEXTRAH Kuka `EventCfg` 를 그대로 옮긴다(mode="reset", num_buckets 250).
    #   ★restitution 0.0 → **1.0**. 구 주석은 "(1.0,1.0) 을 그대로 복사하면 컵이
    #     완전탄성이 된다"고 경고했는데, Kuka 는 `bounce_threshold_velocity=0.2` 와
    #     **짝으로** 쓴다 — 상대속도 0.2 m/s 이하 접촉에는 반발이 적용되지 않으므로
    #     느린 파지 접촉에서는 무해하다. 구 우리 조합은 (restitution 0, bounce 0.01)로
    #     내부적으로는 일관됐지만 Kuka 와 달랐다. 둘 다 Kuka 값으로 간다.
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
    # ★아래 셋은 Kuka 에 있고 우리엔 없던 EventTerm 이다. 공칭 파라미터에서는 전부
    #   항등(scale 1.0)이거나 0 이라 **거동이 바뀌지 않지만**, ADR 을 붙일 때 여기가
    #   확장 지점이다(Kuka adr_cfg_dict: stiffness/damping (0.5,2.) ·
    #   joint_friction (0.,5.) · mass (0.5,3.)).
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
    # ★Kuka 는 `starting_robot_dof_friction_coefficients`(팔 1.0 / 손 0.01)를
    #   `default_joint_friction_coeff` 에 넣어 두고, 이 term 이 **scale (0,0)** 으로
    #   곱해 **실효 마찰 0** 으로 만든다(그리고 실제 write 는 주석 처리돼 있다).
    #   즉 우리가 `friction=0.213/…` 를 제거한 결과와 최종값이 같다.
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
            # ★★08.25 DEXTRAH Kuka(`assets/kuka_allegro/kuka_allegro.py`) 값으로 전환.
            #   `disable_gravity=True` 는 원래 일치했다.
            #   ★`max_depenetration_velocity` 1.0 → 1000.0 은 **되돌릴 후보 1순위**다.
            #     우리가 5.0→1.0 으로 낮춘 근거는 실측이다(전형 접촉 13~20N 인데
            #     관통 복원 스파이크 7218N). Kuka 는 손 링크가 4개(allegro)이고 우리는
            #     27개 convexDecomposition 이라 관통 빈도가 다르다.
            #     접촉력 스파이크가 관측되면 여기부터 본다.
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
                # ★True (08.21 자산 재생성으로 해소). 구 자산에서는 홈 자세의 상시 접촉쌍이
                #   반발력을 만들어 유휴 480스텝에 palm 이 180mm 표류했으나, 신규 USD 가
                #   감사로 찾은 6쌍을 PhysicsFilteredPairsAPI 로 필터하고 convexHull(64-vert
                #   외접) → convex_decomposition 으로 되돌려 유령 접촉을 없앴다.
                #   실측: 유휴 480스텝 palm 드리프트 0.00mm·|qd| 0.00,
                #        full-grip 손관절합 24.1(OFF) → 18.8(ON) = 손가락 겹침 차단.
                # ★★True 복귀 (08.24) — hand_repulsion(18mm 벽)이 **감쌈 자체를 금지**하는
                #   것이 lstm_test4 실측으로 확정됐다(사용자 판정: "fabric 과 관절공간
                #   제어는 성립할 수 없다"). tip_cyl 전환과 함께 repulsion 을 끄므로
                #   관통 방지는 PhysX 로 되돌린다 — 둘 다 끄면 막는 장치가 없다.
                #   처리량 −52% 는 파지가 성립한 뒤 다시 볼 문제다.
                # (이하는 08.23 OFF 시도의 기록 — 근거가 뒤집힌 경위 보존)
                # ★★False (08.23) — 손가락 관통 방지를 Fabrics `hand_repulsion` 이
                #   **계획 단계**에서 맡는다(cfg.use_hand_repulsion). 자기충돌 검출은
                #   스텝 시간의 55~64% 를 쓰는데(자매 트랙 실측), 그 비용의 원인은
                #   solver 도 접촉 빈도도 아닌 **손 27링크의 convexDecomposition 조각
                #   수**였다(08.23 selfcollision-cost-anatomy: 2.2~2.8배·자세 무관).
                #   자매 트랙 검증: repulsion ON 이면 fabric_q 의 다른 손가락 구 최소거리
                #   20.1mm·18mm 미만 0.0% — 계획에 관통 해가 없으므로 정책이 관통으로
                #   이득 보는 전략을 학습할 수 없다. 잔여 관통은 PD 추종오차의 결과라
                #   정책이 조종할 수 없다.
                # ★되돌릴 때는 use_hand_repulsion 도 함께 볼 것 — 둘 다 끄면 관통을
                #   막는 장치가 하나도 없다.
                # ★★08.25 상수 → cfg 스위치(`enable_self_collisions`)로 뺐고 기본을
                #   **False** 로 내렸다. 근거: Kuka-Allegro 자산은 True 지만, DEXTRAH
                #   저자들이 **같은 로봇(OpenArm+Tesollo)으로 포팅할 때는 껐다** —
                #   `assets/open_tesollo/open_tesollo.py:43` 과 `open_l_tesollo_r.py:43`
                #   둘 다 `enabled_self_collisions=False`. 우리 로봇에 대한 원저자
                #   선택이 곧 이 값이다. 전면 convexHull 전환과 짝이다(조각이 하나가
                #   되면 인접 링크 껍질끼리 상시 겹쳐 자기충돌이 오히려 늘어난다).
                # ★★정정(같은 날): "관통 방지는 use_hand_repulsion 이 맡는다"고 적었으나
                #   그 값은 **False** 다. 게다가 fabric 의 `body_points` 그룹은 쌍 목록을
                #   **빈 리스트로 넘긴다**(openarm_tesollo_pose_fabric.py:407 —
                #   `_add_repulsion_group("body_points", frames, radii, [], p)`).
                #   따라서 지금 자기충돌을 막는 장치는 **하나도 없다**.
                #   Kuka 는 fabric 에서 13쌍(전부 `손 링크 ↔ iiwa7_link_2`)을 실제로 걸고
                #   **그 위에 PhysX self-collision 도 True** 다. 우리는 둘 다 비었다.
                #   손가락끼리는 외전 관절 `_1` 이 open/grip 양쪽 0.0 이라 핀 고정이라
                #   측면으로 모일 수 없지만(구조적 안전), **팔↔몸통은 무방비**다.
                #   선택지: ①`use_hand_repulsion=True`(손 쌍은 params 에 이미 있고
                #   joint_slice 로 팔 결합 차단됨) ②이 값을 True 로 되돌림
                #   ③Kuka 처럼 `손↔팔뚝` 쌍을 body_points 에 추가.
                enabled_self_collisions=enable_self_collisions,
                # ★P-9 실측(2048env)으로 16 확정: 32 는 fps 를 25% 깎는데(11.9k→8.9k)
                #   파지 품질이 나아지지 않았다. 오히려 접촉력이 더 높았다
                #   (solver32 Fa 28N/Fb 25N vs solver16 Fa 18N/Fb 8N) = 상호침투를 더
                #   밀어내고 있었다는 뜻. self-collision + convex_decomposition 자산이
                #   이미 관통을 막으므로 32 는 불필요한 비용이다.
                # ★★08.25 Kuka 값(8 / 0)으로 전환. 구 16/1 은 P-9 실측으로 32 대비
                #   비용만 크고 이득 없음을 확인해 고른 값이고, Kuka 는 8/0 이다.
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            # Kuka: drive_type="force".
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
class GraspSensorEnvCfg(DirectRLEnvCfg):
    # ---- 로봇 선택 (서브클래스가 덮어씀) ----------------------------------------
    profile_name: str = "tesollo_right"

    # ---- 시뮬레이션: 물리 120 Hz / 정책 60 Hz ------------------------------------
    # ★08.25 DEXTRAH Kuka 원본값(10.0 = 600 스텝). 구 8.0(480)은 우리 임의값이었다.
    episode_length_s: float = 10.0           # 600 스텝 (Kuka 동일)
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # ★씬 기본 물리 재질. 지정이 없으면 IsaacLab 기본 μ=0.5 로 돌고, PhysX 결합모드가
        #   **average** 라 컵만 0.75 로 올려도 실효는 (0.75+0.5)/2 = 0.625 다.
        #   → 씬 기본 + 로봇 + 컵 + 테이블 **네 곳 전부** 0.75 여야 한다.
        #   파지 용량 ∝ (μcosα + sinα): 0.565 → 0.814 (×1.44).
        #   restitution 0.0 — grasp_v1 의 (1.0,1.0) 은 "중립 1.0배" 주석과 달리 **절대값**이라
        #   그대로 복사하면 컵이 완전탄성이 된다(randomize_rigid_body_material 은 배율 아님).
        # ★★08.25 DEXTRAH Kuka 값으로 전환 — μ 0.75 → 1.0, bounce 0.01 → 0.2.
        #   구 0.75 는 "파지 용량 ∝ (μcosα + sinα) 를 1.44배" 근거로 우리가 고른 값이고,
        #   Kuka 는 1.0/1.0 이다. GPU 버퍼는 Kuka 가 4·5·2^15 만 지정하고 나머지는 기본값
        #   이지만, 우리는 손 27링크 convexDecomposition 때문에 기본값이면 부팅이 깨진
        #   이력이 있어 **용량 계열은 유지**한다(물리 거동에 영향 없는 메모리 한도다).
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
    # ★Kuka: num_envs 4096 · env_spacing 2.0 · replicate_physics **False**.
    #   replicate_physics=False 는 Kuka 가 env 마다 **다른 물체**를 스폰하기 때문이고,
    #   우리는 단일 컵이라 True 가 맞다(False 로 두면 filter_collisions 를 수동으로
    #   해야 하고 부팅이 느려진다 — MultiAsset 규약). **이 한 항목만 의도적 유지.**
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True,
    )

    # ---- 공간 (프로필에서 __post_init__ 이 계산) ----------------------------------
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # ---- Fabrics (grasp_sensor 검증값 승계) ----------------------------------------
    # ★★08.25 DEXTRAH Kuka 원본값(1/60)으로 복귀. 구 주석은 "1/60 × 2 는 2배속이 된다"며
    #   실시간 정합을 이유로 절반(1/120)을 썼는데, **원본이 바로 그 2배속**이다
    #   (Kuka: sim_dt 1/120 · decimation 2 · fabrics_dt 1/60 · fabric_decimation 2
    #    → 정책 스텝 1/60 s 당 fabric 시간 1/30 s).
    #   ★실측으로는 **지속 slew 에 영향이 거의 없다**(절대 매핑·목표를 박스 끝에 두고
    #     +x/+y/+z: 4.873/7.202/5.370 → 4.292/7.440/5.173 mm/step = 0.88/1.03/0.96배).
    #     과도구간 최댓값만 커진다(11.0 → 14.4mm). 즉 지속 상한은 fabric 의 속도·가속
    #     한계가 정하지 이 dt 가 아니다 — "fabrics_dt 가 속도를 반토막" 가설은 기각됐다.
    #   그래도 원본값을 쓴다: 편차를 남길 이유가 없고 손해도 없다.
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    # ★08.25 Kuka ADR `fabric_damping.gain = (10., 20.)` — ADR 은 min 에서 시작해
    #   max 로 확장하므로 **학습 시작값은 10**. 우리에겐 ADR 이 없어 초기값을 쓴다.
    fabrics_damping_gain: float = 10.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # ★★팔 PD 속도 피드포워드(08.25 복구). DEXTRAH 원본은 fabric 이 만든 `fabric_qd` 를
    #   그대로 속도 목표로 준다(`velocity_target_factor` 공칭 1.0). 이 저장소는 전 트랙이
    #   **0** 을 넣고 있었고, 그러면 implicit PD 의 감쇠항이 참조 궤적의 움직임 자체를
    #   반대로 밀어 **속도에 비례하는 추종 지연**이 생긴다:
    #     피드포워드 0 → kp·err = kd·v + τ_마찰  →  err ≈ (kd/kp)·v = 0.2·v [rad]
    #     피드포워드 1 → kp·err = τ_마찰만        →  err ≈ 0.49/400 = 0.0012 rad = 0.07°
    #   lstm_test9 실측 joint_err_max 0.5~1.3 rad 이 위 식과 일치한다. 구 leash(목표를
    #   실측±5cm 로 재클램프)는 이 지연의 **증상 억제기**였고, 원인을 안 고친 채 제거해
    #   palm_err 51~65mm → 90mm(최대 435mm)로 악화됐다.
    #   ★이 값이 바뀌면 폐쇄 속도·파지중심 등 제어 위에서 잰 상수는 전부 재측정 대상이다.
    fabric_velocity_ff_scale: float = 1.0
    # ★★로봇 자기충돌 — **Kuka-Allegro 구성(True)**.
    #   08.25 중 한 번 False 로 내렸었다. 근거는 DEXTRAH 의 OpenArm+Tesollo 포팅이
    #   False 였다는 것인데, 사용자 결정으로 그 포팅은 참조에서 제거했다. 기준은
    #   **Kuka-Allegro** 하나다(`assets/kuka_allegro/kuka_allegro.py:42` = True).
    #   Kuka 는 self-collision 을 켜고 그 위에 fabric 반발까지 이중으로 건다.
    #   전면 convexHull 이라 조각 수가 링크당 1개여서 decomposition 때의 2.2~2.8배
    #   비용이 나오지 않는다(그 비용의 본체가 조각 수였다 — 08.23 해부).
    enable_self_collisions: bool = True
    # ★★손 PD 속도 피드포워드(08.25 3차 감사). Kuka 는 `set_joint_velocity_target` 을
    #   **actuated 23관절 전체**(팔 7 + 손 16)에 준다 — 손도 fabric 이 plant 라
    #   `fabric_qd` 가 그대로 손 관절에 들어간다. 우리는 팔 7개에만 주고 있어서 손은
    #   암묵적 목표 0 → 감쇠항 kd·(0 − q̇) 가 닫는 동작을 상시 반대로 민다.
    #   우리 손은 fabric 밖(시너지 램프)이라 `fabric_qd` 를 쓸 수 없다 — fabric_q 의 손
    #   구간을 매 스텝 덮어쓰므로 그 qd 는 램프의 도함수가 아니다. 대신 **램프 자체의
    #   도함수**(Δtarget / 정책 dt)를 쓴다. 이것이 Kuka 와 같은 의미의 신호다.
    #   산수: kd/kp = 0.1/3.0 = 0.0333 s. 폐쇄 0.005 rad/step @60Hz = 0.3 rad/s
    #        → 지연 err ≈ 0.010 rad(0.57°). 정지 시(v=0) 0 이므로 **과도구간 전용**이다.
    hand_velocity_ff_scale: float = 1.0
    # ★★policy obs 의 측정 속도 계수. Kuka 는 ADR `observation_annealing.coefficient`
    #   가 (0., 0.) 이라 전 구간 0 을 곱한다(env.py:1321, 1337) — 측정 속도가
    #   sim2real 에서 가장 안 맞는 채널이고, distillation student 와 차원을 맞추려고
    #   채널만 남긴 것이다.
    # ★08.25 사용자 결정으로 **1.0**(살림). 당장 student 학습을 하지 않으므로 27D 를
    #   상수 0 으로 버릴 이유가 없다. student 를 붙일 때 이 값을 0 으로 되돌리면
    #   차원 변경 없이 Kuka 배선으로 복귀한다.
    obs_measured_velocity_scale: float = 1.0
    # ★palm leash 는 제거됐다(08.22) — 정책이 팔 목표에 대해 전권을 갖는다.
    #   목표의 유일한 상한은 아래 워크스페이스 박스(profile.palm_box_*)다.
    #   근거는 grasp_sensor_env.py `_pre_physics_step` 주석 참조.

    # ---- 액션 스케일 (스텝당 delta, 60 Hz) ----------------------------------------
    # ★★08.25 실측 정합 0.01 → 0.002. 지령 상한을 **팔이 실제로 낼 수 있는 속도**에
    #   맞춘다. 초과분은 이동이 되지 않고 목표 인플레로만 쌓이며, 그 구간의 정책 액션은
    #   효과도 gradient 도 없다(구 leash 제거 당시 "팔 액션의 절반이 버려지고 있었다"의
    #   정체). probe_armscale 실측 — 지령 대비 실제 palm 변위:
    #     0.50mm 98.1% · 1.00mm 94.6% · 2.00mm 81.4% · 3.00mm 68.3%
    #     (5mm 이상은 워크스페이스 박스 포화 100% 라 측정 오염 — 무릎은 2mm 부근)
    #   구 0.01(10mm/스텝)은 달성률 **28.2%** — 액션의 71.8% 가 버려지고 있었다.
    #   ★상한은 fabric 자체가 정한다(속도 피드포워드 on/off 와 무관하게 동일: 2.050 vs
    #     2.048mm) — PD 수정으로는 안 없어진다.
    #   에피소드 예산 검산: 479스텝 × 2mm = 958mm ≫ 홈→컵 파지중심 208mm.
    #   ★arm_rot_scale 은 **재지 않았다** — 회전 상한은 미측정이므로 그대로 둔다.
    # ★★08.25 DEXTRAH 동일 제어로 전환하면서 **둘 다 미사용**이 됐다. 팔 목표는 이제
    #   델타 누산이 아니라 워크스페이스 박스 안의 **절대 pose** 다(`_pre_physics_step`).
    #   실측 근거 — 누산 방식에서 액션 1.0 의 달성률:
    #     위치 arm_pos_scale 0.01 → 28.2%(palm_err 151mm) · 0.002 → 81.4%(18mm)
    #     회전 arm_rot_scale 0.05 → rz 25.3% / ry 57.0% / rx 27.7%, euler 오차 최대 84°
    #   즉 어떤 스케일을 골라도 "지령 속도 vs 팔 능력"을 사람이 맞춰야 했고, 그 상한은
    #   자세·방향마다 다르다. 절대 매핑에는 그 튜닝 자체가 없다 — fabric attractor 가
    #   속도를 정한다. 삭제하지 않고 남기는 것은 다른 hand_control 경로·구 체크포인트
    #   해석에 참조가 남아 있어서다.
    arm_pos_scale: float = 0.002   # 미사용(절대 매핑)
    arm_rot_scale: float = 0.05    # 미사용(절대 매핑)
    # 손: relative joint position (dexsuite RelativeJointPositionAction scale=0.1 동일)
    hand_joint_scale: float = 0.1

    # ---- 보상 (dexsuite lift 가중치 그대로) ---------------------------------------
    # ---- 접근 기하 (v1/v2 이식, 08.22 인벨롭 재설계) --------------------------------
    # exp(−s·(d_palm + d_side)). 그룹-min reaching(rim-hook 의 원인)을 대체.
    # sharpness 8.0 = 합거리 12.5cm 에서 e⁻¹ (v1/v2 검증치).
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    # 파지중심 z 오프셋(물체 원점 기준)·대향점 반경 — P-A probe 실측으로 확정
    grasp_z_offset: float = 0.0
    side_radius: float = 0.03
    # 감쌈 직접 보상: envelope_frac = 0.5·(중간마디 접촉비율 + 원위마디 접촉비율).
    # ★2.0 초과 금지 — goal 계열(13.5)의 지배가 깨지면 "테이블 위 감싸고 정지" 국소최적
    #   (reward-audit Check 1).
    envelope_weight: float = 2.0
    contact_weight: float = 0.5
    contact_force_threshold: float = 1.0     # [N] dexsuite 동일 (구 보상·자매 트랙 공유)
    # ★감쌈을 **손바닥 접촉만** 인정한다(08.23, reward-audit ACCEPT).
    #   근거 실측(lstm_test3 ep5000): middle_4 가 접촉 시간의 100% 를 손등으로 접촉했고
    #   envelope_frac 은 그걸 감쌈으로 세어 0.746 을 보고했다. 손바닥만 세면 ≈0.55 로
    #   성공 임계 0.6 미달 — 지표가 실패를 성공으로 통과시키고 있었다. 실제로 제대로
    #   감싸는 건 엄지+검지 둘뿐(약지 0.20·새끼 0)이라 5지 인벨롭이 아니라 2지 핀치다.
    #   손등 파지는 force-closure 가 아니라서 pour 의 손목 회전에서 그대로 빠진다.
    # ★★이 값을 켠 채 **구 체크포인트를 이어받지 말 것** — env_frac 이 떨어지면
    #   g_eff(goal 계열 전체의 곱)가 같이 떨어져 보상이 통째로 축소된다. fresh 학습만.
    #   구 판정과의 비교선은 task/envelope_frac_raw 로 계속 로깅된다.
    #   False = 구 판정(크기만). 프로필에 palmar_axis_local 이 없으면 부팅이 죽는다.
    require_palmar_contact: bool = True

    # ---- 손을 Fabrics 가 소유한다 (08.23 자매 트랙 grasp_lift_fabric 검증분 이식) ----
    # "pd"  = 구 배선. 손은 Fabrics 밖(직접 관절 PD). fabric 은 FK 로 손을 보기만 한다.
    # "fabric" = hand_mode="direct"(20x20 항등)로 손 20-DOF 를 fabric attractor 가 제어.
    #   액션 차원·의미는 그대로다(관절 절대목표) — 달라지는 건 **중재자**뿐.
    # ★"tip"(손끝 IK)은 자매 트랙에서 **실측 기각**됐다: 다섯 손끝은 한 손에 결합돼
    #   있는데 정책이 15D 를 독립 지시하면 대부분 기구학적으로 불가능해 fabric 이
    #   모순된 목표를 절충하다 아무데도 못 간다(추종오차 학습 중 85mm,
    #   23,400 스텝 동안 contact_gate 0.000). 게인 문제가 아니라 목표가 불가능한 것.
    #   → 이 트랙은 tip 을 쓰지 않는다. 08.23 손끝 attractor probe 의 2.94mm 는
    #     "손끝 다섯을 **일관되게** 안으로 당긴" 목표라 정책 액션 분포와 다르다.
    # "tip_cyl"(08.24, 기본) — 손가락별 손끝 IK + 원통 액션. 근거 실측 3건:
    #   ①단일 15D 기각(08.23)의 원인이 "기구학적 불가능"이 아니라 taskmap 의 팔 열
    #     오염이었음이 갈렸다(Subchain 수리 후 임의목표 85mm→15.9mm).
    #   ②손가락별 분리(tip_per_finger)로 13.8mm — 겹침(불가능) 목표에서도 엄지만
    #     32.8mm 를 지고 검지·중지·약지는 2.6~4.6mm(간섭이 metric 층에서 사라짐).
    #   ③관절공간 fabric(direct)은 lstm_test4 에서 기각 — attractor 가 명령을 보존하지
    #     않고(잠금 외전 0→−0.234, ring_3 −0.318→−0.750) hand_repulsion(18mm 벽)이
    #     감쌈 자체를 금지해 2지 파지로 붕괴(사용자 판정: "fabric 과 관절공간 제어는
    #     성립할 수 없다").
    #   ④★"synergy"(08.25 신설, 기본값): 관절공간 시너지 그립. tip_cyl 이 파워그립을
    #     **만들 수 없음**이 실측으로 확정돼 도입했다 — r 을 86→14mm 로 전 범위 훑어도
    #     검지 [외전,MCP,PIP,DIP] 가 [0.01,−0.01,−0.07,−0.01] 로 파워그립 기준
    #     [0,1.90,1.80,1.80] 의 1.5% 에 그쳤고(자유공간 실측), 원위마디 접촉이 오프셋×반경
    #     20 조합 중 17 조합에서 정확히 0.00 이었다. 손끝 위치를 4관절로 푸는 IK 는
    #     자유도가 하나 남고, 그 잉여가 "펴진 채 안쪽을 가리키는" 해로 풀린다.
    #     synergy 는 **관절 목표를 직접 보간**하므로 말아 쥐는 것이 구조적으로 보장된다.
    #     팔은 그대로 fabric 이 몰고, 손만 fabric 밖으로 나간다(자매 트랙 "pd" 와 동형).
    hand_control: str = "synergy"
    # tip_cyl 액션 스케일 — 단위=핑거팁 지름(STL 실측 16.1x19.6mm→18mm).
    tip_diameter: float = 0.018
    # r 중심 = 컵 최대반경(45mm)+팁반경(9mm) 대역의 중심. a_r=±1 이 ±2팁지름이라
    # r ∈ [14, 86]mm — 파지 요구 45~54mm 와 완전 개방을 모두 덮는다(워크스페이스 실측
    # r 5~95% = 9~121mm 안쪽).
    tip_r_center: float = 0.050
    tip_action_span: float = 2.0     # a=±1 → ±(span × tip_diameter)

    # ---- synergy 그립 상수 (grasp_v1 검증값) --------------------------------------
    # ★액션은 "속도"가 아니라 **절대 폐쇄도 목표**[0,1]이고, 이 값은 그 목표를 향한
    #   **변화율 상한**이다. 속도 명령(advance = speed×cmd ≥ 0)으로 두면 단조 증가만
    #   가능해 탐색 노이즈 평균(cmd≈0.5)만으로 80스텝에 완전 폐쇄에 도달하고 되돌릴 수
    #   없다 — 정책이 "얼마나 닫을지"를 표현하지 못하게 된다(grasp_v1 실증).
    # ★★08.25 실측 재조정 0.05 → 0.005. 0.05 는 **스윕에서 가장 나쁜 값**이었다.
    #   "연 채로 접근 완료 후 서서히 닫기" 스윕(속도 피드포워드 복구 후, wrap4):
    #     0.050(15스텝) 0.45 · 0.020(37) 0.47 · 0.010(75) 0.58 ·
    #     **0.005(150) 0.64** · 0.002(375) 0.80
    #   단조적으로 느릴수록 좋다. ★grasp_v1 원본값이 0.05 지만 **실측값을 택한다**
    #   (사용자 결정) — 원본 0.05 가 이 로봇·이 컵 기하에서 가장 나쁜 값이었다.
    #   0.002 가 최선이지만 완전 폐쇄에 375스텝이 들어 에피소드(600스텝)에서 접근·
    #   리프트·이송 예산이 빠듯하다. 0.005 는 150스텝으로 예산을 남기면서
    #   0.05 대비 감쌈을 0.45→0.64 로 올린다.
    #   ★이건 **상한**이지 강제 램프가 아니다 — 접촉 동결이 마디별로 먼저 멈추므로
    #     정책은 자유공간에서 빨리, 접촉 근처에서 천천히 닫는 것을 배울 수 있다.
    synergy_close_speed: float = 0.005
    # 접촉 시 관절 동결을 켤 것인가. ★이것이 감쌈 생성 메커니즘이다 — 접촉한 마디가
    #   그 자리에 멈춰 컵 형상에 손가락이 드리워진다. 끄면 손가락이 컵 반경보다 작게
    #   말려 손끝만 닿는 핀치가 된다(grasp_v1 실증: full_envelope 0.176→0.035,
    #   five_tip 동시접촉 0.42→0.68 = 접촉이 마디에서 손끝으로 이동).
    synergy_contact_freeze: bool = True
    # 검지~소지를 채널별 평균으로 묶어 "특정 손가락만 안 닫힘"을 표현 불가하게 한다.
    #   3지(또는 1지) 국소최적을 액션 공간에서 원천 차단. 엄지는 대향을 위해 독립.
    #   ★lstm_test8 이 정확히 이 실패였다(검지만 wrap 0.73, 나머지 0.00).
    couple_four_fingers: bool = True
    # direct 모드 attractor 게인. 자매 트랙 스윕 채택값(이동량 부족 26%→8%).
    hand_attractor_gain: float | None = 400.0
    # ★손가락↔손가락 반발을 Fabrics **계획 단계**에서 건다. PhysX self-collision 을
    #   끄기 위한 전제다(자기충돌은 스텝 시간의 55~64%).
    #   자매 트랙 검증: repulsion ON 이면 fabric_q 의 다른 손가락 구 최소거리 20.1mm ·
    #   18mm 미만 0.0% (OFF 는 계획에 관통 해가 남는다) · palm 추종오차 0.7mm 로 무결.
    #   계획에 관통 해가 없으므로 정책이 관통으로 이득 보는 전략을 학습할 수 없다.
    use_hand_repulsion: bool = False
    # ★★fabric body 반발 쌍을 실제로 걸지 여부(08.25 신설). Kuka 는 13쌍을 건다 —
    #   `palm_link` + 4지 × 링크 1/2/3, 전부 ↔ `iiwa7_link_2`(팔뚝). 손가락↔손가락은
    #   **한 쌍도 없다**. 우리 params 도 같은 패턴으로 교체했고 5지라 16쌍이 된다.
    #   ★공유 fabric 클래스의 기본값은 False 라 다른 트랙 거동은 불변이다.
    use_body_repulsion_pairs: bool = True
    tracking_weight: float = 2.0
    tracking_std: float = 0.1
    success_weight: float = 10.0
    success_std: float = 0.05
    # 리프트 부분 진척 — gate 곱. goal_height_offset(0.15m)에서 포화.
    # dexsuite 는 이 항을 뺐지만 그 전제(goal=물체 근처 랜덤 pose)가 우리와 다르다.
    lift_weight: float = 1.5
    # 직립 **양수** 보상(08.23 사용자 지시: "패널티 말고 양수의 보상으로").
    # (물체 local +z · world +z)^k × 리프트진척 × 유효게이트. 근거는 rewards.upright_reward.
    # ★k=4 는 cos 의 소각 평탄성을 보정해 15~30° 대 판별력을 만든다(도당 기울기가
    #   tilt_penalty 의 약 12 배). ★리프트진척 곱이 없으면 컵이 스폰부터 서 있으므로
    #   "테이블 위 컵 건드리고 정지"가 공짜 수확이 된다.
    upright_weight: float = 3.0
    upright_exponent: float = 4.0
    # 전도 페널티 — 20° 여유대(정상 파지 흔들림 무징계) 초과분 비례, 최대 −0.5.
    # 60° 초과 = 사실상 넘어짐 → **truncation 으로 env 전체 리셋**(08.22, 사용자 지시).
    # 컵 단독 리스폰은 폐기 — 텔레포트 전이가 학습 데이터를 오염시키고, 손 위로
    # 겹쳐 소환되는 결함이 있었다. value_bootstrap(yaml)과 반드시 짝이어야 한다:
    # bootstrap 없는 truncation 은 termination 과 같아져 회피 학습(agn_test2)이 재발한다.
    tilt_penalty_weight: float = -0.5
    tilt_free_deg: float = 20.0
    tilt_reset_deg: float = 60.0
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    # 관절한계 위반 종료 페널티 (diff IK 는 관절한계 무방비 → 종료+페널티로 처리)
    abnormal_penalty: float = -1.0

    # ---- 스폰 높이: 단일 소스 -------------------------------------------------------
    # ★이중 패딩 재발 차단(08.21). 프로필이 완성값(0.282)을 들고 env 가 +5mm 를 또 얹어
    #   컵이 정착고보다 9.7mm 높이 스폰됐다 → 정지 상태 height_delta −9.7mm, lift 보상의
    #   첫 9.7mm 데드존, 실효 목표 159.7mm. 게다가 두 프로필이 0.282/0.297 로 갈렸다.
    #   이제 여기 세 값에서만 파생한다(프로필의 object_spawn_z 필드는 삭제됨).
    table_surface_z: float = 0.200           # env.usd top_plate 상면(점군 실측)
    object_origin_offset_z: float = 0.0773   # cup_big USD 원점 ↔ 바닥
    object_spawn_pad: float = 0.005          # 스폰 침투 반동 방지
    # 위 셋에서 __post_init__ 이 파생시키는 캐시. 직접 쓰지 말 것(단일 소스 유지).
    object_spawn_z: float = 0.0

    # ---- goal / 성공 판정 ---------------------------------------------------------
    goal_height_offset: float = 0.15         # goal = 물체 스폰 위치 + z 0.15
    # dexsuite 규약은 **success 항의 pos_std/2**(dexsuite_env_cfg.py:432). 우리 success_std 가
    # 0.05 이므로 0.025 다. 구 값 0.05 는 참조를 tracking_std/2 로 잘못 적은 것이라 판정선이
    # 보상선보다 5배 헐거웠다 — 정책이 못 하는 난이도로 계속 승급했다.
    success_pos_tolerance: float = 0.025
    # 이전 12,000ep 런과의 연속성 비교 전용(보상·커리큘럼 미사용, 로깅만)
    success_pos_tolerance_loose: float = 0.05
    # ★성공 판정 3조건(08.22, 사용자 결정): goal 근접 AND envelope_frac AND 직립.
    #   "인벨롭으로 세워 든 것"만 성공 — 다음 태스크(pour)의 전제. 커리큘럼 승급도 이 기준.
    # 0.6 = 테솔로(env 4지)는 3지 이상 감쌈, 2지 그리퍼는 양 jaw(1.0). 0.5 는 두
    # 손가락 rim-hook(0.5)이 통과해 버린다 — P-B probe 반증.
    success_envelope_min: float = 0.6
    # ★g_eff 의 gradient 포화점 — 성공 임계(0.6)와 **별도 상수**(08.24 R2).
    #   같은 상수를 쓰면 임계 위에서 감쌈 gradient 가 0 이라 정책이 3지에서 멈춘다
    #   (lstm_test3: env 0.65 에 2,500 에폭 고착 = 정확히 그 포화점). 판정은 0.6 그대로,
    #   goal 계열의 감쌈 유인만 0.85 까지 연장. 4지 감쌈 달성 가능은 tip_cyl probe 실증.
    envelope_gate_saturation: float = 0.85

    # ---- tip_cyl 전용 보상 (08.24 총 재설계 — compute_tip_cyl_rewards) ------------
    # 5항+정규화·게이트 1개. 값은 레퍼런스 원값(DEXTRAH sharpness 10/8.5·dexsuite
    # σ·가중치)에서 approach sharpness 만 8 로(완전 파지 시 d_max≈컵반경+α≈7cm 라
    # 10 이면 만점 지점이 0.5 로 깎여 후반 gradient 가 얕다).
    # 위 approach_*~tilt_penalty_* 상수들은 구 보상(공유 함수) 전용 — pd/fabric 모드와
    # 타 트랙(grasp_lift_fabric) 경로에서만 읽힌다.
    # ---- tip_cyl 3차 재설계(08.25): **소프트 계층** — 하드 스위치 없음 -------------
    # 2차안(6단계 + 이진 대향 게이트)이 lstm_test7 에서 실패한 원인 3건(실측):
    #  ①`upright` 가 독립 가산항이라 **테이블에 선 컵이 만점**(2.674 = 총보상의 48%).
    #    들면 흔들려 그걸 잃으므로 **보상이 "들지 마라"를 가르쳤다** — ep757 h=2.5mm
    #    (최고) → ep1324 h=0.3mm 로 실제로 되돌아갔다.
    #  ②`lift = exp(−8.5·|goal_z−obj_z|)` 가 **h=0 에서 0.28 지급**(실측 0.954).
    #  ③게이트가 손가락 **총접촉**(팁 포함)이라 2지 팁 핀치로 열린다. 실측 수렴 자세는
    #    파지중심에서 106mm 밖의 컵을 엄지·검지 팁으로 집은 것이고, 중지·약지·소지는
    #    1,594 에폭 전 구간 `touch` 자체가 0.000 이었다(컵이 손 안에 들어온 적 없음).
    #
    # ★래치(`pre_lift_gate`)를 쓰지 않는다. 이 저장소가 두 번 제거한 장치다 —
    #   grasp_v2 "순수 grasp_v1 이식 3,271ep, latch 이후 항 전 구간 정확히 0 → 제거",
    #   grasp-sensor lstm_test3 "latch=순손실, 엄지 접촉만 포기하면 영구 차단 → 제거".
    #   레퍼런스 grasp_v1 의 98% 는 보상이 아니라 **스크립트 리프트**(래치 후 정책 palm
    #   액션을 폐기하고 z 를 120스텝에 +10cm 램프)의 성과이고, 우리는 팔이 100% 정책
    #   제어라 전제가 다르다. 절벽 산수: 손익분기 높이 0.197m > 목표 0.15m.
    #
    # 계층은 **소프트 인자의 곱셈 깊이**로 만든다(전 인자 [0,1] 연속 → 불연속 없음):
    #   ③lift 3개 곱 · ④transport 4개 · ⑤stabilize 5개.
    # 상한 사다리: 접근2 → +파지6=8 → +리프트12=20 → +이송8=28 → +정지4=32 → +성공20=52.
    # ★tip_cyl 전용 접촉 임계 — 0.1N (사용자 결정, grasp_v1 검증값). 위
    #   `contact_force_threshold`(1.0)는 구 보상·**자매 트랙 계약**이 공유하므로 못 바꾼다.
    #   grasp_v1 근거: "손끝 접촉력 p95=7.77 / max=10.37N, 비영 접촉 하위 5분위 =
    #   1.86N ≫ 0.1 → 접촉 판정이 놓치는 구간 없음". 1.0 은 그 5분위의 절반이라 얕은
    #   마디 접촉을 잘라낼 수 있다 — lstm_test7 의 "중지·약지·소지 wrap 정확히 0.000"이
    #   진짜 미접촉인지 임계 미달인지 P-1b probe 로 갈린다.
    stage_contact_threshold: float = 0.1
    # ★tip_cyl 전용 성공 감쌈 임계 — 분모가 **4지(엄지 제외)** 라 0.75 = 3지 이상.
    #   엄지는 대향이라 mid/dist 감쌈이 구조적으로 불가하다(e2e probe: 이상적 파지에서도
    #   엄지 wrap 0.00). 위 `success_envelope_min`(0.6)은 자매 트랙 공유라 불변.
    stage_success_envelope_min: float = 0.75
    # ★파지중심을 5점평균에서 대향중점 쪽으로 옮기는 비율. 0 = 5점평균(엄지가 영원히
    #   못 닿음, 4지는 관통 31.6N), 1 = 대향중점(4지가 전혀 못 닿음, wrap4=0.00).
    #   0.39 = 오프셋 스윕 실측 최적 28mm ÷ 대향중점까지 72mm.
    #   실측(off, thumb, wrap4, G): (0, 0.00, 1.00, 0.25) (15, 0.31, 1.00, 0.41)
    #   **(28, 1.00, 1.00, 0.83)** (35, 0.94, 0.77, 0.58) (45, 0.56, 0.14, 0.07)
    #   (60, 1.00, 0.00, 0.00). 자산이 바뀌면 이 스윕을 다시 돌려야 한다.
    # ★08.25 재측정(시너지 그립 전환 후): 0.39(28mm) → 0.67(48mm).
    #   손 제어가 tip_cyl(펴진 손) → synergy(말아 쥠)로 바뀌면서 손가락이 모이는 지점이
    #   달라졌다. 오프셋 스윕 실측(close 0.7 고정, 값은 wrap4/deep4/oppose/G):
    #     −40mm 0.09/0.06/0.00/0.022   −20mm 0.48/0.41/0.00/0.117
    #     +0mm  0.59/0.44/0.81/0.483   **+20mm 1.00/0.92/1.00/0.984**
    #     +40mm 1.00/0.69/1.00/0.938
    #   +20mm 에서 **5지 전부 손바닥면 접촉 + 4지 두 마디 동시 + 엄지 대향**이 성립하고
    #   G 가 0.48→0.98 로 2배가 된다. 손 제어를 바꾸면 이 스윕을 다시 돌려야 한다.
    stage_gc_opposition_frac: float = 0.67
    # ★★08.25 파지중심 직접 지정(자유 컵 실측). 위 보간 유도는 컵을 텔레포트로
    #   **붙잡아 놓고** 잰 값이라 무효였다 — 자유 컵에서는 손이 닫히며 컵을 쓸어내려
    #   실제로 무는 지점이 z 로 39mm 더 손바닥 쪽이다. 두 독립 측정 일치:
    #     probe_seqclose(ff=0) [58, 1, 64]mm · probe_seqclose(ff=1) [56, −3, 64]mm
    #   None 이면 위 보간식으로 되돌아간다. 자산·손 제어·폐쇄 속도가 바뀌면 재측정.
    stage_gc_local_override: tuple | None = (0.057, -0.001, 0.064)
    stage_approach_weight: float = 2.0
    stage_approach_sharpness: float = 8.0
    # ★★08.25 5단계 소프트 게이트 재편(사용자 지정):
    #   approach → grasp(5지+palm 밀착) → lift → transfer&stabilize → stay
    # ─ 자세(approach 인자) ─────────────────────────────────────────────────────
    #   palm_ee **+x 가 손바닥 법선**이고 컵 축(+z)과 **수직**이어야 한다(사용자 규약).
    #   실측: 홈 자세에서 palm_ee_x·cup_z = −0.0025 로 이미 만족 = 액션 박스 중심이 정답.
    #   구 `align` 은 접근축이 컵을 겨누는가(1자유도)만 봐서 롤·피치를 안 잡았고,
    #   컵이 손보다 아래라 **숙이면 approach 가 +16%** 오르는 역유인까지 있었다.
    stage_perp_exponent: float = 2.0     # (1 − |cos(palm_x, cup_z)|) ** e
    #   ★법선 수직만으로는 법선 둘레 롤이 남는다(90° 굴리면 손가락이 세로 평면으로
    #     감싸는데 perp 는 1.0 그대로). palm_ee +y ∥ cup_z 가 그 자유도를 잠근다.
    #     과하면 이 값을 0.0 으로 → roll_q 가 항상 1.0 이 되어 항이 꺼진다.
    stage_roll_exponent: float = 4.0
    stage_orient_floor: float = 0.15     # 자세가 최악이어도 approach 의 15% 는 남긴다
    # ─ grasp(②) ───────────────────────────────────────────────────────────────
    #   G = five_frac · exp(−d_gc/τ). **새 센서를 쓰지 않는다** — 실기 센서는 tip 에만
    #   있어 palm 접촉은 배포 불가다. 파지중심은 palm 에 강체로 붙은 점이라 d_gc → 0 이
    #   곧 "물체가 손 깊숙이 = palm 밀착"이고 FK 로만 계산된다.
    #   ★형상 어댑티브: 크기 가정 없음. 엄지 하나 터치는 five_frac 0.2 라 구조적으로 죽는다.
    # ★★08.25 폐기 — `near_q = exp(−d_gc/τ)` 는 `five_frac` 과 정면으로 싸우고 있었다.
    #   lstm_test14 ep1102 정점 실측: five_frac 0.780 인데 deep4 0.155 · full_tip 0.0007,
    #   손가락별 touch 0.63~0.67 vs wrap 0.28~0.48 = **손끝 스침**이었고 near_q 는
    #   그것을 0.124 로 정확히 깎았다. G = 0.10 고착 → grasp·lift·transfer·stay 전부
    #   정격의 10% → 이진 success 가 총보상의 67~79% → 두 런 모두 그 직후 붕괴.
    #   대체 = 접촉 기하 기반 `Q_g`(rewards_tip_cyl.py). d_gc 는 approach 에만 남는다.
    # stage_grasp_near_tau: 삭제
    # ── 파지 품질 Q_g 배합 (합 1.0, 부팅 어서션) ────────────────────────────────
    # ★도달 불가능한 손가락 — 접촉 분모에서 제외한다. 자세표를 고치면 () 로 되돌린다.
    #   근거: pinky `_1`/`_2` 가 hand_open_pose == hand_grip_pose 라 lerp 가 상수다.
    hand_unusable_fingers: tuple[str, ...] = ("pinky",)
    stage_graspq_touch: float = 0.25     # 닿았나
    stage_graspq_deep: float = 0.55      # **두 마디 동시** = 실제 감쌈. 팁 스침으론 불가
    stage_graspq_persist: float = 0.20   # 유지하는가
    stage_graspq_thumb_floor: float = 0.30   # 엄지 없이 4지만 긁으면 상한 30%
    stage_thumb_force_ref: float = 0.5   # [N] 소프트 대향. 접촉임계 0.1N 의 5배,
                                         #      실측 p95 7.77N 의 1/15 = 센서 스케일
    # ─ stay(⑤) ────────────────────────────────────────────────────────────────
    #   S = exp(−|v_obj|/v_ref). 구 S 는 액션 변화량이라 "액션을 안 바꾼다"였지
    #   "안 움직인다"가 아니었다. 물체 실제 선속도로 바꾼다.
    stage_stay_speed_ref: float = 0.05   # 0.05 m/s
    stage_stay_hold_steps: int = 30      # 성공률 로깅용 — 0.5 초 연속 유지
    # ─ 가중 사다리 ────────────────────────────────────────────────────────────
    #   인자가 깊어질수록 곱이 작아지므로 상한을 키운다(lstm_test8: 네 인자 곱이
    #   0.008 로 소멸해 어느 방향으로도 gradient 가 없었다).
    #     ① approach 2.0 · ② grasp 12.0 · ③ lift 12.0 · ④ transfer 16.0 · ⑤ stay 24.0
    stage_transfer_weight: float = 7.0
    stage_stay_weight: float = 10.0
    # ★08.25 5단계 재편으로 죽은 상수 5개 삭제(코드 참조 0 확인):
    #   stage_transport_weight / stage_stabilize_weight / stage_stabilize_sharpness
    #     → ④transfer(16.0) · ⑤stay(24.0) 로 대체
    #   stage_upright_tau_deg  → U 를 각도 exp 가 아니라 **축 정렬 cos** 로 직접 계산
    #   stage_lift_envelope_mix → G 가 five_frac·near_q 로 바뀌어 혼합비가 사라짐
    # ★`stage_open_penalty` 삭제 — 게이트가 대신한다. G = five_frac·near_q 라 닿지도
    #   않고 쥔 주먹은 G ≈ 0 이고, 주먹이 컵 진입을 막으면 d_gc 가 안 줄어 approach 도
    #   같이 떨어진다. 무엇보다 그 벌점이 **엄지 단독 터치 전략을 만든 장본인**이었다
    #   (contact_frac 이 손가락 하나만 닿아도 20% 면제 → 엄지가 통로를 막음).
    # 정렬 배수의 바닥 — align=−1(손등 쪽)일 때 남기는 비율. 0 이면 초기 오정렬에서
    # approach gradient 가 사라져 접근 자체를 못 배운다(reward-audit Check1).
    stage_align_floor: float = 0.25
    # ②grasp = w·(reach 몫 + G 몫). 두 몫의 합이 1 이라 재정규화된다.
    # ★★08.25 grasp_v1 구조로 전면 교체(사용자 지시 "grasp-v1과 동일 구조로").
    #   구 구성 `6·(0.4·reach + 0.6·G)` 를 폐기한다. `reach` 는 **거리 항**이었고
    #   (mid/dist 링크 → 물체), 컵이 손 밖에 있으면 **손가락을 펼수록 커진다**.
    #   lstm_test9 실측 — 정책이 감쌈을 버리고 보상을 올렸다:
    #     ep550  6·(0.4·0.473 + 0.6·0.025) = 1.225   wrap4 0.125
    #     ep750  6·(0.4·0.534 + 0.6·0.008) = 1.310   wrap4 0.042  ← 감쌈 −66%, 보상 +0.09
    #   폐쇄도 스윕(d_gc 60mm, 정책 운전점): wrap4 0.50 → R_grasp 1.157,
    #   wrap4 0.00 → 1.279. **감쌈을 버리는 쪽이 이득인 지형이었다.**
    #   grasp_v1 은 grasp_quality 네 항이 **전부 접촉**이라 이 계곡이 구조적으로 없다.
    #   거리 shaping : 접촉 보상 비율 — grasp_v1 1:6, 구 우리 것 1.22:1 (7배 어긋남).
    stage_contact_weight: float = 1.0    # 게이트 없음 — λ=1·μ=0 사각지대 방지 shaping
    stage_grasp_weight: float = 3.0          # grasp_v1 grasp_weight
    # grasp_quality = 0.15s·tip + 0.20s·full_tip + 0.25s·persist + credit·deep4
    #   s = (1 − credit)/0.60 로 합이 1 로 재정규화된다(credit 을 올려도 최대치 불변 →
    #   "감쌈만 하고 안 드는" 국소최적을 구조적으로 못 만든다 — grasp_v1 reward-audit).
    stage_grasp_envelope_credit: float = 0.55  # grasp_v1 grasp_envelope_credit
    # 접촉 지속 — 접촉 손가락 수가 임계 이상인 스텝을 세고 이 스텝수로 정규화.
    stage_contact_persistence_steps: int = 20  # grasp_v1 grasp_contact_persistence_reward_steps
    stage_persistence_min_contacts: int = 4    # grasp_v1 stage0_lift_start_min_contacts
    # 리프트 계열 접촉 게이팅 — grasp_v1 graded_contact.
    #   Q_lift = (1−mix)·tip_frac + mix·envelope_frac,  envelope_frac = 0.5(wrap4 + deep4)
    #   ★항등식: mean(mid)+mean(dist) = mean(mid∨dist)+mean(mid∧dist) 이므로
    #     grasp_v1 의 0.5(mid_frac+dist_frac) 가 우리 0.5(wrap4+deep4) 와 정확히 같다.
    # ★엄지 바닥값(구 stage_gq_thumb_floor 0.25)은 **삭제**한다. 그 배수가 첫 접촉을
    #   순손실로 만들었다(reach 소등 −0.272 vs G 상승 +0.18). grasp_v1 처럼 엄지는
    #   `tip_frac`(5팁) 안에서 자연히 계상된다. 성공 판정은 `oppose` 를 그대로 요구한다.
    # ★컵 밀기·기울임 벌점(grasp_v1 approach 항). 우리에겐 없었고, lstm_test9 는
    #   컵을 평균 ~50mm 밀면서도 벌점을 한 푼도 안 물었다.
    # ★★08.25 lstm_test12 실측으로 25.0 → 8.0. 25.0 은 **접근 보상을 통째로 삼켰다**:
    #   ep300 에 xy_disp 0.050 → 벌점 25.0·(0.050−0.025) = 0.625 인데 그 시점 approach
    #   양의 항은 ~0.40 이라 `reward/approach = −0.714`. 음수가 되면 `orient_q` 가
    #   **양의 항에만 곱해지므로** 자세 gradient 가 소멸한다 — 실제로 perp_q 가
    #   0.929(ep100) → 0.425(ep300) 로 무너졌고, 손이 기울자 컵도 기울어(12.2°)
    #   높이가 0.143 → 0.036 으로 붕괴했다. 파지는 접촉을 요구하고 접촉은 컵을 반드시
    #   미세하게 민다 — 25.0 은 사실상 "닿지 마라"였다.
    #   8.0 이면 xy_disp 0.050 에서 벌점 0.20 = 양의 항의 절반이라 억제는 남는다.
    stage_approach_xy_penalty: float = 8.0     # grasp_v1 approach_xy_penalty_weight
    stage_approach_xy_margin: float = 0.025    # grasp_v1 grasp_xy_threshold
    stage_approach_tilt_penalty: float = 0.08  # grasp_v1 approach_tilt_penalty_weight
    stage_approach_tilt_margin_deg: float = 8.0  # grasp_v1 grasp_upright_threshold_deg
    stage_lift_weight: float = 5.0
    # ★목표(goal_height_offset=0.15)와 **정렬**한다. 포화점을 목표보다 낮게 두면 그
    #   위에서 gradient 가 0 이라 정책이 포화점에 고착한다(lstm_test3: env 0.65 에
    #   2,500 에폭 고착 = 정확히 그 포화점. grasp_v1: 4cm 포화 → 평형 3.1cm).
    stage_lift_height_ref: float = 0.15
    stage_tracking_std: float = 0.1
    # 직립은 **독립 항이 아니라 곱셈 인자**. 테이블 위 컵에 지급되지 않는다.
    # ★★08.25 lstm_test12 실측으로 `U = cos(tilt)` 를 **폐기**한다. cos 는 작은 각에서
    #   평평해 판별력이 없다: 실측 U 가 전 구간 0.996~0.952 상수였고, 컵이 15.98° 로
    #   누웠을 때도 0.952 였다(손해 4%). 기울임을 막는 인자가 사실상 없었던 것.
    #   → 가우시안 exp(−(tilt°/τ)²), τ=10°:  3° 0.914 · 7.5° 0.570 · 15° 0.105 · 20° 0.018
    #     15° 에서 0.1 로 떨어져 `stage_success_tilt_deg=15.0` 과 자연히 정렬된다.
    #   ★acos 없이 `exp(−2(1−cos)/τ_rad²)` 로 계산한다 — 1−cos ≈ θ²/2 근사라 20° 까지
    #     오차 1% 미만이고, acos 의 cos=±1 미분 발산을 피한다.
    # ★★08.26 사용자 규격(학습 영상). 직립을 **단계별로 다르게** 요구한다:
    #   "이송 중에 20도 내로 기울여져 있는 상태는 괜찮음. 오히려 목표 좌표 5cm 내로
    #    오면 가만히 있되 컵을 똑바로 world +z 와 컵 +z 가 같은 곳을 보게 하고 정지."
    #   구 (25,10) 은 리프트·이송 내내 10° 직립을 강요하면서 정작 stay 에는 직립
    #   인자가 없었다 — 요구가 뒤집혀 있었다.
    stage_tilt_tolerance_deg: tuple[float, float] = (30.0, 20.0)  # lift/transfer 관용
    stage_upright_gate_deg: tuple[float, float] = (15.0, 5.0)     # stay/success 직립
    # 컵 밀림 감쇠 — 제곱역수. 선형 (1−d/L) 은 d≥L 에서 정확히 0 이 되어 하드 게이트와
    # 같아지고 gradient 가 소실된다(실측: 밀림 0.207 이 300 에폭간 전혀 안 줄어듦).
    stage_disp_limit: float = 0.06
    # ── 계층 게이트 λ→μ→ν→ρ (DexPour, IROS 2025 식 3~6) ────────────────────────
    # 논문값(d_approach 0.1 · c_finger 4 · h_lift 0.15 · d_pour 0.17)을 우리 실측
    # 스케일로 옮긴 것이다. 논문은 컵을 0.5m 들지만 우리 목표는 스폰+0.15m 다.
    stage_gate_approach_m: float = 0.12   # d_gc 중앙값 130mm · 최선 83mm → 운전권에서 열림
    stage_gate_contact_n: float = 3.0     # 가용 4지 중 3지. 논문은 4지 전부
    stage_gate_lift_m: float = 0.05       # 목표 0.15 의 1/3 에서 이송이 열린다
    stage_gate_transfer_m: float = 0.08   # d_goal 시작 0.15 의 절반
    # ── 성공 — 이진 5중 AND 폐기, 연속 곱. 전이 구간은 실측 분포가 걸친 곳 ────────
    stage_success_weight: float = 6.0
    stage_succ_height_band: tuple[float, float] = (0.04, 0.12)   # 실측 이봉 0.061/0.12 사이
    stage_succ_graspq_band: tuple[float, float] = (0.35, 0.70)
    stage_succ_tilt_band_deg: tuple[float, float] = (18.0, 6.0)   # 직립 요구(내려가는 전이)
    # ★구 success 에는 **속도 조건이 아예 없었다** — 목표를 스쳐 지나가도 성공으로
    #   셀 수 있었다. 사용자 규격이 "가만히 있되"를 명시한다.
    stage_succ_speed_band: tuple[float, float] = (0.10, 0.03)     # 정지(내려가는 전이)
    # ★stay 단계 판정(로깅) 전용. `success_pos_tolerance`/`success_tilt_max_deg` 는
    #   grasp_lift_fabric 과 **동기 계약**이라(그쪽 test_task_contract 가 검사) 건드리지
    #   않는다. 사용자 규격 "목표 5cm 내에서 정지 + 직립"은 여기로 낸다.
    stage_stay_pos_tol_m: float = 0.05
    stage_stay_tilt_deg: float = 10.0
    stage_succ_goal_band_m: tuple[float, float] = (0.09, 0.05)    # 사용자 규격 5cm
    # ── 코리더 래치 (08.26 사용자 승인: "래치 + 느슨한 시작(20cm/50°)") ─────────────
    # probe_lift_trajectory 실측(ep10400 결정론): 낚아챔 = xy 정점 253mm·tilt 49°·
    # 수평속도 1074mm/s, 60스텝 통행료 ~7% 로 순간 게이트를 우회했다. 래치는 그
    # 허점을 닫는다 — 에피소드 중 **한 번이라도** 코리더(스폰 기준 xy 이탈·기울기)를
    # 넘으면 그 에피소드의 ν 이후(lift·transfer·stay·success)를 몰수한다.
    # (initial, final) — per-env 난이도 0→만렙으로 선형 보간. ADR 확장과 같은 축이라
    # "난이도가 오르면 보상 요구도 조여진다"(사용자 설계 방향).
    # 시작값 근거: 20cm/50° 는 현 낚아챔(25cm/49°)은 걸리고 정상 정착(3~4cm/6°)은
    # 여유 5배. 최종값 근거: xy 5cm = stage_succ_goal_band 하한과 정렬, tilt 20° =
    # stage_tilt_tolerance_deg 포화점과 정렬.
    stage_corridor_xy_m: tuple[float, float] = (0.20, 0.05)
    stage_corridor_tilt_deg: tuple[float, float] = (50.0, 20.0)
    # ── palm 지령 rate limit (08.26 계획서 승인 — 제어층 대책, 보상과 역할 분리) ────
    # 절대 매핑은 한 스텝에 박스 대각 983mm 를 점프할 수 있고 σ=1 탐색 지터가
    # 축별 96~212mm/step 이다(계획서 §1). 리미터는 지령 변화율만 묶는다 — 절대
    # 규약(목표=박스 안 절대 좌표)은 유지되고 누산식 목표 인플레도 없다.
    # ★0.0 = 비활성(기본). corridor_test1 등 기존 런은 무영향. 값은 probe A/B
    #   ({0, 0.10, 0.05} 재생 대조) 후 다음 fresh 런에서 켠다. 후보 0.05/0.10 은
    #   사용자 지정(좌팔 0.02 는 이 트랙 기준 과도).
    palm_cmd_rate_limit_m: float = 0.0
    # 회전 지령 rate limit [deg/step]. probe A/B 실측 — 회전 지령도 평균 84°/step
    # 텔레포트(위치만 묶으면 정책이 회전으로 우회할 통로가 남는다). 0.0 = 비활성.
    # 값 비례 논리: 위치 0.1 ≈ 박스 대각의 10%/step → 회전 등가 ≈ 15°/step.
    palm_cmd_rate_limit_rot_deg: float = 0.0
    # ── close_bridge (08.26) — "가까이서 조이기 시작" 구간의 gradient 공백 다리 ────
    # B(corridor_lim01) 실측: 62~74mm 접근 상태 500ep 동안 폐쇄 미발생, ep778 폐쇄
    # 시도 시 R 25분의 1 → 회귀. 접근 보상은 손가락 상태 무관·폐쇄 보상은 접촉부터라
    # "근접+폐쇄 시작"만 무보상 지대다. r = w·λ·syn_close(가용 평균):
    #   멀면 λ=0 → 펴고 접근 유지(현행 행동 불변) · 가까우면 조임에 소액 지급 ·
    #   접촉하면 contact/grasp 가 덮는다(★접촉 시 끄지 않음 — grip-contact-cliff 함정).
    # 0.0 = 비활성(기본). B ep1500 판정 후 오버라이드로만 켠다. reward-audit ACCEPT.
    stage_close_bridge_weight: float = 0.0

    dex_approach_weight: float = 2.0
    dex_approach_sharpness: float = 8.0
    dex_tracking_weight: float = 2.0
    dex_tracking_std: float = 0.1
    dex_upright_weight: float = 3.0
    dex_upright_std: float = 0.35        # [rad] ≈ 20° — 성공 임계와 같은 스케일
    dex_lift_weight: float = 1.0
    dex_lift_sharpness: float = 8.5
    dex_success_weight: float = 10.0
    dex_success_pos_std: float = 0.05
    dex_success_rot_std: float = 0.35
    success_tilt_max_deg: float = 20.0

    # ---- 커리큘럼 (per-env 난이도 0~10) --------------------------------------------
    curriculum_max_level: int = 10
    # ★08.25 `gravity_min_frac` 삭제 — 참조 0건인 죽은 상수였다. 08.22 에 물체 반중력
    #   보상력을 없앴는데(만중력 고정) 상수와 주석만 남아, 읽는 사람에게 "커리큘럼이
    #   유효 중력을 스케일한다"고 잘못 알려주고 있었다. 현재 중력 배선은 Kuka 와 같다:
    #   씬 중력 기본 (0,0,−9.81) · 로봇 `disable_gravity=True` · 물체 `False` · 보상력 없음.
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
            # ★★08.25 물체 물리도 DEXTRAH Kuka 값으로 전환
            #   (`dextrah_kuka_allegro_env.py:553` object_cfg.rigid_props).
            #   구 값(solver 16/1 · max_vel 100 · depenetration 1.0)은 "빠지면 PhysX
            #   기본 4회라 파지 조임 중 손끝이 컵 벽을 파고든다(사용자 영상 08.20)"는
            #   근거로 우리가 고른 것이다. Kuka 는 8/0 · 1000 · 1000 을 쓴다.
            #   ★로봇 쪽 `max_depenetration_velocity` 와 함께 되돌릴 후보다 —
            #     접촉력 스파이크가 보이면 여기부터 본다.
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

    # 물체 rigid body 는 USD 안 baseLink 에 있다(cup_big_rl.usd 규약)
    object_contact_filter: tuple = ("/World/envs/env_.*/Object/baseLink",)

    # 물리 재질 이벤트(로봇·컵). 테이블은 scene 자산이 아니라 정적 프림이라
    # env 가 clone 전에 bind_physics_material 로 직접 건다.
    events: GraspSensorEventCfg = GraspSensorEventCfg()
    surface_friction: float = _FRICTION

    robot_cfg: ArticulationCfg = None  # __post_init__ 에서 프로필로 조립

    def __post_init__(self):
        profile = PROFILES[self.profile_name]
        self.robot_cfg = _build_robot_cfg(
            profile, bool(self.enable_self_collisions))
        num_joints = profile.num_arm_joints + profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D delta + 손:
        #   pd/fabric  : **자유** 손관절 delta(고정 관절은 정책이 안 건드린다)
        #   tip_cyl    : 손가락별 (r, z) 절대 2D × 5 = 10D — θ 는 공칭값 고정.
        #     원통 좌표가 감쌈 기하를 좌표계에 넣는다: r=개구도(조임이 스칼라 하나),
        #     θ 고정=두 손가락이 같은 각도에 못 옴(겹침을 구조로 차단),
        #     z=컵의 어느 높이. 단위=핑거팁 지름이라 "얼마나 가까이"가 물리량이다.
        if self.hand_control == "synergy":
            # 손가락 × 채널. 채널 수는 프로필의 채널 대응에서 파생(로봇 무관).
            _nch = len(set(profile.hand_channel_of_joint.values()))
            self.action_space = 6 + _nch * len(profile.finger_sensor_bodies)
        elif self.hand_control == "tip_cyl":
            self.action_space = 6 + 2 * len(profile.finger_sensor_bodies)
        else:
            self.action_space = 6 + profile.num_hand_joints - profile.num_locked_hand_joints
        # 관측 = q + qd + palm pose(7) + tips(3T) + obj pose(7) + goal(3)
        #        + 손가락별 접촉력 크기(F) + last action + **fabric 상태(3×num_joints)**
        # ★★08.25 fabric_q/qd/qdd 신설(DEXTRAH Kuka policy obs 에 있고 우리엔 없었다).
        #   액션이 **절대 목표**로 바뀐 지금 정책은 "참조 궤적이 지금 어디 있는지"를
        #   모른 채 목표를 지정하고 있었다 — 절대 액션과 짝인 관측이다.
        # ★접촉력(num_fingers)은 Kuka policy obs 에 없지만 **유지**한다. 이 트랙의
        #   존재 이유가 촉각 s2r 이고(tip F/T 15D 실기 배포 규약), 빼면 트랙이
        #   성립하지 않는다. Kuka 는 critic 에만 hand_forces 를 둔다.
        # policy = 관절각·속도(2·nj) + palm pos(3) + **palm 회전 2열(6)** + 손끝(3·nt)
        #          + 물체 pose(7) + goal(3) + last action + fabric q/qd/qdd(3·nj)
        # ★08.25 변경 2건(사용자 결정):
        #   · 접촉력(num_fingers) 을 **critic 전용**으로 이동 — Kuka 배선과 동일
        #   · palm 쿼터니언(4) → **회전행렬 두 열(6)**. 부호 이중성(q ≡ −q) 제거,
        #     보상의 자세 항(perp_q·roll_q)이 정확히 이 두 축의 함수라 정합이 맞는다.
        #   · 물체 **자세**(obj_quat 4) 도 critic 으로 — 실기 인식으로 안정적으로 얻기
        #     어렵다(사용자 결정). policy 는 물체 **위치**만 본다.
        #   · `fabric_qd`/`fabric_qdd` 54D 제거 — Kuka 가 둘 다 0 으로 죽이고, LSTM 이
        #     `fabric_q` 시퀀스에서 속도를 뽑으며, 손 구간은 실측상 상수였다.
        self.observation_space = (
            2 * num_joints + 3 + 6 + 3 * num_tips + 3 + 3 + self.action_space
            + num_joints
        )
        # critic = 관측 + **접촉력(num_fingers)** + 물체 속도(6) + 난이도(1)
        #          + 측정 관절토크(num_joints)
        # ★measured_joint_torque 는 Kuka critic obs 에 있다(privileged).
        # ★08.25 구 `joint_vel_true`(num_joints) 삭제 — policy obs 의 joint_vel 이
        #   이제 실측 원값(scale 1.0)이라 critic 에 같은 값을 또 넣을 이유가 없다.
        self.state_space = (
            self.observation_space + num_fingers + 4 + 7 + num_joints)
        # 스폰 높이는 cfg 세 값에서만 파생(단일 소스) — 프로필은 xy 만 준다
        self.object_spawn_z = (
            self.table_surface_z + self.object_origin_offset_z + self.object_spawn_pad)
        self.object_cfg.init_state.pos = [
            profile.object_spawn_center[0], profile.object_spawn_center[1], self.object_spawn_z,
        ]




@configclass
class GraspSensorTesolloRightEnvCfg(GraspSensorEnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspSensorGripperLeftEnvCfg(GraspSensorEnvCfg):
    profile_name: str = "gripper_left"
