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

_FRICTION = 0.75


@configclass
class GraspSensorEventCfg:
    """물리 재질 — startup 1회. 값은 절대값이다(배율 아님).

    ★씬 기본(SimulationCfg.physics_material)만으로는 부족하다: 로봇·컵 콜라이더는 각자
    재질을 갖고, PhysX 결합이 average 라 한쪽만 올리면 실효 μ 가 중간값이 된다.
    mode="startup" 인 이유 = 이 term 은 CPU 텐서로 버킷을 만들어 초기화 때만 쓰라고
    상류가 명시한다(events.py:167-170). 고정값이라 reset 마다 다시 걸 이유도 없다.
    """

    robot_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    object_material = EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (_FRICTION, _FRICTION),
            "dynamic_friction_range": (_FRICTION, _FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

_HDGP_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *([".."] * 6)))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


def _build_robot_cfg(profile: RobotProfile) -> ArticulationCfg:
    """프로필 → ArticulationCfg. 조인트 이름은 전부 프로필에서."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, profile.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                # 5.0 → 1.0: 관통 시 밀어내는 속도가 크면 충격량이 폭증한다
                # (fabric 트랙 실측: 전형 13~20N 인데 스파이크 7218N).
                max_depenetration_velocity=1.0,
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
                enabled_self_collisions=True,
                # ★P-9 실측(2048env)으로 16 확정: 32 는 fps 를 25% 깎는데(11.9k→8.9k)
                #   파지 품질이 나아지지 않았다. 오히려 접촉력이 더 높았다
                #   (solver32 Fa 28N/Fb 25N vs solver16 Fa 18N/Fb 8N) = 상호침투를 더
                #   밀어내고 있었다는 뜻. self-collision + convex_decomposition 자산이
                #   이미 관통을 막으므로 32 는 불필요한 비용이다.
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
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
    episode_length_s: float = 8.0            # 480 스텝
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
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.75, dynamic_friction=0.75, restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True,
    )

    # ---- 공간 (프로필에서 __post_init__ 이 계산) ----------------------------------
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # ---- Fabrics (grasp_sensor 검증값 승계) ----------------------------------------
    # ★정책 스텝(1/60)당 fabrics_dt × fabric_decimation 만큼 fabric 시간이 흐른다.
    #   1/120 × 2 = 1/60 으로 실시간과 일치시킨다(1/60 × 2 는 2배속이 된다).
    fabrics_dt: float = 1.0 / 120.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 20.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    # ★palm leash 는 제거됐다(08.22) — 정책이 팔 목표에 대해 전권을 갖는다.
    #   목표의 유일한 상한은 아래 워크스페이스 박스(profile.palm_box_*)다.
    #   근거는 grasp_sensor_env.py `_pre_physics_step` 주석 참조.

    # ---- 액션 스케일 (스텝당 delta, 60 Hz) ----------------------------------------
    # 팔: palm 절대 목표 누산 delta — pos 1cm/스텝(최대 0.6 m/s), rot 0.05 rad/스텝.
    arm_pos_scale: float = 0.01
    arm_rot_scale: float = 0.05
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
    hand_control: str = "tip_cyl"
    # tip_cyl 액션 스케일 — 단위=핑거팁 지름(STL 실측 16.1x19.6mm→18mm).
    tip_diameter: float = 0.018
    # r 중심 = 컵 최대반경(45mm)+팁반경(9mm) 대역의 중심. a_r=±1 이 ±2팁지름이라
    # r ∈ [14, 86]mm — 파지 요구 45~54mm 와 완전 개방을 모두 덮는다(워크스페이스 실측
    # r 5~95% = 9~121mm 안쪽).
    tip_r_center: float = 0.050
    tip_action_span: float = 2.0     # a=±1 → ±(span × tip_diameter)
    # direct 모드 attractor 게인. 자매 트랙 스윕 채택값(이동량 부족 26%→8%).
    hand_attractor_gain: float | None = 400.0
    # ★손가락↔손가락 반발을 Fabrics **계획 단계**에서 건다. PhysX self-collision 을
    #   끄기 위한 전제다(자기충돌은 스텝 시간의 55~64%).
    #   자매 트랙 검증: repulsion ON 이면 fabric_q 의 다른 손가락 구 최소거리 20.1mm ·
    #   18mm 미만 0.0% (OFF 는 계획에 관통 해가 남는다) · palm 추종오차 0.7mm 로 무결.
    #   계획에 관통 해가 없으므로 정책이 관통으로 이득 보는 전략을 학습할 수 없다.
    use_hand_repulsion: bool = False
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
    stage_gc_opposition_frac: float = 0.39
    stage_approach_weight: float = 2.0
    stage_approach_sharpness: float = 8.0
    # 정렬 배수의 바닥 — align=−1(손등 쪽)일 때 남기는 비율. 0 이면 초기 오정렬에서
    # approach gradient 가 사라져 접근 자체를 못 배운다(reward-audit Check1).
    stage_align_floor: float = 0.25
    # ②grasp = w·(reach 몫 + G 몫). 두 몫의 합이 1 이라 재정규화된다.
    stage_grasp_weight: float = 6.0
    stage_graspq_reach: float = 0.4
    stage_graspq_g: float = 0.6
    stage_grip_sharpness: float = 8.0
    # 팁 반경 — **손 기하 상수**(물체 형상 아님). 접촉 감지 실패 시 압입 무한보상 차단.
    stage_grip_dist_floor: float = 0.009
    # 파지품질 G 의 내부 몫 — 얕은 감쌈(wrap)보다 깊은 감쌈(deep=같은 손가락 mid AND
    # dist)에 무게를 준다. 2지 팁 핀치는 wrap4=0.25·deep4≈0 → G≈0.06, 완전 인벨롭은
    # G=1.0 으로 **16배** 차이가 난다(이것이 핀치를 죽이는 유일한 장치).
    stage_gq_thumb_floor: float = 0.25   # 엄지 미대향 시 남기는 비율
    # ★08.25 실측으로 재가중(구 0.25/0.75 → 0.8/0.2). 엄지 대향이 성립하는 어떤
    #   조건에서도 `deep4`(같은 손가락 mid AND dist)가 **0.25 를 못 넘는다** — 오프셋
    #   스윕과 반경 스윕 두 축에서 독립적으로 확인됐다. 손 대향폭 ~100mm 에 컵 반경
    #   45mm 라 대향 중점에서는 4지가 원위 마디로만 닿는 기하다. deep 에 0.75 를 주면
    #   최선의 파지도 G 가 0.44 에 갇혀 리프트가 44% 로 깎인다(도달 불가 목표 =
    #   "엄지를 감쌈 분모에 넣어 상한 0.8" 과 같은 계열의 실수).
    #   판별력은 wrap4 가 이미 충분하다: test7 핀치 0.25 vs 인벨롭 1.00 = 4배.
    #   재가중 후 — 인벨롭+대향 0.85 / 대향없는 4지 0.25 / test7 핀치 0.20.
    stage_gq_wrap: float = 0.8
    stage_gq_deep: float = 0.2
    stage_lift_weight: float = 12.0
    # ★목표(goal_height_offset=0.15)와 **정렬**한다. 포화점을 목표보다 낮게 두면 그
    #   위에서 gradient 가 0 이라 정책이 포화점에 고착한다(lstm_test3: env 0.65 에
    #   2,500 에폭 고착 = 정확히 그 포화점. grasp_v1: 4cm 포화 → 평형 3.1cm).
    stage_lift_height_ref: float = 0.15
    stage_transport_weight: float = 8.0
    stage_tracking_std: float = 0.1
    # 직립은 **독립 항이 아니라 곱셈 인자** exp(−tilt°/τ). 테이블 위 컵에 지급되지 않는다.
    stage_upright_tau_deg: float = 8.0
    stage_stabilize_weight: float = 4.0
    stage_stabilize_sharpness: float = 1.5
    # 컵 밀림 감쇠 — 제곱역수. 선형 (1−d/L) 은 d≥L 에서 정확히 0 이 되어 하드 게이트와
    # 같아지고 gradient 가 소실된다(실측: 밀림 0.207 이 300 에폭간 전혀 안 줄어듦).
    stage_disp_limit: float = 0.06
    stage_success_weight: float = 20.0
    stage_success_height: float = 0.12
    stage_success_wrap_min: float = 0.75
    stage_success_tilt_deg: float = 15.0
    stage_success_pos_tol: float = 0.05

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
    # 물체 반중력 보상력: 유효 중력 = g × max(min_frac, level/max).
    # min_frac 0.15 = level 0 에서도 컵이 테이블에 앉아 마찰로 고정되게(완전 무중력이면
    # 수직항력 0 → 부유 표류. probe 실측: 120스텝에 컵이 100mm 떠다님).
    gravity_min_frac: float = 0.15
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
            # ★물체에도 solver/depenetration 을 명시한다 — 빠지면 PhysX 기본(4회)이라
            #   파지 조임 중 손끝이 컵 벽을 파고든다(사용자 영상 08.20).
            #   값은 grasp_v1·grasp_lift_fabric 과 동일.
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
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
        self.robot_cfg = _build_robot_cfg(profile)
        num_joints = profile.num_arm_joints + profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        num_fingers = len(profile.finger_sensor_bodies)
        # 액션 = palm 6D delta + 손:
        #   pd/fabric  : **자유** 손관절 delta(고정 관절은 정책이 안 건드린다)
        #   tip_cyl    : 손가락별 (r, z) 절대 2D × 5 = 10D — θ 는 공칭값 고정.
        #     원통 좌표가 감쌈 기하를 좌표계에 넣는다: r=개구도(조임이 스칼라 하나),
        #     θ 고정=두 손가락이 같은 각도에 못 옴(겹침을 구조로 차단),
        #     z=컵의 어느 높이. 단위=핑거팁 지름이라 "얼마나 가까이"가 물리량이다.
        if self.hand_control == "tip_cyl":
            self.action_space = 6 + 2 * len(profile.finger_sensor_bodies)
        else:
            self.action_space = 6 + profile.num_hand_joints - profile.num_locked_hand_joints
        # 관측 = q + qd + palm pose(7) + tips(3T) + obj pose(7) + goal(3)
        #        + 손가락별 접촉력 크기(F) + last action
        self.observation_space = (
            2 * num_joints + 7 + 3 * num_tips + 7 + 3 + num_fingers + self.action_space
        )
        # critic = 관측 + 물체 속도(6) + 난이도(1)
        self.state_space = self.observation_space + 7
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
