# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""환경 설정: 5g_grasp_left_v1

right/grasp_v1(grasp_right_env_cfg)의 좌우 미러. v7: Fabrics 팔 학습(6D palm) +
per-finger lerp(5D) + sim2real 가능 obs
- Action: 11D (6D palm pose + 5D per-finger lerp)
- Observation: actor 114D(106 base + 8 물체 onehot) / critic 151D (asymmetric)
- Episode: Grasp phase (Fabrics arm + finger 정책) + left-grip lift-wait (frozen hand)
- Contact: fingertip FT sensor (actor, real-compatible) + distal/middle sensors (critic only)

★자산 전환: right/grasp_v1 은 openarm_tesollo_sensor_rl.usd(오른손 Tesollo만
20관절, 왼팔은 단순 2-DOF 그리퍼 l_hj_gripper_1/2)를 쓴다 — 이 USD 에는 왼손
Tesollo 관절이 없어 좌팔 제어가 불가능하다. left/grasp_v2 선례를 따라
openarm_tesollo_bi_rl.usd(양팔 Tesollo 20관절)로 전환했다 — grasp_left_preset.py
docstring 및 robot_cfg 주석 참조.

2026-07-26 MultiAsset(8종)+DR 이식 (design: docs/superpowers/specs/2026-07-26-
tesollo-grasp-v1-multiasset-dr-design.md): 단일 cup_big_sdf → cup_big×4 scale +
shaker_body + cyl 3종(높이 12cm 통일). reward/성공판정/side approach 로직은 불변.
물체 자산·경로는 좌우 공유이므로 right 원본과 동일하게 유지한다.
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from .grasp_left_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .grasp_left_preset import (
    HAND_BODY_NAMES_USD,
    RIGHT_ARM_AND_GRIPPER_JOINT_NAMES,
    RIGHT_ARM_REST_JOINT_POS,
    LEFT_ACTUATED_JOINT_NAMES,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_VISDEX_ROOT = _os.path.join(_ASSETS_DIR, "visdex_objects", "USD")

# ---------------------------------------------------------------------------
# 물체 구성 — 8종 (design §물체 구성). 논리 ID(=onehot·bbox 조회 키) 순서가
# env_id % 8 결정론적 배정과 MultiAssetSpawnerCfg assets_cfg 순서를 동시에 정한다.
# 물체 자산은 좌우 공유이므로 right 원본과 동일하게 유지한다(경로 치환 금지).
# ---------------------------------------------------------------------------
# 전 물체 공통 기본 질량 [kg]. pour_v1 실컵(cup_big_sdf, density 파생)과 동일.
# ADR mass DR 의 곱셈 기준이 되므로 여기를 바꾸면 실효 질량 범위 전체가 이동한다.
_BASE_OBJECT_MASS: float = 0.134

# ★08.16 SDF 콜라이더 자산(scripts/tools/make_sdf_grasp_assets.py 산출, right 이식).
#   visdex cup_big/shaker 는 physics:approximation="sdf" 를 적어놓고도 apiSchemas 에
#   PhysxSDFMeshCollisionAPI 와 physxSDFMeshCollision:sdfResolution 이 없어 PhysX 가
#   **convexHull 로 폴백**했다(런타임 경고 실증). convexHull 은 컵의 오목한 내부를 메우므로
#   grasp 는 "속이 찬 원통", pour(cup_big_sdf.usd, 진짜 SDF)는 "속이 빈 컵"이 되어
#   파지 자세가 전달되지 않았다(08.16 실측 pour 통과율 4.0%).
#   메시는 두 자산이 **완전 동일**(1765 pts/3526 faces/bbox 일치)이라 authoring 만 맞추면 된다.
#   ⚠️visdex 원본을 제자리 수정하지 않은 이유: grasp_v2 가 visdex_objects/USD 를 sorted-glob 해
#   물체 수로 obs 차원이 정해지므로 그 디렉토리는 불가침. 산출물을 assets/cup/ 에 둔다.
_SDF_ASSET_ROOT = _os.path.join(_ASSETS_DIR, "cup")

_ACTIVE_OBJECT_SPECS: tuple[dict, ...] = (
    # ★기본 질량 8종 통일 = _BASE_OBJECT_MASS(0.134kg, pour 실컵 질량). right 08.16 이식.
    #   ① pour 실컵(cup_big_sdf, density 파생 ≈0.134kg)과 학습 질량 정합.
    #   ② SDF 컵 질량이 scale s³로 몰래 바뀌는 함정 차단(질량은 scale 무관 고정).
    #   ③ USD 기본질량이 제각각(shaker 0.263 / 원기둥 0.100)이면 동일한 ADR scale 을 걸어도
    #      물체마다 절대 질량이 최대 2.6배 벌어져, 무거운 물체만 미검증 외삽 영역으로 튄다.
    #      통일하면 **모든 물체가 동일한 절대 질량 범위 전체를 학습**한다(다양성은 ADR DR 이
    #      매 에피소드 샘플링으로 제공 — 오히려 커버리지가 넓어짐).
    {"id": "cup_big_s085", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (0.85, 0.85, 0.85), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s100", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.00, 1.00, 1.00), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s115", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.15, 1.15, 1.15), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s130", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.30, 1.30, 1.30), "mass": _BASE_OBJECT_MASS},
    # ★08.17 shaker_body → shaker_closed: 원본은 **양쪽 뚫린 관**(축 근처 정점 0개)이라
    #   비드가 바닥으로 그냥 빠졌다(유지율 0.000 vs cup_big 0.975~1.000). 공동 반경은
    #   오히려 셰이커가 넓어(28~38mm) 비드를 줄여도 해결되지 않는다 — 바닥이 없는 것이 원인.
    #   `scripts/tools/make_closed_shaker_asset.py` 가 하단에 얇은 원기둥 콜라이더를 덧붙인
    #   사본을 만든다. 플러그 반경 29.5mm < 셰이커 최대 44mm 라 파지 외곽 프로파일은 불변.
    {"id": "shaker_closed", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "shaker_closed_rl.usd"), "scale": (1.0, 1.0, 1.0), "mass": _BASE_OBJECT_MASS},
    # ★08.17 cyl 3종 제거 → **컵 변형으로 치환** (both/pour_v1 = 컵 다양화 학습).
    #   왜 삭제가 아니라 치환인가: `multi_object_idx_onehot` 의 폭이 `len(_object_names)` 에서
    #   파생되어 물체 수가 곧 actor obs 차원이다(146 = 138 + onehot 8). 3종을 지우면
    #   obs 146→143 이 되어 학습된 체크포인트가 전부 무효가 된다. 8종을 유지하면 obs 불변이라
    #   기존 체크포인트로 학습을 이어갈 수 있고, 컵 다양성은 4종→7종으로 오히려 늘어난다.
    #
    #   cyl 을 뺀 이유: visdex 원본(`large_*_cyl`)은 apiSchemas 에 SDF API 가 없어 PhysX 가
    #   **convexHull 로 폴백**한다 = 속이 찬 원통. pour 는 컵 안에 비드를 담아 붓는 과제라
    #   ① 속 찬 물체는 비드를 담을 수 없고(실측: 그 안에 비드를 소환하면 유지율 0.089,
    #      실제 컵은 0.95~1.00) ② pour source/receiver 는 항상 컵이므로 전이 가치도 없다.
    {"id": "cup_big_s090", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (0.90, 0.90, 0.90), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s105", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.05, 1.05, 1.05), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s120", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.20, 1.20, 1.20), "mass": _BASE_OBJECT_MASS},
)
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = tuple(_s["id"] for _s in _ACTIVE_OBJECT_SPECS)


def _object_usd_cfg(spec: dict) -> "sim_utils.UsdFileCfg":
    """단일 물체 USD spawn cfg. rigid/articulation 속성은 tesollo 기존 cup_cfg 값 그대로.

    spec에 "mass"(kg)가 있으면 mass_props로 명시 고정 — USD 내장값(명시 mass 또는
    density 파생)을 덮어써 scale·자산 교체와 무관하게 질량을 결정론화한다.
    """
    _mass_props = (
        sim_utils.MassPropertiesCfg(mass=float(spec["mass"])) if "mass" in spec else None
    )
    return sim_utils.UsdFileCfg(
        usd_path=spec["usd_path"],
        activate_contact_sensors=True,
        scale=spec["scale"],
        mass_props=_mass_props,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=100.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )


# env_id % 8 결정론적 배정 (rh56f1 grasp_v1 이식). replicate_physics=False 필요.
_GRASP_OBJECT_SPAWN = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[_object_usd_cfg(_s) for _s in _ACTIVE_OBJECT_SPECS],
    random_choice=False,
)


@configclass
class EventCfg:
    """물체 physics DR (design §DR — friction/mass, 매 reset per-env 연속 랜덤).

    ★08.16 ADR 스케줄 대상으로 전환(right 이식, grasp_v2 방식): 아래 값은 **초기(중립)
    범위**이고, ADR increment 마다 adr_physics_cfg 의 terminal 로 선형 확장된다. 구조상
    정적 고정 범위는 "처음부터 최대 난이도"라 커리큘럼이 아니었다(질량·마찰이 ADR 밖).
    """

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup", body_names=".*"),
            # 초기=중립(1.0 배). terminal 은 adr_physics_cfg 참조.
            "static_friction_range":  (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range":      (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            # 초기=중립(원 질량 그대로). terminal 은 adr_physics_cfg 참조.
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class GraspLeftEnvCfg(DirectRLEnvCfg):
    """5g_grasp_left_v1 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    # Episode: 10s = 600 steps @ 60Hz (8s grasp + 2s lift)
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 114 (actor, 106 base + 8 물체 onehot)
    action_space:      int = NUM_ACTIONS               # 11
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 151 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric:            bool  = False
    max_pose_angle:             float = 45.0
    fabrics_max_objects_per_env: int  = 8
    fabrics_damping_gain:       float = 20.0  # 10→20: Fabrics 속도 감쇠 증가 → grasp phase 떨림 감소

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # -----------------------------------------------------------------------
    pregrasp_fabric_steps: int   = 60
    reset_fabric_chunk_size: int = 128
    # 17×17 grid(1cm 간격, ±8cm) IK 사전 계산 → reset 시 lookup. design §위치 ADR:
    # spawn xy_range가 ADR로 0.02→0.08까지 커지므로 캐시는 항상 최대범위(±8cm)를 커버해야
    # ADR가 range를 넓혀도 lookup이 grid 밖으로 벗어나지 않는다(2026-07-26, 13×13/±6cm→17×17/±8cm).
    # ★2026-08-18 고정 홈 리셋 (perception_plus_plus 연결용).
    #   구 동작은 컵 **참값**으로 pregrasp 를 계산해 팔을 텔레포트했다 — 실기에서는
    #   컵 위치가 지각 결과라 재현 불가였고, 컵 y=+0.10 일 때 손이 스폰 박스 안에서
    #   출발해 side 접근도 성립하지 않았다. 이제 컵과 무관한 고정 홈에서 출발한다.
    #
    #   홈 자세 근거 (URDF openarm_tesollo_bi_s_rl FK/IK 실측, base=env 원점):
    #     위치오차 0.00mm · 자세오차 0.0deg · 컵 스폰 박스 밖 80mm
    #     팔 최저 z=0.377(테이블면 0.208) · 헤드 카메라 시선 최근접 72.7mm
    #     (링크 반경 45mm) → 초기 프레임에서 컵 가림 없음
    #   right/grasp_v1 홈 (0.28, -0.38, 0.42, 90, 0, 90) 의 y·자세 미러.
    reset_from_fixed_home: bool = True
    # [x, y, z, ez, ey, ex] — 회전은 도(deg), env 에서 라디안 변환.
    reset_home_palm_pose: tuple = (0.28, 0.38, 0.42, -90.0, 0.0, -90.0)

    # cache_pregrasp_reset: reset_from_fixed_home=True 면 쓰이지 않는다.
    cache_pregrasp_reset:  bool  = True
    pregrasp_cache_xy_range: float = 0.08   # 캐시 grid 반경(고정) — object_spawn_xy_range(ADR 초기값)와 별개
    # pregrasp offset: right 는 -Y(-0.07), left 는 y=0 대칭 반전(+0.07)
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = 0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Demo reset (pour_v1_a11~a20 grasp start and lift target)
    # -----------------------------------------------------------------------
    # right 데모 데이터셋(오른팔 레코딩)만 존재 — 왼팔 전용 데이터셋이 없으므로
    # False 유지(demo_grasp_reset.py 참조). 2026-07-26 MultiAsset 다물체 이식으로도
    # demo pose 는 단일 컵 전용 고정 자세라 8종 물체에 부적합(rh56f1/grasp_v2 동일 결정).
    enable_demo_grasp_reset: bool = False
    demo_grasp_pose_paths: tuple[str, ...] = tuple(
        _os.path.join(_HDGP_ROOT, "..", "datasets", f"pour_v1_a{i}.hdf5") for i in range(11, 21)
    )

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # actor obs에만 적용; critic obs는 privileged clean state 유지
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01    # joint position σ [rad]
    obs_noise_joint_vel: float = 0.05    # joint velocity σ [rad/s]
    obs_noise_body_pos:  float = 0.005   # FK body position σ [m] (palm, fingertip)
    obs_noise_cup_pos:   float = 0.015   # cup position observation σ [m]

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    # cup_grasp_z_offset: cup_big(높이 17.8cm) 기준 보정값. 2026-07-26부터 GraspLeftEnv가
    # object_bbox.json 반높이 비율로 물체별 텐서(cup_grasp_z_offset_buf)를 파생 — 이 필드는
    # 파생 시 기준 비율(0.06/cup_big 반높이)로만 쓰이고 reward/리셋에 직접 대입되지 않는다.
    cup_grasp_z_offset:  float = 0.06
    lift_success_height: float = 0.04
    # ★2026-08-19 수직 palm 리프트 높이. 래치 palm 에서 z 를 이만큼 올린다(120스텝 램프).
    lift_height_delta: float = 0.10
    # lift 보상 정규화 기준을 성공 임계와 분리한다.
    #   구: lift_height_quality = (height_delta / lift_success_height(0.04)).clamp(max=1.0)
    #       → 4cm 에서 포화. Δ=0.10 인데 컵이 5cm 만 따라와도(=손 안에서 5cm 미끄러짐)
    #         보상이 만점이라 미끄러짐을 구분하지 못한다.
    #   신: 정규화만 0.10 으로 열어 0~10cm 에 gradient 를 편다. 최대 보상 크기는 그대로.
    #   ※ lift_height_bonus_clamp 를 올리는 우회는 무효 — 그 분기의 lift_height_bonus_weight
    #     가 grasp_v1 에서 0(기본)이라 꺼져 있다.
    lift_height_ref: float = 0.10

    # -----------------------------------------------------------------------
    # Delta palm action (pregrasp 기준 상대 오프셋)
    # action=0 → pregrasp 위치 유지, action=±1 → pregrasp ± delta
    # -----------------------------------------------------------------------
    # ★2026-08-18 스칼라 0.15 → 축별 (x, y, z). 고정 홈 리셋으로 바꾸면서 y 만 확대.
    #   홈 palm y=+0.38 에서 컵 y=+0.10 까지 0.28m 를 액션으로 덮어야 한다.
    #   x·z 는 파지 직전 미세조정 분해능 유지를 위해 0.15 그대로.
    palm_delta_xyz:     tuple = (0.15, 0.35, 0.15)   # ±m, (x, y, z)
    palm_delta_rot_deg: float = 20.0   # ±20° per axis

    # -----------------------------------------------------------------------
    # Reward 파라미터
    # -----------------------------------------------------------------------
    # RH56F1 shared grasp-v2 reward contract.
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    # ★2026-08-19 5.0 → 25.0. 컵 밀기 페널티는 approach 안에 있는데 approach 전체가
    #   총보상의 1.6% 뿐이라, 컵을 10cm 밀어도 페널티 0.375 vs 성공보너스 4.572(비 1:12)로
    #   미는 게 합리적인 구조였다. 25 는 lift+success(9.16)의 ~10% 수준.
    #   ⚠ 60 이상 금지 — "컵에 안 다가감" 이 국소최적이 된다(reward-audit Check 2).
    approach_xy_penalty_weight: float = 25.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    lift_reward_weight: float = 30.0
    stabilize_weight: float = 10.0
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02
    grasp_xy_threshold: float = 0.025
    # ★2026-08-19 컵 밀림 soft 감쇠 한계. lift/success_bonus 에 (1-clamp(disp/limit)) 를 곱한다.
    #   0 = 비활성(기존 동작 불변). 0.08 이면 4cm 밀 때 보상 50%, 8cm 면 0.
    #   하드 게이트를 쓰지 않는 이유: 보상 86% 를 한 번에 끄면 '컵을 안 밀면서 잡는 법' 을
    #   아직 모르는 초기 구간에서 gradient 가 approach(1.6%)만 남아 탐색이 붕괴한다
    #   (reward-audit Check 1 실패, 과거 pour_gate 지연이 같은 이유로 실패).
    cup_xy_disp_limit: float = 0.08
    grasp_upright_threshold_deg: float = 8.0
    success_upright_max_deg: float = 20.0
    stabilize_upright_max_deg: float = 5.0
    stabilize_upright_reward_scale_deg: float = 5.0
    stabilize_action_sharpness: float = 1.5
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.2
    stage0_lift_start_min_contacts: int = 4  # (persistence·hold용 tip 접촉 임계, 유지)
    lift_start_min_grip_fingers: int = 3  # ★lift 진입: 아무 마디(tip|mid|distal) 닿은 손가락 수 임계.
    # 사용자 의도="어느 위치든 닿으면 lift 진행"+제어(g3)가 distal까지 감아 인벨롭 유도.
    # 부실 방지=hold_steps 연속유지+success(lifted+stable+tilt). 3=엄지+대향2지. success는 success_min_grip_fingers(4).
    success_min_grip_fingers: int = 4  # success 그립 손가락 수(엄지-컵 접촉 AND 강제와 함께 사용). 5(전손가락 동시)는 wrap 진동 이력.
    grasp_ready_hold_steps: int = 8   # 접촉 N개를 연속 hold하면 lift 래치 (잡으면 바로 리프트)
    lift_start_min_envelope_fingers: int = 0  # latch 인벨롭 게이트 제거(0=비활성). envelope은 grasp/lift 보상 credit으로 유도(hard 게이트 대체)
    finger_close_speed: float = 0.05  # ① 접촉-게이트 적응 폐쇄: 손가락 폐쇄 진행 속도/step (중간마디 접촉 시 동결)
    # ★couple_four_fingers(08.02, left-only): 검지~소지(4지)를 공통신호(평균)로 묶어 "특정
    # 손가락만 안 닫힘"을 표현 불가하게 함 → 3지 국소최적 원천 차단. 엄지(0)는 opposition
    # 위해 독립. per-joint 접촉게이트(g3/g4)는 유지되어 물체 형상에 개별 적응(닿는 곳서 동결).
    # grasp_v2 에서 3지 고착 해소 검증된 방법(grasp_v2 left ADR50 0.908). grasp_v1 은 07-28에
    # 제거했으나 left per-finger 가 3지로 수렴(right 는 우연히 5지)해 08.02 left 만 재도입.
    # reward/obs/action 차원 불변(action 전처리) → 좌우 reward 동일 제약 유지·right 무영향.
    # 08.03 되돌림(True→False): couple 은 lift 미러버그를 우회하려던 잘못된 방향이었다.
    # 진짜 원인=joint7 lift 미러(위 수정). lift 고치면 right 와 동일 per-finger 순수 미러로
    # 5지 리프트 수렴 기대 → couple 불필요(오히려 warmstart per-finger 정책과 상충해 0.565 정체).
    # 08.03-2 재도입(False→True): lift 수정 warmstart(lstm_test11)가 lifted 0.15→0.69 로 lift
    # 근본원인 해결을 확정했으나, per-finger 파지가 grip 3.4·success 0.74 천장에서 정체(right 4.4/0.89
    # 미달). 병목이 lift 에서 per-finger firmness 로 이동 → couple 로 4지 공통닫힘(3지 국소최적 차단).
    # 이번엔 fresh(warmstart 없음)+lift 수정 유지라 과거 couple 실패 두 원인(warmstart 상충·lift 버그)
    # 모두 제거됨. 여전히 reward/obs/action 불변·left-only·right 무영향.
    couple_four_fingers: bool = True

    # -----------------------------------------------------------------------
    # 감쌈(envelope) 품질 — 08.16 개편. reward-audit ACCEPT/REVISE 반영.
    #
    # 배경(실측): ADR 만렙 후 난이도가 상수인 2,687 epoch 동안 성공률·리프트는 평탄한데
    # **감쌈만 단조 침식**했다(envelope 1.94→1.74, full_envelope 0.24→0.17,
    # middle_count 2.44→2.16, 엄지 cup_mid 0.59→0.48). 손끝 접촉은 오히려 줄었으므로
    # "손끝 전환"이 아니라 **중간마디(PIP)를 버려 감쌈이 얕아지는 것**이고, 그게 공짜였다:
    #   · grasp(12.0, envelope credit)는 pre_lift_gate 로 리프트 순간 꺼진다
    #   · post_lift_contact_loss 는 grip_frac(마디 무관 OR)이라 mid→distal 이동이 무비용
    # -----------------------------------------------------------------------
    # grasp 보상 안 감쌈 비중(core 기본 0.40). 합이 1로 재정규화되므로 올려도 grasp
    # 최대치는 불변 — 국소최적을 구조적으로 못 만든다. 래치 **전** shaping 이라
    # 래치 시점 자세가 깊어지고, 그 자세가 곧 유지 페널티의 기준선이 된다.
    grasp_envelope_credit: float = 0.55
    # 리프트 후 게이트의 감쌈 비중(core 기본 0.5): graded_contact=(1-mix)*tip+mix*envelope.
    # ★여기엔 느슨한 envelope_frac 이 들어간다 — 엄격한 깊이(wrap)를 넣으면 실측 0.349→0.144
    #   (0.41배)라 lift 30.0·stabilize 10.0·stability 1.0 을 20~30% 일괄 삭감한다.
    #   그래서 mix 만 소폭 올린다(0.5→0.6, graded 0.518→0.484 로 −6.6%).
    lift_envelope_mix: float = 0.6
    # 감쌈 유지 페널티 가중치. **래치 시점 대비 감소분**에만 걸린다(절대 깊이 아님).
    # 유지하면 정확히 0 이라 보상 기준선이 이동하지 않고, 잃을 때만 비용이 생긴다.
    # 크기 산정: 실측 wrap 침식폭 ≈0.05(0.19→0.14) × 이 가중치 = 스텝당 −0.30,
    # reward/lift 실측 5.8 대비 5% 수준 — 신호는 되되 리프트를 억제하지 않는 선.
    wrap_retention_loss_weight: float = -6.0
    # ★★08.16 실패 실증 → False 로 되돌림(wrapfix1_retighten_FAILED_ep400).
    # 시도: 래치 후 동결 게이트를 풀어 "더 조일" 권한을 준다.
    # 결과: **감쌈이 파괴됐다.** 같은 epoch(300~450) 대조 —
    #   full_envelope 0.176→0.035(0.20x), wrap 0.447→0.214(0.48x),
    #   middle_count 2.496→1.190(0.48x), distal_count 0.736→0.420(0.57x)
    #   그런데 총 접촉은 4.07→4.41 로 늘고 five_tip 동시접촉률은 0.42→0.68(1.61x)
    #   = 접촉이 사라진 게 아니라 중간·원위마디에서 **손끝으로 이동**(인벨롭→핀치).
    # 원인: 동결 게이트는 "컵 반경보다 더 말아쥐지 못하게" 하는 장치였다. 풀어주면 손가락이
    #   컵 반경(4.5cm)보다 작은 반경으로 말리고, 그러면 기하학적으로 컵에 닿을 수 있는 점은
    #   가장 끝(손끝) 하나뿐이라 중간·원위가 들린다(close_frac 0.96 = 거의 주먹).
    # 교훈: **위치 목표를 더 미는 것은 "조이는" 동작이 아니라 "더 작은 반경으로 마는" 동작**이다.
    #   반경이 정해진 물체에서는 후자가 감쌈을 파괴한다. 파지력을 키우려면 자세를 더 말지 않고
    #   같은 자세에서 토크를 키워야 하는데, 현 구조(위치 목표가 유일 입력)에는 그 채널이 없다.
    retighten_after_latch: bool = False

    # -----------------------------------------------------------------------
    # ★08.16 squeeze(가압) 철회 — 액션에서 제거(NUM_SQUEEZE_ACTION=0). 근거는 constants 참조:
    #   도입 전제("동결 때문에 오버슈트가 0")가 오진이었고 실제 원인은 토크 포화였다.
    #   게인 재설정으로 포화는 풀렸으나, PIP/DIP 분리 채널이 같은 권한을 더 자연스럽게 준다.
    # ★틸팅 종료 억제를 스크립트 램프(LIFT_PHASE_STEPS) 구간으로만 한정. 기존엔 래치 이후
    # 전 구간을 억제했는데 그 구간이 정확히 회전 외란이 걸리는 구간이라, 외란의 유일한
    # 실패 신호가 꺼져 있었다. 램프 후 hold 구간에서 되살린다.
    tipping_active_after_lift_ramp: bool = True
    grasp_contact_persistence_reward_steps: int = 20
    enclosure_sharpness: float = 15.0
    # cup_radius_approx: cup_big 기준값(반경). 2026-07-26부터 per-object bbox 텐서
    # (cup_radius_approx_buf = bbox half_x/half_y 평균)가 enclosure target·obs 진단에 쓰인다.
    # 이 필드는 object_bbox.json 로딩 실패 시 즉시 예외(fail loud) — fallback 미사용.
    cup_radius_approx: float = 0.045
    enclosure_thumb_weight: float = 0.6

    # -----------------------------------------------------------------------
    # 물체 외란 wrench (08.16 right 이식, DEXTRAH apply_object_wrench)
    # pour 연결 강건화: pour 는 hold 후 비드 20×1g 소환(+15% 계단 하중)+슬로싱(자유 구름)
    # +deep tilt 회전을 가하는데, 기존 grasp_v1 학습은 외란 경험 0회(right e2c0e7a: pour
    # 검증 9~15%). force = mass × accel × 랜덤방향(등방), torque = force × torsional_radius.
    # 게이트: palm이 물체 반경(wrench_hand_dist_threshold) 내일 때만 인가(DEXTRAH 원본).
    # 크기는 ADR 커리큘럼 0→max — 급격 도입 시 grip 붕괴 방지.
    # -----------------------------------------------------------------------
    wrench_enable: bool = True
    # ★08.16 15→8 하향(S2 실측, right 동일). 하중 = m·(9.81+a) 최악 코너를 실측 파지
    # 능력 안에 맞춘 값. 강건성 평가에서 3배(24 m/s²)까지 생존 98.4% 확인.
    wrench_max_accel: float = 8.0      # m/s² ADR 종점. 컵 154g 기준 최대 2.3N = 중력(1.51N)의 1.5배
    wrench_torsional_radius: float = 0.045  # m — pour source_inner_radius(0.041) 비드 쏠림 레버암 재현
    wrench_trigger_every: int = 15      # step(0.25s @60Hz)마다 새 랜덤 wrench — 비드 슬로싱 주기 근사
    wrench_hand_dist_threshold: float = 0.3  # m — palm-물체 거리 게이트
    # 회전 외란(Exp4-A 물리판·reward-audit ACCEPT): lift latch 이후 env에 수평 랜덤축 torque를
    # 추가 인가 — pour deep-tilt 중 컵이 손안에서 회전하려는 하중을 재현(회전에 안 놓는 grip 단련).
    # reward/gate 불변(물리만) — 실패 신호는 기존 tipping 종료(60°)+success upright 게이트가 제공.
    hold_rotation_perturb_enable: bool = True
    # a=12·r=0.045 → τ = 0.154×12×0.045 = 0.083 N·m. pour 110° 기울임에서 컵+비드(0.154kg)가
    # 파지점 레버암 ~0.05m 로 만드는 중력 토크(≈0.076 N·m) 수준을 재현한다.
    # ★08.16 12→8 하향(right 동일): 질량 종점과 조합해 pour deep-tilt 토크 재현 유지.
    hold_rotation_perturb_max_accel: float = 8.0

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    # ★08.16 케이던스 완화(200→1000): 질량·마찰·wrench·회전이 모두 이 카운터 하나를
    # 공유하게 되면서 구 케이던스(50증분×200step≈epoch 500 만렙)는 "커리큘럼 없이 즉시
    # 최대 난이도"와 같아졌다(right lstm_test7 실측: ep~500 만렙 도달). 1000step 이면
    # 만렙까지 최소 ~1,560 epoch — 20k 런에서 실질 커리큘럼이 된다.
    adr_increment_interval: int  = 1000
    # threshold 는 "무너지는 중엔 난이도를 올리지 않는다"는 가드(순간 성공률 기준).
    # 0.25: 순간값(success_flag 평균)은 에피소드 단위 성공률보다 구조적으로 낮다
    # — 에피소드 중 success 유지 구간이 일부이기 때문. 실측은 TB adr/trigger_metric 으로 재보정.
    adr_trigger_threshold: float = 0.25

    # design §위치 ADR: spawn xy_range 0.02→0.08 점진 확대(초기 좁게 학습 후 확장).
    # grasp_adr.get_param("spawn","xy_range")로 GraspLeftEnv._reset_idx가 조회.
    # ★2026-08-18 스칼라 xy_range → 축별 분리 (right/grasp_v1 와 동일).
    #   목표 스폰 범위 x 0.25~0.35(±0.05), y +0.10~+0.30(±0.10).
    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "x_range": (0.02, 0.05),
            "y_range": (0.02, 0.10),
        },
        # 외란 커리큘럼: 0에서 시작해 increment마다 선형 증가(급격 도입 시 grip 붕괴 방지).
        # spawn과 동일 카운터 공유 — 학습 곡선에서 latch 후 붕괴가 보이면
        # adr_increment_interval 상향으로 램프 속도 완화(스케줄 자체 변경은 하지 않음).
        "object_wrench": {
            "max_linear_accel": (0.0, 8.0),
        },
        "hold_rotation": {
            "max_accel": (0.0, 8.0),
        },
    })

    # ★08.16 물리 DR ADR (right 이식, grasp_v2 방식): EventCfg 의 초기(중립) 범위를
    # increment 마다 아래 terminal 로 선형 확장한다. 질량·마찰이 정적 고정 범위였던
    # 구조를 커리큘럼으로 전환 — 초기엔 원 질량/마찰로 파지를 익히고, 성공할수록
    # 무겁고 미끄러운 물체로 확장.
    adr_physics_cfg: dict = field(default_factory=lambda: {
        "object_physics_material": {
            "static_friction_range":  (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),   # 하한 0.3 = 미끄러운 컵
            "restitution_range":      (0.8, 1.0),
        },
        "object_scale_mass": {
            # ★08.17 ×4.0 복원 (강건성 평가 실측). 08.16 에 S2 프로브(총 하중 5N)를 근거로
            # 2.0 으로 내렸으나, 그 수치는 **스크립트 팁핀치**(wrap 0.03) 기준이었고 학습된
            # 인벨롭 정책(wrap 1.43)의 실측 능력은 2.8배였다: 536g(×4.0)+외란 2.0×에서
            # 생존 88.4%, cup_big 계열은 0.937 이상. "컵 외 300g 적재" 요구도 이 복원으로
            # 재충족(536-134=402g). 원통 3종은 이 영역에서 어려움(마찰 의존) — 학습 과제로 남김.
            "mass_distribution_params": (0.5, 4.0),
        },
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 60.0
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    # right 는 y∈[-0.60,0.25](왼팔 쪽 워크스페이스로 이송). left 는 y=0 대칭 반전.
    obj_out_y_min:  float = -0.25
    obj_out_y_max:  float = 0.60
    obj_fallen_z:   float = 0.20

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    # ★2026-08-18 x 중심 0.27 → 0.30 (목표 범위 0.25~0.35 의 중앙).
    object_spawn_x_center: float = 0.30
    # right: y=-0.10 (demo 데이터와 일치). left: y=0 대칭 반전 → +0.10
    # ★2026-08-18 스폰 y 를 ∓0.10 → ∓0.20 으로 벌린다 (both/pour_v1 양손 파지용).
    #   **정책은 바꾸지 않는다** — 스폰 위치만 옮긴다. 재학습 불필요함을 실측으로 확인:
    #   같은 체크포인트(ep_8000/8500)로 ∓0.16 · ∓0.20 에서 각각 256/256 을 수집했다.
    #   실기에서도 컵을 벌려 놓고 시작하므로 이 조건이 s2r 과 더 정합한다.
    #
    #   왜 필요한가: grasp 의 리프트가 joint7 만 0.31rad 돌리는 스크립트 동작이라 잡은 컵을
    #   측면으로 스윙시켜 **좌우 컵이 중앙으로 모인다**. pour_v1 에서 둘을 합치면 컵이 겹친다.
    #   실측(실제 좌/우 페어 8000쌍, 대칭 가정 아님):
    #     스폰 ∓0.10 → warm y 우 -0.040 / 좌 +0.039, 간격  80mm, 겹침 44.3%
    #     스폰 ∓0.16 → 간격 129mm, 겹침 6.9%
    #     스폰 ∓0.20 → warm y 우 -0.085 / 좌 +0.112, 간격 197mm, 겹침 **0.05%**
    #   (좌팔이 y 를 덜 끌어당겨 대칭 예상보다 오히려 더 벌어졌다)
    #   남는 0.05% 는 pour_v1 리셋의 겹침 페어 재추첨이 걸러낸다.
    #
    #   ⚠ 스폰을 더 벌리는 것(∓0.24 등)은 도달 한계에 가까워져 권하지 않는다.
    #   ⚠ `j1` 회전으로 사후 보정하려던 시도는 폐기했다 — j1 은 베이스 요가 아니다
    #      (FK 실측 Δy=0.000). pour_v1 cfg `warm_arm_yaw_spread_rad` 주석 참조.
    object_spawn_y_center: float = 0.20   # ∓0.20 (구 +0.10)
    # object_spawn_z: cup_big(반높이 0.08881200104951859) 기준 테이블 안착 높이. 2026-07-26부터
    # GraspLeftEnv가 이 값에서 테이블 표면 z(=object_spawn_z - cup_big 반높이)를 역산해
    # 물체별 반높이를 더한 object_spawn_z_buf(N,)를 파생한다 — reset에서 물체별 텐서를 쓴다.
    object_spawn_z:        float = 0.297
    # object_spawn_{x,y}_range: enable_adr=False 일 때만 쓰이는 fallback. 기본은
    # enable_adr=True → adr_custom_cfg["spawn"]["x_range"/"y_range"]가 실제 범위를 결정.
    # ★2026-08-18 스칼라를 축별로 분리하고 ADR 종점과 맞췄다.
    object_spawn_x_range: float = 0.05
    object_spawn_y_range: float = 0.10
    # 활성 물체군(8종, MultiAsset assets_cfg 순서와 일치) — onehot·bbox 조회·env_id%8 배정용.
    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
    object_bbox_path: str = _os.path.join(_ASSETS_DIR, "object_bbox.json")

    # -----------------------------------------------------------------------
    # Warm-state export (grasp 성공 → 디스크 캐시 → pour warmstart 재사용)
    # -----------------------------------------------------------------------
    # 학습 루프에는 영향 없음 (기본 False). collect 스크립트/play 에서만 True.
    # success 이후 왼손 grasp arm pose 를 유지하고 joint7 만 lift-wait 로 이동한 상태를 저장한다.
    # demo cup/phase 구분을 신뢰하지 않고 실제 sim 손/컵 grasp 결과를 그대로 유지한다.
    # 손가락 접촉은 기본 2개 이상, lift-wait arm match 는 기본 1 step 만 요구한다.
    enable_warm_state_export: bool = False
    warm_state_export_path: str = _os.path.normpath(
        _os.path.join(_HDGP_ROOT, "..", "datasets", "grasp_warm_left_v1.hdf5")
    )
    warm_state_target_count: int = 2048
    warm_min_contacts: int = 2
    warm_contact_stable_steps: int = 1
    warm_lift_wait_arm_tol: float = 0.035
    warm_lift_wait_hold_steps: int = 1
    # ★joint7 lift 미러 수정(08.03): joint7 은 _ARM_SIGN=-1 미러 관절(left 실측 ≈ -0.97).
    # right 의 양수 delta(+0.31)·클램프[0.20,1.50]를 그대로 쓰면 left joint7 을 -0.97→+0.20 으로
    # 강제(+1.17rad 급젖힘)해 물체를 떨궈 lifted 0.15 고착이었음(right 는 0.8). 방향성 리터럴이라
    # 미러 함수를 안 타 좌우 동일하게 남아있던 버그(rl-mirror-port "하드코딩 리터럴" 전형).
    # delta·범위를 부호 미러: +0.31→-0.31, [0.20,1.50]→[-1.50,-0.20].
    # ⚠2026-08-19 이후 미사용 — 수직 palm 리프트(lift_height_delta)로 대체.
    #   구 j7 리프트는 palm 자세를 17.76° 회전시켜 쥔 컵을 같이 기울였다.
    lift_wait_joint7_delta: float = -0.31
    warm_cup_upright_min: float = 0.90   # legacy override 호환용; lift-wait export 에서는 미사용
    warm_j7_min: float = -1.50
    warm_j7_max: float = -0.20

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
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

    # -----------------------------------------------------------------------
    # 물체 physics DR (design §DR — friction/mass, 2026-07-26)
    # -----------------------------------------------------------------------
    events: EventCfg = field(default_factory=EventCfg)

    # -----------------------------------------------------------------------
    # 씬 설정
    # -----------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        # MultiAsset(env 별 다른 물체) spawn 은 physics 복제 불가(2026-07-26, rh56f1 동일).
        replicate_physics=False,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇 설정 (openarm_tesollo_bi_rl.usd: 양팔 tesollo, 통일 네이밍 r_aj/r_hj/r_hl +
    # l_hj tesollo 20관절. right/grasp_v1 의 openarm_tesollo_sensor_rl.usd — 왼팔 단순
    # 그리퍼 —는 왼손 제어에 필요한 l_hj_<finger>_j 관절이 없어 사용 불가하므로
    # left/grasp_v2 선례를 따라 bi_rl 자산으로 전환했다.)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # ★08.17 openarm_tesollo_bi_rl → openarm_tesollo_bi_s_rl (DG-5F → DG-5FS).
            # 이름은 동일하나 기구학 전면 재정의(축 0 0 1, 마디 0.0388→0.0334,
            # palm 0.0698→0.015, 한계 10/20 변경) → HAND_*_POSE·워크스페이스·PCA·
            # warm state·체크포인트 전부 무효. Fabrics 는 P0(95caa19)에서 갱신됨.
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_bi_s_rl/openarm_tesollo_bi_s_rl.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                # 제어 왼팔: right ARM_START_POSE의 부호 미러 [-0.5,-0.1,-0.4,0.60,0.2,0,0]
                "l_aj_1": -0.5,
                "l_aj_2": -0.1,
                "l_aj_3": -0.4,
                "l_aj_4":  0.60,
                "l_aj_5":  0.2,
                "l_aj_6":  0.0,
                "l_aj_7":  0.0,
                # 제어 왼손 approach 자세(오른손 approach 의 부호 미러): thumb _2=+1.57,_3=+0.5
                "l_hj_thumb_1": 0.0, "l_hj_thumb_2": 1.57, "l_hj_thumb_3": 0.5, "l_hj_thumb_4": 0.0,
                "l_hj_index_1": 0.0, "l_hj_index_2":  0.0,  "l_hj_index_3":  0.0, "l_hj_index_4": 0.0,
                "l_hj_middle_1": 0.0, "l_hj_middle_2":  0.0,  "l_hj_middle_3":  0.0, "l_hj_middle_4": 0.0,
                "l_hj_ring_1": 0.0, "l_hj_ring_2":  0.0,  "l_hj_ring_3":  0.0, "l_hj_ring_4": 0.0,
                "l_hj_pinky_1": 0.0, "l_hj_pinky_2":  0.0,  "l_hj_pinky_3":  0.0, "l_hj_pinky_4": 0.0,
                # 고정 오른팔 + 오른손 rest
                **RIGHT_ARM_REST_JOINT_POS,
            },
        ),
        actuators={
            # head pan/tilt 카메라 관절(신규 USD). revolute라 DOF로 잡히므로
            # actuator 커버리지 필수(없으면 크래시/자유회전). 기본각 0 고정 hold.
            # obs/action은 이름 기반이라 head 미포함 유지 → 차원 불변.
            "head_camera": ImplicitActuatorCfg(
                joint_names_expr=["head_j_(pan|tilt)"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # ★08.16 pour 게인 정합(right 이식, 좌우 통일): kp/kd를 pour 학습값과 동일하게
            #   (팔 400/80, abduction 200/35). 근거=right e2c0e7a 실측 — 캘리브 유연팔(67.6/6.4)
            #   +강한 abduction(600/40)에서 형성된 파지가 pour 게인(400/80·200/35)에서 미끄러져
            #   컵 이탈·bead 유실. 수집 시점 치환만으론 부족 — 학습 물리 = 소비 물리 정합.
            #   both/pour_sensor 의 왼팔도 l_aj 400/80 이라 좌팔 소비 게인과도 일치한다.
            #   friction 은 07.29 real2sim 실측값 유지(실물 마찰은 실재하므로 보존).
            #   캘리브 원본(되돌릴 때 사용): kp 67.587/66.979/12.019, kd 6.376/5.635/2.154
            #   — assets/robot/openarm_tesollo_sensor_rl/calibration/right_arm.json(좌우 동일 HW).
            "left_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-3]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.213,
            ),
            "left_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_4"],
                stiffness=400.0,
                damping=80.0,
                friction=0.493,
            ),
            "left_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[5-7]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.151,
            ),
            # 손 stiffness/damping: pour-v5/6 검증값 채택. 기존 30/5(물렁)은 엄지 _3/_4가
            # 컵을 감을 때 반력이 엄지 대향을 뒤로 밀어냄(play 렌더 관찰). 단단히 유지.
            # ★08.16 손 게인 전면 재설정 (S1~S4 실측, right 와 동일 — HW 좌우 동일).
            #   구 400/60 은 토크 포화 영역: effort limit 7.5 N·m vs 요구 143 N·m(19배),
            #   접촉 관절 80.6% 상시 포화 → 파지력 제어 원리적 불가 + 실기(effort p=1.5) 도달 불가.
            #   S1 포화 이탈선 k≈5 / S4 kd 스윕 → kd 2.0 (포화 0.8% + 최저 채터).
            #   실기 d=0.0 이므로 damping 은 기계마찰의 sim 대역품 — r2s 후 armature 로 교체할 것.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_1"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_2"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_3"],
                stiffness=5.0,
                damping=2.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_4"],
                stiffness=5.0,
                damping=2.0,
            ),
            # 오른손 tesollo (bi USD): 학습 미사용, rest 자세 유지용 hold (고정)
            "tesollo_right_hand": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                stiffness=400.0,
                damping=60.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # Actor: fingertip 5개 개별 센서 (Cup-only, real-compatible FT sensor)
    # Critic: distal + middle 통합 센서 (sim-only)
    # -----------------------------------------------------------------------
    left_tip_contact_links: tuple = (
        "l_hl_thumb_tip",
        "l_hl_index_tip",
        "l_hl_middle_tip",
        "l_hl_ring_tip",
        "l_hl_pinky_tip",
    )

    # distal/middle 도 tip 처럼 Cup-only 필터(force_matrix_w). 무필터(net_forces)면
    # 손가락이 컵이 아닌 다른 손가락/palm 에 self-contact 해도 grip 으로 잡혀,
    # 엄지가 컵을 안 닿고도 success(num_grip>=5)를 거짓 충족하던 버그를 차단한다.
    #
    # 2026-07-26 MultiAsset: 8종 전부 visdex 표준 "Cup/baseLink" rigid body 구조로 통일
    # (물체 자산은 좌우 공유이므로 right 원본과 동일).
    object_contact_filter: tuple = (
        "/World/envs/env_.*/Cup/baseLink",
    )
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_hl_[a-z]+_4",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup/baseLink"],
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_hl_[a-z]+_3",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup/baseLink"],
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # 컵 설정 — 2026-07-26: 단일 cup_big_sdf → MultiAsset 8종(_GRASP_OBJECT_SPAWN).
    # prim 이름 "Cup" 은 유지(ContactSensor filter·env.py 참조 재사용). 좌우 공유.
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=_GRASP_OBJECT_SPAWN,
    )

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = LEFT_ACTUATED_JOINT_NAMES
    fixed_arm_joint_names: list = RIGHT_ARM_AND_GRIPPER_JOINT_NAMES
