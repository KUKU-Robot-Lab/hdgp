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

"""환경 설정: 5g_grasp_right_v1

v7: Fabrics 팔 학습(6D palm) + per-finger lerp(5D) + sim2real 가능 obs
- Action: 11D (6D palm pose + 5D per-finger lerp)
- Observation: actor 114D(106 base + 8 물체 onehot) / critic 151D (asymmetric)
- Episode: Grasp phase (Fabrics arm + finger 정책) + right-grip lift-wait (frozen hand)
- Contact: fingertip FT sensor (actor, real-compatible) + distal/middle sensors (critic only)

2026-07-26 MultiAsset(8종)+DR 이식 (design: docs/superpowers/specs/2026-07-26-
tesollo-grasp-v1-multiasset-dr-design.md): 단일 cup_big_sdf → cup_big×4 scale +
shaker_body + cyl 3종(높이 12cm 통일). reward/성공판정/side approach 로직은 불변.
"""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
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
# ★[both/pour_v1 대응] warm 수집 시 컵을 채울 비드. 정의는 common 한 곳에만 둔다
#   (pour 와 물성이 같아야 bead_state 복원이 물리적으로 성립).
from openarm.common.bead_assets import (
    DEFAULT_BEAD_COUNT as _DEFAULT_BEAD_COUNT,
    make_beads_cfg as _make_beads_cfg,
)
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_VISDEX_ROOT = _os.path.join(_ASSETS_DIR, "visdex_objects", "USD")

# ---------------------------------------------------------------------------
# 물체 구성 — 8종 (design §물체 구성). 논리 ID(=onehot·bbox 조회 키) 순서가
# env_id % 8 결정론적 배정과 MultiAssetSpawnerCfg assets_cfg 순서를 동시에 정한다.
# cup_big×4: 동일 USD, scale만 등방(0.85/1.0/1.15/1.30)으로 다르게.
# shaker_body: cocktail 자산(다른 USD 뱅크, metersPerUnit=1 정상).
# cyl 3종: 높이 12cm 통일(large_5_cyl 그대로, large_8/12_cyl은 z-scale로 승격).
# ---------------------------------------------------------------------------
# 전 물체 공통 기본 질량 [kg]. pour_v1 실컵(cup_big_sdf, density 파생)과 동일.
# ADR mass DR 의 곱셈 기준이 되므로 여기를 바꾸면 실효 질량 범위 전체가 이동한다.
_BASE_OBJECT_MASS: float = 0.134

# ★08.16 SDF 콜라이더 자산(scripts/tools/make_sdf_grasp_assets.py 산출).
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
    # ★기본 질량 8종 통일 = _BASE_OBJECT_MASS(0.134kg, pour 실컵 질량).
    #   ① pour_v1 실컵(cup_big_sdf, density 파생 ≈0.134kg)과 학습 질량 정합.
    #   ② collect_sdf_cup_assets=True 경로에서 SDF 컵 질량이 scale s³로 몰래 바뀌는 함정 차단
    #      (질량은 scale 무관 고정 — cup-shrink 계획문서의 force-ratio DR 붕괴 경고).
    #   ③ 08.16 전 물체로 확대: USD 기본질량이 제각각(shaker 0.263 / 원기둥 0.100)이면 동일한
    #      ADR scale 을 걸어도 물체마다 절대 질량이 최대 2.6배 벌어져, 무거운 물체만 미검증
    #      외삽 영역(shaker 3.0×=0.79kg, grasp_v2 실증 상한 0.536kg의 1.5배)으로 튄다.
    #      기본질량을 통일하면 **모든 물체가 동일한 절대 질량 범위 전체를 학습**한다
    #      (다양성은 ADR DR 이 매 에피소드 샘플링으로 제공 — 오히려 커버리지가 넓어짐).
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

    ★08.16 ADR 스케줄 대상으로 전환(grasp_v2 방식): 아래 값은 **초기(중립) 범위**이고,
    ADR increment 마다 adr_physics_cfg 의 terminal 로 선형 확장된다. 구조상 정적 고정
    범위는 "처음부터 최대 난이도"라 커리큘럼이 아니었다(질량·마찰이 ADR 밖에 있었음).
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
            # 초기=중립(원 질량 그대로). terminal (0.5,3.0)=grasp_v2 동일 —
            # pour 실조건(컵 0.134+비드 0.02=0.154kg=1.54×)을 여유 있게 포함.
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v1 환경 설정."""

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
    #   구 동작: 리셋마다 컵 **참값**으로 pregrasp(cup + pregrasp_offset)를 계산해
    #   팔을 그 자세로 텔레포트했다. 실기에서는 컵 위치가 지각 결과이므로
    #   "참값으로 계산한 자세에서 시작"은 재현할 수 없고, 컵 y=-0.10 일 때는
    #   손이 스폰 박스 **안**(y=-0.22)에서 출발해 side 접근도 성립하지 않았다.
    #   새 동작: 컵과 무관한 고정 홈 palm 자세에서 출발.
    #   ★2026-08-19(A0): 물리 리셋만 홈이고 **액션 기준점은 컵-정준 pregrasp**(env
    #   _reset_idx) — action=0 이면 Fabrics 가 홈에서 정렬된 pregrasp 까지 스스로 접근.
    #   구("접근 전 구간을 정책이 학습")는 tilted-palm 2지 국소최적로 실패(lstm_test1).
    #
    #   홈 자세 근거 (URDF openarm_tesollo_bi_s_rl FK/IK 실측, base=env 원점):
    #     위치오차 0.00mm · 자세오차 0.0deg · 관절한계 여유 ≥0.36rad
    #     컵 스폰 박스까지 80mm 밖 · 팔 최저 z=0.378(테이블면 0.208)
    #     헤드 카메라(head_cam_view [0.037,0,0.852]) → 컵 9개 표본 시선 최근접 93mm
    #       (링크 반경 45mm 기준) → 초기 프레임에서 컵 가림 없음
    reset_from_fixed_home: bool = True
    # [x, y, z, ez, ey, ex] — 회전은 도(deg), env 에서 라디안 변환.
    reset_home_palm_pose: tuple = (0.28, -0.38, 0.42, 90.0, 0.0, 90.0)

    # cache_pregrasp_reset: reset_from_fixed_home=True 면 쓰이지 않는다
    # (홈이 단일 자세라 grid 캐시가 필요 없다 — 시작 시 IK 1회만 푼다).
    cache_pregrasp_reset:  bool  = True
    pregrasp_cache_xy_range: float = 0.08   # 캐시 grid 반경(고정) — object_spawn_xy_range(ADR 초기값)와 별개
    pregrasp_offset_x:     float = -0.06
    pregrasp_offset_y:     float = -0.07
    pregrasp_offset_z:     float = 0.00
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Demo reset (pour_v1_a11~a20 grasp start and lift target)
    # -----------------------------------------------------------------------
    # 2026-07-26 MultiAsset 다물체 이식: demo pose는 단일 컵 전용 고정 자세라 8종 물체(높이·
    # 위치 ADR)에 부적합 → False(rh56f1/grasp_v2 동일 결정, "grasp_v2: cup demo pose 는
    # 다물체에 부적합 → demo-free reset 사용"). off 시 아래 FABRICS pregrasp cache 경로 사용.
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
    # cup_grasp_z_offset: cup_big(높이 17.8cm) 기준 보정값. 2026-07-26부터 GraspRightEnv가
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
    # Delta palm action (리셋 기준 자세 대비 상대 오프셋)
    # action=0 → 기준 자세 유지, action=±1 → 기준 ± delta
    # -----------------------------------------------------------------------
    # ★2026-08-19(A0) y 0.35 → 0.15 원복 (축 균일). 구 y=0.35 는 "액션 기준점=홈"이라
    #   홈 y=-0.38 에서 컵까지 0.28m 를 액션으로 덮어야 했던 보정이었다. 이제 기준점이
    #   컵-정준 pregrasp(env _reset_idx)라 action=0 이 이미 컵 옆이고, delta 는
    #   파지 직전 미세조정 + 잔차 보정 용도만 남는다 → 축 균일 0.15 로 해상도 회복.
    palm_delta_xyz:     tuple = (0.15, 0.15, 0.15)   # ±m, (x, y, z)
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
    # ★2026-08-20 0.025 → 0.05 (audit ACCEPT). approach 의 xy 페널티 사각지대.
    #   test4 ep6500 probe 실측: 밀림 0.084 에서 페널티 25*(0.084-0.025)=1.48 이
    #   접근 보상(최대 2.0, 실효 ~0.6)을 압도해 **reward/approach 가 상시 음수(-0.0995)**.
    #   음수 구간은 "빨리 끝내는 게 이득"이라 정책이 손끝 4지+엄지가 닿자마자 latch(60스텝)
    #   → latch 후 palm 은 스크립트 램프 지배라 palm-컵 9.5cm 에 영구 고정 → PIP 가 이미
    #   0.994 포화라 손끝만 접촉 → 리프트 중 원위 접촉 1.52→0.16 붕괴(wrap 0.036).
    #   0.05 = 파지 접촉에서 불가피한 밀림은 면제, 그 이상은 25.0 기울기 그대로.
    #   가중치 25.0 은 불변이고 lift/success 의 disp 감쇠(limit 0.08)도 그대로다.
    #   ⚠ 대조군: 동일 margin 의 Stage1(텔레포트)은 정상(distal 1.86) — 접근 구간이 없어
    #   음수 approach 를 겪지 않았을 뿐. 고정홈 전용 결함.
    grasp_xy_threshold: float = 0.05
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
    # ★2026-08-19(A3, audit ACCEPT) 3→4 + env 에서 엄지 접촉 AND 추가.
    # latch 는 비가역 + approach/grasp shaping 차단 스위치인데, 구 3지(엄지 무관)는
    # "엄지+소지+1지" 얕은 latch 를 허용해 shaping 을 조기·영구 소멸시켰다
    # (Stage1 텔레포트 실측: wrap_at_latch 0.03~0.05 8000ep 고착 = primary bottleneck).
    # 4+thumb = success_min_grip_fingers(4)+thumb AND 와 일원화 — latch 시점=성공 요건
    # 충족 시점. Stage1 실측 grip_finger_count 4.5·thumb_cup 0.94 라 도달 가능 임계.
    # 폴백(ep6000 lifted_rate 0 시): 4→3 완화 1단계만, thumb AND 는 유지.
    lift_start_min_grip_fingers: int = 4  # lift 진입: 아무 마디(tip|mid|distal) 닿은 손가락 수 임계 + 엄지 AND(env).
    success_min_grip_fingers: int = 4  # success 그립 손가락 수(엄지-컵 접촉 AND 강제와 함께 사용). 5(전손가락 동시)는 wrap 진동 이력.
    grasp_ready_hold_steps: int = 8   # 접촉 N개를 연속 hold하면 lift 래치 (잡으면 바로 리프트)
    lift_start_min_envelope_fingers: int = 0  # latch 인벨롭 게이트 제거(0=비활성). envelope은 grasp/lift 보상 credit으로 유도(hard 게이트 대체)
    finger_close_speed: float = 0.05  # ① 접촉-게이트 적응 폐쇄: 손가락 폐쇄 진행 속도/step (중간마디 접촉 시 동결)
    # ★couple_four_fingers(08.15, left 08.02 이식 — 좌우 통일): 검지~소지(4지)를 공통신호(평균)로
    # 묶어 "특정 손가락만 안 닫힘"을 표현 불가하게 함 → 3지 국소최적 원천 차단. 엄지(0)는 opposition
    # 위해 독립. per-joint 접촉게이트(g3/g4)는 유지되어 물체 형상에 개별 적응(닿는 곳서 동결).
    # left fresh 학습(lstm_test9)서 검증(grasp_v2 left ADR50 0.908도 동일 기법). reward/obs/action
    # 차원 불변(action 전처리만). ⚠️기존 right per-finger 체크포인트와 상충하므로 warmstart 금지
    # — left 08.03 이력: couple+per-finger warmstart 조합이 0.565 정체 유발. fresh 학습 전제.
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
    # 물체 외란 wrench (08.15, DEXTRAH apply_object_wrench — rh56f1 grasp_v1 구현 이식)
    # pour 연결 강건화: pour_v1은 hold 후 비드 20×1g 소환(+15% 계단 하중)+슬로싱(자유 구름)
    # +deep tilt 회전을 가하는데, 기존 grasp_v1 학습은 외란 경험 0회(e2c0e7a: pour 검증 9~15%).
    # force = mass × accel × 랜덤방향(등방), torque = force × torsional_radius × 랜덤방향.
    # 게이트: palm이 물체 반경(wrench_hand_dist_threshold) 내일 때만 인가(DEXTRAH 원본 게이트).
    # 크기는 ADR 커리큘럼 0→max (adr_custom_cfg["object_wrench"]) — 급격 도입 시 grip 붕괴 방지.
    # -----------------------------------------------------------------------
    wrench_enable: bool = True
    # ★08.16 15→8 하향(S2 실측). 실제 인가는 accel = max·rand(0,1), 질량도 per-reset 독립
    # 샘플이므로 하중 = m·(9.81+a) 의 최악 코너는 m=0.268·a=8 → 4.77 N. 측정된 파지 능력
    # (총 5 N, k=5 팁핀치)에 딱 들어온다. 구 15 는 6.65 N 이상을 요구해 코너가 물리적으로 불가.
    wrench_max_accel: float = 8.0       # m/s² ADR 종점
    wrench_torsional_radius: float = 0.045  # m — pour source_inner_radius(0.041) 비드 쏠림 레버암 재현 (grasp_v2 0.03보다 큼)
    wrench_trigger_every: int = 15      # step(0.25s @60Hz)마다 새 랜덤 wrench — 비드 슬로싱 주기 근사(08.16 20→15)
    wrench_hand_dist_threshold: float = 0.3  # m — palm-물체 거리 게이트
    # 회전 외란(08.15, Exp4-A 물리판·reward-audit ACCEPT): lift latch 이후 env에 수평 랜덤축
    # torque를 추가 인가 — pour deep-tilt 중 컵이 손안에서 회전하려는 하중을 재현(회전에 안 놓는
    # grip 단련). reward/gate 불변(물리만) — 실패 신호는 기존 tipping 종료(60°)+success upright
    # 게이트가 제공. slip 페널티(Exp4-B)는 기존 신호 중복+latch 회피 gradient 위험으로 REJECT.
    hold_rotation_perturb_enable: bool = True
    # ★08.16 6→12 상향(pour deep-tilt 정합): pour 110° 기울임에서 컵+비드(0.154kg)가
    # 파지점 레버암 ~0.05m 로 만드는 중력 토크 ≈ 0.154×9.81×0.05 = 0.076 N·m.
    # a=12·r=0.045 → τ = 0.154×12×0.045 = 0.083 N·m 로 그 수준을 재현한다(구 6=0.042,
    # 실제 tilt 하중의 절반에 불과했음).
    # ★08.16 12→8 하향: 위 wrench 와 동일 근거(구 값은 포화 게인 전제). pour deep-tilt
    # 재현 목표치(τ≈0.076 N·m)는 질량 종점 2.0 과 조합하면 0.268×8×0.045 = 0.096 N·m 로
    # 여전히 충족한다 — 난이도를 깎은 게 아니라 질량 쪽으로 옮겨 담았다.
    hold_rotation_perturb_max_accel: float = 8.0

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    # ★08.16 케이던스 완화(200→1000): 질량·마찰·wrench·회전이 모두 이 카운터 하나를
    # 공유하게 되면서 구 케이던스(50증분×200step≈epoch 500 만렙)는 "커리큘럼 없이 즉시
    # 최대 난이도"와 같아졌다(lstm_test7 실측: ep~500 만렙 도달). 1000step 이면 만렙까지
    # 최소 ~1,560 epoch — 20k 런에서 실질 커리큘럼이 된다. 페이싱은 이 interval 이 담당.
    adr_increment_interval: int  = 1000
    # threshold 는 "무너지는 중엔 난이도를 올리지 않는다"는 가드(순간 성공률 기준).
    # 0.25: 순간값(success_flag 평균)은 에피소드 단위 성공률(test7 0.82)보다 구조적으로
    # 낮다 — 에피소드 중 success 유지 구간이 일부이기 때문. grasp_v2 는 0.4 를 쓰지만
    # v2 메트릭(goal 반경 도달)보다 v1 success_flag(latch AND grasped AND valid)가 엄격해
    # 그대로 쓰면 램프 미시작 위험. 실측은 TB adr/trigger_metric 으로 보고 재보정한다.
    adr_trigger_threshold: float = 0.25

    # design §위치 ADR: spawn 반경을 점진 확대(초기 좁게 학습 후 확장).
    # grasp_adr.get_param("spawn","x_range"/"y_range")로 GraspRightEnv._reset_idx가 조회.
    # ★2026-08-18 스칼라 xy_range → 축별 분리. x 와 y 의 요구 폭이 다르기 때문:
    #   목표 스폰 범위 x 0.25~0.35(±0.05), y -0.30~-0.10(±0.10).
    #   구 스칼라로는 둘 중 하나만 맞출 수 있었다.
    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "x_range": (0.02, 0.05),
            "y_range": (0.02, 0.10),
        },
        # 08.15 외란 커리큘럼: 0에서 시작해 increment마다 선형 증가(급격 도입 시 grip 붕괴 방지).
        # spawn과 동일 카운터 공유(grasp_v1 ADR 구조) — 학습 곡선에서 latch 후 붕괴가 보이면
        # adr_increment_interval 상향으로 램프 속도 완화(스케줄 자체 변경은 하지 않음).
        # ★08.16 종점 하향(15→8, 12→8): 아래 wrench_max_accel 주석 참조. 구 값은 포화 게인
        # 전제였고, 실측 파지 능력(총 5N)에 대해 질량 종점 2.0 과 조합하면 4.77N 으로 들어온다.
        "object_wrench": {
            "max_linear_accel": (0.0, 8.0),
        },
        "hold_rotation": {
            "max_accel": (0.0, 8.0),
        },
    })

    # ★08.16 물리 DR ADR (grasp_v2 방식 이식): EventCfg 의 초기(중립) 범위를
    # increment 마다 아래 terminal 로 선형 확장한다. 질량·마찰이 정적 고정 범위였던
    # 구조를 커리큘럼으로 전환 — 초기엔 원 질량/마찰로 파지를 익히고, 성공할수록
    # 무겁고 미끄러운 물체로 확장. terminal 값은 grasp_v2(cfg:480-487)와 동일.
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
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    # ★2026-08-18 x 중심 0.27 → 0.30 (목표 범위 0.25~0.35 의 중앙).
    object_spawn_x_center: float = 0.30
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
    object_spawn_y_center: float = -0.20  # ∓0.20 (구 -0.10: demo 데이터 일치값)
    # object_spawn_z: cup_big(반높이 0.08881200104951859) 기준 테이블 안착 높이. 2026-07-26부터
    # GraspRightEnv가 이 값에서 테이블 표면 z(=object_spawn_z - cup_big 반높이)를 역산해
    # 물체별 반높이를 더한 object_spawn_z_buf(N,)를 파생한다 — reset에서 물체별 텐서를 쓴다.
    object_spawn_z:        float = 0.297
    # object_spawn_{x,y}_range: enable_adr=False 일 때만 쓰이는 fallback. 기본은
    # enable_adr=True → adr_custom_cfg["spawn"]["x_range"/"y_range"]가 실제 범위를 결정.
    # ★2026-08-18 스칼라 object_spawn_xy_range(0.06) 를 축별로 분리하고 ADR 종점과 맞췄다.
    object_spawn_x_range: float = 0.05
    object_spawn_y_range: float = 0.10
    # 활성 물체군(8종, MultiAsset assets_cfg 순서와 일치) — onehot·bbox 조회·env_id%8 배정용.
    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
    object_bbox_path: str = _os.path.join(_ASSETS_DIR, "object_bbox.json")

    # -----------------------------------------------------------------------
    # Warm-state export (grasp 성공 → 디스크 캐시 → pour warmstart 재사용)
    # -----------------------------------------------------------------------
    # 학습 루프에는 영향 없음 (기본 False). collect 스크립트/play 에서만 True.
    # success 이후 오른손 grasp arm pose 를 유지하고 joint7 만 lift-wait 로 이동한 상태를 저장한다.
    # demo cup/phase 구분을 신뢰하지 않고 실제 sim 손/컵 grasp 결과를 그대로 유지한다.
    # 손가락 접촉은 기본 2개 이상, lift-wait arm match 는 기본 1 step 만 요구한다.
    enable_warm_state_export: bool = False
    warm_state_export_path: str = _os.path.normpath(
        _os.path.join(_HDGP_ROOT, "..", "datasets", "grasp_warm_v1.hdf5")
    )
    warm_state_target_count: int = 2048
    warm_min_contacts: int = 2
    warm_contact_stable_steps: int = 1
    warm_lift_wait_arm_tol: float = 0.035
    warm_lift_wait_hold_steps: int = 1
    # ⚠2026-08-19 이후 미사용 — 수직 palm 리프트(lift_height_delta)로 대체.
    #   구 j7 리프트는 palm 자세를 17.76° 회전시켜 쥔 컵을 같이 기울였다.
    lift_wait_joint7_delta: float = 0.31
    warm_cup_upright_min: float = 0.90   # legacy override 호환용; lift-wait export 에서는 미사용
    warm_j7_min: float = 0.20
    warm_j7_max: float = 1.50

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
    # 로봇 설정 (openarm_tesollo_sensor_rl.usd: 통일 네이밍 r_aj/r_hj/r_hl, r_hl_*_tip ContactSensor 포함)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # ★08.18 grasp_sensor: sim2real(a1) 전용 태스크 — 로봇은 **openarm_tesollo_sensor_rl**
            # (DG-5F 우손 + 좌측 2-DOF 그리퍼 + D435i 헤드) 고정. grasp_v1(bi_s_rl/DG-5FS)의
            # 08.18 세팅(고정 홈 리셋·축별 palm_delta·스폰 박스·ADR 물리)을 그대로 가져오되
            # 자산만 되돌렸다. 두 손은 조인트 이름이 같고 기구학이 다르므로(축·마디·palm 오프셋)
            # bi_s_rl 산 warm 뱅크/체크포인트와 절대 혼용 금지 — 뱅크 meta/robot_usd 가드가 막는다.
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"),
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
                "r_aj_1":  0.5,
                "r_aj_2":  0.1,
                "r_aj_3":  0.4,
                "r_aj_4":  0.60,
                "r_aj_5": -0.2,
                "r_aj_6":  0.0,
                "r_aj_7":  0.0,
                "r_hj_thumb_1": 0.0, "r_hj_thumb_2": -1.57, "r_hj_thumb_3": -0.5, "r_hj_thumb_4": 0.0,
                "r_hj_index_1": 0.0, "r_hj_index_2":  0.0,  "r_hj_index_3":  0.0, "r_hj_index_4": 0.0,
                "r_hj_middle_1": 0.0, "r_hj_middle_2":  0.0,  "r_hj_middle_3":  0.0, "r_hj_middle_4": 0.0,
                "r_hj_ring_1": 0.0, "r_hj_ring_2":  0.0,  "r_hj_ring_3":  0.0, "r_hj_ring_4": 0.0,
                "r_hj_pinky_1": 0.0, "r_hj_pinky_2":  0.0,  "r_hj_pinky_3":  0.0, "r_hj_pinky_4": 0.0,
                **LEFT_ARM_REST_JOINT_POS,
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
            # ★08.15 pour 게인 정합(사용자 지시): kp/kd를 pour_v1 학습값과 동일하게(팔 400/80,
            #   abduction 200/35). 근거=e2c0e7a 실측 — 캘리브 유연팔(67.6/6.4)+강한 abduction(600/40)
            #   에서 형성된 파지가 pour 게인(400/80·200/35)에서 미끄러져 컵 이탈·bead 유실.
            #   수집 시점 치환(collect_pour_matched_gains)만으론 부족 — 학습 물리 = 소비 물리 정합.
            #   friction은 07.29 real2sim 실측값 유지(pour cfg에 friction 항 부재 = PhysX 기본 0과
            #   다르지만, 실물 마찰은 실재하므로 보존). 캘리브 원본:
            #   assets/robot/openarm_tesollo_sensor_rl/calibration/right_arm.json (kp 67.587/66.979/12.019,
            #   kd 6.376/5.635/2.154) — 되돌릴 때 이 값 사용. group 경계=CALIBRATION.md(r_aj_[1-3]/4/[5-7]).
            "right_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-3]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.213,
            ),
            "right_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_4"],
                stiffness=400.0,
                damping=80.0,
                friction=0.493,
            ),
            "right_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[5-7]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.151,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # 손 stiffness/damping: pour-v5/6 검증값 채택. 기존 30/5(물렁)은 엄지 _3/_4가
            # 컵을 감을 때 반력이 엄지 대향(_2=-1.57)을 뒤로 밀어냄(play 렌더 관찰). 단단히 유지.
            # ★08.16 손 게인 전면 재설정 (S1~S4 실측). 구 400/60 은 **토크 포화 영역**이었다:
            #   URDF effort limit 7.5 N·m 인데 접촉 시 요구 토크가 143 N·m(한계의 19배),
            #   접촉 관절의 80.6% 가 상시 포화(probe_torque_saturation). 즉 위치 제어가 아니라
            #   "모터 최대 토크로 계속 쥐는 bang-bang 그리퍼"로 동작했고, 그래서 목표를 더 밀어도
            #   힘이 1 N·m 도 안 올랐다(retighten·squeeze 실패의 구조적 원인).
            #   또한 실기는 effort 명령 + P게인 1.5(dg5f_right_controller.yaml)라 이 영역에
            #   구조적으로 도달 불가 — sim2real 이 성립하지 않았다.
            # S1(포화 이탈선): 포화% 83.7(k=400) → 24.1(k=5) → 8.6(k=2) → 0(k=1.5). 상한 k≈5.
            # S2(능력 곡선): k=5 에서 이탈가속 27.2 m/s² = 새 wrench 종점(8)의 3.4배. 게인을
            #   80배 낮춰도 외란 내성은 37→27 로만 감소(외란 내성은 파지력보다 기하에 지배됨).
            # S4(damping): kd=60 은 k=400 기준값. k=5 에서 kd 스윕 실측 —
            #   kd 6.71→포화 20.5%(감쇠항 자체가 토크를 포화시킴), kd≤0.5→정착속도 2배(채터),
            #   kd 2.0 이 포화 0.8% + 최저 채터로 양쪽 최적. 실기 d=0.0 이므로 이 damping 은
            #   실기 기계마찰의 sim 대역품 — r2s 복구 후 armature/joint friction 실측치로 교체할 것.
            # ★2026-08-19(A4) effort_limit_sim=1.5 N·m 신설 (구: 미설정 → URDF 7.5 그대로).
            #   실측(_probe_abd, ep16500 ckpt): 7.5 에서 전 손관절이 3~5 N·m 를 상시 유지하며
            #   ~2 N 컵을 쥐고, 그 합력 모멘트가 thumb_1(외전, 고정 0 명령)을 하드스톱
            #   (-0.384rad) 넘어 -0.94rad 까지 밀었다(viol=1.00, k=200 으로도 불변 = effort
            #   포화가 원인). 실기 DG-5F 연속토크는 ~1.5 N·m 수준이라 7.5 레짐은 sim2real
            #   무효 + 모터 무리(사용자 지시). 1.5 는 과압착 레짐을 물리적으로 제거한다.
            #   ⚠ ADR 만렙 질량(mass_scale_hi)이 1.5 로 유지 불가하면 질량 종점을 낮출 것
            #   (실기가 못 드는 질량은 학습 대상이 아니다) — Phase B probe 로 판정.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_1"],
                stiffness=5.0,
                damping=2.0,
                effort_limit_sim=1.5,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_2"],
                stiffness=5.0,
                damping=2.0,
                effort_limit_sim=1.5,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_3"],
                stiffness=5.0,
                damping=2.0,
                effort_limit_sim=1.5,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_4"],
                stiffness=5.0,
                damping=2.0,
                effort_limit_sim=1.5,
            ),
            # ★08.18 grasp_sensor(sensor_rl): 좌측은 2-DOF 프리즈매틱 그리퍼(l_hj_gripper_1/2).
            # 커버리지를 안 주면 무구동으로 자유이동한다. 이 태스크는 좌측을 쓰지 않으므로
            # rest(0.044 개방) 유지만 하면 된다 — right/pour_v1 과 동일 게인(400/80).
            # ⚠ l_hj_gripper_2 는 USD 에서 PhysX mimic(gearing=-1)으로 gripper_1 을 따른다.
            #   regex 를 [1-2] 로 두는 것은 기존 sensor_rl 태스크들(pour_v1/pour_sensor)과
            #   동일 규약 — 드라이브가 mimic 과 같은 목표(0.044)라 싸우지 않는다.
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_gripper_[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # Actor: fingertip 5개 개별 센서 (Cup-only, real-compatible FT sensor)
    # Critic: distal + middle 통합 센서 (sim-only)
    # -----------------------------------------------------------------------
    right_tip_contact_links: tuple = (
        "r_hl_thumb_tip",
        "r_hl_index_tip",
        "r_hl_middle_tip",
        "r_hl_ring_tip",
        "r_hl_pinky_tip",
    )

    # distal/middle 도 tip 처럼 Cup-only 필터(force_matrix_w). 무필터(net_forces)면
    # 손가락이 컵이 아닌 다른 손가락/palm 에 self-contact 해도 grip 으로 잡혀,
    # 엄지가 컵을 안 닿고도 success(num_grip>=5)를 거짓 충족하던 버그를 차단한다.
    #
    # 2026-07-26 MultiAsset: 8종 전부 visdex 표준 "Cup/baseLink" rigid body 구조로 통일.
    #   cup_big/large_*_cyl 은 원래 baseLink 중첩. shaker_body 는 원본이 Xform 루트에 직접
    #   RigidBodyAPI 인 비표준이라 GPU 에서 "/Cup/baseLink" 8종 중 7종만 매치("expected 8,
    #   found 7") + "/Cup" 루트 filter 는 GPU contact 미지원(07-26 실측). → fix_shaker_asset.py
    #   로 shaker 를 visdex 표준(baseLink[RB+MASS])으로 재이식 → 단일 filter 로 해결.
    object_contact_filter: tuple = (
        "/World/envs/env_.*/Cup/baseLink",
    )
    distal_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_4",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup/baseLink"],
        history_length=1,
        track_air_time=False,
    )

    middle_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_[a-z]+_3",
        filter_prim_paths_expr=["/World/envs/env_.*/Cup/baseLink"],
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # [warm 수집→pour 정합] True면 cup_big 4종의 usd_path를 cup_big_sdf.usd(SDF collider)로 치환.
    #   visdex cup_big은 dynamic body에서 convexHull fallback(오목 내부 채워짐) → 그 표면 기준
    #   파지 자세가 pour의 SDF 컵에서 관통/공극을 만들어 warm 리셋 시 컵 이탈·bead 유실
    #   (08.15 zero-action probe로 실증: SDF 수집 구캐시=유지 1.0, convexHull 신캐시=유실 다발).
    #   외벽 기하는 동일하므로 학습 정책의 zero-shot 파지에는 유효. 수집 CLI에서만 켠다.
    collect_sdf_cup_assets: bool = False
    # [warm 수집→pour 정합] True면 우팔/손 actuator 게인을 pour_sensor 값으로 치환:
    #   팔 r_aj_1-7 → kp400/kd80, abduction r_hj_*_1 → kp200/kd35.
    #   08.15부터 학습 기본 게인 자체가 pour 정합값(위 actuators 주석)이라 이 플래그는 no-op —
    #   과거 캘리브 게인 체크포인트로 수집할 때의 하위호환용으로만 유지.
    collect_pour_matched_gains: bool = False
    # -----------------------------------------------------------------------
    # ★[both/pour_v1 대응] warm 수집 시 source 컵에 **비드를 미리 채운다**.
    # -----------------------------------------------------------------------
    # 왜 필요한가: pour 의 warm 리셋은 파지 자세를 텔레포트한 뒤 손을 그 자세로 **동결**한다
    #   (`freeze_grasp_hand_during_episode=True`) — 즉 재조임하는 정책이 없는 **수동 스프링**이다.
    #   게다가 손 강성이 5.0(토크 포화 이탈선)으로 낮다. 빈 컵으로 만든 파지를 텔레포트한 뒤
    #   비드(20 × 1g)를 나중에 넣으면 그 하중을 수동으로 흡수해야 해서 컵을 놓칠 수 있다.
    #   비드를 **수집 시점에** 채워두면 정책이 하중이 있는 상태에서 파지를 형성하므로,
    #   기록되는 관절 목표가 "실제로 찬 컵을 들었던" 값이 된다.
    #
    # 학습에는 영향이 없다(기본 False) — 수집 CLI 에서만 켠다.
    #   `collect_grasp_v1_warm_states.py --robot tesollo_right --extra env.collect_with_beads=true`
    #
    # 접촉 판정 오염 없음: tip/distal/middle ContactSensor 는 `object_contact_filter`(컵 전용)로
    #   걸려 있어 비드 접촉이 `num_contacts` 에 들어가지 않는다 → warm 성공 판정 규약 불변.
    #
    # MultiAsset 주의: 비드는 전 env **동일 자산 하나**라 MultiAssetSpawner 의 env_i % N
    #   배정 규칙과 무관하다. 다만 씬 생성 순서는 컵과 같이 clone 이후로 둔다.
    collect_with_beads: bool = False
    # 비드 개수 — pour 의 `bead_count` 와 **반드시 같아야** warm 복원이 성립한다.
    collect_bead_count: int = _DEFAULT_BEAD_COUNT
    # 소환 배치는 cfg 파라미터로 두지 않는다 — `openarm.common.bead_assets.bead_offsets_in_cup`
    #   의 **검증된 단일 배치**를 쓴다. 배치를 조절 가능하게 만들었다가 값이 어긋나
    #   비드가 소환 순간 겹쳐 터진 사고(2026-08-17)가 있었다. 공용본만 신뢰한다.
    # 컵 설정 — 2026-07-26: 단일 cup_big_sdf → MultiAsset 8종(_GRASP_OBJECT_SPAWN).
    # prim 이름 "Cup" 은 유지(ContactSensor filter·env.py 참조 재사용).
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
    # ★[both/pour_v1 대응] 비드 컬렉션 설정. `collect_with_beads=True` 일 때만 씬에 올린다.
    #   cfg 객체 자체는 항상 만들어 두어도 무해하다(씬 등록을 안 하면 스폰되지 않는다).
    beads_cfg: RigidObjectCollectionCfg = _make_beads_cfg(_ASSETS_DIR, _DEFAULT_BEAD_COUNT)

    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
