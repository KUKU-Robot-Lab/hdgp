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
    {"id": "shaker_body",  "usd_path": _os.path.join(_SDF_ASSET_ROOT, "shaker_body_rl.usd"), "scale": (1.0, 1.0, 1.0), "mass": _BASE_OBJECT_MASS},
    {"id": "large_5_cyl",     "usd_path": _os.path.join(_VISDEX_ROOT, "large_5_cyl", "large_5_cyl.usd"),   "scale": (1.0, 1.0, 1.0), "mass": _BASE_OBJECT_MASS},
    {"id": "large_8_cyl_h12", "usd_path": _os.path.join(_VISDEX_ROOT, "large_8_cyl", "large_8_cyl.usd"),   "scale": (1.0, 1.0, 1.5), "mass": _BASE_OBJECT_MASS},
    {"id": "large_12_cyl_h12", "usd_path": _os.path.join(_VISDEX_ROOT, "large_12_cyl", "large_12_cyl.usd"), "scale": (1.0, 1.0, 2.4), "mass": _BASE_OBJECT_MASS},
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

    # -----------------------------------------------------------------------
    # Delta palm action (pregrasp 기준 상대 오프셋)
    # action=0 → pregrasp 위치 유지, action=±1 → pregrasp ± delta
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.15   # ±0.15m per axis
    palm_delta_rot_deg: float = 20.0   # ±20° per axis

    # -----------------------------------------------------------------------
    # Reward 파라미터
    # -----------------------------------------------------------------------
    # RH56F1 shared grasp-v2 reward contract.
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    lift_reward_weight: float = 30.0
    stabilize_weight: float = 10.0
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02
    grasp_xy_threshold: float = 0.025
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
    # ★래치 후 손가락 동결 해제(재조임 권한). 파지력은 stiffness×(target−actual) 오버슈트가
    # 전부인데 동결이 첫 접촉(0.1N)에서 걸려 오버슈트≈0 으로 고정된다 — 외란이 와도 더 조일
    # 수단이 없다. 래치 후 동결을 풀어 정책이 폐쇄 진행도를 더 밀 수 있게 한다.
    # ⚠️배포 동기화 필수: sim2real/scripts/grasp_action_decoder.py 의 GraspFingerController.
    retighten_after_latch: bool = True
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
    wrench_max_accel: float = 15.0      # m/s² ADR 종점. 컵 154g 기준 최대 2.3N = 중력(1.51N)의 1.5배(08.16 10→15 상향)
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
    hold_rotation_perturb_max_accel: float = 12.0

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

    # design §위치 ADR: spawn xy_range 0.02→0.08 점진 확대(초기 좁게 학습 후 확장).
    # grasp_adr.get_param("spawn","xy_range")로 GraspRightEnv._reset_idx가 조회.
    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "xy_range": (0.02, 0.08),
        },
        # 08.15 외란 커리큘럼: 0에서 시작해 increment마다 선형 증가(급격 도입 시 grip 붕괴 방지).
        # spawn과 동일 카운터 공유(grasp_v1 ADR 구조) — 학습 곡선에서 latch 후 붕괴가 보이면
        # adr_increment_interval 상향으로 램프 속도 완화(스케줄 자체 변경은 하지 않음).
        "object_wrench": {
            "max_linear_accel": (0.0, 15.0),
        },
        "hold_rotation": {
            "max_accel": (0.0, 12.0),
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
            # ★0.5~4.0× (사용자 지시: pour 에 국한하지 말고 광범위하게, 컵 무게 외
            # 최소 300g 을 여유롭게 파지). 기본질량 0.134kg 통일이므로 전 물체가
            # **0.067 ~ 0.536 kg** 을 동일하게 커버 → 최대 적재량 = 0.536-0.134 = **402g**
            # (요구 300g 대비 +34% 여유). 상한 0.536kg 은 grasp_v2 가 ADR 만렙에서 실제로
            # 성공시킨 최대 절대질량(0.1787kg×3.0)과 정확히 같아, 외삽이 아닌 실증 영역이다.
            # pour 실조건 0.154kg(비드 만재)은 이 분포의 하위 구간에 여유롭게 포함된다.
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
    object_spawn_x_center: float = 0.27   # demo 데이터와 일치 (0.40→0.27)
    object_spawn_y_center: float = -0.10  # demo 데이터와 일치 (-0.15→-0.10)
    # object_spawn_z: cup_big(반높이 0.08881200104951859) 기준 테이블 안착 높이. 2026-07-26부터
    # GraspRightEnv가 이 값에서 테이블 표면 z(=object_spawn_z - cup_big 반높이)를 역산해
    # 물체별 반높이를 더한 object_spawn_z_buf(N,)를 파생한다 — reset에서 물체별 텐서를 쓴다.
    object_spawn_z:        float = 0.297
    # object_spawn_xy_range: enable_adr=False 일 때만 쓰이는 fallback(±6cm). 기본은
    # enable_adr=True → adr_custom_cfg["spawn"]["xy_range"](0.02→0.08)가 실제 범위를 결정.
    object_spawn_xy_range: float = 0.06
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
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_1"],
                stiffness=200.0,   # 08.15 pour 정합: 600→200 (pour_v1 abduction 200/35와 동일). 600 이력=roll 억제 절충이었으나 pour 소비 게인과 불일치가 파지 미끄러짐 유발(e2c0e7a).
                damping=35.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_2"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_3"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_4"],
                stiffness=400.0,
                damping=60.0,
            ),
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
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
