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
_ACTIVE_OBJECT_SPECS: tuple[dict, ...] = (
    {"id": "cup_big_s085", "usd_path": _os.path.join(_VISDEX_ROOT, "cup_big", "cup_big.usd"), "scale": (0.85, 0.85, 0.85)},
    {"id": "cup_big_s100", "usd_path": _os.path.join(_VISDEX_ROOT, "cup_big", "cup_big.usd"), "scale": (1.00, 1.00, 1.00)},
    {"id": "cup_big_s115", "usd_path": _os.path.join(_VISDEX_ROOT, "cup_big", "cup_big.usd"), "scale": (1.15, 1.15, 1.15)},
    {"id": "cup_big_s130", "usd_path": _os.path.join(_VISDEX_ROOT, "cup_big", "cup_big.usd"), "scale": (1.30, 1.30, 1.30)},
    {"id": "shaker_body",  "usd_path": _os.path.join(_VISDEX_ROOT, "shaker_body", "shaker_body.usd"), "scale": (1.0, 1.0, 1.0)},
    {"id": "large_5_cyl",     "usd_path": _os.path.join(_VISDEX_ROOT, "large_5_cyl", "large_5_cyl.usd"),   "scale": (1.0, 1.0, 1.0)},
    {"id": "large_8_cyl_h12", "usd_path": _os.path.join(_VISDEX_ROOT, "large_8_cyl", "large_8_cyl.usd"),   "scale": (1.0, 1.0, 1.5)},
    {"id": "large_12_cyl_h12", "usd_path": _os.path.join(_VISDEX_ROOT, "large_12_cyl", "large_12_cyl.usd"), "scale": (1.0, 1.0, 2.4)},
)
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = tuple(_s["id"] for _s in _ACTIVE_OBJECT_SPECS)


def _object_usd_cfg(spec: dict) -> "sim_utils.UsdFileCfg":
    """단일 물체 USD spawn cfg. rigid/articulation 속성은 tesollo 기존 cup_cfg 값 그대로."""
    return sim_utils.UsdFileCfg(
        usd_path=spec["usd_path"],
        activate_contact_sensors=True,
        scale=spec["scale"],
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

    ADR 스케줄 없이 고정 범위(정적) — object_spawn(xy_range)만 ADR 대상(아래 adr_custom_cfg).
    """

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup", body_names=".*"),
            "static_friction_range":  (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range":      (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "mass_distribution_params": (0.7, 1.3),
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
    stage0_lift_start_min_contacts: int = 4
    success_min_grip_fingers: int = 4  # success 그립 손가락 수(엄지-컵 접촉 AND 강제와 함께 사용). 5(전손가락 동시)는 wrap 진동 이력.
    grasp_ready_hold_steps: int = 8   # 접촉 N개를 연속 hold하면 lift 래치 (잡으면 바로 리프트)
    lift_start_min_envelope_fingers: int = 0  # latch 인벨롭 게이트 제거(0=비활성). envelope은 grasp/lift 보상 credit으로 유도(hard 게이트 대체)
    finger_close_speed: float = 0.05  # ① 접촉-게이트 적응 폐쇄: 손가락 폐쇄 진행 속도/step (중간마디 접촉 시 동결)
    grasp_contact_persistence_reward_steps: int = 20
    enclosure_sharpness: float = 15.0
    # cup_radius_approx: cup_big 기준값(반경). 2026-07-26부터 per-object bbox 텐서
    # (cup_radius_approx_buf = bbox half_x/half_y 평균)가 enclosure target·obs 진단에 쓰인다.
    # 이 필드는 object_bbox.json 로딩 실패 시 즉시 예외(fail loud) — fallback 미사용.
    cup_radius_approx: float = 0.045
    enclosure_thumb_weight: float = 0.6

    # -----------------------------------------------------------------------
    # ADR
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 200
    adr_trigger_threshold: float = 0.02

    # design §위치 ADR: spawn xy_range 0.02→0.08 점진 확대(초기 좁게 학습 후 확장).
    # grasp_adr.get_param("spawn","xy_range")로 GraspLeftEnv._reset_idx가 조회.
    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "xy_range": (0.02, 0.08),
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
    object_spawn_x_center: float = 0.27   # demo 데이터와 일치 (0.40→0.27)
    # right: y=-0.10 (demo 데이터와 일치). left: y=0 대칭 반전 → +0.10
    object_spawn_y_center: float = 0.10
    # object_spawn_z: cup_big(반높이 0.08881200104951859) 기준 테이블 안착 높이. 2026-07-26부터
    # GraspLeftEnv가 이 값에서 테이블 표면 z(=object_spawn_z - cup_big 반높이)를 역산해
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
    # 로봇 설정 (openarm_tesollo_bi_rl.usd: 양팔 tesollo, 통일 네이밍 r_aj/r_hj/r_hl +
    # l_hj tesollo 20관절. right/grasp_v1 의 openarm_tesollo_sensor_rl.usd — 왼팔 단순
    # 그리퍼 —는 왼손 제어에 필요한 l_hj_<finger>_j 관절이 없어 사용 불가하므로
    # left/grasp_v2 선례를 따라 bi_rl 자산으로 전환했다.)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.usd"),
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
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # 손 stiffness/damping: pour-v5/6 검증값 채택. 기존 30/5(물렁)은 엄지 _3/_4가
            # 컵을 감을 때 반력이 엄지 대향을 뒤로 밀어냄(play 렌더 관찰). 단단히 유지.
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_1"],
                stiffness=600.0,   # 2000→600: 2000은 _1을 0에 고정했으나 컵 반력을 큰 교정토크로 되받아 파지 교란→붕괴. 600은 roll을 크게 줄이되 반력 교란 완화.
                damping=40.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_2"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_3"],
                stiffness=400.0,
                damping=60.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_[a-z]+_4"],
                stiffness=400.0,
                damping=60.0,
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
