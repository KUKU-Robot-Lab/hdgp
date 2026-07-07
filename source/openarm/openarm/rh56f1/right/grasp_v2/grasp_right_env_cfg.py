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

"""환경 설정: open-rh56f1_r_grasp_v1."""

from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm import OPENARM_ROOT_DIR
from .grasp_right_constants import (
    NUM_OBSERVATIONS,
    NUM_OBSERVATIONS_NO_MASS,
    NUM_ACTIONS,
    NUM_CRITIC_OBSERVATIONS,
    LIFT_PHASE_STEPS,
    LIFT_Z_DELTA,
    STABILIZE_PHASE_STEPS,
    STABILIZE_START_STEP,
)
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_HAND_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_HAND_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
)
from .real2sim_actuator_cfg import get_actuator_params, load_real2sim_calibration

_HDGP_ROOT  = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_REAL2SIM_CALIBRATION = load_real2sim_calibration(
    _os.environ.get("OPENARM_REAL2SIM_ACTUATOR_CALIBRATION", "")
)

# ---------------------------------------------------------------------------
# grasp_v2 파지 대상 물체 (다물체): primitives  [tesollo grasp_v2 에서 이식]
# ---------------------------------------------------------------------------
# 이름: 숫자=파지 폭(XY, cm). large 5/8/12cm, small 2.5/4/6cm. cyl=원통 / cuboid=직육면체.
# STAGE1=large 6종(폭 5~12cm). small 은 STAGE2 확장. _ACTIVE_OBJECT_NAMES 만 바꾸면 물체군 변경.
_PRIMITIVE_CURRICULUM_STAGE1: tuple = (
    "large_5_cyl", "large_8_cyl", "large_12_cyl",
    "large_5_cuboid", "large_8_cuboid", "large_12_cuboid",
)
_PRIMITIVE_ALL: tuple = _PRIMITIVE_CURRICULUM_STAGE1 + (
    "small_5_cyl", "small_8_cyl", "small_12_cyl",
    "small_5_cuboid", "small_8_cuboid", "small_12_cuboid",
)
# 현재 활성 물체군(초기 학습 = STAGE1 large 6종).
_ACTIVE_OBJECT_NAMES: tuple = _PRIMITIVE_CURRICULUM_STAGE1


def _primitive_usd_cfg(name: str) -> "sim_utils.UsdFileCfg":
    """단일 primitive USD spawn cfg (cup 과 동일 rigid 속성)."""
    return sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSETS_DIR, "primitives/USD", name, f"{name}.usd"),
        activate_contact_sensors=True,
        scale=(1.0, 1.0, 1.0),
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


# env 마다 랜덤 물체 선택(random_choice). MultiAsset → replicate_physics=False 필요.
_GRASP_OBJECT_SPAWN = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[_primitive_usd_cfg(_n) for _n in _ACTIVE_OBJECT_NAMES],
    random_choice=True,
)

# 비드 4단계 이산 질량: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g
# cup_middle용 소형 bead. mass는 그대로 10g으로 유지해 hidden-mass bin을 바꾸지 않는다.
_DEFAULT_BEAD_COUNT = 30
_DEFAULT_BEAD_MASS = 0.010
_DEFAULT_BEAD_SCALE = 0.35


def _actuator_params(group_name: str, default_stiffness: float, default_damping: float) -> dict:
    return get_actuator_params(
        group_name,
        _REAL2SIM_CALIBRATION,
        default_stiffness=default_stiffness,
        default_damping=default_damping,
    )


def _make_beads_cfg() -> RigidObjectCollectionCfg:
    """컵 내부 무게 도메인 랜덤화용 bead 설정 (30개, 각 10g, mesh 0.35x)."""
    rigid_objects: dict = {}
    for i in range(_DEFAULT_BEAD_COUNT):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            scale=(_DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE, _DEFAULT_BEAD_SCALE),
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=_DEFAULT_BEAD_MASS),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=0.5,
                angular_damping=0.5,
                max_depenetration_velocity=5.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        )
        bead_spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.1,
            dynamic_friction=0.08,
            restitution=0.1,
            friction_combine_mode="min",
            restitution_combine_mode="max",
        )
        rigid_objects[f"bead_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Bead_{i:02d}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42, -0.15, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=bead_spawn_cfg,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v11 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0   # grasp 7s + lift 2s + stabilize 1s
    decimation:       int   = 2
    fabrics_dt:       float = 1.0 / 60.0
    fabric_decimation: int  = 2
    use_cuda_graph:   bool  = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS          # 98 (actor; +oracle mass=99)
    action_space:      int = NUM_ACTIONS               # 11 (palm 6 + PCA 5)
    state_space:       int = NUM_CRITIC_OBSERVATIONS   # 121 (critic, privileged)

    num_observations: int = NUM_OBSERVATIONS
    num_actions:      int = NUM_ACTIONS
    num_states:       int = NUM_CRITIC_OBSERVATIONS
    actor_observe_bead_mass: bool = False

    # -----------------------------------------------------------------------
    # Pour warm-state export (play/collector only)
    # -----------------------------------------------------------------------
    enable_warm_state_export: bool = False
    warm_state_export_path: str = "/home/oem/rl_ws/datasets/grasp_warm_v11_pour.hdf5"
    warm_state_target_count: int = 2048
    warm_state_success_source: str = "stabilize"

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    # 실험1: fabric 이 손 제어, hand_mode="direct"(6-DOF 직접, PCA 1D 붕괴 회피).
    # PCA 복귀: hand_mode="pca" + constants NUM_ACTIONS 11 + _pre_physics scale 되돌림.
    use_hand_fabric:            bool  = True
    hand_mode:                  str   = "direct"
    # 실험2→3a(접근각): ±90 은 top 파지 확보하나 fabric IK 부하로 fps 급락 → ±70 로 축소.
    # ex 20~160° 유지(top-down 커버) + fps 회복. 복귀=45.0(side 전용).
    max_pose_angle:             float = 70.0
    fabrics_max_objects_per_env: int  = 8   # open_tesollo_boxes_no_table 객체 7개 → ≥7 필요
    fabrics_damping_gain:       float = 20.0
    stabilize_upright_orientation_enabled: bool = True
    # 07.03 upright 강화: pour가 똑바로 든 컵을 요구 → 컵 세우기 보정 강화(gain 1.5→3.0, max 25→45°).
    stabilize_upright_orientation_gain: float = 3.0
    stabilize_upright_orientation_max_deg: float = 45.0
    stabilize_upright_orientation_blend_steps: int = STABILIZE_PHASE_STEPS // 2
    # lift phase부터 upright 보정 적용 → stabilize 전에 미리 컵을 세워 righting 시간 확보.
    upright_orientation_from_lift: bool = False  # 07.03 lift중 보정이 그립 흔들어 리프트 붕괴 → stabilize만
    # Backward-compatible aliases for older launch overrides.
    stabilize_spawn_xy_hold_enabled: bool = True
    stabilize_spawn_xy_hold_gain: float = 2.0
    stabilize_spawn_xy_hold_max_delta: float = 0.10

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK rollout)
    # pregrasp_offset_* is the desired palm sensor offset from the cup.
    # The environment converts it to the Fabric palm_link target before IK.
    # -----------------------------------------------------------------------
    # ez=180(palm+z=+y 수평, side grasp) 규약에서 리셋 fabric 은 안정 수렴 → 전량 수렴 사용.
    pregrasp_fabric_steps: int   = 100
    reset_fabric_chunk_size: int = 128
    cache_pregrasp_reset:  bool  = True
    # +y side grasp offset (probe 실측, cup x_center=0.34 기준):
    #   offx=-0.07, offy=-0.08 → 엄지 opposition(1.57) 유지 시 pregrasp 관통 없는 standoff.
    #     정책이 이 자세에서 컵을 손가락-엄지 gap 으로 넣어 감싸도록 학습.
    #   offz=-0.15→ palm 을 아래로 끌어내림. r_aj7_bias 와 합쳐 palm 을 컵 높이로.
    pregrasp_offset_x:     float = -0.07
    pregrasp_offset_y:     float = -0.08
    pregrasp_offset_z:     float = -0.15
    # r_aj_7(손목)을 이만큼 낮춰 palm 을 컵 rim(z~0.35)→컵 중심(z~0.29)으로 내림.
    # fabric 은 +y 수평 유지 위해 r_aj_7 을 높게(≈1.27) 잡아 palm 이 rim 에 뜸(probe 확정).
    # 0.3 낮추면 palm z≈0.29(컵 높이)·수평 유지·엄지 관통 없음(offx 와 함께). bias 후 palm
    # anchor 는 실제 FK 로 재정합해 정책 시작 시 palm 튐 방지.
    pregrasp_r_aj7_bias:   float = 0.3
    pregrasp_noise_x:      float = 0.01
    pregrasp_noise_y:      float = 0.01
    pregrasp_noise_z:      float = 0.005

    # -----------------------------------------------------------------------
    # Demo reset (optional): Tesollo 20-DOF demo 데이터(pour_v1_a*) → RH56F1(6-DOF)엔 무효.
    # RH56F1 포팅: 비활성화하고 절차적 13-DOF 리셋(robot_start_joint_pos) 사용.
    # -----------------------------------------------------------------------
    enable_demo_grasp_reset: bool = False
    demo_grasp_pose_paths: tuple[str, ...] = tuple(
        f"/home/oem/rl_ws/datasets/pour_v1_a{i}.hdf5" for i in range(11, 21)
    )

    # -----------------------------------------------------------------------
    # Observation noise (sim2real domain randomization)
    # -----------------------------------------------------------------------
    obs_noise_joint_pos: float = 0.01
    obs_noise_joint_vel: float = 0.05
    obs_noise_body_pos:  float = 0.005
    obs_noise_cup_pos:   float = 0.015

    # -----------------------------------------------------------------------
    # Real2Sim actuator randomization
    # -----------------------------------------------------------------------
    real2sim_actuator_randomization_enabled: bool = bool(_REAL2SIM_CALIBRATION)
    real2sim_stiffness_scale_range: tuple[float, float] = (0.8, 1.25)
    real2sim_damping_scale_range: tuple[float, float] = (0.7, 1.5)
    real2sim_friction_scale_range: tuple[float, float] = (0.7, 1.3)

    # -----------------------------------------------------------------------
    # 접촉 감지
    # -----------------------------------------------------------------------
    lift_success_height: float = 0.04
    lift_target_z_delta: float = LIFT_Z_DELTA
    success_hold_steps: int = 20   # Phase2-b: 30→20 (decay 0.5/steps 30은 hold_count 4.7서 정체 = success_held ~0). 문턱 완화
    success_hold_miss_decay: float = 0.25   # Phase2-b: 0.5→0.25 (순증 +0.16→+0.30/step at success_now 0.44). leaky: miss 시 0 리셋 대신 감쇠. -1=기존 hard reset
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.4   # Phase B: stable 판정 완화 (0.2는 과도하게 빡빡 → success_held=0)

    # Phase curriculum:
    # 0 = grasp/lift only, 1 = add stabilize.
    enable_phase_curriculum: bool = False
    phase_curriculum_initial_stage: int = 1
    phase_curriculum_min_episodes: int = 100
    phase_curriculum_lift_success_threshold: float = 0.70
    phase_curriculum_stabilize_success_threshold: float = 0.70
    terminate_on_lift_failure: bool = True

    # -----------------------------------------------------------------------
    # Delta palm action
    # -----------------------------------------------------------------------
    palm_delta_xyz:     float = 0.03
    palm_delta_rot_deg: float = 5.0
    ema_action_alpha: float = 0.7
    approach_min_steps: int = 10
    approach_timeout_steps: int = 90
    approach_palm_radial_min: float = 0.025
    # +y side grasp 기하: pregrasp palm radial ~0.099. 0.105 는 너무 느슨(접근 없이 즉시 grasp
    # phase 진입→손가락 먼저 닫혀 fingertip pinch). 0.085 로 조여 palm 이 컵으로 ~1.4cm 접근한
    # 뒤에만 손가락이 닫히게(palm-first). floor~0.075(offx x분리) 위라 도달가능, 미도달 시 timeout fallback.
    approach_palm_radial_max: float = 0.085
    approach_palm_local_z_min: float = -0.015
    approach_palm_local_z_max: float = 0.095
    # 2 는 너무 빡셈: palm 이 radial 게이트 통과해도 열린 손가락(target 0, 더 못 물러남)이
    # 컵에 부수 접촉(~2.9개)해 approach_ready 를 막고 timeout(0.77) 으로 진입 → palm-first 무력화.
    # 5(손가락 tip 총수)로 완화해 approach_ready 를 palm-위치(radial≤0.085) 기반으로 전환.
    approach_max_tip_contacts: int = 5
    approach_upright_max_deg: float = 20.0
    approach_timeout_grasp_reward_scale: float = 0.25
    grasp_palm_delta_scale: float = 1.0
    grasp_palm_inward_offset: float = 0.11   # palm 더 적극적으로 깊숙히: grasp 중 palm 을 컵쪽으로 깊이(0.08→0.11)
    lift_palm_delta_xyz: float = 0.03
    lift_palm_delta_rot_deg: float = 15.0

    # -----------------------------------------------------------------------
    # Finger action semantics
    # RH56F1 v1은 6D absolute synergy target을 사용한다.
    # grasp:      HAND_APPROACH_POSE(-1) ~ HAND_GRASP_POSE(+1)
    # post-grasp: HAND_GRASP_POSE(-1)    ~ HAND_FULL_GRIP_POSE(+1)
    # 아래 delta_scale 항목은 이전 run/config 호환을 위해 유지되며 현재 env에서는 사용하지 않는다.
    # -----------------------------------------------------------------------
    finger_delta_scale:      float = 0.08
    lift_finger_delta_scale: float = 0.08
    enable_grasp_phase_full_grip_blend: bool = True
    grasp_phase_full_grip_contact_threshold: int = 4
    grasp_phase_full_grip_progress_threshold: float = 0.65

    # -----------------------------------------------------------------------
    # Shared 5-tip grasp reward parameters
    # -----------------------------------------------------------------------
    grasp_upright_threshold_deg: float = 8.0
    grasp_xy_threshold: float = 0.025
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    # envelope 강제(test4: envelope 형성됐다 붕괴 → tip-success 로 회귀). lift 게이팅·grasp
    # credit 의 envelope 비중을 올려 감싸야만 lift/grasp 보상을 받게 → envelope 유지 유도.
    # (공통 코어 cfg-configurable, tesollo 는 기본값 0.5/0.40 유지.)
    lift_envelope_mix:      float = 0.58   # lift 접촉 게이팅 envelope 비중 (기본 0.5). test5의 0.65는
                                           # success 억제+upright 소멸 → 완화. 얇은 컵이 envelope 물리 enabling 담당.
    grasp_envelope_credit:  float = 0.47   # grasp_quality envelope credit (기본 0.40). 0.55에서 완화(위와 동일 근거).
    # Exp1 anti-roll: post-lift hold 중 cup_ang_vel 직접 페널티 가중치. 컵 굴림(ang_vel 1.0~1.8 >
    # 임계 0.5)이 held=0의 근본 → 회전 억제로 rigid hold 유도(pour 회전 강건성 기반). 0=무효.
    cup_ang_vel_penalty_weight: float = 0.0  # Exp1 실패(굴림 못 잡음) → 0으로 끔. Exp2 단독 검증. firm 그립 생기면 재검토.
    stabilize_weight: float = 10.0
    stabilize_spawn_xy_scale: float = 0.03
    stabilize_upright_reward_scale_deg: float = 5.0
    stabilize_action_sharpness: float = 1.5
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    post_lift_contact_loss_weight: float = -8.0
    action_smooth_weight: float = -0.02
    palm_action_delta_reward_scale: float = 0.25
    finger_action_delta_reward_scale: float = 1.0

    # Legacy names kept for compatibility with older launch overrides.
    palm_approach_weight:    float = 1.0
    palm_approach_sharpness: float = 10.0

    enclosure_weight:       float = 3.0
    enclosure_sharpness:    float = 15.0
    # palm-seat: 컵을 palm 에 밀착(enclosing grasp)하도록 유도. sparse(접촉) 보상은 palm 이
    # 애초에 안 닿아 gradient=0 부트스트랩 실패 → dense 근접 보상 exp(-sharpness×palm_to_cup_dist)
    # 로 안 닿아도 가까워질수록 보상↑. grip 중(num_contacts≥1)에만 → palm-shove 방지.
    palm_seat_weight:       float = 6.0
    palm_seat_sharpness:    float = 15.0
    cup_radius_approx:      float = 0.025   # 균일 실린더 반경(2.5cm)과 정합. enclosure 게이트 타깃(cup_center±axis×r)에 직접 사용됨.
    enclosure_thumb_weight: float = 0.6
    # palm-first envelope: approach 중 thumb_1(엄지 abduction)을 palm 이 컵에 이 거리 이내로
    # 안착할 때까지 approach 값(opposition, 1.57)에 고정 → 엄지-손가락 사이 통로로 컵이 들어와
    # palm 에 앉은 뒤에야 엄지가 감싸도록(fingertip pinch 탈피). 근접 후 정책이 thumb_1 제어.
    # 같은 게이트로 approach reward 의 fingertip enclosure 항도 켠다. cup_middle 은 작아
    # 0.05 는 palm_sensor→컵 거리 floor(~0.064, 리세스3.9+반경3.5)보다 작아 palm_near 가 절대
    # 안 켜짐 → fingertip enclosure 유도·엄지 release 가 영영 OFF → 팔이 컵으로 파고들어 감싸는
    # 학습이 안 되고 pinch 로 수렴(test2 확인). floor 위 0.08 로 올려 palm 근접 시 enclosure+
    # 엄지 release 활성화 → 컵을 손가락 사이로 넣어 감싸도록 유도. (reward-audit ACCEPT)
    thumb_freeze_release_dist: float = 0.08
    # thumb_1(엄지 abduction) freeze: True면 palm 근접 전까지 1.57에 고정(palm-first). False면
    # 1.57은 초기자세로만, 처음부터 정책이 자유 제어 → 엄지가 굽을 위치를 스스로 찾도록(엄지 활성 시도).
    thumb_freeze_enabled: bool = False

    lift_reward_weight: float = 30.0
    grasp_contact_persistence_reward_steps: int = 30
    approach_tip_contact_penalty_weight: float = -4.0
    # palm 이 컵으로 이동해 seat 하도록 완화(-8→-3, -2→-1): 기존 값이 palm 이동을 억제해 contact/palm=0.
    grasp_palm_anchor_penalty_weight: float = -3.0
    palm_target_motion_penalty_weight: float = -1.0
    stabilize_spawn_xy_reward_weight: float = 40.0

    action_smoothness_palm_weight:   float = -0.10
    action_smoothness_finger_weight: float = -0.01

    # 질량 파라미터 (force-ratio/bin logging용 privileged variable)
    cup_base_mass:  float = 0.170          # kg (빈 컵 질량)
    bead_single_mass: float = _DEFAULT_BEAD_MASS  # kg per bead
    bead_scale: float = _DEFAULT_BEAD_SCALE

    # 07.03: 2로 하드게이트 시도 → test9 success 0.0/1233ep 완전 블록(손 기하가 중간마디 ~1개만
    #   접촉해 2 도달 불가). rh56f1은 하드게이트 불가 확정 → 0 유지, envelope는 geometry/credit로.
    min_middle_contacts_for_success: int = 0

    # Lift-entry grip readiness gate (state tracking용, reward가 아님)
    # Phase A: starting hurdle for the contact_adr 3→4→5 curriculum (was fixed 4).
    stage0_lift_start_min_contacts: int = 3
    stage0_lift_start_hold_steps:   int = 20
    # Exp2: 리프트 전 firm 그립 강제. lift-start를 'tip&근위 두 점 접촉 손가락 수 >= N'이 유지된
    # 후로 지연 → 정책이 먼저 컵을 엄지-손바닥 사이에 넣어 firm 그립을 만들도록 압박(리프트 shortcut
    # 차단). timeout_steps 지나면 fallback 허용(dead episode 방지). 0=firm 게이트 무효.
    lift_start_min_firm_fingers: int = 0   # Exp2(게이트) 실패 → 0으로 끔. 실린더 컵 단독 효과 검증(plain RL).
    lift_start_timeout_steps:    int = 150
    # success grip 판정(tip+근위 firm 관점): "손가락당 tip AND 근위 두 점 접촉(firm) 손가락 수 >= N
    # + 엄지 접촉". firm grip이 컵 회전을 구속 → held 가능. full envelope(원위/palm)는 불필요.
    # (tip|근위 union >=4 에서 tip&근위 firm >=3 으로 재구성 — firm은 더 엄격해 N 완화.) tunable.
    success_min_grip_fingers: int = 3
    lift_contact_hold_steps: int = 30
    full_grip_hold_steps:    int = 30
    lift_min_force_ratio:    float = 1.8

    # Slip proxy (no_slip_gate 계산용, 게임로직용)
    slip_proxy_threshold:                float = 1.0
    slip_proxy_contact_delta_weight:     float = 0.5
    # 0.5 → 0.0: middle 접촉을 이제 실제로 채우므로, 이번 단계(계측+critic)를 reward-neutral 로
    # 유지하기 위해 slip proxy 의 middle 기여를 0 으로 둔다. envelope 를 reward 로 적극 유도하려면
    # reward-audit 통과 후 재활성화.
    slip_proxy_middle_contact_delta_weight: float = 0.0
    slip_proxy_tilt_delta_weight:        float = 0.5
    slip_proxy_tilt_delta_scale:         float = 8.0

    # Stabilize 판정 임계값 (full_grip_ready gate용)
    stabilize_cup_lin_vel_threshold:  float = 0.04
    stabilize_cup_ang_vel_threshold:  float = 0.50
    stabilize_force_delta_threshold:  float = 0.35
    stabilize_contact_delta_threshold: float = 1.0
    stabilize_spawn_xy_success_threshold: float = 0.01

    # Legacy delta-control knob (absolute synergy semantics에서는 미사용)
    thumb_curl_downward_action_scale: float = 0.25
    thumb_curl_max_downward_delta:    float = 0.05

    # -----------------------------------------------------------------------
    # ADR — contact curriculum (threshold=0.1, 먼저 진행)
    # -----------------------------------------------------------------------
    # slip/adaptive/full_contact gate의 최소 총 접촉 수: 2 → 5
    enable_contact_adr:             bool  = True
    contact_adr_num_increments:     int   = 50
    contact_adr_increment_interval: int   = 400
    contact_adr_trigger_threshold:  float = 0.5   # Phase A: lift_started_rate 50%에서 다음 허들로 (트리거 신호를 success_rate→lift_started_rate로 교체)

    # 6.2: ADR trigger moving-window 크기
    # 최근 N episode 성공률을 ADR trigger에 사용 (0: 기존 cumulative 방식 유지)
    adr_window_size: int = 500

    contact_adr_custom_cfg: dict = field(default_factory=lambda: {
        "contact": {
            # int(round(value)) 로 사용. Phase A: 3 → 5 (전 손가락 FULL-GRASPING)
            "min_contacts": (3.0, 5.0),
        },
    })

    # -----------------------------------------------------------------------
    # ADR — 난이도 (threshold=0.8, contact ADR 이후 진행)
    # -----------------------------------------------------------------------
    enable_adr:            bool  = True
    adr_num_increments:    int   = 50
    adr_increment_interval: int  = 400
    # 실험3b: wrench 커리큘럼 트리거 = lifted_rate. 리프트 30% 달성 시 난이도(외란)↑.
    adr_trigger_threshold: float = 0.3   # lifted_rate 30%에서 ADR increment (외란 점증)

    adr_custom_cfg: dict = field(default_factory=lambda: {
        "spawn": {
            "object_spawn_xy_range": (0.01, 0.06),
        },
        "noise": {
            "obs_noise_cup_pos": (0.005, 0.025),
        },
        "finger": {
            "delta_scale": (0.05, 0.15),
        },
        "reward_weights": {
            "enclosure_weight": (10.0, 20.0),
        },
        # 실험3b: apply_object_wrench 외란 크기 ADR 점증(0→강). firm grip 커리큘럼.
        # lifted_rate>=trigger_threshold 마다 increment → 외란 강화 → 정책이 더 꽉 잡아야 함.
        "object_wrench": {
            "max_linear_accel": (0.0, 10.0),
        },
    })

    # -----------------------------------------------------------------------
    # 종료 조건
    # -----------------------------------------------------------------------
    cup_tipping_max_deg: float = 35.0
    success_upright_max_deg: float = 12.0   # 07.03 pour용 upright(20→12°): 달성가능+16°보다 개선
    stabilize_upright_max_deg: float = 12.0  # 20→12°. stabilize 보정강화로 달성 목표
    obj_out_x_min:  float = 0.05
    obj_out_x_max:  float = 0.85
    obj_out_y_min:  float = -0.60
    obj_out_y_max:  float = 0.25
    obj_fallen_z:   float = 0.20

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.34
    object_spawn_y_center: float = -0.10
    object_spawn_z:        float = 0.30    # DEXTRAH drop height(크기무관 고정). settle 로 안착.
    object_spawn_xy_range: float = 0.06
    settle_steps:          int   = 25      # episode 초기 N step 손 pregrasp 유지 + PCA close 억제 → 낙하 안착

    # --- 실험3b: apply_object_wrench (firm grip, DEXTRAH 이식) ---
    # 손이 물체를 잡았을 때(hand_to_object<threshold) 랜덤 외란 힘/토크 → 정책이 견디며 꽉 잡아야 함.
    enable_object_wrench:          bool  = True
    wrench_trigger_every:          int   = 60      # step(1s@60Hz)마다 새 외란 방향
    torsional_radius:              float = 0.01     # m (토크 = mass·accel·radius)
    hand_to_object_dist_threshold: float = 0.15     # 이 거리 이내(파지)일때만 외란 (MAX거리 스케일)
    wrench_lifted_z:               float = 0.26     # object z > 이면 lifted (ADR 트리거 metric)

    # -------- DEXTRAH goal-driven reward/goal (Phase4 에서 weight 튜닝 + reward-audit) --------
    # 물체를 goal 위치로 옮겨 들어올리는 태스크. weight/sharpness = DEXTRAH 기본값(dextrah_kuka_allegro).
    object_goal_pos:            tuple = (0.34, -0.10, 0.45)   # rh56f1 workspace(x_center 0.34) 기준 lift 목표
    object_goal_tol:            float = 0.10                   # success region 반경(m)
    hand_to_object_weight:      float = 1.0
    hand_to_object_sharpness:   float = 10.0
    object_to_goal_weight:      float = 5.0
    object_to_goal_sharpness:   float = -15.0
    lift_weight:                float = 5.0
    lift_sharpness:             float = 8.5
    finger_curl_reg_weight:     float = 0.0    # Phase4 에서 작은 정규화값으로 활성(reward-audit)
    object_out_of_reach_z:      float = 0.20   # z < 이면 낙하 판정 → done

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
    # 씬 설정
    # -----------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        replicate_physics=False,  # MultiAsset(env 별 다른 물체) → physics 복제 불가
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
    # 로봇 설정
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd"),
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
                # RH56F1 우측 손 drive (approach pose)
                "r_hj_thumb_1":  HAND_APPROACH_POSE[0],   # 0.60
                "r_hj_thumb_2":  HAND_APPROACH_POSE[1],   # 0.15
                "r_hj_index_1":  HAND_APPROACH_POSE[2],   # 0.30
                "r_hj_middle_1": HAND_APPROACH_POSE[3],   # 0.30
                "r_hj_ring_1":   HAND_APPROACH_POSE[4],   # 0.30
                "r_hj_pinky_1": HAND_APPROACH_POSE[5],   # 0.30
                # mimic 추종 (= drive × multiplier, 결합 init 으로 snap 방지)
                "r_hj_thumb_3":  HAND_APPROACH_POSE[1] * 1.1425,            # 0.171
                "r_hj_thumb_4":  HAND_APPROACH_POSE[1] * 1.1425 * 0.7508,   # 0.129
                "r_hj_index_2":  HAND_APPROACH_POSE[2] * 1.1169,            # 0.335
                "r_hj_middle_2": HAND_APPROACH_POSE[3] * 1.1169,
                "r_hj_ring_2":   HAND_APPROACH_POSE[4] * 1.1169,
                "r_hj_pinky_2": HAND_APPROACH_POSE[5] * 1.1169,
                **LEFT_ARM_REST_JOINT_POS,
                **LEFT_HAND_REST_JOINT_POS,
            },
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-7]"],
                **_actuator_params("openarm_right_arm", 400.0, 80.0),
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                **_actuator_params("openarm_left_arm", 400.0, 80.0),
            ),
            # RH56F1 우측 손 굴곡 5 (thumb_2 + 4손가락_1) — 07.02: 30→400 (tesollo pour curl/pip/dip 참조).
            "rh56f1_right_flexion": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"
                ],
                **_actuator_params("rh56f1_right_flexion", 400.0, 60.0),
            ),
            # abduction(thumb_1) — 30→200 (tesollo abduction 참조, 굴곡보다 낮게: 반력교란 회피).
            "rh56f1_right_abduction": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_thumb_1"],
                **_actuator_params("rh56f1_right_abduction", 200.0, 35.0),
            ),
            # RH56F1 우측 손 mimic(원위) 6 — 0→400 (tesollo dip 참조). PhysxMimicJoint 미결합 시
            # 원위가 흐물해 컵을 못 감쌈 → 강성 부여.
            "rh56f1_right_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"
                ],
                stiffness=400.0, damping=60.0,
            ),
            # RH56F1 좌측 손 drive 6 (학습 비사용 → 0 hold)
            "rh56f1_left_drive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"
                ],
                **_actuator_params("rh56f1_left_drive", 30.0, 5.0),
            ),
            # RH56F1 좌측 손 mimic 추종 6 (passive)
            "rh56f1_left_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"
                ],
                stiffness=0.0, damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # ContactSensor 설정
    # -----------------------------------------------------------------------
    # fingertip 힘센서 (실 *_force_sensor → 병합된 말단 링크). 순서: thumb,index,middle,ring,little
    right_tip_contact_links: tuple = (
        "r_hl_thumb_4",
        "r_hl_index_2",
        "r_hl_middle_2",
        "r_hl_ring_2",
        "r_hl_pinky_2",
    )
    # 근위(proximal, finger_1) 마디 접촉 = envelope 그립 signature (tip pinch 와 구분).
    # sim-only ContactSensor(실물엔 없음) → 계측/critic privileged 용. 엄지는 mid 마디 thumb_3.
    right_middle_contact_links: tuple = (
        "r_hl_thumb_3",
        "r_hl_index_1",
        "r_hl_middle_1",
        "r_hl_ring_1",
        "r_hl_pinky_1",
    )

    tip_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_(thumb_4|index_2|middle_2|ring_2|pinky_2)",
        history_length=1,
        track_air_time=False,
    )

    # palm force sensor body = r_hl_palm_sensor (OLD rh56f1_right_plam_force_sensor 대응).
    # 구 r_al_7(팔 손목)는 palm-cup 접촉 신호가 죽어 파지 저하 → 07.01 복구.
    palm_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_hl_palm_sensor",
        # grasp_v2: MultiAsset(replicate_physics=False)는 GPU contact filter 미지원 →
        # filter_prim_paths_expr 제거, net_forces_w 사용(tesollo grasp_v2 동일).
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # 컵 설정
    # -----------------------------------------------------------------------
    # grasp_v2: 단일 cup → primitives 다물체(MultiAsset, env 별 random_choice).
    # prim 명 "Cup" 유지 → tip/palm ContactSensor(Robot 링크 기준) 재사용. env.py self.cup 참조 불변.
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            # 크기무관 고정 drop height (DEXTRAH 방식). env reset 에서 xy 재배치 + settle 안착.
            pos=[0.34, -0.10, 0.30],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=_GRASP_OBJECT_SPAWN,
    )

    # -----------------------------------------------------------------------
    # 컵 마찰계수 도메인 랜덤화
    # -----------------------------------------------------------------------
    # μ_static  ~ Uniform[cup_friction_min, cup_friction_max]  (에피소드별 리셋)
    # μ_dynamic = μ_static × 0.9
    # 목적: max-grip이 Pareto-optimal이 되는 것을 방지 → 마찰 변동으로 적응형 파지 학습
    cup_friction_min: float = 0.15
    cup_friction_max: float = 0.60

    # 6.4: friction ablation — 고정값 (>= 0)이면 DR 비활성화, -1이면 랜덤화
    cup_friction_fixed: float = -1.0

    # -----------------------------------------------------------------------
    # Bead 무게 도메인 랜덤화
    # cup_middle에서는 물리 bead가 컵을 관통/튕김시키므로 기본 비활성화한다.
    # -----------------------------------------------------------------------
    beads_cfg: RigidObjectCollectionCfg = field(default_factory=_make_beads_cfg)
    num_beads: int = _DEFAULT_BEAD_COUNT              # 30
    physical_beads_enabled: bool = False
    # 가상질량 도메인 랜덤화: {0,10,20,30}개 × 10g → 컵 실효질량 {170,270,370,470}g.
    # 물리 bead 없이 hidden-mass 로 grip force 하중강건성 학습(actor 는 질량 미관측, critic oracle).
    bead_count_min: int = 0
    bead_count_max: int = 30
    bead_spawn_z_offset: float = 0.035

    # Keep dynamic insertion disabled for hidden-mass static-bin grasp/lift training.
    dynamic_bead_spawn_enabled: bool = False
    dynamic_bead_spawn_step: int = STABILIZE_START_STEP + 30
    bead_initial_count_min: int = 0
    bead_initial_count_max: int = 0
    dynamic_bead_add_count_min: int = 10
    dynamic_bead_add_count_max: int = 20

    # Eval-only: lift 중 oracle/effective mass만 바꿔 force 반응을 측정한다.
    eval_mass_shift_enabled: bool = False
    eval_mass_shift_step: int = LIFT_PHASE_STEPS // 2
    eval_mass_shift_target_bead_count: int = 30

    # -----------------------------------------------------------------------
    # Hand / joint 이름
    # -----------------------------------------------------------------------
    hand_body_names:      list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_HAND_JOINT_NAMES


class GraspRightEnvCfgNoActorMass(GraspRightEnvCfg):
    """Asymmetric teacher config: actor excludes oracle mass, critic keeps it."""

    observation_space: int = NUM_OBSERVATIONS_NO_MASS
    num_observations: int = NUM_OBSERVATIONS_NO_MASS
    actor_observe_bead_mass: bool = False
