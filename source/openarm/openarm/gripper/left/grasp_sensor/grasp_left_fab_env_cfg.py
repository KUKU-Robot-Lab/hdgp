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

"""Fabrics 팔 액션 변형 — `open-grip_l_grasp_sensor_fab`.

관절공간판(`GraspLeftGripperEnvCfg`, test17 검증)에서 **팔 액션 하나만** Fabrics 절대
palm 6D 로 바꾼다. 보상·씬·커맨드·커리큘럼·물리 플래그는 전부 부모 것 그대로다 —
그래야 test17 과 제어기만 다른 직접 비교가 성립한다(IK 변형 때와 같은 패턴).

왜 바꾸나: test17 은 이송까지 성공했지만 목표에서 못 멈춘다(잔류 0.17 m/s). 원인은
정책 raw 지령의 상시 포화이고, Fabrics 는 그 진동을 2차 적분으로 흡수한다.
근거 전문은 `grasp_left_fabric_action.py` docstring.

액션 차원 8 → **7** (팔 6D + 그리퍼 1). obs 의 `last_action` 은 자동 적응(36→35D).
"""

from __future__ import annotations

from isaaclab.envs import mdp
# ★`object_position_in_robot_root_frame` 는 isaaclab.envs.mdp 가 아니라 **lift 태스크의**
#   mdp 에 있다. 부모(grasp_left_env_cfg)도 이쪽을 쓴다 — 같은 자를 써야 한다.
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import grasp_left_curriculums as curriculums
from . import grasp_left_events as events
from . import grasp_left_obs_noise as obs_noise
from . import grasp_left_observations as obs_mdp
from . import grasp_left_fabric_action as fab
from . import grasp_left_preset as P
from . import grasp_left_rewards as rewards
from .grasp_left_env_cfg import GraspLeftGripperEnvCfg


def _require_log_only_reward_terms() -> None:
    """weight 0 항이 **log-only 로 동작하는지** 임포트 시점에 확인한다.

    ★★fab_test27 사고. 진단 항(`diag_*`)을 weight 0 으로 걸었는데 TB 에 **정확히 0.0000**
      만 찍혔다. 원인은 우리 코드가 아니라 **학습 호스트의 IsaacLab 판본**이었다:

        # upstream (vision-3090 에 설치돼 있던 것)
        # skip if weight is zero (kind of a micro-optimization)
        if term_cfg.weight == 0.0:
            self._step_reward[:, term_idx] = 0.0
            continue          # ← func 을 아예 호출하지 않는다

      로컬 IsaacLab 에는 log-only 분기가 있었는데 그건 **우리 로컬 수정**이었고, 학습
      호스트에는 전파돼 있지 않았다. 나는 로컬 소스만 읽고 같을 거라 가정했다.

    ⚠ 이 실패는 **조용하다** — 값이 0 으로 찍힐 뿐 에러가 없어서, 몇 시간을 태운 뒤에야
      "정확히 0.0000" 을 보고 알게 된다. 이 저장소가 죽은 접촉센서로 이미 당한 서명이다.
      그래서 정적 가드로 바꾼다: 패치가 없으면 **학습이 시작조차 안 된다.**

    호스트 패치: reward_manager.compute 의 weight==0 분기를 로컬과 같게 맞출 것.
    """
    import inspect

    from isaaclab.managers.reward_manager import RewardManager

    src = inspect.getsource(RewardManager.compute)
    if "log-only" not in src:
        raise RuntimeError(
            "이 호스트의 IsaacLab RewardManager 는 weight==0 항을 건너뛴다 — "
            "`diag_*` 진단 항이 전부 정확히 0 으로 찍힌다(조용한 실패). "
            "reward_manager.compute 의 weight==0 분기를 log-only 로 패치할 것."
        )


_require_log_only_reward_terms()


@configclass
class GraspLeftGripperFabEnvCfg(GraspLeftGripperEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이송 — Fabrics 팔 제어."""

    def __post_init__(self):
        super().__post_init__()
        # 팔 액션만 교체. 그리퍼(BinaryJointPositionAction 양조 지령)는 부모 그대로.
        self.actions.arm_action = fab.FabricPalmActionCfg()

        # ★★fab_test21 원본 정합: 정책 주기를 원본과 같은 **60 Hz** 로.
        #   원본 kuka: sim_dt 1/120 · decimation 2 → 60 Hz. 우리는 sim 0.01(100 Hz) ·
        #   decimation 2 → 50 Hz 였다. fabric 시간이 정책 스텝에 묶여 있으므로 주기가
        #   다르면 원본과 같은 비율을 맞춰도 절대 시간이 어긋난다.
        self.sim.dt = 1.0 / 120.0

        # ── 시뮬 / 에피소드 (kuka 원본값) ────────────────────────────
        self.sim.physx.bounce_threshold_velocity = P.PHYSX_BOUNCE_THRESHOLD_VELOCITY
        self.sim.physx.gpu_max_rigid_patch_count = P.PHYSX_GPU_MAX_RIGID_PATCH_COUNT
        self.episode_length_s = P.EPISODE_LENGTH_S
        # ★씬 기본 마찰 (kuka SimulationCfg.physics_material static/dynamic 1.0)
        self.sim.physics_material.static_friction = P.SCENE_STATIC_FRICTION
        self.sim.physics_material.dynamic_friction = P.SCENE_DYNAMIC_FRICTION
        self.scene.env_spacing = P.SCENE_ENV_SPACING
        # ★★fab_test23: 팔 중력을 끈다(kuka `KUKA_ALLEGRO_CFG` `disable_gravity=True`).
        #   원본에는 중력 보상 항이 없다 — 저수준에서 이미 보상된 팔을 모델링한 것이다.
        #   같이 꺼지는 것: 우리 `_droop` 적분항(preset `GRAVITY_COMP_ENABLED`).
        self.scene.robot.spawn.rigid_props.disable_gravity = P.ROBOT_DISABLE_GRAVITY
        self.scene.robot.spawn.rigid_props.retain_accelerations = P.ROBOT_RETAIN_ACCELERATIONS
        self.scene.robot.spawn.articulation_props.sleep_threshold = P.ROBOT_SLEEP_THRESHOLD
        self.scene.robot.spawn.articulation_props.stabilization_threshold = (
            P.ROBOT_STABILIZATION_THRESHOLD)
        self.scene.robot.spawn.joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type=P.ROBOT_DRIVE_TYPE)
        # ★솔버·강체 속성 (kuka 로봇 asset). 우리는 16/1 로 원본보다 촘촘했고
        #   max_depenetration_velocity 도 5.0 으로 200 배 작았다.
        for _spawn in (self.scene.robot.spawn, self.scene.object.spawn):
            if getattr(_spawn, "articulation_props", None) is not None:
                _spawn.articulation_props.solver_position_iteration_count = (
                    P.ARTICULATION_SOLVER_POSITION_ITER)
                _spawn.articulation_props.solver_velocity_iteration_count = (
                    P.ARTICULATION_SOLVER_VELOCITY_ITER)
            if getattr(_spawn, "rigid_props", None) is not None:
                _spawn.rigid_props.max_depenetration_velocity = P.RIGID_MAX_DEPENETRATION_VELOCITY
                _spawn.rigid_props.max_linear_velocity = P.RIGID_MAX_LINEAR_VELOCITY
                _spawn.rigid_props.max_angular_velocity = P.RIGID_MAX_ANGULAR_VELOCITY
                _spawn.rigid_props.linear_damping = P.RIGID_LINEAR_DAMPING
                _spawn.rigid_props.angular_damping = P.RIGID_ANGULAR_DAMPING
                _spawn.rigid_props.solver_position_iteration_count = (
                    P.ARTICULATION_SOLVER_POSITION_ITER)
                _spawn.rigid_props.solver_velocity_iteration_count = (
                    P.ARTICULATION_SOLVER_VELOCITY_ITER)

        # ── 손가락 접촉 센서 — critic 특권 관측용 (원본 `hand_forces`) ──
        # ⚠ USD 스폰에서 contact reporter API 를 켜지 않으면 센서 초기화가 실패한다
        #   ("could not find any bodies with contact reporter API"). 실제로 당했다.
        self.scene.robot.spawn.activate_contact_sensors = True
        # ⚠ body 마다 **개별** 센서여야 한다. 다중 body 단일 센서는 force_matrix_w 가
        #   조용히 0 이 된다(자매 트랙 실측 함정). Object 만 필터.
        for _b in P.GRIPPER_FINGER_BODIES:
            setattr(self.scene, f"contact_{_b}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{_b}",
                # ★★fab_test33: `/Object` 가 아니라 **`/Object/baseLink`**. 시뮬레이터가
                #   1024 env × 2 센서 = 2,048 번 경고를 찍고 있었다:
                #     [omni.physx.tensors.plugin] GPU contact filter for collider
                #     '/World/envs/env_N/Object' is not supported
                #   그래서 `force_matrix_w` 가 **최대까지 정확히 0** 이었다(실측).
                #   컵 자산의 RigidBodyAPI 는 `/object_shaker_body/baseLink` 에 있고
                #   프림 루트에는 없다 — 필터는 강체 프림을 가리켜야 한다.
                #   ⚠ 이 저장소 메모리에 같은 함정이 이미 적혀 있었다
                #     ("접촉필터는 RigidBodyAPI 붙은 baseLink로"). 두 번째로 밟았다.
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Object/{P.CUP_BODY_NAME}"],
                history_length=1,
                track_air_time=False,
            ))

        # ── 관측: fabric 내부 상태 + 노이즈 (원본 정합) ────────────────
        # ★★정책이 fabric 상태를 못 보고 있었다 — 전문은 grasp_left_observations.py.
        self.observations.policy.fabric_q = ObsTerm(func=obs_mdp.fabric_q)
        self.observations.policy.fabric_qd = ObsTerm(func=obs_mdp.fabric_qd)
        # ★★fab_test31 제거: `fabric_qdd`(관절 **가속도**). 실측 |x| 평균 4.39 · 최대 20.0 이라
        #   `clip_observations 5.0` 에서 **표본의 39.2% 가 잘린다.** 잘린 값은 정보가 아니라
        #   상수에 가까운 가짜 신호다 — 7 차원이 통째로 그렇게 죽어 있었다.
        #   단위가 rad/s² 인 값을 clip 5 짜리 obs 에 넣은 것이 애초에 잘못이었다.
        self.observations.policy.palm_pose_target = ObsTerm(func=obs_mdp.palm_pose_target)

        # ── 2-스케일 액션의 문맥 (fab_test40) ────────────────────────
        # ★★이게 빠지면 POMDP 가 된다. 같은 액션 벡터가 FINE/COARSE 에 따라 다른 절대
        #   지령이 되는데 그 문맥이 관측에 없으면 정책이 구분할 수 없다.
        #   **policy** 에 넣는다 — critic 전용이면 정책 쪽 POMDP 가 그대로 남는다.
        self.observations.policy.palm_action_scale = ObsTerm(func=obs_mdp.palm_action_scale)
        self.observations.policy.palm_action_anchor = ObsTerm(func=obs_mdp.palm_action_anchor)

        # ── GUI 마커: TCP 대신 **액션 지령 6D** (사용자 지시) ────────
        # `object_pose` 커맨드의 debug_vis 가 그리는 것이 body_pose(=TCP)와 goal_pose 다.
        # 그쪽을 끄고, 액션 텀이 정책의 **실제 지령**(큰 프레임)과 목표(작은 프레임)를 그린다.
        self.commands.object_pose.debug_vis = False
        self.actions.arm_action.debug_vis = True

        # ★★fab_test23 원본 정합 — 노이즈는 `ObsTerm.noise`(Unoise) 가 아니라 전용
        #   모듈이 건다. 원본은 폭을 env 마다 다시 뽑고 에피소드 고정 bias 를 얹는데
        #   Unoise 로는 둘 다 표현할 수 없다(상태를 못 든다). obs_noise 모듈 참조.
        _left_all = SceneEntityCfg("robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"])
        _jaws = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
        self.observations.policy.joint_pos = ObsTerm(
            func=obs_mdp.joint_pos_noisy, params={"asset_cfg": _left_all})
        self.observations.policy.joint_vel = ObsTerm(
            func=obs_mdp.joint_vel_noisy,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"])})
        self.observations.policy.object_position = ObsTerm(func=obs_mdp.object_position_noisy)
        # ★원본 policy obs 에 있던 세 항목이 우리에겐 없었다 — 물체 자세, 손 직교 위치,
        #   손 직교 속도. 이 태스크의 보상은 전부 턱–컵 기하로 정의돼 있는데 그 기하를
        #   만드는 입력이 관측에 없었다(정책이 관절각에서 FK 를 스스로 배워야 했다).
        self.observations.policy.object_rotation = ObsTerm(
            func=obs_mdp.object_rotation, params={"noisy": True})
        self.observations.policy.hand_pos = ObsTerm(
            func=obs_mdp.hand_body_pos,
            params={"asset_cfg": _jaws, "noisy": True, "lever": P.HAND_POINT_NOISE_LEVER})
        # ★fab_test31 제거: `hand_vel`(12D). `joint_vel`(9) + `fabric_qd`(7) 와 중복이고,
        #   두 턱이 함께 움직이므로 절반이 잉여다. 85 차원 중 12 를 쓰고 있었다.
        # ⚠ `enable_corruption` 은 이제 아무 항에도 걸리지 않는다(noise cfg 가 없다).
        #   노이즈 경로가 하나뿐이어야 "노이즈를 껐는데 왜 흔들리지"를 안 겪는다.
        self.observations.policy.enable_corruption = False

        # ── critic 전용 특권 관측 (비대칭 actor-critic) ────────────────
        # ★`critic` 이라는 이름의 그룹은 RlGamesVecEnvWrapper 가 자동으로 states 로 넘긴다.
        #   원본 `compute_critic_observations` 를 따른다 — 노이즈 없는 실측 + 접촉력 +
        #   관절토크 + 물체 속도.
        @configclass
        class _CriticCfg(ObsGroup):
            joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg(
                                    "robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"])})
            joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                                params={"asset_cfg": SceneEntityCfg(
                                    "robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"])})
            # ★fab_test31 제거: `arm_torque`. 실측 |x| 평균 5.83 · 최대 40.0 →
            #   clip 5.0 에서 **41.1% 포화**. 단위가 N·m 라 애초에 안 맞았다.
            hand_pos = ObsTerm(func=obs_mdp.hand_body_pos,
                               params={"asset_cfg": SceneEntityCfg(
                                   "robot", body_names=list(P.GRIPPER_FINGER_BODIES))})
            object_position = ObsTerm(func=lift_mdp.object_position_in_robot_root_frame)
            object_rotation = ObsTerm(func=obs_mdp.object_rotation)
            object_vel = ObsTerm(func=obs_mdp.object_lin_ang_vel)
            contact_forces = ObsTerm(
                func=obs_mdp.finger_contact_forces,
                params={"sensor_names": tuple(
                    f"contact_{b}" for b in P.GRIPPER_FINGER_BODIES)})
            target_object_position = ObsTerm(func=mdp.generated_commands,
                                             params={"command_name": "object_pose"})
            actions = ObsTerm(func=mdp.last_action)
            fabric_q = ObsTerm(func=obs_mdp.fabric_q)
            fabric_qd = ObsTerm(func=obs_mdp.fabric_qd)

            def __post_init__(self):
                self.enable_corruption = False   # critic 은 실측을 본다
                self.concatenate_terms = True

        self.observations.critic = _CriticCfg()
        # ★2-스케일 문맥은 critic 에도 준다(policy 쪽은 위에서 등록했다).
        #   ⚠ critic 그룹은 여기서 만들어지므로 이 줄들은 **반드시 그 뒤**여야 한다.
        self.observations.critic.palm_action_scale = ObsTerm(func=obs_mdp.palm_action_scale)
        self.observations.critic.palm_action_anchor = ObsTerm(func=obs_mdp.palm_action_anchor)

        # ══════════════════════════════════════════════════════════════
        # ★★fab_test43: **실패 종료 3종을 전부 없앤다.** 남는 종료는 `time_out` 뿐이고
        #   에피소드는 항상 만기까지 간다.
        #
        # 왜 벌점 전환과 한 몸인가 — `approach` 가 벌점이 되면 파지 전 보상이 전부
        # 음수다. 그 상태에서 `terminated`(bootstrap 없음)가 남아 있으면 V<0 이므로
        # **종료가 V=0 을 주는 이득**이 되어 "일부러 컵을 쓰러뜨리기/떨어뜨리기"가
        # 최적이 된다. 이 트랙이 이미 겪은 실패다(test6/test7: 보상 0 → 에피소드
        # 130 → 13 → 총보상 −0.46). 하나라도 남기면 그게 탈출구가 된다.
        #
        # 대체 수단:
        #   전도   → `rewards.tip` 벌점 (60° 에서 1.5/스텝, 계속 문다)
        #   낙하   → 컵이 바닥이면 진입깊이·높이 오차가 캡까지 차 자동으로 최대 벌점
        #   이탈   → 같은 이유 (턱축 이탈이 캡 0.15 m 까지 찬다)
        # ⚠ 되살리려면 반드시 `approach` 를 양수로 되돌린 뒤에 해야 한다.
        # ══════════════════════════════════════════════════════════════
        self.terminations.object_out_of_workspace = None
        self.terminations.object_tipped = None
        self.terminations.object_dropping = None

        # ── 리셋 관절 노이즈 (원본 `robot_spawn`) ─────────────────────
        # ★우리는 항상 같은 홈에서 시작했다. 원본은 ADR 로 관절 pos/vel 을 흔든다.
        self.events.arm_spawn_noise = EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_aj_[1-7]"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        # ★fab_test21 원본 정합: 로봇 표면 물성도 랜덤화한다(kuka `robot_physics_material`).
        #   파지는 두 표면의 접촉인데 우리는 컵 쪽만 흔들고 있었다.
        self.events.robot_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                "static_friction_range": P.ADR_ROBOT_STATIC_FRICTION[0],
                "dynamic_friction_range": P.ADR_ROBOT_DYNAMIC_FRICTION[0],
                "restitution_range": P.ADR_ROBOT_RESTITUTION[0],
                "num_buckets": 250,
            },
        )

        # ★★fab_test21 원본 정합: 팔 PD 를 DEXTRAH kuka 원본의 **테이퍼**로 바꾼다.
        #   원본 iiwa7: j1-4 300/45 · j5 100/20 · j6 50/15 · j7 25/15 (근위 단단·원위 무름).
        #   구값 400/80 은 균일했고, 그 값은 DEXTRAH open_tesollo 판본에서 **학습에 쓰지
        #   않는 반대쪽 팔을 잠그는** 게인과 같다 — 구동축 게인이 아니었다.
        #   ⚠ 중력은 켠 채 간다(실기에 중력이 있다). 남는 처짐은 `_droop` 적분이 흡수한다.
        self.scene.robot.actuators["left_arm"].stiffness = dict(P.ARM_FABRIC_STIFFNESS)
        self.scene.robot.actuators["left_arm"].damping = dict(P.ARM_FABRIC_DAMPING)

        # ── 목표 근처 지령 배회 페널티 — **1차**, 게이트 내장 ──────────
        # ★★fab_test20. jerk(2차) 를 여기서 뺐다. fab_test19 층 분해가 2차 성분은
        #   fabric 이 전부 흡수해(팔 관절 방향반전 0.0%) 제어에 도달하지 않음을 보였고,
        #   벌금은 접근·이송에만 물려 dwell 을 1.02 → 0.005 로 무너뜨렸다.
        #   실제로 보이는 진동은 dwell 구간 지령 배회(1.16~5.2 mm/step)이고 그건 1차다.
        #   근거 전문: grasp_left_rewards.py `palm_command_rate_at_goal` docstring.
        #
        # ⚠ 이 항은 커리큘럼 게이트가 필요 없다 — 게이트가 항 안에 있다(목표에서 멀면 0).
        #   fab_test14/19 실패는 "억제가 과제 성립 전에 걸린 것"이었는데, 이 항은 구조적으로
        #   과제가 성립한 상태(들고 + 목표 근처)에서만 존재한다.
        # ⚠ fab 전용이다. 관절공간 변형은 palm 지령 자체가 없다.

        # ── 접촉 보상 — 컵을 **건드리는 것 자체**에 값을 매긴다 (fab_test38) ──
        # ★★t22~t37 열세 판의 공통 서명이 `drop` ep50 안 0.000 이었다. 컵을 안 만지니
        #   파지를 찾을 표본이 없다. 만져서 얻는 게 아무것도 없기 때문이다 —
        #   낙하는 페널티가 아니라 종료라 위험만 있고, 파지 계열 보상은 `grasp_quality`
        #   를 지나야 하는데 거기 도달하려면 이미 잘 잡고 있어야 한다.
        #   DexPour(IROS 2025) III-A Stage 2 `r_contact` 이식. 그 논문 ablation 의
        #   Config.2 가 정확히 같은 실패를 기록한다("avoiding cup movement to minimize
        #   penalties", 파지 성공률 0%).
        # ⚠ 가중 2.0 은 실측 산정 — t37 살아있던 항의 순간율 합 0.486 대비 69%.
        #   지배하지 않으면서 접근과 동급이다. 근거는 preset 주석.
        self.rewards.contact_engage = RewTerm(
            func=rewards.contact_engage,
            weight=P.CONTACT_ENGAGE_WEIGHT,
            params={
                "sensor_names": tuple(f"contact_{b}" for b in P.GRIPPER_FINGER_BODIES),
                "force_threshold": P.CONTACT_FORCE_THRESHOLD,
                "all_bonus": P.CONTACT_ALL_BONUS,
            },
        )

        # ── 진단 항 (weight 0 = log-only) ─────────────────────────────
        # ★★fab_test26. **지령과 실제를 나란히 TB 에 띄운다.** 지금까지 이 트랙은 둘을
        #   같이 본 적이 없어서, 추종오차 90 mm 도 회피 국소최적도 전부 사후 프로브로만
        #   발견했다. weight 0 이면 총보상에 안 들어가고 원값만 로깅된다
        #   (`reward_manager.compute` 의 log-only 분기 — 값은 에피소드 **시간평균**).
        # ⚠ SceneEntityCfg 는 가변 객체다 — term 마다 **새 인스턴스**를 준다.
        _diag_jaw = lambda: SceneEntityCfg(  # noqa: E731
            "robot", body_names=list(P.GRIPPER_FINGER_BODIES))
        for _ax in ("x", "y", "z"):
            setattr(self.rewards, f"diag_cmd_{_ax}", RewTerm(
                func=rewards.diag_palm_cmd, weight=0.0,
                params={"action_term_name": "arm_action", "axis": _ax}))
            setattr(self.rewards, f"diag_jaw_{_ax}", RewTerm(
                func=rewards.diag_jaw_pos, weight=0.0,
                params={"axis": _ax, "pad_offset": P.JAW_PAD_OFFSET,
                        "jaw_cfg": _diag_jaw()}))
        # 추종오차 — 지령을 팔이 따라가고 있는가
        self.rewards.diag_cmd_jaw_gap = RewTerm(
            func=rewards.diag_cmd_jaw_gap, weight=0.0,
            params={"action_term_name": "arm_action", "pad_offset": P.JAW_PAD_OFFSET,
                    "jaw_cfg": _diag_jaw()})
        # 스텝당 지령 이동 — **리미터가 실제로 무는지**를 이걸로 확인한다
        self.rewards.diag_cmd_step = RewTerm(
            func=rewards.diag_cmd_step, weight=0.0,
            params={"action_term_name": "arm_action"})
        # ★정규화 분모 — 이 항의 로깅값이 곧 (에피소드 길이 / episode_length_s) 다.
        self.rewards.diag_duty = RewTerm(func=rewards.diag_duty, weight=0.0, params={})
        # 턱–컵 날 거리 — `reaching_object` 는 커널을 거쳐서 거리로 못 읽는다
        self.rewards.diag_jaw_cup_dist = RewTerm(
            func=rewards.diag_jaw_cup_dist, weight=0.0,
            params={"pad_offset": P.JAW_PAD_OFFSET, "jaw_cfg": _diag_jaw()})
        # ★fab_test39 신설 두 개 — t38 은 이 둘을 TB 에서 못 봐서 4000 epoch 내내
        #   사후 프로브로만 확인할 수 있었다.
        # 컵 **최저점** 상승 — `lifting_object` 가 0 일 때 "안 들었다"인지 "게이트가
        #   막았다"인지를 가른다. 기울여도 0 이라 가짜 리프트가 안 섞인다.
        self.rewards.diag_lift_height = RewTerm(
            func=rewards.diag_lift_height, weight=0.0, params={})
        # 턱축까지의 수직거리 — `grasp_ok` 의 1차 조건이자 D2 가 겨냥하는 값.
        #   t38 실측 62.4 mm(최선 27.4) vs 문턱 30 mm.
        self.rewards.diag_jaw_lateral = RewTerm(
            func=rewards.diag_jaw_lateral, weight=0.0,
            params={"pad_offset": P.JAW_PAD_OFFSET, "jaw_cfg": _diag_jaw()})

        # ══════════════════════════════════════════════════════════════
        # DexPour 계층 보상 (fab_test41) — 구 보상 전면 교체
        #
        # 논문 Fig.3: r_t = (1−λ)·p + μ·r_grasping + μ·r_lift + ν·r_transporting + ρ·r_pouring
        # 우리 5단계(사용자 규격): approach/align → grasp → lift → transfer → stay
        #
        # ★★**보상 슬롯 이름 = TB 태그**다(IsaacLab 이 슬롯 이름으로 로깅한다).
        #   구 이름은 `LiftEnvCfg` 에서 물려받은 것이라 내용과 어긋나 있었다 —
        #   `reaching_object` 안에 `approach_opposed` 가 들어 있었다. 전부 교체한다.
        #   ⚠ 과거 30여 런과 TB 태그 비교가 끊긴다. 대조는
        #     `docs/eval/fab_runs/README.md` 의 표로만 남는다(사용자 지시).
        # ★★가중이 **단조 증가**한다(2→1→3→5→7→10). 근거는 preset STAGE_* 주석.
        # ══════════════════════════════════════════════════════════════
        _stage_args = dict(
            jaw_cfg=SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
            sensor_names=tuple(f"contact_{b}" for b in P.GRIPPER_FINGER_BODIES),
        )

        def _sa():
            # ⚠ SceneEntityCfg 는 매니저가 제자리 변경하는 가변 객체다 — term 마다 새 인스턴스.
            return dict(
                jaw_cfg=SceneEntityCfg(
                    "robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                sensor_names=_stage_args["sensor_names"],
            )

        # 구 항 전부 제거 — 이름이 남아 있으면 TB 가 두 체계로 갈린다.
        for _old in ("reaching_object", "lifting_object", "object_goal_tracking",
                     "object_goal_tracking_fine_grained", "cup_between_jaws",
                     "grip_closure_when_enclosed", "grasp_pose", "settled_at_goal", "palm_cmd_rate",
                     "dwell_at_goal", "contact_engage", "gate_rate"):
            if hasattr(self.rewards, _old):
                setattr(self.rewards, _old, None)

        # ★★fab_test43: approach 는 **벌점**이다(weight 음수 · func 는 양수 크기).
        #   근거 전문은 `rewards.stage_approach` docstring — 요약하면 t42 에서 양수
        #   shaping 의 자세 인자가 거리 인자를 3배 눌러 "멀리서 각도만 맞추기"가
        #   1153 epoch 동안 최적이었다.
        self.rewards.approach = RewTerm(
            func=rewards.stage_approach, weight=P.STAGE_APPROACH_WEIGHT, params=_sa())
        # 전도 벌점 — 아래에서 `object_tipped` **종료를 제거**하고 이것으로 대체한다.
        self.rewards.tip = RewTerm(
            func=rewards.stage_tip, weight=P.STAGE_TIP_WEIGHT, params=_sa())
        self.rewards.contact = RewTerm(
            func=rewards.stage_contact, weight=P.STAGE_CONTACT_WEIGHT, params=_sa())
        self.rewards.grasp = RewTerm(
            func=rewards.stage_grasp, weight=P.STAGE_GRASP_WEIGHT, params=_sa())
        self.rewards.lift = RewTerm(
            func=rewards.stage_lift, weight=P.STAGE_LIFT_WEIGHT, params=_sa())
        self.rewards.transfer = RewTerm(
            func=rewards.stage_transfer, weight=P.STAGE_TRANSFER_WEIGHT, params=_sa())
        self.rewards.stay = RewTerm(
            func=rewards.stage_stay, weight=P.STAGE_STAY_WEIGHT, params=_sa())

        # ⑦ 평활화 페널티는 **부모의 `action_rate`·`joint_vel` 둘만** 쓴다.
        # ★★fab_test41: `palm_cmd_rate` 를 **제거**했다(사용자 지적).
        #   ⑴ 전 이력에서 **정확히 0** 이었다 — 게이트(`held_and_near_goal`)가 한 번도
        #      안 열렸으므로 지금까지 학습에 아무 영향이 없었다.
        #   ⑵ fabric 이 ③fabric 관절목표·④실제 관절에서 방향반전을 **0.0%** 로 지운다
        #      (fab_test19 층 분해 실측). 평활화 항이 셋일 이유가 없다.
        #   ⑶ DexPour 도 agnostic 도 평활화는 두 항(action_l2 · action_rate_l2)뿐이다.
        #   ⚠ 되살리려면 게이트를 사다리와 **같은 ρ** 로 두고, 그때도 억제는 과제가
        #     성립한 뒤에만 켠다([[suppression-terms-need-task-first]]).

        # ── 단계 진단 — "TB events logging 값 매칭"(사용자 지시) ──────
        for _d in ("lam", "mu", "nu", "rho", "tilt_deg", "perp_q", "d_goal",
                   "enter_s", "jaw_l", "height_h"):
            setattr(self.rewards, f"diag_stage_{_d}", RewTerm(
                func=getattr(rewards, f"stage_diag_{_d}"), weight=0.0, params=_sa()))

        # ── 도메인 랜덤화 (fab_test18 신설, 사용자 지시) ──────────────
        # 이 트랙엔 DR 이 사실상 없었다(hold_idle_joints 는 유휴 관절 고정이지 DR 이 아니다).
        # grasp_v1/v2 의 요소를 옮긴다. ⚠ **초기값은 전부 중립**이어야 한다 — 처음부터
        # 미끄럽거나 무거우면 파지 자체를 못 배운다(fab_test14 가 jerk 로 같은 실수를 했다).
        # 실제 범위는 아래 ADR 커리큘럼이 성공에 따라 넓힌다.
        self.events.cup_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": P.ADR_CUP_STATIC_FRICTION[0],
                "dynamic_friction_range": P.ADR_CUP_DYNAMIC_FRICTION[0],
                "restitution_range": P.ADR_CUP_RESTITUTION[0],
                "num_buckets": 250,
            },
        )
        self.events.cup_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": P.ADR_CUP_MASS_SCALE[0],
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # 외란 — 컵에 무작위 렌치. ★★fab_test23: IsaacLab 기본 항을 버리고 원본
        #   `apply_object_wrench` 를 그대로 옮겼다(등방 방향 · 질량 비례 · 토크 포함 ·
        #   손이 가까울 때만). 세 가지가 어떻게 갈렸는지는 events 모듈 docstring 참조.
        self.events.cup_disturbance = EventTermCfg(
            func=events.apply_object_wrench,
            mode="interval",
            interval_range_s=P.ADR_DISTURB_INTERVAL_S,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "jaw_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                "torsional_radius": P.DISTURB_TORSIONAL_RADIUS,
                "hand_dist_threshold": P.DISTURB_HAND_DIST_THRESHOLD,
            },
        )
        # 관측 노이즈/바이어스 재추첨 — 원본은 `_reset_idx` 에서 한다.
        self.events.obs_noise_resample = EventTermCfg(func=obs_noise.resample, mode="reset")

        # ★★fab_test21 원본 정합: PD 게인·관절 마찰 도메인 랜덤화.
        #   원본 kuka ADR: stiffness/damping ×(0.5, 2.0) · joint friction (0., 5.).
        #   우리에겐 아예 없었다 — 속도 피드포워드·fabric damping 과 함께 **세 번째**로
        #   "원본 ADR 항목을 통째로 빠뜨리거나 하드 끝값으로 고정한" 사례다.
        #   ⚠ 초기값은 중립(×1.0, 마찰 0) — ADR 이 넓힌다.
        self.events.arm_gains = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_aj_[1-7]"]),
                "stiffness_distribution_params": P.ADR_ARM_GAIN_SCALE[0],
                "damping_distribution_params": P.ADR_ARM_GAIN_SCALE[0],
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.arm_friction = EventTermCfg(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["l_aj_[1-7]"]),
                "friction_distribution_params": P.ADR_ARM_FRICTION[0],
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # ── ADR: 성공하면 난이도를 한 단계씩 넓힌다 ────────────────
        # 원리: 난이도를 올리는 요소는 **과제가 성립한 뒤에** 켠다. fab_test14 가 그 반대를
        # 해서(억제 항을 epoch 0 부터) 이송 학습을 통째로 잃었다.
        # 전문은 grasp_left_curriculums.py docstring.
        self.curriculum.adr = CurrTerm(
            func=curriculums.adr_expand_on_dwell,
            params={
                "metric_term": P.ADR_METRIC_TERM,
                "trigger": P.ADR_TRIGGER,
                "levels": P.ADR_LEVELS,
                "min_steps_between": P.ADR_MIN_STEPS_BETWEEN,
                "ema_alpha": P.ADR_METRIC_EMA_ALPHA,
            },
        )


@configclass
class GraspLeftGripperFabEnvCfg_PLAY(GraspLeftGripperFabEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
