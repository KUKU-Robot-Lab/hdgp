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

"""환경 설정: gripper/left/grasp_sensor — 왼팔 2지 그리퍼로 shaker 집어 옮기기.

IsaacLab `Isaac-Lift-Cube-OpenArm-v0` 레시피를 그대로 물려받고 **로봇·물체·씬만** 바꾼다.
그래서 이 파일이 짧다 — 보상·관측·커맨드·이벤트·커리큘럼은 손대지 않는다.

왜 이 방식인가
--------------
처음에는 우측 다지 손 태스크(Direct RL + Fabrics + 정확 6D TCP 포즈 attractor)를 이식했다가
막혔다. 2지 그리퍼는 jaw 가 수평이어야 파지가 성립해 팔에 특정 6-DOF 자세를 강제하는데,
이 팔은 손목 j6 가 ±45°·손목 3축 effort 가 7 N·m 뿐이라 낼 수 있는 자세가 얇은 곡선이다.
거기에 "정확한 포즈를 내라"는 가장 빡빡한 제어를 얹은 셈이었다(실측 자세 오차 28°, j5 한계 고착).

lift 레시피는 정반대다:
  · 팔 = 관절 위치 델타(JointPositionAction) → 정책이 내는 모든 액션이 항상 유효한 타깃
  · 그리퍼 = 이진 스칼라(BinaryJointPositionAction) → 파지력·개도를 학습할 필요 없음
  · 보상에 **회전 항이 하나도 없다** → "자세 도달성" 문제가 발생할 지점이 없다

★바꾸지 말 것: scale=0.5, use_default_offset=True, BinaryJointPositionAction,
  커리큘럼 2개, decimation=2 / episode_length_s=5.0, reward weight 조합.
  이것들이 이 레시피가 학습되는 이유다.

단 하나 덧댄 것: 팔 액션에 **목표 변화율 상한**(= 관절 속도 한계)을 씌웠다. 위 성질은
그대로다 — scale·offset·도달 범위·액션 차원이 전부 같고, 절대 목표를 만드는 부모의
계산 뒤에 clamp 만 붙는다. 근거는 grasp_left_actions.py 의 docstring 에 실측과 함께 있다.
"""

from __future__ import annotations

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

# ★IsaacLab 원본을 먼저, 없으면 vendored 사본. vision-3090 의 IsaacLab(5c2ec81c)에는
#   openarm lift 레시피가 아직 없어 ModuleNotFoundError 로 죽는다(08.22 실측) —
#   그 머신의 IsaacLab 을 올리면 퍼셉션 쪽 소비자가 위험해 사본을 동봉했다.
try:
    from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import (
        LiftEnvCfg,
    )
except ModuleNotFoundError:
    from ._vendored_lift_openarm_env_cfg import LiftEnvCfg

from openarm import OPENARM_ROOT_DIR

from . import grasp_left_actions as actions
from . import grasp_left_events as events
from . import grasp_left_preset as P
from . import grasp_left_rewards as rewards

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspLeftGripperEnvCfg(LiftEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이동."""

    def __post_init__(self):
        super().__post_init__()

        # ── PhysX GPU 버퍼 ─────────────────────────────────────────
        # ★★self-collision 을 켜면 접촉 패치가 폭증한다. 기본값으로 돌리면 첫 스텝부터
        #   "Patch buffer overflow detected, please increase its size to at least 239679"
        #   가 쏟아진다(실측, 1024 env). 오버플로는 **접촉이 조용히 유실**되는 것이라
        #   물리가 신뢰할 수 없게 된다 — 학습을 태우기 전에 반드시 올려야 한다.
        #   값은 형제 트랙 `agnostic/tasks/grasp_lift_fabric` 에서 2048 env 로 검증된 것을 쓴다.
        #   ⚠ 값은 **GPU 메모리에 맞춰야** 한다. 처음엔 98 GB 서버 기준(2**22·8M·2**28)으로
        #     잡았는데 24 GB(RTX 3090)에서 4096 env 는 CUDA OOM, 2048 env 는 22.9/24.5 GB 로
        #     포화해 PhysX 가 "Scene state is corrupted" 를 2733 회 뱉으며 epoch 31 에서
        #     멈췄다(08.22 실측). 필요량은 1024 env 에서 패치 24 만이므로 2**20 이면 4 배 여유다.
        #   ★줄인 뒤에는 반드시 오버플로 카운트 0 을 확인할 것 — 부족하면 접촉이 조용히 유실된다.
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 20
        # ★2 ** 21 로는 부족했다 — 실측 요구 **3,191,536**(vision-3090 2048 env).
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 22
        # ★★08.22 실측으로 올렸다. 2 * 1024 * 1024 로는 **부족했다** — vision-3090 2048 env
        #   에서 PhysX 가 "increase foundLostAggregatePairsCapacity to **4562626**" 를 냈다.
        #   ⚠ 이건 죽지 않고 경고만 내면서 **접촉을 조용히 놓치는** 종류다("the simulation
        #     will miss interactions"). fab_test1 이 이 상태로 4000 epoch 을 돌 뻔했고,
        #     내 모니터링 grep 이 "Patch buffer|buffer overflow" 만 봐서 놓쳤다.
        #     → 모니터링 패턴에 반드시 `PxGpuDynamicsMemoryConfig` 를 넣을 것.
        #   요구치 4.56M 에 1.8 배 여유. 쌍 버퍼라 VRAM 증가는 수십 MB 수준이다.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 8 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 4 * 1024 * 1024
        # ★2 ** 26(67.1M) 이 실측 요구 **68,960,016** 에 아슬아슬하게 못 미쳤다
        #   ("Contacts have been dropped"). 한 단 올린다.
        self.sim.physx.gpu_collision_stack_size = 2 ** 27

        # ── 로봇 ────────────────────────────────────────────────────
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_os.path.join(_ASSETS_DIR, P.ROBOT_USD_REL),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # ★중력 켠 채 학습한다. 우측 태스크와 IK 경로는 disable_gravity 를 쓰지만
                    #   그건 포즈 추종을 위한 타협이라 실기 이식성을 해친다.
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    # ★08.22 켰다. 자산이 self-collision-safe 로 재빌드됐고(urdf 5023977:
                    #   콜라이더 전부 convexDecomposition + 감사 WARN 쌍 전부 filtered_pairs),
                    #   좌우 팔을 **이 태스크의 실제 학습 자세**로 놓고 감사하면 깨끗하다:
                    #     audit_self_collision.py --pose (좌팔 홈 7개 + r_aj_2=0.3 r_aj_4=2.0)
                    #       → PASS (0 fail, 4 warn), WARN 4개 전부 우측·이미 필터됨
                    #   ⚠ 폐기된 `grasp_sensor_fabrics_ABORTED` 홈(j6=−0.67, j7=+1.36)에서는
                    #     `l_al_5 ↔ l_al_7` 이 5.4 kN 으로 유령접촉한다(raw 여유 3.2 mm).
                    #     **홈이 다르면 자기충돌 결론이 이식되지 않는다** — 트랙별로 재감사할 것.
                    #     이 태스크 홈은 j6=+0.0003 / j7=−0.3306 으로 안전 구간이다.
                    enabled_self_collisions=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    # ★액션 0 = 이 자세다(use_default_offset). 파지 준비 자세여야 한다.
                    **P.LEFT_ARM_HOME_JOINT_POS,
                    P.GRIPPER_JOINT_NAMES[0]: P.GRIPPER_OPEN_POS,
                    P.GRIPPER_JOINT_NAMES[1]: P.GRIPPER_OPEN_POS,
                    **P.RIGHT_REST_JOINT_POS,
                },
            ),
            actuators={
                # 팔: IsaacLab OpenArm 값. ★400/80 은 IK 추종용이라 쓰지 않는다.
                # ★★`velocity_limit_sim` 을 반드시 함께 준다. 레퍼런스 `OPENARM_UNI_CFG` 가
                #   명시하는데 처음 이식할 때 빠뜨렸고, 그러면 USD/URDF 기본값(5.4~20.9 rad/s,
                #   레퍼런스의 2.5~9.6 배)이 쓰인다. damping 이 4 뿐이라 팔이 과속으로 오버슈트
                #   하며 흔들리고("시작할 때 진자처럼 흔들린다"는 렌더 관찰), TCP 로 컵을
                #   정조준할 수 없게 된다. 20.9 rad/s 면 한 스텝(0.02 s)에 0.42 rad — 액션
                #   범위(±0.5 rad)를 한 스텝에 소화해 버린다. 2.175 면 0.0435 rad 로 부드럽다.
                "left_arm": ImplicitActuatorCfg(
                    joint_names_expr=["l_aj_[1-7]"],
                    velocity_limit_sim=P.ARM_VELOCITY_LIMIT,
                    effort_limit_sim=P.ARM_EFFORT_LIMIT,
                    stiffness=80.0,
                    damping=4.0,
                ),
                # 그리퍼: 두 관절 모두 커버리지를 준다(없으면 무구동 자유이동).
                # ★지령도 두 관절 모두에 간다 — USD 에 mimic 이 없다(preset 주석 참조).
                "left_gripper": ImplicitActuatorCfg(
                    joint_names_expr=["l_hj_gripper_[1-2]"],
                    velocity_limit_sim=0.2,
                    effort_limit_sim=333.33,
                    stiffness=2e3,
                    damping=1e2,
                ),
                # 유휴 오른팔·오른손: rest 자세 유지만 하면 된다.
                # ★★`effort_limit_sim` 을 반드시 올린다. URDF 의 팔 effort 는
                #   j1/j2=40, j3/j4=27, **j5~j7=7 N·m** 뿐이라 stiffness 400 이 무의미하게
                #   포화하고, 20 관절 손(약 1.4 kg)을 단 오른팔이 중력에 그대로 처진다.
                #   실측(프로브 1a): 관절 오차 최대 49.9°·평균 27°, 손끝이 테이블 상면
                #   바로 위(0.223)까지 내려와 **테이블에 얹힌다**. 렌더에서 사용자가 지적한
                #   "오른팔이 바닥에 닿아 있다"가 이것이다.
                #   이 팔은 학습에 쓰이지 않는 배경이고 실기로 배포되지도 않으므로,
                #   sim 에서 자세만 고정되면 된다.
                "idle_right_arm": ImplicitActuatorCfg(
                    joint_names_expr=["r_aj_[1-7]"],
                    stiffness=400.0, damping=80.0, effort_limit_sim=1000.0,
                ),
                # 유휴 오른손도 같은 이유로 올린다. effort 1.5 는 실기 정합값이지만
                # 그건 **파지를 학습하는 손**에 필요한 것이고, 여기 오른손은 배경이다.
                "idle_right_hand": ImplicitActuatorCfg(
                    joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                    stiffness=20.0, damping=4.0, effort_limit_sim=50.0,
                ),
                "head_camera": ImplicitActuatorCfg(
                    joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        # ── 액션 ────────────────────────────────────────────────────
        # ★레퍼런스의 `JointPositionActionCfg` 에 **목표 변화율 상한**만 씌운 것이다.
        #   scale=0.5 / use_default_offset=True 는 그대로 — 액션 0 = 홈 자세, 도달 범위도 동일.
        #   상한을 넣은 이유는 grasp_left_actions.py 의 모듈 docstring 에 실측과 함께 있다.
        #   요약: 결정론 정책이 관절 속도 한계의 **7 배**를 지령하고 있었고, 그건 보상으로
        #   고칠 수 없다(action_rate 는 탐색 노이즈에 오염돼 σ 만 줄인다).
        self.actions.arm_action = actions.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["l_aj_[1-7]"],
            scale=0.5,
            use_default_offset=True,
            rate_limit=P.ARM_TARGET_RATE_LIMIT,
        )
        # ⚠ gripper_2 는 USD PhysX mimic 이라 지령 대상에서 뺀다. BinaryJointPositionAction 은
        #   joint_names 가 하나라도 안 풀리면 ValueError 로 즉사하므로 정규식이 아니라 정확한
        #   이름을 쓴다.
        # ★08.22 **두 조 모두에 지령한다.** 예전에는 `gripper_1` 에만 줬는데, 그건 USD 에
        #   PhysX mimic 제약이 있어 `gripper_2` 가 따라온다는 전제였다. **그 전제가 깨졌다** —
        #   자산 재빌드(urdf 6d065f7) 후 USD 의 `l_hj_gripper_2` 에는 mimic API 가 없고
        #   `PhysicsDriveAPI` 만 있다(실측: 적용 스키마에 PhysxMimicJointAPI 없음).
        #   액션 대상이 아닌 관절은 PD 목표가 0 이므로 두 번째 조가 **닫힌 채 고정**됐다:
        #       open 지령에도 j1=44.00 mm / **j2=0.00 mm**, 조 간격 56 mm (예전 자산은 j2=40.26)
        #   컵 몸통이 58~88 mm 라 이 상태로는 물리적으로 물 수 없다.
        #   두 조는 축이 서로 반대(`0 -1 0` vs `0 1 0`)라 같은 값을 주면 함께 벌어진다.
        #   ※ 자산 쪽에서 mimic 을 복원하면 이 지령은 무해하게 중복될 뿐이다.
        # ★★08.24 **접근 성공 하드 게이트**. 접근 전에는 그리퍼를 강제로 연다.
        #   근거: Fabrics 가 우연한 리프트를 없앴다(관절 목표 변화 test17 2.79 rad/s vs
        #   fab_test5 0.38 rad/s → 컵 상승 +138 mm vs +17 mm). 정책이 "열기·위치·닫기·들기"
        #   연접을 우연히 맞춰야 하는 문제를, 앞 두 칸을 코드가 강제해 없앤다.
        #   ⚠ **부모에서 바꿔야** 관절공간·IK·Fabrics 세 변형에 전파된다
        #     (fab cfg 에서의 그리퍼 재정의는 계약으로 금지돼 있다).
        self.actions.gripper_action = actions.GatedBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(P.GRIPPER_JOINT_NAMES),
            open_command_expr={j: P.GRIPPER_OPEN_POS for j in P.GRIPPER_JOINT_NAMES},
            close_command_expr={j: P.GRIPPER_CLOSED_POS for j in P.GRIPPER_JOINT_NAMES},
            finger_body_names=tuple(P.GRIPPER_FINGER_BODIES),
            object_name="object",
            pad_offset=P.JAW_PAD_OFFSET,
            lateral_ok=P.GRASP_GATE_LATERAL_OK,
            along_ok=P.GRASP_GATE_ALONG_OK,
            release_lateral=P.GRASP_GATE_RELEASE_LAT,
        )

        # ── 씬: 테이블 (로컬 자산) ──────────────────────────────────
        # 레퍼런스는 클라우드 Nucleus 의 SeattleLabTable 을 쓰는데 이 머신에 캐시가 없다.
        # ★씬 전체가 `assets/env/usd/env.usd` 한 덩어리다(사용자 지정, 08.20).
        #   **Env 원점 = 로봇 base link 원점**이라 오프셋 없이 (0,0,0) 에 붙인다.
        #   상속받은 `scene.table` 슬롯을 그대로 쓴다 — 이름만 table 이고 prim 은 Env 다.
        #   ⚠ `rigid_props` 를 주면 안 된다. env.usd 의 메시는 PhysicsCollisionAPI 만 가진
        #     **정적 삼각메시 콜라이더**인데 RigidBodyPropertiesCfg 를 씌우면 rigid body API
        #     가 붙어 동적 강체가 된다. 레퍼런스도 테이블에 rigid_props 를 주지 않는다.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Env",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=P.ENV_POS, rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(usd_path=_os.path.join(_ASSETS_DIR, P.ENV_USD_REL)),
        )
        # 바닥면: env.usd 의 base_plate 밑면(-0.025)에 맞춘다. 판 밖으로 떨어진 컵은
        # 여기까지 내려가고, 그전에 object_dropping 이 이미 종료시킨다.
        self.scene.plane.init_state.pos = (0.0, 0.0, P.ENV_FLOOR_Z - 0.010)

        # ── 씬: 물체 (shaker) ──────────────────────────────────────
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_Y_CENTER, P.CUP_SPAWN_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=_os.path.join(_ASSETS_DIR, "cup", P.CUP_USD_NAME),
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=P.CUP_MASS),
            ),
        )

        # ── EE 프레임 (보상 계산용, 액션과 무관) ────────────────────
        # `l_hl_gripper_tcp` 는 physics USD 에 강체로 없다 → base + z 오프셋으로 TCP 를 만든다
        # (Franka 가 panda_hand + offset 0.1034 를 쓰는 것과 같은 패턴).
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{P.GRIPPER_BASE_BODY}",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z)),
                ),
            ],
        )

        # ── 물체 스폰 랜덤화 ────────────────────────────────────────
        # ★asset_cfg 를 반드시 교체한다. 레퍼런스는 큐브 prim 이름 `"Object"` 를 박아 두는데
        #   우리 shaker 의 강체는 `baseLink` 라, 매니저가 이름을 resolve 하는 순간 죽는다.
        #   이 이벤트는 root state 만 쓰므로 body_names 자체가 불필요하다.
        #   ⚠ 로컬에서는 sim 이 playing 이 아닌 타이밍이라 resolve 가 스킵돼 통과하고,
        #     서버 학습 기동에서만 터졌다. 아래 계약 테스트로 고정해 둔다.
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_X_RANGE),
            "y": (-P.CUP_SPAWN_Y_RANGE, P.CUP_SPAWN_Y_RANGE),
            "z": (0.0, 0.0),
        }

        # ── 학습 영상: env 하나만 정면에서 ──────────────────────────
        # 기본 뷰어는 여러 env 가 한 화면에 잡혀 파지 자세를 판별할 수 없다.
        # `origin_type="env"` + `env_index=0` 으로 env 0 에 고정하고, 로봇 정면에서
        # 컵·그리퍼를 바라본다(파지 시 jaw 가 수평인지 보이는 각도).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = P.VIEWER_EYE
        self.viewer.lookat = P.VIEWER_LOOKAT
        self.viewer.resolution = (1280, 720)

        # ── 유휴 관절 자세 고정 ────────────────────────────────────
        # ★★없으면 오른팔이 **차렷으로 내려가 바닥에 닿는다**. init_state 는 관절의 상태만
        #   정하고 PD 목표는 정하지 않는데, 액션 대상이 아닌 관절은 아무도 목표를 써 주지
        #   않아 0 인 채로 남기 때문이다. 자세한 경위는 grasp_left_events 참조.
        self.events.hold_idle_joints = EventTermCfg(
            func=events.hold_joints_at_target,
            mode="reset",
            params={
                "joint_targets": {
                    **P.RIGHT_REST_JOINT_POS,
                    "head_j_pan": 0.0,
                    "head_j_tilt": 0.0,
                },
            },
        )

        # ── 목표 커맨드 ────────────────────────────────────────────
        self.commands.object_pose.body_name = P.GRIPPER_BASE_BODY
        self.commands.object_pose.ranges.pos_x = P.GOAL_POS_X
        self.commands.object_pose.ranges.pos_y = P.GOAL_POS_Y
        self.commands.object_pose.ranges.pos_z = P.GOAL_POS_Z

        # ── 관측: 왼팔 관절만 ──────────────────────────────────────
        # 기본값은 로봇 전체(오른팔 27관절 포함)라, 이 팔이 못 건드리는 값이 관측을 채운다.
        # ★term 마다 **새 인스턴스**를 만든다. SceneEntityCfg 는 매니저가 resolve() 로
        #   제자리 변경(joint_ids 를 채워 넣음)하는 가변 객체다. 하나를 공유하면 두 번째
        #   term 에서 "joint_names 와 joint_ids 가 불일치" 로 env 생성이 죽는다(실측).
        def _left_joints() -> SceneEntityCfg:
            return SceneEntityCfg(
                "robot", joint_names=["l_aj_[1-7]", "l_hj_gripper_[1-2]"]
            )

        # ★★게이트 상태를 관측에 노출한다. 하드 게이트는 정책이 볼 수 없는 숨은 상태라,
        #   phase 0 에서 정책의 그리퍼 지령은 기록되지만 실행되지 않는다(그 차원 gradient 가
        #   환경 응답과 무관해진다). obs 가 1 늘어난다 — **fresh 학습 전용**.
        self.observations.policy.gripper_gate = ObsTerm(func=rewards.gripper_gate_open)
        self.observations.policy.joint_pos.params["asset_cfg"] = _left_joints()
        self.observations.policy.joint_vel.params["asset_cfg"] = _left_joints()
        self.rewards.joint_vel.params["asset_cfg"] = _left_joints()

        # ── 리프트 판정: **이진 하드 게이트** ───────────────────────
        # ★★리프트 임계는 **놓인 컵의 원점 + 4 cm** 다(하드 게이트).
        #   08.22 연속 램프에서 되돌렸다 — 램프의 근거였던 "IK test3 이 총보상 149 인데
        #   3.6 mm 만 올렸다"는 게이트 모양이 아니라 **임계값이 스폰보다 낮았던 것**이
        #   원인이었다(0.27709 < 0.29209). 같은 하드 게이트를 제대로 준 관절공간 런은
        #   실제로 들어 올렸다: test13 lift 0.83 / test16 lift 0.84.
        #   근거 전문은 `grasp_left_rewards._held` docstring 과
        #   `log/rl_games/open-grip/left/grasp-sensor/analysis.md`.
        for _term in (
            self.rewards.lifting_object,
            self.rewards.object_goal_tracking,
            self.rewards.object_goal_tracking_fine_grained,
        ):
            _term.params["minimal_height"] = P.MINIMAL_LIFT_HEIGHT
            _term.params["ramp_zero_z"] = P.LIFT_RAMP_ZERO_Z
            _term.params["enclose_half_width"] = P.JAW_ENCLOSE_HALF_WIDTH
            _term.params["pad_offset"] = P.JAW_PAD_OFFSET
            _term.params["lat_ok"] = P.GRASP_GATE_LATERAL_OK
            _term.params["along_ok"] = P.GRASP_GATE_ALONG_OK
            # ★SceneEntityCfg 는 매니저가 제자리 변경하는 가변 객체다 — term 마다 새 인스턴스.
            _term.params["jaw_cfg"] = SceneEntityCfg(
                "robot", body_names=list(P.GRIPPER_FINGER_BODIES)
            )

        # ── 리프트 판정에 "쥐고 있는가"를 AND ────────────────────────
        # ★★weight 는 그대로 두고 **판정 함수만** 바꾼다. z 만 보는 레퍼런스 판정으로는
        #   컵을 위로 쳐 날리는 것이 최적 전략이 되기 때문이다(test3 실증: 리프트 판정
        #   85.9% 동안 TCP–컵 평균 3044 mm). 자세한 근거는 grasp_left_rewards 참조.
        #   ⚠ goal-tracking 두 개도 내부에서 z 게이트를 직접 계산하므로 함께 교체해야 한다 —
        #     하나라도 남기면 그쪽으로 같은 hack 이 되살아난다.
        self.rewards.lifting_object.func = rewards.object_is_held_and_lifted
        self.rewards.lifting_object.params["max_ee_distance"] = P.GRASP_MAX_EE_DISTANCE
        for _term in (
            self.rewards.object_goal_tracking,
            self.rewards.object_goal_tracking_fine_grained,
        ):
            _term.func = rewards.object_goal_distance_when_held
            _term.params["max_ee_distance"] = P.GRASP_MAX_EE_DISTANCE

        # ── 파지 자세 보너스 (신설) ─────────────────────────────────
        # ★★자세는 **연속 보너스로만** 유도한다. 게이트로 넣으면 파지 중 필연적인 흔들림이
        #   전부 차단돼 양의 보상이 0 이 되고, 남은 것이 페널티뿐이라 **에피소드를 빨리
        #   끝내는 것이 최적**이 된다. 실제로 그렇게 죽였다(test6/test7):
        #       lifting 6.14 → 0.0000 / 에피소드 130 → 13 / 총보상 +34.9 → −0.46
        #   별도 term 이라 TFEvents 에 로깅돼 자세 개선을 학습 중 관측할 수 있다.
        # ── 평활화 페널티 커리큘럼 시점만 뒤로 민다 ─────────────────
        # 항도 weight 도 레퍼런스 그대로다. **켜는 시점**만 옮긴다. 근거는 프리셋
        # `ACTION_PENALTY_CURRICULUM_STEPS` 주석에 test15 붕괴 로그와 함께 있다.
        for _curr in (self.curriculum.action_rate, self.curriculum.joint_vel):
            _curr.params["num_steps"] = P.ACTION_PENALTY_CURRICULUM_STEPS

        # ── 액션 jerk 페널티는 **배선하지 않는다** ──────────────────
        # ★한때 넣었다가 뺐다. 근거: 그 처방은 test12 의 고주파 채터링(방향 반전 68.6%,
        #   2차 차분 > 1차)을 보고 쓴 것인데, test13 에서 그 증상이 사라졌다(반전 19.9%).
        #   그리고 레퍼런스식 커리큘럼(action_rate/joint_vel 를 10000 step 에 1000 배)이
        #   **이미 작동 중**임을 학습 곡선이 보여준다 — 관절 목표 도약이
        #       epoch 1150: 10.8° → 1450: 7.45°/스텝 으로 단조 감소, 끝에서도 감소 중.
        #   즉 평활화에 부족한 것은 새 항이 아니라 **학습 시간**이었다(1500 에서 잘렸다).
        #   `rewards.ActionJerkL2` 는 남겨 두되, 커리큘럼이 평탄해진 뒤에도 진동이 남을
        #   때에만 꺼내 쓴다. 한 런에 한 가설만 바꾼다.

        # ── 도달 보상의 **목표 높이 교정** (08.22) ──────────────────
        # ★★레퍼런스 `object_ee_distance` 는 컵 **원점**을 겨냥하는데, 우리 shaker 는
        #   원점이 상면 +92 mm 로 **그리퍼 통과 대역(+10~85 mm) 밖**이다. 그 높이의 컵
        #   지름 88 mm > 개구 84.5 mm 라 턱이 물리적으로 못 들어간다.
        #   즉 도달 보상이 학습 내내 **들어갈 수 없는 높이**를 가리키고 있었다.
        #   G3 실측: 컵 원점 겨냥 시 진입 TCP 오차 100.2 mm → 파지 대역 겨냥 시 70.7 mm.
        #   std·weight 는 레퍼런스 그대로(0.1 / 1.1) — 목표점만 옮긴다.
        self.rewards.reaching_object = RewTerm(
            func=rewards.ee_grasp_point_distance,
            weight=1.1,
            params={"std": 0.1, "grasp_offset": P.CUP_ORIGIN_TO_GRASP_Z},
        )

        # ── 컵이 턱 사이에 들어왔는가 (08.22 신설) ─────────────────
        # ★★"이번 런에서 파지 실패가 확인되면 그때 꺼내 쓴다"고 적어 뒀던 조건이 **충족됐다.**
        #   fab_test1 이 684 epoch 동안 lifting 정확히 0 이었고, 결정론 프로브가 원인을 짚었다:
        #       턱축까지 수직 최선 36.5 mm(≈컵 반경) · 개도 3.1 mm · '열기' 지령 0.0%
        #   = **주먹을 쥔 채 컵 옆구리를 누르고 있었다.** 닫힌 턱에는 컵이 들어갈 자리가 없다.
        #   성공한 test17 은 같은 자로 수직 최선 0.4 mm · 개도 26.5 mm 다.
        #   ⚠ 옛 `gripper_closure_on_cup` 을 그대로 꺼내 쓰면 안 됐다 — closure 를 곱해
        #     **닫을수록 커지므로** 관측된 실패 행동을 그대로 보상한다. enclose 로 교체했다.
        #   ⚠ 관절공간 트랙(test17)에도 함께 들어간다 — 항이 태스크의 올바른 서술이고,
        #     test17 실측 상태에서 이미 만점에 가까워 반대 압력이 없다. 다만 총보상 기준선이
        #     최대 +3.0 이동하므로 test17 의 171.7 과 직접 비교하지 말 것.
        self.rewards.cup_between_jaws = RewTerm(
            func=rewards.cup_between_jaws,
            weight=P.BETWEEN_JAWS_REWARD_WEIGHT,
            params={
                "along_std": P.JAW_ALONG_STD,
                "lateral_std": P.JAW_LATERAL_STD,
                "enclose_half_width": P.JAW_ENCLOSE_HALF_WIDTH,
                "enclose_floor": P.JAW_ENCLOSE_FLOOR,
                "pad_offset": P.JAW_PAD_OFFSET,
                # ★SceneEntityCfg 는 가변 객체다 — term 마다 **새 인스턴스**여야 한다.
                "robot_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
            },
        )

        # ── 감싼 상태에서 닫기 (08.23 신설) ────────────────────────
        # ★★fab_test4 가 이 구멍을 드러냈다: enclose 0.845 로 턱은 컵을 잘 감쌌는데
        #   '열기' 지령 78.0% · 거의 닫힘 0.0% — **한 번도 닫지 않는다.**
        #   닫는 것을 보상하는 항이 없고, 닫다가 컵이 밀리면 cup_between_jaws 를 잃으니
        #   닫지 않는 것이 최적이었다. 리프트는 닫아야만 생기는데 닫을 이유가 없다(닭-달걀).
        #   ⚠ enclose 를 곱하므로 옛 주먹 해킹(fab_test1, enclose 0.026)은 0 이다.
        self.rewards.grip_closure_when_enclosed = RewTerm(
            func=rewards.grip_closure_when_enclosed,
            weight=P.CLOSURE_WHEN_ENCLOSED_WEIGHT,
            params={
                "along_std": P.JAW_ALONG_STD,
                "lateral_std": P.JAW_LATERAL_STD,
                "enclose_half_width": P.JAW_ENCLOSE_HALF_WIDTH,
                "pad_offset": P.JAW_PAD_OFFSET,
                "open_pos": P.GRIPPER_OPEN_POS,
                "drive_joint": P.GRIPPER_DRIVE_JOINT,
                # ★SceneEntityCfg 는 가변 객체다 — term 마다 새 인스턴스여야 한다.
                "robot_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
            },
        )

        # ── 목표에서 정지 보너스 (신설) ─────────────────────────────
        # 레퍼런스 goal-tracking 은 **거리만** 본다. "옮겨서 가만히 세워 둔다"를 표현하려면
        # 속도 항이 필요하다. 여기도 게이트가 아니라 보너스다.
        self.rewards.settled_at_goal = RewTerm(
            func=rewards.object_settled_at_goal,
            weight=P.SETTLE_REWARD_WEIGHT,
            params={
                "std": P.SETTLE_POS_STD,
                "lin_vel_std": P.SETTLE_LIN_VEL_STD,
                "ang_vel_std": P.SETTLE_ANG_VEL_STD,
                "minimal_height": P.MINIMAL_LIFT_HEIGHT,
                "ramp_zero_z": P.LIFT_RAMP_ZERO_Z,
                "enclose_half_width": P.JAW_ENCLOSE_HALF_WIDTH,
                "pad_offset": P.JAW_PAD_OFFSET,
                "lat_ok": P.GRASP_GATE_LATERAL_OK,
                "along_ok": P.GRASP_GATE_ALONG_OK,
                "jaw_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                "max_ee_distance": P.GRASP_MAX_EE_DISTANCE,
                "command_name": "object_pose",
            },
        )

        # ── dwell 보너스 (fab_test13 신설) ─────────────────────────
        # settled_at_goal(순간 품질)은 스쳐 지나가도 지급된다 — fab_test12 는 목표 100 mm
        # 옆 순회가 국소최적으로 굳었다. dwell 은 품질 > 임계의 **연속 유지**에만 지급해
        # "머무름"과 "지나감"의 리턴을 가른다. 임계 근거는 preset DWELL_* 주석(실측).
        self.rewards.dwell_at_goal = RewTerm(
            func=rewards.DwellSettledAtGoal,
            weight=P.DWELL_REWARD_WEIGHT,
            params={
                "q_thresh": P.DWELL_Q_THRESH,
                "hold_steps": P.DWELL_HOLD_STEPS,
                "std": P.SETTLE_POS_STD,
                "lin_vel_std": P.SETTLE_LIN_VEL_STD,
                "ang_vel_std": P.SETTLE_ANG_VEL_STD,
                "minimal_height": P.MINIMAL_LIFT_HEIGHT,
                "ramp_zero_z": P.LIFT_RAMP_ZERO_Z,
                "enclose_half_width": P.JAW_ENCLOSE_HALF_WIDTH,
                "pad_offset": P.JAW_PAD_OFFSET,
                "lat_ok": P.GRASP_GATE_LATERAL_OK,
                "along_ok": P.GRASP_GATE_ALONG_OK,
                # ★SceneEntityCfg 는 가변 객체 — settled_at_goal 과 공유 금지, 새 인스턴스.
                "jaw_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                "max_ee_distance": P.GRASP_MAX_EE_DISTANCE,
                "command_name": "object_pose",
            },
        )

        self.rewards.grasp_pose = RewTerm(
            func=rewards.held_with_good_pose,
            weight=P.GRASP_POSE_REWARD_WEIGHT,
            params={
                "minimal_height": P.MINIMAL_LIFT_HEIGHT,
                "ramp_zero_z": P.LIFT_RAMP_ZERO_Z,
                "enclose_half_width": P.JAW_ENCLOSE_HALF_WIDTH,
                "pad_offset": P.JAW_PAD_OFFSET,
                "lat_ok": P.GRASP_GATE_LATERAL_OK,
                "along_ok": P.GRASP_GATE_ALONG_OK,
                "jaw_cfg": SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES)),
                "max_ee_distance": P.GRASP_MAX_EE_DISTANCE,
                "body_name": P.GRIPPER_BASE_BODY,
                "upright_zero_at_cos": P.CUP_UPRIGHT_ZERO_AT_COS,
            },
        )
        # ★진단(weight 0) — 게이트 진입 비율. 이번 런의 1차 관전 지표라 반드시 로깅한다.
        # ⚠ weight 0 이면 RewardManager 가 w·term·dt 를 로깅해 **항상 0.0000** 이 찍힌다
        #   (fab_test12 에서 1차 관전 지표를 통째로 잃었다). 0.001 은 총보상 ~110/step 대비
        #   1e-5 수준이라 학습에 무영향이면서 TFEvents 에서 게이트 개방률을 읽을 수 있다.
        self.rewards.gate_rate = RewTerm(func=rewards.gripper_gate_rate, weight=0.001, params={})

        self.terminations.object_dropping.params["minimum_height"] = P.OBJECT_DROP_HEIGHT


@configclass
class GraspLeftGripperEnvCfg_PLAY(GraspLeftGripperEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
