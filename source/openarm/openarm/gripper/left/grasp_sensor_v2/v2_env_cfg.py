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

"""`open-grip_l_grasp_sensor_v2` — 계단식 보상 재설계판.

**골격은 v1 을 그대로 쓰고 보상·관측만 새로 쓴다.** v1 의 CLAUDE.md 가 "바꾸지 말 것"으로
못 박은 성질들(액션 `scale=0.5`/`use_default_offset` · 홈 자세 · 이진 그리퍼 · fabric 배선 ·
`decimation=2` · `episode_length_s` · 종료 조건 · 자산 · 스폰)은 lift 레시피가 단순한 제어로도
학습되는 이유라 하나라도 깨면 그 장점이 사라진다. v2 가 바꾸는 것은 **보상 계층과 관측
계층**뿐이다.

바뀌는 것 요약 (근거는 각 항 주석):
  1. 가산형 10 항 → **계단 4 단** (Lee et al.)
  2. goal 채점 **TCP → 컵**, 목표 상자를 실측 파지 오프셋만큼 평행이동
  3. 조건부 추종을 강제하는 **지시함수 단계** 신설 → 08.29 네 조건 **AND** 로 재정의
  4. obs 에 `목표 − 컵` 상대벡터와 컵 직립 추가

★★08.29 라운드 3 — 계단을 **연속·단조**로 재구성했다. 값이 "그 단계의 양"이 아니라
  "다음 문턱까지의 진행도"가 되어 고원과 절벽이 동시에 사라진다. 근거는
  `v2_preset.py` 의 D1~D3 주석과 `v2_stages.all_stages` 를 볼 것.
"""

from __future__ import annotations

import os as _os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..grasp_sensor.grasp_left_fab_env_cfg import GraspLeftGripperFabEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import v2_curriculum as C
from . import v2_events as E
from . import v2_observations as obs
from . import v2_preset as P
from . import v2_rewards as R
from . import v2_terminations as T


def _jaw() -> SceneEntityCfg:
    """턱 기하용 cfg. ★`SceneEntityCfg` 는 가변 객체다 — term 마다 **새 인스턴스**여야
    한다(같은 객체를 공유하면 매니저 resolve 가 서로 덮어쓴다)."""
    return SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))


def _stage_params() -> dict:
    """네 단계가 **같은 기하**를 보도록 인자를 한 곳에서 만든다."""
    return {
        "command_name": "object_pose",
        "robot_cfg": SceneEntityCfg("robot"),
        "jaw_cfg": _jaw(),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "object_cfg": SceneEntityCfg("object"),
    }


# ---------------------------------------------------------------------------
# 환경변수 스위치 — **하나만 남았다**
# ---------------------------------------------------------------------------
# 09.03 정리: 라운드 3~30 의 실험 스위치 22 종은 v2E29 확정과 함께 제거하고, 이긴 쪽
# 값을 아래 cfg 필드의 기본값으로 **동결**했다. 기각된 분기는 코드에서 지웠다
# (경위·근거는 `docs/grasp_sensor_v2_rounds.md`).
#
# `DWELL_END` 만 남긴 이유: 판정 프로토콜이 이걸 끄고 잰다. 학습은 "목표 체류 N 스텝"
# 에서 에피소드를 끊지만(=1), 결정론 프로브는 250 스텝을 끝까지 굴려 결말을 봐야
# 하므로 `HDGP_V2_DWELL_END=0` 으로 호출한다.
_V2_DWELL_END = _os.environ.get("HDGP_V2_DWELL_END", "1") == "1"


@configclass
class GraspLeftV2EnvCfg(GraspLeftGripperFabEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이송 — Lee 계단식 보상."""

    # ── v2E29 확정 설정 (09.03 동결) ─────────────────────────────
    # 값은 cfg 필드에 남으므로 `env.yaml` dump 가 "무슨 설정으로 돌았는지"를 증명한다.
    # 실험적으로 바꿔야 하면 hydra CLI 로 덮는다 (`env.v2_dr=False` 등).
    # ⚠⚠ hydra 오버라이드는 `__post_init__` **뒤에** 적용된다. 그래서 여기서 읽어
    #   다른 객체에 **구워 넣은 값**은 오버라이드가 안 닿는다(항이 등록되냐 마냐를
    #   가르는 `if self.v2_*` 분기도 마찬가지다). 런타임에 다시 읽어야 하는 값은
    #   term params 로 굽지 말고 해당 term 이 `env.cfg` 에서 직접 읽게 할 것.
    #   ★F2 가 이걸로 E29 의 완전한 재실행이 되어 200 epoch 을 버렸다.
    v2_rot_wide: bool = True             # palm 회전 박스 ±60°
    v2_hold_premium: bool = True         # 지속 정착 누진 프리미엄
    v2_upright_shaping: bool = True      # 이송 구간 직립 셰이핑(처방 A)
    v2_adr_fixed_level: int = 4          # ADR 만렙 고정. −1 이면 사다리 승급·강등
    # ★★09.03 — **재소환을 끈다**(사용자 결정). 라운드 8 에서 "실패의 비용을 줄인다"는
    #   의도로 넣었는데, 비용을 0 으로 만들어 **전도가 학습 대상에서 빠졌다**:
    #     E29 결정론 실측 — 컵을 60° 이상 넘어뜨린 env **84.0%** · 78° 이상 52.1% ·
    #     재소환 누적 3,233회/1024env(에피소드당 3.2회). B25 는 89.6% · 4,264회로 더 심하다.
    #   실기에는 되돌림이 없으므로 그대로 **접근→전도→후퇴→재접근** 궤적이 된다
    #   (S2R 실기 관찰과 일치). 끄면 부모의 `object_dropping`(root z < 0.255)이 살아나
    #   전도가 다시 에피소드를 끝낸다 — 컵이 옆으로 누우면 원점이 0.244 로 내려간다.
    #   ⚠ 라운드 8 이전의 위험이 되살아난다: "낙하=종료=잔여 몰수"가 손익분기 실패율
    #     22.9% 로 가혹했다. 지금은 목표 체류 truncation 이 있어 전제가 다르다 — 이 판이
    #     그 재검증이다.
    v2_respawn: bool = False             # 낙하/전도 시 에피소드 내 재소환 (끔)
    v2_dr: bool = True                   # DR(마찰) + ADR(질량·스폰·목표·obs bias)
    v2_dwell_end: bool = _V2_DWELL_END   # 목표 체류 N 스텝 → 에피소드 종료(truncation)
    v2_zfloor: bool = True               # 경로 손끝-테이블 z 마진 벌점
    v2_home_low: bool = True             # 리셋 홈 B100 (낮고 덜 기운 자세)
    v2_vendor_gains: bool = True         # 팔 액추에이터를 실기 벤더 kp/kd 로
    # ★★09.03 — **리프트 전용 과제**(사용자 결정: "goal 은 필요없고 lift 만").
    #   계단이 grasp → lift 2 단이 되고 리프트가 만점이다. 목표 상자는 obs 에만 남는다.
    #   끊는 조건도 목표 도달 → 리프트 유지로 바뀐다(`LiftDwellDone`).
    v2_lift_only: bool = True

    def __post_init__(self):
        super().__post_init__()

        # ── 목표 상자를 **컵 좌표**로 평행이동 ──────────────────────
        # v1 은 goal 을 TCP 로 채점하면서 합격은 컵으로 판정했다. t79 best 프로브 실측
        # 리프트 후 `컵 − TCP` = 37.2 mm ⇒ TCP 가 목표에 완벽히 도달해도 컵은 37 mm
        # 남아 합격 예산 57 mm 의 65% 를 계통 오프셋이 먼저 먹었다.
        # 상자를 통째로 그만큼 옮기면 **컵이 새 상자에 놓일 때 TCP 는 옛 상자(= TCP 제약
        # IK 로 도달성이 검증된 자리)** 에 있다 — 검증을 버리지 않고 채점만 정렬한다.
        # ★오프셋은 구 값(z +12 mm)으로 확정했다. 라운드 4 에서 실측 오프셋(z +30 mm)과
        #   대조한 결과, 상자를 30 mm 올리면 목표가 팔의 도달 봉투 위쪽으로 밀려
        #   이득이 없었다. v2E29 까지 전 판이 이 값으로 돌았다.
        _off = P.GRASP_OFFSET_ROOT_V1
        _goal = tuple(P.GOAL_POINT[i] + _off[i] for i in range(3))
        self.commands.object_pose.ranges.pos_x = (_goal[0] - P.GOAL_JITTER_V2[0],
                                                  _goal[0] + P.GOAL_JITTER_V2[0])
        self.commands.object_pose.ranges.pos_y = (_goal[1] - P.GOAL_JITTER_V2[1],
                                                  _goal[1] + P.GOAL_JITTER_V2[1])
        self.commands.object_pose.ranges.pos_z = (_goal[2] - P.GOAL_JITTER_V2[2],
                                                  _goal[2] + P.GOAL_JITTER_V2[2])

        # ── 보상 전면 교체 ────────────────────────────────────────
        # 부모(v1)가 세운 10 항 + 게이트 스택을 **전부 지우고** 계단으로 간다.
        # 가산형이 만든 두 병리:
        #   · "높이 들고 가만히"가 충분히 이득 → 이송을 안 배운다
        #   · 36 점이 게이트 하나를 공유 → **이동이 도박**(프로브: 이동 중 0.944 vs 정지 1.000)
        for _name in list(self.rewards.__dict__.keys()):
            if not _name.startswith("_"):
                setattr(self.rewards, _name, None)

        self.rewards.staircase = RewTerm(
            func=R.Staircase,
            weight=P.STAIRCASE_WEIGHT,
            params={**_stage_params(), "use_sr": False,
                    # ★리프트 전용에서는 둘 다 끈다 — hold 는 목표 도달(`settle_success`)
                    #   기준이고 upright 셰이핑은 `idx >= 2`(이송) 게이트라 둘 다
                    #   도달 불가다. 켜두면 죽은 계산만 남는다.
                    "hold_weight": (0.0 if self.v2_lift_only
                                    else (P.HOLD_WEIGHT if self.v2_hold_premium else 0.0)),
                    "upright_weight": (0.0 if self.v2_lift_only
                                       else (P.UPRIGHT_WEIGHT
                                             if self.v2_upright_shaping else 0.0))},
        )
        # ⚠ `use_sr`(Hundt 진행도 역전 차단)은 기본 off. 원문은 이산 primitive 태스크라
        #   안전하지만 우리는 50 Hz 연속제어라 경계 근처에서 매 스텝 보상을 0 으로 만들
        #   위험이 있다. 켜려면 **단일 변수로** 켠다.

        self.rewards.action_l2 = RewTerm(func=R.action_l2, weight=P.ACTION_L2_WEIGHT)
        self.rewards.action_rate = RewTerm(func=R.action_rate_l2,
                                           weight=P.ACTION_RATE_WEIGHT)
        # ★커리큘럼을 걸지 않는다 — t73/t75 가 발동 시점(ep1500)에 정확히 꺾였고,
        #   `action_rate_l2` 가 재는 것의 대부분은 정책의 거칢이 아니라 σ 였다.
        #   라운드 23 에서 계단 상승 커리큘럼을 다시 시도했으나 채택하지 않았다.
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # ── 진단 (weight 0) ───────────────────────────────────────
        # 계단은 보상 항이 하나라 이게 없으면 안이 안 보인다. 이 저장소의
        # `reward_manager.py` 패치가 weight 0 항의 **원값**을 누적한다.
        for _n, _f in (("diag_stage", R.diag_stage_index),
                       ("diag_v_stage", R.diag_v_stage),
                       ("diag_r_grasp", R.diag_r_grasp),
                       ("diag_r_lift", R.diag_r_lift),
                       ("diag_r_transport", R.diag_r_transport),
                       # ★구 `diag_r_settle`(네 인자 곱, 6 판 최대 0.0106)의 후계.
                       #   뜻이 완전히 달라져 같은 키를 재사용하지 않는다.
                       ("diag_stage3_ok", R.diag_stage3_ok),
                       # ★라운드 27 — 접근 자세. `diag_appr_angle / diag_appr_steps`
                       #   가 평균 접근각(도)이고, tcp_* 도 같은 분모를 쓴다.
                       ("diag_appr_steps", R.diag_appr_steps),
                       ("diag_appr_angle", R.diag_appr_angle),
                       ("diag_tcp_x", R.diag_tcp_x),
                       ("diag_tcp_y", R.diag_tcp_y),
                       ("diag_tcp_z", R.diag_tcp_z)):
            setattr(self.rewards, _n, RewTerm(func=_f, weight=0.0, params=_stage_params()))

        _goal_only = {"command_name": "object_pose",
                      "robot_cfg": SceneEntityCfg("robot"),
                      "object_cfg": SceneEntityCfg("object")}
        # ★최종 합격 판정 지표. v2 는 보상도 같은 값으로 재므로 보상 최적점과 합격
        #   기준이 같은 곳을 가리킨다.
        self.rewards.diag_cup_goal_dist = RewTerm(func=R.diag_cup_goal_dist,
                                                  weight=0.0, params=dict(_goal_only))
        # ★조건부 추종의 직접 지표 — 고정점 전략의 이론 상한이 37.4% 다.
        self.rewards.diag_at_goal = RewTerm(func=R.diag_at_goal,
                                            weight=0.0, params=dict(_goal_only))
        self.rewards.diag_cup_speed = RewTerm(func=R.diag_cup_speed, weight=0.0, params={})
        self.rewards.diag_cup_upright = RewTerm(func=R.diag_cup_upright, weight=0.0, params={})
        # ★합격 판정 지표. 순간속도(`diag_cup_speed`)와 **나란히** 찍어야 진동을
        #   정지로 오독했는지가 로그만으로 드러난다.
        self.rewards.diag_cup_net_speed = RewTerm(func=R.diag_cup_net_speed,
                                                  weight=0.0, params={})
        # ★진행도 3 종 — `v_2 = r_close · P_dist · P_still · P_upright` 의 인자들.
        #   따로 찍어야 "이송이 왜 안 되는가"가 로그만으로 갈린다: 거리를 못 좁히는
        #   것인지 · 못 멈추는 것인지 · 컵이 기우는 것인지.
        for _n, _f in (("diag_p_dist", R.diag_p_dist),
                       ("diag_p_still", R.diag_p_still),
                       ("diag_p_upright", R.diag_p_upright),
                       ("diag_p_center", R.diag_p_center)):
            setattr(self.rewards, _n, RewTerm(func=_f, weight=0.0, params=dict(_goal_only)))
        # ★접근각을 **보상으로** 세우려던 라운드 20~22 는 전부 기각됐다(5 종 시도).
        #   실제 해법은 파지 대역을 판 위 80~150 mm 로 올리는 것이었다 — 낮은 파지점이
        #   팔을 과신전시켜 기울기를 강제하고 있었다. 접근각 118° → 98.6°.

        # ── 라운드 18: 경로 z 마진 벌점 ───────────────────────────
        #   ★파지와 **역방향**인 항이다. 마진(판 위 30 mm) 위에서는 정확히 0 이라
        #     파지 구간에 gradient 를 만들지 않는다. v2 파지 대역은 판 위 80~150 mm
        #     이므로 이 항과 파지 구간은 **아예 안 겹친다**.
        if self.v2_zfloor:
            self.rewards.tip_floor = RewTerm(
                func=R.tip_floor_penalty, weight=P.TIP_FLOOR_WEIGHT,
                params={"tip_cfg": _jaw()})
            self.rewards.diag_tip_height = RewTerm(
                func=R.diag_tip_height, weight=0.0, params={"tip_cfg": _jaw()})
            self.rewards.diag_tip_violation = RewTerm(
                func=R.diag_tip_violation, weight=0.0, params={"tip_cfg": _jaw()})

        # ── 리셋 홈 B100 ─────────────────────────────────────────
        #   ★액션 0 = 이 자세다(`use_default_offset`). 홈을 바꾸면 액션 상자가 통째로
        #     옮겨가고 fabric 의 널스페이스 앵커까지 바뀌므로 **기존 체크포인트와
        #     호환되지 않는다** — 홈을 건드리는 판은 항상 FRESH 다.
        #   ★홈 후보 4 종(J147 · A94 수평 · B100 · C96 · 라운드 30 고홈)을 전부 돌린
        #     결과 B100 이 이겼다. 특히 "홈을 판에서 띄우면 턱 긁힘이 준다"는 직관은
        #     **반대로** 나왔다(라운드 30: 접근각 116° · 긁힘 42 배). 홈은 첫 순간만
        #     정하고, 그 뒤는 fabric 이 액션 박스 중심으로 끌고 간다.
        if self.v2_home_low:
            self.scene.robot.init_state.joint_pos.update(P.LEFT_ARM_HOME_LOW)

        # ── 라운드 24: 실기 벤더 게인 정합 ────────────────────────
        #   부모(`grasp_left_fab_env_cfg`)가 꽂은 ARM_IK_STIFFNESS/DAMPING 을 덮어쓴다.
        #   ★sim 이 실기보다 4~10배 뻣뻣해 정책 진동이 팔에 그대로 실린다 — 벤더값이
        #     진실이므로 그쪽에 맞춘다. 근거·주의는 `v2_preset.LEFT_ARM_VENDOR_*` 주석.
        #   ⚠ 동특성이 바뀌므로 기존 체크포인트와 **호환되지 않는다**(FRESH 전용).
        if self.v2_vendor_gains:
            self.scene.robot.actuators["left_arm"].stiffness = dict(
                P.LEFT_ARM_VENDOR_STIFFNESS)
            self.scene.robot.actuators["left_arm"].damping = dict(
                P.LEFT_ARM_VENDOR_DAMPING)

        # ── 라운드 17: 목표 체류 N 스텝이면 에피소드 종료 ──────────
        #   ⚠⚠ `time_out=True` (truncation) 여야 한다. 진짜 종료로 두면 부트스트랩이
        #      끊겨 성공이 곧 남은 보상 포기가 되고, 정책은 목표 밖을 맴돌며 stage 3
        #      보상을 계속 빠는 쪽을 배운다. 계약 테스트가 이 플래그를 고정한다.
        if self.v2_dwell_end and self.v2_lift_only:
            # ★리프트를 든 채 유지하면 끊는다. 보상의 stage 1 과 **같은 자**를 쓴다.
            self.terminations.goal_dwell = DoneTerm(
                func=T.LiftDwellDone, time_out=True,
                params={"jaw_cfg": _jaw(), "object_cfg": SceneEntityCfg("object")})
            self.rewards.diag_goal_dwell = RewTerm(
                func=T.diag_lift_dwell, weight=0.0, params={})
        elif self.v2_dwell_end:
            self.terminations.goal_dwell = DoneTerm(
                func=T.GoalDwellDone, time_out=True,
                params={"command_name": "object_pose",
                        "robot_cfg": SceneEntityCfg("robot"),
                        "jaw_cfg": _jaw(),
                        "object_cfg": SceneEntityCfg("object")})
            self.rewards.diag_goal_dwell = RewTerm(
                func=T.diag_goal_dwell, weight=0.0, params={})

        # ── 라운드 8 후보: 낙하/전도 재소환 ──────────────────────
        if self.v2_respawn:
            # ★종료 항을 꺼야 한다 — 재소환 문턱과 같은 높이라 종료가 먼저 발화한다.
            #   time_out(truncation+bootstrap)은 그대로 남아 에피소드는 항상 완주한다.
            self.terminations.object_dropping = None
            self.events.respawn_cup = EventTermCfg(
                func=E.respawn_dropped_cup, mode="interval",
                # step_dt(0.02 s)와 같게 — 사실상 매 스텝 검사. 한 스텝 늦어도
                # 떨어진 컵은 떨어진 채라 놓치지 않는다.
                interval_range_s=(0.02, 0.02),
                params={"object_cfg": SceneEntityCfg("object"),
                        "ee_frame_cfg": SceneEntityCfg("ee_frame")})

        # ── 라운드 9: DR + ADR ────────────────────────────────────
        if self.v2_dr:
            # 마찰 = 정적 DR (버킷은 startup 에만 배정 가능 — 사다리 불가).
            #   ★현재 컵 마찰은 조용한 기본값 0.5 다("안 적은 물리 파라미터" 메모리).
            self.events.dr_cup_friction = EventTermCfg(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg("object", body_names=".*"),
                        "static_friction_range": P.DR_FRICTION_STATIC,
                        "dynamic_friction_range": P.DR_FRICTION_DYNAMIC,
                        "restitution_range": P.DR_RESTITUTION,
                        "num_buckets": P.DR_FRICTION_BUCKETS,
                        "make_consistent": True})
            # 질량 = 리셋마다 scale. 레벨 0 = (1.0, 1.0) — ADR 이 넓힌다.
            self.events.dr_cup_mass = EventTermCfg(
                func=mdp.randomize_rigid_body_mass, mode="reset",
                params={"asset_cfg": SceneEntityCfg("object", body_names=".*"),
                        "mass_distribution_params": (1.0, 1.0),
                        "operation": "scale", "distribution": "uniform",
                        "recompute_inertia": True})
            # obs 노이즈 — 컵 위치 관측에만 낀다(보상·판정은 ground truth).
            #   스텝 잡음 ±3 mm(정적) + 에피소드 bias(ADR knob ④, 레벨 0 = 0).
            self.observations.policy.object_position = ObsTerm(
                func=obs.object_position_noisy,
                params={"robot_cfg": SceneEntityCfg("robot"),
                        "object_cfg": SceneEntityCfg("object"),
                        "jaw_cfg": _jaw(),
                        "step_noise": P.DR_OBS_STEP_NOISE})
            self.events.dr_obs_bias = EventTermCfg(
                func=E.resample_obs_bias, mode="reset",
                params={"bias_range": 0.0})
            # ★08.30 — 턱 마찰도 랜덤화한다. 지금까지 컵만 랜덤화하고 그리퍼 턱은
            #   고정값 하나로 학습했다(실물 패드와 다르면 그대로 sim2real 격차).
            self.events.dr_jaw_friction = EventTermCfg(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": _jaw(),
                        "static_friction_range": P.DR_JAW_FRICTION_STATIC,
                        "dynamic_friction_range": P.DR_JAW_FRICTION_DYNAMIC,
                        "restitution_range": P.DR_RESTITUTION,
                        "num_buckets": P.DR_FRICTION_BUCKETS,
                        "make_consistent": True})
            # ADR 사다리 — 스폰·목표·질량·obs bias 를 넓힌다. `Curriculum/adr`.
            #   ★`v2_adr_fixed_level >= 0` 이면 그 레벨에 **고정**하고 승급·강등을 끈다.
            #     라운드 13 대조에서 사다리(A)와 만렙 고정(B)을 갈랐고 B 가 이겼다 —
            #     처음부터 전범위로 두는 쪽이 사다리보다 낫다.
            #     노브를 여기서 직접 건드리지 않는 이유: 스폰 상자는 부모 골격 소유라
            #     계약이 금지한다. 같은 `_apply` 경로를 써야 만렙 정의가 일치한다.
            #   ⚠ `fixed_level` 을 params 로 넘기지 않는다 — 여기서 넘기면 값이
            #     `__post_init__` 시점에 구워져 hydra 오버라이드가 조용히 무시된다.
            #     `ADRLadder` 가 `env.cfg.v2_adr_fixed_level` 을 런타임에 읽는다.
            self.curriculum.adr = CurrTerm(func=C.ADRLadder)

        # ★라운드 7 — hold 카운터 관찰(성과 지표: 흔들기는 이 값을 못 올린다)
        self.rewards.diag_hold = RewTerm(func=R.diag_hold, weight=0.0, params={})
        # ★★합격 판정 — 승급 문턱에서 뺀 정지·직립이 여기 남아 과제 정의를 지킨다.
        self.rewards.diag_success = RewTerm(
            func=R.diag_success, weight=0.0,
            params={"command_name": "object_pose", "robot_cfg": SceneEntityCfg("robot"),
                    "jaw_cfg": _jaw(), "object_cfg": SceneEntityCfg("object")})

        # 축 포화 — 절대 태스크공간 액션에서 상설 감시 지표다(v1 에서 y 99.1% 사망 이력).
        for _ax, _i in (("x", 0), ("y", 1), ("z", 2)):
            setattr(self.rewards, f"diag_act_{_ax}_mu",
                    RewTerm(func=R.diag_action_axis_mu, weight=0.0, params={"axis": _i}))
            setattr(self.rewards, f"diag_act_{_ax}_sat",
                    RewTerm(func=R.diag_action_axis_sat, weight=0.0, params={"axis": _i}))

        # ── 파지 대역을 **게이트에도** 넣는다 ────────────────────
        #   v2 는 판 위 80~150 mm 를 잡는다(v1 은 10~85). 보상(`v2_stages`)만 바꾸고
        #   그리퍼 게이트를 두면 "보상은 받는데 그리퍼가 안 열리는" 상태가 생긴다.
        self.actions.gripper_action.grasp_band = P.CUP_GRASP_BAND_AXIS

        # ── 제어 층 — palm 회전 박스 ±20° → ±60° ────────────────
        #   euler 박스 포화 98% 실측. 접근(ey 양)과 이송(ey 음)의 부호가 뒤집혀
        #   최소 40° 스팬이 필요하다.
        #   ★접근각을 이 박스의 `ey` 상한으로 자르려던 라운드 28 은 기각됐다 —
        #     fabric 은 위치·자세가 상충하면 **자세를 조용히 포기**해서, 명령을 잘라도
        #     실제 접근각이 안 따라온다.
        if self.v2_rot_wide:
            self.actions.arm_action.palm_max_pose_angle = P.PALM_MAX_POSE_ANGLE_WIDE

        # ★★회전 축(3·4·5) 진단 — **지금까지 계측 공백이었다.** 액션은 6 차원인데
        #   `diag_act_*` 는 위치 축 0·1·2 만 등록돼 있어 회전 포화를 한 번도 못 봤다.
        #   v1 에서 y 축 포화 99.1% 로 죽은 이력이 있는데 회전엔 같은 감시가 없었다.
        for _ax, _i in (("ez", 3), ("ey", 4), ("ex", 5)):
            setattr(self.rewards, f"diag_act_{_ax}_mu",
                    RewTerm(func=R.diag_action_axis_mu, weight=0.0, params={"axis": _i}))
            setattr(self.rewards, f"diag_act_{_ax}_sat",
                    RewTerm(func=R.diag_action_axis_sat, weight=0.0, params={"axis": _i}))

        # ── 관측 추가 ─────────────────────────────────────────────
        # 부모가 이미 `tcp_pos` + `palm_rot`(HDGP_OBS_SET=pose 기본)을 붙였다. 여기서는
        # **조건부 관계를 직접 주는 항**만 더한다. 45D → 49D.
        if P.OBS_GOAL_MINUS_CUP:
            self.observations.policy.goal_minus_cup = ObsTerm(func=obs.goal_minus_cup)
        self.observations.policy.cup_upright = ObsTerm(func=obs.cup_upright)


@configclass
class GraspLeftV2EnvCfg_PLAY(GraspLeftV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
