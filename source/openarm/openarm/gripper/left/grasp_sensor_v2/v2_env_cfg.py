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
# ★★08.29 라운드 3 실험 스위치 — 단일 변수는 **속도 입력 하나**뿐이다
# ---------------------------------------------------------------------------
#   arm A = False → 속도 입력 = 순간속도 (재구성 단독)
#   arm B = True  → 속도 입력 = 순변위   (재구성 + 순변위)
#
# 이 값은 `P_still` 과 stage 3 승급 조건의 속도 항 **양쪽 모두**에 같은 입력으로
# 들어간다. 식은 완전히 동일하고 입력만 바뀌므로 단일 변수가 구조적으로 성립한다.
#
# ★라운드 1·2 의 스위치는 제거했다(전부 실측 기각):
#   `HDGP_V2_REWARD_FIX`(R1 순변위 still + R2 stage 2 속도 shaping + R3 오프셋 z)
#   `HDGP_V2_ACTION_FIX`(A1 이송 국면 리미터 1/4)
#   R3(오프셋 z 30 mm)는 채점점·합격점 정렬이라 v2 의 전제로 **상시 적용**한다.
#
# ⚠ 환경변수로 받되 **cfg 필드에 박는다** — 그래야 `env.yaml` dump 에 남아
#   "무슨 설정으로 돌았는지"를 나중에 증명할 수 있다(학습 전 확인 규칙 ②).
_V2_STILL_NET = _os.environ.get("HDGP_V2_STILL_NET", "0") == "1"

# ★★08.29 라운드 4 — R3(목표 상자 이동)를 **다시 스위치로 되돌린다.**
#   라운드 3 에서 "v2 의 전제"라며 상시 적용했는데, 그 결과 baseline(run0-s43)과
#   목표 z 가 18 mm 달라져(0.415~0.515 vs 0.397~0.497) **7 판 치 비교가 통째로
#   근사치가 됐다**. `cupd` 124 mm vs 89 mm 격차의 최대 절반이 이 교란이다.
#   ⇒ 끌 수 있게 두고, 라운드 4 의 한 팔을 baseline 과 같은 상자로 돌려
#     "보상 효과"를 처음으로 깨끗하게 잰다.
_V2_GOAL_MEASURED = _os.environ.get("HDGP_V2_GOAL_MEASURED", "1") == "1"

# ★★08.29 라운드 6 — 제어 층 스위치 2 종. 기본 off 라 켜는 것은 명시적 행위다.
#   ROT   : palm 회전 박스 ±20° → ±60°  (euler 박스 포화 98% 실측)
#   ANCHOR: env 별 cspace 재앵커        (앵커가 액션 박스 밖 · env 공용 하나)
_V2_ROT_WIDE = _os.environ.get("HDGP_V2_ROT_WIDE", "0") == "1"
_V2_ANCHOR_RELATCH = _os.environ.get("HDGP_V2_ANCHOR_RELATCH", "0") == "1"

# ★★08.29 라운드 7 — 지속 정착 프리미엄(hold). '정점 후 붕괴'의 기대값 구조를 친다.
#   근거·수치는 `v2_preset.HOLD_WEIGHT` 주석. 기본 off.
_V2_HOLD = _os.environ.get("HDGP_V2_HOLD", "0") == "1"
_V2_UPRIGHT = _os.environ.get("HDGP_V2_UPRIGHT", "0") == "1"
_V2_STILL_SHAPE = _os.environ.get("HDGP_V2_STILL_SHAPE", "0") == "1"
_V2_STILL_GOAL = _os.environ.get("HDGP_V2_STILL_GOAL", "0") == "1"
_V2_ADR_FIXED = int(_os.environ.get("HDGP_V2_ADR_FIXED", "-1"))  # >=0 이면 레벨 고정

# ★★08.29 라운드 8 후보 — 낙하/전도 재소환. 종료 절벽을 회복 가능한 비용으로 바꾼다.
#   근거·수치는 `v2_events.py` 와 `v2_preset.RESPAWN_*` 주석. 기본 off.
_V2_RESPAWN = _os.environ.get("HDGP_V2_RESPAWN", "0") == "1"

# ★★08.30 라운드 9 — DR + ADR 사다리(질량·마찰·스폰·목표). 기본 off.
#   근거·범위는 `v2_preset.ADR_*`/`DR_*` 주석과 `v2_curriculum.py`.
_V2_DR = _os.environ.get("HDGP_V2_DR", "0") == "1"

# ★★08.31 라운드 17 — S2R 에서 드러난 두 건(사용자 지시 + 좌팔 보정 실측 문서).
#   HOME_J147 : 리셋 홈을 **J1·J4·J7 만** 쓰는 자세로 교체. 판 위 여유가 최저 링크
#               기준 59 → 122 mm 로 2.1 배가 된다(테이블 절대높이 미실측에 대한 마진).
#   DWELL_END : 목표 반경 안 연속 N 스텝이면 에피소드를 **truncation** 으로 끊는다.
#               이송까지가 과제이고 그 뒤 자세는 IK 가 잡는다.
#   근거·수치는 `v2_preset.LEFT_ARM_HOME_J147` / `EPISODE_DWELL_STEPS` 주석.
_V2_HOME_J147 = _os.environ.get("HDGP_V2_HOME_J147", "0") == "1"
_V2_DWELL_END = _os.environ.get("HDGP_V2_DWELL_END", "0") == "1"

# ★★08.31 라운드 18 — 경로 z 마진(좌팔 보정 문서 §2-2 권고). 기본 off.
#   손끝·TCP 가 판 위 `TIP_FLOOR_MARGIN` 아래로 내려가면 벌점. 근거는 preset 주석.
_V2_ZFLOOR = _os.environ.get("HDGP_V2_ZFLOOR", "0") == "1"

# ★★08.31 라운드 20 — 접근 자세 제약. 기본 off.
#   파지 전 그리퍼 +z 가 아래로 기울면 벌점(90° 이하는 0). 근거는 preset 주석 —
#   실측 접근 각도 117.5°, 90° 초과 env 100%.
_V2_APPR = _os.environ.get("HDGP_V2_APPR", "0") == "1"
# 라운드 22 Part 1 — 곱셈형 방향(접근 평균 품질 × 파지 후 계단)
_V2_DIRMUL = _os.environ.get("HDGP_V2_DIRMUL", "0") == "1"
# 라운드 22 Part 2 — 접근축을 수평으로 세운 리셋 홈
_V2_HOME_LEVEL = _os.environ.get("HDGP_V2_HOME_LEVEL", "0") == "1"
# 라운드 22b — 낮고 덜 기운 홈(B100). HOME_LEVEL(A94)과 배타적이다
_V2_HOME_LOW = _os.environ.get("HDGP_V2_HOME_LOW", "0") == "1"
# 라운드 22c — 중간 홈 C96(구 홈 높이 유지 + 각도 −14°)
_V2_HOME_MID = _os.environ.get("HDGP_V2_HOME_MID", "0") == "1"


@configclass
class GraspLeftV2EnvCfg(GraspLeftGripperFabEnvCfg):
    """왼팔 2지 그리퍼 shaker 파지·이송 — Lee 계단식 보상."""

    # 실험 스위치(런타임에 환경변수로 세팅되지만 값은 cfg 에 남는다)
    v2_still_net: bool = _V2_STILL_NET   # 속도 입력: False=순간속도 · True=순변위
    # 목표 상자 오프셋: True=v2 실측 46.1 mm(z+30) · False=구 값(z+12, run0 과 동일)
    v2_goal_measured: bool = _V2_GOAL_MEASURED
    v2_rot_wide: bool = _V2_ROT_WIDE            # palm 회전 박스 ±60°
    v2_anchor_relatch: bool = _V2_ANCHOR_RELATCH  # env 별 cspace 재앵커
    v2_hold_premium: bool = _V2_HOLD            # 지속 정착 누진 프리미엄
    v2_upright_shaping: bool = _V2_UPRIGHT     # 이송 구간 직립 셰이핑(처방 A)
    v2_still_shaping: bool = _V2_STILL_SHAPE   # 목표 접근 감속 셰이핑(정지 처방)
    v2_still_goal: bool = _V2_STILL_GOAL       # 목표 **안** 안정화 셰이핑(라운드 16)
    v2_adr_fixed_level: int = _V2_ADR_FIXED    # >=0 이면 그 레벨 고정(승급·강등 끔)
    v2_respawn: bool = _V2_RESPAWN              # 낙하/전도 시 에피소드 내 재소환
    v2_dr: bool = _V2_DR                        # DR(마찰) + ADR 사다리(질량·스폰·목표)
    v2_home_j147: bool = _V2_HOME_J147          # 리셋 홈을 J1·J4·J7 자세로 교체
    v2_dwell_end: bool = _V2_DWELL_END          # 목표 체류 N 스텝 → 에피소드 종료
    v2_zfloor: bool = _V2_ZFLOOR                # 경로 손끝-테이블 z 마진 벌점
    v2_approach_tilt: bool = _V2_APPR           # 파지 전 접근 자세(수평 이상) 제약
    v2_dirmul: bool = _V2_DIRMUL                # 방향 품질을 파지 후 계단에 곱한다
    v2_home_level: bool = _V2_HOME_LEVEL        # 접근축 수평 홈 (FRESH 전용)
    v2_home_low: bool = _V2_HOME_LOW            # 낮고 덜 기운 홈 B100 (FRESH 전용)
    v2_home_mid: bool = _V2_HOME_MID            # 중간 홈 C96 (FRESH 전용)

    def __post_init__(self):
        super().__post_init__()

        # ── 목표 상자를 **컵 좌표**로 평행이동 ──────────────────────
        # v1 은 goal 을 TCP 로 채점하면서 합격은 컵으로 판정했다. t79 best 프로브 실측
        # 리프트 후 `컵 − TCP` = 37.2 mm ⇒ TCP 가 목표에 완벽히 도달해도 컵은 37 mm
        # 남아 합격 예산 57 mm 의 65% 를 계통 오프셋이 먼저 먹었다.
        # 상자를 통째로 그만큼 옮기면 **컵이 새 상자에 놓일 때 TCP 는 옛 상자(= TCP 제약
        # IK 로 도달성이 검증된 자리)** 에 있다 — 검증을 버리지 않고 채점만 정렬한다.
        # ★R3 — 라운드 4 부터 다시 **스위치**다(위 `_V2_GOAL_MEASURED` 주석 참조).
        #   True  = v2 정책 실측 오프셋(z +30 mm). 컵이 목표에 있을 때 TCP 가 제약 IK 로
        #           도달성이 검증된 상자 안에 정확히 놓인다 — 물리적으로 옳은 쪽.
        #   False = 구 값(z +12 mm) = **run0-s43 과 같은 목표 상자**. 보상 효과를
        #           단일 변수로 재기 위한 대조군 조건.
        _off = P.GRASP_OFFSET_ROOT if self.v2_goal_measured else P.GRASP_OFFSET_ROOT_V1
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
                    "still_net": self.v2_still_net,
                    "hold_weight": P.HOLD_WEIGHT if self.v2_hold_premium else 0.0,
                    "upright_weight": (P.UPRIGHT_WEIGHT
                                       if self.v2_upright_shaping else 0.0),
                    "still_weight": (P.STILL_WEIGHT
                                     if self.v2_still_shaping else 0.0),
                    "still_goal_weight": (P.STILL_GOAL_WEIGHT
                                          if self.v2_still_goal else 0.0),
                    "dirmul_gain": P.DIRMUL_GAIN if self.v2_dirmul else 0.0},
        )
        # ⚠ `use_sr`(Hundt 진행도 역전 차단)은 기본 off. 원문은 이산 primitive 태스크라
        #   안전하지만 우리는 50 Hz 연속제어라 경계 근처에서 매 스텝 보상을 0 으로 만들
        #   위험이 있다. 켜려면 **단일 변수로** 켠다.

        self.rewards.action_l2 = RewTerm(func=R.action_l2, weight=P.ACTION_L2_WEIGHT)
        self.rewards.action_rate = RewTerm(func=R.action_rate_l2,
                                           weight=P.ACTION_RATE_WEIGHT)
        # ★커리큘럼을 걸지 않는다 — t73/t75 가 발동 시점(ep1500)에 정확히 꺾였고,
        #   `action_rate_l2` 가 재는 것의 대부분은 정책의 거칢이 아니라 σ 였다.
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
                       ("diag_stage3_ok", R.diag_stage3_ok)):
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
        # ── 라운드 20: 접근 자세 제약 (파지 전, 수평 이상) ─────────
        #   ★높이가 아니라 **자세**를 친다. z 마진이 +1.7 mm 로 끝난 것은 정책이
        #     기울기를 유지한 채 전체를 살짝 들었기 때문이다(라운드 19 실측).
        if self.v2_approach_tilt:
            # ★★라운드 20b — 순수 벌점(`approach_tilt_penalty`)은 기각됐다. 90° 이하가
            #   전부 0 이라 정책이 그리퍼를 위로 세워 벌점만 피하고 파지를 포기했다.
            #   방향에 **+보상**을 주되 `stage_reach` 를 곱해 "서서 안 잡기"를 막는다.
            self.rewards.approach_dir = RewTerm(
                func=R.approach_dir_bonus, weight=P.APPROACH_DIR_WEIGHT,
                params={"jaw_cfg": _jaw(),
                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                        "object_cfg": SceneEntityCfg("object")})
            self.rewards.diag_approach_deg = RewTerm(
                func=R.diag_approach_deg, weight=0.0, params={})
            self.rewards.diag_approach_down = RewTerm(
                func=R.diag_approach_down, weight=0.0,
                params={"jaw_cfg": _jaw(), "object_cfg": SceneEntityCfg("object")})

        # ── 라운드 22 Part 1: 곱셈형 방향의 진단 ──────────────────
        #   가중 0 항이라 보상에 안 들어가고 값만 기록된다(hdgp reward_manager 패치).
        if self.v2_dirmul:
            self.rewards.diag_dir_quality = RewTerm(
                func=R.diag_dir_quality, weight=0.0, params={})
            self.rewards.diag_approach_deg = RewTerm(
                func=R.diag_approach_deg, weight=0.0, params={})
            self.rewards.diag_approach_down = RewTerm(
                func=R.diag_approach_down, weight=0.0,
                params={"jaw_cfg": _jaw(), "object_cfg": SceneEntityCfg("object")})

        # ── 라운드 18: 경로 z 마진 벌점 ───────────────────────────
        #   ★파지와 **역방향**인 항이다. 하한(판 위 20 mm) 위에서는 정확히 0 이라
        #     파지 구간에 gradient 를 만들지 않는다. 컵 파지 대역이 10~85 mm 이므로
        #     하단 10 mm 만 포기하고 65 mm 가 남는다.
        if self.v2_zfloor:
            self.rewards.tip_floor = RewTerm(
                func=R.tip_floor_penalty, weight=P.TIP_FLOOR_WEIGHT,
                params={"tip_cfg": _jaw()})
            self.rewards.diag_tip_height = RewTerm(
                func=R.diag_tip_height, weight=0.0, params={"tip_cfg": _jaw()})
            self.rewards.diag_tip_violation = RewTerm(
                func=R.diag_tip_violation, weight=0.0, params={"tip_cfg": _jaw()})

        # ── 라운드 17: 리셋 홈 교체 (J1·J4·J7) ────────────────────
        #   ★액션 0 = 이 자세다(`use_default_offset`). 홈을 바꾸면 액션 상자가
        #     통째로 옮겨가므로 **기존 체크포인트와 호환되지 않는다** — FRESH 전용.
        if self.v2_home_j147:
            self.scene.robot.init_state.joint_pos.update(P.LEFT_ARM_HOME_J147)

        # ── 라운드 22 Part 2: 접근축 수평 홈 ──────────────────────
        #   ★홈 각도(103.9°)가 곧 "각도 유지 비용"이다. G(w=7.0) 는 이 14° 를 정책이
        #     스스로 세우느라 액션 6축 중 4축을 포화시키고도 파지점 48 mm 앞에서
        #     멈췄다. 홈에서 이미 수평이면 그 비용이 0 이 된다.
        #   ⚠ 홈 교체는 액션 상자를 통째로 옮긴다 — **기존 체크포인트와 비호환**.
        if self.v2_home_level:
            self.scene.robot.init_state.joint_pos.update(P.LEFT_ARM_HOME_LEVEL)

        # ── 라운드 22b: 낮고 덜 기운 홈 B100 ──────────────────────
        #   ★A94 는 각도만 세우고 TCP 를 파지점보다 80mm 위에 두는 바람에 정책이
        #     다시 기울었다. B100 은 각도·높이를 동시에 개선한다.
        #   ⚠ HOME_LEVEL 과 동시에 켜면 뒤엣것이 이긴다 — 둘 중 하나만 켠다.
        if self.v2_home_low:
            self.scene.robot.init_state.joint_pos.update(P.LEFT_ARM_HOME_LOW)

        # ── 라운드 22c: 중간 홈 C96 ───────────────────────────────
        #   ★B100 은 낮춰서 리프트를 잃었다(컵 +4.9mm, 램프는 +6.0mm). C96 은 구 홈과
        #     **같은 높이**를 유지하면서 각도만 14° 세운다.
        if self.v2_home_mid:
            self.scene.robot.init_state.joint_pos.update(P.LEFT_ARM_HOME_MID)

        # ── 라운드 17: 목표 체류 N 스텝이면 에피소드 종료 ──────────
        #   ⚠⚠ `time_out=True` (truncation) 여야 한다. 진짜 종료로 두면 부트스트랩이
        #      끊겨 성공이 곧 남은 보상 포기가 되고, 정책은 목표 밖을 맴돌며 stage 3
        #      보상을 계속 빠는 쪽을 배운다. 계약 테스트가 이 플래그를 고정한다.
        if self.v2_dwell_end:
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
            # ADR 사다리 — 스폰·목표·질량·obs bias 를 성공에 맞춰 넓힌다. `Curriculum/adr`.
            #   ★라운드 13 대조군 B: `HDGP_V2_ADR_FIXED=4` 면 **레벨을 만렙에 고정**하고
            #     승급·강등을 끈다(커리큘럼 없이 처음부터 전범위). 노브를 여기서 직접
            #     건드리지 않는 이유 — 스폰 상자는 부모 골격 소유라 계약이 금지한다.
            #     같은 `_apply` 경로를 쓰므로 A 와 B 의 만렙 정의가 구조적으로 동일하다.
            self.curriculum.adr = CurrTerm(func=C.ADRLadder,
                                           params={"fixed_level": _V2_ADR_FIXED})

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

        # ── 라운드 6: 제어 층 스위치 ──────────────────────────────
        if self.v2_rot_wide:
            self.actions.arm_action.palm_max_pose_angle = P.PALM_MAX_POSE_ANGLE_WIDE
        if self.v2_anchor_relatch:
            self.actions.arm_action.anchor_relatch_cup_z = P.ANCHOR_RELATCH_CUP_Z

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
