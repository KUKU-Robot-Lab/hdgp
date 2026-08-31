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

"""계단 합성 · 벌점 · 진단 항.

계단은 **하나의 보상 항**이다. 그래서 단계별 값이 TFEvents 에 안 찍히므로 weight 0
진단 항을 함께 등록한다 — 이 저장소의 `reward_manager.py` 패치가 weight 0 항의 원값을
`_episode_sums` 에 누적한다(fab_test74 에서 그 패치가 없는 호스트의 진단이 전부 정확히
0 으로 찍혀 하루를 오진한 이력이 있다).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

# 액션 진단은 v1 것을 그대로 쓴다 — 축 포화는 이 트랙의 상설 감시 지표다.
from ..grasp_sensor.grasp_left_rewards import (  # noqa: F401
    _cup_upright_cos,
    diag_action_axis_mu,
    diag_action_axis_sat,
)
from . import v2_preset as P
from . import v2_stages as S

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stage_index(r_grasp, r_lift, r_transport, ok3) -> torch.Tensor:
    """도달한 최상단 단계 (0=grasp · 1=lift · 2=transport · 3=settle).

    ★네 번째 인자만 의미가 바뀌었다(08.29): 예전엔 연속량 `r_settle`, 지금은
      네 조건 AND 의 결과 `ok3` ∈ {0, 1}. `> STAGE_THRESHOLD(0.1)` 로 재는 식은
      두 경우 모두 같은 뜻이라 **판정 배선은 한 글자도 안 바뀐다**.
    """
    idx = torch.zeros_like(r_grasp)
    idx = torch.where(r_lift > P.STAGE_THRESHOLD, torch.ones_like(idx), idx)
    idx = torch.where(r_transport > P.STAGE_THRESHOLD, torch.full_like(idx, 2.0), idx)
    idx = torch.where(ok3 > P.STAGE_THRESHOLD, torch.full_like(idx, 3.0), idx)
    return idx


def _stage_value(idx, v0, v1, v2, v3) -> torch.Tensor:
    """그 단계의 값 하나만 고른다 — 계단은 **더하지 않는다**.

    ★인자는 `all_stages` 의 `val` = 진행도 네 벌이다. 각각 [0,1] 이고 다음 문턱에서
      1 이 되므로, 여기서 하나만 골라도 `(idx + v)/N` 이 경계에서 연속이 된다.
    """
    v = v0
    v = torch.where(idx >= 1.0, v1, v)
    v = torch.where(idx >= 2.0, v2, v)
    v = torch.where(idx >= 3.0, v3, v)
    return v


class Staircase(ManagerTermBase):
    """Lee et al. 계단식 보상 — `r = (단계 인덱스 + 그 단계 값) / N`.

    ★★왜 계단인가. v1 은 lift 15 + goal 16 + fine 5 + jaws 3 + closure 5 + pose 5 를
      **동시에** 받는 가산형이었다. 결과:
        · "높이 들고 가만히"가 충분히 이득이어서 이송을 안 배웠다
        · 36 점이 게이트 하나를 공유해 **이동이 도박**이 됐다
          (t79 프로브 실측: 이동 중 게이트 0.944 vs 정지 중 1.000)
      계단에서는 한 번에 한 단계 값만 받는다. stage 1(lift) 천장이 (1+1)/4 = 0.5 로
      고정되고 stage 2(transport)는 0.5~0.75 라 **전진이 항상 유리**하며, 후퇴 손실도
      한 계단(0.25)으로 **유계**다.

    ★★08.29 라운드 3 — 계단을 **연속·단조**로 재구성했다. 값 `v_k` 는 "다음 문턱까지의
      진행도"라 문턱에서 정확히 1 이고, 다음 단계 값은 0 에서 시작한다 ⇒ 경계 점프가
      1→2 에서 0.003 · 2→3 에서 정확히 0 이 된다. `v_1` 에 목표 거리가 들어가
      **stage 1 고원(∂r/∂dist = 0)이 사라진다** — 그 고원이 시드 의존의 기계적
      원인이었다(붕괴 후 `cupd` 146·160·170 mm = stage 2 문턱 150~161 mm 바깥).

    ★`still_net=True` 면 속도 입력이 **순변위**, False 면 **순간속도**다. 그 하나만
      바뀌고 식은 동일하다 — 라운드 3 의 단일 변수(arm A ↔ arm B).

    ⚠ `1_SR`(Hundt 의 진행도 역전 차단)은 **기본 off** 다. 우리는 50 Hz 연속제어라
      경계 근처에서 매 스텝 보상을 0 으로 만들 위험이 있다. 별도 단일 변수로 켠다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_idx = torch.zeros(env.num_envs, device=env.device)
        # ★라운드 7 — 지속 정착 카운터. success_ok 가 깨지면 0 으로 리셋(단조 아님).
        self._hold = torch.zeros(env.num_envs, device=env.device)
        self._tracker = S._NetSpeedTracker(env.num_envs, P.NET_SPEED_WINDOW, env.device)
        # ★라운드 22 Part 1 — 접근 구간 방향 품질의 누적(합·개수). stage 0 에서만
        #   전진하므로 파지 후에는 얼어붙는다 = 별도 래치가 필요 없다.
        self._dir_sum = torch.zeros(env.num_envs, device=env.device)
        self._dir_cnt = torch.zeros(env.num_envs, device=env.device)
        # 진단 항이 같은 스텝에 다시 계산하지 않도록 캐시한다(트래커 중복 갱신 방지).
        self._cache_step = -1
        self._cache: tuple | None = None

    def reset(self, env_ids=None):
        # 에피소드 경계에서 0 으로. 안 하면 리셋 직후가 가짜 후퇴로 보이고,
        # 순변위 버퍼는 리셋 텔레포트를 거대한 변위로 읽는다.
        if env_ids is None:
            self._prev_idx[:] = 0.0
            self._hold[:] = 0.0
            self._dir_sum[:] = 0.0
            self._dir_cnt[:] = 0.0
        else:
            self._prev_idx[env_ids] = 0.0
            self._hold[env_ids] = 0.0
            self._dir_sum[env_ids] = 0.0
            self._dir_cnt[env_ids] = 0.0
        self._tracker.reset(env_ids)

    def net_speed(self, env) -> torch.Tensor:
        """컵 순변위 속도 (m/s). 스텝당 한 번만 실제 갱신된다."""
        obj = env.scene["object"]
        return self._tracker.get(env, obj.data.root_pos_w)

    def stages(self, env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg,
               object_cfg, still_net: bool) -> tuple[tuple, tuple]:
        """`(pos, val)`. 같은 스텝 안에서는 캐시를 돌려준다 — 진단 항이 여러 번
        불러도 트래커가 두 번 전진하지 않도록."""
        step = int(getattr(env, "common_step_counter", -1))
        if step == self._cache_step and self._cache is not None and step >= 0:
            return self._cache
        ns = self.net_speed(env) if still_net else None
        out = S.all_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg,
                           object_cfg, net_speed=ns)
        self._cache_step, self._cache = step, out
        return out

    def __call__(self, env, command_name: str, robot_cfg: SceneEntityCfg,
                 jaw_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg,
                 object_cfg: SceneEntityCfg, use_sr: bool = False,
                 still_net: bool = False,
                 hold_weight: float = 0.0,
                 upright_weight: float = 0.0,
                 still_weight: float = 0.0,
                 still_goal_weight: float = 0.0,
                 dirmul_gain: float = 0.0) -> torch.Tensor:
        pos, val = self.stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg,
                               object_cfg, still_net)
        idx = _stage_index(*pos)          # ★판정 = 위치·파지·AND 조건
        v = _stage_value(idx, *val)       # ★값 = 다음 문턱까지의 진행도
        r = (idx + v) / P.N_STAGES
        # ── ★라운드 7: 지속 정착 프리미엄 ────────────────────────────
        #   흔들기(0.814/step)와 정착(1.0/step)의 기대값이 낙하 종료 앞에서 역전되는
        #   구조를 깬다 — success_ok 를 **연속 유지**해야만 누진으로 벌 수 있고,
        #   흔들기는 순간속도 조건에 걸려 카운터가 계속 리셋된다.
        if hold_weight > 0.0:
            succ = S.settle_success(env, command_name, robot_cfg, jaw_cfg, object_cfg)
            self._hold = torch.where(succ > 0.5, self._hold + 1.0,
                                     torch.zeros_like(self._hold))
            r = r + hold_weight * (self._hold / P.HOLD_RAMP_STEPS).clamp(max=1.0)
        # ── ★라운드 11: 이송 구간 직립 셰이핑 (처방 A) ────────────────
        #   기울기는 파지 순간에 생기는데 직립 보상은 목표 반경 안에만 있었다
        #   (~200 스텝 시차). 이송 중에 gradient 를 줘 원인 시점에서 고치게 한다.
        #   ★곱이 아니라 **가산**이다 — 라운드 3 이 stage 2 에 곱해 2.7 배 약화로
        #     실패했다. 가중치 < 계단 한 칸(0.25)이라 단계를 건너뛸 수 없고,
        #     stage 2·3 에 **동일하게** 붙어 2→3 무점프 성질도 그대로다.
        if upright_weight > 0.0:
            gate = (idx >= P.UPRIGHT_MIN_STAGE).to(r.dtype)
            r = r + upright_weight * S.upright_shaped(env, object_cfg) * gate
        # ── ★라운드 14: 감속 셰이핑 (정지 처방) ────────────────────────
        #   처방 A 와 **같은 구조**다 — 감속도 도착 전에 시작해야 하는 행동인데
        #   `p_still` 이 stage 3 안에만 있어 신용 할당에 시차가 있었다.
        #   `p_near` 가 150 mm 밖에서 0 이라 멀리서 멈춰 점수를 벌 수 없다.
        if still_weight > 0.0:
            gate = (idx >= P.STILL_MIN_STAGE).to(r.dtype)
            r = r + still_weight * S.still_shaped(env, command_name, robot_cfg,
                                                  object_cfg) * gate
        # ── ★라운드 16: 목표 안 안정화 셰이핑 ─────────────────────────
        if still_goal_weight > 0.0:
            r = r + still_goal_weight * S.still_at_goal(env, command_name,
                                                        robot_cfg, object_cfg)
        # ── ★라운드 22 Part 1: 곱셈형 방향 ──────────────────────────
        #   접근 구간(stage 0)의 **평균** 방향 품질을 파지 후 계단 전체에 곱한다.
        #   · 배수는 1.0~1.5 — **감액이 아니라 가산**이라 0→1 경계에 절벽이 없다.
        #   · 파지해야만 받는다 ⇒ 머무는 유인이 생기지 않는다(stage 0 천장 0.25).
        #   · 순간값이 아닌 평균이라 "파지 직전에만 잠깐 세우기"로 못 딴다.
        if dirmul_gain > 0.0:
            pre = idx < 1.0
            self._dir_sum = torch.where(pre, self._dir_sum + _approach_dirq(env),
                                        self._dir_sum)
            self._dir_cnt = torch.where(pre, self._dir_cnt + 1.0, self._dir_cnt)
            r = torch.where(pre, r, r * (1.0 + dirmul_gain * self.dir_quality()))
        if use_sr:
            r = torch.where(idx < self._prev_idx, torch.zeros_like(r), r)
        self._prev_idx[:] = idx
        return r

    def dir_quality(self) -> torch.Tensor:
        """접근 구간 평균 방향 품질 (0~1). 파지 전에는 진행 중인 평균이다."""
        return self._dir_sum / self._dir_cnt.clamp(min=1.0)

    def hold_norm(self) -> torch.Tensor:
        """진단용 — 정규화된 hold 카운터 (0~1)."""
        return (self._hold / P.HOLD_RAMP_STEPS).clamp(max=1.0)


# ---------------------------------------------------------------------------
# 벌점
# ---------------------------------------------------------------------------
def action_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """`mean(a²)`. ★합(sum)은 상시 포화해 gradient 가 0 인 죽은 항이 된다(자매 트랙 실측).
    평균이면 액션 차원이 바뀌어도 스케일이 불변이다."""
    return torch.mean(torch.square(env.action_manager.action), dim=1)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """`mean((a − a_prev)²)`.

    ⚠ v1 실측: 이 항은 σ 에 오염돼 **정책 평활도를 못 고친다** — σ≈1·6 차원이면 독립
      샘플 차분 기댓값이 2σ²×6 = 12 라 재는 것의 대부분이 탐색 노이즈다. 그리고
      t73/t75 가 커리큘럼 발동 시점(ep1500)에 정확히 꺾였다. ⇒ 작게 두고
      **커리큘럼을 걸지 않는다.**
    """
    a = env.action_manager.action
    return torch.mean(torch.square(a - env.action_manager.prev_action), dim=1)


# ---------------------------------------------------------------------------
# 진단 (weight 0) — 계단이 항 하나라 이게 없으면 안이 안 보인다
# ---------------------------------------------------------------------------
def _staircase(env) -> "Staircase | None":
    """등록된 `Staircase` term 인스턴스를 찾는다.

    ★진단이 `S.all_stages` 를 직접 부르면 **순변위 트래커가 스텝마다 여러 번 전진**해
      창 길이가 사실상 1/6 이 된다(진단 5 개 + 본항). 반드시 같은 인스턴스의 캐시를 쓴다.
    """
    for term in env.reward_manager._term_cfgs:
        f = getattr(term, "func", None)
        if isinstance(f, Staircase):
            return f
    return None


def _diag_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    sc = _staircase(env)
    if sc is None:      # 계약상 있어야 하지만, 없으면 run 0 식으로 폴백
        return S.all_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)
    sn = bool(sc.cfg.params.get("still_net", False))
    return sc.stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg, sn)


def diag_stage_index(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg,
                     object_cfg) -> torch.Tensor:
    """평균 단계 인덱스 (0~3). ★위치 기준(`pos`) — 보상 판정과 같은 정의다."""
    pos, _ = _diag_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)
    return _stage_index(*pos)


def _pos_pick(env, which, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """판정용 양(`pos`). ★**과거 6 판과 같은 정의**라 로그를 그대로 이어 비교할 수 있다
    (`r_grasp` · `r_lift` · `r_transport` 는 run 0 이래 식이 안 바뀌었다)."""
    pos, _ = _diag_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)
    return pos[which]


def diag_r_grasp(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """`r_grasp` — reach × close 보너스. 정의 불변."""
    return _pos_pick(env, 0, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)


def diag_r_lift(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """`r_lift` = `r_close × move_up`. 정의 불변 — 6 판 비교의 기준선이다."""
    return _pos_pick(env, 1, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)


def diag_r_transport(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """`r_transport` = `r_lift × D(dist)`. 정의 불변.
    ★`transp/lift` 비율이 "리프트가 이송으로 전환되는가"의 직접 지표다."""
    return _pos_pick(env, 2, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)


def diag_stage3_ok(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """stage 3 승급 조건(**도달 + 파지**) 만족 비율.

    ★구 `diag_r_settle`(네 인자 곱, 6 판 최대 0.0106)을 대체한다.
    ★★08.29 라운드 4 로 뜻이 또 바뀌었다 — 속도·직립이 값 쪽으로 옮겨가면서 이 지표는
      "목표 반경 안에서 쥐고 있는 비율" 이 됐다. **합격 판정은 `diag_success` 다.**
      라운드 3 값(속도·직립 포함)과 직접 비교하지 말 것.
    """
    return _pos_pick(env, 3, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)


def diag_success(env, command_name: str,
                 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 jaw_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """★★**합격 판정** — 도달 + 정지 + 직립 + 파지 유지의 AND.

    승급 문턱에서 뺀 두 조건이 여기 남아 있어 **과제 정의는 그대로**다. 보상 지형만
    부드럽게 하고 합격선은 안 낮춘다 — 지형과 합격선을 같이 낮추면 "쉬워져서 올랐다"를
    "고쳐서 올랐다"로 오독한다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    dist = S.cup_goal_distance(env, command_name, robot_cfg, object_cfg)
    sc = _staircase(env)
    if sc is not None and bool(sc.cfg.params.get("still_net", False)):
        speed = sc.net_speed(env)
    else:
        speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    return S.success_ok(dist, speed, _cup_upright_cos(env, object_cfg),
                        S.stage_close(env, jaw_cfg, object_cfg))


def diag_v_stage(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg):
    """실제로 보상에 들어간 진행도 `v_idx` (0~1).

    `diag_stage`(정수 인덱스)와 나란히 보면 `r = (idx + v)/4` 를 로그만으로 복원할 수
    있다 — 어느 단계에서 얼마나 진행했는지가 드러난다."""
    pos, val = _diag_stages(env, command_name, robot_cfg, jaw_cfg, ee_frame_cfg, object_cfg)
    return _stage_value(_stage_index(*pos), *val)


def _progress(env, which, command_name, robot_cfg, object_cfg):
    """`(P_dist, P_still, P_upright)` 중 하나. 보상 경로와 **같은 인스턴스**의 속도를 쓴다."""
    sc = _staircase(env)
    ns = None
    if sc is not None and bool(sc.cfg.params.get("still_net", False)):
        ns = sc.net_speed(env)
    return S.progress_terms(env, command_name, robot_cfg, object_cfg, net_speed=ns)[which]


def diag_p_dist(env, command_name: str,
                robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """거리 진행도. ★셋을 따로 찍어야 "이송이 왜 안 되는가"가 로그만으로 갈린다 —
    거리를 못 좁히는 것인지(이 항), 못 멈추는 것인지, 컵이 기우는 것인지."""
    return _progress(env, 0, command_name, robot_cfg, object_cfg)


def diag_p_still(env, command_name: str,
                 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """정지 진행도. arm A 는 순간속도 · arm B 는 순변위 — 라운드 3 의 단일 변수."""
    return _progress(env, 1, command_name, robot_cfg, object_cfg)


def diag_p_upright(env, command_name: str,
                   robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """직립 진행도. ★라운드 4 부터 `v_3` 의 인자다(`v_2` 에서 뺐다)."""
    return _progress(env, 2, command_name, robot_cfg, object_cfg)


def diag_p_center(env, command_name: str,
                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """목표 **반경 안**에서의 중심 접근도. `v_3` 의 첫 인자이자 2→3 연속성의 근거 —
    문턱(dist = SETTLE_RADIUS)에서 정확히 0 이다."""
    return _progress(env, 3, command_name, robot_cfg, object_cfg)


def diag_hold(env):
    """★라운드 7 — 정규화 hold 카운터 평균. 이게 오르면 '굳히기'를 배우는 중이다.
    흔들기는 success_ok 리셋 때문에 이 값을 못 올린다 — 해킹과 진짜 정착을 가른다."""
    sc = _staircase(env)
    if sc is None:
        return torch.zeros(env.num_envs, device=env.device)
    return sc.hold_norm()


def diag_cup_net_speed(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """★컵 **순변위** 속도 (m/s). 합격 기준이 이 값이다(< 0.05).
    순간속도는 왕복의 반환점에서 0 이 되어 진동을 정지로 오독한다."""
    sc = _staircase(env)
    if sc is None:
        obj: RigidObject = env.scene[object_cfg.name]
        return torch.norm(obj.data.root_lin_vel_w, dim=1)
    return sc.net_speed(env)


def diag_cup_goal_dist(env, command_name: str,
                       robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                       object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """**컵** ↔ 목표 거리 (m). ★최종 합격 판정이 이 값이다(기준 < 0.057).

    v2 는 보상도 이 값으로 재므로 **보상 최적점과 합격 기준이 같은 곳**을 가리킨다 —
    v1 은 보상이 TCP 라 37 mm 어긋나 있었다.
    """
    return S.cup_goal_distance(env, command_name, robot_cfg, object_cfg)


def diag_at_goal(env, command_name: str,
                 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """반경 `SETTLE_RADIUS` 안에 든 비율. **조건부 추종의 직접 지표**이자 라운드 3 의
    사전 등록 판정 지표다(baseline run0-s43 정점 0.285).

    고정점 전략의 이론 상한이 37.4% 이므로 이 값이 그걸 넘으면 추종을 배운 것이다.
    ★`diag_stage3_ok` 는 여기에 정지·직립·파지 조건을 AND 로 더한 것이라 항상
      이 값 이하다 — 둘의 차이가 "도달은 하는데 못 멈춘다"의 크기다."""
    return (S.cup_goal_distance(env, command_name, robot_cfg, object_cfg)
            < P.SETTLE_RADIUS).float()


def diag_cup_speed(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """컵 선속도 크기 (m/s). 합격 보조 기준 < 0.05."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.norm(obj.data.root_lin_vel_w, dim=1)


def diag_cup_upright(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """컵 직립 코사인 (1 = 세워짐)."""
    return _cup_upright_cos(env, object_cfg)


def _tip_height(env: "ManagerBasedRLEnv", tip_cfg: SceneEntityCfg) -> torch.Tensor:
    """손끝·TCP 중 **가장 낮은** 높이 (테이블 상면 기준, m).

    ★한 점만 보면 안 된다 — 긁는 것은 가장 낮은 부위다. 두 턱 링크와 TCP 를 모두
      넣고 최소를 취한다.
    """
    robot = env.scene[tip_cfg.name]
    z = robot.data.body_pos_w[:, tip_cfg.body_ids, 2]
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0, 2]
    origin_z = env.scene.env_origins[:, 2]
    low = torch.minimum(z.min(dim=1).values, ee) - origin_z
    return low - P.TABLE_SURFACE_Z


def tip_floor_penalty(env: "ManagerBasedRLEnv", tip_cfg: SceneEntityCfg) -> torch.Tensor:
    """판 위 `TIP_FLOOR_MARGIN` 아래로 내려간 정도 (0~1). **위에서는 정확히 0.**

    선형 힌지다 — 마진에서 0, 상면(높이 0)에서 1, 상면 아래로는 1 에서 포화한다.
    포화시키는 이유: 이미 관통한 상태에서 더 깊이 들어간 것을 무한히 벌하면
    리셋 직후 한 프레임의 관통이 에피소드 전체를 지배한다.
    """
    h = _tip_height(env, tip_cfg)
    return ((P.TIP_FLOOR_MARGIN - h) / P.TIP_FLOOR_MARGIN).clamp(0.0, 1.0)


def diag_tip_height(env: "ManagerBasedRLEnv", tip_cfg: SceneEntityCfg) -> torch.Tensor:
    """진단 — 최저 손끝 높이(판 위, m). 벌점이 실제로 무엇을 보고 있는지 남긴다."""
    return _tip_height(env, tip_cfg)


def diag_tip_violation(env: "ManagerBasedRLEnv", tip_cfg: SceneEntityCfg) -> torch.Tensor:
    """진단 — 마진을 어긴 스텝의 비율(0/1). 벌점 크기와 달리 **빈도**를 본다."""
    return (_tip_height(env, tip_cfg) < P.TIP_FLOOR_MARGIN).float()


def _approach_az(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """그리퍼 +z 축의 world z 성분 (= 회전행렬 R[2,2]).

    +1 = 똑바로 위 · 0 = 수평(90°) · −1 = 똑바로 아래. 사잇각은 `acos` 로 얻는다.
    """
    robot = env.scene["robot"]
    bi = robot.body_names.index(P.GRIPPER_BASE_BODY)
    q = robot.data.body_quat_w[:, bi, :]
    return (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0)


def approach_tilt_penalty(env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg,
                          object_cfg: SceneEntityCfg) -> torch.Tensor:
    """**파지 전** 그리퍼가 아래로 기운 정도 (0~1). 90° 이하에서 정확히 0.

    파지 후에는 0 이다 — 무는 동작(j7 을 드는 것)과 리프트·이송은 이 항의 대상이
    아니다. 게이트는 파지 판정과 같은 `stage_close` 를 쓴다.
    """
    held = S.stage_close(env, jaw_cfg, object_cfg) > 0.5
    return (-_approach_az(env)).clamp(0.0, 1.0) * (~held).to(torch.float32)


def _approach_dirq(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """방향 품질 0~1 — **90° 이하는 만점 평지, 초과는 완만한 감쇠**.

    ★★라운드 21(I2) 이 이 모양 때문에 죽었다. 지시받은 "90° 초과면 0" 을 그대로
      `where(az < 0, 0, ...)` 로 구현했더니 `az<0` 이 **값도 기울기도 0 인 평지**가
      됐고, 정책이 110° 에 자리잡은 뒤로 되돌아올 신호가 없어 1000 epoch 동안
      각도가 미동도 안 했다(수령률 상한의 0.05%).
      의도(90° 이하 만점 · 아래로 기울면 벌)는 그대로 두고 하향만 감쇠로 바꾼다.
        103.9°(홈) 0.527 · 110° 0.273 · 117.5° 0.093
    """
    az = _approach_az(env)
    below = torch.exp(-((az / P.APPROACH_DIR_SIGMA_DN) ** 2))
    return torch.where(az >= 0.0, torch.ones_like(az), below)


def diag_dir_quality(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """진단 — 접근 구간 평균 방향 품질(계단이 누적한 값)."""
    sc = _staircase(env)
    return sc.dir_quality() if sc is not None else torch.zeros(env.num_envs,
                                                               device=env.device)


def diag_approach_deg(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """진단 — 접근 각도(도). 90° = 수평, 초과면 아래로 기욺."""
    return torch.rad2deg(torch.acos(_approach_az(env)))


def diag_approach_down(env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg,
                       object_cfg: SceneEntityCfg) -> torch.Tensor:
    """진단 — 파지 전에 90° 를 넘긴 스텝의 비율(0/1)."""
    held = S.stage_close(env, jaw_cfg, object_cfg) > 0.5
    return ((_approach_az(env) < 0.0) & (~held)).to(torch.float32)


def approach_dir_bonus(env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg,
                       ee_frame_cfg: SceneEntityCfg,
                       object_cfg: SceneEntityCfg) -> torch.Tensor:
    """**접근 방향 보너스** — 수평(90°)에서 최대, 아래로 기울면 0. 파지 전에만.

    `r_grasp`(접근 진행도)를 곱한다. 이게 없으면 "수평으로 서서 안 잡기"가 해킹면이
    된다 — 곱하면 컵 근처까지 가야만 보너스가 생긴다.

    ★순수 벌점판(`approach_tilt_penalty`)이 실패한 이유는 90° 이하가 전부 0 이라
      **위로 세우는 것이 최적**이었기 때문이다. 여기서는 수평에서만 최대라 그 경로가
      막힌다(위로 세우면 `APPROACH_DIR_BASE` = 1% 만 받는다).
    """
    az = _approach_az(env)
    peak = torch.exp(-((az / P.APPROACH_DIR_SIGMA) ** 2))
    dirq = torch.where(az < 0.0, torch.zeros_like(az),
                       P.APPROACH_DIR_BASE + (1.0 - P.APPROACH_DIR_BASE) * peak)
    held = S.stage_close(env, jaw_cfg, object_cfg) > 0.5
    # ★`stage_reach` = 순수 접근 근접도(TCP → 파지점). `stage_grasp` 는 닫기까지
    #   포함하므로 여기엔 안 맞다 — 우리가 곱하고 싶은 것은 "얼마나 다가왔는가" 뿐이다.
    r_reach = S.stage_reach(env, ee_frame_cfg, object_cfg)
    return dirq * r_reach * (~held).to(dirq.dtype)


class ApproachDirPBRS(ManagerTermBase):
    """접근 방향 shaping — **차분 지급(PBRS)**.

    ★레벨 지급(`approach_dir_bonus`)은 실패했다. 각도는 목표대로 117.5° → **90.1°**
      로 내려갔지만, 정책이 그 자세로 **250 스텝 중 239 스텝을 떠 있기만** 하고 컵을
      안 잡았다(⑤ 0.0%). 원인은 크기가 아니라 **지급 방식과 종료 조건의 상호작용**이다 —
      과제를 완수하면 에피소드가 54 스텝에 끝나는데(목표 체류 10 스텝 종료), 떠 있으면
      250 스텝 내내 받는다. 버티기와 완수의 가치가 같아졌다.

    차분으로 주면 그 경로가 사라진다: Φ 가 변하지 않으면 **0 원**이고, 자세를 수평으로
    **개선할 때만** 받는다. 저장소 원칙 "절단은 truncated, 지급은 차분(PBRS)으로" 그대로다.

        Φ = dir(a_z) · r_reach,   r = Φ_t − Φ_{t−1}
        dir(a_z) = 0                                  a_z < 0  (90° 초과 = 아래로 기욺)
                 = BASE + (1−BASE)·exp(−(a_z/σ)²)     a_z ≥ 0  (수평에서 1.0)

    ⚠ 파지 후에는 Φ 를 **직전 값으로 동결**한다. 게이트로 0 을 만들면 파지하는 순간
      −Φ 의 절벽이 생겨 **파지를 벌하게** 된다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        # 에피소드 경계에서 0 으로. 안 하면 리셋 텔레포트가 거대한 차분으로 읽힌다.
        if env_ids is None:
            self._prev[:] = 0.0
        else:
            self._prev[env_ids] = 0.0

    def __call__(self, env, jaw_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg,
                 object_cfg: SceneEntityCfg) -> torch.Tensor:
        az = _approach_az(env)
        peak = torch.exp(-((az / P.APPROACH_DIR_SIGMA) ** 2))
        dirq = torch.where(az < 0.0, torch.zeros_like(az),
                           P.APPROACH_DIR_BASE + (1.0 - P.APPROACH_DIR_BASE) * peak)
        phi = dirq * S.stage_reach(env, ee_frame_cfg, object_cfg)
        held = S.stage_close(env, jaw_cfg, object_cfg) > 0.5
        # 파지 후에는 Φ 를 동결 ⇒ 차분 0 (절벽 없음)
        phi = torch.where(held, self._prev, phi)
        out = phi - self._prev
        self._prev = phi
        return out
