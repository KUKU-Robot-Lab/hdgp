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

"""계단식 보상의 **단계 함수** — 전부 `[0, 1]` 을 돌려준다.

Lee et al., *Beyond Pick-and-Place*(CoRL 2021) 의 stage 구조를 컵 파지·이송에 맞춘 것이다.
계단 합성과 벌점은 `v2_rewards.py` 에 있다.

★★Lee 와 갈라지는 지점 둘.
  ① 원문의 stage 4·5 는 물체를 **놓고 물러나는** 것이다. 우리 최종 목적은
     `both/pour_sensor` 에서 왼팔이 컵을 **쥔 채 유지**하는 것이라 반대다 ⇒ 최상단을
     "목표에 정지 + upright + 파지 유지"로 바꿨다(`stage3_ok`). 이는 v1 에서
     `SETTLE_REWARD_WEIGHT = 0.0` 으로 꺼져 있던 `settled_at_goal` 의 부활이자 참고
     문서 "강한 권고 4"의 우리 버전이다.
  ② **08.29 라운드 3** — 원문은 각 단계가 이산 primitive 라 "그 단계를 완성"하는 것이
     곧 다음 단계 진입이다. 우리는 50 Hz 연속제어라 **한 단계 안에 무한히 머물 수
     있다**. 그래서 계단 값을 그 단계의 양(`r_lift` 등)이 아니라 **다음 문턱까지의
     진행도**로 재정의했다(`all_stages` 의 `val`). 그러지 않으면 stage 1 안에 목표
     거리가 없어 ∂r/∂dist = 0 인 고원이 생기고, 그 고원을 언제 벗어나느냐가 시드
     운에 달린다 — 실측 6 판에서 붕괴 후 `cupd` 가 146·160·170 mm 로 stage 2 문턱
     (150~161 mm) 바깥에 정확히 주차했다.

★기하는 v1 에서 그대로 가져온다. `grasp_ok` / `_jaw_geometry` / `_cup_upright_cos` 는
  2 지 그리퍼 실측으로 굳은 판단이라 다시 쓰면 같은 함정을 다시 밟는다:
    · `enclose` 는 판별력이 없다 — fab_test11 이 컵 축에서 옆으로 85.5 mm 떨어진 채
      enclose 0.824 를 받으면서 컵을 0.2 mm 도 못 들었다(성공 정책 test17 은 0.804).
    · 그래서 파지 판정은 lateral 을 직접 보는 `grasp_ok` 여야 한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

# v1 기하 헬퍼 재사용 — 실측으로 굳은 판단이라 다시 쓰지 않는다.
from ..grasp_sensor.grasp_left_rewards import (
    _cup_upright_cos,
    _jaw_geometry,
    grasp_ok,
)
from . import v2_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Lee 의 거리 shaping
# ---------------------------------------------------------------------------
def d_shape(dist: torch.Tensor, s: float, tau: float = 0.0) -> torch.Tensor:
    """`D(a,b,s,τ) = 1 − tanh²(‖a−b‖·atanh(√0.95)/s)`, `‖a−b‖<τ` 이면 1.

    `s` 는 **보상이 0.05 까지 떨어지는 거리**다 — 이 성질이 계약 테스트로 고정돼 있다.
    v1 의 `1 − tanh(d/std)` 와 달리 스케일에 물리적 의미가 있어, `std` 를 감으로 고르다
    두 번 실패한 이력(coarse t63 기각 · fine t64 역행)을 반복하지 않는다.
    """
    q = torch.tanh(dist * (P.D_SHAPE_K / s))
    out = 1.0 - q * q
    if tau > 0.0:
        out = torch.where(dist < tau, torch.ones_like(out), out)
    return out


# ---------------------------------------------------------------------------
# ★★R1 — 순변위 속도 (진동을 걸러내는 정지 판정)
# ---------------------------------------------------------------------------
class _NetSpeedTracker:
    """컵의 **순변위 속도** `‖p_t − p_{t−K}‖ / (K·dt)`.

    ★왜 순간속도를 쓰면 안 되는가. 왕복 운동의 **반환점에서 순간속도가 0** 이 된다.
      그래서 `still = D(‖v‖)` 로 정지를 요구하면, 진짜로 멈추는 것(정밀 제어라 어렵다)
      보다 **흔들어서 반환점을 목표 안에 두는 것**이 쉽고 보상이 같아진다.
      순변위는 왕복을 상쇄하므로 그 트릭이 통하지 않는다.
      저장소 이력도 같은 결론이다: "순간속도 0.07 m/s = 서브밀리미터 솔버 버즈
      (순변위 2.3 mm/s), 게인·damping 전부에 불변 → 합격기준을 순변위로".

    ⚠ 스텝당 **한 번만** 갱신해야 한다. 진단 항들이 `all_stages` 를 다시 부르므로
      `common_step_counter` 로 중복 갱신을 막고 캐시를 돌려준다.
    ⚠ 에피소드 경계에서 버퍼를 비운다. 안 그러면 리셋 텔레포트가 거대한 순변위로 보인다.
    """

    def __init__(self, num_envs: int, window: int, device) -> None:
        self._w = int(window)
        self._buf = torch.zeros(self._w, num_envs, 3, device=device)
        self._n = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ptr = 0
        self._last_step = -1
        self._value = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._n[:] = 0
        else:
            self._n[env_ids] = 0

    def get(self, env: "ManagerBasedRLEnv", pos: torch.Tensor) -> torch.Tensor:
        step = int(getattr(env, "common_step_counter", -1))
        if step == self._last_step and step >= 0:
            return self._value
        self._last_step = step
        span = torch.minimum(self._n, torch.full_like(self._n, self._w))
        # ★★버퍼가 아직 안 찬 env(리셋 직후)는 `buf[_ptr]` 이 **직전 에피소드의 데이터**다.
        #   `_ptr` 은 "가장 오래된 슬롯"이지 "span 스텝 전"이 아니기 때문이다. 그대로 쓰면
        #   리셋 텔레포트가 거대한 변위로 잡힌다 — 실측 스모크에서 순변위 2.10 m/s 가
        #   순간속도 0.025 m/s 의 **84 배**로 나왔다(순변위 ≤ 경로길이라 불가능한 값).
        #   ⇒ env 마다 `(_ptr − span) mod w` 슬롯을 본다. span=w 면 가장 오래된 것,
        #     span=1 이면 바로 직전 것 — 둘 다 정확히 span 스텝 전이다.
        idx = (self._ptr - span) % self._w
        oldest = self._buf[idx, torch.arange(self._buf.shape[1], device=pos.device)]
        dt = float(env.step_dt)
        disp = torch.norm(pos - oldest, dim=1)
        denom = span.clamp(min=1).float() * dt
        self._value = torch.where(span > 0, disp / denom, torch.zeros_like(disp))
        self._buf[self._ptr] = pos.detach()
        self._ptr = (self._ptr + 1) % self._w
        # ⚠ 무한 증가 방지 — span 계산에만 쓰므로 w 에서 멈춰도 무손실이다.
        self._n = torch.minimum(self._n + 1, torch.full_like(self._n, self._w))
        return self._value


def smoothstep(x: torch.Tensor, far: float, near: float) -> torch.Tensor:
    """`far` 에서 0 · `near` 에서 1 · 사이는 `3t²−2t³`. 경계에서 기울기가 0 이라
    **값도 미분도 연속**이다.

    `far > near` 면 감소형(거리·속도), `far < near` 면 증가형(직립 코사인) — 부호를
    식에서 자동으로 흡수하므로 세 진행도가 **같은 함수 하나**를 쓴다.

    ★자매 트랙(`agnostic/grasp_sensor`)의 `succ_soft = s_h·s_c·s_o·s_t·s_d·s_v` 와
      같은 패턴이다. 그쪽은 이송에 성공했고, 우리가 Lee 계단으로 백지 재설계하면서
      버린 것이 바로 이 "조건마다 부드러운 진행도, 곱으로 합성" 구조였다.

    ⚠ `d_shape` 와 역할이 다르다. `d_shape` 는 **꼬리가 무한히 이어지는** 거리 shaping
      (문턱 판정용 `r_transport` 에 쓴다)이고, 이쪽은 **밴드 밖에서 정확히 0/1** 이라
      "조건을 만족했다"를 표현할 수 있다. 계단 값이 다음 문턱에서 정확히 1 이 되려면
      후자가 필요하다.
    """
    t = ((x - far) / (near - far)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 씬 조회 헬퍼
# ---------------------------------------------------------------------------
def goal_pos_w(env: "ManagerBasedRLEnv", command_name: str,
               robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """목표 위치를 world 로. 명령은 **로봇 베이스 기준**이다(레퍼런스와 동일)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    return des_pos_w


def cup_goal_distance(env: "ManagerBasedRLEnv", command_name: str,
                      robot_cfg: SceneEntityCfg,
                      object_cfg: SceneEntityCfg) -> torch.Tensor:
    """**컵 원점** ↔ 목표 거리 (m).

    ★★v2 의 핵심 변경: v1 은 이것을 **TCP** 로 쟀는데(fab_test73) 합격 판정은 컵이었다.
      t79 best 프로브 실측 리프트 후 `컵 − TCP` = 37.2 mm ⇒ TCP 가 목표에 완벽히
      도달해도 컵은 37 mm 남아, 합격 예산 57 mm 의 65% 를 계통 오프셋이 먼저 먹었다.
      v2 는 컵으로 재고 **목표 상자를 그 오프셋만큼 평행이동**해(`GOAL_POINT_V2`)
      도달성 검증(TCP 제약 IK)과 채점 대상을 다시 일치시킨다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.norm(goal_pos_w(env, command_name, robot_cfg) - obj.data.root_pos_w, dim=1)


def grasp_point_w(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """파지점 — 컵 원점에서 **컵 로컬 축**을 따라 내린 점.

    world z 로 내리면 컵이 기울었을 때 파지점이 컵 밖으로 나간다(v1 실측 근거).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_z = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
    return obj.data.root_pos_w + cup_z * P.CUP_ORIGIN_TO_GRASP_Z


# ---------------------------------------------------------------------------
# Stage 1 — reach & grasp
# ---------------------------------------------------------------------------
def stage_reach(env: "ManagerBasedRLEnv",
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """`D(p_tcp, p_grasp, s_r, 0)`. 목표는 컵 원점이 아니라 **파지점**이다.

    컵 원점(상면 +92 mm)을 겨냥하면 안 된다 — v1 이 실측으로 잡은 함정이다.
    ※09.02 정정: 그 근거였던 "지름 88 mm 가 개구 84.5 mm 보다 넓어 못 들어간다"는
      **틀렸다**. PhysX 실측 개구는 100 mm 이고 상면 +92 mm 의 지름은 68 mm 다.
      결론(대역으로 clamp)은 유효하나 근거가 바뀌었다.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    tcp = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(grasp_point_w(env, object_cfg) - tcp, dim=1)
    return d_shape(dist, P.REACH_S, P.REACH_TAU)


def stage_close(env: "ManagerBasedRLEnv",
                jaw_cfg: SceneEntityCfg,
                object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Lee 의 `R_close` — 파지가 성립하면 1, 아니면 **닫으려는 노력에 절반**만 준다.

    Lee 원문은 "grasp sensor triggered" 를 쓴다. 우리 대응물은 `grasp_ok`(기하 판정)다.
    부분 점수는 v1 의 `grip_closure_when_enclosed`(align × enclose × closure) 구조를
    그대로 쓴다 — "감싸지 않은 폐쇄"가 정확히 0 이어야 옛 주먹 해킹이 안 살아난다.
    """
    # ★`band=` 를 반드시 넘긴다 — v2 의 파지 대역은 v1(판 위 10~85 mm)이 아니라
    #   판 위 80~150 mm 다. 안 넘기면 v1 기본값으로 조용히 되돌아간다.
    ok = grasp_ok(env, P.GRASP_GATE_LATERAL_OK, P.GRASP_GATE_ALONG_OK,
                  P.JAW_PAD_OFFSET, jaw_cfg, object_cfg,
                  band=P.CUP_GRASP_BAND_AXIS).float()
    align, enclose = _jaw_geometry(env, P.JAW_ALONG_STD, P.JAW_LATERAL_STD,
                                   P.JAW_ENCLOSE_HALF_WIDTH, P.JAW_PAD_OFFSET,
                                   jaw_cfg, object_cfg, band=P.CUP_GRASP_BAND_AXIS)
    robot: Articulation = env.scene[jaw_cfg.name]
    drive = robot.data.joint_pos[:, robot.joint_names.index(P.GRIPPER_DRIVE_JOINT)]
    closure = (1.0 - drive / P.GRIPPER_OPEN_POS).clamp(0.0, 1.0)
    partial = 0.5 * align * enclose * closure
    return torch.where(ok > 0.5, torch.ones_like(partial), partial)


def stage_grasp(env: "ManagerBasedRLEnv",
                jaw_cfg: SceneEntityCfg,
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Lee: `R_grasp = R_reach · (0.5 + R_close/2  if R_reach>0.9  else 0.5)`.

    ★접근 품질이 충분히 좋아진 **뒤에야** 닫기가 보상된다. 이 곱셈 게이트가
      "멀리서 닫고 서 있기" 국소최적을 원천 차단한다(v1 은 그것을 액션 게이트로 막았다).
    """
    r_reach = stage_reach(env, ee_frame_cfg, object_cfg)
    r_close = stage_close(env, jaw_cfg, object_cfg)
    bonus = torch.where(r_reach > 0.9, 0.5 + 0.5 * r_close, torch.full_like(r_reach, 0.5))
    return r_reach * bonus


# ---------------------------------------------------------------------------
# Stage 2 — lift
# ---------------------------------------------------------------------------
def stage_lift(env: "ManagerBasedRLEnv",
               jaw_cfg: SceneEntityCfg,
               object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Lee: `R_lift = R_close · R_move_up`. 램프는 v1 값(스폰 +6 mm → +40 mm)."""
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    move_up = ((z - P.LIFT_RAMP_ZERO_Z)
               / (P.MINIMAL_LIFT_HEIGHT - P.LIFT_RAMP_ZERO_Z)).clamp(0.0, 1.0)
    return stage_close(env, jaw_cfg, object_cfg) * move_up


# ---------------------------------------------------------------------------
# Stage 3 — transport
# ---------------------------------------------------------------------------
def stage_transport(env: "ManagerBasedRLEnv",
                    command_name: str,
                    robot_cfg: SceneEntityCfg,
                    jaw_cfg: SceneEntityCfg,
                    object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """문서 6.3: `R_transport = R_lift · D(p_cup, p_goal, s_t, τ_t)`.

    리프트에 **곱해져** 활성화된다 — "transport 는 grasp/lift 성공 후에만"(강한 권고 3).
    거리는 **컵 기준**이다(`cup_goal_distance` docstring 참조).
    """
    dist = cup_goal_distance(env, command_name, robot_cfg, object_cfg)
    return stage_lift(env, jaw_cfg, object_cfg) * d_shape(dist, P.TRANSPORT_S, P.TRANSPORT_TAU)


# ---------------------------------------------------------------------------
# Stage 4 — settle (쥔 채 정지) : **곱 문턱 → AND 조건**
# ---------------------------------------------------------------------------
def stage3_ok(dist: torch.Tensor, r_close: torch.Tensor) -> torch.Tensor:
    """stage 3 승급 판정 — **도달 + 파지** 두 조건의 AND (1.0 / 0.0).

    ★★08.29 라운드 4 수정. 라운드 3 은 여기에 속도·직립까지 넣었는데, 그 둘은
      **값(`v_3`)으로 옮겼다.** 이유:
        · 문턱에 넣으면 "정지·직립을 못 하면 4 층 자체가 안 열려" 개선 방향을 못 배운다
          (구 설계 D3 와 같은 절벽). 실측 `stage3_ok` 라운드 3 최대 0.0024.
        · 값에 두면 4 층 안에서 중심 접근·감속·직립 셋 다 gradient 가 산다.
      ⚠ 그래서 `stage3_ok` 는 더 이상 "성공"이 아니다 — 성공 판정은 `success_ok` 다.

    ★★현행 `r_settle = r_transport·at_goal·still·upright > 0.1` 은 **네 인자 곱**이
      0.1 을 넘기를 요구한다. 실측 6 판에서 `r_settle` 최대가 0.0106, 대부분 정확히 0 —
      **4 단 계단의 4 층이 학습 신호로 존재한 적이 없다**. 자매 트랙도 같은 함정에
      빠진 이력이 있다(`lstm_test8`: 네 인자 곱이 0.008 로 소멸해 어느 방향으로도
      gradient 가 없었다).

    AND 로 바꾸면 각 조건이 독립적으로 명확해지고, 진행도(`v_2`)가 그 셋을 연속으로
    잇는다 — 판정은 이산, 값은 연속이라는 분업이다.

    ★`r_close` 조건은 reward-audit Check 3 의 REVISE 다. 초안에는 파지 항이 없어
      목표 근처에서 컵을 **놓아도** 값이 유지됐다.
      ⚠ `stage_close` 는 `grasp_ok` 성립 시 정확히 1.0, 아니면 `0.5·align·enclose·closure`
        ≤ 0.5 다. 따라서 `r_close > 0.5` ⟺ `r_close == 1.0` — 이 성질 덕분에 승급
        순간 `v_2` 가 정확히 1 이 되어 **문턱이 완전 연속**이다(계약으로 잠근다).
    """
    return ((dist < P.SETTLE_RADIUS) & (r_close > P.STAGE3_GRASP_MIN)).float()


def settle_success(env: "ManagerBasedRLEnv",
                   command_name: str,
                   robot_cfg: SceneEntityCfg,
                   jaw_cfg: SceneEntityCfg,
                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """씬에서 직접 `success_ok` 를 계산한다 — hold 프리미엄(라운드 7)의 판정 입력.

    ⚠ 속도는 **순간속도**다. 순변위로 바꾸면 제자리 진동(순변위 ≈ 0)이 만점을 받아
      hold 가 흔들기에 프리미엄을 주게 된다(라운드 1 R1 의 함정).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    dist = cup_goal_distance(env, command_name, robot_cfg, object_cfg)
    speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    return success_ok(dist, speed, _cup_upright_cos(env, object_cfg),
                      stage_close(env, jaw_cfg, object_cfg))


def success_ok(dist: torch.Tensor, speed: torch.Tensor,
               upright_cos: torch.Tensor, r_close: torch.Tensor) -> torch.Tensor:
    """**합격 판정** — 도달 + 정지 + 직립 + 파지 유지 (보상 아님, 진단 전용).

    참고 문서 "강한 권고 4"(성공은 도달만이 아니라 정지·upright 유지까지)의 우리 구현.
    승급 문턱에서 뺀 두 조건이 여기 남아 있어 **과제 정의는 그대로**다 —
    보상 지형만 부드럽게 하고 합격선은 안 낮춘다.
    """
    # ★★08.31 라운드 15 — **속도 조건 제거**(사용자 지시). 근거는 `CENTER_BAND`
    #   주석 참조: 진동체가 반환점마다 속도 0 을 지나 통과했고(평균 0.145 m/s인데
    #   ⑤ 49.2%), 최장 연속 합격이 1 스텝이었다. 합격은 **콜라이더 체류**로 잰다.
    #   `speed` 인자는 호출부 호환을 위해 남기되 판정에 쓰지 않는다.
    del speed
    return ((dist < P.SETTLE_RADIUS)
            & (upright_cos > P.STAGE3_UPRIGHT_MIN)
            & (r_close > P.STAGE3_GRASP_MIN)).float()


# ---------------------------------------------------------------------------
# 계단 전체를 한 번에
# ---------------------------------------------------------------------------
def all_stages(env: "ManagerBasedRLEnv",
               command_name: str,
               robot_cfg: SceneEntityCfg,
               jaw_cfg: SceneEntityCfg,
               ee_frame_cfg: SceneEntityCfg,
               object_cfg: SceneEntityCfg,
               net_speed: torch.Tensor | None = None,
               lift_only: bool = False) -> tuple[tuple, tuple]:
    """`(pos, val)` — 판정용 양과 보상용 진행도. **의미가 다른 두 벌**이다.

      · `pos = (r_grasp, r_lift, r_transport, ok3)` → **stage 인덱스 판정**에만 쓴다.
        앞 셋은 v1 부터 이어진 정의 그대로이고(로그 비교 가능성 유지), `ok3` 만
        네 조건 AND 로 새로 잡았다. `_stage_index` 가 `> 0.1` 로 재는데 `ok3` 는
        0/1 이라 판정 배선은 한 글자도 안 바뀐다.
      · `val = (v_0, v_1, v_2, v_3)` → **보상 크기**. 각각 [0,1] 이고 **다음 문턱에서
        정확히 1** 이 되도록 정의해 계단 경계의 점프를 없앴다.

    ★★왜 이렇게 나누는가 (08.29 라운드 3, 6 판 실측 근거):
      계단 값이 `r_lift` 그 자체였을 때 **stage 1 안에 목표 거리가 없어** ∂r/∂dist = 0
      인 고원이 생겼다. stage 2 문턱은 컵–목표 150~161 mm 인데 붕괴 후 실측 `cupd` 가
      146 · 160 · 170 mm — 세 판 모두 그 고원에 주차했다. 문턱을 일찍 넘느냐가
      시드 운에 달렸고, 그것이 시드 의존(0.095 vs 0.285)의 기계적 원인이다.
      값을 "다음 문턱까지의 진행도"로 재정의하면 고원과 절벽이 동시에 사라진다.

    `net_speed=None` 이면 속도 입력이 **순간속도**, 주면 **순변위 속도**다. 그 하나만
    바뀌고 **식은 완전히 동일하다** — 라운드 3 의 단일 변수가 구조적으로 보장된다.

    ⚠ 단계들이 서로를 곱하는 구조라 개별 함수를 따로 부르면 하위 단계가 중복 계산된다.
      학습 경로(`v2_rewards.Staircase`)는 반드시 이 함수를 쓴다.
    """
    r_reach = stage_reach(env, ee_frame_cfg, object_cfg)
    r_close = stage_close(env, jaw_cfg, object_cfg)
    bonus = torch.where(r_reach > 0.9, 0.5 + 0.5 * r_close, torch.full_like(r_reach, 0.5))
    r_grasp = r_reach * bonus

    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    move_up = ((z - P.LIFT_RAMP_ZERO_Z)
               / (P.MINIMAL_LIFT_HEIGHT - P.LIFT_RAMP_ZERO_Z)).clamp(0.0, 1.0)
    r_lift = r_close * move_up

    # ── ★★09.03 리프트 전용 (사용자 결정: "goal 은 필요없고 lift 만") ──────────
    #   계단을 **2 단(grasp → lift)** 으로 줄이고 리프트를 **최종 단계**로 만든다.
    #   왜: 4 단에서는 리프트가 중간 계단이라 "잡고 가만히"(v_0)와 이득 차이가 작았고,
    #   재소환을 끄자 정책이 그 국소최적에 갇혔다(G1 실측 — r_grasp 0.765 인데
    #   r_lift 0.0115, 400 epoch 평평). 리프트를 만점으로 두면 그 이득 차가 2 배가 된다.
    #   ⚠ 이송·정지 능력은 **의도적으로 버린다.** 목표 상자는 obs 에만 남는다.
    if lift_only:
        zero = torch.zeros_like(r_grasp)
        return (r_grasp, r_lift, zero, zero), (r_grasp, move_up, zero, zero)

    dist = cup_goal_distance(env, command_name, robot_cfg, object_cfg)
    r_transport = r_lift * d_shape(dist, P.TRANSPORT_S, P.TRANSPORT_TAU)

    speed = (torch.norm(obj.data.root_lin_vel_w, dim=1)
             if net_speed is None else net_speed)
    upright_cos = _cup_upright_cos(env, object_cfg)
    ok3 = stage3_ok(dist, r_close)

    # ── 판정 (pos) — 문턱은 그대로, stage 3 만 AND 조건 ────────────────────
    #   `_stage_index` 는 `> STAGE_THRESHOLD(0.1)` 로 판정하는데 `ok3` 는 0 또는 1 이라
    #   식을 바꾸지 않아도 정확히 같은 뜻이 된다(배선 무변경).
    pos = (r_grasp, r_lift, r_transport, ok3)

    # ── 값 (val) — 각각 [0,1] 이고 **다음 문턱에서 정확히 1** ──────────────
    #   v_0 : 불변. grasp/lift 는 6 판 내내 개선만 됐다(Check 4).
    #   v_1 : ★D1(stage 1 고원)과 D2(램프 조기 포화)를 동시에 없앤다. dist 가 들어와
    #         z 상승(r_lift↑)과 목표 접근(D↑) 양쪽에 gradient 가 산다. 다음 문턱
    #         (r_transport = 0.1)에서 정확히 1 이라 경계 점프가 0.003 로 줄어든다.
    #         ★리프트 램프 상수는 **안 건드린다** — 램프가 40 mm 에서 포화해도
    #           D(dist) 가 계속 위로 끌어올린다(회귀 위험이 더 작다).
    #   ★★★라운드 5 — v_0·v_1·v_2 를 **baseline 원본으로 되돌린다.**
    #   env 별 결말 프로브(결정론 1024 env)가 밝힌 것: baseline 은 ①~④ 를 이미 푼다.
    #       파지 실패 0.6% · 리프트 미도달 5.8% · **도달 93.7%** · 성공 0.0%
    #       컵–목표 최저 거리 중앙 28.1 mm
    #   7 판 내내 `diag_at_goal`(스텝 비율) 0.19 를 "이송 실패"로 읽었는데, 그건
    #   "스텝의 19% 만 반경 안"이지 "env 의 19% 만 도달"이 아니었다. 스텝 평균은
    #   **"못 간다"와 "갔다가 지나친다"를 같은 값으로 뭉갠다.**
    #   ⇒ 남은 문제는 ⑤ 정지 하나뿐이므로, 0~2 단계는 **건드리지 않는다**.
    #
    #   구 주석(라운드 4, 폐기): v_2 를 거리 진행도만으로 바꿨다 — 그 결과 stage 2
    #   신호가 얇아져 σ 가 1.3 까지 팽창했고(신호 고갈), 결정론 정책이 무너졌다
    #   (확률적 파지 0.836 vs 결정론 0.190 · 컵 85% 전도).
    #   v_2(폐기) : ★★거리 진행도만. 라운드 4 의 핵심 수정이었다.
    #         라운드 3 은 `v_2 = r_close·P_dist·P_still·P_upright` 로 네 인자를 곱했다.
    #         설계 시 지형 스캔이 `upright=1·speed=0` 인 **이상적 슬라이스**만 봐서
    #         "유인 +0.250" 으로 읽혔는데, 실측 운전점(`P_still` 0.43 · `P_upright` 0.71)
    #         에서는 곱이 0.31 로 축소돼 **유인이 +0.077** 이었다 — 구 설계(+0.207)의
    #         2.7 배 약화. ep1053 까지 `atgoal` 정점 0.046(baseline 0.285)로 실측 확인.
    #         ⇒ 문턱에서 뺐던 네 인자 곱을 **값에 다시 넣은 것**이 실패 원인이었다.
    #         이제 stage 2 는 속도·직립에 **불변**이라 운전점과 무관하게 +0.250 이다.
    #   v_3 : 속도·직립은 **여기로 모은다.** 4 층은 이미 목표 반경 안이라 셋 다 달성
    #         가능하고, 곱이어도 각 인자에 gradient 가 산다.
    #         `P_center` 가 문턱에서 0 이라 2→3 경계는 여전히 **완전 연속**이다.
    #   v_3 : ⑤ 를 담당하는 **유일한 신규 항**. 구 보상은 직립·정지에 gradient 가
    #         하나도 없었다 — 담당 항 `r_settle` 이 네 인자 곱이라 6 판 통틀어 문턱
    #         (0.1)을 한 번도 못 넘었기 때문이다(최대 0.0106).
    #         실측 병목(도달 env 기준): 파지 99.8% · 정지 56.1% · **직립 23.5%** ·
    #         네 조건 **동시 0.0%**. 곱으로 두면 "동시"가 곧 최적화 대상이 된다.
    #         밴드 near 가 합격선(`STAGE3_*`)보다 엄격하므로 `v_3 = 1` ⟹ 합격이다.
    p_still = smoothstep(speed, *P.P_STILL_BAND)
    p_upright = smoothstep(upright_cos, *P.P_UPRIGHT_BAND)
    # ★★08.31 라운드 15 — `v_3` 를 **중심 접근 × 직립**으로. 속도는 뺀다.
    #   "정중앙에 올수록 큰 보상"(사용자 지시). 체류 보상은 hold 프리미엄이 맡는다 —
    #   속도 조건이 빠지면서 `hold_t` 가 곧 **콜라이더 연속 체류 스텝**이 된다.
    p_center = smoothstep(dist, *P.CENTER_BAND)
    v3 = p_center * p_upright

    # 진단 전용 — 보상 경로에는 안 들어간다(`progress_terms` 가 같은 식을 쓴다).
    return pos, (r_grasp, r_lift, r_transport, v3)


def upright_shaped(env: "ManagerBasedRLEnv",
                   object_cfg: SceneEntityCfg) -> torch.Tensor:
    """컵 직립도의 셰이핑 값 [0,1] — `v_3` 이 쓰는 것과 **같은 밴드**다.

    ★라운드 11(처방 A). 계단 밖 가산항 전용. 같은 밴드를 쓰므로 이송 중에 올린
      직립이 stage 3 의 `p_upright` 로 그대로 이어진다(두 신호가 안 싸운다).
    """
    return smoothstep(_cup_upright_cos(env, object_cfg), *P.P_UPRIGHT_BAND)


def progress_terms(env: "ManagerBasedRLEnv",
                   command_name: str,
                   robot_cfg: SceneEntityCfg,
                   object_cfg: SceneEntityCfg,
                   net_speed: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """진단용 `(P_dist, P_still, P_upright, P_center)`. 보상 경로와 **같은 식**을 쓴다.

    넷을 따로 찍어야 "이송이 왜 안 되는가"가 로그만으로 갈린다 — 거리를 못 좁히는
    것인지(`P_dist`), 못 멈추는 것인지(`P_still`), 컵이 기우는 것인지(`P_upright`),
    목표 안에서 중심을 못 잡는 것인지(`P_center`).
    ★라운드 3 실패의 진단이 정확히 이 분해에서 나왔다: `P_dist` 는 0.12~0.36 인데
      `P_still`·`P_upright` 가 곱해져 stage 2 값이 0.31 배로 깎이고 있었다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    dist = cup_goal_distance(env, command_name, robot_cfg, object_cfg)
    speed = (torch.norm(obj.data.root_lin_vel_w, dim=1)
             if net_speed is None else net_speed)
    return (smoothstep(dist, *P.P_DIST_BAND),
            smoothstep(speed, *P.P_STILL_BAND),
            smoothstep(_cup_upright_cos(env, object_cfg), *P.P_UPRIGHT_BAND),
            smoothstep(dist, P.SETTLE_RADIUS, 0.0))
