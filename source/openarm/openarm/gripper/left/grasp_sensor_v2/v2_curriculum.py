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

"""v2 ADR 사다리 — 질량·스폰·목표 범위를 성공에 맞춰 단계적으로 넓힌다 (라운드 9).

레벨 0 = 현행 과제와 완전히 동일. 승급 = `P.ADR_SIGNAL_TERM` 의 에피소드 누적 EMA 가
게이트를 넘고 최소 간격이 지났을 때 한 칸. TFEvents 에 `Curriculum/adr` 로 레벨이 찍힌다.

⚠⚠ 단위는 **초**다. `RewardManager` 가 weight 0 항을 `+= raw_value * dt`(dt=0.02 s)로
  누적한다 — "스텝 수"가 아니다. 라운드 10 에서 이걸 스텝으로 오독해 게이트를 실측의
  17 배로 잡았고, 사다리가 완주까지 레벨 0 에 고정됐다.
⚠ 승급 신호는 σ 에 견디는 항이어야 한다. per-step 4조건 동시 만족 플래그(구
  `diag_success`)는 탐색 노이즈 아래서 거의 0 이라 승급 신호로 못 쓴다.
⚠ 커리큘럼은 reward sum 리셋 **전에** 불린다(`_reset_idx` 순서 실측) — 그래서
  `_episode_sums[P.ADR_SIGNAL_TERM]` 가 방금 끝난 에피소드의 누적을 담고 있다.
⚠ 승급이 조작하는 것은 네 곳뿐이다: 목표 명령 ranges · 스폰 pose_range ·
  질량 이벤트 params · obs bias 이벤트 params.
  마찰은 startup 버킷이라 여기서 못 건드린다(정적 DR).
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg  # noqa: F401

from . import v2_preset as P


class ADRLadder(ManagerTermBase):
    """`CurriculumTermCfg(func=ADRLadder)` 로 등록한다. 반환값 = 현재 레벨."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # ★라운드 13 — `fixed_level >= 0` 이면 그 레벨에 **고정**하고 승급·강등을 끈다.
        #   대조군 B(사다리 없이 처음부터 만렙)를 같은 코드 경로로 만든다.
        # ★★09.03 — 값을 **env.cfg 에서 직접** 읽는다. term params 로 받으면
        #   `__post_init__` 실행 시점에 구워져 hydra CLI 오버라이드
        #   (`env.v2_adr_fixed_level=-1`)가 **조용히 무시된다** — 실제로 F2 가
        #   그렇게 E29 의 완전한 재실행이 되어 버렸다. params 는 명시된 경우에만
        #   우선하고, 없으면 cfg 를 본다.
        _p = getattr(cfg, "params", {}) or {}
        if "fixed_level" in _p:
            self._fixed = int(_p["fixed_level"])
        else:
            self._fixed = int(getattr(env.cfg, "v2_adr_fixed_level", -1))
        self._level = max(self._fixed, 0)
        self._ema = 0.0
        self._last_promo = 0
        # 목표 상자 중심 — 현재 ranges 에서 역산해 보관(GOAL_MEASURED 스위치와 무관하게 옳다)
        rg = env.command_manager.get_term("object_pose").cfg.ranges
        self._goal_center = tuple(0.5 * (lo + hi) for lo, hi in
                                  (rg.pos_x, rg.pos_y, rg.pos_z))

    def _apply(self, env) -> None:
        f = self._level / (P.ADR_LEVELS - 1)
        # ① 목표 상자 (x 는 불변 — 판 앞모서리)
        jx = P.GOAL_JITTER_V2[0]
        jy = P.GOAL_JITTER_V2[1] + f * (P.ADR_GOAL_JITTER_MAX[1] - P.GOAL_JITTER_V2[1])
        jz = P.GOAL_JITTER_V2[2] + f * (P.ADR_GOAL_JITTER_MAX[2] - P.GOAL_JITTER_V2[2])
        rg = env.command_manager.get_term("object_pose").cfg.ranges
        cx, cy, cz = self._goal_center
        rg.pos_x = (cx - jx, cx + jx)
        rg.pos_y = (cy - jy, cy + jy)
        rg.pos_z = (cz - jz, cz + jz)
        # ② 스폰 상자 — **절대 상자 보간**(레벨 0 → P1 실측 봉투).
        #   ★★08.30 라운드 13 — 라운드 12 의 "하한을 자르고 +x 로 넓힌다" 클램프를
        #     **폐기**한다. P1 정책 스윕이 방향이 반대임을 보였다:
        #       x 0.344 에서 ① 파지 실패 0~7%(멀쩡)  ·  x 0.417 에서 46~70%
        #       x 0.436 에서 **100%**  ⇒ 벽은 −x 가 아니라 **+x(≈0.41)** 다.
        #     `SPAWN_X_SAFE_MIN`(구 홈의 관통 경계)은 지금 홈에서 무효라 참조하지 않는다.
        #   ★봉투가 중심에 대해 비대칭이라 ± 오프셋으로 표현 불가 — 절대 경계를 보간하고
        #     이벤트가 요구하는 **중심 기준 오프셋**으로 마지막에 변환한다(부호 함정).
        b0, bm = P.ADR_SPAWN_BOX_L0, P.ADR_SPAWN_BOX_MAX
        xlo, xhi, ylo, yhi = (b0[i] + f * (bm[i] - b0[i]) for i in range(4))
        pr = env.event_manager.get_term_cfg("reset_object_position").params["pose_range"]
        pr["x"] = (xlo - P.CUP_SPAWN_X_CENTER, xhi - P.CUP_SPAWN_X_CENTER)
        pr["y"] = (ylo - P.CUP_SPAWN_Y_CENTER, yhi - P.CUP_SPAWN_Y_CENTER)
        # ③ 컵 질량 scale
        lo = 1.0 + f * (P.ADR_MASS_SCALE_MAX[0] - 1.0)
        hi = 1.0 + f * (P.ADR_MASS_SCALE_MAX[1] - 1.0)
        env.event_manager.get_term_cfg("dr_cup_mass").params[
            "mass_distribution_params"] = (lo, hi)
        # ④ 컵 obs 의 에피소드 bias (실기 /cup_pose 캘리브 오차 모사)
        ob = f * P.ADR_OBS_BIAS_MAX
        env.event_manager.get_term_cfg("dr_obs_bias").params["bias_range"] = ob
        print(f"[ADR] level {self._level}  goal_j=({jx:.3f},{jy:.3f},{jz:.3f})"
              f"  spawn=x[{xlo:.3f},{xhi:.3f}] y[{ylo:.3f},{yhi:.3f}]"
              f"  mass=({lo:.2f},{hi:.2f})  obs_bias=±{ob*1000:.0f}mm", flush=True)

    def __call__(self, env, env_ids, fixed_level: int = -1) -> torch.Tensor:
        if self._fixed >= 0:
            if not getattr(self, "_fixed_applied", False):
                self._apply(env)
                self._fixed_applied = True
            return torch.tensor(float(self._level), device=env.device)
        sums = getattr(env.reward_manager, "_episode_sums", {}).get(P.ADR_SIGNAL_TERM)
        if sums is not None and len(env_ids) > 0:
            val = float(sums[env_ids].mean())
            if math.isfinite(val):
                self._ema = (1 - P.ADR_EMA_ALPHA) * self._ema + P.ADR_EMA_ALPHA * val
        step = int(getattr(env, "common_step_counter", 0))
        if (self._level < P.ADR_LEVELS - 1
                and self._ema >= P.ADR_SUCCESS_GATE
                and step - self._last_promo >= P.ADR_MIN_STEPS_BETWEEN):
            self._level += 1
            self._last_promo = step
            # ★승급 시 EMA 를 반토막 — 새(더 어려운) 분포에서 다시 증명하게 한다.
            #   안 하면 이전 레벨의 성공 잔고로 연쇄 승급해 사다리가 계단이 아니게 된다.
            self._ema *= 0.5
            self._apply(env)
        # ── ★★08.30 라운드 12 — **강등**. 사다리가 한 방향뿐이라 소화 못 하는
        #   레벨에 영구 고착됐다: 레벨 2 승급(ep960) 후 승급 신호가 2.50 → 0.62 초로
        #   단조 하락했는데 1540 epoch 동안 아무 대응이 없었다(간격 150 판은 레벨 3
        #   에서 400 epoch 째 succ 0.000). 못 버티면 한 칸 내려가 다시 오르게 한다.
        #   ⚠ 문턱을 승급선보다 **낮게**(절반) 둬서 승급↔강등 진동을 막는다(히스테리시스).
        elif (self._level > 0
                and self._ema < P.ADR_DEMOTE_GATE
                and step - self._last_promo >= P.ADR_MIN_STEPS_BETWEEN):
            self._level -= 1
            self._last_promo = step
            self._apply(env)
            print(f"[ADR] ↓강등 level {self._level}  (신호 EMA {self._ema:.2f}"
                  f" < {P.ADR_DEMOTE_GATE})", flush=True)
        return torch.tensor(float(self._level), device=env.device)
