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

"""gripper/left/grasp_sensor reward — 2지 평행 그리퍼 전용.

right/grasp_sensor(grasp_reward.py)의 **8-term 계약을 그대로 유지**한다
(GRASP_V2_REWARD_TERMS). 계약을 지키면 기존 TFEvents 파싱·분석 도구·모니터가
수정 없이 그대로 붙는다. 바뀐 것은 `grasp` term 의 내부 품질 정의뿐이다.

왜 grasp_quality 를 바꿔야 하나
-------------------------------
우측의 품질은 5지 손의 **감쌈(envelope/wrap)** = "중간마디와 원위마디가 함께 닿는 깊이"로
정의돼 있다. 2지 평행 그리퍼에는 감쌈이라는 자유도가 물리적으로 없다 — 마디가 하나뿐이고
개폐 1 DOF 다. 그래서 감쌈 대신 **대향(opposition)** 과 **압착(squeeze)** 으로 바꾼다:

  · 대향  = 두 접촉점이 컵 단면의 지름 양끝에 있는가 (force-closure 의 최소 조건)
  · 압착  = 그리퍼가 컵에 막혀 생긴 관절 위치 오차 (지령 − 실측 = 파지력 대리)

reward-audit (2026-08-19)
-------------------------
Check 1 Local Min : ✓ 가중합을 1.0 으로 재정규화 → grasp 최대치가 우측과 동일(12.0).
                      lift(30) + success(20) 대비 0.24배라 "쥐기만 하고 안 듦" 수렴 불가.
                      게다가 grasp 은 pre_lift_gate 라 래치 후 꺼진다.
Check 2 Hacking   : ✓ **단, 조건부**. squeeze 를 관절오차 그대로 쓰면 컵 없이도 farm 된다 —
                      URDF/USD 상 q=0(완전폐쇄)에서 두 핑거가 3.5 mm 겹치므로 빈 그리퍼를
                      닫아도 오차가 남을 수 있다(self-collision 설정에 의존).
                      → `squeeze` 에 **접촉 게이트를 곱해** 컵 접촉 없이는 0 이 되게 한다.
                      컵을 테이블에 눌러 압착만 하는 경로는 approach_xy_penalty(25.0)와
                      pre_lift_gate 가 억제한다.
Check 3 Grasp충돌  : ✓ 2지에는 tilt↔감쌈 상충 구조가 없다.
Check 4 기존파괴   : ✓ 신설 태스크. 우측 파일은 한 줄도 건드리지 않는다.
Check 5 측정      : ✓ 8-term 계약 유지 + gates 에 opposition/squeeze 노출.
판정: ACCEPT

⚠ latch 게이트는 **완화하지 않는다**. "양 핑거 접촉 AND + 연속 hold" 를 유지한다.
   1지 접촉만으로 래치되게 풀면 부실 파지 국소최적이 생긴다
   (좌측 grasp_v1 에서 게이트 완화로 lifted 0.72→0.002 붕괴한 이력).
"""

from __future__ import annotations

import torch

from openarm.common.grasp_v2_contract import GRASP_V2_REWARD_TERMS


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_gripper_grasp_reward_terms(
    *,
    # ── 접촉 ────────────────────────────────────────────────────────
    contact_frac: torch.Tensor,          # (N,) 컵에 닿은 핑거 비율 0 / 0.5 / 1
    both_contact: torch.Tensor,          # (N,) bool 두 핑거 모두 접촉
    contact_persistence_frac: torch.Tensor,
    opposition: torch.Tensor,            # (N,) 0~1 두 접촉점의 대향도
    squeeze_frac: torch.Tensor,          # (N,) 0~1 그리퍼 관절오차 정규화 (접촉 게이트 적용 전)
    # ── 기하 ────────────────────────────────────────────────────────
    tcp_to_cup_dist: torch.Tensor,
    finger_side_dist: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_xy_displacement: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    upright_quality: torch.Tensor,
    # ── 상태 ────────────────────────────────────────────────────────
    lift_latched: torch.Tensor,
    action_delta_norm: torch.Tensor,
    stabilize_reward_gate: torch.Tensor | None = None,
    success_now: torch.Tensor | None = None,
    stable: torch.Tensor | None = None,
    stability_quality: torch.Tensor | None = None,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """2지 그리퍼 grasp reward. 반환 계약은 right/grasp_sensor 와 동일."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    contact_frac = contact_frac.clamp(0.0, 1.0)
    contact_persistence_frac = contact_persistence_frac.clamp(0.0, 1.0)
    opposition = opposition.clamp(0.0, 1.0)
    both = both_contact.float()

    # ★Check 2 완화: 컵 접촉이 없으면 압착 신호를 0 으로 만든다.
    #   (빈 그리퍼를 닫아 생기는 관절오차로 보상을 farm 하는 경로 차단)
    any_contact = (contact_frac > 0.0).float()
    squeeze = any_contact * squeeze_frac.clamp(0.0, 1.0)

    graded_contact = contact_frac

    lifted_bool = cup_height_delta >= _cfg_float(cfg, "lift_success_height", 0.04)
    lifted_gate = lifted_bool.float()
    reward_upright_success = cup_tilt_deg <= _cfg_float(cfg, "success_upright_max_deg", 20.0)
    final_upright_success = cup_tilt_deg <= _cfg_float(cfg, "stabilize_upright_max_deg", 5.0)
    stabilize_gate = (
        lift_gate if stabilize_reward_gate is None else stabilize_reward_gate.float()
    )

    # ── approach ────────────────────────────────────────────────────
    xy_margin = _cfg_float(cfg, "grasp_xy_threshold", 0.0)
    tilt_margin = _cfg_float(cfg, "grasp_upright_threshold_deg", 0.0)
    approach = pre_lift_gate * (
        _cfg_float(cfg, "approach_weight", 0.0)
        * torch.exp(
            -_cfg_float(cfg, "approach_sharpness", 1.0)
            * (tcp_to_cup_dist + finger_side_dist)
        )
        - _cfg_float(cfg, "approach_xy_penalty_weight", 0.0)
        * torch.relu(cup_xy_displacement - xy_margin)
        - _cfg_float(cfg, "approach_tilt_penalty_weight", 0.0)
        * torch.relu(cup_tilt_deg - tilt_margin)
    )

    # ── grasp: 2지 대향 파지 품질 ───────────────────────────────────
    # 가중합 = 1.0 (재정규화). 최대치가 우측(envelope 판)과 같으므로 grasp_weight 를
    # 그대로 물려받아도 항목 간 비율이 보존된다(Check 1).
    #   contact_frac      닿았는가 (가장 기본, 조밀한 gradient)
    #   both × opposition 두 점이 지름 양끝인가 (force-closure 최소조건)
    #   persistence       놓치지 않고 유지하는가
    #   squeeze           실제로 힘이 걸렸는가 (접촉 게이트 적용됨)
    _w_opp = _cfg_float(cfg, "grasp_opposition_credit", 0.25)
    _w_sqz = _cfg_float(cfg, "grasp_squeeze_credit", 0.20)
    _w_rest = max(0.0, 1.0 - _w_opp - _w_sqz)
    grasp_quality = (
        _w_rest * (0.55 * contact_frac + 0.45 * contact_persistence_frac)
        + _w_opp * both * opposition
        + _w_sqz * squeeze
    )
    grasp = _cfg_float(cfg, "grasp_weight", 0.0) * pre_lift_gate * grasp_quality

    # ── lift ────────────────────────────────────────────────────────
    _h_ref = _cfg_float(cfg, "lift_height_ref", 0.0)
    if _h_ref <= 0.0:
        _h_ref = _cfg_float(cfg, "lift_success_height", 0.04)
    lift_height_quality = (cup_height_delta / max(_h_ref, 1e-6)).clamp(min=0.0, max=1.0)
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * graded_contact
        * lift_height_quality
        * upright_quality
    )

    # ── stabilize / stability ───────────────────────────────────────
    action_quality = torch.exp(
        -_cfg_float(cfg, "stabilize_action_sharpness", 1.0) * action_delta_norm
    )
    stable_gate = (
        torch.ones_like(cup_height_delta) if stable is None else stable.float()
    )
    stability_quality_f = (
        stable_gate if stability_quality is None else stability_quality
    ).clamp(min=0.0, max=1.0)
    stabilize = (
        _cfg_float(cfg, "stabilize_weight", 0.0)
        * lift_gate * lifted_gate * graded_contact * upright_quality * action_quality
    )
    stability = (
        _cfg_float(cfg, "stability_reward_weight", 0.0)
        * stabilize_gate * lifted_gate * graded_contact * upright_quality
        * stability_quality_f
    )

    # ── 리프트 후 접촉 상실 ─────────────────────────────────────────
    # 우측의 wrap_retention_loss(래치 시점 감쌈 대비 감소분)는 **이식하지 않는다** —
    # 2지에는 "중간마디를 잃고 손끝으로 미끄러진다"는 실패 양상 자체가 없다.
    # 접촉을 잃으면 contact_frac 이 곧바로 떨어지므로 아래 한 항으로 충분하다.
    post_lift_contact_loss = (
        _cfg_float(cfg, "post_lift_contact_loss_weight", 0.0)
        * lift_gate * lifted_gate * torch.relu(1.0 - contact_frac)
    )

    success_now_bool = (
        torch.zeros_like(lift_latched, dtype=torch.bool)
        if success_now is None
        else success_now.bool()
    )
    success_bonus = _cfg_float(cfg, "success_bonus_weight", 0.0) * success_now_bool.float()

    # 컵 밀림 soft 감쇠 — "밀어서라도 성공"을 이득에서 제거(우측과 동일 구조).
    _disp_limit = _cfg_float(cfg, "cup_xy_disp_limit", 0.0)
    if _disp_limit > 0.0:
        disp_factor = 1.0 - (cup_xy_displacement / _disp_limit).clamp(0.0, 1.0)
        lift = lift * disp_factor
        success_bonus = success_bonus * disp_factor

    action_smooth = _cfg_float(cfg, "action_smooth_weight", 0.0) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "stabilize": stabilize,
        "success_bonus": success_bonus,
        "post_lift_contact_loss": post_lift_contact_loss,
        "action_smooth": action_smooth,
        "stability": stability,
    }
    missing_terms = set(GRASP_V2_REWARD_TERMS) - set(terms)
    if missing_terms:
        raise RuntimeError(f"missing grasp reward terms: {sorted(missing_terms)}")

    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "both_contact": both,
        "contact_persistence": contact_persistence_frac,
        "opposition": opposition,
        "squeeze": squeeze,
        "upright_success": reward_upright_success.float(),
        "final_upright_success": final_upright_success.float(),
        "success_now": success_now_bool.float(),
        "stability_quality": stability_quality_f,
        "stable": stable_gate,
        "action_quality": action_quality,
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, gates
