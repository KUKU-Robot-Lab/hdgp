"""pour_fabric 보상 — 9항.

grasp_lift_fabric 의 보상 규약을 그대로 잇는다:
  · **래치 금지** — 모든 게이트는 매 스텝 현재 상태로 재평가된다(감쌈이 풀리면
    즉시 깎여 반대 gradient 가 생긴다). pour_v1 의 ready_latch 는 계승하지 않는다.
  · 곱셈 게이트 ≤ 2단, 완화 곱수(0.5~1.0)는 0 으로 떨어지지 않는다.
  · 페널티는 **상한 클램프**(무한대면 회복 불가 상태가 회피 국소최적을 만든다).
  · action_rate 는 mean(차원 불변 — robot-agnostic 트랙에서 sum 금지).
  · 상태 보상(레벨) 대신 **증분(Δ) 보상** — pour_v1 이 weight_bead_in=0 으로
    끈 계보: 레벨 보상은 "채워둔 채 가만히" farming 을 만든다.

pour_v1 에서 계승하지 않는 것(1차 포팅 제외 — 근거는 CLAUDE.md/플랜):
    demo_pose_reward · R(β) nullspace · deep_tilt_boot · corridor escape penalty.
"""

from __future__ import annotations

import torch


def _cfg(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def contact_gate(group_a_force: torch.Tensor, group_b_force: torch.Tensor,
                 threshold: float) -> torch.Tensor:
    """대향 접촉 = 파지 성립 (grasp_lift_fabric 과 동일 함수)."""
    a = (group_a_force > threshold).any(dim=-1)
    b = (group_b_force > threshold).any(dim=-1)
    return a & b


def compute_rewards(
    *,
    # ---- 파지 유지 (양손) ------------------------------------------------------
    src_envelope_frac: torch.Tensor,   # (N,) source 손 감쌈 비율
    src_grip_frac: torch.Tensor,
    src_group_a_force: torch.Tensor,   # (N,Fa)
    src_group_b_force: torch.Tensor,
    rcv_envelope_frac: torch.Tensor,
    rcv_grip_frac: torch.Tensor,
    rcv_group_a_force: torch.Tensor,
    rcv_group_b_force: torch.Tensor,
    # ---- 조준/기울임 ------------------------------------------------------------
    aim_dist: torch.Tensor,            # (N,) source 주둥이 ↔ target 개구 거리
    tilt_deg: torch.Tensor,            # (N,) source 컵 기울기 [deg]
    pour_dir_cos: torch.Tensor,        # (N,) 기울임 방향 ↔ target 방향 cos(xy)
    # ---- 비드 증분 ---------------------------------------------------------------
    d_in_target: torch.Tensor,         # (N,) Δ bead_in_target_frac
    d_released: torch.Tensor,          # (N,) Δ (1 - bead_in_source_frac)
    d_spill: torch.Tensor,             # (N,) Δ spill_frac
    # ---- 이산 사건 ---------------------------------------------------------------
    success_now: torch.Tensor,         # (N,) bool
    dropped_now: torch.Tensor,         # (N,) bool — 이번 스텝에 낙하 판정(일회)
    # ---- 액션 --------------------------------------------------------------------
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict, dict]:
    """Returns: (total, terms, gates)."""
    thr = _cfg(cfg, "contact_force_threshold", 1.0)
    gate_src = contact_gate(src_group_a_force, src_group_b_force, thr)
    gate_rcv = contact_gate(rcv_group_a_force, rcv_group_b_force, thr)

    # 1/2. hold — 래치 없이 상시. 감쌈이 풀리면 즉시 깎인다(침식 반대 gradient).
    _env_c = _cfg(cfg, "hold_envelope_credit", 0.6)
    _grip_c = _cfg(cfg, "hold_grip_credit", 0.4)

    def _hold(w: float, gate: torch.Tensor, env_f: torch.Tensor,
              grip_f: torch.Tensor) -> torch.Tensor:
        return w * gate.float() * (
            _env_c * env_f.clamp(0.0, 1.0) + _grip_c * grip_f.clamp(0.0, 1.0))

    hold_source = _hold(_cfg(cfg, "hold_source_weight", 2.0),
                        gate_src, src_envelope_frac, src_grip_frac)
    hold_receiver = _hold(_cfg(cfg, "hold_receiver_weight", 1.0),
                          gate_rcv, rcv_envelope_frac, rcv_grip_frac)

    # 3. aim — 게이트 없음(정렬 전 유일한 dense gradient).
    #    pour_v1 06.18 "corridor penalty → 순수 positive pull" 결정 계승.
    aim = _cfg(cfg, "aim_weight", 1.5) * torch.exp(
        -_cfg(cfg, "aim_sharpness", 4.0) * aim_dist)

    # 4. tilt — 근접(연속 게이트, 래치 아님) × 진행 × 방향 완화 곱수.
    #    ★근접 게이트가 없으면 "빈 데서 기울이기" farming 이 열린다.
    #    방향(내회전)은 0 으로 떨어지지 않는 0.5~1.0 완화 — grasp up_mul 과 동형.
    prox = torch.exp(-aim_dist / max(_cfg(cfg, "tilt_prox_std", 0.10), 1e-6))
    tilt_prog = (tilt_deg / max(_cfg(cfg, "tilt_target_deg", 110.0), 1e-6)).clamp(0.0, 1.0)
    dir_mul = 0.5 + 0.5 * pour_dir_cos.clamp(0.0, 1.0)
    tilt = _cfg(cfg, "tilt_weight", 2.0) * prox * tilt_prog * dir_mul

    # 5. pour_delta — 증분만 보상(레벨 금지). relu 라 되돌아가도 음수 없음(그건 spill 담당).
    #    ★release 항 기본 0 — spill 페널티가 작은 초기엔 "바닥에 붓기"를 보상하게 된다.
    #      capture 가 안 늘 때만 부트스트랩으로 켠다.
    pour_delta = (
        _cfg(cfg, "pour_capture_weight", 25.0) * d_in_target.clamp(min=0.0)
        + _cfg(cfg, "pour_release_weight", 0.0) * d_released.clamp(min=0.0)
    )

    # 6. success — 지표형 보상(판정은 env 가 계산: fill ∧ spill ∧ xy align).
    success = _cfg(cfg, "success_weight", 10.0) * success_now.float()

    # 7. spill — 증분 페널티, 스텝당 상한(전량 손실=1.0 이 자연 상한이지만 명시).
    _spill_cap = _cfg(cfg, "spill_step_cap", 0.5)
    spill_penalty = _cfg(cfg, "spill_weight", -2.0) * d_spill.clamp(
        min=0.0, max=_spill_cap)

    # 8. drop — 일회 페널티(env 가 낙하 판정 + 종료를 담당).
    drop_penalty = _cfg(cfg, "drop_penalty_weight", -5.0) * dropped_now.float()

    # 9. action_rate — mean·무클램프 (grasp_lift_fabric 실측 근거 그대로:
    #    sum 은 차원 비례, clamp 는 12배 포화로 gradient 0).
    action_rate = _cfg(cfg, "action_rate_weight", -0.3) * torch.mean(
        (actions - prev_actions) ** 2, dim=-1)

    terms = {
        "hold_source": hold_source,
        "hold_receiver": hold_receiver,
        "aim": aim,
        "tilt": tilt,
        "pour_delta": pour_delta,
        "success": success,
        "spill_penalty": spill_penalty,
        "drop_penalty": drop_penalty,
        "action_rate": action_rate,
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    gates = {"gate_src": gate_src, "gate_rcv": gate_rcv,
             "prox": prox, "tilt_prog": tilt_prog}
    return total, terms, gates
