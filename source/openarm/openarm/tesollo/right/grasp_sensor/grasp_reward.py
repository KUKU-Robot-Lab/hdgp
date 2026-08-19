"""grasp_sensor 전용 reward — 2026-08-20 전면 재설계 (simple is best).

## 왜 다시 썼나

고정 홈 fresh 학습이 3연속으로 **서로 다른 국소최적**에 빠졌다:
  test1 tilted-palm 2지 파지 / test3 엄지 접촉 포기로 latch 회피 / test4 손끝 파지
원인을 하나씩 고쳤으나(P1 절벽 제거, MCP 동결) 매번 새 국소최적이 나타났다.
실측 진단은 전부 같은 뿌리였다 — **latch 절벽 + 5겹 곱셈 게이트**.

구 설계(8항 / 최대 5겹 곱 / 페널티 5개)의 붕괴 경로(test4 ep6000 실측):
  · `grasp`(12.0)만 게이트가 없어 총보상의 80%, 그 안에서 손끝 항이 95%
    → 손끝 파지가 문자 그대로 최적해였다(wrap 0.036, distal 0.51).
  · `lift`(30)·`success_bonus`(20)는 5겹 곱 + `disp_factor` 하드게이트로 실수령 0.24/0.40.
    `disp_factor = 1-clamp(disp/0.08)` 가 실측 밀림 0.084 에서 **정확히 0**.
  · 곱셈 factor 가 TB 에 하나도 안 나가 어느 factor 가 죽이는지 로그로 분해 불가.

## 설계 원칙

  1. **단계적 상승**: approach(1) → +grasp(3) → +lift(8) → +hold(12).
     누적 최대 1 → 4 → 12 → 24 로 단조 증가. 다음 단계로 가면 이전 보상을 잃지 않는다.
  2. **latch 게이트 0개**. latch 는 제어 트리거(수직 램프)·ADR·로깅 전용이고
     reward 는 오직 물리 상태(거리/접촉/높이/기울기)만 본다.
     "잡고 안 들기" 는 구조적으로 불가능하다 — 파지를 만들면 램프가 컵을 올리고,
     그때 lift+hold(20)가 approach+grasp(4)의 5배로 들어온다.
  3. **곱셈 최대 2겹**. 페널티는 곱이 아니라 덧셈(상한 있음)으로 — 곱하면 하드게이트가 된다.
  4. 모든 곱셈 factor 를 로깅한다(재발 방지).

레퍼런스: `tesollo/right/grasp_v2`(4항·2겹·게이트 0, clean succ 0.723). 그 파일 주석에
grasp_v1 식 8항·5겹·latch 구조를 이식했다가 3271 epoch 성공 0 으로 실패하고 되돌린 기록이 있다.

동기화 규칙: 이 파일은 grasp_sensor 전용이다(grasp_v1 좌/우와 더 이상 미러가 아니다).
"""

from __future__ import annotations

import torch


# grasp_sensor 전용 계약. 공유 GRASP_V2_REWARD_TERMS(8항)는 grasp_v1·v10_3·core 가
# 계속 쓰므로 건드리지 않고, 여기서만 독립 목록을 갖는다.
GRASP_SENSOR_REWARD_TERMS: tuple[str, ...] = (
    "approach",
    "grasp",
    "lift",
    "hold",
    "push_penalty",
    "action_smooth",
)

GRASP_SENSOR_REWARD_FACTORS: tuple[str, ...] = (
    "contact_quality",
    "height_frac",
    "upright_soft",
    "push_norm",
)


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_grasp_reward_terms(
    *,
    tip_contact_frac: torch.Tensor,      # 손끝 접촉 손가락 비율 (N,) ∈[0,1]
    envelope_frac: torch.Tensor,         # 0.5*(중간마디 + 원위마디) 접촉 비율 (N,) ∈[0,1]
    palm_to_cup_dist: torch.Tensor,      # palm → 파지중심 거리 [m]
    fingertip_side_dist: torch.Tensor,   # 엄지가중 손끝-컵 거리 [m]
    cup_height_delta: torch.Tensor,      # 컵 상승량 [m]
    cup_xy_displacement: torch.Tensor,   # 컵 수평 밀림 [m]
    cup_tilt_deg: torch.Tensor,          # 컵 기울기 [deg]
    action_delta_norm: torch.Tensor,     # 액션 변화량 norm
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """4항 + 페널티 2개. 게이트 없음, 곱셈 최대 2겹.

    returns: (total, terms, factors)  — factors 는 TB 로깅용 진단값.
    """

    # ---- 공통 factor (전부 [0,1], 전부 로깅된다) --------------------------------
    # 접촉 품질: envelope(마디 감쌈)이 지배. 손끝만 닿으면 ~0.33, 감싸 쥐면 ~1.5 로 4.5배 격차.
    # 구 설계는 손끝 항이 95% 라 손끝 파지가 최적해였다 — 그 비중을 뒤집는 것이 이 항의 목적.
    _env_credit = _cfg_float(cfg, "contact_envelope_credit", 0.75)
    contact_quality = (
        (1.0 - _env_credit) * tip_contact_frac.clamp(0.0, 1.0)
        + _env_credit * envelope_frac.clamp(0.0, 1.0)
    )
    # 리프트 진행도: 0~ref 구간에 gradient 를 편다(ref=목표 상승량).
    _h_ref = max(1e-6, _cfg_float(cfg, "lift_height_ref", 0.10))
    height_frac = (cup_height_delta / _h_ref).clamp(0.0, 1.0)
    # 정립도: 구 exp(-tilt/5°)는 10° 에서 0.135 로 과가혹해 lift·stabilize·stability 를
    # 일괄 붕괴시켰다. 15° 스케일이면 10° 에서 0.51 로 gradient 가 살아 있다.
    _tilt_scale = max(1e-6, _cfg_float(cfg, "upright_soft_scale_deg", 15.0))
    upright_soft = torch.exp(-cup_tilt_deg.clamp(min=0.0) / _tilt_scale)
    # 컵 밀림 정규화. **곱이 아니라 덧셈 페널티의 입력**이다 — 곱하면 한계에서 0 이 되어
    # gradient 가 소실된다(구 disp_factor 가 정확히 그 실패였다).
    _push_ref = max(1e-6, _cfg_float(cfg, "push_penalty_ref", 0.10))
    push_norm = (cup_xy_displacement / _push_ref).clamp(0.0, 1.0)

    # ---- 단계 1: approach — 손을 컵으로 (1겹) -----------------------------------
    approach = _cfg_float(cfg, "approach_weight", 1.0) * torch.exp(
        -_cfg_float(cfg, "approach_sharpness", 8.0)
        * (palm_to_cup_dist + fingertip_side_dist)
    )

    # ---- 단계 2: grasp — 감싸 쥐기 (1겹) ----------------------------------------
    grasp = _cfg_float(cfg, "grasp_weight", 3.0) * contact_quality

    # ---- 단계 3: lift — 들어올리기 (1겹) ----------------------------------------
    # 접촉 곱을 걸지 않는다: 리프트는 정책이 고르는 행동이 아니라 latch 가 트리거하는
    # 스크립트 램프이고, latch 자체가 접촉을 전제한다. 컵을 놓치면 height_frac 이 즉시
    # 떨어져 물리적 인과가 게이트를 대신한다.
    lift = _cfg_float(cfg, "lift_weight", 8.0) * height_frac

    # ---- 단계 4: hold — 똑바로 든 채 유지 (2겹) ---------------------------------
    # sim2real 흐름의 종착점(들어올려 대기). 가장 큰 보상이라 여기서 수렴하는 것이 목표다.
    hold = _cfg_float(cfg, "hold_weight", 12.0) * height_frac * upright_soft

    # ---- 페널티 (덧셈, 상한 있음) ------------------------------------------------
    push_penalty = -_cfg_float(cfg, "push_penalty_weight", 2.0) * push_norm
    action_smooth = _cfg_float(cfg, "action_smooth_weight", -0.02) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "hold": hold,
        "push_penalty": push_penalty,
        "action_smooth": action_smooth,
    }
    missing = set(GRASP_SENSOR_REWARD_TERMS) - set(terms)
    if missing:
        raise RuntimeError(f"missing grasp_sensor reward terms: {sorted(missing)}")

    factors = {
        "contact_quality": contact_quality,
        "height_frac": height_frac,
        "upright_soft": upright_soft,
        "push_norm": push_norm,
    }

    # 물리 발산 시 리턴 폭주 방어(lstm_test1 iter 14111 전례).
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, factors
