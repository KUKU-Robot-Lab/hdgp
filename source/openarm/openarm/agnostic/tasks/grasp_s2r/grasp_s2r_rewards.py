"""grasp_s2r 보상 — grasp_v1 8항 이식 + 이송 2항(transfer·stay) 신설.

`tesollo/right/grasp_v1/grasp_reward.py` 의 구조·근거를 그대로 가져오되 두 곳이 다르다.

1. **이송 2항 신설.** grasp_v1 은 제자리 리프트가 종점이었다. 여기서는 목표 지점까지
   옮기고 멈추는 것이 과제라 `transfer`(목표 접근)·`stay`(도달 후 유지)를 더한다.
   둘 다 `graded_contact` 를 곱해 **접촉 없이는 0** 이다(파지 없이 밀어 옮기기 차단).

2. **컵 밀림 감쇠 기준을 래치 시점 스냅샷으로.** grasp_v1 의 `disp_factor` 는 스폰점
   대비 **실시간** 수평 변위로 lift·success 를 깎았다. 밀어서 성공하는 경로를 막는
   장치였는데, 이 트랙은 수평 이송이 **과제 자체**라 그대로 쓰면 의도된 이송을
   처벌한다. 래치(파지 성립) 시점의 변위만 기준으로 삼으면 "접근 중 밀지 마라"는
   원래 의도는 유지되고 이송은 자유롭다.

항 계약(`GRASP_S2R_REWARD_TERMS`)은 **이 트랙 로컬**이다 —
`openarm.common.grasp_v2_contract.GRASP_V2_REWARD_TERMS` 는 8항 고정이고 여러 트랙이
공유하므로 건드리지 않는다.
"""

from __future__ import annotations

import torch

# 이 트랙의 보상 항 계약. 순서는 로깅 순서이기도 하다.
GRASP_S2R_REWARD_TERMS: tuple[str, ...] = (
    "approach",
    "grasp",
    "lift",
    "transfer",
    "stay",
    "stabilize",
    "stability",
    "success_bonus",
    "post_lift_contact_loss",
    "action_smooth",
)


def _f(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_grasp_s2r_rewards(
    *,
    # ---- 접촉 ----
    tip_contact_frac: torch.Tensor,       # (N,) 팁 접촉 손가락 비율
    full_tip_contact: torch.Tensor,       # (N,) bool — 전 팁 접촉
    contact_persistence_frac: torch.Tensor,
    wrap_frac: torch.Tensor,              # (N,) per-finger (middle AND distal) 비율 = 감쌈 **깊이**
    wrap_at_latch: torch.Tensor,          # (N,) 래치 시점 깊이 스냅샷
    grip_frac: torch.Tensor,              # (N,) tip|mid|distal OR 비율
    # ---- 기하 ----
    palm_to_cup_dist: torch.Tensor,
    fingertip_side_dist: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_xy_disp_now: torch.Tensor,        # 접근 벌점용 — 실시간 수평 변위
    cup_xy_disp_ref: torch.Tensor,        # 감쇠용 — **래치 시점** 변위 스냅샷
    cup_tilt_deg: torch.Tensor,
    goal_dist: torch.Tensor,              # ‖obj − goal‖
    # ---- 상태 ----
    upright_quality: torch.Tensor,
    lift_latched: torch.Tensor,           # (N,) bool — 파지 성립(보상 단계 표시 전용)
    stay_frac: torch.Tensor,              # (N,) 목표 유지 연속시간 / stay_hold_steps, [0,1]
    stable: torch.Tensor,                 # (N,) bool
    stability_quality: torch.Tensor,
    success_now: torch.Tensor,            # (N,) bool
    action_delta_norm: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(total, terms, gates) 반환. 전부 (N,) 텐서."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    persistence = contact_persistence_frac.clamp(0.0, 1.0)

    # 접촉 품질 — 하드 게이트(전 팁 접촉)는 검지가 구조적으로 미접촉이라 영원히 0 이
    # 된다. 부분 접촉에도 gradient 가 남도록 graded 로 간다.
    # 여기에 감쌈 비중을 섞어 **손끝만으로 드는 것**을 부드럽게 억제한다.
    _emix = _f(cfg, "lift_envelope_mix", 0.6)
    graded_contact = (1.0 - _emix) * tip_contact_frac.clamp(0.0, 1.0) \
        + _emix * wrap_frac.clamp(0.0, 1.0)

    lifted_gate = (cup_height_delta >= _f(cfg, "lift_success_height", 0.04)).float()
    at_goal = (goal_dist <= _f(cfg, "goal_pos_tolerance", 0.025)).float()
    stable_gate = stable.float()

    # ---- approach : 래치 전에만 -----------------------------------------------------
    # ★★08.27 벌금에 **상한**을 씌웠다(상한 = approach_weight). 씌우기 전 실측
    #   (s2r_a1): 접촉이 시작되자(touch_frac 0→0.014) grasp 가 +0.43 오르는 동시에
    #   approach 가 −0.96 → −2.02 로 커져 **순증분이 음수**였다 — 컵에 닿을수록
    #   손해라 접촉 탐색 자체가 금지됐고, 정책은 16스텝 만에 컵을 기울여 끝내는
    #   자살 경로에 240 iter 고착했다(스텝당 보상이 순음수면 조기 종료가 최적).
    #   상한을 씌우면 approach 항의 최솟값이 0 이라 "물러서기"가 최적이 아니게 되고,
    #   grasp(12.0)·transfer(15.0) 로 가는 길만 상승 방향으로 남는다.
    # ★밀림 억제 자체는 죽지 않는다 — lift·transfer·success 에 곱해지는
    #   `disp_factor`(래치 시점 스냅샷)가 별도로 계속 감쇠한다.
    _aw = _f(cfg, "approach_weight", 2.0)
    _penalty = (
        _f(cfg, "cup_disp_penalty", 25.0)
        * torch.relu(cup_xy_disp_now - _f(cfg, "cup_disp_tolerance", 0.025))
        + _f(cfg, "cup_tilt_penalty", 0.08)
        * torch.relu(cup_tilt_deg - _f(cfg, "cup_tilt_free_deg", 8.0))
    ).clamp(max=_aw)
    approach = pre_lift_gate * (
        _aw * torch.exp(
            -_f(cfg, "approach_sharpness", 8.0)
            * (palm_to_cup_dist + fingertip_side_dist))
        - _penalty
    )

    # ---- grasp : 래치 전에만. credit 이 참조하는 감쌈은 **깊이**(wrap_frac) ----------
    # 합이 1 로 재정규화되므로 credit 을 올려도 grasp 최대치는 불변 —
    # "감쌈만 하고 안 드는" 국소최적을 구조적으로 못 만든다.
    _ecred = _f(cfg, "grasp_envelope_credit", 0.55)
    _tip_scale = (1.0 - _ecred) / 0.60
    grasp_quality = (
        0.15 * _tip_scale * tip_contact_frac.clamp(0.0, 1.0)
        + 0.20 * _tip_scale * full_tip
        + 0.25 * _tip_scale * persistence
        + _ecred * wrap_frac.clamp(0.0, 1.0)
    )
    grasp = _f(cfg, "grasp_weight", 12.0) * pre_lift_gate * grasp_quality

    # ---- lift : 높이 정규화 기준은 성공 임계와 분리(손 미끄러짐 구분) ----------------
    _h_ref = max(_f(cfg, "lift_height_ref", 0.10), 1e-6)
    lift_height_quality = (cup_height_delta / _h_ref).clamp(min=0.0, max=1.0)
    lift = (
        _f(cfg, "lift_weight", 30.0)
        * lift_gate * graded_contact * lift_height_quality * upright_quality
    )

    # ---- transfer(신설) : 들린 상태에서 목표까지 좁히기 ------------------------------
    # ★`graded_contact` 를 곱해 접촉 없이는 0 — 파지 없이 밀어 옮기는 경로를 차단한다.
    # ★`lifted_gate` 가 있어 테이블 위로 끌고 가는 것도 보상되지 않는다.
    transfer = (
        _f(cfg, "transfer_weight", 15.0)
        * lift_gate * lifted_gate * graded_contact * upright_quality
        * torch.exp(-_f(cfg, "transfer_sharpness", 6.0) * goal_dist)
    )

    # ---- stay(신설) : 목표에서 **머무르기** ------------------------------------------
    # ★도달 순간이 아니라 **연속 유지 시간**에 비례한다(stay_frac). 도달만 반복하는
    #   "찍고 빠지기"로는 최대치를 못 받는다. 정지(stable)와 파지 유지가 동시 조건.
    stay = (
        _f(cfg, "stay_weight", 8.0)
        * at_goal * stable_gate * graded_contact * stay_frac.clamp(0.0, 1.0)
    )

    # ---- stabilize / stability -------------------------------------------------------
    action_quality = torch.exp(-1.5 * action_delta_norm)
    stabilize = (
        _f(cfg, "stabilize_weight", 10.0)
        * lift_gate * lifted_gate * graded_contact * upright_quality * action_quality
    )
    stability = (
        _f(cfg, "stability_weight", 1.0)
        * lift_gate * lifted_gate * graded_contact * upright_quality
        * stability_quality.clamp(0.0, 1.0)
    )

    # ---- 파지 상실 벌점 ---------------------------------------------------------------
    # grip_frac(tip|mid|distal OR)은 중간마디를 잃고 손끝으로 미끄러져도 비용이 0 이다.
    # 그래서 **래치 대비 감쌈 감소분**을 따로 처벌한다 — 유지하면 정확히 0, 잃을 때만 비용.
    # 절대 깊이를 처벌하면 기준선이 통째로 내려가 리프트를 억제한다(grasp_v1 Check4 근거).
    # ⚠잔여 위험: 얕게 래치하면 잃을 게 없어 0 인 회피 경로. 래치 전 grasp credit 이
    #   반대 압력을 주고, `wrap_at_latch` 로 감시한다.
    post_lift_contact_loss = (
        _f(cfg, "post_lift_contact_loss_weight", -8.0)
        * lift_gate * lifted_gate
        * torch.relu(1.0 - grip_frac.clamp(0.0, 1.0))
    ) + (
        _f(cfg, "wrap_retention_weight", -6.0)
        * lift_gate * lifted_gate
        * torch.relu(wrap_at_latch.clamp(0.0, 1.0) - wrap_frac.clamp(0.0, 1.0))
    )

    success_bonus = _f(cfg, "success_weight", 20.0) * success_now.float()

    # ---- 컵 밀림 감쇠 — **래치 시점** 변위 기준 ---------------------------------------
    # ★실시간 변위를 쓰면 의도된 수평 이송이 통째로 처벌된다(이 트랙의 과제가 이송이다).
    #   래치 시점 스냅샷을 쓰면 "접근 중 밀지 마라"는 원래 의도만 남는다.
    # ★하드 게이트가 아니라 제곱역수 감쇠인 이유: 선형(1−d/L)은 d≥L 에서 정확히 0 이
    #   되어 gradient 가 사라진다(실측: 밀림이 300 epoch 간 전혀 안 줄었다).
    #   제곱역수는 0 에 닿지 않아 어떤 밀림에서도 "덜 밀면 더 받는" 단조 신호가 남는다.
    _limit = _f(cfg, "disp_falloff", 0.16)
    if _limit > 0.0:
        _r = cup_xy_disp_ref / _limit
        disp_factor = 1.0 / (1.0 + _r * _r)
        lift = lift * disp_factor
        transfer = transfer * disp_factor
        success_bonus = success_bonus * disp_factor
    else:
        disp_factor = torch.ones_like(goal_dist)

    action_smooth = _f(cfg, "action_smooth_weight", -0.02) * action_delta_norm

    terms = {
        "approach": approach,
        "grasp": grasp,
        "lift": lift,
        "transfer": transfer,
        "stay": stay,
        "stabilize": stabilize,
        "stability": stability,
        "success_bonus": success_bonus,
        "post_lift_contact_loss": post_lift_contact_loss,
        "action_smooth": action_smooth,
    }
    _missing = set(GRASP_S2R_REWARD_TERMS) - set(terms)
    if _missing:
        raise RuntimeError(f"보상 항 누락: {sorted(_missing)}")

    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "at_goal": at_goal,
        "stable": stable_gate,
        "graded_contact": graded_contact,
        "full_tip_contact": full_tip,
        "contact_persistence": persistence,
        "action_quality": action_quality,
        "disp_factor": disp_factor,
        "success_now": success_now.float(),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, gates
