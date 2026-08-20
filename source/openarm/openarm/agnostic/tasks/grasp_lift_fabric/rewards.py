"""grasp_lift_fabric 보상 — 7항.

**항 범주는 grasp_v2(`common/grasp_reward_core.py`)에서, 게이트 구조는 새로 짰다.**

왜 grasp_lift(diffIK 트랙)의 5항을 쓰지 않는가:
    그쪽 success 는 "물체가 goal 위치에 있으면 끝"이라 **파지 품질·자세·유지에
    gradient 가 하나도 없다**(보상에 회전 항이 아예 없다). 그래서 엄지가 컵 입구를
    가로지르는 평면 핀치 같은 축퇴를 구조적으로 허용한다.

왜 grasp_v2 구현을 그대로 쓰지 않는가 (실측 3건):
    ① `lift_latched` 래치가 항을 pre_lift/lift 로 이분한다. 래치가 걸리면
       `grasp`(가중 12.0)가 즉시 꺼져 기여가 0.0362 → 0.0358 로 **평탄**했다.
       "만렙 후 감쌈 침식(중간마디 상실)"의 직접 원인이다. → **래치를 쓰지 않는다.**
    ② `stabilize` 는 게이트 5중첩(lift × lifted × contact × upright × action)이라
       하나가 0이면 전부 0 → 어느 인자가 막았는지 분리 불가(기여 0.965 = 품질 0.097).
       → **곱셈은 최대 2단**, 자세는 0 이 아니라 **0.5~1.0 완화 곱수**로 넣는다.
    ③ 컵 밀기 페널티가 `approach` 안에 묻혀 approach 전체 기여가 1.6% 였고,
       밀기 0.375 vs 성공 4.572 = **1:12** 라 밀어서 성공하는 게 최적이었다.
       → **독립 항으로 분리하고 따로 로깅**한다.

reward-audit 판정 이력 (REVISE → ACCEPT):
    · `upright` 독립 항은 컵이 애초에 서 있어 사실상 공짜였다(접촉 보너스 중복)
      → 독립 항 제거, lift/success 의 완화 곱수로만 사용 + tilt_penalty 신설.
    · penalty 가 무한대라 멀리 밀린 뒤 회복 불가 → **상한 클램프**(접근 회피 방지,
      agn_test2 의 종료-회피와 같은 실패 축).
    · `persistence` 는 래치를 없앤 뒤 중복 → 제거(envelope 0.6 / grip 0.4).

정체(테이블 위 감쌈 유지) 4.0 : 추가(리프트+성공) 12.0 = 1:3.0.
"""

from __future__ import annotations

import torch


def _cfg(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def contact_gate(
    group_a_force: torch.Tensor, group_b_force: torch.Tensor, threshold: float
) -> torch.Tensor:
    """대향 접촉 = 파지 성립. 5지면 엄지 AND 나머지, 2지면 조1 AND 조2 — 같은 코드."""
    a = (group_a_force > threshold).any(dim=-1)
    b = (group_b_force > threshold).any(dim=-1)
    return a & b


def upright_quality(object_tilt_deg: torch.Tensor, max_deg: float) -> torch.Tensor:
    """1(수직) → 0(max_deg 이상 기움)."""
    return (1.0 - object_tilt_deg / max(max_deg, 1e-6)).clamp(0.0, 1.0)


def compute_rewards(
    *,
    palm_to_object: torch.Tensor,     # (N,)  palm ↔ 물체 거리
    tip_side_dist: torch.Tensor,      # (N,)  손끝 평균 ↔ 물체 거리
    envelope_frac: torch.Tensor,      # (N,)  감쌈(중간·원위 마디) 접촉 손가락 비율
    grip_frac: torch.Tensor,          # (N,)  아무 마디든 접촉한 손가락 비율
    object_height_delta: torch.Tensor,
    object_to_goal: torch.Tensor,
    object_xy_displacement: torch.Tensor,
    object_tilt_deg: torch.Tensor,
    group_a_force: torch.Tensor,
    group_b_force: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """Returns: (total, terms, gate, upright_q)."""
    gate = contact_gate(
        group_a_force, group_b_force, _cfg(cfg, "contact_force_threshold", 1.0)
    )
    gate_f = gate.float()
    up_q = upright_quality(object_tilt_deg, _cfg(cfg, "upright_max_deg", 30.0))
    # ★자세는 곱수를 0 으로 떨어뜨리지 않는다. 0.5~1.0 완화 — grasp_v2 의
    #   "인자 하나가 0 이면 항 전체가 0" 붕괴를 구조적으로 막는다.
    up_mul = 0.5 + 0.5 * up_q

    # 1. approach — 게이트 없음(접촉 전 유일한 gradient)
    approach = _cfg(cfg, "approach_weight", 1.0) * torch.exp(
        -_cfg(cfg, "approach_sharpness", 4.0) * (palm_to_object + tip_side_dist)
    )

    # 2. grasp_quality — ★래치 없이 **상시** 켜져 있다. 리프트 후 감쌈이 풀리면
    #    바로 깎이므로 "만렙 후 감쌈 침식"에 반대 gradient 가 생긴다.
    _env_credit = _cfg(cfg, "grasp_envelope_credit", 0.6)
    _grip_credit = _cfg(cfg, "grasp_grip_credit", 0.4)
    grasp_quality = (
        _cfg(cfg, "grasp_quality_weight", 3.0)
        * gate_f
        * (_env_credit * envelope_frac.clamp(0.0, 1.0)
           + _grip_credit * grip_frac.clamp(0.0, 1.0))
    )

    # 3. lift
    height_q = (
        object_height_delta / max(_cfg(cfg, "lift_success_height", 0.10), 1e-6)
    ).clamp(0.0, 1.0)
    lift = _cfg(cfg, "lift_weight", 2.0) * gate_f * height_q * up_mul

    # 4. success
    success = (
        _cfg(cfg, "success_weight", 10.0)
        * (1.0 - torch.tanh(object_to_goal / _cfg(cfg, "success_pos_std", 0.05))) ** 2
        * gate_f
        * up_mul
    )

    # 5. push_penalty — ★상한 클램프. 무한대로 두면 멀리 밀린 뒤 회복 불가라
    #    "컵에 아예 안 다가감"이 국소최적이 된다(agn_test2 종료-회피와 같은 축).
    _push_cap = _cfg(cfg, "push_penalty_cap", 0.10)
    push_penalty = _cfg(cfg, "push_penalty_weight", -2.0) * (
        (object_xy_displacement - _cfg(cfg, "push_margin", 0.025))
        .clamp(min=0.0, max=_push_cap)
    )

    # 6. tilt_penalty — push 와 같은 형태(물체를 교란하지 말 것). 마진 안은 면제.
    _tilt_margin = _cfg(cfg, "tilt_margin_deg", 10.0)
    _tilt_cap = _cfg(cfg, "tilt_penalty_cap_deg", 30.0)
    tilt_penalty = _cfg(cfg, "tilt_penalty_weight", -1.0) * (
        (object_tilt_deg - _tilt_margin).clamp(min=0.0, max=_tilt_cap) / _tilt_cap
    )

    # 7. action_rate — ★**mean** 이고 clamp 이 없다. 둘 다 실측 근거가 있다.
    #   · sum 이면 액션 차원에 비례한다(같은 지터에서 26D=12.0 / 18D=8.3 / 7D=3.2).
    #     robot-agnostic 트랙에서 같은 weight 가 로봇마다 다른 압력이 되므로 mean 을 쓴다.
    #   · clamp(max=1.0) 은 실측 sum 12.0 에서 **12배 포화**해 페널티를 상수로 만들었다.
    #     지터가 12→1 로 줄 때까지 gradient 가 0 이었다(fab_test2 iter15 실측).
    #     grasp_v2 의 래치가 grasp 항을 죽인 것과 같은 부류의 결함이다.
    #   mean((Δa)²) 는 자연 상한 4.0(매 스텝 완전 반전) 이라 clamp 가 필요 없다.
    action_rate = _cfg(cfg, "action_rate_weight", -0.3) * torch.mean(
        (actions - prev_actions) ** 2, dim=-1
    )

    terms = {
        "approach": approach,
        "grasp_quality": grasp_quality,
        "lift": lift,
        "success": success,
        "push_penalty": push_penalty,
        "tilt_penalty": tilt_penalty,
        "action_rate": action_rate,
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, gate, up_q
