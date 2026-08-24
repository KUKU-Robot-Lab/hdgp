"""tip_cyl 전용 보상 — 08.25 3차 재설계(소프트 계층). 로봇 무관 텐서 함수.

★별도 파일인 이유(공유 세션 분리): `rewards.py` 는 grasp_lift_fabric(타 세션 트랙)이
공유하고, 그쪽 계약 테스트가 파일 전체의 `cfg.X` 키를 자기 cfg 와 대조한다.
tip_cyl 상수(stage_*)는 이 트랙 전용이므로 파일을 가르는 것이 경계다.
공용 소부품(action_l2 계열)만 rewards 에서 가져온다.
"""

from __future__ import annotations

import torch

from .rewards import action_l2_clamped, action_rate_l2_clamped


def compute_tip_cyl_rewards(
    palm_pos: torch.Tensor,           # (N, 3) env-local · 정렬 항의 기준점
    grasp_center_pos: torch.Tensor,   # (N, 3) env-local · palm 부착 파지중심(손가락 무관)
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    pull_dist: torch.Tensor,          # (N, F) 손가락별 "당길 마디"까지 거리(역할별)
    touched: torch.Tensor,            # (N, F) bool · 그 손가락 당김을 끌 조건(역할별)
    wrap_c: torch.Tensor,             # (N, B) bool · 4지 손바닥면 감쌈(mid ∨ dist)
    deep_c: torch.Tensor,             # (N, B) bool · 4지 깊은 감쌈(mid ∧ dist)
    oppose: torch.Tensor,             # (N,)   bool · 엄지 대향(팁 포함 총접촉)
    height_delta: torch.Tensor,       # (N,) 스폰 기준 상승 [m]
    tilt_deg: torch.Tensor,           # (N,) 물체 기울기 [deg]
    xy_disp: torch.Tensor,            # (N,) 스폰 기준 수평 밀림 [m]
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """소프트 계층 — 하드 스위치 없음. 계층은 [0,1] 인자의 **곱셈 깊이**로 만든다.

    ★2차안(이진 대향 게이트)이 lstm_test7 에서 실패한 원인 3건(TFEvents 실측):
      ①`upright` 가 독립 가산항이라 **테이블에 선 컵이 만점**(2.674 = 총보상의 48%).
        들면 흔들려 그걸 잃으므로 보상이 "들지 마라"를 가르쳤다 — h 가 실제로
        2.5mm(ep757) → 0.3mm(ep1324) 로 **되돌아갔다**.
      ②`lift = exp(−8.5·|goal_z−obj_z|)` 가 **h=0 에서 지급**(실측 0.954).
      ③게이트가 손가락 **총접촉**(팁 포함)이라 2지 팁 핀치로 열린다. 수렴 자세는
        파지중심 106mm 밖의 컵을 엄지·검지 팁으로 집은 것이고, 중지·약지·소지는
        1,594 에폭 전 구간 `touch` 자체가 0.000 이었다(컵이 손 안에 온 적 없음).

    ★래치(`pre_lift_gate`)를 쓰지 않는 이유 — 이 저장소가 두 번 제거한 장치다:
      grasp_v2 "순수 grasp_v1 이식 3,271ep, latch 이후 항 전 구간 정확히 0 → 제거",
      lstm_test3 "latch=순손실, 엄지 접촉만 포기하면 영구 차단 → 제거".
      레퍼런스 grasp_v1 의 98% 는 보상이 아니라 **스크립트 리프트**(래치 후 정책 palm
      액션 폐기 + z 120스텝 램프)의 성과이고 우리는 팔이 100% 정책 제어다.
      절벽 산수: 래치는 h=0 에서 일어나 래치 후 수입이 0 인데 래치 전 수입은 전액
      소멸 → 손익분기 높이 0.197m > 목표 0.15m = 회수 불가능한 순손실.

    설계 원칙 4개:
      P1 리프트 계열은 **h=0 에서 정확히 0**. `exp(−k·|Δz|)` 커널 금지.
      P2 직립은 독립 항이 아니라 **곱셈 인자** — 테이블 위 컵에 지급되지 않는다.
      P3 파지 품질 G 는 팁이 아니라 **마디 감쌈 깊이**를 본다 → 핀치가 죽는다.
      P4 형상 비의존 — 물체 pose·링크 위치·접촉력만 쓴다. CAD·반지름·높이 미사용.
         조임 깊이도 지정하지 않는다(접촉 시 당김 소등 + 상실 위험이 정한다).

    returns (total, terms, grip_ok, wrap4) — 구 함수와 동일 4-tuple 계약(env 배선 호환).
    """
    # ---- 소프트 인자 (전부 [0,1] 연속) ------------------------------------------
    wrap4 = wrap_c.float().mean(dim=-1)          # 4지 감쌈 비율 — 상한 1.0(엄지 제외)
    deep4 = deep_c.float().mean(dim=-1)          # 4지 깊은 감쌈 비율
    opp_f = oppose.float()

    # 파지 품질 G — 얕은 감쌈보다 깊은 감쌈에 무게. deep 몫에 wrap4 를 곱해
    # "한 손가락만 깊게"가 만점이 되지 않게 한다.
    # deep_c ⊆ wrap_c (mid∧dist ⇒ mid∨dist) 이므로 deep4 ≤ wrap4 가 자동 보장된다 —
    # deep 몫에 wrap4 를 다시 곱할 필요가 없다.
    _fl = float(cfg.stage_gq_thumb_floor)
    G = (_fl + (1.0 - _fl) * opp_f) * (
        float(cfg.stage_gq_wrap) * wrap4 + float(cfg.stage_gq_deep) * deep4)

    H = (height_delta / float(cfg.stage_lift_height_ref)).clamp(0.0, 1.0)   # h=0 → 0
    U = torch.exp(-tilt_deg / float(cfg.stage_upright_tau_deg))
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    T = 1.0 - torch.tanh(d_goal / float(cfg.stage_tracking_std))
    _dn = torch.norm(actions - prev_actions, dim=-1) / (actions.shape[-1] ** 0.5)
    S = torch.exp(-float(cfg.stage_stabilize_sharpness) * _dn)
    _r = xy_disp / float(cfg.stage_disp_limit)
    F = 1.0 / (1.0 + _r * _r)                    # 제곱역수 — 0 에 닿지 않아 gradient 유지

    # ---- ① 접근(팔) — 무게이트 ---------------------------------------------------
    d_gc = torch.norm(grasp_center_pos - object_pos, dim=-1)
    align = torch.nn.functional.cosine_similarity(
        grasp_center_pos - palm_pos, object_pos - palm_pos, dim=-1, eps=1e-6)
    _al = float(cfg.stage_align_floor)
    approach = (torch.exp(-float(cfg.stage_approach_sharpness) * d_gc)
                * (_al + (1.0 - _al) * 0.5 * (1.0 + align)))

    # ---- ② 파지(손가락) — 무게이트. reach 는 접촉 전 shaping, G 는 접촉 후 품질 ----
    reach = ((~touched).float() * torch.exp(
        -float(cfg.stage_grip_sharpness)
        * pull_dist.clamp(min=float(cfg.stage_grip_dist_floor)))).mean(dim=-1)
    grasp = float(cfg.stage_graspq_reach) * reach + float(cfg.stage_graspq_g) * G

    # ---- ③④⑤ 소프트 곱셈 계층 (인자 3 → 4 → 5) ----------------------------------
    lift = G * H * U * F
    transport = G * H * U * T
    stabilize = G * H * U * T * S

    # ---- ⑥ 성공(이진 보너스) -----------------------------------------------------
    success_now = (
        (height_delta >= float(cfg.stage_success_height))
        & (wrap4 >= float(cfg.stage_success_wrap_min))
        & oppose
        & (tilt_deg <= float(cfg.stage_success_tilt_deg))
        & (d_goal <= float(cfg.stage_success_pos_tol))
    )

    terms = {
        "approach": float(cfg.stage_approach_weight) * approach,
        "grasp": float(cfg.stage_grasp_weight) * grasp,
        "lift": float(cfg.stage_lift_weight) * lift,
        "transport": float(cfg.stage_transport_weight) * transport,
        "stabilize": float(cfg.stage_stabilize_weight) * stabilize,
        "success": float(cfg.stage_success_weight) * success_now.float() * F,
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)

    # 진단용(보상 아님) — env 가 로깅에서 pop 한다.
    terms["_d_gc"] = d_gc
    terms["_align"] = align
    terms["_G"] = G
    terms["_H"] = H
    terms["_U"] = U
    terms["_deep4"] = deep4
    terms["_reach"] = reach
    # 3번째 반환값은 구 `gate` 자리 — 로깅 호환을 위해 "쓸만한 파지" 이진값을 싣는다.
    grip_ok = G > 0.5
    return total, terms, grip_ok, wrap4
