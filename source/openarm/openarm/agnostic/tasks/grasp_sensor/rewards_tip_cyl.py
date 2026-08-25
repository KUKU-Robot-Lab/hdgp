"""tip_cyl 전용 보상 — 08.25 3차 재설계(소프트 계층). 로봇 무관 텐서 함수.

★별도 파일인 이유(공유 세션 분리): `rewards.py` 는 grasp_lift_fabric(타 세션 트랙)이
공유하고, 그쪽 계약 테스트가 파일 전체의 `cfg.X` 키를 자기 cfg 와 대조한다.
tip_cyl 상수(stage_*)는 이 트랙 전용이므로 파일을 가르는 것이 경계다.
공용 소부품(action_l2 계열)만 rewards 에서 가져온다.
"""

from __future__ import annotations

import math

import torch

from .rewards import action_l2_clamped, action_rate_l2_clamped


def smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[0,1] 로 부드럽게 올라가는 전이. `hi < lo` 면 내려가는 전이가 된다.

    게이트 전용. `lo` 아래에서 정확히 0, `hi` 위에서 정확히 **1.0 에 포화**하고
    사이는 3차 다항으로 잇는다(양 끝 미분 0). 계층 보상에서 게이트가 포화해야
    깊은 단계가 앞 단계에 의해 깎이지 않는다.
    """
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_tip_cyl_rewards(
    palm_pos: torch.Tensor,           # (N, 3) env-local · 정렬 항의 기준점
    grasp_center_pos: torch.Tensor,   # (N, 3) env-local · palm 부착 파지중심(손가락 무관)
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    tip_c: torch.Tensor,              # (N, F) bool · 손가락별 팁 접촉(엄지 포함 5지)
    persist_frac: torch.Tensor,       # (N,) [0,1] · 접촉 지속 스텝 / 정규화 스텝수
    wrap_c: torch.Tensor,             # (N, B) bool · 4지 손바닥면 감쌈(mid ∨ dist)
    deep_c: torch.Tensor,             # (N, B) bool · 4지 깊은 감쌈(mid ∧ dist)
    oppose: torch.Tensor,             # (N,)   bool · 엄지 대향(팁 포함 총접촉)
    height_delta: torch.Tensor,       # (N,) 스폰 기준 상승 [m]
    tilt_deg: torch.Tensor,           # (N,) 물체 기울기 [deg]
    xy_disp: torch.Tensor,            # (N,) 스폰 기준 수평 밀림 [m]
    touch_c: torch.Tensor,            # (N,C) bool · **가용 손가락**의 접촉(mid∨dist∨tip)
    thumb_force: torch.Tensor,        # (N,) 엄지 총접촉력 [N] — 소프트 대향용
    palm_x: torch.Tensor,             # (N,3) palm_ee +x = **손바닥 법선**(실측 확정)
    palm_y: torch.Tensor,             # (N,3) palm_ee +y — 롤 잠금용
    ref_up: torch.Tensor,             # (N,3) **로봇 베이스 +z** — 자세 항의 기준축
    obj_up: torch.Tensor,             # (N,3) 물체 +z(컵 축). 인식 pose 에서 나온다
    obj_speed: torch.Tensor,          # (N,) 물체 선속도 크기 [m/s] — stay 판정
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
    # ---- 접촉 지표 (전부 [0,1] 연속) --------------------------------------------
    wrap4 = wrap_c.float().mean(dim=-1)          # 4지 감쌈 비율 — 상한 1.0(엄지 제외)
    deep4 = deep_c.float().mean(dim=-1)          # 4지 깊은 감쌈(mid ∧ dist)
    tip_frac = tip_c.float().mean(dim=-1)        # 5지 팁 접촉 비율(엄지 포함)
    full_tip = tip_c.all(dim=-1).float()
    # ★grasp_v1 `envelope_frac` = 0.5(mid_frac + dist_frac). 항등식
    #   mean(mid)+mean(dist) = mean(mid∨dist)+mean(mid∧dist) 로 우리 지표와 정확히 같다.
    envelope = 0.5 * (wrap4 + deep4)

    # 리프트 계열 접촉 게이팅 — grasp_v1 `graded_contact`.
    # ★구 G 의 엄지 배수 (0.25 + 0.75·oppose) 는 **삭제**했다. 그 배수 때문에 첫 접촉이
    #   순손실이었다(reach 소등 −0.272 vs G 상승 +0.18 = −0.09/step). 엄지는 이제
    #   tip_frac(5팁) 안에서 계상되고, 성공 판정만 oppose 를 그대로 요구한다.
    # ★★08.25 5단계 재편. 사용자 지정 순서:
    #     approach → grasp(5지 + palm 밀착) → lift → transfer&stabilize → stay
    #   계층은 [0,1] 인자의 **곱셈 깊이**로 만든다(하드 스위치·래치 없음).
    d_gc = torch.norm(grasp_center_pos - object_pos, dim=-1)

    # ── ② 파지 품질 Q_g — **접촉 기하로만 잰다** ───────────────────────────────
    # ★★08.25 구 `G = five_frac · exp(−d_gc/τ)` 폐기. 실측 근거(lstm_test14 ep1102 정점):
    #     five_frac 0.780 인데 deep4 0.155 · full_tip 0.0007
    #     손가락별 touch 0.63~0.67 vs wrap(mid∨dist) 0.28~0.48
    #   = 정책의 "최고 파지"는 감싼 게 아니라 **손끝으로 스친 것**이었다.
    #   `five_c = mid∨dist∨tip` 이라 팁이 표면을 스치기만 해도 1점이고, `near_q` 는
    #   그것을 0.124 로 정확히 깎았다 — **두 반쪽이 서로 싸워 G 가 0.10 에 고착**했고
    #   G 가 grasp·lift·transfer·stay 전부에 곱해져 연속 보상 전체가 정격의 10% 였다.
    #   남은 것은 이진 success(가중 20)뿐이라 그것이 총보상의 67~79% 를 먹었고,
    #   그 항이 지배한 직후 두 런이 붕괴했다.
    # ★`deep` 은 **같은 손가락의 두 마디가 동시에** 닿아야 1 이라 팁 스침으로는 못 만든다.
    #   컵 반지름·높이 같은 형상 상수를 하나도 쓰지 않는다(형상 비의존 유지).
    # ★실기 센서 제약(tip only)은 **obs** 에 걸리는 것이고 보상은 sim 전용 privileged
    #   신호다. 현행 `five_c` 도 이미 mid·dist 를 쓰고 있었다 — obs 는 불변이다.
    touch_f = touch_c.float().mean(dim=-1)       # 가용 손가락의 접촉 비율
    deep_f = deep_c.float().mean(dim=-1)         # 두 마디 동시 접촉 = 실제 감쌈
    # 이진 `oppose`(>0.1N) 를 소프트로. 0.5N 기준은 접촉임계 0.1N 의 5배이자 실측
    # p95 7.77N 의 1/15 — 형상이 아니라 **센서 스케일**에서 온 값이다.
    opp = 1.0 - torch.exp(-thumb_force / float(cfg.stage_thumb_force_ref))
    _pf = persist_frac.clamp(0.0, 1.0)
    _tf = float(cfg.stage_graspq_touch)
    _df = float(cfg.stage_graspq_deep)
    _sf = float(cfg.stage_graspq_persist)
    _of = float(cfg.stage_graspq_thumb_floor)
    Q_g = (_of + (1.0 - _of) * opp) * (_tf * touch_f + _df * deep_f + _sf * _pf)
    # 엄지+검지 팁 핀치: touch_f 0.5 · deep_f 0 · persist 0 → Q_g 0.125. 구조적으로 죽는다.
    # 엄지 없이 4지만 긁으면 상한이 `_of` 배 = 대향 없는 전략이 죽는다. 바닥 `_of` 는
    # 첫 접촉이 순손실이 되는 것을 막는다(구 엄지 배수 삭제 사유 회피).

    # ── 진척(포화하지 않음 — 각 단계가 **자기 몫으로** 버는 값) ──────────────────
    H = (height_delta / float(cfg.stage_lift_height_ref)).clamp(0.0, 1.0)   # h=0 → 0
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    T = 1.0 - torch.tanh(d_goal / float(cfg.stage_tracking_std))

    # ── 계층 게이트 λ → μ → ν → ρ (DexPour, IROS 2025 식 3~6) ────────────────────
    # 논문 구조를 그대로 따른다: **이진 누적 곱**. 각 게이트가 앞 게이트를 곱하므로
    # 접근이 열려야 파지가, 파지가 열려야 리프트가, 그 위에 이송이 열린다.
    #   λ = [d_hand_cup < d_approach]
    #   μ = λ · [접촉 손가락 수 ≥ c_finger]
    #   ν = μ · [height ≥ h_lift]
    #   ρ = ν · [d_goal < d_transfer]
    # ★소프트(smoothstep)가 아니라 이진인 것이 논문의 선택이고, 사용자가 요구한 것도
    #   이 계층 구조다. 저장소가 과거에 제거한 것은 **래치**(한 번 열리면 유지·역방향
    #   차단)였고 이것은 매 스텝 재평가되는 **순간 술어**라 성질이 다르다.
    # ★λ=1 인데 μ=0 인 사각지대에서 보상이 0 이 되지 않도록, 게이트 밖에 항상 열려
    #   있는 shaping 두 개(`approach`, `contact`)를 둔다 — 논문의 r_finger_cup_dist /
    #   r_contact 와 같은 역할이다.
    touch_n = touch_c.float().sum(dim=-1)
    lam = (d_gc < float(cfg.stage_gate_approach_m)).float()
    mu = lam * (touch_n >= float(cfg.stage_gate_contact_n)).float()
    nu = mu * (height_delta >= float(cfg.stage_gate_lift_m)).float()
    rho = nu * (d_goal < float(cfg.stage_gate_transfer_m)).float()
    # 컵 축 기울기 — `tilt_deg`(한 스텝 낡음) 대신 즉석 `obj_up` 으로 낸다.
    _cos_up = torch.nn.functional.cosine_similarity(
        obj_up, torch.tensor([0.0, 0.0, 1.0], device=obj_up.device).expand_as(obj_up),
        dim=-1).clamp(-1.0, 1.0)
    _tilt = torch.rad2deg(torch.acos(_cos_up.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))
    U = smoothstep(_tilt, float(cfg.stage_upright_gate_deg[0]),
                   float(cfg.stage_upright_gate_deg[1]))
    # ★⑤ stay = **실제로 안 움직이는가**. 구 S 는 액션 변화량이라 "액션을 안 바꾼다"
    #   였지 "안 움직인다"가 아니었다. 물체 선속도로 바꾼다.
    S = torch.exp(-obj_speed / float(cfg.stage_stay_speed_ref))
    _r = xy_disp / float(cfg.stage_disp_limit)
    F = 1.0 / (1.0 + _r * _r)                    # 제곱역수 — 0 에 닿지 않아 gradient 유지

    # ---- ① 접근(팔) — 무게이트 + 컵 밀기·기울임 벌점(grasp_v1) ---------------------
    # ★벌점은 approach_weight 로 곱하지 **않는다**(grasp_v1 배선 그대로 — 벌점이
    #   접근 가중에 비례해 희석되면 "빨리 가되 밀어도 된다"가 된다).
    # ★★08.25 align 을 **수평 성분만** 보도록 고쳤다.
    #   구 정의는 접근축과 컵방향의 3D 코사인이었는데, 접근 중 컵이 palm 보다 아래에
    #   있으므로(홈에서 컵방향 z = −0.594) **손을 숙일수록 align 이 올랐다**:
    #   홈 0.639 → 완전히 겨누면 1.0 = approach +16%. 기울이기를 **보상**하고 있었다.
    #   방금 자세 항(perp/roll)을 베이스 기준으로 바꿨는데 align 이 컵 기준으로 남으면
    #   두 항이 정면으로 싸운다.
    #   → 두 벡터를 기준축(베이스 +z)에 수직인 평면으로 투영해 **방위각만** 본다.
    #     "컵 쪽을 향해라"는 유지되고 "숙여라"는 사라진다. 피치는 perp_q 가 맡는다.
    def _proj(v):
        return v - (v * ref_up).sum(dim=-1, keepdim=True) * ref_up

    align = torch.nn.functional.cosine_similarity(
        _proj(grasp_center_pos - palm_pos), _proj(object_pos - palm_pos),
        dim=-1, eps=1e-6)
    _al = float(cfg.stage_align_floor)
    # ★★자세 인자(08.25 사용자 규약). palm_ee **+x 가 손바닥 법선**이고, 그것이
    #   컵 축(+z)과 **수직**이어야 한다 — 실측으로 홈 자세가 이미 −0.0025 로 만족한다.
    #   `align` 은 접근축이 컵을 겨누는가(1자유도)만 보므로 롤·피치를 전혀 안 잡았고,
    #   오히려 컵이 손보다 아래라 **숙이면 approach 가 +16%** 오르는 역유인이 있었다.
    _perp = 1.0 - torch.nn.functional.cosine_similarity(
        palm_x, ref_up, dim=-1, eps=1e-6).abs()          # 법선 ⊥ 베이스축 → 1.0
    perp_q = _perp.clamp(0.0, 1.0) ** float(cfg.stage_perp_exponent)
    # ★법선 수직만으로는 **법선 둘레의 롤**이 남는다(90° 굴리면 손가락이 세로 평면으로
    #   감싸는데 perp 는 그대로 1.0). palm_ee +y 가 컵 축과 정렬되면 그 자유도가 잠긴다.
    #   과하면 cfg 로 이 항만 끈다(stage_roll_exponent = 0 → 항상 1.0).
    roll_q = torch.nn.functional.cosine_similarity(
        palm_y, ref_up, dim=-1, eps=1e-6).clamp(0.0, 1.0) ** float(cfg.stage_roll_exponent)
    _of = float(cfg.stage_orient_floor)
    orient_q = _of + (1.0 - _of) * perp_q * roll_q

    approach = (
        float(cfg.stage_approach_weight)
        * torch.exp(-float(cfg.stage_approach_sharpness) * d_gc)
        * (_al + (1.0 - _al) * 0.5 * (1.0 + align))
        * orient_q
        - float(cfg.stage_approach_xy_penalty)
        * torch.relu(xy_disp - float(cfg.stage_approach_xy_margin))
        - float(cfg.stage_approach_tilt_penalty)
        * torch.relu(tilt_deg - float(cfg.stage_approach_tilt_margin_deg))
    )

    # ---- 계층 사다리 — 각 칸은 자기 게이트 × 자기 진척 -----------------------------
    # 게이트가 이진이므로 열린 칸의 지급 = 가중 × 진척. 가중이 단조 증가하면
    # **실지급도 단조 증가한다**(구 구조는 인자곱이 매번 <1 이라 역전됐다:
    # 실지급 grasp 1.469 > lift 0.757 > transfer 0.661 > stay 0.334).
    contact = touch_f                              # 게이트 없음 — 사각지대 방지 shaping
    grasp = mu * Q_g                               # 파지가 열려야
    lift = nu * U * H                              # 리프트가 열려야 (직립 품질 곱)
    transfer = nu * T                              # 리프트 위에서 목표로
    stay = rho * S                                 # 목표권 안에서 정지

    # ---- 성공 — 이진 AND 를 **연속 곱**으로 (사용자 요구: 소프트) --------------------
    # 구 판정은 5중 AND 이진이라 총보상의 67~79% 를 차지하면서 임계에서 깜빡였다
    # (실측 height 0.101~0.125 vs 임계 0.12 · d_goal 0.067~0.077 vs 임계 0.05).
    # 전이 구간 하한은 전부 **현재 실측 분포가 걸쳐 있는 곳**에 둔다 = 실패 반쪽에도
    # gradient 가 생긴다(구 판정은 정확히 0 이었다).
    s_h = smoothstep(height_delta, *cfg.stage_succ_height_band)
    s_c = smoothstep(Q_g, *cfg.stage_succ_graspq_band)
    s_o = 1.0 - torch.exp(-thumb_force / float(cfg.stage_thumb_force_ref))
    s_t = smoothstep(_tilt, *cfg.stage_succ_tilt_band_deg)
    s_d = smoothstep(d_goal, *cfg.stage_succ_goal_band_m)
    succ_soft = s_h * s_c * s_o * s_t * s_d

    terms = {
        # ★approach 는 이미 가중·벌점이 반영된 값이다(grasp_v1 배선) — 다시 곱하지 않는다.
        "approach": approach,
        "contact": float(cfg.stage_contact_weight) * contact,
        "grasp": float(cfg.stage_grasp_weight) * grasp,
        "lift": float(cfg.stage_lift_weight) * lift,
        "transfer": float(cfg.stage_transfer_weight) * transfer,
        "stay": float(cfg.stage_stay_weight) * stay,
        "success": float(cfg.stage_success_weight) * succ_soft * F,
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)

    # 진단용(보상 아님) — env 가 로깅에서 pop 한다.
    terms["_d_gc"] = d_gc
    terms["_align"] = align
    terms["_H"] = H
    terms["_U"] = U
    terms["_deep4"] = deep4
    terms["_tip_frac"] = tip_frac
    terms["_full_tip"] = full_tip
    terms["_persist"] = _pf
    terms["_envelope"] = envelope
    terms["_grasp_q"] = Q_g
    terms["_touch_frac"] = touch_f
    terms["_deep_frac"] = deep_f
    terms["_opp_soft"] = opp
    terms["_perp_q"] = perp_q
    terms["_roll_q"] = roll_q
    terms["_orient_q"] = orient_q
    terms["_T"] = T
    terms["_S"] = S
    # ★계층 게이트 — "어느 단계까지 열렸나"가 로그로 보여야 한다. 지금까지는 어느
    #   조건이 막는지 알 수 없었다(구 5중 AND 도 같은 병이었다).
    terms["_lam"] = lam
    terms["_mu"] = mu
    terms["_nu"] = nu
    terms["_rho"] = rho
    terms["_succ_soft"] = succ_soft
    terms["_succ_s_h"] = s_h
    terms["_succ_s_c"] = s_c
    terms["_succ_s_d"] = s_d
    terms["_succ_s_t"] = s_t
    # 3번째 반환값은 구 `gate` 자리 — 계층에서 "파지가 열렸나" = μ 가 그 의미다.
    grip_ok = mu > 0.5
    return total, terms, grip_ok, wrap4
