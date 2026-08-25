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
    tip_c: torch.Tensor,              # (N, F) bool · 손가락별 팁 접촉(엄지 포함 5지)
    persist_frac: torch.Tensor,       # (N,) [0,1] · 접촉 지속 스텝 / 정규화 스텝수
    wrap_c: torch.Tensor,             # (N, B) bool · 4지 손바닥면 감쌈(mid ∨ dist)
    deep_c: torch.Tensor,             # (N, B) bool · 4지 깊은 감쌈(mid ∧ dist)
    oppose: torch.Tensor,             # (N,)   bool · 엄지 대향(팁 포함 총접촉)
    height_delta: torch.Tensor,       # (N,) 스폰 기준 상승 [m]
    tilt_deg: torch.Tensor,           # (N,) 물체 기울기 [deg]
    xy_disp: torch.Tensor,            # (N,) 스폰 기준 수평 밀림 [m]
    five_c: torch.Tensor,             # (N,5) bool · **5지 전부**의 손바닥면 접촉(mid∨dist∨tip)
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

    # ── ② grasp 품질 = 5지 접촉 × palm 밀착 ─────────────────────────────────
    # ★"palm 밀착"에 새 센서를 쓰지 않는다 — 실기 센서는 **tip 에만** 있어서
    #   palm 접촉은 배포 불가다. 파지중심은 palm 에 강체로 붙은 점이므로
    #   `d_gc → 0` 이 곧 "물체가 손 깊숙이 = palm 밀착"이다. FK 로만 계산된다.
    # ★형상 어댑티브: 크기 가정이 없다. 어떤 모양이든 5지가 닿고 손 깊숙이 들어오면
    #   감싼 것이다. 엄지 하나로 툭 건드리는 전략은 five_frac 이 0.2 라 구조적으로 죽는다.
    five_frac = five_c.float().mean(dim=-1)      # 엄지 포함 5지 — 1지 터치는 0.2
    near_q = torch.exp(-d_gc / float(cfg.stage_grasp_near_tau))
    G = five_frac * near_q

    H = (height_delta / float(cfg.stage_lift_height_ref)).clamp(0.0, 1.0)   # h=0 → 0
    # ★④ 컵의 +z 가 중력 반대인가 — 기울기 각도가 아니라 축 정렬로 직접 잰다.
    U = torch.nn.functional.cosine_similarity(
        obj_up, torch.tensor([0.0, 0.0, 1.0], device=obj_up.device).expand_as(obj_up),
        dim=-1).clamp(0.0, 1.0)
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    T = 1.0 - torch.tanh(d_goal / float(cfg.stage_tracking_std))
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

    # ---- ②~⑤ 소프트 게이트 계층 -------------------------------------------------
    # 인자 수가 단계마다 1 → 2 → 2 → 4 → 5 로 깊어지고, 각 인자가 [0,1] 이라 깊을수록
    # 값이 작아진다. 가중을 그만큼 키워야 단계가 실제로 열린다(lstm_test8 실측:
    # 12·G·H·U·F 의 네 인자가 각각 0.2~0.5 인데 곱이 0.008 로 소멸해 gradient 부재).
    grasp = G                                   # 5지 접촉 × palm 밀착
    lift = G * H
    transfer = G * H * U * T                    # ④ 이송 + 컵 직립
    stay = G * H * U * T * S                    # ⑤ 목표에서 정지

    # ---- ⑥ 성공(이진 보너스) -----------------------------------------------------
    success_now = (
        (height_delta >= float(cfg.stage_success_height))
        & (five_frac >= float(cfg.stage_success_wrap_min))
        & oppose
        & (tilt_deg <= float(cfg.stage_success_tilt_deg))
        & (d_goal <= float(cfg.stage_success_pos_tol))
    )

    terms = {
        # ★approach 는 이미 가중·벌점이 반영된 값이다(grasp_v1 배선) — 다시 곱하지 않는다.
        "approach": approach,
        "grasp": float(cfg.stage_grasp_weight) * grasp,
        "lift": float(cfg.stage_lift_weight) * lift,
        "transfer": float(cfg.stage_transfer_weight) * transfer,
        "stay": float(cfg.stage_stay_weight) * stay,
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
    terms["_tip_frac"] = tip_frac
    terms["_full_tip"] = full_tip
    terms["_persist"] = persist_frac.clamp(0.0, 1.0)
    terms["_envelope"] = envelope
    terms["_grasp_q"] = grasp
    terms["_five_frac"] = five_frac
    terms["_near_q"] = near_q
    terms["_perp_q"] = perp_q
    terms["_roll_q"] = roll_q
    terms["_orient_q"] = orient_q
    terms["_T"] = T
    terms["_S"] = S
    # 3번째 반환값은 구 `gate` 자리 — 로깅 호환을 위해 "쓸만한 파지" 이진값을 싣는다.
    grip_ok = G > 0.5
    return total, terms, grip_ok, wrap4
