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
    grip_close: torch.Tensor,         # (N,) [0,1] · 정책의 실제 폐쇄도(관절 평균)
    contact_frac: torch.Tensor,       # (N,) [0,1] · 접촉한 손가락 비율(mid∨dist∨tip)
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
    _mix = float(cfg.stage_lift_envelope_mix)
    G = (1.0 - _mix) * tip_frac + _mix * envelope

    H = (height_delta / float(cfg.stage_lift_height_ref)).clamp(0.0, 1.0)   # h=0 → 0
    U = torch.exp(-tilt_deg / float(cfg.stage_upright_tau_deg))
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    T = 1.0 - torch.tanh(d_goal / float(cfg.stage_tracking_std))
    _dn = torch.norm(actions - prev_actions, dim=-1) / (actions.shape[-1] ** 0.5)
    S = torch.exp(-float(cfg.stage_stabilize_sharpness) * _dn)
    _r = xy_disp / float(cfg.stage_disp_limit)
    F = 1.0 / (1.0 + _r * _r)                    # 제곱역수 — 0 에 닿지 않아 gradient 유지

    # ---- ① 접근(팔) — 무게이트 + 컵 밀기·기울임 벌점(grasp_v1) ---------------------
    # ★벌점은 approach_weight 로 곱하지 **않는다**(grasp_v1 배선 그대로 — 벌점이
    #   접근 가중에 비례해 희석되면 "빨리 가되 밀어도 된다"가 된다).
    d_gc = torch.norm(grasp_center_pos - object_pos, dim=-1)
    align = torch.nn.functional.cosine_similarity(
        grasp_center_pos - palm_pos, object_pos - palm_pos, dim=-1, eps=1e-6)
    _al = float(cfg.stage_align_floor)
    approach = (
        float(cfg.stage_approach_weight)
        * torch.exp(-float(cfg.stage_approach_sharpness) * d_gc)
        * (_al + (1.0 - _al) * 0.5 * (1.0 + align))
        - float(cfg.stage_approach_xy_penalty)
        * torch.relu(xy_disp - float(cfg.stage_approach_xy_margin))
        - float(cfg.stage_approach_tilt_penalty)
        * torch.relu(tilt_deg - float(cfg.stage_approach_tilt_margin_deg))
    )

    # ---- ② 파지(손가락) — **전부 접촉 항**(grasp_v1 grasp_quality) -----------------
    # ★거리 항(구 `reach`)을 폐기했다. `reach` 는 mid/dist 링크 → 물체 거리라 컵이 손
    #   밖에 있으면 **손가락을 펼수록 커졌고**, 정책이 감쌈을 버리고 그걸 취했다
    #   (lstm_test9: wrap4 0.125→0.042 인데 R_grasp 1.224→1.310). 접촉 개수만 세면
    #   그 계곡이 구조적으로 생길 수 없다.
    # ★합이 1 로 재정규화되므로 credit 을 올려도 grasp 최대치는 불변 —
    #   "감쌈만 하고 안 드는" 국소최적을 못 만든다(grasp_v1 reward-audit 근거).
    _cred = float(cfg.stage_grasp_envelope_credit)
    _ts = (1.0 - _cred) / 0.60
    grasp = (
        0.15 * _ts * tip_frac
        + 0.20 * _ts * full_tip
        + 0.25 * _ts * persist_frac.clamp(0.0, 1.0)
        + _cred * deep4
    )

    # ---- ③④⑤ 소프트 곱셈 계층 (인자 3 → 4 → 5) ----------------------------------
    # ★리프트에서 U(직립)·F(밀림)를 뺀다(08.25). 사용자 단계 순서가
    #   **접근 → 파지 → 리프트 → 기울기 → 이송 → 안정화** 인데, U 를 리프트에 곱하면
    #   "똑바로 세운 채로만 들 수 있다"가 되어 리프트와 기울기가 한 단계로 뭉개진다.
    #   lstm_test8 실측: lift = 12·G(0.58)·H(0.025)·U(0.25)·F(0.18) = 0.008 —
    #   네 인자가 각각 0.2~0.5 인데 곱하면 소멸해 어느 방향으로도 gradient 가 없었다.
    #   이제 **기울어진 채로 들어도 리프트는 받고**, 세우면 이송 8 이 추가로 열린다.
    lift = G * H
    transport = G * H * U * T
    stabilize = G * H * U * T * S

    # ---- ⑦ 헛닫힘 벌점 — "닿지도 않았는데 주먹" ----------------------------------
    # ★사용자 렌더링 관찰(증거 2순위): palm 이 컵에서 멀어져도 손가락이 주먹을 쥔 채
    #   배회한다. 멀어지면 손을 다시 벌려야 재접근이 되는데 그 압력이 없었다.
    #   실측 근거: 보상의 **97.5%가 approach** 인데(0.714/0.732) approach 는 손 상태를
    #   전혀 안 본다. 쥐는 것이 보상도 벌점도 아닌 **공짜**라 그쪽으로 흘렀다.
    #   그리고 주먹은 중립이 아니다 — probe_lateral 실측에서 닫으면 컵이 밀려나
    #   d_gc 23 → 64.5mm, 접촉력 5~8.7N 인데 손바닥면 wrap 은 0.12~0.50 뿐이었다
    #   (바깥에서 미는 중). 보상이 그 손해를 못 보고 있었다.
    #
    # ★★거리 상수를 쓰지 않는다. `d_gc` 는 **물체 원점까지의 거리**라 물체가 커지면
    #   표면이 더 일찍 닿고 작아지면 더 깊이 들어와야 한다 — "닿기 직전 거리"가
    #   크기마다 다르므로 하드 임계는 다물체에서 깨진다. 대신 **접촉 자체**를 쓴다.
    #   큰 물체는 낮은 폐쇄도에서 닿아 벌점이 일찍 사라지고, 작은 물체는 더 닫아야
    #   하지만 닿는 순간 똑같이 사라진다 = 크기에 자동 적응.
    #
    # ★제곱인 이유: 접촉을 만들려면 어차피 좀 닫아야 한다. 선형이면 그 탐색을 막는다.
    #     close 0.3(탐색) → 0.09·w   ·   close 1.0(주먹) → 1.00·w
    # ★뺄셈인 이유: approach 에 곱하면 벌점이 exp(−k·d_gc) 를 따라가 **컵에 가까울수록
    #   커진다** — 닫아야 할 바로 그 자리에서 최대가 되는 역방향이다.
    open_pen = (float(cfg.stage_open_penalty)
                * grip_close.clamp(0.0, 1.0) ** 2
                * (1.0 - contact_frac.clamp(0.0, 1.0)))

    # ---- ⑥ 성공(이진 보너스) -----------------------------------------------------
    success_now = (
        (height_delta >= float(cfg.stage_success_height))
        & (wrap4 >= float(cfg.stage_success_wrap_min))
        & oppose
        & (tilt_deg <= float(cfg.stage_success_tilt_deg))
        & (d_goal <= float(cfg.stage_success_pos_tol))
    )

    terms = {
        # ★approach 는 이미 가중·벌점이 반영된 값이다(grasp_v1 배선) — 다시 곱하지 않는다.
        "approach": approach,
        "grasp": float(cfg.stage_grasp_weight) * grasp,
        "lift": float(cfg.stage_lift_weight) * lift,
        "transport": float(cfg.stage_transport_weight) * transport,
        "stabilize": float(cfg.stage_stabilize_weight) * stabilize,
        "success": float(cfg.stage_success_weight) * success_now.float() * F,
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
        "open_pen": -open_pen,
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
    terms["_contact_frac"] = contact_frac
    terms["_grip_close"] = grip_close
    # 3번째 반환값은 구 `gate` 자리 — 로깅 호환을 위해 "쓸만한 파지" 이진값을 싣는다.
    grip_ok = G > 0.5
    return total, terms, grip_ok, wrap4
