"""grasp-sensor 보상 — 인벨롭 그립 성립 중심 (08.22 재설계). 로봇 무관 텐서 함수.

설계 방침(사용자 결정): 최우선 목표는 **인벨롭 그립 성립** — 다음 태스크(pour)의 전제.
이전 dexsuite 이식(그룹-min reaching + 이진 접촉 AND 게이트)은 rim-hook(림을 한 손가락
으로 걸고 엄지를 대는 파지, 기울임 20~30°)이 보상을 전부 만족해 인벨롭을 만들지
못했다(lstm_test3 영상·TB 실측). 채택 구조:
  ① approach — v1/v2 검증 기하 이식: palm→파지중심 + 손끝을 물체 양옆 대향점으로.
    프리그래스프 **자세 자체**를 shaping — rim-hook 차단의 근본.
  ② envelope — 중간/원위 마디 접촉 비율을 직접 보상 (v2 envelope_credit 계보).
  ③ 게이트는 기존 이진 대향 AND 유지(사용자 결정 — 게이트 인벨롭화는 채택 안 함).
  ④ 성공 판정 3조건: goal 근접 AND envelope_frac AND 직립 (env 쪽에서 판정).
"""

from __future__ import annotations

import torch


def approach_reward(
    palm_pos: torch.Tensor,           # (N, 3) env-local
    fingertip_pos: torch.Tensor,      # (N, T, 3) env-local
    grasp_center: torch.Tensor,       # (N, 3) 물체중심 + z offset
    group_a_tip_idx: torch.Tensor,    # 엄지/조1 그룹의 tip 인덱스
    group_b_env_tip_idx: torch.Tensor,  # 나머지 그룹 ∩ envelope 손가락의 tip 인덱스
    side_radius: float,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """exp(−s·(d_palm + d_side)) — v1/v2 검증 접근 기하 (grasp_reward.py L90 계보).

    d_side 는 palm→물체 xy 방향에 수직인 축 n 위의 대향 두 점 c±n·r 까지의 거리:
    A 그룹(엄지/조1)은 자기 쪽 점으로 min, B 그룹(감쌈 손가락)은 반대쪽 점으로 mean.
    n 의 부호는 **현재 A 그룹 팁이 있는 쪽**으로 동적 선택 — 좌우 미러/그리퍼에서
    같은 코드가 동작한다(robot-agnostic, 고정 부호는 반대손에서 감쌈을 뒤집는다).

    returns (reward_raw, d_palm, d_side) — 가중치는 호출부에서 곱한다.
    """
    d_palm = torch.norm(palm_pos - grasp_center, dim=-1)
    # 대향축: (palm−c) 의 xy 를 90° 회전 → 접근 방향에 수직
    a_xy = palm_pos[:, :2] - grasp_center[:, :2]
    a_xy = a_xy / a_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    n = torch.stack([-a_xy[:, 1], a_xy[:, 0],
                     torch.zeros_like(a_xy[:, 0])], dim=-1)
    tip_a = fingertip_pos[:, group_a_tip_idx]                 # (N, Fa, 3)
    sign = torch.sign(
        ((tip_a.mean(dim=1) - grasp_center) * n).sum(dim=-1, keepdim=True)
    )
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n = n * sign
    target_a = grasp_center + n * side_radius
    target_b = grasp_center - n * side_radius
    d_a = torch.norm(tip_a - target_a[:, None, :], dim=-1).min(dim=-1).values
    d_b = torch.norm(
        fingertip_pos[:, group_b_env_tip_idx] - target_b[:, None, :], dim=-1
    ).mean(dim=-1)
    d_side = 0.6 * d_a + 0.4 * d_b                            # v1 enclosure_thumb_weight
    return torch.exp(-sharpness * (d_palm + d_side)), d_palm, d_side


def envelope_fraction(
    mid_force: torch.Tensor,   # (N, E) envelope 손가락별 중간마디(_3) 접촉력
    dist_force: torch.Tensor,  # (N, E) envelope 손가락별 원위마디(_4) 접촉력
    threshold: float,
) -> torch.Tensor:
    """감싼 손가락 비율: mean_f 1[mid ∨ dist 접촉] ∈ [0,1].

    ★v2 계보의 0.5·(mid평균+dist평균)는 P-B probe 에서 반증됐다 — 두 손가락만 깊게
      림에 걸어도 0.56 이 나와 rim-hook 이 인벨롭보다 높았다(접촉 **깊이**를 재는
      공식). 인벨롭 vs rim-hook 을 가르는 건 **감싼 손가락 수**이므로 손가락별
      wrap 판정(마디 하나라도 접촉)의 비율로 바꾼다: rim-hook(검지+엄지)=0.5,
      실제 정책의 rim-hook(검지만 마디 접촉)=0.25, 전지 감쌈=1.0.
    분모는 profile.envelope_fingers 로 한정된 E 개(호출부에서 열 선별) — pinky 를
    분모에 넣으면 상한이 0.8 로 깎인다(pinky 굴곡축 부재). 팁 접촉은 여기 없다:
    핀치(팁만)로는 이 항이 0 — 감쌈만 이 항을 채운다.
    """
    wrap = (mid_force > threshold) | (dist_force > threshold)
    return wrap.float().mean(dim=-1)


def contact_gate(group_a_force: torch.Tensor, group_b_force: torch.Tensor, threshold: float) -> torch.Tensor:
    """대향 접촉 게이트: (A그룹 아무 손가락) AND (B그룹 아무 손가락). bool (N,).

    group_*_force: (N, F) 손가락별 접촉력 크기 [N].
    dexsuite 의 "thumb AND (index|middle|ring)" 의 일반화 — 2지 그리퍼는 jaw1 AND jaw2.
    """
    a = (group_a_force > threshold).any(dim=-1)
    b = (group_b_force > threshold).any(dim=-1)
    return a & b


def envelope_gate(gate: torch.Tensor, envelope_frac: torch.Tensor,
                  target_frac: float) -> torch.Tensor:
    """유효 게이트 = 대향 이진 접촉 × clamp(env_frac / target, 0, 1) ∈ [0,1].

    ★lstm_test1 실측: goal 계열(13.5)이 **팁 접촉만으로 성립하는 이진 게이트**로만
      막혀 있어 감쌈이 순수 비용이었다 — 정책이 ep500 에 스스로 배운 감쌈(0.335)을
      lift 가 오르면서 되팔았다(ep1000 0.215). ∂(goal 계열)/∂(env_frac)=0 이 원인.
      영상: 엄지 미접촉·네 손가락으로 컵을 손바닥에 눌러 12cm 리프트(press-grip).
    이 인자로 감쌈이 goal 계열의 **선행 조건**이 된다. 연속(곱셈)이라 부분 감쌈은
    부분 보상을 주므로 절벽이 아니고, target(=success_envelope_min)에서 포화한다.
    이진 대향 게이트 자체는 유지(08.22 사용자 결정).
    """
    return gate.float() * (envelope_frac / max(target_frac, 1e-6)).clamp(0.0, 1.0)


def tracking_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, std: float,
                    gate: torch.Tensor) -> torch.Tensor:
    """(1 − tanh(‖obj − goal‖ / std)) × 유효게이트."""
    d = torch.norm(object_pos - goal_pos, dim=-1)
    return (1.0 - torch.tanh(d / std)) * gate


def success_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, pos_std: float,
                   gate: torch.Tensor) -> torch.Tensor:
    """(1 − tanh(‖obj − goal‖ / pos_std))² × 유효게이트.

    goal 이 스폰+15cm 고정이라 "쳐올려 저글링" 비접촉 수확이 가능해 게이트 필수
    (reward-audit Check 1). hold 중 상시 참이라 절벽이 아니다.
    """
    d = torch.norm(object_pos - goal_pos, dim=-1)
    return ((1.0 - torch.tanh(d / pos_std)) ** 2) * gate


def lift_reward(height_delta: torch.Tensor, target_height: float,
                gate: torch.Tensor) -> torch.Tensor:
    """스폰 대비 상승분의 선형 진척(0~1) × 유효게이트.

    tracking(std 0.1)은 초기 몇 cm 리프트에서 tanh 포화 구간이라 gradient 가 사실상
    0 이었다(agn_test12: 잡고 정지 수렴). 부분 진척을 선형으로 되돌려준다.
    """
    return (height_delta / max(target_height, 1e-6)).clamp(0.0, 1.0) * gate


def tilt_penalty(object_tilt_deg: torch.Tensor, free_deg: float) -> torch.Tensor:
    """전도 페널티: free_deg 초과분을 90°로 정규화한 값(0~1).

    ★agn_test11 ep3000: 쓰러진 컵을 눌러 게이트만 채우는 수렴 — 기울기를 보는 항
    부재가 원인. free_deg 여유대는 정상 파지 흔들림 무징계용. 60° 초과는 truncation
    리셋(env)이 별도 처리.
    """
    return ((object_tilt_deg - free_deg).clamp(min=0.0) / 90.0).clamp(0.0, 1.0)


def action_l2_clamped(actions: torch.Tensor, clamp: float = 1.0) -> torch.Tensor:
    """★sum+clamp 는 죽은 항이었다(상시 포화 → gradient 0). mean 은 액션 차원 불변."""
    return torch.mean(actions**2, dim=-1).clamp(max=clamp)


def action_rate_l2_clamped(actions: torch.Tensor, prev_actions: torch.Tensor,
                           clamp: float = 1.0) -> torch.Tensor:
    return torch.mean((actions - prev_actions) ** 2, dim=-1).clamp(max=clamp)


def compute_grasp_sensor_rewards(
    *,
    palm_pos: torch.Tensor,           # (N, 3) env-local
    fingertip_pos: torch.Tensor,      # (N, T, 3) env-local
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    group_a_force: torch.Tensor,      # (N, Fa) 손가락별(합산) 접촉력
    group_b_force: torch.Tensor,      # (N, Fb)
    group_a_tip_idx: torch.Tensor,    # fingertip_pos 안에서의 A 그룹 인덱스
    group_b_env_tip_idx: torch.Tensor,  # B 그룹 ∩ envelope 손가락 인덱스
    env_mid_force: torch.Tensor,      # (N, E) envelope 손가락 중간마디 접촉력
    env_dist_force: torch.Tensor,     # (N, E) envelope 손가락 원위마디 접촉력
    object_tilt_deg: torch.Tensor,    # (N,)
    height_delta: torch.Tensor,       # (N,) 스폰 대비 상승분 [m]
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """returns (total, terms, gate, envelope_frac)."""
    thr = float(cfg.contact_force_threshold)
    g = contact_gate(group_a_force, group_b_force, thr)
    grasp_center = object_pos.clone()
    grasp_center[:, 2] = grasp_center[:, 2] + float(cfg.grasp_z_offset)
    app, d_palm, d_side = approach_reward(
        palm_pos, fingertip_pos, grasp_center,
        group_a_tip_idx, group_b_env_tip_idx,
        float(cfg.side_radius), float(cfg.approach_sharpness),
    )
    env_frac = envelope_fraction(env_mid_force, env_dist_force, thr)
    # goal 계열은 **감쌈을 선행 조건으로** 한다 — envelope_gate() 주석 참조.
    # contact(0.5)만 순수 이진 게이트로 남겨 접촉 자체의 사다리 한 칸을 유지한다.
    g_eff = envelope_gate(g, env_frac, float(cfg.success_envelope_min))
    terms = {
        "approach": float(cfg.approach_weight) * app,
        "envelope": float(cfg.envelope_weight) * env_frac,
        "contact": float(cfg.contact_weight) * g.float(),
        "tracking": float(cfg.tracking_weight)
        * tracking_reward(object_pos, goal_pos, float(cfg.tracking_std), g_eff),
        "success": float(cfg.success_weight)
        * success_reward(object_pos, goal_pos, float(cfg.success_std), g_eff),
        "lift": float(cfg.lift_weight)
        * lift_reward(height_delta, float(cfg.goal_height_offset), g_eff),
        "tilt_penalty": float(cfg.tilt_penalty_weight)
        * tilt_penalty(object_tilt_deg, float(cfg.tilt_free_deg)),
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    # 진단용 원거리도 terms 에 실어 로깅 경로 하나로 (가중 0 취급 아님 — env 가 분리 기록)
    terms["_d_palm"] = d_palm
    terms["_d_side"] = d_side
    terms["_gate_eff"] = g_eff      # 이진 게이트 vs 인벨롭 인자 분리 진단(reward-audit Check 5)
    return total, terms, g, env_frac
