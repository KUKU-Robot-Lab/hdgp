"""grasp_v1 전용 reward — 공유 core(openarm.common.grasp_reward_core)에서 분리.

분리 이유(2026-08-16 사용자 지시): grasp_v1 은 "인벨롭 그립 + 외란 하 파지력 유지"라는
고유 목표를 위해 감쌈 깊이·유지 페널티·재조임을 계속 손대야 하는데, 공유 core 를 건드리면
grasp_v2 / grasp_v7_2 / grasp_v10_3 / grasp_adapt 가 전부 영향을 받는다. 실제로 08.16 에
공유 core 를 수정했다가 되돌렸다. 여기서는 grasp_v1 만의 계약으로 자유롭게 진화시킨다.

★공유 core 와의 차이(grasp_v1 고유):
  1. grasp credit 이 참조하는 감쌈을 **깊이**(per-finger middle AND distal)로 사용.
     공유 core 의 envelope_frac(=0.5*(middle+distal) 평균)은 서로 다른 손가락이어도
     값이 올라가 "얕게 여러 곳" 을 깊은 감쌈으로 오인한다.
  2. **감쌈 유지 페널티** 신설 — 래치 시점 대비 감소분만 처벌(절대 깊이 아님).
     공유 core 의 post_lift_contact_loss 는 grip_frac(tip|mid|distal OR)이라
     중간마디를 잃고 손끝으로 미끄러져도 비용이 0 이었다.
  3. grasp_envelope_credit / lift_envelope_mix 를 cfg 로 노출(기본값 상향).

동기화 규칙: 이 파일은 left/right 미러다. 한쪽만 고치지 말 것.
"""

from __future__ import annotations

import torch

from openarm.common.grasp_v2_contract import GRASP_V2_REWARD_TERMS


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def compute_grasp_reward_terms(
    *,
    num_tip_contacts: torch.Tensor,
    tip_contact_frac: torch.Tensor,
    full_tip_contact: torch.Tensor,
    contact_persistence_frac: torch.Tensor,
    envelope_frac: torch.Tensor | None = None,
    grip_frac: torch.Tensor | None = None,
    # ★grasp_v1 고유: 감쌈 "깊이" = per-finger (middle_i AND distal_i) 비율.
    # envelope_frac 은 서로 다른 손가락이어도 값이 오르는 느슨한 지표라 깊이를 못 잰다.
    wrap_frac: torch.Tensor,
    # 래치 시점 깊이 스냅샷 — 유지 페널티의 기준선.
    wrap_at_latch: torch.Tensor,
    palm_to_cup_dist: torch.Tensor,
    fingertip_side_dist: torch.Tensor,
    cup_height_delta: torch.Tensor,
    cup_xy_displacement: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    upright_quality: torch.Tensor,
    lift_latched: torch.Tensor,
    action_delta_norm: torch.Tensor,
    stabilize_reward_gate: torch.Tensor | None = None,
    success_now: torch.Tensor | None = None,
    stable: torch.Tensor | None = None,
    stability_quality: torch.Tensor | None = None,
    cup_lin_vel: torch.Tensor | None = None,
    cup_ang_vel: torch.Tensor | None = None,
    palm_alignment: torch.Tensor | None = None,
    palm_down_alignment: torch.Tensor | None = None,
    palm_to_cup_dist_xy: torch.Tensor | None = None,
    palm_to_cup_dist_z: torch.Tensor | None = None,
    palm_lin_vel: torch.Tensor | None = None,
    fingertip_min_z: torch.Tensor | None = None,
    cfg: object = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Shared grasp v3 reward with decoupled XY standoff & Z height alignment."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    contact_persistence_frac = contact_persistence_frac.clamp(0.0, 1.0)
    graded_contact = tip_contact_frac.clamp(0.0, 1.0)
    if envelope_frac is not None:
        env_quality = envelope_frac.clamp(0.0, 1.0)
        _emix = _cfg_float(cfg, "lift_envelope_mix", 0.5)
        graded_contact = (1.0 - _emix) * graded_contact + _emix * env_quality

    lifted_bool = cup_height_delta >= _cfg_float(cfg, "lift_success_height", 0.04)
    lifted_gate = lifted_bool.float()
    reward_upright_success = cup_tilt_deg <= _cfg_float(cfg, "success_upright_max_deg", 20.0)
    final_upright_success = cup_tilt_deg <= _cfg_float(cfg, "stabilize_upright_max_deg", 5.0)

    stabilize_gate = (
        lift_gate if stabilize_reward_gate is None else stabilize_reward_gate.float()
    )

    xy_margin = _cfg_float(cfg, "grasp_xy_threshold", 0.0)
    tilt_margin = _cfg_float(cfg, "grasp_upright_threshold_deg", 0.0)

    # -----------------------------------------------------------------------
    # Reach v4: Decoupled XY Standoff & Z Height Alignment & Damping & Guards
    # -----------------------------------------------------------------------
    standoff_target = _cfg_float(cfg, "standoff_target_dist", 0.08)

    # 1. [XY 수평 거리 보상] 컵 중심으로부터 수평 8cm(표면 앞 3.5cm) 정렬
    dist_xy = palm_to_cup_dist if palm_to_cup_dist_xy is None else palm_to_cup_dist_xy
    dist_xy_error = torch.abs(dist_xy - standoff_target)
    w_xy_coarse = _cfg_float(cfg, "approach_xy_weight", 0.30)
    w_xy_fine = _cfg_float(cfg, "approach_xy_fine_weight", 0.35)
    std_xy = _cfg_float(cfg, "approach_xy_fine_std", 0.05)
    reward_xy = -w_xy_coarse * dist_xy_error + w_xy_fine * (1.0 - torch.tanh(dist_xy_error / std_xy))

    # 2. [Z 높이 일치 보상] 손바닥 높이를 컵 허리 정중앙 높이(Z_error -> 0)로 강제 밀착
    height_error = torch.zeros_like(dist_xy_error) if palm_to_cup_dist_z is None else palm_to_cup_dist_z
    w_z_coarse = _cfg_float(cfg, "approach_z_weight", 0.30)
    w_z_fine = _cfg_float(cfg, "approach_z_fine_weight", 0.35)
    std_z = _cfg_float(cfg, "approach_z_fine_std", 0.03)
    reward_z = -w_z_coarse * height_error + w_z_fine * (1.0 - torch.tanh(height_error / std_z))

    # ★ [보상 해킹 방지 게이트] Z 높이가 컵 높이에 10cm 이내로 들어와야만 XY 수평 정렬 양수 보상을 지급
    # (테이블 아래에 서서 수평 정렬만으로 보상을 챙기는 Shortcut Learning 원천 차단)
    height_gate = (1.0 - (height_error / 0.10).clamp(max=1.0))
    reward_xy = torch.where(reward_xy > 0.0, reward_xy * height_gate, reward_xy)

    # 3. [자세 정렬 보상] 손바닥 정면(+X) 컵 대면 + 손가락(+Z) 수평(Horizontal) 자세 안정
    align_facing = upright_quality if palm_alignment is None else palm_alignment.clamp(0.0, 1.0)
    align_down = torch.ones_like(align_facing) if palm_down_alignment is None else palm_down_alignment.clamp(0.0, 1.0)
    w_align_facing = _cfg_float(cfg, "approach_align_weight", 0.20)
    w_align_down = _cfg_float(cfg, "approach_down_align_weight", 0.10)
    reward_align = (w_align_facing * align_facing + w_align_down * align_down) * height_gate

    # 4. [목표 지점 감속 정지 보상] Standoff 근접 게이트 내에서 손바닥 선속도 감속 유도
    w_vel_damp = _cfg_float(cfg, "approach_vel_damping_weight", 0.0)
    std_vel = _cfg_float(cfg, "approach_vel_damping_std", 0.05)
    gate_xy_std = _cfg_float(cfg, "approach_vel_gate_xy_std", 0.03)
    gate_z_std = _cfg_float(cfg, "approach_vel_gate_z_std", 0.02)

    if palm_lin_vel is not None and w_vel_damp > 0.0:
        palm_spd = palm_lin_vel.norm(dim=-1)
        near_gate = torch.exp(
            -0.5 * (dist_xy_error / gate_xy_std) ** 2
            - 0.5 * (height_error / gate_z_std) ** 2
        )
        reward_vel_damp = w_vel_damp * near_gate * torch.exp(-(palm_spd / std_vel) ** 2)
    else:
        reward_vel_damp = torch.zeros_like(reward_xy)

    # 5. [테이블 바닥 긁힘 방지] 손끝이 테이블 안전선 아래로 파고들 때 힌지 페널티
    w_table = _cfg_float(cfg, "table_clearance_weight", 0.0)
    table_z = _cfg_float(cfg, "table_surface_z", 0.208)
    table_margin = _cfg_float(cfg, "table_clearance_margin", 0.015)
    table_safe_z = table_z + table_margin

    if fingertip_min_z is not None and w_table > 0.0:
        table_penetration = torch.relu(table_safe_z - fingertip_min_z)
        table_clearance_penalty = w_table * table_penetration
    else:
        table_clearance_penalty = torch.zeros_like(reward_xy)

    # 6. [외란 감점] 컵 선속도/각속도/변위 발생 시 강력 감점
    w_lin = _cfg_float(cfg, "cup_lin_vel_penalty_weight", 2.0)
    w_ang = _cfg_float(cfg, "cup_ang_vel_penalty_weight", 0.5)
    w_disp = _cfg_float(cfg, "approach_xy_penalty_weight", 5.0)
    w_tilt = _cfg_float(cfg, "approach_tilt_penalty_weight", 0.08)

    cup_lin_spd = cup_lin_vel.norm(dim=-1) if cup_lin_vel is not None else torch.zeros_like(cup_xy_displacement)
    cup_ang_spd = cup_ang_vel.norm(dim=-1) if cup_ang_vel is not None else torch.zeros_like(cup_xy_displacement)

    disturbance_penalty = (
        w_lin * cup_lin_spd
        + w_ang * cup_ang_spd
        + w_disp * torch.relu(cup_xy_displacement - xy_margin)
        + w_tilt * torch.relu(cup_tilt_deg - tilt_margin)
    )

    # 최종 접근 보상 통합
    approach = pre_lift_gate * (
        reward_xy
        + reward_z
        + reward_align
        + reward_vel_damp
        - disturbance_penalty
        - table_clearance_penalty
    )
    if envelope_frac is None:
        grasp_quality = (
            0.25 * tip_contact_frac + 0.35 * full_tip + 0.40 * contact_persistence_frac
        )
    else:
        # envelope(중간/원위 wrap)을 grasp 보상에 credit → 지배적 grasp 보상이 wrap을
        # 당기는 gradient가 됨(tip-farming 차단). pre_lift_gate라 "wrap만 하고 안 듦"
        # 수렴은 불가(리프트 후 꺼짐).
        # envelope credit(기본 0.40). cfg 로 올리면 grasp 보상이 wrap 을 더 강하게 당김.
        # 나머지 tip 항은 (1-credit) 로 비례 축소해 합=1 유지. (기본값 유지 시 기존 동작 불변.)
        # ★합이 1로 재정규화되므로 credit 을 올려도 grasp 최대치는 불변 —
        #   "감쌈만 하고 안 드는" 국소최적을 구조적으로 못 만든다(reward-audit Check1 근거).
        _ecred = _cfg_float(cfg, "grasp_envelope_credit", 0.40)
        _tip_scale = (1.0 - _ecred) / 0.60
        # grasp_v1 고유: credit 이 참조하는 감쌈은 **깊이(wrap_frac)**. 래치 **전** shaping
        # 이라 래치 시점 자세가 깊어지고, 그 자세가 곧 유지 페널티의 기준선이 된다.
        # graded_contact(리프트 후 게이트)에는 넣지 않는다 — 실측 0.349→0.144(0.41배)라
        # lift 30.0·stabilize 10.0·stability 1.0 을 20~30% 일괄 삭감한다(Check4 REVISE).
        grasp_quality = (
            0.15 * _tip_scale * tip_contact_frac
            + 0.20 * _tip_scale * full_tip
            + 0.25 * _tip_scale * contact_persistence_frac
            + _ecred * wrap_frac.clamp(0.0, 1.0)
        )
    grasp = _cfg_float(cfg, "grasp_weight", 0.0) * pre_lift_gate * grasp_quality
    lift_height_quality = (
        cup_height_delta
        / max(_cfg_float(cfg, "lift_success_height", 0.04), 1e-6)
    ).clamp(min=0.0, max=1.0)
    # Phase K2: lift height bonus는 4cm 위까지 gradient를 살려야 평형이 4cm 이상에 형성된다.
    # lift_height_quality(4cm clamp)를 그대로 쓰면 4cm 위 gradient=0이라 평형이 ~3.1cm에
    # 고착(test2 확인). bonus 전용 quality를 lift_height_bonus_clamp(기본 1.0=4cm,
    # v10-3은 1.5=6cm)까지 열어 컵이 4cm를 안정적으로 넘게 한다.
    lift_height_bonus_quality = (
        cup_height_delta
        / max(_cfg_float(cfg, "lift_success_height", 0.04), 1e-6)
    ).clamp(min=0.0, max=_cfg_float(cfg, "lift_height_bonus_clamp", 1.0))
    lift = (
        _cfg_float(cfg, "lift_reward_weight", 0.0)
        * lift_gate
        * graded_contact
        * lift_height_quality
        * upright_quality
        # Phase K: contact-독립 height 보상. 풀그립(Phase J)으로 graded_contact가 0.85까지
        # 치솟아 lift 보상(=contact×height)이 2.9cm에서 이미 포화 → 정책이 더 들 유인을
        # 잃고 local optimum 고착. height 자체에 contact와 무관한 보상을 더해 4cm 이상까지
        # 들도록 유도. v1은 weight 0(기본)이라 영향 없음.
        + _cfg_float(cfg, "lift_height_bonus_weight", 0.0)
        * lift_gate
        * lift_height_bonus_quality
    )
    action_quality = torch.exp(
        -_cfg_float(cfg, "stabilize_action_sharpness", 1.0) * action_delta_norm
    )
    if stable is None:
        stable_bool = torch.ones_like(cup_height_delta, dtype=torch.bool)
    else:
        stable_bool = stable.bool()
    stable_gate = stable_bool.float()
    stability_quality_f = (
        stable_gate if stability_quality is None else stability_quality
    ).clamp(min=0.0, max=1.0)
    post_lift_contact_loss = (
        _cfg_float(cfg, "post_lift_contact_loss_weight", 0.0)
        * lift_gate
        * lifted_gate
        # envelope-consistent: 임의 마디 접촉(grip_frac)이 주어지면 tip 대신 사용.
        # envelope wrap이 tip을 mid/dist로 옮겨도 그립 손실로 처벌하지 않음
        # (tip-only면 wrap을 처벌 → tip↔envelope 진동). RH56F1은 None→tip 유지.
        * torch.relu(
            1.0 - (tip_contact_frac if grip_frac is None else grip_frac.clamp(0.0, 1.0))
        )
    )
    # ★grasp_v1 고유: 감쌈 유지 페널티.
    # 위 post_lift_contact_loss 는 grip_frac(tip|mid|distal OR)이라 **중간마디를 잃고
    # 손끝으로 미끄러져도 비용이 0** 이다. 난이도가 상수인 2,687 epoch 동안 감쌈만 단조
    # 침식한 실측(middle_count 2.44→2.16, full_envelope 0.24→0.17)의 구조적 원인.
    # 절대 깊이를 처벌하면 기준선이 통째로 내려가 리프트를 억제하므로(−1.10→−3.98 로
    # reward/lift 5.8 에 필적, Check4 ✗) **래치 대비 감소분**만 처벌한다 →
    # 유지하면 정확히 0, 잃을 때만 비용.
    # ⚠️잔여 위험: 얕게 래치하면 잃을 게 없어 0 인 회피 경로. 래치 전 credit(깊이 참조)이
    # 반대 압력을 주고, contact/wrap_at_latch 로 감시한다.
    post_lift_contact_loss = post_lift_contact_loss + (
        _cfg_float(cfg, "wrap_retention_loss_weight", 0.0)
        * lift_gate
        * lifted_gate
        * torch.relu(wrap_at_latch.clamp(0.0, 1.0) - wrap_frac.clamp(0.0, 1.0))
    )

    stabilize = (
        _cfg_float(cfg, "stabilize_weight", 0.0)
        * lift_gate
        * lifted_gate
        * graded_contact
        * upright_quality
        * action_quality
    )
    stability = (
        _cfg_float(cfg, "stability_reward_weight", 0.0)
        * stabilize_gate
        * lifted_gate
        * graded_contact
        * upright_quality
        * stability_quality_f
    )
    success_now_bool = (
        torch.zeros_like(lift_latched, dtype=torch.bool)
        if success_now is None
        else success_now.bool()
    )
    success_bonus = _cfg_float(cfg, "success_bonus_weight", 0.0) * success_now_bool.float()
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
        raise RuntimeError(f"missing grasp v2 reward terms: {sorted(missing_terms)}")
    gates = {
        "pre_lift": pre_lift_gate,
        "lift": lift_gate,
        "lifted": lifted_gate,
        "full_tip_contact": full_tip,
        "contact_persistence": contact_persistence_frac,
        "upright_success": reward_upright_success.float(),
        "final_upright_success": final_upright_success.float(),
        "success_now": success_now_bool.float(),
        "stability_quality": stability_quality_f,
        "stable": stable_gate,
        "action_quality": action_quality,
    }
    total = torch.nan_to_num(
        sum(terms.values()),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return total, terms, gates
