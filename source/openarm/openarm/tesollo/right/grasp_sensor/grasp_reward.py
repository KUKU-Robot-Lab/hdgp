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
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Shared grasp v2 reward used by Tesollo and RH56F1 grasp tasks."""

    lift_gate = lift_latched.float()
    pre_lift_gate = 1.0 - lift_gate
    full_tip = full_tip_contact.float()
    contact_persistence_frac = contact_persistence_frac.clamp(0.0, 1.0)
    # graded contact factor: hard full_tip(>=5) 곱셈 게이트는 TESOLLO에서 검지가
    # 구조적으로 미접촉이라 영원히 0 → lift/stabilize/stability 전부 닫힘. 부분 접촉에도
    # gradient를 주도록 tip_contact_frac(=num_tip/5)로 graded 게이팅한다.
    # success/termination 판정은 별도로 full_tip>=5를 유지(env에서 처리).
    graded_contact = tip_contact_frac.clamp(0.0, 1.0)
    # envelope-aware 접촉 품질: envelope_frac(중간/원위 마디 wrap 비율)이 주어지면
    # lift/stabilize 접촉 게이팅을 tip+envelope 혼합으로 → 손끝-only lift를 부드럽게 억제
    # (hard latch 게이트 대체). RH56F1은 envelope_frac=None → tip-only 기존 동작 유지.
    if envelope_frac is not None:
        env_quality = envelope_frac.clamp(0.0, 1.0)
        # lift 접촉 게이팅의 envelope 비중(기본 0.5=tip/env 반반). cfg 로 올리면 tip-only lift
        # 억제 강화 → envelope 유지 강제. (기본값 유지 시 tesollo 등 기존 동작 불변.)
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
    approach = pre_lift_gate * (
        _cfg_float(cfg, "approach_weight", 0.0)
        * torch.exp(
            -_cfg_float(cfg, "approach_sharpness", 1.0)
            * (palm_to_cup_dist + fingertip_side_dist)
        )
        - _cfg_float(cfg, "approach_xy_penalty_weight", 0.0)
        * torch.relu(cup_xy_displacement - xy_margin)
        - _cfg_float(cfg, "approach_tilt_penalty_weight", 0.0)
        * torch.relu(cup_tilt_deg - tilt_margin)
    )
    if envelope_frac is None:
        grasp_quality = (
            0.25 * tip_contact_frac + 0.35 * full_tip + 0.40 * contact_persistence_frac
        )
    else:
        # envelope(중간/원위 wrap)을 grasp 보상에 credit → 지배적 grasp 보상이 wrap을
        # 당기는 gradient가 됨(tip-farming 차단). "wrap만 하고 안 듦" 수렴은 불가 —
        # latch(4지+엄지)가 성립하면 스크립트 램프가 자동으로 들어올린다(P1 주석 참조).
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
    # ★2026-08-19(P1, audit ACCEPT) pre_lift_gate 곱 제거 — grasp shaping 을 latch 후에도 유지.
    #   구(latch 절벽): latch 순간 grasp+approach 수입(실측 1.63/step)이 0 이 되고 비가역이라,
    #   어설픈 파지의 lift 수입 < 절벽 손실 → 정책이 latch 를 능동 회피(lstm_test3 ep1000:
    #   엄지 접촉 포기 thumb_cup 0.56→0.007 로 thumb-AND latch 를 외과적으로 차단,
    #   lift_ready 0.47→0.00). latch 후에도 quality 수입이 이어지면 latch 는 순이득
    #   (+lift 30/stabilize 10/success 20)이 된다. latch 후 palm 은 스크립트 램프가 지배라
    #   "latch 하고 안 드는" hacking 은 구조적으로 불가. 파지를 놓치면 quality 자동 소멸.
    #   ⚠ lift 구간 수입 증가로 reward/total 절대값은 이전 run 과 비교 금지.
    grasp = _cfg_float(cfg, "grasp_weight", 0.0) * grasp_quality
    # ★2026-08-19 보상 정규화 기준을 성공 임계와 분리. lift_height_ref 미설정(0)이면
    #   기존대로 lift_success_height 를 쓴다(동작 불변).
    _h_ref = _cfg_float(cfg, "lift_height_ref", 0.0)
    if _h_ref <= 0.0:
        _h_ref = _cfg_float(cfg, "lift_success_height", 0.04)
    lift_height_quality = (cup_height_delta / max(_h_ref, 1e-6)).clamp(min=0.0, max=1.0)
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

    # ★2026-08-19 컵 밀림 soft 감쇠. 보상의 86% 를 차지하는 lift/success_bonus 를
    #   컵이 밀린 만큼 연속적으로 깎는다 → "밀어서라도 성공" 이 더 이상 이득이 아니다.
    #   하드 게이트가 아니라 감쇠인 이유는 cfg 주석(cup_xy_disp_limit) 참조.
    _disp_limit = _cfg_float(cfg, "cup_xy_disp_limit", 0.0)
    if _disp_limit > 0.0:
        disp_factor = 1.0 - (cup_xy_displacement / _disp_limit).clamp(0.0, 1.0)
        lift = lift * disp_factor
        success_bonus = success_bonus * disp_factor
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
