"""grasp-lift 보상 — IsaacLab Dexsuite(Kuka-Allegro lift) 이식. 로봇 무관 텐서 함수.

원본: IsaacLab/source/isaaclab_tasks/.../manipulation/dexsuite/mdp/rewards.py
설계 원칙(dexsuite 가 코드로 증명한 것):
  · 항 5개 + 정규화 2개, 곱셈 게이트는 접촉 1개뿐.
  · lift 항 없음 — 중력 커리큘럼(여기서는 물체 반중력 보상력)이 대체.
    "removes the need for a special 'Lift' reward" (dexsuite_env_cfg.py:305-317 주석)
  · reaching 은 손끝 거리의 **max** — 전 손가락을 물체로 끌어당긴다(1지 치팅 차단).
  · tracking 은 접촉 게이트 곱 — 접촉 없이 물체를 쳐서 goal 로 보내는 경로 차단.
"""

from __future__ import annotations

import torch


def reaching_reward(fingertip_pos: torch.Tensor, object_pos: torch.Tensor, std: float) -> torch.Tensor:
    """1 − tanh(max_i ‖tip_i − obj‖ / std).  fingertip_pos: (N, T, 3)."""
    d = torch.norm(fingertip_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1.0 - torch.tanh(d / std)


def group_reaching(fingertip_pos: torch.Tensor, object_pos: torch.Tensor, std: float,
                   group_idx: torch.Tensor) -> torch.Tensor:
    """대향 그룹별 접근 보상 — 그룹 안에서 물체에 **가장 가까운** 손끝 기준.

    ★max-over-all-tips 는 "엄지가 반대편에 와야 한다"를 표현하지 못한다(agn_test10:
    4지는 컵을 감싸는데 엄지 접촉력이 전 구간 0 → 게이트 영구 미발화). 접촉 게이트가
    쓰는 그룹 구조를 접근 단계에도 그대로 적용해 **양쪽 그룹이 각각 물체에 닿도록**
    유도한다. 2지 그리퍼(조1/조2)에도 동일하게 성립 — max 판정보다 robot-agnostic.
    """
    d = torch.norm(fingertip_pos[:, group_idx] - object_pos[:, None, :], dim=-1).min(dim=-1).values
    return 1.0 - torch.tanh(d / std)


def contact_gate(group_a_force: torch.Tensor, group_b_force: torch.Tensor, threshold: float) -> torch.Tensor:
    """대향 접촉 게이트: (A그룹 아무 손가락) AND (B그룹 아무 손가락). bool (N,).

    group_*_force: (N, F) 손가락별 접촉력 크기 [N].
    dexsuite 의 "thumb AND (index|middle|ring)" 의 일반화 — 2지 그리퍼는 jaw1 AND jaw2.
    """
    a = (group_a_force > threshold).any(dim=-1)
    b = (group_b_force > threshold).any(dim=-1)
    return a & b


def tracking_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, std: float,
                    gate: torch.Tensor) -> torch.Tensor:
    """(1 − tanh(‖obj − goal‖ / std)) × 접촉게이트."""
    d = torch.norm(object_pos - goal_pos, dim=-1)
    return (1.0 - torch.tanh(d / std)) * gate.float()


def success_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, pos_std: float,
                   gate: torch.Tensor) -> torch.Tensor:
    """(1 − tanh(‖obj − goal‖ / pos_std))² × 접촉게이트.

    ★dexsuite 원본은 게이트가 없지만(goal 이 임의 pose 명령+재샘플), 우리 goal 은
    스폰+15cm 고정 + 저중력 커리큘럼이라 "쳐올려 저글링" 으로 비접촉 수확이 가능
    (reward-audit Check 1 REVISE). 최종 행동이 "잡고 유지"이므로 접촉 게이트는
    태스크 정의 그 자체 — hold 중 상시 참이라 절벽이 아니다.
    """
    d = torch.norm(object_pos - goal_pos, dim=-1)
    return ((1.0 - torch.tanh(d / pos_std)) ** 2) * gate.float()


def lift_reward(height_delta: torch.Tensor, target_height: float,
                gate: torch.Tensor) -> torch.Tensor:
    """스폰 대비 상승분의 선형 진척(0~1) × 접촉게이트.

    ★dexsuite 원본은 lift 항을 빼고 중력 커리큘럼으로 대체했지만, 그 전제는
    goal 이 **물체 근처 랜덤 pose** 라 tracking gradient 가 살아 있다는 것이다.
    우리 goal 은 스폰+15cm 고정이라 tracking(std 0.1)이 tanh 포화 구간에 있어
    초기 몇 cm 리프트의 gradient 가 사실상 0 이었다(agn_test12: 잡고 정지 수렴,
    d 0.16→0.15 일 때 항 값 0.078→0.095). 부분 진척을 선형으로 되돌려준다.
    """
    return (height_delta / max(target_height, 1e-6)).clamp(0.0, 1.0) * gate.float()


def tilt_penalty(object_tilt_deg: torch.Tensor, free_deg: float) -> torch.Tensor:
    """전도 페널티: free_deg 초과분을 90°로 정규화한 값(0~1).

    ★agn_test11 ep3000: 접근 중 컵을 넘어뜨린 뒤 **쓰러진 컵을 테이블에 눌러** 접촉
    게이트 0.9 를 만족시키고 리프트는 영영 못 하는 상태로 수렴했다. 기울기를 보는
    항이 하나도 없어 "넘어뜨려도 손해가 없었다". free_deg 여유대는 정상 파지 중의
    자연스러운 흔들림을 벌하지 않기 위한 것.
    """
    return ((object_tilt_deg - free_deg).clamp(min=0.0) / 90.0).clamp(0.0, 1.0)


def action_l2_clamped(actions: torch.Tensor, clamp: float = 1.0) -> torch.Tensor:
    return torch.sum(actions**2, dim=-1).clamp(max=clamp)


def action_rate_l2_clamped(actions: torch.Tensor, prev_actions: torch.Tensor,
                           clamp: float = 1.0) -> torch.Tensor:
    return torch.sum((actions - prev_actions) ** 2, dim=-1).clamp(max=clamp)


def compute_grasp_lift_rewards(
    *,
    fingertip_pos: torch.Tensor,      # (N, T, 3) env-local
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    group_a_force: torch.Tensor,      # (N, Fa)
    group_b_force: torch.Tensor,      # (N, Fb)
    group_a_tip_idx: torch.Tensor,    # fingertip_pos 안에서의 A 그룹 인덱스
    group_b_tip_idx: torch.Tensor,
    object_tilt_deg: torch.Tensor,    # (N,) 물체 z축과 월드 z축 사이각
    height_delta: torch.Tensor,       # (N,) 스폰 대비 상승분 [m]
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """returns (total, terms, gate)."""
    g = contact_gate(group_a_force, group_b_force, float(cfg.contact_force_threshold))
    _std = float(cfg.reaching_std)
    _w = 0.5 * float(cfg.reaching_weight)
    terms = {
        # 대향 그룹별 접근(각 0.5×w) — 합은 기존 reaching 최대치와 동일해 스케일 불변
        "reaching_a": _w * group_reaching(fingertip_pos, object_pos, _std, group_a_tip_idx),
        "reaching_b": _w * group_reaching(fingertip_pos, object_pos, _std, group_b_tip_idx),
        "contact": float(cfg.contact_weight) * g.float(),
        "tracking": float(cfg.tracking_weight)
        * tracking_reward(object_pos, goal_pos, float(cfg.tracking_std), g),
        "success": float(cfg.success_weight)
        * success_reward(object_pos, goal_pos, float(cfg.success_std), g),
        "lift": float(cfg.lift_weight)
        * lift_reward(height_delta, float(cfg.goal_height_offset), g),
        "tilt_penalty": float(cfg.tilt_penalty_weight)
        * tilt_penalty(object_tilt_deg, float(cfg.tilt_free_deg)),
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, g
