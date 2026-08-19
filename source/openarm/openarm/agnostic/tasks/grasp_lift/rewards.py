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


def success_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, pos_std: float) -> torch.Tensor:
    """(1 − tanh(‖obj − goal‖ / pos_std))² — dexsuite success (rot 없음 = lift 변형)."""
    d = torch.norm(object_pos - goal_pos, dim=-1)
    return (1.0 - torch.tanh(d / pos_std)) ** 2


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
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """returns (total, terms, gate)."""
    g = contact_gate(group_a_force, group_b_force, float(cfg.contact_force_threshold))
    terms = {
        "reaching": float(cfg.reaching_weight)
        * reaching_reward(fingertip_pos, object_pos, float(cfg.reaching_std)),
        "contact": float(cfg.contact_weight) * g.float(),
        "tracking": float(cfg.tracking_weight)
        * tracking_reward(object_pos, goal_pos, float(cfg.tracking_std), g),
        "success": float(cfg.success_weight)
        * success_reward(object_pos, goal_pos, float(cfg.success_std)),
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, g
