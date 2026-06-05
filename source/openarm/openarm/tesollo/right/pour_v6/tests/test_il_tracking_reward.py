"""IL tracking reward 계산 로직 단위 테스트.

Isaac Sim 의존 없음 — 순수 PyTorch 텐서 연산으로 검증.
"""
from __future__ import annotations

import torch


def _compute_il_reward(
    arm_joint_pos: torch.Tensor,   # (N, 7)
    hand_joint_pos: torch.Tensor,  # (N, 20)
    demo_arm: torch.Tensor,        # (N, 7)
    demo_hand: torch.Tensor,       # (N, 20)
    weight_arm: float = 5.0,
    weight_hand: float = 2.0,
) -> torch.Tensor:
    """il_env._get_rewards의 핵심 수식 (Isaac 의존 없이 테스트)."""
    r_arm  = -weight_arm  * (arm_joint_pos  - demo_arm ).pow(2).sum(-1)
    r_hand = -weight_hand * (hand_joint_pos - demo_hand).pow(2).sum(-1)
    return r_arm + r_hand


def test_zero_error_gives_zero_reward() -> None:
    """demo target과 일치하면 reward = 0."""
    arm  = torch.zeros(4, 7)
    hand = torch.zeros(4, 20)
    r = _compute_il_reward(arm, hand, arm.clone(), hand.clone())
    assert torch.allclose(r, torch.zeros(4)), f"zero error → reward must be 0, got {r}"


def test_reward_negative_with_error() -> None:
    """오차가 있으면 reward < 0."""
    arm  = torch.zeros(4, 7)
    hand = torch.zeros(4, 20)
    demo_arm  = torch.ones(4, 7)
    demo_hand = torch.ones(4, 20)
    r = _compute_il_reward(arm, hand, demo_arm, demo_hand)
    assert (r < 0).all(), f"nonzero error → reward must be negative, got {r}"


def test_larger_error_gives_lower_reward() -> None:
    """오차 클수록 reward 더 낮음."""
    arm   = torch.zeros(1, 7)
    hand  = torch.zeros(1, 20)
    small_err_arm = torch.full((1, 7), 0.1)
    large_err_arm = torch.full((1, 7), 1.0)
    zero_hand = torch.zeros(1, 20)

    r_small = _compute_il_reward(arm, hand, small_err_arm, zero_hand)
    r_large = _compute_il_reward(arm, hand, large_err_arm, zero_hand)
    assert r_large.item() < r_small.item(), (
        f"larger error must give lower reward: r_small={r_small.item():.3f}, r_large={r_large.item():.3f}"
    )


def test_weight_scales_reward() -> None:
    """weight_arm이 클수록 arm error의 패널티 증가."""
    arm  = torch.zeros(1, 7)
    hand = torch.zeros(1, 20)
    demo_arm = torch.ones(1, 7)

    r_w1 = _compute_il_reward(arm, hand, demo_arm, hand.clone(), weight_arm=1.0, weight_hand=0.0)
    r_w5 = _compute_il_reward(arm, hand, demo_arm, hand.clone(), weight_arm=5.0, weight_hand=0.0)
    assert abs(r_w5.item()) == abs(r_w1.item()) * 5.0, (
        f"weight_arm=5 should give 5× penalty: {r_w1.item()}, {r_w5.item()}"
    )


def test_arm_and_hand_rewards_additive() -> None:
    """r_arm + r_hand 합산 검증."""
    arm  = torch.zeros(1, 7)
    hand = torch.zeros(1, 20)
    demo_arm  = torch.ones(1, 7)
    demo_hand = torch.ones(1, 20)

    r_arm_only  = _compute_il_reward(arm, hand, demo_arm, hand.clone(),   weight_arm=5.0, weight_hand=0.0)
    r_hand_only = _compute_il_reward(arm, hand, arm.clone(),  demo_hand,  weight_arm=0.0, weight_hand=2.0)
    r_total     = _compute_il_reward(arm, hand, demo_arm, demo_hand,      weight_arm=5.0, weight_hand=2.0)

    assert torch.allclose(r_arm_only + r_hand_only, r_total), (
        f"total must equal arm+hand: {r_arm_only.item()} + {r_hand_only.item()} ≠ {r_total.item()}"
    )


def test_batch_independent() -> None:
    """각 env(배치 차원)가 독립적으로 계산됨."""
    N = 8
    arm  = torch.randn(N, 7)
    hand = torch.randn(N, 20)
    demo_arm  = torch.randn(N, 7)
    demo_hand = torch.randn(N, 20)

    r_batch = _compute_il_reward(arm, hand, demo_arm, demo_hand)
    for i in range(N):
        r_single = _compute_il_reward(
            arm[i:i+1], hand[i:i+1], demo_arm[i:i+1], demo_hand[i:i+1]
        )
        assert torch.allclose(r_batch[i:i+1], r_single), (
            f"env {i}: batch={r_batch[i]:.4f} vs single={r_single.item():.4f}"
        )
