from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from openarm.common.grasp_reward_core import compute_grasp_reward_terms
from openarm.common.grasp_v2_contract import (
    GRASP_V2_COMMON_SCALAR_TAGS,
    GRASP_V2_REWARD_TERMS,
    compute_action_delta_norm,
    compute_grasp_v2_stability,
    compute_stationary_grasp_success,
)


@dataclass
class RewardCfg:
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    lift_reward_weight: float = 30.0
    stabilize_weight: float = 10.0
    stability_reward_weight: float = 1.0
    success_bonus_weight: float = 20.0
    action_smooth_weight: float = -0.02
    post_lift_contact_loss_weight: float = -8.0
    lift_success_height: float = 0.04
    success_upright_max_deg: float = 20.0
    stabilize_upright_max_deg: float = 5.0
    success_hold_steps: int = 30
    grasp_xy_threshold: float = 0.025
    grasp_upright_threshold_deg: float = 8.0
    stabilize_action_sharpness: float = 1.5
    stability_cup_lin_vel_threshold: float = 0.04
    stability_cup_ang_vel_threshold: float = 0.5
    stability_contact_delta_threshold: float = 1.0
    stability_action_delta_threshold: float = 0.2


def _success_inputs(cfg: RewardCfg, **overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    values = {
        "stabilize_started": torch.ones(1, dtype=torch.bool),
        "cup_height_delta": torch.tensor([cfg.lift_success_height]),
        "full_contact": torch.ones(1, dtype=torch.bool),
        "cup_tilt_deg": torch.tensor([cfg.stabilize_upright_max_deg]),
        "stable": torch.ones(1, dtype=torch.bool),
        "previous_success_hold_count": torch.zeros(1, dtype=torch.long),
    }
    values.update(overrides)
    return values


def test_success_gate_requires_stabilize_phase_and_stability() -> None:
    cfg = RewardCfg()

    high_velocity = compute_stationary_grasp_success(
        cfg=cfg,
        **_success_inputs(cfg, stable=torch.zeros(1, dtype=torch.bool)),
    )
    before_stabilize = compute_stationary_grasp_success(
        cfg=cfg,
        **_success_inputs(cfg, stabilize_started=torch.zeros(1, dtype=torch.bool)),
    )

    assert high_velocity.success_now.tolist() == [False]
    assert high_velocity.success_held.tolist() == [False]
    assert before_stabilize.success_now.tolist() == [False]
    assert "at_goal" not in before_stabilize.gates


def test_success_gate_holds_for_thirty_stable_steps() -> None:
    cfg = RewardCfg()
    hold_count = torch.zeros(1, dtype=torch.long)

    for _ in range(cfg.success_hold_steps - 1):
        success = compute_stationary_grasp_success(
            cfg=cfg,
            **_success_inputs(cfg, previous_success_hold_count=hold_count),
        )
        hold_count = success.hold_count
        assert success.success_now.tolist() == [True]
        assert success.success_held.tolist() == [False]

    success = compute_stationary_grasp_success(
        cfg=cfg,
        **_success_inputs(cfg, previous_success_hold_count=hold_count),
    )

    assert success.hold_count.item() == cfg.success_hold_steps
    assert success.success_held.tolist() == [True]


def test_stability_gate_uses_physical_velocity_not_noisy_action_delta() -> None:
    # Phase D: action_delta_norm은 stochastic sampled action의 RMS라 탐색 노이즈
    # 바닥(σ·√2 ≈ 0.5)이 임계를 항상 초과 → stable hard-gate에서 제외됐다.
    # 물리적 정지는 cup_lin_vel/cup_ang_vel만으로 판정한다.
    cfg = RewardCfg()
    stability = compute_grasp_v2_stability(
        cup_lin_vel=torch.tensor([[0.01, 0.01, 0.01], [0.10, 0.0, 0.0], [0.01, 0.01, 0.01]]),
        cup_ang_vel=torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
        contact_delta=torch.tensor([0.0, 0.0, 0.0]),
        # 3번째 케이스: action_delta 큼(0.9)이지만 cup 정지 → stable=True (게이트 제외 증명)
        action_delta_norm=torch.tensor([0.05, 0.05, 0.9]),
        cfg=cfg,
    )

    assert stability.stable.tolist() == [True, False, True]
    assert stability.cup_lin_vel_norm[1] > cfg.stability_cup_lin_vel_threshold


def test_action_delta_norm_is_dimension_normalized() -> None:
    prev_12d = torch.zeros(1, 12)
    cur_12d = torch.ones(1, 12) * 0.1
    prev_27d = torch.zeros(1, 27)
    cur_27d = torch.ones(1, 27) * 0.1

    assert torch.allclose(
        compute_action_delta_norm(cur_12d, prev_12d),
        compute_action_delta_norm(cur_27d, prev_27d),
    )


def test_reward_terms_are_exact_common_v2_contract() -> None:
    cfg = RewardCfg()
    total, terms, gates = compute_grasp_reward_terms(
        num_tip_contacts=torch.tensor([5]),
        tip_contact_frac=torch.ones(1),
        full_tip_contact=torch.ones(1),
        contact_persistence_frac=torch.ones(1),
        palm_to_cup_dist=torch.zeros(1),
        fingertip_side_dist=torch.zeros(1),
        cup_height_delta=torch.tensor([0.06]),
        cup_xy_displacement=torch.zeros(1),
        cup_tilt_deg=torch.zeros(1),
        upright_quality=torch.ones(1),
        lift_latched=torch.ones(1, dtype=torch.bool),
        action_delta_norm=torch.zeros(1),
        stabilize_reward_gate=torch.ones(1, dtype=torch.bool),
        success_now=torch.ones(1, dtype=torch.bool),
        stable=torch.ones(1, dtype=torch.bool),
        stability_quality=torch.ones(1),
        cfg=cfg,
    )

    assert set(terms) == set(GRASP_V2_REWARD_TERMS)
    assert set(terms) == {
        "approach",
        "grasp",
        "lift",
        "stabilize",
        "success_bonus",
        "post_lift_contact_loss",
        "action_smooth",
        "stability",
    }
    assert terms["stability"].item() > 0.0
    assert gates["success_now"].item() == 1.0
    assert total.item() > 0.0


def test_target_env_sources_use_common_v2_helpers_and_common_tags() -> None:
    root = Path(__file__).resolve().parents[2]
    tesollo_env = (
        root / "tesollo/right/grasp_v10_3/grasp_right_env.py"
    ).read_text(encoding="utf-8")
    rh_env = (
        root / "rh56f1/right/grasp_v1/grasp_right_env.py"
    ).read_text(encoding="utf-8")
    tesollo_cfg = (
        root / "tesollo/right/grasp_v10_3/grasp_right_env_cfg.py"
    ).read_text(encoding="utf-8")
    rh_cfg = (
        root / "rh56f1/right/grasp_v1/grasp_right_env_cfg.py"
    ).read_text(encoding="utf-8")

    for cfg in (tesollo_cfg, rh_cfg):
        assert "episode_length_s: float = 10.0" in cfg
        assert "lift_success_height: float = 0.04" in cfg
        assert "success_hold_steps: int = 30" in cfg
        assert "transport_goal_dist_threshold" not in cfg
        assert "stabilize_upright_max_deg: float = 12.0" in cfg
        assert "stability_cup_ang_vel_threshold: float = 0.5" in cfg
        assert "stability_contact_delta_threshold: float = 1.0" in cfg
        assert "stability_action_delta_threshold: float = 0.4" in cfg

    # Phase G: cup_lin_vel 임계는 per-task로 분기 (의도된 divergence).
    # TESOLLO는 잡힌 컵 잔류속도 ~0.045가 0.04를 넘겨 stable 깜빡임 → 0.06 완화.
    # RH56F1은 미검증이라 0.04 유지 (v1 정렬 보류).
    assert "stability_cup_lin_vel_threshold: float = 0.06" in tesollo_cfg
    assert "stability_cup_lin_vel_threshold: float = 0.04" in rh_cfg

    for env in (tesollo_env, rh_env):
        assert "compute_grasp_v2_stability(" in env
        assert "compute_stationary_grasp_success(" in env
        assert "compute_action_delta_norm(" in env
        assert "log_grasp_v2_common_scalars(" in env
        assert "transport_goal" not in env
        assert 'self.extras["reward/hand_residual_magnitude"]' not in env
        for tag in GRASP_V2_COMMON_SCALAR_TAGS:
            assert f'"{tag}"' in env
