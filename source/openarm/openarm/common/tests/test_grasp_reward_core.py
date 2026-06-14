from __future__ import annotations

from dataclasses import dataclass

import torch

from openarm.common.grasp_reward_core import compute_grasp_reward_terms


@dataclass
class RewardCfg:
    approach_weight: float = 2.0
    approach_sharpness: float = 8.0
    approach_xy_penalty_weight: float = 5.0
    approach_tilt_penalty_weight: float = 0.08
    grasp_weight: float = 12.0
    lift_reward_weight: float = 20.0
    stabilize_weight: float = 6.0
    success_bonus_weight: float = 20.0
    action_smooth_weight: float = -0.02
    lift_success_height: float = 0.04
    success_upright_max_deg: float = 20.0
    grasp_xy_threshold: float = 0.025
    stabilize_spawn_xy_scale: float = 0.03
    grasp_upright_threshold_deg: float = 8.0
    stabilize_action_sharpness: float = 1.5
    post_lift_contact_loss_weight: float = -8.0


def _reward(
    contacts: torch.Tensor,
    *,
    lifted: bool = False,
    tilted: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    height = 0.06 if lifted else 0.0
    tilt = 25.0 if tilted else 0.0
    return compute_grasp_reward_terms(
        num_tip_contacts=contacts,
        tip_contact_frac=contacts.float() / 5.0,
        full_tip_contact=(contacts >= 5).float(),
        palm_to_cup_dist=torch.full_like(contacts.float(), 0.02),
        fingertip_side_dist=torch.full_like(contacts.float(), 0.02),
        cup_height_delta=torch.full_like(contacts.float(), height),
        cup_xy_displacement=torch.zeros_like(contacts.float()),
        cup_tilt_deg=torch.full_like(contacts.float(), tilt),
        upright_quality=torch.exp(-torch.full_like(contacts.float(), tilt) / 10.0),
        lift_latched=torch.full_like(contacts, lifted, dtype=torch.bool),
        action_delta_norm=torch.zeros_like(contacts.float()),
        cfg=RewardCfg(),
    )


def test_grasp_reward_orders_zero_four_and_five_contacts_before_lift() -> None:
    total, terms, gates = _reward(torch.tensor([0, 4, 5], dtype=torch.long))

    assert terms["grasp"][0] < terms["grasp"][1] < terms["grasp"][2]
    assert total[0] < total[1] < total[2]
    assert gates["success_now"].tolist() == [0.0, 0.0, 0.0]


def test_lift_and_success_require_full_tip_contact_when_lifted() -> None:
    total, terms, gates = _reward(torch.tensor([4, 5], dtype=torch.long), lifted=True)

    assert terms["lift"][0].item() == 0.0
    assert terms["lift"][1].item() > 0.0
    assert terms["post_lift_contact_loss"][0].item() < 0.0
    assert terms["post_lift_contact_loss"][1].item() == 0.0
    assert terms["success_bonus"][0].item() == 0.0
    assert terms["success_bonus"][1].item() > 0.0
    assert gates["success_now"].tolist() == [0.0, 1.0]
    assert total[1] > total[0]


def test_tilted_lift_loses_success_bonus_and_upright_quality() -> None:
    upright_total, upright_terms, upright_gates = _reward(
        torch.tensor([5], dtype=torch.long),
        lifted=True,
    )
    tilted_total, tilted_terms, tilted_gates = _reward(
        torch.tensor([5], dtype=torch.long),
        lifted=True,
        tilted=True,
    )

    assert tilted_terms["lift"].item() < upright_terms["lift"].item()
    assert tilted_gates["success_now"].item() == 0.0
    assert upright_gates["success_now"].item() == 1.0
    assert tilted_total.item() < upright_total.item()


def test_stabilize_reward_prefers_spawn_xy_recovery() -> None:
    contacts = torch.tensor([5, 5], dtype=torch.long)
    total, terms, gates = compute_grasp_reward_terms(
        num_tip_contacts=contacts,
        tip_contact_frac=torch.ones(2),
        full_tip_contact=torch.ones(2),
        palm_to_cup_dist=torch.zeros(2),
        fingertip_side_dist=torch.zeros(2),
        cup_height_delta=torch.full((2,), 0.06),
        cup_xy_displacement=torch.tensor([0.01, 0.05]),
        cup_tilt_deg=torch.zeros(2),
        upright_quality=torch.ones(2),
        lift_latched=torch.ones(2, dtype=torch.bool),
        action_delta_norm=torch.zeros(2),
        cfg=RewardCfg(),
    )

    assert gates["spawn_xy_quality"][0] > gates["spawn_xy_quality"][1]
    assert terms["stabilize"][0] > terms["stabilize"][1]
    assert terms["success_bonus"][0] > terms["success_bonus"][1]
    assert total[0] > total[1]
