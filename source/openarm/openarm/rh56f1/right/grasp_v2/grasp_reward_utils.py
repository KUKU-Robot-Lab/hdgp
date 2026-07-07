from __future__ import annotations

import math

import torch


def compute_upright_success_mask(
    cup_z_cos: torch.Tensor,
    threshold_deg: float,
) -> torch.Tensor:
    threshold_cos = math.cos(math.radians(float(threshold_deg)))
    return cup_z_cos >= threshold_cos


def compute_middle_contact_gate(
    middle_binary_contact: torch.Tensor,
    min_middle_contacts: int,
) -> torch.Tensor:
    if int(min_middle_contacts) <= 0:
        return torch.ones(
            middle_binary_contact.shape[0],
            dtype=torch.bool,
            device=middle_binary_contact.device,
        )
    return middle_binary_contact.sum(dim=-1) >= int(min_middle_contacts)


def compute_slip_proxy(
    cup_xy_velocity: torch.Tensor,
    cup_tilt_delta_deg: torch.Tensor,
    contact_delta_abs: torch.Tensor,
    middle_contact_delta_abs: torch.Tensor,
    xy_velocity_scale: float,
    tilt_delta_scale: float,
    contact_delta_scale: float,
    middle_contact_delta_scale: float,
    contact_delta_weight: float,
    middle_contact_delta_weight: float,
    tilt_delta_weight: float,
) -> torch.Tensor:
    xy_term = cup_xy_velocity / max(float(xy_velocity_scale), 1e-6)
    tilt_term = cup_tilt_delta_deg / max(float(tilt_delta_scale), 1e-6)
    contact_term = contact_delta_abs / max(float(contact_delta_scale), 1e-6)
    middle_contact_term = middle_contact_delta_abs / max(float(middle_contact_delta_scale), 1e-6)
    proxy = (
        xy_term
        + float(tilt_delta_weight) * tilt_term
        + float(contact_delta_weight) * contact_term
        + float(middle_contact_delta_weight) * middle_contact_term
    )
    return torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)


def compute_lift_readiness(
    num_contacts: torch.Tensor,
    is_grasp_phase: torch.Tensor,
    previous_hold_count: torch.Tensor,
    previous_latched: torch.Tensor,
    min_contacts: int,
    hold_steps: int,
    num_firm_fingers: torch.Tensor | None = None,
    min_firm_fingers: int = 0,
    timeout_reached: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """접촉(손끝) + firm(tip&근위) + hold 로 lift 진입 판정.

    Exp2: min_firm_fingers>0 이면 'tip&근위 두 점 접촉 손가락 수 >= min_firm_fingers'를
    AND 게이트로 추가 → 리프트 전 firm 그립 강제(손끝만으로 일찍 리프트 차단).
    timeout_reached: firm 미형성이 오래되면 리프트 허용(dead episode 방지 fallback).
    """
    lift_contact_now = num_contacts >= int(min_contacts)
    if num_firm_fingers is not None and int(min_firm_fingers) > 0:
        lift_contact_now = lift_contact_now & (
            num_firm_fingers >= int(min_firm_fingers)
        )
    next_hold_count = torch.where(
        lift_contact_now & is_grasp_phase,
        previous_hold_count + 1,
        torch.where(
            previous_latched,
            previous_hold_count,
            torch.zeros_like(previous_hold_count),
        ),
    )
    lift_contact_ready_now = next_hold_count >= int(hold_steps)
    if timeout_reached is not None:
        lift_contact_ready_now = lift_contact_ready_now | (
            timeout_reached & (num_contacts >= int(min_contacts)) & is_grasp_phase
        )
    next_latched = previous_latched | lift_contact_ready_now
    return next_hold_count, lift_contact_ready_now, next_latched


def compute_late_grasp_full_grip_mask(
    num_contacts: torch.Tensor,
    is_grasp_phase: torch.Tensor,
    episode_length_buf: torch.Tensor,
    grasp_phase_steps: int,
    contact_threshold: int,
    progress_threshold: float,
) -> torch.Tensor:
    grasp_progress = (
        episode_length_buf.float() / max(float(grasp_phase_steps), 1.0)
    ).clamp(min=0.0, max=1.0)
    return is_grasp_phase & (
        (num_contacts >= int(contact_threshold))
        | (grasp_progress >= float(progress_threshold))
    )


def compute_transport_success_mask(
    goal_dist: torch.Tensor,
    upright_success: torch.Tensor,
    contact_grasped: torch.Tensor,
    middle_grasped: torch.Tensor,
    no_slip: torch.Tensor,
    goal_dist_threshold: float,
) -> torch.Tensor:
    at_goal = goal_dist <= float(goal_dist_threshold)
    return at_goal & upright_success & contact_grasped & middle_grasped & no_slip
