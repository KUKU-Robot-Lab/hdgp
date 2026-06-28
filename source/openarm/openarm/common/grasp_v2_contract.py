from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch


GRASP_V2_REWARD_TERMS: tuple[str, ...] = (
    "approach",
    "grasp",
    "lift",
    "stabilize",
    "success_bonus",
    "post_lift_contact_loss",
    "action_smooth",
    "stability",
)

GRASP_V2_COMMON_SCALAR_TAGS: tuple[str, ...] = (
    "phase/approach",
    "phase/grasp",
    "phase/lift",
    "phase/stabilize",
    "reward/approach",
    "reward/grasp",
    "reward/lift",
    "reward/stabilize",
    "reward/success_bonus",
    "reward/post_lift_contact_loss",
    "reward/action_smooth",
    "reward/stability",
    "reward/total",
    "contact/count",
    "contact/full_contact_rate",
    "contact/palm",
    "contact/palm_force",
    "contact/grasp_ready_hold",
    "contact/contacts_at_lift_start",
    "contact/palm_at_lift_start",
    "cup/height_delta",
    "cup/tilt_deg",
    "cup/upright_quality",
    "cup/grasp_tilt_deg",
    "cup/lift_tilt_deg",
    "cup/xy_displacement",
    "task/lifted_rate",
    "task/upright_rate",
    "task/stable_rate",
    "task/cup_lin_vel",
    "task/cup_ang_vel",
    "task/action_delta_norm",
    "task/contact_delta",
    "task/grasp_ready_rate",
    "task/five_tip_contact_rate",
    "task/prelift_five_tip_contact_rate",
    "task/lift_five_tip_contact_rate",
    "task/lift_started_rate",
    "task/lift_success_now",
    "task/stabilize_success_now",
    "task/lift_success_rate",
    "task/stabilize_success_rate",
    "task/success_rate",
    "task/common_success_now",
    "task/success_held_rate",
    "task/success_hold_count",
)

GRASP_V2_COMMON_PREFIXES: tuple[str, ...] = (
    "phase/",
    "reward/",
    "contact/",
    "cup/",
    "task/",
)


@dataclass(frozen=True)
class GraspV2Stability:
    stable: torch.Tensor
    quality: torch.Tensor
    cup_lin_vel_norm: torch.Tensor
    cup_ang_vel_norm: torch.Tensor
    contact_delta: torch.Tensor
    action_delta_norm: torch.Tensor


@dataclass(frozen=True)
class StationaryGraspSuccess:
    success_now: torch.Tensor
    success_held: torch.Tensor
    hold_count: torch.Tensor
    gates: dict[str, torch.Tensor]


def _cfg_float(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: object, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _norm_or_abs(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1:
        return torch.nan_to_num(value.norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(value.abs(), nan=0.0, posinf=0.0, neginf=0.0)


def compute_action_delta_norm(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
) -> torch.Tensor:
    """Return a dimension-normalized action delta for cross-hand comparison."""

    delta = torch.nan_to_num(actions - previous_actions, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt(delta.pow(2).mean(dim=-1).clamp(min=0.0))


def compute_grasp_v2_stability(
    *,
    cup_lin_vel: torch.Tensor,
    cup_ang_vel: torch.Tensor,
    contact_delta: torch.Tensor,
    action_delta_norm: torch.Tensor,
    cfg: object,
) -> GraspV2Stability:
    cup_lin_vel_norm = _norm_or_abs(cup_lin_vel)
    cup_ang_vel_norm = _norm_or_abs(cup_ang_vel)
    contact_delta_abs = _norm_or_abs(contact_delta)
    action_delta_norm = _norm_or_abs(action_delta_norm)

    lin_threshold = _cfg_float(cfg, "stability_cup_lin_vel_threshold", 0.04)
    ang_threshold = _cfg_float(cfg, "stability_cup_ang_vel_threshold", 0.5)
    contact_threshold = _cfg_float(cfg, "stability_contact_delta_threshold", 1.0)
    action_threshold = _cfg_float(cfg, "stability_action_delta_threshold", 0.2)

    # action_delta_norm은 stochastic하게 샘플된 raw action의 RMS(aₜ−aₜ₋₁)라
    # 탐색 노이즈 바닥(σ·√2 ≈ 0.5, σ≈0.44@entropy16)이 임계(0.2/0.4)를 항상 초과한다.
    # → 정지 정책도 학습 중 stable 통과 불가(success_held=0의 근본원인). palm freeze도 무효.
    # 물리적 정지는 cup_lin_vel/cup_ang_vel이 노이즈 없이 직접 측정하므로 hard gate에서 제외.
    # action_delta_norm은 아래 quality(reward shaping)에 남겨 stillness gradient는 유지한다.
    stable = (
        (cup_lin_vel_norm <= lin_threshold)
        & (cup_ang_vel_norm <= ang_threshold)
        & (contact_delta_abs <= contact_threshold)
    )
    quality = (
        torch.exp(-cup_lin_vel_norm / max(lin_threshold, 1e-6))
        * torch.exp(-cup_ang_vel_norm / max(ang_threshold, 1e-6))
        * torch.exp(-contact_delta_abs / max(contact_threshold, 1e-6))
        * torch.exp(-action_delta_norm / max(action_threshold, 1e-6))
    )
    quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
    return GraspV2Stability(
        stable=stable,
        quality=quality,
        cup_lin_vel_norm=cup_lin_vel_norm,
        cup_ang_vel_norm=cup_ang_vel_norm,
        contact_delta=contact_delta_abs,
        action_delta_norm=action_delta_norm,
    )


def compute_stationary_grasp_success(
    *,
    stabilize_started: torch.Tensor,
    cup_height_delta: torch.Tensor,
    full_contact: torch.Tensor,
    cup_tilt_deg: torch.Tensor,
    stable: torch.Tensor,
    previous_success_hold_count: torch.Tensor,
    cfg: object,
) -> StationaryGraspSuccess:
    # Phase N2: success lifted 게이트도 success_lift_height(있으면)로 — lift 보상 saturation
    # (lift_success_height)과 분리해 컵이 stabilize서 문턱 위 진동/슬립해도 success가 발화하도록 margin.
    # RH56F1은 success_lift_height 미정의라 fallback으로 lift_success_height(4cm) 유지.
    lifted = cup_height_delta >= _cfg_float(
        cfg, "success_lift_height", _cfg_float(cfg, "lift_success_height", 0.04)
    )
    full_contact_bool = full_contact.bool()
    upright = cup_tilt_deg <= _cfg_float(cfg, "stabilize_upright_max_deg", 5.0)
    stable_bool = stable.bool()
    success_now = (
        stabilize_started.bool()
        & lifted
        & full_contact_bool
        & upright
        & stable_bool
    )
    hold_count = torch.where(
        success_now,
        previous_success_hold_count + 1,
        torch.zeros_like(previous_success_hold_count),
    )
    success_held = hold_count >= _cfg_int(cfg, "success_hold_steps", 30)
    gates = {
        "lifted": lifted.float(),
        "full_contact": full_contact_bool.float(),
        "upright": upright.float(),
        "stable": stable_bool.float(),
        "success_now": success_now.float(),
        "success_held": success_held.float(),
    }
    return StationaryGraspSuccess(
        success_now=success_now,
        success_held=success_held,
        hold_count=hold_count,
        gates=gates,
    )


def clear_grasp_v2_common_scalars(extras: MutableMapping[str, torch.Tensor]) -> None:
    for tag in list(extras):
        if tag.startswith(GRASP_V2_COMMON_PREFIXES):
            extras.pop(tag, None)


def log_grasp_v2_common_scalars(
    extras: MutableMapping[str, torch.Tensor],
    values: Mapping[str, torch.Tensor],
) -> None:
    """Replace all common comparison scalars with the canonical v2 tag set."""

    clear_grasp_v2_common_scalars(extras)
    device = None
    for value in values.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break
    zero = torch.zeros((), device=device)
    for tag in GRASP_V2_COMMON_SCALAR_TAGS:
        extras[tag] = values.get(tag, zero)
