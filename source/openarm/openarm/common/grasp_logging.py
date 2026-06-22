"""Shared TensorBoard logging helpers for grasp tasks.

Pure functions that turn per-finger contact buffers and joint states into
flat ``{tag: scalar_tensor}`` dicts. They are embodiment-agnostic: the number
of fingers/sensors is read from tensor shapes, so the same helper works for
RH56F1 and Teosllo. No observation/action tensors are touched — these only
add scalars to ``self.extras``.
"""

from __future__ import annotations

import torch


def per_finger_contact_scalars(
    *,
    tip_force: torch.Tensor,             # (N, F) per-fingertip force magnitude
    tip_binary: torch.Tensor,            # (N, F) bool per-fingertip contact
    distal_force: torch.Tensor | None = None,   # (N, D)
    middle_force: torch.Tensor | None = None,   # (N, M)
    palm_force: torch.Tensor | None = None,      # (N,)
    prefix: str = "contact",
) -> dict[str, torch.Tensor]:
    """Expose each fingertip/phalanx sensor as its own TB scalar.

    Aggregate tags (``contact/count`` etc.) stay where they are; this adds the
    per-index breakdown so per-finger slip/contact can be read individually.
    """
    out: dict[str, torch.Tensor] = {}

    num_tips = tip_force.shape[1]
    for i in range(num_tips):
        out[f"{prefix}/tip_force_{i + 1}"] = tip_force[:, i].mean()
        out[f"{prefix}/tip_contact_{i + 1}"] = tip_binary[:, i].float().mean()

    if distal_force is not None:
        for i in range(distal_force.shape[1]):
            out[f"{prefix}/distal_force_{i + 1}"] = distal_force[:, i].mean()

    if middle_force is not None:
        for i in range(middle_force.shape[1]):
            out[f"{prefix}/middle_force_{i + 1}"] = middle_force[:, i].mean()

    if palm_force is not None:
        out[f"{prefix}/palm_force_scalar"] = palm_force.mean()

    return out


def joint_state_scalars(
    *,
    arm_pos: torch.Tensor,      # (N, A)
    arm_vel: torch.Tensor,      # (N, A)
    finger_pos: torch.Tensor,   # (N, H)
    finger_vel: torch.Tensor,   # (N, H)
    prefix: str = "debug/joint",
) -> dict[str, torch.Tensor]:
    """Summarize arm/finger joint state for training-time observability.

    Velocity is the trembling signal of interest, so both mean and max of the
    absolute velocity are exposed alongside the position mean.
    """
    return {
        f"{prefix}/arm_pos_mean": arm_pos.mean(),
        f"{prefix}/arm_vel_abs_mean": arm_vel.abs().mean(),
        f"{prefix}/arm_vel_abs_max": arm_vel.abs().max(),
        f"{prefix}/finger_pos_mean": finger_pos.mean(),
        f"{prefix}/finger_vel_abs_mean": finger_vel.abs().mean(),
        f"{prefix}/finger_vel_abs_max": finger_vel.abs().max(),
    }
