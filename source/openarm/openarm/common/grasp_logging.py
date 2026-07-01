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
    per_joint: bool = False,
    prefix: str = "joint_state",
) -> dict[str, torch.Tensor]:
    """Summarize arm/finger joint state for training-time observability.

    Velocity is the trembling signal of interest, so both mean and max of the
    absolute velocity are exposed alongside the position mean. ``per_joint``
    additionally exposes each joint's position/velocity mean so a single drifting
    joint (e.g. thumb pushed backward) is visible without a render.
    """
    out = {
        f"{prefix}/arm_pos_mean": arm_pos.mean(),
        f"{prefix}/arm_vel_abs_mean": arm_vel.abs().mean(),
        f"{prefix}/arm_vel_abs_max": arm_vel.abs().max(),
        f"{prefix}/finger_pos_mean": finger_pos.mean(),
        f"{prefix}/finger_vel_abs_mean": finger_vel.abs().mean(),
        f"{prefix}/finger_vel_abs_max": finger_vel.abs().max(),
    }
    if per_joint:
        for i in range(arm_pos.shape[1]):
            out[f"{prefix}/arm/j{i + 1}_pos"] = arm_pos[:, i].mean()
            out[f"{prefix}/arm/j{i + 1}_vel_abs"] = arm_vel[:, i].abs().mean()
        for i in range(finger_pos.shape[1]):
            out[f"{prefix}/finger/q{i}_pos"] = finger_pos[:, i].mean()
            out[f"{prefix}/finger/q{i}_vel_abs"] = finger_vel[:, i].abs().mean()
    return out


def action_policy_scalars(
    *,
    action: torch.Tensor,                 # (N, A) raw policy output ~[-1,1]
    prev_action: torch.Tensor | None = None,
    palm_dims: int = 6,
    palm_names: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz"),
    prefix: str = "action_policy",
) -> dict[str, torch.Tensor]:
    """Expose the raw policy action per component for training-time observability.

    palm(앞 palm_dims) = 6D pose action(x/y/z/rx/ry/rz), 나머지 = finger 명령.
    각 차원 평균 + 그룹 abs_mean/norm + step delta. embodiment-agnostic
    (차원 수는 tensor shape에서 읽음). RH56F1·Teosllo grasp 공용.
    """
    out: dict[str, torch.Tensor] = {}
    num = action.shape[1]
    pd = min(int(palm_dims), num)
    for i in range(pd):
        nm = palm_names[i] if i < len(palm_names) else f"p{i}"
        out[f"{prefix}/palm/{nm}_mean"] = action[:, i].mean()
    out[f"{prefix}/palm/abs_mean"] = action[:, :pd].abs().mean()
    if num > pd:
        finger = action[:, pd:]
        for j in range(finger.shape[1]):
            out[f"{prefix}/finger/f{j + 1}_mean"] = finger[:, j].mean()
        out[f"{prefix}/finger/abs_mean"] = finger.abs().mean()
    out[f"{prefix}/all/abs_mean"] = action.abs().mean()
    out[f"{prefix}/all/norm_mean"] = action.norm(dim=-1).mean()
    if prev_action is not None:
        out[f"{prefix}/all/delta_abs_mean"] = (action - prev_action).abs().mean()
    return out
