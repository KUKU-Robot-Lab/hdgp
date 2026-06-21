from __future__ import annotations

import torch


def _quat_multiply_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def compose_incremental_palm_pose(
    current_pose: torch.Tensor,
    delta_position: torch.Tensor,
    delta_rotation_vector: torch.Tensor,
    position_mins: torch.Tensor,
    position_maxs: torch.Tensor,
) -> torch.Tensor:
    """Compose a bounded world-frame delta with an XYZW palm pose."""

    target = current_pose.clone()
    target[:, :3] = torch.max(
        torch.min(current_pose[:, :3] + delta_position, position_maxs.unsqueeze(0)),
        position_mins.unsqueeze(0),
    )
    angle = delta_rotation_vector.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle.clamp(min=1e-8),
        torch.full_like(angle, 0.5),
    )
    delta_quat = torch.cat(
        (delta_rotation_vector * scale, torch.cos(half_angle)), dim=-1
    )
    target[:, 3:7] = torch.nn.functional.normalize(
        _quat_multiply_xyzw(delta_quat, current_pose[:, 3:7]), dim=-1, eps=1e-6
    )
    return target


def compute_lift_stabilize_palm_targets(
    episode_length_buf: torch.Tensor,
    grasp_palm_pose: torch.Tensor,
    lift_start_pose: torch.Tensor,
    stabilize_delta: torch.Tensor,
    palm_mins: torch.Tensor,
    palm_maxs: torch.Tensor,
    lift_start_step: int,
    stabilize_start_step: int,
    lift_raise_steps: int,
    lift_raise_z_delta: float,
) -> torch.Tensor:
    is_lift_or_stabilize = episode_length_buf >= int(lift_start_step)
    is_stabilize = episode_length_buf >= int(stabilize_start_step)

    raise_progress = (
        (episode_length_buf - int(lift_start_step)).clamp(min=0).float()
        / max(int(lift_raise_steps), 1)
    ).clamp(max=1.0)

    raise_pose = lift_start_pose.clone()
    raise_pose[:, 2] = raise_pose[:, 2] + float(lift_raise_z_delta) * raise_progress

    stabilize_anchor = lift_start_pose.clone()
    stabilize_anchor[:, 2] = stabilize_anchor[:, 2] + float(lift_raise_z_delta)
    stabilize_pose = stabilize_anchor + stabilize_delta
    stabilize_pose[:, 2] = stabilize_anchor[:, 2]

    lift_pose = torch.where(is_stabilize.unsqueeze(1), stabilize_pose, raise_pose)
    lift_pose = torch.max(torch.min(lift_pose, palm_maxs.unsqueeze(0)), palm_mins.unsqueeze(0))

    return torch.where(is_lift_or_stabilize.unsqueeze(1), lift_pose, grasp_palm_pose)
