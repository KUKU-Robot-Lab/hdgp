from __future__ import annotations

import torch


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
