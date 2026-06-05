from __future__ import annotations

import torch

from .pour_gate_success import compute_gate_terms
from .pour_utils import ACTOR_OBSERVATION_DIM, assemble_actor_observation


def compute_pour_observation_metrics(
    *,
    cup_pos_w: torch.Tensor,
    source_pour_point_w: torch.Tensor,
    target_opening_w: torch.Tensor,
    source_up_axis_w: torch.Tensor,
    cfg,
) -> dict[str, torch.Tensor]:
    mouth_delta = target_opening_w - source_pour_point_w
    mouth_distance = torch.norm(mouth_delta, dim=-1)
    mouth_xy_distance = torch.norm(mouth_delta[:, :2], dim=-1)
    mouth_z_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
    source_up_dot_world = source_up_axis_w[:, 2].clamp(-1.0, 1.0)

    mouth_dir_xy = mouth_delta[:, :2] / mouth_delta[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mouth_tilt_dir_xy = source_up_axis_w[:, :2]
    mouth_tilt_dir_xy = mouth_tilt_dir_xy / mouth_tilt_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    directional_tilt_cos = (mouth_tilt_dir_xy * mouth_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)

    mouth_dir = mouth_delta / mouth_distance.unsqueeze(1).clamp(min=1e-6)
    pour_heading_xy = source_up_axis_w[:, :2]
    pour_heading_xy_norm = pour_heading_xy.norm(dim=-1, keepdim=True)
    effective_heading_xy = torch.where(
        pour_heading_xy_norm > 1e-4,
        pour_heading_xy / pour_heading_xy_norm.clamp(min=1e-6),
        torch.zeros_like(pour_heading_xy),
    )
    effective_pour_heading = torch.cat(
        [effective_heading_xy, torch.zeros(source_up_axis_w.shape[0], 1, device=source_up_axis_w.device)],
        dim=-1,
    )
    mouth_alignment_cos = (effective_pour_heading * mouth_dir).sum(dim=-1).clamp(-1.0, 1.0)

    cup_center_xy_dist = torch.norm(cup_pos_w[:, :2] - target_opening_w[:, :2], dim=-1)
    gate_terms = compute_gate_terms(
        cfg=cfg,
        mouth_xy_distance=mouth_xy_distance,
        mouth_z_clearance=mouth_z_clearance,
        directional_tilt_cos=directional_tilt_cos,
    )

    return {
        "mouth_delta": mouth_delta,
        "mouth_distance": mouth_distance,
        "mouth_xy_distance": mouth_xy_distance,
        "mouth_z_clearance": mouth_z_clearance,
        "source_up_dot_world": source_up_dot_world,
        "directional_tilt_cos": directional_tilt_cos,
        "mouth_alignment_cos": mouth_alignment_cos,
        "cup_center_xy_dist": cup_center_xy_dist,
        **gate_terms,
    }


def build_actor_observation(
    *,
    right_joint_pos: torch.Tensor,
    right_joint_vel: torch.Tensor,
    left_arm_joint_pos: torch.Tensor,
    left_arm_joint_vel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    cup_pos_w: torch.Tensor,
    cup_quat_w: torch.Tensor,
    cup_lin_vel_w: torch.Tensor,
    cup_ang_vel_w: torch.Tensor,
    target_opening_w: torch.Tensor,
    bead_centroid_w: torch.Tensor,
    env_origins: torch.Tensor,
    prev_actions: torch.Tensor,
    mouth_delta: torch.Tensor,
    mouth_xy_distance: torch.Tensor,
    mouth_z_clearance: torch.Tensor,
    source_up_dot_world: torch.Tensor,
    directional_tilt_cos: torch.Tensor,
    mouth_alignment_cos: torch.Tensor,
    bead_cross_fraction: torch.Tensor,
    bead_in_target_fraction: torch.Tensor,
    bead_in_source_fraction: torch.Tensor,
    spill_ratio: torch.Tensor,
    g_ready: torch.Tensor,
    g_pour: torch.Tensor,
    num_observations: int,
) -> torch.Tensor:
    target_opening_local = target_opening_w - env_origins
    bead_centroid_local = bead_centroid_w - env_origins
    fingertip_pos_flat = fingertip_pos.reshape(fingertip_pos.shape[0], -1)
    cup_pose_and_vel = torch.cat(
        [cup_pos_w, cup_quat_w, cup_lin_vel_w, cup_ang_vel_w],
        dim=-1,
    )
    actor_obs = assemble_actor_observation(
        right_joint_pos=right_joint_pos,
        right_joint_vel=right_joint_vel,
        left_arm_joint_pos=left_arm_joint_pos,
        left_arm_joint_vel=left_arm_joint_vel,
        fingertip_pos=fingertip_pos_flat,
        cup_pose_vel=cup_pose_and_vel,
        target_opening_pos=target_opening_local,
        bead_centroid_pos=bead_centroid_local,
        prev_actions=prev_actions,
        mouth_delta=mouth_delta,
        mouth_xy_distance=mouth_xy_distance.unsqueeze(-1),
        mouth_z_clearance=mouth_z_clearance.unsqueeze(-1),
        source_up_dot_world=source_up_dot_world.unsqueeze(-1),
        directional_tilt_cos=directional_tilt_cos.unsqueeze(-1),
        mouth_alignment_cos=mouth_alignment_cos.unsqueeze(-1),
        bead_cross_fraction=bead_cross_fraction.unsqueeze(-1),
        bead_in_target_fraction=bead_in_target_fraction.unsqueeze(-1),
        bead_in_source_fraction=bead_in_source_fraction.unsqueeze(-1),
        spill_ratio=spill_ratio.unsqueeze(-1),
        g_ready=g_ready.unsqueeze(-1),
        g_pour=g_pour.unsqueeze(-1),
    )
    if num_observations != ACTOR_OBSERVATION_DIM:
        raise RuntimeError(
            f"[pour_v1] cfg obs dim mismatch: {num_observations} != {ACTOR_OBSERVATION_DIM}"
        )
    if actor_obs.shape[1] != num_observations:
        raise RuntimeError(
            f"[pour_v1] actor obs dim mismatch: {actor_obs.shape[1]} != {num_observations}"
        )
    return actor_obs
