"""Pour-start reset curriculum geometry (axis 1 of the v3 redesign).

The pour policy was trapped in a deceptive plateau: positioning shaping
(pour_xy + zband + transport + demo) was fully maximizable *without ever
tilting*, and the real reward (beads crossing into the target) sat behind a
wall the policy never explored. The fix is to make the real reward
*reachable*: reset a fraction of envs into a configuration where the source cup
is already tilted over the target opening, so the policy collects ``bead_in``
reward from step one and PPO can propagate that value back toward upright starts.

This module holds only pure-torch geometry so it can be unit-tested without
Isaac Sim. ``compute_pour_start_pose`` rigidly rotates the grasped (cup+hand)
about the cup's pour-point by ``tilt_deg`` toward the target, then translates so
the pour-point lands at ``z_clearance`` above the target opening. Because the
transform is rigid, the finger<->cup grasp relationship is preserved; the caller
then solves Fabrics IK for the rotated palm pose to obtain arm joints.

All inputs/outputs are env-local (env-origin subtracted). Cup quaternions are
``wxyz`` (Isaac convention); palm quaternions are ``xyzw`` (the layout the env's
``palm_pose_targets`` / Fabrics reset use).
"""

from __future__ import annotations

import math

import torch


def _axis_angle_to_quat_wxyz(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """(n,3) unit axis + scalar angle -> (n,4) wxyz quaternion."""
    half = 0.5 * angle
    s = math.sin(half)
    w = torch.full((axis.shape[0], 1), math.cos(half), dtype=axis.dtype, device=axis.device)
    xyz = axis * s
    return torch.cat([w, xyz], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (n,4) wxyz quaternions: a ⊗ b."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate (n,3) vectors v by (n,4) wxyz quaternions q."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def _safe_normalize(v: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(v, dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, v / norm.clamp(min=1e-6), fallback)


def compute_pour_start_pose(
    cup_pos: torch.Tensor,            # (n,3) env-local, upright cup center
    palm_pos: torch.Tensor,           # (n,3) env-local
    palm_quat_xyzw: torch.Tensor,     # (n,4) xyzw
    rim_b: torch.Tensor,              # (3,) pour-point in cup body frame
    target_opening: torch.Tensor,     # (n,3) env-local target opening
    *,
    tilt_deg: float,
    z_clearance: float,
) -> dict[str, torch.Tensor]:
    """Rigidly tip the grasped cup over the target opening.

    Returns dict with cup_pos (n,3), cup_quat_wxyz (n,4), palm_pos (n,3),
    palm_quat_xyzw (n,4).
    """
    n = cup_pos.shape[0]
    device, dtype = cup_pos.device, cup_pos.dtype
    rim_b = rim_b.to(device=device, dtype=dtype).unsqueeze(0).expand(n, -1)

    world_up = torch.zeros(n, 3, device=device, dtype=dtype)
    world_up[:, 2] = 1.0

    # Upright cup => body offset == world offset. Current pour point pivot.
    pour_point0 = cup_pos + rim_b

    # Horizontal direction from pour point toward the target opening.
    to_target = target_opening - pour_point0
    horiz = to_target.clone()
    horiz[:, 2] = 0.0
    fallback_dir = torch.zeros(n, 3, device=device, dtype=dtype)
    fallback_dir[:, 0] = 1.0
    dir_xy = _safe_normalize(horiz, fallback_dir)

    # Axis that tips the cup-up toward the target: cross(up, dir).
    tilt_axis = _safe_normalize(
        torch.linalg.cross(world_up, dir_xy, dim=-1),
        torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(n, -1),
    )
    theta = math.radians(tilt_deg)
    rot_q = _axis_angle_to_quat_wxyz(tilt_axis, theta)  # (n,4) wxyz

    # Target pour-point position.
    p_star = target_opening.clone()
    p_star[:, 2] = p_star[:, 2] + z_clearance

    # Rigid map x' = R*(x - pivot) + p_star  (pour point maps to p_star).
    new_cup_pos = _quat_rotate_wxyz(rot_q, cup_pos - pour_point0) + p_star
    new_palm_pos = _quat_rotate_wxyz(rot_q, palm_pos - pour_point0) + p_star

    # Orientations: world-frame left-multiplication.
    identity_wxyz = torch.zeros(n, 4, device=device, dtype=dtype)
    identity_wxyz[:, 0] = 1.0
    new_cup_quat_wxyz = _quat_mul_wxyz(rot_q, identity_wxyz)  # cup was upright

    palm_quat_wxyz = palm_quat_xyzw[:, [3, 0, 1, 2]]
    new_palm_quat_wxyz = _quat_mul_wxyz(rot_q, palm_quat_wxyz)
    new_palm_quat_xyzw = torch.nn.functional.normalize(
        new_palm_quat_wxyz[:, [1, 2, 3, 0]], dim=-1
    )
    new_cup_quat_wxyz = torch.nn.functional.normalize(new_cup_quat_wxyz, dim=-1)

    return {
        "cup_pos": new_cup_pos,
        "cup_quat_wxyz": new_cup_quat_wxyz,
        "palm_pos": new_palm_pos,
        "palm_quat_xyzw": new_palm_quat_xyzw,
    }


def pour_start_ratio(
    global_step: int,
    start_ratio: float,
    end_ratio: float,
    *,
    anneal_start_step: int,
    anneal_steps: int,
) -> float:
    """Linear anneal of the fraction of envs reset into the pour-start state.

    Holds ``start_ratio`` until ``anneal_start_step``, then linearly moves to
    ``end_ratio`` over ``anneal_steps`` and stays there.
    """
    if global_step <= anneal_start_step:
        return start_ratio
    if anneal_steps <= 0:
        return end_ratio
    frac = min(max((global_step - anneal_start_step) / float(anneal_steps), 0.0), 1.0)
    return start_ratio + (end_ratio - start_ratio) * frac
