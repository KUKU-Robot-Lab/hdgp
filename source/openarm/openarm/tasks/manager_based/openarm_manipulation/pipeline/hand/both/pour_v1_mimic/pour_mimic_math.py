from __future__ import annotations

import torch

RIGHT_PALM_DELTA_XYZ_SCALE = 0.30
RIGHT_PALM_DELTA_ROT_SCALE = 0.30
LEFT_ARM_DELTA_JOINT_SCALE = 0.10


def _skew(vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vec[..., 0])
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert batched axis-angle vectors to rotation matrices."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = torch.where(angle > 1e-8, rotvec / angle.clamp(min=1e-8), torch.zeros_like(rotvec))
    k = _skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[:-1] + (3, 3))
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * k + (1.0 - cos) * torch.matmul(k, k)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to axis-angle vectors."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    vee = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    scale = 0.5 / torch.sin(angle).unsqueeze(-1).clamp(min=1e-8)
    axis = vee * scale
    return torch.where(angle.unsqueeze(-1) > 1e-6, axis * angle.unsqueeze(-1), 0.5 * vee)


def quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)
    two = quat.new_tensor(2.0)
    return torch.stack(
        (
            torch.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1),
            torch.stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), dim=-1),
            torch.stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def pose7_xyzw_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(4, dtype=pose.dtype, device=pose.device).expand(pose.shape[:-1] + (4, 4)).clone()
    matrix[..., :3, :3] = quat_xyzw_to_matrix(pose[..., 3:7])
    matrix[..., :3, 3] = pose[..., :3]
    return matrix


def action_to_target_pose(current_pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Map 18D pour action to a right palm target pose matrix."""
    if action.shape[-1] != 18:
        raise ValueError(f"expected action last dimension 18, got {action.shape[-1]}")
    target_pose = current_pose.clone()
    target_pose[..., :3, 3] = current_pose[..., :3, 3] + action[..., :3] * RIGHT_PALM_DELTA_XYZ_SCALE
    delta_rot = axis_angle_to_matrix(action[..., 3:6] * RIGHT_PALM_DELTA_ROT_SCALE)
    target_pose[..., :3, :3] = torch.matmul(delta_rot, current_pose[..., :3, :3])
    return target_pose


def target_pose_to_action(
    current_pose: torch.Tensor,
    target_pose: torch.Tensor,
    gripper_action: torch.Tensor,
) -> torch.Tensor:
    """Map a right palm target pose and gripper command to the 18D pour action."""
    if gripper_action.shape[-1] == 1:
        curl = gripper_action.expand(gripper_action.shape[:-1] + (5,))
    elif gripper_action.shape[-1] == 5:
        curl = gripper_action
    else:
        raise ValueError(f"expected gripper action last dimension 1 or 5, got {gripper_action.shape[-1]}")

    action = torch.zeros(current_pose.shape[:-2] + (18,), dtype=current_pose.dtype, device=current_pose.device)
    action[..., :3] = (target_pose[..., :3, 3] - current_pose[..., :3, 3]) / RIGHT_PALM_DELTA_XYZ_SCALE
    delta_rot = torch.matmul(target_pose[..., :3, :3], current_pose[..., :3, :3].transpose(-1, -2))
    action[..., 3:6] = matrix_to_axis_angle(delta_rot) / RIGHT_PALM_DELTA_ROT_SCALE
    action[..., 6:11] = curl
    return action


def actions_to_gripper_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] != 18:
        raise ValueError(f"expected action last dimension 18, got {actions.shape[-1]}")
    return actions[..., 6:11]
