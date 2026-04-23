from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from ..pour_v1.pour_env import PourEnv
from .pour_mimic_math import (
    action_to_target_pose,
    actions_to_gripper_actions,
    pose7_xyzw_to_matrix,
    target_pose_to_action,
)


class PourMimicEnv(PourEnv, ManagerBasedRLMimicEnv):
    """Mimic-compatible facade over the DirectRL pour_v1 environment."""

    def _select_env_ids(self, env_ids: Sequence[int] | None):
        return slice(None) if env_ids is None else env_ids

    def _current_right_palm_pose7_xyzw(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        ids = self._select_env_ids(env_ids)
        if getattr(self, "palm_body_index", -1) >= 0 and hasattr(self, "robot"):
            pos = self.robot.data.body_pos_w[ids, self.palm_body_index, :]
            if hasattr(self, "scene") and hasattr(self.scene, "env_origins"):
                pos = pos - self.scene.env_origins[ids]
            quat_wxyz = self.robot.data.body_quat_w[ids, self.palm_body_index, :]
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
            return torch.cat((pos, quat_xyzw), dim=-1)
        if hasattr(self, "palm_pose_targets"):
            return self.palm_pose_targets[ids]
        return self.pregrasp_palm_pose_buf[ids]

    def get_robot_eef_pose(self, eef_name: str = "right", env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return the current end-effector pose as a batched 4x4 matrix."""
        if eef_name not in ("right", "palm", "right_palm"):
            raise KeyError(f"unsupported eef_name {eef_name!r}; pour_v1_mimic exposes 'right'")
        return pose7_xyzw_to_matrix(self._current_right_palm_pose7_xyzw(env_ids))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert a right-palm target pose and gripper command into the 18D pour action."""
        target_pose = target_eef_pose_dict.get("right", next(iter(target_eef_pose_dict.values())))
        gripper_action = gripper_action_dict.get("right", next(iter(gripper_action_dict.values())))
        if target_pose.ndim == 2:
            target_pose = target_pose.unsqueeze(0)
        if gripper_action.ndim == 1:
            gripper_action = gripper_action.unsqueeze(0)

        current_pose = self.get_robot_eef_pose("right", env_ids=[env_id])
        action = target_pose_to_action(current_pose, target_pose[:1].to(current_pose), gripper_action[:1].to(current_pose))

        if action_noise_dict is not None:
            noise_scale = action_noise_dict.get("right", next(iter(action_noise_dict.values())))
            action[:, :6] = torch.clamp(action[:, :6] + noise_scale * torch.randn_like(action[:, :6]), -1.0, 1.0)
        return action[0]

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert 18D pour actions into right-palm target pose matrices."""
        current_pose = self.get_robot_eef_pose("right", env_ids=None)
        return {"right": action_to_target_pose(current_pose, action.to(current_pose))}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract the 5D right-hand curl command from pour actions."""
        return {"right": actions_to_gripper_actions(actions)}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return source and target cup poses for Mimic object-centric transforms."""
        ids = self._select_env_ids(env_ids)
        source_pose = torch.cat(
            (self.cup.data.root_pos_w[ids], self.cup.data.root_quat_w[ids][:, [1, 2, 3, 0]]),
            dim=-1,
        )
        target_pose = torch.cat(
            (
                self.left_target_cup.data.root_pos_w[ids],
                self.left_target_cup.data.root_quat_w[ids][:, [1, 2, 3, 0]],
            ),
            dim=-1,
        )
        return {
            "source_cup": pose7_xyzw_to_matrix(source_pose),
            "target_cup": pose7_xyzw_to_matrix(target_pose),
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return automatic phase completion flags for Mimic annotation."""
        ids = self._select_env_ids(env_ids)
        self._compute_intermediate_values()

        contact_force = getattr(self, "contact_force_raw", None)
        if contact_force is None:
            grasp_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            grasp_done = torch.max(contact_force, dim=-1).values >= float(self.cfg.grasp_force_threshold)
        grasp_done = grasp_done & (self.cup.data.root_pos_w[:, 2] >= self.object_init_pos[:, 2] + 0.01)

        lift_done = self.cup.data.root_pos_w[:, 2] >= float(self.cfg.lift_threshold_z)
        align_done = self._mouth_xy_distance <= float(self.cfg.align_threshold_xy)
        tilt_threshold = math.cos(math.radians(float(self.cfg.pour_threshold_tilt_deg)))
        pour_done = self._source_up_dot_world <= tilt_threshold

        return {
            "grasp_done": grasp_done[ids],
            "lift_done": lift_done[ids],
            "align_done": align_done[ids],
            "pour_done": pour_done[ids],
        }

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = self._select_env_ids(env_ids)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return {
            "grasp_start": done[ids],
            "lift_start": self.get_subtask_term_signals(env_ids)["grasp_done"],
            "align_start": self.get_subtask_term_signals(env_ids)["lift_done"],
            "pour_start": self.get_subtask_term_signals(env_ids)["align_done"],
        }
