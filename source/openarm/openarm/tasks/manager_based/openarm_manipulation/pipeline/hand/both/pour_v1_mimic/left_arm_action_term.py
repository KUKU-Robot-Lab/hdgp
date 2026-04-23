"""Left arm delta ActionTerm for pour_v1_mimic.

Action slice [11:18] of the 18D pour action — 7D joint-space delta for the
left arm.  The term also holds the left-arm gripper in its closed preset and
maintains the left target cup's kinematic attachment pose.

Action convention:  action=0 → hold rest pose; ±1 → ±left_arm_delta_joint rad
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_NUM_LEFT_ARM = 7


@configclass
class LeftArmDeltaActionTermCfg(ActionTermCfg):
    """Configuration for the left arm delta action term."""

    class_type: type = None  # set below

    asset_name: str = "robot"

    # Maximum per-step joint delta [rad] (un-normalised)
    left_arm_delta_joint: float = 0.10

    # Joint names for the left arm (7D) and gripper (2D)
    left_arm_joint_names: tuple = (
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
    )
    left_gripper_joint_names: tuple = (
        "openarm_left_finger_joint1",
        "openarm_left_finger_joint2",
    )


class LeftArmDeltaActionTerm(ActionTerm):
    """Applies a 7D joint-space delta to the left arm.

    Left gripper is always held at the rest (closed) pose.
    """

    cfg: LeftArmDeltaActionTermCfg

    def __init__(self, cfg: LeftArmDeltaActionTermCfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)

        self._robot = self._asset
        self._device = env.device
        num_envs = env.num_envs

        self._delta_scale = cfg.left_arm_delta_joint

        # DOF indices (resolved lazily)
        self._arm_ids: torch.Tensor | None = None
        self._gripper_ids: torch.Tensor | None = None

        # Rest pose (filled lazily from articulation soft limits)
        from ..pour_v1.pour_preset import LEFT_ARM_REST_JOINT_POS
        self._rest_vals = LEFT_ARM_REST_JOINT_POS
        self._rest_pos: torch.Tensor | None = None
        self._gripper_closed: torch.Tensor | None = None

        # Joint targets (reset sets these)
        self._arm_targets: torch.Tensor = torch.zeros(num_envs, _NUM_LEFT_ARM, device=self._device)
        self._arm_limits: torch.Tensor | None = None

        # Raw / processed buffers
        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=self._device)
        self._processed = torch.zeros(num_envs, _NUM_LEFT_ARM, device=self._device)

    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return _NUM_LEFT_ARM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    # ------------------------------------------------------------------

    def _resolve_ids(self) -> None:
        if self._arm_ids is not None:
            return
        names = self._robot.joint_names
        self._arm_ids = torch.tensor(
            [names.index(n) for n in self.cfg.left_arm_joint_names],
            device=self._device, dtype=torch.long,
        )
        self._gripper_ids = torch.tensor(
            [names.index(n) for n in self.cfg.left_gripper_joint_names],
            device=self._device, dtype=torch.long,
        )
        # rest positions from soft limits (midpoint) or preset dict
        arm_lims = self._robot.data.soft_joint_pos_limits[0, self._arm_ids, :]
        rest_list = [
            self._rest_vals.get(n, float(arm_lims[i, 0] + arm_lims[i, 1]) / 2.0)
            for i, n in enumerate(self.cfg.left_arm_joint_names)
        ]
        self._rest_pos = torch.tensor(rest_list, dtype=torch.float32, device=self._device)
        self._arm_limits = arm_lims  # (7, 2)

        gripper_lims = self._robot.data.soft_joint_pos_limits[0, self._gripper_ids, :]
        gripper_list = [
            self._rest_vals.get(n, float(gripper_lims[i, 0]))
            for i, n in enumerate(self.cfg.left_gripper_joint_names)
        ]
        self._gripper_closed = torch.tensor(gripper_list, dtype=torch.float32, device=self._device)

        num_envs = self._arm_targets.shape[0]
        self._arm_targets = self._rest_pos.unsqueeze(0).expand(num_envs, -1).clone()

    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """Decode 18D action slice [11:18] and update arm targets."""
        self._resolve_ids()

        delta = actions[:, 11:18] * self._delta_scale
        self._arm_targets = torch.clamp(
            self._arm_targets + delta,
            self._arm_limits[:, 0].unsqueeze(0),
            self._arm_limits[:, 1].unsqueeze(0),
        )
        self._raw_actions.copy_(actions[:, 11:18])
        self._processed.copy_(self._arm_targets)

    def apply_actions(self) -> None:
        """Write joint position targets to the robot articulation."""
        self._robot.set_joint_position_target(self._arm_targets, joint_ids=self._arm_ids)
        self._robot.set_joint_velocity_target(
            torch.zeros_like(self._arm_targets), joint_ids=self._arm_ids
        )
        closed = self._gripper_closed.unsqueeze(0).expand(self._arm_targets.shape[0], -1)
        self._robot.set_joint_position_target(closed, joint_ids=self._gripper_ids)
        self._robot.set_joint_velocity_target(
            torch.zeros_like(closed), joint_ids=self._gripper_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._resolve_ids()
        ids = slice(None) if env_ids is None else env_ids
        n = self._arm_targets[ids].shape[0]
        self._arm_targets[ids] = self._rest_pos.unsqueeze(0).expand(n, -1)


LeftArmDeltaActionTermCfg.class_type = LeftArmDeltaActionTerm
