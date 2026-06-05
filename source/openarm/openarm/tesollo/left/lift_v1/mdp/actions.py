# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Custom action terms for 5-finger hand with synergy control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Finger 2-4 joint names (excluding thumb + pinky)
NON_THUMB_JOINTS_RIGHT = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(2, 5)  # fingers 2-4
    for joint in range(1, 5)   # joints 1-4
]

NON_THUMB_JOINTS_LEFT = [
    f"lj_dg_{finger}_{joint}"
    for finger in range(2, 5)
    for joint in range(1, 5)
]

# Default open/close poses for RIGHT hand fingers 2-4 (12 joints)
DEFAULT_OPEN_POSE_RIGHT = {
    f"rj_dg_{f}_{j}": 0.0 for f in range(2, 5) for j in range(1, 5)
}

DEFAULT_CLOSE_POSE_RIGHT = {
    # Finger 2 (index)
    "rj_dg_2_1": 0.6108652, "rj_dg_2_2": 2.0071286, "rj_dg_2_3": 1.5707963, "rj_dg_2_4": 1.5707963,
    # Finger 3 (middle)
    "rj_dg_3_1": 0.6108652, "rj_dg_3_2": 1.9547686, "rj_dg_3_3": 1.5707963, "rj_dg_3_4": 1.5707963,
    # Finger 4 (ring)
    "rj_dg_4_1": 0.418879, "rj_dg_4_2": 1.9024088, "rj_dg_4_3": 1.5707963, "rj_dg_4_4": 1.5707963,
}

# Default open/close poses for LEFT hand fingers 2-4 (12 joints)
DEFAULT_OPEN_POSE_LEFT = {
    f"lj_dg_{f}_{j}": 0.0 for f in range(2, 5) for j in range(1, 5)
}

DEFAULT_CLOSE_POSE_LEFT = {
    # Finger 2 (index)
    "lj_dg_2_1": 0.0, "lj_dg_2_2": 2.0071286, "lj_dg_2_3": 1.5707963, "lj_dg_2_4": 1.5707963,
    # Finger 3 (middle)
    "lj_dg_3_1": 0.0, "lj_dg_3_2": 1.9547686, "lj_dg_3_3": 1.5707963, "lj_dg_3_4": 1.5707963,
    # Finger 4 (ring)
    "lj_dg_4_1": 0.0, "lj_dg_4_2": 1.9024088, "lj_dg_4_3": 1.5707963, "lj_dg_4_4": 1.5707963,
}

# Backwards compatibility aliases
DEFAULT_OPEN_POSE = DEFAULT_OPEN_POSE_RIGHT
DEFAULT_CLOSE_POSE = DEFAULT_CLOSE_POSE_RIGHT


class FingerSynergyAction(ActionTerm):
    """Synergy-based action for RIGHT hand fingers 2-5.

    Maps a single scalar input (grip strength) to all 16 non-thumb finger joints.
    - Input: 1 value in [-1, 1] range
    - Output: 16 joint positions interpolated between open and close poses

    grip_strength = -1: fully open
    grip_strength = +1: fully closed
    """

    cfg: "FingerSynergyActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "FingerSynergyActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve joint IDs
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        # Create open/close pose tensors
        self._open_pose = torch.zeros(self._num_joints, device=self.device)
        self._close_pose = torch.zeros(self._num_joints, device=self.device)

        for i, name in enumerate(self._joint_names):
            self._open_pose[i] = cfg.open_pose.get(name, 0.0)
            self._close_pose[i] = cfg.close_pose.get(name, 0.5)

        # Raw action storage (1 value)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        # Processed actions (num_joints values)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        """Action dimension is 1 (grip strength)."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process grip strength into joint positions."""
        self._raw_actions[:] = actions

        # Clamp to [-1, 1]
        grip = torch.clamp(actions[:, 0], -1.0, 1.0)

        # Map from [-1, 1] to [0, 1] for interpolation
        # -1 = open (t=0), +1 = close (t=1)
        t = (grip + 1.0) / 2.0  # [0, 1]

        # Interpolate: open_pose * (1-t) + close_pose * t
        self._processed_actions[:] = (
            self._open_pose.unsqueeze(0) * (1 - t).unsqueeze(1) +
            self._close_pose.unsqueeze(0) * t.unsqueeze(1)
        )

    def apply_actions(self) -> None:
        """Apply processed joint positions to the asset."""
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset action state for specified environments."""
        self._raw_actions[env_ids] = 0.0


@configclass
class FingerSynergyActionCfg(ActionTermCfg):
    """Configuration for RIGHT hand finger synergy action."""

    class_type: type[ActionTerm] = FingerSynergyAction

    asset_name: str = "robot"
    """Name of the asset in the scene."""

    joint_names: list[str] = None
    """List of joint names for synergy control (fingers 2-5)."""

    open_pose: dict[str, float] = None
    """Joint positions for fully open hand."""

    close_pose: dict[str, float] = None
    """Joint positions for fully closed hand."""

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = NON_THUMB_JOINTS_RIGHT
        if self.open_pose is None:
            self.open_pose = DEFAULT_OPEN_POSE_RIGHT
        if self.close_pose is None:
            self.close_pose = DEFAULT_CLOSE_POSE_RIGHT


class FingerSynergyActionLeft(ActionTerm):
    """Synergy-based action for LEFT hand fingers 2-5.

    Maps a single scalar input (grip strength) to all 16 non-thumb finger joints.
    - Input: 1 value in [-1, 1] range
    - Output: 16 joint positions interpolated between open and close poses

    grip_strength = -1: fully open
    grip_strength = +1: fully closed
    """

    cfg: "FingerSynergyActionLeftCfg"
    _asset: Articulation

    def __init__(self, cfg: "FingerSynergyActionLeftCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve joint IDs
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        # Create open/close pose tensors
        self._open_pose = torch.zeros(self._num_joints, device=self.device)
        self._close_pose = torch.zeros(self._num_joints, device=self.device)

        for i, name in enumerate(self._joint_names):
            self._open_pose[i] = cfg.open_pose.get(name, 0.0)
            self._close_pose[i] = cfg.close_pose.get(name, 0.5)

        # Raw action storage (1 value)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        # Processed actions (num_joints values)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        """Action dimension is 1 (grip strength)."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process grip strength into joint positions."""
        self._raw_actions[:] = actions

        # Clamp to [-1, 1]
        grip = torch.clamp(actions[:, 0], -1.0, 1.0)

        # Map from [-1, 1] to [0, 1] for interpolation
        # -1 = open (t=0), +1 = close (t=1)
        t = (grip + 1.0) / 2.0  # [0, 1]

        # Interpolate: open_pose * (1-t) + close_pose * t
        self._processed_actions[:] = (
            self._open_pose.unsqueeze(0) * (1 - t).unsqueeze(1) +
            self._close_pose.unsqueeze(0) * t.unsqueeze(1)
        )

    def apply_actions(self) -> None:
        """Apply processed joint positions to the asset."""
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset action state for specified environments."""
        self._raw_actions[env_ids] = 0.0


@configclass
class FingerSynergyActionLeftCfg(ActionTermCfg):
    """Configuration for LEFT hand finger synergy action."""

    class_type: type[ActionTerm] = FingerSynergyActionLeft

    asset_name: str = "robot"
    """Name of the asset in the scene."""

    joint_names: list[str] = None
    """List of joint names for synergy control (fingers 2-5)."""

    open_pose: dict[str, float] = None
    """Joint positions for fully open hand."""

    close_pose: dict[str, float] = None
    """Joint positions for fully closed hand."""

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = NON_THUMB_JOINTS_LEFT
        if self.open_pose is None:
            self.open_pose = DEFAULT_OPEN_POSE_LEFT
        if self.close_pose is None:
            self.close_pose = DEFAULT_CLOSE_POSE_LEFT
