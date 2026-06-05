# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, SceneEntityCfg

from .object_pose_command import ObjectPoseCommand
from .world_pose_command import WorldPoseCommand

@configclass
class ObjectPoseCommandCfg(CommandTermCfg):
    """Configuration for the object pose command term.

    This command term generates a command that tracks the pose of a specified object,
    with a dynamically changing offset based on the object's height.
    """
    # The name of the robot asset that the command is generated for.
    asset_name: str = MISSING
    # The configuration of the target object asset.
    asset_cfg: SceneEntityCfg = MISSING
    # Resampling time range for the command.
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    # Offset for pre-grasp phase (e.g., just above the object)
    pre_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
    # Offset for hold phase (e.g., higher above the object)
    hold_offset: tuple[float, float, float] = (0.0, 0.0, 0.10)
    # Z-threshold to determine if the object is considered "lifted"
    lift_threshold_z: float = 0.05

    def __post_init__(self):
        """Post-initialization."""
        super().__post_init__()
        # The command dimension is 7 (3 for pos, 4 for quat)
        self.command_dim = 7
        # Link the implementation class.
        self.class_type = ObjectPoseCommand


@configclass
class WorldPoseCommandCfg(CommandTermCfg):
    """Configuration for the world-frame pose command generator."""

    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    make_quat_unique: bool = False

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands (world frame)."""

        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    def __post_init__(self):
        super().__post_init__()
        self.command_dim = 7
        self.class_type = WorldPoseCommand
