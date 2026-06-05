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
from .initial_object_pose_command import InitialObjectPoseCommand


@configclass
class InitialObjectPoseCommandCfg(CommandTermCfg):
    """Command that targets the object's initial pose with a fixed offset."""

    asset_name: str = MISSING
    asset_cfg: SceneEntityCfg = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    goal_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        self.command_dim = 7
        self.class_type = InitialObjectPoseCommand
