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

import math

from .pour_preset import (
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    palm_pose_mins,
    palm_pose_maxs,
    RIGHT_ARM_START_POSE,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF   = len(RIGHT_ARM_JOINT_NAMES)    # 7
NUM_HAND_DOF  = len(RIGHT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF     # 27
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION   = 6   # 6D palm pose (Fabrics IK)
NUM_FINGER_ACTION = 5   # per-finger lerp
NUM_LEFT_ARM_ACTION = 7  # left arm joint-space delta
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION + NUM_LEFT_ARM_ACTION  # 18

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
POUR_EPISODE_STEPS = 600    # 10s: transport + tilt + pour
EPISODE_STEPS = POUR_EPISODE_STEPS

# ---------------------------------------------------------------------------
# FABRICS pregrasp
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60

# ---------------------------------------------------------------------------
# Cup geometry
# ---------------------------------------------------------------------------
CUP_RADIUS_APPROX = 0.045  # m, cup_big 반경 (enclosure target 계산용)

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
ARM_START_POSE      = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
