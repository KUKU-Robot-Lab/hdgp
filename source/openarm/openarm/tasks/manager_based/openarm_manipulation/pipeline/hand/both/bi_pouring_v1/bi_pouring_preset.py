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

"""Preset metadata for bi_pouring_v1.

The structure deliberately keeps a left-holder init-pose seam for future
curriculum randomization via FABRICS or an equivalent pose generator.
"""

import math

RIGHT_SOURCE_CUP_ATTACH_FRAME_NAME = "rl_dg_palm"
LEFT_TARGET_CUP_ATTACH_FRAME_NAME = "openarm_left_hand"

RIGHT_POLICY_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
LEFT_HOLDER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

# Symmetric holder-like baseline for early pouring-motion learning.
# The source cup is already attached to the right hand, and reset-time joint
# noise is applied around this baseline for initial pose randomization.
RIGHT_ARM_POUR_READY_POSE = [-0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]
# Backward-compatible alias for existing references inside the task.
RIGHT_ARM_HOME_POSE = RIGHT_ARM_POUR_READY_POSE

LEFT_HOLDER_FIXED_JOINT_POS = {
    "openarm_left_joint1": 0.0,
    "openarm_left_joint2": 0,
    "openarm_left_joint3": 0,
    "openarm_left_joint4": 1.5,
    "openarm_left_joint5": 0,
    "openarm_left_joint6": 0,
    "openarm_left_joint7": 0,
    "openarm_left_finger_joint1": 0.044,
    "openarm_left_finger_joint2": 0.044,
}

# Static cup-to-tool attachment definitions.
# Keep these separate from any future left-holder pose generator so FABRICS-based
# randomization only changes the holder start pose, not the attachment model.
# TODO: replace / validate with measured cup-to-tool transforms for sim2real.
# URDF: rl_dg_palm -> palm_link is identity, and palm_link -> palm_ee is
# xyz = [0.028, 0.0, 0.04]. Since palm_ee is not a simulated rigid body in
# Isaac Lab body_names, attach from rl_dg_palm and bake the helper-frame
# offset directly in the palm local frame.
RIGHT_SOURCE_CUP_ATTACH_POS_B = [0.06, 0.02, 0.04]
RIGHT_SOURCE_CUP_ATTACH_QUAT_WXYZ_B = [0.70710678, -0.70710678, 0.0, 0.0]
LEFT_TARGET_CUP_ATTACH_POS_B = [0.0, 0.0, 0.06]
LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B = [0.70710678, 0.0, 0.70710678, 0.0]

RIGHT_HAND_GRASP_JOINT_POS = {
    "rj_dg_1_1": 0.0,
    "rj_dg_1_2": -1.57,
    "rj_dg_1_3": 0.5,
    "rj_dg_1_4": 0.5,
    "rj_dg_2_1": 0.0,
    "rj_dg_2_2": 0.7,
    "rj_dg_2_3": 0.5,
    "rj_dg_2_4": 0.5,
    "rj_dg_3_1": 0.0,
    "rj_dg_3_2": 0.7,
    "rj_dg_3_3": 0.5,
    "rj_dg_3_4": 0.5,
    "rj_dg_4_1": 0.0,
    "rj_dg_4_2": 0.7,
    "rj_dg_4_3": 0.5,
    "rj_dg_4_4": 0.5,
    "rj_dg_5_1": 0.0,
    "rj_dg_5_2": 0.0,
    "rj_dg_5_3": 1.2,
    "rj_dg_5_4": 0.8,
}

# Single-bead spawn offset in the source cup local frame.
# TODO: replace with measured inner-cup calibration once the source cup mesh and
# bead diameter are validated in simulation and on hardware.
BEAD_SPAWN_POS_SOURCE_CUP_B = [0.0, 0.0, 0.035]
BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ = [1.0, 0.0, 0.0, 0.0]

# Pour-observation anchor defaults.
# These remain cup-local so future left-holder pose randomization only changes
# the target cup world pose, not the observation definition.
# TODO: replace with measured lip/opening anchors from the actual cup mesh.
SOURCE_CUP_POUR_POINT_POS_B = [0.045, 0.0, 0.080]
TARGET_CUP_OPENING_POS_B = [0.0, 0.0, 0.080]
SOURCE_CUP_POUR_AXIS_B = [1.0, 0.0, 0.0]
SOURCE_CUP_UP_AXIS_B = [0.0, 0.0, 1.0]
TARGET_CUP_UP_AXIS_B = [0.0, 0.0, 1.0]

# ---------------------------------------------------------------------------
# FABRICS arm control
# ---------------------------------------------------------------------------
# FABRICS 초기 arm joint 위치 (RIGHT_ARM_POUR_READY_POSE alias)
ARM_START_POSE = RIGHT_ARM_POUR_READY_POSE
