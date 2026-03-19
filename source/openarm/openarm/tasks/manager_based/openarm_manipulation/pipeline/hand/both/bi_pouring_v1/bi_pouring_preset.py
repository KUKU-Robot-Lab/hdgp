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

RIGHT_SOURCE_CUP_ATTACH_FRAME_NAME = "openarm_right_eef_link"
LEFT_TARGET_CUP_ATTACH_FRAME_NAME = "openarm_left_eef_link"

RIGHT_POLICY_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
LEFT_HOLDER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

# Nominal v1 pouring-ready pose for the policy-controlled right arm.
# This is intentionally conservative: cup is already near the receiving cup,
# but the wrist starts close to upright so the policy only needs to solve the
# short transfer motion rather than grasp acquisition.
RIGHT_ARM_POUR_READY_POSE = [0.58, -0.18, 0.62, 1.18, -0.22, 0.08, 0.10]
# Backward-compatible alias for existing references inside the task.
RIGHT_ARM_HOME_POSE = RIGHT_ARM_POUR_READY_POSE

LEFT_HOLDER_FIXED_JOINT_POS = {
    "openarm_left_joint1": -0.55,
    "openarm_left_joint2": -0.30,
    "openarm_left_joint3": 0.55,
    "openarm_left_joint4": 0.85,
    "openarm_left_joint5": 0.00,
    "openarm_left_joint6": 0.00,
    "openarm_left_joint7": -0.80,
    "openarm_left_finger_joint1": 0.00,
    "openarm_left_finger_joint2": 0.00,
}

# Static cup-to-tool attachment definitions.
# Keep these separate from any future left-holder pose generator so FABRICS-based
# randomization only changes the holder start pose, not the attachment model.
# TODO: replace / validate with measured cup-to-tool transforms for sim2real.
RIGHT_SOURCE_CUP_ATTACH_POS_B = [0.0, 0.0, 0.0]
RIGHT_SOURCE_CUP_ATTACH_QUAT_WXYZ_B = [1.0, 0.0, 0.0, 0.0]
LEFT_TARGET_CUP_ATTACH_POS_B = [0.0, 0.0, 0.0]
LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B = [1.0, 0.0, 0.0, 0.0]

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
