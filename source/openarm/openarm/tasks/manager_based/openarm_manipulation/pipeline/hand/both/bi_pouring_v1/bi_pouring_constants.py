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

"""Initial constants for bi_pouring_v1.

v1 scope:
  - only the right arm is policy controlled
  - left arm / left gripper act as a fixed target holder
  - cups are intended to be rigidly attached to the respective end-effectors
  - bead transfer logic is intentionally left as TODO
"""

from .bi_pouring_preset import LEFT_HOLDER_JOINT_NAMES, RIGHT_POLICY_ARM_JOINT_NAMES

NUM_RIGHT_ARM_DOF = len(RIGHT_POLICY_ARM_JOINT_NAMES)
NUM_LEFT_HOLDER_DOF = len(LEFT_HOLDER_JOINT_NAMES)

# Actor observation layout:
#   right_arm_joint_pos: 7
#   right_arm_joint_vel: 7
#   source_to_target_relative_pose: 7
#   source_pour_point_to_target_opening: 3
#   source_cup_velocity_summary: 2
#   tilt_alignment_summary: 3
#   last_actions: 7
NUM_OBSERVATIONS = 36
NUM_ACTIONS = NUM_RIGHT_ARM_DOF

# Critic-only extras:
#   bead_pos_env: 3
#   bead_lin_vel_world: 3
#   task_flags: 3
#   stable_step_counter: 1
#   spill_flags: 2
NUM_STATES = 48
