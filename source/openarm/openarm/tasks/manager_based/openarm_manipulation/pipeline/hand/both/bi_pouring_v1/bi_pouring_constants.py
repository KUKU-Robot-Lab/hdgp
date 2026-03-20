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
NUM_ARM_DOF = NUM_RIGHT_ARM_DOF  # 7

# Actor observation layout:
#   right_arm_joint_pos: 7
#   right_arm_joint_vel: 7
#   source_to_target_relative_pose: 7  (pos 3 + quat 4)
#   source_pour_point_to_target_opening: 3
#   source_cup_velocity_summary: 2
#   tilt_alignment_summary: 3
#   last_actions: 7  (right arm joint delta 7D)
NUM_OBSERVATIONS = 36
NUM_ACTIONS = 7  # right arm joint delta 7D (PD 제어)

# Critic-only extras (DexPour 계층적 리워드 구조):
#   bead_centroid_pos: 3
#   bead_centroid_vel: 3
#   bead_in_target_ratio: 1
#   pour_stage_active: 1
#   task_flags: 3  (bead_has_entered, success, bead_exited_after_entry)
#   stable_retention_steps: 1
#   spill_flags: 2  (bead_spilled, major_spill)
#   bead_spill_ratio: 1
NUM_STATES = 51
