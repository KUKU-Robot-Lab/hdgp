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

"""Constants for the pre-pouring version of bi_pouring_v1.

Observation layout (39D):
  right_arm_joint_pos          : 7
  right_arm_joint_vel          : 7
  right_palm_env_pos           : 3
  source_to_target_mouth_vec   : 3
  source_cup_quat              : 4
  target_cup_quat              : 4
  transport_summary            : 5
  last_palm_pose_action        : 6
"""

from .bi_pouring_preset import RIGHT_POLICY_ARM_JOINT_NAMES

NUM_RIGHT_ARM_DOF = len(RIGHT_POLICY_ARM_JOINT_NAMES)
NUM_ARM_DOF = NUM_RIGHT_ARM_DOF

NUM_PALM_POSE_ACTIONS = 6

NUM_OBSERVATIONS = 39
NUM_ACTIONS = NUM_PALM_POSE_ACTIONS
NUM_STATES = 0
