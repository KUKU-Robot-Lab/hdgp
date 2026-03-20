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

"""Constants for bi_pouring_v1 (ManagerBasedRLEnv).

Actor observation layout (36D):
  right_arm_joint_pos        : 7
  right_arm_joint_vel        : 7
  cup_relative_pose          : 7  (pos3 + quat4)
  pour_point_to_opening      : 3
  source_cup_velocity_summary: 2
  tilt_alignment_summary     : 3
  last_arm_action            : 7
  ─────────────────────────────
  Total                      : 36
"""

from .bi_pouring_preset import LEFT_HOLDER_JOINT_NAMES, RIGHT_POLICY_ARM_JOINT_NAMES

NUM_RIGHT_ARM_DOF = len(RIGHT_POLICY_ARM_JOINT_NAMES)   # 7
NUM_LEFT_HOLDER_DOF = len(LEFT_HOLDER_JOINT_NAMES)      # 9 (7 arm + 2 gripper)
NUM_ARM_DOF = NUM_RIGHT_ARM_DOF

# ManagerBasedRLEnv에서 obs/action 크기는 manager가 자동 계산하지만
# YAML 설정 참조용으로 유지.
NUM_OBSERVATIONS = 36
NUM_ACTIONS = 7
NUM_STATES = 0
