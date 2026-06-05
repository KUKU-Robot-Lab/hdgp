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

"""상수 정의: inspire_r_pour_v1 (RH56F1 6-DOF 손)

Tesollo 20-DOF(robot 27D) → RH56F1 6 actuated DOF(robot 13D) 포팅.
손은 pour 중 grasp pose 로 고정(per-finger lerp), 팔(6D palm)만 학습.

Action (11D) — pour 와 동일 유지:
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp: -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE
         (5 손가락 → 6 drive 관절 매핑: thumb 은 2관절[thumb_1,thumb_2], 나머지 1관절씩)

Actor Observation (60D) — pour-flow 중심 (로봇 비의존, 기존과 동일):
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_grasp_progress:    5  (per-finger grasp 유지 요약, 5 손가락)
  right_cup_pos_rel_palm:   3
  right_cup_quat:           4
  left_cup_pos_rel_palm:    3
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  transport_summary:        8
  last_palm_actions:        6
  bead_in_source_fraction:  1
  bead_in_target_fraction:  1
  bead_cross_fraction:      1
  spill_ratio:              1
  flow_summary:             4
  Total:                   60

Critic Base (82D) — sim-only full-state:
  arm_pos 7 + arm_vel 7 + finger_pos 6 + finger_vel 6
  + rcup_rel 3 + rcup_quat 4 + lcup_rel 3 + lcup_quat 4
  + opening_rel 3 + source_pour_axis 3 + source_up_axis 3
  + transport_stack 8 + binary_contact 5 + tip_force 5 + last_actions 11
  + bead(source/target/cross/spill) 4
  = 82

Critic Extra (50D) — sim-only privileged:
  left_arm_joint_pos:      19   (7 arm + 12 left RH56F1 hand)
  left_arm_joint_vel:      19
  distal_contact_binary:    5   (RH56F1: distal = fingertip)
  distal_contact_norm:      5
  cup_height_delta:         1
  rho:                      1
  Total:                   50

Critic Total: 82 + 50 = 132D

Episode (@ 60 Hz):
  Pour phase: Fabrics arm policy + frozen hand
"""

from .pour_right_preset import (
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
NUM_HAND_DOF  = len(RIGHT_HAND_JOINT_NAMES)   # 6 (RH56F1 drive)
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF    # 13
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION   = 6   # 6D palm pose (Fabrics IK)
NUM_FINGER_ACTION = 5   # per-finger lerp (5 손가락)
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 11

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 60         # Actor: 7+7+5+3+4+3+3+3+3+8+6+1+1+1+1+4 = 60
NUM_DISTAL_SENSORS  = 5       # RH56F1: distal = fingertip (tip 센서 재사용)
NUM_MIDDLE_SENSORS  = 5       # RH56F1: 중간 phalanx 없음 → zeros
NUM_CRITIC_BASE_OBSERVATIONS = 82
NUM_CRITIC_EXTRAS   = 50      # left_arm(38) + distal(10) + cup_h(1) + rho(1)
NUM_CRITIC_OBSERVATIONS = NUM_CRITIC_BASE_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 132

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
POUR_EPISODE_STEPS = 600    # 10s: transport + tilt + pour

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  정규화 분모
MIN_CONTACTS_FOR_SUCCESS = 2      # 성공 판정 최소 접촉 손가락 수

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
ARM_START_POSE      = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
