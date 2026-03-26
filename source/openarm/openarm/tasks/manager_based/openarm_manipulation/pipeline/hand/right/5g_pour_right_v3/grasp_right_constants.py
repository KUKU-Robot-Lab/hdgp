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

"""상수 정의: 5g_pour_right_v3

v3: source cup을 오른손 EE에 고정한 arm-only 물붓기 태스크.
    finger/grasp 상태 없이 transport + tilt + pour 모션만 학습한다.

Action (6D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF

Actor Observation (51D) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  right_cup_pos_rel_palm:   3
  right_cup_quat:           4
  left_cup_pos_rel_palm:    3
  left_cup_quat:            4
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  target_up_axis:           3
  transport_summary:        5  [mouth_dist, mouth_xy_dist, z_clearance,
                                source_up_dot, dir_tilt_cos]
  last_actions:             6
  Total:                   51

Critic Extra (30D) — sim-only privileged:
  left_arm_joint_pos:       9
  left_arm_joint_vel:       9
  bead_pos_rel_source_cup:  3
  bead_pos_rel_target_cup:  3
  mouth_distance:           1
  mouth_xy_distance:        1
  mouth_z_clearance:        1
  source_up_dot_world:      1
  directional_tilt_cos:     1
  mouth_alignment_cos:      1
  bead_cross_fraction:      1
  Total:                   30

Critic Total: 51 + 30 = 81D

Episode (6s @ 60Hz = 360 steps):
  Pour phase (0~359): Fabrics arm policy + EE-attached source cup
"""

import math

from .grasp_right_preset import (
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
NUM_ACTIONS = NUM_PALM_ACTION  # 6

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 51         # Actor: sim2real 가능
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 30
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 81

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
POUR_EPISODE_STEPS = 360    # 6s: transport + tilt + pour (bi_pouring_v1과 동일)
EPISODE_STEPS = POUR_EPISODE_STEPS

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  정규화 분모
MIN_CONTACTS_FOR_SUCCESS = 2      # 성공 판정 최소 접촉 손가락 수

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
