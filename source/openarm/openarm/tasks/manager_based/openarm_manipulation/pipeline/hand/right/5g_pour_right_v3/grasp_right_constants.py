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

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (freeze_grasp=True → 항상 1.0 강제)

Actor Observation (105D) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  right_cup_pos_rel_palm:   3
  right_cup_quat:           4
  left_cup_pos_rel_palm:    3
  left_cup_quat:            4
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  transport_summary:        7  [mouth_dist, mouth_xy_dist, z_clearance,
                                source_up_dot, dir_tilt_cos,
                                mouth_alignment_cos, g_ready]
  fingertip_contact_binary: 5
  tip_force_norm:           5  (v8처럼, 실로봇 FT 센서 직결, sim2real 가능)
  last_actions:            11
  Total:                  105

Critic Extra (50D) — sim-only privileged:
  left_arm_joint_pos:       9
  left_arm_joint_vel:       9
  distal_contact_binary:    5  (rl_dg_*_4)
  distal_contact_norm:      5
  cup_height_delta:         1
  bead_pos_rel_source_cup:  3
  bead_pos_rel_target_cup:  3
  mouth_distance:           1
  mouth_xy_distance:        1
  mouth_z_clearance:        1
  source_up_dot_world:      1
  directional_tilt_cos:     1
  mouth_alignment_cos:      1
  g_align_xy:               1
  g_clear:                  1
  g_tilt:                   1
  g_ready:                  1
  g_pour:                   1
  bead_cross_fraction:      1
  bead_in_source_fraction:  1
  bead_in_target_fraction:  1
  spill_ratio:              1
  Total:                   50

Critic Total: 105 + 50 = 155D

Episode (10s @ 60Hz = 600 steps):
  Pour phase: Fabrics arm policy + frozen hand
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
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 11

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 106        # Actor: sim2real 가능 (cup_center_xy_dist 추가로 transport_summary 7→8)
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 51      # stage gate + bead/spill + cup_center_xy_dist
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 157

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
POUR_EPISODE_STEPS = 600    # 10s: transport + tilt + pour
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
