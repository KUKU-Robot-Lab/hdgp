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

"""상수 정의: 5g_grasp_right_v7_3

v7-3: v7-2 11D action + goal-aware lift/stabilize/transport sequence

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (thumb, index, middle, ring, pinky)
         -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE

Actor Observation (110D) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15  (5 × 3D)
  palm_to_cup_pos:          3
  cup_to_fingertip:        15  (5 × 3D)
  fingertip_contact_binary: 5  (FT sensor, 실 로봇 가능)
  last_actions:            11
  cup_to_goal:              3
  phase_step_ratio:         1
  Total:                  110

Critic Extra (36D) — sim-only privileged:
  cup_lin_vel:              3
  cup_ang_vel:              3
  cup_rot (quat):           4
  cup_height_delta:         1
  distal_contact_binary:    5  (rl_dg_*_4)
  distal_contact_norm:      5
  middle_contact_binary:    5  (rl_dg_*_3)
  middle_contact_norm:      5
  fingertip_to_cup_signed_dist: 5
  Total:                   36

Critic Total: 110 + 36 = 146D

Episode (18s @ 60Hz = 1080 steps):
  Grasp     phase (0~479):    Fabrics arm + per-finger policy
  Lift      phase (480~719):  palm target lift + frozen hand
  Stabilize phase (720~839):  lifted-pose hold
  Transport phase (840~1079): palm target transport to sampled goal
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
NUM_OBSERVATIONS = 110        # Actor: sim2real 가능
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 36
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 146

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS     = 480    # 8s: Fabrics arm + per-finger policy
LIFT_PHASE_STEPS      = 240    # 4s: palm target lift + frozen hand
STABILIZE_PHASE_STEPS = 120    # 2s: lifted-pose hold
TRANSPORT_PHASE_STEPS = 240    # 4s: palm target transport
LIFT_START_STEP       = GRASP_PHASE_STEPS    # 480
STABILIZE_START_STEP  = LIFT_START_STEP + LIFT_PHASE_STEPS  # 720
TRANSPORT_START_STEP  = STABILIZE_START_STEP + STABILIZE_PHASE_STEPS  # 840
EPISODE_STEPS         = (
    GRASP_PHASE_STEPS
    + LIFT_PHASE_STEPS
    + STABILIZE_PHASE_STEPS
    + TRANSPORT_PHASE_STEPS
)  # 1080

LIFT_Z_DELTA = 0.10    # 10cm 수직 상승 (j4 += 0.31 근사)

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
