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

"""상수 정의: 5g_grasp_right_v5

v5: FABRICS pregrasp reset + 정책 손가락 grasp formation + Scripted lift checker.

Episode 구조:
  Grasp phase (0~GRASP_PHASE_STEPS-1): arm joints = pregrasp 고정, 정책이 finger만 제어
  Lift phase  (GRASP_PHASE_STEPS~끝):  arm joints = pregrasp→prelift 선형 보간 (reset 시 미리 계산)
                                        finger joints = grasp phase 종료 시점 포지션으로 고정

Action (5D):
  [0:5]  per-finger lerp factor  ([-1,1] → lerp(HAND_APPROACH, HAND_GRASP) per finger)

Observation (101D) — actor (real-compatible):
  palm_to_cup_pos:        3  (world frame)
  cup_rot:                4  (quaternion w,x,y,z)
  fingertip_to_cup_pos:  15  (5 × 3D, world frame)
  finger_joint_pos:      20
  finger_joint_vel:      20
  arm_joint_pos:          7
  arm_joint_vel:          7
  binary_contact:         5  (ContactSensor Cup-filtered, per-fingertip)
  fingertip_force_xyz:   15  (get_contact_force_matrix: Cup-only fx,fy,fz × 5tips, [-1,1])
  last_actions:           5
  Total:                101
"""

import math

from .grasp_right_preset import (
    HAND_GRASP_POSE,
    HAND_START_POSE,
    OBJECT_GOAL_POS,
    OBJECT_SPAWN_CENTER,
    OBJECT_SPAWN_RANGE_XY,
    PREGRASP_OFFSET,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_START_POSE,
    RIGHT_HAND_JOINT_NAMES,
    palm_pose_maxs,
    palm_pose_mins,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF  = len(RIGHT_ARM_JOINT_NAMES)    # 7
NUM_HAND_DOF = len(RIGHT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF   # 27

NUM_FINGER_ACTION = 5    # per-finger lerp (thumb, index, middle, ring, pinky)
NUM_ACTIONS = NUM_FINGER_ACTION                        # 5

NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 101  # Actor (101D): 3+4+15+20+20+7+7+5+15+5

# Critic privileged sensors
NUM_DISTAL_SENSORS = 5   # rl_dg_1_4 ~ rl_dg_5_4 (distal phalanx)

# Critic extras over actor obs (18D):
#   cup_lin_vel(3) + cup_ang_vel(3)
#   + distal_binary(5) + distal_force(5)
#   + lift_phase_flag(1) + cup_height_delta(1)
NUM_CRITIC_EXTRAS = 18
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS   # 119

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz policy rate)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS = 480    # 8s: 정책 완전 제어 (손가락 닫기)
LIFT_PHASE_STEPS  = 120    # 2s: 스크립트 z 상승 + 정책 손가락 유지 (1s → 2s, 리프트 시간 확보)
TOTAL_EPISODE_STEPS = GRASP_PHASE_STEPS + LIFT_PHASE_STEPS   # 600 = 10s

LIFT_Z_DELTA = 0.10         # 10cm 수직 상승 (Scripted Lift Checker)

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  force_norm 정규화 분모 (clamp 후 /max → [0,1])
MIN_CONTACTS_FOR_SUCCESS = 2      # 성공 판정 최소 접촉 손가락 수

# ---------------------------------------------------------------------------
# FABRICS pregrasp steps
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60   # 리셋 시 FABRICS rollout 스텝 수

# ---------------------------------------------------------------------------
# v1/v4/v5 alias
# ---------------------------------------------------------------------------
ARM_START_POSE     = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
