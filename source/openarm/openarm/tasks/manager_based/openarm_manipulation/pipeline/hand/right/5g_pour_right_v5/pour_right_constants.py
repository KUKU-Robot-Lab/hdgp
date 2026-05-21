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

"""상수 정의: 5g_pour_right_v5

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (freeze_grasp=True → 항상 1.0 강제)

Actor Observation (60D) — pour-flow 중심:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_grasp_progress:    5  (per-finger grasp 유지 요약)
  right_cup_pos_rel_palm:   3
  right_cup_quat:           4
  left_cup_pos_rel_palm:    3
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  transport_summary:        8  [mouth_dist, mouth_xy_dist, cup_center_xy_dist,
                                z_clearance, source_up_dot, dir_tilt_cos,
                                mouth_alignment_cos, g_ready]
  last_palm_actions:        6
  bead_in_source_fraction:  1  (소스 컵 잔량 — "다 쏟았나" 인식)
  bead_in_target_fraction:  1  (타겟 컵 유입량 — "얼마나 넣었나" 인식)
  bead_cross_fraction:      1  (mouth 통과 비율 — r_cross weight=20 대응)
  spill_ratio:              1  (유출 비율 — spill_cost weight=10 대응)
  flow_summary:             4  (source/target/cross/spill fraction step delta)
  Total:                   60

Critic Base (110D) — sim-only full-state:
  기존 actor full-state layout 유지: hand pos/vel, fingertip contact/force,
  last_actions 포함. Actor LSTM에는 넣지 않지만 critic value 추정에는 유지한다.

Critic Extra (33D) — sim-only privileged:
  left_arm_joint_pos:       9
  left_arm_joint_vel:       9
  distal_contact_binary:    5  (rl_dg_*_4)
  distal_contact_norm:      5
  cup_height_delta:         1
  g_align_xy:               1
  g_clear:                  1
  g_tilt:                   1
  g_pour:                   1
  Total:                   33
  (mouth_dist/xy/z, up_dot, tilt_cos, align_cos, g_ready, bead/spill fracs 제거 — actor_obs_clean에 중복)

Critic Total: 110 + 33 = 143D

Episode (10s @ 60Hz = 600 steps):
  Pour phase: Fabrics arm policy + frozen hand
"""

import math

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
NUM_OBSERVATIONS = 60         # Actor: 7+7+5+3+4+3+3+3+3+8+6+4+4 = 60
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_BASE_OBSERVATIONS = 110
NUM_CRITIC_EXTRAS   = 33      # left_arm(18) + distal(10) + cup_h(1) + g_align/clear/tilt/pour(4)
NUM_CRITIC_OBSERVATIONS = NUM_CRITIC_BASE_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 143

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
