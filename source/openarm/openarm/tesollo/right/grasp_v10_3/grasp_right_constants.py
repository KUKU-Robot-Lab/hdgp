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

"""상수 정의: 5g_grasp_right_v10-3.

Policy-driven cup-aware grasp 계약:
  정책은 매 step 다음 palm pose와 손 관절 목표를 직접 출력한다.
  arm 7축은 action에 포함하지 않고 Fabrics IK가 palm target에서 계산한다.

Action (26D):
  [0:3]   incremental palm xyz command
  [3:6]   incremental palm rotation-vector command
  [6:26]  20D hand residual around HAND_GRASP_POSE
         rj_dg_1_1~4, rj_dg_2_1~4, rj_dg_3_1~4, rj_dg_4_1~4, rj_dg_5_1~4

Actor Observation (133D, no oracle mass) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15  (5 × 3D)
  palm_to_cup_pos:          3
  last_actions:            27  (v10.3 policy target action)
  tip_force_xyz_norm:      15  (5 × 3D 법선 방향 힘 벡터, v9.1: 5D norm → 15D vector)
  middle_to_cup_xyz:       15  (5 × 3D FK 기반, sim2real 가능: joint encoder → FK)
  phase_step_ratio:         1  (step counter 기반, 실 로봇 가능)
  [제거] cup_to_fingertip  15D → fingertip_pos_rel_palm - palm_to_cup 항등식 (완전 중복)
  [제거] binary_contact     5D → tip_force_xyz_norm norm의 하위 집합 (함수적 중복)
  Total:                  136

Optional actor debug observation with oracle mass: 137D

Critic Extra (37D) — sim-only privileged:
  bead_mass_normalized:     1  (0=빈 컵, 1=최대 하중)
  cup_lin_vel:              3
  cup_ang_vel:              3
  cup_rot (quat):           4
  cup_height_delta:         1
  distal_contact_binary:    5  (rl_dg_*_4)
  distal_contact_norm:      5
  middle_contact_binary:    5  (rl_dg_*_3)
  middle_contact_norm:      5
  fingertip_to_cup_signed_dist: 5
  Total:                   37

Actor Observation without oracle mass: 136D
Critic Total: 136 + 37 = 173D

Episode (10s @ 60Hz = 600 steps):
  approach/close-grasp/lift/stabilize phase는 상태 기반 reward/gate/diagnostic label이다.
  approach는 reset-local policy target, close-grasp/lift는 captured palm anchor 주변 target을 쓴다.
  lift 성공이 latch되면 제자리 stabilize phase로 승급한다.
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
NUM_PALM_POS_ACTION  = 3
NUM_PALM_ROT_ACTION  = 3
NUM_PALM_ACTION      = NUM_PALM_POS_ACTION + NUM_PALM_ROT_ACTION
NUM_FINGER_ACTION    = 20
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 26
PALM_POS_ACTION_SLICE = slice(0, 3)
PALM_ROT_ACTION_SLICE = slice(3, 6)
FINGER_ACTION_SLICE = slice(6, 26)

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
# Actor obs (136D no-mass, 137D optional mass/debug):
#   arm_joint_pos        7  | arm_joint_vel          7
#   finger_joint_pos    20  | finger_joint_vel       20
#   palm_center_pos      3  | fingertip_pos_rel_palm 15
#   palm_to_cup          3  | cup_to_goal             3
#   last_actions        27  | tip_force_xyz_norm     15
#   middle_to_cup_xyz   15  | phase_step_ratio        1
#   [optional] bead_mass_normalized 1
#   [제거] cup_to_fingertip 15D (항등식), binary_contact 5D (tip_force 하위집합)
NUM_OBSERVATIONS_NO_MASS = 132
NUM_OBSERVATIONS = 133
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 37
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS_NO_MASS + NUM_CRITIC_EXTRAS  # 169

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
EPISODE_STEPS           = 600    # 10s @ 60Hz (grasp/lift/stabilize)
GRASP_PHASE_STEPS       = EPISODE_STEPS  # legacy compatibility only
LIFT_RAISE_PHASE_STEPS  = 0      # state-latched, not step-gated
STABILIZE_PHASE_STEPS   = 0      # state-latched, not step-gated
LIFT_PHASE_STEPS        = EPISODE_STEPS  # legacy config default only
PRELOAD_START_STEP = 400

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  정규화 분모
MIN_CONTACTS_FOR_SUCCESS = 4      # 성공 판정 최소 접촉 손가락 수 (v10: 2→4, ADR 독립 고정값)

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
