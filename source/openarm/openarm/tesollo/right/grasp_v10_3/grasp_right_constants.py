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

"""상수 정의: 5g_grasp_right_v10-3

v10: v9 기반 버그 수정 (MIN_CONTACTS_FOR_SUCCESS=4, has_5_contact 고정, thumb_1=0)
v9: v8 기반 + 20D 손가락 직접 제어 + Slip/Force-Efficiency/Force-Smooth reward
- 5D synergy lerp → 20D per-joint position delta (±finger_delta_scale rad)
- Slip-aware reward: 컵 수평 속도 기반 slip proxy
- Force-efficiency reward: 질량 기반 최소 충분 파지력 유도
- Force-smooth reward: 파지력 변화율 억제

Action (26D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:26] 20D per-joint finger delta (±finger_delta_scale rad)
         rj_dg_1_1~4, rj_dg_2_1~4, rj_dg_3_1~4, rj_dg_4_1~4, rj_dg_5_1~4

Actor Observation (133D) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15  (5 × 3D)
  palm_to_cup_pos:          3
  last_actions:            26  (v8: 11D → v9: 26D)
  bead_mass_normalized:     1  (0=빈 컵, 1=최대 하중)
  tip_force_xyz_norm:      15  (5 × 3D 법선 방향 힘 벡터, v9.1: 5D norm → 15D vector)
  middle_to_cup_xyz:       15  (5 × 3D FK 기반, sim2real 가능: joint encoder → FK)
  phase_step_ratio:         1  (step counter 기반, 실 로봇 가능)
  [제거] cup_to_fingertip  15D → fingertip_pos_rel_palm - palm_to_cup 항등식 (완전 중복)
  [제거] binary_contact     5D → tip_force_xyz_norm norm의 하위 집합 (함수적 중복)
  Total:                  133

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

Actor Observation without oracle mass: 132D
Critic Total: 133 + 36 = 169D

Episode (14s @ 60Hz = 840 steps):
  Grasp phase (0~479):  Fabrics arm + per-joint finger delta
  Raise phase (480~599): scripted 5cm vertical prelift + micro-delta hand
  Stabilize phase (600~839): bounded arm action + micro-delta hand
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
NUM_PALM_ACTION   = 6    # 6D palm pose (Fabrics IK)
NUM_FINGER_ACTION = 20   # 20D per-joint delta (v8: 5D synergy → v9: 20D direct)
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 26

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
# Actor obs (133D):
#   arm_joint_pos        7  | arm_joint_vel          7
#   finger_joint_pos    20  | finger_joint_vel       20
#   palm_center_pos      3  | fingertip_pos_rel_palm 15
#   palm_to_cup          3  | last_actions           26
#   bead_mass_normalized 1  | tip_force_xyz_norm     15
#   middle_to_cup_xyz   15  | phase_step_ratio        1
#   [제거] cup_to_fingertip 15D (항등식), binary_contact 5D (tip_force 하위집합)
NUM_OBSERVATIONS = 133
NUM_OBSERVATIONS_NO_MASS = 132
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 36
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 169

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS       = 480    # 8s: Fabrics arm + per-joint finger delta
LIFT_RAISE_PHASE_STEPS  = 120    # 2s: scripted 5cm vertical prelift
STABILIZE_PHASE_STEPS   = 240    # 4s: bounded arm action while holding lift height
LIFT_PHASE_STEPS        = LIFT_RAISE_PHASE_STEPS + STABILIZE_PHASE_STEPS
LIFT_START_STEP         = GRASP_PHASE_STEPS
STABILIZE_START_STEP    = LIFT_START_STEP + LIFT_RAISE_PHASE_STEPS
EPISODE_STEPS           = GRASP_PHASE_STEPS + LIFT_PHASE_STEPS  # 840
PRELOAD_START_STEP = 400    # lift 직전 80 step: under-grip penalty 활성 구간 (400~479)

LIFT_Z_DELTA = 0.05    # 5cm 수직 상승 후 stabilize phase에서 높이 유지

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
