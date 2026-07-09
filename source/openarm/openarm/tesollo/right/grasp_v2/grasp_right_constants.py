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

"""상수 정의: 5g_grasp_right_v1

v7: Fabrics 팔 학습(6D palm action) + per-finger lerp(5D) + Contact sensor 없는 FK 기반 보상

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (thumb, index, middle, ring, pinky)
         -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE

Actor Observation (106D) — sim2real 가능:
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
  Total:                  106

Critic Extra (37D) — sim-only privileged:
  cup_lin_vel:              3
  cup_ang_vel:              3
  cup_rot (quat):           4
  cup_height_delta:         1
  distal_contact_binary:    5  (rl_dg_*_4)
  distal_contact_norm:      5
  middle_contact_binary:    5  (rl_dg_*_3)
  middle_contact_norm:      5
  phase_step_ratio:         1
  fingertip_to_cup_signed_dist: 5
  Total:                   37

Critic Total: 106 + 37 = 143D

Episode (10s @ 60Hz = 600 steps):
  Grasp phase (0~479):  Fabrics arm + per-finger policy
  Lift-wait phase (480~599): scripted joint7-only lift-wait + frozen hand
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
# Observation space — DEXTRAH teacher 구조 (distillation 대비 동일 구조)
#
# Policy obs (BASE 193 + num_objects onehot):
#   robot_dof_pos_noisy 27 + robot_dof_vel_noisy 27 (annealing으로 0)
#   + hand_pos_noisy 18 (fabric FK: palm+5tip) + hand_vel_noisy 18 (0)
#   + object_pos_noisy 3 + object_rot_noisy 4 + object_goal 3
#   + [onehot num_objects] + object_scale 1 + actions 11
#   + fabric_q 27 + fabric_qd 27 + fabric_qdd 27
#   = 193 + N_obj   (DEXTRAH 원본 "193 + num_objects"와 동일 구조)
#
# Critic obs (BASE 247 + num_objects):
#   robot_dof_pos 27 + robot_dof_vel 27 + hand_pos 18 + hand_vel 36
#   + hand_forces[:, :3] 3 + measured_joint_torque 27
#   + object_pos 3 + object_rot 4 + object_vel 6 + object_goal 3
#   + [onehot] + object_scale 1 + actions 11 + fabric q/qd/qdd 81
#   = 247 + N_obj   (DEXTRAH 원본 "247 + num_objects"와 동일 구조)
# ---------------------------------------------------------------------------
NUM_HAND_POINTS = 6           # palm + 5 fingertips (DEXTRAH hand bodies)
NUM_OBS_BASE        = 193     # onehot 제외 policy obs
NUM_CRITIC_OBS_BASE = 247     # onehot 제외 critic obs
# 실제 차원은 env_cfg 에서 + len(active_object_names) 로 확정
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS      = 480    # 8s: Fabrics arm + per-finger policy
LIFT_WAIT_PHASE_STEPS  = 120    # 2s: keep grasp arm pose, move only joint7
LIFT_PHASE_STEPS       = LIFT_WAIT_PHASE_STEPS
LIFT_START_STEP        = GRASP_PHASE_STEPS                              # 480
EPISODE_STEPS          = GRASP_PHASE_STEPS + LIFT_WAIT_PHASE_STEPS      # 600

LIFT_WAIT_JOINT7_DELTA = 0.31

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
