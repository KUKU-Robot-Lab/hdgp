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

"""상수 정의: inspire_r_grasp_v1 (RH56F1 6-DOF 손)

Tesollo 20-DOF(action 26D, obs 144D) → RH56F1 6 actuated DOF 포팅.

Action (12D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:12] 6D absolute hand synergy target (drive 6관절)
         thumb_1, thumb_2, index_1, middle_1, ring_1, little_1

Actor Observation (99D) — sim2real 가능 (실 센서 + FK):
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:         6   (drive 6)
  finger_joint_vel:         6
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15   (5 × 3D, FK)
  palm_to_cup_pos:          3
  cup_to_goal:              3
  cup_rot (quat):           4
  last_actions:            12
  middle_to_cup_xyz:       15   (5 × 3D, FK)
  phase_step_ratio:         1
  palm_binary:              1   (실 palm 힘센서 접촉)
  palm_force_norm:          1   (실 palm 힘센서 크기)
  tip_force_xyz:           15   (5 × 3D, 실 fingertip 힘센서 — *_force_sensor)
  Total:                   99

  ※ RH56F1 의 *_force_sensor (palm + 5 fingertip) 는 모두 실 하드웨어 센서.
    fingertip force_sensor 링크는 USD 에서 말단 링크(*_2, thumb_4)로 병합되어
    해당 body 의 ContactSensor 가 force_sensor 패드 접촉을 그대로 포착한다.

Actor Observation with oracle mass: 100D

Critic Extra (18D) — sim-only privileged:
  bead_mass_normalized:        1
  cup_lin_vel:                 3
  cup_ang_vel:                 3
  cup_height_delta:            1
  tip_contact_binary:          5   (privileged 정밀 접촉 flag)
  fingertip_to_cup_signed_dist: 5
  Total:                      18

Critic Total: 99 + 18 = 117D

Episode (12s @ 60Hz = 720 steps):
  Grasp / Lift / Stabilize / Transport
"""

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
NUM_HAND_DOF  = len(RIGHT_HAND_JOINT_NAMES)   # 6
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF    # 13
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION   = 6    # 6D palm pose (Fabrics IK)
NUM_FINGER_ACTION = 6    # 6D absolute hand synergy target (drive 6)
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 12

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 99
NUM_OBSERVATIONS_WITH_MASS = 100
NUM_OBSERVATIONS_NO_MASS = NUM_OBSERVATIONS

NUM_TIP_SENSORS   = 5     # fingertip 힘센서 (실 센서, actor) — *_force_sensor → 말단 링크
NUM_PALM_SENSORS  = 1     # palm 힘센서 (실 센서, actor) — palm_force_sensor
NUM_CRITIC_EXTRAS = 18
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 117

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS     = 420    # 7s
LIFT_PHASE_STEPS      = 120    # 2s
STABILIZE_PHASE_STEPS = 60     # 1s
TRANSPORT_PHASE_STEPS = 120    # 2s
LIFT_START_STEP       = GRASP_PHASE_STEPS
STABILIZE_START_STEP  = LIFT_START_STEP + LIFT_PHASE_STEPS
TRANSPORT_START_STEP  = STABILIZE_START_STEP + STABILIZE_PHASE_STEPS
EPISODE_STEPS         = (
    GRASP_PHASE_STEPS
    + LIFT_PHASE_STEPS
    + STABILIZE_PHASE_STEPS
    + TRANSPORT_PHASE_STEPS
)
PRELOAD_START_STEP = 340    # lift 직전 80 step

LIFT_Z_DELTA = 0.10    # 10cm 수직 상승

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  정규화 분모
MIN_CONTACTS_FOR_SUCCESS = 3      # legacy constant; runtime success uses cfg gates.

# ---------------------------------------------------------------------------
# FABRICS pregrasp
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60

# ---------------------------------------------------------------------------
# Cup geometry
# ---------------------------------------------------------------------------
CUP_RADIUS_APPROX = 0.035  # m, cup_middle radius

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
ARM_START_POSE      = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
