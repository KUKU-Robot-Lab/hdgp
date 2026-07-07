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

Actor Observation (96D) — sim2real 가능 (실 센서 + FK):
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:         6   (drive 6)
  finger_joint_vel:         6
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15   (5 × 3D, FK)
  palm_to_cup_pos:          3
  cup_rot (quat):           4
  last_actions:            12
  middle_to_cup_xyz:       15   (5 × 3D, FK)
  phase_step_ratio:         1
  palm_binary:              1   (실 palm 힘센서 접촉)
  palm_force_norm:          1   (실 palm 힘센서 크기)
  tip_force_xyz:           15   (5 × 3D, 실 fingertip 힘센서 — *_force_sensor)
  Total:                   96

  ※ RH56F1 의 *_force_sensor (palm + 5 fingertip) 는 모두 실 하드웨어 센서.
    fingertip force_sensor 링크는 USD 에서 말단 링크(*_2, thumb_4)로 병합되어
    해당 body 의 ContactSensor 가 force_sensor 패드 접촉을 그대로 포착한다.

Actor Observation with oracle mass: 97D

Critic Extra (23D) — sim-only privileged:
  bead_mass_normalized:        1
  cup_lin_vel:                 3
  cup_ang_vel:                 3
  cup_height_delta:            1
  tip_contact_binary:          5   (privileged 정밀 접촉 flag)
  fingertip_to_cup_signed_dist: 5
  middle_contact_binary:       5   (근위 마디 접촉 = envelope, sim-only)
  Total:                      23

Critic Total: 96 + 23 = 119D

Episode (10s @ 60Hz = 600 steps):
  Grasp / Lift / Stabilize
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
NUM_PALM_ACTION = 6    # 6D palm pose (Fabrics IK → arm 7 DOF)
NUM_HAND_PCA    = 5    # 5D hand PCA action (DEXTRAH 물체파지, fabric hand_mode="pca")
NUM_ACTIONS = NUM_PALM_ACTION + NUM_HAND_PCA  # 11 (palm 6 + PCA 5)

# hand PCA action 범위 (uncentered 투영 좌표, fabric LinearMap 과 정합).
# 출처: assets/demograsp_references/rh56f1_grasp_pca5.pt (pca_action_mins/maxs).
# fabric RH56F1_HAND_PCA_MATRIX 와 동일 단일 진실원(compute_rh56f1_grasp_pca.py).
HAND_PCA_MINS = [0.2258177250623703, -0.07913326472043991, -0.37289050221443176,
                 -0.19040870666503906, -0.06522199511528015]
HAND_PCA_MAXS = [3.7262845039367676, 0.8179822564125061, 0.1744416356086731,
                 0.025599155575037003, 0.04043136164546013]

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
# grasp_v1 96D 골격 유지 + DEXTRAH goal-driven 개조:
#   last_actions 12→11 (action 차원), object_goal 3D 추가 → 96 − 1 + 3 = 98
NUM_OBSERVATIONS = 98
NUM_OBSERVATIONS_WITH_MASS = 99
NUM_OBSERVATIONS_NO_MASS = NUM_OBSERVATIONS

NUM_TIP_SENSORS   = 5     # fingertip 힘센서 (실 센서, actor) — *_force_sensor → 말단 링크
NUM_PALM_SENSORS  = 1     # palm 힘센서 (실 센서, actor) — palm_force_sensor
# 18 → 23: middle(proximal) phalanx 접촉 binary 5D 를 critic privileged 로 추가.
# (실물엔 tip 힘센서만 있으나 sim-only ContactSensor 로 envelope 접촉을 critic 에 노출)
NUM_CRITIC_EXTRAS = 23
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 98 + 23 = 121

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS     = 420    # 7s
LIFT_PHASE_STEPS      = 120    # 2s
STABILIZE_PHASE_STEPS = 60     # 1s
LIFT_START_STEP       = GRASP_PHASE_STEPS
STABILIZE_START_STEP  = LIFT_START_STEP + LIFT_PHASE_STEPS
EPISODE_STEPS = 600
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
