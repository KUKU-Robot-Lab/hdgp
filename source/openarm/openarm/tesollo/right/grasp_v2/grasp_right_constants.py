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

Action (16D):
  [0:6]   6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11]  5D 시너지(PCA) 계수 → 20관절 폐쇄 진행도
  [11:16] 5D abduction/opposition 절대 목표
          (thumb_1 벌림, thumb_2 대향, index_1, pinky_1, pinky_2)
          thumb_2(대향)는 엄지를 4지 반대편으로 보내는 축이다. -1.57 고정이던 것을
          자유화 — 그 값은 side 접근 전용 튜닝이라 top-down 파지가 불가능했다.
          시너지 basis 가 0인 관절이라 별도 축으로 뺀다 (HAND_ABDUCTION_* 참조)

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
NUM_PALM_ACTION      = 6   # 6D palm pose (Fabrics IK)
NUM_FINGER_ACTION    = 5   # 시너지(PCA) 계수
NUM_ABDUCTION_ACTION = 5   # thumb_1(벌림) / thumb_2(대향) / index_1 / pinky_1 / pinky_2
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION + NUM_ABDUCTION_ACTION  # 16

# ---------------------------------------------------------------------------
# Observation space — Tesollo-native (08-21 재설계: actor obs 물체 수 무관, 일반화)
#
# Policy obs (BASE 193, 물체 수 무관 — onehot/scale/rotation 없음):
#   robot_dof_pos_noisy 27 + robot_dof_vel_noisy 27 (annealing으로 0)
#   + hand_pos_noisy 18 (fabric FK: palm+5tip) + hand_vel_noisy 18 (0)
#   + object_pos_noisy 3 (FP 배포 채널, pos만) + object_goal 3 + actions 16
#   + fabric_q 27 + fabric_qd 27 + fabric_qdd 27
#   = 193   (물체 identity/scale/rotation 없음 → 미학습 신규 물체에도 obs dim 불변)
#
# Critic obs (BASE 252 + num_objects, privileged 유지 — 비대칭 actor-critic):
#   robot_dof_pos 27 + robot_dof_vel 27 + hand_pos 18 + hand_vel 36
#   + hand_forces[:, :3] 3 + measured_joint_torque 27
#   + object_pos 3 + object_rot 4 + object_vel 6 + object_goal 3
#   + [onehot] + object_scale 1 + actions 16 + fabric q/qd/qdd 81
#   = 252 + N_obj   (DEXTRAH 원본 "247 + num_objects" + abduction action 4)
#
# Student obs (distillation, 190 — 현재 보류. actor obs 가 이미 pos-only 라
#   "object_pos_noisy" 3D 하나만 더 빼면 student 가 됨. FP 직접배포로 distillation
#   자체가 불필요해질 가능성 높음 — 재설계 시 재검토):
#   robot_dof_pos_noisy 27 + robot_dof_vel_noisy 27 + hand_pos_noisy 18
#   + hand_vel_noisy 18 + object_goal 3 + actions 16 + fabric q/qd/qdd 81 = 190
# ---------------------------------------------------------------------------
NUM_HAND_POINTS = 6           # palm + 5 fingertips (DEXTRAH hand bodies)
# fingertip 접촉력 15D(5tip × 3축, force_matrix_w Cup-only) 를 actor obs 에 추가.
# 정책이 접촉을 "보고" force closure 를 조율하게 함(실물 RH56F1/Tesollo FT 센서 대응).
# critic 은 privileged 로 distal/middle 접촉력 norm 10D 도 추가.
NUM_OBS_BASE        = 208     # 193 + 15 (fingertip contact force xyz). 물체 수 무관(고정)
NUM_CRITIC_OBS_BASE = 277     # 252 + 15 (tip xyz) + 10 (distal/middle norm)
NUM_STUDENT_OBS     = 205     # distillation student (190 + 15 접촉력; onehot 무관)
# critic 만 실제 차원이 env_cfg 에서 + len(active_object_names) 로 확정된다.
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
