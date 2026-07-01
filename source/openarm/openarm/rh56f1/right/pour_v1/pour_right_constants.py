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

"""상수 정의: inspire_r_pour_v1 (RH56F1 6-DOF 손) — pour_v6 구조 이식본

Tesollo pour_v6(20-DOF 손, robot 27D)의 학습 구조(nullspace action, sim2real
actor obs, B-full Fabrics, deep-tilt/r_beta)를 RH56F1 6 actuated DOF(robot 13D)로
이식. 손은 pour 중 grasp pose 로 고정(per-finger lerp), 팔(6D palm + 1D nullspace)만 학습.

Action (12D) — pour_v6 채택(nullspace 추가):
  [0:6]   6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6]     1D nullspace α (arm 잉여 1-DOF self-motion, palm pose 보존 elbow-swivel)
  [7:12]  5D per-finger lerp: -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE
          (5 손가락 → 6 drive 관절 매핑: thumb 은 2관절[thumb_1,thumb_2], 나머지 1관절씩)

Actor Observation (51D) — pour_v6 sim2real 레이아웃 채택 (proprio/FK/target-relative):
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_grasp_progress:    5  (per-finger grasp 유지 요약, freeze지만 호환 유지)
  left_arm_joint_pos:       7  (target cup FK 원천, real encoder. 왼손 rest 고정 → arm만)
  left_arm_joint_vel:       7
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  target_up_axis:           3
  last_palm_actions:        6  (palm 6채널만, α 제외)
  Total:                   51

Critic Base (77D) — sim-only full-state:
  arm_pos 7 + arm_vel 7 + finger_pos 6 + finger_vel 6
  + rcup_rel 3 + rcup_quat 4 + lcup_rel 3 + lcup_quat 4
  + opening_rel 3 + source_pour_axis 3 + source_up_axis 3
  + transport_stack 8 + binary_contact 5 + tip_force 5 + last_actions 6
  + bead(source/target/cross/spill) 4
  = 77

Critic Extra (35D) — sim-only privileged:
  left_arm_joint_pos:       7  (7 arm; 왼손 rest 고정)
  left_arm_joint_vel:       7
  distal_contact_binary:    5  (RH56F1: distal = fingertip)
  distal_contact_norm:      5
  cup_height_delta:         1
  rho:                      1
  demo_arm_joint_err:       1  (privileged: 현재 j1-4 ↔ demo pour 자세 거리)
  demo_j5_err:              1  (privileged: 현재 j5 ↔ demo tilt 자세 거리)
  demo_target_arm_q:        7  (privileged: NN-매칭된 demo pour arm 7DOF 목표)
  Total:                   35

Critic Total: 77 + 35 = 112D

Episode (@ 60 Hz):
  Pour phase: Fabrics arm policy (6D palm + 1D nullspace) + frozen hand
"""

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
NUM_HAND_DOF  = len(RIGHT_HAND_JOINT_NAMES)   # 6 (RH56F1 drive)
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF    # 13
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space (palm pose + nullspace + per-finger lerp 손가락 제어)
# ---------------------------------------------------------------------------
NUM_PALM_ACTION      = 6   # 6D palm pose (Fabrics IK)
NUM_NULLSPACE_ACTION = 1   # arm 잉여 1-DOF (elbow-swivel self-motion, palm pose 보존)
NUM_FINGER_ACTION    = 5   # per-finger lerp (5 손가락)
NUM_ACTIONS = NUM_PALM_ACTION + NUM_NULLSPACE_ACTION + NUM_FINGER_ACTION  # 12

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 51         # Actor: 7+7+5+7+7+3+3+3+3+6 = 51
NUM_DISTAL_SENSORS  = 5       # RH56F1: distal = fingertip (tip 센서 재사용)
NUM_MIDDLE_SENSORS  = 5       # RH56F1: 중간 phalanx 없음 → zeros
NUM_CRITIC_BASE_OBSERVATIONS = 77
NUM_CRITIC_EXTRAS   = 35      # left_arm(14) + distal(10) + cup_h(1) + rho(1) + demo(9)
NUM_CRITIC_OBSERVATIONS = NUM_CRITIC_BASE_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 112

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

# ---------------------------------------------------------------------------
# Demo pour 자세 (a11~a20 마지막 프레임 joint 평균) — nullspace default_config용
#   pour_v6 이식: demo(pour_v1_a11~20) 궤적은 arm 7-DOF라 RH56F1 손과 무관하게 동일.
#   deep pour는 j4=1.87(팔꿈치 up)+j5=-1.22(롤)로 달성. j1-4를 이 값으로 nullspace 바이어스.
DEMO_POUR_ARM_POSE  = [0.216, 0.633, -0.371, 1.868, -1.217, 0.038, 0.604]

# nullspace self-motion 축 = demo − robot_start. 7번째 action α가 default_config를
#   이 축으로 이동. (tilt 슬라이더 성격 — palm pose 미보존)
NULLSPACE_OFFSET_ARM = [d - s for d, s in zip(DEMO_POUR_ARM_POSE, ARM_START_POSE)]

# 진짜 palm-task nullspace 축 (elbow-swivel). demo 자세에서 palm 6D Jacobian FK 유한차분
#   → SVD → 최소 특이벡터(null vec). J@n≈0(palm pose 보존). (B-full 복귀용)
N_DEMO_NULLSPACE_OFFSET = [-0.2321, -0.4811, 0.5291, 0.0000, -0.3976, 0.1821, 0.4935]
