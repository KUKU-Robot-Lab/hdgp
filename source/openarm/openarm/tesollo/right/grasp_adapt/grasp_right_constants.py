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

"""상수 정의: 5g_grasp_adapt.

Policy-driven cup-aware grasp 계약:
  정책은 매 step 다음 palm pose와 손 관절 목표를 직접 출력한다.
  arm 7축은 action에 포함하지 않고 Fabrics IK가 palm target에서 계산한다.

Action (27D):
  [0:3]   palm xyz target, normalized [-1, 1] → workspace clamp
  [3:7]   palm quaternion target (x, y, z, w), env에서 unit normalize
  [7:27]  20D hand residual around HAND_GRASP_POSE, normalized [-1, 1] → ±scale rad
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
  tip_force_local_norm:    15  (5 × 3D fingertip body-local 힘 벡터: fx,fy=shear, fz=normal,
                                실 tactile 센서 출력과 정합. grasp_adapt: world→local 변환)
  middle_to_cup_xyz:       15  (5 × 3D FK 기반, sim2real 가능: joint encoder → FK)
  phase_step_ratio:         1  (step counter 기반, 실 로봇 가능)
  [제거] cup_to_goal       3D → transport 단계 제거 (grasp_adapt: grasp/lift만)
  [제거] cup_to_fingertip  15D → fingertip_pos_rel_palm - palm_to_cup 항등식 (완전 중복)
  [제거] binary_contact     5D → tip_force_xyz_norm norm의 하위 집합 (함수적 중복)
  Total:                  133

Optional actor debug observation with oracle mass: 134D

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

Actor Observation without oracle mass: 133D
Critic Total: 133 + 37 = 170D

Episode (6s @ 60Hz = 360 steps):
  approach/grasp/lift/stabilize(height-hold) phase는 상태 기반 reward/gate/diagnostic label이다.
  action override나 scripted lift 없이 policy target만 적용한다.
  lift off 후 컵을 세운 채 lift_target_height(10cm)까지 상승·유지하면 성공이다.

Mass-adaptive force grip (lift/hold 구간 전용):
  - force_quality: grip 법선력 합 / (질량·g) 가 af_target_ratio에 가까울수록 1 (Gaussian).
    질량은 reward 계산에만 쓰는 privileged 값 — actor obs는 무게 hidden (정성적 추론).
  - no_slip_quality: 1 - mean(shear/normal). fingertip→cup 기하 분해로 접선력(shear) 산출.
  - 두 quality는 success_bonus 곱셈계수(+floor) + lift/hold 연속 shaping. done 성공조건은 불변.
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
NUM_PALM_QUAT_ACTION = 4
NUM_PALM_ACTION      = NUM_PALM_POS_ACTION + NUM_PALM_QUAT_ACTION
NUM_FINGER_ACTION    = 20
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 27
PALM_POS_ACTION_SLICE = slice(0, 3)
PALM_QUAT_ACTION_SLICE = slice(3, 7)
FINGER_ACTION_SLICE = slice(7, 27)

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
# Actor obs (133D no-mass, 134D optional mass/debug):
#   arm_joint_pos        7  | arm_joint_vel          7
#   finger_joint_pos    20  | finger_joint_vel       20
#   palm_center_pos      3  | fingertip_pos_rel_palm 15
#   palm_to_cup          3  | last_actions           27
#   tip_force_xyz_norm  15  | middle_to_cup_xyz      15
#   phase_step_ratio     1
#   [optional] bead_mass_normalized 1
#   [제거] cup_to_goal 3D (transport 제거), cup_to_fingertip 15D (항등식), binary_contact 5D
NUM_OBSERVATIONS_NO_MASS = 133
NUM_OBSERVATIONS = 134
NUM_DISTAL_SENSORS  = 5       # rl_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # rl_dg_*_3
NUM_CRITIC_EXTRAS   = 37
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS_NO_MASS + NUM_CRITIC_EXTRAS  # 170

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
EPISODE_STEPS           = 360    # 6s @ 60Hz (grasp/lift/stabilize height-hold)
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
MIN_CONTACTS_FOR_SUCCESS = 4      # (미사용) env는 import하지 않음. 실제 성공 gate는
                                  # grasp_right_env._get_dones 참조 — Phase 1에서 5/5 접촉을
                                  # compute_precision_grasp_mask(엄지+대향2지)로 교체 예정.

# ---------------------------------------------------------------------------
# FABRICS pregrasp
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60

# ---------------------------------------------------------------------------
# Cup geometry
# ---------------------------------------------------------------------------
CUP_RADIUS_APPROX = 0.045  # m, cup_big 반경 (enclosure target 계산용)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
GRAVITY = 9.81  # m/s², adaptive force ratio = ΣF_n / (mass·GRAVITY)

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
ARM_START_POSE      = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
