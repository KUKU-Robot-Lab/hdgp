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

"""상수 정의: 5g_pour_right_v4 (전면 재설계)

Action (6D) — palm pose만 (손은 grasp_hold freeze):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF

Actor Observation (55D) — sim2real 가능한 proprio/FK/target-relative 상태:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_grasp_progress:    5  (per-finger grasp 유지 요약, freeze지만 호환 유지)
  left_arm_joint_pos:       9  (target cup FK 원천, real encoder)
  left_arm_joint_vel:       9
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  target_up_axis:           3
  last_palm_actions:        6
  Total:                   55

Critic Base (105D) — sim-only full-state:
  actor full-state layout: hand pos/vel, fingertip contact/force, last_actions(6) 포함.

Critic Extra (39D) — sim-only privileged:
  left_arm_joint_pos:       9
  left_arm_joint_vel:       9
  distal_contact_binary:    5  (r_hl_*_4)
  distal_contact_norm:      5
  cup_height_delta:         1
  rho:                      1
  demo_arm_joint_err:       1  (privileged: 현재 j1-4 ↔ demo pour 자세 거리)
  demo_j5_err:              1  (privileged: 현재 j5 ↔ demo tilt 자세 거리)
  demo_target_arm_q:        7  (privileged: NN-매칭된 demo pour arm 7DOF 목표)
  Total:                   39

Critic Total: 105 + 39 = 144D

Episode (20s @ 60Hz):
  Pour phase: Fabrics arm policy (6D palm) + frozen hand. warmstart 컵-든-자세 시작.
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
NUM_HAND_DOF  = len(RIGHT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF     # 27
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space (palm pose + nullspace + per-finger lerp 손가락 제어)
# ---------------------------------------------------------------------------
NUM_PALM_ACTION = 6   # 6D palm pose (Fabrics IK)
NUM_NULLSPACE_ACTION = 1   # [2b] arm 잉여 1-DOF (팔꿈치↔손목 self-motion) 정책 제어
NUM_HAND_ACTION = 5   # [v7 복원] per-finger lerp: action[7:12] ∈[-1,1] → open(approach)~grasp 사이
NUM_ACTIONS = NUM_PALM_ACTION + NUM_NULLSPACE_ACTION + NUM_HAND_ACTION  # 12 (obs는 불변: finger_grasp_progress가 손 상태)

# ★좌팔(receiver) TCP 위치 3D — `left_arm_action_enable=True` 일 때만 액션에 붙는다.
#   both/pour_sensor 에서 이식(그쪽은 상시 15D). 관측은 **양쪽 모두 55/144 로 동일**하다
#   — 좌팔 관절(18ch)과 주둥이→입구 상대벡터가 이미 actor obs 에 있어서, 좌팔이 움직이면
#   그 채널들이 저절로 따라간다. 그래서 액션만 늘리면 된다.
NUM_LEFT_TCP_ACTION = 3
NUM_ACTIONS_WITH_LEFT = NUM_ACTIONS + NUM_LEFT_TCP_ACTION  # 15

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 55         # Actor: 7+7+5+9+9+3+3+3+3+6 = 55
NUM_DISTAL_SENSORS  = 5       # r_hl_*_4
NUM_MIDDLE_SENSORS  = 5       # r_hl_*_3
NUM_CRITIC_BASE_OBSERVATIONS = 105   # last_actions 11→6 반영 (110-5)
NUM_CRITIC_EXTRAS = 39   # left_arm(18)+distal(10)+cup_h(1)+rho(1)+demo(arm_err1+j5_err1+target_q7=9)
NUM_CRITIC_OBSERVATIONS = NUM_CRITIC_BASE_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 144

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
#   kinematic 검증(2026-06-19): deep pour는 j4=1.87(팔꿈치 up)+j5=-1.22(롤)로 j6를 거의
#   안 쓰고(sat 0.05) 달성. robot_start(j4=0.60)로 nullspace를 풀면 j6가 포화되어 tilt 막힘.
#   pour_v4는 j1-4만 이 값으로 nullspace를 바이어스(팔꿈치 up), j5-7은 robot_start 유지.
DEMO_POUR_ARM_POSE  = [0.216, 0.633, -0.371, 1.868, -1.217, 0.038, 0.604]

# [2b] nullspace self-motion 축 = demo − robot_start. 7번째 action α가 default_config를
#   이 축으로 이동. v5와 동일 상수(ablation 공통). v4 baseline=demo, v5 baseline=robot_start.
#   ⚠️ 이 축은 palm pose를 보존하지 않음(= tilt 슬라이더). FK 검증: demo−start 방향 이동 시
#   cup-up tilt가 ±0.09 변함 → α가 사실상 tilt를 직접 바꿔, 정책이 drift 회피로 α를 낮춤.
NULLSPACE_OFFSET_ARM = [d - s for d, s in zip(DEMO_POUR_ARM_POSE, ARM_START_POSE)]

# [stage2] 진짜 palm-task nullspace 축 (elbow-swivel). demo 자세에서 palm 6D Jacobian을
#   FK 유한차분 → SVD → 최소 특이벡터(null vec)로 산출. J@n≈0(palm pose 보존).
#   (B-full 복귀: palm 3D nullspace 실험 종결 — soft cspace로 j5 미구동 확정, 6D B-full 복귀)
N_DEMO_NULLSPACE_OFFSET = [-0.2321, -0.4811, 0.5291, 0.0000, -0.3976, 0.1821, 0.4935]
