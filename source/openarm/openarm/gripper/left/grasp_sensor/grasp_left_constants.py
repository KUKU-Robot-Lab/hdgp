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

"""gripper/left/grasp_sensor 차원·에피소드 상수.

Action (7D):
  [0:6]  TCP(palm) 6D delta (x, y, z, ez, ey, ex) → Fabrics IK → l_aj_1..7
         기준점은 홈이 아니라 **컵-정준 pregrasp** (right/grasp_sensor A0 규약과 동일).
  [6]    그리퍼 1D: -1 → 개방(0.044 m), +1 → 폐쇄(0.0 m)
         ⚠ 목표는 l_hj_gripper_1 에만 준다 (gripper_2 는 USD PhysX mimic).

Actor Observation (48D) — 전부 실기 취득 가능:
  arm_joint_pos            7
  arm_joint_vel            7
  gripper_pos              1
  gripper_vel              1
  tcp_pos (env-local)      3
  finger_pos_rel_tcp       6   (2 × 3D)
  tcp_to_cup               3
  cup_to_finger            6   (2 × 3D)
  finger_contact_force     6   (2 × 3D, 손가락 F/T, 10N 정규화)
  gripper_pos_err          1   (지령 − 실측; 막힌 만큼 커지는 파지력 대리 신호)
  last_actions             7
  ------------------------------
  Total                   48

  ★물체 onehot 없음 — 이 태스크는 cup_big 단일 종이다(right/grasp_sensor 는 8종 MultiAsset).

Critic Extra (14D) — sim-only privileged:
  cup_lin_vel              3
  cup_ang_vel              3
  cup_rot (quat)           4
  cup_height_delta         1
  phase_step_ratio         1
  finger_to_cup_signed_dist 2
  ------------------------------
  Critic Total            62

⚠ 아래 숫자는 주석이 아니라 **상수 산술**로만 유지한다. right/grasp_sensor 는 주석의
  산술(146/183)이 실제 값(154/191)과 어긋난 채 남아 있다 — 여기서는 tests 가 실제
  concat 폭과 대조해 고정한다(tests/test_dims_and_contract.py).
"""

from __future__ import annotations

from .grasp_left_preset import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_GRIPPER_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF = len(LEFT_ARM_JOINT_NAMES)            # 7
NUM_GRIPPER_JOINTS = len(LEFT_GRIPPER_JOINT_NAMES)  # 2 (mimic 포함)
NUM_GRIPPER_DOF = 1                                 # 실제 자유도 (l_hj_gripper_1 만 지령)
NUM_FINGERS = 2
NUM_IDLE_ARM_DOF = len(RIGHT_ARM_JOINT_NAMES)       # 7
NUM_IDLE_HAND_DOF = len(RIGHT_HAND_JOINT_NAMES)     # 20

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION = 6
NUM_GRIPPER_ACTION = 1
NUM_ACTIONS = NUM_PALM_ACTION + NUM_GRIPPER_ACTION   # 7

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
_OBS_TERMS = (
    NUM_ARM_DOF,            # arm_joint_pos
    NUM_ARM_DOF,            # arm_joint_vel
    NUM_GRIPPER_DOF,        # gripper_pos
    NUM_GRIPPER_DOF,        # gripper_vel
    3,                      # tcp_pos
    NUM_FINGERS * 3,        # finger_pos_rel_tcp
    3,                      # tcp_to_cup
    NUM_FINGERS * 3,        # cup_to_finger
    NUM_FINGERS * 3,        # finger_contact_force
    NUM_GRIPPER_DOF,        # gripper_pos_err
    NUM_ACTIONS,            # last_actions
)
NUM_OBSERVATIONS = sum(_OBS_TERMS)                   # 48

_CRITIC_EXTRA_TERMS = (
    3,                      # cup_lin_vel
    3,                      # cup_ang_vel
    4,                      # cup_rot (quat)
    1,                      # cup_height_delta
    1,                      # phase_step_ratio
    NUM_FINGERS,            # finger_to_cup_signed_dist
)
NUM_CRITIC_EXTRAS = sum(_CRITIC_EXTRA_TERMS)         # 14
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS   # 62

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz) — right/grasp_sensor 와 동일 구조
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS = 480        # 8s: Fabrics 접근 + 그리퍼 폐쇄
LIFT_PHASE_STEPS = 120         # 2s: 래치 후 수직 리프트 램프
EPISODE_STEPS = GRASP_PHASE_STEPS + LIFT_PHASE_STEPS   # 600

# ---------------------------------------------------------------------------
# Contact / force
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD = 0.1    # N, binary 접촉 판정 (right/grasp_sensor S5 실측 유효값)
CONTACT_FORCE_MAX = 10.0         # N, 정규화 분모

# 그리퍼 관절 위치 오차(지령−실측) 정규화 분모 [m].
# 왜 이 관측인가: 힘 ∝ stiffness × (지령−실측). 손끝 F/T 는 접촉면이 어긋나면 0 을 읽지만
# 관절 오차는 컵에 막힌 만큼 항상 생긴다 — 실기에서도 지령과 joint_states 차이로 계산 가능.
# 스트로크가 0.044 m 이므로 그 절반을 분모로 잡는다(포화 없이 대부분 커버).
GRIPPER_POS_ERR_MAX = 0.022      # m

# ---------------------------------------------------------------------------
# Fabrics
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60
