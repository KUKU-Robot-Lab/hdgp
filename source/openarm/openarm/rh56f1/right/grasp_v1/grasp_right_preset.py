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

"""inspire_r_grasp_v1 preset (RH56F1 6-DOF underactuated 손).

Tesollo 20-DOF → RH56F1 6 actuated DOF 포팅.
  - 손 drive 6관절: thumb_1(측배), thumb_2(굽힘), index_1, middle_1, ring_1, little_1
  - mimic 추종(thumb_3/4, *_2)은 USD PhysxMimicJoint 가 자동 처리 → RL 제어 대상 아님.
  - 센서: palm_force_sensor(실 로봇 센서, actor) + fingertip 접촉(sim 전용, critic)
  - 팔/workspace/palm pose 규약은 Tesollo 와 동일 (palm 가상프레임 기하 일치).
"""

import math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
RIGHT_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]

# RH56F1 actuated drive 6관절 (USD revolute joint 이름, _joint 접미사 포함)
RIGHT_HAND_JOINT_NAMES = [
    "rh56f1_right_right_thumb_1_joint",   # thumb abduction (Z), 0~2.094
    "rh56f1_right_right_thumb_2_joint",   # thumb flexion drive (Z), 0~0.475
    "rh56f1_right_right_index_1_joint",   # index flexion (Z), 0~1.529
    "rh56f1_right_right_middle_1_joint",  # middle flexion
    "rh56f1_right_right_ring_1_joint",    # ring flexion
    "rh56f1_right_right_little_1_joint",  # little flexion
]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# mimic 추종 관절 (RL 비제어, 참고용)
RIGHT_HAND_MIMIC_JOINT_NAMES = [
    "rh56f1_right_right_thumb_3_joint",   # = thumb_2 × 1.1425
    "rh56f1_right_right_thumb_4_joint",   # = thumb_3 × 0.7508
    "rh56f1_right_right_index_2_joint",   # = index_1 × 1.1169
    "rh56f1_right_right_middle_2_joint",
    "rh56f1_right_right_ring_2_joint",
    "rh56f1_right_right_little_2_joint",
]

LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]

# 좌측도 RH56F1 손 (12 DOF: drive 6 + mimic 6). 학습 비사용 → 전체 lock.
# (기존 Tesollo 의 openarm_left_finger_joint1/2 2-DOF 그리퍼는 존재하지 않음)
LEFT_HAND_DRIVE_JOINT_NAMES = [
    "rh56f1_left_left_thumb_1_joint",
    "rh56f1_left_left_thumb_2_joint",
    "rh56f1_left_left_index_1_joint",
    "rh56f1_left_left_middle_1_joint",
    "rh56f1_left_left_ring_1_joint",
    "rh56f1_left_left_little_1_joint",
]
LEFT_HAND_MIMIC_JOINT_NAMES = [
    "rh56f1_left_left_thumb_3_joint",
    "rh56f1_left_left_thumb_4_joint",
    "rh56f1_left_left_index_2_joint",
    "rh56f1_left_left_middle_2_joint",
    "rh56f1_left_left_ring_2_joint",
    "rh56f1_left_left_little_2_joint",
]
LEFT_HAND_JOINT_NAMES = LEFT_HAND_DRIVE_JOINT_NAMES + LEFT_HAND_MIMIC_JOINT_NAMES
LEFT_ARM_AND_HAND_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

# 좌팔 rest 자세 (학습 비사용, 우측 작업공간 침범 방지). 좌손은 0(열림)으로 lock.
LEFT_ARM_REST_JOINT_POS = {
    "openarm_left_joint1": -0.315,
    "openarm_left_joint2": -0.290,
    "openarm_left_joint3":  0.400,
    "openarm_left_joint4":  0.513,
    "openarm_left_joint5":  0.666,
    "openarm_left_joint6": -0.729,
    "openarm_left_joint7": -0.957,
}
LEFT_HAND_REST_JOINT_POS = {name: 0.0 for name in LEFT_HAND_JOINT_NAMES}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
# 보상/관측에 쓰는 USD body (palm + 5 말단 손가락 링크).
# Phase 0 검증: fingertip force_sensor 링크는 병합 소멸 → 생존 말단 링크 사용.
#   [0]=palm(plam_force_sensor, 실 센서 body)
#   [1:6]=thumb_4, index_2, middle_2, ring_2, little_2
HAND_BODY_NAMES_USD = [
    "rh56f1_right_plam_force_sensor",
    "rh56f1_right_right_thumb_4",
    "rh56f1_right_right_index_2",
    "rh56f1_right_right_middle_2",
    "rh56f1_right_right_ring_2",
    "rh56f1_right_right_little_2",
]

# RH56F1 의 *_force_sensor 는 모두 실 하드웨어 힘센서 (palm + 5 fingertip) → actor obs.
# 실 로봇 palm 힘센서 body
PALM_FORCE_SENSOR_BODY = "rh56f1_right_plam_force_sensor"

# fingertip 힘센서 body — *_force_sensor 링크가 USD 에서 말단 링크로 병합되어,
# 말단 링크(thumb_4, *_2)의 ContactSensor 가 force_sensor 패드 접촉을 포착한다.
FINGERTIP_SENSOR_BODIES = [
    "rh56f1_right_right_thumb_4",
    "rh56f1_right_right_index_2",
    "rh56f1_right_right_middle_2",
    "rh56f1_right_right_ring_2",
    "rh56f1_right_right_little_2",
]

# Fabrics FK taskmap body 이름 (openarm_rh56f1.urdf 기준)
#   [0]=palm_link  [1]=palm_x  [2:7]=fingertip 5개
FABRIC_HAND_BODY_NAMES = [
    "palm_link",
    "palm_x",
    "rh56f1_tip_thumb",
    "rh56f1_tip_index",
    "rh56f1_tip_middle",
    "rh56f1_tip_ring",
    "rh56f1_tip_little",
]


# ---------------------------------------------------------------------------
# Hand joint limits (rad) — RH56F1 drive 6관절 (모두 lower=0)
# ---------------------------------------------------------------------------
HAND_JOINT_LIMITS_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HAND_JOINT_LIMITS_MAX = [
    2.0943951,   # thumb_1 abduction
    0.4745550,   # thumb_2 flexion drive
    1.5285594,   # index_1
    1.5285594,   # middle_1
    1.5285594,   # ring_1
    1.5285594,   # little_1
]


# ---------------------------------------------------------------------------
# Hand poses (6D: thumb_1, thumb_2, index_1, middle_1, ring_1, little_1)
# ---------------------------------------------------------------------------
# 완전히 열린 자세
HAND_START_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# FABRICS 접근 자세 (pregrasp): thumb opposition 유지 + thumb_2를 살짝 벌려 컵 spawn clearance 확보
HAND_APPROACH_POSE = [
    1.57,   # thumb_1 abduction → opposition 방향
    0.15,   # thumb_2: reset에서 엄지 말단 링크 clearance 확보
    0.00,   # index_1
    0.00,   # middle_1
    0.00,   # ring_1
    0.00,   # little_1
]

# 파지 자세 (grasp): thumb opposition + non-thumb 우선 굽힘.
# test3에서는 thumb force만 커지고 others_avg_force가 거의 0에 머물렀으므로,
# lift 전 grasp target은 엄지 말단 curl을 낮추고 네 손가락 closure를 앞당긴다.
HAND_GRASP_POSE = [
    1.57,   # thumb_1
    0.24,   # thumb_2
    0.90,   # index_1
    0.90,   # middle_1
    0.90,   # ring_1
    0.90,   # little_1
]

# 완전 파지 자세 (adaptive closure 상한)
HAND_FULL_GRIP_POSE = [
    1.57,   # thumb_1
    0.4745, # thumb_2 (한계)
    1.50,   # index_1
    1.50,   # middle_1
    1.50,   # ring_1
    1.50,   # little_1
]


# ---------------------------------------------------------------------------
# Arm start pose (Tesollo 와 동일 — 팔 동일)
# ---------------------------------------------------------------------------
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal (Tesollo 와 동일)
# ---------------------------------------------------------------------------
OBJECT_SPAWN_CENTER = [0.27, -0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, 0.10, 0.65]

# Pregrasp offset (palm_link 기준)
PREGRASP_OFFSET = [0.0, -0.12, 0.05]


def palm_pose_mins(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.20, -0.55, 0.20,
        (90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.65, 0.22, 0.65,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]
