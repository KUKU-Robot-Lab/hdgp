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

"""Hand/robot preset metadata for 5g_grasp_right_v4.

v3와 동일한 joint/body 구성. v4에서 재사용.
"""

import math
import math as _math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
RIGHT_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    "openarm_left_joint1": -0.5,
    "openarm_left_joint2": -0.5,
    "openarm_left_joint3": 0.6,
    "openarm_left_joint4": 0.7,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": -1.0,
    "openarm_left_finger_joint1": 0.0,
    "openarm_left_finger_joint2": 0.0,
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
HAND_BODY_NAMES_USD = [
    "rl_dg_palm",
    "rl_dg_1_4",
    "rl_dg_2_4",
    "rl_dg_3_4",
    "rl_dg_4_4",
    "rl_dg_5_4",
]

# Fabrics FK taskmap body names (openarm_tesollo_sensor.urdf 기준)
# [0]=palm_link (= rl_dg_palm alias, Fabrics attractor 기준점)
# [1]=palm_x    (palm_link +X 방향 기준, 방향 참조용)
# [2:7]=rl_dg_*_tip (fingertip sensor 링크, 센서 URDF 기준)
FABRIC_HAND_BODY_NAMES = [
    "palm_link",
    "palm_x",
    "rl_dg_1_tip",
    "rl_dg_2_tip",
    "rl_dg_3_tip",
    "rl_dg_4_tip",
    "rl_dg_5_tip",
]


# ---------------------------------------------------------------------------
# Start / grasp poses
# ---------------------------------------------------------------------------
# 완전히 열린 자세 (절대 열림 기준; 사용처 없어도 alias로 유지)
HAND_START_POSE = [
    0.0, 0.0, 0.0, 0.0,   # thumb
    0.0, 0.0, 0.0, 0.0,   # index
    0.0, 0.0, 0.0, 0.0,   # middle
    0.0, 0.0, 0.0, 0.0,   # ring
    0.0, 0.0, 0.0, 0.0,   # pinky
]

# FABRICS 접근 자세 (Approach pose)
# FABRICS pregrasp rollout 동안 유지 + episode 시작 초기 손 자세 + per-finger lerp 기준점
# rj_dg_1_2 (thumb, Z-axis curl, range [-π, 0]) = -1.57 rad
#   → thumb을 opposition 방향으로 pre-curl하여 접근 시 컵과의 collision 방지
#   → episode 중 action[0]=1 → lerp → HAND_GRASP_POSE (thumb_2 = -1.5, ≈ 유지)
#   → 나머지 손가락(1~4)은 0에서 시작하여 lerp로 curl
HAND_APPROACH_POSE = [
    -0.283, -1.241, +0.104, +0.790,   # thumb
    +0.016, +0.527, +0.502, +0.674,   # index
    +0.004, +0.775, +0.170, +1.090,   # middle
    -0.000, +0.668, +0.387, +1.013,   # ring
    +0.000, -0.002, +0.716, +0.889,   # pinky
]ㅊ

# 파지 자세 — v7 test* 학습 결과에서 추출 (100 에피소드 평균, ep_len>=450 AND contacts>=2)
# extract_grasp_pose.py --task 5g_grasp_right-v7 --collect_from_step 450 --min_contacts 2
HAND_GRASP_POSE = [
    -0.354, -1.551, +0.130, +0.988,   # thumb
    +0.020, +0.659, +0.628, +0.843,   # index
    +0.005, +0.969, +0.213, +1.363,   # middle
    -0.000, +0.835, +0.484, +1.266,   # ring
    +0.000, -0.003, +0.895, +1.111,   # pinky
]

# 팔 시작 자세 (Q_REF 근처 안전 자세; old ARM_START_POSE에서 FK ≈ sim (delta≈0))
# Fabrics rollout이 [cup_x-0.167, cup_y-0.09, cup_z+0.04]로 수렴
# j4=0.60: FK z≈0.282, 테이블 안전, 물리 충돌 없음
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal
# ---------------------------------------------------------------------------
# cup spawn center (local frame)
OBJECT_SPAWN_CENTER = [0.40, -0.15, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.40, -0.15, 0.65]

# Pregrasp offset: cup 옆(-Y 방향)에서 접근 (palm_link 기준)
# orientation: ez=90°, ey=0°, ex=90° → palm +X(손바닥 법선)=world +Y, palm +Z(손가락)=world +X
# lift_v1: palm_ee 기준 -6cm → palm_link 기준 ≈ -9cm + rollout 여유 3cm = -12cm
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
        0.65, -0.02, 0.65,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]


# ---------------------------------------------------------------------------
# Direct PD hand control (v4: iCub-style, curl_gate 제거)
# ---------------------------------------------------------------------------

# RL이 직접 제어하는 curl joints (5D action, 손가락당 1D)
HAND_CURL_JOINT_NAMES = [
    "rj_dg_1_2",  # thumb curl (Z, range [-π, 0])
    "rj_dg_2_2",  # index curl (Y, range [0, 2.007])
    "rj_dg_3_2",  # middle curl (Y, range [0, 1.955])
    "rj_dg_4_2",  # ring curl (Y, range [0, 1.902])
    "rj_dg_5_3",  # pinky curl (Y, _1 고정이므로 _3 사용)
]

# 고정 joints (RL 제어 제외)
HAND_FIXED_JOINT_NAMES = [
    "rj_dg_1_1",  # thumb abduction: 0.0 고정
    "rj_dg_2_1",  # index abduction: 0.0 고정
    "rj_dg_3_1",  # middle abduction: 0.0 고정
    "rj_dg_4_1",  # ring abduction: 0.0 고정
    "rj_dg_5_1",  # pinky Z-flex: 0.0 고정
    "rj_dg_5_2",  # pinky abduction: 0.0 고정
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# iCub distal tendon 커플링 (PIP = _3, DIP = _4)
HAND_PIP_JOINT_NAMES = [
    "rj_dg_1_3",  # thumb PIP
    "rj_dg_2_3",  # index PIP
    "rj_dg_3_3",  # middle PIP
    "rj_dg_4_3",  # ring PIP
    "rj_dg_5_4",  # pinky DIP (pinky _3이 curl이므로 _4가 커플링)
]
HAND_DIP_JOINT_NAMES = [
    "rj_dg_1_4",  # thumb DIP
    "rj_dg_2_4",  # index DIP
    "rj_dg_3_4",  # middle DIP
    "rj_dg_4_4",  # ring DIP
]

# 커플링 비율 (HAND_GRASP_POSE 기준)
DISTAL_RATIO_PIP = [0.33, 0.71, 0.71, 0.71, 0.71]
DISTAL_RATIO_DIP = [0.33, 0.71, 0.71, 0.71]

# curl joint 절대 범위 [min, max] (rad)
CURL_JOINT_LIMITS_MIN = [-_math.pi, 0.0,  0.0,  0.0,  0.0]
CURL_JOINT_LIMITS_MAX = [0.0, 2.007, 1.955, 1.902, _math.pi / 2]
