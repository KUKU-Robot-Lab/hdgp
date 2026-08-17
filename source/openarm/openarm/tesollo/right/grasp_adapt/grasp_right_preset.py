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

import math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
# 통일 네이밍(openarm_tesollo_sensor_rl.usd): arm r_aj_/l_aj_, 손 r_hj_<finger>_
# 손가락 순서(finger-major) 보존: thumb,index,middle,ring,pinky
_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
# ★08.17 sensor_rl → bi_s_rl: 좌측이 2-DOF 그리퍼 → DG-5FS 20-DOF 손.
#   존재하지 않는 이름을 init_state 에 남기면 Isaac Lab 이 예외를 던진다.
LEFT_HAND_JOINT_NAMES = [f"l_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    # pour_right_v3 LEFT_ARM_REST_JOINT_POS와 일치시킴:
    # warmstart collection 시 pour env가 이 자세를 사용하므로 OOD 방지
    # FK 결과: target cup pos ≈ [0.268, 0.100, 0.291] (demo target=[0.27, 0.10])
    "l_aj_1": -0.315,
    "l_aj_2": -0.290,
    "l_aj_3":  0.400,
    "l_aj_4":  0.513,
    "l_aj_5":  0.666,
    "l_aj_6": -0.729,
    "l_aj_7": -0.957,
    # 좌손(DG-5FS)은 이 태스크에서 미사용 — 전 관절 0(개방) 고정.
    **{n: 0.0 for n in LEFT_HAND_JOINT_NAMES},
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
HAND_BODY_NAMES_USD = [
    "r_hl_palm",
    "r_hl_thumb_4",
    "r_hl_index_4",
    "r_hl_middle_4",
    "r_hl_ring_4",
    "r_hl_pinky_4",
]

# Fabrics FK taskmap body names (openarm_tesollo_sensor_rl fabrics URDF 기준)
# [0]=r_hl_palm (Fabrics attractor 기준점, old palm_link ↔ 동일 transform 검증됨)
# [1]=r_hl_palm_x (r_hl_palm +X 방향 기준, 방향 참조용; fabrics URDF 전용 helper)
# [2:7]=r_hl_<finger>_tip (fingertip 링크)
FABRIC_HAND_BODY_NAMES = [
    "r_hl_palm",
    "r_hl_palm_x",
    "r_hl_thumb_tip",
    "r_hl_index_tip",
    "r_hl_middle_tip",
    "r_hl_ring_tip",
    "r_hl_pinky_tip",
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
# r_hj_thumb_1 (thumb abduction, X축) = 0.0 고정 (v10: -0.283 → 0.0)
#   → 0으로 고정 시 엄지가 neutral opposition 위치를 유지 (새끼손가락 방향으로 치우치는 현상 방지)
# r_hj_thumb_2 (thumb, Z-axis curl, range [-π, 0]) = -1.241 rad
#   → thumb을 opposition 방향으로 pre-curl하여 접근 시 컵과의 collision 방지
HAND_APPROACH_POSE = [
    +0.000, -1.241, +0.104, +0.790,   # thumb  (v10: r_hj_thumb_1 -0.283→0.0 고정)
    +0.016, +0.527, +0.502, +0.674,   # index
    +0.004, +0.775, +0.170, +1.090,   # middle
    -0.000, +0.668, +0.387, +1.013,   # ring
    +0.000, -0.000, +0.716, +0.889,   # pinky
]

# 파지 자세 — v7 test* 학습 결과에서 추출 후 thumb_1 수동 보정
# r_hj_*_1 = 0.0 고정
HAND_GRASP_POSE = [
    +0.000, -1.570, +0.130, +0.988,   # thumb  (v10: r_hj_thumb_1=0.0 고정)
    +0.000, +0.659, +0.628, +0.843,   # index
    +0.000, +0.969, +0.213, +1.363,   # middle
    -0.000, +0.835, +0.484, +1.266,   # ring
    +0.000, -0.000, +0.895, +1.111,   # pinky
]

# 완전 파지 자세 — HAND_GRASP_POSE 기준 약 20% 더 닫힌 상한/방향
# policy imitation target이 아니라 adaptive closure의 bounded limit로만 사용한다.
# r_hj_*_1 및 thumb r_hj_thumb_2는 HAND_GRASP_POSE와 동일하게 유지한다.
HAND_FULL_GRIP_POSE = [
    +0.000, -1.570, +0.156, +1.186,   # thumb
    +0.000, +0.791, +0.754, +1.012,   # index
    +0.000, +1.163, +0.256, +1.636,   # middle
    -0.000, +1.002, +0.581, +1.519,   # ring
    +0.000, -0.000, +1.074, +1.333,   # pinky
]

# 팔 시작 자세 (Q_REF 근처 안전 자세; old ARM_START_POSE에서 FK ≈ sim (delta≈0))
# Fabrics rollout이 [cup_x-0.167, cup_y-0.09, cup_z+0.04]로 수렴
# j4=0.60: FK z≈0.282, 테이블 안전, 물리 충돌 없음
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal
# ---------------------------------------------------------------------------
# cup spawn center (local frame) — demo 데이터와 일치: source=[0.27,-0.10]
OBJECT_SPAWN_CENTER = [0.27, -0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06

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
    # y_max: -0.02 → 0.22 (source cup y=-0.10에서 target cup y=0.10으로 이송 허용)
    d = math.pi / 180.0
    return [
        0.65, 0.22, 0.65,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]
