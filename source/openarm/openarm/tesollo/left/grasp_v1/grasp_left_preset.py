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

"""Hand/robot preset metadata for 5g_grasp_left_v1 (좌팔 제어 미러).

right/grasp_v1(grasp_right_preset)의 좌우 미러 버전. 미러 규칙은
right/grasp_v2 ↔ left/grasp_v2(grasp_left_preset.py)에서 확립된 것과 동일:
  q_left = SIGN * q_right (관절별 부호 매핑)
    arm  j1~j7:               [-1,-1,-1, 1,-1,-1,-1]
    thumb  _1~_4:             [-1,-1,-1,-1]
    index/middle/ring _1~_4:  [-1, 1, 1, 1]
    pinky  _1~_4:             [-1,-1, 1, 1]
  워크스페이스: y 좌표는 y=0 평면 대칭(부호 반전).

★자산 전환(2026-07-26 마이그레이션): right/grasp_v1 은 단일 Tesollo 손 USD
(openarm_tesollo_sensor_rl.usd — 오른손 Tesollo 20관절 + 왼팔은 단순 2-DOF
그리퍼 l_hj_gripper_1/2)를 쓴다. 이 USD 에는 왼손 Tesollo 관절(l_hj_<finger>_j)이
존재하지 않아 좌팔 제어가 물리적으로 불가능하다. left/grasp_v2 가 이미 동일한
문제를 해결한 전례를 그대로 따라 로봇 자산을 openarm_tesollo_bi_rl.usd(양팔
Tesollo 20관절)로 전환한다(grasp_left_env_cfg.py robot_cfg 참조). 고정되는
오른손도 이제 전체 20관절 Tesollo 이므로, RIGHT_ARM_REST_JOINT_POS 는 그리퍼
값(l_hj_gripper_1/2=0.044) 대신 right/grasp_v2 관례대로 손 전체 0 rest 를 쓴다.

★palm orientation G-frame 이식(2026-07-27): grasp_v1 은 원래 palm pose 회전을
P-frame euler_zyx 로 fabric 에 직접 넘겼다(옛 중심 ez=-90,ey=0,ex=-90). 이는
grasp_v2 가 "가짜 top-down"으로 실증·폐기한 규약이다 — tesollo palm 로컬축
(+X=법선, +Z=손가락)이 Allegro/DEXTRAH 규약(+X=손가락, ±Z=법선)과 90° 어긋나
있어 P-frame euler_zyx(ey≈0)로는 손바닥을 물체 쪽으로 정확히 돌릴 수 없다
(왼팔 grip_probe 실증: palm-물체 손목거리 0.17 vs right 0.09 — 왼팔이 물체
옆에 못 붙음). grasp_v2 left 의 검증된 해법을 그대로 이식한다: 회전은 항상
grasp 프레임(G) 에서 명령하고, 상수 회전 C(PALM_GRASP_FRAME_ROT)로 palm 프레임
행렬에 사상한다(R_world_p = R_cmd(G-euler) · Cᵀ). grasp_v1 은 side 접근
단일모드이므로 grasp_v2 의 SIDE/TOPDOWN 분기 없이 PALM_G_EULER_CENTER_SIDE
하나만 쓴다. C 행렬은 좌우 공용(팔 미러와 무관, grasp_v2 원본 그대로).
"""

import math
import math as _math


# ---------------------------------------------------------------------------
# 좌우 미러 부호 매핑 (q_left = SIGN * q_right) — right/grasp_v2 와 동일 규칙
# ---------------------------------------------------------------------------
_HAND_SIGN = [
    -1.0, -1.0, -1.0, -1.0,   # thumb  (07-27: j3/j4 -1→+1 되돌림 — palm orientation 이 진짜
                              # 원인이었고 이 부호 변경은 무관·오히려 악화였다. grasp_v2 규칙 복원)
    -1.0,  1.0,  1.0,  1.0,   # index
    -1.0,  1.0,  1.0,  1.0,   # middle
    -1.0,  1.0,  1.0,  1.0,   # ring
    -1.0, -1.0,  1.0,  1.0,   # pinky
]
_ARM_SIGN = [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]


def _mirror_hand(pose_right: list) -> list:
    """오른손 20D 관절 포즈 → 왼손(부호 매핑)."""
    return [s * v for s, v in zip(_HAND_SIGN, pose_right)]


def _mirror_arm(pose_right: list) -> list:
    """오른팔 7D 관절 포즈 → 왼팔(부호 매핑)."""
    return [s * v for s, v in zip(_ARM_SIGN, pose_right)]


# ---------------------------------------------------------------------------
# Joint groups (제어=왼팔 l_aj/l_hj)
# ---------------------------------------------------------------------------
# 통일 네이밍(openarm_tesollo_bi_rl.usd): 왼팔 l_aj_, 왼손 l_hj_<finger>_
_L_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
LEFT_HAND_JOINT_NAMES = [f"l_hj_{f}_{j}" for f in _L_FINGERS for j in range(1, 5)]
LEFT_ACTUATED_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

# 고정 대상: 오른팔 + 오른손 (bi USD 우측 체인 hold, right/grasp_v2 와 동일 이름 규약
# — "GRIPPER" 는 legacy 명칭 유지, 실제로는 openarm_tesollo_bi_rl.usd 의 전체 손)
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{i}" for f in _L_FINGERS for i in range(1, 5)]
RIGHT_ARM_AND_GRIPPER_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# 고정 오른팔 rest 자세: left 는 왼팔로 파지하므로 오른팔+오른손(bi_rl 우측 체인)을
# 중립 자세로 hold 고정한다. 값 = grasp_v2 left 검증본(전체 0 + r_aj_4=1.57 팔꿈치 굽힘,
# ADR50 완주로 안정 확인), 오른손 전체 0.
RIGHT_ARM_REST_JOINT_POS = {
    "r_aj_1": 0.0,
    "r_aj_2": 0.0,
    "r_aj_3": 0.0,
    "r_aj_4": 1.57,
    "r_aj_5": 0.0,
    "r_aj_6": 0.0,
    "r_aj_7": 0.0,
    **{_n: 0.0 for _n in RIGHT_HAND_JOINT_NAMES},
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics) — 제어 왼손 l_hl_*
# ---------------------------------------------------------------------------
HAND_BODY_NAMES_USD = [
    "l_hl_palm",
    "l_hl_thumb_4",
    "l_hl_index_4",
    "l_hl_middle_4",
    "l_hl_ring_4",
    "l_hl_pinky_4",
]

# Fabrics FK taskmap body names (openarm_tesollo_bi_rl fabrics URDF 기준, 좌팔 미러)
FABRIC_HAND_BODY_NAMES = [
    "l_hl_palm",
    "l_hl_palm_x",
    "l_hl_thumb_tip",
    "l_hl_index_tip",
    "l_hl_middle_tip",
    "l_hl_ring_tip",
    "l_hl_pinky_tip",
]


# ---------------------------------------------------------------------------
# Start / grasp poses (오른손 값의 부호 미러)
# ---------------------------------------------------------------------------
_HAND_START_POSE_RIGHT = [
    0.0, 0.0, 0.0, 0.0,   # thumb
    0.0, 0.0, 0.0, 0.0,   # index
    0.0, 0.0, 0.0, 0.0,   # middle
    0.0, 0.0, 0.0, 0.0,   # ring
    0.0, 0.0, 0.0, 0.0,   # pinky
]
HAND_START_POSE = _mirror_hand(_HAND_START_POSE_RIGHT)

# FABRICS 접근 자세 (오른손 기준값의 미러) — 오른손: thumb _2=-1.57(opposition), _3=-0.5(PIP curl)
_HAND_APPROACH_POSE_RIGHT = [
    0.0, -1.57, -0.5, 0.0,   # thumb
    0.0,  0.0,   0.0, 0.0,   # index
    0.0,  0.0,   0.0, 0.0,   # middle
    0.0,  0.0,   0.0, 0.0,   # ring
    0.0,  0.0,   0.0, 0.0,   # pinky
]
HAND_APPROACH_POSE = _mirror_hand(_HAND_APPROACH_POSE_RIGHT)

# 파지 자세 (per-finger lerp action=+1 목표) — 오른손 값의 미러
_HAND_GRASP_POSE_RIGHT = [
    0.0, -1.57, 1.5, 1.5,   # thumb
    0.0,  1.6,  1.5, 1.5,   # index
    0.0,  1.6,  1.5, 1.5,   # middle
    0.0,  1.6,  1.5, 1.5,   # ring
    0.0,  0.0,  1.5, 1.5,   # pinky
]
HAND_GRASP_POSE = _mirror_hand(_HAND_GRASP_POSE_RIGHT)

# Lift-phase absolute closure anchor — 오른손 값의 미러
_HAND_FULL_GRIP_POSE_RIGHT = [
    0.0, -1.57, 1.8, 1.8,   # thumb
    0.0,  1.9,  1.8, 1.8,   # index
    0.0,  1.9,  1.8, 1.8,   # middle
    0.0,  1.9,  1.8, 1.8,   # ring
    0.0,  0.0,  1.8, 1.8,   # pinky
]
HAND_FULL_GRIP_POSE = _mirror_hand(_HAND_FULL_GRIP_POSE_RIGHT)

# 팔 시작 자세 (오른팔 start 자세의 부호 미러; right/grasp_v2 LEFT_ARM_START_POSE 와 동일 값)
_RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]
LEFT_ARM_START_POSE = _mirror_arm(_RIGHT_ARM_START_POSE)


# ---------------------------------------------------------------------------
# Workspace / goal (y 좌표 y=0 대칭 반전)
# ---------------------------------------------------------------------------
# right: spawn=[0.27,-0.10], goal=[0.27,0.10] → left: spawn=[0.27,+0.10], goal=[0.27,-0.10]
OBJECT_SPAWN_CENTER = [0.27, 0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]

# Pregrasp offset: cup 옆에서 접근 — right 는 -Y, left 는 +Y 방향
PREGRASP_OFFSET = [0.0, 0.12, 0.05]


# ---------------------------------------------------------------------------
# grasp 프레임 (G) — palm orientation 을 DEXTRAH 규약으로 사상
# (grasp_v2 left, grasp_left_preset.py 의 PALM_GRASP_FRAME_ROT 를 그대로 이식 — 값 검증됨)
# ---------------------------------------------------------------------------
# tesollo palm 로컬축(+X=법선, +Z=손가락)은 Allegro/DEXTRAH 규약(+X=손가락, ±Z=법선)과
# 90° 어긋나 있다. euler_zyx(Rz·Ry·Rx) 에서 col0 = (cos ez·cos ey, sin ez·cos ey, -sin ey)
# 이므로 ey≈0 이면 col0(=법선)가 반드시 수평 → 손바닥을 아래로 못 돌린다.
# 상수 회전 C 로 G 규약에서 명령하면 (ez, ey=0, ex=180) 이 법선을 항상 -Z 로 보낸다.
# C 는 좌우 동일(팔 미러와 무관 — palm 로컬축 규약 자체가 같다).
PALM_GRASP_FRAME_ROT = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
]

# side 자세의 G 등가(grasp_v2 left 검증값): right (0,0,-90) ↔ left (0,0,+90).
# grasp_v1 은 side 접근 단일모드이므로 top-down 분기 없이 이 중심 하나만 쓴다.
PALM_G_EULER_CENTER_SIDE = [0.0, 0.0, 90.0]


def palm_pose_mins(max_pose_angle: float) -> list:
    """palm pose 하한 [x,y,z, ez,ey,ex] (회전은 G 규약). y 는 right 워크스페이스의
    y=0 대칭 반전. 회전 중심은 PALM_G_EULER_CENTER_SIDE ± max_pose_angle.
    """
    d = math.pi / 180.0
    return [0.20, -0.22, 0.20] + [
        (v - max_pose_angle) * d for v in PALM_G_EULER_CENTER_SIDE
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    # y_max: right 의 y_max(-0.02→0.22 확장)를 좌우 대칭 반전 → left y_max=0.55
    d = math.pi / 180.0
    return [0.65, 0.55, 0.65] + [
        (v + max_pose_angle) * d for v in PALM_G_EULER_CENTER_SIDE
    ]


# ---------------------------------------------------------------------------
# Direct PD hand control (v4: iCub-style, curl_gate 제거) — 제어 왼손 l_hj_*
# ---------------------------------------------------------------------------

# RL이 직접 제어하는 curl joints (5D action, 손가락당 1D)
HAND_CURL_JOINT_NAMES = [
    "l_hj_thumb_2",   # thumb curl
    "l_hj_index_2",   # index curl
    "l_hj_middle_2",  # middle curl
    "l_hj_ring_2",    # ring curl
    "l_hj_pinky_3",   # pinky curl (_1 고정이므로 _3 사용)
]

# 고정 joints (RL 제어 제외)
HAND_FIXED_JOINT_NAMES = [
    "l_hj_thumb_1",
    "l_hj_index_1",
    "l_hj_middle_1",
    "l_hj_ring_1",
    "l_hj_pinky_1",
    "l_hj_pinky_2",
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# iCub distal tendon 커플링 (PIP = _3, DIP = _4)
HAND_PIP_JOINT_NAMES = [
    "l_hj_thumb_3",
    "l_hj_index_3",
    "l_hj_middle_3",
    "l_hj_ring_3",
    "l_hj_pinky_4",
]
HAND_DIP_JOINT_NAMES = [
    "l_hj_thumb_4",
    "l_hj_index_4",
    "l_hj_middle_4",
    "l_hj_ring_4",
]

# 커플링 비율 (HAND_GRASP_POSE 기준; 크기값이라 부호 무관)
DISTAL_RATIO_PIP = [0.33, 0.71, 0.71, 0.71, 0.71]
DISTAL_RATIO_DIP = [0.33, 0.71, 0.71, 0.71]

# curl joint 절대 범위 [min, max] (rad) — 오른손 범위의 미러.
# thumb_2: right [-π, 0] → left [0, π] (부호반전 → min/max 스왑). 나머지: 부호 유지.
CURL_JOINT_LIMITS_MIN = [0.0, 0.0,  0.0,  0.0,  0.0]
CURL_JOINT_LIMITS_MAX = [_math.pi, 2.007, 1.955, 1.902, _math.pi / 2]
