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

"""Hand/robot preset metadata for 5g_grasp_left_v2 (좌팔 제어 미러).

right/grasp_v2(grasp_right_preset)의 좌우 미러 버전.
- 제어 대상: 왼팔(l_aj) + 왼손 Teosllo 20관절(l_hj)
- 고정 대상: 오른팔 + 오른손 (RIGHT_ARM_REST_JOINT_POS)

미러 규칙(generate_left_fabric_urdf.py + OpenArmTeoslloLeftPoseFabric 검증 기준):
  q_left = SIGN * q_right (관절별 부호 매핑)
    arm  j1~j7:               [-1,-1,-1, 1,-1,-1,-1]
    thumb  _1~_4:             [-1,-1,-1,-1]
    index/middle/ring _1~_4:  [-1, 1, 1, 1]
    pinky  _1~_4:             [-1,-1, 1, 1]
  워크스페이스: y 좌표는 y=0 평면 대칭(부호 반전).
  palm orientation(euler_zyx): ez,ex 부호 반전, ey 유지 (fabric 기본자세와 일치).
"""

import math
import math as _math


# ---------------------------------------------------------------------------
# 좌우 미러 부호 매핑 (q_left = SIGN * q_right)
# ---------------------------------------------------------------------------
# 손 20관절 순서(finger-major): thumb, index, middle, ring, pinky × (_1.._4)
_HAND_SIGN = [
    -1.0, -1.0, -1.0, -1.0,   # thumb
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
# 손가락 순서(finger-major) 보존: thumb,index,middle,ring,pinky
_L_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
LEFT_HAND_JOINT_NAMES = [f"l_hj_{f}_{j}" for f in _L_FINGERS for j in range(1, 5)]
LEFT_ACTUATED_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

# 고정 대상: 오른팔 + 오른손 (bi USD 우측 체인 hold)
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [
    f"r_hj_{f}_{i}" for f in _L_FINGERS for i in range(1, 5)
]
RIGHT_ARM_AND_GRIPPER_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# 고정 오른팔 rest 자세: grasp_v2는 왼팔만 제어 — 오른팔은 순수 고정(장식)이므로
# pour warmstart 자세를 유지할 이유 없음. 렌더에서 팔이 이상한 자세로 흔들리던 문제
# 해소: 전체 0 + r_aj_4=1.57(팔꿈치 굽힘)로 깔끔한 중립 고정.
# (right/grasp_v2 의 고정 왼팔 중립화 {l_aj_4:1.57} 와 대칭 — arm4 부호 +1이라 값 동일)
RIGHT_ARM_REST_JOINT_POS = {
    "r_aj_1": 0.0,
    "r_aj_2": 0.0,
    "r_aj_3": 0.0,
    "r_aj_4": 1.57,
    "r_aj_5": 0.0,
    "r_aj_6": 0.0,
    "r_aj_7": 0.0,
    # 오른손 tesollo: 미사용 → 전체 0 rest (self-collision disabled)
    **{_n: 0.0 for _n in RIGHT_HAND_JOINT_NAMES},
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics) — 제어 왼손 l_hl_*
# ---------------------------------------------------------------------------
# sim USD(openarm_tesollo_bi_rl) body 이름
HAND_BODY_NAMES_USD = [
    "l_hl_palm",
    "l_hl_thumb_4",
    "l_hl_index_4",
    "l_hl_middle_4",
    "l_hl_ring_4",
    "l_hl_pinky_4",
]

# Isaac USD(l_hl_*) 참조용. grasp env 의 body index 조회에 쓰인다.
# 주의: 좌팔 fabric URDF(openarm_tesollo_left)는 우측 fabric URDF 의 미러이며
# 내부 링크/조인트 이름을 보존한다(palm_link, rl_dg_*_tip 등). fabric 내부
# 프레임 이름은 우측과 동일하다.
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

# FABRICS 접근 자세 (오른손 기준값의 미러)
# 오른손: thumb _2=-1.57(opposition), _3=-0.5(PIP curl)
# 왼손: 부호 반전 → thumb _2=+1.57, _3=+0.5
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

# 팔 시작 자세 (오른팔 start 자세의 부호 미러)
_RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]
LEFT_ARM_START_POSE = _mirror_arm(_RIGHT_ARM_START_POSE)


# ---------------------------------------------------------------------------
# Workspace / goal (y 좌표 y=0 대칭 반전)
# ---------------------------------------------------------------------------
# right: source=[0.27,-0.10], target=[0.27,0.10]
#   → left: source=[0.27,+0.10], target=[0.27,-0.10]
OBJECT_SPAWN_CENTER = [0.27, 0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]

# Pregrasp offset: cup 옆에서 접근 — right 는 -Y, left 는 +Y 방향
PREGRASP_OFFSET = [0.0, 0.12, 0.05]

# Pregrasp/reset palm 접근 방향 euler (deg) — env.py IK 타깃이 참조.
# left=-90 (palm 경계 [-180°,0°] 중앙 = +y측 side-approach). right +90의 미러.
# lstm_test1 실패 원인: env.py의 +90 하드코드가 left 경계에 0°로 clamp되어
# pregrasp 90° 뒤틀림 → 파지 불가·palm_orient hacking 붕괴 (analysis.md 참조).
PREGRASP_EULER_EZ_DEG = -90.0
PREGRASP_EULER_EX_DEG = -90.0

# ---------------------------------------------------------------------------
# Top-down 접근 (기본) / side 접근 (cup 전용) — right 의 y/부호 미러
# ---------------------------------------------------------------------------
# DEXTRAH 와 동일하게 top-down 이 기본. side-approach(|ex|=90)는 감쌀 수직 옆면이
# 있어야 작동하고, 납작한 물체에서는 손끝이 테이블에 눌린 채 수평 전진해 물체를
# 쳐내는 불도저 실패가 난다(right ep_14000 관찰 probe 실증).
# 물체 높이 규칙 분기(lstm_test2)는 ADR 회전이 커지면 원통이 누우면서 "납작"에서
# 빠져 topdown_frac 이 자멸했다 → 물체 이름 기반 고정 분기로 대체.
#
# tesollo palm 축: +X=손바닥 법선, +Z=손가락 방향.
#   ex=-90  → 손가락 수평 = side
#   ex=-180 → 손가락 -Z(아래) = top-down (물체 위에서 손끝으로 집기)
# ey=90 이면 손바닥 법선이 정확히 -Z 가 되지만 euler_zyx gimbal lock 특이점이라
# Fabrics IK 가 불안정 → ey=0 을 유지하고 ex 만 돌린다.
PREGRASP_EULER_EX_TOPDOWN_DEG = -180.0

# pregrasp offset 은 물체 크기(clearance = ‖half_extent‖, 회전 무관 최대 반경)에
# 비례한다. 고정 offset 은 회전 ADR 이 오르면 물체가 palm 을 침범해 PhysX
# depenetration 폭주를 일으킨다 — lstm_test1 실증: ADR 36 부터 리턴 -1e4 스파이크가
# 24회 발생(그 전엔 0회), iter 14111 에서 -4.9e7 로 정책 붕괴.
# top-down: palm 을 물체 위 (clearance + 손가락 길이 여유) 에 둔다.
PREGRASP_TOPDOWN_XY = [0.0, +0.02]
PREGRASP_TOPDOWN_CLEARANCE = 0.10
# side(cup 전용): palm 을 물체 옆 (clearance + palm 두께 여유) 에 둔다.
PREGRASP_SIDE_Z = 0.05
PREGRASP_SIDE_CLEARANCE = 0.03

# side 접근을 유지할 물체 (그 외 전부 top-down). cup 은 내용물을 흘리면 안 되므로
# 위에서 집지 않고 옆면을 감싸 잡는다(grasp_v1 방식).
SIDE_APPROACH_OBJECT_NAMES = ("cup",)

# Fabrics world 파일 — right world의 y-미러(반발체가 오른팔 영역 y<0으로 이동).
# lstm_test2 실패 근본원인: sed 재생성이 right world 문자열을 복귀시켜
# left_arm_body sphere·left_target_cup box가 left pregrasp 목표·물체 spawn을
# 정확히 덮음 → fabric이 왼손을 자기 물체에서 밀어냄 (analysis.md 참조).
FABRIC_WORLD_FILENAME = "open_tesollo_left_boxes_no_table"


def palm_pose_mins(max_pose_angle: float) -> list:
    # y 경계: right [-0.55, 0.22] → left 미러 [-0.22, 0.55]
    # orientation: ez,ex 중심 -90° (right +90° 의 미러), ey 중심 0°
    d = math.pi / 180.0
    return [
        0.20, -0.22, 0.20,
        (-90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (-90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.65, 0.55, 0.65,
        (-90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (-90.0 + max_pose_angle) * d,
    ]


# ---------------------------------------------------------------------------
# Direct PD hand control (curl joints) — 제어 왼손 l_hj_*
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
    "l_hj_middle_1",
    "l_hj_ring_1",
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0]

# ---------------------------------------------------------------------------
# 자유화된 abduction/opposition joints (RL 직접 제어, 4D action) — right 의 부호 미러
#
# 시너지 basis 로는 살릴 수 없다: basis 열이 0 이고 anchor==open 이라
# q* = anchor + coeffs@basis 가 open 값에 고정 → progress 항상 0.
# 그래서 시너지 경로 바깥에서 절대 목표로 덮어쓴다.
#
# 범위는 URDF joint limit 의 "0 을 경계로 한 한쪽 절반" — 손이 안쪽으로만 모인다.
# left URDF limit 자체가 right 의 미러라 부호가 그대로 뒤집힌다.
# enabled_self_collisions=False 이므로 이 범위 제한이 유일한 방어선이다.
#
# 인덱스는 LEFT_HAND_JOINT_NAMES(finger-major) 기준: thumb_1=0, index_1=4,
# pinky_1=16, pinky_2=17.
HAND_ABDUCTION_JOINT_NAMES = [
    "l_hj_thumb_1",  # thumb abduction: 전 범위
    "l_hj_index_1",  # index abduction: 양수만 (엄지 쪽으로 모음 — right 미러)
    "l_hj_pinky_1",  # pinky Z-flex:   음수만 (손바닥 안쪽으로 오므림)
    "l_hj_pinky_2",  # pinky abduction: 음수만
]
HAND_ABDUCTION_LOCAL_INDICES = [0, 4, 16, 17]
HAND_ABDUCTION_LIMITS_MIN = [-0.8901179, 0.0, -1.0471976, -0.6108652]
HAND_ABDUCTION_LIMITS_MAX = [0.3839724, 0.4188790, 0.0, 0.0]

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
# curl 부호맵: thumb_2=-1(반전), index_2/middle_2/ring_2=+1(유지), pinky_3=+1(유지).
# thumb_2: right [-π, 0] → left [0, π] (부호반전 → min/max 스왑·부호반전)
# 나머지: 부호 유지 → 범위 동일.
CURL_JOINT_LIMITS_MIN = [0.0, 0.0, 0.0, 0.0, 0.0]
CURL_JOINT_LIMITS_MAX = [_math.pi, 2.007, 1.955, 1.902, _math.pi / 2]

# ---------------------------------------------------------------------------
# Distillation 카메라 — RealSense D435i (mono RGB-D). right preset 의 좌우 미러.
#
# intrinsics 는 센서 사양이라 좌우 동일(D435i depth 1280x720, HFOV 87°/VFOV 58°).
#   16:9 를 유지해야 FOV 가 맞는다. clipping 근거리 0.3m = D435i 최소 측정거리.
#
# extrinsics: right 배치(GUI 로 확정, 07.13)의 좌우 미러.
#   위치는 y 부호 반전, 자세는 "미러된 목표점을 다시 look-at" 으로 재계산했다.
#   쿼터니언을 그대로 부호 뒤집으면 핸디드니스가 깨진다 — 그렇게 하지 말 것.
#   실물 D435i 를 단 뒤 hand-eye 캘리브레이션 값으로 최종 교체할 것.
CAMERA_IMG_WIDTH  = 320
CAMERA_IMG_HEIGHT = 180        # 16:9 — D435i depth 종횡비
CAMERA_FOCAL_LENGTH       = 20.0
CAMERA_HORIZONTAL_APERTURE = 37.9586   # 2*focal*tan(87°/2) → HFOV 87°
CAMERA_CLIPPING_RANGE = (0.3, 3.0)

CAMERA_POS = [0.0804, 0.0050, 0.9674]
CAMERA_ROT = [0.1530261, -0.6915698, 0.6892421, -0.1525110]  # (w,x,y,z), ros convention

CAMERA_D_MIN = 0.3
CAMERA_D_MAX = 1.5
