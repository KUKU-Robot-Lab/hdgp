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

# ---------------------------------------------------------------------------
# grasp 프레임 (G) — palm orientation 을 DEXTRAH 규약으로 사상 (right 와 동일한 C)
# ---------------------------------------------------------------------------
# tesollo palm 로컬축(+X=법선, +Z=손가락)은 Allegro/DEXTRAH 규약(+X=손가락, ±Z=법선)과
# 90° 어긋나 있다. euler_zyx(Rz·Ry·Rx) 에서 col0 = (cos ez·cos ey, sin ez·cos ey, -sin ey)
# 이므로 ey≈0 이면 col0(=법선)가 반드시 수평 → 손바닥을 아래로 못 돌린다.
# lstm_test3 의 (ez=-90, ey=0, ex=-180) 은 "가짜 top-down" 이었다.
# 상수 회전 C 로 G 규약에서 명령하면 (ez, ey=0, ex=180) 이 법선을 항상 -Z 로 보낸다.
#
# C 는 좌우 동일(팔 미러와 무관 — palm 로컬축 규약 자체가 같다).
# side 자세의 G 등가만 부호가 뒤집힌다: right (0,0,-90) ↔ left (0,0,+90).
#   검증: 현행 left side(ez=-90,ey=0,ex=-90) 의 G 등가 = (0, 0, +90).
PALM_GRASP_FRAME_ROT = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
]

PREGRASP_G_EULER_TOPDOWN = [0.0, 0.0, 180.0]   # ez* = 손가락 방위각 (probe 로 확정)
PREGRASP_G_EULER_SIDE = [0.0, 0.0, 90.0]

PALM_G_EULER_CENTER_TOPDOWN = [0.0, 0.0, 180.0]
PALM_G_EULER_CENTER_SIDE = [0.0, 0.0, 90.0]

# pregrasp offset 은 물체 크기(clearance = ‖half_extent‖, 회전 무관 최대 반경)에
# 비례한다. 고정 offset 은 회전 ADR 이 오르면 물체가 palm 을 침범해 PhysX
# depenetration 폭주를 일으킨다 — lstm_test1 실증: ADR 36 부터 리턴 -1e4 스파이크가
# 24회 발생(그 전엔 0회), iter 14111 에서 -4.9e7 로 정책 붕괴.
# top-down: palm 을 물체 위 (clearance + palm 여유) 에 둔다.
# 0.10 → 0.04: 0.10 은 "손가락이 아래로 10cm 뻗는다"는 가짜 top-down 전제로 정한 값이다.
# 진짜 top-down(G 규약)에서는 손가락이 수평이라 손끝이 옆으로 뻗고, palm 을 높이 띄우면
# hand_to_object 가 보는 MAX(palm,tips) 거리가 오히려 커진다(실측 30.1cm → h2o 0.049).
# 0.04 = palm 두께 이상의 여유. 실측 침투 여유 8.1cm 이므로 겹침 없이 줄일 수 있다.
PREGRASP_TOPDOWN_XY = [0.0, +0.02]
# top-down palm 높이 = 물체중심 + (회전 후 half_z) + FINGER_REACH.
#
# clearance(‖half_extent‖ = 대각선)를 쓰면 직립 물체에서 물체 top 보다 최대 3.2cm
# 과대평가한다(small_12_cyl: clearance 4.4cm vs half_z 1.3cm). 그 결과 palm 이
# 물체 top 위 9.3cm 에 뜨는데 손가락 길이가 ~10cm 라 굽혀도 물체에 닿지 않았다
# (실측: contact/tip 0.00~0.09, object_height 음수 — 한 번도 못 잡음).
#
# 회전 후 half_z = Σ_j |R[2,j]|·half[j] 로 실제 물체 높이를 쓰면
#   - 직립일 때 palm 이 물체 top 바로 위 FINGER_REACH 에 온다 → 손가락이 닿는다
#   - ADR 로 물체가 누우면 half_z 가 커져 palm 도 자동으로 올라간다 → 겹침 방지
PREGRASP_TOPDOWN_FINGER_REACH = 0.06

# 테이블 상면 z. pregrasp 를 "안착 예상 높이"(TABLE_TOP_Z + half_z) 기준으로 잡는 데 쓴다.
#
# 물체는 DEXTRAH 원본처럼 공중에서 떨어뜨린다(원본: object_start_state[:,2]=0.5).
# 낙하하며 굴러 위치·자세가 랜덤해지는 것이 의도된 도메인 랜덤화다 — 없애면 안 된다.
# 다만 pregrasp 를 낙하 "전" spawn z 기준으로 잡으면
#     palm~물체 = (spawn_z + half_z + REACH) - (table + half_z) = 0.157
# 로 half_z 가 소거돼 물체 크기와 무관하게 항상 15.7cm 가 되고, 손가락(~10cm)이 닿지
# 못한다(실측: contact/tip 0.00~0.09, 한 번도 못 잡음). 안착 예상 높이로 잡아야 한다.
# 낙하 중 xy 가 굴러 바뀌는 것은 정책이 obs(실시간 물체 위치)로 보정한다.
TABLE_TOP_Z = 0.200
# side(cup 전용): palm 을 물체 옆 (clearance + palm 두께 여유) 에 둔다.
PREGRASP_SIDE_Z = 0.05
PREGRASP_SIDE_CLEARANCE = 0.03

# side 접근을 유지할 물체 (그 외 전부 top-down). cup 은 내용물을 흘리면 안 되므로
# 위에서 집지 않고 옆면을 감싸 잡는다(grasp_v1 방식).
SIDE_APPROACH_OBJECT_NAMES = ("cup", "cup_big")

# Fabrics world 파일 — right world의 y-미러(반발체가 오른팔 영역 y<0으로 이동).
# lstm_test2 실패 근본원인: sed 재생성이 right world 문자열을 복귀시켜
# left_arm_body sphere·left_target_cup box가 left pregrasp 목표·물체 spawn을
# 정확히 덮음 → fabric이 왼손을 자기 물체에서 밀어냄 (analysis.md 참조).
FABRIC_WORLD_FILENAME = "open_tesollo_left_boxes_no_table"


# ⚠️ z 상한 0.65 → 0.75 (07.13): top-down 에서는 palm 이 물체보다
# (clearance + 0.04) 만큼 위에 있으므로, 물체를 goal(z=0.65)로 올리려면 palm 이
# 0.72~0.81 까지 가야 한다. 상한 0.65 는 물체 절반을 goal tol(0.10) 밖에 가둬
# 성공 자체를 물리적으로 불가능하게 만들었다(in_success 0.000 의 근본 원인).
#   clearance 8.7cm(중앙) → 물체 최대 z 0.523 → goal 과 12.7cm > tol 10cm
# IK 실측: 팔은 z≈0.74 까지 도달한다(0.80 목표 시 실제 0.737). 즉 박스가 팔의
# 실제 도달 범위보다 9cm 낮게 잘려 있었다. side 방식은 palm 이 물체와 같은
# 높이라 이 제약이 없었다(lstm_test1 이 in_success 0.637 을 낸 이유).
# palm workspace 위치 경계 (y: right [-0.55, 0.22] → left 미러 [-0.22, 0.55])
PALM_POS_MINS = [0.20, -0.22, 0.20]
PALM_POS_MAXS = [0.65, 0.55, 0.75]


def palm_pose_mins(max_pose_angle: float, center_deg: list | None = None) -> list:
    """palm pose 하한 [x,y,z, ez,ey,ex]. 회전은 G 규약 중심 ± max_pose_angle."""
    d = math.pi / 180.0
    c = center_deg if center_deg is not None else PALM_G_EULER_CENTER_SIDE
    return PALM_POS_MINS + [(v - max_pose_angle) * d for v in c]


def palm_pose_maxs(max_pose_angle: float, center_deg: list | None = None) -> list:
    d = math.pi / 180.0
    c = center_deg if center_deg is not None else PALM_G_EULER_CENTER_SIDE
    return PALM_POS_MAXS + [(v + max_pose_angle) * d for v in c]


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
    "l_hj_thumb_2",  # thumb opposition: 전 범위 — 엄지를 4지 반대편으로 보내는 핵심 축
    "l_hj_index_1",  # index abduction: 양수만 (엄지 쪽으로 모음 — right 미러)
    "l_hj_pinky_1",  # pinky Z-flex:   음수만 (손바닥 안쪽으로 오므림)
    "l_hj_pinky_2",  # pinky abduction: 음수만
]
HAND_ABDUCTION_LOCAL_INDICES = [0, 1, 4, 16, 17]
HAND_ABDUCTION_LIMITS_MIN = [-0.8901179, 0.0, 0.0, -1.0471976, -0.6108652]
HAND_ABDUCTION_LIMITS_MAX = [0.3839724, 3.1415927, 0.4188790, 0.0, 0.0]

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
# extrinsics: DEXTRAH-like 정면 — 단일 중앙 카메라(right 와 완전 동일). 07.16.
#   실물 카메라 1대라 left/right 가 같은 POS·ROT 를 쓴다(y 미러 아님). 워크스페이스 중앙
#   (0.27,0,0.31) 앞 높이 0.72m·27.1° 하향, dist 0.899m. 왼팔 물체(y=+0.10)도 중앙축 7° 이내.
#   DEXTRAH 원본(~0.9m·정면, clip far 2.0m) 충실. 실물은 hand-eye 캘리브로 교체.
#   (이전 미러 07.13: POS [0.0804,0.0050,0.9674]
#    ROT [0.1530261,-0.6915698,0.6892421,-0.1525110] — 롤백용 보존)
CAMERA_IMG_WIDTH  = 320
CAMERA_IMG_HEIGHT = 180        # 16:9 — D435i depth 종횡비
CAMERA_FOCAL_LENGTH       = 20.0
CAMERA_HORIZONTAL_APERTURE = 37.9586   # 2*focal*tan(87°/2) → HFOV 87°
CAMERA_CLIPPING_RANGE = (0.3, 3.0)

# 단일 중앙 카메라: right 와 정확히 같은 값(실물 1대 반영).
CAMERA_POS = [1.0700, 0.0000, 0.7200]
CAMERA_ROT = [0.3687510, -0.6033429, -0.6033429, 0.3687510]  # (w,x,y,z), ros convention

CAMERA_D_MIN = 0.3
CAMERA_D_MAX = 2.0
