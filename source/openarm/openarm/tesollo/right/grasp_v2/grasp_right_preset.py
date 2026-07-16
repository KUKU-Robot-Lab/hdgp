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
# 새 통일 네이밍(openarm_tesollo_sensor_rl.usd): arm r_aj_/l_aj_, 손 r_hj_<finger>_
# 손가락 순서(finger-major) 보존: thumb,index,middle,ring,pinky
_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
# 양팔 tesollo USD(openarm_tesollo_bi_rl): 왼손 = tesollo 20관절 (gripper 대체)
LEFT_HAND_JOINT_NAMES = [
    f"l_hj_{f}_{i}"
    for f in ("thumb", "index", "middle", "ring", "pinky")
    for i in range(1, 5)
]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    # grasp_v2는 오른팔만 제어 — 왼팔은 순수 고정(장식)이므로 pour warmstart 자세를
    # 유지할 이유 없음. 렌더에서 왼팔이 이상한 자세로 흔들리던 문제 해소:
    # 전체 0 + l_aj_4=1.57(팔꿈치 굽힘)로 깔끔한 중립 고정.
    "l_aj_1": 0.0,
    "l_aj_2": 0.0,
    "l_aj_3": 0.0,
    "l_aj_4": 1.57,
    "l_aj_5": 0.0,
    "l_aj_6": 0.0,
    "l_aj_7": 0.0,
    # 왼손 tesollo: 미사용 → 전체 0 rest (self-collision disabled)
    **{_n: 0.0 for _n in LEFT_HAND_JOINT_NAMES},
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
# sim USD(openarm_tesollo_sensor_rl.usd) body 이름
HAND_BODY_NAMES_USD = [
    "r_hl_palm",
    "r_hl_thumb_4",
    "r_hl_index_4",
    "r_hl_middle_4",
    "r_hl_ring_4",
    "r_hl_pinky_4",
]

# Fabrics FK taskmap body names (openarm_tesollo_sensor_rl fabrics URDF 기준)
# [0]=r_hl_palm  (Fabrics attractor 기준점, old palm_link ↔ 동일 transform 검증됨)
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
# r_hj_thumb_2 (thumb, Z-axis curl, range [-π, 0]) = -1.57 rad
#   → thumb을 opposition 방향으로 pre-curl하여 접근 시 컵과의 collision 방지
#   → episode 중 action[0]=1 → lerp → HAND_GRASP_POSE (thumb_2 = -1.5, ≈ 유지)
#   → 나머지 손가락(1~4)은 0에서 시작하여 lerp로 curl
HAND_APPROACH_POSE = [
    0.0, -1.57, -0.5, 0.0,   # thumb: _2=-1.57(opposition 유지), _3=-0.5(PIP curl → _3 부분이 컵에 먼저 닿는 문제 방지)
    0.0,  0.0,   0.0, 0.0,   # index: fully open
    0.0,  0.0,   0.0, 0.0,   # middle: fully open
    0.0,  0.0,   0.0, 0.0,   # ring: fully open
    0.0,  0.0,   0.0, 0.0,   # pinky: fully open
]

# 파지 자세 (per-finger lerp action=+1 목표)
# index/middle/ring _2: 0.7→1.6 rad (관절 한계 ~2.0 rad의 80%)
# thumb _3/_4: 0.5→0.8 (더 강한 curl)
HAND_GRASP_POSE = [
    0.0, -1.57, 1.5, 1.5,   # thumb
    0.0,  1.6,  1.5, 1.5,   # index
    0.0,  1.6,  1.5, 1.5,   # middle
    0.0,  1.6,  1.5, 1.5,   # ring
    0.0,  0.0,  1.5, 1.5,   # pinky
]

# Lift-phase absolute closure anchor. Fixed abduction/opposition joints are
# unchanged; flexion joints close beyond HAND_GRASP_POSE and are clamped to
# the articulation's runtime soft limits.
HAND_FULL_GRIP_POSE = [
    0.0, -1.57, 1.8, 1.8,   # thumb
    0.0,  1.9,  1.8, 1.8,   # index
    0.0,  1.9,  1.8, 1.8,   # middle
    0.0,  1.9,  1.8, 1.8,   # ring
    0.0,  0.0,  1.8, 1.8,   # pinky
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
OBJECT_GOAL_POS = [0.27, 0.10, 0.65]  # target cup xy와 일치 (demo target=[0.27,0.10])

# Pregrasp offset: cup 옆(-Y 방향)에서 접근 (palm_link 기준)
# orientation: ez=90°, ey=0°, ex=90° → palm +X(손바닥 법선)=world +Y, palm +Z(손가락)=world +X
# lift_v1: palm_ee 기준 -6cm → palm_link 기준 ≈ -9cm + rollout 여유 3cm = -12cm
PREGRASP_OFFSET = [0.0, -0.12, 0.05]

# ---------------------------------------------------------------------------
# grasp 프레임 (G) — palm orientation 을 DEXTRAH 규약으로 사상
# ---------------------------------------------------------------------------
# 문제: tesollo palm 로컬축은 +X=손바닥 법선, +Z=손가락. Allegro(DEXTRAH)는 정반대로
# +X=손가락, ±Z=법선이다. euler_zyx(R = Rz·Ry·Rx) 에서
#     col0 = (cos ez·cos ey, sin ez·cos ey, -sin ey)
# 이므로 ey=0 이면 col0 는 반드시 수평이다. Allegro 는 col0 가 손가락이라 무관하지만,
# tesollo 는 col0 가 법선이라 ey≈0 으로는 손바닥을 아래로 돌리는 것이 원리적으로 불가능하다.
# lstm_test3 의 (ez=90, ey=0, ex=180) 은 손바닥이 옆(+Y)을 본 채 손가락만 아래로 꽂힌
# "가짜 top-down" 이었다 — 실측: 법선 (-0.14, +0.98, +0.10), 손가락 (+0.21, +0.13, -0.97).
# 정책은 물체에 닿을 수 없어 정렬 보상만 먹고 제자리에서 주먹을 쥐었다(in_success 0.000).
#
# 해법: 상수 회전 C 로 palm 프레임 P 를 Allegro 규약 G 로 사상한다.
#     G.x = P.z(손가락),  G.z = P.x(법선),  G.y = -P.y
#     R_world_P = R_cmd(G-euler) · Cᵀ
# G 규약에서 (ez, ey=0, ex=180) 은 ez 와 무관하게 법선을 항상 -Z(테이블)로 보낸다.
# ez 는 손가락 방위각만 결정하고, ey 는 법선 기울임이 된다 → 기존 ±45° 경계가 그대로 유효.
# 특이점(ey=±90)이 필요 없지만, 최종 행렬은 fabric 에 quaternion 으로 넘긴다
# (openarm_tesollo_pose_fabric.set_features 가 "quaternion" convention 지원).
PALM_GRASP_FRAME_ROT = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
]

# G 규약 pregrasp euler (deg) — env.py 하드코드 금지.
#   top-down: 법선 -Z(테이블), 손가락 수평. ez* = 손가락 방위각.
#   side(cup): 기존 side 자세와 정확히 동일한 회전을 재현한다(회귀 없음).
#              검증: 현행 right side(ez=90,ey=0,ex=90) 의 G 등가 = (0, 0, -90).
PREGRASP_G_EULER_TOPDOWN = [0.0, 0.0, 180.0]   # ez* 는 heading — probe 로 확정
PREGRASP_G_EULER_SIDE = [0.0, 0.0, -90.0]

# palm 회전 경계 중심 (G 규약). 폭은 cfg.max_pose_angle(±45°).
PALM_G_EULER_CENTER_TOPDOWN = [0.0, 0.0, 180.0]
PALM_G_EULER_CENTER_SIDE = [0.0, 0.0, -90.0]

# pregrasp offset 은 물체 크기(clearance = ‖half_extent‖, 회전 무관 최대 반경)에
# 비례한다. 고정 offset 은 회전 ADR 이 오르면 물체가 palm 을 침범해 PhysX
# depenetration 폭주를 일으킨다 — lstm_test1 실증: ADR 36 부터 리턴 -1e4 스파이크가
# 24회 발생(그 전엔 0회), iter 14111 에서 -4.9e7 로 정책 붕괴.
# top-down: palm 을 물체 위 (clearance + palm 여유) 에 둔다.
# 0.10 → 0.04: 0.10 은 "손가락이 아래로 10cm 뻗는다"는 가짜 top-down 전제로 정한 값이다.
# 진짜 top-down(G 규약)에서는 손가락이 수평이라 손끝이 옆으로 뻗고, palm 을 높이 띄우면
# hand_to_object 가 보는 MAX(palm,tips) 거리가 오히려 커진다(실측 30.1cm → h2o 0.049).
# 0.04 = palm 두께 이상의 여유. 실측 침투 여유 8.1cm 이므로 겹침 없이 줄일 수 있다.
PREGRASP_TOPDOWN_XY = [0.0, -0.02]
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

# side 접근을 유지할 물체 (그 외 전부 top-down).
# cup 은 내용물을 흘리면 안 되므로 위에서 집으면 안 된다 — 옆면을 감싸 잡는다
# (grasp_v1 방식). pour 로 이어지는 유일한 물체.
SIDE_APPROACH_OBJECT_NAMES = ("cup", "cup_big")

# Fabrics world 파일 (WorldMeshesModel) — env.py가 참조. 문자열 하드코드 금지.
# right world는 왼팔 영역(y>0)에 반발체(left_arm_body sphere·left_target_cup box)를
# 두므로 left에서 그대로 쓰면 왼손이 자기 workspace에서 밀려남(left lstm_test2
# in_success 0 근본원인). left 미러는 open_tesollo_left_boxes_no_table.
FABRIC_WORLD_FILENAME = "open_tesollo_boxes_no_table"


# ⚠️ z 상한 0.65 → 0.75 (07.13): top-down 에서는 palm 이 물체보다
# (clearance + 0.04) 만큼 위에 있으므로, 물체를 goal(z=0.65)로 올리려면 palm 이
# 0.72~0.81 까지 가야 한다. 상한 0.65 는 물체 절반을 goal tol(0.10) 밖에 가둬
# 성공 자체를 물리적으로 불가능하게 만들었다(in_success 0.000 의 근본 원인).
#   clearance 8.7cm(중앙) → 물체 최대 z 0.523 → goal 과 12.7cm > tol 10cm
# IK 실측: 팔은 z≈0.74 까지 도달한다(0.80 목표 시 실제 0.737). 즉 박스가 팔의
# 실제 도달 범위보다 9cm 낮게 잘려 있었다. side 방식은 palm 이 물체와 같은
# 높이라 이 제약이 없었다(lstm_test1 이 in_success 0.637 을 낸 이유).
# palm workspace 위치 경계 (G 규약과 무관 — 회전 표현을 바꿔도 위치는 그대로)
# y_max: -0.02 → 0.22 (source cup y=-0.10에서 target cup y=0.10으로 이송 허용)
PALM_POS_MINS = [0.20, -0.55, 0.20]
PALM_POS_MAXS = [0.65, 0.22, 0.75]


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
# Direct PD hand control (v4: iCub-style, curl_gate 제거)
# ---------------------------------------------------------------------------

# RL이 직접 제어하는 curl joints (5D action, 손가락당 1D)
HAND_CURL_JOINT_NAMES = [
    "r_hj_thumb_2",  # thumb curl (Z, range [-π, 0])
    "r_hj_index_2",  # index curl (Y, range [0, 2.007])
    "r_hj_middle_2",  # middle curl (Y, range [0, 1.955])
    "r_hj_ring_2",  # ring curl (Y, range [0, 1.902])
    "r_hj_pinky_3",  # pinky curl (Y, _1 고정이므로 _3 사용)
]

# 고정 joints (RL 제어 제외)
HAND_FIXED_JOINT_NAMES = [
    "r_hj_middle_1",  # middle abduction: 0.0 고정
    "r_hj_ring_1",  # ring abduction: 0.0 고정
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0]

# ---------------------------------------------------------------------------
# 자유화된 abduction/opposition joints (RL 직접 제어, 4D action)
#
# 시너지 basis 로는 살릴 수 없다: basis 열이 0 이고 anchor==open 이라
# q* = anchor + coeffs@basis 가 open 값에 고정 → progress 항상 0.
# 그래서 시너지 경로 바깥에서 절대 목표로 덮어쓴다.
#
# 범위는 URDF joint limit 의 "0 을 경계로 한 한쪽 절반" — 손이 안쪽으로만 모이게
# 하여 반대 방향 벌어짐(물체를 놓침)과 인접 손가락 관통을 동시에 막는다.
# enabled_self_collisions=False 이므로 물리 자기충돌 검사가 없다 → 이 범위 제한이
# 유일한 방어선이다. 좌우는 URDF limit 자체가 미러라 부호만 뒤집힌다.
#
# 인덱스는 RIGHT_HAND_JOINT_NAMES(finger-major) 기준: thumb_1=0, index_1=4,
# pinky_1=16, pinky_2=17.
HAND_ABDUCTION_JOINT_NAMES = [
    "r_hj_thumb_1",  # thumb abduction: 전 범위 (대향 자세를 물체마다 달리 잡아야 함)
    "r_hj_thumb_2",  # thumb opposition: 전 범위 — 엄지를 4지 반대편으로 보내는 핵심 축
    "r_hj_index_1",  # index abduction: 음수만 (엄지 쪽으로 모음)
    "r_hj_pinky_1",  # pinky Z-flex:   양수만 (손바닥 안쪽으로 오므림)
    "r_hj_pinky_2",  # pinky abduction: 양수만
]
HAND_ABDUCTION_LOCAL_INDICES = [0, 1, 4, 16, 17]
HAND_ABDUCTION_LIMITS_MIN = [-0.3839724, -3.1415927, -0.4188790, 0.0, 0.0]
HAND_ABDUCTION_LIMITS_MAX = [0.8901179, 0.0, 0.0, 1.0471976, 0.6108652]

# iCub distal tendon 커플링 (PIP = _3, DIP = _4)
HAND_PIP_JOINT_NAMES = [
    "r_hj_thumb_3",  # thumb PIP
    "r_hj_index_3",  # index PIP
    "r_hj_middle_3",  # middle PIP
    "r_hj_ring_3",  # ring PIP
    "r_hj_pinky_4",  # pinky DIP (pinky _3이 curl이므로 _4가 커플링)
]
HAND_DIP_JOINT_NAMES = [
    "r_hj_thumb_4",  # thumb DIP
    "r_hj_index_4",  # index DIP
    "r_hj_middle_4",  # middle DIP
    "r_hj_ring_4",  # ring DIP
]

# 커플링 비율 (HAND_GRASP_POSE 기준)
DISTAL_RATIO_PIP = [0.33, 0.71, 0.71, 0.71, 0.71]
DISTAL_RATIO_DIP = [0.33, 0.71, 0.71, 0.71]

# curl joint 절대 범위 [min, max] (rad)
CURL_JOINT_LIMITS_MIN = [-_math.pi, 0.0,  0.0,  0.0,  0.0]
CURL_JOINT_LIMITS_MAX = [0.0, 2.007, 1.955, 1.902, _math.pi / 2]

# ---------------------------------------------------------------------------
# Distillation 카메라 — RealSense D435i (mono RGB-D)
#
# intrinsics: D435i depth 스트림 실측 사양(1280x720, HFOV 87°/VFOV 58°)에서 유도.
#   16:9 를 유지해야 FOV가 맞는다 — DEXTRAH 원본의 4:3(320x240)을 그대로 쓰면
#   같은 HFOV에서 VFOV가 71°로 부풀어 실제 센서와 어긋난다.
#   clipping 근거리 0.3m = D435i 최소 측정거리. 원본의 0.01m는 실기에 존재하지 않는
#   관측이라 그대로 학습시키면 sim2real 갭이 된다.
#
# extrinsics: DEXTRAH-like 정면 3인칭. 07.16 — DEXTRAH 원본 배치 충실 재현.
#   물체(0.27,-0.10,0.297) 앞(+x)에서 로봇 향해 높이 0.72m·27.1° 하향, 시선이 (0.27,-0.10,0.31)
#   을 지난다(dist 0.899m). roll 없는 순수 look-at. left 는 y 미러(회전 동일).
#   근거: DEXTRAH 원본(repo/DEXTRAH)은 카메라를 워크스페이스에서 ~0.9m·정면에 두고 clip far 2.0m,
#   좁은 FOV 48° 로 학습. 우리는 실물 D435i(광각 87°)라 같은 거리에서 배경을 더 많이 봐 depth 가
#   더 죽지만(유효 ~66%), CAMERA_D_MAX=2.0 으로 올려 워크스페이스는 밴드 안에 들어온다.
#   preview_camera.py 검증(07.16): top-down pregrasp 손이 옆모습으로 보여 물체 가림 적음.
#   ★사용자가 GUI(place_camera.py)로 잡은 먼 정면(1.5m·유효 33%)을 이 거리(0.9m)로 교정한 값.
#   실물 D435i 는 hand-eye 캘리브레이션 값으로 최종 교체. 실측 이미지 docs/camera_preview/.
#   (이전 over-shoulder 07.13: POS [0.0804,-0.0050,0.9674]
#    ROT [0.1524955,-0.6891995,0.6916149,-0.1530299] — 롤백용 보존)
CAMERA_IMG_WIDTH  = 320
CAMERA_IMG_HEIGHT = 180        # 16:9 — D435i depth 종횡비
CAMERA_FOCAL_LENGTH       = 20.0
CAMERA_HORIZONTAL_APERTURE = 37.9586   # 2*focal*tan(87°/2) → HFOV 87°
CAMERA_CLIPPING_RANGE = (0.3, 3.0)     # D435i: 최소 0.3m, 실사용 상한 3m

CAMERA_POS = [1.0700, -0.1000, 0.7200]
CAMERA_ROT = [0.3687510, -0.6033429, -0.6033429, 0.3687510]  # (w,x,y,z), ros convention

# depth 유효 밴드 — 이 밖의 픽셀은 0으로 죽인다.
# 카메라~물체 0.87m, 카메라~테이블 뒤편 ~1.3m 를 포괄.
CAMERA_D_MIN = 0.3
CAMERA_D_MAX = 2.0
