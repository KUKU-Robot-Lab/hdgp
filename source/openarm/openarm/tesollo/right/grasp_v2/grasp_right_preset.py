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

# Pregrasp/reset palm 접근 방향 euler (deg) — env.py IK 타깃이 참조.
# right=+90 (palm 경계 [0°,180°] 중앙 = side-approach). left 미러는 -90.
# 주의: env.py에 숫자 하드코드 금지 — left lstm_test1에서 +90 하드코드가 left 경계
# [-180°,0°]에 0°로 clamp되어 pregrasp 90° 뒤틀림 → 파지 불가·orientation hacking 실증.
PREGRASP_EULER_EZ_DEG = 90.0
PREGRASP_EULER_EX_DEG = 90.0

# Fabrics world 파일 (WorldMeshesModel) — env.py가 참조. 문자열 하드코드 금지.
# right world는 왼팔 영역(y>0)에 반발체(left_arm_body sphere·left_target_cup box)를
# 두므로 left에서 그대로 쓰면 왼손이 자기 workspace에서 밀려남(left lstm_test2
# in_success 0 근본원인). left 미러는 open_tesollo_left_boxes_no_table.
FABRIC_WORLD_FILENAME = "open_tesollo_boxes_no_table"


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
    "r_hj_thumb_1",  # thumb abduction: 0.0 고정
    "r_hj_index_1",  # index abduction: 0.0 고정
    "r_hj_middle_1",  # middle abduction: 0.0 고정
    "r_hj_ring_1",  # ring abduction: 0.0 고정
    "r_hj_pinky_1",  # pinky Z-flex: 0.0 고정
    "r_hj_pinky_2",  # pinky abduction: 0.0 고정
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
# extrinsics: !!! PLACEHOLDER — 실제 마운트 후 hand-eye 캘리브레이션 값으로 교체할 것 !!!
#   현재 값은 작업공간(물체 스폰 0.27, -0.10, 0.297 / goal 0.27, -0.10, 0.45)을
#   정면 대각 위에서 내려다보도록 look-at 으로 계산한 것(목표까지 0.87m, 하향 30°).
#   left 미러 시 y 부호와 쿼터니언을 함께 뒤집어야 한다.
# ---------------------------------------------------------------------------
CAMERA_IMG_WIDTH  = 320
CAMERA_IMG_HEIGHT = 180        # 16:9 — D435i depth 종횡비
CAMERA_FOCAL_LENGTH       = 20.0
CAMERA_HORIZONTAL_APERTURE = 37.9586   # 2*focal*tan(87°/2) → HFOV 87°
CAMERA_CLIPPING_RANGE = (0.3, 3.0)     # D435i: 최소 0.3m, 실사용 상한 3m

CAMERA_POS = [1.05, -0.10, 0.75]                       # PLACEHOLDER
CAMERA_ROT = [0.354477, -0.6118382, -0.6118382, 0.354477]  # (w,x,y,z), ros convention

# depth 유효 밴드 — 이 밖의 픽셀은 0으로 죽인다.
# 카메라~물체 0.87m, 카메라~테이블 뒤편 ~1.3m 를 포괄.
CAMERA_D_MIN = 0.3
CAMERA_D_MAX = 1.5
