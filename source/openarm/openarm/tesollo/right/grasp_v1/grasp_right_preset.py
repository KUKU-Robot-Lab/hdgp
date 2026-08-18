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
# ★08.17 로봇 자산 openarm_tesollo_sensor_rl → openarm_tesollo_bi_s_rl 전환.
#   sensor_rl 은 좌측이 2-DOF 그리퍼(l_hj_gripper_1/2)였으나 bi_s_rl 은 좌측도 DG-5FS
#   20-DOF 손이다. 존재하지 않는 이름을 init_state 에 남기면 Isaac Lab 의
#   resolve_matching_names_values 가 예외를 던진다.
LEFT_HAND_JOINT_NAMES = [f"l_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    # ★2026-08-18 유휴 팔을 **파지 팔 홈의 부호 미러**로 둔다(중립 접힘 대체).
    #   값 = Fabrics 가 푼 반대편 q_home (grasp_v1 리셋 홈 palm 의 IK 해).
    #   좌우가 완전 대칭이 되고 양팔 pour 초기 자세와도 그대로 이어진다.
    #   URDF FK 실측(양팔 미러 상태): palm 이 x·z 동일하고 y 만 반전
    #     (r [+0.281,-0.382,+0.412] / l [+0.281,+0.382,+0.412], Δy=-0.764),
    #     유휴 팔→컵 박스 482mm(구 중립 접힘 253mm), 유휴 팔 최저 z 0.369,
    #     헤드 카메라 가림 최악 93.2mm(링크반경 45mm) — 파지 팔이 결정, 불변.
    #   ⚠ reset_home_palm_pose 를 바꾸면 이 값도 같이 바꿔야 한다.
    #     env._build_home_pose 가 시작 시 미러 일치를 검사해 어긋나면 즉시 실패한다.
    "l_aj_1": -0.3082,
    "l_aj_2": -0.5785,
    "l_aj_3": -0.0970,
    "l_aj_4": +0.5811,
    "l_aj_5": -0.2676,
    "l_aj_6": -0.5281,
    "l_aj_7": -0.5792,
    # 좌손(DG-5FS)은 이 태스크에서 쓰지 않는다 — 전 관절 0(개방)으로 고정만 한다.
    # 구 그리퍼 0.044 를 대체. 우손 엄지처럼 opposition(-1.57)을 줄 이유가 없다.
    **{n: 0.0 for n in LEFT_HAND_JOINT_NAMES},
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

# ★08.17 DG-5FS(bi_s_rl) 전환 검증 — HAND_*_POSE 는 **변경 없이 유효**하다. 3단 검증:
#   ① palm 공통 프레임에서 q=0 기준 20관절 회전축이 구 자산과 전부 동일(dot>0.99, 좌손 포함)
#      → 각 값의 방향 의미(굴곡/외전 부호)가 보존된다. ⚠️부모 프레임 축만 비교하면 상류
#      rpy 누적 때문에 직교로 보인다 — 반드시 palm 프레임에서 누적 후 비교할 것.
#   ② 새 관절 한계 대조: FULL_GRIP _3/_4=1.8 의 ±1.571 초과만 걸리는데 이는 구 자산에서도
#      동일한 의도적 과지령(런타임 클램프)이라 변경 불필요.
#   ③ GPU 물리(probe_pretrain_check): 인벨롭 프로파일 wrap 0.159 / middle 1.50
#      (구 자산 0.172 / 1.23) — 새 마디 길이(0.0388→0.0334)에서도 감쌈 성립.
# 바뀐 것은 마디 길이·베이스 위치·한계뿐이고, 그건 접촉 동결이 흡수한다.
#
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
