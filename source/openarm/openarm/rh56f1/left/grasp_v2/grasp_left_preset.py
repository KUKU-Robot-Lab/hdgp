# Copyleft 2025 Enactic, Inc.
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
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]

# RH56F1 actuated drive 6관절 (USD revolute joint 이름, _joint 접미사 포함)
LEFT_HAND_JOINT_NAMES = [
    "l_hj_thumb_1",   # thumb abduction (Z), 0~2.094
    "l_hj_thumb_2",   # thumb flexion drive (Z), 0~0.475
    "l_hj_index_1",   # index flexion (Z), 0~1.529
    "l_hj_middle_1",  # middle flexion
    "l_hj_ring_1",    # ring flexion
    "l_hj_pinky_1",  # little flexion
]
LEFT_ACTUATED_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

# mimic 추종 관절 (RL 비제어, 참고용)
LEFT_HAND_MIMIC_JOINT_NAMES = [
    "l_hj_thumb_3",   # = thumb_2 × 1.1425
    "l_hj_thumb_4",   # = thumb_3 × 0.7508
    "l_hj_index_2",   # = index_1 × 1.1169
    "l_hj_middle_2",
    "l_hj_ring_2",
    "l_hj_pinky_2",
]

RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]

# 좌측도 RH56F1 손 (12 DOF: drive 6 + mimic 6). 학습 비사용 → 전체 lock.
# (기존 Tesollo 의 openarm_right_finger_joint1/2 2-DOF 그리퍼는 존재하지 않음)
RIGHT_HAND_DRIVE_JOINT_NAMES = [
    "r_hj_thumb_1",
    "r_hj_thumb_2",
    "r_hj_index_1",
    "r_hj_middle_1",
    "r_hj_ring_1",
    "r_hj_pinky_1",
]
RIGHT_HAND_MIMIC_JOINT_NAMES = [
    "r_hj_thumb_3",
    "r_hj_thumb_4",
    "r_hj_index_2",
    "r_hj_middle_2",
    "r_hj_ring_2",
    "r_hj_pinky_2",
]
RIGHT_HAND_JOINT_NAMES = RIGHT_HAND_DRIVE_JOINT_NAMES + RIGHT_HAND_MIMIC_JOINT_NAMES
RIGHT_ARM_AND_HAND_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
# env_cfg 호환 별칭 (tesollo grasp_v2 매칭: right_arm_joint_names 로 좌측 전체 hold)
RIGHT_ARM_AND_GRIPPER_JOINT_NAMES = RIGHT_ARM_AND_HAND_JOINT_NAMES

# 좌팔 rest 자세 (학습 비사용, 우측 작업공간 침범 방지). 좌손은 0(열림)으로 lock.
RIGHT_ARM_REST_JOINT_POS = {
    "r_aj_1": -0.315,
    "r_aj_2": -0.440,   # 왼손 palm_sensor y=0.082→0.15 (오른손 워크스페이스서 옆으로 치움, probe 실측)
    "r_aj_3":  0.400,
    "r_aj_4":  0.513,
    "r_aj_5":  0.666,
    "r_aj_6": -0.729,
    "r_aj_7": -0.957,
}
RIGHT_HAND_REST_JOINT_POS = {name: 0.0 for name in RIGHT_HAND_JOINT_NAMES}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
# 보상/관측에 쓰는 USD body (palm + 5 말단 손가락 링크).
# Phase 0 검증: fingertip force_sensor 링크는 병합 소멸 → 생존 말단 링크 사용.
#   [0]=palm force sensor body = l_hl_palm_sensor (OLD의 rh56f1_left_plam_force_sensor 대응).
#       (구 r_al_7는 팔 손목이라 palm-cup 접촉 신호가 죽어 파지 저하 → 07.01 복구)
#   [1:6]=thumb_4, index_2, middle_2, ring_2, little_2
HAND_BODY_NAMES_USD = [
    "l_hl_palm_sensor",
    "l_hl_thumb_4",
    "l_hl_index_2",
    "l_hl_middle_2",
    "l_hl_ring_2",
    "l_hl_pinky_2",
]

# RH56F1 의 *_force_sensor 는 모두 실 하드웨어 힘센서 (palm + 5 fingertip) → actor obs.
# 실 로봇 palm 힘센서 body (OLD rh56f1_left_plam_force_sensor 대응)
PALM_FORCE_SENSOR_BODY = "l_hl_palm_sensor"

# fingertip 힘센서 body — *_force_sensor 링크가 USD 에서 말단 링크로 병합되어,
# 말단 링크(thumb_4, *_2)의 ContactSensor 가 force_sensor 패드 접촉을 포착한다.
FINGERTIP_SENSOR_BODIES = [
    "l_hl_thumb_4",
    "l_hl_index_2",
    "l_hl_middle_2",
    "l_hl_ring_2",
    "l_hl_pinky_2",
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
    1.57,   # thumb_1 abduction → opposition(손가락-엄지 사이 gap 열림). 정책이 컵을 이 gap 으로
            #        넣도록 학습. (엄지를 벌렸다 닿게 하는 재배치는 물리적으로 무리 → opposition 유지.)
    0.00,   # thumb_2: 엄지 원위(thumb_3/4) 완전 폄. test5 렌더서 q1(thumb_2)=0.10 flat →
            #        엄지 능동 굽힘 없이 abduction tip만 수동 접촉. 0에서 시작해 정책이 능동 wrap 학습.
    0.00,   # index_1
    0.00,   # middle_1
    0.00,   # ring_1
    0.00,   # little_1
]

# 파지 자세 (grasp): thumb opposition + non-thumb 우선 굽힘.
# test5에서는 index/middle 접촉이 거의 0에 머물렀으므로,
# lift 전 grasp target은 index/middle closure를 ring/little보다 앞당긴다.
HAND_GRASP_POSE = [
    1.20,   # thumb_1: approach(1.57)보다 낮춰 unpin → action f1 작동. 손바닥쪽으로 내려 컵 지지(수동 확인)
    0.24,   # thumb_2
    1.08,   # index_1
    1.08,   # middle_1
    0.85,   # ring_1
    0.85,   # little_1
]

# 완전 파지 자세 (adaptive closure 상한)
HAND_FULL_GRIP_POSE = [
    1.00,   # thumb_1: full 에서 더 내려 정책이 [1.0~1.57] 범위로 thumb_1 안정값 탐색
    0.4745, # thumb_2 (한계)
    1.50,   # index_1
    1.50,   # middle_1
    1.50,   # ring_1
    1.50,   # little_1
]


# ---------------------------------------------------------------------------
# Arm start pose (Tesollo 와 동일 — 팔 동일)
# ---------------------------------------------------------------------------
LEFT_ARM_START_POSE = [-0.5, -0.1, -0.4, 0.60, 0.2, 0.0, 0.0]  # 우 START 미러(arm sign -1,-1,-1,1,-1,-1,-1)


# ---------------------------------------------------------------------------
# Workspace / goal (Tesollo 와 동일)
# ---------------------------------------------------------------------------
OBJECT_SPAWN_CENTER = [0.27, 0.10, 0.38]  # y-미러
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]  # y-미러

# Pregrasp offset (palm_link 기준)
PREGRASP_OFFSET = [0.0, 0.12, 0.05]  # y-미러


# Pregrasp/reset palm 접근 방향 euler (deg) — env.py IK 타깃이 참조.
# 07.13 접근 자세 분기 (tesollo cd29c62 이식, 안 1→반전): top-down 이 기본, cup 만
# side(내용물 흘림 방지, grasp_v1 방식). 물체 높이 규칙(78592a3)은 ADR 회전이
# 커지면 납작한 물체가 누우면서 분기에서 스스로 빠지는 자기모순이 있어(tesollo
# lstm_test2 실증) 이름 기반 고정 분기로 대체.
# (ez,ey,ex)=(180,0,90) → palm_sensor +z(법선) = world +y (물체 방향, side).
# (ez,ey,ex)=(180,0,180) → palm_sensor +z(법선) = world -z (아래보기, top-down).
# RH56F1 은 local+Z 가 이미 Allegro/DEXTRAH 와 동일 규약(법선)이라 tesollo 의
# "가짜 top-down"(frame 불일치, 12673ea) 버그가 없음을 실측 확인(07.13 probe:
# 법선 |z| 0.849 — tesollo 깨짐값 0.123과 다름) → G-frame 이식 불필요.
# (tesollo palm_link 규약 ez=90과 다름 — 프레임 규약 차이.)
# 주의: env.py에 숫자 하드코드 금지 — tesollo right lstm_test1에서 하드코드가 경계 clamp로
# pregrasp 90° 뒤틀림 → 파지 불가 실증 (tesollo preset 동일 규칙).
# ★좌측 미러 확정(07.17 FK 교차검증, scratchpad/verify_left_mirror_fk2.py):
# l_hl_palm_sensor 프레임 = y-미러 + 로컬 Rz(180) 규약(D=diag(-1,-1,1) 실측, 자세무관 고정)
# → 물리 미러 자세의 euler_zyx 는 "EZ 180→0, EX 유지" (단순 부호반전 아님).
#   side:    right(180,0,90) → left(0,0,90)   — 좌 법선 -y (+y쪽 물체를 마주봄)
#   topdown: right(180,0,180) → left(0,0,180) — 좌우 모두 법선 -z(아래보기)
PREGRASP_EULER_EZ_DEG = 0.0
PREGRASP_EULER_EX_DEG = 90.0
PREGRASP_EULER_EX_TOPDOWN_DEG = 180.0

# pregrasp offset 은 물체 크기(clearance = ‖half_extent‖, 회전 무관 최대 반경)에
# 비례한다 (tesollo 9f0e4f7 이식, 안전 필수): 고정 offset 은 ADR 회전이 오르면
# 큰 물체가 palm 을 침범해 PhysX depenetration 폭주를 일으킨다(tesollo lstm_test1
# 실증: ADR 36부터 리턴 -1e4 스파이크 24회, iter 14111 붕괴 -4.9e7). 상수는
# RH56F1 자체 probe 로 재도출(153종 전수 겹침 0 검증, analysis 참조) — tesollo
# 값(0.04/0.03)을 맹목 복사하지 않음(손 리치·aj7 하강 기구가 다름).
PREGRASP_TOPDOWN_XY = [-0.07, -0.02]  # y-미러
PREGRASP_TOPDOWN_CLEARANCE = 0.05
PREGRASP_SIDE_Z = -0.15
PREGRASP_SIDE_CLEARANCE = 0.03
PREGRASP_R_AJ7_BIAS_TOPDOWN = 0.6

# ---------------------------------------------------------------------------
# palm 절대 pose 박스 위치 재정렬 (07.14, DEXTRAH 원본 재확인)
# ---------------------------------------------------------------------------
# DEXTRAH 는 robot_start_joint_pos 가 고정 상수라 reset 자세 == action=0 자세(둘 다
# 같은 홈 포지션) — 이 문제 자체가 없다. 우리는 tesollo 에서 물려받은 범용 박스
# 중심(0.425,-0.165,0.425)을 그대로 썼는데, 실제 pregrasp reset 위치와 크게
# 어긋나 있었다(probe 실측: side 20cm·topdown 8~9cm 차이). settle 종료 직후
# 미학습 정책(출력≈0)이 잘 계산된 pregrasp 를 버리고 박스 중심으로 끌려가
# 17.9cm→24.7cm 로 더 멀어지는 것을 6 step 만에 확인(매 에피소드 반복).
# 박스 중심을 실측 reset 위치로 재정렬 — 폭은 그대로 유지(안전 여유 보존).
# z 는 두 pose 모두 이동 시 바닥 안전마진(0.20)을 위반해 미이동.
PALM_POS_CENTER_SHIFT_SIDE    = [-0.1985, 0.0725, 0.0]  # y-미러  # side(cup) reset 위치 기준
PALM_POS_CENTER_SHIFT_TOPDOWN = [-0.0828, -0.0820, 0.0]  # y-미러  # topdown(152종) reset 위치 기준

# side 접근을 유지할 물체 (그 외 전부 top-down). cup 은 내용물을 흘리면 안 되므로
# 위에서 집으면 안 된다 — 옆면을 감싸 잡는다(grasp_v1 방식, tesollo와 동일).
SIDE_APPROACH_OBJECT_NAMES = ("cup", "cup_big")


def palm_pose_mins(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.20, -0.22, 0.20,  # y-미러: 우측 y[-0.55,0.22] → 좌측 y[-0.22,0.55]
        # ez: 좌측 side grasp palm_sensor +z 가 -y(컵, +y 접근) → 중심 0°(FK 검증 D=Rz(180)).
        (PREGRASP_EULER_EZ_DEG - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        # ex: E3 top-down — palm_sensor +z(법선)가 -z(테이블)를 향하는 자세 → 중심 180°.
        (PREGRASP_EULER_EX_DEG - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.65, 0.55, 0.65,  # y-미러: 우측 y[-0.55,0.22] → 좌측 y[-0.22,0.55]
        # ez: 좌측 side grasp palm_sensor +z 가 -y(컵, +y 접근) → 중심 0°(FK 검증 D=Rz(180)).
        (PREGRASP_EULER_EZ_DEG + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        # ex: E3 top-down — palm_sensor +z(법선)가 -z(테이블)를 향하는 자세 → 중심 180°.
        (PREGRASP_EULER_EX_DEG + max_pose_angle) * d,
    ]
