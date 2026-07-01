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

"""inspire_r_pour_v1 preset (RH56F1 6-DOF underactuated 손).

Tesollo 20-DOF → RH56F1 6 actuated DOF 포팅 (grasp_r_v1 과 동일 로봇 구성).
  - 손 drive 6관절: thumb_1(측배), thumb_2(굽힘), index_1, middle_1, ring_1, little_1
  - mimic 추종(thumb_3/4, *_2)은 USD PhysxMimicJoint 가 자동 처리 → RL 제어 대상 아님.
  - pour 태스크 고유 기하(left target cup FK, bead spawn, pour point/opening)는 보존.
  - 팔/workspace 규약은 pour Tesollo 와 동일 (palm 가상프레임 기하 일치).
"""

import math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]

# RH56F1 actuated drive 6관절 (USD revolute joint 이름, _joint 접미사 포함)
RIGHT_HAND_JOINT_NAMES = [
    "r_hj_thumb_1",   # thumb abduction (Z), 0~2.094
    "r_hj_thumb_2",   # thumb flexion drive (Z), 0~0.475
    "r_hj_index_1",   # index flexion (Z), 0~1.529
    "r_hj_middle_1",  # middle flexion
    "r_hj_ring_1",    # ring flexion
    "r_hj_pinky_1",  # little flexion
]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# mimic 추종 관절 (RL 비제어, 참고용)
RIGHT_HAND_MIMIC_JOINT_NAMES = [
    "r_hj_thumb_3",   # = thumb_2 × 1.1425
    "r_hj_thumb_4",   # = thumb_3 × 0.7508
    "r_hj_index_2",   # = index_1 × 1.1169
    "r_hj_middle_2",
    "r_hj_ring_2",
    "r_hj_pinky_2",
]

LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]

# 좌측도 RH56F1 손 (12 DOF: drive 6 + mimic 6). 학습 비사용 → 전체 lock.
# (기존 Tesollo 의 openarm_left_finger_joint1/2 2-DOF 그리퍼는 존재하지 않음)
LEFT_HAND_DRIVE_JOINT_NAMES = [
    "l_hj_thumb_1",
    "l_hj_thumb_2",
    "l_hj_index_1",
    "l_hj_middle_1",
    "l_hj_ring_1",
    "l_hj_pinky_1",
]
LEFT_HAND_MIMIC_JOINT_NAMES = [
    "l_hj_thumb_3",
    "l_hj_thumb_4",
    "l_hj_index_2",
    "l_hj_middle_2",
    "l_hj_ring_2",
    "l_hj_pinky_2",
]
LEFT_HAND_JOINT_NAMES = LEFT_HAND_DRIVE_JOINT_NAMES + LEFT_HAND_MIMIC_JOINT_NAMES
LEFT_ARM_AND_HAND_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES

# 좌팔 rest 자세 (target cup 을 드는 자세; pour 태스크 demo a11-a20 pour 구간 평균값).
# 좌손은 0(열림)으로 lock. (Tesollo 의 openarm_left_finger_joint1/2 항목 제거)
LEFT_ARM_REST_JOINT_POS = {
    "l_aj_1": -0.315,
    "l_aj_2": -0.079,
    "l_aj_3":  0.217,
    "l_aj_4":  0.513,
    "l_aj_5":  0.666,
    "l_aj_6": -0.729,
    "l_aj_7": -0.957,
}
LEFT_HAND_REST_JOINT_POS = {name: 0.0 for name in LEFT_HAND_JOINT_NAMES}


# ---------------------------------------------------------------------------
# Left target cup — FK 자동 계산
# ---------------------------------------------------------------------------
# LEFT_ARM_REST_JOINT_POS를 수정하면 아래 상수가 자동으로 재계산됨.
# URDF: openarm_modular_dual.urdf 기준 (body_link0 → left_link0~7 → hand).
# 왼팔은 OpenArm 으로 동일하므로 hand 변경(RH56F1)과 무관하게 FK 그대로 사용.

def _left_arm_fk_hand_pose(joint_pos_dict: dict) -> tuple:
    """openarm_left_hand body의 robot-base-local 위치(3,)와 회전행렬(3,3)을 반환."""
    import numpy as np

    def _Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def _Ry(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def _Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    def _rpy(r,p,y): return _Rz(y) @ _Ry(p) @ _Rx(r)
    def _T(R, p): M = np.eye(4); M[:3,:3] = R; M[:3,3] = p; return M
    def _Tf(xyz, rpy): return _T(_rpy(*rpy), np.array(xyz, float))
    def _Tj(xyz, rpy, ax, q):
        a = np.array(ax, float); a /= np.linalg.norm(a)
        c, s = math.cos(q), math.sin(q)
        K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        return _T(_rpy(*rpy) @ (np.eye(3) + s*K + (1-c)*(K@K)), np.array(xyz, float))

    g = lambda k: joint_pos_dict.get(k, 0.0)
    Tc = np.eye(4)
    Tc = Tc @ _Tf([0, .031, .698],     [-math.pi/2, 0, 0])
    Tc = Tc @ _Tj([0, 0, .0625],       [0,0,0], [0,0,1],   g("l_aj_1"))
    Tc = Tc @ _Tj([-.0301,0,.06],      [-math.pi/2,0,0], [-1,0,0], g("l_aj_2"))
    Tc = Tc @ _Tj([.0301,0,.06625],    [0,0,0], [0,0,1],   g("l_aj_3"))
    Tc = Tc @ _Tj([0,.0315,.15375],    [0,0,0], [0,1,0],   g("l_aj_4"))
    Tc = Tc @ _Tj([0,-.0315,.0955],    [0,0,0], [0,0,1],   g("l_aj_5"))
    Tc = Tc @ _Tj([.0375,0,.1205],     [0,0,0], [1,0,0],   g("l_aj_6"))
    Tc = Tc @ _Tj([-.0375,0,0],        [0,0,0], [0,-1,0],  g("l_aj_7"))
    Tc = Tc @ _Tf([0, 0, .1001],       [0,0,0])
    return Tc[:3, 3], Tc[:3, :3]


def compute_left_cup_pose_from_fk(joint_pos_dict: dict, local_z: float = 0.04) -> tuple:
    """
    left arm 관절 위치로부터 target cup의 robot-base-local 위치와 쿼터니언을 계산.

    local_z: openarm_left_hand body frame Z 방향 offset
             0.0 = hand origin, 0.08 = tcp, 0.04(default) = midpoint
    Returns:
        pos_env_local: list[float, 3]   robot base(=env origin) 기준 위치
        quat_wxyz:     list[float, 4]   cup Z ≈ world Z (upright)
    """
    import numpy as np

    def _R_to_quat_wxyz(R):
        t = R[0,0]+R[1,1]+R[2,2]
        if t > 0:
            s=2*np.sqrt(t+1); w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif R[1,1] > R[2,2]:
            s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
        q = np.array([w, x, y, z]); return (q / np.linalg.norm(q)).tolist()

    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    R_y90 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    p_hand, R_hand = _left_arm_fk_hand_pose(joint_pos_dict)
    cup_pos  = (p_hand + R_hand @ np.array([0.0, 0.0, local_z])).tolist()
    cup_quat = _R_to_quat_wxyz(R_hand @ R_y90)
    return cup_pos, cup_quat


# LEFT_ARM_REST_JOINT_POS 변경 시 자동 반영 : local_z=0.05m에서 컵 위치 계산 (hand origin에서 5cm 앞)
LEFT_TARGET_CUP_POS_ENV_LOCAL, LEFT_TARGET_CUP_QUAT_WXYZ = compute_left_cup_pose_from_fk(
    LEFT_ARM_REST_JOINT_POS, local_z=0.05
)

# 기존 attach 상수 (레거시 — env.py에서 더 이상 사용 안 함)
LEFT_TARGET_CUP_ATTACH_FRAME_NAME = "openarm_left_hand"
LEFT_TARGET_CUP_ATTACH_POS_B = [0.0, 0.0, 0.10]
LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B = [0.70710678, 0.0, 0.70710678, 0.0]

BEAD_SPAWN_POS_SOURCE_CUP_B = [0.0, 0.0, 0.04]
BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ = [1.0, 0.0, 0.0, 0.0]
SOURCE_CUP_POUR_POINT_POS_B = [0.0, 0.0, 0.100]   # 실제 컵 림(입구) z=+0.100m (origin=컵 중앙)
TARGET_CUP_OPENING_POS_B = [0.0, 0.0, 0.100]   # 실제 컵 림(입구) z=+0.100m (origin=컵 중앙)
SOURCE_CUP_POUR_AXIS_B = [1.0, 0.0, 0.0]
SOURCE_CUP_UP_AXIS_B = [0.0, 0.0, 1.0]
TARGET_CUP_UP_AXIS_B = [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
# 보상/관측에 쓰는 USD body (palm + 5 말단 손가락 링크).
# Phase 0 검증: fingertip force_sensor 링크는 병합 소멸 → 생존 말단 링크 사용.
#   [0]=palm(plam_force_sensor, 실 센서 body)
#   [1:6]=thumb_4, index_2, middle_2, ring_2, little_2
HAND_BODY_NAMES_USD = [
    "r_al_7",
    "r_hl_thumb_4",
    "r_hl_index_2",
    "r_hl_middle_2",
    "r_hl_ring_2",
    "r_hl_pinky_2",
]

# RH56F1 의 *_force_sensor 는 모두 실 하드웨어 힘센서 (palm + 5 fingertip).
PALM_FORCE_SENSOR_BODY = "r_al_7"

# fingertip 힘센서 body — *_force_sensor 링크가 USD 에서 말단 링크로 병합되어,
# 말단 링크(thumb_4, *_2)의 ContactSensor 가 force_sensor 패드 접촉을 포착한다.
FINGERTIP_SENSOR_BODIES = [
    "r_hl_thumb_4",
    "r_hl_index_2",
    "r_hl_middle_2",
    "r_hl_ring_2",
    "r_hl_pinky_2",
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
# pour: per-finger lerp action=-1 → APPROACH, action=+1 → GRASP.
# freeze_grasp 시 grasp pose 로 컵을 단단히 쥔 채 붓기.
# ---------------------------------------------------------------------------
# 완전히 열린 자세
HAND_START_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# FABRICS 접근 자세 (pregrasp): thumb opposition pre-rotate + 약한 굽힘
HAND_APPROACH_POSE = [
    1.57,   # thumb_1 abduction → opposition 방향
    0.0,   # thumb_2 약한 굽힘
    0.00,   # index_1
    0.00,   # middle_1
    0.00,   # ring_1
    0.00,   # little_1
]

# 파지 자세 (grasp): thumb opposition + 손가락 굽힘
HAND_GRASP_POSE = [
    1.57,   # thumb_1
    0.30,   # thumb_2 (한계 0.475 의 84%)
    0.50,   # index_1
    0.50,   # middle_1
    0.50,   # ring_1
    0.50,   # little_1
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
# Arm start pose (pour Tesollo 와 동일 — 팔 동일)
# ---------------------------------------------------------------------------
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal (pour Tesollo 와 동일)
# ---------------------------------------------------------------------------
# cup spawn center (local frame) — demo 데이터와 일치: source=[0.27,-0.10]
OBJECT_SPAWN_CENTER = [0.27, -0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]

# Pregrasp offset: cup 옆(-Y 방향)에서 접근 (palm_link 기준)
PREGRASP_OFFSET = [0.0, -0.12, 0.05]


def palm_pose_mins(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.00, -0.55, 0.20,  # x_min: 0.20→0.00 (target cup이 x≈0.0에 있어 더 들어가야 함)
        (90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    # pour workspace: target cup rim ≈ z=0.44m. palm z=0.48 → cup rim clearance 적정.
    d = math.pi / 180.0
    return [
        0.65, 0.22, 0.48,   # z_max: 0.48 (warmstart z_boost 0.12m → 실제 palm ≤ 0.60m)
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]
