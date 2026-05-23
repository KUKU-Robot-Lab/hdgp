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

"""Hand/robot preset metadata for 5g_pour_right_v3.

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
    "openarm_left_joint1": -0.315,
    "openarm_left_joint2": -0.290,  # j2/j3 수정: target cup y≈0.10 (demo 데이터와 일치)
    "openarm_left_joint3":  0.400,  # FK 결과: [0.268, 0.100, 0.291]
    "openarm_left_joint4":  0.513,
    "openarm_left_joint5":  0.666,
    "openarm_left_joint6": -0.729,
    "openarm_left_joint7": -0.957,
    "openarm_left_finger_joint1": 0.044,
    "openarm_left_finger_joint2": 0.044,
}

# ---------------------------------------------------------------------------
# Left target cup — FK 자동 계산
# ---------------------------------------------------------------------------
# LEFT_ARM_REST_JOINT_POS를 수정하면 아래 상수가 자동으로 재계산됨.
# URDF: openarm_modular_dual.urdf 기준 (body_link0 → left_link0~7 → hand).

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
    Tc = Tc @ _Tj([0, 0, .0625],       [0,0,0], [0,0,1],   g("openarm_left_joint1"))
    Tc = Tc @ _Tj([-.0301,0,.06],      [-math.pi/2,0,0], [-1,0,0], g("openarm_left_joint2"))
    Tc = Tc @ _Tj([.0301,0,.06625],    [0,0,0], [0,0,1],   g("openarm_left_joint3"))
    Tc = Tc @ _Tj([0,.0315,.15375],    [0,0,0], [0,1,0],   g("openarm_left_joint4"))
    Tc = Tc @ _Tj([0,-.0315,.0955],    [0,0,0], [0,0,1],   g("openarm_left_joint5"))
    Tc = Tc @ _Tj([.0375,0,.1205],     [0,0,0], [1,0,0],   g("openarm_left_joint6"))
    Tc = Tc @ _Tj([-.0375,0,0],        [0,0,0], [0,-1,0],  g("openarm_left_joint7"))
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

# 팔 시작 자세 (Q_REF 근처 안전 자세; old ARM_START_POSE에서 FK ≈ sim (delta≈0))
# Fabrics rollout이 [cup_x-0.167, cup_y-0.09, cup_z+0.04]로 수렴
# j4=0.60: FK z≈0.282, 테이블 안전, 물리 충돌 없음
# [test9] j7: 1.1→0.0 (grasp v7-2 학습 시작점 j7=0.0과 일치시켜 warmstart rollout OOD 방지)
# test6에서 1.1로 올렸으나, warmstart rollout 시 grasp v7-2가 OOD 상태에서 실행돼
# cache 자세가 v7-2 정상 분포와 달라지는 문제 발생. demo shaping이 j7 방향을 안내함.
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal
# ---------------------------------------------------------------------------
# cup spawn center (local frame) — demo 데이터와 일치: source=[0.27,-0.10]
OBJECT_SPAWN_CENTER = [0.27, -0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]

# Pregrasp offset: cup 옆(-Y 방향)에서 접근 (palm_link 기준)
# orientation: ez=90°, ey=0°, ex=90° → palm +X(손바닥 법선)=world +Y, palm +Z(손가락)=world +X
# lift_v1: palm_ee 기준 -6cm → palm_link 기준 ≈ -9cm + rollout 여유 3cm = -12cm
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
    # LEFT_ARM_REST_JOINT_POS FK 기준 target cup pos (env-local): x=0.268, y=0.100, z=0.291
    #   pregrasp palm y = spawn_center_y(-0.10) + offset_y(-0.12) = -0.22m
    #   palm_delta_xyz=0.5m → max palm y = min(-0.22+0.50, 0.22) = 0.22m
    #   target cup y = 0.100m → workspace 내에 충분히 포함됨
    #   y_max=0.22m: 탐색 여유 확보 (왼팔 손목 y≈0.10에서 12cm 여유, 충돌 안전)
    #
    # [test9] z_max: 0.65 → 0.48
    #   target cup rim ≈ z=0.44m. palm z=0.48 → cup center ≈ 0.45m → cup rim ≈ 0.55m
    #   clearance ≈ 0.11m (붓기에 충분). 이전 z_max=0.65는 rim=0.75m → clearance=0.31m (너무 높음)
    #   g_clear gradient가 cup을 과도하게 올리는 문제 방지.
    d = math.pi / 180.0
    return [
        0.65, 0.22, 0.48,   # z_max: 0.65→0.48 (과도한 높이 억제)
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
