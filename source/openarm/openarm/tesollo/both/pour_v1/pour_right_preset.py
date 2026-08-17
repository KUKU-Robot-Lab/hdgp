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

"""Hand/robot preset metadata for 5g_pour_right_v4.

v3와 동일한 joint/body 구성. v4에서 재사용.
"""

import math
import math as _math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
# 통일 네이밍(openarm_tesollo_bi_s_rl.usd): arm r_aj_/l_aj_, 손 r_hj_<finger>_/l_hj_<finger>_
#
# ★[both/pour_v1] 왼손이 2-DOF 그리퍼(l_hj_gripper_1/2)에서 **DG-5FS 20관절**로 교체됐다.
#   구 pour_sensor 는 왼손이 그리퍼라 receiver 컵을 kinematic-follow 로 붙여 들고 있었지만,
#   pour_v1 은 양손이 실제로 컵을 파지한다 → 왼손도 오른손과 대칭인 20관절을 쓴다.
_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

_L_FINGERS = list(_R_FINGERS)
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
LEFT_HAND_JOINT_NAMES = [f"l_hj_{f}_{j}" for f in _L_FINGERS for j in range(1, 5)]
LEFT_ARM_AND_HAND_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES
# 구 이름 별칭 — 외부(sim2real·probe)에서 참조하는 곳이 있어 유지한다.
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_AND_HAND_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    "l_aj_1": -0.315,
    "l_aj_2": -0.079,  # test10: demo a11-a20 pour 구간 평균값으로 변경 (기존 -0.290)
    "l_aj_3":  0.217,  # test10: demo a11-a20 pour 구간 평균값으로 변경 (기존 +0.400)
    "l_aj_4":  0.513,
    "l_aj_5":  0.666,
    "l_aj_6": -0.729,
    "l_aj_7": -0.957,
    # 왼손 DG-5FS pre-grasp 자세 — left/grasp_v1 과 부호 규약을 그대로 맞춘다
    #   (오른손 thumb_2/-3 은 -1.57/-0.5, 왼손은 미러라 +1.57/+0.5).
    #   ★이 값은 spawn 기본치일 뿐이다. 실제 에피소드 시작 자세는 왼쪽 warm bank 가 덮어쓴다.
    "l_hj_thumb_1": 0.0, "l_hj_thumb_2": 1.57, "l_hj_thumb_3": 0.5, "l_hj_thumb_4": 0.0,
    "l_hj_index_1": 0.0, "l_hj_index_2": 0.0, "l_hj_index_3": 0.0, "l_hj_index_4": 0.0,
    "l_hj_middle_1": 0.0, "l_hj_middle_2": 0.0, "l_hj_middle_3": 0.0, "l_hj_middle_4": 0.0,
    "l_hj_ring_1": 0.0, "l_hj_ring_2": 0.0, "l_hj_ring_3": 0.0, "l_hj_ring_4": 0.0,
    "l_hj_pinky_1": 0.0, "l_hj_pinky_2": 0.0, "l_hj_pinky_3": 0.0, "l_hj_pinky_4": 0.0,
}

# ---------------------------------------------------------------------------
# Left target cup — FK 자동 계산
# ---------------------------------------------------------------------------
# LEFT_ARM_REST_JOINT_POS를 수정하면 아래 상수가 자동으로 재계산됨.
# URDF: openarm_modular_dual.urdf 기준 (body_link0 → left_link0~7 → hand).

def _left_arm_fk_hand_pose(joint_pos_dict: dict) -> tuple:
    """`l_hl_palm` body의 robot-base-local 위치(3,)와 회전행렬(3,3)을 반환.

    ★[both/pour_v1] 왼손 마운트 체인이 DG-5F(그리퍼) → DG-5FS 로 바뀌어 tail 이 달라졌다.
      구(sensor_rl): l_al_7 --fixed z=0.1001--> l_hl_gripper_base
      신(bi_s_rl)  : l_al_7 --fixed z=0.0495, yaw=+90°--> l_hl_mount
                            --fixed z=0.004--> l_hl_base --fixed z=0.015--> l_hl_palm
      → 합산 z=0.0685 + mount 의 yaw +pi/2. 이 tail 을 안 고치면 receiver 컵 스폰 위치가
        3.2cm 어긋나고 회전이 90° 틀어진다.
    """
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
    # DG-5FS 왼손 마운트 체인 (l_hj_mount → base → palm). urdf 실측값.
    Tc = Tc @ _Tf([0, 0, .0495],       [0, 0, math.pi/2])   # l_hl_mount
    Tc = Tc @ _Tf([0, 0, .004],        [0, 0, 0])           # l_hl_base
    Tc = Tc @ _Tf([0, 0, .015],        [0, 0, 0])           # l_hl_palm
    return Tc[:3, 3], Tc[:3, :3]


def compute_left_cup_pose_from_fk(joint_pos_dict: dict, local_z: float = 0.04) -> tuple:
    """
    left arm 관절 위치로부터 target cup의 robot-base-local 위치와 쿼터니언을 계산.

    local_z: l_hl_palm body frame Z 방향 offset
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

    p_hand, R_hand = _left_arm_fk_hand_pose(joint_pos_dict)
    cup_pos = (p_hand + R_hand @ np.array([0.0, 0.0, local_z])).tolist()

    # ★[both/pour_v1] upright 자세를 **명시적으로 구성**한다.
    #   구 코드는 `R_hand @ R_y90` 으로 upright 를 얻었는데, 이는 DG-5F 그리퍼 프레임에서만
    #   성립하는 보정이었다. DG-5FS 는 마운트에 yaw +90° 가 붙어 같은 보정을 쓰면 컵이
    #   거의 수평(world z 와 dot 0.045)으로 눕는다.
    #   → 컵 z = world up 으로 고정하고, 손의 접근방향(R_hand 의 z)을 수평 투영해 yaw 만 따른다.
    up = np.array([0.0, 0.0, 1.0])
    fwd = R_hand @ np.array([0.0, 0.0, 1.0])          # 손 접근방향
    fwd_h = np.array([fwd[0], fwd[1], 0.0])
    if np.linalg.norm(fwd_h) < 1e-6:                   # 손이 정확히 수직이면 yaw 임의
        fwd_h = np.array([1.0, 0.0, 0.0])
    x_axis = fwd_h / np.linalg.norm(fwd_h)
    y_axis = np.cross(up, x_axis)
    R_cup = np.column_stack([x_axis, y_axis, up])
    return cup_pos, _R_to_quat_wxyz(R_cup)


# LEFT_ARM_REST_JOINT_POS 변경 시 자동 반영 : local_z=0.05m에서 컵 위치 계산 (hand origin에서 5cm 앞)
LEFT_TARGET_CUP_POS_ENV_LOCAL, LEFT_TARGET_CUP_QUAT_WXYZ = compute_left_cup_pose_from_fk(
    LEFT_ARM_REST_JOINT_POS, local_z=0.05
)

# ---------------------------------------------------------------------------
# [both/pour_v1] 왼손 body 이름
# ---------------------------------------------------------------------------
# DG-5FS 마운트 체인은 fixed 조인트 4개(mount→adapter→base→palm→palm_alias)로 이어져
# Isaac URDF 임포트에서 일부 링크가 부모로 병합될 수 있다. 이름을 하나로 못 박으면
# `body_names.index()` 실패 시 -1(=마지막 body)로 조용히 빗나가므로 후보를 우선순위로 둔다.
# 실제 해석은 env.py 의 `_resolve_body_index()` 가 하고, 전부 없으면 예외를 던진다.
LEFT_HAND_BODY_NAME_CANDIDATES = ("l_hl_palm", "l_hl_palm_alias", "l_hl_base", "l_hl_mount")
RIGHT_HAND_BODY_NAME_CANDIDATES = ("r_hl_palm", "r_hl_palm_alias", "r_hl_base", "r_hl_mount")
LEFT_FINGERTIP_BODY_NAMES = tuple(f"l_hl_{f}_tip" for f in _L_FINGERS)

# 기존 attach 상수 (레거시 — env.py에서 더 이상 사용 안 함)
LEFT_TARGET_CUP_ATTACH_FRAME_NAME = "l_hl_palm"
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
    "r_hl_palm",
    "r_hl_thumb_4",
    "r_hl_index_4",
    "r_hl_middle_4",
    "r_hl_ring_4",
    "r_hl_pinky_4",
]

# Fabrics FK taskmap body names (openarm_tesollo_sensor.urdf 기준)
# [0]=palm_link (= r_hl_palm alias, Fabrics attractor 기준점)
# [1]=palm_x    (palm_link +X 방향 기준, 방향 참조용)
# [2:7]=rl_dg_*_tip (fingertip sensor 링크, 센서 URDF 기준)
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
OBJECT_GOAL_POS = [0.27, -0.10, 0.65]

# Pregrasp offset: cup 옆(-Y 방향)에서 접근 (palm_link 기준)
# orientation: ez=90°, ey=0°, ex=90° → palm +X(손바닥 법선)=world +Y, palm +Z(손가락)=world +X
# lift_v1: palm_ee 기준 -6cm → palm_link 기준 ≈ -9cm + rollout 여유 3cm = -12cm
PREGRASP_OFFSET = [0.0, -0.12, 0.05]


def palm_pose_mins(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        # [REDESIGN v5] 깊은 pour tilt 시 palm이 target 너머/아래로 스윙 → x_min/z_min 완화.
        # (rotation은 _apply_action 위치 클램프 대상 아님 → tilt 자유, 위치 박스만 병목)
        # [test11] x_min -0.15→-0.30: rim-pivot이 80°↑ 깊은 tilt에서 pour_point 고정 위해
        #   palm.xy를 박스 밖으로 스윙해야 하나 클램프(palm_clamp_active 0.15→0.30)가 막아 tilt 80° plateau.
        #   외회전(박스 축소 사유)은 internal_rot_gate=0.98로 해결됨 → 박스 완화 안전.
        -0.30, -0.55, 0.10,  # x_min -0.15→-0.30 [test11], z_min 0.10
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
    # target cup rim ≈ z=0.44m. palm z=0.48 → cup center ≈ 0.45m → cup rim ≈ 0.55m
    #   clearance ≈ 0.11m (붓기에 충분). 이전 z_max=0.65는 rim=0.75m → clearance=0.31m (너무 높음)
    #   g_clear gradient가 cup을 과도하게 올리는 문제 방지.
    d = math.pi / 180.0
    return [
        # [H9] 상한 축소: demo a20 sim 측정 palm 도달범위(world) x 0.21~0.33, y -0.20~-0.10, z 0.30~0.58.
        #   기존 박스가 demo 폭의 x6.7·y7.7·z2.2배라 palm이 극단(z_max 0.72→j4 한계, xy 사방)으로
        #   이동해 외회전 귀결. 상한을 demo max + 여유로 축소 (min은 시작점 보호 위해 유지).
        #   x 0.65→0.45, y 0.22→0.10, z 0.72→0.62 (demo z끝 0.58 + 4cm).
        # [test11] x_max 0.45→0.65, y_max 0.10→0.25 (H9 이전 복원): 깊은 tilt rim-pivot 스윙 여유.
        #   H9 축소는 외회전 억제 목적이었으나 internal_rot_gate=0.98로 해결 → 박스가 tilt 벽으로 작동.
        #   per-axis 클램프 로깅으로 binding bound 확정 후 안 쓰는 방향은 재축소 예정.
        # [test4] z_max 0.62→0.68. lstm_test3 진단: palm_clamp_active 0.87(=Z 단독, xy위반≈0),
        #   viol_z max 0.043m ≈ palm_ee offset Z(0.04). 원인=제어점이 r_hl_palm(손바닥 아래)이라
        #   deep tilt 시 palm_ee(컵 중심) offset이 회전하며 보상 상승분이 z_max에 걸림.
        # [palm_ee 제어] 박스 기준이 이제 r_hl_palm이 아닌 palm_ee(진짜 손바닥 중심). offset 회전
        #   binding은 IK가 흡수 → z_max 0.68은 generous 값으로 유지, 첫 palm_ee run의 viol_z로 재보정.
        #   (0.72는 과거 g_clear 과상승/외회전 유발 → 그 아래 유지, 외회전은 internal_rot_gate로 제어)
        0.65, 0.25, 0.68,
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
