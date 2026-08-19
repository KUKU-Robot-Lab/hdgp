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

"""gripper/left/grasp_sensor 프리셋 — 조인트 이름·자세·파지 기하 단일 출처.

로봇 자산: assets/robot/openarm_tesollo_sensor_rl (비대칭 양팔)
  · 왼팔  = 7 DOF + **2지 프리즈매틱 그리퍼** (이 태스크가 제어)
  · 오른팔 = 7 DOF + Tesollo DG-5F 20관절 (이 태스크에서는 rest 고정)

이 파일의 수치는 전부 프로브 **실측**이다. 유도 과정은 각 상수 주석 참조:
  scripts/probes/probe_gripper_opening.py      (그리퍼 개구 / 컵 단면)
  scripts/probes/probe_left_gripper_reach.py   (도달성 / 기준 파지자세)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Joint groups  (openarm_tesollo_sensor_rl 통일 네이밍: arm l_aj_/r_aj_, 손 r_hj_<finger>_)
# ---------------------------------------------------------------------------
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]

# ★그리퍼는 2관절이지만 자유도는 1이다. `l_hj_gripper_2` 는 USD 에서 PhysX mimic
#   (gearing=-1, referenceJoint=l_hj_gripper_1)으로 gripper_1 을 따라간다.
#   따라서 **목표는 gripper_1 에만 준다**. 액추에이터 커버리지는 두 관절 모두에 준다
#   (커버리지가 없으면 무구동으로 자유이동) — 기존 sensor_rl 태스크들과 동일 규약.
LEFT_GRIPPER_JOINT_NAMES = ["l_hj_gripper_1", "l_hj_gripper_2"]
LEFT_ACTUATED_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]

# ---------------------------------------------------------------------------
# 그리퍼 스트로크  (URDF l_hj_gripper_1 limit)
# ---------------------------------------------------------------------------
GRIPPER_OPEN_POS = 0.044    # m, 완전 개방 (limit upper)
GRIPPER_CLOSED_POS = 0.0    # m, 완전 폐쇄 (limit lower)

# ---------------------------------------------------------------------------
# 왼팔 홈 자세 (파지 팔)
# ---------------------------------------------------------------------------
# right/grasp_sensor 가 유휴 왼팔 rest 로 쓰는 값과 **동일**하다. 그쪽에서 이 값은
# `_build_home_pose` 가 출력한 실측 미러값이고(= 우팔 홈의 부호 미러), 이 태스크에서는
# 그 자세가 곧 파지 팔의 시작 홈이 된다. 두 태스크가 같은 홈을 공유하므로 양팔 pour
# 초기 자세와도 그대로 이어진다.
# ⚠ 이 태스크는 홈을 IK 로 풀지 않는다 — 이미 측정된 관절값이라 그대로 쓴다.
#   (우측이 홈 palm 자세를 IK 로 푸는 것은 그쪽 홈이 palm 6D 로 정의돼 있기 때문)
LEFT_ARM_HOME_JOINT_POS = {
    "l_aj_1": -0.0431,
    "l_aj_2": -0.6706,
    "l_aj_3": -0.0961,
    "l_aj_4": +0.7342,
    "l_aj_5": -0.3750,
    "l_aj_6": -0.5678,
    "l_aj_7": -0.6709,
}
# 홈에서의 TCP 자세 (FK 실측, 참고용 — reward/리셋에 직접 쓰지 않는다)
LEFT_HOME_TCP_POS = (0.3362, 0.3910, 0.4073)

# ---------------------------------------------------------------------------
# 유휴 오른팔 rest  (파지 팔 홈의 부호 미러)
# ---------------------------------------------------------------------------
# 부호 벡터는 right/grasp_sensor 의 _ARM_MIRROR_SIGN 과 동일: [-1,-1,-1,+1,-1,-1,-1].
# 결과값은 right/grasp_sensor 가 실제로 쓰는 우팔 q_home 과 일치한다 → 좌우 대칭 유지.
_ARM_MIRROR_SIGN = (-1.0, -1.0, -1.0, +1.0, -1.0, -1.0, -1.0)
RIGHT_ARM_REST_JOINT_POS = {
    name.replace("l_aj_", "r_aj_"): s * LEFT_ARM_HOME_JOINT_POS[name]
    for name, s in zip(LEFT_ARM_JOINT_NAMES, _ARM_MIRROR_SIGN)
}
# 유휴 오른손: DG-5F 개방(approach) 자세. right/grasp_sensor HAND_APPROACH_POSE 와 동일.
_RIGHT_HAND_APPROACH = {
    "thumb": (0.0, -1.57, -0.5, 0.0),
    "index": (0.0, 0.0, 0.0, 0.0),
    "middle": (0.0, 0.0, 0.0, 0.0),
    "ring": (0.0, 0.0, 0.0, 0.0),
    "pinky": (0.0, 0.0, 0.0, 0.0),
}
RIGHT_HAND_REST_JOINT_POS = {
    f"r_hj_{f}_{j + 1}": v
    for f, vals in _RIGHT_HAND_APPROACH.items()
    for j, v in enumerate(vals)
}
RIGHT_REST_JOINT_POS = {**RIGHT_ARM_REST_JOINT_POS, **RIGHT_HAND_REST_JOINT_POS}

# ---------------------------------------------------------------------------
# 링크 이름
# ---------------------------------------------------------------------------
# ⚠ `l_hl_gripper_tcp` 는 physics USD 에서 고정 프레임이 강체로 병합돼 **존재하지 않는다**.
#   ContactSensor 대상으로도, body_pos_w 조회 대상으로도 쓸 수 없다.
#   TCP 는 `l_hl_gripper_base` 에서 FK 오프셋(TCP_OFFSET_IN_BASE_Z)으로 계산한다.
GRIPPER_BASE_BODY = "l_hl_gripper_base"
GRIPPER_FINGER_BODIES = ("l_hl_gripper_left_finger", "l_hl_gripper_right_finger")
TCP_OFFSET_IN_BASE_Z = 0.08     # m, l_hj_gripper_tcp origin

# Fabrics (openarm_tesollo_sensor_left_gripper URDF) — palm_link == 그리퍼 TCP
FABRIC_ROBOT_DIR = "openarm_tesollo_sensor_left_gripper"
FABRIC_PALM_BODY = "palm_link"

# ---------------------------------------------------------------------------
# 파지 기하 (probe_gripper_opening.py / probe_left_gripper_reach.py 실측)
# ---------------------------------------------------------------------------
# ★그리퍼 최대 개구 = 84.5 mm. 조인트 origin(∓0.006)+스트로크(0.044)로 계산한 100 mm 가
#   아니다 — 충돌 근사가 convexHull 이고 통과폭은 가장 안쪽 점인 **핑거 팁**이 지배한다.
GRIPPER_MAX_OPENING = 0.0845

# ★cup_big 은 원통이 아니라 **원뿔형**이다. bbox 반경 0.045(지름 90mm)는 림의 최대치이고
#   몸통은 하단 62~71 / 중단 83~86 / 림 93.4 mm. 따라서 스케일 축소 없이 scale 1.0 을 쓰되
#   **테이블 위 35~60 mm 구간에서만** 파지 가능하다(통과지름 64.3mm, 편측 여유 10.1mm).
#   h>=70mm 는 편측 여유 <2mm 로 불가.
GRASP_HEIGHT_ABOVE_TABLE = 0.055        # m, 통과대역 35~60mm 의 상단부 (팔 도달성에 유리)
GRASP_HEIGHT_BAND = (0.035, 0.060)      # m, 그리퍼가 통과 가능한 파지 높이 범위

# ★기준 파지자세: jaw 축이 **수평**이어야 두 접촉점이 컵 단면 지름 양끝(대향)에 놓인다.
#   접근축까지 수평으로 고정하면 이 팔은 자세를 못 낸다(손목 j6 가 ±45° 뿐).
#   probe_left_gripper_reach.py 가 스폰 박스 전 격자점 공통해로 도출한 값:
#     jaw 방위 θ = -35°, 접근축을 수평에서 아래로 φ = 55° (= 대각 측면 파지)
#     → 전 격자점 최소 관절여유 0.101 rad
GRASP_JAW_AZIMUTH_DEG = -35.0
GRASP_APPROACH_TILT_DEG = 55.0

# 위 (θ, φ) 를 Fabrics 가 받는 euler_zyx(ez, ey, ex) 로 변환한 값 [deg].
# R = Rz(ez)·Ry(ey)·Rx(ex) 로 역산 검증됨 (오차 2e-16).
GRASP_PALM_EULER_ZYX_DEG = (145.0, 35.0, 180.0)


def grasp_axes() -> tuple[tuple[float, float, float], ...]:
    """기준 파지자세의 (핑거폭축, jaw축, 접근축) — world 단위벡터.

    palm 프레임 축 정의: +x 핑거 폭, +y jaw 개폐, +z 접근.
    """
    t, f = math.radians(GRASP_JAW_AZIMUTH_DEG), math.radians(GRASP_APPROACH_TILT_DEG)
    jaw = (-math.sin(t), math.cos(t), 0.0)
    approach = (math.cos(t) * math.cos(f), math.sin(t) * math.cos(f), -math.sin(f))
    width = (
        jaw[1] * approach[2] - jaw[2] * approach[1],
        jaw[2] * approach[0] - jaw[0] * approach[2],
        jaw[0] * approach[1] - jaw[1] * approach[0],
    )
    return width, jaw, approach


# 파지 시 TCP 를 컵 축보다 접근축 방향으로 이만큼 더 밀어넣는다(컵이 jaw 안쪽에 물리도록).
GRASP_DEPTH = 0.02
# 액션 기준점(pregrasp)은 파지 자세에서 접근축 **반대**로 이만큼 물러난 곳.
# action=0 이면 Fabrics 가 홈에서 여기까지 스스로 접근하고, 정책은 마지막 진입과 폐쇄를 학습한다.
PREGRASP_RETREAT = 0.06

# ---------------------------------------------------------------------------
# 씬 (테이블/컵)
# ---------------------------------------------------------------------------
TABLE_SURFACE_Z = 0.2082          # right/grasp_sensor 와 동일 테이블 표면
CUP_BOTTOM_TO_ORIGIN = 0.0773     # cup_big_rl.usd 메시 bottom → 원점 (실측)
CUP_SPAWN_Z = TABLE_SURFACE_Z + CUP_BOTTOM_TO_ORIGIN     # 컵 원점 높이 = 0.2855

# ★스폰 중심 x=0.25 는 right/grasp_sensor(0.30)의 단순 미러가 **아니다**.
#   그리퍼가 컵 하단만 잡을 수 있어 파지점이 우측보다 6~10cm 낮은데, x=0.30 에서는
#   그 낮은 점에 팔이 못 미친다(실측 잔차 11~20mm). x=0.25 로 당기면 전 구간 도달된다.
CUP_SPAWN_X_CENTER = 0.25
CUP_SPAWN_Y_CENTER = 0.20         # 양팔 pour 좌우 컵 분리 규약(∓0.20)과 정합
CUP_SPAWN_X_RANGE = 0.03          # ±m — x 는 도달성이 민감해 우측(±0.05)보다 좁다
CUP_SPAWN_Y_RANGE = 0.10          # ±m

# ---------------------------------------------------------------------------
# palm(TCP) workspace — 액션 클램프 절대 한계
# ---------------------------------------------------------------------------
_ROT_HALF_RANGE_DEG = 45.0


def palm_pose_mins() -> list[float]:
    """[x, y, z, ez, ey, ex] 하한 (위치 m / 회전 rad)."""
    ez, ey, ex = GRASP_PALM_EULER_ZYX_DEG
    h = _ROT_HALF_RANGE_DEG
    return [0.10, 0.00, TABLE_SURFACE_Z - 0.02] + [
        math.radians(v - h) for v in (ez, ey, ex)
    ]


def palm_pose_maxs() -> list[float]:
    """[x, y, z, ez, ey, ex] 상한."""
    ez, ey, ex = GRASP_PALM_EULER_ZYX_DEG
    h = _ROT_HALF_RANGE_DEG
    return [0.50, 0.50, TABLE_SURFACE_Z + 0.40] + [
        math.radians(v + h) for v in (ez, ey, ex)
    ]
