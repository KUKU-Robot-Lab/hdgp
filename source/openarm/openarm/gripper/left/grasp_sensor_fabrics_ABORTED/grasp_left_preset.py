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
# ★이 홈은 right/grasp_sensor 의 유휴 왼팔 rest 와 **다르다**. 처음에는 그 값(= 우팔
#   DG-5F 홈의 부호 미러)을 썼는데, 20관절 손 기준으로 잡힌 자세라 2지 그리퍼의 파지
#   자세군과 손목이 ~100° 어긋난다. Isaac 실측 결과 Fabrics 가 파지 자세를 못 내고
#   jaw 가 28.5° 기울어 수평 파지가 불성립했다(j5 가 한계에 붙음).
#   Fabrics 는 IK 솔버가 아니라 홈에서 출발하는 **기울기 흐름**이라, 홈이 파지 자세군
#   밖에 있으면 흐름이 거기까지 못 간다.
#   → 홈을 파지 자세군 안에서 다시 뽑았다(scripts/probes/probe_left_gripper_home.py):
#     홈 = **스폰 박스 중심의 pregrasp 자세 그 자체**. 액션 기준점이 어차피 per-episode
#     pregrasp 이므로 홈을 그 중심에 두면 Fabrics 는 스폰 편차(±3cm/±6cm)만 따라가면 되고
#     손목 재배치가 아예 없다. 홈→파지 최대 관절변위 **0.836 rad**(구 홈 1.73~1.80 = 분기 경계).
#     위·바깥으로 물린 후보는 전부 관절 한계에 붙었다(홈여유 0.000) — 이 팔은 이 작업영역에서
#     원래 한계 근처로 동작한다.
# ⚠ 홈이 바뀌면 아래 RIGHT_ARM_REST_JOINT_POS(부호 미러)도 함께 바뀐다 —
#   두 팔 모두 한계 안이고 여유 0.30 rad 이상임을 확인했다.
LEFT_ARM_HOME_JOINT_POS = {
    "l_aj_1": -0.1569,
    "l_aj_2": -0.5984,
    "l_aj_3": +1.4065,
    "l_aj_4": +1.2005,
    "l_aj_5": +1.0895,
    "l_aj_6": -0.6695,
    "l_aj_7": +1.3563,
}
# 홈에서의 TCP 자세 (FK 실측, 참고용 — reward/리셋에 직접 쓰지 않는다)
LEFT_HOME_TCP_POS = (0.2391, 0.2443, 0.2947)

# ---------------------------------------------------------------------------
# 유휴 오른팔 rest
# ---------------------------------------------------------------------------
# ★왼팔 홈의 **부호 미러가 아니다**. 처음에는 미러로 뒀는데, 왼팔 홈이 그리퍼 파지 전용
#   자세(손목이 파지 방향으로 돌아간 자세)로 바뀌면서 그 미러가 오른팔에는 아무 의미 없는
#   자세가 됐다 — 렌더에서 오른팔이 기괴한 자세로 서 있는 것으로 드러났다.
#   유휴 팔은 그냥 **그 팔의 자연스러운 홈**이면 된다. right/grasp_sensor 가 실제로 쓰는
#   우팔 q_home 실측값을 그대로 쓴다(그쪽 preset 의 좌팔 rest 를 부호 반전한 값).
RIGHT_ARM_REST_JOINT_POS = {
    "r_aj_1": +0.0431,
    "r_aj_2": +0.6706,
    "r_aj_3": +0.0961,
    "r_aj_4": +0.7342,
    "r_aj_5": +0.3750,
    "r_aj_6": +0.5678,
    "r_aj_7": +0.6709,
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
# ★08.21 자산 이력 주의. 08:02 재빌드(ffe4239)는 고정조인트를 병합해
#   `l_hl_gripper_base` 가 강체에서 사라졌고 태스크가 `body_names.index()` 에서 죽었다.
#   13:49 재빌드(81dfcf0 "keep fixed joints unmerged")로 **되돌아왔다** — 실측 확인.
#   같은 수정으로 `l_hl_gripper_tcp` 도 이제 강체로 존재한다(강체 39 → 57).
#   그래도 앵커는 base + 오프셋을 유지한다: 기존 보상·IK·테스트가 전부 이 규약으로
#   검증돼 있고, 바꿀 이유가 없다. tcp 바디는 필요해지면 그때 쓴다.
GRIPPER_BASE_BODY = "l_hl_gripper_base"
GRIPPER_FINGER_BODIES = ("l_hl_gripper_left_finger", "l_hl_gripper_right_finger")
TCP_OFFSET_IN_BASE_Z = 0.08     # m, l_hj_gripper_tcp origin
# 그리퍼 base 원점 기준 팁 거리 — 앵커가 바뀌어도 이 값은 그리퍼 고유값이라 불변이다.
_TIP_FROM_GRIPPER_BASE_Z = 0.0954
_TCP_FROM_GRIPPER_BASE_Z = 0.08

# Fabrics (openarm_tesollo_sensor_left_gripper URDF) — palm_link == 그리퍼 TCP
FABRIC_ROBOT_DIR = "openarm_tesollo_sensor_left_gripper"
FABRIC_PALM_BODY = "palm_link"

# ---------------------------------------------------------------------------
# 파지 기하 (probe_gripper_opening.py / probe_left_gripper_reach.py 실측)
# ---------------------------------------------------------------------------
# ★그리퍼 최대 개구 = 84.5 mm. 조인트 origin(∓0.006)+스트로크(0.044)로 계산한 100 mm 가
#   아니다 — 충돌 근사가 convexHull 이고 통과폭은 가장 안쪽 점인 **핑거 팁**이 지배한다.
GRIPPER_MAX_OPENING = 0.0845

# ★shaker 는 원통이 아니라 **계단형 원뿔**이다. bbox 반경 0.044(지름 88mm)는 상단 최대치이고
#   몸통은 하단 58 / 68 / 78 / 상단 88 mm. 따라서 스케일 축소 없이 scale 1.0 을 쓰되
#   **테이블 위 10~85 mm 구간에서만** 파지 가능하다(h=65mm 에서 통과지름 68mm, 편측 여유 8.2mm).
#   h>=90mm 는 지름 78mm 로 편측 여유 3.2mm 라 불가.
#   (참고: 이전 자산 cup_big 은 대역이 35~60mm 로 훨씬 좁았다)
GRASP_HEIGHT_ABOVE_TABLE = 0.045
GRASP_HEIGHT_BAND = (0.010, 0.085)      # m, 그리퍼가 통과 가능한 파지 높이 범위

# ★기준 파지자세: jaw 축이 **수평**이어야 두 접촉점이 컵 단면 지름 양끝(대향)에 놓인다.
#   접근축까지 수평으로 고정하면 이 팔은 자세를 못 낸다(손목 j6 가 ±45° 뿐).
#   probe_left_gripper_reach.py 가 스폰 박스 전 격자점 공통해로 도출한 값:
#     jaw 방위 θ = 25°, 접근축을 수평에서 아래로 φ = 15° (거의 수평 측면 파지)
#     → 스폰 박스 전 격자점 최소 관절여유 0.166 rad, 홈에서 도달 검증됨
#   ★파지 높이는 그리퍼 여유와 팔 도달성이 **반대 방향**이라 대역 안에서 스윕해 정했다.
#   ★★자세는 **정확 자세로 전 격자점 IK 가 풀리는지**로 골라야 한다. "±8° 근방에 해가
#     있다"로 고르면 실제로 명령하는 정확 자세는 도달 불가일 수 있다(그 실수로 한 번 실패).
GRASP_JAW_AZIMUTH_DEG = 25.0
GRASP_APPROACH_TILT_DEG = 15.0

# 위 (θ, φ) 를 Fabrics 가 받는 euler_zyx(ez, ey, ex) 로 변환한 값 [deg].
# R = Rz(ez)·Ry(ey)·Rx(ex) 로 역산 검증됨 (오차 3e-16). tests 가 (θ,φ) 와의 일치를 고정한다.
GRASP_PALM_EULER_ZYX_DEG = (-155.0, 75.0, 180.0)

# 파지 높이에서의 컵 단면 반경 [m] — approach shaping 의 표면 거리 기준.
GRASP_CUP_RADIUS = 0.034


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

# 핑거 팁이 TCP 보다 접근축 방향으로 앞서 있는 거리 [m].
# gripper_base 기준 팁 z=0.0954, TCP z=0.08 (probe_gripper_opening 실측).
FINGERTIP_AHEAD_OF_TCP = _TIP_FROM_GRIPPER_BASE_Z - _TCP_FROM_GRIPPER_BASE_Z

# 액션 기준점(pregrasp)은 파지 자세에서 접근축 **반대**로 물러난 곳.
# action=0 이면 Fabrics 가 홈에서 여기까지 스스로 접근하고, 정책은 마지막 진입과 폐쇄를 학습한다.
#
# ★상수로 찍지 말고 **기하에서 유도할 것**. 처음 0.06 으로 찍었더니 action=0 기준점에서
#   핑거 팁이 컵 표면 **안쪽 9.4 mm** 에 놓였다 — 즉 액션 0 이 그리퍼를 컵에 밀어넣는
#   자세였고, Isaac 실측에서 접근 중 컵이 58.6 mm 밀리고 원치 않는 래치가 걸렸다.
#   팁이 컵 표면에서 PREGRASP_CLEARANCE 만큼 떨어지도록 역산한다.
PREGRASP_CLEARANCE = 0.025
PREGRASP_RETREAT = (
    GRASP_DEPTH + FINGERTIP_AHEAD_OF_TCP + GRASP_CUP_RADIUS + PREGRASP_CLEARANCE
)   # ≈ 0.094 m

# ---------------------------------------------------------------------------
# 씬 (테이블/컵)
# ---------------------------------------------------------------------------
# 파지 대상 = **shaker**. `shaker_body` 가 아니라 `shaker_closed` 를 쓴다 —
# 원본 shaker_body 는 양쪽이 뚫린 관이라(축 근처 정점 0개) 내용물이 그대로 빠진다.
# scripts/tools/make_closed_shaker_asset.py 가 하단에 얇은 원기둥 콜라이더를 덧붙인 것이
# shaker_closed 이고, right/grasp_sensor 도 이쪽을 쓴다. 양팔 pour 의 receiver 로 이어지려면
# 내용물을 받을 수 있어야 하므로 closed 가 유일한 선택이다.
CUP_USD_NAME = "shaker_closed_rl.usd"

# ★테이블 상면 z = 0.215 — table.usd 의 Cube extent ±(0.3625, 0.58, 0.015) 와
#   cfg init pos z=0.2 에서 나온다(0.2 + 0.015).
#   ⚠ 이 값을 두 번 틀렸다:
#     · right/grasp_sensor 의 0.2082 는 "컵 bbox 반높이로 역산한 중간값"이지 상면이 아니다.
#     · USD BBoxCache 로 읽으면 Cube 의 extent 에 xformOp:scale 이 **또** 곱해져 0.2004 가
#       나온다(extent 는 이미 스케일이 반영된 값). extent 를 직접 읽어야 한다.
#   상면을 6.8mm 낮게 잡았더니 컵이 테이블 **속에** 스폰돼 PhysX 가 밀어내며 넘어졌다
#   (Isaac 실측: 접촉 없이 기울기 36.6°). "컵이 밀린다"로 보이지만 원인은 스폰 높이다.
TABLE_SURFACE_Z = 0.215
TABLE_POS = (0.5725, 0.003, 0.2)  # 기본 위치 유지 (판 x∈[0.210, 0.935])
TABLE_HALF_X = 0.3625             # table.usd Cube extent 실측
# 메시 bottom → 원점 (probe_gripper_opening 실측). ★bbox 반높이가 아니다 —
# shaker 원점은 기하 중심이 아니라서 반높이로 역산하면 컵이 테이블에 파묻히거나 뜬다.
CUP_BOTTOM_TO_ORIGIN = 0.092090   # shaker_closed_rl.usd
CUP_SPAWN_Z = TABLE_SURFACE_Z + CUP_BOTTOM_TO_ORIGIN     # 컵 원점 높이 = 0.30029

# ★스폰 중심 x=0.25 는 right/grasp_sensor(0.30)의 단순 미러가 **아니다**.
#   그리퍼는 컵 굵기가 개구보다 좁은 낮은 구간에서만 잡을 수 있어 파지점이 우측보다
#   낮은데, x=0.30 에서는 그 낮은 점에 팔이 못 미친다(실측 잔차 11~20mm).
# 중심은 **tesollo/left/grasp_v1 과 동일**(x 0.30, y +0.20) — 이미 검증된 좌팔 스폰 위치이고
# 우측(0.30, -0.20)의 좌우 미러라 양팔 pour 배치와도 그대로 이어진다.
# 범위만 좁혔다: 2지 그리퍼는 파지 **자세**가 훨씬 빡빡해(손목 j6 ±45°) 우측의 ±0.05/±0.10
# 을 다 덮지 못한다. 실제 도달 가능 범위는 probe_left_gripper_home.py 로 확정한다.
CUP_SPAWN_X_CENTER = 0.28
# ★사용자 요청(y=0.1~0.2)에 맞춘 값. **y=0.10 은 어떤 (θ,φ)·파지높이로도 자세가 안 나온다**
#   — 팔이 몸쪽으로 접히면서 jaw 수평을 못 만든다(실측: 파지높이 30/45/65/85mm × θ 20~60°
#   전 조합에서 X). 실현 가능한 하한이 y≈0.13 이라 중심을 0.175 로 둔다.
#   파지높이도 65 → 45mm 로 낮췄다 — 그래야 이 y 대역이 열린다(65mm 에서는 y>=0.22 만 가능).
CUP_SPAWN_Y_CENTER = 0.175
CUP_SPAWN_X_RANGE = 0.02          # ±m — x 0.26~0.30
CUP_SPAWN_Y_RANGE = 0.045         # ±m — y 0.13~0.22

# ---------------------------------------------------------------------------
# palm(TCP) workspace — 액션 클램프 절대 한계
# ---------------------------------------------------------------------------
_ROT_HALF_RANGE_DEG = 45.0
# ★euler_zyx 는 ey = ±90° 에서 짐벌 특이점이라 그 근처에서 표현이 퇴화한다.
#   기준자세의 ey 가 75° 라 액션(±20°)만으로도 90° 를 넘길 수 있으므로 ey 만 따로 막는다.
#   (Fabrics 는 euler 를 회전행렬로 바꿔 쓰지만, 우리가 euler 공간에서 클램프하기 때문에
#    퇴화 구간에 들어가면 같은 회전이 여러 euler 로 갈려 클램프가 의미를 잃는다.)
_EY_ABS_LIMIT_DEG = 85.0


def _rot_bounds(sign: float) -> list[float]:
    ez, ey, ex = GRASP_PALM_EULER_ZYX_DEG
    h = _ROT_HALF_RANGE_DEG
    ey_bound = ey + sign * h
    ey_bound = max(-_EY_ABS_LIMIT_DEG, min(_EY_ABS_LIMIT_DEG, ey_bound))
    return [math.radians(ez + sign * h), math.radians(ey_bound), math.radians(ex + sign * h)]


def palm_pose_mins() -> list[float]:
    """[x, y, z, ez, ey, ex] 하한 (위치 m / 회전 rad)."""
    return [0.10, 0.00, TABLE_SURFACE_Z - 0.02] + _rot_bounds(-1.0)


def palm_pose_maxs() -> list[float]:
    """[x, y, z, ez, ey, ex] 상한."""
    return [0.50, 0.50, TABLE_SURFACE_Z + 0.40] + _rot_bounds(+1.0)
