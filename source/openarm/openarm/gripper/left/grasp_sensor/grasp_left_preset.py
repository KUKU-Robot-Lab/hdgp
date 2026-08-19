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

"""gripper/left/grasp_sensor 프리셋 — 조인트 이름·자세·씬 기하 단일 출처.

로봇: assets/robot/openarm_tesollo_sensor_rl (비대칭 양팔)
  · 왼팔  = 7 DOF + 2지 프리즈매틱 그리퍼  ← 이 태스크가 제어
  · 오른팔 = 7 DOF + Tesollo DG-5F 20관절  ← rest 고정

여기 수치는 전부 **자산에서 직접 잰 값**이다. 유도 과정은 각 상수 주석 참조.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 조인트 / 링크 이름 (openarm_tesollo_sensor_rl 통일 네이밍)
# ---------------------------------------------------------------------------
LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]

# ★그리퍼는 관절 2개지만 자유도는 1이다. `l_hj_gripper_2` 는 USD 에서 PhysX mimic
#   (gearing=-1, referenceJoint=l_hj_gripper_1)으로 gripper_1 을 따라간다.
#   IsaacLab 관례는 "mimic 을 시뮬에 안 넣고 두 손가락에 같은 타깃을 뿌리는 것"이지만
#   우리 USD 에는 진짜 mimic 제약이 있으므로 **gripper_1 에만 지령**한다.
#   액추에이터 커버리지는 두 관절 모두에 준다 — 없으면 무구동으로 자유이동한다.
GRIPPER_DRIVE_JOINT = "l_hj_gripper_1"
GRIPPER_JOINT_NAMES = ["l_hj_gripper_1", "l_hj_gripper_2"]

GRIPPER_BASE_BODY = "l_hl_gripper_base"
# ⚠ `l_hl_gripper_tcp` 는 URDF 에는 있지만 physics USD 에서 고정 프레임이 강체로 병합돼
#   **사라진다**. FrameTransformer 대상으로 쓸 수 없어 base + z 오프셋으로 TCP 를 만든다.
TCP_OFFSET_IN_BASE_Z = 0.08     # m, l_hj_gripper_tcp origin

RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]

# ---------------------------------------------------------------------------
# 그리퍼 스트로크 (URDF l_hj_gripper_1 limit) — IsaacLab OpenArm 과 동일 값
# ---------------------------------------------------------------------------
GRIPPER_OPEN_POS = 0.044
GRIPPER_CLOSED_POS = 0.0

# ---------------------------------------------------------------------------
# 씬 기하
# ---------------------------------------------------------------------------
# ★테이블 상면 z = 0.215. table.usd 의 Cube extent ±(0.3625, 0.58, 0.015) 와
#   init pos z=0.2 에서 나온다(0.2 + 0.015). 판 범위 x∈[0.210, 0.935].
#   ⚠ 이 값을 두 번 틀렸다:
#     · right/grasp_sensor 의 0.2082 는 "컵 bbox 반높이로 역산한 중간값"이지 상면이 아니다.
#     · USD BBoxCache 로 읽으면 extent 에 xformOp:scale 이 **또** 곱해져 0.2004 가 나온다
#       (extent 는 이미 스케일 반영값). extent 를 직접 읽어야 한다.
#   상면을 낮게 잡으면 컵이 판 속에 스폰돼 PhysX 가 밀어내며 넘어진다(Isaac 실측).
TABLE_POS = (0.5725, 0.003, 0.2)
TABLE_HALF_X = 0.3625
TABLE_HALF_Y = 0.58
TABLE_SURFACE_Z = 0.215

# 파지 대상 = shaker. `shaker_body` 가 아니라 `shaker_closed` 를 쓴다 — 원본은 양쪽 뚫린
# 관이라 내용물이 그대로 빠진다. 양팔 pour 의 receiver 로 이어지려면 받을 수 있어야 한다.
CUP_USD_NAME = "shaker_closed_rl.usd"
CUP_MASS = 0.134                  # kg, right/grasp_sensor 전 물체 공통값과 동일
# 메시 bottom → 원점 (probe_gripper_opening.py 실측).
# ★bbox 반높이(0.0875)가 아니다 — shaker 원점은 기하 중심이 아니라서 반높이로 역산하면
#   컵이 테이블에 파묻힌다.
CUP_BOTTOM_TO_ORIGIN = 0.09209
# ★컵의 강체 이름. 레퍼런스는 큐브 prim 이름인 `"Object"` 를 SceneEntityCfg 에 박아 두는데
#   우리 shaker 는 `baseLink` 라 그대로 두면 매니저가 이름을 resolve 하는 순간 죽는다
#   (서버 학습 기동 시 실제로 터졌다: "Object: [] / Available strings: ['baseLink']").
#   ⚠ 로컬에서는 sim 이 playing 이 아닌 타이밍이라 resolve 가 스킵돼 **드러나지 않는다**.
CUP_BODY_NAME = "baseLink"
CUP_SPAWN_Z = TABLE_SURFACE_Z + CUP_BOTTOM_TO_ORIGIN     # 0.30709

# ★★스폰 x 하한은 **홈 자세의 팔이 점유한 공간 바깥**이어야 한다.
#   홈은 컵을 감싼 파지 자세라, 컵을 그 자리에 스폰하면 손가락·팔 메시가 컵을 관통해
#   PhysX 가 컵을 수백 mm 날려버린다(zero-action 실측: 최대 886 mm, tilt 85°).
#   probe_lift_left_gripper_smoke.py 의 1e 스윕(49 조합)으로 잰 경계:
#     · x ≥ 0.31 : y ∈ [0.17, 0.23] 전 구간 조용 (이동 0.00 mm)
#     · x = 0.30 : y ≤ 0.18 에서만 조용
#     · x < 0.30 : **전 구간 관통** — 팔이 그 공간에 있다
#   그래서 하한을 0.32 로 잡아 10 mm 여유를 둔다.
#   ⚠ "컵을 앞에 둔다"와 "홈을 뒤로 물린다"는 로봇 기준 상대 배치가 같아 물리적으로 동등하다.
#     결과적으로 홈이 pre-grasp 자세가 되며, 이는 lift 레시피가 원하는 초기 조건이다.
#   랜덤화 폭은 처음에 좁게 두고 학습이 붙은 뒤 넓힌다.
SPAWN_X_SAFE_MIN = 0.31           # 실측 관통 경계 (이 아래는 홈 자세와 충돌)
CUP_SPAWN_X_CENTER = 0.36
CUP_SPAWN_Y_CENTER = 0.20
CUP_SPAWN_X_RANGE = 0.04          # ±m → x ∈ [0.32, 0.40]
CUP_SPAWN_Y_RANGE = 0.03          # ±m → y ∈ [0.17, 0.23] (스윕 검증 범위 내)

# ---------------------------------------------------------------------------
# 리프트 판정
# ---------------------------------------------------------------------------
# ★★`mdp.object_is_lifted` 는 물체 **root 원점**의 절대 world z 를 본다.
#   기준선은 테이블 상면이 아니라 **컵이 놓여 있을 때의 원점 z**, 즉 CUP_SPAWN_Z 다.
#   레퍼런스가 상면 기준으로 맞아떨어지는 건 큐브의 원점이 기하 중심이라서일 뿐이고,
#   shaker 는 원점이 바닥에서 92 mm 위라 상면만 더하면 임계가 놓인 상태보다 **낮아진다**.
#
#   ⚠ 실제로 이 실수를 저질렀다(test1-r2). `TABLE_SURFACE_Z + 0.04 = 0.255` 인데 놓인
#     컵의 원점이 이미 0.30709 라, lifting 보상(weight 15)이 **상시 1**이었다:
#       Episode_Reward/lifting_object 14.63 / 상한 15.0
#     goal-tracking 게이트도 늘 열려 있으니 정책이 컵을 건드릴 이유가 없고, action_rate·
#     joint_vel 페널티까지 있어 **가만히 있는 것이 최적**이 된다. 실제로 reaching_object 가
#     0.024 → 0.007 로 계속 떨어졌다(그리퍼가 컵에서 도망갔다).
LIFT_HEIGHT_ABOVE_TABLE = 0.04
MINIMAL_LIFT_HEIGHT = CUP_SPAWN_Z + LIFT_HEIGHT_ABOVE_TABLE       # 0.34709
# ★종료 임계는 "낙하"뿐 아니라 **쓰러짐**도 잡아야 한다.
#   shaker 는 가늘고 길어 잘 쓰러지는데, 2지 그리퍼로는 쓰러진 컵을 다시 세울 수 없다.
#   그런데 원점 z 는 기울기에 둔감해서(실측: 완전히 누운 컵 0.25199, 60° 기울면 약 0.299)
#   레퍼런스식 임계(상면 −0.05 = 0.165)로는 **안 잡히고 에피소드가 끝까지 낭비된다**
#   (렌더 관찰 + 프로브 1a3 으로 확인).
#   0.27 이면: 정상 0.30709 통과 / 완전히 누움 0.252 종료 / 테이블 밖 낙하 0.092 종료.
#   60° 기울기(0.299)는 통과시킨다 — 그 정도는 아직 회복 가능하고, 조기 종료를 늘리면
#   lift 레시피의 "신호가 끊기지 않는다"는 장점을 깎는다.
CUP_TIPPED_ORIGIN_Z = 0.25199   # 테이블 위에 완전히 누운 컵의 원점 z (프로브 1a3 실측)
OBJECT_DROP_HEIGHT = 0.27

# ---------------------------------------------------------------------------
# 자세
# ---------------------------------------------------------------------------
# ★왼팔 초기 자세 = **파지 준비 자세**. 이게 lift 레시피가 학습되는 핵심 이유 중 하나다 —
#   액션이 `use_default_offset=True` 라 **액션 0 = 이 자세**이고, 정책 초기화 시점부터
#   해답 근처에서 ±0.5 rad 국소 탐색을 한다. 초기 자세가 엉뚱하면 탐색이 통째로 낭비된다.
#   값은 probe_left_gripper_home.py 가 스폰 박스 중심의 파지 자세로 뽑은 IK 해에서 왔다
#   (그 프로브의 나머지 판정 기준은 lift 방식에선 불필요하지만, 이 자세 자체는 유효하다).
LEFT_ARM_HOME_JOINT_POS = {
    "l_aj_1": -0.1569,
    "l_aj_2": -0.5984,
    "l_aj_3": +1.4065,
    "l_aj_4": +1.2005,
    "l_aj_5": +1.0895,
    "l_aj_6": -0.6695,
    "l_aj_7": +1.3563,
}

# 유휴 오른팔 = **팔꿈치를 접어 든 자세**.
#   ★왼팔 홈의 부호 미러가 아니다(왼팔 홈이 그리퍼 전용 파지 자세라 미러가 무의미하고,
#     렌더에서 기괴한 자세로 드러났다).
#   ★`right/grasp_sensor` 의 우팔 q_home 도 아니다. 그 자세는 이 씬에서 손이 테이블 상면
#     바로 위(0.291)까지 내려와 컵을 들어올릴 때 간섭할 수 있고, 자세 오차도 5.1° 로 컸다.
#   probe_idle_right_arm_rest.py 로 후보 5 종을 전수 평가해 고른 값이다:
#     자세오차 2.09° · 최저 링크 z 0.4886(상면보다 274 mm 위) · 컵까지 473 mm.
#     (차렷은 자세는 지키지만 손이 **바닥**에 닿는다: 최저 z 0.009)
RIGHT_ARM_REST_JOINT_POS = {
    "r_aj_1": +0.0,
    "r_aj_2": +0.3,
    "r_aj_3": +0.0,
    "r_aj_4": +2.0,
    "r_aj_5": +0.0,
    "r_aj_6": +0.0,
    "r_aj_7": +0.0,
}
# 유휴 오른손: DG-5F 개방(approach) 자세.
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
# 목표(goal) 커맨드 범위 — 왼팔 워크스페이스
# ---------------------------------------------------------------------------
# 레퍼런스 uni 버전은 x(0.2,0.4) y(-0.2,0.2) z(0.15,0.4) 로 줄여 뒀다(작은 로봇 기준).
# 우리는 **테이블 상면이 z=0.215** 이므로 z 를 그만큼 올려야 목표가 판 위에 뜬다.
# x 하한은 스폰과 같은 이유로 0.32 이상. 그보다 앞은 홈 자세의 팔이 점유한 공간이라
# 컵을 들고 그리로 가려면 팔을 접어야 하는데, 관절 델타 ±0.5 rad 로는 무리다.
GOAL_POS_X = (0.32, 0.44)
GOAL_POS_Y = (0.15, 0.28)
# ★목표도 컵 **원점** 좌표다. 하한이 리프트 임계보다 낮으면 "먼저 들어라 → 옮겨라" 순서가
#   무너진다(게이트는 닫혀 있는데 목표는 이미 발밑에 있는 상태). 놓인 원점 기준으로 잡는다.
GOAL_POS_Z = (CUP_SPAWN_Z + 0.08, CUP_SPAWN_Z + 0.20)

# ---------------------------------------------------------------------------
# 그리퍼 기하 (probe_gripper_opening.py 실측 — 참고값, 보상/제어에 직접 쓰지 않음)
# ---------------------------------------------------------------------------
# 최대 개구 84.5 mm. 조인트 origin(∓0.006)+스트로크(0.044)로 계산한 100 mm 가 아니다 —
# 충돌 근사가 convexHull 이라 통과폭은 가장 안쪽 점인 핑거 팁이 지배한다.
GRIPPER_MAX_OPENING = 0.0845
# shaker 는 계단형 원뿔(58/68/78/88 mm)이라 테이블 위 10~85 mm 에서만 개구를 통과한다.
GRASP_HEIGHT_BAND = (0.010, 0.085)
