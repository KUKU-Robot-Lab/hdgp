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

"""상수 정의: 5g_grasp_left_v1

right/grasp_v1(grasp_right_constants)의 좌우 미러. v7: Fabrics 팔 학습(6D palm action)
+ per-finger lerp(5D) + Contact sensor 없는 FK 기반 보상.

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (thumb, index, middle, ring, pinky)
         -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE

Actor Observation (106D) — sim2real 가능:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  palm_center_pos (world):  3
  fingertip_pos_rel_palm:  15  (5 × 3D)
  palm_to_cup_pos:          3
  cup_to_fingertip:        15  (5 × 3D)
  fingertip_contact_binary: 5  (FT sensor, 실 로봇 가능)
  last_actions:            11
  Base Total:             106
  object_onehot:            8  (2026-07-26 MultiAsset 8종, 뒤에 append)
  Total:                  114

Critic Extra (37D) — sim-only privileged:
  cup_lin_vel:              3
  cup_ang_vel:              3
  cup_rot (quat):           4
  cup_height_delta:         1
  distal_contact_binary:    5  (lj_dg_*_4 미러, 실제 l_hl_*_4)
  distal_contact_norm:      5
  middle_contact_binary:    5  (l_hl_*_3)
  middle_contact_norm:      5
  phase_step_ratio:         1
  fingertip_to_cup_signed_dist: 5
  Total:                   37

Critic Total: 114 + 37 = 151D

Episode (10s @ 60Hz = 600 steps):
  Grasp phase (0~479):  Fabrics arm + per-finger policy
  Lift-wait phase (480~599): scripted joint7-only lift-wait + frozen hand
"""

import math

from .grasp_left_preset import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    palm_pose_mins,
    palm_pose_maxs,
    LEFT_ARM_START_POSE,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF   = len(LEFT_ARM_JOINT_NAMES)    # 7
NUM_HAND_DOF  = len(LEFT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF     # 27
NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION   = 6   # 6D palm pose (Fabrics IK)
# ★08.16 PIP/DIP 분리 — 손가락 action 5D → 15D (5 손가락 × 3 채널).
#
# 왜: 구 5D 는 손가락당 스칼라 하나를 4관절에 그대로 복사(repeat_interleave(4))했다.
#   그러면 관절 목표가 open_pose → full_grip_pose **직선 하나** 위에만 존재하고,
#   관절 사이의 각도 비율이 full_grip_pose 로 고정된다. 즉 "MCP 는 깊게 굽혀 근위 마디를
#   컵에 붙이고 PIP/DIP 는 얕게 두어 원위 마디가 말리지 않고 표면에 눕는" 자세 —
#   **진짜 인벨롭 자세가 액션 공간에 존재하지 않았다.**
#   또한 _1(외전)·_2(MCP)는 접촉 게이트가 없어 물체 지름과 무관하게 항상 full_grip 까지
#   닫혔다(큰 물체 s130·12_cyl 취약과 연결).
# 근거: S2 실측 — stiffness 를 20배 낮춰 접촉력이 30→9N 으로 떨어져도 이탈가속은 20~28로
#   평탄했다. 외란 내성은 **파지력이 아니라 감쌈 기하에 지배**된다. 따라서 지렛대는
#   힘 채널(squeeze)이 아니라 형상 자유도다.
# 채널: [0]=_1 외전, [1]=_2 MCP, [2]=_3/_4 PIP·DIP 공통.
#   _3/_4 를 묶어 두는 이유 = 이 둘은 접촉 동결이 이미 개별 적응을 담당한다(g3/g4).
#   전 관절 독립(full-joint 20D)은 그 동결 래칫 구조를 버리게 되어 3지 국소최적 회귀 위험.
NUM_FINGER_CHANNELS = 3
NUM_FINGER_ACTION = NUM_FINGERTIPS * NUM_FINGER_CHANNELS   # 15
# ★08.16 squeeze(가압) 채널 철회 — 2D → 0D.
# 도입 근거였던 "동결 때문에 오버슈트가 0 이라 더 조일 수단이 없다"는 진단은 틀렸다:
#   실제로는 접촉 관절의 80.6% 가 effort limit 7.5 N·m 에 **포화**해 있었고(요구 토크 143 N·m),
#   즉 이미 모터 최대로 쥐고 있어 목표를 더 밀어도 힘이 오르지 않았다. 실측에서도 가압을
#   올릴수록 접촉력이 36→25N 으로 **감소**했다(기하만 말려 접촉 마디를 잃음).
# 게인 재설정(k=400→5)으로 포화는 해소됐지만, PIP/DIP 분리 채널이 같은 권한을 더 자연스럽게
#   주므로(관절 비율 자체를 정책이 정함) 별도 힘 채널은 중복이다. 변수를 늘리지 않는다.
# 되살릴 조건: 재학습 후에도 "감쌈은 좋은데 외란에 힘이 모자라 놓친다"가 실측되면 그때.
NUM_SQUEEZE_ACTION = 0
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION + NUM_SQUEEZE_ACTION  # 6+15 = 21

# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------
# 2026-07-26 MultiAsset(8종)+DR 이식: 물체 onehot 8D를 base obs 뒤에 append.
# ★08.16: fingertip 접촉 obs 를 binary 5D → **3축 힘 15D**(tip-local, 10N 정규화)로 교체.
# 106 - 5 + 15 = 116. 근거: Tesollo DG-5F 손끝에 6축 F/T 가 내장되어 실기도 3축 force 를
# 발행하며(/dg5f_left/fingertip_*/wrench), binary(임계 0.1N)로는 정책이 자기 파지력을
# 관측·조절할 수 없다(재조임 권한을 열어도 무의미).
# last_actions 11D→13D(+2) 및 관절 위치 오차 20D 추가(+20) → 116+2+20 = 138.
# ★08.16 PIP/DIP 분리로 last_actions 13D→21D(+8) → 138+8 = 146.
NUM_OBSERVATIONS_BASE = 146   # Actor base: sim2real 가능 (물체 onehot 제외)
NUM_OBJECT_CLASSES    = 8     # 물체군: cup_big×4 scale + shaker_body + cyl×3(직경5/8/12, 높이12cm 통일)
NUM_OBSERVATIONS = NUM_OBSERVATIONS_BASE + NUM_OBJECT_CLASSES    # 138+8=146
NUM_DISTAL_SENSORS  = 5       # ll_dg_*_4
NUM_MIDDLE_SENSORS  = 5       # ll_dg_*_3
NUM_CRITIC_EXTRAS   = 37
NUM_CRITIC_OBSERVATIONS = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 146+37=183

# ---------------------------------------------------------------------------
# Episode structure (@ 60 Hz)
# ---------------------------------------------------------------------------
GRASP_PHASE_STEPS      = 480    # 8s: Fabrics arm + per-finger policy
LIFT_WAIT_PHASE_STEPS  = 120    # 2s: keep grasp arm pose, move only joint7
LIFT_PHASE_STEPS       = LIFT_WAIT_PHASE_STEPS
LIFT_START_STEP        = GRASP_PHASE_STEPS                              # 480
EPISODE_STEPS          = GRASP_PHASE_STEPS + LIFT_WAIT_PHASE_STEPS      # 600

LIFT_WAIT_JOINT7_DELTA = 0.31

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
# ★08.16 S5 재검증 — 새 게인 영역(k=5·kd=2.0)에서도 두 값 모두 **유효 확인, 변경 없음**.
#   손끝 접촉력 분포: p95=7.77 / max=10.37 N → MAX=10.0 이 여전히 적정(포화 없이 대부분 커버).
#   비영 접촉의 하위 5분위 = 1.86 N ≫ THRESHOLD 0.1 → 접촉 판정이 놓치는 구간 없음.
CONTACT_FORCE_THRESHOLD  = 0.1    # N  binary contact 판정
CONTACT_FORCE_MAX        = 10.0   # N  정규화 분모
# ★08.16 관절 위치 오차(지령−실측) 정규화 분모 [rad].
# 왜 이 관측인가: 힘 ∝ stiffness × (지령−실측) 이고, **어느 마디가 닿든 막힌 만큼 오차가 생긴다**.
#   손끝 F/T 는 인벨롭 그립에서 팁이 안 닿으면 0 을 읽는다(실측: 엄지 팁 접촉 0.619 vs
#   아무 마디 0.844 — 약 40% 구간에서 팁 무접촉). 즉 **파지가 좋을수록 힘 신호가 사라진다**.
#   반면 관절 오차는 중간·원위마디 접촉도 잡아내므로 실기 배포 가능한 유일한 전마디 힘 관측이다.
# 실기 계산법: 우리가 보낸 지령 − /dg5f_right/joint_states 실측. 추가 센서 불필요.
# ★08.16 0.10 → 1.2 확정 (S5 실측, k=5·kd=2.0 영역). PIP/DIP 오차 분포는
#   p50=0.003 / p95=1.165 / max=1.763 rad — 대부분의 관절은 자유 추종이라 ~0 이고,
#   **막힌 관절만 큰 값을 갖는 롱테일**이다(그 꼬리가 곧 파지력 신호). 구 0.10 은 그 꼬리를
#   전부 클립해 신호를 지워버렸다. p95 를 분모로 잡아 꼬리를 살린다.
#   구 값이 sim 400 기준으로도 틀렸던 게 아니라, 애초에 실측 없이 넣은 잠정치였다.
# ⚠️실기 손 P게인은 1.5 라 같은 힘에서 오차가 3.3배 크다(sim 5.0 대비) — r2s 게인 재측정
#   후 이 분모를 재확인할 것. sim/실기 게인이 정합되면 분모도 그대로 쓸 수 있다.
JOINT_POS_ERR_MAX        = 1.2    # rad
MIN_CONTACTS_FOR_SUCCESS = 2      # 성공 판정 최소 접촉 손가락 수

# ---------------------------------------------------------------------------
# FABRICS pregrasp
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60

# ---------------------------------------------------------------------------
# Cup geometry
# ---------------------------------------------------------------------------
CUP_RADIUS_APPROX = 0.045  # m, (legacy fallback) 07.26부터 per-object bbox(GraspLeftEnv.cup_radius_approx_buf)가 대체 — 미사용

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
ARM_START_POSE      = LEFT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
