# OpenArm Tesollo Reach v4 기구학 한계값(Bounds) 정합 및 3D 뷰어 추가 내역서

**작성 일시**: 2026년 9월 1일 23:50:00 (KST)  
**작업 브랜치**: `dev/beomsu`  
**대상 태스크**: `open-tesol_r_reach_v4`  
**관련 커밋**: `8dcfec52`

---

## 1. 📌 개요 및 목적
`reach_v4` 태스크에서 초기 목표 자세를 `(0°, -90°, 0°)`로 수정한 후 발생했던 **손목이 바닥 대각선으로 꺾이는 비정상 회전 왜곡 현상**을 기구학적으로 정밀 분석하여, 허용 각도 범위(Limits Bounds) 및 성공 판정 신호 불일치를 근본 해결하고 윈도우용 독립 3D 기구학 랩 뷰어를 구축 완료했습니다.

---

## 2. 🛠️ 주요 수정 및 개선 파일 상세 내역

### ① [`grasp_right_preset.py`](file:///c:/Users/User/RL/KUKU_hdgp/source/openarm/openarm/tesollo/right/reach_v4/grasp_right_preset.py) — 회전 작업공간(Bounds) 한계 완전 정합

* **수정 위치**: L159 - L177
* **문제점**:
  * 과거 구버전 `(90°, 0°, 90°)` 기준 울타리 `[55°~125°, -35°~35°, 55°~125°]`가 남아있어, 신규 목표 `(0°, -90°, 0°)`로 초기화 시 `(55°, -35°, 55°)`로 강제 클램프되어 로봇 손목이 바닥 대각선으로 꺾이던 치명적 결함 발생.
* **변경 내용**:
  ```python
  def palm_pose_mins(max_pose_angle: float) -> list:
      d = math.pi / 180.0
      return [
          0.20, -0.55, 0.20,
          (0.0 - max_pose_angle) * d,
          (-90.0 - max_pose_angle) * d,
          (0.0 - max_pose_angle) * d,
      ]

  def palm_pose_maxs(max_pose_angle: float) -> list:
      d = math.pi / 180.0
      return [
          0.65, 0.22, 0.65,
          (0.0 + max_pose_angle) * d,
          (-90.0 + max_pose_angle) * d,
          (0.0 + max_pose_angle) * d,
      ]
  ```
* **수정 효과**: 목표 오일러각 중심인 `(0°, -90°, 0°)` $\pm 35^\circ$ 범위 안에서 로봇 손이 벽에 걸리지 않고 자유롭게 정석 Side-Approach 자세로 탐색 및 제어 가능.

---

### ② [`grasp_right_env.py`](file:///c:/Users/User/RL/KUKU_hdgp/source/openarm/openarm/tesollo/right/reach_v4/grasp_right_env.py) — 성공 판정 손바닥 법선 일치

* **수정 위치**: L1652
* **문제점**:
  * 보상 함수에서는 진짜 손바닥 피부인 `+Y` (`[0.0, 1.0, 0.0]`)를 쓰고 있으나, `reach_condition_now` 성공 판정 루프에서는 구버전 `+X` (`[1.0, 0.0, 0.0]`)를 검사하여 신호 불일치 발생.
* **변경 내용**:
  ```python
  palm_normal_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
  ```
* **수정 효과**: 보상 함수와 성공 조건의 손바닥 대면 판정 벡터가 `+Y`로 100% 일치.

---

### ③ [`docs/robot_kinematics_lab.html`](file:///c:/Users/User/RL/KUKU_hdgp/docs/robot_kinematics_lab.html) — 독립형 3D 로봇 기구학 시뮬레이터 (신규 추가)

* **기능**:
  1. GPU / Isaac Sim / 우분투 없이 윈도우 크롬/엣지 브라우저에서 더블클릭으로 즉시 실행되는 3D WebGL 시뮬레이터.
  2. OpenArm 7-DOF 관절 슬라이더 ($q_1 \sim q_7$) 및 목표 오일러 회전 (Euler ZYX) 실시간 60fps 렌더링.
  3. 손바닥 3축 화살표 (초록=+Y, 파랑=-Z, 빨강=+X) 및 테이블, 실제 학습 컵(직경 8.4cm, 높이 12cm) 스폰.
  4. 손바닥-컵 대면 내적 점수 및 손가락 전방 수평 정렬 점수 실시간 연산/표시.
  5. 정석 Side Approach 자세와 이전 왜곡 버그 자세(`55, -35, 55`)를 원클릭으로 비교 검증.

---

## 3. 🚀 검증 및 실행 권장

```bash
# 1. 서버에서 최신 코드 동기화
git pull origin dev/beomsu

# 2. 1024 환경 학습 재기동 (test12)
./train.sh open-tesol_r_reach_v4 test12 --num_envs 1024
```
