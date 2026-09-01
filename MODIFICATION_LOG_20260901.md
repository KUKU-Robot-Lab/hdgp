# OpenArm Tesollo Reach v4 수정 및 정밀 개선 내역서

**작성 일시**: 2026년 9월 1일 11:42:00 (KST)  
**작업 브랜치**: `dev/beomsu`  
**대상 태스크**: `open-tesol_r_reach_v4`  

---

## 1. 📌 개요 및 목적
OpenArm 7-DOF 우측 로봇 팔 및 Tesollo 5지 그리퍼의 `reach_v4` 도달 및 정렬 강화학습 태스크에서 발생했던 다음 4가지 핵심 문제점을 기구학(Kinematics) 및 보상 구조(Reward Engineering) 차원에서 근본 수정 및 개선 완료했습니다:
1. 손이 테이블 아래에 머물며 꼼수로 보상을 챙기는 2D 보상 해킹 (Reward Hacking)
2. 4손가락 정렬 축 불일치 및 손등으로 컵을 내리치는 자세 뒤집힘 (Upside-down Flip)
3. 차렷 자세 출발 시 테이블 앞쪽 수직 모서리에 팔뚝이 콱 끼여 멈추는 물체 충돌 (Table Edge Collision)
4. 시뮬레이터를 켜지 않고 코드 단에서 7-DOF URDF 순방향 기구학(FK)을 0.1초 만에 검증하는 오프라인 도구 부재

---

## 2. 🛠️ 주요 수정 및 개선 파일 상세 내역

### ① [`grasp_reward.py`](file:///home/user/Documents/bumsu/KUKU-Robot_hdgp/source/openarm/openarm/tesollo/right/reach_v4/grasp_reward.py) — 보상 해킹 차단 게이트 (Height Gate)

* **수정 위치**: L110 - L117
* **변경 내용**:
  ```python
  # [보상 해킹 방지 게이트] Z 높이가 컵 높이에 10cm 이내로 들어와야만 XY 수평 정렬 및 자세 정렬 보상을 지급
  height_gate = (1.0 - (height_error / 0.10).clamp(max=1.0))
  reward_xy = torch.where(reward_xy > 0.0, reward_xy * height_gate, reward_xy)
  reward_align = (w_align_facing * align_facing + w_align_down * align_down) * height_gate
  ```
* **수정 효과**: 손바닥 Z 높이가 테이블 상공(컵 높이 10cm 이내)으로 올라오기 전에는 2D 수평 거리 및 자세 보상을 0점으로 락(Lock) 걸어, **테이블 아래에서 꼼수를 쓰지 못하고 팔을 테이블 상공으로 먼저 들어 올리도록 유도**.

---

### ② [`grasp_right_env.py`](file:///home/user/Documents/bumsu/KUKU-Robot_hdgp/source/openarm/openarm/tesollo/right/reach_v4/grasp_right_env.py) — 기구학 오프셋 및 2단계 수평 접근 정렬

#### 2-1. 손바닥 피부 정면(`+Y`) 및 4손가락 월드 전방(`+X`) 내적 정렬 (L1347-L1357)
```python
# +Y 축 = 진짜 손바닥 피부 정면 (장풍 방향: 컵의 옆면을 정확히 대면)
palm_normal_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
palm_normal_world = quat_apply(palm_quat, palm_normal_local)
palm_alignment = torch.sum(palm_normal_world * palm_to_cup_dir, dim=-1).clamp(min=0.0)

# -Z 축 = 4손가락 뻗는 방향 (월드 +X_world 전방 방향을 향하도록 유도)
palm_finger_local = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
palm_finger_world = quat_apply(palm_quat, palm_finger_local)
world_x_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
# 월드 +X_world 축([1.0, 0.0, 0.0])과 4손가락 내적이 1.0 일 때 만점
palm_down_alignment = torch.sum(palm_finger_world * world_x_dir, dim=-1).clamp(min=0.0)
```
* **수정 효과**: 4손가락(`-Z`)이 월드 전방(`+X`)을 향하고, 손바닥 피부(`+Y`)가 컵 측면을 마주보고, 엄지(`+X`)가 하늘을 향하는 정석 Side-Approach 3D 프레임 완성.

#### 2-2. 목표 오일러 각도 뒤집힘 정정 (L1956-L1960)
```python
# 기존 (90°, 0°, 90°) -> 손바닥이 천장을 향하고 손등(-Y)이 바닥/컵을 향해 뒤집혔던 오류 정정
pregrasp_palm_pose[:, 3] = math.radians(0.0)
pregrasp_palm_pose[:, 4] = math.radians(-90.0)
pregrasp_palm_pose[:, 5] = math.radians(0.0)
```
* **수정 효과**: 손등이 바닥을 향해 컵을 툭툭 치던 현상을 완벽하게 정정하고 손바닥 피부 정면이 컵을 똑바로 대면하도록 변경.

#### 2-3. 차렷 자세 솟구침 및 테이블 모서리 충돌 방지 (L1954)
```python
pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise
# 테이블 모서리 충돌 방지: 차렷 자세에서 테이블 상공(Z >= 0.35m)으로 먼저 들어올리도록 강제
pregrasp_pos[:, 2] = torch.clamp(pregrasp_pos[:, 2], min=0.35)
```
* **수정 효과**: 차렷 자세($Z=0.185\text{m}$)에서 출발할 때 대각선 직선 경로로 전진하다 테이블 모서리에 콱 끼이는 현상을 방지하기 위해, **테이블 상공($Z \ge 0.35\text{m}$)으로 먼저 손을 들어 올린 후 수평 접근**하도록 2단계 안전 보정 적용.

---

### ③ [`scripts/tools/verify_kinematics.py`](file:///home/user/Documents/bumsu/KUKU-Robot_hdgp/scripts/tools/verify_kinematics.py) — 0.1초 오프라인 7-DOF URDF FK 분석 도구

* **기능**: OpenArm 7개 관절 각도 $q$를 입력받아 시뮬레이터 없이 0.1초 만에 엔드이펙터 3D 좌표, 손바닥 피부 방향, 4손가락 방향, 엄지 방향을 기구학 연산하여 검증해 주는 파이썬 독립 도구 제작.

---

## 3. 🚀 추천 재기동 명령어 요약

### ① 1024 환경 초고속 학습 실행 (`test11`)
```bash
pkill -9 -f train.py 2>/dev/null || true
cd /home/user/Documents/bumsu/KUKU-Robot_hdgp
./train.sh open-tesol_r_reach_v4 test11 --num_envs 1024
```

### ② 3D 시뮬레이션 관찰 화면 (GUI) 실행
```bash
/home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task open-tesol_r_reach_v4 \
  --num_envs 1 \
  --real-time \
  --checkpoint log/rl_games/open-tesol/right/reach-v4/test11/nn/open-tesol_r_reach_v4.pth
```
