# open-tesol_r_reach_v4 태스크 개발 및 수정 이력

## 1. 개요 (Task Overview)
- **로봇 자산**: `openarm_tesollo_sensor_rl` (우팔 7-DOF OpenArm + 20-DOF Delto 5지 그리퍼, 좌측 2-DOF 프리즈매틱 그리퍼 유휴)
- **태스크 목표**: 수직 차렷 대기 자세(`[0, 0, 0, 0, 0, 0, 0]`)에서 시작하여 테이블 위 물체(8종 다중 자산)의 측면으로 자연스럽게 접근(Reach) 및 정밀 손바닥 정렬/파지
- **제어 방식**: 11D Action Space (Fabrics IK 기반 6D Palm Pose Target + 5D Per-Finger Lerp)

---

## 2. `test7` 분석 및 병목 진단 (Diagnosis)

W&B 롤아웃 비디오 및 500 에포크 시계열 지표 분석 결과:
1. **손바닥 하향 고정 (Palm Downward Locking)**:
   - 차렷 자세 초기화 시 손바닥 Euler 각도가 `(Roll 0°, Pitch 90°, Yaw 0°)`로 설정되어, 기구학 변환 시 손바닥 피부 법선(`+X`)이 World `-Z`(바닥)를 향함.
   - 기존 pour 태스크의 110도 각도 구속 잔재로 인해 `palm_delta_rot_deg`가 $\pm 20^\circ$로 묶여 있어 정책이 손바닥을 컵 옆면(수평)으로 회전시키지 못하고 바닥을 본 채 고정됨.
2. **손바닥 법선 벡터 축 불일치**:
   - 보상 함수(`_get_rewards`)에서는 `[0, 1, 0]`(+Y)을 참조하고, 종료/성공 판정(`_get_dones`)에서는 `[1, 0, 0]`(+X)을 참조하여 신호 불일치 발생.
   - `sensor_rl` URDF 검증 결과 손가락이 오므라드는 안쪽 면(진짜 손바닥 피부)은 **`+X` 축 (`[1.0, 0.0, 0.0]`)**임.
3. **카메라 뷰 원거리 문제**:
   - 기존 뷰어가 World 7.5m 상공 원거리 뷰로 설정되어 로봇 손과 컵 사이의 미세 접촉/정렬 상태 관찰 불가.

---

## 3. 코드 수정 내역 (Changes for `test8`)

| 항목 | 파일 위치 | 기존 설정 (test7) | 수정 설정 (test8) | 변경 사유 |
| :--- | :--- | :--- | :--- | :--- |
| **손바닥 정면 법선** | `grasp_right_env.py` (`_get_rewards`) | `[0.0, 1.0, 0.0]` (+Y) | **`[1.0, 0.0, 0.0]`** (+X) | `sensor_rl` URDF 기준 손바닥 피부 법선 벡터 정합 |
| **목표 손바닥 오일러각** | `grasp_right_env.py` (`_reset_idx`) | `(0°, 90°, 0°)` (Euler ZYX) | **`(90°, 0°, 90°)`** (Euler ZYX) | 팔 시작은 차렷 유지하되, Fabrics 타겟은 컵 측면 대면(Side Approach)으로 정렬 |
| **손바닥 회전 델타 범위** | `grasp_right_env_cfg.py` | `palm_delta_rot_deg = 20.0` | **`palm_delta_rot_deg = 35.0`** | 기존 붓기(pour) 110도 꺾기 구속 해제 및 접근 회전 자유도 확보 |
| **단일 로봇 전신 뷰어** | `grasp_right_preset.py`<br>`grasp_right_env_cfg.py` | World 원거리 뷰 (미지정) | **`eye = (1.4, -0.5, 0.75)`**<br>**`lookat = (0.35, -0.05, 0.35)`**<br>`origin_type = "env"`, `env_index = 0` | 단일 로봇(env 0)의 전신 + 테이블 + 컵이 선명하게 들어오는 정면-우측 HD(1280x720) 뷰 고정 |
| **검증 스크립트 기본 뷰** | `scripts/.../play.py` | `is_pour`만 근접뷰 | **`reach`/`grasp`도 신규 근접 전신 뷰 자동 적용** | 검증 및 MP4 비디오 렌더링 시 최적 앵글 자동 제공 |

---

## 4. 실행 및 검증 방법

### 학습 실행 (GPU 머신)
```bash
# W&B 실시간 그래프 자동 전송 (train.sh 내장)
./train.sh open-tesol_r_reach_v4 test8 --num_envs 1024

# 비디오 함께 렌더링 시
./train.sh open-tesol_r_reach_v4 test8 --num_envs 1024 --video
```

### 롤아웃 검증 및 비디오 생성 (Play)
```bash
isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task open-tesol_r_reach_v4 \
    --checkpoint log/rl_games/open-tesol/right/reach_v4/test8/nn/open-tesol_r_reach_v4.pth \
    --num_envs 1 \
    --headless \
    --video \
    --video_length 200
```
