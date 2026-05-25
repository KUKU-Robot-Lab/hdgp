# 5g_pour_right_v5 — 태스크 로컬 규칙

## 현재 상태 (2026-05-25)

- **진행 중인 test**: test6 (막 시작됨)
- **핵심 미해결 문제**: bead_in_target = 0.000 (전체 이력 동안 성공 에피소드 없음)
- **최근 분석 결론**: step-indexed demo 시간축 불일치가 test4 실패의 주원인

---

## 현재 파라미터 (test6 기준)

| 파라미터 | 현재 값 | 주의 사항 |
|---------|--------|---------|
| weight_tilt | 5.0 | test4에서 8.0→5.0 하향됨 |
| weight_align | 10.0 | 과도하게 높음 (test4 진단) |
| weight_demo_arm_pose | 2.00 | test1의 9.0에서 크게 하향됨 |
| weight_demo_palm_pose | 2.00 | |
| cup_collision_margin | 0.12m | pour 중 두 컵 근접 불가 (test3 붕괴 원인) |
| pour_tilt_target_deg | 120.0 | v3 값 복원 |
| pour_tilt_sharpness | 2.0 | gradient 범위 확대용 |

---

## 알려진 실패 패턴 (절대 반복 금지)

| 변경 | 어느 test | 실패 이유 |
|-----|---------|---------|
| weight_demo_arm_pose = 9.0 | test1 | tilt 보상의 13배 지배 → demo local min |
| weight_align = 10.0 | test4 | pour 탐색보다 정렬 최적화 집중 → pour 미발현 |
| cup_collision_margin = 0.12m | test3 | pour 중 두 컵 근접 불가 → collapse |
| step-indexed demo (warmstart 오프셋 없음) | test4 | warmstart 상태(t=0)와 demo[0] 불일치 |
| pour_gate 지연 50k step | test1 | approach만 학습 후 tilt gradient 없음 |

---

## Cross-Test 요약

| Test | Max Rew | Final Rew | 에포크 | 상태 | 핵심 이슈 |
|------|---------|-----------|-------|------|----------|
| before/test3 | 56,900 | 55,064 | 5,150 | 수렴 | 구 reward, pour 됨 |
| before/test4 | 69,361 | 57,002 | 5,300 | 상승 | 구 reward, pour 됨 |
| test1 | 14,551 | 14,129 | 10,000 | Plateau | weight_demo_arm=9.0 로컬 min |
| test2 | 36,120 | 35,180 | 10,000 | Plateau | 일부 pour 있음 |
| test3 | 27,024 | 7,307 | 10,000 | 붕괴 | cup_collision+spill 복합 상충 |
| test4 | 14,652 | 13,873 | 3,000 | Low Plateau | step-indexed demo 오프셋 없음 |
| test6 | - | - | 진행중 | ? | |

---

## 분석 시 필수 확인 지표

```
Episode/log/bead_in_target        ← 가장 중요 (0이면 pour 없음)
Episode/log/bead_in_source        ← 구슬 소스컵 유지율
Episode/log/directional_tilt_cos  ← 0.75 이상이면 tilt 41도 미달
Episode/log/pour_gate             ← pour 단계 활성화율
Episode/log/cup_center_xy_dist    ← 두 컵 거리
Episode/log/demo_arm_joint_err    ← demo 추종 오차
Episode/cost/cup_collision        ← 충돌 패널티 (test3 붕괴 원인)
bc/loss_sim                       ← 0이면 성공 에피소드 없음
```

---

## v5 특수 구조 (v3 대비 차이점)

- **LSTM + BC auxiliary loss**: `lstm_bc_agent.py`, `demo_bc_buffer.py`, `success_traj_buffer.py`
- **데모 방식**: `warmstart_logic.py` → 에피소드 시작 시 demo 중간 상태로 warmstart
- **demo 인덱싱**: `demo_pose_reference.py` — step-indexed 사용 시 반드시 warmstart 오프셋 적용
  - `demo[t + start_frame]`, `start_frame = demo_start_fraction * T_demo`
- **pour_gate**: 활성화 조건 충족 시 tilt/pour 보상 켜짐

---

## 다음 수정 후보 (분석 우선순위 순)

1. **step-indexed demo 오프셋 수정**: `pour_right_env.py` demo_idx 계산 부분
   - 검증 기준: demo_arm_joint_err 초기값 0.5 → 0.1 수준 감소
2. **weight_align 하향**: 10.0 → 4.0~5.0
   - reward-audit 통과 후 진행 필수
3. **cup_collision_margin 완화**: 0.12m → 0.08m (또는 pour phase 비활성)

---

## 코드 수정 금지 사항 (v5 특이사항)

- `demo_bc_buffer.py`, `success_traj_buffer.py` 구조 변경 금지
- `lstm_bc_agent.py` 수정 금지 (명시적 요청 없이)
- obs 차원 변경 금지 (LSTM hidden state 재학습 불가)
