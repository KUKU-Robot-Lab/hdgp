# 5g_pour_right_v3 — 태스크 로컬 규칙

## 현재 상태 (2026-05-25)

- **진행 중인 test**: test6 (막 시작됨)
- **태스크 특성**: bead_in_target 달성 이력 있는 안정적 학습 태스크
- **v3의 역할**: v5 설계 기준선 — v3에서 작동한 구조를 v5 벤치마크로 참조

---

## 안정 학습 파라미터 (v3 참조 기준값)

| 파라미터 | 값 | 비고 |
|---------|---|------|
| weight_tilt | 8.0 | v5 test1에서 3.0으로 낮춰 실패 |
| weight_align | 6.0 | |
| weight_demo_arm_pose | 0.0 | 비활성. v5 test1에서 9.0 높여 실패 |
| weight_demo_palm_pose | 0.0 | 비활성 |
| weight_spill | 40.0 | |
| pour_tilt_target_deg | 120.0 | 이 값에서 bead 흐름 달성 |
| enable_spill_adr | True | ADR 기반 점진적 spill 패널티 |
| success_target_fill_ratio | 0.50 | |
| success_spill_max | 0.40 | |

---

## Cross-Test 요약

| Test | Max Rew | Final Rew | 에포크 | 상태 |
|------|---------|-----------|-------|------|
| test1 | 8,368 | 5,632 | 4,750 | spike 후 하락 |
| test2 | 4,720 | 4,720 | 150 | 조기 종료 (데이터 부족) |
| test3 | 6,604 | 6,523 | 1,800 | 안정 수렴 |
| test4 | (미기록) | | | |
| test6 | - | - | 진행중 | ? |

---

## v3 구조 특징 (v5 대비)

- **BC auxiliary loss 없음**: demo_bc_buffer, success_traj_buffer 없음
- **데모 방식 없음**: nearest-neighbor / step-indexed 모두 미사용
- **pour_gate**: 즉시 활성 (pour_reward_start_step 없음)

---

## 분석 시 참조 지표

```
Episode/log/bead_in_target        ← v3에서 > 0 달성 가능
Episode/log/cup_center_xy_dist    ← < 0.15 달성 여부
Episode/log/rho                   ← > 0.9면 approach 학습됨
Episode/log/directional_tilt_cos  ← < -0.17이면 tilt 120도 달성
Episode/cost/spill                ← ADR 진행 여부 확인
```

---

## v5 분석 시 v3 벤치마크 활용

v3 달성 기준값을 v5 bottleneck 진단에 활용:

```
v3 달성 기준:
  - cup_center_xy_dist < 0.15m
  - rho > 0.9
  - bead_in_target > 0 (실제 pour 발생)
  - directional_tilt_cos < -0.17 (tilt ≥ 100도)

v5가 이 기준을 못 달성하는 지표 → bottleneck 후보
```

---

## 코드 수정 주의사항

- v3는 안정 기준선이므로 대규모 구조 변경 비권장
- 수정 전 기준 파라미터를 `test_history.md`에 반드시 기록
- obs/action 차원 변경 금지 (명시적 요청 없이)
