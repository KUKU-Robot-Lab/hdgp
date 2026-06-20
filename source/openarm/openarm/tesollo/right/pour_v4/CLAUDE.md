# 5g_pour_right_v4 — 태스크 로컬 규칙

## 현재 상태 (2026-06-19 생성)

- **v4 = pour-v5 reward/control 구조 그대로 + v3 demo pose reward 이식.**
- **근거**: pour-v5 lstm_test2 붕괴 진단 = deep tilt 시 j6/j7 손목 range 소진 → pour-point z escape.
  "실현 가능한 tilt joint_state를 탐색으로 못 찾음"이 근본 원인.
- **해법**: a11~a20 pour 분포로 j1-4 + j5(틸트 주역) 앵커(감쇠+floor) → 적절 joint_state 조기 발견.
  + arm 중립 nullspace 제거(demo reward와 충돌) → demo reward가 arm config 유도.
- 상세: `log/rl_games/open-tesol/right/pour-v4/analysis.md`
- 아래 메커니즘 문서(bead 커리큘럼 등)는 v5에서 상속, 그대로 유효.

---

## 핵심 메커니즘: bead-count 커리큘럼

- 물리 비드 `_DEFAULT_BEAD_COUNT=30` 고정 spawn. 커리큘럼이 **활성 N**만 사용(앞 N 슬라이스).
- 비활성 비드는 hide(z=-10). 모든 bead fraction(in_target/source/spill/near/centroid)은 **활성 N 정규화**.
- N 스케줄: `(1,5,8,10,20,30)`. stage-windowed success_rate ≥ 0.5 (interval 20k step, 최소 5회) 시 advance.
- **원리**: 1-bead면 우연한 틸트가 그 1개를 넣어 bead_in(200) 보상을 *처음 경험* → tilt 부트스트랩.
  sparse reward를 reachable하게 만드는 게 핵심 (DexPour APA+curriculum).

### 모듈
- `bead_curriculum.py` — `BeadCountCurriculum` (순수 로직, 단위테스트 `tests/test_bead_curriculum.py` 8/8)
- env 배선: `_active_bead_count`, `_compute_bead_flags` 슬라이스, `_sample_bead_states_inside_cup` hide,
  ADR 영역 advance 훅, `_bead_curric_*` windowed 집계.

### 로그 지표
```
log/active_bead_count    ← 현재 활성 비드 수 (커리큘럼 진행)
log/bead_curric_winrate  ← stage-windowed success rate (advance 트리거)
log/bead_in_target       ← 활성 N 정규화된 채움 (0이면 pour 없음)
```

---

## 회귀 방지 (Phase 3)

- N 증가 시 bead_in(200)이 anti-spill·penalty를 압도해야 park 회귀 안 함.
  착지율 15%만 유지돼도 틸트(bead_in≈30/step) > park(dense≈25/step).
- 커리큘럼이 success 기반이라 못 하면 advance 안 됨(자동 가드).

---

## 미해결 (Step 3 후보 — 틸트 부트스트랩 보강)

커리큘럼 단독으로 fresh tilt가 안 풀리면:
- 초기 탐험 자유 (entropy_coef/sigma 높게 시작)
- pre-pour dense cap (직립 farming 차단)
- demo_j5 cold-gradient 수정 (exp sharpness↓ or 선형) — reward-audit 필수

## 코드 수정 규칙

- reward/gate/weight 변경 전 reward-audit 필수 (`~/.claude/skills/reward-audit/`)
- obs/action 차원 변경 금지 (명시적 요청 없이)
- v3는 별개 태스크(불변) — v5만 수정
- 학습 전 `record_test_snapshot.py --task open-tesol_r_pour_v5 --test <name>`
