# 5g_pour_right_v5 — 태스크 로컬 규칙

## 현재 상태 (2026-06-10 재설계)

- **v5 = v3 검증 구조(stageB 래치, 20% 성공) + bead-count 커리큘럼.** BC 시스템 전부 제거.
- **목표: FRESH ONE-SHOT** (warmstart/teleport 없이 scratch에서 pour 학습).
- **근거**: v3 구조는 warmstart로만 20% 도달, fresh는 직립-park deadlock. DexPour ablation —
  커리큘럼 없는 full reward는 premature park 수렴(=fresh deadlock). 커리큘럼이 부트스트랩 enabler.

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
