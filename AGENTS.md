# hdgp 에이전트 정의

## 역할 분리 원칙

```
분석  →  감사  →  수정  →  기록
 ↑                           ↓
 └─────── 다음 사이클 ────────┘
```

- **분석(Analyzer)**: 로그 읽기, bottleneck 특정. 코드 수정 금지.
- **감사(Audit)**: 수정 제안의 위험도 평가. 코드 수정 금지.
- **수정(Modifier)**: 감사 통과 후에만 코드 수정.
- **기록**: analysis.md + test_history.md 자동 업데이트.

---

## rl-training-analyzer

**언제**: 학습 결과를 분석해야 할 때.

**입력**: test 폴더 경로 (예: `5g_pour_right_v5/test6`)

**책임**:
- TFEvents 파싱 (`parse_tfevents.py` 사용)
- 체크포인트 reward 추세 파싱
- git 타임라인으로 해당 test의 코드 변경 특정
- 이전 test들과 지표 비교
- **하나의 primary bottleneck만 출력** (여러 개 나열 금지)
- analysis.md에 결과 기록

**출력 형식**:
```
## 상태: [🟢상승 / 🟡불안정 / 🔴붕괴 / ⚪Plateau]
## 핵심 근거: <지표명=수치> 기반 1~2문장
## 이전 test 대비: <지표명> X → Y
## 주 원인: <하나만>
## 다음 수정 후보: <하나만>
## 검증 기준: <어떤 지표가 어떻게 변하면 성공인지>
```

**금지**: 코드 수정, 수정 제안 목록 나열, 일반 RL 조언.

---

## rl-pour-modifier

**언제**: 분석 결과를 바탕으로 코드를 수정해야 할 때.

**전제조건**: `rl-training-analyzer`의 분석 결과가 있어야 함.

**책임**:
1. `test_history.md` 읽기 (이전에 시도된 변경인가?)
2. `analysis.md` 읽기 (분석 근거 확인)
3. **reward-audit 체크리스트 실행** (`~/.claude/skills/reward-audit/SKILL.md`)
4. audit 통과 시에만 외과적 수정
5. git diff 요약 → `test_history.md` 기록

**금지**: 
- audit 없이 reward/weight/gate 변경
- 이미 ✗ 판정된 변경 반복 (명시적 이유 없이)
- 관련 없는 파일 수정
- obs/action 차원 변경 (명시적 요청 없이)

---

## 스킬: reward-audit

**언제**: reward weight, gate, 새 reward term 추가/변경 전.

**사용법**: `rl-pour-modifier`가 Phase 2에서 자동 실행.

상세 내용: `~/.claude/skills/reward-audit/SKILL.md`

---

## 스킬: rl-pour-diagnostics

**언제**: pour 태스크 특화 지식이 필요할 때 (지표 해석, 알려진 실패 패턴).

상세 내용: `~/.claude/skills/rl-pour-diagnostics/SKILL.md`

---

## 실행 순서 예시

```
사용자: "test6 분석해줘"
  → rl-training-analyzer 실행
  → analysis.md 기록
  → "주 원인: directional_tilt_cos ~0.8 (41°), weight_tilt 약화"
  → "다음 후보: weight_tilt 5.0→8.0"

사용자: "수정해줘"
  → rl-pour-modifier 실행
  → test_history.md 확인 (test2에서 weight_tilt=8.0 시도? → 그 당시와 조건 다름)
  → reward-audit: weight_tilt ↑ → tilt gradient 강화 → local optima 변화?
  → audit 통과 → 수정 실행
  → test_history.md 기록

사용자: "./train.sh 5g_pour_right_v5 test7"
  → record_test_snapshot.py 자동 실행
  → 학습 시작
```
