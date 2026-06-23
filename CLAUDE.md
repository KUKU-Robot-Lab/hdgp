# hdgp — 학습 실험 저장소 공통 규칙

> **이 저장소는 일반 SW 개발이 아니라 실험 루프다.** 코드 품질보다 가설 검증이 우선.
>
> **도메인 해석·진단 방법론·reward 구조·핵심 지표는 각 프로젝트 전용 CLAUDE.md를 기준으로 한다.**
> - pour: `source/openarm/openarm/tesollo/right/pour_v5/CLAUDE.md` (v6는 `@import`)
> - grasp 등: 해당 프로젝트 폴더의 CLAUDE.md
>
> hdgp는 **모든 학습 프로젝트 공통**만 담는다. 포괄적 지표·프로젝트 도메인 규칙은 여기 두지 않는다.

Isaac Lab / Isaac Sim 기반. OpenArm(7 DOF) + Teosllo(20 DOF) 조작.

---

## 핵심 규칙: 로그 먼저

**로그(TFEvents) 수치 근거 없이 코드를 수정하지 않는다.**

```
1. TFEvents + 체크포인트에서 수치 추출
2. 이전 test와 지표 비교
3. 가설 주도로 primary bottleneck 특정
   (고정 체크리스트 ✗ — 프로젝트 전용 CLAUDE.md의 진단 방법론을 따른다)
4. 최소 수정 → reward-audit 통과
5. record_test_snapshot(train.sh 자동) → analysis.md 누적 → 주제 완료 시 Notion
```

---

## 증거 우선순위

하위 항목이 상위 항목을 뒤집을 수 없다.

```
1 TFEvents 스칼라 (parse_tfevents.py)
2 play.py 렌더링 / 사용자의 직접 관찰
3 analysis.md 이전 분석 이력
4 test_history.md (git diff 기반)
5 소스 코드 직접 읽기
6 일반 RL 직관
```
TFEvents 수치와 모순되는 일반 RL 조언은 무시한다.

---

## 분석 원칙

- **단일 로깅값·단일 시점 판단 금지.** 표면 지표(reward·entropy·성공률·frac)는 **출발점일 뿐**이다.
- **가설 주도 인과 추적**: 표면에서 멈추지 말고 메커니즘 — action 추종(raw→gate→applied), gate 경로,
  물리 좌표, joint 한계, reward 구성비 — 까지 직접 측정으로 검증한다.
- **봐야 할 지표는 가설마다 다르다.** 고정 N축 체크리스트만으론 RL 실험의 *원인*을 알 수 없다.
- 프로젝트별 진단 절차·핵심 지표·인과 모델은 **해당 프로젝트 CLAUDE.md 참조.**

---

## 실행 환경 (분산)

- **로컬** (`user@10.102.101.234`, RTX 5090): 분석·코드 수정·모니터링.
- **server** (`oem@10.102.101.240`, RTX PRO 6000 Blackwell ×2, 98GB): 학습.
- **코드 동기화**: 로컬 수정 → `git push` → server `git pull` (branch `pour`).
- **학습**: SSH + conda(miniforge3 `proj-hdgp-py311`) →
  `CUDA_VISIBLE_DEVICES=N NOTE="" ./train.sh <task> <label> --num_envs 2048 --headless` (setsid 백그라운드).
- **모니터링**: 로컬서 `sync_server.sh`(60s rsync) → `watch_pour.sh`.

---

## 코드 수정 규칙

```
1. 한 번에 하나의 가설만 검증 (사용자가 통합 명시 시 예외).
2. obs/action 차원 변경 금지 (명시 요청 없이).
3. 관련 없는 파일 수정 금지 — 요청된 프로젝트 폴더만.
4. 새 reward term 전에 기존 weight 조정으로 해결 가능한지 먼저 확인.
5. 모든 reward/gate/weight 변경 전 reward-audit 통과 (~/.claude/skills/reward-audit/).
6. 변경 후 예상 지표 이동 방향 명시.
```

---

## 기록 규칙

- **학습 전**: `record_test_snapshot.py` (train.sh가 자동 기록).
- **분석 후**: `analysis.md` 누적.
- **주제 완료 시 Notion**:
  `notion_log.py --target <grasping|pouring> --topic "..." --content-file <file>`
  6단계 구조: **질문 → 문제 → 원인 → 변경 → 근거 → 결과**.

---

## 도구

| 도구 | 위치 | 용도 |
|-----|------|------|
| `parse_tfevents.py` | `scripts/tools/` | TFEvents 바이너리 파싱 (tensorflow 불필요) |
| `monitor_pour.py` / `watch_pour.sh` | `scripts/tools/` | 여러 실험 통합 비교표 |
| `sync_server.sh` | `scripts/tools/` | server TFEvents → 로컬 mirror (60s) |
| `notion_log.py` | `scripts/tools/` | Notion 연구일지 자동 기록 |
| `record_test_snapshot.py` | `scripts/tools/` | 학습 전 git diff + 파라미터 스냅샷 |
| `openarm_fk.py` | `scripts/tools/` | OpenArm FK (numpy only) |
| `train.sh` | 루트 | 스냅샷 자동 기록 + 학습 실행 |

---

## 에이전트

에이전트 세부 역할은 `AGENTS.md` 참조.

| 에이전트 | 역할 | 코드 수정 |
|---------|------|----------|
| `rl-training-analyzer` | TFEvents 분석 + bottleneck 특정 | ❌ (analysis.md만) |
| `rl-pour-modifier` | reward-audit 통과 후 외과적 수정 | ✅ |
| 스킬: `reward-audit` | 수정 전 위험도 평가 | — |
