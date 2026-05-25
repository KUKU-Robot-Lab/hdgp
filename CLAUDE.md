# hdgp — Claude 행동 규칙

## 프로젝트 성격

Isaac Lab / Isaac Sim 기반 RL 실험 저장소. OpenArm(7 DOF) + Teosllo(20 DOF) 로봇 조작.

**이 저장소는 일반 소프트웨어 개발이 아니라 실험 루프다.**
코드 품질보다 실험 가설 검증이 우선이다.

주요 태스크:
- `5g_pour_right_v3` — 안정적 학습 확인됨 (bead_in_target 달성)
- `5g_pour_right_v5` — 진행 중 (tilt local minimum 해결 중)

---

## 핵심 규칙

**로그를 먼저 읽기 전까지 코드를 수정하지 않는다.**

올바른 루프:

```
1. TFEvents + 체크포인트에서 수치 근거 추출
2. 이전 test와 지표 비교
3. 하나의 primary bottleneck만 특정
4. 그 bottleneck에 대한 하나의 최소 수정 제안
5. reward-audit 통과 확인
6. 수정 실행
7. test_history.md 기록 (record_test_snapshot.py)
8. analysis.md 업데이트
```

---

## 증거 우선순위

반드시 이 순서로 판단한다. 하위 항목이 상위 항목을 뒤집을 수 없다.

```
1순위: TFEvents 스칼라 (parse_tfevents.py로 파싱)
2순위: play.py 렌더링 / 사용자의 직접 관찰 보고
3순위: analysis.md 이전 분석 이력
4순위: test_history.md 코드 변경 이력 (git diff 기반)
5순위: 소스 코드 직접 읽기
6순위: 일반 RL 직관

TFEvents 수치와 모순되는 일반 RL 조언은 무시한다.
```

---

## 5g_pour_right 태스크 필수 점검 지표

분석 시 반드시 확인해야 할 TFEvents 태그:

**성공 지표 (가장 중요)**
- `Episode/log/bead_in_target` — 실제 pour 성공. 0이면 pour 없음.
- `Episode/log/bead_cross` — 경계 통과 순간 신호
- `Episode/log/bead_in_source` — 소스컵 유지율

**Stage 진행 지표**
- `Episode/log/pour_gate` — pour stage 활성화율 (0→1)
- `Episode/log/rho` — 근거리 조건 달성율 (cup 접근)
- `Episode/log/contact_gate` — 접촉 게이트
- `Episode/log/directional_tilt_cos` — 컵 기울기 방향 (cos)

**위치 지표**
- `Episode/log/cup_center_xy_dist` — 두 컵 중심 XY 거리
- `Episode/log/mouth_xy_dist` — 컵 입구 XY 거리

**비용/패널티 지표**
- `Episode/log/spill_ratio` — 흘린 비율
- `Episode/cost/cup_collision` — 컵 충돌 패널티
- `Episode/cost/grasp_loss` — 파지 손실
- `Episode/cost/premature_tilt` — 조기 기울기 패널티

**학습 안정성**
- `info/kl` — KL divergence (>0.2이면 불안정)
- `Episode/reward/pour_tilt` — tilt 보상 실제 수치
- `Episode/reward/bead_progressive` — 비드 이동 보상

**BC 시스템 (v5)**
- `bc/loss_demo` — demo BC 손실 (0이면 demo 미로드)
- `bc/loss_sim` — sim BC 손실 (0이면 성공 에피소드 없음)

---

## 코드 수정 규칙

```
1. 한 번에 하나의 가설만 검증한다.
   - 여러 파라미터를 동시에 바꾸면 어떤 게 효과인지 알 수 없다.

2. 관측(obs) 차원과 액션 차원은 명시적 요청 없이 변경 금지.
   - NUM_OBSERVATIONS, NUM_ACTIONS 변경은 학습 불연속 발생.

3. 관련 없는 파일 수정 금지.
   - 요청된 태스크 폴더 파일만 수정.

4. reward 항목 추가 전에 기존 weight 조정으로 해결 가능한지 먼저 확인.
   - 새 reward term은 복잡성을 높인다.

5. 모든 reward/gate/weight 변경 전 reward-audit 체크리스트 통과 필요.
   - 체크리스트: ~/.claude/skills/reward-audit/SKILL.md

6. 변경 후 예상 지표 이동 방향을 명시한다.
   - "weight_tilt 올리면 directional_tilt_cos < 0으로 이동 예상"
```

---

## 수정 기록 규칙

```
수정 전:
  python3 scripts/tools/record_test_snapshot.py --task <task> --test <next_test>

수정 후 분석:
  python3 scripts/tools/parse_tfevents.py <events_file> --tags "Episode/" --epochs 50
  → 결과를 analysis.md에 누적 기록
```

---

## 도구

| 도구 | 위치 | 용도 |
|-----|------|------|
| `parse_tfevents.py` | `scripts/tools/` | TFEvents 바이너리 파싱 (tensorflow 불필요) |
| `record_test_snapshot.py` | `scripts/tools/` | 학습 전 git diff + 파라미터 스냅샷 기록 |
| `train.sh` | 루트 | 스냅샷 자동 기록 + 학습 실행 |

---

## 에이전트

에이전트 세부 역할은 `AGENTS.md` 참조.

| 에이전트 | 역할 | 코드 수정 여부 |
|---------|------|------------|
| `rl-training-analyzer` | TFEvents 분석 + bottleneck 특정 | ❌ (analysis.md만 기록) |
| `rl-pour-modifier` | reward-audit 통과 후 외과적 수정 | ✅ |
| 스킬: `rl-pour-diagnostics` | pour 도메인 지식 참조 | — |
| 스킬: `reward-audit` | 수정 전 위험도 평가 | — |
