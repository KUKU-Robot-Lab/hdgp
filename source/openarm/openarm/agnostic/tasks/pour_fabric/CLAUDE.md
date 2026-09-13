# pour_fabric — 태스크 규칙 (09.13 재작성: text2reward 트랙)

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) (로그 먼저 · 증거 우선순위 · 분석 원칙)

## 태스크 정체성

**양팔 잡기 → 들기 → 붓기.** 테이블 위 컵 2개(소스=우, 비드 20개 / 리시버=좌, 빈 컵)에서
시작한다. 팔은 Fabrics(palm 6D = 시작자세 앵커 + 비대칭 델타), 손은 grasp_s2r 의 관절공간
시너지(접촉 동결)를 양팔로 복제했다. 액션 42 = (palm 6 + 손 15) × 2. obs 286 / critic 316.

**보상은 이 트랙에 없다.** `cfg.reward_code_path` 의 생성 코드가 `RewardContext`
(`modules/t2r/context.py`)를 읽어 `(reward (N,), {항: (N,)})` 를 돌려준다. 성공 판정
(fill ≥ 0.5 ∧ spill ≤ 0.4 ∧ 컵 xy < 0.2)은 env 가 계산해 `ctx.success` 로 넘긴다 —
보상이 바꿀 수 없는 **기준 지표**다(`task/success_now`, `task/episode_success`).

## 보상 생성 루프 (`scripts/reward_gen/t2r.py`)

```
render  → reward_gen/<track>/iter_NN/prompt.md      (RewardContext 스텁을 소스에서 렌더)
(LLM)   → response.md                                (프롬프트 파일만 받는 새 세션/에이전트)
ingest  → compute_reward.py + validation.json        (정적 검사 + 가짜 ctx 드라이런)
(audit 없음 — 09.13 사용자 결정: 생성 보상의 평가는 학습 지표뿐, 세션 의견 주입 금지)
train   → env.reward_code_path=<abs path>            (서버, git push→pull)
reflect → feedback.md + 다음 iter prompt.md          (TFEvents reward/* · task/* 요약)
```

- 생성기에는 **프롬프트 파일만** 준다. 저장소를 읽은 세션이 쓰면 "프롬프트만으로 생성"이 아니다.
- `reward/<항>` 태그는 생성 코드의 dict 키가 그대로 탄다 — 항 이름을 바꾸면 피드백 표도 바뀐다.
- rl_games `reward_shaper.scale_value=0.01` 이 총보상에 곱해진다. 이 스케일 뒤의
  값이 `score_to_win`·value 정규화에 들어간다.

## 검증 게이트 (학습 전)

1. `probe_pour_fabric_boot.py` 무작위 300스텝: NaN 0 · runaway 0 · hold 종료 in_source ≥ 0.9.
2. 같은 프로브 `--script pour`: 접근→닫기→들기→기울이기 스크립트로 **제어 파이프라인이
   물리적으로 과제를 수행할 수 있는지**(파지 게이트·리프트·tilt 도달). 안 되면 보상 탐색은 무의미.
3. `ingest` 검증 PASS (기계적 검증만 — 보상 품질 판단은 루프의 지표 피드백이 한다).

## 알려진 함정

- **a=0 = 앵커.** 델타 박스가 비대칭이라 선형 매핑(0.5(a+1)(hi−lo)+lo)을 쓰면 a=0 이 박스
  중점이 된다(09.13 부팅 실측). `side_rig.compose_palm_target` 은 부호별 매핑이다.
- `openarm/tasks/__init__.py` 는 등록 모듈의 ImportError 를 조용히 삼킨다 — env 가 안 뜨면
  "unknown task" 가 아니라 import 오류일 수 있다. 프로브는 config 를 명시 import 한다.
- 닫기 게이트는 palm↔컵 거리(0.22 m 램프 0.3). 시작 자세 palm↔컵 ≈ 0.16 m.
- palm 박스는 양팔 모두 **미검증**(`palm_box_verified=False`) — probe 로 승격할 것.
