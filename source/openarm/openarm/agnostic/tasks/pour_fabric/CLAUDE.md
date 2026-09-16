# pour_fabric — 태스크 규칙 (09.13 재작성: text2reward 트랙)

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) (로그 먼저 · 증거 우선순위 · 분석 원칙)

## 태스크 정체성

**양팔 잡기 → 들기 → 붓기.** 테이블 위 컵 2개(소스=우, 비드 20개 / 리시버=좌, 빈 컵)에서
시작한다. 팔은 Fabrics(palm 6D = 시작자세 앵커 + 비대칭 델타), 손은 grasp_s2r 의 관절공간
시너지(접촉 동결)를 양팔로 복제했다. 액션 18 = (palm 6 + 손 3) × 2, obs 223 / critic 293
(09.15 grip3 + 09.16 채움 정도 1칸. 구 `hand_action_mode="synergy15"` 는 액션 42, obs 247 / critic 317 —
t2r_i05/i07 보관 체크포인트(obs 222)는 이 코드로 재생 불가, 커밋 1dc8e099 이전을 체크아웃).

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

## sim2real 관측·DR (09.14 라운드 3)

- actor 관측 246 = 팔당 99(팔 q/qd·손 q·palm·손끝·**지각된** 컵 pose 파생·관절오차·컵 up) ×2 + 공통 48.
  **hand_qd 는 actor 에 없다**(실기 DG-5F 드라이버 velocity ≠ 관절속도, 09.07 실측) — critic 만 316.
- 컵 pose 는 지각 링버퍼(지연 0→3 스텝, ADR)+코히런트 노이즈(xyz 0→1.5 cm, rot 0→3°)를 거친다(`_perceive`).
  actor 의 컵 파생 관측 전부(상대 pose·손끝 상대·up·컵 간·개구 간)가 **같은 지각 pose** 에서 나온다.
- 관절/FK 상시 노이즈 qpos 0.002 rad·qvel 0.05·body 5 mm.
- 물리 DR(EventTerm, mode=reset): 컵 질량 ×0.5~2.5·관절 게인 ×0.5~2.0 은 ADR 확장, 컵 마찰 0.7~1.2 는 고정
  (재질 term 은 런타임 확장이 무증상 no-op — grasp_s2r 주석). 들린 컵 외란 0→5 N/kg(`WrenchDR`).
- ADR 트리거 = 순간 성공률 ≥ 0.3, 30단계, 3000 스텝 간격(`modules/adr.TaskADR`). 로그 `adr/*`, `dr/*`.
- 충돌 신호: `ctx.cup_cup_force`(소스 컵 센서→리시버 필터), `ctx.*_hand_foreign_force`(net − 자기컵 필터),
  지표 `task/cup_collision_rate`·`task/*_hand_foreign_rate`. 벌점은 생성 보상이 넣는다(프롬프트 지식 11).

## 액션 처리·성공 판정 (09.15 라운드 7 — t2r_i05 궤적 계측 근거)

- **손 grip3**: 손당 3칸 = [엄지 대향(thumb ch1), 엄지 닫힘(thumb ch2), 4지 닫힘]. 4지 닫힘은 `_2`(ch1)·`_3`·`_4`(ch2)
  에 같은 값(`side_rig.expand_grip3`). i05 는 4지 ch1 을 내리고 ch2 를 올려 손끝으로 누르는 굴림(손바닥 접촉 7→2 N)을
  보였고 42 중 22 차원이 null 이었다. ★실기 정책 노드도 같은 3→15 매핑을 써야 한다.
- **palm EMA**: palm 6D 액션 y ← α·a + (1−α)·y, `palm_action_ema_alpha=0.25`. i05 소스 회전 액션이 매 스텝 부호 교대
  (81~84 %)해 palm 목표가 스텝당 68° 튀고 팔에 4~5 Hz 진동. actor 관측의 이전 palm 액션 칸은 **거른 값**(Markov),
  `ctx.actions` 는 거르기 전 원출력. ★실기 노드도 같은 α·같은 관측 규약.
- **리시버 직립 성공 조건**: 리시버 기울기 ≤ `success_rcv_tilt_max_deg=20°`. i05 는 붓는 동안 46°(최대 55°).
- i05 보관 체크포인트 재생: `env.hand_action_mode=synergy15 env.palm_action_ema_alpha=1.0`.

## 비드 부피 DR · 조준 전 틸트 래치 (09.16 라운드 9 — t2r_i07 영상·계측 근거, 사용자 결정 A)

- **왜**: i07 은 잡자마자 테이블 위에서 기울여(20° at step 77·입구 거리 0.30 m) 들면서 90° — 12 mm 비드 20개는 컵의
  3 % 라 흘리지 않았을 뿐(계측: 첫 이탈 20개 82°·60개 90~93°). 실현 가능한 개수로는 흘림 각도가 안 변한다(기하 76→73°).
- **비드 30 mm × 활성 개수 DR** (`pour_rules.py`, isaaclab 무관·로컬 테스트): 스폰 `bead_count`=20 고정, 리셋마다
  `bead_active_range` 안에서 활성 n 을 뽑고(상한은 ADR `beads/active_hi` 12→20) 비활성은 env 별 테이블 뒤 지면 격자에
  파킹(한 점에 모으면 브로드페이즈 폭발). `compute_bead_flags(active_mask=)` 가 비율·무게중심에서 파킹을 뺀다.
  질량은 밀도 유지(1 g@12 mm → 15.6 g). 프로브 실측: 26개는 림을 넘어 과적(in_source 0.81), 20개 정착 0.988(거의 가득).
  가득(20)일 때 첫 이탈 72°·20 % 80°·50 % 86°, 조금(6)일 때 첫 이탈 97°·50 % 100° — 좁은 컵에 큰 구라 낟알 걸림으로
  기하 예측(≈55°)보다 늦다. 부피 DR 은 흘림 각도를 72~97° 로 바꾸지만 "30° 에서 흘림" 은 못 만들며, 들면서 90° 로 가는
  행동만 가득 찬 에피소드에서 물리 벌(흘림 20~50 %)을 받는다. `bead_floor_tol_m` 3 mm: 바닥 비드 중심이 판정 경계에
  정확히 놓여 여유 없이는 0.3~0.6개/env 가 "컵 밖·흘림" 으로 흔들렸다(게이트 FAIL 원인).
- **채움 정도 obs**(actor·critic +1): hold 끝에 활성 비드 평균 높이 × 2 / 내부 높이(0~1, 에피소드 고정). 실기에서는 사람이
  어림잡아 넣는 명령 입력 — 센서가 아니라 sim2real 규칙과 충돌 없음. `ctx.bead_fill_level`.
- **래치**: 입구 xy 거리 > `premature_lip_xy_m`(0.10) 인 동안 소스 > `premature_tilt_max_deg`(30°) 면 리셋까지 성공 무효
  (`ctx.premature_tilt`, 지표 `task/premature_tilt_rate`·`task/tilt_far_deg`). 리시버 정지 대기는 `task/rcv_palm_speed` 로 본다.
- 프롬프트 지식 5항의 "110°" 는 12 mm·20개에서만 참이라 채움별 각도(2/3 → 65°, 거의 가득 → 40°)로 바꿨다.

## 알려진 함정

- **a=0 = 앵커.** 델타 박스가 비대칭이라 선형 매핑(0.5(a+1)(hi−lo)+lo)을 쓰면 a=0 이 박스
  중점이 된다(09.13 부팅 실측). `side_rig.compose_palm_target` 은 부호별 매핑이다.
- `openarm/tasks/__init__.py` 는 등록 모듈의 ImportError 를 조용히 삼킨다 — env 가 안 뜨면
  "unknown task" 가 아니라 import 오류일 수 있다. 프로브는 config 를 명시 import 한다.
- 닫기 게이트는 palm↔컵 거리(0.22 m 램프 0.3). 시작 자세 palm↔컵 ≈ 0.16 m.
- palm 박스는 양팔 모두 **미검증**(`palm_box_verified=False`) — probe 로 승격할 것.
