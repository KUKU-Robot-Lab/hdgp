# IKER 1단계 파지 — t2r 보상 생성 루프 설계

> 기준: `2026-09-14-iker-auto-loop-design.md`(이하 "루프 스펙"), `2026-09-14-iker-learned-grasp-left-arm-design.md`(이하 "파지 스펙").
> 이 문서는 1단계 파지의 **보상 출처**와 루프의 1단계 단계를 개정한다. 수확 이후(수확·VLM 목표·2단계·재질의·완수)와
> 1단계 env 의 관측·행동·종료·잡힘/성공 판정·성공 캡처는 기존 스펙 그대로다.
> 상태: 사용자 결정(2026-09-15) — 설계 섹션 1~4 승인.

## 1. 왜 바꾸나

손설계 1단계 보상은 파지 스펙 §16 개정 3-1~3-6(트랙 r2~r8)으로 한 번에 한 조건씩 고쳐 왔다. r8 은 epoch 550 게이트를
통과했지만(래치 2.66 %) 성공은 0 이었다. r8 ep550 결정론 롤아웃(512 env, 첫 에피소드, `log/iker_diag/r8_ep550_after_latch.log`):

| 측정 | 값 |
|---|---|
| 래치한 에피소드 | 40/512 (7.8 %) — 40 개 전부 놓침 종료, 성공 0 |
| 래치 뒤 최대 연속 잡힘 | q10/50/90 3/3/5 스텝 (20 도달 0, 11 이상 1) |
| 래치 → 에피소드 끝 | q50 4 스텝 |
| 래치 뒤 잡힘이 끊긴 조건(전체/그것만) | 미끄럼 ≥ 0.05 82.4 %/38.4 % · 5 cm 아래 30.8 % · 시작 이탈 27.0 % · 엄지 18.2 % · 15 cm 초과 16.4 % |
| 래치 뒤 신발 속도 | q50 0.23 m/s |

IKER 원형(기준 스펙 §2)은 보상이 고정 5항이고 VLM 은 목표 코드만 쓰며 파지는 학습하지 않는다 — 파지 보상 생성은 IKER
범위 밖이다.

사용자 결정(2026-09-15):
1. 파지 부분만 **t2r(text2reward) 구조**로 보상을 생성·반복 학습하고, 성공 파지를 IKER(수확 이후)로 넘긴다.
2. 생성기는 **보상만** 쓴다. 성공 판정(파지 스펙 §16 의 잡힘·래치·성공)은 넘김 계약으로 고정한다.
3. 첫 생성은 **완전 zero-shot** — 과제문·성공 판정·env API 만 준다. 손설계 보상·개정 이력·r2~r8 진단은 넣지 않는다.
4. 접근안 A — IKER 자동 루프 안의 `stage1_t2r` 단계(별도 루프·수동 넘김이 아니다).
5. 라운드마다 새로 학습(이어학습 아님), 라운드 500 epoch·조기 종료 250·수확 시도 성공 ≥ 5 %·최대 6 라운드.
6. 보상 전용 접촉 센서를 넣는다(관측 불변).
7. `stage1_a`·`calibrate`·`stage1_b` 단계와 그 실행기·테스트를 뺀다.

t2r 공통 규칙(기존 사용자 결정): reward-audit 을 쓰지 않는다 · 생성기 notes 는 측정·영상으로 확인한 사실만 · 트랙끼리 코드를
공유하지 않는다(복사 포크) · 보상 루프는 PPO 4096 env · obs/actor 는 sim2real 출처가 있는 입력만(보상 전용 특권 정보는 허용).

## 2. 구조

```
stage1_t2r ──(commit_bank)──▶ vlm_target ─▶ stage2_train ─▶ observe_requery ─▶ completion_review ─▶ done
   │  라운드 iter_NN (최대 6)
   │  write_t2r_prompt → t2r_generate → ingest_reward → run_t2r_smoke → launch_t2r → 판정
   └─ 판정: 성공 ≥ 5 % 체크포인트 → run_harvest → commit_bank / record_harvest_miss
            500 epoch·조기 종료 → end_round → 다음 iter
```

- `PHASES = ("stage1_t2r", "vlm_target", "stage2_train", "observe_requery", "completion_review", "done")` — 수확은 별도 단계가
  아니라 `stage1_t2r` 안에서 한다(수확 판정 규칙·`HARVEST` 출력·뱅크 커밋은 기존 수확 단계와 같다).
- `TRAINING_RUNS = ("stage1_t2r", "stage2")`, `SIDE_RUNS` 에 `t2r_smoke` 추가, `calibrate` 제거.
- `PHASE_RUNS["stage1_t2r"] = ("stage1_t2r", "t2r_smoke", "harvest")`.
- env 원칙: 생성 코드는 **1단계 보상 합만** 바꾼다. 부모 env 의 `stage1_step` 은 그대로 돌아 잡힘·래치·성공·놓침·종료·
  성공 캡처·wrench DR(래치 기준)을 계산한다. 그래서 수확·2단계는 바뀌지 않는다.

## 3. stage1_t2r 라운드

### 3.1 상태

`state["t2r"] = {"iter": int, "requests": int, "rounds": [RoundRecord, ...]}` — `RoundRecord` =
`{"iter", "label", "reward_sha256", "end_epoch", "success_max", "latched_max", "harvest_tried": [checkpoint...], "ended": "handover|round_epochs|early|crash"}`.
라운드 산출물: `iker_runs/shoe_place/config_00/loop/stage1_t2r/iter_NN/{prompt.md, response.md, response_attempt_K.md, compute_reward.py, validation.json, generator.json, smoke.json, feedback.md}`.

### 3.2 정책 키

| 키 | 값 | 뜻 |
|---|---|---|
| `t2r_round_epochs` | 500 | 라운드 학습 최대 epoch(`--max_iterations`), ≈ 3 h(22 s/epoch) |
| `t2r_early_epoch` | 250 | 조기 종료 판정 epoch |
| `t2r_early_latched` | 0.005 | 조기 종료: 이 epoch 에서 성공 bin 0 이고 래치 bin 이 이 값 미만 |
| `t2r_harvest_success` | 0.05 | 체크포인트 성공 bin(`bin_epochs` 10) 이 이 값 이상이면 수확 시도 |
| `t2r_max_rounds` | 6 | 넘김 없이 이 라운드 수를 채우면 pause |
| `t2r_max_requests` | 3 | 한 iter 의 생성 요청(첫 응답 + 재생성 2) |
| `t2r_num_envs` | 4096 | PPO 학습 env |
| `t2r_smoke_envs` / `t2r_smoke_steps` | 64 / 150 | 라운드 스모크 |

`harvest_min` 64·`side_gpu_limit_mib`·`bin_epochs` 등 기존 키는 유지한다. `gate_epoch`·`gate_latched`·`calibrate_latched`·
`b_g_min`·`b_max_epochs` 는 뺀다. `labels["stage1_t2r"]` 는 라벨 접두사(`iker_grasp_c00_t2r`)이고 실제 라벨은 `<접두사>_iNN`.

### 3.3 판정 순서 (`_stage1_t2r(state, probe)`)

1. 현재 iter 의 `prompt.md` 가 없다 → `write_t2r_prompt`(iter 0: render / iter > 0: 이전 iter 의 코드와 TFEvents 로 reflect).
2. `response.md` 가 없다 → `t2r_generate`(요청 수 < `t2r_max_requests`), 넘으면 `pause`.
3. `validation.json` 이 없다 → `ingest_reward`. `ok=false` 이면 응답을 `response_attempt_K.md` 로 옮기고 요청 수가 남았으면
   `t2r_generate`(같은 프롬프트, 오류 문구를 붙이지 않는다), 없으면 `pause`(오류 목록).
4. `smoke.json` 이 없다 → `run_t2r_smoke`(부수 런, GPU 한도 규칙). 실패 → `pause`.
5. 학습 런 기록이 없다 → `launch_t2r`(새로 학습, 라벨 `<접두사>_iNN`, `env.reward_code_path=<절대경로>`,
   `--max_iterations t2r_round_epochs`, `--num_envs t2r_num_envs`).
6. 수확: 수확 부수 런이 돌고 있으면 `wait`. 끝났으면 기존 수확 판정 — 통과 + 검증 ≥ `harvest_min` + 뱅크가 그 체크포인트의
   학습 뱅크 → `commit_bank`(학습 런 PID 종료, `ended=handover`, 단계를 `vlm_target` 으로); 그 밖 → `record_harvest_miss`
   (`harvest_tried` 에 추가) 후 계속. 수확이 돌지 않을 때, 아직 시도하지 않은 가장 최근 체크포인트의 성공 bin ≥
   `t2r_harvest_success` → `run_harvest`(학습은 계속 돈다).
7. 라운드 끝: 수확 부수 런이 없고, 런이 끝났다(`t2r_round_epochs`) 또는 epoch ≥ `t2r_early_epoch` 에서 성공 bin 0 이고 래치 bin
   < `t2r_early_latched` → `end_round`(런이 살아 있으면 PID 종료, `RoundRecord` 기록). 끝난 라운드 수 < `t2r_max_rounds` 면
   `iter += 1`, 아니면 `pause`("6 rounds without a handover"). 끝난 런의 마지막 체크포인트가 수확 조건을 만족하고 아직
   시도하지 않았으면 라운드를 끝내기 전에 먼저 수확한다(6 번).
8. 크래시·조기 사망 → 기존 규칙대로 `pause`(로그 끝 줄).

판정 지표는 고정 판정의 TFEvents 태그다: 성공 `Episode/grasp_episode/success`, 래치 `Episode/grasp_episode/latched`.

### 3.4 생성기 격리

`.claude/agents/iker-t2r-generator.md` — 도구 Read·Write, 브리프가 이름 붙인 `prompt.md` 하나만 읽고 `response.md` 하나만 쓴다
(`iker-vlm-generator` 와 같은 격리 문구, 영상 없음). `loop.py status` 가 `t2r_generate` 에 브리프를 붙이고, LOOP_PROMPT 는
`vlm_generate` 와 같은 절차로 에이전트를 띄운다(브리프 그대로, 설명·힌트·이력 덧붙이지 않음). `generator.json` 에 요청 기록.

## 4. 파일 배치

| 파일 | 역할 |
|---|---|
| `tasks/iker_shoe/t2r/context.py` | `RewardContext`(§5), `context_stub_source()` |
| `tasks/iker_shoe/t2r/loader.py` | `load_reward_fn`, `call_reward_fn`, `zero_reward` (grasp_fj_t2r 포크에서 복사) |
| `tasks/iker_shoe/t2r/validator.py` | `static_check`, `make_fake_context`, `dry_run`, `validate`, `cuda_usable` (복사·필드 적응) |
| `tasks/iker_shoe/t2r/prompts.py` | `PromptSpec`, `render_prompt`, `render_feedback_table`, `FEEDBACK_TAG_PREFIXES` |
| `tasks/iker_shoe/iker_shoe_grasp_t2r_env.py` (+cfg·등록) | 보상 훅·접촉 센서·로그(§7), gym id `open-sens_l_iker_shoe_grasp_t2r`(+Play) |
| `scripts/iker/t2r_smoke.py` | `--mode wire` / `--mode round`(§8) |
| `scripts/iker/t2r_reward.py` | `render` / `ingest` / `reflect` CLI(루프 실행기가 호출, Isaac python 으로 ingest) |
| `.claude/agents/iker-t2r-generator.md` | 격리 생성기 |
| `modules/iker/loop_state.py` · `loop_probe.py` · `scripts/iker/loop.py` · `scripts/iker/LOOP_PROMPT.md` | §3 단계·실행기·틱 절차 |
| `scripts/iker/harvest_grasp_bank.py` · `tasks/iker_shoe/grasp_bank.py` | 뱅크 메타데이터(§10) |

바꾸지 않는 것: `grasp_stage.py` 판정, 부모 `iker_shoe_grasp_env.py`, 2단계 env, 붓기 `modules/t2r`·`scripts/reward_gen/t2r*.py`,
`tasks/grasp_fj_t2r/`. `scripts/iker/measure_grasp_quality.py` 파일은 남긴다(루프에서 호출하지 않는다).

## 5. RewardContext

frozen dataclass, 모든 텐서는 env-local·복사본(`clone`), 각도 rad·힘 N·거리 m. P = 신발 표면 부분표본 점 수.

| 묶음 | 필드 |
|---|---|
| 상수(python) | `table_top_z`, `rack_x_range`, `rack_y_range`, `episode_steps`(120), `control_dt`(0.1), `lift_height`, `lift_max`, `hold_xy_radius`, `hold_radius`, `hold_slip_speed`, `thumb_curl_min`, `success_steps`, `success_speed` — 판정 상수는 `Stage1RewardCfg` 에서 읽는다 |
| 손 | `palm_pos (N,3)`, `palm_quat (N,4 wxyz)`, `palm_normal (N,3)`, `link_pos (N,5,3,3)`(엄지·검지·중지·약지·새끼 × `_3`·`_4`·`tip`), `hand_q`, `hand_qd`, `hand_q_norm`, `hand_target_norm (N,20)`, `thumb_curl (N,)`, `hand_z_min (N,)` |
| 팔 | `arm_q`, `arm_qd (N,7)` |
| 신발 | `shoe_pos`, `shoe_quat`, `shoe_lin_vel`, `shoe_ang_vel`, `shoe_start_xy (N,2)`, `shoe_surface (N,P,3)`, `dz_free (N,)`(받침 영역과 겹치면 0 — 필드 정의), `shoe_shift_xy (N,)`, `palm_gap (N,)`(손바닥–표면 최근접), `palm_shoe_dist (N,)`, `slip_speed (N,)`(손바닥 기준 미끄럼), `link_shoe_gap (N,5,3)` |
| 접촉(보상 전용) | `link_shoe_force (N,5,3)`, `palm_shoe_force (N,)` |
| 상태(고정 판정) | `held`, `hold_count`, `latched`, `success (N,)`, `episode_progress (N,)`(0~1) |
| 행동 | `actions`, `prev_actions (N,26)`(리셋 직후 0) |

넣지 않는 것: 파지 품질 q·w_f(손설계 계측), 키포인트·2단계 목표, 손설계 보상 항 값.

## 6. 프롬프트

grasp_fj_t2r 포크와 같은 절 순서:
1. 역할 문장.
2. 로봇 설명 — 왼팔 OpenArm 7관절을 손바닥 상대 6D IK 로 조종(위치·회전 배율은 cfg 에서), DG-5F 20관절 행동 범위 표(부팅 값과
   같은 profile 순서, `thumb_2`·`pinky_2` 고정), 손 목표 EMA, 10 Hz.
3. 보상 구조 일반 안내(포크의 5부분 문장 그대로 — 설계 조언 아님).
4. `RewardContext` 원문(`context_stub_source()`).
5. 추가 사실(코드로 확인한 env 사실만): 배치·마스크 규칙, 좌표·단위, 잡힘·래치·성공 판정 식(상수는 cfg 값으로 렌더), 종료
   (성공·낙하·래치 뒤 놓침·120 스텝 시간 끝), ctx 읽기 전용, 항 이름이 `t2r_reward/<항>` 로 기록됨.
6. 과제문: "Grasp the shoe with the left hand and lift it 5–15 cm; hold it steady near where it was picked up." + 출력 규칙(시그니처,
   코드 블록 하나).
7. iter ≥ 1: 이전 코드 + 피드백 표. 사용자 notes 는 사람이 검증된 사실을 적었을 때만 붙인다(루프가 자동으로 만들지 않는다).

피드백 표(`render_feedback_table`, 곡선마다 균등 간격 10 점 + min·mean·max): 접두사 `t2r_reward/`, `grasp_episode/success`,
`grasp_episode/latched`, `grasp/held_frac`, `grasp/dz_free`, `grasp/shift_xy`, `grasp/thumb_curl`, `grasp/rel_speed`, `grasp/lost_frac`,
`grasp/over_rack_raised_frac`, `grasp/arm_speed_sum`, `grasp/hand_command_rate`, `contact/`, `episode_lengths`, `rewards`
(TFEvents 의 `Episode/` 접두·`/iter` 접미는 떼고 읽는다).

## 7. env 훅 (`IkerShoeGraspT2rEnv(IkerShoeGraspEnv)`)

- `__init__`: `super().__init__` 전에 `load_reward_fn(cfg.reward_code_path)`(빈 경로 = 영 보상, 없는 파일 = 부팅 오류).
  부모의 무행동 수입 부팅 검사는 손설계 보상 감사용이라 t2r env 는 거치지 않는다(부팅 줄에 코드 경로·sha256 을 찍는다).
- `_setup_scene`: 부모 장면 + 손가락 링크 15 개·손바닥 1 개에 **몸체마다** `ContactSensor`(묶으면 `force_matrix_w` 가 조용히 0),
  필터 경로는 cfg 문자열이 아니라 스테이지에서 신발의 `RigidBodyAPI` 프림을 찾아 만든다(정확히 1 개가 아니면 부팅 오류).
- 보상: 부모 `_get_dones` 가 `stage1_step` 을 계산한 **같은 스텝 상태**로 `_build_context` → `call_reward_fn` →
  `nan_to_num` → `_get_rewards` 가 그 합을 돌려준다. ctx 텐서는 전부 `clone`.
- 로그: 생성 항은 `t2r_reward/<항>`·`t2r_reward/total`, 부모의 손설계 `grasp_reward/*` 키는 로그에서 뺀다(혼동 방지),
  `grasp/*` 진단은 그대로. 접촉 진단 `contact/links_touching`·`contact/fingers_touching`·`contact/palm_touching`(0.1 N 기준, 로그 전용).
- `_reset_idx`: 부모 리셋 뒤 `prev_actions` 행을 0 으로.
- 관측·행동·종료·성공 캡처는 부모 그대로(오버라이드하지 않는다).

## 8. 검증·스모크

- `ingest_reward`(Isaac python): 응답의 마지막 python 코드 블록 → `static_check`(최상위 `compute_reward(ctx)` 하나, import
  `torch·math·typing` 만, `exec·eval·open·__import__·compile·getattr·setattr·delattr·globals·locals·vars` 금지, ctx 필드 대입·
  제자리 연산·첨자 대입 금지, 없는 ctx 필드 오류) → `dry_run` 3 seed × cpu·cuda(`make_fake_context`, 형상은 §5) → 합·항 유한 →
  `validation.json {ok, errors, warnings, term_stats, total_stats, fields_used}`.
- `t2r_smoke.py --mode wire`(구현 수용 검사, 1 회): 사전 파지 뱅크 자세에서 손을 쥔 자세로 닫아 엄지와 다른 손가락 ≥ 1 개의
  접촉력 > 0.1 N · 신발을 멀리 옮긴 env 는 모든 행 0 · 손가락별 행 대응 · 프롬프트 관절표 = 부팅 측정(±2e-3) · `GATE PASS/FAIL`.
- `t2r_smoke.py --mode round`(라운드마다, 그 iter 의 보상): 64 env · 150 스텝(무작위 + 무행동) · NaN 0 · `t2r_reward/total` 기록 ·
  리셋 뒤 `prev_actions` 0 · 부모 판정 값(held·latched·success·lost) 계산됨 · 무행동 보상 평균은 **보고만**(게이트 아님) →
  `smoke.json {passed, checks, idle_reward_mean}`.

## 9. 실패 처리

| 상황 | 동작 |
|---|---|
| 생성 요청 3 번에도 `response.md` 없음 | `pause` |
| ingest 실패 3 번 | `pause`(마지막 오류 목록) |
| 라운드 스모크 실패 | `pause`(`smoke.json` 실패 검사) |
| 학습 크래시·조기 사망 | `pause`(로그 끝) |
| 판정 태그 없음(TFEvents 에 success·latched 없음) | `pause` |
| 수확 통과했지만 검증 < `harvest_min` | `record_harvest_miss`, 다음 체크포인트 |
| 6 라운드 넘김 없음 | `pause` |
| 부수 런 중 GPU > `side_gpu_limit_mib` | `wait` |

## 10. 수확·뱅크 메타데이터

`harvest_grasp_bank.py` 는 `--task` 로 t2r gym id 를 받아 같은 성공 캡처·복원 검증을 한다(`--g-min` 인자는 뺀다). 뱅크
메타데이터의 `stage1_reward` 는 `{"source": "t2r", "iter": NN, "reward_code_path": ..., "reward_code_sha256": ...}` 로 바꾸고,
`LEARNED_BOOT_KEYS` 대조는 그대로다. 2단계 env 의 뱅크 부팅 대조는 바뀌지 않는다.

## 11. 테스트

순수(테스트 먼저):
- t2r 포크: 컨텍스트 원문 = 프롬프트 env 설명 · 가짜 ctx 형상 = §5 · 좋은 코드 통과 · 없는 필드·import·ctx 변경·제자리 연산·
  cuda device 불일치 거부 · 로컬 텐서 제자리 연산 허용 · loader 반환 형상 · 빈 경로 영 보상 · 프롬프트 판정 수치 = `Stage1RewardCfg`
  (리터럴 아님) · 관절표 = profile 순서 · 피드백 표 렌더(접두사·10 점).
- env 계약(AST, `source_contract`): 오버라이드는 `__init__`·`_setup_scene`·`_get_rewards`·`_get_dones`(로그 교체만)·`_reset_idx`·
  `_build_context`·`_shoe_contact_filter`·`_link_shoe_forces` 뿐 · ctx 는 같은 스텝 상태 · 텐서 복사 · 몸체마다 센서 ·
  필터는 스테이지에서 · `reward_code_path` 기본 "".
- `loop_state.decide`(가짜 Probe): §3.3 전이 전부(프롬프트 → 생성 → ingest → 재생성 한도 → 스모크 → 기동 → 대기 → 성공 ≥ 5 %
  수확 → 통과 뱅크 → vlm_target / 미달 계속 · 500 끝 · 250 조기 끝 · 6 라운드 pause · 크래시 · 태그 없음).
- `loop.py` 실행기: 기동 인자(새 학습·보상 경로·max_iterations·num_envs) · reflect 가 `feedback.md` 와 다음 `prompt.md` 를 씀 ·
  `end_round` 기록 · 수확 메타데이터(iter·sha256).
- 뱅크 메타데이터 테스트 갱신(`test_grasp_bank*.py`).

Isaac: `t2r_smoke.py --mode wire` GATE PASS, `--mode round` 를 테스트 전용 고정 보상(fixture)으로 PASS, 기존 `grasp_smoke.py` 통과.

## 12. 전환 절차

1. 이 스펙·구현 계획 승인 → 구현·리뷰(SDD).
2. r8 A(`iker_grasp_c00_r8_a`) PID 종료, r8 루프 상태를 `config_00/archive/2026-09-15_r8_hand_reward_no_hold/loop/` 로 `git mv`.
3. `loop.py init --track iker_shoe_c00_t2r --policy '{...}'`(adopt 없음 — 첫 틱이 `write_t2r_prompt`).
4. LOOP_PROMPT 크론 트랙 이름 변경, 크론 재생성(세션 한정·7 일 만료).

## 13. 위험·범위 밖

- zero-shot·audit 없음이라 첫 보상이 무행동에도 수입을 줄 수 있다 — 스모크가 무행동 보상 평균을 보고하고 피드백 표가
  보여준다(설계상 게이트로 막지 않는다).
- 접촉 필터 함정(스테이지 경로·몸체별 센서) — wire 스모크가 잡는다.
- 라운드마다 새로 학습해 라운드당 ≈ 3 h, 6 라운드 최대 ≈ 18 h.
- 범위 밖: 2단계 보상(IKER 고정 5항)·VLM 목표, 손설계 `grasp_stage` 보상 항 코드 삭제(부모 env 에서 계속 계산만 된다),
  서버 학습, SAPG.

## 14. 구현 계획 판정 (2026-09-15, `docs/superpowers/plans/2026-09-15-iker-stage1-t2r.md`)

구현 계획을 쓰며 위 절을 이렇게 구체화한다(설계 결정은 그대로):
- §5 상수: 받침 범위는 스칼라 `rack_x_min`·`rack_x_max`·`rack_y_min`·`rack_y_max`(검증기의 스칼라 필드는 float·int), 래치 연속 수
  `latch_steps` 를 더한다(추가 사실 6 의 래치 정의가 읽는다).
- §7 훅: 생성 보상·로그 교체는 `_get_rewards` 에서 한다(`DirectRLEnv.step` 은 `_get_dones` → `_get_rewards` → 리셋 순이라 같은 스텝
  상태다). `_get_dones` 는 덮지 않는다. 부모 `__init__` 의 무행동 부팅 검사는 부모 코드 그대로 돌고(기본 cfg 에서 통과), t2r env 는
  자기 부팅 줄(코드 경로·sha256·센서 수·필터)을 찍는다.
- §3.3 3 번: ingest 실패 때 `response.md`·`validation.json`·`compute_reward.py` 를 `*_attempt_K` 로 옮긴다 — 다음 판정이 파일 유무만
  보고 재생성한다.
- §4 파일: 라운드 파일 단계(render·ingest·reflect)는 순수 모듈 `tasks/iker_shoe/t2r/pipeline.py`, 루프 쪽 생성기 브리프·기록·이름은
  `modules/iker/loop_t2r.py`(torch 없음).
- §10 수확: 부모 1단계 태스크(`open-sens_l_iker_shoe_grasp`)로 돈다 — 관측·행동·종료·성공 캡처가 t2r env 와 같고 접촉 센서는 보상
  전용이다. `--g-min` 을 빼고(g_min < 1 은 보정 파일을 요구한다) `--t2r-iter`·`--reward-code` 로 `stage1_reward` 메타데이터를 쓴다.
- 프롬프트 손 관절표는 관절 이름을 profile(`TESOLLO_LEFT_SHORT.hand_joint_names`)에서 읽고 범위는 순서대로 상수로 둔다(생산 코드에
  관절 이름 리터럴 금지).
