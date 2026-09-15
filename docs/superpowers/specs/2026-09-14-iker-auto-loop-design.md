# IKER 신발 놓기 — 자동 루프 설계 (계획 2c + 3 통합)

> 기준 스펙: `2026-09-13-iker-vlm-keypoint-reward-design.md`(이하 "기준 스펙"). 학습 파지 설계:
> `2026-09-14-iker-learned-grasp-left-arm-design.md`(이하 "파지 스펙", 개정 2 까지). 이 문서는 파지 스펙 §2 의 계획 2c
> ([2] 수확·[3] 2단계·[4] 평가)와, 기준 스펙·파지 스펙이 계획 3 으로 미룬 비전 재질의 루프를 **한 자동 루프**로 묶는다.
> 상태: 설계 섹션 1~4 사용자 승인(2026-09-14).

## 1. 왜

사용자 질문 "iker 은 자동화 프레임워크가 아니낙?"(2026-09-14). IKER(기준 스펙 §2)는 번호 찍힌 영상 → VLM 목표 코드 →
고정 보상 5항 RL → 새 관측으로 재질의(`done=True` 까지)를 사람 없이 돈다. 지금 저장소는 그렇지 않다.

- VLM 호출, 프롬프트 작성기(기준 스펙의 `generate.py`), 평가(`evaluate.py`), 실행 뒤 관측 스크립트가 없다(계획 3 몫이었다).
- 구성 0 에 VLM 시도 기록(`attempt_NN`)이 없다 — 2단계 학습은 사람 기준선 목표로만 돌았다.
- 1단계 게이트·보정·수확·2단계 착수·평가를 대화로 한 단계씩 진행하고 있다.

## 2. 사용자 결정 (2026-09-14)

| # | 결정 |
|---|---|
| 1 | VLM 백엔드 = **Claude Code 세션**(기준 스펙 §1 그대로, API 키 없음). 새 에이전트에 `prompt.md`·`snapshot.png` 경로만 준다 |
| 2 | 계획 2c 와 3 을 **자동 루프 한 계획**으로 합친다 |
| 3 | 루프가 **1단계부터** 몬다: A 게이트 → q 보정 → B 이어학습 → 수확 |
| 4 | 수확 시점 = B 저장 체크포인트 중 **재현 들기 검증 통과 ≥ 64 개**가 처음 나오는 것 |
| 5 | 과제 완수 = 2단계 결정론 평가 **성공 ≥ 50 %** + 실행 뒤 장면으로 재질의한 VLM 이 **done** + **영상 확인** |
| 6 | 사람 확인은 **완수 판정과 실패 때만** |
| 7 | 방식 = 세션 크론 틱 + 상태 기계(저장소 t2r 루프와 같은 틀, t2r 코드는 import 하지 않는다) |
| 8 | 현재 A 런(`iker_grasp_c00_a`)은 epoch 100 게이트 미달(잠김 1.52 %) 뒤 **epoch 200 에서 다시 판정**한다(재생창 없이) |

## 3. 구조

```text
scripts/iker/
  loop.py                 상태 기계 CLI: status | act <action>   (Isaac 은 하위 프로세스로만 띄운다)
  LOOP_PROMPT.md          틱 절차서(크론 프롬프트가 이 경로를 가리킨다)
  harvest_grasp_bank.py   1단계 체크포인트 → 성공 순간 캡처 → env 안 재현 들기 검증 → grasp_bank.json
  eval_iker.py            결정론 평가(스크래치 프로브를 옮김) → eval JSON
  observe.py              주어진 최종 상태(로봇 관절·신발 자세)로 헤드 카메라 스냅샷과 keypoints.json
source/openarm/openarm/agnostic/modules/iker/
  loop_state.py           순수: 상태 스키마·판정 함수·phase 전이(테스트 대상)
source/openarm/openarm/agnostic/tasks/iker_shoe/
  iker_shoe_grasp_env.py  + 성공 순간 캡처 버퍼(cfg 플래그로만 켬)·캡처 상태 복원
  config/__init__.py      2단계 id 를 open-sens_l_iker_shoe[-play] 로(로그 left/iker-shoe)
iker_runs/shoe_place/config_00/loop/
  LOOP_STATE.json, history.jsonl
  stage_01/  prompt.md, snapshot.png, attempt_KK/{response.md, gate.json}, eval_epNNN.json,
             observe/{snapshot.png, keypoints.json, prompt.md, response.md}, video.txt(영상 경로)
```

- `LOOP_STATE.json` 키:
  - 식별·진행: `schema`, `track`(`"iker_shoe_c00"`), `phase`, `status`(`running`·`awaiting`·`done`), `awaiting`(사유와 필요한 사람 행동), `stage`
  - 런: `runs`(phase 별 `label`·`pid`·`started`·`commit`·`checkpoint`)
  - 판정 기록: `gate1`(epoch·값), `calibration`, `bank`(경로·체크포인트·통과 수), `attempts`(단계별 게이트 시도 수), `eval`(체크포인트별 성공률)
  - `updated`
- phase 순서: `stage1_a` → `calibrate` → `stage1_b` → `harvest` → `vlm_target` → `stage2_train` → `observe_requery` →
  `completion_review` → `done`.
- 틱 한 번의 동작:
  1. `loop.py status` 로 판정한다. 순수 판정 함수에 TFEvents·체크포인트·`/proc` 조사 결과를 넣는다.
  2. **다음 동작 하나만** `loop.py act <action>` 으로 실행한다.
  3. 동작은 다시 실행해도 결과가 같다 — 이미 끝난 일은 건너뛴다.

## 4. 흐름과 판정

| phase | 판정 입력 | 진행 | 멈춤(awaiting) |
|---|---|---|---|
| stage1_a | TFEvents `Episode/grasp_episode/latched` 10 epoch 구간 평균, 저장 체크포인트 | 게이트 epoch(현재 200)에 마지막 구간 ≥ 2 % 면 계속. 구간 평균 ≥ 10 % 인 첫 저장 체크포인트가 나오면 calibrate | 게이트 epoch 에 < 2 %. 받침 위 들림이 잠김의 2 배를 넘으면 틱 보고에 표시만 한다 |
| calibrate | `measure_grasp_quality.py --checkpoint`(학습 옆, 헤드리스) | `QUALITY ... passed True` 면 보정 파일 커밋 후 stage1_b | 64 개 미만이면 다음 저장 체크포인트로 재시도. A 가 끝날 때까지 실패하면 멈춘다 |
| stage1_b | 부팅 줄의 `g_min=0.5` 와 보정 파일 q_lo·q_hi | A 를 PID 로 종료하고, 보정에 쓴 체크포인트에서 이어학습(`--checkpoint --no-reset_epoch --max_iterations 1000 env.grasp_reward.g_min=0.5`) | 부팅 줄 값이 다르다 |
| harvest | 새 B 저장 체크포인트(50 epoch 간격) | `harvest_grasp_bank.py`(512 env × 시드 0~3) 통과 ≥ 64 면 grasp_bank.json 커밋, B 종료 후 vlm_target | B max epoch 까지 64 미만 |
| vlm_target | `ingest.py` 종료 코드와 gate.json | 새 에이전트가 response.md 를 쓰고 게이트를 통과하면 interaction_vlm.json 커밋 후 stage2_train. 실패하면 같은 프롬프트로 재생성한다(최대 2 회, 오류 문구를 넣지 않는다 — 기준 스펙) | 3 회 실패 |
| stage2_train | 250 epoch 마다 `eval_iker.py`(512 env, 첫 에피소드, 학습 잡음 켬) | 성공 ≥ 50 % 면 observe_requery | 750 epoch 끝에도 < 50 %. 학습 오류 |
| observe_requery | `observe.py` 가 최고 체크포인트로 잡음 없는 결정론 롤아웃(512 env, 첫 에피소드)을 직접 돌려 env 마다 최종 로봇 관절·신발 자세·끝 거리를 기록한다 | 성공 env 중 끝 거리 중앙값 env 의 최종 상태로 렌더 → 다단계 프롬프트(이력 = 단계 1 영상 표식과 코드) → 새 에이전트 → `None`(done) 이면 영상(play `--video` 200 스텝, 16 env) 후 completion_review | done 이 아닌 새 단계를 내놓는다. 성공 env 가 0 개 |
| completion_review | 사람 | 사용자가 영상을 보고 승인하면 done | 항상 멈춘다 |

## 5. 학습 파지 뱅크 수확 (파지 스펙 §7 구체화)

- **캡처.** 1단계 env 에 cfg 플래그 `capture_success_states`(기본 False)를 둔다. 켜면 `_get_dones` 에서 `step.success` 인 env 의 다음 값을 리셋과 무관한 버퍼에 `torch.where` 로 기록하고 캡처 스텝 번호를 남긴다(스텝 경로에 호스트 동기 없음).
  - 전 관절 위치
  - 관절 목표: 팔 = 현재 관절 위치, 손 = EMA 목표 `_hand_targets`, 나머지 = `_joint_targets`
  - 신발 자세, 손바닥 자세
  - 같은 스텝의 리셋이 관절 상태·목표·신발 자세를 덮어쓰기 때문에 버퍼가 필요하다.
- **롤아웃.** env 마다 첫 에피소드만, 결정론 행동, 학습 잡음·외란은 끈다(`measure_grasp_quality.py` 와 같은 설정).
- **검증(env 안).**
  1. 캡처 상태를 복원한다.
  2. 손 행동은 캡처한 EMA 목표를 그대로 내는 정규화 값으로 둔다(EMA 고정점).
  3. 팔 행동은 +z 최대(0.02 m/스텝)로 5 스텝(10 cm) 들고, 10 스텝(1 s) 유지한다.
  4. 통과 = `grasp_bank.lift_held`: 신발 상승 ≥ 5 cm, 유지 중 신발–손바닥 상대 변화 < 1 cm.
  5. 검증 env 는 시작 DR·잡음·외란을 끈 공칭 물리다(`make_grasp_bank.py` 와 같다). 같은 env 클래스라 게인·자산이 1단계·2단계와 같다.
- **파일.** 기존 스키마(`bank_document`)를 쓰고 관절 열은 이름으로 정렬한다. 64 개 미만이면 파일을 쓰지 않는다.
  - metadata = 부팅 대조 9 키 + `source="learned_grasp"`·`side_sign`·`checkpoint`·`checkpoint_sha256`·`stage1_reward`(Stage1RewardCfg 값)·`seeds`·`captured`·`verified`
- 커밋된 오른팔 K1 뱅크는 이 파일로 대체된다(git 이력에 남는다). 뱅크 파일 테스트의 측면 skip 이 풀린다.

## 6. VLM 단계

- **프롬프트 작성(`loop.py act prompt`).**
  - 단계 1 = `prompts.fill_single_step(layout.TASK_INSTRUCTION)` + 구성 0 `snapshot.png`(정착 자세). 목표는 받침과 다른 신발 기준이라 시작 자세와 무관하다.
  - 재질의 = `fill_multi_step(task, [StageRecord("Stage 1 image", 단계 1 코드)])` + `observe/snapshot.png`.
- **생성기 격리.** 틱이 새 에이전트를 띄워 `prompt.md`·`snapshot.png` 경로만 준다. 에이전트는 영상과 프롬프트만 읽고 `response.md` 를 쓴다 — 저장소, 이전 응답, 사람 기준선을 보지 않는다.
- **파싱과 게이트.** `interaction.py`·`gate.py`·`ingest.py` 를 그대로 쓴다. 재질의 응답은 파싱만 한다: `None` = done, 그 밖(새 단계)은 멈춤.

## 7. 2단계

- id `open-sens_l_iker_shoe`(로그 `log/rl_games/open-sens/left/iker-shoe/<RUN_LABEL>`). env 는 파지 스펙 §8 그대로다: 행동 6D, 손 = 뱅크 손 목표 고정, IKER 보상 5항과 종료, `env.interaction_source=vlm`.
- RUN_LABEL `iker_vlm_c00_s1`, 750 epoch, 4096 env. 250 epoch 마다 평가하고, 성공률이 가장 높은 체크포인트를 observe 에 쓴다.
- 비교 기준: 오른팔 K1 사람 목표 26.8 %(ep250)·29.3 %(ep750), 논문 VLM sim 77.8 %.

## 8. 실패 처리와 안전

- **동시 실행.** 학습은 한 번에 하나. 읽기 전용 실행(보정·수확·평가·관측·영상)은 GPU 사용량 ≤ 20 GB 일 때만 학습 옆에서 하나씩 돌린다.
- **프로세스.**
  - 식별은 `/proc/<pid>/environ` 의 `RUN_LABEL` 로 한다.
  - 종료는 PID 로만 한다(pkill·killall 금지). SIGTERM 뒤 최대 5 분 기다린다.
  - 강제 종료는 멈춘 Isaac(결과를 쓰고 2 분 넘게 살아 있음)에만 쓴다.
- **진행 판정.** TFEvents(`Episode/*` 는 step = epoch)와 `nn/` 체크포인트로 한다 — nohup 콘솔은 버퍼링돼 epoch 줄이 늦다.
- **학습 오류.** Traceback·KeyError·OOM 은 멈춤이다. 루프는 코드를 고치지 않는다.
- **커밋.** phase 전환 때 해당 산출물만 커밋한다(보정 파일, grasp_bank.json, interaction_vlm.json, 평가 요약, LOOP_STATE·history). 트레일러 두 줄, `git add -A` 금지.
- **크론.** 세션 CronCreate `7,37 * * * *`(7 일 만료), 프롬프트는 트랙 이름과 `LOOP_PROMPT.md` 경로를 적는다. 틱 보고는 한두 줄이다. 세션이 끝나면 루프도 멈추고, 다음 세션은 상태 파일로 이어간다.
- **트랙 격리.** t2r 트랙과 파일·GPU·크론을 공유하지 않는다.

## 9. 검증

- **순수 테스트** `modules/iker/tests/test_loop_state.py`: phase 전이, 게이트·보정·수확·평가 판정 경계값, 재시도 횟수, awaiting 사유, 상태 파일 왕복.
- **순수 테스트** — 나머지: 프롬프트 작성(단일·다단계 경로와 이력), 평가 요약 판정, 수확 metadata 조립.
- **Isaac 스모크.**
  - 수확 스모크: 현재 A 체크포인트로 64 env × 시드 1. 캡처·복원·검증이 오류 없이 돌고 통과 수를 보고한다(0 도 허용).
  - 관측 스모크: 사람 기준선 목표 자세로 신발을 놓고 렌더한다. 모든 신발 키포인트가 여백 15 px 안에 있어야 한다.
  - 2단계 env 스모크: `l` id, 학습 뱅크가 생긴 뒤.
  - 2단계 학습 스모크: 10 epoch. 에피소드 200 스텝이 3 epoch × 32 스텝보다 길다(관찰 0244).
- **루프 모의 실행.** VLM 응답 대신 사람 기준선 목표를 `get_interaction_data` 코드로 적은 응답을 넣고, vlm_target → stage2(짧은 학습) → observe_requery 까지 상태 전이를 한 번 따라간다.

## 10. 제약 수치 (승인 전 계산)

- **시간.** 1단계 500 epoch ≈ 2.3 h(6,400~6,900 fps, epoch 당 131,072 프레임). 2단계 750 epoch ≈ 3.4 h(K1 런 ≈ 8,000 fps).
- **수확.** 512 env × 4 시드 = 첫 에피소드 2,048 개. 성공률 × 검증 통과율 ≥ 3.1 % 면 64 개가 모인다.
- **화면.** 사람 기준선 목표 자세의 신발 키포인트는 스냅샷 화면 약 (342~352, 91~346) px 에 들어온다(640×480, 여백 15 px 안). 재질의 영상에 놓인 신발이 보인다.
- **GPU.** 1단계 학습이 12.8 GB·94 % 를 쓴다. 읽기 전용 512 env 실행을 더해도 32 GB 안이다.

## 11. 범위 밖

VLM 이 새 단계를 내놓을 때 그 단계를 자동 학습하는 것(이번에는 멈춘다), 구성 1 이상, 실기와 실제 장면 재구성,
API·Qwen 백엔드, 루프의 코드 자동 수정.

## 12. 위험

- 1단계 성공률이 낮아 수확 64 개가 안 모일 수 있다(현재 A epoch 100 에 성공 0).
- 받침 위 들림이 잠김보다 빨리 오른다(epoch 100 에 1.92 %). B 와 수확에서 받침에 기댄 파지가 뱅크에 들어갈 수 있다 — 공중 10 cm 검증 들기가 이를 거른다.
- VLM 목표가 게이트를 3 번 넘지 못할 수 있다.
- 학습 뱅크 시작(신발을 든 채)은 시작 DR(질량 ×0.3~2.0)에서 떨어질 수 있다 — 평가 `dropped` 로 본다.
- 세션 크론은 7 일에 만료되고, 세션이 끝나면 멈춘다.

## 13. 개정 1 — 2단계 스모크 실패 뒤 재시작 (2026-09-15)

첫 루프는 vlm_target 뒤 2단계 env 스모크에서 멈췄고 사용자가 루프를 끝냈다. 진단과 1단계 수정은 파지 스펙 §16 에 있다.

- **2단계 env 스모크 검사 2.** 신발 z 판정을 **팔을 붙잡은 채 손바닥 기준 신발 미끄럼** 판정으로 바꾼다: 첫 무행동 스텝 뒤 팔 관절을 그 자리 PD 로 붙잡고(IK 행동 경로 우회) 49 스텝 뒤 손바닥 좌표계 신발 위치 변화 > 3 cm 인 env 비율 ≤ 25 %. 손바닥 z 변화는 찍기만 한다. 상대 IK 영명령의 처짐(진단 q10 −40 cm)은 새 뱅크(테이블 위 5~15 cm)에서 신발을 테이블에 내려놓아 좋은 파지도 미끄럼으로 읽히게 한다 — 붙잡은 팔 진단: 미끄럼 q90 10 mm(DR 끔)/35 mm(DR 켬).
- **2단계 env 스모크 검사 5.** 받침 목표 칸 도달은 최악 env 가 아니라 **20 mm 이내 env 비율 ≥ 50 %** 로 판정하고 최악 오차와 반대편 칸은 찍기만 한다 — 2단계는 뱅크 항목을 무작위로 뽑고, 위에서 잡는 손바닥 자세도 진단에서 51~66 % 였다(IK 시작 자세 탓 실패 포함). 최종 리뷰(2026-09-15) 반영.
- **보존.** 첫 루프 산출물을 `iker_runs/shoe_place/config_00/archive/2026-09-14_midair/` 로 옮겨 커밋한다: `loop/`,
  `grasp_bank.json`, `grasp_quality_calibration.json`. 옛 파일이 남으면 새 루프가 옛 보정·뱅크 metadata 를 읽는다.
  뱅크 파일 테스트는 뱅크가 없으면 skip 한다(수확 전). 옛 학습 런 디렉토리는 그대로 둔다.
- **새 트랙.** `track = iker_shoe_c00_r2`, 라벨 `stage1_a = iker_grasp_c00_r2_a`·`stage1_b = iker_grasp_c00_r2_b`·
  `stage2 = iker_vlm_c00_r2_s1`. A 는 전과 같이 사람이 띄우고(`RUN_LABEL` = A 라벨, 로그 `log/rl_games/open-sens/left/train_<라벨>.log`)
  `loop.py init --track iker_shoe_c00_r2 --policy '{"labels": …}' --adopt stage1_a` 로 넘긴다. 이후 흐름·판정은 §4 그대로다.
- **가동 시점.** 코드 수정·리뷰·스모크 통과 뒤 사용자에게 확인하고 A 를 띄운다.
