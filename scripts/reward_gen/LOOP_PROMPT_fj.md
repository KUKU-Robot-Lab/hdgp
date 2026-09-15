# grasp_fj text2reward 루프 — 한 틱의 절차 (원본 t2r interactive 형식 · 트랙 공용)

사용자 결정(09.14):
- "t2r 은 루프 틱을 검사하면서 보상함수 설계가 제대로 되고 있는지 피드백 구조" · t2r 에선 reward-audit 안 씀.
- 라운드 피드백 = 원본 text2reward interactive(`skill_gen/text2reward/code_generation/interactive/`) 그대로 —
  학습한 로봇을 **영상으로 보고** "I can see from the robot that …"(관찰) + "the feedback for improvement is …"(개선 피드백)을
  지난 코드와 함께 전부 넣고 "Re-imagine which steps is missed or wrong". **Claude 가 영상·지표로 초안 → 사용자 승인** 후에만
  생성기에 넘긴다(승인 없이 다음 보상을 만들지 않는다).
- fj 실험은 **GPU0 만** · **붓기 t2r 에 영향 금지**(GPU1 런 · `LOOP_PROMPT.md` · `t2r.py`/`t2r_round.py` · `reward_gen/pour_bi` · 붓기 cron).
- 관측·actor 입력은 **sim2real 가능한 구조로만** 바꾼다(사용자 09.14) — 루프는 보상 코드만 바꾸고 env 는 건드리지 않는다.
- "컵에 접근, 파지, 리프트가 잘 되는지 틱을 확인" — 틱 요약은 에피소드 퍼널(`stage/<단계>_ep`)이다.
- "보상 구조가 확실하지 않은데 SAPG·env 수를 너무 늘린 게 아닌지" → reach 보상 루프는 **PPO-LSTM 4096 env**, 파지·리프트가
  되는 보상이 나온 뒤 최종 정책만 SAPG 12,288. 막힌 단계가 200 epoch 정체하면 e1000 전이라도 라운드 끝.

트랙(`t2r_fj_round.py` `TRACKS`): `grasp_fj_envelope`(라벨 `fj_t2r_iNN`, B leaf, SAPG 12,288 — 정지) · `grasp_fj_reach`(라벨 `fj_reach_iNN`,
테이블 가장자리 시작·cup_family·0.3 rad/s·15 s — 09.14 최종 목표, PPO-LSTM 4096 · i00 만 SAPG 12,288 · i05 에서 정지) ·
★`grasp_fj_stage`(라벨 `fj_stage_iNN`, 09.15 사용자 3단계 재구성 — 같은 reach env + 보상 게이트 래치, **새 이력**, 과제 문장
`tasks/grasp_fj_stage.txt`, 조임 적응은 이번 범위 밖). cron 프롬프트가 트랙을 지정한다.
★09.15 23:0x 사용자 "T2R 제대로 적용하면서 진행되는건지?" → grasp_fj_stage 는 iter_02 부터 **시작 상태 커리큘럼(가까운 출발 50 %) ·
C자 사전파지 접근 래치 · env/래치/과제 문장 고정 · 조기 판정 끔**(`TRACKS.early_stop False` — 체크포인트·막힘·단계 연장 없음, 라운드는
3000 epoch 또는 4 h 끝까지 → 영상 판정). 틱 보고는 출발 그룹별(`stage/far_*`·`stage/near_*`)로 적는다. 라운드 사이에 env 를 바꾸지 않는다.
★판정 창은 **프레임 기준**(사용자 09.14): `track_policy` 가 ROUND_POLICY 의 epoch 값(라운드·창·평균)을 12,288/env 수 배로 늘린다 —
아래의 "200 epoch" 은 기준값이고 reach(4096)는 **600 epoch**, 라운드 끝은 **3000 epoch**(≈3.3 h). 시간 상한 4 h 는 그대로.
상태: `reward_gen/<track>/LOOP_STATE.json` = {"track","iter","label","round","awaiting","best","success_ticks",...}
이력: `reward_gen/<track>/history.jsonl` — 라운드별 (코드 · 관찰 · 피드백), 다음 프롬프트에 전부 들어간다.
판정 수치: `scripts/reward_gen/t2r_fj_round.py` 의 `ROUND_POLICY` 한 곳. 실행 위치 `cd ~/rl_ws/hdgp`. 아래 모든 명령에 `--track <track>`.

1. `python3 scripts/reward_gen/t2r_fj_round.py status --track <track> --label <label> --iter reward_gen/<track>/iter_NN` → verdict.
2. `LOOP_STATE.awaiting` 이 있으면(초안 작성·사용자 승인 대기) 한 줄 요약만 남기고 끝낸다. 영상·초안을 다시 만들지 않는다.
   단 `awaiting` 이 `extend_<label>_until_eNNNN`(사용자가 보상은 그대로 두고 학습 연장을 고름 — `LOOP_STATE.extend`)이면
   epoch < NNNN 동안은 한 줄 요약만, NNNN 이상이면 4(라운드 끝: 영상 → 기존 초안을 새 지표로 갱신 → 승인 요청, awaiting=approval_iter_NN)로 간다.
3. verdict 별
   - `continue` / `continue(success)` / `continue(curriculum)` / `continue(stage)`: 한 줄 요약만. success_ticks 0.
     요약 = epoch · 성공 · tol · **퍼널(에피소드 비율, 200 epoch 변화): 접근 → 파지 → 인벨롭 → 리프트 → 성공** · 손바닥↔컵 간극 ·
     성공 순간 손가락/손바닥. ★사용자 09.14 "컵에 접근, 파지, 리프트가 잘 되는지 틱을 확인" — 막힌 단계(앞 단계보다 크게 떨어지는 곳)를
     한 단어로 적는다. 퍼널만 보고 보상을 바꾸지 않는다(바꾸는 것은 라운드 끝 영상 → 승인 뒤).
     (`continue(curriculum)` = 성공은 2.0 아래지만 공차가 최근 200 epoch 안에 조여졌다 — 조일 때마다 성공이 떨어지는 게 정상이다.)
     (`continue(stage)` = 성공·공차는 안 움직였지만 퍼널 어느 단계가 최근 200 epoch 에 2%p 이상 올랐다 — 2×ROUND_EPOCHS 까지.)
   - `crashed` / `dead`: 콘솔 `~/rl_ws/our_source/fj_t2r_runs/<label>.out` 의 Traceback 확인.
       · 생성 코드 런타임 오류 → 오류 문장을 관찰로, 고칠 점을 피드백으로 초안 → 4c(승인 요청).
       · env/인프라 오류(OOM·PhysX 등) → 사용자 보고·루프 정지.
   - `done_candidate`: success_ticks += 1. 2 틱 연속이면 4(라운드 끝 — 영상으로 종료 여부를 올린다).
   - `advance` / `advance(envelope)` / `advance(stuck:<단계>)`: 4.
     (`advance(stuck:<단계>)` = 앞 단계가 200 epoch 내내 ≥0.9 인데 그 단계가 ≈0 이고 뒤 단계도 안 오른다 — e1000 전이라도 라운드 끝.)
   - `stop(checkpoint:approach)` / `stop(checkpoint:envelope)`: 4. ★09.15 사용자 "의도한 동작이 전혀 안 나오는데 학습이 진행되는 게 잘못".
     보상 게이트 래치(`grasp_gates.py`: 기본 손 자세로 접근 완료 → 그 뒤 인벨롭 완료)의 에피소드 비율(최근 창 평균)이
     12,288 env 기준 e200 까지 접근 ≥ 0.3 · e500 까지 인벨롭 ≥ 0.05 에 못 미치면 곧바로 라운드 끝(reach·stage 4096 = e600 · e1500).
     ★기동할 때 체크포인트 시각(학습 시작 + epoch × 약 4 s)에 1회 틱 cron 을 건다 — 3 h 틱을 기다리지 않는다.
4. 라운드 끝 — 영상 → 초안 → 승인 요청
   a. `python3 scripts/reward_gen/t2r_fj_round.py video --track <track> --label <label> --iter reward_gen/<track>/iter_NN`
      (백그라운드, 수 분 · 런이 쓴 태스크는 launch.json 에서 읽는다) → 로컬 `~/rl_ws/our_source/fj_t2r_videos/<label>_<ts>.mp4` ·
      `frames_<…>/sheet.png` · 정지 프레임 5장.
   b. 시트·정지 프레임과 `iter_NN/status.json` 지표를 보고 초안 두 파일을 쓴다(영어 — 생성기 프롬프트 언어):
      - `iter_NN/observation.md`: 영상에서 **보이는 동작** + 지표 사실(수치). 보이지 않는 것은 "안 보인다"고 쓴다.
        ★초안을 쓴 뒤 생성 직전에 지표 수치를 한 번 더 확인한다(승인 대기 중에도 공차 등이 움직인다).
      - `iter_NN/improvement.md`: 개선 피드백 — 되고 있는 것과 고칠 것. 가중치 숫자·코드 조각은 쓰지 않는다.
   c. 사용자에게 승인 요청: 시트·영상(SendUserFile)과 두 초안(한국어 요약 포함)을 보내고
      `LOOP_STATE.awaiting = "approval_iter_NN"` 으로 저장한 뒤 끝낸다. done_candidate 에서 온 경우 영상이 인벨롭·직립·정지면
      "종료 제안"으로 올린다(루프 종료는 사용자가 정한다).
5. 사용자가 승인하면(대화에서 — 수정 요청이면 파일을 고쳐 다시 승인받는다):
   a. `python3 scripts/reward_gen/t2r_fj_round.py advance --track <track> --label <label> --iter reward_gen/<track>/iter_NN --description reward_gen/<track>/iter_NN/observation.md --feedback reward_gen/<track>/iter_NN/improvement.md [--no-metrics]`
      → `history.jsonl` 갱신 + `iter_(NN+1)/prompt.md`(전 이력 + 참고 지표 표, 환경 변종은 meta 의 variant).
   과제 문장(`scripts/reward_gen/tasks/<track>.txt`)을 사용자 결정으로 바꾼 라운드만 `--task-file` 을 붙인다 — 없으면 지난 meta 의 task 승계
   (★09.15 사용자 "컵에 다가가는 palm_ee_x · 손가락 방향" → grasp_fj_stage 접근 손 방향).
   b. 생성: Agent(general-purpose, 새 에이전트)에게 **`iter_(NN+1)/prompt.md` 경로만** —
      "그 파일만 읽고 요청대로 답을 `iter_(NN+1)/response.md` 에 써라. 저장소의 다른 파일을 열거나 실행하지 마라."
   c. `../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/<track>/iter_(NN+1)` (cpu+cuda 드라이런).
      FAIL 이면 오류를 붙여 재생성(최대 2회), 그래도 FAIL 이면 보고·정지.
   d. 커밋(이 경로만, 메시지는 파일이나 -m 여러 개로 — heredoc 두 개를 한 명령에 넣지 않는다):
      `git add reward_gen/<track>/iter_NN reward_gen/<track>/iter_(NN+1) reward_gen/<track>/history.jsonl reward_gen/<track>/LOOP_STATE.json`
      → `git commit` → `git log -1 --format=%s` 확인 → `git push origin main`.
   e. `python3 scripts/reward_gen/t2r_fj_round.py launch --track <track> --label <prefix>(NN+1) --iter reward_gen/<track>/iter_(NN+1) --kill-label <label>`
      (검증·push 확인 → 서버 reset·HEAD 대조 → 이전 런 PID 종료(RUN_LABEL·CUDA0, 5분 대기, GPU0 에 남의 런이면 중단) →
      run_fj.sh(트랙 기본: reach 4096 PPO-LSTM · envelope 12,288 SAPG) → 새 PID 확인 → `launch.json`). 실패 메시지는 그대로 보고·정지.
      SIGKILL 은 사용자 확인 후.
   f. LOOP_STATE: iter · label · round+1 · awaiting null · success_ticks 0. round > MAX_ROUNDS 면 기동 전에 보고·정지.

금지: pkill/killall · GPU1 · 붓기 런/cron/파일 접촉 · env/cfg 코드 수정(보상 코드만 바뀐다) · reward-audit ·
**사용자 승인 없는 관찰·피드백 사용** · 다른 경로 git add.
task-observer: 라운드가 바뀌거나 루프가 멈출 때 관찰 기록을 확인한다.
