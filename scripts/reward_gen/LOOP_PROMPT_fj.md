# grasp_fj text2reward 루프 — 한 틱의 절차 (원본 t2r interactive 형식)

사용자 결정(09.14):
- "t2r 은 루프 틱을 검사하면서 보상함수 설계가 제대로 되고 있는지 피드백 구조" · t2r 에선 reward-audit 안 씀.
- 라운드 피드백 = 원본 text2reward interactive(`skill_gen/text2reward/code_generation/interactive/`) 그대로 —
  학습한 로봇을 **영상으로 보고** "I can see from the robot that …"(관찰) + "the feedback for improvement is …"(개선 피드백)을
  지난 코드와 함께 전부 넣고 "Re-imagine which steps is missed or wrong". **Claude 가 영상·지표로 초안 → 사용자 승인** 후에만
  생성기에 넘긴다(승인 없이 다음 보상을 만들지 않는다).
- fj 실험은 **GPU0 만** · **붓기 t2r 에 영향 금지**(GPU1 런 · `LOOP_PROMPT.md` · `t2r.py`/`t2r_round.py` · `reward_gen/pour_bi` · 붓기 cron).

상태: `reward_gen/grasp_fj_envelope/LOOP_STATE.json` = {"iter","label","round","awaiting","best","success_ticks",...}
이력: `reward_gen/grasp_fj_envelope/history.jsonl` — 라운드별 (코드 · 관찰 · 피드백), 다음 프롬프트에 전부 들어간다.
판정 수치: `scripts/reward_gen/t2r_fj_round.py` 의 `ROUND_POLICY` 한 곳. 실행 위치 `cd ~/rl_ws/hdgp`.

1. `python3 scripts/reward_gen/t2r_fj_round.py status --label <label> --iter reward_gen/grasp_fj_envelope/iter_NN` → verdict.
2. `LOOP_STATE.awaiting` 이 있으면(사용자 승인 대기) 한 줄 요약만 남기고 끝낸다 — 런은 계속 학습한다. 영상·초안을 다시 만들지 않는다.
3. verdict 별
   - `continue` / `continue(success)` / `continue(curriculum)`: 한 줄 요약(epoch · 성공 · tol · lifted · 접촉 · envelope 항)만. success_ticks 0.
   - `crashed` / `dead`: 콘솔 `~/rl_ws/our_source/fj_t2r_runs/<label>.out` 의 Traceback 확인.
       · 생성 코드 런타임 오류 → 오류 문장을 관찰로, 고칠 점을 피드백으로 초안 → 4c(승인 요청).
       · env/인프라 오류 → 사용자 보고·루프 정지.
   - `done_candidate`: success_ticks += 1. 2 틱 연속이면 4(라운드 끝 — 영상으로 종료 여부를 올린다).
   - `advance` / `advance(envelope)`: 4.
4. 라운드 끝 — 영상 → 초안 → 승인 요청
   a. `python3 scripts/reward_gen/t2r_fj_round.py video --label <label> --iter reward_gen/grasp_fj_envelope/iter_NN`
      (백그라운드, 수 분) → 로컬 `~/rl_ws/our_source/fj_t2r_videos/<label>_<ts>.mp4` · `frames_<…>/sheet.png` · 정지 프레임 5장.
   b. 시트·정지 프레임과 `iter_NN/status.json` 지표를 보고 초안 두 파일을 쓴다(영어 — 생성기 프롬프트 언어):
      - `iter_NN/observation.md`: 영상에서 **보이는 동작** + 지표 사실(수치). 보이지 않는 것은 "안 보인다"고 쓴다.
      - `iter_NN/improvement.md`: 개선 피드백 — 되고 있는 것과 고칠 것. 가중치 숫자·코드 조각은 쓰지 않는다.
   c. 사용자에게 승인 요청: 시트·영상(SendUserFile)과 두 초안(한국어 요약 포함)을 보내고
      `LOOP_STATE.awaiting = "approval_iter_NN"` 으로 저장한 뒤 끝낸다. done_candidate 에서 온 경우 영상이 인벨롭·직립·정지면
      "종료 제안"으로 올린다(루프 종료는 사용자가 정한다).
5. 사용자가 승인하면(대화에서 — 수정 요청이면 파일을 고쳐 다시 승인받는다):
   a. `python3 scripts/reward_gen/t2r_fj_round.py advance --label <label> --iter .../iter_NN --description .../iter_NN/observation.md --feedback .../iter_NN/improvement.md [--no-metrics]`
      → `history.jsonl` 갱신 + `iter_(NN+1)/prompt.md`(전 이력 + 참고 지표 표).
   b. 생성: Agent(general-purpose, 새 에이전트)에게 **`iter_(NN+1)/prompt.md` 경로만** —
      "그 파일만 읽고 요청대로 답을 `iter_(NN+1)/response.md` 에 써라. 저장소의 다른 파일을 열거나 실행하지 마라."
   c. `../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/grasp_fj_envelope/iter_(NN+1)` (cpu+cuda 드라이런).
      FAIL 이면 오류를 붙여 재생성(최대 2회), 그래도 FAIL 이면 보고·정지.
   d. 커밋(이 경로만): `git add reward_gen/grasp_fj_envelope/iter_NN reward_gen/grasp_fj_envelope/iter_(NN+1) reward_gen/grasp_fj_envelope/history.jsonl reward_gen/grasp_fj_envelope/LOOP_STATE.json`
      → `git commit -m "t2r_fj: iter_(NN+1) …"` → `git push origin main`.
   e. `python3 scripts/reward_gen/t2r_fj_round.py launch --label fj_t2r_i(NN+1) --iter reward_gen/grasp_fj_envelope/iter_(NN+1) --kill-label <label>`
      (검증·push 확인 → 서버 reset·HEAD 대조 → 이전 런 PID 종료(RUN_LABEL·CUDA0, GPU0 에 남의 런이면 중단) → run_fj.sh 12,288 env →
      새 PID 확인 → `launch.json`). 실패 메시지는 그대로 보고·정지.
   f. LOOP_STATE: iter · label · round+1 · awaiting null · success_ticks 0. round > MAX_ROUNDS 면 기동 전에 보고·정지.

금지: pkill/killall · GPU1 · 붓기 런/cron/파일 접촉 · env/cfg 코드 수정(보상 코드만 바뀐다) · reward-audit ·
**사용자 승인 없는 관찰·피드백 사용** · 다른 경로 git add.
task-observer: 라운드가 바뀌거나 루프가 멈출 때 관찰 기록을 확인한다.
