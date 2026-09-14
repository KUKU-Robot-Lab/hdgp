# grasp_fj text2reward 자동 루프 — 한 틱의 절차

사용자 결정(09.14): "t2r 은 루프 틱을 검사하면서 보상함수 설계가 제대로 되고 있는지 피드백 구조" · t2r 에선 reward-audit 안 씀 ·
fj 실험은 **GPU0 만** · **붓기 t2r 에 영향 금지**(GPU1 런 · `LOOP_PROMPT.md` · `t2r.py`/`t2r_round.py` · `reward_gen/pour_bi` · 붓기 cron).

상태 파일: `reward_gen/grasp_fj_envelope/LOOP_STATE.json` = {"iter","label","round","best","success_ticks",...}
실행 위치: `cd ~/rl_ws/hdgp` · 판정 수치의 유일한 출처는 `scripts/reward_gen/t2r_fj_round.py` 의 `ROUND_POLICY`.

1. `python3 scripts/reward_gen/t2r_fj_round.py status --label <label> --iter reward_gen/grasp_fj_envelope/iter_NN` → verdict.
2. verdict 별 행동
   - `continue` / `continue(success)` / `continue(curriculum)`: 한 줄 요약(epoch · 성공 prev_ep · tol · lifted ·
     성공 순간 손가락/손바닥 · envelope 항)만 남기고 종료. `success_ticks` 는 0 으로.
     (`continue(curriculum)` = 성공은 2.0 아래지만 공차가 최근 200 epoch 안에 조여졌다 — 조일 때마다 성공이 떨어지는 게 정상이다.)
   - `done_candidate`: `success_ticks += 1`. 2 틱 연속이면 6(종료 게이트), 아니면 요약만.
   - `crashed` / `dead`: 서버 콘솔 `~/rl_ws/our_source/fj_t2r_runs/<label>.out` 의 마지막 Traceback 을 읽는다.
       · 생성 코드 런타임 오류 → 오류 문장을 notes 파일로 만들어 3 → 4 → 5.
       · OOM·PhysX 버퍼·그 밖의 env/인프라 오류 → **사용자에게 보고하고 루프 정지**(루프는 env 수·env 코드를 바꾸지 않는다).
   - `advance` / `advance(envelope)`: 3 → 4 → 5.
3. `python3 scripts/reward_gen/t2r_fj_round.py advance --label <label> --iter reward_gen/grasp_fj_envelope/iter_NN [--notes <파일>]`
   → `iter_NN/feedback.md` + `iter_(NN+1)/prompt.md`.
   ★best 갱신: status.json metrics 의 성공(prev_ep)·성공 순간 손가락/손바닥으로 LOOP_STATE.best 와 비교해 좋으면 교체.
     previous_code 는 **직전 iter**(Eureka 방식). best 가 다르면 notes 에 best iter 번호·항 이름·지표를 한 줄로(설계 의견 금지).
4. 생성: Agent(general-purpose, 새 에이전트)에게 **`iter_(NN+1)/prompt.md` 경로만** 준다. 지시문은 iter_00 과 같다 —
   "그 파일만 읽고 요청대로 답을 `iter_(NN+1)/response.md` 에 써라. 저장소의 다른 파일을 열거나 실행하지 마라."
   완료 후 `../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/grasp_fj_envelope/iter_(NN+1)`
   (Isaac python = cpu+cuda 드라이런). FAIL 이면 오류를 notes 로 붙여 재생성(최대 2회), 그래도 FAIL 이면 사용자 보고·루프 정지.
5. 기동(★audit 없음 — 기계적 검증 PASS 면 바로):
   - 커밋: `git add reward_gen/grasp_fj_envelope/iter_NN reward_gen/grasp_fj_envelope/iter_(NN+1) reward_gen/grasp_fj_envelope/LOOP_STATE.json`
     → `git commit -m "t2r_fj: iter_(NN+1) …"` → `git push origin main`. ★다른 경로는 add 하지 않는다(붓기 세션 파일이 섞인다).
   - `python3 scripts/reward_gen/t2r_fj_round.py launch --label fj_t2r_i(NN+1) --iter reward_gen/grasp_fj_envelope/iter_(NN+1) --kill-label <label>`
     (검증 PASS·push 확인 → 서버 reset·HEAD 대조 → 이전 런 PID 종료(RUN_LABEL·CUDA 0 대조, GPU0 에 남의 런 있으면 중단) →
     run_fj.sh 12,288 env → 새 PID 확인 → `launch.json`). 실패 메시지가 나오면 그대로 사용자에게 보고하고 정지.
   - LOOP_STATE 갱신: iter·label·round+1·success_ticks 0. round > MAX_ROUNDS 이면 기동하지 말고 보고·정지.
   - 다음 틱의 status 가 crashed 가 아니면 부팅 성공(첫 epoch 는 약 3분).
6. 종료 게이트: `done_candidate` 2 틱 연속 → **서버에서 play 영상**을 찍어 프레임을 눈으로 본다(지표만으로 끝내지 않는다 —
   붓기 09.13 컵 끼워넣기 교훈). 볼 것: 손바닥이 컵 옆면에 닿는가 · 다섯 손가락이 둘레를 감싸는가 · 엄지가 반대편인가 · 들고 정지하는가.
   영상 절차는 메모리 `grasp-fj-five-finger-grasp-factor` 의 영상 항목(서버 스냅샷·12 env·카메라)을 따르되 play id 는
   `open-short_r_grasp_fj_t2r-play-lstm-sapg`, 저장은 `~/rl_ws/our_source/fj_t2r_videos/`.
   - 인벨롭이면: 런은 그대로 두고 루프를 멈추고 사용자에게 보고(best iter · 지표 · 체크포인트 · 영상 경로).
   - 아니면: 영상에서 **보이는 사실만** notes 로 적어 3 → 4 → 5.

금지: pkill/killall · GPU1 · 붓기 런/cron/파일 접촉 · env/cfg 코드 수정(보상 코드만 바뀐다) · reward-audit ·
프롬프트·notes 에 이 세션의 설계 의견 넣기 · 다른 경로 git add.
task-observer: 라운드가 바뀌거나 루프가 멈출 때 관찰 기록을 확인한다.
