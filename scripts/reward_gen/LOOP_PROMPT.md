# text2reward 자동 루프 — 한 틱의 절차 (사용자 승인 09.13: 자동 기동 허용 · GPU1 단독 · env 확대 가능)

상태 파일: `reward_gen/pour_bi/LOOP_STATE.json` = {"iter": N, "label": "...", "round": k, "best": {...}}
실행 위치: `cd ~/rl_ws/hdgp`

0. ★HOLD: `LOOP_STATE.hold.label == label` 이면 **사용자가 해제를 요청할 때까지 그 런을 계속 학습**한다
   (09.14 사용자 지시 "이번엔 요청 전까지 계속 학습"). advance·kill·재생성 금지, 틱 보고만.
   status 는 이때 advance 대신 `continue(hold)` 를 낸다. crashed/dead 면 같은 iter 를 재기동한다.
   성공 조건에 닿아도 종료하지 않고 보고만 한다(영상 확인은 해도 된다). 해제는 사용자 요청으로만 `hold` 키를 지운다.
   ★수렴 판정(09.15 사용자 "현재 학습 수렴할 때까지 진행"): 매 틱 `hold.convergence.history` 에
   {epoch, recent10, adr} 를 추가하고, `adr/progress == 1.0` 이후 두 틱 연속 episode_success recent10 변화가
   ±0.02 이내면 "수렴"으로 보고한다. 수렴해도 런은 유지하고 중단·eval 은 사용자 결정을 기다린다.
1. `python3 scripts/reward_gen/t2r_round.py status --label <label> --iter reward_gen/pour_bi/iter_NN`
   → verdict 를 읽는다.
2. verdict 별 행동
   - `continue` / `continue(success)`: 한 줄 요약(epoch·episode_success·grasp·in_target)만 남기고 종료.
   - `crashed` / `dead`: 서버 로그 마지막 Traceback 을 읽는다.
       · PhysX 버퍼/OOM 이면 env 수를 절반으로 낮춰 **같은 iter** 를 재기동(라벨에 e512 등).
       · 생성 코드 런타임 오류면 그 오류를 notes 로 넣어 다음 iter 생성(3 → 4 → 5).
   - `advance`: 아래 3 → 4 → 5.
3. `python3 scripts/reward_gen/t2r_round.py advance --label <label> --iter reward_gen/pour_bi/iter_NN`
   → `iter_NN/feedback.md` + `iter_(NN+1)/prompt.md`. 관찰이 있으면 `--notes` 로.
   ★best 갱신: status.metrics 의 task/episode_success.max, bead/in_target.max, grasp 평균으로
     LOOP_STATE.best 와 비교해 더 좋으면 best 를 이 iter 로 바꾼다. 다음 프롬프트의 previous_code 는
     **직전 iter**(Eureka 방식: 최신 피드백 기준)로 두되, best 가 다르면 notes 에 best 의 항·지표를 한 줄 적는다.
4. 생성: Agent(general-purpose, fresh) 에게 **`iter_(NN+1)/prompt.md` 경로만** 주고 `response.md` 를 쓰게 한다
   (저장소 탐색 금지 지시 그대로). 완료 후
   `python3 scripts/reward_gen/t2r.py ingest --iter reward_gen/pour_bi/iter_(NN+1)`.
   검증 FAIL 이면 오류를 notes 로 붙여 한 번 재생성(최대 2회), 그래도 FAIL 이면 사용자에게 보고하고 루프 종료.
5. ★audit 없음(사용자 결정 09.13): 보상의 좋고 나쁨은 학습 지표가 판정한다 — 이 세션의 설계 의견을
   프롬프트로 흘리지 않는다(생성기 오염 금지). 기계적 검증(ingest) PASS 면 바로 기동:
   - 커밋: `git add reward_gen/pour_bi/iter_(NN+1) && git commit -m "t2r: iter_(NN+1) …" && git push origin main`
   - 서버: `git fetch && git reset --hard origin/main`, 이전 런은 **RUN_LABEL 로 PID 확정 후 kill**(pkill 금지),
     ★SIGTERM 후 30 s 안에 python 이 안 죽으면(09.14 i03: R 상태로 16.5 GB 유지) RUN_LABEL 재확인 후 `kill -9 <pid>`,
     GPU 메모리가 비워진 것을 보고 기동한다. 런처는 `scripts/experiments/run_pour_t2r.sh`(f6f38389 에서 루트→이동).
     `nohup bash ~/logs/t2r/launch_t2r.sh <label> reward_gen/pour_bi/iter_(NN+1) <num_envs> > ~/logs/t2r/<label>.log 2>&1 < /dev/null &`
     **num_envs = 4096**(사용자 지시 09.13: 적은 env 는 분산이 큼 · 1024 에서 19 GB → 4096 ≈ 40 GB 예상). PhysX overflow/OOM 이면 2048 → 1024 로 후퇴.
   - 로컬: `iter_(NN+1)/launch.json` {"label","num_envs","started","commit","gpu":1} 기록, LOOP_STATE 갱신.
   - 첫 epoch 확인(3분 내 Traceback 이면 2 의 crashed 절차).
6. 종료 조건: episode_success 최근 평균 ≥ 0.5 가 두 틱 연속 **그리고** task/nested_rate < 0.1 · bead/spill 정상
   → 그 런을 끝까지 두고, **종료 선언 전에 서버에서 play 영상을 찍어 프레임을 눈으로 확인**한다
   (09.13 실측: 지표는 0.73 성공인데 영상은 컵 끼워넣기였다 — 지표만으로 종료하지 않는다).
   영상이 붓기가 아니면 그 hacking 을 env 판정에 막고(보상이 아니라 판정) 같은 보상으로 재학습.
   또는 round ≥ MAX_ROUNDS(8). 종료 시 사용자에게 best iter·지표·체크포인트·영상 경로를 보고한다.

금지: pkill/killall · GPU0 사용 · 남의 런 접촉 · env 파일(pour_fabric) 수정(보상 코드만 바뀐다).
task-observer: 각 라운드 종료(deliverable flush)마다 관측 기록 확인.
