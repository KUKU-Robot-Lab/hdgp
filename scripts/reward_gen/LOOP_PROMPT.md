# text2reward 자동 루프 — 한 틱의 절차 (사용자 승인 09.13: 자동 기동 허용 · GPU1 단독 · env 확대 가능)

상태 파일: `reward_gen/pour_bi/LOOP_STATE.json` = {"iter": N, "label": "...", "round": k, "best": {...}}
실행 위치: `cd ~/rl_ws/hdgp`

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
5. audit(reward-audit 5 체크, `iter/audit.md` 에 기록). REJECT 면 사유를 notes 로 붙여 재생성(최대 2회).
   ACCEPT 면 기동:
   - 커밋: `git add reward_gen/pour_bi/iter_(NN+1) && git commit -m "t2r: iter_(NN+1) …" && git push origin main`
   - 서버: `git fetch && git reset --hard origin/main`, 이전 런은 **RUN_LABEL 로 PID 확정 후 kill**(pkill 금지),
     `nohup bash ~/logs/t2r/launch_t2r.sh <label> reward_gen/pour_bi/iter_(NN+1) <num_envs> > ~/logs/t2r/<label>.log 2>&1 < /dev/null &`
   - 로컬: `iter_(NN+1)/launch.json` {"label","num_envs","started","commit","gpu":1} 기록, LOOP_STATE 갱신.
   - 첫 epoch 확인(3분 내 Traceback 이면 2 의 crashed 절차).
6. 종료 조건: episode_success 최근 평균 ≥ 0.5 가 두 틱 연속(→ 그 런을 끝까지 두고 루프 종료 보고),
   또는 round ≥ MAX_ROUNDS(8). 종료 시 사용자에게 best iter·지표·체크포인트 경로를 보고한다.

금지: pkill/killall · GPU0 사용 · 남의 런 접촉 · env 파일(pour_fabric) 수정(보상 코드만 바뀐다).
task-observer: 각 라운드 종료(deliverable flush)마다 관측 기록 확인.
