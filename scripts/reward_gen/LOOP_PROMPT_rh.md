# text2reward 자동 루프 — pour_bi_rh(RH56F1 · pour_fabric_mimic · vision-3090) 한 틱의 절차

원본 `LOOP_PROMPT.md`(pour_bi · server GPU1)와 **파일·GPU·cron 을 공유하지 않는다**. 이 트랙은:
- 상태 `reward_gen/pour_bi_rh/LOOP_STATE.json` · iter `reward_gen/pour_bi_rh/iter_NN/`
- 호스트 **vision-3090**(RTX 3090 24 GB, GPU 1장) · 로그 `~/rl_ws/hdgp/log/pour_fabric_mimic/<label>.log`
- 판정기 `scripts/reward_gen/t2r_rh_round.py`(ROUND_HOURS 8 — 3090 은 서버의 ~1/3) · 런처 `scripts/experiments/pour_t2r_rh_train.sh`
- 생성 도구는 트랙 전용 `t2r_rh.py`(공용 모듈과 분리, 09.17) 에 **`--num-actions 24 --robot-file scripts/reward_gen/tasks/pour_bi_rh_robot.md`** 를 항상 붙인다
  (24 = (palm 6 + 손 6) × 2 · 로봇 설명은 RH56F1 언더액추 문장, 출처표 `pour_bi_rh_robot.SOURCES.md`).
실행 위치: `cd ~/rl_ws/hdgp`

1. `python3 scripts/reward_gen/t2r_rh_round.py status --label <label> --iter reward_gen/pour_bi_rh/iter_NN` → verdict.
2. verdict 별 행동 — 원본 LOOP_PROMPT.md 2 와 같다. 추가로 **`ctrl/mimic_err_max`** 가 10 rad 을 넘는 epoch 이 반복되면
   보상이 아니라 물리(종속 한계)가 문제다 → `env.mimic_dep_limit_margin_rad` 를 올려 같은 iter 재기동(라벨 `_m2` 등).
   OOM/PhysX overflow 면 env 를 절반으로(1024 → 512).
   **`continue(extended)`**(09.15): `LOOP_STATE.round_extension` 이 이 라벨에 걸려 기본 600 epoch 을 넘겼지만 연장 기준 안이다 —
   continue 와 같이 한 줄 요약만 하되, `task/src_grasped`(와 이후 `rcv_grasped`)가 150 epoch 동안 평탄하면 **자동 advance 하지 말고**
   사용자에게 advance 를 제안한다. 연장 기준(epoch/시간)을 넘으면 판정기가 `advance` 를 낸다.
3. `python3 scripts/reward_gen/t2r_rh_round.py advance --label <label> --iter reward_gen/pour_bi_rh/iter_NN`
   → `iter_NN/feedback.md` + `iter_(NN+1)/prompt.md`. best 갱신 규칙은 원본과 같다.
   ★원본 text2reward 의 피드백은 **사람이 학습된 정책을 보고 쓴 관찰**이다(관측 #0233). 지표 표만으로 advance 하지 말고
   play 영상 프레임을 보고 한 줄 관찰을 `--notes` 로 넣는다(사용자 승인 후).
4. 생성: Agent(general-purpose, fresh)에게 **`iter_(NN+1)/prompt.md` 경로만** 주고 `response.md` 를 쓰게 한다. 완료 후
   `python3 scripts/reward_gen/t2r_rh.py --num-actions 24 --robot-file scripts/reward_gen/tasks/pour_bi_rh_robot.md ingest --iter reward_gen/pour_bi_rh/iter_(NN+1)`.
   FAIL 이면 오류를 notes 로 붙여 재생성(최대 2회).
5. audit 없음(사용자 결정 09.13/09.14). ingest PASS 면 기동:
   - 커밋·push(main) 후 vision-3090 에서 `git pull --ff-only`. 이전 런은 **RUN_LABEL 로 PID 확정 후 kill**(pkill 금지).
     `ssh vision-3090 'export TERM=xterm-256color; cd ~/rl_ws/hdgp && nohup ./scripts/experiments/pour_t2r_rh_train.sh <label> reward_gen/pour_bi_rh/iter_(NN+1) --num_envs 1024 > log/pour_fabric_mimic/<label>.log 2>&1 < /dev/null &'`
     **num_envs 1024 시작**(3090 24 GB: short 트랙 1024 env 실측 19 GB · RH56F1 은 링크가 적어 여유). OOM 이면 512.
   - 로컬: `iter_(NN+1)/launch.json` {"label","num_envs","started","commit","host":"vision-3090"} 기록, LOOP_STATE 갱신.
   - 첫 epoch 확인(5분 내 Traceback 이면 2 의 crashed 절차).
6. 종료 조건: 원본과 같다(episode_success ≥ 0.5 두 틱 연속 ∧ nested_rate < 0.1 ∧ cup_collision_rate < 0.05) **+ 영상 확인**.
   종료 시 best iter·지표·체크포인트·영상 경로를 보고한다. round ≥ MAX_ROUNDS(8) 도 종료.

금지: pkill/killall · 다른 호스트/GPU 사용 · pour_fabric(원본)·pour_fabric_mimic env 수정(보상 코드만 바뀐다) ·
루트에 .sh 생성(런처는 scripts/experiments) · 로그를 log/pour_fabric_mimic 밖에 두기.
task-observer: 각 라운드 종료(deliverable flush)마다 관측 기록 확인.
