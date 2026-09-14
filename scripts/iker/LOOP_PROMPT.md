# IKER 신발 놓기 자동 루프 — 한 틱의 절차

설계: `docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md`. 상태: `iker_runs/shoe_place/config_00/loop/LOOP_STATE.json`.
실행 위치: `cd ~/rl_ws/hdgp-iker` (공유 체크아웃 `~/rl_ws/hdgp` 에서는 아무것도 실행하지 않는다).

1. `python3 scripts/iker/loop.py status` 를 실행해 JSON 의 `action`·`reason`·`notes`·`summary` 를 읽는다.
2. `action` 별로 한다.
   - `wait`: 아무것도 하지 않는다.
   - `vlm_generate`: 새 에이전트(Agent, general-purpose)를 띄워 `vlm.brief` 를 **그대로** 준다.
     설명·힌트·사람 기준선·이전 응답·이 대화의 맥락을 덧붙이지 않는다(생성기 격리, 설계 §6).
     에이전트가 끝나면 `vlm.response` 파일이 생겼는지만 보고 1 로 돌아간다.
   - 그 밖(`pause` 포함): `python3 scripts/iker/loop.py act <action>`.
3. 한 틱은 Isaac 을 띄우는 동작(`run_*`·`launch_*`), `vlm_generate`, `pause`, `wait` 중 하나를 마치면 끝낸다.
   그 전의 기록·파일 동작(`record_*`·`advance`·`commit_*`·`write_*`·`ingest`·`parse_requery`·`next_calibration`·`store_video`·`kill_stale`)은
   끝나는 대로 1 로 돌아가 이어서 한다. 한 틱에 `act` 는 최대 8 번.
4. 틱 보고는 1~2 줄: `summary` + 추세 해석 + 다음 틱에 볼 것. `notes` 가 있으면 함께 적는다.
   `pause` 를 기록했으면 `reason` 과 `params.needs` 를 사용자에게 그대로 알린다. 그 뒤 틱은 `wait` 만 낸다.
5. 추가 확인
   - `commit_bank` 뒤: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py -q -p no:cacheprovider`
     결과를 보고에 적는다. 실패해도 루프는 코드를 고치지 않는다 — 사용자에게 알린다.
   - `completion_review` 의 pause: `stage_01/video.txt` 의 영상 경로를 사용자에게 준다.
6. 사람의 말이 있을 때만: `act approve`(영상 확인 뒤 완수 승인), `act resume [--policy '{"gate_epoch": 300}']`.
   정책 변경은 `resume --policy` 로만 한다(LOOP_STATE.json 손편집 금지).

금지: `pkill`·`killall`(종료는 loop.py 가 RUN_LABEL 로 찾은 PID 로만 한다) · 코드·보상·env 수정 · `git add -A`·push ·
다른 트랙(t2r·pour_fabric)의 런·GPU·크론 접촉.

크론: 세션 CronCreate `7,37 * * * *`, 프롬프트 "IKER 자동 루프 틱: ~/rl_ws/hdgp-iker/scripts/iker/LOOP_PROMPT.md 절차대로 한 틱을
수행한다 (track iker_shoe_c00)". 세션이 끝나면 루프도 멈춘다. 다음 세션은 `status` 로 이어가고, 커밋 트레일러는
`python3 scripts/iker/loop.py --session-url <그 세션 URL> act ...` 로 그 세션 것을 쓴다.

task-observer: 틱 보고(산출물 전달) 때 관측 기록을 확인한다.
