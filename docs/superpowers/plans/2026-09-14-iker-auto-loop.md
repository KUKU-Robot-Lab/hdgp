# IKER 신발 놓기 자동 루프 구현 계획 (계획 2c + 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1단계 학습 파지부터 VLM 목표·2단계 학습·결정론 평가·실행 뒤 재질의·영상까지, 세션 크론 틱과 상태 기계로 사람 없이 이어 가는 IKER 자동 루프를 만들고 구성 0 에서 가동한다.

**Architecture:**
- 순수 모듈 세 개가 규칙을 가진다.
  - `modules/iker/loop_state.py`: 상태 스키마·판정·전이. 모든 문턱값이 `DEFAULT_POLICY` 에 있다.
  - `modules/iker/loop_probe.py`: 로그·TFEvents·체크포인트·`/proc`·루프 파일을 `Probe` 로 읽는다.
  - `modules/iker/loop_vlm.py`: 생성기 에이전트용 프롬프트와 재질의 파싱.
- `scripts/iker/loop.py` 는 `status | act <action>` CLI 다. Isaac 은 RUN_LABEL 을 단 자식 프로세스로만 띄우고, PID 로만 멈추고, 단계 산출물만 커밋한다.
- Isaac 쪽 스크립트:
  - 1단계 env: 성공 순간 캡처·복원(cfg 플래그)
  - `harvest_grasp_bank.py`: 수확
  - `eval_iker.py`: 결정론 평가와 최종 상태 기록
  - `observe.py`: 실행 뒤 장면 렌더(`head_snapshot.py` 를 `snapshot.py` 와 공유)
- 2단계 gym id 는 `open-sens_l_iker_shoe[-play]` 로 바꾼다.

**Tech Stack:** Isaac Sim 5.1 / Isaac Lab 0.50.5, rl_games PPO, torch, pytest(시스템 python3), Claude Code 세션(VLM·크론).

**Spec:** `docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md` (§3 구조, §4 흐름과 판정, §5 수확, §6 VLM, §7 2단계, §8 안전, §9 검증, §10 수치). 딸린 스펙:
- 기준 스펙 `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`
- 파지 스펙 `docs/superpowers/specs/2026-09-14-iker-learned-grasp-left-arm-design.md`(개정 2)

**선행 조건:**
- 계획 2b(1단계 파지 env·A 단계 학습)가 브랜치 `iker-front-end` 에 있다.
- A 런 `iker_grasp_c00_a`(RUN_LABEL 같음)가 돌고 있거나 끝나 체크포인트를 남겼다.
- 오른팔 K1 런 `log/rl_games/open-sens/right/iker-shoe/iker_human_c00_k1` 은 비교 기준이고 스모크 체크포인트로도 쓴다(삭제 금지).

**실행 위치:** 워크트리 `~/rl_ws/hdgp-iker`(브랜치 `iker-front-end`). 공유 체크아웃 `~/rl_ws/hdgp` 는 건드리지 않는다.

**검증 상태 (2026-09-14 20:00)**

미러는 스크래치 `scratchpad/plan2c/hdgp` 다: 워크트리 `61983cb8` 의 `source`·`scripts` 복사본에 `assets`·`iker_runs`·`log` 심링크를 걸고, 부모 폴더에 `sim2real` 심링크를 걸었다. 이 계획의 코드 블록은 그 미러에서 실제로 돌린 파일을 `assemble_plan.py` 로 옮겨 넣은 것이다. 새 파일은 전문이고, 수정 파일은 교체 블록이다.

- **순수 테스트:** `194 passed, 1 skipped`(skip 은 오른팔 K1 뱅크의 측면 불일치). 태스크별 누계는 T1 155 · T2 174 · T3 186 · T4 191 · T8 194 이다.
- **컴파일:** 수정·신규 파이썬 파일 17 개 모두 통과.
- **`loop.py init --adopt stage1_a` + `status`**(돌고 있는 A 런, 스크래치 상태 폴더):
  - PID 3109974 를 입양했다.
  - 판정은 `record_gate`(epoch 200, 191~200 구간 잠김 9.02 %, 통과)였다.
  - 그 시각 A 는 epoch 331, 마지막 구간 잠김 25.3 % 였다.
- **Isaac 스모크**(학습 옆, GPU 12.3~12.5 GB 에서 시작):
  - **1단계 캡처·복원(`grasp_smoke.py`):** `SMOKE success capture: valid True, shoe pose error 0.00e+00, restored joint error 0.00e+00, hand target error 0.00e+00, episode length 0` 이 나오고 `GRASP SMOKE passed True`.
  - **평가(`eval_iker.py`,** K1 ep250 체크포인트 + 접근 자세 뱅크, 64 env, 최종 상태 기록): `EVAL {... "episodes": 64, "steps": 199, "success_5cm_end": 0.0 ...}`.
  - **관측 렌더(`observe.py --human-target`):**
    - 출력은 `OBSERVE config 00 source human_target env None margin +55.0 px settle 0.00 mm passed True` 였다.
    - 렌더에서 두 신발이 받침 위에 나란히 보이고, 키포인트는 여백 안이다.
  - **스냅샷 회귀(`snapshot.py --out-dir`, 공용 모듈 전환):** 커밋된 구성 0 `keypoints.json` 과 비교해 world 0.00 mm, 픽셀 0.00 px, 여백 50.6 px 로 같다.
  - **수확(`harvest_grasp_bank.py`,** A ep350, 128 env × 시드 1, `--g-min 1.0 --min-entries 1`, 스크래치 출력):
    - 출력은 `HARVEST seed 1 envs 128 first-episode successes 33 verified 24` 였다.
    - 분위수(q10/50/90):

      | 지표 | 값 |
      |---|---|
      | 신발 상승 | −1.7 / 7.9 / 11.7 cm |
      | 손바닥 상승, 들기 뒤 | 1.9 / 3.8 / 8.0 cm |
      | 손바닥 상승, 유지 뒤 | −1.6 / 8.0 / 10.5 cm |
      | 손바닥 좌표계 미끄럼 | 0.0 / 0.1 / 0.7 cm |
      | 손바닥 기울기 | 1.6 / 12.5 / 42.7° |

    - 리셋은 0 이었다.
    - 마지막 줄은 `... captured 33 verified 24 (min 1) passed True -> .../grasp_bank.json` 이었다.
    - 뱅크 파일은 확인을 통과했다: 관절 이름 정렬, `source learned_grasp`, `side_sign -1.0`, 항목 24 개 = `verified`, 부팅 대조 9 키 포함.
- **처음 실패에서 바뀐 것:**
  - 수확 검증은 세 번 고쳤다(계획 결정 12).
    - **1·2 차**(A ep300 64 env, ep350 128 env): 캡처 11·37 개 중 통과 0 개.
      - 손바닥 상승은 10 cm 명령에 q50 3.5 cm·q90 25 cm 였고, 월드 좌표 미끄럼은 q10 부터 1.1 cm 였다.
      - 조치: 복원 뒤 정착 1 스텝, 손바닥 좌표계 미끄럼.
    - **3 차**(ep350 128 env): 캡처 34 개 중 통과 6 개.
      - 미끄럼은 q50 0.0·q90 0.4 cm 로 파지는 버텼다.
      - 손바닥 상승이 들기 뒤 q50 4.9 cm 에서 유지 뒤 2.4 cm 로 처졌고, 기울기는 q90 43° 였다. 0 행동 유지는 상대 IK 가 매 스텝 지금 자세를 목표로 삼는다.
      - 조치: 정착 자세 +10 cm·방향 유지 목표로의 추종(행동 스케일로 나눠 ±1 클램프).
    - **4 차:** 위 수확 줄. 캡처 33 개 중 통과 24 개(73 %)였다.
  - 관측·스냅샷 첫 실행은 미러 부모에 `sim2real/` 이 없어 `head_camera.load_spec` 에서 멈췄다. 미러 환경 문제이고 코드와 무관하다.
- **돌리지 않은 검증:** 루프 모의 실행(Task 9)과 실제 보정·B·2단계 학습은 실행하면서 처음 돈다.

**계획 결정** (스펙이 정하지 않았거나 스펙 문구를 좁힌 것, 틀리면 드는 비용과 함께)

1. **보정 파일 schema 버그 수정을 Task 1 로 둔다.**
   - `measure_grasp_quality.py` 는 `schema` 없는 JSON 을 쓰고, 1단계 env 는 그 파일을 `run_files.read_json` 으로 읽는다. `read_json` 은 `schema == 1` 을 요구하므로 B 단계는 부팅에서 멈춘다.
   - 비용: 없음.
2. **`LOOP_STATE.json` 에 스펙 키 목록 밖의 `policy` 키를 둔다.**
   - 문턱값·런 라벨·epoch 수를 여기에 담아, 모의 루프가 같은 코드로 다른 값을 쓴다.
   - 비용: 키 하나.
3. **관측 롤아웃과 렌더를 두 스크립트로 나눈다.**
   - 스펙 §4 는 observe.py 가 롤아웃하고 렌더한다고, §3 은 주어진 상태를 렌더한다고 적었다.
   - 롤아웃은 `eval_iker.py --no-noise --final-states`(학습 잡음 없는 결정론 512 env, 첫 에피소드)가 하고, `observe.py` 는 고른 env 의 최종 상태를 렌더한다. 512 env 롤아웃과 카메라 장면을 한 Isaac 프로세스에 싣지 않기 위해서다.
   - 비용: 프로세스 하나 더.
4. **2단계 학습 착수 전에 `env_smoke.py` 를 루프 동작 `run_env_smoke` 로 돌린다.**
   - 입력은 학습 뱅크와 VLM 목표이고, 정책 `stage2_env_smoke` 로 끄고 켠다. 스펙 §9 의 "2단계 env 스모크, 학습 뱅크가 생긴 뒤"를 자동화한 것이다.
   - 뱅크·env 불일치를 3.4 h 학습 전에 잡는다. 스모크의 무행동 낙하 한도(25 %)를 넘으면 사람에게 멈춘다.
   - 비용: 약 3 분.
5. **한 틱이 이어서 하는 동작의 범위.**
   - 스펙 §3 은 "틱 한 번에 다음 동작 하나"라고 했다. 이 계획은 Isaac 기동(`run_*`·`launch_*`)·생성기(`vlm_generate`)·`pause`·`wait` 를 틱당 하나로 센다.
   - 그 앞의 기록·파일 동작(`record_*`·`advance`·`commit_*`·`write_*`·`ingest`·`parse_requery`)은 같은 틱에서 새 판정으로 이어 한다. 한 틱에 `act` 는 최대 8 번이다.
   - 30 분 틱마다 기록 하나씩 하면 VLM 단계만 몇 시간이 걸린다. 매 `act` 는 새 probe 로 다시 판정하므로 멱등성은 그대로다.
   - 비용: 없음.
6. **재질의 프롬프트의 이력 이미지 표식은 `[STAGE_1_IMAGE]` 다.** 스펙 문구 `"Stage 1 image"` 를 그대로 쓰면 프롬프트 줄이 `Stage 1 image: Stage 1 image` 가 된다. 생성기 지시문이 표식마다 파일 경로를 준다.
7. **수확은 이미 시도한 체크포인트보다 새 체크포인트만 시도한다.** 스펙 "새 B 저장 체크포인트". 한 틱에 새 체크포인트가 여럿 있으면 가장 새것을 먼저 쓴다.
8. **`snapshot.py` 의 장면·정착·검사는 `tasks/iker_shoe/head_snapshot.py` 로 옮기고, `observe.py` 가 같이 쓴다.**
   - 동작은 바꾸지 않는다. Task 7 이 구성 0 을 스크래치로 다시 찍어 커밋된 keypoints.json 과 대조한다.
   - rl_games 플레이어 배선은 새 `policy_player.py` 로 수확·평가가 공유한다. 돌고 있는 `measure_grasp_quality.py` 의 배선은 외과적 변경 원칙으로 두고, 그 스크립트에는 schema 수정만 넣는다.
9. **루프 커밋의 `Claude-Session` 트레일러는 `loop.py --session-url` 로 받는다.**
   - 기본값은 이 계획을 쓴 세션이다. 다음 세션은 자기 URL 을 넘긴다.
10. **수확 뱅크는 늘 `config_XX/grasp_bank.json` 에 쓴다.** 정책 `grasp_bank_path` 는 2단계 입력(스모크·학습·평가·영상)만 바꾼다. 모의 루프는 접근 자세 뱅크를 준다.
11. **`commit_bank` 뒤 틱이 `test_grasp_bank_file.py` 를 돌려 보고한다.**
    - 학습 뱅크가 들어오면 측면 skip 이 풀리고 손가락 한계 검사를 받는다. 루프는 코드를 고치지 않으므로 실패는 알리기만 한다.
12. **수확 검증의 팔 동작과 미끄럼 측정을 좁혔다.** 스펙 §5 는 "복원 → 팔 +z 최대(0.02 m/스텝) 5 스텝 → 10 스텝 유지, 신발–손바닥 상대 변화 < 1 cm"다. 미러에서 세 번 돌려 다음 셋을 넣었다.
    - **정착 1 스텝.** 복원 직후 첫 스텝은 이전 상태의 손바닥 자세와 야코비안으로 IK 를 풀어 팔이 튄다(10 cm 명령에 손바닥 상승 q50 3.5 cm·q90 25 cm). 팔 0 행동 한 스텝으로 상태를 새로 고친 뒤 시작 자세를 잰다.
    - **손바닥 좌표계 미끄럼.** 월드 좌표 차이는 손바닥 회전을 미끄럼으로 센다(q10 부터 1.1 cm). 신발 위치를 손바닥 좌표계로 옮겨 비교하면 같은 파지의 미끄럼은 q90 0.4 cm 였다.
    - **목표 추종 유지.** 유지 구간의 0 행동은 상대 IK 가 매 스텝 지금 자세를 목표로 삼아 중력 처짐이 쌓인다. 들기 뒤 4.9 cm 이던 손바닥 상승(q50)이 유지 뒤 2.4 cm 로 줄었고, 기울기는 q90 43° 였다.
      - 그래서 팔 행동을 "정착 자세 +10 cm, 방향 유지" 목표로의 오차를 행동 스케일로 나눠 ±1 로 자른 값으로 둔다.
      - 들기 5 스텝은 스펙대로 +z 최대 속도이고, 유지 10 스텝은 그 목표를 붙든다.
    - 셋을 넣은 뒤 같은 체크포인트(A ep350, 128 env)의 검증 통과가 캡처 34 개 중 6 개에서 33 개 중 24 개로 늘었다.
    - 통과 판정(`grasp_bank.lift_held`: 상승 ≥ 5 cm, 미끄럼 < 1 cm)은 그대로다.
    - 스크립트는 시드마다 상승·미끄럼·손바닥 기울기 분위수를 출력해, 루프가 수확에서 멈췄을 때 사람이 판단할 근거로 남긴다.
    - 비용: 한 스텝(0.1 s)과 목표 추종 계산. 틀리면 정책이 쓰지 않은 제어기로 들어 올린 상태가 뱅크에 들어간다.

## 루프 가동 전 1단계 운영

Task 9 가 루프를 가동하기 전까지는 컨트롤러가 스펙 §4 의 stage1 행을 손으로 적용한다. B 단계 착수는 Task 1 이 들어간 뒤에만 한다(보정 파일 schema).

Task 9 의 가동 단계는 그 시점의 1단계 상태에 맞는 `init` 을 고른다.
- 손으로 보정·B 를 끝냈으면 `--phase harvest --adopt stage1_b`.
- 아니면 `--adopt stage1_a`.

## Global Constraints

- **보상.** IKER 보상 5항(`modules/iker/reward.py`)과 2단계 env 의 보상·종료는 바꾸지 않는다(사용자 결정 "IKER 원형 유지"). 1단계 보상도 이 계획에서 바꾸지 않는다.
- **트랙 격리.**
  - t2r(`modules/t2r`, `scripts/reward_gen`)과 grasp_s2r 을 import 하지 않는다.
  - 루프는 t2r·pour_fabric 트랙과 파일·GPU·크론을 공유하지 않는다.
- **로봇 값.**
  - `modules/robot_profiles.py`·`modules/vendor_gains.py` 에서만 읽는다. 제품 코드에 `r_hj_`/`l_hj_` 를 박지 않는다(테스트 픽스처 제외).
  - 공유 프로필 `tesollo_left_short` 는 고치지 않는다.
- **GPU·동시 실행.**
  - GPU 는 로컬만 쓰고, 학습은 한 번에 하나다.
  - 읽기 전용 Isaac 실행(스모크·보정·수확·평가·관측·영상)은 한 번에 하나씩, env ≤ 512 로 돌린다. 학습 옆에서는 `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` 가 20000 이하일 때만 띄운다.
- **돌고 있는 1단계 학습.** RUN_LABEL `iker_grasp_c00_a`·`iker_grasp_c00_b` 는 구현자가 멈추거나 건드리지 않는다(루프·컨트롤러 몫).
- **프로세스 종료.** `/proc/<pid>/environ` 의 RUN_LABEL 로 찾은 PID 로만 한다. `pkill`·`killall` 금지.
- **순수 테스트:** `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider` (scipy·requests 경고 줄은 무시).
- **Isaac 실행:**
  - 명령: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=<라벨> ../IsaacLab/_isaac_sim/python.sh <스크립트> --headless`.
  - `isaaclab.sh` 는 쓰지 않는다(공유 체크아웃 openarm 을 PYTHONPATH 앞에 넣는다).
  - 10 분이 넘을 수 있는 실행은 백그라운드로 띄우고 짧은 Bash 호출로 로그를 폴링한다. **실행 중에 턴을 끝내지 않는다.**
  - 성공 판정은 출력 표식과 산출 파일로 한다: `GRASP SMOKE passed True`, `SMOKE passed True`, `HARVEST config ... passed`, `EVAL {`, `OBSERVE config ... passed True`, `SNAPSHOT config ... passed True`, `QUALITY ... passed True`, `MAX EPOCHS NUM!`.
- **산출물 위치.**
  - 스모크 산출물은 세션 스크래치패드나 `log/`(gitignored) 에만 쓴다. `iker_runs/shoe_place/config_00/` 의 커밋된 파일을 덮어쓰지 않는다.
  - 영상은 `~/rl_ws/our_source` 에 둔다.
  - 공유 체크아웃 `~/rl_ws/hdgp` 에서는 어떤 명령도 실행하지 않는다.
- **git.**
  - `git add -A`·`git add .` 금지. 각 커밋 단계의 경로만 add 하고, push·병합 금지.
  - 커밋은 따로 Bash 호출로 한다(훅이 `git commit` 과 ` -n` 을 한 호출에서 막는다).
  - 커밋 메시지 끝 두 줄: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- **서브에이전트.** 구현자는 서브에이전트를 띄우지 않는다.

## 파일 구조

| 파일 | 역할 | 태스크 |
|---|---|---|
| `tasks/iker_shoe/grasp_stage.py` (수정) | 보정 파일 문서·읽기(schema), 성공 캡처(`SuccessCapture`·`capture_rows`·`holding_targets`) | 1, 4 |
| `tasks/iker_shoe/tests/test_grasp_stage.py` (수정) | 보정 파일 1개 + 캡처·EMA 고정점 3개 | 1, 4 |
| `tasks/iker_shoe/iker_shoe_grasp_env.py` (수정) | 보정 파일 읽기, 성공 캡처·복원, `_boot_metadata`, `_bottom_z` | 1, 4 |
| `scripts/iker/measure_grasp_quality.py` (수정) | 보정 파일을 schema 문서로 쓴다 | 1 |
| `modules/iker/loop_state.py` (신규) | 상태 스키마·판정·전이 | 2 |
| `modules/iker/tests/test_loop_state.py` (신규) | 19개 | 2 |
| `modules/iker/loop_probe.py`, `loop_vlm.py` (신규) | 세계 읽기, VLM 프롬프트·재질의 파싱 | 3 |
| `modules/iker/tests/test_loop_probe.py`, `test_loop_vlm.py` (신규) | 8개, 4개 | 3 |
| `tasks/iker_shoe/grasp_bank.py` (수정) | 부팅 대조 키, 관절 열 이름 정렬, 학습 뱅크 metadata | 4 |
| `tasks/iker_shoe/tests/test_grasp_bank.py` (수정) | 2개 | 4 |
| `tasks/iker_shoe/iker_shoe_grasp_env_cfg.py` (수정) | `capture_success_states = False` | 4 |
| `scripts/iker/grasp_smoke.py` (수정) | 검사 6: 캡처·복원 | 4 |
| `tasks/iker_shoe/policy_player.py` (신규) | rl_games 플레이어 배선(수확·평가 공유) | 5 |
| `scripts/iker/harvest_grasp_bank.py` (신규) | 학습 파지 뱅크 수확 | 5 |
| `tasks/iker_shoe/config/__init__.py` (수정) | 2단계 id `open-sens_l_iker_shoe[-play]` | 6 |
| `scripts/iker/env_smoke.py` (수정) | `l` id, `--interaction`·`--grasp-bank` | 6 |
| `scripts/iker/eval_iker.py` (신규) | 결정론 평가·최종 상태(스크래치 프로브를 옮김) | 6 |
| `tasks/iker_shoe/head_snapshot.py` (신규) | 헤드 카메라 장면·정착·주석·검사(스냅샷에서 옮김) | 7 |
| `scripts/iker/snapshot.py` (교체) | 공용 모듈 사용, `--out-dir` | 7 |
| `scripts/iker/observe.py` (신규) | 실행 뒤 장면 렌더 | 7 |
| `scripts/iker/loop.py`, `scripts/iker/LOOP_PROMPT.md` (신규) | 루프 CLI, 틱 절차서 | 8 |
| `modules/iker/tests/test_loop_cli.py` (신규) | 3개 | 8 |
| `iker_runs/shoe_place/config_00/loop/{LOOP_STATE.json,history.jsonl}` (생성) | 가동한 루프의 상태 | 9 |

(경로 접두 `source/openarm/openarm/agnostic/` 생략.)

---

### Task 1: 보정 파일 schema — B 단계 부팅 버그

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
- Modify: `scripts/iker/measure_grasp_quality.py`

**Interfaces:**
- Produces: `gs.quality_calibration_document(q_lo: float, q_hi: float, **details) -> dict` (`schema` 포함, `0 <= q_lo < q_hi <= 1` 아니면 `ValueError`), `gs.read_quality_calibration(path) -> tuple[float, float]`.

- [ ] **Step 1: 실패하는 테스트** — `tests/test_grasp_stage.py` 의 import 에 `from openarm.agnostic.modules.iker import run_files` 를 `from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs` 바로 위에 넣고, 파일 끝에 붙인다.

````python
def test_quality_calibration_document_round_trips_and_a_file_without_schema_is_rejected(tmp_path):
    doc = gs.quality_calibration_document(0.12, 0.34, checkpoint="/a/ep250.pth", latch_events=80)
    assert doc["schema"] == 1 and doc["checkpoint"] == "/a/ep250.pth" and (doc["q_lo"], doc["q_hi"]) == (0.12, 0.34)
    path = run_files.write_json(tmp_path / "grasp_quality_calibration.json", doc)
    assert gs.read_quality_calibration(path) == (0.12, 0.34)
    # what measure_grasp_quality.py wrote before 2026-09-14: the environment could not read it
    run_files.write_json(tmp_path / "old.json", {"checkpoint": "/a/ep250.pth", "q_lo": 0.12, "q_hi": 0.34})
    with pytest.raises(ValueError, match="schema"):
        gs.read_quality_calibration(tmp_path / "old.json")
    for lo, hi in ((0.3, 0.3), (-0.1, 0.2), (0.1, 1.2), (math.nan, 0.2)):
        with pytest.raises(ValueError, match="q_lo < q_hi"):
            gs.quality_calibration_document(lo, hi)
    with pytest.raises(ValueError, match="may not set"):
        gs.quality_calibration_document(0.1, 0.2, schema=2)
````

- [ ] **Step 2: 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py -q -p no:cacheprovider -k calibration`
Expected: FAIL — `AttributeError: module 'openarm.agnostic.tasks.iker_shoe.grasp_stage' has no attribute 'quality_calibration_document'`

- [ ] **Step 3: 구현** — `grasp_stage.py` 의 `import torch` 다음 줄에 빈 줄과 `from openarm.agnostic.modules.iker import run_files` 를 넣고, 파일 끝에 붙인다.

````python
def quality_calibration_document(q_lo: float, q_hi: float, **details) -> dict:
    """The grasp quality calibration file (design §5): ``0 <= q_lo < q_hi <= 1`` and the measurement details, under the
    run-file schema the environment reads it with (``run_files.read_json`` rejects a document without it)."""
    reserved = sorted({"schema", "q_lo", "q_hi"} & set(details))
    if reserved:
        raise ValueError(f"calibration details may not set {reserved}")
    lo, hi = float(q_lo), float(q_hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi <= 1.0):
        raise ValueError(f"grasp quality calibration needs 0 <= q_lo < q_hi <= 1, got q_lo {lo}, q_hi {hi}")
    return {"schema": run_files.SCHEMA_VERSION, **details, "q_lo": lo, "q_hi": hi}


def read_quality_calibration(path) -> tuple[float, float]:
    """(q_lo, q_hi) of a file written from ``quality_calibration_document``."""
    doc = run_files.read_json(path)
    checked = quality_calibration_document(doc["q_lo"], doc["q_hi"])
    return checked["q_lo"], checked["q_hi"]
````

- [ ] **Step 4: 통과 확인** — Step 2 명령. Expected: `1 passed`.

- [ ] **Step 5: env 가 이 함수로 읽는다** — `iker_shoe_grasp_env.py` 에서

```python
            calibration = run_files.read_json(calibration_path)
            reward_cfg = replace(reward_cfg, q_lo=float(calibration["q_lo"]), q_hi=float(calibration["q_hi"]))
```

를 다음으로 바꾼다.

```python
            q_lo, q_hi = gs.read_quality_calibration(calibration_path)
            reward_cfg = replace(reward_cfg, q_lo=q_lo, q_hi=q_hi)
```

- [ ] **Step 6: 보정 스크립트가 schema 문서를 쓴다** — `scripts/iker/measure_grasp_quality.py` 를 네 곳 고친다.
  - import:
    ```python
    from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
    ```
    바로 아래에 다음 줄을 넣는다.
    ```python
    from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
    ```
  - `result.update({` 부분:
    ```python
        result.update({
            "q_lo": float(torch.quantile(moments, Q_LO_PERCENTILE)),
            "q_hi": float(torch.quantile(moments, Q_HI_PERCENTILE)),
            "q_quantiles_10_25_50_75_90": [round(float(torch.quantile(moments, p)), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
    ```
    를 다음으로 바꾼다.
    ```python
        q_lo, q_hi = float(torch.quantile(moments, Q_LO_PERCENTILE)), float(torch.quantile(moments, Q_HI_PERCENTILE))
        result.update({
            "q_quantiles_10_25_50_75_90": [round(float(torch.quantile(moments, p)), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
    ```
  - `passed = result["q_lo"] < result["q_hi"]` 를 `passed = q_lo < q_hi` 로 바꾼다.
  - 쓰기와 출력:
    ```python
        if passed:
            run_files.write_json(out, result)
        print(f"QUALITY config {args.config_index:02d} {result} passed {passed} -> {out}", flush=True)
    ```
    를 다음으로 바꾼다.
    ```python
        if passed:
            # the environment reads the file with run_files.read_json, which requires the schema key
            run_files.write_json(out, gs.quality_calibration_document(q_lo, q_hi, **result))
        print(f"QUALITY config {args.config_index:02d} {result} q_lo {q_lo} q_hi {q_hi} passed {passed} -> {out}", flush=True)
    ```

- [ ] **Step 7: 전체 순수 테스트와 컴파일**

Run: 순수 테스트 명령, 그리고 `cd ~/rl_ws/hdgp-iker && python3 -m py_compile scripts/iker/measure_grasp_quality.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
Expected: `155 passed, 1 skipped`, 컴파일 오류 없음.

- [ ] **Step 8: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py scripts/iker/measure_grasp_quality.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "fix(iker): q 보정 파일에 run-file schema — env 가 read_json 으로 읽어 B 단계가 부팅에서 멈추던 문제, 보정 문서·읽기를 grasp_stage 한 곳으로

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 2: 루프 상태 기계 (순수)

**Files:**
- Create: `source/openarm/openarm/agnostic/modules/iker/loop_state.py`
- Create: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py`

**Interfaces:**
- Consumes: `run_files.SCHEMA_VERSION`, `run_files.read_json`, `run_files.write_json`.
- Produces (later tasks use these names exactly):
  - 상수: `SCHEMA`, `TRACK = "iker_shoe_c00"`, `LEARNED_BANK_SOURCE = "learned_grasp"`, `PHASES`, `STATUSES`, `TRAINING_RUNS = ("stage1_a", "stage1_b", "stage2")`, `SIDE_RUNS`, `DEFAULT_POLICY`, `PHASE_RUNS`, `LAUNCH_RUNS`, `SESSION_ACTIONS = ("wait", "vlm_generate")`, `MANUAL_ACTIONS = ("approve", "resume")`, `FILE_ACTIONS`.
  - dataclass: `RunStatus(started, alive, finished, passed, crashed, marker, idle_s)`, `IDLE`, `AttemptStatus(index, gate_passed)`, `Probe(gpu_used_mib, runs, latched, over_rack, checkpoints, boot_reward, calibration, bank_meta, attempts, evals, final_rows, requery, files)`, `Decision(action, reason, params, notes)`.
  - 함수: `bin_mean(points, end_epoch, width)`, `last_epoch(points)`, `first_calibration_checkpoint(latched, checkpoints, threshold, width)`, `eval_epochs(policy)`, `best_eval(evals)`, `pick_observe_env(rows)`, `new_state(now, *, track, phase, policy)`, `validate_state(state)`, `load_state(path)`, `save_state(path, state)`, `decide(state, probe) -> Decision`, `apply(state, decision, now, result) -> dict`.
  - `apply` 의 `result` 키: `run`(런 기록), `stopped_runs`(PID 로 멈춘 런 이름), `path`(뱅크), `passed`(ingest).
  - 판정이 내는 action: `wait`, `pause`, `kill_stale`, `record_gate`, `advance`, `run_calibrate`, `next_calibration`, `commit_calibration`, `launch_b`, `run_harvest`, `record_harvest_miss`, `commit_bank`, `write_prompt`, `vlm_generate`, `ingest`, `commit_interaction`, `run_env_smoke`, `launch_stage2`, `run_eval`, `record_eval`, `run_observe_rollout`, `run_observe_render`, `write_requery_prompt`, `parse_requery`, `run_video`, `store_video`.
  - Probe `files` 키: `stage_prompt`, `observe_snapshot`, `requery_prompt`, `requery_response`, `video_raw`, `video`.

- [ ] **Step 1: 실패하는 테스트** — 아래 파일을 만든다.

````python
"""loop_state — the IKER auto-loop state machine (design 2026-09-14-iker-auto-loop §3, §4, §9; no Isaac)."""

from dataclasses import replace

import pytest

from openarm.agnostic.modules.iker import loop_state as ls

NOW = "2026-09-14T20:00:00"
RUNNING = ls.RunStatus(started=True, alive=True)
ENDED = ls.RunStatus(started=True, finished=True)
DONE_PASS = ls.RunStatus(started=True, finished=True, passed=True, marker="X passed True")
DONE_FAIL = ls.RunStatus(started=True, finished=True, passed=False, marker="X passed False")


def _state(phase="stage1_a", **changes):
    return {**ls.new_state(NOW, phase=phase), **changes}


def _flat(value, upto, start=1):
    return tuple((epoch, value) for epoch in range(start, upto + 1))


def test_new_state_merges_policy_overrides_and_rejects_unknown_keys():
    state = ls.new_state(NOW, policy={"stage2_epochs": 10, "labels": {"stage2": "iker_vlm_mock"}})
    assert state["policy"]["stage2_epochs"] == 10 and state["policy"]["eval_every"] == 250
    assert state["policy"]["labels"] == {"stage1_a": "iker_grasp_c00_a", "stage1_b": "iker_grasp_c00_b", "stage2": "iker_vlm_mock"}
    assert ls.DEFAULT_POLICY["labels"]["stage2"] == "iker_vlm_c00_s1"
    with pytest.raises(ValueError, match="unknown policy keys"):
        ls.new_state(NOW, policy={"gate_epochs": 300})


def test_state_round_trips_and_validation_rejects_inconsistent_states(tmp_path):
    state = _state()
    path = ls.save_state(tmp_path / "loop" / "LOOP_STATE.json", state)
    assert ls.load_state(path) == state
    with pytest.raises(ValueError, match="awaiting"):
        ls.validate_state({**state, "status": "awaiting"})
    with pytest.raises(ValueError, match="phase"):
        ls.validate_state({**state, "phase": "stage3"})
    with pytest.raises(ValueError, match="unknown runs"):
        ls.validate_state({**state, "runs": {"stage1_c": {}}})


def test_bin_mean_takes_the_epochs_after_end_minus_width_through_end():
    points = ((190, 1.0), (191, 2.0), (200, 4.0), (201, 8.0))
    assert ls.bin_mean(points, 200, 10) == 3.0
    assert ls.bin_mean(points, 180, 10) is None
    assert ls.last_epoch(points) == 201 and ls.last_epoch(()) == 0


def test_first_calibration_checkpoint_is_the_earliest_saved_epoch_over_the_threshold():
    latched = _flat(0.05, 100) + _flat(0.12, 150, start=101)
    checkpoints = {50: "a", 100: "b", 150: "c"}
    assert ls.first_calibration_checkpoint(latched, checkpoints, 0.10, 10) == 150
    assert ls.first_calibration_checkpoint(latched, {50: "a", 100: "b"}, 0.10, 10) is None


def test_pick_observe_env_best_eval_and_eval_epochs():
    rows = [{"env": 0, "success": False, "end_dist": 0.01}, {"env": 1, "success": True, "end_dist": 0.04},
            {"env": 2, "success": True, "end_dist": 0.02}, {"env": 3, "success": True, "end_dist": 0.03},
            {"env": 4, "success": True, "end_dist": 0.01}]
    assert ls.pick_observe_env(rows) == 2
    assert ls.pick_observe_env(rows[:1]) is None
    evals = {"250": {"success": 0.4}, "500": {"success": 0.6}, "750": {"success": 0.6}}
    assert ls.best_eval(evals) == (500, {"success": 0.6})
    assert ls.best_eval({}) is None
    assert ls.eval_epochs(ls.DEFAULT_POLICY) == (250, 500, 750)


def test_a_paused_loop_only_waits():
    paused = ls.apply(_state(), ls.Decision("pause", "why", {"needs": "what"}), NOW)
    assert paused["status"] == "awaiting" and paused["awaiting"] == {"reason": "why", "needs": "what", "phase": "stage1_a"}
    assert ls.decide(paused, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="not recorded"):
        ls.apply(_state(), ls.Decision("wait", "nothing"), NOW)


def test_stage1_a_waits_for_the_gate_epoch_then_records_the_last_bin():
    state = _state()
    assert ls.decide(state, ls.Probe(latched=_flat(0.05, 199), runs={"stage1_a": RUNNING})).action == "wait"
    passing = ls.decide(state, ls.Probe(latched=_flat(0.0, 190) + _flat(0.03, 205, start=191), runs={"stage1_a": RUNNING}))
    assert passing.action == "record_gate" and passing.params["passed"] and passing.params["latched"] == pytest.approx(0.03)
    failing = ls.decide(state, ls.Probe(latched=_flat(0.019, 200), runs={"stage1_a": RUNNING}))
    assert failing.action == "record_gate" and not failing.params["passed"]
    paused = ls.apply(state, failing, NOW)
    assert paused["status"] == "awaiting" and paused["gate1"]["passed"] is False


def test_stage1_a_moves_to_calibrate_at_the_first_checkpoint_over_ten_percent():
    state = _state(gate1={"epoch": 200, "latched": 0.08, "passed": True})
    checkpoints = {200: "/a/ep200.pth", 250: "/a/ep250.pth"}
    latched = _flat(0.08, 240) + _flat(0.11, 255, start=241)
    decision = ls.decide(state, ls.Probe(latched=latched, checkpoints=checkpoints, runs={"stage1_a": RUNNING}))
    assert decision.action == "advance" and decision.params == {"to": "calibrate", "epoch": 250, "checkpoint": "/a/ep250.pth"}
    moved = ls.apply(state, decision, NOW)
    assert moved["phase"] == "calibrate"
    assert moved["calibration"] == {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    below = _flat(0.08, 255)
    assert ls.decide(state, ls.Probe(latched=below, checkpoints=checkpoints, runs={"stage1_a": RUNNING})).action == "wait"
    assert ls.decide(state, ls.Probe(latched=below, checkpoints=checkpoints, runs={"stage1_a": ENDED})).action == "pause"


def test_a_crash_pauses_only_the_phase_that_uses_the_run():
    crashed = ls.RunStatus(started=True, crashed=True, marker="Traceback (most recent call last)")
    decision = ls.decide(_state(gate1={"epoch": 200, "latched": 0.08, "passed": True}), ls.Probe(runs={"stage1_a": crashed}))
    assert decision.action == "pause" and decision.reason.startswith("stage1_a crashed: Traceback")
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"stage1_a": crashed})).action == "write_prompt"


def test_a_side_run_alive_two_minutes_after_its_result_is_killed_as_hung():
    hung = ls.RunStatus(started=True, alive=True, finished=True, passed=True, idle_s=121.0)
    decision = ls.decide(_state("vlm_target"), ls.Probe(runs={"video": hung}))
    assert decision.action == "kill_stale" and decision.params == {"run": "video"}
    fresh = replace(hung, idle_s=60.0)
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"video": fresh})).action == "write_prompt"


def test_calibrate_launches_below_the_gpu_limit_and_commits_a_passing_measurement():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    state = _state("calibrate", calibration=calibration)
    probe = ls.Probe(gpu_used_mib=13000, checkpoints={250: "/a/ep250.pth"}, runs={"stage1_a": RUNNING})
    launch = ls.decide(state, probe)
    assert launch.action == "run_calibrate" and launch.params == {"key": "/a/ep250.pth", "checkpoint": "/a/ep250.pth", "epoch": 250}
    assert ls.decide(state, replace(probe, gpu_used_mib=21000)).action == "wait"
    assert ls.decide(state, replace(probe, runs={"stage1_a": RUNNING, "video": RUNNING})).action == "wait"
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_shoe_c00_calibrate", "pid": 11}})
    assert launched["runs"]["calibrate"] == {"label": "iker_shoe_c00_calibrate", "pid": 11, "key": "/a/ep250.pth", "epoch": 250}
    runs = {"stage1_a": RUNNING, "calibrate": DONE_PASS}
    doc = {"checkpoint": "/a/ep250.pth", "q_lo": 0.1, "q_hi": 0.3}
    commit = ls.decide(launched, replace(probe, runs=runs, calibration=doc))
    assert commit.action == "commit_calibration"
    assert ls.apply(launched, commit, NOW)["phase"] == "stage1_b"
    other = {**doc, "checkpoint": "/a/ep200.pth"}
    assert ls.decide(launched, replace(probe, runs=runs, calibration=other)).action == "pause"


def test_calibrate_retries_at_the_next_checkpoint_and_pauses_when_phase_a_is_over():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    state = _state("calibrate", calibration=calibration, runs={"calibrate": {"label": "c", "key": "/a/ep250.pth"}})
    later = {250: "/a/ep250.pth", 300: "/a/ep300.pth"}
    retry = ls.decide(state, ls.Probe(checkpoints=later, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING}))
    assert retry.action == "next_calibration" and retry.params == {"epoch": 300, "checkpoint": "/a/ep300.pth"}
    moved = ls.apply(state, retry, NOW)
    assert moved["calibration"]["tried"] == ["/a/ep250.pth"] and moved["calibration"]["epoch"] == 300
    assert ls.decide(moved, ls.Probe(checkpoints=later, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING})).action == "run_calibrate"
    only = {250: "/a/ep250.pth"}
    assert ls.decide(state, ls.Probe(checkpoints=only, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING})).action == "wait"
    assert ls.decide(state, ls.Probe(checkpoints=only, runs={"calibrate": DONE_FAIL, "stage1_a": ENDED})).action == "pause"


def test_stage1_b_stops_a_launches_from_the_calibrated_checkpoint_and_checks_the_boot_line():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": True}
    state = _state("stage1_b", calibration=calibration, runs={"stage1_a": {"label": "iker_grasp_c00_a", "key": "a"}})
    launch = ls.decide(state, ls.Probe(runs={"stage1_a": RUNNING}))
    assert launch.action == "launch_b" and launch.params["checkpoint"] == "/a/ep250.pth"
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_grasp_c00_b"}, "stopped_runs": ["stage1_a"]})
    assert launched["runs"]["stage1_a"]["stopped"] == NOW and launched["runs"]["stage1_b"]["key"] == "/a/ep250.pth"
    probe = ls.Probe(runs={"stage1_b": RUNNING}, calibration={"checkpoint": "/a/ep250.pth", "q_lo": 0.11, "q_hi": 0.33})
    assert ls.decide(launched, probe).action == "wait"
    good = ls.decide(launched, replace(probe, boot_reward={"g_min": 0.5, "q_lo": 0.11, "q_hi": 0.33}))
    assert good.action == "advance" and good.params == {"to": "harvest"}
    assert ls.decide(launched, replace(probe, boot_reward={"g_min": 1.0, "q_lo": 0.11, "q_hi": 0.33})).action == "pause"


def test_harvest_tries_the_newest_new_checkpoint_and_commits_only_a_learned_bank():
    state = _state("harvest", runs={"stage1_b": {"label": "b", "key": "b"}})
    probe = ls.Probe(checkpoints={300: "/b/ep300.pth", 350: "/b/ep350.pth"}, runs={"stage1_b": RUNNING})
    launch = ls.decide(state, probe)
    assert launch.action == "run_harvest" and launch.params == {"key": "/b/ep350.pth", "checkpoint": "/b/ep350.pth", "epoch": 350}
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_shoe_c00_harvest"}})
    assert ls.decide(launched, replace(probe, runs={"stage1_b": RUNNING, "harvest": RUNNING})).action == "wait"
    miss = ls.decide(launched, replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL}))
    assert miss.action == "record_harvest_miss"
    missed = ls.apply(launched, miss, NOW)
    assert missed["bank"] == {"tried": ["/b/ep350.pth"], "last_epoch": 350}
    assert ls.decide(missed, replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL})).action == "wait"
    newer = replace(probe, checkpoints={**probe.checkpoints, 400: "/b/ep400.pth"}, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL})
    assert ls.decide(missed, newer).params["checkpoint"] == "/b/ep400.pth"
    assert ls.decide(missed, replace(probe, runs={"stage1_b": ENDED, "harvest": DONE_FAIL})).action == "pause"
    learned = {"source": "learned_grasp", "checkpoint": "/b/ep350.pth", "verified": 80}
    passed = replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_PASS}, bank_meta=learned)
    commit = ls.decide(launched, passed)
    assert commit.action == "commit_bank" and commit.params == {"checkpoint": "/b/ep350.pth", "verified": 80}
    committed = ls.apply(launched, commit, NOW, {"path": "/c/grasp_bank.json", "stopped_runs": ["stage1_b"]})
    assert committed["phase"] == "vlm_target" and committed["bank"]["verified"] == 80
    assert committed["runs"]["stage1_b"]["stopped"] == NOW
    assert ls.decide(launched, replace(passed, bank_meta={"side_sign": 1.0, "seed": 0})).action == "pause"


def test_vlm_target_writes_the_prompt_generates_ingests_and_stops_after_three_failures():
    state = _state("vlm_target")
    assert ls.decide(state, ls.Probe()).action == "write_prompt"
    files = {"stage_prompt": True}
    first = ls.decide(state, ls.Probe(files=files))
    assert first.action == "vlm_generate" and first.params == {"kind": "target", "attempt": 0}
    pending = ls.decide(state, ls.Probe(files=files, attempts=(ls.AttemptStatus(0, None),)))
    assert pending.action == "ingest" and pending.params == {"attempt": 0}
    assert ls.apply(state, pending, NOW, {"passed": False})["attempts"] == {"1": 1}
    two_failed = (ls.AttemptStatus(0, False), ls.AttemptStatus(1, False))
    assert ls.decide(state, ls.Probe(files=files, attempts=two_failed)).params == {"kind": "target", "attempt": 2}
    three_failed = two_failed + (ls.AttemptStatus(2, False),)
    assert ls.decide(state, ls.Probe(files=files, attempts=three_failed)).action == "pause"
    passed = ls.decide(state, ls.Probe(files=files, attempts=(ls.AttemptStatus(0, False), ls.AttemptStatus(1, True))))
    assert passed.action == "commit_interaction" and ls.apply(state, passed, NOW)["phase"] == "stage2_train"


def test_stage2_train_smokes_launches_evaluates_and_advances_on_the_success_target():
    state = _state("stage2_train")
    smoke = ls.decide(state, ls.Probe())
    assert smoke.action == "run_env_smoke"
    smoked = ls.apply(state, smoke, NOW, {"run": {"label": "s"}})
    assert ls.decide(smoked, ls.Probe(runs={"env_smoke": DONE_FAIL})).action == "pause"
    runs = {"env_smoke": DONE_PASS}
    assert ls.decide(smoked, ls.Probe(runs={**runs, "stage1_b": RUNNING})).action == "wait"
    launch = ls.decide(smoked, ls.Probe(runs=runs))
    assert launch.action == "launch_stage2" and launch.params == {"key": "iker_vlm_c00_s1"}
    training = ls.apply(smoked, launch, NOW, {"run": {"label": "iker_vlm_c00_s1"}})
    runs = {**runs, "stage2": RUNNING}
    checkpoints = {250: "/s/ep250.pth"}
    evaluate = ls.decide(training, ls.Probe(runs=runs, checkpoints=checkpoints))
    assert evaluate.action == "run_eval" and evaluate.params == {"key": "ep250", "epoch": 250, "checkpoint": "/s/ep250.pth"}
    evaluating = ls.apply(training, evaluate, NOW, {"run": {"label": "e"}})
    summary = {"success_5cm_end": 0.3, "checkpoint": "/s/ep250.pth", "dropped": 0.04}
    record = ls.decide(evaluating, ls.Probe(runs={**runs, "eval": DONE_PASS}, checkpoints=checkpoints, evals={250: summary}))
    assert record.action == "record_eval"
    recorded = ls.apply(evaluating, record, NOW)
    assert recorded["eval"] == {"250": {"success": 0.3, "checkpoint": "/s/ep250.pth", "dropped": 0.04}}
    assert ls.decide(recorded, ls.Probe(runs=runs, checkpoints=checkpoints, evals={250: summary})).action == "wait"
    good = {**recorded, "eval": {**recorded["eval"], "500": {"success": 0.55, "checkpoint": "/s/ep500.pth", "dropped": 0.0}}}
    advance = ls.decide(good, ls.Probe(runs=runs, checkpoints=checkpoints, evals={250: summary, 500: summary}))
    assert advance.action == "advance" and advance.params == {"to": "observe_requery", "epoch": 500}
    low = {key: {"success": 0.3, "checkpoint": f"/s/ep{key}.pth", "dropped": 0.0} for key in ("250", "500", "750")}
    ended = {**recorded, "eval": low}
    all_checkpoints = {250: "a", 500: "b", 750: "c"}
    probe = ls.Probe(runs={"env_smoke": DONE_PASS, "stage2": ENDED}, checkpoints=all_checkpoints, evals={250: {}, 500: {}, 750: {}})
    assert ls.decide(ended, probe).action == "pause"
    assert ls.decide(ls.new_state(NOW, phase="stage2_train", policy={"stage2_env_smoke": False}), ls.Probe()).action == "launch_stage2"


def test_observe_requery_rolls_out_renders_requeries_and_stores_the_video():
    state = _state("observe_requery", eval={"500": {"success": 0.6, "checkpoint": "/s/ep500.pth", "dropped": 0.0}})
    rollout = ls.decide(state, ls.Probe())
    assert rollout.action == "run_observe_rollout" and rollout.params["checkpoint"] == "/s/ep500.pth"
    rolled = ls.apply(state, rollout, NOW, {"run": {"label": "o"}})
    failures = ({"env": 0, "success": False, "end_dist": 0.2},)
    assert ls.decide(rolled, ls.Probe(runs={"observe_rollout": DONE_PASS}, final_rows=failures)).action == "pause"
    rows = ({"env": 3, "success": True, "end_dist": 0.02},)
    render = ls.decide(rolled, ls.Probe(runs={"observe_rollout": DONE_PASS}, final_rows=rows))
    assert render.action == "run_observe_render" and render.params == {"key": "env3", "env": 3}
    files = {"observe_snapshot": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).action == "write_requery_prompt"
    files = {**files, "requery_prompt": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).params == {"kind": "requery"}
    files = {**files, "requery_response": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).action == "parse_requery"
    new_stage = {"done": False, "detail": "new stage: move shoe_other"}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files, requery=new_stage)).action == "pause"
    done = {"done": True, "detail": "done"}
    video = ls.decide(rolled, ls.Probe(final_rows=rows, files=files, requery=done))
    assert video.action == "run_video"
    recording = ls.apply(rolled, video, NOW, {"run": {"label": "v"}})
    probe = ls.Probe(runs={"video": DONE_PASS}, final_rows=rows, files={**files, "video_raw": True}, requery=done)
    assert ls.decide(recording, probe).action == "store_video"
    review = ls.decide(recording, replace(probe, files={**files, "video_raw": True, "video": True}))
    assert review.action == "advance" and review.params == {"to": "completion_review"}


def test_completion_waits_for_the_user_and_resume_moves_the_gate():
    review = ls.decide(_state("completion_review"), ls.Probe())
    assert review.action == "pause"
    done = ls.apply(ls.apply(_state("completion_review"), review, NOW), ls.Decision("approve", "user approved"), NOW)
    assert (done["phase"], done["status"]) == ("done", "done") and ls.decide(done, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="approve"):
        ls.apply(_state(), ls.Decision("approve", "too early"), NOW)
    low = ls.apply(_state(), ls.Decision("record_gate", "low", {"epoch": 200, "latched": 0.01, "passed": False}), NOW)
    resumed = ls.apply(low, ls.Decision("resume", "user", {"policy": {"gate_epoch": 300}}), NOW)
    assert resumed["status"] == "running" and resumed["gate1"] is None and resumed["policy"]["gate_epoch"] == 300
    with pytest.raises(ValueError, match="unknown policy keys"):
        ls.apply(low, ls.Decision("resume", "user", {"policy": {"gate": 1}}), NOW)


def test_over_rack_rising_past_twice_the_latched_rate_is_noted():
    state, latched = _state(), _flat(0.01, 150)
    noted = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.03, 150), runs={"stage1_a": RUNNING}))
    assert noted.action == "wait" and noted.notes[0].startswith("over-rack raised 3.00 % > 2 x latched 1.00 %")
    quiet = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.015, 150), runs={"stage1_a": RUNNING}))
    assert quiet.notes == ()
````

- [ ] **Step 2: 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'loop_state'`

- [ ] **Step 3: 구현** — 아래 파일을 만든다.

````python
"""IKER auto-loop state machine (design 2026-09-14-iker-auto-loop §3, §4, §8). Pure: no Isaac, torch or subprocess.

A tick reads the world into a ``Probe`` (loop_probe.collect), asks ``decide`` for the one next action, lets the CLI
(scripts/iker/loop.py) carry out its side effect, and records the outcome with ``apply``. Every threshold of the loop is
a key of ``DEFAULT_POLICY`` and is stored in LOOP_STATE.json, so a run's rules travel with its state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from . import run_files

SCHEMA = run_files.SCHEMA_VERSION
TRACK = "iker_shoe_c00"
LEARNED_BANK_SOURCE = "learned_grasp"
PHASES = (
    "stage1_a", "calibrate", "stage1_b", "harvest", "vlm_target", "stage2_train", "observe_requery", "completion_review",
    "done",
)
STATUSES = ("running", "awaiting", "done")
TRAINING_RUNS = ("stage1_a", "stage1_b", "stage2")
SIDE_RUNS = ("calibrate", "harvest", "env_smoke", "eval", "observe_rollout", "observe_render", "video")
STALE_AFTER_RESULT_S = 120.0  # an Isaac process alive this long after writing its result is hung (§8)

DEFAULT_POLICY: Mapping = {
    "gate_epoch": 200,  # user decision 8 (2026-09-14): phase A judged again at epoch 200
    "gate_latched": 0.02,
    "calibrate_latched": 0.10,
    "bin_epochs": 10,
    "b_g_min": 0.5,
    "b_max_epochs": 1000,
    "harvest_min": 64,  # user decision 4
    "vlm_max_attempts": 3,  # the first response and two regenerations (base spec §6)
    "stage2_env_smoke": True,
    "stage2_epochs": 750,
    "stage2_num_envs": 4096,
    "eval_every": 250,
    "success_target": 0.5,  # user decision 5
    "side_gpu_limit_mib": 20000,
    "minibatch_size": 0,  # 0 keeps the PPO yaml value
    "grasp_bank_path": "",  # "" = config_XX/grasp_bank.json
    "labels": {"stage1_a": "iker_grasp_c00_a", "stage1_b": "iker_grasp_c00_b", "stage2": "iker_vlm_c00_s1"},
}

PHASE_RUNS: Mapping[str, tuple[str, ...]] = {  # runs whose crash stops the phase
    "stage1_a": ("stage1_a",),
    "calibrate": ("stage1_a", "calibrate"),
    "stage1_b": ("stage1_b",),
    "harvest": ("stage1_b", "harvest"),
    "vlm_target": (),
    "stage2_train": ("env_smoke", "stage2", "eval"),
    "observe_requery": ("observe_rollout", "observe_render", "video"),
    "completion_review": (),
    "done": (),
}
LAUNCH_RUNS: Mapping[str, str] = {
    "run_calibrate": "calibrate", "launch_b": "stage1_b", "run_harvest": "harvest", "run_env_smoke": "env_smoke",
    "launch_stage2": "stage2", "run_eval": "eval", "run_observe_rollout": "observe_rollout",
    "run_observe_render": "observe_render", "run_video": "video",
}
SESSION_ACTIONS = ("wait", "vlm_generate")  # carried out by the tick session, never recorded by the CLI
MANUAL_ACTIONS = ("approve", "resume")  # only on the user's word
FILE_ACTIONS = ("write_prompt", "write_requery_prompt", "parse_requery", "store_video", "kill_stale")  # files are the record


@dataclass(frozen=True)
class RunStatus:
    started: bool = False  # the state holds a launch record for the run
    alive: bool = False  # an Isaac process carries the run's RUN_LABEL
    finished: bool = False  # training: the max-epoch checkpoint or MAX EPOCHS NUM!; side run: its result marker or file
    passed: bool | None = None  # side run: "passed True/False" of the result marker (True for a marker without it)
    crashed: bool = False  # a Traceback or FAILED marker, or the process ended without a result
    marker: str = ""
    idle_s: float = 0.0  # seconds since the run's log last changed


IDLE = RunStatus()


@dataclass(frozen=True)
class AttemptStatus:
    index: int
    gate_passed: bool | None  # None until attempt_NN/gate.json exists


@dataclass(frozen=True)
class Probe:
    gpu_used_mib: int = 0
    runs: Mapping[str, RunStatus] = field(default_factory=dict)
    latched: tuple[tuple[int, float], ...] = ()  # Episode/grasp_episode/latched of the phase's stage-1 run, by epoch
    over_rack: tuple[tuple[int, float], ...] = ()  # Episode/grasp/over_rack_raised_frac, by epoch
    checkpoints: Mapping[int, str] = field(default_factory=dict)  # the phase's training run: epoch -> checkpoint path
    boot_reward: Mapping[str, float] | None = None  # phase B's printed reward: g_min, q_lo, q_hi
    calibration: Mapping | None = None  # grasp_quality_calibration.json
    bank_meta: Mapping | None = None  # metadata of the grasp bank stage 2 loads
    attempts: tuple[AttemptStatus, ...] = ()  # stage-1 VLM attempts holding a response, in order
    evals: Mapping[int, Mapping] = field(default_factory=dict)  # epoch -> eval_iker.py summary
    final_rows: tuple[Mapping, ...] | None = None  # noise-free rollout rows (env, success, end_dist); None before the file
    requery: Mapping | None = None  # observe/requery.json
    files: Mapping[str, bool] = field(default_factory=dict)  # see loop_probe.collect


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    params: Mapping = field(default_factory=dict)
    notes: tuple[str, ...] = ()


# ------------------------------------------------------------------ measures


def bin_mean(points: Sequence[tuple[int, float]], end_epoch: int, width: int) -> float | None:
    """Mean of the values logged at epochs ``end_epoch - width < e <= end_epoch``; None without any."""
    values = [value for epoch, value in points if end_epoch - width < epoch <= end_epoch]
    return sum(values) / len(values) if values else None


def last_epoch(points: Sequence[tuple[int, float]]) -> int:
    return max((epoch for epoch, _ in points), default=0)


def first_calibration_checkpoint(
    latched: Sequence[tuple[int, float]], checkpoints: Mapping[int, str], threshold: float, width: int
) -> int | None:
    """The earliest saved epoch whose bin of latched episodes reaches ``threshold`` (§4 stage1_a)."""
    for epoch in sorted(checkpoints):
        mean = bin_mean(latched, epoch, width)
        if mean is not None and mean >= threshold:
            return epoch
    return None


def eval_epochs(policy: Mapping) -> tuple[int, ...]:
    return tuple(range(policy["eval_every"], policy["stage2_epochs"] + 1, policy["eval_every"]))


def best_eval(evals: Mapping[str, Mapping]) -> tuple[int, Mapping] | None:
    """(epoch, record) with the highest success; the earlier epoch wins a tie."""
    if not evals:
        return None
    key = max(evals, key=lambda k: (evals[k]["success"], -int(k)))
    return int(key), evals[key]


def pick_observe_env(rows: Sequence[Mapping]) -> int | None:
    """The succeeded env with the (lower) median end distance (§4 observe_requery); None without a success."""
    successes = sorted((row for row in rows if row["success"]), key=lambda row: (row["end_dist"], row["env"]))
    return int(successes[(len(successes) - 1) // 2]["env"]) if successes else None


# --------------------------------------------------------------------- state


def new_state(now: str, *, track: str = TRACK, phase: str = "stage1_a", policy: Mapping | None = None) -> dict:
    overrides = copy.deepcopy(dict(policy or {}))
    unknown = sorted(set(overrides) - set(DEFAULT_POLICY))
    if unknown:
        raise ValueError(f"unknown policy keys {unknown}")
    labels = {**DEFAULT_POLICY["labels"], **overrides.pop("labels", {})}
    state = {
        "schema": SCHEMA, "track": track, "phase": phase, "status": "running", "awaiting": None, "stage": 1,
        "policy": {**copy.deepcopy(dict(DEFAULT_POLICY)), **overrides, "labels": labels},
        "runs": {}, "gate1": None, "calibration": None, "bank": None, "attempts": {"1": 0}, "eval": {}, "updated": now,
    }
    validate_state(state)
    return state


def validate_state(state: Mapping) -> None:
    if state.get("schema") != SCHEMA:
        raise ValueError(f"loop state schema {state.get('schema')} != {SCHEMA}")
    if state.get("phase") not in PHASES:
        raise ValueError(f"unknown phase {state.get('phase')!r}")
    if state.get("status") not in STATUSES:
        raise ValueError(f"unknown status {state.get('status')!r}")
    if (state["status"] == "awaiting") != (state.get("awaiting") is not None):
        raise ValueError("an awaiting reason must be set exactly while the status is awaiting")
    policy = state.get("policy", {})
    missing = sorted(set(DEFAULT_POLICY) - set(policy)) + sorted(set(TRAINING_RUNS) - set(policy.get("labels", {})))
    if missing:
        raise ValueError(f"policy lacks {missing}")
    unknown = sorted(set(state.get("runs", {})) - set(TRAINING_RUNS) - set(SIDE_RUNS))
    if unknown:
        raise ValueError(f"unknown runs {unknown}")


def load_state(path) -> dict:
    state = run_files.read_json(path)
    validate_state(state)
    return state


def save_state(path, state: Mapping):
    validate_state(state)
    return run_files.write_json(path, state)


# -------------------------------------------------------------------- decide


def decide(state: Mapping, probe: Probe) -> Decision:
    """The one next action of the loop (§3 tick step 2)."""
    if state["status"] != "running":
        reason = (state.get("awaiting") or {}).get("reason")
        return _wait(f"loop is {state['status']}" + (f": {reason}" if reason else ""))
    hung = next((name for name in SIDE_RUNS if _hung(probe.runs.get(name, IDLE))), None)
    if hung is not None:
        idle = probe.runs[hung].idle_s
        return Decision("kill_stale", f"{hung} wrote its result {idle:.0f} s ago and is still alive", {"run": hung})
    phase = state["phase"]
    crashed = next((name for name in PHASE_RUNS[phase] if probe.runs.get(name, IDLE).crashed), None)
    if crashed is not None:
        return _pause(f"{crashed} crashed: {probe.runs[crashed].marker or 'ended without a result'}",
                      "read the end of its log; the loop never edits code - fix the cause, then `loop.py act resume`")
    decision = _DECIDERS[phase](state, probe)
    if phase in ("stage1_a", "calibrate", "stage1_b", "harvest"):
        decision = replace(decision, notes=decision.notes + _over_rack_note(probe, state["policy"]["bin_epochs"]))
    return decision


def _wait(reason: str) -> Decision:
    return Decision("wait", reason)


def _pause(reason: str, needs: str) -> Decision:
    return Decision("pause", reason, {"needs": needs})


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f} %"


def _hung(run: RunStatus) -> bool:
    return run.alive and run.finished and run.idle_s > STALE_AFTER_RESULT_S


def _launched(state: Mapping, run: str, key: str) -> bool:
    return state["runs"].get(run, {}).get("key") == key


def _busy_side_run(probe: Probe) -> str | None:
    return next((name for name in SIDE_RUNS if probe.runs.get(name, IDLE).alive), None)


def _side_launch(state: Mapping, probe: Probe, action: str, params: Mapping) -> Decision:
    """A read-only Isaac run: one at a time, next to training only below the GPU memory limit (§8)."""
    busy = _busy_side_run(probe)
    if busy is not None:
        return _wait(f"{action} waits for the side run {busy}")
    limit = state["policy"]["side_gpu_limit_mib"]
    if probe.gpu_used_mib > limit:
        return _wait(f"{action} waits: GPU memory {probe.gpu_used_mib} MiB > {limit} MiB")
    return Decision(action, f"{action} {params['key']}", params)


def _over_rack_note(probe: Probe, width: int) -> tuple[str, ...]:
    end = last_epoch(probe.latched)
    latched, over = bin_mean(probe.latched, end, width), bin_mean(probe.over_rack, end, width)
    if latched is None or over is None or over <= 2.0 * latched:
        return ()
    return (f"over-rack raised {_pct(over)} > 2 x latched {_pct(latched)} over epochs {end - width + 1}-{end}",)


def _stage1_a(state: Mapping, probe: Probe) -> Decision:
    policy, width = state["policy"], state["policy"]["bin_epochs"]
    last = last_epoch(probe.latched)
    if state["gate1"] is None:
        if last < policy["gate_epoch"]:
            return _wait(f"phase A at epoch {last}, gate at epoch {policy['gate_epoch']}")
        value = bin_mean(probe.latched, policy["gate_epoch"], width)
        passed = value is not None and value >= policy["gate_latched"]
        return Decision("record_gate", f"epoch {policy['gate_epoch']} latched {_pct(value)} vs {_pct(policy['gate_latched'])}",
                        {"epoch": policy["gate_epoch"], "latched": value, "passed": passed})
    epoch = first_calibration_checkpoint(probe.latched, probe.checkpoints, policy["calibrate_latched"], width)
    if epoch is not None:
        return Decision("advance", f"checkpoint ep {epoch} latched {_pct(bin_mean(probe.latched, epoch, width))}",
                        {"to": "calibrate", "epoch": epoch, "checkpoint": probe.checkpoints[epoch]})
    if not probe.runs.get("stage1_a", IDLE).alive:
        return _pause(f"phase A ended at epoch {last} with no checkpoint at latched >= {_pct(policy['calibrate_latched'])}",
                      "decide whether to extend phase A or revise stage 1")
    return _wait(f"phase A epoch {last}, latched {_pct(bin_mean(probe.latched, last, width))}")


def _calibrate(state: Mapping, probe: Probe) -> Decision:
    calibration = state["calibration"]
    checkpoint, epoch = calibration["checkpoint"], calibration["epoch"]
    if not _launched(state, "calibrate", checkpoint):
        return _side_launch(state, probe, "run_calibrate", {"key": checkpoint, "checkpoint": checkpoint, "epoch": epoch})
    run = probe.runs.get("calibrate", IDLE)
    if not run.finished:
        return _wait(f"measuring grasp quality at ep {epoch}")
    if run.passed:
        doc = probe.calibration
        if doc is None or doc.get("checkpoint") != checkpoint:
            return _pause("QUALITY passed but the calibration file is missing or names another checkpoint",
                          "inspect grasp_quality_calibration.json")
        return Decision("commit_calibration", f"q_lo {doc['q_lo']:.4f} q_hi {doc['q_hi']:.4f} at ep {epoch}", {"checkpoint": checkpoint})
    later = sorted(e for e in probe.checkpoints if e > epoch)
    if later:
        return Decision("next_calibration", f"ep {epoch}: {run.marker}", {"epoch": later[0], "checkpoint": probe.checkpoints[later[0]]})
    if probe.runs.get("stage1_a", IDLE).alive:
        return _wait(f"ep {epoch} could not calibrate; waiting for the next phase-A checkpoint")
    return _pause("grasp quality calibration failed on every phase-A checkpoint", "decide on phase A before phase B")


def _stage1_b(state: Mapping, probe: Probe) -> Decision:
    policy, calibration = state["policy"], state["calibration"]
    if "stage1_b" not in state["runs"]:
        busy = _busy_side_run(probe)
        if busy is not None:
            return _wait(f"launch_b waits for the side run {busy}")
        return Decision("launch_b", f"resume ep {calibration['epoch']} with g_min {policy['b_g_min']}",
                        {"key": calibration["checkpoint"], "checkpoint": calibration["checkpoint"]})
    if probe.boot_reward is None:
        return _wait("phase B has not printed its reward line yet")
    if probe.calibration is None:
        return _pause("phase B runs but the calibration file is gone", "restore grasp_quality_calibration.json from git")
    expected = {"g_min": policy["b_g_min"], "q_lo": probe.calibration["q_lo"], "q_hi": probe.calibration["q_hi"]}
    if any(not math.isclose(probe.boot_reward.get(key, math.nan), value, rel_tol=0.0, abs_tol=1e-9) for key, value in expected.items()):
        return _pause(f"phase B booted with {dict(probe.boot_reward)}, expected {expected}",
                      "stop phase B by PID and relaunch it with the calibration")
    return Decision("advance", f"phase B reward g_min {expected['g_min']} q_lo {expected['q_lo']:.4f} q_hi {expected['q_hi']:.4f}",
                    {"to": "harvest"})


def _harvest(state: Mapping, probe: Probe) -> Decision:
    bank = state["bank"] or {"tried": [], "last_epoch": 0}
    record, run = state["runs"].get("harvest"), probe.runs.get("harvest", IDLE)
    if record is not None and record["key"] not in bank["tried"]:
        if not run.finished:
            return _wait(f"harvesting ep {record['epoch']}")
        if run.passed:
            meta = probe.bank_meta or {}
            if meta.get("source") != LEARNED_BANK_SOURCE or meta.get("checkpoint") != record["key"]:
                return _pause("HARVEST passed but the grasp bank is not the learned bank of that checkpoint", "inspect grasp_bank.json")
            return Decision("commit_bank", f"{meta['verified']} verified grasps at ep {record['epoch']}",
                            {"checkpoint": record["key"], "verified": meta["verified"]})
        return Decision("record_harvest_miss", f"ep {record['epoch']}: {run.marker}", {"checkpoint": record["key"], "epoch": record["epoch"]})
    newer = [epoch for epoch in sorted(probe.checkpoints) if epoch > bank["last_epoch"]]
    if newer:
        epoch = newer[-1]
        return _side_launch(state, probe, "run_harvest", {"key": probe.checkpoints[epoch], "checkpoint": probe.checkpoints[epoch], "epoch": epoch})
    if not probe.runs.get("stage1_b", IDLE).alive:
        return _pause(f"phase B ended before a checkpoint gave {state['policy']['harvest_min']} verified grasps",
                      "decide on stage 1 before harvesting again")
    return _wait("waiting for the next phase-B checkpoint")


def _vlm_target(state: Mapping, probe: Probe) -> Decision:
    if not probe.files.get("stage_prompt"):
        return Decision("write_prompt", "the stage-1 prompt, snapshot and keypoints are not in the stage directory")
    attempts = probe.attempts
    passed = next((attempt for attempt in attempts if attempt.gate_passed), None)
    if passed is not None:
        return Decision("commit_interaction", f"attempt {passed.index:02d} passed the gate", {"attempt": passed.index})
    if attempts and attempts[-1].gate_passed is None:
        return Decision("ingest", f"attempt {attempts[-1].index:02d} awaits the gate", {"attempt": attempts[-1].index})
    if len(attempts) >= state["policy"]["vlm_max_attempts"]:
        return _pause(f"{len(attempts)} VLM responses failed the gate",
                      "read attempt_*/gate.json; resume with a higher vlm_max_attempts, or stop")
    return Decision("vlm_generate", f"attempt {len(attempts):02d}", {"kind": "target", "attempt": len(attempts)})


def _stage2_train(state: Mapping, probe: Probe) -> Decision:
    policy = state["policy"]
    if policy["stage2_env_smoke"]:
        if not _launched(state, "env_smoke", "vlm_target"):
            return _side_launch(state, probe, "run_env_smoke", {"key": "vlm_target"})
        smoke = probe.runs.get("env_smoke", IDLE)
        if not smoke.finished:
            return _wait("stage-2 environment smoke running")
        if not smoke.passed:
            return _pause(f"stage-2 environment smoke failed: {smoke.marker}", "read its SMOKE CHECK FAILED lines")
    if "stage2" not in state["runs"]:
        if any(probe.runs.get(name, IDLE).alive for name in ("stage1_a", "stage1_b")):
            return _wait("a stage-1 training is still alive")
        return Decision("launch_stage2", f"{policy['stage2_epochs']} epochs x {policy['stage2_num_envs']} envs", {"key": policy["labels"]["stage2"]})
    new = sorted(epoch for epoch in probe.evals if str(epoch) not in state["eval"])
    if new:
        summary = probe.evals[new[0]]
        return Decision("record_eval", f"ep {new[0]} success {_pct(summary['success_5cm_end'])}", {"epoch": new[0], "summary": dict(summary)})
    best = best_eval(state["eval"])
    if best is not None and best[1]["success"] >= policy["success_target"]:
        return Decision("advance", f"ep {best[0]} success {_pct(best[1]['success'])}", {"to": "observe_requery", "epoch": best[0]})
    due = [epoch for epoch in eval_epochs(policy) if epoch in probe.checkpoints and str(epoch) not in state["eval"]]
    if due:
        epoch = due[0]
        if _launched(state, "eval", f"ep{epoch}"):
            if not probe.runs.get("eval", IDLE).finished:
                return _wait(f"evaluating ep {epoch}")
            return _pause(f"the evaluation of ep {epoch} printed its result but wrote no summary", "inspect the eval log and output path")
        return _side_launch(state, probe, "run_eval", {"key": f"ep{epoch}", "epoch": epoch, "checkpoint": probe.checkpoints[epoch]})
    if not probe.runs.get("stage2", IDLE).alive and len(state["eval"]) == len(eval_epochs(policy)):
        return _pause(f"stage 2 ended with best success {_pct(best[1]['success']) if best else 'n/a'} < {_pct(policy['success_target'])}",
                      "decide on stage 2: train longer, query a new target, or stop")
    return _wait(f"stage 2 training; evaluated epochs {sorted(int(k) for k in state['eval'])}")


def _observe_requery(state: Mapping, probe: Probe) -> Decision:
    chosen = best_eval(state["eval"])
    if chosen is None:
        return _pause("observe_requery has no stage-2 evaluation to pick a checkpoint from", "inspect LOOP_STATE.json eval")
    epoch, checkpoint = chosen[0], chosen[1]["checkpoint"]
    if probe.final_rows is None:
        if _launched(state, "observe_rollout", checkpoint):
            if not probe.runs.get("observe_rollout", IDLE).finished:
                return _wait("noise-free observation rollout running")
            return _pause("the observation rollout finished without final states", "inspect the observe_rollout log")
        return _side_launch(state, probe, "run_observe_rollout", {"key": checkpoint, "checkpoint": checkpoint, "epoch": epoch})
    env = pick_observe_env(probe.final_rows)
    if env is None:
        return _pause(f"no env succeeded in the noise-free rollout of ep {epoch}", "watch a play video of the checkpoint; decide on stage 2")
    files = probe.files
    if not files.get("observe_snapshot"):
        if _launched(state, "observe_render", f"env{env}"):
            run = probe.runs.get("observe_render", IDLE)
            if not run.finished:
                return _wait(f"rendering env {env}")
            return _pause(f"the observation render failed: {run.marker}", "read its OBSERVE CHECK FAILED lines")
        return _side_launch(state, probe, "run_observe_render", {"key": f"env{env}", "env": env})
    if not files.get("requery_prompt"):
        return Decision("write_requery_prompt", "multi-step prompt with the stage-1 code as history")
    if not files.get("requery_response"):
        return Decision("vlm_generate", "requery on the executed scene", {"kind": "requery"})
    if probe.requery is None:
        return Decision("parse_requery", "the requery response is not parsed yet")
    if not probe.requery["done"]:
        return _pause(f"the VLM did not report done: {probe.requery['detail']}", "a further stage is outside this loop; decide by hand")
    if not files.get("video"):
        if _launched(state, "video", checkpoint):
            if probe.runs.get("video", IDLE).alive:
                return _wait("recording the play video")
            if files.get("video_raw"):
                return Decision("store_video", "copy the play video to our_source", {"checkpoint": checkpoint, "epoch": epoch})
            return _pause("the play video run ended without a file", "inspect the video log")
        return _side_launch(state, probe, "run_video", {"key": checkpoint, "checkpoint": checkpoint, "epoch": epoch})
    return Decision("advance", "the VLM reported done and the video is stored", {"to": "completion_review"})


def _completion_review(state: Mapping, probe: Probe) -> Decision:
    return _pause("stage 2 met the success target and the VLM reported done",
                  "watch the video named in stage_01/video.txt; approve with `loop.py act approve`")


_DECIDERS = {
    "stage1_a": _stage1_a, "calibrate": _calibrate, "stage1_b": _stage1_b, "harvest": _harvest, "vlm_target": _vlm_target,
    "stage2_train": _stage2_train, "observe_requery": _observe_requery, "completion_review": _completion_review,
    "done": lambda state, probe: _wait("done"),
}


# --------------------------------------------------------------------- apply


def apply(state: Mapping, decision: Decision, now: str, result: Mapping | None = None) -> dict:
    """The state after ``decision`` was carried out; ``result`` is what the CLI's side effect returned
    (``run``: a launch record, ``stopped_runs``: runs it stopped by PID, ``passed``: an ingest verdict)."""
    action, params, outcome = decision.action, dict(decision.params), dict(result or {})
    known = action in LAUNCH_RUNS or action in _APPLIERS or action in FILE_ACTIONS
    if not known:
        raise ValueError(f"action {action!r} is not recorded by the loop state")
    new = copy.deepcopy(dict(state))
    new["updated"] = now
    if action in LAUNCH_RUNS:
        new["runs"][LAUNCH_RUNS[action]] = {**outcome.get("run", {}), "key": params["key"], "epoch": params.get("epoch")}
    for name in outcome.get("stopped_runs", ()):
        if name in new["runs"]:
            new["runs"][name] = {**new["runs"][name], "stopped": now}
    if action in _APPLIERS:
        _APPLIERS[action](new, decision, params, outcome)
    validate_state(new)
    return new


def _apply_pause(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["status"], new["awaiting"] = "awaiting", {"reason": decision.reason, "needs": params["needs"], "phase": new["phase"]}


def _apply_record_gate(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["gate1"] = {"epoch": params["epoch"], "latched": params["latched"], "passed": params["passed"]}
    if not params["passed"]:
        _apply_pause(new, decision, {"needs": "phase A missed the gate: resume to go on (a new gate_epoch judges again), or stop it"}, outcome)


def _apply_advance(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["phase"] = params["to"]
    if params["to"] == "calibrate":
        new["calibration"] = {"checkpoint": params["checkpoint"], "epoch": params["epoch"], "tried": [], "passed": False}


def _apply_next_calibration(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    calibration = new["calibration"]
    calibration["tried"].append(calibration["checkpoint"])
    calibration["checkpoint"], calibration["epoch"] = params["checkpoint"], params["epoch"]


def _apply_commit_calibration(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["calibration"]["passed"] = True
    new["phase"] = "stage1_b"


def _apply_record_harvest_miss(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    bank = new["bank"] or {"tried": [], "last_epoch": 0}
    bank["tried"].append(params["checkpoint"])
    bank["last_epoch"] = max(bank["last_epoch"], params["epoch"])
    new["bank"] = bank


def _apply_commit_bank(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    bank = new["bank"] or {"tried": [], "last_epoch": 0}
    new["bank"] = {**bank, "checkpoint": params["checkpoint"], "verified": params["verified"], "path": outcome.get("path")}
    new["phase"] = "vlm_target"


def _apply_ingest(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["attempts"]["1"] = max(new["attempts"].get("1", 0), params["attempt"] + 1)


def _apply_commit_interaction(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["phase"] = "stage2_train"


def _apply_record_eval(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    summary = params["summary"]
    new["eval"][str(params["epoch"])] = {
        "success": summary["success_5cm_end"], "checkpoint": summary["checkpoint"], "dropped": summary.get("dropped"),
    }


def _apply_approve(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    if new["phase"] != "completion_review":
        raise ValueError(f"approve is for completion_review; the loop is at {new['phase']}")
    new["phase"], new["status"], new["awaiting"] = "done", "done", None


def _apply_resume(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    updates = copy.deepcopy(dict(params.get("policy", {})))
    unknown = sorted(set(updates) - set(DEFAULT_POLICY))
    if unknown:
        raise ValueError(f"unknown policy keys {unknown}")
    if updates.get("gate_epoch", new["policy"]["gate_epoch"]) != new["policy"]["gate_epoch"]:
        new["gate1"] = None
    labels = {**new["policy"]["labels"], **updates.pop("labels", {})}
    new["policy"] = {**new["policy"], **updates, "labels": labels}
    new["status"], new["awaiting"] = "running", None


_APPLIERS = {
    "pause": _apply_pause, "record_gate": _apply_record_gate, "advance": _apply_advance,
    "next_calibration": _apply_next_calibration, "commit_calibration": _apply_commit_calibration,
    "record_harvest_miss": _apply_record_harvest_miss, "commit_bank": _apply_commit_bank, "ingest": _apply_ingest,
    "commit_interaction": _apply_commit_interaction, "record_eval": _apply_record_eval, "approve": _apply_approve,
    "resume": _apply_resume,
}
````

- [ ] **Step 4: 통과 확인** — Step 2 명령. Expected: `19 passed`.

- [ ] **Step 5: 전체 순수 테스트** — Expected: `174 passed, 1 skipped`.

- [ ] **Step 6: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/modules/iker/loop_state.py source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 자동 루프 상태 기계 — phase 판정(게이트·보정·B·수확·VLM·2단계·재질의·완수)과 전이, 문턱값은 상태에 싣는 정책으로

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 3: 세계 읽기와 VLM 단계 (순수)

**Files:**
- Create: `source/openarm/openarm/agnostic/modules/iker/loop_probe.py`
- Create: `source/openarm/openarm/agnostic/modules/iker/loop_vlm.py`
- Create: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_probe.py`
- Create: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_vlm.py`

**Interfaces:**
- Consumes: Task 2 의 `loop_state` 전부. `interaction.extract_code_block`, `interaction.run_interaction`, `interaction.InteractionError`, `prompts.fill_single_step`, `prompts.fill_multi_step`, `prompts.StageRecord`, `prompts.IMAGE_MARKER`.
- Produces (`loop_probe`):
  - `LoopPaths.of(root, config_index, track, state_dir=None)` 와 그 속성·메서드.
    - 속성: `state_file`, `history_file`, `stage_dir`, `observe_dir`, `interaction_file`, `video_note`, `calibration_file`, `final_states_file`, `harvest_bank_file`.
    - 메서드: `bank_file(policy)`, `task_dir(run)`, `train_log(label)`, `side_log(run, tag)`, `eval_file(epoch)`, `eval_rows_file(epoch)`.
  - 함수: `read_tail`, `list_checkpoints(nn_dir) -> dict[int, str]`, `find_run_dir(task_dir, label, started_s)`, `label_pids(label, proc_root=Path("/proc")) -> list[int]`, `parse_gpu_used_mib(text)`, `boot_reward(text)`, `training_status(...)`, `side_status(...)`, `attempts(stage_dir)`, `evals(stage_dir)`, `final_rows(path)`, `stage2_video(state, paths)`, `collect(state, paths, *, now_s, gpu_used_mib, load_events, proc_root) -> Probe`.
  - 상수: `LATCHED_TAG`, `OVER_RACK_TAG`, `ISAAC_PYTHON_MARK`.
- Produces (`loop_vlm`): `STAGE_IMAGE_MARKER = "[STAGE_1_IMAGE]"`, `target_prompt(task)`, `requery_prompt(task, stage_response)`, `requery_result(response, keypoints) -> {"done", "detail"}`, `generator_brief(prompt_path, response_path, images) -> str`.
- 파일 규약 (루프 산출물):
  - `stage_01/eval_epNNNN.json` = `{"schema": 1, "summary": {...}}`
  - `log/iker_loop/<track>/observe_final_states.json` = `{"schema": 1, "checkpoint", "noise", "joint_names", "rows": [{"env", "success", "end_dist", "joint_pos", "joint_target", "shoe_move", "shoe_other"}]}`
  - `stage_01/observe/requery.json` = `{"schema": 1, "done", "detail"}`
  - 사이드 런 결과 표식: `QUALITY config`, `HARVEST config`, `SMOKE passed`, `EVAL {`, `OBSERVE config`. 예외 표식: `<이름> FAILED`.

- [ ] **Step 1: 실패하는 테스트** — 아래 두 파일을 만든다.

````python
"""loop_probe — reading runs, logs, checkpoints and loop files into a Probe (design auto-loop §3, §8; no Isaac)."""

import os

from openarm.agnostic.modules.iker import loop_probe as lp
from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import run_files

ISAAC = b"/home/u/IsaacLab/_isaac_sim/kit/python/bin/python3\0scripts/reinforcement_learning/rl_games/train.py\0"


def _proc(root, pid, label, cmdline):
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / "environ").write_bytes(b"HOME=/h\0RUN_LABEL=" + label.encode() + b"\0")
    (entry / "cmdline").write_bytes(cmdline)


def test_list_checkpoints_reads_saved_and_max_epoch_names(tmp_path):
    nn = tmp_path / "nn"
    nn.mkdir()
    for name in ("last_open-sens_l_iker_shoe_grasp_ep_50_rew_49.66499.pth", "last_open-sens_l_iker_shoe_ep_750_rew__582.5_.pth",
                 "open-sens_l_iker_shoe_grasp.pth", "last_x_frame_100_rew_1.pth"):
        (nn / name).write_bytes(b"")
    found = lp.list_checkpoints(nn)
    assert sorted(found) == [50, 750] and found[50].endswith("ep_50_rew_49.66499.pth") and os.path.isabs(found[50])
    assert lp.list_checkpoints(tmp_path / "missing") == {}


def test_label_pids_match_the_exact_label_on_isaac_python_only(tmp_path):
    proc = tmp_path / "proc"
    _proc(proc, 11, "iker_grasp_c00_a", b"/bin/bash\0../IsaacLab/_isaac_sim/python.sh\0train.py\0")
    _proc(proc, 12, "iker_grasp_c00_a", ISAAC)
    _proc(proc, 13, "iker_grasp_c00_ab", ISAAC)
    (proc / "self").mkdir()
    assert lp.label_pids("iker_grasp_c00_a", proc) == [12]


def test_training_status_finishes_on_the_marker_or_max_epoch_and_crashes_when_it_dies_early():
    running = lp.training_status("epoch 3", True, {}, 1000, 5.0)
    assert running.alive and not running.finished and not running.crashed and running.idle_s == 5.0
    assert lp.training_status("MAX EPOCHS NUM!", False, {}, None, 0.0).finished
    assert lp.training_status("", False, {1000: "p"}, 1000, 0.0).finished
    died = lp.training_status("", False, {500: "p"}, 1000, 0.0)
    assert died.crashed and died.marker == ""
    trace = lp.training_status("Traceback (most recent call last):\nKeyError: 'x'\n", True, {}, 1000, 0.0)
    assert trace.crashed and trace.marker.startswith("Traceback")


def test_side_status_reads_the_last_result_marker_and_the_excepthook_marker():
    failed = lp.side_status("calibrate", "QUALITY config 00 {'envs': 512} passed False: 12 latch/success moments < 64\n", False, False, 1.0)
    assert failed.finished and failed.passed is False and not failed.crashed
    text = "HARVEST seed 0 envs 512 first-episode successes 40 verified 30\nHARVEST config 00 checkpoint x captured 90 verified 70 (min 64) passed True -> out\n"
    passed = lp.side_status("harvest", text, True, False, 130.0)
    assert passed.finished and passed.passed and passed.alive and passed.marker.startswith("HARVEST config")
    evaluated = lp.side_status("eval", 'EVAL {"success_5cm_end": 0.3}\n', False, False, 0.0)
    assert evaluated.finished and evaluated.passed
    broke = lp.side_status("observe_render", "Traceback\nOBSERVE FAILED\n", False, False, 0.0)
    assert broke.crashed and broke.marker == "OBSERVE FAILED"
    running = lp.side_status("env_smoke", "SMOKE obs (16, 38) finite True bank 80\n", True, False, 0.0)
    assert not running.finished and not running.crashed
    video = lp.side_status("video", "Traceback (most recent call last): harmless extension warning\n", False, True, 0.0)
    assert video.finished and not video.crashed


def test_boot_reward_parses_the_last_reward_line_and_gpu_memory_reading():
    line = "[iker_grasp] reward Stage1RewardCfg(palm_scale=50.0, latch_steps=3, g_min=0.5, q_lo=0.1234567890123, q_hi=0.4) · idle income 0\n"
    assert lp.boot_reward("noise\n" + line) == {"g_min": 0.5, "q_lo": 0.1234567890123, "q_hi": 0.4}
    assert lp.boot_reward("nothing") is None
    assert lp.parse_gpu_used_mib("13041\n") == 13041


def test_find_run_dir_takes_the_newest_folder_made_since_the_launch(tmp_path):
    task = tmp_path / "iker-shoe"
    old, new = task / "iker_vlm_c00_s1", task / "iker_vlm_c00_s1-r1"
    old.mkdir(parents=True)
    new.mkdir()
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (5000.0, 5000.0))
    assert lp.find_run_dir(task, "iker_vlm_c00_s1", 4000.0) == new
    assert lp.find_run_dir(task, "iker_vlm_c00_s1", 9000.0) is None
    assert lp.find_run_dir(tmp_path / "missing", "iker_vlm_c00_s1", 0.0) is None


def test_collect_reads_runs_events_checkpoints_and_the_calibration(tmp_path):
    paths = lp.LoopPaths.of(tmp_path / "hdgp", 0, "iker_shoe_c00")
    state = ls.new_state("t", phase="calibrate")
    label = state["policy"]["labels"]["stage1_a"]
    run_dir = paths.task_dir("stage1_a") / label
    (run_dir / "nn").mkdir(parents=True)
    (run_dir / "summaries").mkdir()
    (run_dir / "summaries" / "events.out.tfevents.1").write_bytes(b"")
    (run_dir / "nn" / "last_open-sens_l_iker_shoe_grasp_ep_250_rew_1.0.pth").write_bytes(b"")
    train_log = paths.train_log(label)
    train_log.write_text("epoch\n")
    side_log = paths.side_log("calibrate", "ep250")
    side_log.parent.mkdir(parents=True)
    side_log.write_text("QUALITY config 00 {} passed True -> x\n")
    state["runs"] = {
        "stage1_a": {"label": label, "log": str(train_log), "started_s": 0.0, "key": label},
        "calibrate": {"label": "iker_shoe_c00_calibrate", "log": str(side_log), "started_s": 0.0, "key": "k"},
    }
    proc = tmp_path / "proc"
    _proc(proc, 7, label, ISAAC)
    run_files.write_json(paths.calibration_file, {"schema": 1, "checkpoint": "k", "q_lo": 0.1, "q_hi": 0.3})
    events = {lp.LATCHED_TAG: [(1, 0.1), (2, 0.2)], lp.OVER_RACK_TAG: [(1, 0.0)]}
    probe = lp.collect(state, paths, now_s=train_log.stat().st_mtime + 3.0, gpu_used_mib=13000,
                       load_events=lambda path: events, proc_root=proc)
    assert probe.runs["stage1_a"].alive and not probe.runs["stage1_a"].crashed and probe.runs["stage1_a"].idle_s >= 3.0
    assert probe.runs["calibrate"].finished and probe.runs["calibrate"].passed and not probe.runs["calibrate"].alive
    assert probe.latched == ((1, 0.1), (2, 0.2)) and probe.over_rack == ((1, 0.0),) and list(probe.checkpoints) == [250]
    assert probe.calibration["q_hi"] == 0.3 and probe.gpu_used_mib == 13000 and probe.boot_reward is None
    assert probe.attempts == () and probe.final_rows is None and probe.files["stage_prompt"] is False


def test_attempts_evals_final_rows_and_files_come_from_the_stage_directory(tmp_path):
    paths = lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00")
    stage = paths.stage_dir
    for index, passed in ((0, False), (1, None)):
        attempt = stage / f"attempt_{index:02d}"
        attempt.mkdir(parents=True)
        (attempt / "response.md").write_text("```python\n```\n")
        if passed is not None:
            run_files.write_json(attempt / "gate.json", {"schema": 1, "passed": passed, "failures": []})
    (stage / "attempt_02").mkdir()
    run_files.write_json(paths.eval_file(250), {"schema": 1, "summary": {"success_5cm_end": 0.3}})
    run_files.write_json(paths.final_states_file, {"schema": 1, "rows": [{"env": 4, "success": True, "end_dist": 0.02, "joint_pos": []}]})
    for name in ("prompt.md", "snapshot.png", "keypoints.json"):
        (stage / name).write_text("x")
    assert lp.attempts(stage) == (ls.AttemptStatus(0, False), ls.AttemptStatus(1, None))
    assert lp.evals(stage) == {250: {"success_5cm_end": 0.3}}
    assert lp.final_rows(paths.final_states_file) == ({"env": 4, "success": True, "end_dist": 0.02},)
    probe = lp.collect(ls.new_state("t", phase="vlm_target"), paths, now_s=0.0, gpu_used_mib=0, load_events=lambda path: {},
                       proc_root=tmp_path / "no_proc")
    assert probe.files["stage_prompt"] and not probe.files["requery_prompt"] and not probe.files["video_raw"]
    assert probe.bank_meta is None and probe.checkpoints == {} and len(probe.attempts) == 2
````

````python
"""loop_vlm — prompts for the generator agent and the re-query parse (design auto-loop §6; no Isaac)."""

from openarm.agnostic.modules.iker import loop_vlm, prompts

TASK = "Place the shoe on the rack next to the other shoe."
KEYPOINTS = {1: (0.38, -0.06, 0.38), 2: (0.14, -0.06, 0.38), 3: (0.26, -0.11, 0.38), 4: (0.26, -0.02, 0.38)}
STAGE_RESPONSE = '''Plan: move the shoe onto the rack.

```python
import numpy as np
def get_interaction_data(keypoint_coordinates):
    """Stage 1: place shoe_move beside shoe_other."""
    done = False
    if done:
        return
    object_to_interact = "shoe_move"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for key in ["1", "2", "3", "4"]:
        keypoint_coordinates[key] = keypoint_coordinates[key] + np.array([0.0, -0.2, 0.12])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
'''
DONE_RESPONSE = '''```python
def get_interaction_data(keypoint_coordinates):
    """The shoe is on the rack next to the other shoe."""
    done = True
    if done:
        return
```
'''


def test_target_prompt_is_the_single_step_prompt():
    assert loop_vlm.target_prompt(TASK) == prompts.fill_single_step(TASK)


def test_requery_prompt_carries_the_stage_code_and_marks_the_stage_image():
    text = loop_vlm.requery_prompt(TASK, STAGE_RESPONSE)
    assert text.count(prompts.IMAGE_MARKER) == 1
    assert f"Stage 1 image: {loop_vlm.STAGE_IMAGE_MARKER}" in text
    assert 'object_to_interact = "shoe_move"' in text
    assert text.index("## History") < text.index(prompts.QUERY_HEADER)


def test_requery_result_is_done_only_when_the_function_returns_none():
    assert loop_vlm.requery_result(DONE_RESPONSE, KEYPOINTS) == {"done": True, "detail": "get_interaction_data returned None (done=True)"}
    new_stage = loop_vlm.requery_result(STAGE_RESPONSE, KEYPOINTS)
    assert new_stage["done"] is False and new_stage["detail"] == "new stage: move shoe_move keypoints [1, 2, 3, 4]"
    broken = loop_vlm.requery_result("no code block here", KEYPOINTS)
    assert broken["done"] is False and broken["detail"].startswith("G1: expected exactly one python code block")


def test_generator_brief_names_only_the_given_files():
    brief = loop_vlm.generator_brief("/s/prompt.md", "/s/attempt_00/response.md", {prompts.IMAGE_MARKER: "/s/snapshot.png"})
    assert brief.splitlines()[0] == "Read the prompt file /s/prompt.md and answer it exactly as it instructs."
    assert "The image for [IMAGE_WITH_KEYPOINTS] is /s/snapshot.png; open it with the Read tool." in brief
    assert "/s/attempt_00/response.md" in brief and "Do not open, list or search any other file" in brief
````

- [ ] **Step 2: 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_loop_probe.py source/openarm/openarm/agnostic/modules/iker/tests/test_loop_vlm.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'loop_probe'` / `'loop_vlm'`

- [ ] **Step 3: 구현** — 아래 두 파일을 만든다.

````python
"""Read the IKER auto loop's world into a ``loop_state.Probe`` (design 2026-09-14-iker-auto-loop §3, §8).

No Isaac, torch or subprocess: the checkout root, the /proc root, a TFEvents reader and the GPU memory reading are passed
in, so the tests run on temporary trees. A process belongs to a run only through the RUN_LABEL in its environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from . import loop_state as ls
from . import run_files

RL_LOG_PARTS = ("log", "rl_games", "open-sens", "left")
TASK_DIRS = {"stage1_a": "iker-shoe-grasp", "stage1_b": "iker-shoe-grasp", "stage2": "iker-shoe"}
PHASE_TRAINING = {
    "stage1_a": "stage1_a", "calibrate": "stage1_a", "stage1_b": "stage1_b", "harvest": "stage1_b",
    "stage2_train": "stage2", "observe_requery": "stage2",
}
LATCHED_TAG = "Episode/grasp_episode/latched"
OVER_RACK_TAG = "Episode/grasp/over_rack_raised_frac"
CHECKPOINT_RE = re.compile(r"^last_.+_ep_(\d+)_rew_.*\.pth$")
ATTEMPT_RE = re.compile(r"^attempt_(\d{2})$")
EVAL_RE = re.compile(r"^eval_ep(\d+)\.json$")
TRAIN_FINISHED = "MAX EPOCHS NUM!"
TRAIN_CRASH = ("Traceback (most recent call last)", "CUDA out of memory", "Error executing job")
SIDE_MARKERS: Mapping[str, tuple[str, str]] = {  # run -> (result line prefix, crash marker printed by its excepthook)
    "calibrate": ("QUALITY config", "QUALITY FAILED"),
    "harvest": ("HARVEST config", "HARVEST FAILED"),
    "env_smoke": ("SMOKE passed", "SMOKE FAILED"),
    "eval": ("EVAL {", "EVAL FAILED"),
    "observe_rollout": ("EVAL {", "EVAL FAILED"),
    "observe_render": ("OBSERVE config", "OBSERVE FAILED"),
    "video": ("", ""),  # play.py prints no result line: the video file is the result
}
PASSED_RE = re.compile(r"\bpassed (True|False)\b")
REWARD_LINE_RE = re.compile(r"\[iker_grasp\] reward Stage1RewardCfg\(([^)]*)\)")
FIELD_RE = re.compile(r"(\w+)=([-+0-9.eE]+)")
ISAAC_PYTHON_MARK = b"kit/python/bin/python3"
LOG_TAIL_BYTES = 4 << 20
VIDEO_NAME = "rl-video-step-0.mp4"


@dataclass(frozen=True)
class LoopPaths:
    root: Path  # the hdgp checkout
    config_dir: Path  # iker_runs/shoe_place/config_XX
    state_dir: Path  # LOOP_STATE.json, history.jsonl, stage_01/
    log_dir: Path  # side-run logs and uncommitted outputs (gitignored)

    @classmethod
    def of(cls, root: Path, config_index: int, track: str, state_dir: Path | None = None) -> "LoopPaths":
        config_dir = root / "iker_runs" / "shoe_place" / f"config_{config_index:02d}"
        return cls(root, config_dir, state_dir if state_dir is not None else config_dir / "loop", root / "log" / "iker_loop" / track)

    @property
    def state_file(self) -> Path:
        return self.state_dir / "LOOP_STATE.json"

    @property
    def history_file(self) -> Path:
        return self.state_dir / "history.jsonl"

    @property
    def stage_dir(self) -> Path:
        return self.state_dir / "stage_01"

    @property
    def observe_dir(self) -> Path:
        return self.stage_dir / "observe"

    @property
    def interaction_file(self) -> Path:
        return self.stage_dir / "interaction_vlm.json"

    @property
    def video_note(self) -> Path:
        return self.stage_dir / "video.txt"

    @property
    def calibration_file(self) -> Path:
        return self.config_dir / "grasp_quality_calibration.json"  # iker_shoe_grasp_env.QUALITY_CALIBRATION_FILE

    @property
    def final_states_file(self) -> Path:
        return self.log_dir / "observe_final_states.json"

    @property
    def harvest_bank_file(self) -> Path:
        return self.config_dir / "grasp_bank.json"

    def bank_file(self, policy: Mapping) -> Path:
        """The bank stage 2 loads: the harvested bank unless the policy names another (the mock loop)."""
        return Path(policy["grasp_bank_path"]) if policy["grasp_bank_path"] else self.harvest_bank_file

    def task_dir(self, run: str) -> Path:
        return self.root.joinpath(*RL_LOG_PARTS, TASK_DIRS[run])

    def train_log(self, label: str) -> Path:
        return self.root.joinpath(*RL_LOG_PARTS, f"train_{label}.log")

    def side_log(self, run: str, tag: str) -> Path:
        return self.log_dir / f"{run}_{tag}.log"

    def eval_file(self, epoch: int) -> Path:
        return self.stage_dir / f"eval_ep{epoch:04d}.json"

    def eval_rows_file(self, epoch: int) -> Path:
        return self.log_dir / f"eval_ep{epoch:04d}_rows.json"


def read_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def list_checkpoints(nn_dir: Path) -> dict[int, str]:
    """epoch -> absolute path of rl_games' ``last_<name>_ep_<E>_rew_<R>.pth`` files."""
    if not nn_dir.is_dir():
        return {}
    found = {}
    for path in nn_dir.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if match:
            found[int(match.group(1))] = str(path.resolve())
    return found


def find_run_dir(task_dir: Path, label: str, started_s: float) -> Path | None:
    """The folder train.py made for ``label`` (``label``, or ``label-rN`` when that existed), modified since the launch."""
    candidates = [task_dir / label, *task_dir.glob(f"{label}-r*")] if task_dir.is_dir() else []
    fresh = [path for path in candidates if path.is_dir() and path.stat().st_mtime >= started_s - 60.0]
    return max(fresh, key=lambda path: path.stat().st_mtime) if fresh else None


def label_pids(label: str, proc_root: Path = Path("/proc")) -> list[int]:
    """Isaac python processes whose environment holds ``RUN_LABEL=<label>`` (the python.sh bash wrapper is not one)."""
    wanted = b"RUN_LABEL=" + label.encode()
    pids = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ, cmdline = (entry / "environ").read_bytes(), (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if wanted in environ.split(b"\0") and ISAAC_PYTHON_MARK in cmdline:
            pids.append(int(entry.name))
    return sorted(pids)


def parse_gpu_used_mib(text: str) -> int:
    """Largest ``memory.used`` in ``nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`` output."""
    values = [int(line.strip()) for line in text.splitlines() if line.strip()]
    if not values:
        raise ValueError(f"no GPU memory reading in {text!r}")
    return max(values)


def boot_reward(text: str) -> dict[str, float] | None:
    """g_min, q_lo and q_hi of the last ``[iker_grasp] reward Stage1RewardCfg(...)`` line."""
    lines = REWARD_LINE_RE.findall(text)
    if not lines:
        return None
    fields = dict(FIELD_RE.findall(lines[-1]))
    return {key: float(fields[key]) for key in ("g_min", "q_lo", "q_hi")}


def training_status(text: str, alive: bool, checkpoints: Mapping[int, str], max_epochs: int | None, idle_s: float) -> ls.RunStatus:
    finished = TRAIN_FINISHED in text or (max_epochs is not None and max_epochs in checkpoints)
    crash = next((marker for marker in TRAIN_CRASH if marker in text), "")
    return ls.RunStatus(started=True, alive=alive, finished=finished, crashed=bool(crash) or (not alive and not finished),
                        marker=crash, idle_s=idle_s)


def side_status(run: str, text: str, alive: bool, result_ready: bool, idle_s: float) -> ls.RunStatus:
    prefix, crash_marker = SIDE_MARKERS[run]
    lines = [line for line in text.splitlines() if prefix and line.startswith(prefix)]
    marker = lines[-1] if lines else ""
    finished = bool(marker) or result_ready
    found = PASSED_RE.search(marker)
    passed = (found.group(1) == "True") if found else (True if finished else None)
    crash = bool(crash_marker) and crash_marker in text
    return ls.RunStatus(started=True, alive=alive, finished=finished, passed=passed, crashed=crash or (not alive and not finished),
                        marker=marker or (crash_marker if crash else ""), idle_s=idle_s)


def attempts(stage_dir: Path) -> tuple[ls.AttemptStatus, ...]:
    """VLM attempts that hold a response.md, with their gate verdict once gate.json exists."""
    if not stage_dir.is_dir():
        return ()
    found = []
    for path in sorted(stage_dir.iterdir()):
        match = ATTEMPT_RE.match(path.name)
        if match and (path / "response.md").is_file():
            gate = _optional_json(path / "gate.json")
            found.append(ls.AttemptStatus(int(match.group(1)), None if gate is None else bool(gate["passed"])))
    return tuple(found)


def evals(stage_dir: Path) -> dict[int, dict]:
    if not stage_dir.is_dir():
        return {}
    return {int(match.group(1)): run_files.read_json(path)["summary"]
            for path in sorted(stage_dir.iterdir()) if (match := EVAL_RE.match(path.name))}


def final_rows(path: Path) -> tuple[dict, ...] | None:
    doc = _optional_json(path)
    if doc is None:
        return None
    return tuple({"env": int(row["env"]), "success": bool(row["success"]), "end_dist": float(row["end_dist"])} for row in doc["rows"])


def stage2_video(state: Mapping, paths: LoopPaths) -> Path | None:
    """The play video file of the stage-2 run folder (it may not exist yet)."""
    record = state["runs"].get("stage2", {})
    run_dir = find_run_dir(paths.task_dir("stage2"), state["policy"]["labels"]["stage2"], record.get("started_s", 0.0))
    return run_dir / "videos" / "play" / VIDEO_NAME if run_dir is not None else None


def collect(state: Mapping, paths: LoopPaths, *, now_s: float, gpu_used_mib: int,
            load_events: Callable[[str], Mapping[str, list]], proc_root: Path = Path("/proc")) -> ls.Probe:
    policy = state["policy"]
    runs = {name: _run_status(name, record, state, paths, now_s, proc_root) for name, record in state["runs"].items()}
    latched, over_rack, checkpoints, boot = (), (), {}, None
    training = PHASE_TRAINING.get(state["phase"])
    if training is not None:
        record = state["runs"].get(training, {})
        run_dir = find_run_dir(paths.task_dir(training), policy["labels"][training], record.get("started_s", 0.0))
        if run_dir is not None:
            checkpoints = list_checkpoints(run_dir / "nn")
            if training != "stage2":
                series = _events(run_dir, load_events)
                latched, over_rack = _series(series, LATCHED_TAG), _series(series, OVER_RACK_TAG)
        if training == "stage1_b" and record:
            boot = boot_reward(read_tail(Path(record["log"])))
    return ls.Probe(
        gpu_used_mib=gpu_used_mib, runs=runs, latched=latched, over_rack=over_rack, checkpoints=checkpoints, boot_reward=boot,
        calibration=_optional_json(paths.calibration_file), bank_meta=(_optional_json(paths.harvest_bank_file) or {}).get("metadata"),
        attempts=attempts(paths.stage_dir), evals=evals(paths.stage_dir), final_rows=final_rows(paths.final_states_file),
        requery=_optional_json(paths.observe_dir / "requery.json"), files=_files(state, paths),
    )


def _optional_json(path: Path) -> dict | None:
    return run_files.read_json(path) if path.is_file() else None


def _series(series: Mapping[str, list], tag: str) -> tuple[tuple[int, float], ...]:
    return tuple((int(step), float(value)) for step, value in series.get(tag, ()))


def _events(run_dir: Path, load_events: Callable[[str], Mapping[str, list]]) -> Mapping[str, list]:
    files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    return load_events(str(files[-1])) if files else {}


def _run_status(name: str, record: Mapping, state: Mapping, paths: LoopPaths, now_s: float, proc_root: Path) -> ls.RunStatus:
    log = Path(record["log"])
    alive = bool(label_pids(record["label"], proc_root))
    idle_s = now_s - log.stat().st_mtime if log.is_file() else 0.0
    text = read_tail(log)
    if name in ls.TRAINING_RUNS:
        run_dir = find_run_dir(paths.task_dir(name), record["label"], record.get("started_s", 0.0))
        checkpoints = list_checkpoints(run_dir / "nn") if run_dir is not None else {}
        max_epochs = {"stage1_a": None, "stage1_b": state["policy"]["b_max_epochs"], "stage2": state["policy"]["stage2_epochs"]}[name]
        return training_status(text, alive, checkpoints, max_epochs, idle_s)
    return side_status(name, text, alive, name == "video" and _video_ready(state, paths), idle_s)


def _video_ready(state: Mapping, paths: LoopPaths) -> bool:
    record, video = state["runs"].get("video"), stage2_video(state, paths)
    if record is None or video is None or not video.is_file():
        return False
    stat = video.stat()
    return stat.st_size > 0 and stat.st_mtime >= record.get("started_s", 0.0)


def _files(state: Mapping, paths: LoopPaths) -> dict[str, bool]:
    stage, observe = paths.stage_dir, paths.observe_dir
    return {
        "stage_prompt": all((stage / name).is_file() for name in ("prompt.md", "snapshot.png", "keypoints.json")),
        "observe_snapshot": all((observe / name).is_file() for name in ("snapshot.png", "keypoints.json", "state.json")),
        "requery_prompt": (observe / "prompt.md").is_file(),
        "requery_response": (observe / "response.md").is_file(),
        "video_raw": _video_ready(state, paths),
        "video": paths.video_note.is_file(),
    }
````

````python
"""VLM steps of the IKER auto loop (design 2026-09-14-iker-auto-loop §6).

The generator is a fresh Claude Code agent that receives only ``generator_brief``: the prompt file, the images and the
response path. The first stage uses the single-step prompt; the re-query after execution uses the multi-step prompt
with the first stage's code as history and is only parsed: ``None`` from the function means done.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import interaction, prompts

STAGE_IMAGE_MARKER = "[STAGE_1_IMAGE]"


def target_prompt(task: str) -> str:
    return prompts.fill_single_step(task)


def requery_prompt(task: str, stage_response: str) -> str:
    """Multi-step prompt whose history holds the code of the response that set the stage-1 target."""
    code = interaction.extract_code_block(stage_response)
    return prompts.fill_multi_step(task, [prompts.StageRecord(STAGE_IMAGE_MARKER, code)])


def requery_result(response: str, keypoints: Mapping[int, Sequence[float]]) -> dict:
    """{"done": bool, "detail": str}: done only when ``get_interaction_data`` returns None (``done=True``)."""
    try:
        result = interaction.run_interaction(interaction.extract_code_block(response), keypoints)
    except interaction.InteractionError as exc:
        return {"done": False, "detail": f"G1: {exc}"}
    if result.done:
        return {"done": True, "detail": "get_interaction_data returned None (done=True)"}
    return {"done": False, "detail": f"new stage: move {result.object_name} keypoints {list(result.keypoint_ids)}"}


def generator_brief(prompt_path, response_path, images: Mapping[str, str]) -> str:
    """The whole message a fresh generator agent receives (§6 generator isolation)."""
    lines = [f"Read the prompt file {prompt_path} and answer it exactly as it instructs."]
    lines += [f"The image for {marker} is {path}; open it with the Read tool." for marker, path in images.items()]
    lines += [
        f"Write your complete answer, with the single python code block the prompt asks for, to {response_path}.",
        "Read only these files. Do not open, list or search any other file or directory, and do not run commands.",
    ]
    return "\n".join(lines)
````

- [ ] **Step 4: 통과 확인** — Step 2 명령. Expected: `12 passed`.

- [ ] **Step 5: 전체 순수 테스트** — Expected: `186 passed, 1 skipped`.

- [ ] **Step 6: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/modules/iker/loop_probe.py source/openarm/openarm/agnostic/modules/iker/loop_vlm.py source/openarm/openarm/agnostic/modules/iker/tests/test_loop_probe.py source/openarm/openarm/agnostic/modules/iker/tests/test_loop_vlm.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 자동 루프의 세계 읽기(로그 표식·TFEvents·체크포인트·RUN_LABEL 프로세스·루프 파일)와 VLM 단계(생성기 지시문·재질의 done 파싱)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 4: 1단계 성공 캡처·복원과 학습 뱅크 metadata

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py`
- Modify: `scripts/iker/grasp_smoke.py`

**Interfaces:**
- Consumes: Task 2 `loop_state.LEARNED_BANK_SOURCE`. Task 1 의 `grasp_stage` 변경.
- Produces:
  - `gs.SuccessCapture(joint_pos, joint_target, shoe_pose, palm_pose, step, valid)` 와 `.empty(num_envs, num_joints, device)`.
  - `gs.capture_rows(capture, mask, *, joint_pos, joint_target, shoe_pose, palm_pose, step) -> SuccessCapture`, `gs.holding_targets(joint_targets, joint_pos, arm_ids, hand_ids, hand_targets)`.
  - `gb.BOOT_METADATA_KEYS`, `gb.JOINT_COLUMNS`, `gb.sort_joint_columns(entries, joint_names) -> (entries, names)`, `gb.learned_bank_metadata(boot, *, side_sign, checkpoint, checkpoint_sha256, stage1_reward, seeds, captured, verified) -> dict`.
  - env:
    - cfg `capture_success_states: bool = False`
    - 속성 `env._capture`(`SuccessCapture`), `env._boot_metadata`(부팅 대조 9 키)
    - 메서드 `env.clear_success_captures()`, `env.restore_success_captures(env_ids)`
    - 기존 속성 `_hand_ids`·`_hand_lo`·`_hand_hi`·`_reward_cfg`·`_last`·`_palm`·`_shoe`

- [ ] **Step 1: 실패하는 테스트** — `tests/test_grasp_stage.py` 끝에 붙인다.

````python
def test_capture_rows_keep_each_envs_first_success_until_cleared():
    capture = gs.SuccessCapture.empty(3, 4)
    first = dict(joint_pos=torch.ones(3, 4), joint_target=2 * torch.ones(3, 4), shoe_pose=3 * torch.ones(3, 7), palm_pose=4 * torch.ones(3, 7))
    capture = gs.capture_rows(capture, torch.tensor([True, False, True]), step=5, **first)
    later = {key: 10 * value for key, value in first.items()}
    capture = gs.capture_rows(capture, torch.tensor([True, True, False]), step=9, **later)
    assert capture.valid.tolist() == [True, True, True]
    assert capture.step.tolist() == [5, 9, 5]
    assert capture.joint_pos[:, 0].tolist() == [1.0, 10.0, 1.0]
    assert capture.joint_target[:, 0].tolist() == [2.0, 20.0, 2.0]
    assert capture.shoe_pose[:, 6].tolist() == [3.0, 30.0, 3.0] and capture.palm_pose[:, 0].tolist() == [4.0, 40.0, 4.0]
    cleared = gs.SuccessCapture.empty(3, 4)
    assert not cleared.valid.any() and cleared.step.tolist() == [-1, -1, -1]


def test_holding_targets_pin_the_arm_at_its_position_and_the_hand_at_its_ema_target():
    targets = torch.zeros(2, 6)
    joint_pos = torch.arange(12, dtype=torch.float32).view(2, 6)
    hand = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    out = gs.holding_targets(targets, joint_pos, [0, 1], torch.tensor([3, 4]), hand)
    assert out.tolist() == [[0.0, 1.0, 0.0, 7.0, 8.0, 0.0], [6.0, 7.0, 0.0, 9.0, 10.0, 0.0]]
    assert float(targets.abs().sum()) == 0.0


def test_the_normalized_ema_target_is_a_fixed_point_of_the_hand_law():
    lo, hi = torch.tensor([-0.5, 0.0]), torch.tensor([0.5, 1.2])
    previous = torch.tensor([[0.1, 0.9], [-0.5, 1.2]])
    held = gs.hand_targets(gs.normalized_targets(previous, lo, hi), lo, hi, previous)
    assert torch.allclose(held, previous, atol=1e-6)
````

`tests/test_grasp_bank.py` 끝에 붙인다.

````python
BOOT = {key: index for index, key in enumerate(gb.BOOT_METADATA_KEYS)}


def test_sorted_joint_columns_load_back_in_articulation_order():
    names = ["r_aj_1", "l_hj_thumb_1", "head_pan", "l_aj_1"]
    entries = {
        "joint_pos": torch.tensor([[1.0, 2.0, 3.0, 4.0]]), "joint_target": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        "shoe_pose": torch.zeros(1, 7), "palm_pose": torch.ones(1, 7),
    }
    moved, sorted_names = gb.sort_joint_columns(entries, names)
    assert sorted_names == ["head_pan", "l_aj_1", "l_hj_thumb_1", "r_aj_1"]
    assert moved["joint_pos"].tolist() == [[3.0, 4.0, 2.0, 1.0]] and moved["palm_pose"] is entries["palm_pose"]
    bank = gb.load_bank(gb.bank_document(moved, sorted_names, {"k": 1}), names, {"k": 1}, "cpu")
    assert bank.joint_pos.tolist() == [[1.0, 2.0, 3.0, 4.0]] and bank.joint_target.tolist() == [[5.0, 6.0, 7.0, 8.0]]


def test_learned_bank_metadata_keeps_the_boot_keys_and_records_the_origin():
    meta = gb.learned_bank_metadata(BOOT, side_sign=-1.0, checkpoint="/b/ep350.pth", checkpoint_sha256="ab",
                                    stage1_reward={"g_min": 0.5}, seeds=(0, 1), captured=90, verified=70)
    assert {key: meta[key] for key in gb.BOOT_METADATA_KEYS} == BOOT
    assert (meta["source"], meta["side_sign"], meta["seeds"], meta["captured"], meta["verified"]) == ("learned_grasp", -1.0, [0, 1], 90, 70)
    without_gains = {key: value for key, value in BOOT.items() if key != "gains"}
    with pytest.raises(ValueError, match="missing"):
        gb.learned_bank_metadata(without_gains, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=1)
    with pytest.raises(ValueError, match="verified"):
        gb.learned_bank_metadata(BOOT, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=2)
````

- [ ] **Step 2: 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py -q -p no:cacheprovider -k "capture or holding or fixed_point or sorted_joint or learned_bank"`
Expected: FAIL — `AttributeError: ... has no attribute 'SuccessCapture'` / `'BOOT_METADATA_KEYS'` (`fixed_point` 테스트는 기존 함수만 써서 통과한다 — 수확 검증이 기대는 EMA 고정점을 고정하는 테스트다)

- [ ] **Step 3: 구현** — `grasp_stage.py` 끝에 붙인다.

````python
@dataclass(frozen=True)
class SuccessCapture:
    """Each env's state at its first success since the last clear, kept apart from the same step's reset (auto-loop §5)."""

    joint_pos: torch.Tensor  # (N, J) articulation joint positions
    joint_target: torch.Tensor  # (N, J) targets that hold the state (``holding_targets``)
    shoe_pose: torch.Tensor  # (N, 7) env-local position + wxyz
    palm_pose: torch.Tensor  # (N, 7) env-local position + wxyz
    step: torch.Tensor  # (N,) long, the environment's common step counter at the capture, -1 before
    valid: torch.Tensor  # (N,) bool

    @classmethod
    def empty(cls, num_envs: int, num_joints: int, device: str | torch.device = "cpu") -> "SuccessCapture":
        return cls(
            joint_pos=torch.zeros(num_envs, num_joints, device=device),
            joint_target=torch.zeros(num_envs, num_joints, device=device),
            shoe_pose=torch.zeros(num_envs, 7, device=device),
            palm_pose=torch.zeros(num_envs, 7, device=device),
            step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
            valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )


def capture_rows(
    capture: SuccessCapture,
    mask: torch.Tensor,
    *,
    joint_pos: torch.Tensor,
    joint_target: torch.Tensor,
    shoe_pose: torch.Tensor,
    palm_pose: torch.Tensor,
    step: int,
) -> SuccessCapture:
    """A new capture holding the rows of ``mask`` that hold none yet; a row's later successes are ignored until a clear.
    Tensor ops only, so the step path does not synchronise with the host."""
    take = mask & ~capture.valid
    rows = take[:, None]
    return SuccessCapture(
        joint_pos=torch.where(rows, joint_pos, capture.joint_pos),
        joint_target=torch.where(rows, joint_target, capture.joint_target),
        shoe_pose=torch.where(rows, shoe_pose, capture.shoe_pose),
        palm_pose=torch.where(rows, palm_pose, capture.palm_pose),
        step=torch.where(take, torch.full_like(capture.step, step), capture.step),
        valid=capture.valid | take,
    )


def holding_targets(
    joint_targets: torch.Tensor,
    joint_pos: torch.Tensor,
    arm_ids: Sequence[int] | torch.Tensor,
    hand_ids: Sequence[int] | torch.Tensor,
    hand_targets: torch.Tensor,
) -> torch.Tensor:
    """(N, J) joint targets that hold a state: the arm at its measured position, the hand at its EMA target and every
    other joint at its command."""
    out = joint_targets.clone()
    out[:, arm_ids] = joint_pos[:, arm_ids]
    out[:, hand_ids] = hand_targets
    return out
````

`grasp_bank.py` 의 `import torch` 다음 줄에 빈 줄과 `from openarm.agnostic.modules.iker.loop_state import LEARNED_BANK_SOURCE` 를 넣고, 파일 끝에 붙인다.

````python
BOOT_METADATA_KEYS = (
    "config_index", "physics_dt", "friction", "solver_position_iterations", "solver_velocity_iterations", "gains",
    "robot_usd", "shoe_meta_sha256", "scene_config",
)
JOINT_COLUMNS = ("joint_pos", "joint_target")


def sort_joint_columns(entries: Mapping[str, torch.Tensor], joint_names: Sequence[str]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Bank entries with their joint columns ordered by joint name (auto-loop §5); ``load_bank`` reorders by name."""
    order = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
    moved = {
        key: value[:, torch.tensor(order, device=value.device)] if key in JOINT_COLUMNS else value for key, value in entries.items()
    }
    return moved, [joint_names[i] for i in order]


def learned_bank_metadata(
    boot: Mapping,
    *,
    side_sign: float,
    checkpoint: str,
    checkpoint_sha256: str,
    stage1_reward: Mapping,
    seeds: Sequence[int],
    captured: int,
    verified: int,
) -> dict:
    """Metadata of a bank harvested from a stage-1 checkpoint: the environments' boot comparison keys and its origin."""
    missing = [key for key in BOOT_METADATA_KEYS if key not in boot]
    extra = sorted(set(boot) - set(BOOT_METADATA_KEYS))
    if missing or extra:
        raise ValueError(f"boot metadata keys differ from {BOOT_METADATA_KEYS}: missing {missing}, extra {extra}")
    _check_side_sign(float(side_sign))
    if not 0 <= verified <= captured:
        raise ValueError(f"verified {verified} must lie within 0..captured {captured}")
    return {
        **{key: boot[key] for key in BOOT_METADATA_KEYS},
        "source": LEARNED_BANK_SOURCE,
        "side_sign": float(side_sign),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "stage1_reward": dict(stage1_reward),
        "seeds": [int(seed) for seed in seeds],
        "captured": int(captured),
        "verified": int(verified),
    }
````

- [ ] **Step 4: 통과 확인** — Step 2 명령, 그리고 전체 순수 테스트. Expected: `5 passed`, 전체 `191 passed, 1 skipped`.

- [ ] **Step 5: cfg 플래그** — `iker_shoe_grasp_env_cfg.py` 의 `    quality_calibration_path = ""` 줄 다음에 넣는다.

```python

    # harvest_grasp_bank.py: keep each env's first success state apart from the same step's reset (auto-loop design §5)
    capture_success_states = False
```

- [ ] **Step 6: env 캡처·복원** — `iker_shoe_grasp_env.py` 다섯 곳.
  - (a) `        self._bank = gb.load_bank(bank_doc, joint_names, expected, dev)` 다음 줄:
    ```python
            self._boot_metadata = expected  # harvest_grasp_bank.py writes these comparison keys into the learned bank
    ```
  - (b) `        self._episode_log = dict.fromkeys(EPISODE_LOG_KEYS, 0.0)  # statistics of the most recently finished episodes` 다음 줄:
    ```python
            self._capture = gs.SuccessCapture.empty(n, self._robot.num_joints, dev)  # written only with cfg.capture_success_states
    ```
  - (c) `_get_dones` 의 `        self._q_at_success = torch.where(step.success, q, self._q_at_success)` 다음:
    ```python
            if self.cfg.capture_success_states:
                # the same step's reset overwrites the joints, their targets and the shoe pose (auto-loop design §5)
                self._capture = gs.capture_rows(
                    self._capture,
                    step.success,
                    joint_pos=self._robot.data.joint_pos,
                    joint_target=gs.holding_targets(self._joint_targets, self._robot.data.joint_pos, self._arm_ids, self._hand_ids, self._hand_targets),
                    shoe_pose=torch.cat([shoe_pos, self._shoe.data.root_quat_w], dim=-1),
                    palm_pose=torch.cat([palm_pos, self._robot.data.body_quat_w[:, self._palm]], dim=-1),
                    step=self.common_step_counter,
                )
    ```
  - (d) `_reset_idx` 의 시작 바닥 높이 계산:
    ```python
            start_points = quat_apply(
                shoe_pose[:, None, 3:].expand(count, self._surface_local.shape[0], 4).reshape(-1, 4),
                self._surface_local.expand(count, -1, 3).reshape(-1, 3),
            ).view(count, -1, 3) + shoe_pose[:, None, :3]
            self._start_bottom_z[env_ids] = start_points[..., 2].min(dim=-1).values
    ```
    를 다음으로 바꾼다.
    ```python
            self._start_bottom_z[env_ids] = self._bottom_z(shoe_pose)
    ```
  - (e) `    # ----------------------------------------------------------------- reset` 줄 바로 위에 넣는다.
    ```python
        # ------------------------------------------------------ success capture

        def clear_success_captures(self) -> None:
            self._capture = gs.SuccessCapture.empty(self.num_envs, self._robot.num_joints, self.device)

        def restore_success_captures(self, env_ids: torch.Tensor) -> None:
            """Write the captured success states of ``env_ids`` back as the start of a fresh episode (auto-loop design §5)."""
            if not bool(self._capture.valid[env_ids].all()):
                raise ValueError("restore_success_captures: some of the envs hold no capture")
            count, dev = len(env_ids), self.device
            joint_pos, joint_target = self._capture.joint_pos[env_ids], self._capture.joint_target[env_ids]
            self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
            self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
            self._joint_targets[env_ids] = joint_target
            self._hand_targets[env_ids] = joint_target[:, self._hand_ids]
            shoe_pose = self._capture.shoe_pose[env_ids].clone()
            self._start_bottom_z[env_ids] = self._bottom_z(shoe_pose)
            shoe_pose[:, :3] += self.scene.env_origins[env_ids]
            self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
            self._shoe.write_root_velocity_to_sim(torch.zeros(count, 6, device=dev), env_ids=env_ids)
            self._stage = self._stage.reset_rows(env_ids)
            self._q_at_latch[env_ids] = 0.0
            self._q_at_success[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            self._wrench.reset(env_ids)
            self.actions[env_ids] = 0.0
            self._ik.reset(env_ids)

        def _bottom_z(self, shoe_pose: torch.Tensor) -> torch.Tensor:
            """(K,) height of the lowest surface point of env-local shoe poses (K, 7)."""
            count, points = shoe_pose.shape[0], self._surface_local.shape[0]
            surface = quat_apply(
                shoe_pose[:, None, 3:].expand(count, points, 4).reshape(-1, 4), self._surface_local.expand(count, -1, 3).reshape(-1, 3)
            ).view(count, points, 3)
            return (surface[..., 2] + shoe_pose[:, None, 2]).min(dim=-1).values

    ```

- [ ] **Step 7: 스모크 검사 6** — `scripts/iker/grasp_smoke.py` 세 곳.
  - docstring:
    ```text
       (rl_games reads each step's log with the keys of the epoch's first step).
    ```
    를 다음으로 바꾼다.
    ```text
       (rl_games reads each step's log with the keys of the epoch's first step);
    6. with ``capture_success_states`` the success call of a forced hold is captured, and restoring it writes the captured joints
       and hand targets back as a fresh episode (the learned grasp harvest, auto-loop design §5).
    ```
  - `    cfg.grasp_reward.hold_radius_m = FORCED_HOLD_RADIUS_M` 다음 줄:
    ```python
        cfg.capture_success_states = True
    ```
  - `        failures.append("random-action rewards, the episode-end log, or log keys that differ between steps")` 다음:
    ```python

        env.reset()
        env.clear_success_captures()
        forced_hold(env, SUCCESS_CALL + 1)  # the success call is captured before anything could overwrite it
        capture = env._capture
        pose_error = float((capture.shoe_pose[:, :3] - (env._shoe.data.root_pos_w - origins)).abs().max())
        home = env._robot.data.default_joint_pos.clone()
        env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
        env.episode_length_buf[:] = 7
        env.restore_success_captures(torch.arange(n, device=dev))
        joint_error = float((env._robot.data.joint_pos - capture.joint_pos).abs().max())
        hand_error = float((env._hand_targets - capture.joint_target[:, env._hand_ids]).abs().max())
        length = int(env.episode_length_buf.max())
        print(f"SMOKE success capture: valid {bool(capture.valid.all())}, shoe pose error {pose_error:.2e}, restored joint error "
              f"{joint_error:.2e}, hand target error {hand_error:.2e}, episode length {length}", flush=True)
        if not bool(capture.valid.all()) or pose_error > 1e-4 or joint_error > 1e-5 or hand_error > 1e-6 or length != 0:
            failures.append("the success capture or its restore does not reproduce the captured state")
    ```

- [ ] **Step 8: Isaac 스모크** — GPU 메모리를 확인한 뒤(≤ 20000 MiB) 돌린다.

Run: `cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_loop_t4_grasp_smoke ../IsaacLab/_isaac_sim/python.sh scripts/iker/grasp_smoke.py --headless 2>&1 | grep -aE "SMOKE|Traceback|Error"`
Expected:
- 기존 검사 줄과 함께 `SMOKE success capture: valid True, shoe pose error ... restored joint error ... hand target error 0.00e+00, episode length 0` 이 나온다.
- 오차는 pose ≤ 1e-4, joint ≤ 1e-5 다.
- 마지막 줄은 `GRASP SMOKE passed True` 다.

- [ ] **Step 9: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py scripts/iker/grasp_smoke.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 1단계 env 성공 순간 캡처·복원(cfg 플래그, 리셋과 무관한 버퍼)과 학습 뱅크 metadata·관절 열 이름 정렬, 스모크 검사 6

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 5: 학습 파지 뱅크 수확

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/policy_player.py`
- Create: `scripts/iker/harvest_grasp_bank.py`

**Interfaces:**
- Consumes: Task 4 의 env 캡처 API, `gs.normalized_targets`, `gb.lift_held`, `gb.sort_joint_columns`, `gb.learned_bank_metadata`, `gb.bank_document`, `gb.side_sign`.
- Produces:
  - `policy_player.load_player(env, task, checkpoint) -> (wrapped, agent)`, `reset_player(wrapped, agent) -> obs`, `step_player(wrapped, agent, obs) -> (obs, dones)`.
  - CLI `harvest_grasp_bank.py --checkpoint C [--config-index 0] [--num-envs 512] [--seeds 0 1 2 3] [--min-entries 64] [--g-min 0.5] [--out PATH]`.
  - 출력: 시드마다 `HARVEST seed <s> envs <n> first-episode successes <k> verified <v> shoe rise m q10/50/90 [..] palm rise [..] slip m [..] resets <r>` 줄(성공 0 이면 `... successes 0 verified 0` 까지만)과 마지막 `HARVEST config <cc> checkpoint <name> captured <c> verified <v> (min <m>) passed <True|False> -> <path|not written>`.
  - 통과할 때만 뱅크 파일을 쓴다. 예외 시 `HARVEST FAILED`, 종료 코드 0 = passed.

- [ ] **Step 1: 플레이어 헬퍼** — 아래 파일을 만든다.

````python
"""rl_games player on an IKER gym env for evaluation scripts (auto-loop §5, §7). Import after the Isaac app launcher.

The wiring is the one measure_grasp_quality.py uses: the task's registered PPO config, the Isaac Lab rl_games wrapper
and ``Runner.create_player``, with deterministic actions.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def load_player(env, task: str, checkpoint: Path):
    """(wrapped env, player) with ``checkpoint`` restored."""
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(
        env, agent_cfg["params"]["config"].get("device", "cuda:0"), params.get("clip_observations", math.inf),
        params.get("clip_actions", math.inf), params.get("obs_groups"), params.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(checkpoint))
    return wrapped, agent


def reset_player(wrapped, agent) -> torch.Tensor:
    """Reset every env and the player's state; returns the first observation batch."""
    agent.reset()
    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    return obs


def step_player(wrapped, agent, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic policy step; returns (next observation batch, dones)."""
    obs, _, dones, _ = wrapped.step(agent.get_action(agent.obs_to_torch(obs), is_deterministic=True))
    return (obs["obs"] if isinstance(obs, dict) else obs), dones
````

- [ ] **Step 2: 수확 스크립트** — 아래 파일을 만든다.

````python
"""Harvest the learned grasp bank from a stage-1 checkpoint (design 2026-09-14-iker-auto-loop §5).

For each seed every env runs its first episode with deterministic actions, no noise, no disturbance and no startup
randomisation; the environment captures each success moment before the same step's reset overwrites it. The captured
states are restored into their envs and lifted: the hand action is the normalized captured EMA target (a fixed point of
the hand law). One zero arm action refreshes the restored state (the first step after a write still reads the old palm
pose and Jacobian); then the arm servoes the palm toward a goal 10 cm above that pose with its orientation kept, each
action component clamped at full scale: +z at 0.02 m per step for 5 steps, then 10 steps (1 s) holding the goal (a zero
delta would let the arm sag). A state is kept when ``grasp_bank.lift_held`` holds, with the shoe's position measured in the
palm frame, and its env did not reset.
grasp_bank.json is written only with ``--min-entries`` verified grasps.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/harvest_grasp_bank.py --checkpoint <stage-1 .pth> --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Harvest the learned grasp bank from a stage-1 checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--min-entries", type=int, default=64)
parser.add_argument("--g-min", type=float, default=0.5, help="grasp factor floor the checkpoint trained with (phase A 1.0, B 0.5)")
parser.add_argument("--out", default="", help="bank file (default: iker_runs/shoe_place/config_XX/grasp_bank.json)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("HARVEST FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import ARM_ACTION_DIM, IkerShoeGraspPlayEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp"
SETTLE_STEPS, LIFT_STEPS, HOLD_STEPS = 1, 5, 10  # 0.02 m per policy step at full action: 10 cm, then 1 s at 10 Hz
LIFT_HEIGHT_M = 0.10
BANK_FIELDS = ("joint_pos", "joint_target", "shoe_pose", "palm_pose")


def make_env():
    cfg = IkerShoeGraspPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seeds[0]
    cfg.grasp_reward.g_min = args.g_min
    cfg.wrench_prob_range = (1e-9, 1e-9)
    cfg.capture_success_states = True
    cfg.events.shoe_mass = None  # nominal physics, as make_grasp_bank.py
    cfg.events.shoe_material = None
    cfg.events.shoe_com = None
    return gym.make(TASK, cfg=cfg)


def first_episode_successes(u, wrapped, agent) -> torch.Tensor:
    """(N,) bool: envs whose first episode ended in success; their captures hold that moment."""
    done_first = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    succeeded = torch.zeros_like(done_first)
    obs = reset_player(wrapped, agent)
    for _ in range(u.max_episode_length + 5):
        live = ~done_first
        obs, dones = step_player(wrapped, agent, obs)
        succeeded |= u._last.success & live
        done_first |= dones.bool()
        if bool(done_first.all()):
            break
    return succeeded & u._capture.valid


def quantiles(values: torch.Tensor) -> list[float]:
    if values.numel() == 0:
        return []
    return [round(float(v), 3) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))]


def palm_in_base(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) position and (N, 4) quaternion of the palm in the robot base frame, the frame of the arm action."""
    root, palm = u._robot.data.root_pose_w, u._robot.data.body_pose_w[:, u._palm]
    return subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])


def servo_arm_actions(u, goal_pos: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
    """(N, 6) arm actions toward a base-frame palm goal, each component at most the full-scale step: +z at full rate while
    far below the goal, and holding the goal (against the gravity sag a zero delta would accumulate) once there."""
    pos, quat = palm_in_base(u)
    move = ((goal_pos - pos) / u.cfg.action_pos_scale).clamp(-1.0, 1.0)
    turn = (axis_angle_from_quat(quat_mul(goal_quat, quat_inv(quat))) / u.cfg.action_rot_scale).clamp(-1.0, 1.0)
    return torch.cat([move, turn], dim=-1)


def palm_state(u, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(palm z env-local, palm quaternion, shoe position in the palm frame) of ``rows``."""
    palm = u._robot.data.body_pose_w[rows, u._palm]
    shoe_in_palm = quat_apply_inverse(palm[:, 3:7], u._shoe.data.root_pos_w[rows] - palm[:, :3])
    return palm[:, 2] - u.scene.env_origins[rows, 2], palm[:, 3:7].clone(), shoe_in_palm


def verify(u, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per restored capture (len(rows),): ``held`` and the lift diagnostics (shoe and palm rise, slip, palm tilt, reset)."""
    capture = u._capture
    u.restore_success_captures(rows)
    actions = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
    actions[:, ARM_ACTION_DIM:] = gs.normalized_targets(capture.joint_target[:, u._hand_ids], u._hand_lo, u._hand_hi)
    reset = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)

    def step(arm: torch.Tensor | None) -> None:
        nonlocal reset
        actions[:, :ARM_ACTION_DIM] = 0.0 if arm is None else arm
        _, _, terminated, truncated, _ = u.step(actions)
        reset = reset | terminated | truncated

    for _ in range(SETTLE_STEPS):
        step(None)
    goal_pos, goal_quat = palm_in_base(u)
    goal_pos = goal_pos + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M], device=u.device)
    origins_z = u.scene.env_origins[rows, 2]
    shoe_z_start = u._shoe.data.root_pos_w[rows, 2] - origins_z
    palm_z_start, palm_quat_start, _ = palm_state(u, rows)
    for _ in range(LIFT_STEPS):
        step(servo_arm_actions(u, goal_pos, goal_quat))
    palm_z_lift, _, rel_after_lift = palm_state(u, rows)
    for _ in range(HOLD_STEPS):
        step(servo_arm_actions(u, goal_pos, goal_quat))
    palm_z_end, palm_quat_end, rel_after_hold = palm_state(u, rows)
    shoe_z_end = u._shoe.data.root_pos_w[rows, 2] - origins_z
    tilt = torch.rad2deg(2.0 * torch.acos((palm_quat_start * palm_quat_end).sum(dim=-1).abs().clamp(max=1.0)))
    return {
        "held": gb.lift_held(shoe_z_start, shoe_z_end, rel_after_lift, rel_after_hold) & ~reset[rows],
        "rise": shoe_z_end - shoe_z_start,
        "palm_lift": palm_z_lift - palm_z_start,
        "palm_rise": palm_z_end - palm_z_start,
        "slip": (rel_after_hold - rel_after_lift).norm(dim=-1),
        "tilt_deg": tilt,
        "reset": reset[rows],
    }


def main() -> int:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env = make_env()
    u = env.unwrapped
    wrapped, agent = load_player(env, TASK, checkpoint)
    kept = {field: [] for field in BANK_FIELDS}
    captured = 0
    with torch.inference_mode():
        for seed in args.seeds:
            u.seed(seed)
            u.clear_success_captures()
            rows = torch.nonzero(first_episode_successes(u, wrapped, agent)).squeeze(-1)
            if len(rows) == 0:
                print(f"HARVEST seed {seed} envs {u.num_envs} first-episode successes 0 verified 0", flush=True)
                continue
            result = verify(u, rows)
            for field in BANK_FIELDS:
                kept[field].append(getattr(u._capture, field)[rows[result["held"]]].clone())
            captured += len(rows)
            print(f"HARVEST seed {seed} envs {u.num_envs} first-episode successes {len(rows)} verified {int(result['held'].sum())} "
                  f"q10/50/90: shoe rise m {quantiles(result['rise'])} palm rise after lift {quantiles(result['palm_lift'])} "
                  f"after hold {quantiles(result['palm_rise'])} slip in palm m {quantiles(result['slip'])} "
                  f"palm tilt deg {quantiles(result['tilt_deg'])} resets {int(result['reset'].sum())}", flush=True)
    entries = {field: torch.cat(parts) if parts else torch.zeros(0) for field, parts in kept.items()}
    verified = int(entries["joint_pos"].shape[0])
    out = Path(args.out) if args.out else layout.RUNS_DIR / f"config_{args.config_index:02d}" / "grasp_bank.json"
    passed = verified >= args.min_entries
    if passed:
        columns, names = gb.sort_joint_columns(entries, list(u._robot.data.joint_names))
        metadata = gb.learned_bank_metadata(
            u._boot_metadata, side_sign=gb.side_sign(robot.profile().hand_joint_names), checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), stage1_reward=asdict(u._reward_cfg),
            seeds=args.seeds, captured=captured, verified=verified,
        )
        run_files.write_json(out, gb.bank_document(columns, names, metadata))
    print(f"HARVEST config {args.config_index:02d} checkpoint {checkpoint.name} captured {captured} verified {verified} "
          f"(min {args.min_entries}) passed {passed} -> {out if passed else 'not written'}", flush=True)
    return 0 if passed else 1


os._exit(main())
````

- [ ] **Step 3: 컴파일** — `cd ~/rl_ws/hdgp-iker && python3 -m py_compile scripts/iker/harvest_grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/policy_player.py`. Expected: 출력 없음.

- [ ] **Step 4: 수확 스모크** (스펙 §9: A 체크포인트, 64 env × 시드 1, 통과 수 0 도 허용)
  - 쓰기 경로까지 거치게 `--min-entries 1` 로 돌린다.
  - 스크래치에만 쓴다.
  - GPU ≤ 20000 MiB 를 확인한다.

Run:
```bash
cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits && CK=$(ls -t log/rl_games/open-sens/left/iker-shoe-grasp/iker_grasp_c00_a/nn/last_*_ep_*_rew_*.pth | head -1) && echo "$CK" && OUT=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t5_harvest && mkdir -p $OUT && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_loop_t5_harvest_smoke ../IsaacLab/_isaac_sim/python.sh scripts/iker/harvest_grasp_bank.py --checkpoint "$CK" --num-envs 64 --seeds 1 --g-min 1.0 --min-entries 1 --out $OUT/grasp_bank.json --headless 2>&1 | grep -aE "HARVEST|Traceback|Error"
```
Expected:
- `HARVEST seed 1 envs 64 first-episode successes K verified V` 줄과 `HARVEST config 00 checkpoint ... captured K verified V (min 1) passed ...` 줄이 나온다.
- V ≥ 1 이면 파일이 생긴다.
- `git status --short iker_runs` 에 변화가 없다.

- [ ] **Step 5: 파일 확인** (V ≥ 1 일 때만)

Run:
```bash
cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -c "
from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb
d = run_files.read_json('/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t5_harvest/grasp_bank.json')
m = d['metadata']
assert d['joint_names'] == sorted(d['joint_names']) and m['source'] == 'learned_grasp' and m['side_sign'] == -1.0
assert len(d['entries']['joint_pos']) == m['verified'] and all(k in m for k in gb.BOOT_METADATA_KEYS)
print('BANK FILE ok', m['verified'], m['captured'], m['seeds'], m['stage1_reward']['g_min'])"
```
Expected: `BANK FILE ok V K [1] 1.0`

- [ ] **Step 6: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/policy_player.py scripts/iker/harvest_grasp_bank.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 학습 파지 뱅크 수확 — 결정론 첫 에피소드 성공 순간 캡처, env 안 복원·손 EMA 고정점·팔 +10 cm 들기 1 s 유지 검증, 통과 64 개 이상일 때만 뱅크

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 6: 2단계 id `l` 과 결정론 평가

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py`
- Modify: `scripts/iker/env_smoke.py`
- Create: `scripts/iker/eval_iker.py`

**Interfaces:**
- Consumes: Task 5 `policy_player`. 2단계 env 속성 `_keypoint_distance`, `_keypoints_local()`, `_targets`, `_success_count`, `_failure_count`, `_shoe`, `_other`, `_robot`, `_palm`, `_joint_targets`, `cfg.reward.sustain_steps`, `cfg.eval_success_distance_m`.
- Produces:
  - gym id `open-sens_l_iker_shoe`·`open-sens_l_iker_shoe-play`. 로그는 `log/rl_games/open-sens/left/iker-shoe/<RUN_LABEL>`, 체크포인트는 `last_open-sens_l_iker_shoe_ep_<E>_rew_<R>.pth`.
  - `env_smoke.py --interaction P --grasp-bank P`.
  - `eval_iker.py --checkpoint C [--config-index 0] [--num-envs 512] [--seed 7] [--no-noise] [--interaction P] [--grasp-bank P] --out S [--rows-out R] [--final-states F]`:
    - S = `{"schema": 1, "summary": {..., "success_5cm_end", "checkpoint"(절대경로), ...}}`
    - 출력 `EVAL {json}`, 예외 시 `EVAL FAILED`
    - F 는 Task 3 의 파일 규약을 따른다.

- [ ] **Step 1: gym id** — `config/__init__.py` 에서 두 곳을 고친다.
  - 첫 `for` 줄 `for _suffix, _cfg_name in (("", "IkerShoeEnvCfg"), ("-play", "IkerShoePlayEnvCfg")):` 바로 위에 넣는다.
    ```python
    # Stage 2 moves the shoe with the left arm since 2026-09-14 (logs log/rl_games/open-sens/left/iker-shoe/); the right-arm
    # K1 runs under right/iker-shoe/ stay as the comparison.
    ```
  - `        id=f"open-sens_r_iker_shoe{_suffix}",` 를 `        id=f"open-sens_l_iker_shoe{_suffix}",` 로 바꾼다.

- [ ] **Step 2: env 스모크** — `scripts/iker/env_smoke.py` 에서 세 곳을 고친다.
  - docstring `5. the right arm reaches the bank's grasp palm pose` 의 `right arm` 을 `grasping arm` 으로 바꾼다.
  - `parser.add_argument("--config-index", type=int, default=0)` 다음 줄에 넣는다.
    ```python
    parser.add_argument("--interaction", default="", help="interaction file (default: the configuration's interaction_human.json)")
    parser.add_argument("--grasp-bank", default="", help="grasp bank file (default: the configuration's grasp_bank.json)")
    ```
  - 다음 두 줄을
    ```python
        cfg.config_index = args.config_index
        env = gym.make("open-sens_r_iker_shoe", cfg=cfg).unwrapped
    ```
    다음으로 바꾼다.
    ```python
        cfg.config_index = args.config_index
        cfg.interaction_path = args.interaction
        cfg.grasp_bank_path = args.grasp_bank
        env = gym.make("open-sens_l_iker_shoe", cfg=cfg).unwrapped
    ```

- [ ] **Step 3: 옛 id 가 남지 않았다**

Run: `cd ~/rl_ws/hdgp-iker && grep -rn "open-sens_r_iker_shoe" source scripts --include=*.py`
Expected: 출력 없음(종료 코드 1).

- [ ] **Step 4: 평가 스크립트** — 아래 파일을 만든다.

````python
"""Deterministic first-episode evaluation of a stage-2 IKER checkpoint (design 2026-09-14-iker-auto-loop §4, §7).

Every env's FIRST episode is recorded (all envs start at the same reset, so there is no short-episode bias). Failures
split into drop / timeout-far / timeout-near and the end error into centroid translation vs rotation (Kabsch on the 4
keypoints). ``--final-states`` also records every env's final joints, joint targets and both shoe poses: the scene
observe.py renders for the re-query. Moved from the scratch probe that measured the right-arm K1 runs (2026-09-14).

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/eval_iker.py --checkpoint <stage-2 .pth> --out <summary.json> --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic first-episode evaluation of a stage-2 IKER checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no-noise", action="store_true", help="turn the training observation and action noise off")
parser.add_argument("--interaction", default="", help="interaction file (default: the configuration's interaction_human.json)")
parser.add_argument("--grasp-bank", default="", help="grasp bank file (default: the configuration's grasp_bank.json)")
parser.add_argument("--out", required=True, help="summary JSON")
parser.add_argument("--rows-out", default="", help="per-env rows JSON")
parser.add_argument("--final-states", default="", help="per-env final robot and shoe states JSON")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("EVAL FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe"
HOLD_RADIUS_M = 0.15  # palm-to-shoe-root distance treated as "still in hand"
NEAR_M = 0.10
ROW_KEYS = ("env", "length", "end_dist", "min_dist", "first_5cm", "sustained", "dropped", "centroid_err", "rot_deg", "shoe_z", "palm_shoe")


def kabsch_angle_deg(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N,) rotation angle in degrees of the best rigid fit of ``current`` (N, K, 3) onto ``target`` (N, K, 3)."""
    a = current - current.mean(dim=1, keepdim=True)
    b = target - target.mean(dim=1, keepdim=True)
    u, _, vh = torch.linalg.svd(a.transpose(1, 2) @ b)
    d = torch.sign(torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    rotation = vh.transpose(1, 2) @ fix @ u.transpose(1, 2)
    cos = ((rotation.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def make_env():
    cfg = IkerShoeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.add_noise = not args.no_noise
    cfg.interaction_path = args.interaction
    cfg.grasp_bank_path = args.grasp_bank
    return gym.make(TASK, cfg=cfg)


class FirstEpisodeRecorder:
    """Wraps the env's ``_get_dones`` and ``_log_episode_end`` to record each env's first episode before its reset."""

    def __init__(self, u, keep_states: bool):
        n, dev = u.num_envs, u.device
        self.u, self.keep_states = u, keep_states
        self.running_min = torch.full((n,), math.inf, device=dev)
        self.first_hit = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.recorded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rows = {key: [] for key in ROW_KEYS}
        self.states = []
        self._get_dones, self._log_episode_end = u._get_dones, u._log_episode_end
        u._get_dones, u._log_episode_end = self.get_dones, self.log_episode_end

    def get_dones(self):
        out = self._get_dones()
        u, distance = self.u, self.u._keypoint_distance
        hit = (distance <= u.cfg.eval_success_distance_m) & (self.first_hit < 0)
        self.first_hit[hit] = u.episode_length_buf[hit]
        torch.minimum(self.running_min, distance, out=self.running_min)
        return out

    def log_episode_end(self, env_ids):
        u = self.u
        finished = env_ids[(u.episode_length_buf[env_ids] > 0) & ~self.recorded[env_ids]]
        if len(finished):
            self._record(finished)
            self.recorded[finished] = True
        self.running_min[env_ids] = math.inf
        self.first_hit[env_ids] = -1
        self._log_episode_end(env_ids)

    def _record(self, finished: torch.Tensor) -> None:
        u = self.u
        origins = u.scene.env_origins[finished]
        current, target = u._keypoints_local()[finished], u._targets.expand(len(finished), 4, 3)
        shoe, palm = u._shoe.data.root_pos_w[finished], u._robot.data.body_pos_w[finished, u._palm]
        sustain = u.cfg.reward.sustain_steps
        values = {
            "env": finished, "length": u.episode_length_buf[finished], "end_dist": u._keypoint_distance[finished],
            "min_dist": self.running_min[finished], "first_5cm": self.first_hit[finished],
            "sustained": u._success_count[finished] > sustain, "dropped": u._failure_count[finished] > sustain,
            "centroid_err": (current.mean(1) - target.mean(1)).norm(dim=-1), "rot_deg": kabsch_angle_deg(current, target),
            "shoe_z": shoe[:, 2] - origins[:, 2], "palm_shoe": (palm - shoe).norm(dim=-1),
        }
        for key, value in values.items():
            self.rows[key].extend(value.cpu().tolist() if value.dtype == torch.bool else value.float().cpu().tolist())
        if self.keep_states:
            self._record_states(finished, origins)

    def _record_states(self, finished: torch.Tensor, origins: torch.Tensor) -> None:
        u = self.u
        shoe = torch.cat([u._shoe.data.root_pos_w[finished] - origins, u._shoe.data.root_quat_w[finished]], dim=-1)
        other = torch.cat([u._other.data.root_pos_w[finished] - origins, u._other.data.root_quat_w[finished]], dim=-1)
        distance = u._keypoint_distance[finished]
        for i, env in enumerate(finished.tolist()):
            self.states.append({
                "env": env, "success": bool(distance[i] <= u.cfg.eval_success_distance_m), "end_dist": float(distance[i]),
                "joint_pos": u._robot.data.joint_pos[env].tolist(), "joint_target": u._joint_targets[env].tolist(),
                "shoe_move": shoe[i].tolist(), "shoe_other": other[i].tolist(),
            })


def summarize(rows: dict, steps: int, u) -> dict:
    table = {key: torch.tensor(values) for key, values in rows.items()}
    distance = u.cfg.eval_success_distance_m
    succeeded = table["end_dist"] <= distance
    timeout = ~table["sustained"] & ~table["dropped"]
    holding = table["palm_shoe"] < HOLD_RADIUS_M
    failed = ~succeeded
    kept = failed & ~table["dropped"]

    def frac(mask):
        return round(float(mask.float().mean()), 4) if mask.numel() else None

    def q(values):
        return [round(float(v), 4) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9]))] if values.numel() else None

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()), "num_envs": u.num_envs, "episodes": len(rows["env"]), "steps": steps,
        "noise": not args.no_noise, "success_5cm_end": frac(succeeded), "sustained": frac(table["sustained"]),
        "dropped": frac(table["dropped"]), "timeout": frac(timeout), "reached_5cm_ever": frac(table["min_dist"] <= distance),
        "failures": {
            "n": int(failed.sum()), "dropped": frac(table["dropped"][failed]), "timeout_holding": frac((timeout & holding)[failed]),
            "timeout_released": frac((timeout & ~holding)[failed]), "ever_reached_5cm": frac((table["min_dist"] <= distance)[failed]),
            "min_dist_q10_50_90": q(table["min_dist"][kept]), "end_dist_q10_50_90": q(table["end_dist"][kept]),
            "centroid_err_q10_50_90": q(table["centroid_err"][kept]), "rot_deg_q10_50_90": q(table["rot_deg"][kept]),
            "near_but_rotated": frac(((table["centroid_err"] < NEAR_M) & (table["rot_deg"] > 30))[kept]),
        },
        "successes": {
            "rot_deg_q10_50_90": q(table["rot_deg"][succeeded]), "length_q10_50_90": q(table["length"][succeeded]),
            "first_5cm_q10_50_90": q(table["first_5cm"][succeeded]),
        },
    }


def main() -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env = make_env()
    u = env.unwrapped
    recorder = FirstEpisodeRecorder(u, keep_states=bool(args.final_states))
    wrapped, agent = load_player(env, TASK, checkpoint)
    steps = 0
    with torch.inference_mode():
        obs = reset_player(wrapped, agent)
        while not bool(recorder.recorded.all()) and steps < 3 * u.max_episode_length:
            obs, _ = step_player(wrapped, agent, obs)
            steps += 1
    if not recorder.rows["env"]:
        raise RuntimeError(f"no episode finished in {steps} steps")
    summary = summarize(recorder.rows, steps, u)
    run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "summary": summary})
    if args.rows_out:
        run_files.write_json(Path(args.rows_out), {"schema": run_files.SCHEMA_VERSION, "rows": recorder.rows})
    if args.final_states:
        run_files.write_json(Path(args.final_states), {
            "schema": run_files.SCHEMA_VERSION, "checkpoint": summary["checkpoint"], "noise": summary["noise"],
            "joint_names": list(u._robot.data.joint_names), "rows": sorted(recorder.states, key=lambda row: row["env"]),
        })
    print("EVAL " + json.dumps(summary), flush=True)
    return 0


os._exit(main())
````

- [ ] **Step 5: 평가 스모크** — 배선만 확인하는 스모크다.
  - 오른팔 K1 ep250 체크포인트(관측 38·행동 6 이 같다)와 접근 자세 뱅크를 쓴다.
  - 64 env 로 돌리고, 스크래치에 최종 상태를 기록한다.
  - 성공률은 보지 않는다.
  - GPU ≤ 20000 MiB 를 확인한다.

Run:
```bash
cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits && K1=$(ls log/rl_games/open-sens/right/iker-shoe/iker_human_c00_k1/nn/last_*_ep_250_rew_*.pth) && OUT=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t6_eval && mkdir -p $OUT && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_loop_t6_eval_smoke ../IsaacLab/_isaac_sim/python.sh scripts/iker/eval_iker.py --checkpoint "$K1" --num-envs 64 --grasp-bank iker_runs/shoe_place/config_00/pregrasp_bank.json --out $OUT/summary.json --rows-out $OUT/rows.json --final-states $OUT/final_states.json --headless 2>&1 | grep -aE "^EVAL|Traceback|Error" | cut -c1-300
```
Expected: `EVAL {"checkpoint": ".../last_open-sens_r_iker_shoe_ep_250_rew_...pth", "num_envs": 64, "episodes": 64, ...` 한 줄.

- [ ] **Step 6: 파일 확인**

Run:
```bash
cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -c "
from openarm.agnostic.modules.iker import loop_probe as lp, loop_state as ls, run_files
o = '/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t6_eval/'
s = run_files.read_json(o + 'summary.json')['summary']; f = run_files.read_json(o + 'final_states.json')
rows = lp.final_rows(__import__('pathlib').Path(o + 'final_states.json'))
assert s['episodes'] == 64 and len(f['rows']) == 64 and len(f['rows'][0]['joint_pos']) == len(f['joint_names']) and len(f['rows'][0]['shoe_move']) == 7
print('EVAL FILES ok success', s['success_5cm_end'], 'pick', ls.pick_observe_env(rows))"
```
Expected: `EVAL FILES ok success <x> pick <env|None>`

- [ ] **Step 7: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py scripts/iker/env_smoke.py scripts/iker/eval_iker.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 2단계 gym id 를 open-sens_l_iker_shoe 로(왼팔, 로그 left/iker-shoe), 결정론 첫 에피소드 평가·최종 상태 기록 스크립트, env 스모크 입력 경로 인자

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 7: 헤드 스냅샷 공용화와 실행 뒤 장면 렌더

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/head_snapshot.py`
- Replace: `scripts/iker/snapshot.py`
- Create: `scripts/iker/observe.py`

**Interfaces:**
- Consumes: Task 6 의 final states 파일 규약. `gate.kabsch`, `rotations.matrix_to_quat_wxyz`, `run_files.keypoints_document`.
- Produces:
  - `head_snapshot` 모듈:
    - `build_scene(poses, device) -> HeadScene`, `head_target(arm, joint_target) -> (target, HeadAim)`, `settle(scene, target, settle_steps, render_steps)`
    - `capture(scene, traces, meta, aim) -> Snapshot`(`failures` 는 `margin`·`head`·`settle`·`rack_depth`·`support` 키)
    - `write(snapshot, out_dir, scene_config, ignore=()) -> bool`
  - `snapshot.py --out-dir`.
  - `observe.py` 인자와 출력:
    - 인자 `--config-index 0`, `--states F --env K` 또는 `--human-target`, `--out-dir D`
    - `D` 에 `snapshot_raw.png`·`snapshot.png`·`keypoints.json`·`state.json` 을 쓴다.
    - 출력 `OBSERVE config .. passed ...`, 예외 시 `OBSERVE FAILED`

- [ ] **Step 1: 공용 모듈** — 아래 파일을 만든다(`snapshot.py` 의 `build_scene`·`settle`·주석·검사 본문을 옮긴 것).

````python
"""Head-camera scene, settling and keypoint snapshot shared by snapshot.py and observe.py (design spec §5, auto-loop §4).

Moved out of scripts/iker/snapshot.py unchanged in behaviour. Import after the Isaac app launcher has started with
cameras enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch
from PIL import Image

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg, SimulationContext

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.projection import CameraPose, camera_pose_from_link
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix
from openarm.agnostic.modules.iker.scene_image import AnnotatedSnapshot, PointGroup, annotate_snapshot
from openarm.sensors.head_camera import head_camera_cfg, load_spec, urdf_head_angles

from . import layout, robot, scene_cfg

PHYSICS_DT = 1.0 / 120.0
ENV_PRIM = "/World/envs/env_0"
SETTLE_WINDOW_STEPS = 30
MAX_SETTLE_MOTION_M = 0.005
MAX_HEAD_ERROR_DEG = 0.5
MAX_RACK_DEPTH_ERROR_M = 0.005


@dataclass(frozen=True)
class HeadScene:
    sim: SimulationContext
    arm: Articulation
    shoes: Mapping[str, RigidObject]
    camera: Camera
    spec: object


@dataclass(frozen=True)
class HeadAim:
    pan_id: int
    tilt_id: int
    pan_deg: float
    tilt_deg: float


@dataclass(frozen=True)
class Snapshot:
    rgb: np.ndarray
    annotated: AnnotatedSnapshot
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]]
    camera_pose: CameraPose
    intrinsic: np.ndarray
    checks: Mapping[str, object]  # head_error_deg, settle_motion_m, rack_depth_error_max_m
    failures: Mapping[str, str]  # check name (margin, head, settle, rack_depth, support) -> message


def build_scene(poses: Mapping[str, tuple], device: str) -> HeadScene:
    """Table, rack, light, the robot and the shoes at ``poses`` under env_0, with the head camera."""
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, render_interval=1, device=device))
    prim_utils.create_prim(ENV_PRIM, "Xform")
    for cfg, prim in ((scene_cfg.table_cfg(), f"{ENV_PRIM}/Table"), (scene_cfg.rack_cfg(), f"{ENV_PRIM}/Rack")):
        cfg.spawn.func(prim, cfg.spawn, translation=cfg.init_state.pos)
    light = scene_cfg.light_cfg()
    light.spawn.func(light.prim_path, light.spawn)
    arm = Articulation(robot.robot_cfg().replace(prim_path=f"{ENV_PRIM}/Robot"))
    shoes = {
        name: RigidObject(scene_cfg.shoe_cfg(name, pos, quat).replace(prim_path=f"{ENV_PRIM}/{name}"))
        for name, (pos, quat) in poses.items()
    }
    spec = load_spec()
    camera = Camera(
        head_camera_cfg(spec, data_types=("rgb", "distance_to_image_plane")).replace(
            prim_path=f"{ENV_PRIM}/Robot/{spec.link}/head_cam_real"
        )
    )
    sim.reset()
    if not camera.is_initialized:  # the camera is created after the robot; see openarm.sensors.head_camera
        camera._initialize_impl()
        camera._is_initialized = True
    return HeadScene(sim, arm, shoes, camera, spec)


def head_target(arm: Articulation, joint_target: torch.Tensor) -> tuple[torch.Tensor, HeadAim]:
    """``joint_target`` (1, J) with the head joints at the snapshot angles."""
    aim = HeadAim(
        arm.find_joints(robot.HEAD_PAN_JOINT)[0][0],
        arm.find_joints(robot.HEAD_TILT_JOINT)[0][0],
        *urdf_head_angles(layout.HEAD_PAN_ENCODER_DEG, layout.HEAD_TILT_ENCODER_DEG),
    )
    target = joint_target.clone()
    target[:, aim.pan_id], target[:, aim.tilt_id] = math.radians(aim.pan_deg), math.radians(aim.tilt_deg)
    return target, aim


def settle(scene: HeadScene, target: torch.Tensor, settle_steps: int, render_steps: int) -> dict[str, np.ndarray]:
    """Hold ``target`` (re-commanded every step) with arm gravity compensation; returns each shoe's position trace."""
    gravity_ids, _ = scene.arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    traces = {name: [] for name in scene.shoes}
    for step in range(settle_steps):
        scene.arm.set_joint_position_target(target)
        tau = scene.arm.root_physx_view.get_gravity_compensation_forces()
        scene.arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
        scene.arm.write_data_to_sim()
        render = step >= settle_steps - render_steps
        scene.sim.step(render=render)
        scene.arm.update(PHYSICS_DT)
        for name, shoe in scene.shoes.items():
            shoe.update(PHYSICS_DT)
            traces[name].append(shoe.data.root_pos_w[0].cpu().numpy().copy())
        if render:
            scene.camera.update(PHYSICS_DT)
    return {name: np.asarray(trace) for name, trace in traces.items()}


def capture(scene: HeadScene, traces: Mapping[str, np.ndarray], meta: Mapping, aim: HeadAim) -> Snapshot:
    """Render, annotate the keypoints and run the snapshot checks."""
    arm = scene.arm
    link = arm.find_bodies(scene.spec.link)[0][0]
    camera_pose = camera_pose_from_link(
        arm.data.body_pos_w[0, link].cpu().numpy(), arm.data.body_quat_w[0, link].cpu().numpy(), scene.spec.pos, scene.spec.quat_wxyz
    )
    intrinsic = scene.camera.data.intrinsic_matrices[0].cpu().numpy()
    rgb = scene.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    depth = scene.camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
    poses = {name: (s.data.root_pos_w[0].cpu().numpy(), s.data.root_quat_w[0].cpu().numpy()) for name, s in scene.shoes.items()}

    groups = [
        PointGroup(name, False, pos + np.asarray(meta["objects"][name]["keypoint_offsets"]) @ quat_wxyz_to_matrix(quat).T)
        for name, (pos, quat) in poses.items()
    ]
    groups.append(PointGroup(layout.RACK, True, layout.rack_keypoints()))
    anchor = np.concatenate([group.points_w for group in groups]).mean(axis=0)
    annotated = annotate_snapshot(rgb, depth, camera_pose, intrinsic, groups, anchor, layout.OVERLAP_MIN_PX)

    head_error = max(
        abs(math.degrees(arm.data.joint_pos[0, aim.pan_id].item()) - aim.pan_deg),
        abs(math.degrees(arm.data.joint_pos[0, aim.tilt_id].item()) - aim.tilt_deg),
    )
    settle_motion = max(
        float(np.linalg.norm(trace[-SETTLE_WINDOW_STEPS:] - trace[-1], axis=1).max()) for trace in traces.values()
    )
    rack_errors = [abs(r.render_depth - r.depth) for r in annotated.records if r.is_static and r.visible and r.kept]
    rack_depth_error = max(rack_errors) if rack_errors else None
    failures = {}
    if not annotated.min_margin_px >= layout.MIN_BORDER_MARGIN_PX:
        failures["margin"] = f"keypoint border margin {annotated.min_margin_px:.1f} px < {layout.MIN_BORDER_MARGIN_PX}"
    if not head_error <= MAX_HEAD_ERROR_DEG:
        failures["head"] = f"head angle error {head_error:.2f} deg > {MAX_HEAD_ERROR_DEG}"
    if not settle_motion <= MAX_SETTLE_MOTION_M:
        failures["settle"] = f"shoes still moving {settle_motion * 1000:.1f} mm over the last {SETTLE_WINDOW_STEPS} steps"
    if rack_depth_error is None or not rack_depth_error <= MAX_RACK_DEPTH_ERROR_M:
        failures["rack_depth"] = f"rack keypoint depth vs render depth {rack_depth_error} m > {MAX_RACK_DEPTH_ERROR_M}"
    try:
        run_files.movable_objects(
            {"keypoints": [vars(r) for r in annotated.records], "objects": {n: {"position": p.tolist()} for n, (p, _) in poses.items()}},
            meta,
            layout.supports(),
        )
    except ValueError as exc:
        failures["support"] = str(exc)
    checks = {"head_error_deg": head_error, "settle_motion_m": settle_motion, "rack_depth_error_max_m": rack_depth_error}
    return Snapshot(rgb, annotated, poses, camera_pose, intrinsic, checks, failures)


def write(snapshot: Snapshot, out_dir: Path, scene_config, ignore: tuple[str, ...] = ()) -> bool:
    """snapshot_raw.png, snapshot.png and keypoints.json; True when every check outside ``ignore`` passed."""
    failures = [message for name, message in snapshot.failures.items() if name not in ignore]
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot.rgb).save(out_dir / "snapshot_raw.png")
    snapshot.annotated.image.save(out_dir / "snapshot.png")
    head = {"pan_encoder_deg": layout.HEAD_PAN_ENCODER_DEG, "tilt_encoder_deg": layout.HEAD_TILT_ENCODER_DEG}
    doc = run_files.keypoints_document(
        scene_config=vars(scene_config), camera=snapshot.camera_pose, intrinsic=snapshot.intrinsic, snapshot=snapshot.annotated,
        image_size=(snapshot.rgb.shape[1], snapshot.rgb.shape[0]), object_poses=snapshot.poses, head=head,
        checks={"passed": not failures, "failures": failures, **snapshot.checks},
    )
    run_files.write_json(out_dir / "keypoints.json", doc)
    return not failures
````

- [ ] **Step 2: snapshot.py 교체** — 파일 전체를 아래로 바꾼다.

````python
"""Head-camera snapshot of one IKER configuration (design spec §5).

Spawns the scene, holds the head at the snapshot angles while the shoes settle, renders RGB and
depth, and writes iker_runs/shoe_place/config_XX/{snapshot_raw.png, snapshot.png, keypoints.json}.
The files are written even when a check fails (``checks.passed`` is false) and the exit code is 1.
The scene, settling and checks live in tasks/iker_shoe/head_snapshot.py (shared with observe.py).

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/snapshot.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of one IKER configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
parser.add_argument("--out-dir", default="", help="output directory (default: iker_runs/shoe_place/config_XX)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("SNAPSHOT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import head_snapshot as hs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402


def main() -> int:
    if not args.settle_steps > max(hs.SETTLE_WINDOW_STEPS, args.render_steps):
        raise ValueError("--settle-steps must exceed both the settle window and --render-steps")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    scene = hs.build_scene(layout.shoe_start_poses(config, meta), args.device)

    home = scene.arm.data.default_joint_pos.clone()
    scene.arm.write_joint_state_to_sim(home, torch.zeros_like(home))
    target, aim = hs.head_target(scene.arm, home)
    traces = hs.settle(scene, target, args.settle_steps, args.render_steps)
    snapshot = hs.capture(scene, traces, meta, aim)

    out_dir = Path(args.out_dir) if args.out_dir else layout.RUNS_DIR / f"config_{config.index:02d}"
    passed = hs.write(snapshot, out_dir, config)
    annotated, checks = snapshot.annotated, snapshot.checks
    print(
        f"SNAPSHOT config {config.index:02d} derotation {annotated.derotation_deg:+.1f} deg margin {annotated.min_margin_px:+.1f} px "
        f"gap {annotated.min_gap_px} px head_err {checks['head_error_deg']:.2f} deg settle {checks['settle_motion_m'] * 1000:.2f} mm "
        f"rack_depth_err {checks['rack_depth_error_max_m']} m passed {passed} -> {out_dir}",
        flush=True,
    )
    for failure in snapshot.failures.values():
        print(f"SNAPSHOT CHECK FAILED: {failure}", flush=True)
    return 0 if passed else 1


os._exit(main())
````

- [ ] **Step 3: 관측 렌더** — 아래 파일을 만든다.

````python
"""Head-camera snapshot of an executed IKER scene (design 2026-09-14-iker-auto-loop §4 observe_requery, §9).

``--states <final_states.json> --env K`` places the robot joints, their targets and both shoes at env K's final state of a
noise-free rollout (eval_iker.py --final-states). ``--human-target`` places the moving shoe at the human baseline target
pose with the robot at home instead (the observation smoke). The head holds the snapshot angles while the scene settles;
the render is annotated as in snapshot.py and written, with the placed state (state.json), to --out-dir. Every snapshot
check applies except settling, which is reported only: a closed hand may still be adjusting its hold.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/observe.py --states <final_states.json> --env <k> --out-dir <dir> --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of an executed IKER scene.")
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--states", default="", help="final_states.json written by eval_iker.py --final-states")
parser.add_argument("--env", type=int, default=-1, help="the env of --states to place")
parser.add_argument("--human-target", action="store_true", help="place the moving shoe at the human baseline target instead")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("OBSERVE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import head_snapshot as hs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402

REPORTED_CHECKS = ("settle",)
SHOE_KEYS = ((layout.MOVING_SHOE, "shoe_move"), (layout.OTHER_SHOE, "shoe_other"))


def rollout_state(path: Path, env: int) -> dict:
    doc = run_files.read_json(path)
    row = next((row for row in doc["rows"] if row["env"] == env), None)
    if row is None:
        raise ValueError(f"{path} holds no env {env}")
    return {
        "schema": run_files.SCHEMA_VERSION, "source": "noise_free_rollout", "checkpoint": doc["checkpoint"], "env": env,
        "joint_names": doc["joint_names"],
        **{key: row[key] for key in ("success", "end_dist", "joint_pos", "joint_target", "shoe_move", "shoe_other")},
    }


def human_target_state(config_dir: Path, meta: dict) -> dict:
    """The moving shoe at the rigid fit of its keypoints onto the human baseline targets; the other shoe as snapshotted."""
    interaction = run_files.read_json(config_dir / "interaction_human.json")
    snapshot = run_files.read_json(config_dir / "keypoints.json")
    offsets = np.asarray(meta["objects"][layout.MOVING_SHOE]["keypoint_offsets"], dtype=float)
    rotation, center = kabsch(offsets, np.asarray(interaction["target_keypoints"], dtype=float))
    center = center + np.array([0.0, 0.0, layout.SPAWN_CLEARANCE_M])
    other = snapshot["objects"][layout.OTHER_SHOE]
    return {
        "schema": run_files.SCHEMA_VERSION, "source": "human_target", "checkpoint": None, "env": None, "joint_names": None,
        "success": None, "end_dist": None, "joint_pos": None, "joint_target": None,
        "shoe_move": [*center.tolist(), *matrix_to_quat_wxyz(rotation).tolist()],
        "shoe_other": [*other["position"], *other["quat_wxyz"]],
    }


def main() -> int:
    if args.human_target == bool(args.states):
        raise ValueError("give exactly one of --states (with --env) and --human-target")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    config_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    state = human_target_state(config_dir, meta) if args.human_target else rollout_state(Path(args.states), args.env)
    poses = {name: (np.asarray(state[key][:3]), np.asarray(state[key][3:])) for name, key in SHOE_KEYS}
    scene = hs.build_scene(poses, args.device)
    joint_pos = scene.arm.data.default_joint_pos.clone()
    joint_target = joint_pos.clone()
    if state["joint_names"] is not None:
        order = [state["joint_names"].index(name) for name in scene.arm.data.joint_names]
        joint_pos[0] = torch.tensor(state["joint_pos"], device=joint_pos.device)[order]
        joint_target[0] = torch.tensor(state["joint_target"], device=joint_pos.device)[order]
    target, aim = hs.head_target(scene.arm, joint_target)
    joint_pos[:, aim.pan_id], joint_pos[:, aim.tilt_id] = target[:, aim.pan_id], target[:, aim.tilt_id]
    scene.arm.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    traces = hs.settle(scene, target, args.settle_steps, args.render_steps)
    snapshot = hs.capture(scene, traces, meta, aim)
    out_dir = Path(args.out_dir)
    passed = hs.write(snapshot, out_dir, config, ignore=REPORTED_CHECKS)
    run_files.write_json(out_dir / "state.json", state)
    print(f"OBSERVE config {config.index:02d} source {state['source']} env {state['env']} margin "
          f"{snapshot.annotated.min_margin_px:+.1f} px settle {snapshot.checks['settle_motion_m'] * 1000:.2f} mm passed {passed} -> {out_dir}",
          flush=True)
    for name, failure in snapshot.failures.items():
        print(f"OBSERVE CHECK {'REPORTED' if name in REPORTED_CHECKS else 'FAILED'}: {failure}", flush=True)
    return 0 if passed else 1


os._exit(main())
````

- [ ] **Step 4: 스냅샷 회귀** — 구성 0 을 스크래치로 다시 찍어 커밋된 파일과 대조한다. GPU ≤ 20000 MiB 를 확인한다.

Run:
```bash
cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits && OUT=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t7_snapshot && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_loop_t7_snapshot ../IsaacLab/_isaac_sim/python.sh scripts/iker/snapshot.py --config-index 0 --out-dir $OUT --headless 2>&1 | grep -aE "SNAPSHOT|Traceback" && python3 - <<'PY'
import json
new = json.load(open("/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t7_snapshot/keypoints.json"))
old = json.load(open("iker_runs/shoe_place/config_00/keypoints.json"))
pairs = list(zip(new["keypoints"], old["keypoints"]))
assert [n["label"] for n, _ in pairs] == [o["label"] for _, o in pairs] and len(pairs) == len(old["keypoints"])
world = max(abs(a - b) for n, o in pairs for a, b in zip(n["world"], o["world"]))
pixel = max(abs(a - b) for n, o in pairs for a, b in zip(n["uv"], o["uv"]))
print(f"SNAPSHOT REGRESSION world {world * 1000:.2f} mm pixel {pixel:.2f} px passed {new['checks']['passed']} vs {old['checks']['passed']}")
PY
```
Expected: `SNAPSHOT config 00 ... passed True -> .../t7_snapshot`, `SNAPSHOT REGRESSION world ≤ 1.00 mm pixel ≤ 1.00 px passed True vs True`. `git status --short iker_runs` 에 변화가 없다.

- [ ] **Step 5: 관측 스모크**(스펙 §9) — 사람 기준선 목표 자세로 신발을 놓고 렌더한다.

Run: `cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_loop_t7_observe ../IsaacLab/_isaac_sim/python.sh scripts/iker/observe.py --human-target --out-dir /tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t7_observe --headless 2>&1 | grep -aE "OBSERVE|Traceback"`
Expected:
- `OBSERVE config 00 source human_target env None margin +M px settle ... passed True -> ...` 가 나오고, M ≥ 15 다.
- `settle` 이 넘쳤다면 `OBSERVE CHECK REPORTED` 줄만 허용한다.
- 산출 `snapshot.png` 를 Read 로 열어 두 신발이 받침 위에 나란히 보이는지 확인한다.

- [ ] **Step 6: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/head_snapshot.py scripts/iker/snapshot.py scripts/iker/observe.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 헤드 스냅샷 장면·정착·검사를 공용 모듈로(동작 불변, 구성 0 재촬영 대조), 실행 뒤 최종 상태·사람 목표 자세 렌더 observe.py

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 8: 루프 CLI 와 틱 절차서

**Files:**
- Create: `scripts/iker/loop.py`
- Create: `scripts/iker/LOOP_PROMPT.md`
- Create: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_cli.py`

**Interfaces:**
- Consumes:
  - Task 2·3 전부.
  - Task 1·5·6·7 의 스크립트와 인자: `measure_grasp_quality.py --checkpoint`, `harvest_grasp_bank.py --g-min --min-entries --out`, `env_smoke.py --interaction --grasp-bank`, `eval_iker.py --no-noise --out --rows-out --final-states`, `observe.py --states --env --out-dir`.
  - `train.py --no-reset_epoch`(워크트리 `scripts/reinforcement_learning/rl_games/train.py`), `play.py --video --video_length`.
  - `scripts/tools/parse_tfevents.load_tfevents`, `ingest.py --run-dir --attempt`.
- Produces:
  - 명령:
    - `loop.py [--state-dir D] [--no-commit] [--session-url U] init [--track T] [--phase P] [--policy JSON] [--adopt RUN ...]`
    - `status`: JSON 으로 `phase`, `status`, `awaiting`, `action`, `reason`, `params`, `notes`, `summary`, `vlm_generate` 일 때 `vlm`{prompt, response, images, brief}
    - `act ACTION [--policy JSON]`
  - 사이드 런 라벨은 `<track>_<run>` 이다. 로그는 `log/iker_loop/<track>/<run>_<tag>.log` 에 둔다.
  - 모듈 함수 `main(argv)`, `vlm_request(paths, decision)`.

- [ ] **Step 1: 실패하는 테스트** — 아래 파일을 만든다.

````python
"""scripts/iker/loop.py — init, the user's actions and the generator request (design auto-loop §3, §6; no Isaac, no probe)."""

import importlib.util
import json

import pytest

from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import prompts
from openarm.agnostic.tasks.iker_shoe import layout

SPEC = importlib.util.spec_from_file_location("iker_loop_cli", layout.HDGP_ROOT / "scripts" / "iker" / "loop.py")
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


def _history(state_dir):
    return [json.loads(line) for line in (state_dir / "history.jsonl").read_text().splitlines()]


def test_init_writes_the_policy_overrides_once(tmp_path, capsys):
    state_dir = tmp_path / "mock_loop"
    policy = json.dumps({"stage2_epochs": 10, "labels": {"stage2": "iker_vlm_mock"}})
    assert loop.main(["--state-dir", str(state_dir), "init", "--track", "iker_shoe_c00_mock", "--phase", "vlm_target", "--policy", policy]) == 0
    state = ls.load_state(state_dir / "LOOP_STATE.json")
    assert (state["track"], state["phase"], state["policy"]["stage2_epochs"]) == ("iker_shoe_c00_mock", "vlm_target", 10)
    assert state["policy"]["labels"]["stage2"] == "iker_vlm_mock" and state["runs"] == {}
    assert _history(state_dir)[0]["action"] == "init"
    with pytest.raises(SystemExit, match="exists"):
        loop.main(["--state-dir", str(state_dir), "init"])


def test_resume_and_approve_are_recorded_without_a_probe(tmp_path, capsys):
    state_dir = tmp_path / "mock_loop"
    loop.main(["--state-dir", str(state_dir), "init", "--phase", "completion_review"])
    paused = ls.apply(ls.load_state(state_dir / "LOOP_STATE.json"), ls.Decision("pause", "review", {"needs": "watch"}), "t")
    ls.save_state(state_dir / "LOOP_STATE.json", paused)
    assert loop.main(["--state-dir", str(state_dir), "--no-commit", "act", "resume", "--policy", '{"success_target": 0.6}']) == 0
    resumed = ls.load_state(state_dir / "LOOP_STATE.json")
    assert resumed["status"] == "running" and resumed["policy"]["success_target"] == 0.6
    assert loop.main(["--state-dir", str(state_dir), "--no-commit", "act", "approve"]) == 0
    done = ls.load_state(state_dir / "LOOP_STATE.json")
    assert (done["phase"], done["status"]) == ("done", "done")
    assert [record["action"] for record in _history(state_dir)] == ["init", "resume", "approve"]


def test_vlm_request_gives_the_generator_only_its_files(tmp_path):
    paths = loop.lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00")
    target = loop.vlm_request(paths, ls.Decision("vlm_generate", "attempt 01", {"kind": "target", "attempt": 1}))
    assert target["response"] == str(paths.stage_dir / "attempt_01" / "response.md")
    assert target["images"] == {prompts.IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    requery = loop.vlm_request(paths, ls.Decision("vlm_generate", "requery", {"kind": "requery"}))
    assert requery["prompt"] == str(paths.observe_dir / "prompt.md")
    assert set(requery["images"]) == {prompts.IMAGE_MARKER, loop.loop_vlm.STAGE_IMAGE_MARKER}
    assert requery["brief"].startswith(f"Read the prompt file {paths.observe_dir / 'prompt.md'}")
````

- [ ] **Step 2: 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_loop_cli.py -q -p no:cacheprovider`
Expected: FAIL — collection error `FileNotFoundError: ... scripts/iker/loop.py`

- [ ] **Step 3: CLI** — 아래 파일을 만든다.

````python
#!/usr/bin/env python3
"""IKER auto-loop CLI (design docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md §3, §8).

System python3; Isaac runs only in child processes started here, identified by RUN_LABEL and stopped by PID.

    python3 scripts/iker/loop.py [--state-dir DIR] init [--track T] [--phase P] [--policy JSON] [--adopt RUN ...]
    python3 scripts/iker/loop.py [--state-dir DIR] status
    python3 scripts/iker/loop.py [--state-dir DIR] [--no-commit] act ACTION [--policy JSON]

``act`` decides again from a fresh probe and refuses an action that is no longer the decision. ``approve`` and
``resume`` are the user's words. scripts/iker/LOOP_PROMPT.md is the tick procedure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "openarm"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from parse_tfevents import load_tfevents  # noqa: E402

from openarm.agnostic.modules.iker import loop_probe as lp  # noqa: E402
from openarm.agnostic.modules.iker import loop_state as ls  # noqa: E402
from openarm.agnostic.modules.iker import loop_vlm, prompts, run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402

CONFIG_INDEX = 0
ISAAC_PYTHON = ROOT.parent / "IsaacLab" / "_isaac_sim" / "python.sh"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train.py"
PLAY_SCRIPT = "scripts/reinforcement_learning/rl_games/play.py"
STAGE1_TASK, STAGE2_TASK = "open-sens_l_iker_shoe_grasp", "open-sens_l_iker_shoe"
STAGE1_NUM_ENVS = 4096
VIDEO_ENVS, VIDEO_STEPS = 16, 200
VIDEO_DIR = ROOT.parent / "our_source"
STOP_TIMEOUT_S, PID_WAIT_S = 300.0, 120.0
SESSION_URL = "https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
COMMIT_ACTIONS = (
    "record_gate", "advance", "commit_calibration", "commit_bank", "commit_interaction", "record_eval", "store_video", "pause",
    "approve", "resume",
)
OBSERVE_FILES = ("prompt.md", "response.md", "requery.json", "snapshot.png", "snapshot_raw.png", "keypoints.json", "state.json",
                 "rollout_summary.json")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return lp.parse_gpu_used_mib(out.stdout)


def head_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


# ------------------------------------------------------------------ processes


def launch(label: str, argv: list[str], log: Path, note: str) -> dict:
    """Start a detached Isaac child and return its launch record once a process with its RUN_LABEL is visible."""
    if lp.label_pids(label):
        raise RuntimeError(f"a process already carries RUN_LABEL={label}")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TERM": "xterm", "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONPATH": str(ROOT / "source" / "openarm"),
           "RUN_LABEL": label, "NOTE": note}
    started = time.time()
    with log.open("wb") as out:
        subprocess.Popen([str(ISAAC_PYTHON), *argv], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out,
                         stderr=subprocess.STDOUT, start_new_session=True)
    while not (pids := lp.label_pids(label)):
        if time.time() > started + PID_WAIT_S:
            raise RuntimeError(f"no Isaac process with RUN_LABEL={label} after {PID_WAIT_S:.0f} s; see {log}")
        time.sleep(2.0)
    return {"label": label, "log": str(log), "pid": pids[0], "started": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"),
            "started_s": started, "commit": head_commit(), "argv": argv}


def stop(label: str, force: bool = False) -> list[int]:
    """Stop the Isaac processes of ``label`` by PID (never by a name pattern); SIGKILL only for a hung run (§8)."""
    pids = lp.label_pids(label)
    for pid in pids:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    deadline = time.time() + STOP_TIMEOUT_S
    while any(Path(f"/proc/{pid}").exists() for pid in pids):
        if time.time() > deadline:
            raise RuntimeError(f"RUN_LABEL={label} PIDs {pids} are alive {STOP_TIMEOUT_S:.0f} s after the signal")
        time.sleep(2.0)
    return pids


def commit(paths: list[Path], message: str, session_url: str) -> str | None:
    """Commit exactly ``paths`` (no -A); None when nothing changed."""
    rel = sorted({str(Path(p).resolve().relative_to(ROOT)) for p in paths if Path(p).exists()})
    subprocess.run(["git", "-C", str(ROOT), "add", "--", *rel], check=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "--", *rel]).returncode == 0:
        return None
    body = f"{message}\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: {session_url}"
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", body, "--", *rel], check=True)
    return head_commit()


# ------------------------------------------------------------------ executors


def side_label(state: dict, run: str) -> str:
    return f"{state['track']}_{run}"


def stage2_inputs(state: dict, paths: lp.LoopPaths) -> list[str]:
    bank = state["policy"]["grasp_bank_path"]
    return ["--interaction", str(paths.interaction_file), *(["--grasp-bank", bank] if bank else [])]


def stage2_overrides(state: dict, paths: lp.LoopPaths) -> list[str]:
    """Hydra overrides of the stage-2 inputs; quoted, since a path segment may start with '-'."""
    bank = state["policy"]["grasp_bank_path"]
    return [f"env.interaction_path='{paths.interaction_file}'", *([f"env.grasp_bank_path='{bank}'"] if bank else [])]


def run_calibrate(state, paths, decision):
    epoch = decision.params["epoch"]
    argv = ["scripts/iker/measure_grasp_quality.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX), "--headless"]
    return {"run": launch(side_label(state, "calibrate"), argv, paths.side_log("calibrate", f"ep{epoch}"), f"IKER loop q calibration ep {epoch}")}


def commit_calibration(state, paths, decision):
    return {"commit_paths": [paths.calibration_file], "message": f"iker(loop): 1단계 q 보정 파일 — {Path(decision.params['checkpoint']).name}"}


def launch_b(state, paths, decision):
    policy, labels = state["policy"], state["policy"]["labels"]
    stopped = ["stage1_a"] if stop(labels["stage1_a"]) else []
    argv = [TRAIN_SCRIPT, "--task", STAGE1_TASK, "--num_envs", str(STAGE1_NUM_ENVS), "--headless", "--checkpoint", decision.params["checkpoint"],
            "--no-reset_epoch", "--max_iterations", str(policy["b_max_epochs"]), f"env.grasp_reward.g_min={policy['b_g_min']}"]
    note = f"IKER 1단계 B(g_min {policy['b_g_min']}, 보정 파일) {Path(decision.params['checkpoint']).name} 이어학습, 자동 루프"
    return {"run": launch(labels["stage1_b"], argv, paths.train_log(labels["stage1_b"]), note), "stopped_runs": stopped}


def run_harvest(state, paths, decision):
    policy, epoch = state["policy"], decision.params["epoch"]
    argv = ["scripts/iker/harvest_grasp_bank.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX),
            "--g-min", str(policy["b_g_min"]), "--min-entries", str(policy["harvest_min"]), "--out", str(paths.harvest_bank_file), "--headless"]
    return {"run": launch(side_label(state, "harvest"), argv, paths.side_log("harvest", f"ep{epoch}"), f"IKER loop harvest ep {epoch}")}


def commit_bank(state, paths, decision):
    stopped = ["stage1_b"] if stop(state["policy"]["labels"]["stage1_b"]) else []
    return {"path": str(paths.harvest_bank_file), "stopped_runs": stopped, "commit_paths": [paths.harvest_bank_file],
            "message": f"iker(loop): 학습 파지 뱅크 {decision.params['verified']} 개 — {Path(decision.params['checkpoint']).name}"}


def write_prompt(state, paths, decision):
    paths.stage_dir.mkdir(parents=True, exist_ok=True)
    (paths.stage_dir / "prompt.md").write_text(loop_vlm.target_prompt(layout.TASK_INSTRUCTION), encoding="utf-8")
    for name in ("snapshot.png", "keypoints.json"):
        shutil.copyfile(paths.config_dir / name, paths.stage_dir / name)
    return {}


def ingest(state, paths, decision):
    attempt = decision.params["attempt"]
    done = subprocess.run([sys.executable, "scripts/iker/ingest.py", "--run-dir", str(paths.stage_dir), "--attempt", str(attempt)], cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "source" / "openarm")}, capture_output=True, text=True)
    if not (paths.stage_dir / f"attempt_{attempt:02d}" / "gate.json").is_file():
        raise RuntimeError(f"ingest.py wrote no gate verdict (exit {done.returncode}):\n{done.stdout}{done.stderr}")
    return {"passed": done.returncode == 0, "output": done.stdout.strip()}


def commit_interaction(state, paths, decision):
    stage = paths.stage_dir
    files = [stage / "prompt.md", stage / "snapshot.png", stage / "keypoints.json", paths.interaction_file, *sorted(stage.glob("attempt_*/*"))]
    return {"commit_paths": files, "message": f"iker(loop): 1단계 VLM 목표 — attempt {decision.params['attempt']:02d} 게이트 통과"}


def run_env_smoke(state, paths, decision):
    argv = ["scripts/iker/env_smoke.py", "--config-index", str(CONFIG_INDEX), *stage2_inputs(state, paths), "--headless"]
    return {"run": launch(side_label(state, "env_smoke"), argv, paths.side_log("env_smoke", "vlm_target"), "IKER loop stage-2 env smoke")}


def launch_stage2(state, paths, decision):
    policy, label = state["policy"], state["policy"]["labels"]["stage2"]
    argv = [TRAIN_SCRIPT, "--task", STAGE2_TASK, "--num_envs", str(policy["stage2_num_envs"]), "--max_iterations", str(policy["stage2_epochs"]),
            "--headless", "env.interaction_source=vlm", *stage2_overrides(state, paths)]
    if policy["minibatch_size"]:
        argv.append(f"agent.params.config.minibatch_size={policy['minibatch_size']}")
    return {"run": launch(label, argv, paths.train_log(label), "IKER 2단계 VLM 목표·학습 파지 뱅크, 자동 루프")}


def run_eval(state, paths, decision):
    epoch = decision.params["epoch"]
    argv = ["scripts/iker/eval_iker.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX), *stage2_inputs(state, paths),
            "--out", str(paths.eval_file(epoch)), "--rows-out", str(paths.eval_rows_file(epoch)), "--headless"]
    return {"run": launch(side_label(state, "eval"), argv, paths.side_log("eval", f"ep{epoch}"), f"IKER loop eval ep {epoch}")}


def record_eval(state, paths, decision):
    epoch, success = decision.params["epoch"], decision.params["summary"]["success_5cm_end"]
    return {"commit_paths": [paths.eval_file(epoch)], "message": f"iker(loop): 2단계 ep {epoch} 평가 — 성공 {success}"}


def run_observe_rollout(state, paths, decision):
    argv = ["scripts/iker/eval_iker.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX), "--no-noise",
            *stage2_inputs(state, paths), "--out", str(paths.observe_dir / "rollout_summary.json"), "--final-states", str(paths.final_states_file),
            "--headless"]
    log = paths.side_log("observe_rollout", f"ep{decision.params['epoch']}")
    return {"run": launch(side_label(state, "observe_rollout"), argv, log, "IKER loop noise-free rollout")}


def run_observe_render(state, paths, decision):
    env = decision.params["env"]
    argv = ["scripts/iker/observe.py", "--config-index", str(CONFIG_INDEX), "--states", str(paths.final_states_file), "--env", str(env),
            "--out-dir", str(paths.observe_dir), "--headless"]
    return {"run": launch(side_label(state, "observe_render"), argv, paths.side_log("observe_render", f"env{env}"), "IKER loop observation render")}


def write_requery_prompt(state, paths, decision):
    passed = next(attempt for attempt in lp.attempts(paths.stage_dir) if attempt.gate_passed)
    response = (paths.stage_dir / f"attempt_{passed.index:02d}" / "response.md").read_text(encoding="utf-8")
    paths.observe_dir.mkdir(parents=True, exist_ok=True)
    (paths.observe_dir / "prompt.md").write_text(loop_vlm.requery_prompt(layout.TASK_INSTRUCTION, response), encoding="utf-8")
    return {}


def parse_requery(state, paths, decision):
    keypoints = run_files.vlm_keypoints(run_files.read_json(paths.observe_dir / "keypoints.json"))
    result = loop_vlm.requery_result((paths.observe_dir / "response.md").read_text(encoding="utf-8"), keypoints)
    run_files.write_json(paths.observe_dir / "requery.json", {"schema": run_files.SCHEMA_VERSION, **result})
    return result


def run_video(state, paths, decision):
    argv = [PLAY_SCRIPT, "--task", f"{STAGE2_TASK}-play", "--num_envs", str(VIDEO_ENVS), "--checkpoint", decision.params["checkpoint"],
            "--video", "--video_length", str(VIDEO_STEPS), "--headless", *stage2_overrides(state, paths)]
    return {"run": launch(side_label(state, "video"), argv, paths.side_log("video", f"ep{decision.params['epoch']}"), "IKER loop play video")}


def store_video(state, paths, decision):
    source = lp.stage2_video(state, paths)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    target = VIDEO_DIR / f"iker_shoe_c{CONFIG_INDEX:02d}_s1_ep{decision.params['epoch']:04d}_{datetime.now():%m%d_%H%M}.mp4"
    shutil.copyfile(source, target)
    paths.video_note.write_text(f"{target}\n", encoding="utf-8")
    files = [paths.video_note, *(paths.observe_dir / name for name in OBSERVE_FILES)]
    return {"video": str(target), "commit_paths": files, "message": f"iker(loop): 재질의 done, 2단계 영상 {target.name}"}


def kill_stale(state, paths, decision):
    run = decision.params["run"]
    return {"stopped_runs": [run], "pids": stop(state["runs"][run]["label"], force=True)}


def pause(state, paths, decision):
    return {"commit_paths": sorted(path for path in paths.state_dir.rglob("*") if path.is_file())}


EXECUTORS = {
    "run_calibrate": run_calibrate, "commit_calibration": commit_calibration, "launch_b": launch_b, "run_harvest": run_harvest,
    "commit_bank": commit_bank, "write_prompt": write_prompt, "ingest": ingest, "commit_interaction": commit_interaction,
    "run_env_smoke": run_env_smoke, "launch_stage2": launch_stage2, "run_eval": run_eval, "record_eval": record_eval,
    "run_observe_rollout": run_observe_rollout, "run_observe_render": run_observe_render, "write_requery_prompt": write_requery_prompt,
    "parse_requery": parse_requery, "run_video": run_video, "store_video": store_video, "kill_stale": kill_stale, "pause": pause,
}


# ------------------------------------------------------------------- commands


def vlm_request(paths: lp.LoopPaths, decision: ls.Decision) -> dict:
    """Paths and the verbatim brief for the fresh generator agent (§6)."""
    if decision.params["kind"] == "target":
        folder, response = paths.stage_dir, paths.stage_dir / f"attempt_{decision.params['attempt']:02d}" / "response.md"
        images = {prompts.IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    else:
        folder, response = paths.observe_dir, paths.observe_dir / "response.md"
        images = {prompts.IMAGE_MARKER: str(paths.observe_dir / "snapshot.png"), loop_vlm.STAGE_IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    prompt = folder / "prompt.md"
    return {"prompt": str(prompt), "response": str(response), "images": images, "brief": loop_vlm.generator_brief(prompt, response, images)}


def summary_line(state: dict, probe: ls.Probe, decision: ls.Decision) -> str:
    parts = [f"phase {state['phase']}", f"GPU {probe.gpu_used_mib} MiB"]
    if probe.latched:
        end = ls.last_epoch(probe.latched)
        parts.append(f"epoch {end} latched {100.0 * (ls.bin_mean(probe.latched, end, state['policy']['bin_epochs']) or 0.0):.2f} %")
    if probe.checkpoints:
        parts.append(f"last checkpoint ep {max(probe.checkpoints)}")
    if state["eval"]:
        parts.append("eval " + ", ".join(f"ep{k} {100.0 * v['success']:.1f} %" for k, v in sorted(state["eval"].items(), key=lambda kv: int(kv[0]))))
    return " · ".join(parts) + f" -> {decision.action}: {decision.reason}"


def append_history(paths: lp.LoopPaths, record: dict) -> None:
    paths.history_file.parent.mkdir(parents=True, exist_ok=True)
    with paths.history_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def adopt(state: dict, paths: lp.LoopPaths, run: str) -> dict:
    """The launch record of a training run started before the loop, found by its policy label."""
    if run not in ls.TRAINING_RUNS:
        raise SystemExit(f"only training runs can be adopted, not {run!r}")
    label = state["policy"]["labels"][run]
    run_dir, log = paths.task_dir(run) / label, paths.train_log(label)
    if not run_dir.is_dir() or not log.is_file():
        raise SystemExit(f"cannot adopt {run}: {run_dir} or {log} is missing")
    started, pids = run_dir.stat().st_mtime, lp.label_pids(label)
    return {"label": label, "log": str(log), "pid": pids[0] if pids else None, "started_s": started,
            "started": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"), "commit": None, "key": label, "adopted": now_iso()}


def state_dir_of(args) -> Path | None:
    return Path(args.state_dir).resolve() if args.state_dir else None


def load(args) -> tuple[dict, lp.LoopPaths]:
    state = ls.load_state(lp.LoopPaths.of(ROOT, CONFIG_INDEX, ls.TRACK, state_dir_of(args)).state_file)
    return state, lp.LoopPaths.of(ROOT, CONFIG_INDEX, state["track"], state_dir_of(args))


def cmd_init(args) -> int:
    paths = lp.LoopPaths.of(ROOT, CONFIG_INDEX, args.track, state_dir_of(args))
    if paths.state_file.exists():
        raise SystemExit(f"{paths.state_file} exists; the loop continues from it")
    state = ls.new_state(now_iso(), track=args.track, phase=args.phase, policy=json.loads(args.policy))
    state["runs"] = {run: adopt(state, paths, run) for run in args.adopt}
    ls.save_state(paths.state_file, state)
    append_history(paths, {"time": state["updated"], "phase": state["phase"], "action": "init", "adopted": list(args.adopt)})
    print(json.dumps({"state": str(paths.state_file), "phase": state["phase"], "runs": state["runs"]}, ensure_ascii=False, indent=1))
    return 0


def cmd_status(args) -> int:
    state, paths = load(args)
    probe = lp.collect(state, paths, now_s=time.time(), gpu_used_mib=gpu_used_mib(), load_events=load_tfevents)
    decision = ls.decide(state, probe)
    out = {"phase": state["phase"], "status": state["status"], "awaiting": state["awaiting"], "action": decision.action,
           "reason": decision.reason, "params": decision.params, "notes": list(decision.notes), "summary": summary_line(state, probe, decision)}
    if decision.action == "vlm_generate":
        out["vlm"] = vlm_request(paths, decision)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_act(args) -> int:
    state, paths = load(args)
    if args.action in ls.MANUAL_ACTIONS:
        params = {"policy": json.loads(args.policy)} if args.action == "resume" else {}
        decision = ls.Decision(args.action, f"the user asked to {args.action}", params)
    else:
        probe = lp.collect(state, paths, now_s=time.time(), gpu_used_mib=gpu_used_mib(), load_events=load_tfevents)
        decision = ls.decide(state, probe)
        if decision.action != args.action:
            raise SystemExit(f"the loop now decides {decision.action!r} ({decision.reason}), not {args.action!r}")
        if decision.action in ls.SESSION_ACTIONS:
            raise SystemExit(f"{decision.action!r} is carried out by the tick session (LOOP_PROMPT.md), not by act")
    result = EXECUTORS.get(decision.action, lambda s, p, d: {})(state, paths, decision)
    new_state = ls.apply(state, decision, now_iso(), result)
    ls.save_state(paths.state_file, new_state)
    record = {"time": new_state["updated"], "phase": state["phase"], "action": decision.action, "reason": decision.reason,
              "params": decision.params, "result": {k: v for k, v in result.items() if k not in ("commit_paths", "message")}}
    append_history(paths, record)
    if decision.action in COMMIT_ACTIONS and not args.no_commit:
        message = result.get("message") or f"iker(loop): {decision.action} — {decision.reason}"
        record["commit"] = commit([*result.get("commit_paths", []), paths.state_file, paths.history_file], message, args.session_url)
    print(json.dumps({**record, "phase_after": new_state["phase"], "status_after": new_state["status"]}, ensure_ascii=False, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", default="", help="loop directory (default: iker_runs/shoe_place/config_00/loop)")
    parser.add_argument("--no-commit", action="store_true", help="never commit (a mock loop outside the repository)")
    parser.add_argument("--session-url", default=SESSION_URL, help="Claude-Session trailer of the loop's commits")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--track", default=ls.TRACK)
    init.add_argument("--phase", default="stage1_a", choices=ls.PHASES)
    init.add_argument("--policy", default="{}", help="JSON overrides of loop_state.DEFAULT_POLICY")
    init.add_argument("--adopt", nargs="*", default=[], help="training runs already running under their policy labels")
    commands.add_parser("status")
    act = commands.add_parser("act")
    act.add_argument("action")
    act.add_argument("--policy", default="{}", help="resume: JSON policy updates")
    args = parser.parse_args(argv)
    return {"init": cmd_init, "status": cmd_status, "act": cmd_act}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
````

- [ ] **Step 4: 통과 확인** — Step 2 명령, 그리고 전체 순수 테스트. Expected: `3 passed`, 전체 `194 passed, 1 skipped`.

- [ ] **Step 5: 틱 절차서** — 아래 파일을 만든다.

````markdown
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
````

- [ ] **Step 6: 돌고 있는 1단계 런에 대한 status 시험**
  - 스크래치 상태 폴더를 쓰고, 기동·커밋은 없다.
  - `init` 은 A 런의 RUN_LABEL 폴더·로그·PID 를 입양한다.

Run:
```bash
cd ~/rl_ws/hdgp-iker && D=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t8_dry_loop && rm -rf $D && python3 scripts/iker/loop.py --state-dir $D init --adopt stage1_a && python3 scripts/iker/loop.py --state-dir $D status
```
Expected:
- `init` 출력의 `runs.stage1_a` 에 `label iker_grasp_c00_a`, `log .../train_iker_grasp_c00_a.log` 가 있다. `pid` 는 런이 살아 있으면 숫자, 끝났으면 null 이다.
- `status` 는 `"action": "record_gate"`, `"params": {"epoch": 200, "latched": 0.09..., "passed": true}` 와 `summary` 한 줄을 낸다. 9.02 %(epoch 191~200 평균)는 2026-09-14 기록이다.
- `git status --short` 에 새 변화가 없다.

- [ ] **Step 7: 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add scripts/iker/loop.py scripts/iker/LOOP_PROMPT.md source/openarm/openarm/agnostic/modules/iker/tests/test_loop_cli.py
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "feat(iker): 자동 루프 CLI(status·act, RUN_LABEL 자식 기동·PID 종료·단계 산출물 커밋)와 틱 절차서

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 9: 루프 모의 실행과 가동

**Files:**
- Create (가동): `iker_runs/shoe_place/config_00/loop/LOOP_STATE.json`, `iker_runs/shoe_place/config_00/loop/history.jsonl`

**Interfaces:**
- Consumes: Task 8 CLI 전부.
- Produces: 가동된 루프 상태(구성 0), 모의 루프 보고서(`.superpowers/sdd/2026-09-14-iker-auto-loop/task-9-report.md` 의 모의 전이 표).

모의 실행(스펙 §9)은 VLM 응답 대신 사람 기준선 목표를 `get_interaction_data` 코드로 넣는다. 그리고 vlm_target → stage2_train(짧은 학습) → observe_requery 까지 전이를 한 번 따라간다.
- 상태 폴더는 스크래치를 쓰고 `--no-commit` 이다. 트랙은 `iker_shoe_c00_mock` 이고 2단계 라벨은 `iker_vlm_mock` 이다.
- 2단계 뱅크는 접근 자세 뱅크를 쓴다. 신발이 손에 없으니 성공이 나올 수 없어, `observe_requery` 에서 "성공 env 0" 멈춤까지 가는 것이 기대 결과다.

- [ ] **Step 1: 모의 init**

Run:
```bash
cd ~/rl_ws/hdgp-iker && D=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t9_mock_loop && rm -rf $D && python3 scripts/iker/loop.py --state-dir $D --no-commit init --track iker_shoe_c00_mock --phase vlm_target --policy "{\"labels\": {\"stage2\": \"iker_vlm_mock\"}, \"grasp_bank_path\": \"$PWD/iker_runs/shoe_place/config_00/pregrasp_bank.json\", \"stage2_env_smoke\": false, \"stage2_epochs\": 10, \"stage2_num_envs\": 512, \"eval_every\": 10, \"success_target\": 0.0, \"minibatch_size\": 16384}"
```
Expected: `"phase": "vlm_target"`, `"runs": {}`.

- [ ] **Step 2: 틱을 손으로 돌린다** — 아래 한 틱을 반복한다.
  - 명령: `python3 scripts/iker/loop.py --state-dir $D status` 로 action 을 읽고 `python3 scripts/iker/loop.py --state-dir $D --no-commit act <action>` 을 실행한다.
  - `wait` 이면 60~120 s 뒤 다시 본다(백그라운드 대기 뒤 짧은 폴링, 턴을 끝내지 않는다).
  - `vlm_generate` 이면 에이전트를 띄우지 않고, 아래 모의 응답을 `vlm.response` 경로에 쓴 뒤 `status` 로 돌아간다.

```bash
cd ~/rl_ws/hdgp-iker && D=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t9_mock_loop && mkdir -p $D/stage_01/attempt_00 && python3 - "$D/stage_01/attempt_00/response.md" <<'PY'
import json, sys
doc = json.load(open("iker_runs/shoe_place/config_00/interaction_human.json"))
ids = doc["interaction"]["keypoint_ids"]
lines = ["Mock loop response: the human baseline target of configuration 0.", "", "```python", "import numpy as np",
         "def get_interaction_data(keypoint_coordinates):", '    """Mock: place shoe_move at the human baseline target."""',
         "    done = False", "    if done:", "        return", '    object_to_interact = "shoe_move"',
         f"    keypoint_indices_to_interact = {ids}", "    grasp_mode = True"]
lines += [f"    keypoint_coordinates['{k}'] = np.array({p})" for k, p in zip(ids, doc["target_keypoints"])]
lines += ["    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates", "```", ""]
open(sys.argv[1], "w").write("\n".join(lines))
print("wrote", sys.argv[1])
PY
```

  기대 전이:
  1. `write_prompt`
  2. `vlm_generate`(attempt 0) — 위 모의 응답을 쓴다.
  3. `ingest` — `gate.json` 의 `passed true` 를 확인한다.
  4. `commit_interaction` — phase `stage2_train` 이 된다.
  5. `launch_stage2` — GPU ≤ 20000 MiB 를 확인한다. `log/rl_games/open-sens/left/iker-shoe/iker_vlm_mock` 과 `train_iker_vlm_mock.log` 가 생긴다.
  6. 몇 틱 동안 `wait`.
  7. `run_eval`(ep10)
  8. `record_eval`
  9. `advance` — phase `observe_requery` 가 된다.
  10. `run_observe_rollout`
  11. `pause` — reason `no env succeeded in the noise-free rollout of ep 10`. 성공 env 가 생겼다면 `run_observe_render` → `write_requery_prompt` 까지 가고 `vlm_generate` 에서 멈춘다.

  각 `act` 의 JSON 한 줄 요약을 보고서의 전이 표에 적는다.

- [ ] **Step 3: 모의 결과 확인**

Run:
```bash
cd ~/rl_ws/hdgp-iker && D=/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/t9_mock_loop && python3 scripts/iker/loop.py --state-dir $D status && cut -c1-160 $D/history.jsonl && git status --short
```
Expected:
- `status` 가 `"status": "awaiting"` 와 `"action": "wait"` 를 낸다(또는 성공 env 경로라면 `vlm_generate`).
- `history.jsonl` 에 위 전이가 순서대로 있다.
- `git status --short` 에 `iker_runs`·`source`·`scripts` 변화가 없다. 모의는 스크래치와 `log/` 에만 쓴다.
- 모의 2단계 프로세스가 남아 있지 않다: `grep -l "RUN_LABEL=iker_vlm_mock" /proc/[0-9]*/environ 2>/dev/null` 이 빈 출력이다.

- [ ] **Step 4: 실제 루프 가동** — 이 시점의 1단계 상태에 맞춰 `init` 을 고른다.
  - 기본은 컨트롤러가 손으로 보정·B 를 하지 않은 경우다(`LOOP_STATE.json` 이 아직 없어야 한다).

Run:
```bash
cd ~/rl_ws/hdgp-iker && ls iker_runs/shoe_place/config_00/loop/LOOP_STATE.json 2>/dev/null; python3 scripts/iker/loop.py init --adopt stage1_a && python3 scripts/iker/loop.py status
```
Expected: `init` 이 `iker_runs/shoe_place/config_00/loop/LOOP_STATE.json` 을 만들고, `status` 가 `record_gate`(passed true)를 낸다.
  - 컨트롤러가 이미 손으로 보정 파일을 커밋하고 B(`iker_grasp_c00_b`)를 띄웠다면 `init --phase harvest --adopt stage1_b` 를 쓴다.
  - 아무 `act` 도 하지 않는다. 첫 틱은 컨트롤러의 크론이 한다.

- [ ] **Step 5: 가동 상태 커밋** (따로 Bash 호출)

```bash
cd ~/rl_ws/hdgp-iker && git add iker_runs/shoe_place/config_00/loop/LOOP_STATE.json iker_runs/shoe_place/config_00/loop/history.jsonl
```

```bash
cd ~/rl_ws/hdgp-iker && git commit -m "chore(iker): 구성 0 자동 루프 가동 — 1단계 런 입양, 첫 판정은 크론 틱

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

- [ ] **Step 6: (컨트롤러) 크론 등록과 첫 틱** — 이 단계는 구현자가 아니라 컨트롤러 세션이 한다.
  - CronCreate `7,37 * * * *` 에 프롬프트 "IKER 자동 루프 틱: ~/rl_ws/hdgp-iker/scripts/iker/LOOP_PROMPT.md 절차대로 한 틱을 수행한다 (track iker_shoe_c00)" 를 등록한다.
  - 곧바로 `LOOP_PROMPT.md` 절차로 첫 틱을 한 번 돌린다.
  - 사용자에게 한두 줄로 알린다: 가동 시각, 첫 동작, 다음 틱 시각.
