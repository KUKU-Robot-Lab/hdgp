# grasp_fj_t2r — Track B + text2reward 생성 보상 (인벨롭 파지)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 베이스: [`../grasp_fj`](../grasp_fj/CLAUDE.md) · 붓기 t2r 트랙과 **코드 공유 없음**(09.10 fj 공유 금지)
> 루프 작업 **보고 방식**: [`reward_gen/grasp_fj_stage/CLAUDE.md`](../../../../../../reward_gen/grasp_fj_stage/CLAUDE.md) (09.17 — 경과 중계 금지 · 최종 1회 요약 · 틱 형식은 예외)

## 목적
DG-5F full-joint 손(액션 26 = 팔 7 증분 + 손 19 절대)으로 **인벨롭 파지 → 들기 → 유지**. 보상은 사람이 쓰지 않고
text2reward 방식으로 생성한다: 환경 설명(RewardContext 소스)+과제 문장 → LLM → `compute_reward(ctx)`.

## 트랙 (09.14)
| 트랙 (`reward_gen/<트랙>`) | gym id | 시작 | 컵 | 팔 | 에피소드 |
|---|---|---|---|---|---|
| `grasp_fj_envelope` (정지 — iter_01 미기동) | `open-short_r_grasp_fj_t2r-lstm-sapg` | B 홈 · 손바닥↔컵 0.16 m | shaker_sweep 반경 29–44 mm | 0.15 rad/s | 10 s |
| `grasp_fj_reach` (최종 목표) | 루프 `open-short_r_grasp_fj_t2r_reach-lstm`(PPO-LSTM 4096) · 최종 정책 `-lstm-sapg`(12,288) | 테이블 앞 가장자리 밖 · 0.38 m (cfg `arm_reset_joint_pos_override`) | cup_family 반경 44–81 mm | 0.3 rad/s | 15 s |

★09.20 좌팔판 `grasp_fj_rand_left`(gym `open-short_l_grasp_fj_t2r_rand-lstm`, cfg `GraspFJT2RRandLeftEnvCfg`, 프로필 `tesollo_left_short_tl`)
= 우 `grasp_fj_rand` 의 y 반전 미러. env 는 `cfg.hand_side` 로 법선 방향(−y)·손 정규화 방향(음의 각으로 조이는 관절 1−x)·손바닥 bbox 를 고르고,
프롬프트는 변종 `rand_left`(관절표·ctx 스텁 문구 좌손판). 계약 `tests/test_left_mirror.py`.

★09.14 사용자 "보상 구조가 확실하지 않은데 SAPG·env 수를 너무 늘린 게 아닌지" → 보상 설계 루프는 PPO-LSTM 4096 env
(`grasp_fj/config/agents/rl_games_ppo_lstm_cfg.yaml` — SimToolReal 값 · horizon 16 · 미니배치 16,384), 파지·리프트가 되는 보상이
나온 뒤 최종 정책만 SAPG 12,288. reach i00 은 SAPG 12,288 로 돌았다(부팅 ~15분 · e571 에서 라운드 끝). i00↔i01 은 알고리즘도 달라 1:1 비교 금지.

프롬프트의 환경 사실은 `t2r/prompts.py` `VARIANTS`(`render --variant`, meta 로 다음 iter 에 승계) — reach 값이 등록 cfg 와 어긋나면 테스트가 막는다.
루프 도구는 전부 `--track`. ★관측·actor 는 sim2real 가능한 구조로만 바꾼다(사용자 09.14) — 컵 종류·반경·접촉은 obs 에 없다.
★reach 팔 0.3 rad/s 는 실기 reduced 상한 0.25 를 넘는다(full 2.0 · bridge 기본 0.5) — 배포 전 확인.

## 계약 (`tests/` 가 잠근다)
| 계약 | 왜 |
|---|---|
| env 가 덮는 훅은 `__init__`·`_setup_scene`·`_progress_reward`·`_build_context`·`_link_cup_forces`·`_log_fabric_metrics`·`_reset_idx` 뿐 | 관측·액션·종료·성공 판정·공차 커리큘럼이 B 와 같아야 "보상만 바뀐 같은 과제"다 |
| `_progress_reward` 는 B 모듈을 먼저 돌리고(`out` = lifted 래치·추적기) 총보상·항만 생성 코드 결과로 바꾼다 | 에피소드 상태가 생성 코드에 좌우되면 판정이 흔들린다 |
| 보상 전용 컵 접촉 센서: 마디(`_3`·`_4`·tip) body 하나당 센서 하나 + 손바닥 | 다중 body 를 한 센서에 묶으면 `force_matrix_w` 가 조용히 0. 관측에는 넣지 않는다(09.14 사용자 결정) |
| 컵 필터는 env_0 스테이지의 RigidBodyAPI 프림으로 만든다(`_cup_contact_filter`) — cfg `object_contact_filter` 금지 | ★09.14 스모크: cfg 값이 shaker_sweep 에서 `Object/ShakerFDM5mm`(없는 경로, 0개 매칭)라 PhysX 에러 로그만 남고 힘이 전부 0 이었다. 셰이커 USD 루트가 참조로 `Object` 자체가 된다 |
| `ctx.palm_pos` = 손바닥 중심 **palm_ee**(collision 없는 가상 점) = 손바닥 링크 원점 + R·URDF `r_hj_palm_ee` 오프셋 (28, 0, 40) mm — `palm_frame.py`, rpy ≠ 0 이면 부팅 거부 | ★09.14 사용자 "손바닥의 중심쪽은 palm_ee xform · collision 없는 가상의 점". `palm_idx`(프로필 `r_hl_palm`)는 손목 쪽이라 reach i00 보상이 손목 끝을 컵에 붙였다. 접촉 센서는 collider 가 있는 `r_hl_palm` 그대로(손등·손목 접촉도 센다 — 방향은 보상이 `palm_normal` 로 가려야 한다) |
| 보상 게이트 래치 `ctx.approach_done`·`envelope_done`·`hand_default_q_norm`(`grasp_gates.py`) — `_build_context` 가 매 스텝 갱신 · `_reset_idx` 가 에피소드 EMA(`stage/{approach,envelope}_gate_ep`) 후 리셋 | ★09.15 사용자 3단계 "기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트". 보상 함수는 매 스텝 상태만 봐 순서(과거)를 모른다 — reach 6라운드 동안 단계를 건너뛴 자세가 점수를 받았다. ★09.15 23:2x **접근 = C자 사전파지**(기본 자세 엄지가 손바닥면 118 mm 앞으로 뻗어 "손바닥 중심 2 cm"는 엄지로 컵을 눌러야 섰다): 손바닥면↔컵 옆면(법선 방향) −1~+2 cm(하한 = 컵이 손등 뒤면 안 선다, 서버 스모크에서 발견) · 컵 축이 palm_ee 보다 손가락 방향 R−5~R+20 mm 앞 · palm_ee 가 띠 높이 안 · 시작 방향(법선 +y · 손가락 `ctx.palm_finger_dir` +x, cos ≥ 0.7 — 리셋 FK 대조) · 움직이는 손 관절 기본 자세 ±0.3(09.16 완화 — 기본 자세 = 액션 하한 · 탐색 σ 1 이라 0.15 는 통과 0.8 %) · 손가락·엄지 무접촉(조건별 통과율 `stage/approach_ok_{gap,along,height,orient,pose,no_touch}_now`), 인벨롭 = 접근 뒤 손바닥+엄지+닿은 손가락 ≥ 4 가 5 스텝 연속. 로그 퍼널(관찰용, 접근 ≤ 5 cm)과 따로 둔다. 트랙 `grasp_fj_stage`(새 이력) · 판정 `stop(checkpoint:<단계>)`. ★09.15 사용자 "컵에 다가가는 palm_ee_x · 손가락 방향(palm_ee_z)": i00 보상의 손바닥 방향 항이 시작부터 손을 ~68° 돌리게 해(e1→e17 방향 항 0.19→0.36 · 간극 0.255→0.278 m) 손 방향 조건을 넣었다 |
| 시작 상태 커리큘럼 `near_start_*`(reach leaf: 50 % · 공통 스텝 > 4 · 컵 8종 IK 표) — `_reset_idx` 가 B 리셋 **뒤** 팔 관절 상태·q*·이전 q* 를 함께 덮는다 · 출발 그룹별 래치·퍼널 EMA `stage/{far,near}_*` · `stage/near_start_frac_now` | ★09.15 사용자 "T2R 제대로 적용하면서 진행되는건지?" → 0.38 m 먼 출발만으로는 매 라운드 접근만 고쳐 파지·리프트 피드백이 쌓이지 않았다. 자세 = C자 완료 조금 앞(손바닥면 4.5 cm · 컵 축 R+2.5 cm · 띠 0.8 H · 시작 방향 — 컵 스폰 ±2 cm 에도 겹침·공짜 래치 없음, FK 대조 테스트; 3 cm 판은 스모크에서 32 중 4 env 가 첫 스텝에 래치). 첫 스텝 부팅 시작 거리 가드(common_step_counter ≤ 4)는 먼 출발만 본다. ★09.16 사용자 "가까운 출발 = 접근 완료로 시작" — `_reset_idx` 가 가까운 출발에 접근 래치를 세운다(2단계부터). 창 밖 4.5 cm 판을 1단계로 두자 컵이 손 앞인데도 닫기 보상이 켜지지 않았다(i02 e240 래치 0). **이 변경 뒤 env·래치·과제 문장은 라운드 사이에 바꾸지 않는다.** 스모크 `probes/probe_grasp_fj_t2r_near_start.py` |
| `reward_code_path` 기본 "" = 영 보상 · env `__init__` 에서 읽는다 | 부팅 스모크용. hydra 가 `__post_init__` 에 구워지는 함정 없음 |
| t2r 포크(`t2r/`): 컨텍스트 소스 = 프롬프트 환경 설명 · 검증기(정적+드라이런) · 프롬프트는 **환경 사실만** | 생성기에 이 세션의 설계 의견을 흘리지 않는다(09.14 사용자 결정: 새 에이전트 · 지난 결과 미포함) |

## 절차
```bash
cd ~/rl_ws/hdgp
python3 scripts/reward_gen/t2r_fj.py render --track grasp_fj_envelope --task-file scripts/reward_gen/tasks/grasp_fj_envelope.txt
# → 새 에이전트(prompt.md 경로만)가 reward_gen/grasp_fj_envelope/iter_00/response.md 를 쓴다
# ingest 는 Isaac python 으로 — cuda 드라이런까지 돈다(시스템 python torch 는 5090 커널이 없어 cpu 만, 경고로 표시)
../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/grasp_fj_envelope/iter_00
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
# ★ingest PASS 는 가짜 ctx 드라이런일 뿐이다 — 실제 장치·실제 ctx 로 한 번 굴린다(배선 wire + 무작위 random)
R=$PWD/reward_gen/grasp_fj_envelope/iter_00/compute_reward.py
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_boot.py --mode wire --reward_code_path $R
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_boot.py --mode random --num_envs 64 --steps 300 --reward_code_path $R
```
프로브 GATE: NaN 0 · `reward/total`·`contact/*` 존재 · 리셋 직후 prev_actions 0 · 프롬프트 손 관절표 = 부팅 실측(±2e-3) ·
(wire) 검~새끼 자기 행만 섬 + 전부 묶음에서 엄지 행. ★엄지 행은 한 손가락 묶음에서도 선다(굽힌 손가락이 컵을 엄지에 민다).
★reach 트랙에선 wire GATE FAIL 이 정상이다 — wire 는 컵을 수평으로만 옮기는데 시작 손가락이 컵보다 0.16 m 위(테이블 가장자리 밖)라
기하상 대부분 안 닿는다. 판정은 "닿은 행이 자기 행뿐인가 · 힘이 0 이 아닌가"(09.14: 약지 3.9 N·새끼 0.16 N 자기 행만 → `Object/baseLink` 필터 매칭).
gym id `open-short_r_grasp_fj_t2r-lstm-sapg`, 보상은 `env.reward_code_path=<절대경로>`. 학습 기동은 사용자 승인 후.
t2r 트랙이라 reward-audit 은 쓰지 않는다(사용자 09.14) — 검증은 위 검증기 + 실제 장치 스모크.

서버 기동 — i1/i2 와 같은 런처 `scripts/experiments/run_fj.sh` · 12,288 env · SAPG 2048(6블록) · seed 42 · FRESH · GPU0:
```bash
ssh server "cd ~/rl_ws/hdgp && TASK=open-short_r_grasp_fj_t2r-lstm-sapg RUN=fj_t2r_i00 GPU=0 ENVS=12288 BLK=2048 SERVER=1 \
  EXTRA='agent.params.config.expl_coef_block_size=2048 env.reward_code_path=/home/oem/rl_ws/hdgp/reward_gen/grasp_fj_envelope/iter_00/compute_reward.py' \
  nohup bash scripts/experiments/run_fj.sh > ~/rl_ws/our_source/fj_t2r_runs/fj_t2r_i00.out 2>&1 < /dev/null & echo pid=\$!"
```
★`SERVER=1` 필수 — 서버에도 `../IsaacLab` 이 있어 없으면 conda 대신 isaaclab.sh 로 간다. 종료는 PID 로만(RUN_LABEL·CUDA 대조).
부팅 확인: `[grasp_fj_t2r] 보상 = <경로> · … 필터 ['/World/envs/env_.*/Object']` 줄 · 로그에 `did not match` 0건.

## 판정 지표
`contact/fingers_touching`(0~5) · `contact/finger_<손가락>` · `contact/palm_touching` · `task/grasp_q_at_success`(B 계측 유지) ·
`ctrl/prev_ep_successes_mean` · `ctrl/drop_sticky_frac` · `task/tilt_deg` · `reward/<생성 항>`. 지표만으로 종료하지 않는다 — 영상 확인(붓기 트랙 09.13 교훈).
★성공 **순간** 접촉(이벤트 EMA, −1 = 아직 성공 없음): `contact/{fingers,links,palm}_touching_at_success` · `contact/finger_<손가락>_at_success`.
★에피소드 퍼널(`stage_funnel.py`, 09.14 사용자 "접근·파지·리프트가 잘 되는지 틱 확인" — 로그 전용, 보상·관측 아님):
`stage/{reach,grasp,envelope,lift,success}_ep` = 끝난 에피소드 중 그 단계에 한 번이라도 닿은 비율(이벤트 EMA, −1 = 아직 없음) ·
`stage/palm_cup_gap` = 손바닥 원점 → 컵 파지 띠 거리(스텝 평균). 접근 = 간극 ≤ 5 cm · 파지 = 닿은 손가락 ≥ 3 · 인벨롭 = 손바닥 + 5손가락 동시.
단계끼리 포함 관계를 강제하지 않는다(파지 없이 밀어 올려도 lift 가 선다).

## t2r 루프 (09.14 사용자: "루프 틱을 검사하면서 보상함수 설계가 제대로 되고 있는지 피드백 구조")
- 틱 절차 `scripts/reward_gen/LOOP_PROMPT_fj.md` · 상태 `reward_gen/grasp_fj_envelope/LOOP_STATE.json` · cron(세션 한정).
- `scripts/reward_gen/t2r_fj_round.py status|advance|launch` — 판정 수치는 `ROUND_POLICY` 한 곳(라운드 1000 epoch/4h ·
  유지 = 성공 ≥2.0 **또는 최근 200 epoch 안에 공차가 조여짐** · 종료 후보 성공 ≥4.0 + tol ≤0.03 + 성공 순간 손가락 ≥4 · 손바닥 ≥0.5
  → 2틱 연속이면 **영상 게이트**). ★공차 커리큘럼(3000 프레임≈188 epoch 마다 성공 ≥2.0 이면 ×0.9)이 성공 수를 게이트 2.0 쪽으로
  끌어내린다(i00: 3.46→2.71) — 성공만으로 판정하면 개선 중인 런을 죽인다(09.14 틱에서 발견·수정).
- ★피드백 = **원본 text2reward interactive**(09.14 사용자: "T2R 방식이 제대로 적용된게 맞는지?" → 결정 "Claude 초안 → 사용자 승인"):
  라운드 끝에 `t2r_fj_round.py video` 로 영상 → Claude 가 `iter_NN/observation.md`("I can see from the robot that") ·
  `improvement.md`("feedback for improvement") 초안 → **사용자 승인** → `advance` → `t2r_fj.py reflect` 가 `history.jsonl` 의
  전 이력(코드·관찰·피드백)을 원본 템플릿으로 넣고 "Re-imagine which steps is missed or wrong". 지표 표(`reward/*`·`contact/*`·과제)는
  참고로만 붙고 B 설계 계측(`task/grasp_q*`)은 안 준다. ★처음 만든 루프는 이 사람 관찰 단계를 지표 임계값으로 바꿔 놓았었다(Eureka 식).
- 붓기 루프(`LOOP_PROMPT.md`·`t2r_round.py`·`reward_gen/pour_bi`·GPU1)와 파일·GPU·cron 을 공유하지 않는다.
