# grasp_fj_t2r — Track B + text2reward 생성 보상 (인벨롭 파지)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 베이스: [`../grasp_fj`](../grasp_fj/CLAUDE.md) · 붓기 t2r 트랙과 **코드 공유 없음**(09.10 fj 공유 금지)

## 목적
DG-5F full-joint 손(액션 26 = 팔 7 증분 + 손 19 절대)으로 **인벨롭 파지 → 들기 → 유지**. 보상은 사람이 쓰지 않고
text2reward 방식으로 생성한다: 환경 설명(RewardContext 소스)+과제 문장 → LLM → `compute_reward(ctx)`.

## 트랙 (09.14)
| 트랙 (`reward_gen/<트랙>`) | gym id | 시작 | 컵 | 팔 | 에피소드 |
|---|---|---|---|---|---|
| `grasp_fj_envelope` (정지 — iter_01 미기동) | `open-short_r_grasp_fj_t2r-lstm-sapg` | B 홈 · 손바닥↔컵 0.16 m | shaker_sweep 반경 29–44 mm | 0.15 rad/s | 10 s |
| `grasp_fj_reach` (최종 목표) | `open-short_r_grasp_fj_t2r_reach-lstm-sapg` | 테이블 앞 가장자리 밖 · 0.38 m (cfg `arm_reset_joint_pos_override`) | cup_family 반경 44–81 mm | 0.3 rad/s | 15 s |

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
