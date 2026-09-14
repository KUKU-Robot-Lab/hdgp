# grasp_fj_t2r — Track B + text2reward 생성 보상 (인벨롭 파지)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 베이스: [`../grasp_fj`](../grasp_fj/CLAUDE.md) · 붓기 t2r 트랙과 **코드 공유 없음**(09.10 fj 공유 금지)

## 목적
DG-5F full-joint 손(액션 26 = 팔 7 증분 + 손 19 절대)으로 **인벨롭 파지 → 들기 → 유지**. 보상은 사람이 쓰지 않고
text2reward 방식으로 생성한다: 환경 설명(RewardContext 소스)+과제 문장 → LLM → `compute_reward(ctx)`.

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

서버 기동 — i1/i2 와 같은 런처 `run_fj.sh` · 12,288 env · SAPG 2048(6블록) · seed 42 · FRESH · GPU0:
```bash
ssh server "cd ~/rl_ws/hdgp && TASK=open-short_r_grasp_fj_t2r-lstm-sapg RUN=fj_t2r_i00 GPU=0 ENVS=12288 BLK=2048 SERVER=1 \
  EXTRA='agent.params.config.expl_coef_block_size=2048 env.reward_code_path=/home/oem/rl_ws/hdgp/reward_gen/grasp_fj_envelope/iter_00/compute_reward.py' \
  nohup bash ./run_fj.sh > ~/rl_ws/our_source/fj_t2r_runs/fj_t2r_i00.out 2>&1 < /dev/null & echo pid=\$!"
```
★`SERVER=1` 필수 — 서버에도 `../IsaacLab` 이 있어 없으면 conda 대신 isaaclab.sh 로 간다. 종료는 PID 로만(RUN_LABEL·CUDA 대조).
부팅 확인: `[grasp_fj_t2r] 보상 = <경로> · … 필터 ['/World/envs/env_.*/Object']` 줄 · 로그에 `did not match` 0건.

## 판정 지표
`contact/fingers_touching`(0~5) · `contact/finger_<손가락>` · `contact/palm_touching` · `task/grasp_q_at_success`(B 계측 유지) ·
`ctrl/prev_ep_successes_mean` · `ctrl/drop_sticky_frac` · `task/tilt_deg` · `reward/<생성 항>`. 지표만으로 종료하지 않는다 — 영상 확인(붓기 트랙 09.13 교훈).
