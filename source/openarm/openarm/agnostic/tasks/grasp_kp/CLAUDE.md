# grasp_kp — 태스크 규칙 (Track A: fabric palm 6D + 시너지 15D · 접촉 센서 0)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 설계: `DESIGN.md`(§8 = API 계약) · 보상 감사: `REWARD_AUDIT.md` · 기준선 `grasp_s2r` 은 **불변**

## 정체성
SimToolReal 식 **목표열 + progress-only 보상 8항**. 축대칭 키포인트 4개(yaw 불변) `d(o,g)=max‖kp‖` 로
성공(d ≤ tol, 누적 10 step) → 다음 목표(이전 목표 ±0.08, 박스 클램프) → `max_goals` 50 truncation.
접촉 센서는 만들지도 읽지도 않는다. 손 홀드는 `blocked`(지령↔실측 정체)뿐. 팔은 grasp_s2r 그대로(fabric).

## 핵심 계약 (`tests/test_grasp_kp_contract.py` 가 잠근다)
| 계약 | 왜 |
|---|---|
| obs 129 / state 153 = `_derive_spaces` 공식 = `_get_observations` cat 순서(물체 quat 없음) | 첫 호출 shape 가드가 두 숫자를 들고 죽는다 |
| 목표 박스 ⊂ 팔 지령 범위(앵커 ±(0.10,0.10,0.35) ∩ 클램프 박스, xy 반폭 0.08) — `_assert_goal_box_in_arm_reach` | 넘으면 목표열·tol 커리큘럼이 조용히 멈춘다 |
| play 는 `tol_eval = tol_floor` 고정(커리큘럼 갱신 없음) | 커리큘럼 상태는 체크포인트에 없다 |
| `_latched` = 높이 래치(dz > 0.10, 리셋에서만 해제) | 닫기 게이트·앵커·species 로깅이 그대로 쓴다 |
| 지연 큐 3개(obs 3 / action 3 / object 10), flush = `episode_length_buf == 0` | 리셋 직후 옛 값 누출 방지 |
| 외란은 lifted 에서만, `set_external_force_and_torque(is_global=True)` 매 스텝 | decay 0 = 매 스텝 재추첨 |
| palm 박스 z 하한 0.27 (**올리기만**) | a=0 에서 손이 상판 49 mm 관통(09.06) |
| `value_bootstrap: True` · `gamma 0.99` | time-out·max_goals truncation 이 잦다(rl_games 래퍼 `time_outs`) |
| respawn OFF · ADR OFF · 접촉 동결 OFF | 부팅 가드(`_assert_kp_contract`)가 CLI 오버라이드를 거부한다 |

## 지표 (TFEvents)
`stage/{lifted,goal1,goal2,goal3}` 단조 · `task/lifted_frac`·`task/just_lifted` · `task/kp_dist`·`task/near_goal`·
`task/successes_mean` · `task/tol`(0.06→0.015 계단) · `task/hand_floor_depth_max ≈ 0`·`done/hand_floor` ·
`fabric/joint_err_max < 0.1` · `task/abnormal_rate ≈ 0` · `reward/*` 8항. 리프트 전 총보상 ≈ 1.0/step 이
장기 유지되면 REWARD_AUDIT Check 1 의 국소최적이다(`stage/lifted` 500 epoch 무상승 → 재검토).

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/tests \
  source/openarm/openarm/agnostic/tasks/grasp_kp/tests -q                       # Isaac 불필요
# 로컬 스모크(5090): 16 env × horizon 16 = batch 256 → minibatch 256
./train.sh open-sens_r_grasp_kp-lstm kp_smoke --num_envs 16 --max_iterations 5 \
  agent.params.config.minibatch_size=256 agent.params.config.central_value_config.minibatch_size=256
# server gpu0 (conda proj-hdgp-py311, branch pour): 4096 env
CUDA_VISIBLE_DEVICES=0 NOTE="simtoolreal A" ./train.sh open-sens_r_grasp_kp-lstm kp_a1 --num_envs 4096 --headless
```
로그 `log/rl_games/open-sens/right/grasp-kp/<label>/`. LSTM 은 num_envs 1024 의 배수(minibatch 16384). 스모크 게이트:
`ep_len` 상승 · abnormal ≈ 0 · `joint_err < 0.1` · `hand_floor_depth_max ≈ 0` · `stage/lifted` 상승.
