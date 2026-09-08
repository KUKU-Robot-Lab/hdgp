# grasp_fj — 태스크 규칙 (Track B: 팔 7D 증분+EMA + 손 20관절 독립 · Fabrics 없음 · 접촉 센서 0)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 설계: `../grasp_kp/DESIGN.md` §1 B 열 · 형제 A = `../grasp_kp`(목표열·보상·관측 전부 공유) · 기준선 `grasp_s2r` 은 **불변**

## 목적
**sim2sim 정합 + fabric 노드 제거.** 정책이 팔 관절 위치 목표 `q*` 를 직접 낸다(A 는 palm 6D → fabric → q*).
실기 배포가 4노드(fabric 포함)에서 **3노드**(정책 → pd → 드라이버)가 된다. 실기 JTC 는 velocity 를 쓰지 않으므로
sim 도 팔에 **위치 목표만** 준다.

★**09.08 SimToolReal 과제 의미 정합**(사용자 지시: "팔매핑·팔실효 slew·손속도 제한 동일 / 성공목표에피소드
동일성유지 / **목표델타는 제외** / 외란은 줄일 필요가 있음"). 이 판부터 손도 A 와 갈린다 — 손 20관절을
**독립**으로 지령하고(`hand_direct`), 보상 모듈도 포크했다(`fj_reward.py`). A/B 대조의 변수는 이제 팔+손이다.

## 핵심 계약 (`tests/test_grasp_fj_contract.py` 가 잠근다)
| 계약 | 왜 |
|---|---|
| `q*_t = clamp(q*_{t-1} + k_arm·a)` → EMA α → clamp, **`k_arm 0.025`·`α 0.1` → 실효 slew 0.15 rad/s** | `k_arm ≡ dof_speed_scale × 정책_dt` = 1.5 × 1/60. 정책 dt 가 양쪽 다 1/60 이라 성립 — cfg 가 **환산값 1.5 자체**를 대조한다 |
| 리셋 시 `q*_{-1}` = 홈 q(`_default_q`) | 홈 텔레포트라 지령 = 실측에서 출발 |
| **손 20관절 독립**(`hand_direct=True`, leaf) — 폐쇄도 [0,1] 를 관절별로 지령 | 시너지의 실효 지령은 15 가 아니라 **4개**였다(couple_four_fingers + `_1` 무효). 옆에서 접근하면 손 모양을 못 바꿔 파지가 실패한다 |
| 손 폐쇄 = 램프 `synergy_close_speed` **또는** EMA `synergy_close_ema`, 둘 동시 금지 | 동시에 걸리면 실효 속도를 어떤 지표로도 못 가른다. 폐쇄 속도는 fj_c1/c2/c3 **스윕**으로 정한다 |
| `close_gate`·`blocked` hold 는 EMA 를 켜도 **산다** | 둘 다 증분에 걸리고 EMA 도 증분에 곱하는 양수 스칼라라 순서가 교환된다. `blocked` 는 접촉 항 0개인 이 보상에서 감쌈을 만드는 **유일한** 장치 |
| obs `cmd_state` = `q*_{t-1}`(7) → **actor 136 / critic 160, action 27** | `hand_direct` 가 손 폭을 20 으로. fj_b9(22/131/155)와 **호환 안 됨** — FRESH |
| 성공 = **연속** 10회 · tol 0.1125→0.015 · 커리큘럼 게이트 3.0 · 스텝 예산 **목표당** | SimToolReal `env.py:2437`·`utils.py:238`. 공차·연속·`goal_first_z_range` 는 **3중 잠금쌍** |
| 보상 = `fj_reward.py`(**포크**), `goal_bonus` 만 A 와 다름(성공 순간 1회 전액) | A 와 제어 방식이 다르다. `_get_rewards` 는 여전히 안 덮는다 — A 의 이음매 `_progress_reward` 만 덮는다 |
| 외란 2.7 N/kg · 0.27 N·m/kg (A 의 1/7.5) | Kuka 어깨 300 N·m vs OpenArm 40 N·m. **의도적 divergence** — 원본과 같은 20/2.0 은 팔이 못 버틴다 |
| **목표 델타는 A 값(0.08 m / 0°) 유지** | 사용자가 명시적으로 제외했다. leaf 가 이 둘을 덮으면 지시 위반 |
| fabric 런타임 0(`self.fabric = None`), 부모 버퍼(`fabric_q`·`palm_targets`·`_palm_lo`…)는 모양만 유지 | 부모 리셋·앵커·박스 부트스트랩이 읽는다 |
| A 를 덮는 훅은 팔·손 어댑터 + 이음매뿐 (`_get_rewards`·`_get_dones`·`_hand_command` 금지) | 보상·종료 본체를 덮으면 A/B 가 같은 과제가 아니다 |

### 배치 방화벽 — 과제 의미 값은 **leaf 에만**
형제 `grasp_fj_rh`(RH56F1, 진행 중)가 `GraspFJEnvCfg` 를 **상속**하고, 등록부는 leaf
(`GraspFJTesolloRightEnvCfg`)만 인스턴스화한다. 그래서 공차·성공 술어·예산·외란·손 값은 전부
leaf 에 둔다 — base 에 두면 남의 트랙이 조용히 바뀐다. `k_arm`/`arm_slew_rad_s` 만 base 인 이유는
fj_rh 가 이미 동일 값으로 고정·assert 하고 있어 no-op 이기 때문이다.
`test_task_semantics_live_on_the_leaf_so_grasp_fj_rh_is_not_captured` 가 기계 검사한다.

## 성공 기준
1. **정합**: 같은 체크포인트·같은 액션열을 sim(이 env)과 실기 pd 노드에 넣었을 때 관절 궤적 오차가 A 경로보다 작다.
2. `ctrl/joint_err_max` < 0.1 rad(목표↔실측; 실기 err ≈ (kd/kp)·q̇ 와 같은 양) · `ctrl/arm_limit_sat` ≈ 0.
3. `task/hand_floor_depth_max` ≈ 0 · `done/hand_floor` ≈ 0(테이블 방어는 fabric 이 아니라 종료·벌점뿐이다).
4. 3노드 배포에서 fabric 노드 없이 정책 출력이 pd 노드 계약(`policy_control` v2)에 바로 실린다.

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/tests \
  source/openarm/openarm/agnostic/tasks/grasp_kp/tests source/openarm/openarm/agnostic/tasks/grasp_fj/tests -q
./train.sh open-sens_r_grasp_fj-lstm fj_smoke --num_envs 16 --max_iterations 5 \
  agent.params.config.minibatch_size=256 agent.params.config.central_value_config.minibatch_size=256   # 로컬 스모크
# ★09.08 손 폐쇄속도 3팔 스윕 — 손 폐쇄법만 다르고 나머지는 전부 같다
CUDA_VISIBLE_DEVICES=0 ./train.sh open-sens_r_grasp_fj-lstm-sapg fj_c1 --num_envs 4096 --headless
CUDA_VISIBLE_DEVICES=0 ./train.sh ... fj_c2 env.synergy_close_speed=0.020
CUDA_VISIBLE_DEVICES=1 ./train.sh ... fj_c3 env.synergy_close_speed=1.0 env.synergy_close_ema=0.1
```
로그 `log/rl_games/open-sens/right/grasp-fj/<label>/`. 스모크 게이트: `ctrl/arm_target_step` ≤ 0.0025 rad
(×60 = 0.15 rad/s) · `ctrl/hand_blocked_frac` ≈ 0(물체가 멀 때) · abnormal ≈ 0 · `ctrl/joint_err_max` < 0.1 ·
`hand_floor_depth_max` ≈ 0 · `reset/arm_q_dev_max` 가 b9 대역(목표당 시계가 리셋 진단을 안 깼다는 증거).

⚠ **비교는 matched-tol 로 한다, matched-epoch 금지.** tol 0.1125 + 게이트 3.0 이면 0.015 까지
약 19 커리큘럼 스텝이라 b9(0.06 출발)보다 ~1,300 에폭 더 느슨한 공차에 머문다 — 에폭 맞춤 비교는
새 런을 과대평가한다.

**왜 스윕인가**: 08.25 시너지 스윕은 단조였다(0.050→감쌈 0.45 … 0.005→0.64, 0.002→0.80) —
빠를수록 컵을 쳐냈다. 그런데 그 표는 **결합된 시너지(실효 4자유도)** 에서 잰 값이라 20관절 독립에
그대로 옮겨간다는 보장이 없다. c1(0.005 현재값)·c2(0.020)·c3(EMA 0.1 = SimToolReal)로 재측정한다.
3팔은 동시에 PhysX 비결정성(실패율 ~1/3)에 대한 **3중 복제**다 — 셋이 같은 방식으로 죽으면
그건 손이 아니라 팔 속도(`k_arm` 6.7배 감속)가 원인이고, 그때는 `env.k_arm=0.167
env.arm_slew_rad_s=1.0` 대조군을 돌린다(둘 다 안 주면 검증기가 부팅을 죽인다 = 가드 정상).

지표 나머지는 A CLAUDE.md 와 같다(`fabric/*` 만 `ctrl/*` 로 바뀐다).
