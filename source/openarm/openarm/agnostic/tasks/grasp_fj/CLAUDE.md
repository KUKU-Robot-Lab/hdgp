# grasp_fj — 태스크 규칙 (Track B: 팔 관절 7D 증분+EMA + 시너지 15D · Fabrics 없음 · 접촉 센서 0)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 설계: `../grasp_kp/DESIGN.md` §1 B 열 · 형제 A = `../grasp_kp`(목표열·보상·관측 전부 공유) · 기준선 `grasp_s2r` 은 **불변**

## 목적
**sim2sim 정합 + fabric 노드 제거.** 정책이 팔 관절 위치 목표 `q*` 를 직접 낸다(A 는 palm 6D → fabric → q*).
실기 배포가 4노드(fabric 포함)에서 **3노드**(정책 → pd → 드라이버)가 된다. 실기 JTC 는 velocity 를 쓰지 않으므로
sim 도 팔에 **위치 목표만** 준다(`set_joint_velocity_target` 은 손에만). 손 경로는 A 와 동일 — A/B 대조의 변수는 팔뿐.

## 핵심 계약 (`tests/test_grasp_fj_contract.py` 가 잠근다)
| 계약 | 왜 |
|---|---|
| `q*_t = clamp(q*_{t-1} + k_arm·a)` → EMA α → clamp, `k_arm 0.167`·`α 0.1` → 실효 포화 slew α·k/dt = **1.0 rad/s**(`arm_slew_rad_s`, 브리지 상한) | DESIGN §1 B. EMA 가 누적 목표에 걸려 스텝당 변화 = α·k·a — cfg 가 선언값과 대조한다 |
| 리셋 시 `q*_{-1}` = 홈 q(`_default_q`) | 홈 텔레포트라 지령 = 실측에서 출발 |
| obs `cmd_state` = `q*_{t-1}`(7) → actor 131 / critic 155, action 22 | A 의 `_derive_spaces` 공식 + `_arm_action_dim` 훅 하나 |
| fabric 런타임 0(`self.fabric = None`), 부모 버퍼(`fabric_q`·`palm_targets`·`_palm_lo`…)는 모양만 유지 | 부모 리셋·앵커·박스 부트스트랩이 읽는다 |
| A 를 덮는 훅은 팔 어댑터 9개뿐 | 보상·관측·종료를 덮으면 A/B 가 같은 과제가 아니다 |

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
CUDA_VISIBLE_DEVICES=1 NOTE="simtoolreal B" ./train.sh open-sens_r_grasp_fj-lstm fj_b1 --num_envs 4096 --headless  # server gpu1
```
로그 `log/rl_games/open-sens/right/grasp-fj/<label>/`. 스모크 게이트: `ep_len` 상승 · abnormal ≈ 0 · `ctrl/joint_err_max` < 0.1 ·
`hand_floor_depth_max` ≈ 0 · `stage/lifted` 상승. 지표 나머지는 A CLAUDE.md 와 같다(`fabric/*` 만 `ctrl/*` 로 바뀐다).
