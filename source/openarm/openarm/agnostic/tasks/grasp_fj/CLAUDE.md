# grasp_fj — 태스크 규칙 (Track B: 팔 7D 증분+EMA + 손 20관절 full-joint+EMA · Fabrics 없음 · 접촉 센서 0)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 설계: `../grasp_kp/DESIGN.md` §1 B 열 · 형제 A = `../grasp_kp`(목표열·관측 공유, 보상은 포크) · 기준선 `grasp_s2r` 은 **불변**

## 목적
**sim2sim 정합 + fabric 노드 제거 + (09.08) 컵 grasp-lift 안정 파지.** 정책이 팔 관절 위치 목표 `q*` 와 손 관절 위치
목표를 직접 낸다(A 는 palm 6D → fabric → q*). 실기 배포가 4노드(fabric 포함)에서 **3노드**(정책 → pd → 드라이버)가
된다. 실기 JTC 는 velocity 를 쓰지 않으므로 sim 도 팔·손에 **위치 목표만** 준다.

★**09.08 SimToolReal 정합 + 과제 목적 축소**(사용자 확정): 팔 매핑·팔 실효 slew·시작 거리·손 full-joint·손 EMA·
성공 술어·목표당 예산은 SimToolReal 과 같고, 외란은 어깨 토크비로 줄였으며, **목표열은 "제자리 유지"(D1-a)** 로
바꿨다 — 과제 목적이 "컵을 안정적으로 잡아 드는 것" 뿐이기 때문이다. b9 는 목표 50개 순회에 최적화된 정책이었다.

## 핵심 계약 (`tests/test_grasp_fj_contract.py`·`tests/test_fj_hand_law.py` 가 잠근다)
| 계약 | 왜 |
|---|---|
| 팔 `q*_t = clamp(q*_{t-1} + k_arm·a)` → EMA α → clamp, **`k_arm 0.025`·`α 0.1` → 실효 slew 0.15 rad/s** | `k_arm ≡ dof_speed_scale × 정책_dt` = 1.5 × 1/60. cfg 가 환산값 1.5 자체를 대조한다 |
| 리셋 시 `q*_{-1}` = 홈 q + `arm_reset_offset_rad`(고정 델타 7개) → 손끝→물체 **104.7 mm** | 그들 팔 속도(0.15 rad/s)는 손이 물체 옆에서 시작하는 것과 한 묶음. 컵 상대 텔레포트는 아니다(08.18 결정). `ctrl/start_ft_dist` + 첫 리셋 부팅 가드(70~140 mm) |
| **손 20관절 full-joint**(`hand_direct=True`, leaf): `raw = lo + ½(a+1)(hi−lo)` → 관절 EMA `hand_ema 0.1` → clamp | SimToolReal `action_utils.py:61-69` 와 같은 꼴. 폐쇄도·램프·`close_gate`·`blocked`·open→grip 보정은 **없다**(09.08 사용자 확정). `_hand_blocked` 는 진단(`ctrl/hand_blocked_frac`)으로만 산다 |
| 손 액션한계 [lo, hi] = URDF 한계(soft = 하드, factor 1.0 을 부팅 대조) ∩ 프로필 `hand_action_limit_override` — 다섯 손가락 `_3/_4` 10개 하한 0(엄지 `_3` 포함, 사용자 재확정 "−0.5 는 꺾이는 자세"), 나머지 URDF 실제 범위(`thumb_2` 대향 [−π, 0] 포함, 사용자 확정). 프로필 리셋 자세(엄지 `_3` −0.5)는 B 리셋이 액션한계로 **clamp 해 관절 상태·EMA 시드에 심는다**(`_hand_reset_q`, 이동 상한 `hand_reset_clamp_max_rad` 0.6) | SHARPA 는 PIP/DIP 하한 0 이라 원시 한계가 안전했지만 테솔로 `_3/_4` 는 ±1.571 대칭 — a=−1 이 손등 −90°(08.23 exploit). 접촉 항 0개라 **범위가 유일한 방어선**. 부팅에서 정규식 미매칭(fullmatch)·폭 0 칸·리셋 자세 과이탈을 죽인다. A 의 프로필 init 은 불변 |
| `hand_velocity_ff_scale = 0`(hand_direct 필수, 검증기) | 램프가 없어 `_syn_vel` 이 최대 ≈19 rad/s 까지 뛴다. SimToolReal 은 위치 목표만 |
| obs `cmd_state` = 팔 `q*_{t-1}`(7) · 액션 블록 27 = 팔 지연 액션 7 + **손 정규화 관절 목표 20**(`_action_obs` 이음매) → **actor 136 / critic 160, action 27** | SimToolReal 은 post-EMA `prev_action_targets` 를 관측한다. 손 EMA 상태(τ≈10스텝)는 a_{t-1}·hand_q 로 복원 불가(09.08 리뷰). A 는 `self.actions` 그대로(산술 불변). fj_b9(22/131/155)와 **호환 안 됨** — FRESH |
| 성공 = **연속** 10회 · tol 0.1125→0.015 · 커리큘럼 게이트 **2.0**(= goal_max 5 의 40%; 상류 3/50 = 6%, 3.0 은 60% 라 10배 엄격) · 스텝 예산 **목표당**(성공 시 시계→2, 0/1 은 검증기 거부) | SimToolReal `env.py:2437`·`utils.py:238`. 공차·연속·`goal_first_z_range (0.2125,0.28)` 는 **3중 잠금쌍**(래치 0.10 불변식). 게이트 입력 = `ctrl/prev_ep_successes_mean` |
| **목표열 = 제자리 유지**: `goal_delta_distance 0.0` · `goal_max 5`(> 게이트 2.0, 검증기) · 회전 0° 상속 | D1-a. 첫 목표 = 리프트 높이, 이후 같은 자리 → "들고 정지" 가 성공. 5목표 후 truncated(value_bootstrap). `REWARD_AUDIT.md` 09.08 2판 ACCEPT |
| 보상 = `fj_reward.py`(**포크**), `goal_bonus` 만 A 와 다름(성공 순간 1회 전액) | `_get_rewards` 는 안 덮는다 — A 의 이음매 `_progress_reward` 만 덮는다 |
| 외란 2.7 N/kg · 0.27 N·m/kg (A 의 1/7.5) | Kuka 어깨 300 N·m vs OpenArm 40 N·m. 파지 품질을 강제하는 유일한 장치 |
| fabric 런타임 0(`self.fabric = None`), 부모 버퍼(`fabric_q`·`palm_targets`·`_palm_lo`…)는 모양만 유지 | 부모 리셋·앵커·박스 부트스트랩이 읽는다 |
| A 를 덮는 훅은 팔·손 어댑터 + 이음매뿐 (`_get_rewards`·`_get_dones`·`_hand_command` 금지) | 보상·종료 본체를 덮으면 A/B 가 같은 과제가 아니다 |

### 배치 방화벽 — 과제 의미 값은 **leaf 에만**
형제 `grasp_fj_rh`(RH56F1)가 `GraspFJEnvCfg` 를 **상속**하고, 등록부는 leaf(`GraspFJTesolloRightEnvCfg`·Short)만
인스턴스화한다. 공차·성공 술어·예산·외란·목표열·`hand_direct`·`hand_velocity_ff_scale` 은 leaf. base 는 메커니즘
(`k_arm`·`arm_slew_rad_s`·`hand_ema`·`goal_clock_restart_step`)만 — fj_rh 는 `hand_direct=False`·자체 `_hand_command` 라
안 읽는다. `test_task_semantics_live_on_the_leaf_so_grasp_fj_rh_is_not_captured` 가 기계 검사한다.

## 성공 기준
1. **안정 파지**: `ctrl/drop_sticky_frac`(들었다 놓친 에피소드) 리프트 후 < 0.05 · `task/lifted_frac` ≥ b9 대역(0.8) ·
   영상에서 손등 접근·손가락 교차 없음.
2. **정합**: 같은 체크포인트·같은 액션열을 sim(이 env)과 실기 pd 노드에 넣었을 때 관절 궤적 오차가 A 경로보다 작다.
3. `ctrl/joint_err_max` < 0.1 rad · `ctrl/arm_limit_sat` ≈ 0 · `task/hand_floor_depth_max` ≈ 0 · `done/hand_floor` ≈ 0.
4. 3노드 배포에서 fabric 노드 없이 정책 출력이 pd 노드 계약(`policy_control` v2)에 바로 실린다.

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/tests \
  source/openarm/openarm/agnostic/tasks/{grasp_kp,grasp_s2r,grasp_fj,grasp_fj_rh}/tests -q
./train.sh open-sens_r_grasp_fj-lstm fj_smoke --num_envs 16 --max_iterations 5 \
  agent.params.config.minibatch_size=256 agent.params.config.central_value_config.minibatch_size=256   # 로컬 스모크
# ★09.08 seed 만 다른 **복제 3런**(같은 설정) — 단일 런으로는 판정 불가(a11/a12/a13: 0.85/0.82/0.00)
/home/oem/launch_fj_c1.sh fj_c1 0 --seed 42
/home/oem/launch_fj_c1.sh fj_c2 0 --seed 7
/home/oem/launch_fj_c1.sh fj_c3 1 --seed 123
```
로그 `log/rl_games/open-sens/right/grasp-fj-lstm-sapg/<label>/`. 부팅 라인: 손 액션한계 20관절 표(리셋 자세 범위 안 ✓) ·
`시작 거리 가드 ✓ ~105 mm`. 스모크 게이트: `ctrl/arm_target_step` ≤ 0.0025 rad(×60 = 0.15 rad/s) · abnormal ≈ 0 ·
`ctrl/joint_err_max` < 0.1 · `hand_floor_depth_max` ≈ 0 · `reset/arm_q_dev_max` ≈ **0.38**(= `arm_reset_offset_rad` 최대 성분 0.3813 — 진단이 홈 기준 편차를 재므로 고정 오프셋이 그대로 보인다; b9 의 ~0 과 다른 것이 정상).

**판정 지표(matched-tol, matched-epoch 금지)**: `ctrl/drop_sticky_frac`(1순위) · `task/lifted_frac` · `task/successes_mean`(상한 5)
· `ctrl/prev_ep_successes_mean`(게이트 2.0 까지의 거리) · `episode_lengths`(유능해지면 600 → 200~300; 성공 에피소드의
첫 목표 도달이 330 스텝을 넘으면 γ 0.99 에서 Check 1 의 3× 기준 미달) · `diag/obj_speed_lifted` · `reward/hand_vel`(상한
−0.19/step = URDF 3.14 rad/s × 20 × 0.003, 무작위 정책 실측 −0.11; 리프트 후에도 −0.10 이상 지속이면 개입) ·
`reward/cmd_rate`(D1-a 에서는 노이즈 세금 −0.71σ/step; hand_vel 과 합이 리프트 전 수입의 30% 넘으면 개입) ·
`ctrl/hand_joint_err_max` · `ctrl/arm_limit_sat` · `diag/action_sat`.
게이트: e300 `stage/lifted` ≥ 0.10 을 3런 중 2런 이상 미달이면 설정 문제(손 법칙·팔 속도 순으로 의심).

⚠ tol 0.1125 + 게이트 2.0 이면 0.015 까지 약 19 커리큘럼 스텝이라 b9(0.06 출발)보다 훨씬 느슨한 공차에 오래 머문다 —
에폭 맞춤 비교는 새 런을 과대평가한다.

지표 나머지는 A CLAUDE.md 와 같다(`fabric/*` 만 `ctrl/*` 로 바뀐다).
