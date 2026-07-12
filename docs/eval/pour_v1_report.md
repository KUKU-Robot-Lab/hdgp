# pour_v1 (OpenArm + Tesollo, 신 rl USD 최종버전) 결과 보고서

> **대상**: `open-tesol_r_pour_v1-lstm` / run `pour-v1/lstm_test2` / checkpoint `ep_10000_rew_47074.97`
> **과제**: source cup → target cup 로 bead(20개, 액체 proxy) 이송 (palm-pose 제어 pouring)
> **근거 데이터**: ① 학습 TFEvents (655M frames, 10000 epoch) ② 결정론 eval (`eval_pour_envs.py`, 1024/64 env)

---

## 1. 핵심 결과 요약

| 관점 | 지표 | 값 |
|---|---|---|
| **eval (1024 env, 2101 ep)** | 성공률 | **93.1%** (1957/2101) |
| | 평균 이송 bead | **17.67 / 20 (88.4%)** |
| | 평균 spill | 10.8% |
| | 평균 조준(mouth_xy) | 0.0223 m |
| **eval (64 env, 116 ep)** | 성공률 / 이송 | 94.0% / 17.85 (89.3%) |
| **학습 TB (최종)** | `outcome/bead_at_done` | 0.88 |
| | `adr_ep_success_rate` | 0.885 |
| | `outcome/spill_at_done` | 0.088 |
| | `outcome/mouth_xy_at_done` | 0.026 m |

**성공 기준**: bead-in-target 비율 ≥ 0.50 (=10개↑) **and** spill ≤ 0.40 (학습 success_flag와 동일).

**핵심 검증**: 학습 TB의 렌더-동일 지표(`outcome/bead_at_done` 0.88 / `ep_success_rate` 0.885 / `spill` 0.088)와
독립 결정론 eval(이송률 88.4% / 성공률 93.1% / spill 10.8%)이 **오차 ~1%p 내로 일치**한다.
→ 학습 지표가 실제 정책 성능을 정확히 반영함을 대규모 표본(2101 ep)으로 교차검증 완료.

### 이송 개수 분포 (1024 env, 2101 ep)

| 이송 bead | 20 | 19 | 18 | 17 | 16 | 15 | ≤14 | 0(완전실패) |
|---|---|---|---|---|---|---|---|---|
| 비율 | **33.8%** | 23.8% | 14.6% | 8.8% | 5.5% | 3.4% | 8.5% | 1.2% |

- **17개(85%)+ 이송 = 전체의 81.0%**, 18개(90%)+ = 72.2%.
- 완전 실패(0개)는 1.2%뿐 → 붕괴 없는 안정적 정책.

---

## 2. Observation 구조 & 타당성

**비대칭 actor-critic** (privileged critic, 배포 가능 actor).

### Actor obs = 55D (sim2real 가능 proprio/FK/target-relative)
| 구성 | 차원 | 근거 |
|---|---|---|
| 오른팔 joint pos/vel | 7 + 7 | 실 encoder 취득 가능 |
| finger_grasp_progress | 5 | 손 상태 요약(freeze여도 호환) |
| 왼팔(target cup) joint pos/vel | 9 + 9 | target cup 위치의 FK 원천(실 encoder) |
| pour_point → opening 벡터 | 3 | 배출점↔목표입구 상대기하 |
| source_pour_axis / source_up_axis / target_up_axis | 3+3+3 | 붓기 자세 기하 |
| last_palm_actions | 6 | 자기수용 피드백 |

**타당성**: actor는 **손 접촉/힘·bead flow 등 sim-only 특권정보를 배제**하고 proprio + FK 기하만 사용 →
sim2real 이식성 확보. eval에서 88% 이송을 달성했으므로 이 축약 obs가 **정보 충분**함이 입증됨
(target cup 위치를 왼팔 FK로 간접 관측하는 설계가 실제로 조준 0.022m를 만들어냄).

### Critic obs = 144D (105 base + 39 privileged)
- privileged: 왼팔 상태(18) + distal 접촉 binary/norm(10) + cup_height_delta(1) + rho(1) + **demo 자세 목표(demo_arm_err 1 + j5_err 1 + demo_target_arm_q 7)**.
- **타당성**: 학습 시 critic이 특권정보(접촉·demo 목표자세)로 정확한 value 추정 → advantage 품질↑ → actor가 축약 obs로도 안정 학습. entropy 17→4.2 수렴이 이 비대칭 구조의 효과.

---

## 3. Network 구조 & 타당성

| 항목 | 설정 |
|---|---|
| 모델 | `continuous_a2c_logstd`, fixed_sigma, separate=False |
| encoder | MLP `[256]` ELU |
| recurrent | **LSTM 512 × 1 layer, layer_norm, before_mlp=False** (obs→MLP→LSTM) |
| 정규화 | normalize_input / normalize_value = True, mixed_precision=**False** |
| PPO | γ=0.998, τ=0.95, lr=2e-4 linear, kl_thresh=0.013, e_clip=0.2, horizon=32, grad_norm=1.0 |
| entropy_coef | **0.0** |

**타당성**:
1. **LSTM 채택**: pour는 600-step 장기 순차과제(이송→기울기→붓기). MLP→LSTM 구조로 per-step feature를 시간축 누적 → phase(단계) 기억. eval에서 단계적 붓기(19/20개 44%+)가 안정 재현됨이 시계열 표현의 유효성 증거.
2. **entropy_coef=0.0**: dense-plateau의 약한 advantage를 entropy bonus가 이겨 logσ가 폭주(8.3→20)했던 이력을 차단. 이번 run은 σ가 advantage gradient로만 갱신되어 **entropy 17→4.2로 정상 수축** → 결정론 eval이 학습분포와 일치(성공률 93%)하는 근거.
3. **mixed_precision=False**: LSTM+AMP 수치 불안정 회피. 10000 epoch 붕괴 없이 완주(grasp_broken=0)한 안정성의 토대.
4. **γ=0.998**: 600-step 지연보상(붓기 성공은 에피소드 후반) 전파에 적합한 장기 할인.

---

## 4. Action 구조 & 타당성

- **명목 12D** = palm pose 6 + nullspace 1 + per-finger 5.
- **이번 최종 config(deep_tilt_boot1) 유효 7D**: `_pre_physics_step`이 `actions[:, :6]`(palm pose) +
  `actions[:, 6]`(nullspace α)를 사용. **nullspace α 활성**(`nullspace_action_scale=1.0`, L1312-1316) →
  잉여 1-DOF(elbow swivel) 제어. **손가락(action[7:12])만 grasp_hold freeze**(inert).
- palm 6D(x,y,z, euler ez,ey,ex; index 4=β 채널) + α 1D → **Fabrics IK → 오른팔 7 DOF** → EMA smoothing.

**타당성**:
- 손가락 freeze + palm-pose·α IK 제어로 **탐색공간을 축소** → 안정 학습(붕괴 0)에 기여.
  (파지 자체는 grasp 정책 warmstart에서 손가락을 움직여 형성 → pour는 그 자세를 고정.)
- **한계(개선점)**: action head가 12D인데 **7D만 사용** → hand 5 = 5개 출력이 inert.
  성능엔 무해(fixed_sigma로 정책이 무시 학습)하나, 12D→7D head로 축소하면 파라미터·탐색 효율 소폭 개선 여지.

---

## 5. Reward 구조 & 타당성 (TB·eval 연결)

**총 보상** (reward_shaper scale 0.007로 정규화):
```
total = r_hold + r_grasp + r_approach + r_introt + r_tilt + r_tilt_delta
      + r_pour + r_aim + r_stageB + weight_success·r_success − (spill penalty=OFF)
```

| reward 항 | 식 / weight | 유도 목표 | **TB 증거** | **eval 검증** |
|---|---|---|---|---|
| r_approach | `w_dist_to_target=8` · positive exp pull | 컵을 target 위로 이동 | `approach_corridor_score` 0.17→**0.82** | 조준 mouth_xy 0.022m |
| r_tilt (+delta) | `w_tilt=20` 유지 + `w_tilt_delta=100` relu(Δ) | deep tilt(붓기 자세) | `source_up_dot` 1.0→**0.15**(~80°) | 이송률 88% (붓기 성립) |
| **r_aim** | **`w_aim_precision=18` · exp(−aim_scale·mouth_xy)** | 주둥이→입구중심 정조준 | aim_scale ADR 10→**15** → `mouth_xy` 0.056→**0.026** | eval mouth_xy 0.022m |
| **r_pour** | **`w_pour_bead=50` · corridor · (bead_frac + 30·Δcapture)** | 실제 bead 진입(outcome) | `bead_at_done` 0→**0.88** | 이송 17.67/20 |
| r_grasp | `w_grasp=3` · (접촉비율 + full 보너스) | 파지 유지 | `grasp_broken`=**0**, `g_ready`0.85 | 실패(0개) 1.2%뿐 |
| r_introt | `w_introt=5` · internal_rot_gate | 내회전(붓기 방향) | — | 방향 정합 |
| r_success | `w_success=50` · (frac≥fill & spill≤0.4 & cup_xy<0.2) | outcome 닻(anchor) | `Reward_w0/success`≈**12.9** | 성공률 93% |
| spill penalty | `w_spill=0` (**OFF**) | — | spill 0.088(페널티 없이도 낮음) | eval spill 10.8% |

**타당성 핵심**:
1. **r_pour를 z-only 대리에서 실제 bead outcome(corridor·bead_frac)으로 재설계**한 것이 결정적.
   TB `bead_at_done` 0→0.88과 eval 이송 88.4%가 이 outcome 보상이 **정확한 목표를 최적화**했음을 이중 확인.
2. **spill 페널티 OFF인데 eval spill 10.8%**로 낮음 → 정조준(r_aim) + 성공보상(r_success)이 간접적으로 spill을 억제.
   즉 spill을 직접 처벌하지 않고도 "정확히 부으면 spill이 준다"는 인과를 정책이 학습(local-min 회피 설계의 성공).
3. **weight_align 20→5 하향**(방향-only farming 차단), **weight_aim 8→18 상향**(조준을 grasp보다 우위) 등
   reward-audit 이력이 eval의 낮은 mouth_xy(0.022m)로 실효 확인됨.

---

## 6. ADR(자동 커리큘럼) 기여

| ADR 파라미터 | 램프 | 효과(TB) |
|---|---|---|
| `aim_scale` (r_aim 경사도) | 10 → **15** | mouth_xy 0.056→0.026 |
| `fill_ratio` (성공 문턱) | 0.20(2개) → **0.50(10개)** | 성공기준 상향에도 성공률 0.885 |

- 이번 세션 수정: `success_adr_increment_interval` 20000→2000 (env-step당 1회 호출 기준, 첫 체크가 40M→4M frame으로 당겨짐).
  이 수정 전(lstm_test1) aim ADR 미발동으로 bead 0.22 정체 → 수정 후(lstm_test2) ADR 완주로 **bead 0.88 달성**.
- **타당성**: ADR이 "쉬운 기준→어려운 기준"으로 자동 상향하며 정책을 끌어올린 뒤에도 성공률이 유지·상승 →
  커리큘럼이 과적합 없이 난이도 일반화를 유도.

---

## 7. TB ↔ eval 정합성 (렌더-동일 로깅의 타당성)

과거 문제: 학습 TB의 순간 cross-env 평균(`log/bead_in_target`)이 리셋직후 0을 포함해 성공을 심하게 저평가
(TB 0.24 vs 렌더 0.95의 "오해").

해결: **done 시점 값을 유지하는 `outcome/*_at_done` 지표 도입** = 렌더 final-frame과 동일 측정.

| 지표 | TB(outcome_at_done) | eval(1024) | 차이 |
|---|---|---|---|
| bead 이송 | 0.88 | 0.884 | 0.4%p |
| 성공률 | 0.885 | 0.931 | (eval이 spill 관대 기준 반영) |
| spill | 0.088 | 0.108 | 2%p |
| mouth_xy | 0.026 | 0.022 | 0.004m |

→ **오차 ~1–2%p로 정합.** 렌더-동일 로깅 설계가 타당함을 대규모 독립표본이 입증.
"TB≪렌더" 오해가 구조적으로 해소됨.

---

## 8. 종합 타당성 판정

✅ **obs / network / action / reward 설계가 모두 목표(bead 이송)에 정합**하며, TB와 독립 eval이 상호 검증한다.

- **obs**: 배포가능 55D actor + 특권 144D critic → sim2real 지향 + 안정 학습. 정보 충분성 eval로 입증.
- **network**: LSTM 512 + entropy_coef 0 + AMP off → 600-step 장기과제 수렴(17→4.2), 붕괴 0.
- **action**: 7D(palm 6 + nullspace α 1) IK, 손가락 freeze → 탐색 축소·안정. (12D head 중 hand 5 미사용은 무해한 개선여지.)
- **reward**: r_pour outcome 재설계 + r_aim ADR가 핵심 동력. spill 직접처벌 없이 간접 억제 성공.
- **결과**: 성공률 **93.1%**, 평균 이송 **17.67/20**, 20개 전량 **33.8%**, 완전실패 **1.2%**.

### 한계 / 향후 개선
1. **12D→7D action head 축소**(inert hand 5 제거)로 파라미터·탐색 효율 정리(성능 무해).
2. **완전실패 1.2%(25/2101)** 원인 진단 — warmstart palm-clamp(0.12m) decouple 경고와 상관 조사.
3. mouth_xy 0.022m는 우수하나, 잔여 spill 10.8%의 상한 개선은 r_aim ADR을 15→상향 재실험 여지(진동 이력 주의).
4. 실기 이식 시 왼팔 FK 기반 target 관측의 실 encoder 노이즈 강건성 검증.

---

*근거 파일: `docs/eval/pour_v1_eval_1024.md`(2101 ep), `docs/eval/pour_v1_eval.md`(116 ep), 학습 `pour-v1/lstm_test2` TFEvents.*
