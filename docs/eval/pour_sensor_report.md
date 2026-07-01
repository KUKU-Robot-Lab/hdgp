# pour_sensor (OpenArm 양팔 + Tesollo, 신 rl USD) 양팔 조작 베이스라인 보고서

> **대상**: `open-tesol_b_pour_sensor-lstm` / run `both/pour-sensor/test2` / checkpoint `ep_5100_rew_126989`
> **성격**: **진짜 양팔(bimanual) 조작 물붓기** — 오른팔 붓기 + **왼팔 TCP 제어 해제**로 받는컵을 능동 이동.
> **상태**: ⚠️ **미완성 / 진행형 베이스라인** (학습 지속 중, entropy 미수렴). 완성본이 아니라 양팔 조작의 출발 기준선.
> **근거**: ① 학습 TFEvents (test2, ~671M frames, ep~5125) ② 결정론 eval (`eval_pour_envs.py`, 1024 env)

---

## 1. 핵심 결과 요약

| 관점 | 지표 | 값 |
|---|---|---|
| **eval (1024 env, 1146 ep)** | 성공률 | **88.0%** (1008/1146) |
| | 평균 이송 bead | **13.88 / 20 (69.4%)** |
| | 평균 spill | 10.9% |
| | 평균 조준(mouth_xy) | 0.048 m |
| **학습 TB (ep~5125)** | `outcome/bead_at_done` | 0.77 |
| | `adr_ep_success_rate` | 0.737 |
| | `outcome/spill_at_done` | 0.033 |
| | `losses/entropy` | **19.4 (미수렴)** |

**성공 기준**: bead-in-target ≥ 0.50 (10개↑) & spill ≤ 0.40 (pour_v1과 동일).

**해석**: 성공률 88%로 "받는컵에 절반 이상 붓기"는 안정적으로 달성. 다만 **평균 이송 13.88/20, 완전 이송(20개) 6.9%**로
정밀도는 pour_v1(17.67/20, 20개 33.8%)에 못 미친다. **entropy 19.4 미수렴** = 아직 정책이 확률적·학습 진행 중 →
"완성 아닌 베이스라인"과 정합. 양팔 자유도 증가(6D→10D)로 탐색공간이 커져 수렴이 느리다.

### 이송 개수 분포 (1024 env, 1146 ep)

| 이송 bead | 20 | 19 | 18 | 17 | 16 | 15 | 14 | 13 | 12 | 11 | ≤10 | 0(실패) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 비율 | 6.9% | 7.3% | 9.4% | 11.0% | 9.4% | 9.9% | 11.2% | 9.5% | 8.6% | 3.4% | 5.6% | **8.5%** |

- **14~17개 구간에 집중(≈41%)** — v1의 "20개 몰림"과 달리 넓게 퍼진 미수렴 분포.
- **완전 실패(0개) 8.5%** (v1 1.2%의 7배) → 미수렴 정책의 잔여 실패 모드.

---

## 2. pour_v1 대비 — 무엇이 "양팔"인가

| 구분 | pour_v1 (단팔 붓기) | **pour_sensor (양팔)** |
|---|---|---|
| 오른팔 | 6D palm pose → Fabrics IK | 동일 (6D palm) |
| **오른팔 nullspace** | 미사용 | **1D α 해제** (팔꿈치↔손목 잉여자유도 정책 제어) |
| **왼팔** | 고정(정지) | **3D TCP 위치 delta → DifferentialIK(DLS)** 능동 제어 |
| 받는(target) 컵 | 고정 위치 | **왼손 kinematic-follow** (왼팔이 컵을 옮김) |
| 왼손 orientation/그리퍼 | — | upright 고정 / 닫힘 고정 |
| **유효 action** | 6D | **10D** (palm6 + nullspace1 + leftTCP3) |

→ **받는컵을 왼팔이 능동적으로 붓는 지점으로 가져가는** 진짜 양팔 협조. TB `log/cup_center_xy_dist` 0.145→**0.053**이
왼팔이 두 컵 간 거리를 좁히는(양팔 정렬) 증거.

---

## 3. Observation / Network / Action 구조 & 타당성

### Obs (pour_v1과 동일 레이아웃)
- **Actor 55D**: 오른팔 q/qd(14) + finger_grasp_progress(5) + **왼팔 q/qd(18, target cup FK 원천)** + 기하(pour_point→opening·pour/up axis 12) + last_palm_actions(6).
- **Critic 144D**: 105 base + 39 privileged(demo 목표자세·distal 접촉·rho·cup_height).
- **타당성**: 왼팔 상태(18D)가 이미 actor obs에 있어, 왼팔이 제어대상이 되어도 정책이 자기 상태를 관측 → 양팔 협조 학습 가능.

### Network (pour_v1과 동일)
- MLP[256]ELU → **LSTM 512×1** (layer_norm), continuous_a2c_logstd, fixed_sigma.
- entropy_coef=**0.0**, mixed_precision=False, γ=0.998, horizon=32.
- **타당성/한계**: v1과 동일 아키텍처인데 **entropy 21.3→19.4로 거의 미수축** (v1은 17→4.2). 원인은 아키텍처가 아니라
  **10D 양팔 action의 넓은 탐색공간 + 더 긴 에피소드(1162 step)** → 수렴에 더 많은 업데이트 필요. 즉 아키텍처는 타당,
  **학습량 부족**이 현재 정밀도 격차의 주원인.

### Action (15D 명목 / 10D 유효)
- 명목 15D = palm6 + nullspace1 + hand5 + **leftTCP3**. hand5는 grasp freeze로 inert → 유효 10D.
- **타당성**: 왼팔 orientation·그리퍼를 고정하고 **위치 3D만** 여는 축소 설계로 양팔 문제를 다루기 쉽게 만듦(합리적 baseline 단순화).

---

## 4. Reward & TB 연결

reward 구조는 pour_v1과 동일 골격(오른팔 붓기 항 공유) + 양팔 정렬은 `cup_center_xy_dist`·approach로 간접 유도.

| 항/지표 | TB 증거 | eval 검증 |
|---|---|---|
| deep tilt (r_tilt) | `source_up_dot` 1.0→**-0.2 (~102°, v1보다 깊음)** | 이송 성립(성공 88%) |
| 양팔 정렬 | `cup_center_xy_dist` 0.145→**0.053** | — |
| aim ADR | aim_scale 10→15, `mouth_xy` 0.031→0.044 | eval mouth_xy 0.048m (v1 0.022보다 큼) |
| outcome (r_pour) | `bead_at_done` 0→**0.77 (상승 중)** | 이송 13.88/20 |
| 파지 | `grasp_broken`=0 | 완전실패 8.5% |
| spill(OFF) | `spill_at_done` **0.033** | eval spill 10.9% |

**주목**: TB `bead_at_done` 0.77 vs eval(결정론) 0.694 — **v1과 달리 결정론 eval이 학습 stochastic보다 낮다.**
원인: **entropy 19.4로 정책이 아직 확률적**이라 deterministic(평균) action이 최적이 아님 + 미수렴. 정책이 수렴하면
(entropy↓) 두 값이 v1처럼 수렴할 것으로 예상. **이 격차 자체가 "미완성 베이스라인"의 정량 지표.**

---

## 5. 종합 판정 (베이스라인으로서)

✅ **양팔 조작 물붓기가 동작하는 유효한 베이스라인.** 왼팔이 받는컵을 능동 이동시키며 오른팔이 붓고,
성공률 88%·평균 13.88/20 이송을 달성. deep tilt(~102°)·양팔 정렬(cup_dist 0.053)이 협조 동작을 확인.

⚠️ **아직 미완성** (설계대로): entropy 19.4 미수렴, 완전이송 6.9%·완전실패 8.5%. pour_v1(단팔) 대비 정밀도 격차는
**아키텍처 문제가 아니라 10D 양팔 탐색공간의 학습 미완**이 원인 (TB↔eval 역전이 이를 정량화).

### 베이스라인 → 개선 로드맵
1. **학습 지속**: entropy 수렴(→10 이하)까지 계속 → 결정론 eval이 TB에 수렴하는지 재평가.
2. **완전실패 8.5% 진단**: 왼팔 TCP workspace 클램프 / 양팔 자세 decouple 케이스 분석.
3. **양팔 정렬 보상 명시화**: 현재 간접(cup_center_xy_dist) → 왼팔-오른팔 상대정렬 직접 보상 검토(reward-audit 필요).
4. **왼팔 orientation 자유도 해제** 여부(현재 upright 고정)로 표현력 확장 검토.
5. 수렴 후 pour_v1과 동일 포맷으로 최종 보고서 갱신.

---

*근거 파일: `docs/eval/pour_sensor_eval_1024.md`(1146 ep), 학습 `both/pour-sensor/test2` TFEvents. 비교: `docs/eval/pour_v1_report.md`.*
