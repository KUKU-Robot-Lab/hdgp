# Pour 정책 비교분석: 우리 레포(Tesollo) vs DexPour (IROS 2025)

> **목적**: 사내 pouring 레포(`hdgp`)의 최신 학습 결과를 IROS 2025 논문 **DexPour**와 방법론·성능 양면에서 비교한다.
> **비교 대상**
> - **우리 pour-v1** (단팔 붓기, 완성) — `open-tesol/right/pour-v1/lstm_test2`, ckpt `ep_10000_rew_47074`
> - **우리 pour-sensor** (양팔 붓기, 최신·진행형) — `open-tesol/both/pour-sensor/test6` (558M frames)
> - **DexPour** — Franka + Allegro(23-DoF), APA + hierarchical reward + curriculum
>
> **근거 데이터**
> - 우리: 서버 로그 `hdgp/log/rl_games/...` TFEvents 직접 파싱(2026-07-03), `docs/eval/pour_v1_report.md`, `docs/eval/pour_sensor_report.md`
> - DexPour: `repo/DexPour.../dexpour/DexPour Effective and Efficient High-DoF Robotic.md`
>
> ⚠️ **최신성 주의**: 기존 `pour_sensor_report.md`는 **test2(ep5100)** 기준으로 낡았다. 본 문서는 **test6 실측치**로 갱신한 값을 사용한다(아래 §3 참조).

---

## 1. 한눈 요약 (TL;DR)

| 축 | DexPour | 우리 pour-v1 (단팔) | 우리 pour-sensor test6 (양팔) |
|---|---|---|---|
| **태스크 범위** | approach→grasp→transport→**pour** (풀 파이프라인) | **pour 단계만** (grasp는 별도 정책 warmstart) | pour 단계만 + **양팔 협조** (왼팔이 받는컵 능동 이동) |
| **손/자유도** | Allegro 16-DoF **전관절 직접 RL** (총 23-DoF) | palm-pose 6D IK + **손가락 freeze** (유효 7D) | palm 6D + nullspace 1D + **왼팔 TCP 3D** (유효 10D) |
| **유체 대리** | APA: rigid sphere ≤32개 | bead 20개 (강체 proxy) | bead 20개 |
| **최종 유체 검증** | PBD 유체로 test (92%@70%fill) | 별도 `pour_fluid_eval`(PBD record-replay) 프레임워크 보유 | 동일 프레임워크 |
| **핵심 성능** | 유체이송 **92%**@70%fill, **99%**@30%fill | eval 성공률 **93.1%**, 이송 **17.67/20 (88.4%)** | TB bead_at_done **0.93**, spill **1.7%** (eval 미실시) |
| **수렴 상태** | 완료(논문) | **완전 수렴** (entropy 4.2) | **진행형** (entropy 12.8, 수렴 중) |

**핵심 결론**: 우리 접근은 DexPour와 **다른 문제 분할**을 택했다. DexPour는 grasp/transport까지 한 정책이 통짜로 배우는 반면, 우리는 **grasp를 warmstart로 분리하고 pour만 집중 학습**한다. 이 덕분에 우리는 (a) DexPour에 없는 **bimanual pour**(pour-sensor)로 확장했고, (b) **palm-pose IK + 손가락 freeze**로 탐색공간을 줄여 붕괴 없는 안정 수렴을 얻었다. 반대급부로 DexPour가 자연히 다루는 **바닥 컵 집기·리프트**는 우리 pour 태스크의 스코프 밖이다.

---

## 2. DexPour 핵심 (논문 요약)

- **문제**: 고DoF dexterous hand로 액체 붓기. 유체 시뮬 비용이 학습 병목.
- **기여 3축**
  1. **Hierarchical Reward** — pouring을 4단계로 분해: **Approaching / Grasping / Transporting / Pouring**. 각 단계는 binary trigger `(λ, μ, ν, ρ)`로 순차 활성(선행단계 완료 검증 내장).
     - approach: hand-cup 거리·높이·finger 간격 penalty (pre-grasp 거리까지만)
     - grasp: finger-cup 거리 + contact 카운트 + 4-finger 전접촉 시 큰 보상
     - transport: 선형 lift 보상(threshold 이후 정지) + `e^{-2·dist}` 컵-타겟 거리 + tilt penalty
     - pour: **45°에서 peak인 tilt 보상** + align 보상 `(1+cosθ)/2`
     - trigger 임계값: `d_approach=0.1m, c_finger=4, h_lift=0.15m, d_pour=0.17m`
  2. **APA (Approximated Proxy Abstraction)** — 유체를 rigid sphere로 근사. 가설: (1) 매크로 운동학적 유사성, (2) 정책 학습이 미시 유체 오차에 둔감. → **학습시간 81.6% 절감**, PBD 대비 이송효율 동등.
  3. **Curriculum (3-stage)** — ①16k step: sphere 1개·penalty 낮게 ②32k step: penalty↑(부드러운 동작) ③64k step: penalty 대폭↑·sphere ≤32개.
- **셋업**: Franka 7-DoF + Allegro 16-DoF (23-DoF), Isaac Lab, 2048 env, dt=0.008s, RTX 4060Ti. Actor-critic MLP **(512,512,256,128) ELU**.
- **태스크 난이도**: 바닥(지면)의 머그(8.4cm×18cm)를 지면충돌 없이 집어 → **0.5m 이상 리프트** → 0.4m 높이 bowl(20cm)에 붓기.
- **성능**: 유체이송 **92%@70%fill / 96%@50% / 99%@30%**. Ablation으로 각 stage 보상 필수성 입증(sparse baseline 실패, Config.2 커리큘럼 없으면 조기수렴 실패).

---

## 3. 우리 최신 결과 (TFEvents 실측, 2026-07-03)

세 실험의 수렴부(마지막 5% 평균) 실측 지표:

| 지표 (렌더-동일 `outcome/*_at_done`) | pour-v1 lstm_test2 | pour-sensor **test6** (최신) | pour-sensor test4 (구) |
|---|---|---|---|
| 학습량 (frames) | 655M | 558M | 116M |
| `losses/entropy` | **4.24** (완전수렴) | **13.0** (수렴 중) | 21.6 (미수렴) |
| `outcome/bead_at_done` | 0.862 | **0.932** | 0.683 |
| `adr_ep_success_rate` | 0.884 | 0.868 | 0.594 |
| `outcome/spill_at_done` | 0.109 | **0.017** | 0.115 |
| `outcome/mouth_xy_at_done` | 0.027 m | **0.016 m** | 0.025 m |
| `cup_center_xy_dist` (양팔정렬) | 0.108 | 0.105 | 0.103 |
| `grasp_broken` | 0 | 0 | 0 |
| `episode_lengths` | 765 | 899 | 850 |
| reward | 49.3k | 76.9k | 55.1k |

**독립 eval(pour-v1만, 1024 env·2101 ep)**: 성공률 **93.1%**, 평균 이송 **17.67/20 (88.4%)**, 완전이송(20개) 33.8%, 완전실패 1.2%. TB `outcome`과 오차 ~1–2%p로 교차검증됨.

**pour-sensor test6 관찰**
- test4→test6에서 **크게 발전**: entropy 21.6→13.0, bead_at_done 0.68→**0.93**, spill 0.115→**0.017**, mouth_xy 0.025→**0.016**.
- 즉 **양팔 test6은 outcome 지표상 단팔 v1을 이미 상회**(bead 0.93>0.86, spill 1.7%<11%, mouth 0.016<0.027).
- 단, entropy 13.0으로 아직 v1(4.2)만큼 결정론적이지 않음 → **결정론 eval 미실시**. v1의 이력상 결정론 eval은 TB와 정합했으나, sensor는 미수렴 구간이라 eval 확정 전. **본 지표는 TB stochastic 기준**임에 유의.
- 기존 `pour_sensor_report.md`(test2, bead 0.77·entropy 19.4)는 **낡음** → test6로 갱신 필요.

---

## 4. 방법론 상세 비교

### 4.1 태스크 분할 (가장 큰 차이)

| | DexPour | 우리 |
|---|---|---|
| grasp | **정책이 직접 학습** (finger contact 보상) | **별도 grasp 정책의 성공상태를 warmstart**로 로드 → pour는 파지 자세를 freeze |
| transport(lift) | **정책이 직접 학습** (선형 lift 보상, 실패율 높은 동작으로 강조) | **스코프 밖** (컵은 이미 잡힌 상태에서 시작) |
| pour | 4번째 stage | **전 태스크의 초점** |
| bimanual | 없음 (받는컵 고정) | **pour-sensor에서 왼팔이 받는컵 능동 이동** |

→ DexPour의 강점은 **엔드투엔드 파이프라인**(집기부터 붓기까지 한 정책). 우리의 강점은 **모듈화**로 pour 난제(deep tilt·정조준)에 자원 집중 + **bimanual 확장**.

### 4.2 손 제어 & 자유도

- **DexPour**: Allegro 16관절을 전부 RL 액션으로 직접 제어(23-DoF). 표현력 최대지만 탐색공간이 크고, grasp 붕괴 위험을 contact 보상으로 상쇄.
- **우리**: palm 6D pose를 **Fabrics IK**로 팔 7-DoF에 매핑 + **손가락 freeze**. 탐색공간을 palm-pose(+nullspace/leftTCP)로 축소 → **grasp_broken=0**의 안정성. 반대로 in-hand 재파지 같은 손가락 dexterity는 포기.

### 4.3 유체 대리 (수렴 철학은 동일)

- 양측 모두 **rigid proxy로 유체를 근사**(DexPour APA sphere ≤32 / 우리 bead 20). 핵심 가설("정책은 미시 유체 오차에 둔감")이 동일.
- **최종 검증**: DexPour는 학습 후 PBD 유체로 이송효율 측정. 우리도 `pour_fluid_eval` **record-and-replay**(정책 궤적 기록→raw-app PBD 씬 재생)로 동일 검증 경로 보유 — 단, 이 프레임워크는 **GPU 실행 대기**(문서만 완비).

### 4.4 Reward 구조

| | DexPour | 우리 |
|---|---|---|
| 분해 | 4-stage **binary trigger** `λμνρ` (곱연쇄로 선행완료 강제) | gate 기반 stage(approach/tilt/pour) + **outcome bead 보상**(corridor·bead_frac) |
| tilt 목표 | **45° peak** (과회전 방지) | **deep tilt** `source_up_dot→0.15(~80°)` (v1), `-0.2(~102°)`(sensor) |
| align | `(1+cosθ)/2` | 유사 개념(내회전 gate·approach corridor) |
| 성공 앵커 | 최종 task 완료 보상 | `r_success` (fill≥ratio & spill≤0.4 & cup_xy<0.2) |

→ **주목할 차이**: DexPour tilt는 45°에서 peak(spill 억제 목적)인 반면, **우리는 80~102°의 deep tilt**를 유도한다. 이는 우리 bead가 컵 바닥에서 완전 배출되려면 더 깊은 기울기가 필요했기 때문(v5 진단 이력의 `weight_tilt` 튜닝 근거). 태스크 기하가 다르면 tilt 목표각도 달라짐을 보여주는 대비점.

### 4.5 Curriculum

- **DexPour**: 수동 3-stage (step 구간별 sphere 수·penalty 가중 상향).
- **우리**: **ADR**(Automatic Domain Randomization) — `aim_scale 10→15`, `fill_ratio 0.20→0.50` 자동 상향. "쉬운 기준→어려운 기준" 자동 커리큘럼. pour-v1은 `success_adr_increment_interval` 수정 후 ADR 완주가 bead 0.22→0.88의 결정적 계기였음.

### 4.6 네트워크

- **DexPour**: MLP (512,512,256,128) — recurrent 없음.
- **우리**: MLP[256] → **LSTM 512×1** — pour는 600-step 장기 순차과제라 phase 기억용 recurrence 채택. entropy_coef=0, AMP off로 붕괴 없이 수렴.

---

## 5. 성능 비교 — 직접 대응은 조심스럽게

**공정한 비교의 한계**: DexPour "92%@70%fill"은 **PBD 실유체 이송률**, 우리 "88.4%"는 **bead(강체) 이송률**이라 측정 대상이 다르다. fill 조건(70%/50%/30%)도 우리엔 대응 축이 없다(우리는 bead 20개 고정).

| 관점 | DexPour | 우리 pour-v1 | 우리 pour-sensor test6 |
|---|---|---|---|
| 이송 지표 정의 | PBD 유체 부피비 | bead 개수비 (eval) | bead_at_done (TB) |
| 대표값 | 92% (@70%) / 99% (@30%) | 88.4% 이송 / 93.1% 성공 | 93.2% (bead) |
| 정밀도(spill) | (명시 없음, RMSE로 안정성) | 10.8% spill | **1.7% spill** |
| 검증 강건성 | PBD 실유체·200+ trial | **독립 eval 2101 ep** 교차검증 | TB만 (eval 대기) |

**해석**
- **동급 수준**: 우리 pour-v1의 이송률 88.4%/성공률 93.1%는 DexPour 92%(70%fill)와 **동일 리그**. 단, 측정 정의 차이로 우열 단정 불가.
- **우리 강점**: (a) **독립 대규모 eval(2101 ep)로 TB와 1–2%p 교차검증** — DexPour보다 재현성 근거가 정량적으로 두터움. (b) pour-sensor의 **spill 1.7%**는 매우 낮은 정조준. (c) **bimanual** 확장은 DexPour 범위 밖.
- **DexPour 강점**: (a) **바닥 집기+리프트 포함 풀 파이프라인** — 우리는 grasp를 분리(warmstart)해 이 난이도를 우회. (b) **PBD 실유체 최종검증 완료** — 우리는 프레임워크만 있고 GPU 실행 대기. (c) 학습효율(APA로 81.6% 시간절감)을 정량 ablation.

---

## 6. 시사점 & 다음 액션

1. **pour_sensor_report.md 갱신 필요** — test2(낡음)→**test6** 실측으로. test6은 bead 0.93·spill 1.7%로 이미 v1 상회(단 entropy 13.0, eval 확정 전).
2. **pour-sensor 결정론 eval 실시** — entropy가 v1 수준(~4)으로 더 수렴하면 `eval_pour_envs.py`로 1024env 교차검증 → TB↔eval 정합 재확인.
3. **PBD 실유체 검증 실행** — DexPour의 최종검증 강점을 따라잡으려면 `pour_fluid_eval`(record-and-replay)을 GPU에서 돌려 bead→실유체 이송률 상관을 확보. 이것이 DexPour와의 **가장 큰 검증 갭**.
4. **fill 조건 변량 실험 검토** — DexPour는 30/50/70% fill 강건성을 보였다. 우리도 bead 개수/컵 유량을 변량해 volume robustness를 정량화하면 논문성 비교가 명확해짐.
5. **방법론 포지셔닝** — 우리 기여의 차별점은 **(i) grasp-warmstart 모듈화로 pour 집중 + (ii) palm-IK/finger-freeze 안정화 + (iii) bimanual pour + (iv) ADR 자동 커리큘럼**. DexPour의 "풀 파이프라인 hierarchical reward"와 상호보완적 — 향후 우리 grasp 정책과 결합하면 DexPour식 end-to-end도 재현 가능.

---

*근거: TFEvents 직접 파싱(`scripts/tools/parse_tfevents.py`, 2026-07-03) · `docs/eval/pour_v1_report.md` · `docs/eval/pour_sensor_report.md` · `docs/eval/pour_fluid_eval_README.md` · DexPour(IROS 2025) 논문 md.*
