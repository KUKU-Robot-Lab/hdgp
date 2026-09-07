# 실데이터 의존도 저감 정량분석 — 과제 미팅 브리프

- 작성일: 2026-09-02
- 과제 주제: **실데이터 의존도를 낮춘 휴머노이드 자율 조작 지능 기술 개발**
- 대상 시스템: Bimanual Precision Pouring — RA-L
- 목적: 실데이터 사용량, 학습 효율, 조작 정확성을 정량적으로 비교할 수 있는 지표와 현재 증거 상태를 정리한다.

## 1. 결론 요약

현재 시스템은 실제 pouring demonstration **10회, 총 109.2초**를 사용한다. 실제 trajectory 전체를 정책이 직접 모방하는 대신, demonstration에서 얻은 configuration 정보를 canonical trajectory와 null-space/configuration prior로 사용하고 이후 정책 학습은 병렬 시뮬레이션에서 수행한다.

다만 기존에 정리된 `NS_demo 98.6% vs NS_naive 0.0%` 및 `1.8× 학습 가속`은 최신 실험 설정에서 그대로 인용할 수 없다. 최신 실험 레지스트리에서 기존 비교의 플래그 혼입과 자산 변경 문제가 확인됐고, 새 USD 재평가에서는 결과 방향도 반대로 나타났다.

따라서 현재 미팅에서는 다음과 같이 구분해야 한다.

1. **확정 가능:** 실데이터 규모 10회·109.2초, 100 Hz, 10,920 frames
2. **방법 설명 가능:** 실데이터를 configuration prior로 압축하여 RL 탐색을 보조
3. **재검증 필요:** demo prior의 성능 향상률과 학습속도 향상률
4. **신규 핵심 실험:** demo 수를 0·1·2·5·10개로 변화시킨 data-scaling 비교

## PPT 전환용 구성안

이 절은 추후 발표자료에 그대로 옮길 수 있도록 슬라이드 단위로 구성했다. 현재 확정되지 않은 값은 임의로 채우지 않고 `TBD`로 표시한다.

### Slide 1 — 문제와 접근

**제목:** 최소 실데이터 기반 휴머노이드 양팔 정밀조작 학습

**핵심 메시지:**

> 소량의 실로봇 demonstration을 직접 모방하는 대신 configuration prior로 압축하고, 대규모 정책 학습은 시뮬레이션에서 수행한다.

**숫자 카드:**

| 실로봇 Demo | 실데이터 길이 | 원시 Frames | 압축 표현 |
|---:|---:|---:|---:|
| **10회** | **109.2초** | **10,920** | **21-point `R(β)`** |

**권장 그림:** `Real demos → Configuration prior → Simulation RL → Real robot` 파이프라인 다이어그램

**발표자 주석:** 현재 109.2초는 녹화 trajectory 길이이며 setup/reset을 포함한 총 작업시간은 아니다.

### Slide 2 — 무엇을 정량화할 것인가

**제목:** 실데이터 의존도 저감의 정량지표

| 평가 질문 | 핵심 지표 |
|---|---|
| 목표 성능에 실데이터가 얼마나 필요한가? | 최소 demo 수·수집시간 |
| 적은 데이터에서도 정확한가? | Complete-transfer, ≥75% transfer |
| 학습이 빨라지는가? | 목표 성능 도달 transitions·GPU hours |
| 성능 향상이 안전성을 해치지 않는가? | Spill, failure tail, collision |

**헤드라인 KPI:**

> Complete-transfer 목표를 달성하는 데 필요한 실데이터 수집시간

**발표자 주석:** binary success는 방법 간 품질 차이를 충분히 구분하지 못하므로 complete-transfer를 우선한다.

### Slide 3 — 현재 데이터와 검증 상태

**제목:** 기존 결과는 재현성 검증 후 확정

| 데이터 버전 | NS_demo 성공률 | NS_naive 성공률 | 판정 |
|---|---:|---:|---|
| 구 자산 | 98.6% | 0.0% | 플래그 혼입 가능성 |
| 신규 USD | 81.8% | 98.6% | 기존 주장과 방향 충돌 |

**핵심 메시지:**

> 자산과 ablation 설정에 따라 결과가 역전되어, 기존 향상률은 확정 성과가 아니라 재검증 대상으로 관리한다.

**권장 시각화:** 두 데이터 버전의 NS_demo/NS_naive 성공률을 묶은 bar chart. 구 결과는 회색 또는 점선으로 표시하고 `invalidated/preliminary` 라벨을 붙인다.

**발표자 주석:** 문제를 숨기기보다 동일 자산·동일 초기상태·동일 3-flag mechanism 조건으로 재실험하고 있다는 점을 강조한다.

### Slide 4 — 핵심 실험 설계

**제목:** Real-data scaling으로 의존도 직접 측정

| Demo 수 | 0 | 1 | 2 | 5 | 10 |
|---:|---:|---:|---:|---:|---:|
| 실데이터 시간 | 0초 | 약 11초 | 약 22초 | 약 55초 | 109.2초 |
| Complete-transfer | TBD | TBD | TBD | TBD | TBD |
| T70 transitions | TBD | TBD | TBD | TBD | TBD |
| Spill | TBD | TBD | TBD | TBD | TBD |

**실험 통제:** 조건당 ≥3 seeds, 동일 simulator transitions, 동일 evaluation bank, 1,024-env deterministic evaluation

**권장 그래프:**

- X축: 실데이터 시간 또는 demo 수
- Y축 1: complete-transfer rate
- Y축 2: 목표 성능 도달 transitions
- Seed 평균 선 + 표준편차 또는 95% CI band

### Slide 5 — 기대 산출물과 과제 기여

**제목:** 최종적으로 제시할 정량성과

1. 목표 조작성능 달성에 필요한 실데이터를 `TBD분 → TBD분`으로 절감
2. 동일 실데이터 예산에서 complete-transfer를 `TBD%p` 향상
3. 목표 성능 도달 simulator transitions 또는 GPU hours를 `TBD×` 단축
4. Spill과 failure-tail을 악화시키지 않는 안전성 확인

**최종 주장 템플릿:**

> 제안 방법은 실데이터 **[D_base]분 대비 [D_ours]분**만으로 동일한 complete-transfer 성능을 달성하여 실데이터 요구량을 **[R]% 절감**했으며, 목표 성능 도달 학습량을 **[S]배 단축**했다.

단, 위 문장의 대괄호 값은 real-data scaling 및 기준 방법 실험이 완료된 뒤에만 입력한다.

## 2. 현재 실데이터 사용량

### 2.1 원시 demonstration 규모

| Demo | Frames | 기록시간 |
|---|---:|---:|
| a11 | 1,130 | 11.30초 |
| a12 | 1,165 | 11.65초 |
| a13 | 1,156 | 11.56초 |
| a14 | 1,134 | 11.34초 |
| a15 | 883 | 8.83초 |
| a16 | 1,156 | 11.56초 |
| a17 | 915 | 9.15초 |
| a18 | 1,148 | 11.48초 |
| a19 | 1,115 | 11.15초 |
| a20 | 1,118 | 11.18초 |
| **합계** | **10,920** | **109.20초 (1.82분)** |

- 샘플링 주파수: 100 Hz
- 사용 파일: `datasets/pour_v1_a11.hdf5` ~ `pour_v1_a20.hdf5`
- 원시 데이터에서 21-point canonical `R(β)` arm trajectory를 구성한다.
- `R(β)`는 7-DoF arm configuration을 `β∈[0,1]`의 21개 지점으로 표현한다.

### 2.2 과제 관점의 현재 표현

> 약 109초의 실로봇 demonstration을 저차원 configuration prior로 변환하고, 이후 자율조작 정책 학습과 반복 평가는 대규모 병렬 시뮬레이션에서 수행한다.

현재는 비교 기준이 없으므로 “실데이터를 몇 % 절감했다”는 감소율을 확정할 수 없다. 감소율을 주장하려면 end-to-end imitation/BC 또는 full-demo RL 같은 명시적인 기준 방법과 동일 성능 조건에서 비교해야 한다.

## 3. 기존 성능 데이터

### 3.1 구 자산 결과 — 참고용, 확정 성과 인용 금지

| 지표 | NS_demo | NS_naive | Full | Joint-space |
|---|---:|---:|---:|---:|
| 보고 성공률 | 98.6% | 0.0% | 96.4% | 97.7% |
| 평균 이송량 | 17.56/20 | 0.00/20 | 19.21/20 | 15.58/20 |
| ≥75% transfer | 90.2% | 0.0% | 95.6% | 71.1% |
| ≥90% transfer | 58.2% | 0.0% | 93.8% | 23.2% |
| Complete transfer | 28.3% | 0.0% | 90.2% | 5.3% |
| Failure tail (<25%) | 0.7% | 100.0% | 2.6% | 0.9% |
| Spill | 1.5% | 0.0% | 2.5% | 3.6% |

구 결과에서 NS_demo와 joint-space를 비교하면 다음과 같다.

- ≥75% transfer: **+19.1%p**
- Complete transfer: **+23.0%p**
- 평균 이송량: **+1.98 bead**, 약 **+12.7%**
- Spill: **−2.1%p**

이 결과는 지표 선택의 참고자료로는 유용하지만, 현재 코드·USD 기준의 재현 결과가 아니므로 과제의 확정 정량성과로 사용하지 않는다.

### 3.2 신규 USD 재평가 — 기존 주장과 충돌

| 신규 재평가 | NS_demo | NS_naive | NS_demo − NS_naive |
|---|---:|---:|---:|
| 성공률 | 81.8% | 98.6% | −16.8%p |
| 평균 이송량 | 13.04/20 | 16.14/20 | −3.10 bead |
| 평균 spill | 2.5% | 1.9% | +0.6%p |

신규 결과는 demo prior가 우수하다는 기존 결과를 지지하지 않는다. 또한 최신 실험 레지스트리는 과거 `nullspace_baseline` 단독 변경이 실제 메커니즘을 완전히 끄지 못하는 dead-code/confound 문제를 명시한다.

현재 메커니즘 비교에서는 다음 3개 플래그를 하나의 단위로 함께 제어해야 한다.

- `nullspace_baseline`
- `pour_orient_release`
- `pour_bfull_nullspace`

따라서 현재의 올바른 결론은 **성능 향상률 미확정, 통제 실험 재수행 필요**이다.

## 4. 기존 학습속도 데이터의 상태

구 TFEvents에서 `log/adr_ep_success_rate ≥ 0.70` 최초 도달 iteration은 다음과 같이 정리돼 있었다.

| 조건 | 0.70 도달 iteration | 구 해석 |
|---|---:|---|
| Full | 1,597 | 가장 빠름 |
| NS_demo | 3,396 | JS 대비 1.8× 빠름 |
| Joint-space | 6,069 | 느린 고성능 도달 |
| NS_naive | 미도달 | 최대 0.164 |

그러나 최신 레지스트리는 이 표를 그대로 인용하지 말라고 명시한다. 새 자산에서 NS_naive가 98.6%를 기록하여 “NS_naive가 구조적 천장에 막혔다”는 과거 해석이 더 이상 성립하지 않기 때문이다.

`adr_ep_success_rate` 자체도 최종 성능이 아닌 관대한 학습 proxy이므로 다음 용도로만 사용한다.

- 동일 자산·동일 설정에서 조건별 학습속도 비교
- 목표 임계값 도달 iteration 계산
- learning curve AUC 계산

최종 조작 성능 주장은 deterministic evaluation의 transfer distribution으로 판단한다.

## 5. 과제용 권장 핵심 KPI

### KPI 1 — 실데이터 요구량

동일한 목표 성능을 달성하는 데 필요한 실제 demonstration 수와 수집시간을 측정한다.

\[
D_{target}=\min\{D:\text{CompleteTransfer}(D)\ge\tau\}
\]

- 단위: demonstrations, seconds 또는 minutes
- 예시 목표: complete-transfer 80% 또는 ≥75%-transfer 90%
- 이 지표가 과제의 “실데이터 의존도 저감”과 가장 직접적으로 연결된다.

### KPI 2 — 고정 실데이터 예산 성능

\[
P(D)=\text{CompleteTransferRate after training with }D\text{ demos}
\]

demo 수가 작을 때 성능이 얼마나 유지되는지 평가한다. 단순 binary success보다 complete-transfer 또는 ≥75%-transfer가 구분력이 크다.

### KPI 3 — 학습 sample efficiency

\[
T_{70}=\min\{i:\operatorname{EMA}(ADR\ success_i)\ge0.70\}
\]

\[
\text{Speed-up}(D)=\frac{T_{70}(0\ demo)}{T_{70}(D\ demos)}
\]

- iteration뿐 아니라 simulator transitions와 wall-clock GPU hours를 같이 보고한다.
- 서로 다른 병렬 환경 수에서 iteration만 비교하면 실제 sample efficiency가 왜곡될 수 있다.

### KPI 4 — 실데이터 효율

\[
\text{RealDataEfficiency}(D)
=\frac{\text{CompleteTransferRate}(D)}{\text{Real-data minutes}(D)}
\]

보조지표로 사용할 수 있으나, demo가 0개일 때 정의되지 않고 작은 분모에 민감하므로 KPI 1·2와 함께 제시한다.

### Guardrail — 안전성과 정밀도

- Spill rate
- Failure tail: transfer <25%
- Mean/median transfer ratio
- ≥75%, ≥90%, complete-transfer rate
- Mouth-to-target XY error
- Collision 및 grasp-loss rate

## 6. 우선 수행할 실험

### 6.1 Real-data scaling experiment

실데이터 양만 변화시키고 나머지 조건을 동일하게 유지한다.

| Demo 수 | 예상 실데이터 시간 | Seed 수 | 최종 성공률 | ≥75% | Complete | Spill | T70 | GPU hours |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0초 | ≥3 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| 1 | 약 11초 | ≥3 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| 2 | 약 22초 | ≥3 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| 5 | 약 55초 | ≥3 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| 10 | 109.2초 | ≥3 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |

권장 통제 조건:

- 동일 코드와 USD asset
- 동일 초기 상태 bank
- 동일 reward, curriculum, observation, action space
- 동일 환경 수와 총 simulator transitions
- condition당 최소 3 seeds
- seed별 동일 deterministic evaluation bank
- evaluation: 1,024 environments, 동일 episode horizon

### 6.2 Baseline 정의

최소 다음 세 조건을 비교한다.

1. **0-demo RL:** 실데이터 prior 없이 동일한 RL 구조
2. **N-demo prior + RL:** 제안 방법
3. **Full-demo BC/IL 또는 demo-reward baseline:** 실데이터를 직접 학습하는 기준 방법

가능하면 다음 구조적 ablation도 포함한다.

- task-space + mechanism OFF
- task-space + mechanism ON
- joint-space baseline
- Full system

### 6.3 결과 보고 방식

각 셀은 단일 seed 값이 아니라 다음 형식으로 보고한다.

> mean ± standard deviation across seeds, 95% bootstrap CI across evaluation episodes

demo subset의 선택 편향을 줄이려면 `D=1,2,5`에서 서로 다른 demo subset도 반복한다.

## 7. 미팅 발표용 메시지

### 한 문장 버전

> 현재 시스템은 실제 로봇 demonstration 10회, 총 109.2초를 configuration prior로 사용하고 나머지 정책 학습을 시뮬레이션에서 수행한다. 기존 성능향상 수치는 실험 플래그 혼입이 확인되어 재검증 중이며, 0·1·2·5·10-demo scaling 실험으로 목표 성능당 실데이터 요구량과 학습속도를 정량화할 계획이다.

### 발표 시 강조할 지표

1. 목표 complete-transfer 성능을 달성하는 데 필요한 실데이터 시간
2. demo 수 감소에 따른 complete-transfer 성능 유지율
3. 목표 성공률 도달까지의 simulator transitions 및 GPU hours
4. spill과 failure-tail을 포함한 안전성 guardrail

### 피해야 할 표현

- “실데이터로 성공률이 0%에서 98.6%로 향상됐다.”
- “실데이터 prior가 학습을 1.8배 가속했다.”
- “109초만으로 실데이터 의존도를 몇 % 절감했다.”

위 표현은 현재 자산과 통제 설정에서 재현되기 전까지 확정 성과로 사용하지 않는다.

## 8. 근거 자료

### Notion

- [Bimanual Precision Pouring — RA-L](https://app.notion.com/p/3a193e676f2181f19730d705bb7fbaa1?pvs=204)
- [Manuscript Draft — Bimanual Precision Pouring](https://app.notion.com/p/3a593e676f218141b104cdc16138bbc0?pvs=204)

### Local evidence

- `docs/experiments/registry.md` — 최신 실험 설계와 구 자산 인용 금지 경고
- `docs/eval/ral_pour_master_results.csv` — 구 마스터 평가 결과
- `docs/eval/sample_efficiency.md` — 구 학습속도 분석
- `docs/eval/newusd/eval_NS_demo_newUSD.md` — 신규 USD NS_demo 평가
- `docs/eval/newusd/eval_NS_naive_newUSD.md` — 신규 USD NS_naive 평가
- `docs/eval/pour_distribution_metrics_consolidated.md` — transfer distribution
- `docs/demo_pour_joint_analysis.md` — a11–a20 실제 demonstration 분석
- `source/openarm/openarm/tesollo/both/pour_sensor/r_beta_trajectory.py` — 10 demos에서 생성한 canonical trajectory
- `source/openarm/openarm/tesollo/both/pour_sensor/pour_right_env_cfg.py` — demo 입력 및 실험 설정

## 9. 데이터 품질 메모

- **High severity:** 구 master CSV와 신규 USD 결과가 핵심 비교에서 서로 충돌한다.
- **High severity:** 과거 ablation은 실제 mechanism OFF 조건이 아니었을 가능성이 있다.
- **High severity:** 구 sample-efficiency 표는 최신 레지스트리에서 명시적으로 인용 금지됐다.
- **Medium severity:** 기존 핵심 조건은 단일 seed 중심이므로 seed variance가 부족하다.
- **Medium severity:** episode 수는 충분하지만 동일 evaluation bank 여부를 실험별로 다시 확인해야 한다.
- **Medium severity:** binary success는 JS와 Full을 충분히 구분하지 못하므로 complete-transfer와 분포지표가 필요하다.
- **Open gap:** 실데이터 수집에 필요한 사람 작업시간, setup/reset 시간은 아직 계측되지 않았다. 현재 109.2초는 녹화된 trajectory 시간이며 총 현장 인건시간은 아니다.
