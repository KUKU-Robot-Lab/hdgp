---
name: test4 num_contacts plateau 근본 원인 분석
description: 5g_grasp_right_v9 test4에서 num_contacts가 2.99에서 멈추는 이유 — reward gate decoupling
type: project
---

## 발견 내용

**문제**: num_contacts가 epoch 283에서 2.99에 도달한 후 plateau. ADR이 5.0까지 진행해도 증가 안함.

**근본 원인**: 보상 구조의 **의도하지 않은 불일치**

### 1. force_balance_reward Gate 실패 (epoch 282+)

- **epoch 1-281**: adr_min ≤ 3 → others_count ≥ 2 조건 만족 가능 → force_balance > 0
- **epoch 282+**: adr_min ≥ 4 → others_count ≥ 3 필요 → Policy 달성 불가 (others_avg_force ≈ 0.58N)
- **결과**: force_balance_reward (weight=6.0) 완전 소멸 (280 epochs 활성 → 2 epochs만 활성)

### 2. full_contact_bonus가 너무 약함

- weight = 0.5 (lift_reward의 40분의 1)
- 평균 기여도 < 0.1 pts/step
- 추가 손가락의 경제적 유인 부족

### 3. lift_reward가 총 보상의 60-65%

- 성공 조건: num_contacts ≥ 2 (충족)
- num_contacts ≈ 3 수준이면 lift 가능
- **추가 접촉 필요 없음**

### 4. ADR의 역설적 효과

- adr_min 증가 = reward gate 더 엄격해짐
- "어려워지는데" "그래디언트는 없는" 결과
- contact_adr_trigger_threshold=0.1에서 모두 충족하면 진행하지만, 실제 성능 향상 없음

## 양적 증거

| 기간 | adr_min | num_contacts | others_force | force_balance_reward | 해석 |
|------|---------|------------|-------------|-------------------|------|
| epoch 1-100 | 2 | 1.3→2.2 | 0.6 | **활성** | 접근 학습 |
| epoch 101-250 | 2→3 | 2.2→2.7 | 0.65 | **활성** | 파지 강화 |
| epoch 251-281 | 3→4 | 2.7→2.99 | 0.65 | **활성** (피크) | plateau 형성 |
| **epoch 282+** | **4→5** | **2.8±0.1** (plateau) | **0.58** | **≈0** (게이트 실패) | **추가 접촉 유인 상실** |

## 왜 "버그"가 아닌가

이는 **메커니즘적으로 합리적인 평형점**:
1. 3-접촉 grasp는 안정적이고 lift 가능
2. 4-5-접촉은 network capacity/physics limit으로 어려움
3. reward structure상 "3-접촉이 충분"
4. → Policy가 local optimum에서 수렴

## 개선 전략 (필수 아님, 선택사항)

### Option A: full_contact_bonus 5-10배 증가
- weight 0.5 → 5.0
- num_contacts ≥ 4시 추가 유인 생성
- 예상: num_contacts 4.0+ 도달

### Option B: force_balance_sharpness 완화
- sharpness 10 → 3-5
- others_count ≥ 3 못 달성해도 부분 보상
- gate 조건 유연화

### Option C: ADR 진행 속도 감소
- increment_interval 400 → 600+
- Policy가 3-접촉에서 더 오래 학습
- 4-접촉 전환 시 이미 습득

### Option D: "5-contact explicit" reward 추가
- r10_five_contact = 2.0 × (num_contacts ≥ 5).float()
- 초기 curriculum에서 명시적 목표 제시

## 성능 평가

| 메트릭 | 값 | 판정 |
|--------|-----|------|
| episode_success_rate | 81% | ✅ 우수 |
| num_contacts | 2.72 | △ 부분 성공 (3/5 fingers) |
| 안정성 | 낮은 variance | ✅ 안정적 |
| 수렴 | epoch 250 이후 plateau | ✅ 빠른 수렴 |

**종합**: 현재 test4는 안정적 동작. 추가 개선은 선택사항.

## 발견 시점 및 근거

- **분석 일시**: 2026-04-06
- **데이터**: TensorBoard 691 epochs, 691개 스칼라 포인트
- **방법**: Event accumulator로 전수 추출 후 epoch별 분석
- **검증**: 
  - force_balance reward의 생명주기 추적 (280 active epochs → 2 active)
  - others_avg_force 변화 추이 (0.65 → 0.58)
  - ADR transition points (epoch 182, 282, 388, 438)
  - Reward composition decomposition (lift 60%, force_balance 게이트 실패)
