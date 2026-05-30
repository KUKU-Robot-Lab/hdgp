# Plan: IL+RL 조화 구조 개선 — 5g_pour_right_v6

## Summary
v6의 목표는 "IL로 기본 모션 prior 확립 → RL로 task fine-tuning"이지만,
현재 `weight_pour_xy`(이동 보상)가 BC 손실과 gradient 방향으로 충돌하고,
chunk BC backward에서 LSTM gradient 누적 버그가 있으며,
BC decay 속도가 너무 빨라 RL이 성공하기 전에 IL prior가 소멸된다.
이 계획은 최소한의 수정으로 IL/RL 구조 충돌을 해소하고,
diffusion_policy·UMI에서 검증된 패턴을 반영하여 학습 안정성을 높인다.

## User Story
As a reinforcement learning engineer,
I want IL-based motion prior and RL task reward to cooperate without gradient conflict,
So that the agent first learns the basic pouring motion from demos, then optimizes task success with RL.

## Problem → Solution
**현재 상태**: RL 보상에 `weight_pour_xy`(이동 유도)가 있어 BC 손실의 이동 방향과 충돌.
Chunk BC backward에서 LSTM에 orphan gradient 발생.
Demo BC decay=1000 epoch로 빠르게 소멸 → RL 탐색 없으면 IL prior 없는 상태에서 학습.

**목표 상태**: RL 보상은 순수 task outcome만 (bead capture + spill penalty + success).
BC가 "이렇게 움직여라" 담당, RL은 "결과가 좋냐" 담당.
Chunk BC backward는 chunk_head만 학습, LSTM은 PPO만 학습.
Demo BC가 더 오래 유지되어 RL 탐색 초기에도 motion prior 보존.

## Metadata
- **Complexity**: Medium
- **Source PRD**: N/A
- **PRD Phase**: N/A
- **Estimated Files**: 3 (env_cfg.py, pour_chunk_bc_agent.py, yaml)

---

## UX Design

### Before
```
┌────────────────────────────────────────────────────────┐
│ RL reward:                                             │
│   r_pour_xy = 8.0 × exp(-20 × mouth_xy²)             │
│   + r_capture_spill (bead outcome)                    │
│                                                        │
│ BC loss (demo):                                        │
│   λ_demo × NLL(policy | demo_action)                  │
│                                                        │
│ 충돌: r_pour_xy가 "cup을 target 위로 가져가라"         │
│       BC가 "demo 모션을 따라라"                         │
│   → 두 gradient가 다른 방향을 가리키면 상충            │
│                                                        │
│ Chunk BC backward:                                     │
│   total_chunk.backward()                              │
│   → LSTM에도 gradient 누적 (orphan, 미사용)            │
│                                                        │
│ Demo BC decay: epoch 1000에서 0                        │
│   → 1000 epoch 이후 IL prior 없음                      │
└────────────────────────────────────────────────────────┘
```

### After
```
┌────────────────────────────────────────────────────────┐
│ RL reward (task outcome only):                         │
│   r_capture_spill (bead capture + spill penalty)       │
│   + r_success                                          │
│   ← 이동 유도 보상 제거 (BC가 담당)                    │
│                                                        │
│ BC loss (demo):                                        │
│   λ_demo × NLL(policy | demo_action)                  │
│   decay: epoch 0~500 warm → 3000 decay                │
│   (RL 탐색 기간 동안 motion prior 유지)                 │
│                                                        │
│ Chunk BC backward (수정):                              │
│   chunk_pred = chunk_head(lstm_feat.detach())          │
│   → chunk_head만 학습, LSTM은 PPO 전용                 │
│                                                        │
│ 결과: IL이 "how to move" 담당                          │
│       RL이 "did it work" 담당 (충돌 없음)              │
└────────────────────────────────────────────────────────┘
```

### Interaction Changes

| 변경 지점 | Before | After | 비고 |
|---------|--------|-------|------|
| `weight_pour_xy` | 8.0 | 0.0 | 이동 유도 제거 |
| `weight_dist_to_target` | 5.0 | 0.0 | BC가 transport 담당 |
| `real_demo_bc_decay_epochs` | 1000 | 3000 | BC 더 오래 유지 |
| `chunk BC detach` | lstm_feat 직접 사용 | lstm_feat.detach() | bug fix |
| `chunk_bc_weight_final` | 1.0 | 0.3 | 학습 수렴 후 약한 정규화 |

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---------|------|-------|-----|
| P0 | `5g_pour_right_v6/pour_chunk_bc_agent.py` | 104~153 | chunk BC backward bug 위치 |
| P0 | `5g_pour_right_v6/reward_terms.py` | 전체 | 현재 reward 구조 (r_pour_xy 제거 대상) |
| P0 | `5g_pour_right_v6/pour_right_env_cfg.py` | 260~320 | weight 파라미터 설정 |
| P1 | `5g_pour_right_v6/lstm_bc_agent.py` | 242~378 | PPO+BC 통합 backward 구조 |
| P1 | `config/agents/rl_games_ppo_chunk_bc_cfg.yaml` | 전체 | BC hyperparameter 설정 |
| P2 | `5g_pour_right_v3/pour_right_env_cfg.py` | 248~290 | 안정 학습 기준 reward weight 참조 |

## External Documentation & Research

| Topic | Source | Key Takeaway |
|-------|--------|-------------|
| Action Chunking (ACT) | Chi et al. 2023, Zhao et al. 2023 | Chunk prediction forces temporal consistency; temporal ensemble reduces variance |
| IL+RL 조화 (DAPG) | Rajeswaran et al. 2017 | BC loss를 gradient regularizer로, RL reward는 task outcome으로 분리 |
| BC-regularized RL | Nair et al. (AWAC) 2020 | 데모 분포에서 RL 탐색 시작, BC는 prior 역할 |
| Diffusion Policy (obs history) | Chi et al. 2023 | T_obs=2 과거 obs 스태킹으로 explicit temporal context 제공 |
| UMI (relative action) | Chi et al. 2024 | Cup-local frame 상대 action이 일반화 향상 |

**WEB SEARCH 핵심 발견 (합성):**
```
KEY_INSIGHT: BC loss + RL reward 동시 사용 시 "공통 목적(task 완수)"이 없으면
             두 gradient가 상충. 해결책은 BC=motion prior, RL=outcome signal로 역할 분리.
APPLIES_TO: weight_pour_xy 제거, weight_dist_to_target 제거
GOTCHA: IL prior 너무 빨리 끊으면 RL이 랜덤 탐색에 빠짐. BC decay를 RL 성공률이
        오를 때까지 유지해야 함 (ADR 연계 권장).

KEY_INSIGHT: Chunk BC backward가 LSTM까지 gradient를 전달하면 PPO와 chunk BC가
             같은 파라미터를 서로 다른 목적으로 최적화 → 학습 불안정.
APPLIES_TO: pour_chunk_bc_agent.py chunk BC backward
GOTCHA: lstm_feat.detach()로 LSTM gradient 차단 필수.

KEY_INSIGHT: diffusion_policy의 temporal consistency는 action chunk prediction이
             연속적 obs에서 일관된 chunk를 예측할 때 달성됨.
             현재 v6 chunk BC는 zero-init LSTM을 사용해 이 일관성을 깨트림.
APPLIES_TO: 장기적 개선: warm-init LSTM for BC training (obs warmup K steps)
GOTCHA: 단기적으로는 obs_delta를 chunk BC 입력에 추가하는 것이 더 단순.
```

---

## Patterns to Mirror

### REWARD_WEIGHT_ZERO_PATTERN
```python
# SOURCE: pour_right_env_cfg.py:298-299
weight_action_rate_palm: float = 0.0    # diagnostic only
weight_action_rate_finger: float = 0.0  # diagnostic only
```
→ 진단용 weight를 0.0으로 유지하는 패턴. 이동 유도 weight도 같은 방식.

### DETACH_GRADIENT_PATTERN
```python
# SOURCE: pour_right_env_cfg.py (v5 이전 r_tilt.detach() 패턴)
# Gradient 흐름 차단으로 특정 파라미터만 업데이트
r_pour_dist = ... * r_tilt.detach()  # tilt에서 pour_dist gradient 차단
```
→ `lstm_feat.detach()`에 동일 패턴 적용.

### BC_WEIGHT_SCHEDULE_PATTERN
```python
# SOURCE: lstm_bc_agent.py:36-57
def _bc_weight(epoch, warmup_epochs, decay_epochs, weight_init, weight_final):
    # warmup → plateau → decay
    ...
```
→ `real_demo_bc_decay_epochs` 값만 변경.

### ENV_CFG_WEIGHT_PATTERN
```python
# SOURCE: pour_right_env_cfg.py:260-264
weight_grasp_maintain: float = 0.0
weight_contact_maintain: float = 0.0
weight_force_balance: float = 0.0
weight_finger_curl: float = 0.0
```
→ IL/RL 구조에서 필요 없는 보상은 0.0 처리.

---

## Files to Change

| File | Action | Justification |
|------|--------|---------------|
| `pour_right_env_cfg.py` | UPDATE | `weight_pour_xy`=0, `weight_dist_to_target`=0 |
| `pour_chunk_bc_agent.py` | UPDATE | `lstm_feat.detach()` bug fix |
| `config/agents/rl_games_ppo_chunk_bc_cfg.yaml` | UPDATE | `real_demo_bc_decay_epochs` 증가 |

## NOT Building

- 새 reward term 추가 없음 (기존 reward_terms.py 유지)
- obs 차원 변경 없음 (LSTM hidden state 재학습 불가)
- 새 IL 알고리즘 (Diffusion Policy 직접 포팅) — 단계가 너무 큼
- warmstart 구조 변경 없음

---

## Step-by-Step Tasks

### Task 1: RL 이동 보상 제거 (IL/RL 충돌 해소)
- **ACTION**: `pour_right_env_cfg.py`에서 이동 유도 weight를 0으로 설정
- **IMPLEMENT**:
  ```python
  # 이동 유도 보상 → IL(BC)이 담당
  weight_pour_xy: float = 0.0        # 8.0 → 0.0: BC가 XY alignment 담당
  weight_dist_to_target: float = 0.0 # 5.0 → 0.0: BC가 transport 담당
  ```
  reward_terms.py의 `r_pour_xy`는 유지하되 weight=0으로 비활성화.
- **MIRROR**: ENV_CFG_WEIGHT_PATTERN
- **IMPORTS**: 불필요
- **GOTCHA**: `pour_right_env.py`에서 `compute_simple_pour_reward()` 호출 시
  `xy_weight=cfg.weight_pour_xy`로 전달하는지 확인. 0이면 자동 비활성화.
- **VALIDATE**:
  - `grep -n "weight_pour_xy" pour_right_env.py` → cfg.weight_pour_xy로 전달되는지 확인
  - 변경 후 reward 합계: `r_capture_spill + r_success` 만 남는지 확인

### Task 2: Chunk BC backward lstm_feat detach (Bug Fix)
- **ACTION**: `pour_chunk_bc_agent.py`의 chunk BC 계산에서 lstm_feat detach 추가
- **IMPLEMENT**:
  ```python
  # 기존 (버그):
  lstm_feat = self._lstm_feat  # gradient path intact → LSTM에도 전달됨
  chunk_pred = self.chunk_head(lstm_feat)
  
  # 수정 후:
  lstm_feat = self._lstm_feat.detach()  # chunk BC gradient는 chunk_head만 업데이트
  chunk_pred = self.chunk_head(lstm_feat)
  ```
  위치: `pour_chunk_bc_agent.py` line 137-139
- **MIRROR**: DETACH_GRADIENT_PATTERN
- **IMPORTS**: 불필요 (torch는 이미 import됨)
- **GOTCHA**: `lstm_feat`는 이미 `self._lstm_feat`에 저장됨. detach()는
  `.detach()`가 아닌 `.detach().clone()`도 가능하나 `.detach()` 충분.
- **VALIDATE**:
  - 수정 후 `total_chunk.backward()` 시 LSTM 파라미터에 gradient가 없는지 확인:
    ```python
    total_chunk.backward()
    assert all(p.grad is None for p in self.model.a2c_network.rnn.parameters())
    ```

### Task 3: Demo BC decay 연장 (IL prior 유지)
- **ACTION**: `rl_games_ppo_chunk_bc_cfg.yaml`에서 demo BC decay 연장
- **IMPLEMENT**:
  ```yaml
  # 기존:
  real_demo_bc_decay_epochs: 1000
  real_demo_bc_weight_init: 10.0
  real_demo_bc_weight_final: 0.0
  
  # 수정 후:
  real_demo_bc_warmup_epochs: 200   # 0→200: 초반 BC 안정적 증가
  real_demo_bc_decay_epochs: 3000   # 1000→3000: RL 성공률 오를 때까지 IL prior 유지
  real_demo_bc_weight_init: 10.0    # 유지
  real_demo_bc_weight_final: 0.5    # 0.0→0.5: 완전 소멸 방지 (약한 정규화 유지)
  ```
  schedule: 0~200 warm(0→10), 200~3200 decay(10→0.5), 3200+ 0.5 유지
- **MIRROR**: BC_WEIGHT_SCHEDULE_PATTERN
- **IMPORTS**: 불필요
- **GOTCHA**: `real_demo_bc_weight_final=0.5`로 남기면 영구 BC 신호가 존재.
  만약 학습 후반 BC가 RL 탐색을 방해하면 0.0으로 내려야 함.
  초기에는 0.5로 시작해서 TFEvents에서 `bc/loss_demo` 모니터링.
- **VALIDATE**:
  - `bc/weight_demo` 그래프가 epoch 200에서 peak, epoch 3200에서 0.5 도달 확인
  - `bc/loss_demo` > 0으로 유지되는지 (데모 BC 활성 확인)

### Task 4: Chunk BC weight 조정 (Step BC와 균형)
- **ACTION**: Chunk BC가 Step BC와 경합하지 않도록 weight 조정
- **IMPLEMENT**:
  ```yaml
  # 기존:
  chunk_bc_weight_init: 10.0
  chunk_bc_weight_final: 1.0
  chunk_bc_decay_epochs: 5000
  
  # 수정 후:
  chunk_bc_weight_init: 5.0       # 10→5: Step BC(weight=10)와 초반 균형
  chunk_bc_weight_final: 0.3      # 1.0→0.3: 수렴 후 약한 temporal consistency만
  chunk_bc_decay_epochs: 3000     # 5000→3000: Demo BC와 같은 시간축
  ```
- **MIRROR**: BC_WEIGHT_SCHEDULE_PATTERN
- **GOTCHA**: chunk BC weight가 너무 낮으면 LSTM feature에 temporal prior가 약해짐.
  `bc/chunk_loss`가 step BC `bc/loss_demo`와 비슷한 크기가 되도록 tuning.
- **VALIDATE**:
  - TFEvents에서 `bc/chunk_loss`와 `bc/loss_demo`의 비율이 0.1~10 범위인지 확인

---

## Testing Strategy

### 변경 전 상태 기록
```bash
# 수정 전 반드시 실행
python3 /home/user/rl_ws/hdgp/scripts/tools/record_test_snapshot.py \
  --task 5g_pour_right_v6 --test test1
```

### 핵심 검증 지표 (TFEvents)

| 지표 | 기대값 (1000 epoch 기준) | 의미 |
|-----|---------------------|------|
| `bc/loss_demo` | > 0 (소멸되지 않음) | Demo BC 활성 유지 |
| `bc/weight_demo` | 0.5~10 범위 | BC 스케줄 정상 |
| `Episode/log/bead_in_target` | > 0 (최소 몇 에피소드) | BC가 pour 위치로 유도 |
| `Episode/log/cup_center_xy_dist` | 감소 추세 | BC가 transport 학습 |
| `Episode/log/directional_tilt_cos` | < 0 방향으로 감소 | tilt 학습 |
| `info/kl` | < 0.02 (안정) | PPO+BC 학습 안정 |

### Edge Cases
- [ ] weight_pour_xy=0이 되어도 r_capture_spill은 여전히 ρ gate 안에 있어야 함
- [ ] chunk BC detach 후 chunk_head grad norm이 발생하는지 확인
- [ ] demo BC 오래 유지 시 demo local min에 빠지지 않는지 (cup_center_xy_dist 모니터링)

---

## Validation Commands

### Static Analysis
```bash
cd /home/user/rl_ws/hdgp/source/openarm
python3 -c "from openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.'5g_pour_right_v6' import pour_right_env_cfg; print('OK')"
```
EXPECT: Import 에러 없음

### Unit Tests
```bash
cd /home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_pour_right_v6
python3 -m pytest tests/ -x -v 2>&1 | tail -20
```
EXPECT: 기존 테스트 통과 (obs 차원 변경 없으므로)

### Manual Validation
- [ ] Task 1: `grep weight_pour_xy pour_right_env_cfg.py` → `0.0` 확인
- [ ] Task 2: `grep -A2 "lstm_feat" pour_chunk_bc_agent.py` → `.detach()` 확인
- [ ] Task 3: yaml에서 `real_demo_bc_decay_epochs: 3000` 확인
- [ ] 학습 실행 후 1000 epoch: `bc/weight_demo` > 0.5 확인

---

## Acceptance Criteria
- [ ] `weight_pour_xy = 0.0` (이동 유도 보상 제거)
- [ ] `weight_dist_to_target = 0.0` (transport 보상 제거)
- [ ] Chunk BC에서 `lstm_feat.detach()` 적용
- [ ] `real_demo_bc_decay_epochs = 3000`
- [ ] 기존 테스트 통과
- [ ] 학습 초기 `bc/loss_demo` > 0 (BC 활성 확인)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 이동 보상 제거 후 초반 탐색 부족 | Medium | High | BC가 motion prior를 충분히 제공하는지 `cup_center_xy_dist` 모니터링 |
| Demo BC 오래 유지 시 demo local min | Medium | Medium | `weight_final=0.5`는 낮아서 local min 위험 낮음; `bc/loss_demo` plateau 확인 |
| Chunk BC detach 후 LSTM feature 품질 저하 | Low | Low | chunk BC는 보조 역할; PPO가 주 학습 |
| RL 탐색이 너무 sparse해서 bead_in_target 안 나옴 | High | High | sim BC buffer가 채워지지 않으면 탐색 강화 필요 → weight_tilt 올리거나 exploration 증가 |

---

## Notes

### IL+RL 조화의 핵심 원칙 (diffusion_policy, UMI, DAPG에서 공통 발견)

```
1. IL은 "어떻게 움직이느냐"를 담당 (motion manifold 학습)
   RL은 "결과가 좋으냐"를 담당 (task success 최적화)
   
2. 두 신호가 같은 파라미터를 공유할 때 (LSTM + policy head):
   - IL gradient: demo 분포와 policy 분포 일치 방향
   - RL gradient: 보상 최대화 방향
   - 이 두 방향이 직교하거나 반대면 학습 불안정
   
3. 해결: RL 보상을 task outcome으로만 제한
   - BC가 이미 "올바른 이동 방향"을 알려주고 있음
   - RL에서 추가로 이동 방향을 알려주면 중복 + 충돌
   
4. BC decay 타이밍:
   - 너무 빠름: RL이 random walk에서 시작 → 탐색 실패
   - 너무 느림: BC local min에 갇힘
   - 적절: RL 성공 에피소드가 sim BC buffer를 채울 때까지 유지
           (sim BC가 채워지면 IL prior 없어도 self-reinforcing)
```

### diffusion_policy에서 장기 적용 가능한 추가 개선 (v7 후보)

1. **obs 히스토리 conditioning** (T_obs=2):
   - BC 훈련 시 `(obs[t-1], obs[t])` 스태킹 → zero-init LSTM 문제 완화
   - obs 차원 변경 필요하므로 v6에는 미적용, v7 설계 시 반영

2. **Stateful BC training**:
   - BC 훈련 시 K=8 warmup steps → LSTM state carry → real episode와 일치
   - `_compute_bc_loss()`를 stateful하게 수정

3. **Temporal Ensembling** (inference 시):
   - 여러 chunk 예측 평균 → 실행 분산 감소
   - chunk_head를 inference에도 활용할 때 의미 있음

4. **Action normalization fitting** (diffusion_policy LinearNormalizer):
   - 데모 분포에서 per-dim mean/std 피팅 → clip_actions보다 정확
   - `DemoBCBuffer` 초기화 시 통계 계산 가능

### v3 vs v6 reward 비교 (참조)

v3에서 안정적으로 학습된 구조:
- weight_tilt=8.0, weight_align=6.0, weight_dist_to_target=10.0
- demo 없음 → RL이 모든 탐색 담당
- pour됨 (bead_in_target > 0)

v6의 변화:
- IL(BC)가 transport/tilt 방향을 이미 제공
- RL이 중복으로 transport reward를 주면 BC와 충돌
- 따라서 transport/tilt RL reward 제거 후 BC가 그 역할 담당

v3의 weight_dist_to_target=10이 필요했던 이유는 BC가 없었기 때문.
v6에는 BC가 있으므로 transport reward 불필요.
