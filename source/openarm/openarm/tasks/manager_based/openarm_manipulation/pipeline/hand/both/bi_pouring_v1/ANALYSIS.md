# bi_pouring_v1 태스크 분석 및 DexPour 논문 비교

> 작성일: 2026-03-20
> 분석 대상: `/bi_pouring_v1/` vs DexPour (IROS 2025)

---

## 1. 제어 방식 분석

### 1.1 Action Space

| 항목 | 값 |
|------|-----|
| **차원** | 7D (오른팔 관절만) |
| **타입** | Joint position delta (연속) |
| **제어 수식** | `target = default_pos + clamp(action, -1, 1) × 0.5 rad` |
| **Fabrics** | 미사용 (순수 관절 공간 제어) |

### 1.2 제어 체계

| 부위 | DOF | 제어 방식 | 비고 |
|------|-----|---------|------|
| **오른팔** | 7 | `set_joint_position_target` (PD) | 정책 제어 |
| **오른손 (Teosllo)** | 20 | `write_joint_state_to_sim` (강제) | 고정 파지 자세 |
| **왼팔** | 7 | `write_joint_state_to_sim` (강제) | 고정 홀더 |
| **왼손 그리퍼** | 2 | `write_joint_state_to_sim` (강제) | 고정 위치 |

### 1.3 Cup Kinematic Attachment

매 step마다 EE pose에서 컵 pose를 역산하여 `write_root_pose_to_sim`으로 갱신.
- 오른손 컵 (Source): `rl_dg_ee` 기준, attach_offset=[0.032, 0.02, 0.0]m, rot=90° X-flip
- 왼손 컵 (Target): `ll_dg_ee` 기준, attach_offset=[0.0, 0.0, 0.06]m, rot=90° Z-flip

### 1.4 오른손 고정 파지 자세

```
thumb:  _1=0.0, _2=-1.57 (opposition curl), _3=0.5, _4=0.5
index:  _1=0.0, _2=0.7 (curl),              _3=0.5, _4=0.5
middle: _1=0.0, _2=0.7 (curl),              _3=0.5, _4=0.5
ring:   _1=0.0, _2=0.7 (curl),              _3=0.5, _4=0.5
pinky:  _1=0.0, _2=0.0,  _3=1.2,           _4=0.8
```

---

## 2. 리워드 구조 분석

### 2.1 Stage 분기 메커니즘 (ρ-trigger)

```
dist_cup = ||source_pour_point_w - target_opening_w||
at_pour = (dist_cup < 0.17 m)
_pour_trigger_steps += 1 if at_pour else 0
_pour_stage_active = (_pour_trigger_steps >= 5)
```

- **Stage 0 (Transport)**: `_pour_stage_active = False` → upright penalty 활성
- **Stage 1 (Pour)**: `_pour_stage_active = True` → tilt/align reward 활성

### 2.2 전체 리워드 수식

```python
reward = (
    1.5  × transport_cup_distance     # exp(-2 × dist_cup)
  - 0.5  × transport_upright_penalty  # 1 - (source_up · world_up), stage 0만
  + 1.5  × pour_tilt                  # Gaussian @ 45°, stage 1만
  + 1.5  × pour_align                 # 0.5×(1+cos θ), stage 1만
  + 6.0  × bead_entry                 # 이벤트 (새 진입)
  + 2.5  × bead_stable_retention      # 연속 (안정 보유)
  - 4.0  × bead_spill_penalty         # 흘린 bead 비율
  - 1.0  × collision_penalty          # 근접 위험
  - 0.05 × action_smoothness_penalty  # MSE(Δaction)
)
```

### 2.3 각 Reward Term 상세

#### Transport: Cup Distance
```python
r = exp(-2.0 × dist_cup)  # [0, 1]
```
항상 활성. 컵 pour point → target opening 거리를 줄이도록 유도.

#### Transport: Upright Penalty
```python
p = (1.0 - dot(source_up_w, [0,0,1])).clamp(0, 1)  # [0, 1]
```
Stage 0 (transport)에만 활성. 이송 중 컵이 기울어지지 않도록.

#### Pour: Tilt
```python
u = dot(source_up_w, [0,0,1])  # cos(tilt angle)
r = exp(-((u - 0.707) / 0.2)²)  # Gaussian 피크: cos(45°) = 0.707
```
Stage 1 (pour)에만 활성. 45° 틸팅 목표.

#### Pour: Align
```python
to_target_dir = normalize(target_opening_w - source_pour_point_w)
cos_θ = dot(source_pour_axis_w, to_target_dir)
r = 0.5 × (1 + cos_θ)  # [0, 1]
```
Stage 1에만 활성. pour_axis(컵 +X)가 target을 향하도록.

#### Bead Entry (이벤트)
```python
event = (현재 target 안) AND (이전엔 없었음) AND (source 안에 없음)
r = event.sum().float()  # [0, N_bead]
```
Sparse reward. bead가 처음 target cup에 진입할 때 +6 보상.

#### Bead Stable Retention
```python
stable = (any_in_target) AND (not_exited) AND (centroid_speed ≤ 0.35 m/s)
r = stable × (0.5 + 0.5 × clamp(1 - speed/0.35, 0, 1))  # [0, 1]
```
연속 보상. bead가 target cup 안에서 안정적으로 정체 중일 때.

#### Bead Spill Penalty
```python
major_spill = (not in target) AND (not in source) AND (z ≤ z_threshold) AND (xy ≥ 0.1m)
p = spill_count / n_beads  # [0, 1]
```
양쪽 컵 밖으로 떨어진 bead 비율에 -4 패널티.

#### Collision Penalty
```python
# Rim scraping: source pour point가 target cup rim에 너무 가까울 때
# EE collision: right_ee ↔ left_ee 거리 < 0.12m
# Cross cup: right_cup ↔ left_ee, left_cup ↔ right_ee < 0.11m
p = max(rim_scrape, max(ee_pen, cross_pen))  # proximity_penalty = clamp((thr-d)/thr, 0, 1)
```

#### Action Smoothness
```python
p = mean((action_t - action_{t-1})²)
```

### 2.4 성공 판정

```python
success = (
    any_bead_has_entered_target
    AND NOT any_bead_exited_after_entry
    AND any_bead_currently_in_target
    AND (stable_retention_steps >= 100)  # ≈ 0.83초 연속 안정
)
```

---

## 3. Observation Space (36D)

| 컴포넌트 | 크기 | 내용 |
|---------|------|------|
| `arm_joint_pos` | 7 | 오른팔 관절 위치 (rad) |
| `arm_joint_vel` | 7 | 오른팔 관절 속도 (rad/s) |
| `cup_relative_pose` | 7 | source 기준 target cup 상대 포즈 (pos3 + quat4) |
| `pour_point_to_opening` | 3 | source cup local frame에서 pour point → opening 벡터 |
| `source_cup_velocity_summary` | 2 | [lin_speed, ang_speed] 크기만 |
| `tilt_alignment_summary` | 3 | [up·world, pour_axis·target_dir, up_s·up_t] |
| `last_arm_action` | 7 | 이전 step action |
| **합계** | **36** | |

**설계 근거**:
- `cup_relative_pose` (7D): 절대 위치 대신 상대 포즈 → translation invariance
- velocity_summary (2D): 방향 불필요, 크기만으로 충분 → 네트워크 입력 효율

---

## 4. 환경 설정 요약

| 항목 | 값 |
|------|-----|
| **로봇** | OpenArm (7+7 DOF arm) + Teosllo (20 DOF hand) bimanual |
| **USD** | `openarm_tesollo_sensor.usd` |
| **물체** | Table (kinematic), Source Cup (kinematic), Target Cup (kinematic), Bead (dynamic) |
| **Episode 길이** | 6.0s = 360 control steps (120 Hz, decimation 2) |
| **Env 수** | 2048 (학습), 50 (play) |
| **우측 팔 초기 자세** | `[-0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]` rad |
| **초기 자세 노이즈** | `[0.015, 0.020, 0.020, 0.025, 0.015, 0.010, 0.010]` rad (우측 팔만) |
| **Source Cup 초기 위치** | `[0.42, -0.18, 0.34]` m |
| **Target Cup 초기 위치** | `[0.42,  0.18, 0.34]` m |
| **Bead 초기 위치** | `[0.42, -0.18, 0.38]` m (source cup 내부) |

### Termination 조건

| 조건 | 이유 |
|------|------|
| `major_spill` | bead가 양쪽 컵 밖으로 낙하 |
| `invalid_state` | 컵/bead가 workspace 이탈, z 이상 등 |
| `time_out` | 360 steps (6.0초) 초과 |
| `success` | stable retention 100 steps 달성 |

---

## 5. 학습 알고리즘 설정

| 파라미터 | 값 |
|---------|-----|
| **알고리즘** | PPO (rl_games, `a2c_continuous`) |
| **네트워크** | `[512, 256, 128]` MLP, ELU, shared actor-critic |
| **learning_rate** | 3e-4 (linear schedule) |
| **gamma** | 0.998 |
| **tau (GAE λ)** | 0.95 |
| **entropy_coef** | 0.002 |
| **e_clip** | 0.2 |
| **horizon_length** | 16 steps |
| **minibatch_size** | 16384 |
| **mini_epochs** | 4 |
| **max_epochs** | 20000 |
| **reward_scale** | 0.01 |
| **normalize_input** | True |
| **normalize_value** | True |

---

## 6. DexPour 논문과 비교

> DexPour: "Effective and Efficient High-DoF Robotic Hand Liquid Pouring via Hierarchical Reward with Approximated Proxy Abstraction", IROS 2025

### 6.1 핵심 비교 표

| 항목 | DexPour (논문) | bi_pouring_v1 (구현) |
|------|--------------|---------------------|
| **로봇** | Franka Panda (7 DOF) + Allegro Hand (16 DOF) = 23 DOF | OpenArm (7+7 DOF) + Teosllo (20 DOF) = 36 DOF bimanual |
| **Action Space** | 23D (전체 관절 제어) | 7D (오른팔만) |
| **학습 단계 수** | 4단계 (Approach, Grasp, Transport, Pour) | **2단계** (Transport, Pour만 구현) |
| **Stage Trigger** | λ(접근), μ(파지), ν(들기), ρ(정렬) 4단계 | ρ-trigger만 (거리 < 0.17m, 5 step) |
| **Grasp 학습** | O (Stage 2, finger-cup distance + contact reward) | X (손가락 고정 파지 자세) |
| **Lift 보상** | O (Stage 3, `r_lift` 높이 기반) | X (적용 안 함) |
| **Proxy Spheres** | 1→32 (curriculum 3단계) | **1개 고정** |
| **Curriculum** | 3단계 (16k → 32k → 64k steps, penalty 증가, bead 증가) | **없음** (single stage) |
| **Observation** | joint pos/vel, fingertip pos, cup pose, sphere centroid, prev_actions | 36D (arm state, cup rel pose, tilt summary 등) |
| **학습 프레임워크** | Isaac Lab + PPO | Isaac Lab + rl_games PPO |
| **병렬 환경** | 2048 | 2048 |
| **Physics dt** | 0.008s (125 Hz) | 0.00833s (120 Hz) |
| **성능 (보고)** | 92% @ 70% fill, 99% @ 30% fill | 미측정 (학습 중) |

### 6.2 Task Scope 비교

```
DexPour:  [물체 접근] → [파지] → [이동] → [붓기]
               ↑ λ         ↑ μ       ↑ ν      ↑ ρ
           (4단계 전체 학습)

bi_pouring_v1: [이동 시작] → [붓기]
                     ↑               ↑ ρ
               (이미 파지된 상태에서 시작, 2단계만 학습)
```

**bi_pouring_v1은 DexPour의 Stage 3 (Transport) + Stage 4 (Pour)만 구현.**
Approach와 Grasp는 생략하고 손가락은 고정 자세로 시작.

### 6.3 리워드 Term 대응

| DexPour Term | bi_pouring_v1 Term | 차이 |
|-------------|-------------------|------|
| `r_cup_dist = exp(-2×dist)` | `transport_cup_distance = exp(-2×dist)` | **동일** |
| `p_tilt` (transport upright) | `transport_upright_penalty` | **동일** |
| `r_tilt` (Gaussian @ 45°) | `pour_tilt` | **동일** |
| `r_align = 0.5×(1+cosθ)` | `pour_align = 0.5×(1+cosθ)` | **동일** |
| Sphere 진입/보유 보상 | `bead_entry + bead_stable_retention` | bi_pouring_v1이 세분화 |
| 직접 없음 | `bead_spill_penalty` | bi_pouring_v1 추가 |
| 직접 없음 | `collision_penalty` | bi_pouring_v1 추가 (rim, EE, cross) |
| `p_accel` (curriculum stage 2~3) | `action_smoothness_penalty` | 유사, 상시 적용 |
| `r_lift` (높이 기반, stage 3) | **없음** | bi_pouring_v1 미구현 |
| `p_hand_cup_dist` (접근) | **없음** | bi_pouring_v1 scope 밖 |
| `r_grasp` (파지 contact) | **없음** | bi_pouring_v1 scope 밖 |

### 6.4 설계 철학 차이

#### DexPour: 전체 파이프라인 RL
- 접근→파지→이동→붓기를 **하나의 정책**이 end-to-end 학습
- 고차원 action (23D) 으로 손가락까지 제어
- Curriculum으로 점진적 복잡도 증가

#### bi_pouring_v1: Task Decomposition (붓기 특화)
- **파지는 고정 자세로 전제**, 붓기 동작만 학습
- 낮은 차원 action (7D) 으로 학습 효율화
- 왼팔은 고정 holder로 단순화
- 향후 왼팔 랜덤화, multi-bead, Grasp stage 추가 확장 가능

### 6.5 APA (Approximated Proxy Abstraction) 활용도

| 항목 | DexPour | bi_pouring_v1 |
|------|---------|--------------|
| **Proxy 방식** | 단단한 구 (rigid sphere) | 단단한 구 (rigid sphere) **동일** |
| **초기 bead 수** | 1개 (Stage 1) | **1개 고정** |
| **최대 bead 수** | 32개 (Stage 3) | 1개 고정 (curriculum 없음) |
| **테스트 방식** | 현실 액체 시뮬레이션(PBD)으로 평가 | 미정 |
| **계산 절감** | 81.6% 감소 (vs full simulation) | 동일 원리 적용 |

---

## 7. bi_pouring_v1 현재 한계 및 향후 확장

### 현재 제한

1. **Grasp acquisition 없음**: 손가락 고정 자세로 시작 → 실제 물체 파지 학습 불가
2. **Left holder 고정**: 좌측 팔 동작 없이 target cup은 항상 고정 위치
3. **Bead 1개**: curriculum 없이 단일 sphere만 사용
4. **Curriculum 없음**: DexPour의 점진적 penalty/bead 증가 미적용
5. **Action 7D**: 오른팔만 제어, 손가락 파지력 조절 불가

### 향후 확장 포인트 (코드에 준비됨)

```python
# bi_pouring_env_cfg.py 에 이미 준비된 확장점
use_left_holder_reset_fabric: False   # FABRICS pose sampler 연결 예정
left_holder_init_pose_sampler: "fixed"  # "random"으로 변경 가능
left_holder_init_pos_noise_xyz: (0,0,0)  # randomization 준비
```

| 확장 방향 | 우선순위 | 설명 |
|---------|---------|------|
| Left holder randomization | 높음 | FABRICS pose sampler 연결, target cup 위치 다양화 |
| Multi-bead (2~8개) | 중간 | APA 충실도 향상 |
| Curriculum (penalty 증가) | 중간 | DexPour 방식 3단계 적용 |
| Grasp stage 추가 | 낮음 | 별도 grasp 태스크 완성 후 연결 |
| Bimanual action (7+7D) | 낮음 | 양팔 동시 제어 |

---

## 8. 파일 구조

```
bi_pouring_v1/
├── bi_pouring_env.py           # DirectRLEnv (stage trigger, cup attachment, bead)
├── bi_pouring_env_cfg.py       # Scene, physics, 기하 파라미터
├── bi_pouring_constants.py     # NUM_OBSERVATIONS=36, NUM_ACTIONS=7
├── bi_pouring_preset.py        # 관절 이름, 초기 자세, attachment transform
│
├── mdp/
│   ├── observations.py         # 7개 obs 함수
│   ├── rewards.py              # 9개 reward/penalty 함수
│   ├── terminations.py         # major_spill, invalid_state
│   └── __init__.py
│
├── config/
│   └── agents/
│       └── rl_games_ppo_cfg.yaml  # PPO 하이퍼파라미터
│
└── ANALYSIS.md                 # 이 파일
```

---

## 9. 핵심 요약

| 항목 | bi_pouring_v1 |
|------|--------------|
| **핵심 아이디어** | DexPour Stage 3~4만 추출, 오른팔 7D 제어로 단순화 |
| **제어 방식** | PD joint control (Fabrics 미사용), 손가락 고정 |
| **리워드 구조** | 2단계 stage (transport/pour), bead entry 이벤트 중심 |
| **주요 특이점** | kinematic cup attachment, ρ-trigger 자동 stage 전환 |
| **DexPour와 차이** | Approach/Grasp 단계 없음, curriculum 없음, bead 1개 고정 |
| **장점** | 낮은 action 차원, 안정적 학습 시작점 |
| **한계** | 전체 파이프라인 아님, 실제 파지는 별도 태스크 필요 |
