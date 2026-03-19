# bi_pouring_v1

`bi_pouring_v1`는 오른팔이 소스 컵을 들고, 왼팔은 타겟 컵을 고정 지지하는 초기 버전의 bimanual pouring 학습 태스크이다. 이름은 `manager_based` 경로 아래에 있지만, 실제 구현은 `isaaclab.envs.DirectRLEnv` 기반 direct RL 환경이며 `rl_games`의 PPO 설정을 사용한다.

## 1. Framework

### Environment

- Env class: `BiPouringEnv`
- Env cfg: `BiPouringEnvCfg`
- Play cfg: `BiPouringEnvCfg_PLAY`
- Gym id:
  - `bi_pouring-v1`
  - `bi_pouring-play-v1`

### RL stack

- Simulator: Isaac Lab / PhysX
- Env type: Direct RL (`DirectRLEnv`)
- Trainer: `rl_games`
- Algorithm: PPO (`a2c_continuous`)
- Model: `continuous_a2c_logstd`
- Policy type: Asymmetric Actor-Critic
  - Actor obs: 36D
  - Critic obs: 48D
  - Action: 7D

### Scene 구성

- Robot: `openarm_tesollo_sensor`
- Table: 고정 rigid object
- Right source cup: 오른쪽 EE에 rigid attachment
- Left target cup: 왼쪽 EE에 rigid attachment
- Bead: 단일 bead 1개

### 제어 범위

- Policy가 제어하는 것은 오른팔 7자유도만이다.
- 왼팔과 왼 gripper는 고정 홀더 역할이며 policy 제어 대상이 아니다.
- 손가락 action / finger observation은 사용하지 않는다.
- v1의 목표는 "grasp 획득"이 아니라 이미 컵을 잡고 있는 상태에서 짧은 transfer/pouring motion을 학습하는 것이다.

## 2. 학습 시나리오

### 기본 시나리오

1. 오른팔은 pouring-ready pose 근처에서 시작한다.
2. 왼팔은 target cup을 고정된 자세로 지지한다.
3. 소스 컵과 타겟 컵은 각 end-effector에 rigidly attached 되어 있다.
4. 소스 컵 내부에는 bead 1개가 들어 있다.
5. 정책은 오른팔 7개 관절 목표 위치를 출력한다.
6. 오른쪽 소스 컵의 pour point가 왼쪽 타겟 컵 opening과 정렬되도록 접근한다.
7. 컵을 과하게 흔들지 않으면서 tilt를 만들어 bead를 타겟 컵 안으로 넣는다.
8. bead가 타겟 컵 안에서 일정 시간 안정적으로 유지되면 성공이다.

### Reset

- 에피소드 길이: `6.0s`
- 시뮬레이션 `dt`: `1/120`
- decimation: `2`
- 학습 env 수: `128`
- play env 수: `50`
- 오른팔 초기 joint에는 작은 noise가 들어간다.
- 왼쪽 holder pose는 현재 고정값이다.
- bead 수는 반드시 `1`이어야 한다.

### Termination

다음 중 하나면 종료된다.

- 성공: bead가 target 내부에서 안정적으로 유지되어 `success_retention_steps >= 8`
- major spill 발생
- invalid state 발생
- episode timeout

## 3. ACTOR / CRITIC가 보는 OBS

## Actor Observation (36D)

정책이 직접 보는 입력은 다음 36차원이다.

1. Right arm joint position: 7
2. Right arm joint velocity: 7
3. Source cup 기준 target cup 상대 pose: 7
   - relative position 3
   - relative quaternion 4
4. Source pour point -> target opening vector: 3
5. Source cup velocity summary: 2
   - linear speed 1
   - angular speed 1
6. Tilt alignment summary: 3
   - source cup up axis vs world up
   - source pour axis vs source-to-target direction
   - source cup up axis vs target cup up axis
7. Last actions: 7

정리하면, actor는 "오른팔 상태 + 두 컵의 상대 기하 + pouring 정렬 정보 + 직전 action"만 본다. bead의 절대 위치/속도는 actor에 직접 주지 않는다.

## Critic Observation (48D)

Critic은 actor observation 36D에 다음 privileged information 12D를 추가로 본다.

1. Bead position in env/world-aligned frame: 3
2. Bead linear velocity in world frame: 3
3. Task flags: 3
   - stable alignment 여부
   - bead in target 여부
   - success 여부
4. Stable alignment steps normalized: 1
5. Spill flags: 2

즉 critic은 bead 상태와 task progress를 더 정확히 알 수 있고, actor는 그 정보를 직접 보지 않는 asymmetric actor-critic 구조이다.

## 4. Reward 디자인

최종 reward는 아래 항들의 가중합이다.

```text
reward =
  1.5  * alignment_reward
  0.75 * controlled_tilt_reward
  6.0  * bead_entry_reward
  2.5  * stable_retention_reward
  -4.0 * spill_penalty
  -1.0 * collision_penalty
  -0.05 * smoothness_penalty
```

### 1) Alignment reward

소스 컵의 pour point와 타겟 컵 opening이 가까워지도록 유도한다.

- XY 정렬
- Z 높이 정렬
- threshold 기반 stable alignment step 누적의 기반이 된다.

핵심 파라미터:

- `stable_alignment_xy_threshold = 0.040`
- `stable_alignment_z_min = 0.010`
- `stable_alignment_z_max = 0.120`
- `alignment_z_target = 0.040`

### 2) Controlled tilt reward

두 단계 성격을 가진다.

- 접근 단계에서는 컵을 너무 일찍 기울이지 않고 upright를 유지하도록 유도
- alignment가 확보되면 타겟 방향으로 적절히 tilt 하도록 유도

즉, "가까워지기 전에는 세우고, 가까워진 뒤에는 타겟 방향으로 기울이는" shaping이다.

핵심 파라미터:

- `approach_upright_min_cos = 0.92`
- `controlled_tilt_target_cos = 0.55`
- `controlled_tilt_cos_scale = 0.18`

### 3) Bead entry reward

bead가 source를 떠나 target 내부로 처음 들어가는 순간 이벤트성 reward를 준다.

- one-time event 성격
- target 진입을 강하게 밀어주는 주요 sparse reward

가중치:

- `reward_bead_entry_weight = 6.0`

### 4) Stable retention reward

bead가 target 내부에 있고 속도가 낮을수록 보상을 준다.

- 단순 진입이 아니라 "안정적으로 담기는 것"을 학습 목표로 만든다.
- success 판정과도 연결된다.

핵심 파라미터:

- `stable_retention_speed_threshold = 0.35`
- `success_retention_steps = 8`

### 5) Spill penalty

bead가 target/source 어느 쪽에도 없고 충분히 아래로 떨어지거나 멀리 벗어나면 spill로 간주한다.

- 일반 spill
- major spill

major spill은 종료 조건에도 사용된다.

가중치:

- `penalty_spill_weight = 4.0`

### 6) Collision penalty

두 컵 중심이 지나치게 가까워지는 것을 패널티로 준다.

- 실제 충돌 proxy 역할
- cup center distance 기반

핵심 파라미터:

- `collision_center_distance_threshold = 0.085`

### 7) Smoothness penalty

현재 action과 이전 action 차이의 제곱 평균을 패널티로 사용한다.

- 급격한 관절 명령 변화 억제
- 불필요한 흔들림 감소

가중치:

- `penalty_action_smoothness_weight = 0.05`

## 5. Networks

### Action / Observation dimension

- Action dim: `7`
- Actor obs dim: `36`
- Critic obs dim: `48`

### Policy / Value network

`rl_games` 설정 기준:

- network: `actor_critic`
- shared backbone: `separate: False`
- MLP: `[512, 256, 128]`
- activation: `ELU`
- `d2rl: False`

### Action head

- continuous action
- `mu_activation: None`
- `sigma_activation: None`
- `fixed_sigma: True`
- 초기 sigma는 상수 `0`

즉 평균 action을 출력하는 연속 제어 정책이며, log-std는 고정 형태로 사용한다.

### Central value

- `has_central_value: True`
- critic도 `[512, 256, 128]` MLP 사용
- critic은 privileged observation 48D를 입력으로 받는다.

### PPO 주요 하이퍼파라미터

- learning rate: `3e-4`
- gamma: `0.998`
- tau: `0.95`
- horizon length: `16`
- minibatch size: `512`
- mini epochs: `4`
- entropy coef: `0.002`
- critic coef: `4`
- `normalize_input: True`
- `normalize_value: True`
- reward scale: `0.01`

## 6. v1 범위와 의도

이 태스크는 복잡한 전체 bimanual manipulation 문제를 한 번에 풀기 위한 것이 아니라, 다음 조건으로 문제를 축소한 v1 베이스라인이다.

- 오른팔만 policy control
- 왼쪽 holder는 고정
- 컵은 EE에 rigid attachment
- bead는 1개
- 짧은 거리의 pour transfer만 학습

즉 이 버전의 핵심은 `grasp -> transport -> pour -> retain` 전체를 다루는 것이 아니라, `aligned approach -> controlled tilt -> bead transfer -> stable retention`의 RL 학습 토대를 만드는 것이다.

## 7. 향후 확장 포인트

코드에 이미 seam으로 남겨둔 확장 포인트는 다음과 같다.

- left holder init pose randomization
- FABRICS 기반 pose sampler 연결
- cup-to-tool attach transform 실측값 반영
- source/target cup opening anchor 실측값 반영
- single-bead에서 multi-bead 혹은 실제 liquid proxy로 확장
- sim2real calibration 반영
