# `pour_v3` 발표/포스터 상세 정리

이 문서는 `/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/pour_v3` 코드 기준으로,
학회 발표와 포스터 보강에 필요한 내용을 **구현 관점에서 상세히 설명**하기 위한 문서다.

목표는 아래 3가지를 명확히 정리하는 것이다.

1. `reward structure`와 `Stage A / Stage B`를 수식 기반으로 설명
2. `demo data`가 어떻게 수집되고 Isaac 시뮬레이터 기준의 digital twin으로 변환되어 agent가 이해할 수 있는 형태가 되는지 설명
3. `actor / critic / observation` 관점에서 네트워크가 시뮬레이션 데이터를 어떻게 학습하는지 설명

주의:

- 이 문서는 **현재 `pour_v3` 코드에 실제로 구현된 내용**을 우선 설명한다.
- 사용자가 구상 중인 논문 프레이밍(예: BC prior, teleop augmentation)은 별도로 언급하되,
  **현재 이 디렉터리의 핵심 실행 코드에서 직접 구현된 내용과는 구분해서 설명**한다.

---

## 1. 한 줄 요약

`pour_v3`는 “왼손은 타겟 컵을 고정하고, 오른손은 source cup을 잡은 상태에서,
6D palm pose + Fabrics IK를 이용해 workspace를 먼저 확보한 뒤,
pour-point 정렬과 tilt를 통해 bead transfer를 일으키도록 학습하는 stage-wise RL task”라고 정리할 수 있다.

핵심은 다음 순서다.

1. `grasp/lift`가 이미 된 상태에서 시작한다. (`warmstart reset`)
2. 먼저 컵을 타겟 컵 근처의 유효 workspace로 가져간다. (`Stage A`)
3. 그 다음 rim 위치, 높이, tilt를 조절해 실제 bead flow를 만든다. (`Stage B`)
4. 최종적으로 bead가 target cup 안에 들어가고 spill이 적어야 성공으로 인정한다. (`Stage C + success`)

즉, “붓기”를 하나의 추상 행동으로 보지 않고,
**접근 -> 정렬 -> 기울임 -> 실제 전달**의 인과사슬로 분해해 학습한다.

---

## 2. 코드 기준 전체 학습 구조

현재 `pour_v3`의 구조를 발표용으로 압축하면 아래와 같다.

```text
remote teleoperation demos (약 10개)
-> Isaac 기준 HDF5 / digital twin 정렬
-> pour phase 추출
-> demo pose reference bank 구축

RL 학습 시:
warmstart reset (grasp/lift 성공 상태에서 시작)
-> actor가 6D palm pose + 5D finger action 출력
-> Fabrics IK가 palm command를 arm joint motion으로 변환
-> bead proxy fluid 기반으로 reward 계산
-> PPO / PPO-LSTM으로 업데이트
```

현재 구현의 중요한 점은:

- `demo`는 **현재 `v3`에서 actor 입력으로 직접 들어가는 것이 아니라**
  **reward-side reference**로 사용된다.
- 즉, **demo imitation을 직접 supervised action loss로 거는 구조는 아니다.**
- 대신 demo가 “좋은 pre-pour arm posture”가 무엇인지 알려주는 역할을 한다.

---

## 3. Task 설정 요약

### 3.1 로봇 역할 분담

- 왼팔: target cup을 들고 고정하는 역할
- 오른팔: source cup을 잡고 이동, 정렬, tilt, pour를 수행하는 역할

즉, “bimanual pouring”이지만 실제 learning target은 **오른팔의 source-cup control policy**다.

### 3.2 Action space

코드 상 action은 총 `11D`다.

```text
[0:6]   6D palm pose action
[6:11]  5D per-finger action
```

설명:

- `6D palm pose`: `(x, y, z, ez, ey, ex)` 형태의 palm target
- 이 palm target을 직접 관절로 보내는 것이 아니라 **Fabrics IK**를 통해 arm joint로 변환한다.
- `5D per-finger action`: 손가락별 lerp control

의미:

- 팔 전체의 큰 움직임은 palm-space에서 제어한다.
- 손가락은 이미 grasp된 컵을 놓치지 않는 쪽으로 유지한다.

### 3.3 Observation space

현재 코드는 `asymmetric actor-critic` 구조다.

- Actor observation: `60D`
- Critic observation: `140D`

즉, actor와 critic이 **동일한 세계를 보지 않는다.**

이것이 발표에서 중요한 포인트다.

---

## 4. 왜 palm-space + Fabrics IK를 쓰는가

이 task에서 가장 중요한 설계 판단 중 하나는,
“7개 arm joint를 직접 RL로 모두 때리는 대신 palm pose를 RL이 내고,
그걸 Fabrics IK가 관절공간으로 풀게 한다”는 점이다.

이 구조를 쓰는 이유:

1. 고차원 관절공간 직접 탐색보다 palm-space 탐색이 더 task-aligned다.
2. “컵을 어디로 가져가야 하는가?”라는 문제는 joint보다 palm pose가 더 직접적이다.
3. RL이 gross motion을 찾고, IK가 feasible arm posture를 보장하므로 탐색이 훨씬 안정적이다.

발표용 문장:

> We do not ask RL to discover feasible 7-DoF arm postures from scratch.
> Instead, RL proposes a task-space palm target, and Fabrics IK resolves that target into arm motion.
> This reduces exploration burden and keeps the policy on a physically meaningful manifold.

---

## 5. Stage A / Stage B / Stage C 개념 정리

현재 코드의 reward와 observation 구조를 보면, 사실상 3단계로 보는 것이 맞다.

### Stage A: Available Workspace

목표:

- source cup을 타겟 컵 근처의 “붓기 가능한 영역”으로 옮긴다.
- 동시에 grasp를 유지한다.

이 단계에서 중요한 자유도:

- 주로 `j1 / j2 / j3 / j4`
- 이유: 컵의 global reach, lateral alignment, 높이, gross arm posture를 결정하기 때문

직관:

- 아직 “실제로 붓는 것”보다 먼저,
  source cup이 “붓을 수 있는 위치와 자세 근처”에 와야 한다.

### Stage B: Pour-point Tuning

목표:

- source cup의 rim/pour-point를 target opening 위로 맞춘다.
- rim 높이도 적절히 맞춘다.
- 기울기를 만들어 bead flow가 생길 수 있도록 한다.

이 단계에서 중요한 자유도:

- 주로 `j5 / j6 / j7`
- 이유: wrist orientation, local tilt, final pouring geometry를 결정하기 때문

직관:

- Stage A가 “workspace 확보”라면,
  Stage B는 “정말 쏟아질 수 있는 정교한 기하 조정”이다.

### Stage C: Actual Transfer

목표:

- bead가 실제로 target cup 안으로 들어가야 한다.
- spill은 줄여야 한다.

직관:

- geometry만 맞는다고 성공이 아니다.
- 실제 transfer가 일어나야 한다.

---

## 6. Reward 구조 상세

현재 `pour_v3`의 핵심은 reward를 잘게 나눠서 인과사슬대로 정렬했다는 점이다.

총 reward는 코드상 다음 구조로 계산된다.

```text
total
= r_hold
+ r_dist_to_target
+ r_demo_arm_pose
+ r_pour_geo
+ r_pour_z
+ r_pour_stage
+ weight_success * r_success
+ overfill_bonus
- spill_weight * spill_cost
```

이를 stage별로 묶으면 아래처럼 볼 수 있다.

```text
R_total = R_A + R_B + R_C + R_success - R_spill
```

여기서:

- `R_A`: grasp 유지 + workspace 접근 + demo 기반 arm posture guidance
- `R_B`: rim 위치/높이/tilt 기반 pour geometry shaping
- `R_C`: bead 실제 전달과 직접 연결된 reward

---

## 7. Reward 수식 버전

발표나 문서에서 설명하기 쉽게 정리하면:

```text
R_total = R_A + R_B + R_C + w_success R_success - w_spill C_spill

R_A
= R_hold
+ w_dist exp[-k_d max(d_cup_xy - d_sat, 0)]
+ w_demo exp[-e_demo]

R_B
= g_ready(d_cup_xy) * (
     w_xy   exp[-k_xy d_mouth_xy]
   + w_z    exp[-((Delta z_mouth - z*) / sigma_z)^2]
   + w_rel  tau
   + w_tilt tau_tgt
 )

R_C
= g_ready(d_cup_xy) * (
     w_near  s_bead_near
   + w_in    f_bead_in_target
   + w_cross Delta c_cross
 )

g_ready(d_cup_xy) = sigmoid((d0 - d_cup_xy) / delta)
```

기호 설명:

- `d_cup_xy`: source cup center와 target 쪽 shared workspace 사이의 XY 거리
- `d_mouth_xy`: source cup pour-point와 target opening 사이의 XY 거리
- `Delta z_mouth`: source rim이 target opening보다 얼마나 위/아래에 있는지
- `z*`: 이상적인 rim clearance target
- `tau`: 현재 기울기 정도
- `tau_tgt`: target-oriented tilt depth
- `f_bead_in_target`: target cup 내부에 들어간 bead 비율
- `Delta c_cross`: target mouth를 새로 통과한 bead의 증가량

핵심 포인트:

- geometry reward만 두면 “붓는 척”만 하고 실제 transfer는 안 생길 수 있다.
- transfer reward만 두면 너무 sparse해서 policy가 그 지점까지 도달하지 못한다.
- 그래서 geometry와 bead reward를 **같이** 써야 한다.

---

## 8. Stage A reward 상세

### 8.1 `r_hold`

`r_hold`는 grasp를 유지하기 위한 항이다.

코드상:

```text
r_hold
= weight_grasp_maintain * grasp_maintain_reward
+ weight_contact_maintain * full_grasp_flag * contact_gate
+ r_force_balance * upright_gate
+ r_finger_curl
```

구성 요소:

1. `grasp_maintain_reward`
   - 컵의 palm-local 위치가 초기 grasp 위치와 얼마나 비슷한지 본다.
   - 컵이 손바닥 기준으로 미끄러지지 않게 유지하는 항이다.

2. `full_grasp_flag`
   - thumb 접촉 + 다른 손가락 최소 개수 접촉이 있는지를 본다.

3. `r_force_balance`
   - thumb force와 다른 손가락 평균 force가 너무 불균형하지 않도록 한다.

4. `r_finger_curl`
   - 손가락이 닫힌 상태를 유지하도록 한다.

### 8.2 왜 tilt-phase aware인가

코드에서는 `tilt_amount`에 따라 hold 관련 항을 완화한다.

이유:

- 직립 상태에서는 강한 grasp 유지가 중요하다.
- 깊게 tilt한 상태에서는 접촉 조건이 너무 빡빡하면 policy가 기울이기 자체를 회피한다.

즉:

- upright transport 단계에서는 grip을 강하게 요구
- active pour 단계에서는 grip 조건을 일부 완화

이 설계는 “붓기 시작하면 원래 잡던 접촉 패턴이 바뀌는 것”을 반영한다.

### 8.3 `r_dist_to_target`

이 항은 cup center 기반의 접근 보상이다.

```text
r_dist_to_target = w_dist * exp[-k_dist * max(d_cup_xy - d_sat, 0)]
```

기본값:

- `weight_dist_to_target = 5.0`
- `dist_to_target_exp_scale = 5.0`
- `cup_transport_saturate_xy = 0.17`

왜 saturate하는가:

- target 근처로 충분히 접근한 뒤에는 이 항이 더 이상 policy를 강하게 끌지 않게 해야 한다.
- 그렇지 않으면 pouring stage에서도 transport reward만 과도하게 최적화하게 된다.

### 8.4 `r_demo_arm_pose`

이 항은 demo reference를 이용해 **j1~j4 workspace posture**를 유도한다.

중요:

- frame matching은 `j1~j4`만으로 nearest neighbor를 찾는다.
- `j5~j7`은 frame 선택에서 제외된다.

이유:

- `j5`는 tilt 핵심 자유도다.
- 만약 `j5`까지 demo matching에 묶이면 shallow tilt local minimum에 갇히기 쉽다.

구조:

```text
demo_arm_joint_err
= || (q_arm[:4] - q_demo_target[:4]) / std[:4] || / sqrt(4)

r_demo_arm_pose = gate * w_demo * exp[-demo_arm_joint_err]
```

기본값:

- `weight_demo_arm_pose = 20.0`
- `weight_demo_arm_pose_floor = 5.0`
- `demo_nn_lookahead_frames = 10`

해석:

- demo는 “이 자세가 괜찮다”는 arm posture anchor를 준다.
- 하지만 final pouring depth는 demo가 아니라 RL이 bead reward를 보고 찾아가게 한다.

---

## 9. Stage B reward 상세

Stage B는 “붓기 가능한 기하”를 만드는 단계다.

현재 핵심은 `r_pour_geo + r_pour_z`다.

### 9.1 `g_ready`

```text
g_ready = sigmoid((g_ready_center - cup_center_xy_dist) / g_ready_width)
```

기본값:

- `g_ready_center = 0.20`
- `g_ready_width = 0.03`

왜 binary gate가 아니라 sigmoid인가:

- hard gate는 reward landscape를 갑자기 끊는다.
- sigmoid는 target에 가까워질수록 pour reward가 자연스럽게 커지게 만든다.

### 9.2 `r_pour_xy`

```text
r_pour_xy = weight_pour_xy * exp[-pour_xy_scale * mouth_xy_distance]
```

기본값:

- `weight_pour_xy = 15.0`
- `pour_xy_scale = 8.0`

의미:

- source cup의 rim/pour-point를 target opening 위로 보내는 항이다.

### 9.3 `r_zband`

```text
r_zband = weight_pour_zband * exp[-((mouth_z_clearance - z_target)/sigma_z)^2]
```

기본값:

- `weight_pour_zband = 8.0`
- `pour_zband_target = 0.05`
- `pour_zband_sigma = 0.05`

의미:

- source rim이 target opening보다 너무 낮지도, 너무 높지도 않게 맞춘다.

왜 gaussian band인가:

- 단순히 “높을수록 좋다” 혹은 “낮을수록 좋다”가 아니다.
- 실제 붓기는 적절한 clearance가 있어야 가장 잘 일어난다.

### 9.4 `r_release`

```text
r_release = weight_release * tilt_amount * exp[-pour_xy_scale * mouth_xy_distance]
```

기본값:

- `weight_release = 20.0`

의미:

- target 위로 충분히 왔을 때만 기울이기 시작하도록 유도한다.

왜 필요한가:

- target 위로 오기 전에 tilt를 하면 spill만 증가한다.

### 9.5 `r_tilt_depth`

```text
tilt_target = (1 - cos(theta_target)) / 2
tilt_progress = clamp(tilt_amount / tilt_target, 0, 1)
r_tilt_depth = weight_tilt * tilt_progress * exp[-pour_xy_scale * mouth_xy_distance]
```

기본값:

- `pour_tilt_target_deg = 120`
- `weight_tilt = 40.0`

의미:

- shallow tilt local minimum에서 policy를 빼내기 위한 항이다.
- `j5`를 직접 깊은 tilt 방향으로 밀어주는 역할을 한다.

왜 필요한가:

- demo anchor는 `j1~j4` 중심이라 gross posture는 잡아주지만,
  실제 pour를 만드는 deep tilt는 별도 부트스트랩이 필요하다.

### 9.6 `r_pour_z`

```text
z_violation = max(pour_z_margin - mouth_z_clearance, 0)
r_pour_z = - g_ready * weight_pour_z * z_violation
```

기본값:

- `weight_pour_z = 300.0`
- `pour_z_margin = 0.03`

의미:

- source rim이 target rim 아래로 “박히는 것”만 강하게 막는 barrier다.

왜 단방향 penalty인가:

- 위쪽 자유도는 bead flow가 결정하게 두고,
  아래쪽으로 박는 위험한 행동만 막는 것이 더 자연스럽다.

---

## 10. Stage C reward 상세

Stage C는 geometry가 아니라 **실제 transfer**를 보게 만드는 항이다.

### 10.1 `r_bead_near`

```text
r_bead_near = weight_bead_near * bead_near_score
```

기본값:

- `weight_bead_near = 30.0`
- `bead_near_scale = 12.0`

의미:

- source cup을 떠난 bead가 target 축 근처로 가면 보상을 준다.

왜 필요한가:

- bead가 target 안에 완전히 들어가기 전에도 dense한 중간 신호를 주기 위해서다.

### 10.2 `r_bead_in`

```text
r_bead_in = weight_bead_in * bead_in_target_fraction
```

기본값:

- `weight_bead_in = 200.0`

의미:

- 실제로 target cup 안에 들어간 bead 비율에 비례해 보상한다.

### 10.3 `r_cross`

```text
r_cross = weight_bead_cross * clamp(bead_cross_delta, 0, +inf)
```

기본값:

- `weight_bead_cross = 150.0`

의미:

- bead가 target mouth를 새로 통과하는 순간의 보상이다.

왜 필요한가:

- “통과는 하는데 안 쌓이는 상황”과 “아예 통과도 못 하는 상황”을 분리해서 학습시킬 수 있다.

### 10.4 `r_pour_stage`

```text
r_pour_stage = g_ready * (r_bead_near + r_bead_in + r_cross)
```

이 구조의 의미:

- target 근처에 왔을 때만 bead transfer 관련 reward가 강하게 작동한다.
- 너무 이른 시점에 bead term이 작동하면 신호가 noisy해질 수 있다.

---

## 11. Success / Spill 계산

### 11.1 success 조건

현재 success는 대략 다음 조건이다.

```text
bead_in_target_fraction >= success_fill_ratio
and spill_ratio <= success_spill_max
and cup_center_xy_dist < pour_binary_xy_thresh
```

기본값:

- `success_target_fill_ratio = 0.50`
- `success_spill_max = 0.40`
- `pour_binary_xy_thresh = 0.20`

하지만 success fill ratio는 `success ADR`에 의해 바뀔 수 있다.

### 11.2 success reward

```text
r_success = 1 if success_now else 0
total += weight_success * r_success
```

기본값:

- `weight_success = 100.0`

### 11.3 spill cost

```text
spill_cost = sqrt(max(spill_ratio, 0))
total -= spill_weight * spill_cost
```

현재 기본값:

- `weight_spill = 0.0`
- `enable_spill_adr = False`

즉, 현재 `test6` 기준 계열에서는 spill penalty가 기본적으로 꺼져 있다.

발표에서 이 점을 설명할 때는:

> Early pouring bootstrapping can be harmed if spill is penalized too aggressively from the start.
> In the current setting, spill is still logged and used in success criteria, but its direct penalty weight is disabled.

라고 말할 수 있다.

---

## 12. 왜 이런 reward 구조인가

이 질문은 거의 반드시 나온다.

핵심 대답:

### 12.1 sparse reward 문제

“target cup 안에 bead가 들어갔는가?”만 보상으로 쓰면 너무 늦다.

policy가 실제 reward를 보려면:

1. 컵을 집고 있어야 하고
2. 타겟 근처로 와야 하고
3. rim이 맞아야 하고
4. 적절히 기울어야 하고
5. bead가 실제로 움직여야 한다

이 모든 걸 한 번에 랜덤 탐색으로 찾기는 어렵다.

### 12.2 geometry-only reward 문제

반대로 geometry reward만 있으면:

- 컵이 보기 좋게 target 위에 오더라도
- 실제로는 기울지 않거나
- bead가 안 흐르거나
- spill이 커질 수 있다

따라서 geometry와 transfer reward가 모두 필요하다.

### 12.3 stage-wise causal ordering

현재 구조는 다음 causal ordering을 강제한다.

```text
grasp stability
-> workspace acquisition
-> rim alignment
-> tilt depth
-> bead transfer
-> success under spill constraint
```

즉 reward를 “예쁘게 많이 넣었다”가 아니라,
**task가 실제로 진행되는 순서에 맞춰 보상 구조를 설계했다**는 것이 핵심이다.

---

## 13. Demo data 파이프라인 상세

사용자 설명 기준으로는:

- 원격 조정으로 약 `10회`의 demo 데이터를 모았다.

현재 코드 기준으로는:

- `pour_v1_a11.hdf5`부터 `pour_v1_a20.hdf5`까지, 총 10개 파일을 기본 reference로 쓴다.

```python
demo_pose_paths = tuple(
    _os.path.join(_DEFAULT_DEMO_POSE_DATASET_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
)
```

즉 포스터/발표에서는 자연스럽게

> We collected about ten teleoperated pouring demonstrations.

라고 말해도 현재 코드 구조와 잘 맞는다.

### 13.1 demo 파일에서 읽는 key

`demo_pose_reference.py`는 각 demo에서 아래 key를 요구한다.

```text
obs/right_arm_joint_pos
obs/right_hand_joint_pos
obs/right_hand_reference_joint_pos
obs/datagen_info/eef_pose/right
obs/datagen_info/target_eef_pose/right
timestamps_ns
```

즉 demo는 적어도 아래 정보를 가져야 한다.

- right arm joint trajectory
- right hand joint trajectory
- hand reference trajectory
- right end-effector pose
- target end-effector pose
- timestamps

### 13.2 digital twin의 의미

발표에서 digital twin이라고 부를 수 있는 이유는,
이 demo가 단순 영상이 아니라 아래를 Isaac 기준으로 정렬한 구조이기 때문이다.

1. joint naming
2. end-effector pose frame
3. timestamped trajectory
4. pour phase segmentation이 가능한 task-specific metadata

즉:

- 현실에서 취득한 teleop motion이
- Isaac 시뮬레이터 안에서 같은 semantic structure를 갖는 tensor로 변환된다.

이게 바로 발표에서 말할 digital twin의 핵심이다.

### 13.3 pour phase만 뽑는 방법

`demo_pose_reference.py`는 기본적으로 `phase="pour"`를 사용한다.

우선순위:

1. `pour_start` / `pour_done` 신호가 있으면 그 구간 사용
2. 없으면 마지막 2초 사용
3. 그래도 부족하면 tilt 기준 fallback 사용
4. 마지막 fallback은 trajectory 마지막 25%

의미:

- demo 전체가 아니라 **실제 붓는 구간**을 reference로 사용한다.

### 13.4 agent가 demo를 “이해하는 방법”

현재 `v3`에서 중요한 점:

- actor가 demo 이미지나 raw sequence를 직접 보지 않는다.
- demo는 다음과 같은 compact reference로 변환된다.

예:

- arm joint pose bank
- hand joint pose bank
- target palm pose
- mean / std
- arm velocity / jerk percentile

그리고 실제 학습에서는:

- 현재 arm posture와 demo reference의 차이를 계산하고
- 그 차이를 `r_demo_arm_pose` reward로 바꿔준다.

즉, agent는 demo를 직접 imitation하는 것이 아니라,
**“현재 자세가 demo에서 본 좋은 pre-pour posture와 얼마나 가까운가”를 reward로 느낀다.**

---

## 14. Observation 상세

## 14.1 Actor observation (60D)

actor는 partial / noisy observation을 본다.

구성:

```text
arm_joint_pos            7
arm_joint_vel            7
finger_grasp_progress    5
right_cup_pos_rel_palm   3
right_cup_quat           4
left_cup_pos_rel_palm    3
pour_point_to_opening    3
source_pour_axis         3
source_up_axis           3
transport_summary        8
last_palm_actions        6
bead_in_source_fraction  1
bead_in_target_fraction  1
bead_cross_fraction      1
spill_ratio              1
flow_summary             4
total                   60
```

중요 포인트:

- actor는 **지금 무엇을 해야 하는지**에 필요한 compact한 task geometry를 본다.
- actor obs에는 noise가 들어간다.

noise 기본값:

- `obs_noise_joint_pos = 0.01`
- `obs_noise_joint_vel = 0.05`
- `obs_noise_body_pos = 0.005`
- `obs_noise_cup_pos = 0.015`

즉 actor는 sim2real을 의식한 noisy sensing 환경에서 학습한다.

## 14.2 Critic observation (140D)

critic은 privileged observation을 본다.

critic = actor-clean-state 성격의 `110D` + extra privileged `30D`

extra privileged 부분:

- left arm joint pos `9`
- left arm joint vel `9`
- distal binary contact `5`
- distal contact force `5`
- cup height delta `1`
- rho `1`

왜 critic만 더 많이 보는가:

- actor는 deployable / compact해야 한다.
- critic은 training 시점에만 쓰이는 value estimator라 더 많은 내부정보를 봐도 된다.

발표용 문장:

> The actor receives a compact, partially noisy task representation, while the critic receives additional privileged state to stabilize value learning.

---

## 15. `transport_summary`와 `flow_summary`의 의미

이 두 묶음은 발표에서 설명하기 좋다.

### 15.1 `transport_summary`

```text
mouth_distance
mouth_xy_distance
cup_center_xy_dist
mouth_z_clearance
source_up_dot_world
directional_tilt_cos
mouth_alignment_cos
rho
```

즉 actor는 이 8차원 요약을 통해:

- target 위로 왔는지
- rim이 얼마나 맞았는지
- 높이가 맞는지
- 얼마나 기울었는지
- pour gate 근처인지

를 한 번에 알 수 있다.

### 15.2 `flow_summary`

```text
bead_in_source_delta
bead_in_target_delta
bead_cross_delta
spill_delta
```

즉 actor는 단순 누적량뿐 아니라,
**“방금 직전 step에서 transfer가 좋아졌는가 / 나빠졌는가”**도 볼 수 있다.

이건 긴 horizon pouring task에서 매우 중요하다.

---

## 16. Warmstart reset 구조

이 task는 grasp부터 end-to-end로 배우지 않는다.

현재 `pour_v3`는 기본적으로 `enable_warmstart_reset = True`다.

의미:

- 학습 시작 시점은 “이미 grasp/lift가 성공한 state”다.
- 즉 pouring policy는 grasping 전체를 처음부터 배우지 않는다.

### 16.1 왜 warmstart가 필요한가

만약 grasp부터 배우게 하면:

1. 컵 잡기
2. 컵 들기
3. workspace 이동
4. rim 정렬
5. tilt
6. transfer

를 모두 동시에 찾아야 한다.

이는 exploration burden이 너무 크다.

그래서 `pour_v3`는 grasp/lift를 upstream skill로 보고,
pouring만 집중 학습하는 구조를 쓴다.

### 16.2 reset 시 실제로 일어나는 일

reset에서:

1. warmstart cache에서 arm/hand/palm/cup 상태를 샘플링
2. robot joint state를 그 상태로 기록
3. palm target을 workspace 범위 안으로 clamp
4. cup pose를 sim에 기록
5. left target cup pose도 FK 기준으로 기록
6. bead는 즉시 소환하지 않고 hold 단계 이후 소환

즉 학습 episode는 사실상:

> “already grasped and lifted source cup” 상태에서 시작하는 pour episode

라고 이해하면 된다.

### 16.3 hold 단계가 왜 필요한가

`episode_hold_steps = 120`

의미:

- reset 직후 물리적으로 안정화할 시간을 준다.
- bead도 이 hold 이후에 소환한다.

이유:

- reset 직후 contact가 불안정한 순간에 바로 bead를 넣으면 물리적으로 noisy해진다.

---

## 17. Pour-start curriculum

코드에는 `pour_start_curriculum`도 존재한다.

의도:

- 일부 env를 아예 “target 위에 어느 정도 기울어진 state”로 시작시켜
  bead reward를 초기에 더 자주 경험하게 하려는 장치다.

하지만 현재 기본값:

```text
enable_pour_start_curriculum = False
```

즉:

- 코드 구조는 존재하지만
- 현재 기본 실행 경로에서는 꺼져 있다

발표에서는 이렇게 말하는 것이 안전하다.

> The code contains an optional pour-start curriculum for improving reward reachability, but the current default configuration keeps it disabled.

---

## 18. ADR 구조

현재 코드는 세 가지 ADR 축을 가진다.

### 18.1 noise ADR

켜져 있음:

- `enable_noise_adr = True`

의미:

- 성공률이 오를수록 actor observation noise를 점진적으로 키운다.

목적:

- 후반부에 더 robust한 정책을 만들기 위함

### 18.2 spill ADR

현재 기본값:

- `enable_spill_adr = False`

즉 구조는 있으나 기본 실행에서는 꺼져 있다.

### 18.3 success ADR

켜져 있음:

- `enable_success_adr = True`

의미:

- 성공 판정의 fill ratio를 낮은 기준에서 높은 기준으로 점진적으로 올린다.

기본 범위:

- `0.20 -> 0.50`

즉:

- 처음엔 “조금만 넣어도 성공”
- 나중엔 “절반 이상 넣어야 성공”

구조다.

---

## 19. Done 조건

episode는 아래 조건들로 종료될 수 있다.

1. cup이 workspace 밖으로 벗어남
2. cup이 바닥 아래로 떨어짐
3. tip force가 일정 시간 사라짐 (drop으로 판단)
4. success 달성
5. source cup이 거의 비고 그 상태가 일정 시간 유지됨
6. bead 물리 폭발 감지 (`z < -0.5`)
7. episode time limit 도달

발표 관점에서 중요한 해석:

- “끝까지 100% 다 붓기만 하면 끝”이 아니라,
  **물리적으로 붕괴하거나 이미 source가 다 비어도 episode는 종료된다.**

---

## 20. PPO / PPO-LSTM 설정

현재 디렉터리에는 두 가지 agent config가 있다.

1. `rl_games_ppo_cfg.yaml`
2. `rl_games_ppo_lstm_cfg.yaml`

### 20.1 MLP PPO

특징:

- MLP `[512, 256, 128]`
- actor-critic shared trunk
- asymmetric central value 사용

### 20.2 LSTM PPO

특징:

- actor side encoder 뒤에 LSTM 사용
- `rnn units = 512`
- `seq_length = 8`
- `horizon_length = 32`

왜 LSTM이 중요한가:

- pouring은 one-step reflex task가 아니다.
- transport -> align -> tilt -> transfer의 장기 순차 구조가 있다.

따라서 발표에서는:

> We also provide an LSTM PPO configuration because pouring is inherently sequential and benefits from temporal credit assignment.

라고 설명할 수 있다.

---

## 21. 현재 구현과 논문 프레이밍의 차이

이 부분을 분명히 해두는 것이 중요하다.

### 현재 `v3`에서 직접 구현된 것

- warmstart reset
- demo pose reference bank
- demo-guided reward shaping
- asymmetric actor/critic observation
- bead proxy fluid
- staged reward decomposition
- ADR
- PPO / PPO-LSTM training config

### 사용자가 논문 초록에서 말하고 싶은 더 큰 프레이밍

- teleop demo 10개 수집
- digital twin conversion
- augmentation
- BC prior policy
- RL actor를 prior로 초기화

현재 이 디렉터리만 보면:

- `demo-guided RL`은 분명히 구현되어 있음
- 하지만 `BC prior`와 `augmentation pipeline`은 이 폴더의 핵심 실행 코드에서 직접 보이지 않음

따라서 발표에서는 아래처럼 구분하는 것이 안전하다.

> The current `pour_v3` implementation already realizes demonstration-guided reward shaping and stage-wise RL.
> The broader demonstration-augmented framework, including explicit BC priors and augmentation, is the intended research extension.

---

## 22. 발표용 핵심 문장 정리

### 22.1 reward 관련

> We designed the reward to follow the causal order of pouring rather than relying on a single sparse success signal.

> Stage A secures a physically meaningful workspace and arm posture, while Stage B refines the rim position, height, and tilt required for actual transfer.

> Geometry alone is insufficient, so bead-based reward terms bridge the gap between pose alignment and real material transfer.

### 22.2 demo data 관련

> We collect about ten teleoperated demonstrations and convert them into an Isaac-aligned digital twin representation.

> The current implementation does not directly imitate demo actions. Instead, it extracts compact pose statistics and uses them as reward-side guidance.

### 22.3 network / observation 관련

> The actor and critic do not see the same world. The actor receives a compact noisy observation for deployable policy learning, while the critic receives privileged state for stable value estimation.

---

## 23. 포스터/발표에 넣기 좋은 도식 제안

### 도식 1: Reward와 Stage A/B

```text
Stage A
workspace acquisition
= grasp hold + transport + demo posture

Stage B
pour-point tuning
= rim XY + rim Z + tilt release + tilt depth

Stage C
actual transfer
= bead-near + bead-in + mouth-cross + success - spill
```

### 도식 2: Demo data pipeline

```text
Remote teleoperation
-> trajectory logging
-> Isaac digital twin alignment
-> pour-phase extraction
-> tensorized demo bank
-> reward-side demonstration guidance
```

### 도식 3: Network / observation

```text
Isaac simulation state
-> Actor obs (60D, noisy, partial)
-> Policy action (6D palm + 5D finger)
-> Fabrics IK
-> rollout

Isaac full state
-> Critic obs (140D, privileged)
-> value learning
```

---

## 24. 결과 설명 시 주의할 점

현재 사용자가 말한 중간 결과:

- 약 `40%` transfer
- 약 `40%` spill/loss
- 약 `20%` source cup residual

이 수치는 발표에서 다음처럼 표현하는 것이 안전하다.

> Preliminary convergence behavior indicates partial transfer with substantial remaining spill, suggesting that the current policy has learned the basic transport-to-pour chain but still needs stronger refinement in Stage B and Stage C.

즉:

- 실패라고 말할 필요는 없고
- “기본 chain은 배웠지만 final refinement가 더 필요하다”고 정리하는 편이 좋다.

---

## 25. 최종 정리

현재 `pour_v3`는 다음 문장으로 요약할 수 있다.

> `pour_v3` is a warmstarted, demonstration-guided, stage-wise RL pouring task in which palm-space control and Fabrics IK first secure a feasible workspace, and bead-aware rewards then refine pour geometry into actual transfer.

한국어로 말하면:

> `pour_v3`는 warmstart와 demo-guided shaping을 바탕으로, palm-space 제어와 Fabrics IK를 통해 먼저 붓기 가능한 workspace를 확보하고, 이후 bead-aware reward를 통해 실제 전달이 일어나는 pour geometry를 학습하는 stage-wise RL task이다.

이 문장을 중심축으로 두고,

- reward는 왜 그렇게 쪼갰는지
- demo가 agent에 어떻게 들어가는지
- actor와 critic이 무엇을 보는지

를 설명하면 발표 구조가 가장 자연스럽다.

