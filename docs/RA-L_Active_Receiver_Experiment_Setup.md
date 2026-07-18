# RA-L Active-Receiver Bimanual Pouring 실험 세팅

## 1. 핵심 연구 가설

> Active receiver와 feasible-posture prior가 제한된 작업공간·관절 한계·시각 오차 아래에서 물붓기 성공 영역과 회복 성능을 확장한다.

논문의 중심은 단순한 양팔 물붓기 성공이 아니라, 수용컵을 능동 조정하는 정책이 고정 또는 scripted receiver보다 기하학적 허용범위와 외란 회복 성능을 확장한다는 점을 실험으로 증명하는 것이다.

## 1.5 코드 대응 및 선행 구현 현황

본 실험의 **기준 환경은 `tesollo/both/pour_sensor`**(양팔 active-receiver)이다. `right/pour_v1`은 왼팔 제어 action이 없는 단팔 환경(action 12D)이므로 **historical baseline으로만** 인용하고, 논문 직접 비교는 전부 `pour_sensor`에서 수행한다.

### 1.5.1 코드로 뒷받침되는 설계 (검증 완료)

| 설계 항목 | 실제 코드 근거 |
|---|---|
| M4 Active Receiver = "RL 3D TCP" | `pour_right_constants.py`: `NUM_LEFT_TCP_ACTION=3`, `action[12:15]` = 왼팔 TCP 위치 delta → DifferentialIK(DLS). orientation upright 고정, target 컵 kinematic-follow |
| receiver upright 고정 (§3.1) | `env.py`: `left_tcp_fixed_quat_b` = rest upright quat 고정 |
| Posture-prior 2×2 (P0~P3, §6) | `pour_right_env_cfg.py`: `nullspace_baseline: str`(`robot_start`/`demo`) × `enable_demo_pose_reward: bool` — 두 플래그가 §6 매트릭스와 1:1 매핑 |
| Demo = feasible posture prior | `nullspace_baseline="demo"`는 BC가 아니라 nullspace default_config bias(`NULLSPACE_OFFSET_ARM`) + privileged critic obs. §6 해석과 일치 |
| pour_v1 = 단팔 historical baseline | `pour_v1/constants`: `NUM_ACTIONS=12` (왼팔 action 없음) |
| spill / transfer / success-spill 지표 | `env.py`: `_spill_ratio`, `success_spill_max` 로깅 존재 |

### 1.5.2 선행 구현 현황

1. **`receiver_control_mode` config — ✅ 구현 완료.** `pour_right_env_cfg.py`에 flag 추가, `env.py` `_pre_physics_step`에서 3분기 처리. 기본값 `learned`(M4)는 기존 학습과 완전 동일(scale=1·delay=0).
   - `learned` — 정책 action[12:15] 누적 (M4)
   - `frozen` — 왼팔 TCP rest 고정 (M0, EXP-2 freeze)
   - `scripted` — source pour-point 추종 (M2, §3.1)
   - EXP-2 necessity: `receiver_action_scale`(0.5 등), `receiver_action_delay_steps`(100 ms≈6 step)
   - 계약 테스트: `tests/test_v6_ablation_flags.py` (7 pass). 실행별 주입은 hydra override.
2. **M2 scripted receiver — ✅ 기본 구현.** `scripted` 모드가 pour-point 아래(offset/clearance)로 receiver를 추종. ⚠️ **sim 정성 검증 후** M2 수치 신뢰(offset/clearance 튜닝 필요할 수 있음).
3. **복합 성공 지표 `Success@50/40`·`Success@80/15` — ⬜ 미구현.** 현재 `log/adr_ep_success_rate`·`log/spill_ratio`·`log/success_fill_ratio`만 로깅. §11.1 정의에 맞춰 평가 스크립트에서 집계하거나 임계값 확정 필요.
4. **manifest / evaluation bank — 🔶 부분.** 결과 수집기 `scripts/experiments/ral_collect.py` 구현. 고정 초기상태 bank·manifest 자동필드는 미구현(`record_test_snapshot.py` 확장 필요).

### 1.5.2.1 실행 방법

```bash
# EXP-1 (M4→M0→M2 순차 학습, seed 42, GPU 0). server 학습 env에서:
CUDA_VISIBLE_DEVICES=0 NUM_ENVS=2048 ./scripts/experiments/ral_receiver_ablation.sh 42
#   두 GPU 병렬은 seed별 별도 셸: GPU0=seed42, GPU1=seed43
#   부분집합:  METHODS="M0 M2" ./scripts/experiments/ral_receiver_ablation.sh 42

# 단일 실행(직접):  ./train.sh open-tesol_b_pour_sensor-lstm M0_C0_s42 \
#                     --num_envs 2048 --seed 42 env.receiver_control_mode=frozen

# 결과 비교표:
python3 scripts/experiments/ral_collect.py            # 전체
python3 scripts/experiments/ral_collect.py --filter C0
python3 scripts/experiments/ral_collect.py --list-tags   # 사용 가능한 지표 tag 확인

# EXP-2 necessity (학습된 M4 체크포인트를 eval-time override):
#   play 시  env.receiver_control_mode=frozen  또는  env.receiver_action_scale=0.5
#            env.receiver_action_delay_steps=6
```

### 1.5.3 receiver 능력 상한 (실험 해석 주의)

`left_tcp_workspace_range = (0.08, 0.08, 0.08)` — 왼팔 TCP는 rest 기준 **±8 cm 박스** 안에서만 움직이고, step당 최대 `left_tcp_action_delta_m = 0.01 m`(1 cm)로 제한된다. **C1(workspace boundary)·REAL-2의 target 배치는 이 ±8 cm 보정 여력을 기준으로** 설계·해석해야 하며, 그 이상 벗어난 target은 active receiver로도 도달 불가임을 명시한다.

## 2. 공통 실험 설정

모든 비교 방법에서 다음 항목을 동일하게 고정한다.

- 동일 source/receiver cup 형상
- 동일 bead 수 또는 물의 초기 질량
- 동일 초기 자세 및 target 위치 분포
- 동일 episode 길이와 성공 판정
- 동일 observation noise 및 domain randomization
- 동일 PPO/LSTM 구조와 학습 frame
- 동일 training/evaluation seed
- 동일 평가 스크립트
- 동일 source-arm reward
- 동일 Fabric/IK 설정
- 방법별 차이는 receiver 제어 및 posture prior로 제한

각 결과에는 다음 manifest를 기록한다.

```text
experiment_id
git_commit
checkpoint
action_dimension
config
training_seed
evaluation_seed
training_frames
evaluation_episodes
camera/perception_version
robot/calibration_version
```

현재 `pour_sensor` 평가 문서에는 서로 다른 체크포인트 결과가 존재하므로 결과-체크포인트 매핑을 먼저 분리한다.

## 3. 비교 방법

| ID | 방법 | Receiver | Posture prior | 목적 |
|---|---|---|---|---|
| M0 | Fixed Receiver | 고정 | 동일 | 공정한 단팔형 기준선 |
| M2 | Scripted Receiver | 기하학 규칙 기반 | 동일 | 단순 controller로 충분한지 검증 |
| M4 | Learned Active Receiver | RL 3D TCP | 동일 | 제안 방법 |
| M5 | Oracle Receiver | GT pose 기반 추종 | 동일 | 성능 상한선, 선택사항 |

M0/M2/M4는 모두 `pour_sensor`의 동일 env·동일 15D action 공간을 쓰고, `receiver_control_mode`(§1.5.2-1)로 왼팔 TCP action(`action[12:15]`)의 처리만 달리한다: M0=`frozen`, M2=`scripted`, M4=`learned`. Receiver는 세 방법 모두 upright orientation 고정·±8 cm workspace(§1.5.3)로 동일하다.

기존 `pour_v1` 결과는 historical baseline으로 제시하되, 논문의 직접 비교에는 동일한 `pour_sensor` 환경에서 receiver action만 고정한 M0를 사용한다.

### 3.1 Scripted Receiver 예시

```text
receiver_target_xy =
    source_pour_point_xy
    + desired_offset_xy

receiver_target_z =
    source_pour_point_z
    - desired_clearance
```

- 속도와 workspace 제한은 M4와 동일하게 설정 (`left_tcp_action_delta_m`=1 cm/step, `left_tcp_workspace_range`=±8 cm)
- receiver orientation은 upright로 고정 (`left_tcp_fixed_quat_b`)
- vision 또는 GT 입력 조건도 M4와 동일하게 적용
- **구현 위치**: `receiver_control_mode="scripted"` 분기에서 `left_tcp_target_pos_b`를 위 식으로 직접 설정 (§1.5.2-2)

## 4. 평가 조건

### 4.1 핵심 조건

| ID | 조건 | 설정 | 검증할 주장 |
|---|---|---|---|
| C0 | Nominal | 학습분포 중앙 | 기본 물붓기 성능 |
| C1 | Workspace Boundary | target을 source-arm 도달 한계 주변에 배치 | active receiver의 성공영역 확장 |
| C3 | Target Disturbance | 붓기 전·도중 receiver를 3-5 cm 이동 | 폐루프 회복 능력 |

### 4.2 추가 일반화 조건

| ID | 조건 | 권장 범위 |
|---|---|---|
| C2 | Cup geometry | 입구 직경·높이 ±15-25% |
| C4 | Perception noise | 위치 5/10/20 mm, 회전 2/5/10 deg |
| C5 | Latency/dropout | 50/100/200 ms, 5/10/20% dropout |
| C6 | Fill level | 30/50/70% |
| C7 | Dynamics | 질량·마찰·액체/비드 특성 변화 |
| C8 | Occlusion/lighting | 손 가림, 배경 및 조명 변화 |

논문 메인 표에는 `M0/M2/M4 × C0/C1/C3`을 배치하고 나머지는 일반화 표 또는 부록으로 구성한다.

## 5. 시뮬레이션 본실험

### EXP-1. Active Receiver 효과

```text
M0-C0, M2-C0, M4-C0
M0-C1, M2-C1, M4-C1
M0-C3, M2-C3, M4-C3
```

권장 규모:

- 방법당 학습 seed: 5개
- 평가: 각 seed·조건당 500-1,000 episodes
- 모든 방법에 동일 초기상태 evaluation bank 적용

핵심 분석:

- M4가 C1에서 성공 가능한 target 영역을 얼마나 확장하는가
- M4가 C3에서 M0/M2보다 얼마나 빠르게 복구하는가
- receiver 이동으로 spill 또는 cycle time이 악화되지 않는가

### EXP-2. Receiver Necessity Test

학습된 M4 정책을 다음 조건으로 평가한다.

- 정상 receiver action
- receiver action을 전부 0으로 설정
- receiver action에 100-200 ms 지연 적용
- receiver action scale을 50%로 축소
- source action은 유지하고 receiver만 고정

성능 변화가 거의 없다면 정책이 receiver를 실제로 활용하지 않은 것이므로 Active-Receiver Contribution을 지지할 수 없다.

## 6. Feasible-Posture Prior Ablation

동일한 M4 receiver 설정에서 다음 2×2 실험을 수행한다.

| ID | Nullspace baseline | Demo pose reward | config 값 |
|---|---|---:|---|
| P0 | robot start | OFF | `nullspace_baseline="robot_start"`, `enable_demo_pose_reward=False` |
| P1 | demo feasible posture | OFF | `nullspace_baseline="demo"`, `enable_demo_pose_reward=False` |
| P2 | robot start | ON | `nullspace_baseline="robot_start"`, `enable_demo_pose_reward=True` |
| P3 | demo feasible posture | ON | `nullspace_baseline="demo"`, `enable_demo_pose_reward=True` |

> **기본값 주의**: 현재 코드 기본값은 `nullspace_baseline="demo"`, `enable_demo_pose_reward=False` → **P1 셀**이다. P0(순수 DRL)를 돌리려면 `robot_start`로 명시 전환해야 한다. 계약 테스트 `tests/test_v6_ablation_flags.py`로 플래그 무결성 확인.

핵심 비교는 `P1 > P0`이다. Demonstration은 행동복제 데이터가 아니라 실제 실행 가능한 pre-pour posture 및 redundancy direction을 제공하는 prior로 해석한다.

측정 항목:

- 80% 성공률 도달 frame
- wall-clock 학습시간
- 5-seed 성공률 분산
- joint-limit saturation 비율
- minimum joint margin
- infeasible IK target 비율
- Fabric tracking error
- pour-point alignment error
- transfer 및 spill
- deep-tilt 도달률

추가로 `demo critic feature ON/OFF`를 비교하여 privileged critic의 효과를 분리한다.

## 7. Network/Controller Ablation

| 우선순위 | 비교 | 검증 내용 |
|---:|---|---|
| 1 | LSTM vs MLP | 지연·부분관측·액체 동역학에서 recurrence 필요성 |
| 2 | Privileged critic ON/OFF | asymmetric actor-critic 효과 |
| 3 | Learned receiver vs scripted | 학습 협응의 필요성 |
| 4 | Task-space action vs direct joint action | 구조화된 행동공간 효과 |
| 5 | Nullspace action ON/OFF | redundancy 제어 효과 |
| 6 | Observation noise ADR ON/OFF | Sim-to-Real 강건성 |

RA-L 본문에는 우선순위 1-3과 posture-prior ablation을 포함하고 나머지는 부록으로 배치한다.

## 8. Vision Sim-to-Real 실험

### 8.1 Vision 입력 비교

| ID | 입력 | 역할 |
|---|---|---|
| V0 | GT state | teacher 및 성능 상한 |
| V1 | 추정된 rim 3D pose + proprioception | 단순 실기 정책 |
| V2 | segmented RGB-D crop + proprioception | dense vision 비교 |
| V3 | rim pose + RGB-D crop + proprioception | 추천 최종 방법 |

### 8.2 권장 학습 순서

1. 기존 state teacher 고정
2. Simulation에서 RGB-D 및 rim pose 생성
3. Teacher action을 visual student로 distillation
4. Student rollout 기반 DAgger
5. Privileged critic을 이용한 RL fine-tuning
6. Visual+dynamics randomization 적용
7. 실기 zero-shot 또는 최소 calibration 후 평가

### 8.3 Vision randomization

- camera extrinsic 위치·회전
- focal length 및 depth scale
- RGB brightness/contrast/color
- 배경과 distractor
- cup texture 및 transparency
- segmentation mask erosion/dilation
- depth hole 및 Gaussian noise
- 50-200 ms latency
- frame dropout
- source/target rim 일부 가림
- tracking confidence 감소

### 8.4 Vision 성능 지표

- rim-center position error
- rim-normal angular error
- perception failure rate
- policy success vs pose-error curve
- latency vs success curve
- visual student/state teacher 성능비
- Sim-to-Real gap

## 9. 실제 로봇 장비 세팅

- RGB-D 카메라 및 고정 마운트
- camera-robot extrinsic calibration
- source/receiver cup 세트
- 여러 크기의 OOD cup
- receiver 아래 질량 측정 저울
- spill 수집 tray 또는 별도 저울
- source 초기 물 질량 측정 장치
- target disturbance용 위치 마커 또는 이동 jig
- overhead recording camera
- robot state, vision 및 질량 측정 timestamp 동기화
- 방수 커버, 비상정지 및 누수 보호

AprilTag 또는 marker는 policy 입력이 아니라 ground-truth 평가용으로만 사용한다.

## 10. 실제 로봇 본실험

### REAL-1. Nominal

- M0, M2, M4
- 각 방법 최소 30회
- 동일한 초기상태 순서 사용
- transfer, spill, success, cycle time 측정

### REAL-2. Workspace Boundary

- source-only가 어렵지만 receiver 이동으로 보완 가능한 위치 3단계
- 각 방법·위치당 20-30회
- 위치별 성공률 heatmap 생성

### REAL-3. Target Disturbance

- 붓기 시작 전 이동
- 정렬 완료 후 이동
- 실제 유출 시작 후 이동
- 이동 방향: 좌/우/전/후
- 이동 크기: 3 cm 및 5 cm

측정 항목:

- disturbance 후 recovery time
- 최대 rim error
- capture corridor 재진입 여부
- spill 증가량
- receiver TCP 이동거리

### REAL-4. Vision Robustness

- 정상 조명
- 어두운 조명
- 배경 변경
- 부분 가림
- camera calibration perturbation

### REAL-5. OOD Generalization

- unseen source cup 2-3개
- unseen receiver cup 2-3개
- fill level 30/50/70%
- 조건당 최소 10회

핵심 실기 매트릭스 `M0/M2/M4 × C0/C1/C3 × 30`은 총 270 trials이다.

## 11. 기록 지표

### 11.1 물붓기 품질

- transfer efficiency
- spill ratio
- target 최종 질량
- source 잔여 질량
- 완전실패율
- cycle time
- `Success@50/40` (transfer efficiency ≥50% **and** spill ≤40%)
- `Success@80/15` (transfer efficiency ≥80% **and** spill ≤15%)

> **구현 상태**: 현재 코드는 `_spill_ratio`·transfer efficiency를 개별 로깅하나 위 복합 임계 지표는 미정의(§1.5.2-3). 평가 스크립트에서 두 원지표로 집계하거나, 임계값을 현행 `success_spill_max` 정의에 맞춰 확정한다.

### 11.2 Active Receiver

- receiver TCP 이동거리 및 최대 속도
- source-target rim XY/Z error
- capture corridor 유지시간
- disturbance recovery time
- workspace coverage
- receiver action과 rim error의 상관관계

### 11.3 제어 제약

- joint-limit saturation
- minimum joint margin
- IK failure rate
- unreachable target rate
- Fabric tracking error
- minimum inter-arm distance

현재 `enabled_self_collisions=False`이므로 minimum link distance는 안전 진단 지표로만 보고하고 self-collision avoidance Contribution으로 해석하지 않는다.

### 11.4 학습 성능

- 80% 성공률 도달 frame
- 최종 성능
- seed variance
- wall-clock
- entropy
- train-deterministic evaluation gap

## 12. Figure/Table 생성 목록

- Fig. 1: 전체 Vision-Policy-Fabric/IK 구조
- Fig. 2: source/receiver 역할과 행동공간
- Fig. 3: M0/M2/M4 trajectory 비교
- Fig. 4: target 위치별 성공률 heatmap
- Fig. 5: disturbance recovery 시계열
- Fig. 6: pose noise/latency 대비 성공률
- Fig. 7: Sim-Real gap
- Table I: M0/M2/M4 × C0/C1/C3
- Table II: posture-prior 2×2
- Table III: V0/V1/V2/V3
- Table IV: 실기 OOD 결과
- Table V: 실패 원인 분류

## 13. 권장 실행 순서

0. **선행 구현 (§1.5.2)** — `receiver_control_mode`·M2 scripted·결과 수집기는 ✅ 완료(§1.5.2.1). 잔여: M2 scripted **sim 검증**, `Success@` 집계, 고정 eval bank.
1. 체크포인트-코드-설정 manifest 정리
2. 고정 evaluation bank 생성
3. M0/M2/M4 공정 비교
4. receiver freeze test
5. C1 workspace-boundary 실험
6. C3 target-disturbance 실험
7. posture-prior 2×2 ablation
8. LSTM/critic ablation
9. vision pose student 구축
10. vision randomization 및 stress test
11. 실기 nominal 평가
12. 실기 boundary/disturbance 평가
13. OOD cup 및 fill-level 평가
14. 통계, figure 및 실패 영상 추출

## 14. Go/No-Go 기준

- C1 또는 C3에서 M4가 M0보다 성공률 15%p 이상 개선
- 성능 차이의 95% confidence interval이 0을 넘음
- M4의 spill 악화가 M0 대비 5%p 이내
- receiver-freeze ablation에서 성능이 명확히 감소
- visual student가 state teacher 성능의 90% 이상 유지
- 실기 성공률 70% 이상
- Sim-to-Real gap 15%p 이하

Active receiver가 nominal 조건에서만 움직이고 workspace-boundary 또는 disturbance 조건에서 회복하지 못하면 Active-Receiver Contribution은 No-Go로 판단한다.
