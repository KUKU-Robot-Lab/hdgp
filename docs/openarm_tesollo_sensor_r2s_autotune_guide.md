# OpenArm Real-to-Sim Autotune 적용 가이드

이 문서는 논문의 **Algorithm 1: Real-to-Sim Autotune Module**을 현재 로컬 OpenArm 스택에 맞게 적용하기 위한 최소 구현 가이드다.

대상 asset은 두 개다.

```text
/home/user/rl_ws/hdgp/assets/openarm_tesollo_sensor
/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1
```

목표는 RL 정책을 바로 학습하는 것이 아니라, RL 전에 실제 로봇과 Isaac Lab 로봇의 관절 응답 차이를 줄이는 **system identification / actuator calibration** 파이프라인을 만드는 것이다. 논문에서 말하는 real-to-sim autotune은 이 단계에 해당한다.

---

## 1. 현재 스택 기준 결론

최소 구현은 USD 파일을 직접 대량 수정하는 방식이 아니다.

1. `/home/user/rl_ws/urdf`에서 만든 URDF를 source-of-truth로 유지한다.
2. `/home/user/rl_ws/teleopration_openarm_tesollo`에서 실제 로봇 identification bag/HDF5를 만든다.
3. `real2sim_actuator_calibration.py`로 actuator group별 seed JSON을 만든다.
4. `OPENARM_REAL2SIM_ACTUATOR_CALIBRATION` 환경변수로 JSON을 `hdgp` Isaac Lab env에 주입한다.
5. 논문 Algorithm 1의 population search는 이 seed JSON을 중심으로 후보 scale을 만들고, Isaac Lab 병렬 env에서 같은 trajectory를 replay하여 refine한다.

즉, 1차 적용 경로는 다음이다.

```text
real robot log
  -> HDF5 command/response pair
  -> real2sim_actuator_calibration.json
  -> Isaac Lab ImplicitActuatorCfg override
  -> parallel replay error ranking
  -> best calibration JSON
```

USD physics layer는 최종 고정본을 만들 때만 export 대상으로 둔다. 반복 탐색 단계에서는 `ArticulationCfg.actuators` runtime override가 더 안전하고 빠르다.

---

## 2. 논문 Algorithm 1을 내 로봇에 매핑

논문 알고리즘은 다음 구조다.

```text
E: tune할 environment parameter 집합
N: calibration action sequence 개수
R: real robot hardware environment
M: initial robot model file
P: parameter search space
S_i: candidate parameter p_i가 적용된 sim environment
J: real/sim에 공통 입력할 joint target sequence
error: real tracking과 sim tracking 사이의 MSE
best_params: error가 가장 작은 parameter set
```

현재 로컬 스택에서는 이렇게 매핑한다.

| 논문 변수 | 현재 스택 적용 |
|---|---|
| `E` | actuator stiffness, actuator damping, joint friction, velocity/effort limit, delay steps |
| `N` | step, ramp, hold, low-frequency sine, task-replay validation sequence |
| `R` | `/home/user/rl_ws/teleopration_openarm_tesollo`로 구동하는 실제 OpenArm + hand |
| `M` | `/home/user/rl_ws/urdf`에서 만든 URDF와 `hdgp/assets/*/*.usd` |
| `P` | seed JSON 주변 scale range, 예: stiffness 0.8-1.25, damping 0.7-1.5 |
| `S_i` | `hdgp/source/openarm` Isaac Lab env의 parallel candidate env |
| `J` | 동일한 `q_cmd(t)` 또는 task replay action sequence |
| `R_track` | `q_cmd`, `q_real`, `dq_real`, optional effort/contact/sensor |
| `S_track` | `q_cmd`, `q_sim`, `dq_sim`, optional contact/sensor |
| `error` | group별 weighted MSE + delay penalty + validation replay error |
| `best_params` | `schema_version: 1` calibration JSON |

핵심은 논문처럼 “여러 sim 후보를 만들고 실제 tracking과 가장 가까운 후보를 고른다”는 구조만 가져오고, 현재 repo에서는 그 후보를 USD 복사본이 아니라 actuator config JSON으로 표현하는 것이다.

---

## 3. Source-of-Truth 규칙

### 3.1 URDF 생성 위치

사용자 기준 URDF 생성 repo:

```text
/home/user/rl_ws/urdf
```

여기에서 만든 URDF와 mesh가 asset의 원본이다. `hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.urdf`는 `/home/user/rl_ws/urdf/openarm_description`와 `/home/user/rl_ws/urdf/RH56F1` mesh를 참조한다.

### 3.2 HDGP asset 위치

RL/Isaac Lab에서 spawn되는 asset:

```text
/home/user/rl_ws/hdgp/assets/openarm_tesollo_sensor/openarm_tesollo_sensor.usd
/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.usd
```

각 asset의 physics layer:

```text
/home/user/rl_ws/hdgp/assets/openarm_tesollo_sensor/configuration/openarm_tesollo_sensor_physics.usd
/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1/configuration/openarm_bi_rh56f1_physics.usd
```

초기 autotune에서는 이 USD 파일을 직접 patch하지 않는다. Isaac Lab config에서 actuator 값을 override한 뒤, 안정화된 best JSON만 남긴다.

### 3.3 FABRICS의 역할

FABRICS는 동역학 파라미터 식별 대상이 아니라 **kinematic contract**다.

관련 경로:

```text
/home/user/rl_ws/hdgp/source/FABRICS/src/fabrics_sim/fabrics/openarm_tesollo_pose_fabric.py
/home/user/rl_ws/hdgp/source/FABRICS/src/fabrics_sim/fabrics/openarm_rh56f1_pose_fabric.py
/home/user/rl_ws/hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf
/home/user/rl_ws/hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/openarm_rh56f1.urdf
```

따라서 autotune 구현 시 반드시 확인해야 하는 것은 다음이다.

```text
URDF joint order
Isaac Lab articulation DOF order
FABRICS joint order
teleoperation HDF5 joint_names attribute
calibration JSON actuator group regex
```

이 다섯 개가 어긋나면 MSE가 작아져도 실제 정책에는 도움이 되지 않는다.

### 3.4 디렉토리 통합 원칙

세 repo를 물리적으로 한 디렉토리로 합치지 않는다.

```text
/home/user/rl_ws/urdf                         # URDF/mesh 원본 생성 위치
/home/user/rl_ws/teleopration_openarm_tesollo # 실제 로봇 구동/로그/HDF5 생성 위치
/home/user/rl_ws/hdgp                         # Isaac Lab/RL/autotune 실행 위치
```

합치는 대상은 파일 트리가 아니라 **인터페이스**다.

| 소스 | hdgp로 가져올 것 | 가져오지 말 것 |
|---|---|---|
| `/home/user/rl_ws/urdf` | 최종 URDF path, mesh path, joint/link 이름 contract | build/install/log 전체 복사 |
| `/home/user/rl_ws/teleopration_openarm_tesollo` | HDF5 dataset, calibration JSON, topic/joint mapping 문서 | ROS2 driver source 전체 vendoring |
| `/home/user/rl_ws/hdgp/source/FABRICS` | FABRICS용 URDF, joint order, FK/IK frame 이름 | actuator 동역학 튜닝 로직 |
| `/home/user/rl_ws/hdgp/assets/*` | Isaac Lab spawn용 USD/URDF asset | 매 탐색마다 USD 복사본 생성 |

최종 통합 구조는 다음처럼 둔다.

```text
/home/user/rl_ws
├── urdf/
│   ├── openarm_description/...
│   ├── RH56F1/...
│   └── delto_m_ros2/...
│
├── teleopration_openarm_tesollo/
│   ├── bags/real2sim_identification/       # raw DB3 보존
│   ├── datasets/                           # HDF5 변환본
│   ├── REAL2SIM_ACTUATOR_CALIBRATION.md
│   └── src/openarm_teleop/script/
│       ├── record_real2sim_identification_bag.sh
│       └── real2sim_actuator_calibration.py
│
└── hdgp/
    ├── assets/
    │   ├── openarm_tesollo_sensor/
    │   └── openarm_bi_rh56f1/
    ├── source/
    │   ├── FABRICS/
    │   └── openarm/
    ├── scripts/r2s_autotune/                # 새로 추가할 autotune 실행 코드
    └── logs/r2s_autotune/
        ├── seeds/                           # teleop repo에서 만든 seed JSON 복사/스냅샷
        ├── runs/                            # 후보 replay 결과
        └── results/                         # best calibration JSON
```

권장 데이터 흐름:

```text
urdf
  -> hdgp/assets/*.urdf, *.usd 생성/갱신
  -> FABRICS/models/robots/urdf/* 갱신

teleopration_openarm_tesollo
  -> bags/real2sim_identification/*.db3
  -> datasets/real2sim_identification_*.hdf5
  -> datasets/real2sim_actuator_calibration.json

hdgp
  -> logs/r2s_autotune/seeds/*.json
  -> scripts/r2s_autotune/run_parallel_replay.py
  -> logs/r2s_autotune/results/*_best_calibration.json
  -> OPENARM_REAL2SIM_ACTUATOR_CALIBRATION로 RL env에 적용
```

파일을 복사해야 하는 경우는 두 가지뿐이다.

1. `/home/user/rl_ws/teleopration_openarm_tesollo/datasets/*.hdf5` 또는 calibration JSON을 `hdgp/logs/r2s_autotune/seeds/`에 실험 스냅샷으로 복사한다.
2. `/home/user/rl_ws/urdf`에서 로봇 구조가 바뀐 경우, `hdgp/assets/*/*.urdf`, `hdgp/assets/*/*.usd`, `FABRICS/src/fabrics_sim/models/robots/urdf/*`를 같은 commit/실험 단위로 갱신한다.

그 외에는 symlink나 전체 repo 복사를 만들지 않는다. 특히 `teleopration_openarm_tesollo/build`, `install`, `log`, raw ROS2 driver source를 `hdgp` 안으로 복사하지 않는다.

---

## 4. Real 데이터 수집

이미 teleoperation repo에 real-to-sim 수집 가이드와 스크립트가 있다.

```text
/home/user/rl_ws/teleopration_openarm_tesollo/REAL2SIM_ACTUATOR_CALIBRATION.md
/home/user/rl_ws/teleopration_openarm_tesollo/src/openarm_teleop/script/record_real2sim_identification_bag.sh
```

기본 recording 명령:

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo
source /opt/ros/humble/setup.bash
source install/setup.bash

ROS_DOMAIN_ID=126 ./src/openarm_teleop/script/record_real2sim_identification_bag.sh \
  ./bags/real2sim_identification/run_001 20
```

기록되는 주요 토픽:

```text
/openarm/left/joint_states
/openarm/right/joint_states
/openarm/left/leader/gripper_state
/openarm/right/leader/gripper_state
/dg5f_right/rj_dg_pospid/reference
/dg5f_right/joint_states
/tesollo/right/joint_states
/tesollo/right/sensor
/tf
/tf_static
```

주의점:

- teleoperation task demo는 validation에는 좋지만, stiffness/damping/friction/delay를 분리해 식별하기에는 부족하다.
- identification용으로는 작은 step, ramp, hold, return-to-neutral, 저속 sine을 별도 수집한다.
- Tesollo hand는 command reference가 100 Hz라 100 Hz HDF5가 실용적이다.
- OpenArm arm transient를 더 정확히 보려면 raw DB3를 보존하고 500-1000 Hz 분석용 변환본을 별도로 둔다.

---

## 5. Seed Calibration JSON 생성

기존 스크립트:

```text
/home/user/rl_ws/teleopration_openarm_tesollo/src/openarm_teleop/script/real2sim_actuator_calibration.py
```

출력 schema:

```json
{
  "schema_version": 1,
  "robot_asset": "openarm_tesollo_sensor",
  "source_dataset": "/abs/path/to/dataset.hdf5",
  "groups": {
    "tesollo_hand_curl": {
      "stiffness": 30.0,
      "damping": 5.0,
      "joint_friction": 0.0,
      "delay_steps": 0,
      "fit_error": {},
      "joint_metrics": {}
    }
  }
}
```

Tesollo hand curl seed 예시:

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo

python3 src/openarm_teleop/script/real2sim_actuator_calibration.py \
  --dataset datasets/real2sim_identification_100hz.hdf5 \
  --demo-key demo_0 \
  --group-name tesollo_hand_curl \
  --command-dataset obs/right_hand_reference_joint_pos \
  --measured-dataset obs/right_hand_joint_pos \
  --joint-name-regex 'rj_dg_[1-5]_2' \
  --defaults 30.0,5.0,7.5,3.14159,0.0 \
  --robot-asset openarm_tesollo_sensor \
  --output datasets/real2sim_actuator_calibration.json
```

이 스크립트는 한 번에 한 group을 쓴다. 여러 group을 한 파일에 합치려면 다음 중 하나로 구현한다.

1. 같은 추정 함수를 import하는 작은 merge script를 만든다.
2. group별 JSON을 만든 뒤 `groups` object만 병합한다.

최종 파일은 `schema_version: 1`과 `groups` object를 유지해야 한다. `hdgp`의 loader가 이 schema를 기대한다.

---

## 6. HDGP 적용 방식

`hdgp`에는 이미 calibration JSON을 읽어 `ImplicitActuatorCfg`에 넣는 패턴이 있다.

대표 파일:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v11/real2sim_actuator_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v11/grasp_right_env_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/real2sim_actuator_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env_cfg.py
```

적용 명령:

```bash
export OPENARM_REAL2SIM_ACTUATOR_CALIBRATION=/abs/path/to/real2sim_actuator_calibration.json
```

env cfg 내부에서는 다음 형태로 들어간다.

```python
ImplicitActuatorCfg(
    joint_names_expr=[...],
    **_actuator_params("tesollo_hand_curl", 30.0, 5.0),
)
```

이 방식에서 `best_params`는 USD가 아니라 JSON이다. 재현성은 다음 세 가지로 보장한다.

```text
calibration JSON 경로
source_dataset 경로
fit_error / joint_metrics
```

---

## 7. Tesollo 적용 기준

### 7.1 기본 actuator group

Tesollo right hand 기준으로 먼저 맞출 group:

| group | joint regex | 기본값 |
|---|---|---|
| `openarm_right_arm` | `openarm_right_joint[1-7]` | `400.0,80.0` |
| `openarm_left_arm` | `openarm_left_joint[1-7]` | `400.0,80.0` |
| `tesollo_hand_abduction` | `rj_dg_[1-5]_1` | `30.0,5.0` |
| `tesollo_hand_curl` | `rj_dg_[1-5]_2` | `30.0,5.0` |
| `tesollo_hand_pip` | `rj_dg_[1-5]_3` | `30.0,5.0` |
| `tesollo_hand_dip` | `rj_dg_[1-5]_4` | `30.0,5.0` |
| `openarm_left_gripper` | left gripper joint regex | `400.0,80.0` |

초기에는 `tesollo_hand_curl`부터 맞춘다. curl이 가장 task action과 직접 연결되고, hand response 차이를 크게 만든다.

### 7.2 Calibration sequence

Tesollo 초기 sequence:

```text
1. neutral hold 2s
2. curl group small step 0.15 rad
3. hold 2s
4. return-to-neutral
5. slow ramp open/close
6. same sequence at low/mid/high arm workspace posture
```

처음부터 contact sequence를 쓰지 않는다. 관절 동역학을 먼저 맞춘 뒤, contact/force sensor는 별도 validation으로 둔다.

### 7.3 HDGP target env

우선 적용할 env:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v11
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3
```

이 env들은 이미 `OPENARM_REAL2SIM_ACTUATOR_CALIBRATION` loader와 actuator randomization hook을 갖고 있다.

---

## 8. RH56F1 적용 기준

### 8.1 기본 actuator group

RH56F1은 underactuated hand라 Tesollo처럼 20 DOF 전체를 독립 튜닝하면 안 된다. 먼저 drive/mimic group으로 묶는다.

| group | joint regex | 기본값 |
|---|---|---|
| `openarm_right_arm` | `openarm_right_joint[1-7]` | `400.0,80.0` |
| `openarm_left_arm` | `openarm_left_joint[1-7]` | `400.0,80.0` |
| `rh56f1_right_drive` | `rh56f1_right_right_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint` | `30.0,5.0` |
| `rh56f1_right_mimic` | `rh56f1_right_right_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint` | mimic 추종용 |
| `rh56f1_left_drive` | `rh56f1_left_left_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint` | `30.0,5.0` |
| `rh56f1_left_mimic` | `rh56f1_left_left_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint` | mimic 추종용 |

기존 coverage test:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/tests/test_phase3_actuator_coverage.py
```

이 테스트가 말하는 38 DOF coverage를 깨면 안 된다.

### 8.2 FABRICS 주의점

RH56F1 FABRICS는 13 DOF 기준이다.

```text
7 DOF OpenArm right arm
6 DOF RH56F1 drive hand
```

반면 Isaac Lab asset은 양손/양팔 포함 38 DOF다. 따라서 calibration JSON group은 Isaac Lab articulation 기준으로 만들고, FABRICS는 reset/IK/FK target 생성의 joint order 검증 대상으로만 사용한다.

### 8.3 HDGP target env

우선 적용할 env:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1
```

`grasp_right_env_cfg.py`는 이미 `real2sim_actuator_cfg.py`를 통해 calibration JSON을 읽는다.

---

## 9. 논문식 Parallel Autotune Refinement

seed JSON 생성 후 논문 Algorithm 1을 다음처럼 구현한다.

### 9.1 Parameter space

seed를 중심으로 scale을 샘플링한다.

```yaml
population_size: 128
seed: 7
groups:
  openarm_right_arm:
    stiffness_scale: [0.8, 1.25]
    damping_scale: [0.7, 1.5]
    friction_scale: [0.7, 1.3]
  tesollo_hand_curl:
    stiffness_scale: [0.6, 1.6]
    damping_scale: [0.5, 2.0]
    friction_scale: [0.5, 2.0]
```

처음부터 joint별 독립 탐색을 하지 않는다. group 단위로 묶어야 식별 가능성과 탐색 비용이 맞는다.

### 9.2 Replay

각 candidate env는 같은 `q_cmd(t)`를 replay한다.

```text
candidate_000: seed * sampled_scale_000
candidate_001: seed * sampled_scale_001
...
candidate_K:   seed * sampled_scale_K
```

simulation output:

```python
sim_track = {
    "time": time,
    "q_cmd": q_cmd,
    "q_sim": q_sim,      # [K, T, J]
    "dq_sim": dq_sim,    # [K, T, J]
}
```

real output:

```python
real_track = {
    "time": time,
    "q_cmd": q_cmd,
    "q_real": q_real,    # [T, J]
    "dq_real": dq_real,  # [T, J]
}
```

### 9.3 Error

초기 objective:

```text
error =
  1.0  * MSE(q_sim, q_real)
+ 0.05 * MSE(dq_sim, dq_real)
+ 0.01 * delay_penalty
```

contact sensor와 FT sensor는 관절 동역학이 맞은 뒤 추가한다.

```text
phase 1: free-space actuator tracking
phase 2: light-contact validation
phase 3: task replay validation
```

### 9.4 Export

best candidate는 같은 schema로 저장한다.

```text
logs/r2s_autotune/results/openarm_tesollo_sensor_best_calibration.json
logs/r2s_autotune/results/openarm_bi_rh56f1_best_calibration.json
```

최종 RL 실행 전:

```bash
export OPENARM_REAL2SIM_ACTUATOR_CALIBRATION=/home/user/rl_ws/hdgp/logs/r2s_autotune/results/openarm_tesollo_sensor_best_calibration.json
```

---

## 10. 구현 작업 단위

### Task 0. 디렉토리 정리와 연결 방식 확정

목표:

```text
세 repo를 복사해서 섞지 않고, hdgp가 real log/URDF/FABRICS를 참조하는 방식으로 고정한다.
```

먼저 만들 디렉토리:

```bash
mkdir -p /home/user/rl_ws/hdgp/scripts/r2s_autotune/configs
mkdir -p /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds
mkdir -p /home/user/rl_ws/hdgp/logs/r2s_autotune/runs
mkdir -p /home/user/rl_ws/hdgp/logs/r2s_autotune/results
```

옮길 파일:

```text
teleopration_openarm_tesollo/datasets/real2sim_actuator_calibration.json
  -> hdgp/logs/r2s_autotune/seeds/openarm_tesollo_sensor_seed_calibration.json

teleopration_openarm_tesollo/datasets/real2sim_identification_100hz.hdf5
  -> hdgp/logs/r2s_autotune/seeds/real2sim_identification_100hz.hdf5
```

복사하지 않을 것:

```text
teleopration_openarm_tesollo/build/
teleopration_openarm_tesollo/install/
teleopration_openarm_tesollo/log/
teleopration_openarm_tesollo/src/openarm_ros2/
teleopration_openarm_tesollo/src/delto_m_ros2/
urdf/build/
urdf/install/
urdf/log/
```

왜 이렇게 하냐면, `hdgp`는 RL/Isaac Lab 실행 workspace고, teleoperation repo는 실제 로봇 ROS2 workspace다. 둘을 섞으면 ROS2 build 산출물, Isaac Lab Python path, FABRICS path가 서로 오염된다. 통합 단위는 “파일 복사”가 아니라 “calibration JSON과 HDF5 artifact”다.

최소 변경 파일:

```text
/home/user/rl_ws/hdgp/scripts/r2s_autotune/README.md
/home/user/rl_ws/hdgp/scripts/r2s_autotune/configs/autotune_ranges.yaml
/home/user/rl_ws/hdgp/scripts/r2s_autotune/configs/replay_sequences.yaml
/home/user/rl_ws/hdgp/scripts/r2s_autotune/load_real_track.py
/home/user/rl_ws/hdgp/scripts/r2s_autotune/sample_candidates.py
/home/user/rl_ws/hdgp/scripts/r2s_autotune/compute_tracking_error.py
/home/user/rl_ws/hdgp/scripts/r2s_autotune/export_best_calibration.py
/home/user/rl_ws/hdgp/scripts/r2s_autotune/run_parallel_replay.py
```

기존 `hdgp/source/openarm/.../real2sim_actuator_cfg.py`는 유지한다. 새 loader를 중복 작성하지 말고, 이 파일의 JSON schema에 맞춰 autotune 결과를 export한다.

### Task 1. 문서와 seed pipeline 고정

목표:

```text
teleoperation repo에서 real log -> HDF5 -> calibration JSON까지 안정화한다.
```

사용 파일:

```text
/home/user/rl_ws/teleopration_openarm_tesollo/REAL2SIM_ACTUATOR_CALIBRATION.md
/home/user/rl_ws/teleopration_openarm_tesollo/src/openarm_teleop/script/record_real2sim_identification_bag.sh
/home/user/rl_ws/teleopration_openarm_tesollo/src/openarm_teleop/script/real2sim_actuator_calibration.py
```

검증:

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo
pytest src/openarm_teleop/test/test_real2sim_actuator_calibration.py -v
```

### Task 2. Tesollo env 적용 확인

목표:

```text
Tesollo grasp env가 calibration JSON을 읽고 actuator group에 적용하는지 확인한다.
```

사용 파일:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v11/real2sim_actuator_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v11/grasp_right_env_cfg.py
```

필수 group:

```text
openarm_right_arm
openarm_left_arm
tesollo_hand_abduction
tesollo_hand_curl
tesollo_hand_pip
tesollo_hand_dip
openarm_left_gripper
```

### Task 3. RH56F1 env 적용 확인

목표:

```text
RH56F1 env가 drive/mimic group coverage를 유지하면서 calibration JSON을 읽는지 확인한다.
```

사용 파일:

```text
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/real2sim_actuator_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env_cfg.py
/home/user/rl_ws/hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/tests/test_phase3_actuator_coverage.py
```

검증:

```bash
cd /home/user/rl_ws/hdgp
python3 source/openarm/openarm/rh56f1/right/grasp_v1/tests/test_phase3_actuator_coverage.py
```

### Task 4. Parallel replay autotune 추가

목표:

```text
논문 Algorithm 1의 population search를 hdgp 내부 replay script로 구현한다.
```

권장 위치:

```text
/home/user/rl_ws/hdgp/scripts/r2s_autotune/
```

권장 파일:

```text
configs/autotune_ranges.yaml
configs/replay_sequences.yaml
load_real_track.py
sample_candidates.py
run_parallel_replay.py
compute_tracking_error.py
export_best_calibration.py
README.md
```

각 파일 책임:

| 파일 | 책임 |
|---|---|
| `configs/autotune_ranges.yaml` | asset별 group 이름, stiffness/damping/friction scale range, population size |
| `configs/replay_sequences.yaml` | real HDF5 경로, dataset 이름, replay할 group/joint regex, dt |
| `load_real_track.py` | HDF5에서 `time`, `q_cmd`, `q_real`, `dq_real`, `joint_names`를 읽고 rad 단위로 정렬 |
| `sample_candidates.py` | seed calibration JSON을 읽고 group별 scale 후보 K개 생성 |
| `compute_tracking_error.py` | `q`, `dq`, delay penalty 기반 error 계산 |
| `export_best_calibration.py` | best candidate를 `schema_version: 1` JSON으로 저장 |
| `run_parallel_replay.py` | Isaac Lab env K개를 띄우고 후보별 actuator 값을 적용한 뒤 같은 trajectory replay |
| `README.md` | 실행 순서와 예시 command 기록 |

`autotune_ranges.yaml` 최소 예시:

```yaml
asset: openarm_tesollo_sensor
seed_calibration: /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds/openarm_tesollo_sensor_seed_calibration.json
population_size: 128
random_seed: 7
groups:
  openarm_right_arm:
    stiffness_scale: [0.8, 1.25]
    damping_scale: [0.7, 1.5]
    friction_scale: [0.7, 1.3]
  tesollo_hand_curl:
    stiffness_scale: [0.6, 1.6]
    damping_scale: [0.5, 2.0]
    friction_scale: [0.5, 2.0]
```

`replay_sequences.yaml` 최소 예시:

```yaml
real_track:
  hdf5: /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds/real2sim_identification_100hz.hdf5
  demo_key: demo_0
  command_dataset: obs/right_hand_reference_joint_pos
  measured_dataset: obs/right_hand_joint_pos
  joint_name_regex: "rj_dg_[1-5]_2"
  dt: 0.01
error_weights:
  q: 1.0
  dq: 0.05
  delay: 0.01
```

Tesollo에 먼저 만들고, RH56F1은 config만 바꿔 같은 코드로 돌린다.

RH56F1용 `autotune_ranges.yaml` 차이:

```yaml
asset: openarm_bi_rh56f1
seed_calibration: /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds/openarm_bi_rh56f1_seed_calibration.json
groups:
  rh56f1_right_drive:
    stiffness_scale: [0.7, 1.5]
    damping_scale: [0.6, 1.8]
    friction_scale: [0.5, 2.0]
  rh56f1_right_mimic:
    stiffness_scale: [0.8, 1.3]
    damping_scale: [0.8, 1.5]
    friction_scale: [0.7, 1.5]
```

주의:

- 새 스크립트는 RL 학습을 실행하지 않는다.
- 입력은 real HDF5/NPZ와 seed calibration JSON이다.
- 출력은 best calibration JSON과 error report다.
- USD를 복사하거나 대량 수정하지 않는다.

### Task 5. Validation replay

목표:

```text
identification sequence가 아닌 task teleoperation data로 최종 검증한다.
```

사용 데이터:

```text
/home/user/rl_ws/teleopration_openarm_tesollo/bags/pouring/*.db3
/home/user/rl_ws/teleopration_openarm_tesollo/datasets/pour_v1_a*.hdf5
```

판정 기준:

```text
same-sequence tracking error 감소
held-out validation tracking error 감소
task replay에서 손/팔 trajectory drift 감소
contact timing이 과도하게 빨라지거나 늦어지지 않음
```

---

## 11. 실패 진단

### 11.1 MSE는 낮은데 RL이 나빠짐

가능 원인:

```text
identification sequence에만 과적합
joint order mismatch
controller mode mismatch
task contact distribution과 free-space tuning 불일치
너무 좁은 validation set
```

대응:

```text
held-out task replay를 error objective에 추가
group 단위 scale range 축소
contact/FT는 phase 2 이후만 추가
FABRICS joint order와 Isaac DOF order 재검증
```

### 11.2 후보마다 error 차이가 거의 없음

가능 원인:

```text
trajectory excitation이 약함
command와 measured가 같은 값을 보고 있음
HDF5 joint_names attr 누락 또는 regex mismatch
sim controller가 실제 controller와 다름
```

대응:

```text
step/ramp amplitude를 안전 범위 내에서 키움
hold 구간과 transient 구간 error를 분리
q_cmd, q_real, q_sim plot 저장
regex가 실제 joint_names를 fullmatch하는지 테스트
```

### 11.3 hand만 맞고 arm이 어긋남

가능 원인:

```text
Tesollo 100 Hz 기준으로 arm transient까지 downsample함
OpenArm arm command/state topic과 HDF5 dataset mapping이 다름
FABRICS target과 Isaac action target이 섞임
```

대응:

```text
arm은 raw DB3에서 500-1000 Hz 별도 분석
hand는 100 Hz 유지
combined policy validation은 100 Hz로 resample
arm group과 hand group을 따로 seed/refine
```

---

## 12. 최종 적용 체크리스트

Real 데이터:

```text
[ ] DB3 raw bag 보존
[ ] HDF5에 command/measured dataset 존재
[ ] 각 dataset에 joint_names attr 존재
[ ] timestamp 기준 resample 완료
[ ] rad 단위 통일
```

Calibration JSON:

```text
[ ] schema_version == 1
[ ] robot_asset가 대상 asset과 일치
[ ] groups 이름이 env cfg actuator group과 일치
[ ] fit_error와 joint_metrics 저장
[ ] seed JSON과 refined JSON을 분리 저장
```

HDGP 적용:

```text
[ ] OPENARM_REAL2SIM_ACTUATOR_CALIBRATION 설정
[ ] env cfg가 JSON loader를 import
[ ] ImplicitActuatorCfg에 _actuator_params 적용
[ ] actuator regex coverage test 통과
[ ] held-out replay에서 tracking error 감소
```

FABRICS:

```text
[ ] FABRICS URDF joint order 확인
[ ] Isaac Lab articulation joint order 확인
[ ] teleop HDF5 joint_names 확인
[ ] reset/IK/FK target이 calibration 대상 DOF와 일치
```

이 체크리스트가 통과한 뒤에만 RL 학습 또는 sim-to-real policy test로 넘어간다.
