# r2s_autotune — Real-to-Sim Actuator Autotune

논문 *Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids*의
**Algorithm 1 (Real-to-Sim Autotune Module)** 구현.

RL 학습이 아니다. **학습 이전**에 실제 로봇과 Isaac Lab 로봇의 관절 응답 차이를 줄이는
actuator system identification이다.

```
real robot log → HDF5 (q_cmd, q_real, dq_real)
              → seed calibration JSON
              → parallel replay 후보 탐색   ← 이 디렉토리
              → best calibration JSON
              → OPENARM_REAL2SIM_ACTUATOR_CALIBRATION → RL 학습 → sim2real
```

후보는 USD 복사본이 아니라 **calibration JSON + 런타임 gain override**로 표현한다.
`write_joint_stiffness_to_sim(tensor, env_ids=...)`가 `[num_envs, num_joints]`를 받으므로
후보 K개를 단일 sim의 K개 env에 병렬 배치한다. USD는 복사하지도 수정하지도 않는다.

---

## 설계 원칙

**Replay 씬은 grasp env와 독립이다.** `replay_env.py`는 articulation만 띄운다.
물체·reward·fabric·reset이 얽히면 순수 actuator tracking 측정에 잡음이 된다.

**canonical joint 이름만 쓴다.** `assets/robot/*_rl` 기준 (`r_aj_*`, `r_hj_*`, `l_aj_*`, `l_hj_*`).
teleop HDF5의 legacy 이름(`rj_dg_*`, `openarm_right_joint*`)은 읽는 즉시 manifest의
`source_to_canonical_joints`로 정규화한다.

**group 이름/regex의 source of truth는 RL env_cfg다.** 가이드 문서의 표가 아니다.
group 이름이 어긋나면 `get_actuator_params()`가 **조용히 default로 fallback**한다.
에러가 나지 않으므로 `tests/test_config_matches_env_cfg.py`가 env_cfg를 AST로 파싱해 대조한다.

---

## 실행 순서

### 1. 정합 검증 (Isaac 불필요)

```bash
cd /home/user/rl_ws/hdgp/scripts/r2s_autotune
python3 -m pytest tests/ -q
```

### 2. 합성 궤적 생성 — ground-truth 복원 테스트용

real 데이터가 없는 동안 파이프라인의 정확성을 검증할 유일한 방법이다.
알려진 gain으로 궤적을 만들고, autotune이 그 gain을 복원하는지 본다.

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/r2s_autotune/make_synthetic_track.py \
    --config ../hdgp/scripts/r2s_autotune/configs/bi_rh56f1.yaml --headless
```

산출:
```
logs/r2s_autotune/seeds/bi_rh56f1_synthetic.hdf5
logs/r2s_autotune/seeds/bi_rh56f1_synthetic_ground_truth.json
```

### 3. Parallel replay autotune

```bash
./isaaclab.sh -p ../hdgp/scripts/r2s_autotune/run_parallel_replay.py \
    --config ../hdgp/scripts/r2s_autotune/configs/bi_rh56f1.yaml \
    --output ../hdgp/logs/r2s_autotune/results/bi_rh56f1_best_calibration.json \
    --headless
```

**판정:** best calibration이 `_ground_truth.json`의 stiffness/damping을 복원해야 한다.
복원 오차가 크면 real 데이터를 넣어도 의미가 없다 — 파이프라인 버그다.

출력의 두 경고를 반드시 읽는다.
- `error spread < 1e-3` → excitation이 약하거나 command와 measured가 같은 값을 보고 있다.
- `seed가 모든 후보보다 낫다` → scale range가 잘못됐다.

### 4. RL env에 적용

```bash
export OPENARM_REAL2SIM_ACTUATOR_CALIBRATION=/home/user/rl_ws/hdgp/logs/r2s_autotune/results/bi_rh56f1_best_calibration.json
```

---

## 실물 데이터로 전환하기 (Tesollo 손)

`configs/tesollo_sensor.yaml`의 `real_track.hdf5`만 실제 파일로 바꾸면 같은 코드가 그대로 돈다.

### 명령 경로: isaacsim_bridge를 거치지 않는다

```
r2s_excitation.py → /dg5f_right/rj_dg_pospid/reference  (MultiDOFCommand, 100 Hz)
                              ↓ 드라이버 내부 PID (raw)
```

`/isaacsim/right_hand_cmd`를 쓰면 안 된다. `isaacsim_bridge`가 위치 목표를
`trajectory_time_sec`(기본 0.2s) 짜리 `JointTrajectory`로 보간하므로, 그 경로로 수집하면
액추에이터가 아니라 **브리지의 보간 필터**를 식별하게 된다.

`MultiDOFCommand`는 `dof_names`를 들고 다니므로 녹화된 명령 토픽 자체가 자기서술적이다.

### 실행 순서 (teleop repo)

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo
source /opt/ros/humble/setup.bash && source install/setup.bash

# 1) 하드웨어 없이 궤적 확인
python3 src/openarm_teleop/script/r2s_excitation.py --target hand --dry-run

# 2) 녹화 시작 (별도 터미널). 필요한 두 토픽은 이미 목록에 있다.
ROS_DOMAIN_ID=126 ./src/openarm_teleop/script/record_real2sim_identification_bag.sh \
    ./bags/real2sim_identification/run_001 40

# 3) 저진폭으로 첫 실행 — 반드시 입회 하에
ROS_DOMAIN_ID=126 python3 src/openarm_teleop/script/r2s_excitation.py --target hand --amplitude-scale 0.3

# 4) db3 → identification HDF5 (legacy 이름 그대로. load_real_track이 canonical로 정규화한다)
python3 src/openarm_teleop/script/db3_to_identification_hdf5.py \
    --bag-dir bags/real2sim_identification/run_001 \
    --output /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds/tesollo_identification_100hz.hdf5
```

4단계는 명령이 실제로 움직였는지(`barely moves`), 명령과 측정이 같은 토픽이 아닌지
(`command equals measured`)를 스스로 확인하고 실패하면 종료한다 (가이드 §11.2).

그 뒤 `configs/tesollo_sensor.yaml`의 `real_track.hdf5`를 위 경로로 바꾸고
`run_parallel_replay.py`를 돌린다. population은 128 이상.

## 팔(arm) — 구조적으로는 손보다 더 깨끗한 대상

`openarm_hardware/src/v10_simple_hardware.cpp:276`은 모터를 MIT 모드로 구동한다.

```cpp
arm_params.push_back({kp_[i], kd_[i], pos_commands_[i], vel_commands_[i], tau_commands_[i]});
openarm_->get_arm().mit_control_all(arm_params);   // τ = kp·(q*−q) + kd·(q̇*−q̇), tau_ff = 0
```

Isaac `ImplicitActuator`와 수식이 같다. `JointTrajectoryController`는 PD를 돌지 않고
(`command_interfaces: [position]`) 위치를 통과시킬 뿐이며, `forward_position_controller`를
쓰면 스플라인 보간조차 없다. 팔은 position/velocity/**effort** 상태를 모두 내보낸다.

명령 경로 (teleop 워크스페이스 하나로 끝난다):

```
ros2 launch openarm_bringup openarm.bimanual.launch.py robot_controller:=forward_position_controller
  → /right_forward_position_controller/commands   Float64MultiArray[7], 순서 openarm_right_joint1..7
  → ros2_control (보간 없음) → openarm_hardware → MIT PD
측정 ← /joint_states (position, velocity, effort)
```

`Float64MultiArray`는 이름을 싣지 않는다. 손의 `MultiDOFCommand`와 달리 자기서술적이지
않으므로, 변환기에 컨트롤러 yaml의 `joints` 순서를 명시해야 한다.

### 실물 팔 게인과 sim의 간극

| | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| 실물 kp | 70 | 70 | 70 | 60 | 10 | 10 | 10 |
| 실물 kd | 2.75 | 2.5 | 2.0 | 2.0 | 0.7 | 0.6 | 0.5 |
| sim (`openarm_right_arm`) | 400 | 400 | 400 | 400 | 400 | 400 | 400 |

손목은 강성 40배, 감쇠 160배 차이다. 그리고 실물은 관절마다 게인이 다른데 sim의 group
하나로는 표현할 수 없다. 그래서 `configs/tesollo_sensor.yaml`은 팔을 `arm_proximal`(j1–3),
`arm_elbow`(j4), `arm_wrist`(j5–7) 셋으로 나누고 실물 게인을 seed로 쓴다. kp는 각 그룹 안에서
정확히 균일하다.

**이 세 group 이름은 아직 어떤 RL env_cfg에도 없다.** calibration을 학습에 적용하려면 대상
env의 actuator 블록을 같은 이름으로 쪼개야 한다. 그러지 않으면 `get_actuator_params`가 조용히
default로 fallback한다.

### 팔 실행 순서

```bash
# 브링업을 forward_position_controller 로 (JTC 아님)
ros2 launch openarm_bringup openarm.bimanual.launch.py robot_controller:=forward_position_controller

# group 하나씩만 흔든다. 7축 동시 여기는 중력 커플링으로 식별성이 떨어진다.
python3 src/openarm_teleop/script/r2s_excitation.py --target arm --excite wrist --dry-run
ROS_DOMAIN_ID=126 python3 src/openarm_teleop/script/r2s_excitation.py --target arm --excite wrist --amplitude-scale 0.3

# 팔 명령에는 이름이 없다. 컨트롤러 yaml의 joints 순서를 반드시 넘긴다.
CFG=src/openarm_ros2/openarm_bringup/config/v10_controllers/openarm_v10_bimanual_controllers.yaml
python3 src/openarm_teleop/script/db3_to_identification_hdf5.py \
    --bag-dir bags/real2sim_identification/arm_wrist_001 \
    --command-topic /right_forward_position_controller/commands \
    --measured-topic /joint_states \
    --controller-config "$CFG" --controller-name right_forward_position_controller \
    --output /home/user/rl_ws/hdgp/logs/r2s_autotune/seeds/arm_wrist_100hz.hdf5
```

`record_real2sim_identification_bag.sh`는 위 두 팔 토픽을 이미 녹화한다. 손만 돌릴 때는
그 토픽에 발행자가 없어 비어 있을 뿐이다.

### 아직 없는 것

- **multi-group merge.** teleop의 `real2sim_actuator_calibration.py`는 한 번에 한 group만 쓴다.
  autotune의 seed로 쓰려면 group별 JSON의 `groups` 오브젝트를 합쳐야 한다.

수집 시 주의 (가이드 §4, §11.3):
- teleop task demo는 validation에는 좋지만 stiffness/damping/delay 분리 식별에는 부족하다.
- 손은 100 Hz. 팔 transient를 보려면 raw DB3를 500–1000 Hz로 따로 분석한다.

---

## 파일

| 파일 | 책임 |
|---|---|
| `joint_contract.py` | manifest 로드, legacy→canonical 정규화, regex fullmatch 검증, coverage |
| `config.py` | asset별 yaml 로딩. 로딩 시 group coverage를 강제 검증 |
| `excitation.py` | step/ramp/hold/sine 시퀀스. 관절 한계 clamp 내장 (실물 안전) |
| `load_real_track.py` | HDF5 → canonical 정렬된 `RealTrack` |
| `calibration_io.py` | schema_version 1 JSON 읽기/쓰기 |
| `seed_from_config.py` | config 기본값에서 seed calibration 생성 |
| `sample_candidates.py` | seed 중심 group 단위 scale 샘플링. candidate 0 == seed |
| `gain_matrix.py` | 후보별 gain을 `[K, J]`로 전개 (모든 관절을 기본값으로 먼저 채움) |
| `replay_env.py` | articulation-only 씬, per-env gain 주입, replay |
| `compute_tracking_error.py` | `w_q·MSE(q) + w_dq·MSE(dq) + w_delay·|Δlag|` |
| `export_best_calibration.py` | best 후보를 schema v1 JSON으로 |
| `make_synthetic_track.py` | ground-truth 궤적 생성 (Isaac) |
| `run_parallel_replay.py` | Algorithm 1 본체 (Isaac) |

---

## 검증 상태 (2026-07-10, Tesollo)

`openarm_tesollo_sensor_rl`로 end-to-end 실행 완료.

- **Tesollo USD는 canonical joint 이름이 맞다.** `verify_articulation()` 통과 (`r_hj_thumb_2` 등).
- **정답 파라미터를 넣으면 오차가 정확히 0.0**이다. replay가 생성 조건을 비트 단위로 재현하고
  objective도 옳다는 뜻이다.
- **ground-truth 복원**: 후보 128개로 stiffness 7.3%, damping 9.9% 오차 내 복원
  (seed는 각각 23.1%, 25.0%). 오차 59% 감소.
- 후보 32개로는 복원에 실패했다 (damping이 seed보다 나빠짐). population을 줄이지 말 것.

### 이 검증이 드러낸 함정

**`default_joint_pos`를 neutral로 쓰면 안 된다.** Tesollo `r_hj_(index|middle|ring)_2`는
default가 0인데 하한도 정확히 0이다. 그 자리에서 구동하면 관절이 한계를 뚫고 -1.3 rad까지
빠져나간다 (7.5Nm 액추에이터로는 불가능한 힘). `interior_neutral()`이 excitation 전 구간이
한계 안에 들어오도록 neutral을 밀어낸다. 추적하지 않는 관절도 같은 `rest_pose`로 붙든다.

**`sim.reset()`만으로는 articulation이 default 자세에 있지 않다.** `reset_and_settle()`로
관절 상태를 쓰고 정착시켜야 첫 구간의 과도응답이 actuator 특성을 반영한다.

**damping은 이 excitation으로는 약하게만 식별된다.** 0.25 Hz 사인은 거의 준정적이라
정상상태 droop(=stiffness)만 잘 보인다. damping을 제대로 잡으려면 더 빠른 성분이 필요하다.

## 알려진 제약

- **`delay_steps`는 sim에 반영되지 않는다.** hdgp의 `get_actuator_params()`가 이 필드를
  반환하지 않는다. 지금은 error 항에만 쓴다. action delay buffer를 구현하거나 명시적으로
  미지원 선언할지 결정해야 한다.
- **합성 복원 테스트는 "파이프라인이 옳다"만 증명한다.** sim의 actuator 모델 구조 자체가
  틀리면(마찰, 백래시, 직렬탄성) 어떤 gain으로도 real을 복원할 수 없다. 모델 타당성은
  real 데이터의 residual로만 판정된다.
- **RH56F1은 실물 제어 스택이 없다.** teleop/sim2real 어디에도 드라이버가 없고 URDF만 있다.
  실제 하드웨어는 OpenArm + Tesollo다. 따라서 real calibration의 대상은 Tesollo뿐이고,
  `configs/bi_rh56f1.yaml`은 합성 궤적으로 파이프라인을 검증하는 용도다.
- **팔의 명령 토픽은 브링업에 따라 다르다.** teleop의 bilateral C++ 스택은 `openarm_can`으로
  직접 CAN을 때려 명령 토픽이 없고 `/openarm/*/joint_states`만 낸다. identification에는
  `openarm_bringup`을 `robot_controller:=forward_position_controller`로 띄워야 한다.
- `simulation_app.close()`가 행되므로 두 CLI는 `os._exit(0)`으로 빠져나온다.
- **Tesollo `grasp_v11`은 이 파이프라인과 아직 연결되지 않았다.** 그 env는 legacy joint
  이름과 존재하지 않는 USD 경로(`assets/openarm_tesollo_sensor/`)를 참조한다.
  `configs/tesollo_sensor.yaml`은 `assets/robot/openarm_tesollo_sensor_rl` 기준 canonical만 쓴다.
  grasp_v11의 canonical 마이그레이션은 별도 작업이다.
- **`rh56f1/right/grasp_v2`에는 calibration hook이 없다** (stiffness 하드코딩).
  최신 성공 정책이 v2이므로 v1의 `real2sim_actuator_cfg.py` 패턴을 이식해야 한다.
