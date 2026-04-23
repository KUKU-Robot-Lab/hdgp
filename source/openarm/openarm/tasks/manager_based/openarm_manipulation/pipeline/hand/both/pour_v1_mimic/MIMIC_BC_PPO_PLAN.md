# BC+PPO Pouring 파이프라인 구현 계획

> OpenArm+Teosllo 양팔 pouring 태스크를 위한  
> **텔레옵 데이터 수집 → IsaacLab Mimic 증강 → BC-RNN 학습 → PPO fine-tuning** 전체 구현 명세

---

## 1. 전체 아키텍처

```
사용자 텔레옵 (OpenArm 실물 또는 입력 장치)
  │
  │  ROS2 topics:
  │    /isaacsim/right_arm_cmd    (Float64MultiArray, 7D joint pos)
  │    /isaacsim/right_hand_cmd   (Float64MultiArray, 20D joint pos)
  │    /isaacsim/left_arm_cmd     (Float64MultiArray, 7D joint pos)
  │    /isaacsim/left_gripper_cmd (Float64, scalar)
  ▼
[NEW] ros2_demo_recorder.py
  ├─ 오른팔: Fabrics FK → palm pose → 6D delta action
  ├─ 오른손: 20D joint pos → 5D curl action (HAND_CURL_JOINT_NAMES 기준)
  ├─ 왼팔:   joint pos delta → 7D left arm action
  └─ IsaacLab PourMimicEnv.step() 호출 + HDF5 기록
  │
  ▼
HDF5 demo dataset (18D action: 6D+5D+7D)
  │
  ├─ annotate_demos.py     (subtask 어노테이션)
  ├─ generate_dataset.py   (1000개 자동 증강)
  │
  ▼
robomimic BC-RNN 학습 (18D action space)
  │
  ├─ bc_to_rlgames_converter.py  (가중치 포맷 변환)
  │
  ▼
pour_v1 PPO fine-tuning
  └─ PourLstmBCAgent (이미 존재)
     load_checkpoint: true
     BC auxiliary loss로 warm-start
```

---

## 2. 액션 공간 정의

| 인덱스 | 구성 | 차원 | 설명 |
|--------|------|------|------|
| `[0:6]`   | 오른팔 palm pose delta | 6D | xyz(m) + rot_vec(rad), Fabrics IK 입력 |
| `[6:11]`  | 오른손 curl action | 5D | 손가락별 curl 직접 제어 (HAND_CURL_JOINT_NAMES) |
| `[11:18]` | 왼팔 joint delta | 7D | openarm_left_joint1~7 delta |
| **합계** | | **18D** | `NUM_ACTIONS = 18` (pour_constants.py와 동일) |

### 오른손 5D curl 매핑 (HAND_CURL_JOINT_NAMES 순서)
```
action[6]  → rj_dg_1_2  (thumb curl,  Z축, range [-π, 0])
action[7]  → rj_dg_2_2  (index curl,  Y축, range [0, 2.007])
action[8]  → rj_dg_3_2  (middle curl, Y축, range [0, 1.955])
action[9]  → rj_dg_4_2  (ring curl,   Y축, range [0, 1.902])
action[10] → rj_dg_5_3  (pinky curl,  Y축, range [0, π/2])
```

---

## 3. 관측 공간 정의 (pour_utils.py ACTOR_OBSERVATION_GROUP_DIMS 그대로 사용)

| 그룹 키 | 차원 | 비고 |
|---------|------|------|
| `right_joint_pos` | 27 | 오른팔 7D + 오른손 20D |
| `right_joint_vel` | 27 | |
| `left_arm_joint_pos` | 7 | 왼팔 7D |
| `left_arm_joint_vel` | 7 | |
| `fingertip_pos` | 15 | 5 fingertip × 3D (Fabrics FK) |
| `cup_pose_vel` | 13 | source cup pos(3)+quat(4)+linvel(3)+angvel(3) |
| `target_opening_pos` | 3 | target cup 입구 위치 |
| `bead_centroid_pos` | 3 | |
| `prev_actions` | 18 | 이전 step 액션 |
| `mouth_delta` | 3 | |
| `mouth_xy_distance` | 1 | |
| `mouth_z_clearance` | 1 | |
| `source_up_dot_world` | 1 | |
| `directional_tilt_cos` | 1 | |
| `mouth_alignment_cos` | 1 | |
| `bead_cross_fraction` | 1 | **privileged** → BC에서도 포함 |
| `bead_in_target_fraction` | 1 | **privileged** |
| `bead_in_source_fraction` | 1 | **privileged** |
| `spill_ratio` | 1 | **privileged** |
| `g_ready` | 1 | |
| `g_pour` | 1 | |
| **합계** | **134D** | `ACTOR_OBSERVATION_DIM` |

> **중요**: BC 학습 시 privileged 관측(bead fraction, spill_ratio)은 시뮬에서만 얻을 수 있으므로
> Mimic 환경도 이 값을 제공해야 함. PPO 단계에서 별도 teacher/student 분리 예정.

---

## 4. 생성 파일 체크리스트

### 4.1 IsaacLab Mimic 호환 환경

```
hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/
  pipeline/hand/both/pour_v1_mimic/
    [x] __init__.py
    [x] pour_mimic_env.py
    [x] pour_mimic_env_cfg.py
    [x] pour_mimic_subtask.py
    config/
      [x] __init__.py
```

> 진행 메모(2026-04-22): `Pour-Mimic-V1-v0`, `Pour-Mimic-V1-Mimic-v0` gym 등록 및 IsaacLab cfg parse smoke 통과.
> 미완성/보류: 전체 env reset/step acceptance는 현재 머신의 CUDA driver 미탑재 및 CPU fallback scene setup 오류(`spawn_ground_plane` PhysX material binding)로 막혀 있음. `pour_v1` 런타임/Isaac 환경 정리 후 재검증 필요.

#### `__init__.py` — 필수 내용
```python
import gymnasium as gym

gym.register(
    id="Pour-Mimic-V1-v0",              # BC 평가용 (Mimic 없음)
    entry_point="...pour_mimic_env:PourMimicEnv",
    kwargs={"cfg": PourMimicEnvCfg()},
)
gym.register(
    id="Pour-Mimic-V1-Mimic-v0",        # 데이터 증강용
    entry_point="...pour_mimic_env:PourMimicEnv",
    kwargs={"cfg": PourMimicMimicEnvCfg()},
)
```

#### `pour_mimic_env.py` — 필수 구현 메서드

```python
class PourMimicEnv(ManagerBasedRLMimicEnv):

    # ----------------------------------------------------------------
    # Mimic 필수 메서드 (모두 구현 필수)
    # ----------------------------------------------------------------

    def get_robot_eef_pose(self) -> torch.Tensor:
        """오른팔 palm pose 반환 (N, 7): [pos(3), quat_xyzw(4)]
        - Fabrics FK에서 palm_link 위치+자세 추출
        - Mimic이 subtask 경계에서 trajectory 보간에 사용"""
        ...

    def target_eef_pose_to_action(
        self,
        target_eef_pose: torch.Tensor,  # (N, 7)
        gripper_action: torch.Tensor,   # (N, 1) [-1~+1, open~close]
    ) -> torch.Tensor:
        """목표 SE3 → 18D action 변환
        - target_eef_pose - current_eef_pose = 6D palm delta
        - gripper_action → 5D curl (단일 값 → 5 손가락 동일 적용)
        - 왼팔: current left arm joint pos 유지 (delta=0)"""
        ...

    def action_to_target_eef_pose(
        self,
        action: torch.Tensor,  # (N, 18)
    ) -> torch.Tensor:
        """18D action → 목표 SE3 (N, 7)
        - action[0:6] + current palm pose = target palm pose"""
        ...

    def actions_to_gripper_actions(
        self,
        actions: torch.Tensor,  # (T, N, 18) 또는 (N, 18)
    ) -> torch.Tensor:
        """손 curl 부분 추출: actions[..., 6:11] 반환 (N, 5)"""
        ...

    def get_object_poses(self) -> dict[str, torch.Tensor]:
        """Mimic 데이터 생성 시 object 위치 참조용
        Returns:
            {
                "source_cup": (N, 7),  # pos + quat_xyzw
                "target_cup": (N, 7),
            }
        """
        ...

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """각 subtask 완료 여부 (N,) bool tensor
        Returns:
            {
                "grasp_done":  (N,),  # contact sensor, cup_in_hand
                "lift_done":   (N,),  # source_cup z > LIFT_THRESHOLD
                "align_done":  (N,),  # mouth xy 오차 < ALIGN_THRESHOLD
                "pour_done":   (N,),  # tilt angle > POUR_THRESHOLD
            }
        """
        ...
```

#### `pour_mimic_env_cfg.py` — 필수 설정 항목

```python
@configclass
class PourMimicEnvCfg(ManagerBasedRLEnvCfg):
    # ----------------------------------------------------------------
    # 씬 구성 (pour_env_cfg.py와 동일 asset 사용)
    # ----------------------------------------------------------------
    robot: ArticulationCfg          # openarm_modular_dual.usd
    source_cup: RigidObjectCfg      # cup_big.usd (source)
    target_cup: RigidObjectCfg      # cup_big.usd (target)
    beads: RigidObjectCollectionCfg # 20개 bead

    # ----------------------------------------------------------------
    # 제어기 설정
    # ----------------------------------------------------------------
    # 오른팔: Fabrics (외부 제어기) - ImplicitActuator 사용
    # 왼팔: Joint position control - ImplicitActuator
    right_arm_actuator: ImplicitActuatorCfg = ImplicitActuatorCfg(
        joint_names_expr=["openarm_right_joint[1-7]"],
        stiffness=400.0,
        damping=80.0,
    )
    right_hand_actuator: ImplicitActuatorCfg = ImplicitActuatorCfg(
        joint_names_expr=["rj_dg_.*"],
        stiffness=None,   # USD 물성치 유지
        damping=None,
    )
    left_arm_actuator: ImplicitActuatorCfg = ImplicitActuatorCfg(
        joint_names_expr=["openarm_left_joint[1-7]"],
        stiffness=400.0,
        damping=80.0,
    )
    left_gripper_actuator: ImplicitActuatorCfg = ImplicitActuatorCfg(
        joint_names_expr=["openarm_left_finger_joint[1-2]"],
        stiffness=400.0,
        damping=80.0,
    )

    # ----------------------------------------------------------------
    # 액션/관측 차원
    # ----------------------------------------------------------------
    num_actions: int = 18      # 6D palm + 5D hand + 7D left arm
    num_observations: int = 134  # ACTOR_OBSERVATION_DIM

    # ----------------------------------------------------------------
    # 시뮬 파라미터 (pour_env_cfg.py에서 동기화 필요)
    # ----------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(dt=1.0/60.0, render_interval=2)
    episode_length_s: float = 10.0  # EPISODE_STEPS(600) / 60Hz

    # ----------------------------------------------------------------
    # Fabrics 파라미터
    # ----------------------------------------------------------------
    fabrics_steps: int = 60         # PREGRASP_FABRICS_STEPS
    max_pose_angle: float = 30.0    # palm_pose_mins/maxs 함수 인자
    palm_delta_xyz: float = 0.5
    palm_delta_rot_deg: float = 30.0

    # ----------------------------------------------------------------
    # Subtask threshold (get_subtask_term_signals에서 사용)
    # ----------------------------------------------------------------
    lift_threshold_z: float = 0.45          # source cup z > 0.45m
    align_threshold_xy: float = 0.03        # mouth xy 오차 < 3cm
    pour_threshold_tilt_deg: float = 70.0   # source cup 기울기 > 70°
    grasp_force_threshold: float = 0.5      # contact sensor force (N)


@configclass
class PourMimicMimicEnvCfg(PourMimicEnvCfg):
    """IsaacLab Mimic 데이터 생성 전용 설정 (MimicEnvCfg 추가)"""
    mimic: MimicEnvCfg = MimicEnvCfg(
        name="Pour-Mimic-V1",
        num_substeps=2,
        # subtask 정의 (pour_mimic_subtask.py 참조)
        subtask_configs={
            "right": [
                SubTaskConfig(
                    object_ref="source_cup",
                    subtask_term_signal="grasp_done",
                    ee_entry_interpolation_fraction=0.5,
                    num_interpolation_steps=10,
                    gripper_action_for_eef="right",
                ),
                SubTaskConfig(
                    object_ref="source_cup",
                    subtask_term_signal="lift_done",
                    num_interpolation_steps=5,
                    gripper_action_for_eef="right",
                ),
                SubTaskConfig(
                    object_ref="target_cup",
                    subtask_term_signal="align_done",
                    num_interpolation_steps=15,
                    gripper_action_for_eef="right",
                ),
                SubTaskConfig(
                    object_ref="source_cup",
                    subtask_term_signal="pour_done",
                    num_interpolation_steps=5,
                    gripper_action_for_eef="right",
                ),
            ],
            "left": [
                SubTaskConfig(
                    object_ref="target_cup",
                    subtask_term_signal="grasp_done",   # 왼팔: 항상 고정 (grasp_done과 동기화)
                    num_interpolation_steps=10,
                    gripper_action_for_eef="left",
                ),
            ],
        },
        subtask_constraints=[
            # 왼팔 grasp_done은 오른팔 grasp_done 이전에 완료
            SubTaskConstraintConfig(
                eef_name="left",
                subtask_index=0,
                must_complete_before_eef="right",
                must_complete_before_subtask_index=0,
            ),
        ],
        generation_num_trials=1000,
        generation_success_threshold=0.05,  # 5% 이상 성공 시 저장
    )
```

---

### 4.2 ROS2 → IsaacLab 텔레옵 브리지

```
sim2real/scripts/
  [x] ros2_demo_recorder.py       핵심 수집 스크립트 (신규)
  [x] ros2_teleop_device.py       IsaacLab Se3Device 커스텀 구현 (신규)
```

> 진행 메모(2026-04-22): ROS2 미설치 환경에서도 import 가능한 action 변환 코어와 `Se3ROS2Device` dry path 구현, TDD 계약 테스트 통과.
> 미완성/보류: 실 ROS2 subscription loop, 키보드 `S`/`R` 저장/폐기 UX, Pour env의 실제 Fabrics FK 연결은 하드웨어/IsaacLab 런타임에서 추가 검증 필요. 현재 `ROS2DemoRecorder`는 HDF5 writer와 env step 래퍼를 제공하지만 실시간 노드 spin 통합은 acceptance 단계로 남김.

#### `ros2_demo_recorder.py` — 필수 내용

```python
class ROS2DemoRecorder(Node):
    """ROS2 텔레옵 명령 → IsaacLab Mimic env 스텝 + HDF5 기록

    동작 흐름:
      1. /isaacsim/right_arm_cmd  (7D joint pos)  구독
      2. /isaacsim/right_hand_cmd (20D joint pos) 구독
      3. /isaacsim/left_arm_cmd   (7D joint pos)  구독
      4. 60Hz 루프에서 joint pos → 18D action 변환
      5. env.step(action) 호출 → 관측 + 보상 수집
      6. HDF5에 (obs, action, reward, done) 기록
      7. 'S' 키: episode 성공 저장, 'R' 키: episode 폐기

    18D action 변환 절차:
      [0:6]   오른팔 palm pose delta:
                Fabrics FK(current joints) → current palm pose
                Fabrics FK(target joints)  → target palm pose
                delta = target - current (pos: m, rot: axis-angle rad)
      [6:11]  오른손 curl:
                target_hand[CURL_JOINT_IDX] → [-1, +1] 정규화
      [11:18] 왼팔 joint delta:
                target_left_arm - current_left_arm (rad)
    """

    # 구독 토픽
    RIGHT_ARM_TOPIC   = "/isaacsim/right_arm_cmd"    # Float64MultiArray 7D
    RIGHT_HAND_TOPIC  = "/isaacsim/right_hand_cmd"   # Float64MultiArray 20D
    LEFT_ARM_TOPIC    = "/isaacsim/left_arm_cmd"     # Float64MultiArray 7D
    LEFT_GRIP_TOPIC   = "/isaacsim/left_gripper_cmd" # Float64 scalar

    # CURL_JOINT_NAMES 인덱스 (right_hand 20D 내 위치)
    # rj_dg_X_Y 순서: f in 1..5, j in 1..4 → 인덱스 (f-1)*4 + (j-1)
    CURL_JOINT_IDX = [
        1,   # rj_dg_1_2 (thumb curl,  index 1)
        5,   # rj_dg_2_2 (index curl,  index 5)
        9,   # rj_dg_3_2 (middle curl, index 9)
        13,  # rj_dg_4_2 (ring curl,   index 13)
        10,  # rj_dg_5_3 (pinky curl,  index 18 = (5-1)*4+(3-1))
    ]
    # 정정: rj_dg_5_3 = (5-1)*4+(3-1) = 18
    CURL_JOINT_IDX = [1, 5, 9, 13, 18]

    # curl 관절 범위 (정규화 기준, CURL_JOINT_LIMITS_MIN/MAX와 동일)
    CURL_MIN = [-3.14159, 0.0, 0.0, 0.0, 0.0]
    CURL_MAX = [0.0, 2.007, 1.955, 1.902, 1.5708]
```

#### `ros2_teleop_device.py` — IsaacLab Se3Device 확장

```python
class Se3ROS2Device:
    """IsaacLab teleop_se3_agent.py 호환 커스텀 입력 디바이스

    record_demos.py --teleop_device ros2 로 사용
    ROS2 토픽에서 EEF delta 명령을 읽어 Isaac Lab 형식으로 변환

    반환 형식: (delta_pos: np.ndarray(3,), delta_rot: np.ndarray(3,),
               gripper: float, reset: bool)
    """
```

---

### 4.3 BC → rl_games 가중치 변환기

```
hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/
  pipeline/hand/both/pour_v1/
    [ ] bc_to_rlgames_converter.py
```

#### `bc_to_rlgames_converter.py` — 필수 내용

```python
def convert_robomimic_bc_rnn_to_rlgames(
    robomimic_ckpt_path: str,
    output_path: str,
    obs_dim: int = 134,    # ACTOR_OBSERVATION_DIM
    action_dim: int = 18,  # NUM_ACTIONS
    lstm_hidden: int = 256,
    mlp_units: list = [512, 256, 128],
) -> None:
    """robomimic BC-RNN checkpoint → rl_games PPO init 형식 변환

    매핑 규칙:
      robomimic LSTM weights  →  a2c_network.rnn.weight_*
      robomimic MLP weights   →  a2c_network.actor_mlp.*.weight/bias
      robomimic action head   →  a2c_network.mu.weight/bias
      obs running stats       →  running_mean_std.running_mean/var
                                 (robomimic normalizer에서 추출)

    출력 형식:
      {
        "model": state_dict,
        "epoch": 0,
        "frame": 0,
        "last_mean_rewards": 0.0,
        "optimizer": None,
      }
    """
```

---

### 4.4 설정 파일 수정/신규

```
hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/
  pipeline/hand/both/pour_v1/config/agents/
    [ ] rl_games_ppo_lstm_bc_v2_cfg.yaml   신규 (BC 초기화 포함 버전)
```

---

## 5. 설정값 체크리스트

### 5.1 PourMimicEnv 필수 설정 (pour_v1과 동기화 필수)

| 항목 | 값 | 출처 |
|------|---|------|
| `num_actions` | `18` | `pour_constants.py: NUM_ACTIONS` |
| `num_observations` | `134` | `pour_utils.py: ACTOR_OBSERVATION_DIM` |
| `episode_length_s` | `10.0` | `EPISODE_STEPS(600) / 60Hz` |
| `sim.dt` | `1/60` | 60Hz 제어 |
| `right_arm_start_pose` | `[0.5, 0.1, 0.0, 0.60, -0.2, 0.0, 0.0]` | `RIGHT_ARM_START_POSE` |
| `hand_approach_pose` | `[0,−1.57,−0.5,0, ...]` | `HAND_APPROACH_POSE` |
| `hand_grasp_pose` | `[0,−1.57,1.5,1.5, ...]` | `HAND_GRASP_POSE` |
| `left_arm_rest_pose` | `{j4: 1.5, ...}` | `LEFT_ARM_REST_JOINT_POS` |
| `bead_count` | `20` | `_DEFAULT_BEAD_COUNT` |
| `bead_mass` | `0.010 kg` | `pour_env_cfg.py` |
| `palm_delta_xyz` | `0.5 m` | `pour_env_cfg.py` |
| `palm_delta_rot_deg` | `30.0°` | `pour_env_cfg.py` |
| `max_pose_angle` | `30.0°` | `pour_env_cfg.py` |
| `source_cup_usd` | `assets/cup/cup_big.usd` | |
| `target_cup_usd` | `assets/cup/cup_big.usd` | |
| `robot_usd` | `usds/openarm_modular_dual/openarm_modular_dual.usd` | |

### 5.2 Subtask threshold 설정

| 신호 키 | 조건 | 값 |
|---------|------|---|
| `grasp_done` | contact sensor force > threshold AND cup lifted | force ≥ 0.5N |
| `lift_done` | source_cup.pos_w[2] > threshold | z ≥ 0.45m |
| `align_done` | \|\|mouth_pos_xy − target_opening_xy\|\| < threshold | ≤ 0.03m |
| `pour_done` | source_cup 기울기 > threshold | tilt ≥ 70° |

### 5.3 Mimic 데이터 생성 필수 설정

| 항목 | 값 | 이유 |
|------|---|------|
| `generation_num_trials` | `1000` | GR-1 참조: 40~60% BC 성공률 |
| `num_envs` | `20` | 데이터 생성 속도 (GPU 여유 시 높임) |
| `num_interpolation_steps (grasp)` | `10` | cup 내 컵 삽입 부드러운 보간 |
| `num_interpolation_steps (align)` | `15` | cup-to-cup 정렬 경로 |
| `num_interpolation_steps (pour)` | `5` | tilt 단계는 짧게 |
| `ee_entry_interpolation_fraction` | `0.5` | grasp subtask 진입 시 |

### 5.4 Robomimic BC-RNN 필수 설정

| 항목 | 권장값 | 이유 |
|------|--------|------|
| `algo` | `bc_rnn` | LSTM 포함 BC |
| `obs_keys` | `134D actor obs 그룹` | `ACTOR_OBSERVATION_GROUP_DIMS` 전체 |
| `action_keys` | `actions` (18D) | |
| `lstm_hidden_dim` | `256` | `rl_games_ppo_lstm_bc_cfg.yaml`과 동일 |
| `lstm_num_layers` | `1` | |
| `mlp_layer_dims` | `[512, 256, 128]` | `rl_games` YAML과 동일 |
| `seq_length` | `16` | `rl_games_ppo_lstm_bc_cfg.yaml: seq_length: 16` |
| `train_epochs` | `600~1000` | GR-1 NutPour 참조 |
| `normalize_obs` | `True` | running_mean_std 변환 추출에 필요 |
| `normalize_actions` | `True` | `--normalize_training_actions` 플래그 |
| `dataset_keys` | `["obs/actor_obs", "actions"]` | |

### 5.5 PPO BC warm-start 설정 (`rl_games_ppo_lstm_bc_v2_cfg.yaml`)

```yaml
load_checkpoint: true
load_path: '/path/to/bc_converted.pth'  # bc_to_rlgames_converter.py 출력

config:
  # BC auxiliary loss (PourLstmBCAgent)
  bc_loss_warmup_epochs:  100   # 0→bc_w_init 선형 상승
  bc_loss_decay_epochs:  3000   # bc_w_init→bc_w_final 선형 하강
  bc_loss_weight_init:    1.0   # 초반 강하게
  bc_loss_weight_final:   0.05  # 최종 소량 유지
  bc_min_buffer_size:     20    # trajectory buffer 최소 필요 수
  bc_seq_len:             16    # BPTT sequence length
  bc_batch_size:          64

  # 학습률: BC init 후 낮게 시작
  learning_rate: 3e-5    # 기존 1e-4 → 3e-5 (BC 기준점 유지)
  lr_schedule: linear

  # LSTM
  rnn:
    units: 256            # robomimic BC와 동일 → 가중치 직접 전환
    layers: 1
    before_mlp: true
    concat_input: true
    layer_norm: true

  # 기타 (기존 yaml과 동일)
  gamma: 0.998
  tau: 0.95
  e_clip: 0.20
  horizon_length: 64
  seq_length: 16
  minibatch_size: 2048
  entropy_coef: 0.0008
  bc_dataset: '/path/to/pour_generated_1k.hdf5'
```

---

## 6. 데이터 수집 프로토콜

### 수집 목표

| 등급 | 기준 | 수량 |
|------|------|------|
| S (D_seed, BC 초기화용) | capture ≥ 80%, spill ≤ 10% | **20개** |
| A (D_mix, PPO auxiliary용) | capture ≥ 50%, spill ≤ 20% | 10개 |
| recovery (회복 시연) | 정렬 오차 후 복구 포함 | 5개 |
| **최소 수집 합계** | | **35개** |

### 수집 시 필수 준수 사항

- [ ] 시연 길이 600 step (10s) 이내 유지
- [ ] 왼팔: target cup 파지 후 안정화 → 오른팔 시작 (subtask 순서)
- [ ] 오른팔 접근: cup 옆(-Y 방향)에서 접근 (`PREGRASP_OFFSET` 방향)
- [ ] tilt 시작 전 반드시 cup mouth 위 정렬 확인
- [ ] `R` 키: episode 폐기 (실수 시)
- [ ] `S` 키: 성공 저장 확인

### 수집 빈도 설정

| 항목 | 값 |
|------|---|
| 수집 Hz | 60 Hz |
| 저장 신호 | joint pos/vel, EEF pose, cup pose, teleop action, contact force |
| 자동 phase 라벨 | `subtask_term_signals` 기반 자동 생성 |

---

## 7. 실행 커맨드 (순서대로)

### Step 1: 환경 테스트
```bash
# IsaacLab Mimic 파이프라인 동작 확인 (Franka 예제)
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard --num_demos 5 \
  --dataset_file ./datasets/franka_test.hdf5
```

### Step 2: 데모 수집
```bash
# terminal 1: 실물 로봇 제어 스택 (기존)
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch integrated_control openarm_left_gripper_right_dg5_real.launch.py \
  left_can_interface:=can1 right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 dg5f_right_port:=502

# terminal 2: 데모 수집 브리지
python sim2real/scripts/ros2_demo_recorder.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --output_file ./datasets/pour_demos.hdf5 \
  --num_demos 35 \
  --headless false
```

### Step 3: Subtask 어노테이션
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Pour-Mimic-V1-Mimic-v0 --auto \
  --input_file ./datasets/pour_demos.hdf5 \
  --output_file ./datasets/pour_annotated.hdf5
```

### Step 4: Mimic 증강 (1000개)
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --headless \
  --num_envs 20 --generation_num_trials 1000 \
  --input_file ./datasets/pour_annotated.hdf5 \
  --output_file ./datasets/pour_generated_1k.hdf5
```

### Step 5: BC-RNN 학습
```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Pour-Mimic-V1-v0 --algo bc_rnn \
  --normalize_training_actions \
  --dataset ./datasets/pour_generated_1k.hdf5
```

### Step 6: BC 평가
```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Pour-Mimic-V1-v0 --num_rollouts 50 \
  --horizon 600 \
  --checkpoint /PATH/TO/best_checkpoint.pth
```

### Step 7: BC → rl_games 변환
```bash
python hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/ \
  pipeline/hand/both/pour_v1/bc_to_rlgames_converter.py \
  --input /PATH/TO/robomimic_checkpoint.pth \
  --output ./pour_bc_init.pth \
  --obs_dim 134 --action_dim 18
```

### Step 8: PPO fine-tuning
```bash
# hdgp 학습 스크립트 (기존 방식 그대로)
python train.py task=pour_v1 \
  agent=pour_v1/rl_games_ppo_lstm_bc_v2 \
  num_envs=1024 \
  headless=true
```

---

## 8. 구현 순서 (Phase)

```
Phase 0 (2~3일): 환경 검증
  ├─ [ ] IsaacLab Mimic Franka E2E 실행 확인
  │      TODO: 현재 작업 범위에서는 미실행. Mimic 기본 예제 acceptance 필요.
  ├─ [ ] pour_v1 정상 실행 확인
  │      TODO: CUDA driver 미탑재/CPU scene setup 오류로 전체 reset/step 검증 보류.
  └─ [x] Pour-Mimic-V1 registry/cfg parse smoke 확인

Phase 1 (7~10일): PourMimicEnv 구현  ← 핵심 병목
  ├─ [x] pour_v1_mimic/__init__.py
  ├─ [x] pour_mimic_env.py  (ManagerBasedRLMimicEnv 호환 facade + Mimic contract)
  ├─ [x] pour_mimic_env_cfg.py
  └─ [x] pour_mimic_subtask.py

Phase 2 (5~7일): ROS2 텔레옵 브리지
  ├─ [x] ros2_demo_recorder.py
  │      TODO: 실제 ROS2 spin, 키보드 저장/폐기, 실 FK 연결 acceptance 필요.
  └─ [x] ros2_teleop_device.py
         TODO: IsaacLab record_demos.py 플러그인 등록 방식과 실 ROS2 토픽 검증 필요.

Phase 3 (3~5일): 데이터 수집
  └─ 35개 시연 수집 + 품질 라벨링

Phase 4 (1~2일): Mimic 증강
  └─ 1000개 자동 생성

Phase 5 (2~3일): BC-RNN 학습
  └─ robomimic BC-RNN

Phase 6 (5~7일): PPO 연결
  ├─ bc_to_rlgames_converter.py
  ├─ rl_games_ppo_lstm_bc_v2_cfg.yaml
  └─ PPO fine-tuning 실행

전체 예상: 5~6주
```

---

## 9. 리스크 체크리스트

| 리스크 | 감지 증상 | 대응 |
|--------|----------|------|
| Fabrics + ManagerBased 혼용 충돌 | 환경 초기화 실패 | Fabrics를 env._setup_fabrics()로 분리 |
| subtask 자동 감지 오류 | Mimic 생성 성공률 < 5% | manual annotation으로 전환 |
| palm pose delta 변환 불일치 | BC loss 낮은데 closed-loop 실패 | Fabrics FK 입출력 단위 재검증 |
| left arm subtask 순서 오류 | 왼팔이 오른팔보다 늦게 grasp | SubTaskConstraintConfig 재설정 |
| BC-RNN hidden 크기 불일치 | 가중치 변환 오류 | robomimic lstm_hidden = rl_games rnn.units 동기화 |
| privileged obs 누락 | BC 학습 시 bead_cross_fraction=0 | Mimic env에서 bead 물리 시뮬 동작 확인 |
| ROS2 ↔ IsaacLab 타이밍 불일치 | demo 재현 시 궤적 불안정 | 60Hz 하드 동기화, 큐 버퍼 최소화 |

---

## 10. 성공 기준

| 단계 | 기준 |
|------|------|
| Mimic 데이터 생성 | 생성 성공률 ≥ 20% (pour는 stack보다 어려움) |
| BC 학습 (closed-loop) | 50 rollout 기준 success ≥ 30% |
| PPO warm-start | 1000 epoch 기준 BC 대비 capture↑ 또는 spill↓ |
| PPO 최종 | `capture_50_spill_10` 달성 (capture≥50%, spill≤10%) |

---

*작성일: 2026-04-22*  
*참조: deep-research-report2.md, pour_v1, sim2real, isaaclab_mimic docs*
