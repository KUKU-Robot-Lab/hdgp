# grasp_kp / grasp_fj — SimToolReal 식 목표열·progress 보상 개편 설계 (2026-09-06)

근거: `rl_ws/repo/reports/simtoolreal_적용성_평가.md`. 사용자 결정(09.06):
**접촉 센서 전면 미사용** · A(fabric)/B(full-joint) 두 폴더 · 학습 server gpu0/gpu1.

## 0. 폴더와 상속

```
agnostic/modules/keypoint_goal.py      키포인트 · d(o,g) · 목표열 샘플러 · 진행 추적기 · 허용오차 커리큘럼 (순수 torch)
agnostic/modules/progress_reward.py    progress-only 보상 7항 + hand_floor 기하 벌점 + cmd_rate 지령변화 벌점(09.07) (순수 torch)
agnostic/modules/perception_delay.py   지연 큐(obs/action/object) + 코히런트 자세 노이즈 (순수 torch)
agnostic/modules/object_wrench.py      리프트 후 질량정규화 힘/토크 외란 (순수 torch)
agnostic/tasks/grasp_kp/               A: GraspKPEnvCfg(GraspS2REnvCfg) · GraspKPEnv(GraspS2REnv)  — fabric palm 6D + 시너지 15D
agnostic/tasks/grasp_fj/               B: GraspFJEnvCfg(GraspKPEnvCfg) · GraspFJEnv(GraspKPEnv)   — 팔 7D 증분+EMA + 시너지 15D
```

- `grasp_s2r` 는 **건드리지 않는다**(기준선 보존). 새 트랙은 상속으로 필요한 훅만 덮어쓴다.
- 보상·관측·목표·성공판정·DR 은 전부 `modules/` 단일 출처. A/B 는 액션 어댑터만 다르다.
- 접촉 센서: `_setup_scene` 을 덮어써 **ContactSensor 를 만들지 않는다**. `activate_contact_sensors` 스폰 플래그는
  자산에 무해하므로 그대로 둔다. 시너지 동결은 `synergy_hold_mode="blocked"`(지령↔실측 정체, 센서 무관)만 쓴다.

## 1. 액션

| | A grasp_kp | B grasp_fj |
|---|---|---|
| 차원 | 21 = palm 6D 델타 + 손 15D | 22 = 팔 7D + 손 15D |
| 팔 | grasp_s2r 그대로: anchor+delta → 박스 클램프 → 속도 리미터 → fabric → `set_joint_position_target` + `set_joint_velocity_target(fabric_qd)` | `q*_t = clamp(q*_{t-1} + k_arm·a, 관절한계)`, `q*_t = α·q*_t + (1−α)·q*_{t-1}`, **위치 목표만**(속도 목표 없음). EMA 가 누적 목표에 걸리므로 스텝당 변화는 정확히 α·k_arm·a → `k_arm = 0.167 rad/step`, `α = 0.1` ⇒ **실효 포화 slew 1.0 rad/s**(브리지 상한 = `arm_slew_rad_s`, cfg 가 대조). A 의 palm 리미터(0.02 m/step = 1.2 m/s)와 같은 자릿수. 리셋 시 `q*_{-1} = 홈 q`. 09.07: 팔 액션 1차 차분 `_prev_arm_action` → `cmd_rate` = RMS(Δa)/2 (리프트 후 반전 벌점 측도) |
| 손 | `_synergy_targets` 동일(15D → 20 관절, hold_mode=blocked) | 동일 |
| 액션 지연 | 큐 3 step, 매 스텝 인덱스 재추첨 | 동일 |

B 는 `_setup_fabrics/_init_home_palm/_step_fabric/_apply_action/_arm_command` 만 덮어쓴다. `_fab_to_env = 0`.
B 의 팔 목표 버퍼 `_arm_q_target (N,7)` 은 `self.arm_ids` 순서, 클램프는 `_arm_lo/_arm_hi`.

## 2. 목표 표현 — 축대칭 키포인트 4개 (yaw 불변)

컵은 축대칭이라 yaw 가 과제 무관 → 키포인트를 물체 z 축 위에 둔다:
`offsets = s · {(0,0,1), (0,0,−1), (0,0,1/3), (0,0,−1/3)}`, `s = 0.5 · keypoint_scale · fixed_height`
(`keypoint_scale 1.5`, `fixed_height 0.12` → s = 0.09 m). **보상·성공·관측 모두 고정 박스**를 쓴다
(SimToolReal 은 관측만 물체별 크기인데, 이 저장소 규칙 "obs 에 물체 정체성 없음"을 우선한다).
`d(o,g) = max_i ‖kp_i(o) − kp_i(g)‖`. 모든 위치는 env-local.

목표열: 리셋 시 첫 목표 = `settled_pos + [U(±first_xy), U(first_z_lo, first_z_hi)]`, 자세 = settled quat(직립).
성공(`d ≤ tol` **연속** `success_steps=10` — 09.07 `force_consecutive=True`; 누적이면 공차 안팎을 오가며 흔들려도 성공을 세어 준다. SimToolReal 논문 런처와 동일) 시 다음 목표 = 이전 목표에서 `±delta_distance` 균일 이동, `goal_box` 로 클램프,
회전은 `delta_rotation_deg`(기본 0 = 직립 유지; 붓기 확장 시 올린다). `max_goals` 도달 시 truncation.
에피소드 예산은 **600 step 고정**(목표당 예산 아님; `per_goal_budget=False`).
허용오차 커리큘럼: `tol_start 0.06 → tol_floor 0.015`, 3000 프레임마다 `mean(prev_episode_successes) ≥ 2.0` 이면 ×0.9.

기본값: `first_xy 0.05` · `first_z (0.16, 0.24)` · `delta_distance 0.08` · `goal_box = spawn_center ± (0.08, 0.08)`,
z ∈ [settled+0.10, settled+0.30] · `max_goals 50`.
**도달성(09.06 리뷰)**: A 의 palm 지령은 앵커(스폰+(−0.066,−0.022,0.085)) ± `palm_delta_xyz` 를 클램프 박스로 자른 범위다.
xy 반폭 0.15 는 프로필 palm 박스 x 하한 0.20 때문에 **물리적으로** 불가(≤ 0.096) → 0.08. z 0.30 은 델타를 (0.10, 0.10, **0.35**)
로 키워 덮는다(아래쪽은 `palm_box_min_z_override` 가 자른다 — 앵커를 올리면 a=0 이 "컵에서 도망"이 되는 Phase 0 함정 재발).
env `_assert_goal_box_in_arm_reach` 가 목표 박스 전 코너(스폰 ±range·정착고 극단, tol_floor 여유)를 부팅에서 대조한다(A 만; B 는 관절공간).
평가: `tol_eval > 0` 이면 tol 고정(커리큘럼 갱신 없음) — 커리큘럼 상태는 체크포인트에 없어 play 가 0.06 에서 다시 굴리면
성공수가 비교 불가다. `-play` id 는 `tol_eval = tol_floor`(hydra `env.tol_eval=` 로 변경).

## 3. 보상 (progress-only, 접촉 0) — `modules/progress_reward.py`

```
lifted     = (dz > lift_latch_height) | prev_lifted          dz = obj_z − settled_z,  lift_latch_height 0.10  (에피소드 리셋에서만 해제)
r_ft       = 50  · Σ_5 clamp(d*_ft − d_ft, 0, 10) · (¬lifted)   d_ft = ‖tip_i − obj_pos‖, d* = 지금까지 최소(−1 센티널→첫 관측값)
r_lift     = 20  · clamp(0.05 + dz, 0, 0.5) · (¬lifted)
r_liftbon  = 300 · [lifted 최초 진입]
r_kp       = 200 · clamp(d*_kp − d_kp, 0, 100) · lifted        d* 는 목표 전진 시 −1 로 초기화
r_goal     = (1000/10) · [near_goal]                            near_goal = d_kp ≤ tol
r_armvel   = −0.03  · Σ|q̇_arm|     r_handvel = −0.003 · Σ|q̇_hand|   (실측 관절속도)
r_floor    = −clamp(10 · relu(hand_floor_z − hand_z_min), max 5)   hand_floor_z = 0.215 (기하, 센서 아님)
r_cmd      = −cmd_rate_scale · cmd_rate · [lifted ∧ dz > 0.03] · ARMED   (09.07 A-v/B-v · A 0.1 · B 1.0, 상한 없음)
             ARMED = 전역 sticky 래치, lifted_frac EMA(α 0.002) ≥ 0.30 부터(a7: per-env 게이트만으로는 e25 우연 리프트에 붙어 접근 사망)
             cmd_rate: A = ½(‖Δp_raw‖/0.02 + ‖Δr_raw‖/2.9°) 리미터 **전** 원지령 변화(1 = 리미터에 딱 맞춤, 작동점 ≈10~15)
                       B = RMS_j(a_j − a_prev_j)/2 ∈ [0,1] 팔 액션 반전(전속 이송은 0)
```
항 이름(로깅 순서): `fingertip_progress, lift, lift_bonus, keypoint_progress, goal_bonus, arm_vel, hand_vel, hand_floor, cmd_rate`.
d*_ft 는 에피소드 리셋에서만, d*_kp 는 목표 전진마다 초기화(SimToolReal 동일).

## 4. 관측

actor(정책):
```
arm_q(7)+n  arm_qd(7)+n  hand_q(20)+n  hand_qd(20)+n  palm_pos(3)+n  palm_ax(6)  tips_rel_palm(15)+n
cmd_state(A: palm_targets−anchor 6 / B: q*_prev 7)  kp_rel_palm(12)*  kp_rel_goal(12)*  last_action(A)
```
`*` = 단일 지연(≤10 step)·노이즈(xyz 0.01, rot 5°) 자세 하나에서 파생(코히런트). 그 뒤 전체 벡터에 obs 지연 큐(≤3).
A: 7+7+20+20+3+6+15+6+12+12+21 = **129**, B: **131**. 물체 정체성·크기 없음.
물체 쿼터니언은 **넣지 않는다**(09.06 리뷰): 키포인트가 위치·기울기를 담고 남는 정보는 yaw 와 q≡−q 부호뿐인데, 축대칭 물체의
실기 yaw(FP++)는 임의라 sim 에서 ≈상수였던 채널이 배포 시 분포 밖(clip 5.0 포화)이 된다.
`last_action` 은 **적용된(지연 큐를 통과한) 액션**이고 `cmd_state` 도 그 액션에서 계산된다 — 배포 노드는 자기가 실제로
제어기에 보낸 액션 스트림(지연 0 = 학습 분포 안)을 두 항 모두에 넣어야 한다.

critic(state) = actor 의 clean 판 + obj lin/ang vel(6) + palm lin/ang vel(6) + d*_kp(1) + d*_ft(5) + lifted(1)
+ progress(1)=ep_len/max + successes(1) + reward(1)·0.01 + dz(1) + d_kp(1)  → obs + 24.

## 5. 외란·종료·DR

- 외란(`modules/object_wrench.py`): 매 스텝 `p_force, p_torque ~ logU(0.001, 0.1)`(env 별, 리셋마다 재추첨)로
  `randn·mass·20`(N) / `randn·mass·2`(N·m) 새로 뽑고, 아니면 이전 값 유지(decay 0 → 매 스텝 0 으로 소거, SimToolReal 동일),
  **lifted 일 때만** 적용. `object.set_external_force_and_torque(is_global=True)`.
- 종료: fell(`obj_z < object_min_z`), out_x/out_y, tipped(60°), abnormal, `hand_floor_terminate_depth`(손 링크가 상판보다
  0.03 m 아래 → 종료, B 의 테이블 방어), `max_goals` 는 truncation. `respawn_on_fail=False`(SimToolReal 처럼 낙하는 리셋).
  `extras["time_outs"]` 는 IsaacLab 래퍼가 truncated 로 채운다(value_bootstrap True 필수).
- palm 박스 z 하한: A 는 `palm_box_min_z_override = 0.27`(손 최하단이 palm 원점 −57 mm, 상판 0.205 → 관통 방지;
  09.06 실측 "a=0 에서 49 mm 뚫림"의 원인 제거).
- DR: ADR off, 물리 DR 항등(grasp_s2r 동일), 관측 노이즈 기본값, 지연 큐 3/3/10 켬. 관절속도 노이즈 0.1 rad/s.

## 6. PPO (rl_games, `config/agents/rl_games_ppo_lstm_cfg.yaml`)

grasp_s2r LSTM cfg 복사 후: `value_bootstrap: True`, `gamma: 0.99`, actor mlp `[1024,1024,512,512]`, critic mlp `[1024,1024,512,512]`,
critic lr `1e-4`, kl 0.016/0.016, 나머지 유지(minibatch 16384, mini_epochs 4, horizon 16, seq 16, entropy 0.002, e_clip 0.2).
env 4096(server). `mixed_precision` 은 False 유지.

## 7. 검증 게이트

1. 순수 torch 모듈 단위 테스트(pytest, Isaac 불필요): 키포인트 불변성(yaw 회전에 d=0, tilt/이동에 단조), 진행 보상의
   센티널·단조성·소등 게이트, 지연 큐의 flush/범위, 외란의 lifted 게이트·질량 비례, 커리큘럼 게이트.
2. cfg 계약 테스트: 차원 공식 = 실제 assembling, 접촉 필드 미소비(grep), `synergy_hold_mode=="blocked"`,
   `respawn_on_fail False`, id 등록.
3. 로컬 스모크(5090, 16 env, 수 iter) → 부팅·shape·NaN 확인.
4. server gpu0 = grasp_kp, gpu1 = grasp_fj, 4096 env. 스모크 게이트(grasp_s2r CLAUDE.md): `ep_len` 상승 · abnormal≈0 ·
   `joint_err<0.1`(A) · `hand_floor_depth_max≈0` · `stage/lifted` 상승.

## 8. 모듈 API 계약 (구현자·검토자 공통 — 여기서 벗어나면 안 된다)

모든 모듈은 **torch 만** import 한다(isaaclab 금지 — 시스템 python3 pytest 로 돈다). 쿼터니언은 **wxyz**. 필요한
`quat_apply/quat_mul/quat_from_angle_axis` 는 `keypoint_goal.py` 안에 자체 구현한다(isaaclab.utils.math 와 동일 규약).

### modules/keypoint_goal.py
```python
KEYPOINT_AXIAL_UNIT = ((0,0,1.0),(0,0,-1.0),(0,0,1/3),(0,0,-1/3)); NUM_KEYPOINTS = 4
def quat_apply(q_wxyz, v) / quat_mul(a, b) / quat_from_angle_axis(angle, axis) / random_small_rotation(n, max_deg, device) -> (n,4)  # max_deg==0 → identity
def keypoint_offsets(half_height: float, device) -> (4,3)
def keypoints_world(pos (N,3), quat (N,4), offsets (4,3)) -> (N,4,3)
def keypoint_max_dist(kp_a (N,4,3), kp_b (N,4,3)) -> (N,)
@dataclass(frozen=True) class GoalSeqCfg:
    first_xy_range: float = 0.05; first_z_range: tuple = (0.16, 0.24); first_tilt_deg: float = 0.0
    delta_distance: float = 0.08; delta_rotation_deg: float = 0.0
    box_min: tuple = (-1e9,)*3; box_max: tuple = (1e9,)*3      # env-local 절대 박스(클램프)
    success_steps: int = 10; force_consecutive: bool = False; max_goals: int = 50
def sample_first_goal(settled_pos (N,3), settled_quat (N,4), cfg) -> (pos (N,3), quat (N,4))
def sample_delta_goal(prev_pos (N,3), prev_quat (N,4), cfg) -> (pos, quat)   # 이전 **목표** 기준, 박스 클램프, 회전은 world 프레임 pre-multiply
class GoalTrackers:   # 단순 텐서 컨테이너 (num_envs, num_tips, device)
    closest_kp (N,) float init -1 · closest_ft (N,K) float init -1 · near_goal_steps (N,) long · successes (N,) long · prev_episode_successes (N,) long
    def clear_goal(self, ids)      # closest_kp/closest_ft = -1, near_goal_steps = 0   (목표 전진)
    def full_reset(self, ids)      # prev_episode_successes[ids] = successes[ids]; successes = 0; clear_goal(ids)
def progress_delta(curr, closest) -> (delta, new_closest)   # closest<0(센티널) → delta 0·new=curr, 아니면 delta=clamp(closest-curr, min=0)·new=min(closest,curr). (N,) 또는 (N,K) 모두 지원
def update_near_goal(kp_dist (N,), tol: float, trackers, cfg) -> (near_goal bool (N,), is_success bool (N,))   # near_goal_steps 갱신(force_consecutive 규약), is_success = steps ≥ success_steps
class ToleranceCurriculum:
    def __init__(self, start: float, floor: float, factor=0.9, interval=3000, success_threshold=2.0)
    tol: float (property) · def update(self, prev_episode_successes: Tensor) -> bool   # 프레임 카운터 내부, interval 마다 mean ≥ threshold 면 tol=max(tol*factor, floor)
```

### modules/progress_reward.py
```python
PROGRESS_REWARD_TERMS = ("fingertip_progress","lift","lift_bonus","keypoint_progress","goal_bonus","arm_vel","hand_vel","hand_floor","cmd_rate")
@dataclass(frozen=True) class ProgressRewardCfg:
    ft_scale=50.0; lift_scale=20.0; lift_base=0.05; lift_clip=0.5; lift_bonus=300.0; lift_latch_height=0.10
    kp_scale=200.0; goal_bonus=1000.0; success_steps=10
    arm_vel_scale=0.03; hand_vel_scale=0.003
    hand_floor_penalty=10.0; hand_floor_z=0.215; hand_floor_max=5.0; cmd_rate_scale=0.1; cmd_rate_hold_dz=0.03  (env cfg rw_cmd_rate_scale: A 0.1 · B 1.0)
def compute_progress_reward(*, obj_z (N,), settled_z (N,), lifted_prev (N,) bool, ft_dist (N,K), closest_ft (N,K), kp_dist (N,), closest_kp (N,),
                            near_goal (N,) bool, arm_qd (N,7), hand_qd (N,20), hand_z_min (N,), cmd_rate (N,) ≥0, cfg: ProgressRewardCfg)
    -> (total (N,), terms: dict[str, (N,)] (PROGRESS_REWARD_TERMS 전부, 순서 동일), out: dict(lifted=bool(N,), just_lifted=bool(N,), closest_ft=(N,K), closest_kp=(N,)))
# lifted = (obj_z - settled_z > lift_latch_height) | lifted_prev ; lift = lift_scale·clamp(lift_base + dz, 0, lift_clip)·(¬lifted) ; lift_bonus = lift_bonus·just_lifted
# fingertip_progress = ft_scale·Σ_k clamp(progress_delta)·(¬lifted) ; keypoint_progress = kp_scale·progress_delta·lifted ; goal_bonus = (goal_bonus/success_steps)·near_goal
# arm_vel = -arm_vel_scale·Σ|arm_qd| ; hand_vel = -hand_vel_scale·Σ|hand_qd| ; hand_floor = -clamp(hand_floor_penalty·relu(hand_floor_z - hand_z_min), max=hand_floor_max)
# cmd_rate = -cmd_rate_scale·clamp(cmd_rate, min 0)·(lifted ∧ dz > cmd_rate_hold_dz)   (측도는 env: A 리미터 전 원지령 변화/리미터 상한 평균 · B 팔 액션 Δ RMS/2; 상한 없음; 전역 arm 래치는 env 가 cmd_rate 에 곱해 준다)
# total = nan_to_num(sum(terms))
```

### modules/perception_delay.py
```python
class DelayQueue:
    def __init__(self, num_envs, max_delay: int, dim: int, device)      # buf (N, L=max_delay, dim); max_delay ≥ 1 (1 = 지연 없음)
    def reset(self, ids)                                                 # 0 으로
    def push(self, values (N,dim), flush (N,) bool) -> (N,dim)           # flush env 는 전 슬롯을 values 로 채움 → roll(1, dim=1) → [:,0]=values → idx~randint(0,L) env 별 → buf[arange, idx]
def perturb_quat(quat_wxyz (N,4), max_deg: float) -> (N,4)              # 무작위 축·U(-max,max) 각도, max_deg==0 → 그대로
def noisy_pose(pos (N,3), quat (N,4), xyz_std: float, rot_deg: float) -> (pos, quat)
```

### modules/object_wrench.py
```python
def sample_log_uniform(lo, hi, n, device) -> (n,)
class WrenchDR:
    def __init__(self, num_envs, device, *, force_scale=20.0, torque_scale=2.0, prob_range=(0.001, 0.1))
    forces (N,1,3) · torques (N,1,3) · p_force (N,) · p_torque (N,)
    def reset(self, ids)                                    # 확률 재추첨(log-uniform)·wrench 0
    def step(self, mass (N,), lifted (N,) bool) -> (forces (N,1,3), torques (N,1,3))   # decay 0: 매 스텝 0 으로 소거 → fire ~ rand<p → new=randn·mass·scale → lifted 게이트
```

### grasp_kp env 배선 규약 (GraspKPEnv(GraspS2REnv))
- 덮어쓰는 훅: `_setup_scene`(접촉 센서 생성부 제거, 나머지 mixin 과 동일) · `_init_task_state`(super() 후 키포인트/목표/큐/외란 버퍼 추가) ·
  `_pre_physics_step` = 액션 지연 큐 → `self.actions` → `_arm_command()` → `_hand_command()` → `_post_command()` → 외란 적용 ·
  `_get_observations` · `_get_rewards` · `_get_dones` · `_reset_idx`(super() 호출 후 목표·추적기·큐·외란 리셋).
  A 의 `_arm_command` = grasp_s2r `_pre_physics_step` 648-687 그대로, `_hand_command` = 689-714 그대로, `_post_command` = 717-719(fabric 손 동기화 + `_step_fabric`).
- `self._latched` 를 **높이 래치(lifted)** 로 재정의한다(close gate·anchor·species 로깅이 그대로 쓴다). `_hold_count/_wrap_at_latch/_disp_at_latch` 는 쓰지 않는다.
- 목표 상태: `self.goal_pos (N,3)`(부모 버퍼 재사용) + `self.goal_quat (N,4)`. 물체 정착 자세 = `object_spawn_pos`(부모가 정착고로 씀) + 단위 쿼터니언.
- 리셋 순서: `super()._reset_idx(env_ids)` → `goal_pos/goal_quat = sample_first_goal(settled)` → `trackers.full_reset` → 큐 reset → `wrench.reset` → `_latched=False`.
- `_get_rewards` 순서: 기하(ft_dist·kp_dist·hand_z_min) → `near_goal, is_success = update_near_goal(...)` → `compute_progress_reward(...)` → 상태 갱신(`_latched`, closest) →
  is_success env: `successes += 1` · `clear_goal` · `sample_delta_goal` → `_success_now = is_success` → abnormal 벌점 → extras.
- `_get_dones`: 부모 기하(tilt·out·fell·abnormal) + `hand_floor_terminate_depth` + truncated = time_out | successes ≥ max_goals. respawn 없음.
- `_get_observations`: 물체 (pos,quat) 를 DelayQueue(10, dim 7) 에 push(flush = episode_length_buf==0) → `noisy_pose` → 키포인트 파생 → actor 벡터 →
  DelayQueue(3, obs_dim) → policy. critic 은 clean. **`_derive_spaces` 와 정확히 같은 순서·차원**.
- 관측 노이즈 스칼라는 부모의 `_adr_obs_noise_qpos/_qvel`·`cfg.obs_noise_body` 그대로, 물체 노이즈는 `obs_object_xyz_std 0.01`·`obs_object_rot_deg 5.0`.
- cfg 신설 필드(전부 GraspKPEnvCfg): `keypoint_scale 1.5`, `keypoint_fixed_height 0.12`, `goal_first_xy_range`, `goal_first_z_range`, `goal_delta_distance`, `goal_delta_rotation_deg`,
  `goal_box_xy_halfwidth 0.08`, `goal_box_z_range (0.10, 0.30)`, `goal_success_steps 10`, `goal_max 50`, `tol_start 0.06`, `tol_floor 0.015`, `tol_factor 0.9`, `tol_interval 3000`,
  `tol_success_threshold 2.0`, `tol_eval 0.0`(>0 = 고정 tol, play 는 tol_floor), `reward_*`(ProgressRewardCfg 필드 그대로 접두사 `rw_`), `obs_delay_steps 3`, `action_delay_steps 3`, `object_delay_steps 10`,
  `obs_object_xyz_std 0.01`, `obs_object_rot_deg 5.0`, `obs_noise_qvel 0.1`(SimToolReal 값으로 상향) + `adr_obs_noise_qvel_max 0.1`(부모 단조 가드는 ADR OFF 여도 base ≤ max 를 요구),
  `wrench_force_scale 20`, `wrench_torque_scale 2`, `wrench_prob_range`,
  `hand_floor_terminate_depth 0.03`, `palm_box_min_z_override 0.27`, `arm_cmd_dim 6`. 기존 필드 덮어쓰기: `respawn_on_fail False`, `synergy_hold_mode "blocked"`,
  `synergy_contact_freeze False`, `obs_object_noise_coherent True`, `enable_adr False`, `palm_delta_xyz (0.10, 0.10, 0.35)`(목표 박스 z 도달). B 추가: `k_arm 0.167`, `arm_ema 0.1`, `arm_slew_rad_s 1.0`.
