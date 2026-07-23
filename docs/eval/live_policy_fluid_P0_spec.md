# P0 — 라이브 정책 루프 포팅 스펙 (right/pour_v1)

> **목표(재확인)**: `SimulationContext` 텐서 파이프라인 없이 raw Isaac Sim 루프에서 pour_v1 정책을
> 라이브 실행. 이 문서는 `pour_right_env.py`(3250줄) 중 **정책 obs 조립 + action 파이프라인**을
> 정확히 재현하기 위한 포팅 명세다. GPU 불필요(코드 정독 산물).
>
> 상위 계획: [[live_policy_fluid_plan.md]]. 관련 메모리: `pbd-particles-need-raw-app-loop`.

---

## 0. 핵심 결론 (portability verdict)

| 컴포넌트 | 텐서 파이프라인 의존? | 포팅 방법 |
|---|---|---|
| **Fabrics IK** (`OpenArmTeoslloPoseFabric`, `DisplacementIntegrator`) | **아니오** | 통째로 lift. `num_envs`/`device`/`timestep`만으로 생성, 순수 torch 텐서(`fabric_q/qd/qdd`) 연산. Isaac `SimulationView` 불참조. (env.py 806·820줄) |
| obs 조립 (`_get_observations`) | 읽기만 | `.data.*` 60곳을 USD 쿼리로 대체하면 수식 그대로 |
| geometry (`_compute_intermediate_values`) | 읽기만 | 동일 |
| action geometry / nullspace / β-traj | **아니오** | 순수 torch — 복붙 |
| helper (`scale`, `pour_corridor_score`, `_build_cup_local_tilt_rotvec`, `_compose_world_delta_quat_xyzw`, `_quat_xyzw_from_euler_zyx`) | **아니오** | pour_right_utils.py + static/instance 메서드, 전부 순수 텐서 |

**➡ 포팅 경계 = sim state read(60곳) + write(아래 §5)뿐.** 나머지 로직은 그대로 이식 가능.
이것이 P0 최대 발견 — 최대 리스크는 코드 복잡도가 아니라 **cm-스케일 contact grasp 물리**(P1)와
**USD 상태 read 정확도**(§5)다.

---

## 1. Actor obs = 55D (정책이 실제로 보는 것)

`NUM_OBSERVATIONS = 55` (pour_right_constants.py:81). 조립: env.py:1752.
정책은 **bead/유체 미관측** — 라이브든 리플레이든 행동 동일(계획서 확정 사실).

| # | 채널 | dim | 소스 (clean) | noise σ | 비고 |
|---|---|---|---|---|---|
| 1 | `arm_joint_pos` | 7 | `robot.data.joint_pos[:, arm_dof_indices]` | `σ_qp` | 오른팔 7-DOF |
| 2 | `arm_joint_vel` | 7 | `robot.data.joint_vel[:, arm_dof_indices]` | `σ_qv` | |
| 3 | `finger_grasp_progress` | 5 | `_finger_grasp_progress(finger_joint_pos)` | (qp 통해) | per-finger [0,1], §1.1 |
| 4 | `left_arm_joint_pos` | 9 | `robot.data.joint_pos[:, left_arm_dof_indices]` | `σ_qp` | 왼팔(고정 자세) |
| 5 | `left_arm_joint_vel` | 9 | `robot.data.joint_vel[:, left_arm_dof_indices]` | `σ_qv` | |
| 6 | `pour_point_to_opening` | 3 | `target_opening - source_pour_point` | `σ_cp` (양항) | §2 |
| 7 | `source_pour_axis_clean` | 3 | `_source_pour_axis_w` | 0 (clean) | §2 |
| 8 | `source_up_axis_clean` | 3 | `_source_up_axis_w` | 0 | §2 |
| 9 | `target_up_axis_clean` | 3 | `_target_up_axis_w` | 0 | §2 |
| 10 | `last_palm_actions` | 6 | `self.actions[:, :6]` | — | 직전 palm action(α 제외) |

합 = 7+7+5+9+9+3+3+3+3+6 = **55**. 계약 테스트: `tests/test_actor_observation_layout.py`.

> **noise**: 라이브 평가는 `_warmstart_collect_mode`도 아니고 학습도 아님 → **eval에선 σ=0으로 두는 게
> 정답**(record→replay 재현 기준과 일치). cfg default(`obs_noise_*`)를 0으로 강제하거나 clean 값 직접 사용.
> `source_pour_axis/up_axis/target_up_axis`는 원본에서 이미 clean(noise 미적용).

### 1.1 `_finger_grasp_progress` (env.py:1678)
20개 손가락 관절을 5핑거×4로 접어 평균. `progress = clip((q - hand_open_pose)/(hand_grasp_pose - hand_open_pose), 0, 1)`.
`hand_open_pose`·`hand_grasp_pose`는 reset 시 warmstart 캐시에서 설정되는 버퍼 → **포팅 시 동일 값 필요**.
단, 라이브에서 손은 `grasp_hold`로 freeze(§4 말미)이므로 progress는 사실상 상수. 그래도 obs 정합 위해 정확 계산.

---

## 2. obs가 필요로 하는 중간 기하값 (`_compute_intermediate_values`, env.py:1428)

전부 **컵 2개 포즈 + 로봇 body 포즈**에서 유도. 라이브 포팅 핵심 = 이 값들을 USD 쿼리로 재계산.

- **`_source_rim_center_w`** = `cup.root_pos_w + quat_apply(cup.root_quat_w, _source_cup_pour_point_pos_b)` (1463)
- **`_target_opening_w`** = `left_cup.root_pos_w + quat_apply(left_cup.root_quat_w, _target_cup_opening_pos_b)` (1474)
- **`_source_pour_axis_w`** = `quat_apply(cup.root_quat_w, _source_cup_pour_axis_b)` (1513)
- **`_source_up_axis_w`** = `quat_apply(cup.root_quat_w, _source_cup_up_axis_b)` (1517)
- **`_target_up_axis_w`** = `quat_apply(left_cup.root_quat_w, _target_cup_up_axis_b)` (1521)
- **`_source_pour_point_w`** (1495–1512): **동적 blend 주둥이**. gravity_perp(자세) + smoothstep(tilt 깊이)로
  정적(target 방향)↔동적(실 배출구) 혼합. `source_outer_radius`, `pour_point_dyn_lo/hi` cfg 사용.
  → obs #6과 action rim-pivot 양쪽에서 쓰이므로 **정확 재현 필수**.
- **`palm_center_pos`** (1444–1450) = `palm body_pos_w + quat_apply(palm body_quat_w, _palm_ee_offset_local)`;
  `_palm_ee_offset_local = [0.028, 0, 0.04]` (274). palm_ee = 진짜 손바닥 중심.
- `_mouth_*`, `_rim_antiparallel`, `_directional_tilt_cos*`, `_cup_center_xy_dist`, `_rho`,
  `_internal_rot_gate`, `_rim_facing_cos` → **critic/reward 전용**(actor obs 미사용)이나
  `_pour_ready_latched` 갱신이 corridor_score(§6)에 의존하므로 pour_point/target_opening은 필수.

**body-frame 상수**(`_source_cup_*_b`, `_target_cup_*_b`): reset/init에서 설정되는 컵 기하 상수 →
포팅 시 그대로 로드(값 고정).

---

## 3. Fabrics IK 계층 (lift 대상)

- 생성: `_setup_geometric_fabrics` (env.py:790). `OpenArmTeoslloPoseFabric(num_envs, device, timestep,
  graph_capturable=False, use_hand_fabric=False, palm_position_only=cfg.palm_position_only)`.
- 상태 버퍼: `fabric_q/qd/qdd` (num_joints=27), `hand_pca_targets`(5), `palm_pose_targets`(7),
  `fabric_damping_gain`.
- 스텝: `set_features(hand_pca_targets, palm_pose_targets, "quaternion", fabric_q, fabric_qd,
  object_ids, object_indicator, fabric_damping_gain)` → `integrator.step(...)` ×`fabric_decimation(2)`.
- `world_model = WorldMeshesModel(..., world_filename="open_tesollo_boxes_pour_v5")` → 충돌 회피용
  object_ids/indicator. **월드 메시(장애물)도 라이브 씬과 정합 필요**.
- `get_taskmap_jacobian("palm")` → B-full nullspace(§4)에서 사용. Fabrics 내부 태스크맵, 텐서 독립.

> **의존성 확인 필요(P0 잔여)**: `fabrics_sim`이 warp를 쓰며 `initialize_warp` 호출.
> raw-app 루프에서 warp init이 SimulationContext 없이도 되는지 → **P3 착수 전 단독 스모크 테스트** 권장.

---

## 4. Action 파이프라인 (`_pre_physics_step`, env.py:1071) — 활성 config 경로만

**입력**: `actions` (N,12) = palm 6D(`[:6]`) + α nullspace(`[6]`) + hand 5D(`[7:12]`, **미사용**).

**활성 config — ✅ 평가 체크포인트 저장 cfg로 확정 (2026-07-05, task-a):**
> 정답 소스 = `log/rl_games/open-tesol/right/pour-v1/lstm_test2/params/env.yaml`.
> `record_pour_traj.py._restore_run_cfg_if_available`(스크립트 331줄)가 replay 시 **이 저장 cfg를 복원**하므로,
> η_ft 33.5%는 아래 값으로 생성됨. 라이브 포팅은 이 값에 핀한다.
```
pour_action_mode      = "b_trajectory"   # β setpoint (action[4]=β 채널, beta_action_index=4)
pour_approach_pivot   = "palm"           # action xy가 palm 직접 이동
pour_orient_release   = True             # B-light: ready 후 orientation 풀기 + 주둥이 hold
pour_bfull_nullspace  = True             # B-full: J_spout nullspace 투영으로 demo deep-tilt 강제
pour_spout_z_lock     = True             # 주둥이 z를 target 입구 + pour_z_margin으로 구조 강제
nullspace_baseline    = "demo"           # ✅ 저장 cfg = demo (robot_start 아님)
pour_phase_clamp_enable = False          # post-IK 관절 클램프 OFF
episode_hold_steps    = 120              # 시작 120스텝 palm/α=0 (warmstart 물리 안착)
ema_action_alpha      = 0.7 ; fabric_decimation = 2 ; nullspace_action_scale = 1.0
enable_demo_pose_reward = False ; enable_demo_critic_obs = True   # actor demo off, critic만
obs_noise_joint_pos=0.01 ; joint_vel=0.05 ; body_pos=0.005 ; cup_pos=0.015  # ADR base (아래 주의)
```
> **평가 체크포인트/아티팩트 확정**: task `open-tesol_r_pour_v1-lstm`, run `pour-v1/lstm_test2`,
> ckpt `nn/last_open-tesol_r_pour_v1-lstm_ep_10000_rew_47074.97.pth`. (docs/eval/pour_v1_report.md 정합)
>
> **CLAUDE.md의 "`deep_tilt_boot1` = `robot_start` 순수 DRL" 서술은 stale** — 그건 이전 실험 단계이고,
> 실제 eval 체크포인트(lstm_test2)는 **`demo` + B-full**로 학습됨. 위 파이프라인이 재현에 정확.
>
> ⚠️ **obs noise 주의(P2 결정)**: 저장 cfg의 `obs_noise_*`>0은 `--disable_adr`로도 완전히 0이 안 됨 —
> env.py 1720-1729에서 `enable_noise_adr=False`면 `noise_adr=None`→`else` 분기로 **base σ(0.01 등)가 잔존**.
> record 궤적엔 이 noise가 이미 baked. 라이브는 정책이 noise-robust하므로 **σ=0으로 fresh 실행**해도
> 동일 정책 행동(계획서 확정: 유체 미관측)이나, η_ft 정확 재현엔 same-seed 불가 → **aggregate 수준 대조**가 현실적.

**스텝 순서 (활성 경로):**
1. `prev_actions ← actions`; `palm_action = actions[:,:6]`, `alpha = actions[:,6]`.
2. **hold** (`episode_length < 120`): palm_action·alpha = 0.
3. hold 종료 첫 스텝: bead 지연 소환(라이브에선 PBD 유체 소환으로 대체 — P4).
4. **EMA**: `_ema_palm_action = 0.7·palm_action + 0.3·prev`; α도 동일.
5. `delta_pre_gate = scale(_ema_palm_action, delta_mins, delta_maxs)` (env.py 295·299 정의).
6. **tilt_gate** = clamp((`tilt_action_gate_xy_far` - `_mouth_xy_distance`)/den, 0,1) — 원거리 tilt 억제.
7. **β-traj** (`b_trajectory`): `β = clip(_ema_palm_action[4]·0.5+0.5, 0,1)`; 목표 tilt_amount =
   `β·beta_target_tilt_amount`; `delta[:,4] = clip(beta_tilt_kp·(target - cur_ta), ±max_step)`.
   `cur_ta = (1 - _rim_antiparallel)/2`.
8. `delta[:,3:6] *= tilt_gate`; `delta_rotvec_world = _build_cup_local_tilt_rotvec(delta[:,3:6])`
   (cup-local: spin/tilt-toward/ortho → world rotvec).
9. **palm target** (pivot="palm"): `_palm_ee_target = palm_center_pos + delta[:,:3]`;
   z는 `pour_point_target_z - (R·rim_rel)_z`로 환산(주둥이 z-lock). clamp `palm_mins/maxs`.
   orientation = `_compose_world_delta_quat_xyzw(current_palm_quat, delta_rotvec_world)`.
   palm_ee→palm_link 역변환(`_palm_ee_offset_local` 뺌) → `palm_pose_targets`.
10. **orient_release**(ready만): orientation target=현재 palm quat(틸트 명령 제거), 주둥이 위치 hold
    (`_spout_offset_body` 동결). `_pour_ready_latched` 마스크로 where.
11. **nullspace baseline**: `_baseline_arm = robot_start[:7]`; demo/B-full 분기로 j1-4(+ready 시 j5)를
    `_demo_pour_arm_pose`로 설정. `_null_ref = clamp(baseline + nullspace_action_scale·α·_nullspace_offset_arm)`.
    → `open_tesollo_fabric.default_config`.
12. `set_features(..., "quaternion", ...)` → `integrator.step` ×2.
13. **B-full nullspace**(`pour_bfull_nullspace` & any ready): `get_taskmap_jacobian("palm")`로 J_spout 구성,
    DLS pinv nullspace 투영으로 arm을 `_demo_pour_arm_pose`(deep tilt)로 끌되 주둥이 위치 불변(J_spout·Δq=0).
    `bfull_step`·`bfull_lambda`.
14. `pour_phase_clamp_enable=False` → skip.
15. **hand freeze**: `hand_joint_targets = grasp_hold_hand_pos_buf`; `fabric_q[7:]=hand_target`, qd=0.
    → 손은 warmstart 파지 전구간 freeze(컵-손 rigid, drift 방지).

**`_apply_action` (env.py:1398):**
- 오른팔: `robot.set_joint_position_target(fabric_q[:7], arm_dof_indices)`, vel target=0.
- 오른손: `set_joint_position_target(hand_joint_targets, hand_dof_indices)`, vel=0.
- 왼팔: `left_arm_zero_pos` 고정.
- 왼컵(target): `write_root_pose_to_sim(_get_left_target_cup_fixed_pose())` + vel=0 → **kinematic-follow**.

---

## 5. Sim state 인터페이스 (텐서 → USD/PhysX 대체 매핑)

**READ** (라이브 매 스텝, `.data.*` 60곳):
| 원본 텐서 | 의미 | USD/PhysX 대체 |
|---|---|---|
| `robot.data.joint_pos/vel[:, idx]` | 관절 상태 | `UsdPhysics` joint `state:*:physics:position/velocity` 또는 physx articulation view |
| `robot.data.body_pos_w/quat_w[:, body_idx]` | palm/fingertip/distal 포즈 | `UsdGeom.Xformable.ComputeLocalToWorldTransform` |
| `cup.data.root_pos_w/quat_w` | source 컵 | rigid body USD/physx pose |
| `left_target_cup.data.root_pos_w/quat_w` | target 컵 | 동일(단 kinematic-follow라 우리가 씀) |
| `scene.env_origins` | env 오프셋 | 단일 env면 0 (라이브는 1 env) |
| `contact_force_raw` (tip_force) | fingertip 접촉력 | PhysX contact report API (actor obs 미사용 → 라이브 obs엔 불필요, critic만) |

**WRITE** (매 스텝):
| 원본 | 대체 |
|---|---|
| `robot.set_joint_position_target(arm/hand)` | USD joint drive target 속성 set (`drive:*:physics:targetPosition`) |
| `robot.set_joint_velocity_target(...)=0` | drive target velocity 0 |
| `left_target_cup.write_root_pose_to_sim` | rigid body pose 직접 set(kinematic) |

> actor obs는 **contact_force 미참조** → 라이브 obs 조립엔 PhysX contact report 불필요.
> tip_force는 critic/reward 전용. 라이브 평가는 obs만 있으면 정책 구동 가능.

---

## 6. 스텝 순서 & `_pour_ready_latched` 인과 (주의)

- `_pour_ready_latched`는 **`_get_rewards`에서 갱신**됨 (env.py:2045):
  `latched |= corridor_score >= ready_latch_threshold`.
  `corridor_score = pour_corridor_score(_source_pour_point_w, _target_opening_w, corridor_radius, z_min, z_max, scale)`.
- `_pre_physics_step`는 이 latch를 **직전 스텝 값**으로 읽음(orient_release·B-full 게이트).
- DirectRLEnv 순서: `_pre_physics_step` → sim.step ×decimation → `_get_dones` → `_get_rewards`(여기서 latch/intermediate 갱신).
- **라이브 루프도 동일 순서 유지 필수**: (1) `_compute_intermediate_values` 재계산 → (2) `_pour_ready_latched`
  갱신(corridor_score) → (3) obs 조립 → (4) `player.get_action` (LSTM 상태 유지) → (5) action 파이프라인
  (직전 latch 사용) → (6) fabric integrate → (7) drive target set → (8) `app.update()` ×decimation.
  라이브에선 reward 불필요하나 **latch 갱신 로직만은 이식**(action 게이트가 의존).

---

## 7. 상수/버퍼 핀 (reset/init에서 설정 → 포팅 시 고정 로드)

- `_palm_ee_offset_local = [0.028, 0, 0.04]`
- `palm_mins/maxs = PALM_POSE_MINS/MAXS_FUNC(max_pose_angle)` (workspace clamp)
- `delta_mins/maxs` (env.py:295/299) — action→delta 스케일. **obs #10 해석·action 적용 양쪽 정합 필수**.
- `_demo_pour_arm_pose = DEMO_POUR_ARM_POSE` (7,)
- `_nullspace_offset_arm` (env.py:410)
- `robot_start_joint_pos`, `_arm_joint_min/max`
- `hand_open_pose`, `hand_grasp_pose`, `grasp_hold_hand_pos_buf` — warmstart 캐시 산물(§4·P1 grasp에서 확정)
- `_source_cup_*_b`, `_target_cup_*_b` (컵 body-frame 기하 상수)
- cfg 스칼라: `beta_target_tilt_amount`, `beta_tilt_kp`, `beta_tilt_max_step`, `tilt_action_gate_xy_near/far`,
  `pour_z_margin`, `pour_point_dyn_lo/hi`, `source_outer_radius`, `bfull_step`, `bfull_lambda`,
  `ready_latch_threshold`, `pour_corridor_*`, `cspace_attractor_mass`, `fabrics_damping_gain`.

---

## 8. P0 이후 잔여 리스크 / 결정지점

1. ~~**평가 체크포인트 config 확정**~~ ✅ **완료(task-a)**: lstm_test2 저장 env.yaml = `demo`+B-full 확정(§4).
   잔여는 **obs noise 정합**(base σ 잔존)뿐 — P2에서 σ=0 fresh vs same-noise 결정.
2. **warp init in raw-app**: `initialize_warp` + Fabrics가 SimulationContext 없이 동작하는지 단독 스모크(§3).
3. **warmstart grasp 상태**(hand_open/grasp/grasp_hold, _source_cup_*_b): P1 contact grasp와 함께 확정.
   라이브에서 손 freeze값이 obs `finger_grasp_progress`를 결정.
4. **cm 스케일 contact grasp 안정성**(계획서 최대 리스크): §4 손 freeze가 컵을 실제로 붙잡는지 — P1.
5. **USD read 정확도**: joint/body 포즈를 1e-3 이하로 읽어야 정책 정합(계획서 리스크 2).

---

## 부록: 참조 라인 인덱스 (right/pour_v1/pour_right_env.py)

| 블록 | 라인 |
|---|---|
| `_pre_physics_step` | 1071 |
| `_apply_action` | 1398 |
| `_compute_intermediate_values` | 1428 |
| `_finger_grasp_progress` | 1678 |
| `_get_observations` (actor 55D) | 1695 / actor_obs 1752 |
| `_get_rewards` (ready latch 2045) | 1934 |
| `_get_dones` | 2443 |
| `_reset_idx` | 2546 |
| `_build_warmstart_reset_cache` | 2840 |
| `_setup_geometric_fabrics` | 790 |
| constants (NUM_*) | pour_right_constants.py:65–97 |
| 활성 cfg | pour_right_env_cfg.py (§4) |
