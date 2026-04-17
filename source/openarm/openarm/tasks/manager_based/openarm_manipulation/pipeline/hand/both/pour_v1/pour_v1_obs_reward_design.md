# `pour_v1` Left-Arm Assist Obs/Reward Design

## Goal
`both/pour_v1`에서 오른손은 기존 warmstart grasp prior를 유지하고, 왼팔은 target cup을 source cup 쪽으로 가져와 bead capture를 돕는 역할로 학습한다.

이 버전의 설계 목적은 가장 작은 구현 단위를 만드는 것이다.

## Observation

Actor/critic obs는 동일한 134D로 유지한다.

| Group | Dim | Notes |
| --- | ---: | --- |
| `right_joint_pos` | 27 | 오른팔+오른손 actuated joint position |
| `right_joint_vel` | 27 | 오른팔+오른손 actuated joint velocity |
| `left_arm_joint_pos` | 7 | 왼팔 joint position |
| `left_arm_joint_vel` | 7 | 왼팔 관절 속도 |
| `fingertip_pos` | 15 | env-local fingertip positions |
| `cup_pose_vel` | 13 | source cup pos, quat, lin vel, ang vel |
| `target_opening_pos` | 3 | target cup opening world position의 env-local 값 |
| `bead_centroid_pos` | 3 | proxy bead centroid env-local position |
| `prev_actions` | 18 | 직전 policy action 전체 |
| `mouth_delta` | 3 | `target_opening - source_pour_point` |
| `mouth_xy_distance` | 1 | opening 간 XY 거리 |
| `mouth_z_clearance` | 1 | source pour point가 target opening보다 얼마나 위에 있는지 |
| `source_up_dot_world` | 1 | source cup up-axis의 world z 성분 |
| `directional_tilt_cos` | 1 | 컵이 target 방향으로 기울어졌는지 |
| `mouth_alignment_cos` | 1 | pour heading과 target 방향 정렬 |
| `bead_cross_fraction` | 1 | bead가 target mouth를 지난 비율 |
| `bead_in_target_fraction` | 1 | target cup 내부 bead 비율 |
| `bead_in_source_fraction` | 1 | source cup 내부 bead 비율 |
| `spill_ratio` | 1 | source/target 외부 bead 비율 |
| `g_ready` | 1 | XY 정렬 + Z clearance 준비 게이트 |
| `g_pour` | 1 | `g_ready * g_tilt` |

이 구성은 논문의 observation 카테고리인 `joint positions/velocities`, `fingertip positions`, `cup pose`, `target position`, `proxy sphere centroid`, `previous actions`를 모두 포함한다. 여기에 현재 task에 필요한 mouth geometry와 fill/spill 지표를 추가한 superset이다.

## Reward
보상은 DexPour의 4-stage를 현재 task에 맞게 3-stage로 축약한다.

1. `approach`
   - `r_approach = w_xy * exp(-k_xy * mouth_xy_distance)`
   - `r_clearance = w_clear * g_align_xy * sigmoid(k_clear * (mouth_z_clearance - z_min))`
2. `pre-pour alignment`
   - `r_ready = w_ready * g_ready`
   - `r_prepour = g_align_xy * (w_tilt * tilt_score + w_align * align_score)`
3. `pour / capture`
   - `r_pour = g_pour * (w_cross * bead_cross_fraction + w_capture * bead_in_target_fraction)`
   - `r_success`는 `fill >= 0.30`, `spill <= 0.15`, `g_pour > 0.05`일 때 부여
   - `r_terminal_capture`는 episode 종료 또는 source cup 비움 감지 시 최종 capture 비율에 비례

비용 항목:

- `spill_cost`
- `premature_tilt_cost = (1 - g_ready) * tilt_score`
- `left_action_rate_cost`
- `left_joint_vel_cost`

## Gate / Success
- `g_align_xy = exp(-reward_gate_xy_scale * mouth_xy_distance)`
- `g_clear = sigmoid(reward_gate_clear_scale * (mouth_z_clearance - reward_clearance_min))`
- `g_tilt = sigmoid(reward_gate_tilt_scale * (directional_tilt_cos - reward_tilt_cos_min))`
- `g_ready = g_align_xy * g_clear`
- `g_pour = g_ready * g_tilt`

Success는 per-step 즉시 종료 대신 episode-level success latch로 기록한다.

## Logging
필수 로그:

- `r_approach`
- `r_clearance`
- `r_ready`
- `r_prepour`
- `r_pour`
- `r_success`
- `r_terminal_capture`
- `mouth_xy_dist`
- `mouth_z_clearance`
- `mouth_alignment_cos`
- `directional_tilt_cos`
- `bead_cross_fraction`
- `bead_in_target_fraction`
- `spill_ratio`
- `g_ready`
- `g_pour`
- `success_rate`

## Deferred Work
- right arm action space를 policy에서 완전히 제거하는 일
- privileged critic obs 확장
- full DexPour 4-stage gating 복원
- bead-count curriculum / ADR를 새 reward와 다시 맞추는 일
