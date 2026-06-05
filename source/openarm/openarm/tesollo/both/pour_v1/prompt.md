# `pour_v1` Step-by-Step ECC Prompt

## Step 1. Env Audit
목표:
`both/pour_v1`에서 실제로 reward에 쓸 수 있는 기하/비드 지표가 이미 계산되는지 확인한다.

해야 할 일:
- `pour_env.py`에서 `_compute_intermediate_values()`를 읽는다.
- `source_pour_point_w`, `target_opening_w`, `mouth_delta`, `mouth_xy_distance`, `mouth_z_clearance`, `directional_tilt_cos`, `mouth_alignment_cos`, `bead_cross_fraction`, `bead_in_target_fraction`, `spill_ratio`가 모두 채워지는지 확인한다.
- actor obs와 `_get_rewards()`의 현재 상태를 요약한다.

완료 기준:
- reward/obs에 재사용 가능한 텐서 목록이 정리되어 있다.

## Step 2. Observation Patch
목표:
left-arm assist policy에 필요한 최소 actor obs를 구현한다.

해야 할 일:
- `left_arm_pos_offset`, `left_arm_joint_vel`, mouth geometry, fill/spill stats, gates, `prev_left_arm_action`를 actor obs에 넣는다.
- `cfg.num_observations`와 실제 obs dim이 맞는지 assert를 넣는다.
- critic obs는 우선 actor obs와 동일하게 유지한다.

완료 기준:
- obs dim mismatch 없이 env가 초기화된다.

## Step 3. Reward Patch
목표:
DexPour-inspired 3-stage reward를 left-arm assist task에 맞게 구현한다.

해야 할 일:
- approach, pre-pour, pour/capture 보상을 구현한다.
- spill, premature tilt, left action rate, left joint velocity 비용을 구현한다.
- `success_flag`를 episode-level latch로 기록한다.
- tensorboard용 `extras`를 추가한다.

완료 기준:
- `_get_rewards()`가 0 대신 실제 reward를 반환한다.
- right-hand grasp 유지 항목은 이번 단계에서 건드리지 않는다.

## Step 4. Verification
목표:
적어도 정적 검증과 최소 smoke 검증을 통과시킨다.

해야 할 일:
- `python3 -m py_compile`로 `pour_env.py`, `pour_env_cfg.py`를 검사한다.
- 가능하면 headless smoke run으로 env 초기화와 obs/reward shape를 확인한다.
- reviewer agent로 reward hacking 경로를 한번 더 점검한다.

완료 기준:
- 문법 오류 없음
- obs dim / reward path / logging key가 일관됨
