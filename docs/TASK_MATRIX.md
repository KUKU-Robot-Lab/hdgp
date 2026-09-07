# 태스크 부팅 가능성 매트릭스

> `scripts/tools/task_matrix.py` 가 생성한다. **직접 편집하지 말 것.**
> 정적 판정이라 Isaac Sim·GPU 없이 돈다 — 학습이 GPU 를 쓰는 중에도 갱신 가능하다.

`BLOCK` = 부팅 불가(전제조건 결손). `WARN` = 부팅은 되나 검증 미비.

| 판정 | 태스크 | 구성 | gym id | act/obs/state | 사유 |
|---|---|---|---|---:|---|
| **BLOCK** | `agnostic/grasp_sensor` | `gripper_left` | 미등록 | 7 / 48 / 55 | fabric_class: fabric_class/fabric_robot_dir 가 None — env 가 RuntimeError 로 멈춘다 · (palm_box_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_sensor` | `tesollo_right` | 4 | 23 / 114 / 121 | (palm_box_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `bi_left` | 4 | — | (fabric_manifest) · (palm_box_verified) · (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `bi_right` | 4 | — | (fabric_manifest) · (palm_box_verified) · (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `bis_left` | 4 | — | (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `bis_right` | 4 | 19 / 121 / 127 | (probe_verified) · (perception_seam) |
| **BLOCK** | `agnostic/grasp_lift_fabric` | `rh56_left` | 미등록 | — | fabric_class: fabric_class/fabric_robot_dir 가 None — env 가 RuntimeError 로 멈춘다 · (palm_box_verified) · (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `rh56_right` | 4 | — | (palm_box_verified) · (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `sens_left` | 4 | — | (palm_box_verified) · (probe_verified) · (perception_seam) |
| **WARN** | `agnostic/grasp_lift_fabric` | `sens_right` | 4 | — | (fabric_manifest) · (palm_box_verified) · (probe_verified) · (perception_seam) |
| **BLOCK** | `agnostic/pour_fabric` | `bi` | 4 | — | warm_bank: 없음: pour_fab_warm_bi_src.hdf5 / pour_fab_warm_bi_rcv.hdf5 (수집 필요) · (fabric_manifest) · (fabric_manifest) · (perception_seam) |
| **BLOCK** | `agnostic/pour_fabric` | `bis` | 4 | 9 / 210 / 229 | warm_bank: 없음: pour_fab_warm_bis_src.hdf5 / pour_fab_warm_bis_rcv.hdf5 (수집 필요) · (perception_seam) |
| **BLOCK** | `agnostic/pour_fabric` | `sens` | 4 | — | warm_bank: 없음: pour_fab_warm_sens_src.hdf5 / pour_fab_warm_sens_rcv.hdf5 (수집 필요) · (fabric_manifest) · (perception_seam) |
| **WARN** | `gripper/left/grasp_sensor` | `joint` *(비정본)* | 2 | 8 / 36 / — | (perception_seam) |
| **WARN** | `gripper/left/grasp_sensor` | `ik` *(비정본)* | 2 | 7 / 35 / — | (perception_seam) |
| **WARN** | `gripper/left/grasp_sensor` | `fab` | 2 | 7 / 35 / — | (perception_seam) |

## 게이트 전문

### `agnostic/grasp_sensor` / `gripper_left`

gym id: **미등록** — config 가 SKIPPED 로 건너뛴다

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ⛔ `fabric_class` — fabric_class/fabric_robot_dir 가 None — env 가 RuntimeError 로 멈춘다
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: assets/cup/cup_big_rl.usd
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_sensor/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_sensor/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스를 probe 로 실측하지 않았다 — 다른 로봇 값을 물려받았을 수 있다
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_sensor` / `tesollo_right`

gym id: `open-sens_r_grasp_sensor`, `open-sens_r_grasp_sensor-play`, `open-sens_r_grasp_sensor-lstm`, `open-sens_r_grasp_sensor-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_right/openarm_tesollo_sensor_right.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_right/openarm_tesollo_sensor_right_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: assets/cup/cup_big_rl.usd
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_sensor/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_sensor/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스를 probe 로 실측하지 않았다 — 다른 로봇 값을 물려받았을 수 있다
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `bi_left`

gym id: `open-bi_l_grasp_lift_fab`, `open-bi_l_grasp_lift_fab-play`, `open-bi_l_grasp_lift_fab-lstm`, `open-bi_l_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_left/openarm_tesollo_left.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_left/openarm_tesollo_left_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `bi_right`

gym id: `open-bi_r_grasp_lift_fab`, `open-bi_r_grasp_lift_fab-play`, `open-bi_r_grasp_lift_fab-lstm`, `open-bi_r_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `bis_left`

gym id: `open-bis_l_grasp_lift_fab`, `open-bis_l_grasp_lift_fab-play`, `open-bis_l_grasp_lift_fab-lstm`, `open-bis_l_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_s_rl/openarm_tesollo_bi_s_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s_left/openarm_tesollo_bi_s_left.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s_left/openarm_tesollo_bi_s_left_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ✅ `palm_box_verified` — 통과
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `bis_right`

gym id: `open-bis_r_grasp_lift_fab`, `open-bis_r_grasp_lift_fab-play`, `open-bis_r_grasp_lift_fab-lstm`, `open-bis_r_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_s_rl/openarm_tesollo_bi_s_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s/openarm_tesollo_bi_s.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s/openarm_tesollo_bi_s_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ✅ `palm_box_verified` — 통과
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `rh56_left`

gym id: **미등록** — config 가 SKIPPED 로 건너뛴다

- ✅ `robot_usd` — 있음: assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd
- ⛔ `fabric_class` — fabric_class/fabric_robot_dir 가 None — env 가 RuntimeError 로 멈춘다
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `rh56_right`

gym id: `open-rh56_r_grasp_lift_fab`, `open-rh56_r_grasp_lift_fab-play`, `open-rh56_r_grasp_lift_fab-lstm`, `open-rh56_r_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/openarm_rh56f1.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/openarm_rh56f1_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `sens_left`

gym id: `open-sens_l_grasp_lift_fab`, `open-sens_l_grasp_lift_fab-play`, `open-sens_l_grasp_lift_fab-lstm`, `open-sens_l_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/grasp_lift_fabric` / `sens_right`

gym id: `open-sens_r_grasp_lift_fab`, `open-sens_r_grasp_lift_fab-play`, `open-sens_r_grasp_lift_fab-lstm`, `open-sens_r_grasp_lift_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor_manifest.yaml
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: 뱅크 single_cup (1종)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ⚠️ `palm_box_verified` — palm 박스 미실측 — bis_right 값을 물려받으면 62% 가 도달 불가였다(실측)
- ⚠️ `probe_verified` — 물리/IK probe 미통과 — 선언만 된 프로필
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/pour_fabric` / `bi`

gym id: `open-bi_b_pour_fab`, `open-bi_b_pour_fab-play`, `open-bi_b_pour_fab-lstm`, `open-bi_b_pour_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo_manifest.yaml
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_left/openarm_tesollo_left.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_left/openarm_tesollo_left_manifest.yaml
- ✅ `object_usd` — 있음: assets/cup/cup_big_sdf.usd
- ⛔ `warm_bank` — 없음: pour_fab_warm_bi_src.hdf5 / pour_fab_warm_bi_rcv.hdf5 (수집 필요)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/pour_fabric` / `bis`

gym id: `open-bis_b_pour_fab`, `open-bis_b_pour_fab-play`, `open-bis_b_pour_fab-lstm`, `open-bis_b_pour_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_bi_s_rl/openarm_tesollo_bi_s_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s/openarm_tesollo_bi_s.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s/openarm_tesollo_bi_s_manifest.yaml
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s_left/openarm_tesollo_bi_s_left.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_bi_s_left/openarm_tesollo_bi_s_left_manifest.yaml
- ✅ `object_usd` — 있음: assets/cup/cup_big_sdf.usd
- ⛔ `warm_bank` — 없음: pour_fab_warm_bis_src.hdf5 / pour_fab_warm_bis_rcv.hdf5 (수집 필요)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `agnostic/pour_fabric` / `sens`

gym id: `open-sens_b_pour_fab`, `open-sens_b_pour_fab-play`, `open-sens_b_pour_fab-lstm`, `open-sens_b_pour_fab-play-lstm`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf
- ⚠️ `fabric_manifest` — 없음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor_manifest.yaml
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper_manifest.yaml
- ✅ `object_usd` — 있음: assets/cup/cup_big_sdf.usd
- ⛔ `warm_bank` — 없음: pour_fab_warm_sens_src.hdf5 / pour_fab_warm_sens_rcv.hdf5 (수집 필요)
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_cfg.yaml
- ✅ `agent:rl_games_ppo_lstm_cfg.yaml` — 있음: source/openarm/openarm/agnostic/tasks/pour_fabric/config/agents/rl_games_ppo_lstm_cfg.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — `eval_cup_pos_override` 훅 없음 — 물체 pose 를 sim GT 로만 볼 수 있다

### `gripper/left/grasp_sensor` / `joint`

gym id: `open-grip_l_grasp_sensor`, `open-grip_l_grasp_sensor-play`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: assets/cup/shaker_closed_rl.usd
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/gripper/left/grasp_sensor/config/agents/rl_games_ppo_cfg.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — ManagerBased — obs 가 ObsTerm 이라 pose 주입 기구가 별도로 필요하다

### `gripper/left/grasp_sensor` / `ik`

gym id: `open-grip_l_grasp_sensor_ik`, `open-grip_l_grasp_sensor_ik-play`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: assets/cup/shaker_closed_rl.usd
- ✅ `agent:rl_games_ppo_cfg.yaml` — 있음: source/openarm/openarm/gripper/left/grasp_sensor/config/agents/rl_games_ppo_cfg.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — ManagerBased — obs 가 ObsTerm 이라 pose 주입 기구가 별도로 필요하다

### `gripper/left/grasp_sensor` / `fab`

gym id: `open-grip_l_grasp_sensor_fab`, `open-grip_l_grasp_sensor_fab-play`

- ✅ `robot_usd` — 있음: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- ✅ `scene_usd` — 있음: assets/env/usd/env.usd
- ✅ `object_usd` — 있음: assets/cup/shaker_closed_rl.usd
- ✅ `agent:rl_games_ppo_fab_cfg.yaml` — 있음: source/openarm/openarm/gripper/left/grasp_sensor/config/agents/rl_games_ppo_fab_cfg.yaml
- ✅ `fabric_class` — 통과
- ✅ `fabric_urdf` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper.urdf
- ✅ `fabric_manifest` — 있음: source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper_manifest.yaml
- ✅ `fabric_world` — 있음: source/FABRICS/src/fabrics_sim/worlds/open_gripper_left_boxes_no_table.yaml
- ✅ `assets_tracked` — 통과
- ⚠️ `perception_seam` — ManagerBased — obs 가 ObsTerm 이라 pose 주입 기구가 별도로 필요하다

## 범위 밖 (이 매트릭스가 판정하지 않는 것)

- **런타임 거동** — 부팅 후 물리·보상·수렴은 정적으로 알 수 없다.
- **perception 으로 물체 pose 를 대체한 평가** — 기구는 이미 있다:
  `scripts/eval_s2r/providers.py` 의 `make_provider(live|state_frozen|camera_frozen)` 가
  env 의 `eval_cup_pos_override` 에 써 넣고, env 가 관측을 만들 때 그 값을 쓴다.
  ★그 훅은 `tesollo/{right,left}/grasp_v1` 과 `tesollo/right/grasp_sensor` **3개 env 에만**
  있고 위 네 태스크에는 하나도 없다(perception_seam 게이트). 없으면 물체 pose 를 sim GT
  로만 볼 수 있어 "perception 으로 평가를 그대로" 가 성립하지 않는다.
  → 학습이 끝난 뒤 각 env 의 관측 조립부에 같은 훅 2줄을 넣으면 된다(학습 경로라 지금은 동결).
- **perception_plus_plus 저장소** — 이 머신에 없다(vision-3090 별도 repo).
  `/cup_pose` ROS 경로는 실기 전용이고, `sim2real/config/global_camera_extrinsics.yaml`
  은 아직 PLACEHOLDER 라 실기 구동 금지 상태다.
- **체크포인트 계약** — 학습된 정책의 실제 차원은 런의 `params/env.yaml` 이 진실원천이다
  (`scripts/tools/policy_contract.py`). 위 표의 act/obs/state 는 참고값이다.
