# 5g_pour_right_v2 test1 Reward Analysis

분석 대상:
- 로그: `/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_pour_right_v2/test1`
- 체크포인트: `/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_pour_right_v2/test1/nn/5g_pour_right-v2.pth`
- Summary: `/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_pour_right_v2/test1/summaries/events.out.tfevents.1775139285.user`

## 1. 결론
`test1`은 grasp 유지 자체는 매우 안정적이지만, source cup mouth를 target opening까지 충분히 붙이지 못하고, target 방향 tilt reward도 너무 약해서 bead transfer가 전혀 발생하지 않는 상태다. 따라서 다음 실험은 **grasp 유지 계열 reward를 약간 낮추고, transport-progress / tilt / pour_accuracy 계열 reward와 tilt gate 범위를 강화**하는 쪽으로 조정하는 게 맞다.

## 2. 핵심 지표
| Metric | mean_last_100 | 해석 |
|---|---:|---|
| `success_rate/iter` | 0.000000 | 성공 에피소드가 전혀 없음 |
| `bead_cross_fraction/iter` | 0.000000 | 구슬이 source rim을 넘어 target 쪽으로 이동하지 않음 |
| `bead_in_source_rate/iter` | 0.999520 | 거의 모든 구슬이 계속 source cup 안에 남아 있음 |
| `bead_in_target_rate/iter` | 0.000000 | target cup 유입이 0 |
| `mouth_xy_distance/iter` | 0.088855 | source mouth와 target opening의 XY 거리가 약 8.9 cm로 큼 |
| `mouth_z_clearance/iter` | 0.200052 | source mouth가 target opening보다 약 20 cm 높게 유지됨 |
| `reward_transport/iter` | 0.448332 | transport shaping은 들어오지만 최종 정렬 정확도가 부족함 |
| `reward_transport_progress/iter` | 0.000513 | step-wise 접근 진전이 거의 멈춘 상태 |
| `reward_tilt/iter` | 0.003757 | 최종 tilt reward가 매우 작아 기울이기 학습 신호가 약함 |
| `reward_tilt_raw/iter` | 0.031333 | tilt 자체도 목표 각도 근처로 충분히 가지 못함 |
| `reward_aligned_tilt/iter` | 0.003206 | target 근처 + 올바른 방향 tilt가 거의 안 나옴 |
| `directional_tilt_cos/iter` | 0.459641 | target 방향 tilt 정렬이 약함 |
| `pour_gate_xy/iter` | 0.502169 | target 근처 XY gate가 절반 수준이라 tilt reward가 더 약해짐 |
| `pour_gate_z/iter` | 0.970931 | z gate는 거의 항상 열려 있지만, 실제 clearance는 너무 큼 |
| `grasp_maintain/iter` | 0.991389 | cup-palm 상대 자세 유지가 매우 강함 |
| `contact_maintain/iter` | 0.985781 | thumb + finger contact 유지가 거의 완벽함 |
| `finger_curl_min/iter` | 0.900928 | 손가락이 계속 강하게 닫힌 상태 |
| `spill_ratio/iter` | 0.000438 | spill은 거의 없으므로 현재 실패 원인은 spill penalty 과다가 아님 |

## 3. Reward 구조 관점의 병목
- `2*grasp_maintain + 2*contact_maintain + 3*finger_curl_min` 항이 거의 포화 상태라, 정책이 “컵을 단단히 쥐고 현재 자세를 유지”하는 쪽으로 쉽게 수렴한다.
- 반대로 `10*reward_tilt`, `8*pour_accuracy`는 `reward_tilt≈0.0038`, `pour_accuracy=0`이라 실제 총 보상 기여가 거의 없다.
- `mouth_xy_distance≈0.089 m`에서 `tilt_influence=exp(-(d/0.08)^2)`가 약해지므로, target에 충분히 붙기 전에는 tilt reward가 더 희미해진다.
- `pour_gate_z`는 거의 1이지만 `mouth_z_clearance≈0.20 m`이므로, 현재 z gate는 “너무 높은 상태”를 구분하지 못한다. target cup이 상대적으로 낮게 배치되어 있거나, z clearance 상한을 reward/성공 조건이 충분히 제약하지 못하는 상태로 보인다.

## 4. 다음 run에서 우선 적용할 파라미터 조정안
아래는 `test1` 한 번 분석 기준의 1차 제안이며, 자동 루프에서는 2000 epoch run 이후 summary를 보고 다시 증감시키면 된다.

| Parameter | 현재값 | 제안값 | 이유 |
|---|---:|---:|---|
| `env.weight_transport` | 6.0 | 10.0 | target opening XY 정렬을 grasp 유지보다 더 우선시키기 위함 |
| `env.weight_transport_progress` | 5.0 | 15.0 | `reward_transport_progress≈0.0005` 정체를 깨고 매 step 접근을 강하게 유도 |
| `env.reward_transport_scale` | 10.0 | 8.0 | 멀리 있을 때 transport reward 포화를 줄여 8-10 cm 구간 gradient를 완화 |
| `env.weight_tilt` | 10.0 | 18.0 | `reward_tilt≈0.0038`이 너무 작아서 기울이기 항의 총 보상 기여를 키움 |
| `env.reward_tilt_scale` | 5.0 | 3.0 | 목표 tilt에서 멀 때 reward가 너무 작아지는 것을 완화해 더 dense한 tilt gradient 제공 |
| `env.reward_tilt_distance_scale` | 0.08 | 0.12 | `mouth_xy_distance≈0.089 m`에서도 tilt influence가 더 살아있게 함 |
| `env.weight_pour_accuracy` | 8.0 | 16.0 | bead crossing이 한번이라도 발생했을 때 해당 행동을 빠르게 강화 |
| `env.weight_grasp_maintain` | 2.0 | 1.0 | cup-palm 상대 자세를 너무 고정하지 않게 해서 transport/tilt exploration 여지를 확보 |
| `env.weight_contact_maintain` | 2.0 | 1.2 | 접촉 유지는 남기되 grasp lock-in을 약화 |
| `env.weight_finger_curl` | 3.0 | 1.5 | 항상 강하게 닫는 정책으로 고착되는 것을 줄임 |
| `env.reward_grasp_slip_sharpness` | 3.0 | 2.0 | tilt 중 약간의 cup-palm 상대 이동을 허용 |
| `env.contact_maintain_min_others` | 2 | 1 | 최소 접촉 조건을 조금 완화해 transport 중 손가락 자세 재배치를 허용 |
| `env.pour_gate_xy_near` | 0.035 | 0.050 | tilt reward가 유효하게 켜지는 XY 근접 범위를 조금 넓힘 |
| `env.pour_gate_xy_far` | 0.100 | 0.140 | 8-10 cm 거리에서도 gate가 너무 작아지지 않게 함 |
| `env.left_cup_world_z_offset` | -0.08 | -0.03 | target cup을 현재보다 5 cm 올려 `mouth_z_clearance≈0.20 m`를 줄이는 방향 |
| `env.pour_gate_z_high` | 0.05 | 0.12 | 현재 task 기하에서 z gate가 너무 일찍 포화되지 않게 상한 재설정 |
| `env.success_z_clearance_max` | 0.05 | 0.12 | z 성공 판정도 현재 목표 기하에 맞게 완화한 뒤, 이후 성공률이 생기면 다시 조이는 방향으로 사용 |

## 5. 권장 실험 순서
1. **Run2 후보**: transport/tilt 강화 + grasp lock 완화
   - `env.weight_transport=10.0`
   - `env.weight_transport_progress=15.0`
   - `env.reward_transport_scale=8.0`
   - `env.weight_tilt=18.0`
   - `env.reward_tilt_scale=3.0`
   - `env.reward_tilt_distance_scale=0.12`
   - `env.weight_pour_accuracy=16.0`
   - `env.weight_grasp_maintain=1.0`
   - `env.weight_contact_maintain=1.2`
   - `env.weight_finger_curl=1.5`
   - `env.reward_grasp_slip_sharpness=2.0`
   - `env.contact_maintain_min_others=1`
2. **Run3 후보**: Run2 후에도 `mouth_z_clearance > 0.12`가 유지되면 target cup 높이/게이트를 조정
   - `env.left_cup_world_z_offset=-0.03`
   - `env.pour_gate_z_high=0.12`
   - `env.success_z_clearance_max=0.12`
3. **Run4 이후 후보**: `mouth_xy_distance`는 줄었는데 `directional_tilt_cos`가 낮으면 tilt 방향성만 추가 강화
   - `env.weight_tilt=20.0`
   - `env.target_pour_tilt_deg=95.0`

## 6. 자동 루프 설정 제안
- `num_envs=256`, `max_iterations=2000` 고정
- 매 run 종료 후 `nn/5g_pour_right-v2.pth`, `summaries/events.out.tfevents.*`, `params/env.yaml` 분석
- 최대 10회 반복
- 성공 전까지는 `success_rate`, `bead_cross_fraction`, `mouth_xy_distance`, `reward_tilt`, `reward_transport_progress`, `grasp_maintain`, `finger_curl_min`을 핵심 모니터링 지표로 사용

실행 설정 파일:
- `/home/user/rl_ws/hdgp/atuo/config/experiment_5g_pour_right_v2.json`

실행 명령:
```bash
python3 /home/user/rl_ws/hdgp/atuo/orchestrator.py \
  --config /home/user/rl_ws/hdgp/atuo/config/experiment_5g_pour_right_v2.json
```
