# 5g_pour_right_v2 Reward Guide

이 가이드는 `5g_pour_right-v2` 자동 개선 루프에서 reward/summary를 해석하고 다음 run override를 고를 때 쓰는 기준 문서다.

## 1. Task 목표
- 오른손이 source cup grasp를 유지한 채 target cup opening 근처로 source pour point를 이동한다.
- target 방향으로 충분히 기울여 bead를 source cup에서 target cup으로 넘긴다.
- 성공 기준은 `_bead_cross_count >= success_bead_cross_count`이며 기본값은 1개 이상이다.

## 2. Reward 항 구조
- `weight_grasp_maintain * grasp_maintain`
  - cup center가 palm local frame의 초기 상대 위치에서 얼마나 안 미끄러졌는지 본다.
  - `reward_grasp_slip_sharpness`가 클수록 작은 slip에도 reward가 급격히 줄어든다.
- `weight_contact_maintain * contact_maintain`
  - thumb tip 접촉과 나머지 손가락 `contact_maintain_min_others`개 이상 접촉을 유지하면 보너스를 준다.
- `weight_force_balance * force_balance`
  - `exp(-force_balance_sharpness * |F_thumb - F_others_avg|)` 형태로 접촉력 균형을 유도한다.
- `weight_finger_curl * finger_curl_min`
  - 5개 finger action의 minimum을 기준으로 닫힘을 보상한다. 한 손가락이라도 열리면 값이 낮아진다.
- `weight_transport * reward_transport`
  - `1 - tanh(reward_transport_scale * mouth_xy_distance)`로 source pour point와 target opening의 XY 근접을 보상한다.
- `weight_transport_progress * reward_transport_progress`
  - 직전 step 대비 `mouth_xy_distance`가 줄어든 양만큼 보상한다.
- `weight_tilt * reward_tilt`
  - `tilt_influence * exp(-reward_tilt_scale * tilt_error) * clamp(directional_tilt_cos, 0, 1)` 구조다.
  - `tilt_influence = exp(-(mouth_xy_distance / reward_tilt_distance_scale)^2)`라서 target에서 멀면 tilt reward가 약해진다.
- `weight_pour_accuracy * pour_accuracy`
  - `bead_cross_fraction` 자체를 보상한다.
- `- weight_spill * spill_ratio`
  - source/target cup 밖으로 나간 bead 비율을 벌점으로 준다.
- `- weight_action_rate * ||a_t - a_{t-1}||^2`
  - 액션 변화율을 벌점으로 준다.

## 3. Summary 태그 해석
- `success_rate/iter`, `bead_cross_fraction/iter`, `bead_cross_count/iter`
  - 최종 목적 달성 여부를 직접 본다. 이 값이 0이면 다른 shaping reward가 아무리 높아도 실패다.
- `mouth_xy_distance/iter`, `pour_gate_xy/iter`, `reward_transport/iter`, `reward_transport_progress/iter`
  - source mouth가 target opening의 XY 근처로 실제 이동하는지 본다.
  - `mouth_xy_distance`가 5 cm 이상이고 `reward_transport_progress`가 0에 가깝다면 transport가 정체된 것이다.
- `mouth_z_clearance/iter`, `pour_gate_z/iter`
  - source pour point가 target opening보다 얼마나 위에 있는지 본다.
  - 현재 z gate는 상한 초과를 직접 벌주지 않으므로 `mouth_z_clearance`가 지나치게 커도 `pour_gate_z≈1`이 될 수 있다.
- `reward_tilt/iter`, `reward_tilt_raw/iter`, `reward_aligned_tilt/iter`, `directional_tilt_cos/iter`, `pct_correct_tilt_dir/iter`, `source_up_dot/iter`, `tilt_influence/iter`
  - target 쪽으로 충분히 기울이는지, 그리고 그 tilt가 target 근처에서 유효하게 들어가는지 본다.
- `grasp_maintain/iter`, `contact_maintain/iter`, `finger_curl_min/iter`, `force_balance/iter`, `slip_dist/iter`
  - grasp 유지가 충분한지 본다.
  - 이 항들이 거의 1인데 bead transfer가 0이면 grasp shaping이 과도하게 우세해 transport/tilt exploration을 막는 상태일 수 있다.
- `spill_ratio/iter`, `cost_spill/iter`
  - spill이 늘면 pour를 억제해야 하지만, spill이 거의 0인데 성공도 0이면 일반적으로 spill penalty가 아니라 transport/tilt 신호 부족이 병목이다.

## 4. 대표 실패 패턴과 조정 방향
- `pour_success_zero`, `pour_bead_transfer_fail`
  - 증상: `success_rate≈0`, `bead_cross_fraction≈0`, `bead_in_source_rate≈1`.
  - 조정: `weight_pour_accuracy`, `weight_tilt`, `weight_transport_progress`를 올리고 `reward_tilt_distance_scale`을 넓혀 target 근처 이전에도 tilt gradient가 살아있게 한다.
- `pour_transport_xy_far`, `pour_transport_progress_stalled`
  - 증상: `mouth_xy_distance > 0.05`, `pour_gate_xy < 0.7`, `reward_transport_progress≈0`.
  - 조정: `weight_transport`, `weight_transport_progress`를 올리고 `reward_transport_scale`을 낮춰 멀리서 reward 포화가 덜 생기게 하며 `pour_gate_xy_far`를 넓힌다.
- `pour_height_too_high`
  - 증상: `mouth_z_clearance`가 12 cm 이상인데 `pour_gate_z≈1`.
  - 조정: `left_cup_world_z_offset`를 덜 음수로 만들어 target cup을 올리거나, `pour_gate_z_high`, `success_z_clearance_max`를 현재 기하에 맞게 다시 잡는다.
- `pour_tilt_reward_too_weak`, `pour_wrong_tilt_direction`
  - 증상: `reward_tilt`, `reward_aligned_tilt`이 매우 작고 `directional_tilt_cos`가 낮다.
  - 조정: `weight_tilt`를 키우고 `reward_tilt_scale`을 낮춰 reward를 덜 sparse하게 만든 뒤, 필요하면 `target_pour_tilt_deg`를 약간 올린다.
- `pour_grasp_overconstrained`
  - 증상: `grasp_maintain≈1`, `contact_maintain≈1`, `finger_curl_min≈1`인데 `success_rate≈0`.
  - 조정: `weight_grasp_maintain`, `weight_contact_maintain`, `weight_finger_curl`을 낮추고 `reward_grasp_slip_sharpness`를 완화해 컵이 약간 움직이며 transport/tilt를 탐색할 여지를 만든다.

## 5. 추천 자동 루프 기본 정책
- 각 run은 `num_envs=256`, `max_iterations=2000`으로 고정한다.
- 한 run 종료 후 `5g_pour_right-v2.pth`, `summaries/events...`, `params/env.yaml`을 분석해 다음 override를 선택한다.
- 최대 10회까지만 반복하고, `success_rate`, `bead_cross_fraction`, `mouth_xy_distance`, `reward_tilt`, 적용 override를 누적 보고서로 정리한다.
