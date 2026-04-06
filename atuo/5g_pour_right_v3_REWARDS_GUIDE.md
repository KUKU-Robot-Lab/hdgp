# 5g_pour_right_v3 Reward Guide (<=12k chars)

목적: 오른손이 source cup을 잡고 target cup 입구 5–8 cm 이내로 정렬한 뒤, 기울여 bead를 최소 1개 이상 넘긴다. 성공 판단은 학습 로그 상 성공 프록시(`success_rate` 또는 `bead_in_target/frame`, `bead_cross_fraction`)가 0.9 이상일 때.

## 핵심 지표 해석
- success proxies: `success_rate`, `success`, `bead_in_target/frame`, `bead_cross_fraction` 중 존재하는 값. 0.9 미만이면 실패.
- 위치: `mouth_xy_distance`, `pour_gate_xy`, `reward_transport`, `reward_transport_progress`. 0.05 m 이하가 목표. progress≈0이면 정렬 정체.
- 높이: `mouth_z_clearance`, `pour_gate_z`. 0.12 m 초과면 너무 높음.
- 기울기: `reward_tilt`, `reward_aligned_tilt`, `directional_tilt_cos`, `tilt_influence`. target 근처에서 tilt reward가 0.05 이상, directional_cos >0.7이면 양호.
- 잡기 유지: `grasp_maintain`, `contact_maintain`, `finger_curl_min`, `force_balance`. 모두 0.9 이상인데 성공 0이면 grasp shaping이 과도.
- 페널티: `spill_ratio`, `cost_premature_tilt`. spill 0인데 성공 0이면 주로 정렬/tilt 부족. premature_tilt가 높으면 너무 일찍 기울임.

## 조정 가능한 파라미터(override_policy)
- 정렬/게이트: `env.stage_pour_xy_threshold`, `env.stage_approach_xy_threshold`, `env.pour_gate_xy_near/far`, `env.tilt_action_gate_xy_near/far`, `env.reward_gate_xy_scale`, `env.reward_gate_clear_scale`, `env.reward_gate_tilt_scale`.
- 기울기: `env.weight_tilt`, `env.reward_tilt_scale`, `env.reward_tilt_distance_scale`, `env.target_pour_tilt_deg`, `env.success_tilt_cos_tolerance`, `env.success_directional_tilt_cos`, `env.weight_premature_tilt`.
- 이동: `env.weight_transport`, `env.weight_transport_progress`, `env.reward_transport_scale`.
- 성공/포획: `env.weight_pour_accuracy`, `env.success_mouth_xy_threshold`, `env.success_z_clearance_min/max`, `env.success_hold_steps`.
- 그립: `env.weight_grasp_maintain`, `env.weight_contact_maintain`, `env.weight_finger_curl`, `env.reward_grasp_slip_sharpness`, `env.contact_maintain_min_others`, `env.weight_force_balance`, `env.force_balance_sharpness`.
- 기타: `env.weight_spill`, `env.weight_action_rate`, `env.left_cup_world_z_offset`, `env.drop_force_hold_steps`.

## 대표 실패 패턴 → 추천 수정
1) 성공값 <0.9, bead 이동 거의 없음
   - 좁게 붙이기: `stage_pour_xy_threshold=0.06~0.08`, `tilt_action_gate_xy_near=0.06`, `tilt_action_gate_xy_far=0.12`.
   - 기울임 늦추기/강화: `weight_premature_tilt` ↑ (4~6), `weight_tilt` ↑ (16~20), `reward_tilt_distance_scale` 0.10~0.14.
   - 이동 gradient: `weight_transport_progress` ↑ (12~15), `reward_transport_scale` ↓(6~8)로 멀리서도 포화 덜 하게.

2) XY 정렬 정체 (`mouth_xy_distance>0.05`, progress≈0)
   - `weight_transport`/`weight_transport_progress` 증가, `pour_gate_xy_far` 0.14~0.18, `reward_gate_xy_scale` ↑.

3) 너무 높게 유지 (`mouth_z_clearance>0.12`)
   - `left_cup_world_z_offset`를 0~-0.02로 올리거나, `success_z_clearance_max` 0.12로 클램프.

4) 기울기 방향 오류 (`directional_tilt_cos<0.7`, tilt reward≈0)
   - `weight_tilt` ↑, `reward_tilt_scale` ↓(2~3), 필요 시 `target_pour_tilt_deg` 95~110로 미세 조정.

5) 그립 고정 과도 (`grasp/contact/finger≈1`인데 성공 0)
   - `weight_grasp_maintain/contact_maintain/finger_curl` 소폭 ↓, `reward_grasp_slip_sharpness` 완화(2~3), `contact_maintain_min_others` 1~2.

6) 너무 일찍 기울임 (`cost_premature_tilt` 높음)
   - `weight_premature_tilt` ↑, `tilt_action_gate_xy_near/far`를 더 좁게, `stage_pour_xy_threshold` ↓.

## 루프 운영 가이드 (v3)
- 성공 기준: success proxy ≥0.9.
- run은 최대 10회, 중도 `training_collapse`/`entropy_collapse` 시 정지.
- 각 run 후 tensorboard 요약을 기반으로 위 실패 패턴을 매칭해 override 적용.
- eval은 스킵해도 됨(rl_games), 성공 프록시는 train 로그로 판단.
- 보고: `atuo/runs/<run_id>/report.md`와 누적 `progress_*_testN.md` 확인.

## 체크리스트
- mouth_xy_distance < 0.05 ?
- directional_tilt_cos > 0.7 ?
- bead_cross_fraction 상승 추세 있는가?
- premature tilt cost 감소하는가?
- spill_ratio 안정적인가?

