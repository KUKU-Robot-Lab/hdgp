# pour_bi_rh_robot.md 문장별 출처 (관측 #0229: 생성기 프롬프트의 모든 주장은 코드/실측에 닿아야 한다)
- "7-DOF OpenArm ×2, RH56F1 five-finger": tasks/pour_fabric_mimic/robot_profiles.py (num_arm_joints 7, fingers 5)
- "UNDER-ACTUATED, 6 driven per hand, couplings": modules/robot_profiles.py RH56F1_RIGHT 주석 + assets/robot/openarm_rh56f1_bi_rl/*.urdf <mimic> (thumb_3=1.1425·thumb_2, thumb_4=0.7508·thumb_3, *_2=1.1169·*_1)
- "RIGHT=source, LEFT=receiver": pour_fabric_mimic/bimanual.py (_build_pairs: source=sides["r"])
- "frame origin robot base, +z up, +x forward, +y left": 원본 pour_bi 프롬프트와 동일 규약(env-local, side_rig.palm_pos = body_pos_w − env_origins); 부호는 홈 palm (0.31, −0.30) 이 우팔이라는 부팅 로그로 확인
- "action layout 6+6 per side": pour_fabric_env_cfg.resolve_cfg (num_actions_per_side = 6 + 6), side_rig.synergy_targets(syn_slot: thumb_1,thumb_2,index_1,middle_1,ring_1,pinky_1 = hand_finger_channels)
- "contact freeze / opening always allowed / near its cup": side_rig.synergy_targets(hold & delta>0 · close_gate) — 원본과 동일
- "slim shaker ~6 cm dia, 11 cm tall": pour_fabric_mimic cfg POUR_CUP_USD=shaker_closed_rl(외경 88 mm·높이 175 mm @1.0) × cup_scale 0.65
- "fingertip grasp expected, wrap not required": 사용자 결정 09.14(인벨롭 파지 포기) · 09.14 프로브 7종에서 열린 손끝이 포켓을 막아 감쌈 진입 불가
- "light cup, brushing knocks it over": 09.14 프로브 실측(손가락 3~14 N 접촉에 30~88° 전도)
