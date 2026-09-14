# pour_bi_rh_robot.md 문장별 출처 (관측 #0229: 생성기 프롬프트의 모든 주장은 코드/실측에 닿아야 한다)
- "7-DOF OpenArm ×2, RH56F1 five-finger": tasks/pour_fabric_mimic/robot_profiles.py (num_arm_joints 7, fingers 5)
- "UNDER-ACTUATED, 6 driven per hand, couplings": modules/robot_profiles.py RH56F1_RIGHT 주석 + assets/robot/openarm_rh56f1_bi_rl/*.urdf <mimic> (thumb_3=1.1425·thumb_2, thumb_4=0.7508·thumb_3, *_2=1.1169·*_1)
- "RIGHT=source, LEFT=receiver": pour_fabric_mimic/bimanual.py (_build_pairs: source=sides["r"])
- "frame origin robot base, +z up, +x forward, +y left": 원본 pour_bi 프롬프트와 동일 규약(env-local, side_rig.palm_pos = body_pos_w − env_origins); 부호는 홈 palm (0.31, −0.30) 이 우팔이라는 부팅 로그로 확인
- "action layout 6+6 per side": pour_fabric_env_cfg.resolve_cfg (num_actions_per_side = 6 + 6), side_rig.synergy_targets(syn_slot: thumb_1,thumb_2,index_1,middle_1,ring_1,pinky_1 = hand_finger_channels)
- "contact freeze / opening always allowed / near its cup": side_rig.synergy_targets(hold & delta>0 · close_gate) — 원본과 동일
- "opening ~10 cm, cup ~7 cm": grasp_fj_rh CLAUDE.md 파지 창 105.5 mm(09.07 실측) · cup_big 외경 90 mm × cup_scale 0.8 = 72 mm
