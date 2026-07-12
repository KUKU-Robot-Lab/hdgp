Task 1.0: complete (FK verifier, self-check both PASS 0mm/0deg)
Task 1.1: complete (URDF regen, palm_sensor+axis points right, FK gate PASS 0mm/0deg)
Task 1.2: complete (palm attractor→palm_sensor, static frame match OK, runtime=server)
Task 1.3: complete (collision frames palm_link→palm_sensor, 17=17 OK)
Task 1.4: complete (grasp_v1 offset drop + ex+90 recal, static 9 pass, runtime=server)
Task 1.5: complete (pour_v1 offset drop + ex+90, 68 pass /1 pre-existing)
Phase 1 CODE COMPLETE (Tasks 1.0-1.5). Runtime gate 1.6 = server (IK roundtrip + play).
Task 1.6: complete via NUMERIC gate (frame convention 3 pass). Phase 1 DONE (warp runtime deferred).
Task 2.1: complete (bimanual _rl URDF, cspace 26, FK both PASS). Right arm re-based Tesollo→_rl.
=== SESSION STOP 2026-07-03: 양팔 _rl URDF 완료(FK both 0mm/0). 남음: fabric코드26/params/env (warp서버). Notion 저장됨. ===
Task 2.2: complete (fabric cspace 26, default_config 26 [r_arm7,r_hand6,l_arm7,l_hand6], TIP r_hl_*_tip, add_hand_fabric (6,26) 오른손 슬라이스). commit 202f382
Task 2.3: complete (params 충돌 프레임 _rl 실존 19개, joint_limits 26). commit 202f382
=== SERVER VERIFY 2026-07-03 (직접 SSH oem@...240, GPU0): verify_fabric_load_ik.py 기본값 batch8 [PASS] ===
  Gate1 num_joints==26 OK / Gate2 오른손 IK 잔차 22.13mm(<30mm, plateau=정상상태 droop) / Gate3 왼쪽 drift 0.0000rad.
  → 계획 검증 2단계(IK 왕복) 서버 통과. fabric+params 인프라 확정.
남음: Task 2.4 env(grasp_v1/pour_v1) 26 DOF 구동 — 현재 env fabric_q[:,7:]=hand(6폭) 슬라이싱이 26 DOF와 비호환 → 학습 실행엔 필수. 그 뒤 play 렌더(검증 3단계).
Task 2.4 (grasp_v1): complete + SERVER VERIFIED. commit e74033d.
  env fabric_q 13->26 국소화(fabric_q init 26/default_config·reset_cspace [7:13]/ _run_reset_fabric 26패딩/hand sync·reset·warmstart [:, :NUM_ROBOT_DOF]). 왼팔 종전대로 left_arm_zero_pos.
  정적 73 pass. 서버(open-rh56f1_r_grasp_v1, 512 envs, 3 iter): env 로드+reset+step 정상, epoch3/3 rew 58.03 체크포인트 저장, dim crash 없음. (스모크 test2 정리)
남음: Task 2.4(pour_v1) — grasp 검증 통과 후 동일 패턴 이식. 그 뒤 play 렌더(검증 3단계) 선택.
Task 2.4 (pour_v1): complete + SERVER VERIFIED. commits 0386b4b(26DOF), 5869713(palm_sensor fix).
  26 DOF 국소화(grasp와 동일 패턴). + 서버 첫 실행에 드러난 pre-existing 버그 수정:
  c86d24b가 _palm_ee_offset_local 정의만 삭제/사용처3곳 잔존 → AttributeError. pour palm 프레임이
  palm_1(07.02)/palm_sensor(c86d24b) 반쯤 이식 불일치. 사용자 '완전 정합' 선택 →
  HAND_BODY_NAMES_USD[0] palm_1→palm_sensor, _palm_ee_offset_local=0(3곳 identity), 테스트 갱신.
  서버(open-rh56f1_r_pour_v1, 512 envs, 3 iter): env+reset+step 정상, epoch3/3 rew 2184.55 체크포인트. (test3 정리)
  주의: tilt 명령축(07.02 회전 우려)은 world-델타 합성이라 프레임 무관 판단 — 장기학습 tilt 지표로 최종 확인 권장.
=== Task 2.4 grasp+pour 서버 검증 완료. 남음: (선택) play 렌더 검증 3단계 / 왼손 능동제어(이후 phase). ===
=== 2026-07-04 서버 학습 시작 ===
- warm hdf5 palm_pose 마이그레이션: grasp_warm_rh56f1.hdf5(서버 387개) → palm_sensor FK 재계산 적용
  (migrate_warm_palm_to_sensor.py, 커밋 1418d0a). 백업 .pre_palmsensor_bak. ex+33.5°, pos Δ~7.6cm.
- grasp_v1 2048 학습 시작(GPU0, RUN_LABEL ps26_t1→폴더 testN 자동). epoch 377+ 정상, fps 16k.
- pour_v1 2048 학습(GPU1, migrated hdf5).
- 정정: pour "stall" 오진(2048 첫 epoch 느림을 512 스모크와 불공정 비교, 11분에 조기 kill).
  512-env A/B로 migrated hdf5 4 epoch 완주(rew 2214) 확인 → 마이그레이션 정상.
  pour 학습 palm 제어는 이미 quat+관측 기반(line 1211 _compose_world_delta_quat_xyzw, set_features "quaternion")
  → euler gimbal/orientation-bounds(ex90°)는 warmstart_collect_mode 전용, 학습 미gate. palm_sensor 정합 사실상 완료.
- 미해결(추후): pour palm_pose_mins/maxs orientation 중심 여전히 ex90°(collect용). palm 0.12m 클램프 경고(진단용, 비치명).
- 검증 관전 포인트: pour directional_tilt_cos, bead_in_target로 틸팅/action-control 해소 여부(사용자 가설).
=== 2026-07-04 grasp spawn-penetration 진단·수정 (세션 종료) ===
근본원인 확정: reset 시 r_hl_index_1 이 컵을 ~2.8cm 관통(spawn_probe.py) → 첫스텝 force_ratio 100~327 이젝션 → 파지 불가.
  원인=pregrasp standoff 0.6배 과축소. 수정: pregrasp_offset (-0.027,-0.033)→(-0.06,-0.07) Tesollo 스케일 복원. commit pushed.
검증: spawn_probe penetrating [False], force_ratio 0. 재학습 test4(2048): reward 280→1315(4.5배), 컵 안정(tilt 20→3.5°, xy 11→4.9cm), 이젝션 소멸.
probe_sweep: standoff <9cm 는 관통 재발(6cm=2.75cm, 7cm=2.13cm) → 9.2cm 밑 불가.
그러나 test4 epoch 1078 정체: height_delta=0/palm_near=0/envelope=0. 딜레마 = 9.2cm standoff에서 palm_to_cup 8.5cm 고정 → palm-first 게이트(5cm) 미발화 → 엄지 release 안 됨 → envelope/lift 불가.
다음후보(미실행): (A) HAND_APPROACH_POSE 손가락 덜 펴 clearance 확보→standoff 축소 허용(권장), (B) palm-first 게이트 임계값 9.2cm 재보정(reward-audit). 관련 [[palm-sensor-breaks-distance-thresholds]].
도구: spawn_probe.py(관통 실측), probe_sweep.py(standoff sweep) — /tmp 서버, tfx.py/tflatest.py(TFEvents 추출).
