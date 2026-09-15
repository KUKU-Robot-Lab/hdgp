# pour_fabric_mimic — 태스크 규칙 (09.14 신설: 저차원·언더액추 손 전용 text2reward 트랙)

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 원본 트랙 [`../pour_fabric/CLAUDE.md`](../pour_fabric/CLAUDE.md)
> (보상 생성 루프·검증 게이트·s2r 관측/DR 규약은 원본과 같다 — 여기엔 **다른 점만** 적는다)

## 정체성

`pour_fabric`(DG-5F 20관절)의 **사본**을 RH56F1(구동 6 + PhysX mimic 종속 6)용으로 고친 트랙.
사용자 결정 09.14: "원본 냅두고, 저차원 핸드 전용을 새로 만든다". 원본 파일은 이 트랙 때문에 바뀌지 않는다
(`tests/test_mimic_contract.py::test_original_pour_fabric_untouched_by_this_track` 가 잠근다).

- gym id `open-rh_b_pour_fab_mimic` (+`-play`, `-lstm`) · 로그 `log/rl_games/open-rh/both/pour-fab-mimic/`
- 액션 **24** = (palm 6 + 손 6) × 2 · actor obs **182** · critic **224** (09.14 손끝 촉각 5칸×2 추가 전 172/214)
- 보상: `reward_gen/pour_bi_rh/iter_NN/compute_reward.py` (`RewardContext` 는 원본과 **동일 클래스** — F=5 손가락, 손 폐쇄도·접촉력 필드 의미 그대로)
- 학습 호스트: **vision-3090**(RTX 3090 24 GB) — 루프 도구 `scripts/reward_gen/t2r_rh_round.py` · `LOOP_PROMPT_rh.md` · `run_pour_t2r_rh.sh`

## 원본과 다른 곳 (전부 실측 근거)

| 항목 | pour_fabric (DG-5F) | pour_fabric_mimic (RH56F1) | 근거 |
|---|---|---|---|
| 프로필 | `modules/robot_profiles` TESOLLO_*_SHORT | **트랙 로컬** `robot_profiles.py` RH56F1_{RIGHT,LEFT}_FAB | 모듈 RH56F1_RIGHT 는 fabric_class=None(Track B 전용) · 모듈 PROFILES 에 넣으면 grasp config 가 gym id 를 찍는다 |
| Fabrics | 팔당 13 DOF 클래스 2종(우/좌 URDF) | **양팔 26 DOF 클래스 1종** ×2 인스턴스(side="right"/"left"), 슬라이스 `bimanual.fabric_slots` · 매 스텝 `sync_other` | `openarm_rh56f1_pose_fabric.py` 구조 |
| 손 액션 | 채널 3 × 5지 = 15 | **per_finger 6**(`hand_finger_channels` 1:1) | 채널×손가락 격자는 슬롯 4개가 죽는다(grasp_fj_rh 09.07) |
| 4지 결합 | 채널별 평균 | 같은 접미사끼리 평균(결합 행렬) | 동일 의미 |
| 접촉 동결 | 접미사 "3"(중간) / 원위 분리 | **finger 스코프만**(`hand_freeze_suffixes`) | 손가락당 구동관절 1개 — 감쌈은 하드웨어 결합(_2 = _1 × 1.1169) |
| 엄지 대향 노브 | `oppose_grip_delta_rad −0.6` | **0.0** (cfg 검증이 강제) | RH ch1 = 엄지 굴곡(0~0.475) → −0.6 이면 잘려 엄지가 안 접힌다(09.02) |
| 자기충돌 | ON | **OFF** | ON 이면 시작 자세 `r_hl_index_tip` 에 226 N 상시 자기접촉 + 폐쇄 0.7 에서 mimic 오차 0.9 rad(09.14) |
| 종속관절 | 없음 | 액추에이터 **0/0** + 부팅 한계 ±1.5 rad 확장 + URDF↔USD gearing 대조 | grasp_fj_rh 09.07 (rh_b1 폭주 → rh_b2 해결) |
| palm 박스 | 검증됨 | x ≥ 0.10 · z ≤ 0.70 (미검증, probe 후 승격) | 원본 델타 박스가 Track B 박스에 잘렸다(부팅 경고 x,z) |
| palm 델타 z 하한 | −0.12 | **−0.18** | 홈 palm 이 컵 원점보다 13.7 cm 위(09.14 부팅 실측) |
| close_gate 반경 | 0.22 | 0.30 | 시작 palm↔컵 ≈ 0.21 m |

## 좌손 미러 (09.14, rl-mirror-port Step 1~2)
순수 FK(`assets/robot/openarm_rh56f1_bi_rl/*.urdf`)로 확정: **팔 (−1,−1,−1,+1,−1,−1,−1) · 손 6관절 전부 +1**.
같은 q 에서 4지 손끝 |l − mirror(r)| = 0.00 mm. 엄지 5.7 mm·palm_sensor 2.7 mm 는 벤더 자산 비대칭(부호 오류면 수 cm).
부팅 fabric FK 게이트: 양팔 0.14 mm / 0.9° 통과. ★좌 yaw 가 ±π 경계에 있어 각도 차는 wrap 해서 잰다(side_rig).

## ★09.14 사용자 결정 — 대본 파지 게이트 없이 학습 시작 · shaker 계열 · 손끝 파지 허용
프로브 7종(직진·옆·손끝쪽·오목면·Track B 대각·2단 xy→z, 컵 0.8/0.6/0.5)이 전부 접촉 즉시 전도(30~88°)였다.
기하 덤프: '열린'(q=0) 손가락이 45° 안으로 굽어 손끝이 포켓 중심을 막고 엄지는 검지 위(z)에 있어 감쌈 진입이 불가하다.
→ 물체를 `shaker_closed_rl` × 0.65(지름 57 mm, Track B shaker_one 과 동일)로 바꾸고, **인벨롭 파지를 포기**(손끝 파지 OK),
대본 파지 없이 학습을 시작한다(Track B 선례: 대본 전도 611회여도 RL 은 학습). 비드는 바닥 기준 재적층.

## 09.14 폭주 리셋 (사용자 결정)
한계 여유 3.0 으로도 리프트 시작 순간 mimic 오차 154 rad·종속 속도 519 rad/s 가 재현됐다 → 여유는 처방이 아니다. `mimic_runaway_dep_qd`(100 rad/s)를 넘는 env 는 runaway 로 종료·리셋한다(`done/mimic_runaway` 지표).

## 검증 게이트 (학습 전) — 원본 3종 + 언더액추(★2 는 이 트랙에서 면제)
1. `probe_pour_fabric_mimic_boot.py` 무작위 300스텝: NaN 0 · runaway 0 · in_source ≥ 0.9 (09.14 PASS).
2. `probe_pour_fabric_mimic_hand.py --approach cup`: 케이지→컵 접근·폐쇄·리프트 대본으로 파지 성립(`grasped`)·mimic 오차 ≤ 0.1 rad.
3. `probe_pour_fabric_mimic_boot.py --script pour --tilt_slot 5`: 붓기 축 도달.
4. `t2r.py ingest` PASS(`--num-actions 24`).
★`ctrl/mimic_err_max` 건전 기준선(09.07): 자유 폐쇄 ≤0.05 · 파지 ≤0.04 · 10 rad 이상 반복이면 종속 한계 부족.

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/pour_fabric_mimic/tests -q
~/rl_ws/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_mimic_boot.py --num_envs 8 --steps 300
./run_pour_t2r_rh.sh <label> reward_gen/pour_bi_rh/iter_00 --num_envs 1024     # vision-3090 에서
```

### 09.14 라운드 1 종료 후 env 변경 (사용자 결정 "2,3 추가")
- **느린 mimic 폭주도 종료**: `mimic_runaway_err_rad` 3.0 — 속도 100 rad/s 아래로 오차만 벌어진 폭주 2회(epoch 321-323, 348-351)가 속도 기준을 빠져나갔다. 지표 `done/mimic_err_runaway`.
- **손끝 촉각 actor obs**: 손당 5칸 = 손끝 링크 전체 접촉력(net, 컵 필터 아님) · 노이즈 0.1 N · 클립 10 N. 실기 출처 RH56F1 `TouchData1.finger_forces[5]`(0.01 N 단위). sim 손가락 순서(thumb, index, middle, ring, pinky) ≠ 벤더 순서일 수 있다 — 배포 시 재배열. obs 변경이라 iter_01 부터 새로 학습.
- 계기: 라운드 1 영상 — 오른손 엄지가 컵 입구 테두리에 걸려 들지 못함, 왼손은 컵에 닿지 않음.

### 09.15 자산 collider 정리 · 접촉 링크 `(_1, _sensor)` (사용자 결정 "sensor 링크는 실제 힘측정 부위 — 살린다")
- **손끝 마디 사본 제거:** 벤더 STL 의 `{f}_2`·`{f}_force_sensor`·`{f}_tip`(엄지 thumb_4·sensor·tip)은 같은 입체였다(면·부피·관성 동일). collider 는 `_sensor` 에만 둔다. `_2`·`_tip` 은 질량·visual·프레임을 유지하고 `_tip` 은 손끝 위치로만 쓴다(`urdf/tools/generate_rl_urdf.COLLISION_DROP_LINKS`).
- **손바닥:** palm_1·palm_2 hull 이 palm_sensor 패드 면을 1~5 mm 덮어 두 껍질만 convexDecomposition 으로 굽는다(`DECOMPOSITION_LINKS`). 20 mm 큐브를 패드에 대면 `palm_sensor` 에 잡힌다. 40 mm 평판은 CAD 껍질 테두리(+2 mm)에 먼저 닿는데, 이것은 정상이다.
- **몸통:** body_link 도 decomposition 으로 굽는다. 팔 영자세에서 GPU hull 이 수십 mm 부풀어 중지 2.25 N·엄지 외전 25.6 mm 가짜 접촉이 났다. 사용자 원칙은 **가짜 형상 충돌을 필터로 가리지 않고 형상을 고치는 것**이다. 필터는 설계상 박힘인 thumb_2↔palm_2(allowlist `force_filter`)에만 쓴다.
- **env 변경:** 트랙 프로필 `finger_sensor_bodies`=(`_1`, `_sensor`)(엄지 thumb_3, thumb_sensor). side_rig 2원소는 원위=팁=센서로 읽는다. 촉각 obs·접촉 동결·손가락 접촉력이 모두 `_sensor` 를 읽는다. **obs 의미가 바뀌었으니 이전 체크포인트는 이어 쓰지 않는다.**
- **모듈 프로필 수정(cb1fd8fe):** `RH56F1_RIGHT` 가 (_1, _sensor) 로 바뀌었다. Track B 도 이 값을 쓰고 grasp_fj 2원소 규약도 고쳤다. 이 트랙은 프로필을 상속한다.
- **손 게인 30/0.3(사용자 결정 "30/0.3 으로 통일"):** 자산 USD 와 같은 값이다. 벤더 PD 는 없고 사양은 4지 >10 N·엄지 >15 N·전 범위 1 s 이다. 자기충돌 OFF 스윕에서 옛 5/2 는 추종이 0.51 rad 늦고 엄지가 중력에 7 mrad 처졌다. 30/0.3 은 0.01 rad·1.4 mrad 이다.
- **t2r 재시작(사용자 결정):** 새 자산·obs·게인으로 iter_03 보상을 처음부터 학습한다. 라벨은 `t2r_rh_i03_a2` 이고, 옛 기록은 `iter_03/*_t2r_rh_i03.json` 에 있다. 자기충돌은 OFF 그대로 둔다.
- **검증:** `probe_rh56f1_finger_sweep.py --contacts` 는 자기충돌 ON 에서 손 링크끼리 접촉이 0 이어야 한다. `--obstacle_mm`·`--contact_partners /World/obstacle` 로 외부 접촉이 `_sensor`/`palm_sensor` 에 잡히는지 본다.

### 09.15 엄지 입구 걸림 접근 로그 지표 (사용자 "다음부터 지표로깅으로 확인 가능하게")
- 이 지표는 로그 전용이다. obs 와 RewardContext 에는 넣지 않았다. 보상 코드와 무관하게 env 가 직접 계산하므로 iter 가 달라도 서로 비교할 수 있다.
  - `task/{src,rcv}_near_rate`: palm↔컵 원점 거리가 10 cm 안인 env 비율
  - `task/{side}_thumb_over_rim_near`: 위 env 중 엄지 끝이 입구 높이 −2 cm 이상이고 입구 위(벽+2.5 cm 안)에 있는 비율
  - `task/{side}_thumb_above_rim_mm_near`: 같은 env 들에서 엄지 끝 높이 − 입구 높이의 평균(mm). 양수면 엄지가 입구보다 위에 있다.
- 기하는 iter_03 `rim_hook` 과 같다. 기준값은 cfg `thumb_rim_*` 에 둔다.
- t2r_rh_i03_a2 는 epoch 300 에서 영상을 확인한 뒤 이어 학습할 때부터 이 지표가 기록된다.

### 09.15 보상 입력 손바닥 축 정정 (사용자 결정 "env 가 계약을 지키게")
- RewardContext 계약은 `palm_axes` 앞 3칸 = 손바닥 법선. RH56F1 `*_hl_palm_sensor` 는 URDF 상 **열 2 가 법선**(기저 +x, 엄지 기저 쪽), 열 0 = 손가락 가로(±y), 열 1 = 손가락 길이(+z) — 양손 동일(`test_rh56f1_palm_sensor_column2_is_palmar`).
- iter_00~02 는 palm_axes 를 안 썼다. iter_03 이 처음 `normal = palm_axes[:, 0:3]` 로 방향 보상을 만들어 발견.
- env 는 `ctx_palm_normal_col=2`, `ctx_palm_second_col=1` 로 [법선, 손가락 길이] 를 넣는다. 정책 obs(`_side_obs` 의 R 열 0·1)는 그대로.

