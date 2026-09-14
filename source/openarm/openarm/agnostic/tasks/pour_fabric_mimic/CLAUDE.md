# pour_fabric_mimic — 태스크 규칙 (09.14 신설: 저차원·언더액추 손 전용 text2reward 트랙)

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 원본 트랙 [`../pour_fabric/CLAUDE.md`](../pour_fabric/CLAUDE.md)
> (보상 생성 루프·검증 게이트·s2r 관측/DR 규약은 원본과 같다 — 여기엔 **다른 점만** 적는다)

## 정체성

`pour_fabric`(DG-5F 20관절)의 **사본**을 RH56F1(구동 6 + PhysX mimic 종속 6)용으로 고친 트랙.
사용자 결정 09.14: "원본 냅두고, 저차원 핸드 전용을 새로 만든다". 원본 파일은 이 트랙 때문에 바뀌지 않는다
(`tests/test_mimic_contract.py::test_original_pour_fabric_untouched_by_this_track` 가 잠근다).

- gym id `open-rh_b_pour_fab_mimic` (+`-play`, `-lstm`) · 로그 `log/rl_games/open-rh/both/pour-fab-mimic/`
- 액션 **24** = (palm 6 + 손 6) × 2 · actor obs **172** · critic **214**
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
