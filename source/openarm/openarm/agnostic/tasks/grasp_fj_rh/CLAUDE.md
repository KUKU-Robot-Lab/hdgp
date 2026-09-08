# grasp_fj_rh — 태스크 규칙 (Track B · RH56F1 언더액추 손 · 로컬 2,048 env)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 베이스 트랙 `../grasp_fj`(팔 어댑터) ·
> 보상·목표열 설계 `../grasp_kp/DESIGN.md` · 기준선 `../grasp_s2r` 는 **불변**

## 목적
Track B(팔 7D 관절 증분+EMA · Fabrics 없음 · 접촉 센서 0)를 **손만 바꿔** RH56F1 로 옮긴다.
같은 보상·같은 목표열·같은 관측 구조에서 **엔드이펙터가 바뀌면 무엇이 필요한가**를 재는 트랙이고,
동시에 RH56F1 실기 배포 경로(정책 → pd → 드라이버, 3노드)를 그대로 쓴다.

★fabric 트랙(grasp_ua)이 아니라 B 를 베이스로 한 이유: RH56F1 fabric 은 양팔 26 DOF 클래스라
슬롯·전용 params·손끝 FK 게이트·짐벌락 회전중심이 전부 따라온다(09.02 에 일반화 축 5개를
새로 만들어야 했다). B 는 팔이 관절공간이라 그 배선이 통째로 필요 없다.

## 이 손이 tesollo 와 다른 지점 (전부 09.07 `probe_fjrh_calib` 실측)
| 항목 | tesollo(dg5f-m) | RH56F1 |
|---|---|---|
| 손 관절 | 20 전부 구동 | **구동 6 + 종속 6**(PhysX mimic, 자산이 들고 있다) |
| 손 액션 | 채널 3 × 5지 = 15 | **손가락별 6슬롯**(`hand_layout=per_finger`) |
| 액션·관측 | 22 / 131 / 155 | **13 / 94 / 118** |
| 엄지 ch1 | 대향(_2) | **굴곡**(`thumb_2`, 한계 0~0.475) → `oppose_grip_delta_rad=0.0` |
| 파지 창 | — | 열림 **105.5mm** → 완전 폐쇄 **46.6mm**(엄지↔4지) |
| 물체 | cup_family(105~161mm) | **shaker_small**(48~66mm) — cup_family 는 물리적으로 안 들어간다 |
| 손 게인 | 벤더 1.5/0 | **5.0/2.0**(벤더 PD 없음 — `NO_VENDOR_PD["rh56f1_hand"]`) |
| 종속관절 | 없음 | **0/0 필수**(제약이 위치를 정한다) |

팔 PD 게인은 양쪽 다 벤더 `control_gains.yaml` 하나뿐이다(70/70/70/60/10/10/10).

## 핵심 계약 (`tests/test_grasp_fj_rh_contract.py` 가 잠근다)
| 계약 | 왜 |
|---|---|
| 종속 12관절 액추에이터 = **0/0** | 09.07 실측: damping 0.1 만 줘도 mimic 제약과 싸워 구동관절이 하한 **483 rad** 밖으로 나가며 발산했다 |
| 부팅에서 USD `physxMimicJoint:rotZ:gearing` 을 URDF 배율과 대조 | 결합이 사라지면(09.02: headless 빌드가 12개를 통째로 잃었다) 종속관절이 제약도 드라이브도 없는 자유 관절이 되어 손가락이 흐물거린다 — 지표에는 "파지가 안 된다"로만 보인다 |
| mimic 배율은 코드가 아니라 **자산 URDF** 에서 읽는다 | 코드에 적으면 자산과 갈린다 |
| 부팅에서 open ≠ grip · grip ∈ 관절한계 | 노브 하나가 손가락을 조용히 죽인다(09.02 엄지: 300스텝 폐쇄 지령에 0.0001 rad) |
| env 가 덮는 것은 훅 3개(+헬퍼 3개)뿐 | 팔·손 경로를 덮으면 A/B/RH 대조가 성립하지 않는다 |
| per_finger 팔 폭은 mixin 훅 `_arm_slot_width()` | 리터럴 6(palm)이 남아 있으면 B 에서 부팅이 죽는다 |
| 전 DOF 액추에이터 커버리지(중복 0) | 빠진 관절은 조용히 무구동 자유회전한다 |
| 부팅에서 종속관절 한계를 ±1.5 rad **확장**(`mimic_dep_limit_margin_rad`) | 없으면 테이블 접촉에서 mimic 제약과 관절한계가 동시 불만족이 되어 폭주한다(아래 실측) |

## 실측 상수 (09.07 · `our_source/rh56f1_fj_calib/` · env 부팅 보고와 일치)
```
홈 palm(r_hl_palm_sensor)  (0.3100, −0.2999, 0.4188) · euler_zyx (0.4°, −64.9°, −90.4°)
                           → 프로필 `palm_rot_center_deg=(0, −65, −90)` 이 홈과 같다
홈 케이지 중심            (0.3810, −0.2661, 0.4006) · 반경 53mm
케이지 − 컵               (+0.3, −105, +95) mm  ← 스폰 (0.38, −0.16) 에서 x 가 이미 정렬돼 있다
손 최하단 z(open)          0.3478  (테이블 0.205 · 벌점선 0.215 위)
palm − 손 최하단           0.071 m
파지 창(엄지↔4지)         open 105.5mm → grip 46.6mm
mimic 추종(자유)           배율 대비 오차 ≤0.05 rad (리더 0.1/0.4/0.8/1.2/1.5 스윕)
접촉 스트레스(자유물체)    종속 |qd|max 7.2 rad/s · 한계위반 0 · mimic 오차 ≤0.04 rad
접촉 스트레스(고정물체)    종속 |qd|max 47.3 rad/s · 한계위반 0.018 rad — **비현실 조건**
```
★`object_spawn_center` 는 케이지 x 에 정렬해야 접근이 y-z 2D 가 된다. 홈을 바꾸면 다시 재라.
★★프로브는 **중력보상을 env 와 같게** 걸어야 한다. 처음 실행에서 이걸 빼먹어 팔이 처진
  자세를 홈으로 재고(palm z 0.345 vs 실제 0.419) "스폰 x 를 0.334 로 옮겨야 한다"는 잘못된
  결론까지 갔다. 진실원천은 env 부팅 로그의 `홈 케이지 중심 … 케이지−컵` 줄이다.

## reward-audit (09.07 · 게이트 임계 변경분)
바꾼 것은 **게이트 하나**(`blocked_err_thr_rad` 1.00 → 0.25)와 자세 노브·물체 뱅크다.
보상 항·가중치는 `grasp_fj` 그대로다(A/B/RH 대조의 전제).

| Check | 판정 | 근거 |
|---|---|---|
| 1 국소최적 | ✓ | 이 게이트는 보상을 주지 않는다. 폐쇄 **진행만** 멈춘다 — 새 고보상 경로가 없다 |
| 2 hacking | ✓ | 일찍 멈추면 파지가 약해져 리프트·목표 보상이 **줄어든다**(유인이 반대 방향) |
| 3 파지 상충 | ✓ | 0.25 는 토크 포화 오차(effort/kp = 0.20)보다 위 → 진짜 정체된 관절만 얼린다. 자유공간 정상 오차 0.095 의 2.6배 |
| 4 기존 파괴 | ✓ | 신규 트랙(체크포인트 없음). 형제 트랙 기본값은 보존 — 계약 테스트가 잠근다 |
| 5 측정 | ✓ | `ctrl/hand_blocked_frac` 신설(A/B 는 접촉 진단 경로에만 있어 이 트랙에서 안 나왔다) |

관측 로그(OPEN)가 요구하는 추가 검사:
- **#0005 목표박스 도달성** — Track B 는 부모의 도달성 단언이 일찍 반환한다(palm 박스가 지령
  한계가 아니므로). 그래서 IK 로 직접 쟀다: 목표 박스 8꼭짓점+중심 **최악 1.9mm · 한계 관절 0개**.
- **#0012 표본 상태 분포** — 팔은 증분+EMA(복원력 없음)라 스텝당 목표 변화 α·k = 0.0167 rad.
  600스텝 무작위 보행 σ = 0.0167·√600 = **0.41 rad/관절**, 과제 거리(케이지→컵 0.14 m ≈ 0.28 rad)
  보다 크다 → 탐색이 과제 거리를 덮는다.
- **#0010 총량 대 트레이너 임계** — 최대 raw 에피소드 보상 ≈ goal 1000×50 + lift 300 ≈ 5.0e4
  vs `score_to_win` 1e6 → 20배 여유.
- **#0006 실효율** — α·k/dt = 1.002 rad/s(선언 1.0)를 cfg 가 부팅에서 대조한다.

**판정: ACCEPT.**

## ★09.07 학습 1차 실패와 해결 — 종속관절 한계 확장
`rh_b1`(한계 확장 없음)은 부팅·차원·게이트를 전부 통과하고도 **222 epoch 중 118 epoch 에서**
`ctrl/mimic_err_max` 가 0.15 → 400~916 rad 로 폭주했다. 상관은 하나뿐이었다:

| | mimic_err max | 중앙값 | >10 rad epoch | dep_qd max | 손 최하단 min |
|---|---|---|---|---|---|
| `rh_b1` 확장 없음 | **916.3** | 35.13 | **118 / 222** | 1715.9 | 0.145 |
| `rh_b2_depmargin` ±1.5 rad | **7.8** | 1.35 | **0 / 299** | 971.1 | 0.136 |

★두 런의 **테이블 관통 깊이는 같다**(0.145 vs 0.136). 즉 고친 것은 "덜 부딪히게" 가 아니라
**제약이 동시 만족 불가가 되지 않게** 한 것이다 — 09.02 가 지목한 그 메커니즘이 맞았다.
접촉이 구동관절을 하한 밑으로 역구동하면 mimic 요구치가 종속 한계 밖이 되고 솔버가
에너지를 주입한다. 한계는 백스톱일 뿐이라 넓히는 것은 물리적으로 공짜다.

⚠남은 관찰거리: `ctrl/hand_dep_qd_max` 는 확장 후에도 순간 971 rad/s 까지 뛴다(접촉 킥).
  mimic 오차가 ≤8 rad 에 머무는 한 제약은 서 있는 것이므로 지금은 감시 항목이다.

★검토했다가 기각한 것: `max_depenetration_velocity` 1000 → 1 (고정 원통 87.7 → 91.3 · 자유
  물체 7.2 · 테이블 접촉에서도 개선 없음) · 종속관절 damping 0 → 0.1 (**발산**, 구동관절이
  하한 483 rad 밖). 09.02 기록에서 가져온 처방을 그대로 쓰지 않고 현 자산에서 다시 쟀다.

## 성공 기준
1. 스모크(16 env): 부팅 로그에 언더액추 확인·홈 케이지·차원이 찍히고 `abnormal ≈ 0`.
2. `ctrl/mimic_err_max` — **전 env 최댓값**. 건전 기준선은 `rh_b2` 실측 **중앙 1.3 · 최대 7.8**
   (300 epoch). 10 rad 을 넘는 epoch 이 반복되면 종속 한계가 다시 모자란 것이다 —
   `mimic_dep_limit_margin_rad` 를 올린다. `ctrl/hand_dep_qd_max` 는 접촉 킥으로 수백까지
   뛰지만, mimic 오차가 같이 오르지 않으면 제약은 서 있는 것이다.
3. `ctrl/joint_err_max` < 0.1 rad(무작위 초기 0.13 실측) · `task/hand_floor_depth_max` ≈ 0.
4. `ctrl/hand_blocked_frac` 이 접촉 구간에서 0 을 벗어난다(0 고정이면 임계가 크다).
5. `stage/lifted` 상승 → `task/successes` 상승. ★FRESH 는 3,000~4,000 epoch 전에 판정하지 않는다.

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm python3 -m pytest \
  source/openarm/openarm/agnostic/modules/tests \
  source/openarm/openarm/agnostic/tasks/grasp_kp/tests \
  source/openarm/openarm/agnostic/tasks/grasp_fj/tests \
  source/openarm/openarm/agnostic/tasks/grasp_fj_rh/tests -q

# 캘리브 프로브(자산·홈을 바꿨을 때)
~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_fjrh_calib.py

# 스모크
./train.sh open-rh_r_grasp_fj_rh-lstm rh_smoke --num_envs 16 --max_iterations 5 \
  agent.params.config.minibatch_size=256 agent.params.config.central_value_config.minibatch_size=256

# 본 학습 (로컬 · SAPG 없음 · 오른팔 전용 자산)
NOTE="RH56F1 Track B fresh" ./train.sh open-rh_r_grasp_fj_rh-lstm rh_b1 \
  --num_envs 2048 --headless env.profile_name=rh56f1_right_only
```
로그 `log/rl_games/open-rh/right/grasp-fj-rh/<label>/`.
★SAPG id 는 없다 — 벤더 rl_games 포크와 블록 ≥2 가 필요해 서버 전용이다.
★★**오른팔 전용 자산으로 돌린다**(`rh56f1_right_only`). 왼팔 링크 35·관절 35 를 합성에서
  끈 오버레이(`assets/robot/openarm_rh56f1_bi_rl/*_right.usda`,
  생성기 `scripts/tools/make_right_only_overlay.py`)다. 2,048 env 3 epoch 실측:
  step fps 31.3~35.1k → **45.5~50.9k**, total 28.8k → **40.5k**. 차원·부팅 계약은 동일하고
  (13/94/118, mimic 게이트 통과) 오른쪽 물리는 그대로다. ★체크포인트는 양팔 자산과 비호환이다.
