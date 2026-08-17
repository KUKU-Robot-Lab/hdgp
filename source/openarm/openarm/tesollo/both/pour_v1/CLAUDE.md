# both/pour_v1 — 양손 **물리 파지** pour (pour_sensor fork)

@../pour_v5/CLAUDE.md

> 위 공통 규칙(목표·reward 구조·진단 지표·reward-audit 이력·코드 수정 규칙)을 그대로 따른다.
> 아래는 **pour_v1 고유 사항만**.

---

## 왜 포크했는가

실기를 **DG-5FS 양손**으로 운용하기로 확정됐다. 그런데 `both/pour_sensor` 는

- 로봇 자산이 `openarm_tesollo_sensor_rl` = **오른손만 Tesollo 20관절, 왼손은 2-DOF 그리퍼**
- receiver 컵이 `kinematic_enabled=True` + `disable_gravity=True` 인 **kinematic body** 로,
  매 스텝 왼손 pose 를 그대로 써 넣는 **kinematic-follow** (= 물리 파지가 아니다)

라서 "양손이 각자 컵을 쥔 상태에서 붓기 시작" 을 표현할 수 없다.

`pour_sensor` 는 **RA-L 실험 환경이므로 손대지 않고 보존**한다 (구/신 구조 비교 대조군).
`pour_v1` 이 그 포크다.

## pour_sensor 와의 구조 차이

| 항목 | both/pour_sensor | **both/pour_v1** |
|---|---|---|
| 로봇 USD | `openarm_tesollo_sensor_rl` | `openarm_tesollo_bi_s_rl` |
| 왼손 | `l_hj_gripper_[1-2]` (2-DOF) | `l_hj_<finger>_[1-4]` (**20-DOF DG-5FS**) |
| receiver 컵 | kinematic + 중력off, 왼손에 follow | **dynamic 강체, 왼손이 물리적으로 파지** |
| 왼팔 초기자세 | `LEFT_ARM_REST` 고정 | **왼쪽 warm bank** (left/grasp_v1 산출물) |
| warm bank | 1개 (오른팔) | **2개** (`warm_state_paths` / `left_warm_state_paths`) |
| 왼팔 TCP rest | 전 env 공통 FK 상수 | **per-env** (리셋 후 첫 스텝 캡처) |
| 종료 사유 | out_x/out_y/fallen/dropped_by_force/bead_fallen/grasp_broken | **+ `left_cup_dropped`** |
| z-lock 기준 | `_target_opening_w[:,2]` 직접 | **1차 저역통과** (`pour_spout_z_lock_lpf_alpha`) |
| action / obs | 15D / 55·144 | **동일 (불변)** |

## 불변으로 지킨 것 — action 15D

왼손 20관절은 **warm 파지자세 position-hold** 만 하고 action 에 넣지 않는다.

이유는 단계형 학습이다: `receiver_control_mode=frozen`(1단계) → `learned`(2단계) 로
넘어갈 때 **action 차원이 같아야 체크포인트가 그대로 인계**된다. `frozen` 은
`action[12:15]` 를 무시할 뿐 차원을 바꾸지 않는다.

왼손 tip 접촉센서도 같은 이유로 붙이지 않았다 — obs 차원이 바뀌면 재학습 강제.
왼컵 낙하는 접촉이 아니라 **기하**(왼손 palm↔컵 거리 + 컵 z 낙하)로 판정한다.

## 학습 순서 (확정)

```
1. left/grasp_v1, right/grasp_v1 (신 USD) 학습 완주
2. warm HDF5 좌/우 순차 수집
     collect_grasp_v1_warm_states.py --robot tesollo_right --with_beads   ← source: 비드 채움
     collect_grasp_v1_warm_states.py --robot tesollo_left                 ← receiver: 빈 컵
3. 양손 공존 게이트 probe (아래 "미검증 가정" 참조)
4. receiver_control_mode=frozen 학습   ← 왼팔 고정
5. receiver_control_mode=learned 학습  ← 왼팔 제어 해제, 4의 체크포인트 인계
```

## 비드는 **수집 시점**에 채운다

pour 의 오른손은 warm 자세로 **동결된 수동 스프링**이다
(`freeze_grasp_hand_during_episode=True`, 강성 5.0). 재조임하는 정책이 없다.
그래서 빈 컵으로 만든 파지를 텔레포트한 뒤 비드(20 × 1g)를 나중에 넣으면 그 하중을
수동으로 흡수해야 하고, 컵을 놓칠 수 있다. 구 pour 는 k=400 에서 그렇게 했지만
게인을 grasp_v1 값(5.0)으로 통일한 지금은 성립을 보장할 수 없다.

→ **수집 시 컵을 채우고, 그 상태를 warm HDF5 로 넘겨 pour 가 복원한다.**

| 단계 | 무엇 |
|---|---|
| `grasp_v1` (수집 전용, 기본 off) | `collect_with_beads=True` → 컵 안에 비드 소환 후 정책이 파지 형성 |
| warm HDF5 | 선택 데이터셋 `bead_state` (N, k, 13) env-local + `meta/collected_with_beads` |
| `pour_v1` 리셋 | `bead_state` 있으면 복원 + `_beads_spawned=True`(재소환 금지) / 없으면 기존 hold-end 소환 |

- 비드 정의는 **`openarm/common/bead_assets.py` 한 곳**에만 둔다. 수집측과 소비측의 질량·마찰·
  반발·솔버가 다르면 같은 좌표를 복원해도 동역학이 달라진다.
- 개수 불일치(`cache k ≠ cfg.bead_count`)나 일부 파일만 보유하면 **경고 후 기존 소환으로 degrade** 한다.
- 접촉 판정 오염 없음: grasp 의 tip/distal/middle 센서는 `object_contact_filter`(컵 전용)라
  비드 접촉이 `num_contacts` 에 안 들어간다 → warm 성공 판정 규약 불변.
- `receiver`(왼팔)는 빈 컵이라 비드를 붙이지 않는다. 컵 자산이 실제 SDF(`cup_big_rl.usd`,
  sdfAPI=True/res=64)라 나중에 비드를 받을 수 있다 — convexHull 폴백은 **visdex 원본**의
  문제였고 08-16 에 SDF 사본으로 교체돼 이미 해결됐다.

### 자산 출처 검증

좌/우 grasp 의 warm export 가 `meta/robot_usd` 를 기록하고, pour 의 로더가
`_EXPECTED_ROBOT_USD="openarm_tesollo_bi_s_rl"` 와 대조한다. 기록이 없으면 경고,
있고 다르면 **hard fail**. 2026-08-17 DG-5F→DG-5FS 사고(텐서 차원이 같아 구 캐시가
조용히 로드됨)의 재발 방지장치다.

## 공존 게이트 probe (학습 전 필수)

```bash
isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py --num_envs 64 --steps 200
```

정책을 끄고(action=0) 좌/우 warm 을 동시에 적용해 5개를 재고 PASS/FAIL 을 찍는다.

| 지표 | 기준 | 왜 |
|---|---|---|
| 우컵(source) 유지율 | ≥0.90 | 파지가 풀리지 않는가 |
| 좌컵(receiver) 유지율 | ≥0.90 | 물리 파지로 바뀐 컵을 놓치지 않는가 |
| source 비드 유지율 | ≥0.90 | 수집 시 채운 비드가 리셋 직후 쏟아지지 않는가 |
| 양팔 최소거리 | ≥20 mm | 두 자세 조합이 서로 파고들지 않는가 |
| 손 토크 포화율 | <0.50 | k=5 로 내린 뒤에도 파지가 성립하는가 |

- **좌/우 뱅크가 둘 다 없으면 거부한다**(`--allow_no_warm` 로만 우회). 뱅크 없이 재면
  FK 고정배치 degrade 경로를 재는 것이라 게이트가 성립하지 않는다.
- 양팔 간격은 **거리 대리지표**다 — articulation 이 `enabled_self_collisions=False` 라
  팔끼리 물리 접촉이 아예 생성되지 않는다(접촉력으로는 간섭을 볼 수 없다).
  - 근위부(팔 0~3)는 **기본 제외**한다. 어깨 마운트가 몸통에서 구조적으로 붙어 있어
    min 을 지배하면 정작 보고 싶은 손 주변 간섭이 가려진다(`--gap_all_bodies` 로 해제).
  - probe 는 **최소를 만든 body 쌍 이름을 함께 찍는다.** 숫자만 보면 어깨인지 손인지
    구분이 안 돼 해석 불가다. 실측(warm 없는 점검 실행)에서 최소 쌍은
    `r_hl_middle_4 ↔ l_hl_middle_tip` 18.8mm — 즉 **두 손의 중지**가 최근접이었다.
    20mm 기준이 실제 기하에 근접한 값이라는 뜻이므로, 첫 실측 후 조정 여지가 있다.
- `env.step()` 은 종료된 env 를 자동 리셋하므로, 각 지표는 **첫 에피소드 동안의 값만**
  얼려서 집계한다(리셋 후 값이 섞이면 게이트가 무의미해진다).

## ⚠ 미검증 가정 — 학습 전 반드시 재기

1. **좌/우 warm state 공존.** 좌 warm 은 오른팔이 REST 인 상태에서, 우 warm 은 왼팔이
   REST 인 상태에서 각각 수집된다. `_sample_left_warm()` 은 좌/우를 **독립 샘플링**하는데,
   이는 "두 팔이 y 로 20cm 떨어져 서로 간섭하지 않는다" 는 가정이다.
   자기충돌이 관측되면 검증된 페어만 쓰도록 바꿔야 한다.

2. **손 게인을 grasp_v1 로 통일한 결과** (사용자 확정, 아래 "grasp_v1 정합" 참조).
   좌우 손 5.0 / 2.0. 구 pour 값(200~400 / 35~60)을 폐기했으므로 deep-tilt 거동이
   달라진다. **예상**: 접촉력·토크 포화율 대폭↓, tilt 완충 여지↑ → `tilt_frac` 유지·개선,
   반면 파지 마진↓ → `grasp_broken` / `left_cup_dropped` 상승 가능.
   Phase 1 게이트에서 이 두 지표를 **가장 먼저** 확인한다.

3. **왼컵 낙하 임계값** (`left_cup_drop_dist_m=0.06`, `left_cup_drop_z_m=0.08`) 은
   물리 관측 없이 정한 초기값이다. 정상 파지 중 흔들림이 이를 넘으면 오탐이 난다.

4. **왼손 body 이름.** DG-5FS 마운트 체인은 fixed 조인트 4개라 Isaac 임포트에서 링크가
   병합될 수 있다. `_resolve_body_index()` 가 후보를 순회하고 전부 없으면 예외를 던진다
   (구 코드의 `else -1` 조용한 오인덱싱을 대체).

## grasp_v1 정합 — "수집 물리 = 소비 물리" (사용자 확정)

pour_v1 은 grasp_v1 이 만든 warm state 에서 시작하므로, **warm 을 만든 물리와 재생하는
물리가 같아야** 리셋 직후 파지가 유지된다. 전면 대조 결과는 아래와 같다.

### 따라간 것

| 항목 | 값 | 비고 |
|---|---|---|
| 손 게인 (좌·우) | **5.0 / 2.0** | grasp_v1 의 "파지 손" 값. pour_v1 은 양손이 파지하므로 둘 다 |
| 팔 그룹 | `{side}_arm_proximal/elbow/wrist` | r2s 캘리브가 이 이름을 기대 — 다르면 주입이 조용히 fallback |
| 팔 게인 | 400 / 80 | 기존 pour 와 동일값 (거동 변화 없음) |
| 팔 friction | 0.213 / 0.493 / 0.151 | **신규** — pour_v1 엔 실측 마찰이 아예 없었다 (07.29 real2sim) |
| 로봇 spawn/articulation | solver 16/1, self-collision off | 이미 동일 |
| source 컵 질량 | 0.134 kg | 이미 동일 (`test_source_cup_mass_fixed` 로 잠김) |
| 물체 마찰 | USD 기본 | grasp ADR 중립(×1.0) = pour 기본. 수집 시 ADR off 라 동일 |

### 일부러 따라가지 **않은** 것 — 따라가면 pour 가 깨진다

| 항목 | pour_v1 | grasp_v1 | 따라가면 |
|---|---|---|---|
| source 컵 자산 | `cup_big_sdf.usd` (속 빈 컵) | `cup_big_rl.usd` (속 찬 원통) | 비드가 컵에 못 들어감 = 붓기 불가 |
| palm 워크스페이스 | x_min **−0.30**, z_min 0.10 | x_min 0.20 | 깊은 tilt 시 palm 스윙이 막혀 tilt plateau (test11 근거) |
| `episode_length_s` | 20.0 | 10.0 | 붓기 완료 전에 종료 |
| 비드 물리 / GPU 버퍼 | 비드 전용 튜닝 | 해당 없음 | — |

이 "따라가지 않음" 도 테스트로 잠갔다(`test_palm_workspace_intentionally_differs_from_grasp`,
`test_source_cup_asset_intentionally_differs_from_grasp`) — 정합 작업이 여기까지 번지는 것을 막는다.

## 코드 수정 주의

- **`pour_sensor` 와 동기화하지 않는다.** pour_sensor 는 RA-L 대조군으로 동결이다.
  reward 코어를 고칠 때 pour_v5 ↔ pour_sensor 동기화 규칙이 pour_v1 에는 적용되지 않는다.
- 왼팔 warm 적용은 reset 경로 4곳이 공유하는 헬퍼
  (`_sample_left_warm` / `_write_left_warm_joints` / `_place_left_cup`) 에만 둔다.
  경로별로 직접 쓰면 표류한다 — 계약 테스트가 호출 횟수를 검사한다.
- 계약 테스트: `tests/test_bimanual_grasp_start.py` (정적, Isaac 불필요).
