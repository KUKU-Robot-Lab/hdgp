# 실험 레지스트리 — 논문 A: 양손 dexterous 정밀 pouring

> **이 파일이 정본이다.** 무엇을 왜 돌리는지, 어떤 게이트를 통과해야 다음으로 가는지,
> 지금 어디까지 됐는지를 여기서만 관리한다. 러너(`scripts/experiments/run_pour_v1_queue.sh`)와
> 노션 기록은 이 파일을 따른다.

**결정 (2026-08-17)**: 논문을 하나로 통합한다. 환경은 `both/pour_v1`(양손 DG-5FS **물리 파지**,
자산 `a2`=openarm_tesollo_bi_s_rl) 하나만 쓴다.

`both/pour_sensor`(구 RA-L 환경)는 **동결**한다 — 왼손이 2-DOF 그리퍼(`l_hj_gripper_[1-2]`)
전제이고 현재 자산에는 그 관절이 0개라 실행 자체가 불가능하다. 구 자산 기준 자료는
`log_archive/2026-08-17_pre_dg5fs/open-tesol/both/pour-sensor/` 에 11런(TFEvents 포함)
남아 있으므로 필요하면 이력으로 인용한다.

---

## 런 이름 규약

    <논문>-<실험>-<조건>-a<자산>-s<시드>
    예) A-E2-Full-a2-s42

* 정의·검증: `scripts/tools/run_naming.py` (`ASSET_TAGS`: a1=sensor_rl, a2=bi_s_rl)
* **자산 게이트**: 라벨의 `a<N>` 과 실제 로봇 USD 가 다르면 `train.py` 가 학습을 거부한다.
  자산이 이름·기록에 없어 구/신 수치가 뒤섞였던 2026-08-17 사고의 재발 방지 장치다.
* 런 폴더명 = 라벨 (같은 라벨 재실행 시 `-r2`, `-r3` 로 분기).
* **상태를 이름에 붙이지 않는다.** 런 폴더의 `STATUS` 파일 한 줄로 표시한다
  (`running` / `done` / `collapsed` / `partial` / `aborted`).
* `test_history.md` 헤더에 **로봇 자산 경로**가 자동 기록된다.

---

## 선행 조건 (E0) — 이걸 통과하지 못하면 아래 전부 무의미

| # | 항목 | 상태 | 게이트 |
|---|---|---|---|
| E0-1 | `left/grasp_v1`, `right/grasp_v1` (a2) 학습 완주 | **진행 중** | ADR increment 최대 + 성공률 안정 |
| E0-2 | warm HDF5 좌/우 수집 | 대기 | 각 뱅크 ≥2048, `meta/robot_usd` = bi_s_rl |
| E0-3 | 양손 공존 게이트 probe | **PASS** | 아래 5지표 — 전부 통과 |
| E0-4 | 컵 스폰 ∓0.20 재수집 | **완료** | 좌/우 각 2048, 겹침 0.019% |

E0-2 명령:
```bash
python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_right --with_beads
python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_left
```
`--with_beads` 는 **source(우팔)만**. receiver 는 빈 컵으로 시작한다.

### 컵 스폰 분리 (2026-08-18) — 양손 공존의 전제

grasp 의 리프트는 joint7 만 0.31rad 돌리는 스크립트 동작이라 잡은 컵을 **몸쪽으로 스윙**시킨다.
좌/우 뱅크를 독립 샘플링해 합치면 두 컵이 같은 자리를 차지하고, 겹친 채 리셋하면 PhysX 가
침투를 밀어내며 **파지를 뜯어낸다**.

대책은 **grasp 스폰을 ∓0.20 으로 벌리는 것**이다(`object_spawn_y_center`). 정책은 바꾸지
않는다 — 같은 체크포인트가 벌어진 위치에서도 그대로 파지함을 실측했다(∓0.16·∓0.20 각 256/256).
**실기에서도 컵을 벌려 놓고 시작하므로 이 조건이 s2r 과 더 정합한다.**

| 스폰 | warm y (우/좌) | 좌우 간격 | 겹침 | 최소 여유 |
|---|---|---:|---:|---:|
| ∓0.10 (구) | −0.040 / +0.039 | 79.9 mm | **44.3%** | −97.7 mm |
| ∓0.16 | −0.065 / — | 129.2 mm | 6.9% | −55.9 mm |
| **∓0.20 (현)** | **−0.085 / +0.112** | **196.6 mm** | **0.05%** | −24.6 mm |

실제 좌/우 페어 8000쌍 측정(대칭 가정 아님). 좌팔이 y 를 덜 끌어당겨 대칭 예상보다 더 벌어졌다.
남는 0.05% 는 pour_v1 리셋의 **겹침 페어 재추첨**(`left_right_cup_min_gap_m`,
`left_right_cup_redraw_tries`)이 걸러낸다.

> ⚠ `warm_arm_yaw_spread_rad` 로 j1 을 돌려 사후 보정하려던 시도는 **폐기**했다. j1 은 베이스
> 요가 아니다 — FK 실측 `l_aj_1 +0.15` → Δpalm=[−0.057, **0.000**, −0.034]. 적용 시 손과 컵이
> 분리돼 게이트가 더 나빠졌다(비드 0.181, 접촉 0.03, 64env 중 0 완주).

E0-3 명령·게이트:
```bash
isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py --num_envs 64 --steps 200
```
우컵 유지 ≥0.90 · 좌컵 유지 ≥0.90 · 비드 유지 ≥0.90 · 양팔 최소거리 ≥20mm · 손 토크 포화 <0.50

> ★**게이트에 빠진 항목이 있었다 (2026-08-18)**: 위 5지표는 전부 **리셋 직후 품질**만 본다.
> 그래서 스폰 ∓0.20 이 게이트를 5/5 로 통과했는데도 E1 이 학습되지 않았다 — 팔이 뻗은
> 자세를 유지하지 못해 zero-action 200 스텝에서 **128 env 전부 `out_x` 로 죽었다**
> (구 ∓0.10 뱅크는 36 생존). 에피소드가 ~120 스텝에 끝나니 접근할 시간도, 붓기 보상이
> 발화할 기회도 없었다. **`zero-action 200스텝 생존율`을 게이트에 추가해야 한다.**

**2026-08-18 판정: PASS** (num_envs=64, steps=200, 잠정 체크포인트 뱅크)

| 항목 | 값 | 게이트 |
|---|---:|---|
| 우컵 파지유지 | 0.922 | ≥0.90 |
| 좌컵 파지유지 | 0.938 | ≥0.90 |
| 비드 유지 | 0.948 | ≥0.90 |
| 양팔 최소거리 | 60.3 mm (`r_hl_index_tip ↔ l_hl_middle_4`) | ≥20mm |
| 손 토크 포화 | max 0.350 | <0.50 |

기준선(1스텝 후): 우 grip 58.9mm · 좌 107.4mm. 200스텝 동안 우 grip 변화 최대 18.9mm.

> ⚠ **첫 에피소드 완주 0/64 는 게이트 항목이 아니다.** zero-action 은 위치를 붙잡지 않아
> posture prior 가 팔을 끌고, 컵이 `obj_out_x_min` 까지 밀리며 `out_x` 로 끝난다
> (실측: 컵 x 0.326→0.097 하는 동안 grip 거리는 59→64mm 유지). warm state 품질과 무관하다.

> ⚠ **폐기된 수치**: 이전 게이트의 우컵 유지 0.188 / 0.141 은 파지 무결성 측정이 아니었다.
> probe 에 버그 2개가 있었다 — (a) 기준선을 리셋 직후 **stale 버퍼**에서 캡처(리셋 직후
> `root_pos_w` 는 `write_*_to_sim` 반영 전이라 `object_pos` 가 0.000), (b) world
> (`root_pos_w`) 에서 env-local(`palm_center_pos`) 을 빼는 **프레임 혼용**(grip 7610mm).
> 두 버그가 서로를 부분 상쇄해 212mm 라는 그럴싸한 값을 만들었다. 같은 실행의 **양팔 간격
> (world−world 쌍)과 비드 유지(env 계산값)는 유효**하므로 스폰 분리의 효과 판정은 오염되지
> 않았다.

> 비드 수집 경로는 이미 실측 검증됐다 — ep_6000 체크포인트, N=128 에서 컵 내부 비드
> 유지율 전체 0.990 / pour 사용 spec(0-3) 0.995.

---

## E1 — 양손 파지 pour 성립 (기반)

**묻는 것**: 양손이 컵을 쥔 상태에서 붓기가 성립하는가.

| 라벨 | 조건 | override | STATUS |
|---|---|---|---|
| `A-E1-learned-a2-s42` | 왼팔 TCP 학습 | 메커니즘 ON + `receiver_control_mode=learned` | 진행 중 |
| ~~`A-E1-frozen-a2-s42`~~ | 왼팔 TCP 고정 | `receiver_control_mode=frozen` | **aborted** |

### ★ frozen → learned 로 바꾼 이유 (2026-08-18, 실측)

`A-E1-frozen` 은 2442 epoch 동안 **모든 과제 지표가 평탄**했다(epoch 245 이후 변화 없음):

| epoch | 접근 xy | z 여유 | bead_in_target | 에피소드 길이 |
|---|---:|---:|---:|---:|
| 245 | 0.2255 | −0.0095 | 0.000 | 117 |
| 2442 | 0.2265 | −0.0093 | **0.000** | 118 |

원인은 **기하가 성립 불가**였다:

1. pour_v1 은 왼손이 컵을 **들고** 있어 receiver 가 pour_sensor 대비 **7.4cm 높다**
   (z 0.291 → 0.365). source 컵도 z 0.367 → **두 컵이 같은 높이**에서 시작한다.
   pour_sensor 에서는 source 가 8.3cm 위였다. 붓기는 원리상 source 가 위여야 한다.
   (스폰 ∓0.10 좌팔 뱅크도 z 0.369 — **스폰 변경과 무관한 구조적 결과**다.)
2. 그런데 `left_tcp_z_down_m = 0.0` 이라 **왼팔이 컵을 내릴 수 없었다.** 이 값은
   pour_sensor 에서 receiver 가 `kinematic-follow` 라 하강 시 테이블을 관통하기 때문에
   필요했던 것으로, 왼컵이 dynamic 인 pour_v1 에서는 근거가 사라졌는데 값만 따라왔다.
3. 게다가 demo 자세 prior(palm x 0.167)가 receiver(x 0.320)와 **반대 방향**이라
   팔이 −x 로 밀려 `obj_out_x_min=0.05` 에 닿아 ~120 스텝에 종료됐다
   (`command/palm_target_dx≈+0.0003` 인데도 컵 x 0.326→0.097).

**조치**: `left_tcp_z_down_m` 0.0 → 0.08 (필요 하강 약 6.5cm, 그 아래는 테이블이 물리적으로
막는다) + E1 조건을 `learned` 로 변경해 왼팔이 스스로 receiver 를 낮추게 한다.
구 pour_sensor 도 왼팔을 푼 조건에서 동작한 전례가 있다.

**게이트**: `done/left_cup_dropped` < 0.10 · `done/grasp_broken` < 0.20 · bead_in_target 상승 추세
**산출**: E2 의 인계 체크포인트. 실패 시 E2 이후 전부 보류.

### ⚠ 알려진 리스크 — 두 컵 높이가 같다 (2026-08-18 실측)

리셋 로그: `mouth_z_clearance mean=-0.0032 min=-0.1028 max=+0.0882`
(= 주둥이 z − receiver 입구 z. **음수 = 붓는 컵이 받는 컵보다 낮다**, 약 절반의 env)

`pour_sensor` 는 receiver 가 테이블 위(FK z≈0.291)라 항상 낮았다. `pour_v1` 은 왼손이
실제로 컵을 들고 있고 좌우가 같은 스크립트 리프트를 쓰므로 두 컵 높이가 거의 같아진다
(source cup_z 0.373 / 좌컵 z 0.373). `pour_spout_z_lock` 이 매 스텝 주둥이를
`입구 z + pour_z_margin(0.03)` 으로 강제하므로 초기값 자체가 치명적이진 않지만,
min −103mm 인 env 는 시작부터 13cm 를 들어야 한다.

**E1 실패 시 1순위 용의자.** 대응 후보(측정 후 택일):
좌팔 수집 시 리프트 높이 축소 · receiver 컵을 더 낮게 배치 · `pour_z_margin` 확대.

> 참고: 같은 로그의 `[WARN] palm pos clamped by 0.1200m` 은 **오탐**이었다.
> `_ws_clamp_delta` 가 boost 적용 후 값을 boost **이전** 원본과 비교해, 클램프가 없어도
> `warmstart_palm_z_boost`(0.12) 를 그대로 보고했다. clamp 전/후 비교로 수정했다.

---

## ⛔ E1 차단 — pour_v1 제어가 자세를 유지하지 못한다 (2026-08-18, 미해결)

E1 은 두 번 시도했고 둘 다 **과제 지표가 완전히 평탄**했다(`A-E1-frozen` 2442 epoch,
`A-E1-learned` 319 epoch). 둘 다 STATUS `aborted`. 원인은 보상도 정책도 아니었다.

**증상**: zero-action 으로 굴려도 팔이 작업공간 밖으로 밀려 `out_x` 로 죽는다.
컵 x 0.325 → 0.05(경계). 에피소드가 ~120 스텝에 끝나 접근할 시간이 없고, 그래서
`pour`/`bead_in`/`drain`/`success` 보상이 **한 번도 발화하지 못했다**(전부 정확히 0).
gradient 가 없으니 정책은 접근을 포기하고(approach 1.33→1.10, mouth_xy 0.174→0.22)
자세 항만 챙겼다. 총 보상 6배 상승은 **에피소드 길이 6배**(18.7→120)가 전부다.

### 찾아서 고친 결함 2건

| 결함 | 근거 | 효과 |
|---|---|---|
| `hold` 이 목표를 고정하지 않음 — 주석은 "warmstart pose 강제 유지" 인데 코드는 `palm_action` 만 0 으로 하고 목표는 live 계산 | step 120 에 컵 x 0.325→0.119, 58/128 사망 | hold 중 사망 **58 → 0** |
| 명령 상태(`_cmd_spout_env`)를 step 1 에 초기화 — hold 동안 팔이 이동한 뒤 목표가 120 스텝 전으로 되돌아감 | 전환 계단 | 사망 중앙 **119 → 208** |

### 반증된 가설 (재시도 금지 — 근거 있음)

| 가설 | 반증 근거 |
|---|---|
| receiver 가 7.4cm 높아 붓기 불가 | z-lock 이 상대 높이를 이미 묶는다. 7.2cm 내렸더니 우팔도 따라 내려가 z여유 −0.006→−0.026 |
| 메커니즘 3플래그가 팔을 끌어당김 | ON/OFF 사망 117 vs 115, out_x 102 vs 110 |
| 팔 액추에이터 게인 | pour_sensor 와 동일 (stiffness 400 / damping 80) |
| 워크스페이스 경계값 | pour_sensor 와 동일 (`obj_out_x_min` 0.05 등) |
| ~~목표를 실측에 재앵커~~ | **오판정이었다** — `"rim"` 분기만 고쳐 측정했다. 실제 원인이 맞다(위 해결 절) |
| `fabric_q` 를 실제 관절과 동기화 | 드리프트는 0 이 되지만 **팔이 부동**(+y 3cm 명령 280스텝에 palm −11mm) |
| prelift(`warmstart_palm_z_boost`) 과도 | 0/0.06/0.12 사망 동일 (점프 **방향**만 바뀜) |
| 전환 계단 킥 | 램프 0/30/90 에서 out_x 67/70/56 — 미미 |

### ★해결 (2026-08-18) — 목표가 실측 palm 에 매 스텝 재앵커되고 있었다

동작하는 참조 `grasp_v1` 과 **같은 +y 최대명령 시험**을 돌려 갈랐다(같은 fabric·로봇·게인):

| | +y 최대명령 300스텝 |
|---|---|
| grasp_v1 | **+124.8 mm** |
| pour_v1 (구) | **−2.4 mm** |

코드 차이:

```python
# grasp_v1 — 고정 앵커 + 15cm offset → 오차 지속 → 강한 인력
palm_pose = pregrasp_palm_pose_buf + delta
# pour_v1 (구) — 매 스텝 현재 palm 에서 3cm 앞 → 오차 항상 3cm → 약한 인력
_palm_ee_target = self.palm_center_pos + delta[:, :3]
```

pour 의 목표는 **속도 명령**처럼 동작해 오차가 늘 3cm 로 작았고, damping 50 에 눌려 속도가
붙지 않았다. action 만 적분하는 **명령 상태**(`_cmd_spout_env`, plant 를 되읽지 않음)로 바꿨다.

**효과** (zero-action / 스크립트 접근, 128 env):

| 지표 | 전 | 후 |
|---|---:|---:|
| +y 명령 추종 | −2.4 mm | **+114.4 mm** |
| `out_x` 사망 | 72~104/128 | **0** |
| 300스텝 생존 | 0~4 | **70/128** |
| 접근 5cm 이내 도달 | 0.008 | **0.109** |
| 최소 mouth_xy 평균 | 0.15 m | **0.103 m** |

> ⚠ 이 가설은 한 번 "반증" 으로 잘못 처리했었다. `pour_approach_pivot` 이 `"palm"` 인데
> `"rim"` 분기의 목표만 고쳐 측정했기 때문이다. **같은 개념이 두 분기에 나뉘어 있으면
> 어느 분기가 실행되는지 cfg 기본값으로 먼저 확인할 것.**

남은 사망은 `dropped_by_force` 41 · `left_cup_dropped` 12 로 **파지 지속성** 쪽이다
(팔이 이제 실제로 움직여 300스텝을 버티므로 비로소 드러난 문제).

> 스폰 ∓0.20 은 이 문제를 **악화**시키지만 원인은 아니다(구 ∓0.10 뱅크도 out_x 20/128).
> 팔이 더 뻗을수록 중력 토크가 커져 드리프트가 빨라진다.

---

## E2 — 핵심 조건 비교 + 학습 효율 (Table I, Fig 학습곡선)

**묻는 것**: 여자유도 해소 메커니즘과 task-space 정식화가 무엇을 얼마나 바꾸는가.
E1 체크포인트에서 인계해 왼팔 제어를 푼다(`learned`).

메커니즘은 **3플래그를 한 단위로** 켜고 끈다 — `nullspace_baseline` 단독은 죽은 코드다
(`pour_orient_release=True` 분기가 baseline 과 무관하게 demo 자세를 강제한다).

| 라벨 | 조건 | override |
|---|---|---|
| `A-E2-Full-a2-s42` | 제안 방식 (메커니즘 ON + boot) | MECH_ON + `enable_deep_tilt_boot=True` |
| `A-E2-NSdemo-a2-s42` | 메커니즘 ON, boot OFF | MECH_ON + boot False |
| `A-E2-NSnaive-a2-s42` | 메커니즘 OFF | MECH_OFF + boot False |
| `A-E2-JS-a2-s42` | joint-space 대조군 | MECH_OFF + `right_arm_jointspace=True` |

* MECH_ON = `nullspace_baseline=demo` + `pour_orient_release=True` + `pour_bfull_nullspace=True`
* MECH_OFF = `nullspace_baseline=robot_start` + `pour_orient_release=False` + `pour_bfull_nullspace=False`
* 전 조건 공통: `receiver_control_mode=learned`, `enable_demo_pose_reward=False`

**산출**
* Table I — 결정론 eval(1024env·seed100·1200step) 성공률·완전배출·spill
* **학습 효율** — TFEvents 에서 `log/adr_ep_success_rate` 임계(0.10/0.30/0.50/0.70) 도달
  iteration. ⚠ 이 지표는 관대한 학습 프록시다(구 자산에서 NS_naive 0.164 ↔ 결정론 0.0%).
  **효율 비교에만 쓰고 성능 주장에는 쓰지 않는다.**

> ⚠ 구 자산 `sample_efficiency.md`(NS_demo 3,396 vs JS 6,069 iter, NS_naive 천장 0.164)를
> 그대로 인용하지 말 것. 신 자산에서 NS_naive 가 98.6% 였으므로 "천장에 막힌다"가 성립하지
> 않는다. 효율 표는 E2 결과로 새로 만든다.

---

## E3 — reward ablation (Table II)

**묻는 것**: 각 보상 항이 실제로 기여하는가. `A-E2-NSdemo` base, boot OFF.

| 라벨 | 제거 항 | override |
|---|---|---|
| `A-E3-Rnoaim-a2-s42` | 조준 정밀도 | `weight_aim_precision=0.0` |
| `A-E3-Rnoalign-a2-s42` | 정렬 | `weight_align=0.0` |
| `A-E3-Rnointrot-a2-s42` | 내회전 | `weight_introt=0.0` |
| `A-E3-Rnotiltdelta-a2-s42` | tilt delta | `weight_tilt_delta=0.0` |

**산출**: Table II — 항 제거 시 성공률·완전배출 하락폭

---

## E4 — 컵 기하 일반화 (Fig, eval 전용)

**묻는 것**: 학습 분포를 벗어난 컵에서도 되는가. **학습 없이 eval 만** 돌린다.

학습은 source/receiver 각 `(0.85, 1.0, 1.15, 1.30)` 로 이미 섞여 있다
(`source_cup_scale_set` / `left_target_cup_scale_set`).

| 스윕 | CLI |
|---|---|
| 학습 분포 내 | (기본) |
| 균등 축소/확대 | `--cup_scale 0.8` / `1.2` |
| 입구만 좁힘 | `--cup_scale_xy 0.8` / `0.9` |
| 비드 증량 | `--bead_fixed 30` |

> ⚠ **`--cup_scale` 은 receiver 에 그대로 쓸 수 없다** (2026-08-18 확인).
> 구 pour_sensor 는 왼컵이 kinematic-follow 라 스케일을 바꿔도 파지와 무관했다. pour_v1 은
> 왼손이 컵을 **실제로 쥐므로**, 컵만 s배 하고 좌팔 warm 파지자세를 그대로 두면 손가락이
> 허공을 잡거나 컵 벽을 파고든다. 학습 시 섞는 `left_target_cup_scale_set` 은
> `left_warm_spec_map` 으로 spec 매칭이 되어 안전하지만, 단일 `--cup_scale` override 경로는
> 매칭이 없다. E4 에서 receiver 를 스윕하려면 **그 스케일로 수집한 좌팔 warm 뱅크**가 필요하다.
> (source 컵 스윕은 우팔 뱅크에 같은 제약이 걸린다.)

**산출**: Fig — 조건×스윕 완전배출 히트맵. 대상 = E2 4조건 체크포인트.

---

## E5 — sim2real (Fig)

**묻는 것**: 양손 물리 파지가 실기로 넘어가는가.

기존 스택(`sim2real/scripts/pour_sensor_{bimanual,inference}.py`)은 왼손이 2-DOF 그리퍼
전제라 **왼손 20관절 그립 유지 명령을 추가해야 한다**. 그 전에 sim↔브리지 parity 게이트.

**실기 컵 배치**: 좌/우 컵을 로봇 base y 기준 **∓0.20 m** 에 놓는다. 학습 스폰과 같은 조건이며
(E0-4), 양손이 서로를 밀어내지 않는 유일한 배치다. 이 값은 sim 의 `object_spawn_y_center` 와
같은 수치이므로 실기 셋업 문서와 sim cfg 가 어긋나면 **둘 다** 고칠 것.

| # | 항목 | 상태 |
|---|---|---|
| E5-1 | 왼손 그립 유지 명령 추가 (DG-5FS 좌수) | 대기 |
| E5-2 | sim↔브리지 parity 게이트 | 대기 |
| E5-3 | 실기 trial | 대기 |

---

## 실행 순서

```
E0-1(학습 중) → E0-2 → E0-3 게이트
      ↓
E1 (1런)  →  게이트
      ↓
E2 (4런, 2GPU × 2사이클)  →  Table I + 효율 표
      ↓
E3 (4런, 2GPU × 2사이클)  →  Table II
      ↓
E4 (eval 전용, GPU 짧게)  →  일반화 Fig
      ↓
E5 (실기)
```

러너: `scripts/experiments/run_pour_v1_queue.sh E1` / `E2` / `E3`
(각 런 종료 후 결정론 eval 을 자동 실행하고 `docs/eval/pour_v1/` 에 남긴다)

## 상태 갱신 규칙

* 런 시작 시 러너가 `STATUS` 에 `running` 기록, 정상 종료 시 `done`.
* 붕괴·중단은 **사람이** `STATUS` 를 고친다(`collapsed`/`partial`/`aborted`) + 이 파일의
  해당 실험 행에 한 줄 사유를 남긴다. 런 이름은 절대 바꾸지 않는다.
