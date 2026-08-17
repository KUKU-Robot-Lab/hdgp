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
| E0-3 | 양손 공존 게이트 probe | 대기 | 아래 5지표 |

E0-2 명령:
```bash
python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_right --with_beads
python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_left
```
`--with_beads` 는 **source(우팔)만**. receiver 는 빈 컵으로 시작한다.

E0-3 명령·게이트:
```bash
isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py --num_envs 64 --steps 200
```
우컵 유지 ≥0.90 · 좌컵 유지 ≥0.90 · 비드 유지 ≥0.90 · 양팔 최소거리 ≥20mm · 손 토크 포화 <0.50

> 비드 수집 경로는 이미 실측 검증됐다 — ep_6000 체크포인트, N=128 에서 컵 내부 비드
> 유지율 전체 0.990 / pour 사용 spec(0-3) 0.995.

---

## E1 — 양손 파지 pour 성립 (기반)

**묻는 것**: 양손이 컵을 쥔 상태에서 붓기가 성립하는가. 왼팔을 고정한 채 오른팔만 학습.

| 라벨 | 조건 | override |
|---|---|---|
| `A-E1-frozen-a2-s42` | 왼팔 TCP 고정 | `env.receiver_control_mode=frozen` + 메커니즘 ON |

**게이트**: `done/left_cup_dropped` < 0.10 · `done/grasp_broken` < 0.20 · bead_in_target 상승 추세
**산출**: E2 의 인계 체크포인트. 실패 시 E2 이후 전부 보류.

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

**산출**: Fig — 조건×스윕 완전배출 히트맵. 대상 = E2 4조건 체크포인트.

---

## E5 — sim2real (Fig)

**묻는 것**: 양손 물리 파지가 실기로 넘어가는가.

기존 스택(`sim2real/scripts/pour_sensor_{bimanual,inference}.py`)은 왼손이 2-DOF 그리퍼
전제라 **왼손 20관절 그립 유지 명령을 추가해야 한다**. 그 전에 sim↔브리지 parity 게이트.

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
