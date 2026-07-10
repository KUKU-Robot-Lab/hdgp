# Real2Sim Autotune 프레임워크 구현 계획

작성일: 2026-07-10
기준 문서: `repo/Sim-to-Real .../r2s_autotune_guide.md` (논문 Algorithm 1)
최종 목표: real2sim 정합 → sim2real 정책 배포

---

## 1. 요구사항 재진술

논문 Algorithm 1(Real-to-Sim Autotune)을 hdgp 스택에 구현한다. RL 정책 학습이 아니라 **학습 이전 단계의 actuator system identification**이다.

```
real robot log → HDF5 (q_cmd, q_real, dq_real)
              → seed calibration JSON
              → parallel replay 후보 탐색 (Isaac Lab)
              → best calibration JSON
              → OPENARM_REAL2SIM_ACTUATOR_CALIBRATION 으로 RL env 주입
              → RL 학습 → sim2real 배포
```

대상 asset (사용자 확정: `hdgp/assets/robot/` 세팅 기준, canonical 이름):

```
hdgp/assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
hdgp/assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd
```

사용자 확정 사항:
- **Real 데이터는 나중.** 합성 궤적으로 파이프라인을 먼저 만들고 검증한다.
- **두 로봇 병행.** 공통 코드 + asset별 config 분리.
- **canonical만.** legacy joint 이름(`rj_dg_*`, `openarm_right_joint*`)은 신규 코드에서 쓰지 않는다.

---

## 2. 현장 검증 결과 (가이드 문서와 실제 저장소의 차이)

가이드는 설계는 맞지만 세부 사실이 낡았다. 신규 코드는 **가이드의 표가 아니라 실제 코드**를 source of truth로 삼는다.

| # | 가이드 주장 | 실제 | 영향 |
|---|---|---|---|
| 1 | `assets/openarm_tesollo_sensor/*.usd` | 존재하지 않음. 실제는 `assets/robot/openarm_tesollo_sensor_rl/` | grasp_v11 `env_cfg:402`가 죽은 경로 참조 |
| 2 | Tesollo env가 canonical 이름 사용 | grasp_v11은 legacy (`rj_dg_[1-5]_2` 등, 4파일 78토큰) | canonical USD로 바꾸면 regex 0개 매칭 |
| 3 | group `rh56f1_right_drive` | 실제는 `rh56f1_right_flexion`(k=400) + `rh56f1_right_abduction`(k=200) 분리 | 가이드대로 JSON 만들면 조용히 무시됨 |
| 4 | grasp_v1이 대상 | 최신 성공 정책은 grasp_v2. v2에는 calibration hook 없음(하드코딩) | hook 이식 필요 |
| 5 | identification 데이터 존재 전제 | bag/HDF5/JSON 전부 없음. excitation 명령 노드도 없음. db3→identification HDF5 변환기도 없음 | 최대 블로커 |
| 6 | seed 추정기가 system ID | `real2sim_actuator_calibration.py:212`는 휴리스틱 스케일 공식 | seed로만 유효, refinement 필수 |

추가 발견:
- `get_actuator_params()`는 `delay_steps`를 **반환하지 않는다.** JSON에 기록되지만 Isaac Lab에 전달되지 않는 죽은 필드다.
- `test_real2sim_actuator_calibration.py` 4개 통과. manifest `source_to_canonical_joints` 매핑 완비.
- `write_joint_stiffness_to_sim(stiffness, joint_ids, env_ids)`가 `(num_envs, num_joints)` 텐서를 받는다 → **후보 K개를 단일 sim의 K개 env에 병렬 배치 가능.**

---

## 3. 핵심 설계 결정

**D1. 후보는 USD 복사본이 아니라 JSON + 런타임 gain override.**
가이드 §1에 동의. `write_joint_stiffness_to_sim(..., env_ids=)`로 env마다 다른 gain을 쓴다.

**D2. Replay env는 grasp env를 재사용하지 않는다.**
`r2s_autotune/replay_env.py`에 articulation-only 최소 씬을 새로 만든다. 이유: grasp env는 물체·reward·fabric·reset 로직이 얽혀 있어 순수 actuator tracking 측정에 잡음이 된다. 또한 grasp_v11의 legacy 이름 문제를 R2S 작업에서 분리할 수 있다.

**D3. 합성 ground-truth 복원 테스트를 검증의 축으로 삼는다.**
real 데이터가 없는 동안 파이프라인의 정확성을 확인할 방법은 이것뿐이다.

```
알려진 파라미터 A로 sim 궤적 생성
  → 그 궤적을 "real"로 위장해 입력
  → autotune이 A를 복원하는가?
```

복원 오차가 크면 파이프라인 버그다. 이 테스트는 real 데이터가 와도 회귀 테스트로 계속 쓴다.

**D4. group 이름/regex는 env cfg가 source of truth.**
가이드 §7.1/§8.1 표를 쓰지 않는다. `configs/*.yaml`에 실제 group을 적고, contract 테스트로 env cfg와 대조한다.

---

## 4. Phase 계획

### Phase 0 — Contract 고정 및 정합 검증

디렉토리:
```
hdgp/scripts/r2s_autotune/
  configs/{tesollo_sensor,bi_rh56f1}_ranges.yaml
  configs/{tesollo_sensor,bi_rh56f1}_replay.yaml
  joint_contract.py
  tests/test_joint_contract.py
hdgp/logs/r2s_autotune/{seeds,runs,results}/
```

`joint_contract.py` 책임:
- manifest 로드 → `control_joint_order`, `source_to_canonical_joints`
- legacy 이름 → canonical 정규화
- regex가 canonical joint를 `fullmatch`하는지 확인, group 간 중복/누락 검출

**검증:** `pytest scripts/r2s_autotune/tests/ -q` (Isaac 불필요, 순수 파싱)
- rh56f1: env_cfg 5개 group regex가 manifest canonical 이름을 빠짐없이 덮는가
- tesollo: 35개 control joint에 대해 canonical group 표를 새로 정의하고 동일 검증
- 추가로 Isaac articulation을 한 번 띄워 실제 DOF 이름을 덤프해 manifest와 대조 (Tesollo USD가 정말 canonical인지 미검증 상태)

### Phase 1 — 합성 real track + 로더

```
make_synthetic_track.py   # 알려진 gain으로 sim 구동 → HDF5 (teleop과 동일 schema)
load_real_track.py        # HDF5 → canonical 정렬 (manifest normalize)
excitation.py             # step / ramp / hold / return / slow sine 시퀀스 생성
```

`excitation.py`는 sim과 실물이 **같은 시퀀스**를 쓰도록 한 곳에 둔다. 관절 한계·속도 한계 clamp를 내장한다 (실물 안전).

**검증:** 합성 트랙 왕복 로드. `joint_names` attr 존재, rad 단위, `q_cmd`/`q_real` shape 일치.

### Phase 2 — Parallel replay + error + export (Algorithm 1 본체)

```
sample_candidates.py      # seed 중심 group 단위 scale 샘플링 (joint 독립 탐색 금지)
replay_env.py             # articulation-only 씬, K envs
run_parallel_replay.py    # env_ids별 gain 주입 → 동일 q_cmd replay
compute_tracking_error.py # 1.0*MSE(q) + 0.05*MSE(dq) + 0.01*delay_penalty
export_best_calibration.py# schema_version:1 JSON
```

**검증 (핵심): ground-truth 복원 테스트.**
합성 트랙 생성에 쓴 stiffness/damping을 autotune이 복원하는지 측정. 목표는 group 평균 상대오차 10% 이내. 실패 시 파이프라인 버그이며 real 데이터를 넣어도 의미 없다.

부차 검증: error가 후보 간 유의미하게 갈리는가 (가이드 §11.2의 "excitation 부족" 신호 조기 검출).

### Phase 3 — Real seed pipeline (실물 접근 가능 시점에)

teleop repo 쪽 신규 작업:
- **excitation 명령 노드** — 현재 `record_real2sim_identification_bag.sh`는 녹화만 한다. 명령을 주는 주체가 없어 그대로 돌리면 정지 데이터만 쌓인다. Phase 1의 `excitation.py`를 ROS2 퍼블리셔로 감싼다.
- **db3 → identification HDF5 변환기** — 현재 없음 (`db3_to_pour_mimic_hdf5.py`는 pour 전용).
- **multi-group merge script** — 가이드 §5가 남긴 미해결 항목. 현 스크립트는 1회 1 group.

**검증:** 실물 수집 → seed JSON. group 이름이 env cfg group과 정확히 일치. `fit_error`/`joint_metrics` 기록.

수집 순서는 가이드 §7.2를 따르되 arm과 hand를 분리한다 (§11.3: arm은 raw db3 500–1000 Hz, hand는 100 Hz).

### Phase 4 — RL env 적용

- `rh56f1/right/grasp_v2`에 calibration hook 이식 (v1의 `real2sim_actuator_cfg.py` 패턴 재사용, 새 loader 작성 금지)
- `delay_steps` 처리 결정: (a) 명시적으로 미지원 선언하고 JSON에서 drop, 또는 (b) action delay buffer 구현. **(a)를 권장** — 지금은 조용히 무시되어 오해를 부른다.
- actuator coverage test 확장 (rh56f1 38 DOF 유지 — `test_phase3_actuator_coverage.py`)

**검증:** `OPENARM_REAL2SIM_ACTUATOR_CALIBRATION` 설정 후 env 스폰 → 각 actuator group의 실제 gain이 JSON 값과 일치하는지 런타임 assert.

### Phase 5 — Validation 및 sim2real

- held-out task replay (`datasets/pour_v1_a*.hdf5`)로 최종 검증. identification 시퀀스 과적합 검출.
- best calibration을 중심으로 domain randomization 범위 재설정 (정합된 값 주변으로 좁힘)
- `sim2real/scripts/sim2real_inference.py` 경로로 정책 배포

**판정 기준(가이드 §10 Task 5):** same-sequence 오차 감소 + held-out 오차 감소 + task replay drift 감소 + contact timing이 과도하게 앞당겨지거나 늦어지지 않음.

---

## 5. 리스크

| 수준 | 리스크 | 대응 |
|---|---|---|
| HIGH | 합성 검증을 통과해도 real gap이 남는다. sim의 actuator 모델 구조 자체가 틀리면(마찰·백래시·직렬탄성) 어떤 gain으로도 복원 불가 | Phase 2 복원 테스트는 "파이프라인이 옳다"만 증명한다. 모델 구조 타당성은 Phase 3 real 데이터의 residual로 판정 |
| HIGH | Tesollo canonical group 표가 아직 없다. grasp_v11은 legacy 4파일 78토큰 | Phase 0에서 R2S용 group 표를 새로 정의. grasp_v11 마이그레이션은 D2에 따라 **이 작업 범위 밖**으로 분리 |
| MEDIUM | Tesollo USD가 실제로 canonical인지 런타임 미검증 (USD crate 바이너리라 grep 불가) | Phase 0의 DOF 덤프로 확인. 아니면 즉시 중단하고 사용자에게 보고 |
| MEDIUM | `delay_steps`가 loader에서 죽어 있다 | Phase 4에서 명시적 결정 |
| MEDIUM | excitation 명령을 실물에 주는 것은 안전 문제 | `excitation.py`에 관절/속도 한계 clamp 내장. 첫 실행은 사용자 입회 하에 저진폭 |
| LOW | per-env gain override | 검증 완료 — `write_joint_stiffness_to_sim`이 `env_ids` 지원 |

---

## 6. 복잡도

**HIGH.** Phase 0–2(합성 파이프라인)가 이번 작업의 본체이고, Phase 3은 실물 접근에 의존하며, Phase 4–5는 그 이후다.

신규 코드는 전부 `hdgp/scripts/r2s_autotune/` 아래에 격리된다. 기존 RL env는 Phase 4까지 건드리지 않는다.

---

## 7. 즉시 착수 범위 (승인 시)

Phase 0 + Phase 1 + Phase 2 = 합성 궤적으로 end-to-end 동작하고 ground-truth를 복원하는 autotune 파이프라인.
real 데이터가 도착하면 `configs/*_replay.yaml`의 HDF5 경로만 바꿔 그대로 재사용한다.
