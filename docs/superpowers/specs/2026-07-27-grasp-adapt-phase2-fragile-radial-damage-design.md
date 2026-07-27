# Phase 2 — Stage A Fragile: radial-압축 virtual damage 설계

> 상위 계획: `docs/superpowers/plans/2026-07-27-grasp-adapt-fingertip-fragile-rebuild.md` (Phase 2).
> 프로젝트 규칙: `hdgp/CLAUDE.md`(로그 먼저, reward-audit, 실험 루프) + `grasp_adapt/CLAUDE.md`.

## 배경 / 동기

Phase 1(팁 precision 파지)에서 **geometry 기반 envelope penalty(middle 접촉 벌점, w=4)로는 손끝-only 강제에 실패**했다. `lstm_test3` 결과: 초반 억제(middle_contact_rate 0.11)됐다가 후반 감싸기로 회귀(middle 0.18·five_tip 0.77·success 0.5 진동). 근본원인은 **secure(컵 slip_speed 기반) 보상이 감싸기에 구조적으로 유리**해서, penalty로 상쇄하려는 것이 밑 빠진 독이었기 때문이다. reward-audit의 Check 3(secure 역방향) 경고가 실증됐다.

**방향 전환(사용자 판정):** 손끝-only를 인위적 penalty로 강제하지 않고, **fragile 물리(감싸 쥐면 컵 벽이 좌굴=파손)로 자연 유도**한다. 종이컵은 사방에서 안으로 압박하면(감싸기) radial 압축 좌굴이 나고, 손끝으로 위·국부를 집으면 radial 압축이 작다. 즉 radial 압축 damage가 감싸기를 물리적으로 벌하고 손끝 파지를 남긴다. 이때 손끝 6축 F/T가 비로소 reward의 핵심이 된다(설계 문서의 본래 의도).

## 목표

Stage A(rigid cup + 힘 기반 가상 damage)에서:
1. radial 압축 damage를 도입해 **손끝-only 파지가 reward 구조에서 자연히 나오게** 한다(별도 geometry penalty 없이).
2. fragile 파손(좌굴)을 명시적으로 모델링해 안전 파지력 구간을 학습한다.
3. Phase 1 envelope penalty를 제거하고 radial 신호로 일원화한다.

## 설계

### 1. radial 압축력 계산 (핵심 신호)

컵 중심축(연직, `object_rot`로 회전된 z) 기준, 각 접촉력 벡터의 **컵 중심을 향하는 수평(inward) 성분**을 합산한다.

```
컵 up축:        z_cup = quat_apply(object_rot, [0,0,1])              # (N,3)
접촉점 c_i, 힘 f_i (world):
  radial_out_i = normalize( (c_i - object_pos) - proj_onto(z_cup) )  # 축에 수직인 바깥방향
  radial_inward_i = relu( -(f_i · radial_out_i) )                    # 안으로 미는 성분(양수)
radial_compression = Σ_i radial_inward_i
```

포함 접촉:
- **손끝(tip)**: `contact_force_xyz_raw` (N,5,3, world, Cup 필터) + `fingertip_pos` — 즉시 사용 가능.
- **중간/원위 마디(middle/distal)**: `middle_sensor`/`distal_sensor`의 `net_forces_w` (N,5,3 벡터) + 각 body 위치(FK, 구현 시 body pos 배선 필요). **감싸기의 radial 압박은 주로 중간마디에서 나오므로 middle 포함이 손끝 vs 감싸기 구분의 핵심.**

> **열린 구현 이슈 (구현 계획에서 해결):**
> (a) middle/distal `net_forces_w`는 Cup 필터가 없어 로봇 자기접촉이 섞일 수 있다 → 접촉 binary(이미 Cup 근접 판정)로 마스킹하거나 크기 임계로 걸러낸다.
> (b) middle/distal body 위치를 body_pos에서 조회하도록 배선(현재 critic obs `middle_to_cup`가 middle body pos 사용 중 → 재활용 가능).
> (c) 1차 구현은 tip+middle radial로 시작하고, distal은 진단 로깅부터(손끝 인접이라 회색지대) — 데이터 보고 포함 결정.

### 2. reward / 종료

```
r_damage = -damage_penalty_weight · hold_gate · relu(radial_compression − F_safe)   # 초과분 순간 penalty
buckle   = radial_compression > F_buckle                                            # 좌굴(파손)
```
- `buckle` 발생 시 **에피소드 종료**(파손) + success 배제 + 음의 종료 보상(`buckle_penalty`).
- 파손 종료는 `_get_dones`의 `terminated` 조립에 `buckle` OR 추가.
- **envelope_penalty_weight = 0** (Phase 1 penalty 제거, radial로 일원화). cfg 필드는 존치하되 0.

### 3. 물성값 (placeholder, 실측 전)

- `F_safe`, `F_buckle`은 종이컵 추정 placeholder로 시작(설계 문서 §4: `F_safe ≈ 0.6~0.8·F_yield`). 초기값은 학습 초기 radial_compression 분포를 보고 보정(로그 먼저).
- **파손 종료 커리큘럼:** 학습 초기 잦은 파손이 gradient를 죽이지 않도록, `F_buckle`을 ADR로 느슨→엄격 점진 강화(또는 hold 진입 후에만 buckle 판정). 초기엔 penalty 위주, 종료는 완화.
- Stage 4에서 `F_safe`/`F_buckle` randomization(후속 Phase).

### 4. 로깅 / 검증 (Phase 2 exit 기준)

신규 TFEvents 태그:
- `task/radial_compression` (평균), `task/buckle_rate` (파손 종료율), `reward/damage`
- 기존 유지: `task/middle_contact_rate`, `task/distal_contact_rate`, `task/success_rate`, `contact/count`, `task/five_tip_contact_rate`

**Phase 3 진입 exit 기준:**
- **손끝-only 확립:** `middle_contact_rate` 낮게 수렴 + `radial_compression`이 `F_safe` 아래로 수렴(감싸기 억제).
- **파지·리프트 성립:** `success_rate` 유의미(>0.5) + `cup/height_delta` 10cm + `buckle_rate` 낮음(<몇 %).
- `play.py` 육안: 손끝으로 집고 컵을 안 찌그러뜨리는지.
- **실패 시:** radial이 안 내려가면 `damage_penalty_weight`↑ / `F_safe`↓. 손끝만으로 못 들면(success 붕괴) 손끝 파지가 물리적으로 가능한지 재검토(컵 무게·마찰·크기) — 이 경우 물성/물체 조정이 선행.

## 아키텍처 / 파일 영향

- `grasp_right_env.py` `_get_rewards`: radial_compression 계산 + `r_damage` + total 반영. envelope penalty 제거. 로깅 추가.
- `grasp_right_env.py` `_get_dones`: `buckle` 종료 조건 추가.
- `grasp_right_env.py` `_update_contact_forces` 또는 신규 헬퍼: radial 계산(tip+middle force→inward 성분). middle/distal body pos 배선.
- `grasp_right_env_cfg.py`: `damage_penalty_weight`, `F_safe`, `F_buckle`, `buckle_penalty` 신규. `envelope_penalty_weight=0`. (파손 종료 커리큘럼 ADR 필드).
- `grasp_right_utils.py` 또는 신규 모듈: radial 압축 순수 함수(테스트 가능하게).
- `tests/`: radial 압축 계산 순수 함수 계약 테스트(감싸기 형태 입력→radial↑, 손끝 형태→radial↓).
- **차원 변경 없음**(reward/gate만). 단 fresh 재학습(성공/종료 조건 변경).

## 리스크

- **radial 계산이 감싸기/손끝을 실제로 구분하는가:** 순수 함수 테스트 + 초기 학습 radial 분포로 검증. middle 포함이 관건.
- **손끝만으로 물리적 파지 가능성:** 안 되면 어떤 damage 모델도 success를 못 살림 → 물성/물체 선행 조정.
- **파손 종료의 초기 학습 방해:** 커리큘럼으로 완화, hold 진입 후 판정.
- **middle/distal net force 오염(자기접촉):** 접촉 binary 마스킹으로 완화.

## YAGNI (이번 범위 밖)

- 누적 damage dose(설계 §6) / Lagrangian constraint → Phase 5.
- segmented compliant shell(Stage B) / FEM(Stage C) → Phase 6.
- tactile residual adapter(base freeze) → Phase 3.
- 실제 종이컵 물성 측정(물리 실험) → placeholder를 실측으로 교체하는 후속.
