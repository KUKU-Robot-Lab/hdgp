# rh56f1 grasp — pour용 robust cup hold 실험 계획

**Goal:** grasp가 컵을 rigid하게(안 굴러가게) 잡아 pour의 두 외란(bead 무게↑, pour 회전)에 안 풀리도록 만든다. pour 학습이 이 grasp 위에 올라간다.

**진단 요약 (lstm_test1~3):**
- 파지·리프트는 잘 학습(tilt 9°, lifted 46%), 성공조건 grip-based fix로 success_rate 안정(0.2).
- **막힌 것: 컵이 그립 안에서 굴러(cup_ang_vel 1.0~1.8 > 임계 0.5) held=0.**
- firm 손가락(tip&근위) ~1.3 정체 = 물리 아니라 **"컵을 엄지-손바닥 사이로 넣는 탐색"이 어려움**(사용자 확인). mimic 원위는 컵 반력에 멈춤(정상 물리) — 힘/강성으로 못 고침.
- → geometry(컵 크기)는 이미 scale 0.9, 더 못 줄임 → **reward + 커리큘럼**으로 해결.

## Global Constraints
- **obs/action 차원 불변** (명시 요청 없이 금지).
- **reward/gate 변경은 reward-audit 통과 후에만.**
- **한 실험 = 한 가설** (격리 검증). 각 실험 후 keep/revert 판정 뒤 다음.
- **log-first**: TFEvents 근거로 판정.
- **Tesollo 무영향**: 공유 `grasp_reward_core.py`는 cfg 기본값(0/무효)으로 rh56f1만 활성.
- baseline: lstm_test3 (커밋 82374c1, grip-based 성공 + tip&근위 firm 재구성).

---

## Experiment 1 (lstm_test4): Anti-roll 전용 페널티 (#2-A)

**가설:** hold 중 cup_ang_vel을 직접 페널티하면 정책이 "안 굴러가는 grip/자세"를 찾아 → ang_vel↓ → stable→held 발화.

**Files:**
- Modify: `source/openarm/openarm/common/grasp_reward_core.py` — cfg-gated 항 추가
- Modify: `source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env.py` — cup_ang_vel_norm 전달
- Modify: `.../grasp_v1/grasp_right_env_cfg.py` — weight cfg

**변경:**
- reward_core: `roll_penalty = -_cfg_float(cfg,"cup_ang_vel_penalty_weight",0.0) * stabilize_gate * lifted_gate * cup_ang_vel_norm` 추가(terms에 편입). cup_ang_vel_norm은 env에서 인자로 전달(기존 stability.cup_ang_vel_norm 재사용).
- cfg: `cup_ang_vel_penalty_weight: float = 0.5` (rh56f1만; tesollo 기본 0).
- action_delta 오염 없는 순수 회전 페널티(기존 stability_quality의 노이즈 바닥 문제 회피).

**reward-audit:** 회전 억제는 rigid hold 목표와 정합. hacking 없음(컵 들고+접촉+정지 요구). local-min 없음(회전만 처벌). Check 통과 예상 → 구현 전 정식 audit.

**검증:** 정적 테스트(test_phase4_env_static, test_v7_2_reward_contract, test_grasp_v2_contract) PASS → GPU 학습 → 모니터.
**성공 기준:** `cup_ang_vel` 1.0+→0.5 근처↓, `stable_rate`↑, **`success_held_rate` 0→>0**.
**분기:** held 발화 시 keep(Exp2로). ang_vel 안 내려가면 weight↑ 또는 물리 한계 재판정.

---

## Experiment 2 (lstm_test5): 잡고-나서-들기 firm lift 게이트 (#3-1)

**가설:** 리프트를 firm 그립 형성 후로 지연하면, 정책이 **먼저 컵을 엄지-손바닥 사이에 넣어 firm grip을 만들도록 강제** → 리프트 shortcut 차단 → firm 그립↑ → 굴림↓.

**Files:**
- Modify: `.../grasp_v1/grasp_right_env.py` — lift-start 조건(`_lift_contact_ready_latched_buf` 산출부, L1098 부근)
- Modify: `.../grasp_v1/grasp_right_env_cfg.py`

**변경:**
- lift-start를 `num_firm_fingers(tip&근위)>=lift_start_min_firm_fingers & 엄지접촉`이 `lift_start_firm_hold_steps` 유지된 후로 게이트.
- cfg: `lift_start_min_firm_fingers: int=2`, `lift_start_firm_hold_steps: int=10`, `lift_start_timeout_steps: int=120`(firm 미형성 시 fallback 허용 — dead episode 방지).

**reward-audit:** 게이트(상태전이) 변경. lift 보상 경로 자체는 불변, 진입 조건만 강화. hacking 없음. timeout fallback으로 학습 교착 방지. → audit.

**검증:** 정적 PASS → GPU 학습 → 모니터.
**성공 기준:** lift-start 시점 `num_firm_fingers`↑(≥2), 그립 firmer, `cup_ang_vel`↓, held 유지/개선. Exp1과 병행 시 시너지.
**리스크:** firm 형성이 어려워 timeout fallback만 계속 → firm 개선 미미. 그럼 Exp3(warm-start)로.

---

## Experiment 3 (lstm_test6): Warm-start from firm grasp (#3-2)

**가설:** 컵이 이미 엄지-손바닥 사이 firm grip 상태로 시작하는 초기상태를 일부 주입하면, 어려운 삽입 탐색을 건너뛰고 "유지+리프트"부터 학습 → firm grip + held 도달, 이후 cold env로 일반화.

**Files:**
- 신규/수정: warm-state **import/inject** 인프라 (rh56f1 grasp_v1은 현재 **export만** 존재).
  - `.../grasp_v1/warm_state_cache.py` 또는 reset 경로에 warm-state 로드·주입 추가 (tesollo 방식 참조).
  - collect: Exp1/2 성공 run 또는 스크립트로 firm-grasp 성공상태 수집(기존 export 활용) → hdf5.
- Modify: `.../grasp_v1/grasp_right_env_cfg.py` — `warm_state_import_path`, `warm_state_inject_frac`(예: 0.5).

**검증:** 정적(warm import 로드 테스트) → GPU 학습 → 모니터.
**성공 기준:** warm env에서 firm grip+held 조기 발화 → cold env로 held 전파. 전체 held_rate↑.
**리스크:** import 인프라 신규(공수 큼). warm 성공상태 수집이 선행(Exp1/2 성공 run 필요). → Exp1/2에서 firm/held가 조금이라도 나오면 그 상태를 warm 소스로.

---

## Experiment 4 (lstm_test7): 회전 외란 커리큘럼 (#3-3, pour 직결)

**가설:** hold 중 컵에 pour-like 회전/틸트를 실제로 가하고 그립이 풀리면(접촉 상실·slip) 페널티하면, **회전에 안 놓는 grip**을 직접 단련 → pour 회전에 강건.

**Files:**
- Modify: `.../grasp_v1/grasp_right_env.py` — stabilize phase에서 컵에 외부 각속도/토크 인가(root_ang_vel 또는 external wrench) + 접촉 상실 페널티.
- Modify: cfg — `hold_rotation_perturb_enabled`, `hold_rotation_perturb_mag`, ADR 진행(점진 강화).

**reward-audit:** slip 페널티 항. hacking 없음(접촉 유지 요구). ADR로 점진 도입(급격한 외란이 학습 붕괴 방지). → audit.

**검증:** 정적 PASS → GPU 학습 → 모니터.
**성공 기준:** 외란(회전) 인가 중 `contact/count`·`num_firm_fingers` 유지(slip 최소), held 유지. = pour 회전 강건성 확보.
**리스크:** 외란이 세면 grip 붕괴 → ADR로 약하게 시작.

---

## 무게 외란 (bead) — 신규 실험 아님, 점검만
- 기존 hidden-mass DR(컵 실효질량 170~470g, actor 미관측) + force gate로 이미 훈련됨.
- **점검:** pour 실제 최대 bead 하중이 470g 초과면 `bead_count_max`/`cup_base_mass` DR 범위 확장.

---

## 실행 순서·판정
1. **Exp1(anti-roll)** — 단일 reward 변경, 격리 검증. held 발화 여부.
2. **Exp2(firm lift 게이트)** — 삽입 학습 압박. (Exp1 keep 위에)
3. **Exp3(warm-start)** — firm이 여전히 어려우면 탐색 우회. (Exp1/2 성공상태를 warm 소스로)
4. **Exp4(회전 외란)** — pour 강건성 마감.

각 실험: 구현 → (reward/gate면) reward-audit → 정적 테스트 → GPU 학습(train.sh, -lstm) → parse_tfevents 모니터 → keep/revert 판정 → 다음.

**최종 성공 기준(전체):** `success_held_rate > 0` 달성 + hold 중 회전 외란에도 `num_firm_fingers`·접촉 유지 = pour가 컵을 rigid하게 잡고 기울일 수 있는 grip.
