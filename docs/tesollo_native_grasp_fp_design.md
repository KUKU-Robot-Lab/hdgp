# Tesollo-Native Grasping + FoundationPose 배포 — 설계 문서

> 상태: §3.4 관측(pos-only) + §3.1 접근 자세(side-to-side 기본) 구현 완료(커밋 1345fbe, 서버 정적테스트 62 pass). **학습 미기동** — reward/H1(5지 관여)은 §5 재검증 대기. 방향 전환 결정 근거는 memory `dextrah-unsuitable-tesollo-pivot.md`.
> 대상: `source/openarm/openarm/tesollo/right/grasp_v2` 재설계 → Tesollo-native 코어 복원 + FP 배포.

---

## 0. 요약

DEXTRAH 적응을 중단하고, **Tesollo 하드웨어에 맞는 grasp_v1 계보(접촉-게이트 폐쇄 제어 + staged 보상)**를 다물체 스캐폴딩 위에서 재구성한다. **vision distillation 대신 FoundationPose로 teacher를 직접 배포**한다. 이후 양팔로 확장.

---

## 1. 배경: 왜 전환하는가 (DEXTRAH 부적합 5대 실증, 07.20)

| # | 축 | 실증 |
|---|---|---|
| 1 | 제어 | ~~synergy-position → 3지 파지~~ **[07-20 정정, 아래 참조]** |
| 2 | 보상 | opposition force_closure → 약한 hedge 파지. grasp_v1 staged(stabilize/hold)는 firm |
| 3 | 증류 | teacher→vision student 정체(window BPTT +14%, clean eval 0.132→0.150). enveloping=near-total occlusion으로 DEXTRAH tip-grasp보다 불리 |
| 4 | 센싱 | teacher **actor obs 전부 FP+FK+tip F/T+fabric로 실기 취득 가능** → distillation 불필요, 직접배포 |
| 5 | ADR | 3cm over-harden → grip 약화(ADR15 grip 1.86→ADR50 1.34). FP 저노이즈(1-2cm)엔 과잉 |

→ DEXTRAH 3대 축(vision distillation·high-ADR) 중 최소 2개는 우리 센싱/형태에 안 맞는다. 제어 축은 아래 정정 참조.

### ★ 07-20 정정: 실증 #1("synergy 제어가 원인")은 이 teacher에 대해 틀림

분석 대상 teacher(`lstm_test1`, 07-17 학습 시작)의 서버 실측 `params/env.yaml`을 직접 확인한 결과 **이미 `finger_control_mode: per_finger`로 학습돼 있었다**(커밋 `52e0fb9`, 07-14에 grasp_v2 기본값이 synergy→per_finger로 전환됨, lstm_test1은 그 3일 뒤 시작). 즉 "PCA가 20관절을 커플링해 3지 파지가 난다"는 설명은 **이 체크포인트엔 적용되지 않는다** — per_finger로도 동일하게 3지 파지가 나왔다.

middle/ring tip 접촉 0.00 자체는 손가락 길이 때문에 tip 전에 distal/middle이 먼저 닿는 **기하 구조 문제로 여전히 유효**(제어 방식과 무관).

**진짜 미검증 변수**는 per_finger 안에 이미 존재하는 `synergy_freeze_enable` 플래그(접촉 시 해당 관절만 동결하는 grasp_v1식 게이트)다. lstm_test1은 이 값이 `false`(동결 없이 계속 조임)로 학습됐다. 마지막으로 `true`였던 건 07-08(`cd91f22`) 단 한 번뿐인데, 그때는 per_finger도, 힘-크기 기반 force_closure(tip+distal+middle 합산)도, 148물체·cup 스케일 수정도 전부 없던 시절이라 "동결→파지력 0.90 정체"로 되돌려졌다. **현재 조합(per_finger + 힘-기반 force_closure + 148물체)에서 동결을 켜본 적은 없다.** → §3.1·§4·§5 재기술.

---

## 2. 설계 원칙

1. **하드웨어 정합 우선** — Tesollo 5지 enveloping을 제어가 담당(접촉까지 감싸기), 보상은 목표/단계만.
2. **배포 가능한 관측만** — teacher actor obs = FP(object pose) + FK(hand) + tip F/T(tactile) + fabric + proprio. privileged는 critic-only.
3. **배포 노이즈에 맞춘 ADR** — FP 정확도(~1-2cm)를 상한으로. over-harden 금지.
4. **정적 물체 가정 활용** — FP로 폐색 전 pose lock → in-place 파지 → lift 운동학. 단 파지가 물체를 밀지 않아야 lock 유효(in-place 학습 필수).

---

## 3. 아키텍처

### 3.1 손가락 제어 (핵심 변경 — 재기술)
**[정정] per_finger(손가락별 독립 진행도 제어)는 grasp_v2에 이미 기본값으로 존재한다(커밋 52e0fb9). "이식"할 대상이 아니다.** 남은 건 grasp_v1의 **관절별 접촉-게이트 동결**(`synergy_freeze_enable` 플래그, 이미 코드에 있으나 현재 `False`)을 현재 조합 위에서 `True`로 켜 재검증하는 것:
- `_1`(외전)·`_2`(MCP): 무게이트 full close(근위 마디 밀착).
- `_3`(PIP): **middle 마디 접촉 시 동결**.
- `_4`(DIP): **distal|tip 접촉 시 동결**.
- `advance = finger_close_speed · cmd · (1 − gate)` → 각 손가락이 **닿을 때까지** 독립 폐쇄 → 물체 형상에 드리움(envelope). action = 손가락별 폐쇄 속도 5D.
- 팔: Fabrics IK palm 6D(유지). thumb opposition/abduction 축은 유지·정합.
- **리스크(reward-audit Check 1)**: 동결이 "닿기만 하고 안 누르기" 국소최적으로 샐 위험(07-08 grip 0.90 정체 전례). 완화 요인: 현재 force_closure_reward는 접촉이 아닌 힘 **크기**를 요구(tanh(force/scale)) — 07-08엔 없던 항. 단 미검증이므로 격리 실험 필수(§5 Stage 2). lstm_test1~3 실측(07-20~21)으로 3지(엄지+2) 국소최적을 fc_gate 강화(`force_closure_min_others` ADR 1→3)로 대응 시작 — H1으로 §5에 재편입.

**[08-21 추가] 접근 자세도 side-to-side가 기본** — grasp_v1 원형(`enclosure_axis`: 접근 방향에 수직인 축을 잡아 엄지 vs 4지가 물체를 사이에 두고 양옆에서 마주 조임)을 전 물체 기본으로 복원(DEXTRAH top-down 하강 접근은 기본에서 제외, 코드는 남김). `side_approach_object_names` 기본값을 cup 등 예외 목록 → 활성 물체 전체로 변경(구현 완료, 커밋 1345fbe).

### 3.2 리프트 트리거
grasp_v1 **접촉 latch**(`compute_lift_readiness`, grasp_v2엔 dead code로 존재): tip≥N + envelope 손가락≥M → 리프트 진입. step-480 scripted 대체 → **감싸 잡으면 리프트**(herding 완화).

### 3.3 보상 (staged, grasp_v1 core `compute_grasp_reward_terms`)
approach(2) → grasp(12) → lift(30) → **stabilize(10)+stability(1)+post_lift_contact_loss(−8)** → success_bonus(20), enclosure_thumb(0.6). 접촉 게이팅은 tip+envelope 혼합(`lift_envelope_mix`)으로 tip-only 억제. **stabilize+post_lift_contact_loss가 "가만히 버티기"를 직접 보상**.

### 3.4 관측 — [08-21 확정·구현 완료] 물체 pos만, identity/scale/rotation 없음
사용자 확인: "obs에는 물체의 pos 정보만 있으면 됨. 어떤 물체인지는 중요하지 않음." §0의 "알려진 물체 배포"에서 한 단계 더 나아가 **미학습 신규 물체 일반화**로 목표를 상향 — FP는 CAD/mesh만 있으면 신규 물체도 pose를 주지만, bbox 같은 형상 특징을 실시간 obs로 안정적으로 공급하는 건 별도 파이프라인이라 s2r 가정이 약하다(초기 설계 오판, 정정됨).
- **actor(teacher) obs**: proprio + hand FK(palm+5tip) + **object_pos(FP-deployable, 위치만)** + goal + tactile(tip F/T 3축) + fabric q/qd/qdd + actions. **object identity(onehot)·scale·rotation 없음.** `NUM_OBS_BASE=208`, 물체 수 무관(고정) — 구현 완료(`grasp_right_env.py`/`_env_cfg.py`/`_constants.py`, 커밋 1345fbe).
- 근거: 인벨롭 그립(접촉-게이트 폐쇄)은 목표를 FULL_GRIP까지 밀면 닿거나 포화된 관절이 멈추는 **형태-적응** 제어라, 물체가 어떤 모양/크기인지 정책이 몰라도 물리(환경의 실시간 접촉 게이트)가 감싼다. "존재를 알고(pos) 다가가서 손가락을 닫으면 인벨롭된다"가 핵심 원리 — 정책 obs에 형상 정보를 넣는 순간 오히려 물체별 지름길 암기를 유발해 일반화를 저해한다.
- critic(privileged, 비대칭 actor-critic — 배포는 actor만 씀): object_rot, object_vel, onehot, scale, distal/mid contact, torque. 학습 안정화용으로 유지.

### 3.5 배포 (FoundationPose, distillation 없음)
- 고정 **외부 global 카메라** + **일회성 camera→base 외참**(hand-eye 아님).
- 물체 정적: FP로 가시 구간에 base-frame pose **lock** → in-place 파지(폐색 무관) → lift는 `object=hand∘상대` 운동학.
- 상대 feature는 lock pose + live FK로 폐색 중 재구성.
- tip F/T로 접촉/slip 감지(파지 중 물체 밀림 보완).
- **vision student·depth encoder 없음.**

### 3.6 학습
다물체 148 스폰(유지) · **ADR 상한 ~25(FP 노이즈 정합, over-harden 방지)** · LSTM · warmstart.

#### ★ ADR 건강성 기준 (학습 모니터링 게이트)
ADR 증가 시 전체 reward는 각 increment 직후 **일시적 dip 또는 plateau는 허용**되나, **bump을 넘어 지속적 하락 추세가 되면 안 된다.** 하락 추세 = ADR이 정책 용량을 앞질러 over-harden(정책이 hedge/약한 파지로 퇴화) = **실패 신호**.
- 건강: 각 ADR bump 후 steady-state reward가 이전 수준으로 **회복**한 뒤 다음 increment.
- 실패: steady-state reward(및 in_success·grip)가 bump 넘어 **단조 하락**.
- 메커니즘: ADR은 in_success>0.4 **순간값** 트리거로 증가 → 순간값이 잠깐 넘으면 steady-state가 낮아도 계속 증가 → over-harden.
- 대응: 하락 추세 감지 시 **ADR 상한 하향 또는 increment 속도 완화**(트리거를 순간값 대신 이동평균으로).
- 이력: 구 DEXTRAH run(lstm_test1)은 reward 2197(ep6500)→1042(ep15000) **단조하락 = 이 실패**였고, "ADR이라 정상"으로 오독됐다. 재발 방지.

### 3.7 양팔 확장
`both/` 구조로 확장(기존 `both/pour_sensor` 패턴 참조). 초기엔 각 팔 독립 파지, 협조 파지는 후순위.

---

## 4. 이식 범위 (grasp_v1 → 새 grasp env)

| 컴포넌트 | 출처 | 비고 |
|---|---|---|
| per-finger 독립 제어 | **grasp_v2 이미 기본값**(`52e0fb9`) | 이식 불필요, 재확인만 |
| 접촉-게이트 동결 | grasp_v2 `synergy_freeze_enable` 플래그(현 False, grasp_v1식 게이트 이미 구현됨) | `True`로 전환 + 격리 재검증 |
| 접촉 latch 리프트 | grasp_v1 + `compute_lift_readiness`(grasp_v2 dead) | 활성화 |
| staged 보상 | `common/grasp_reward_core.py` | grasp_v1 weight 스킴 |
| 다물체 스폰·obs·ADR·Fabrics·LSTM | grasp_v2 유지 | — |
| FP obs 채널·운동학 브리지 | 신규 | 배포/sim DR |

---

## 5. 구현 단계

1. **reward-audit** — 완료(07-20). 결론: per_finger는 이미 기본값, 미검증 변수는 `synergy_freeze_enable` 단일 플래그. 판정 REVISE(격리 실험 선행).
2. **`synergy_freeze_enable=True` 단독 파일럿** — 현재 baseline(per_finger+148물체+cup수정+force_closure) 위에서 이 플래그 하나만 변경, 다른 reward/제어는 불변. contact/grip·contact/fc_gate_frac·drift 모니터링.
3. (파일럿 결과가 개선 확인 시) **reward core 이식** — grasp_v1 staged(stabilize/post_lift_contact_loss).
4. **lift latch 활성화** — `compute_lift_readiness` 배선(현재 dead code).
5. 정적 테스트(pytest) → 소수 물체 학습 검증.
6. **다물체 재학습**(capped ADR ~25).
7. **in-place 검증**(eval_pose_hold: drift↓, FP-lock 성립 확인).
8. **FP 배포 파이프라인**(실기, 병렬): 외부카메라·외참·FP→base→obs.

---

## 6. 리스크 / 열린 질문

- **다물체 × 접촉게이트 미검증** — grasp_v1은 소수 물체. ADR·다물체 상호작용 확인.
- **action 의미 변경** → 정책 재학습 필수(기존 체크포인트 무효).
- **파지 중 물체 밀림 잔여** — 접촉게이트 폐쇄가 herding을 줄이나 0은 아님 → tip F/T 보완, in-place 지표로 검증.
- **cup(side approach) 여전히 난이도** — 우선 top-down 물체군.
- **양팔 동시 파지 협조** — 범위 밖, 후순위.
- **FP mesh 의존** — 알려진 물체 한정(open-set 아님).
