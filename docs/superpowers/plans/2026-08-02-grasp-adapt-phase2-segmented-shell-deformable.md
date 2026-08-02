# grasp_adapt Phase 2 — Segmented-Shell Deformable Cup (진짜 fragile 적응)

## 목적 / 근거

현 정책은 **과파지(power-grip)** 로 수렴함(실측: 정적 base에서 빈 컵 grip_ratio 3.0, 무거운 컵 1.5 = 절대 파지력 거의 일정, 무게 176%↑에 파지력 38%만↑). 원인:
- tactile-only라 질량 모름 → worst-case(무거움)에 맞춰 firm하게 쥠.
- **과파지가 "공짜"**: radial ~1.4 < f_safe 3 → damage 페널티 미발동. efficient(w2.0)·friction DR(0.15~0.60)도 max-grip을 못 막음(간접유인 한계 실증).

**해결 원리**: 컵을 **실제 변형하는 물체**로 만들어 **과파지 = 실제 crush(실패)** 로 만든다. 그러면 과파지(부숨)도 under-grip(떨굼)도 불가 → 정책이 **"딱 필요한 만큼"의 적응 파지**를 강제로 학습. sim에서 변형/온전을 **영상으로 시연** 가능.

**FEM 아님**: 2048-env 실시간 소프트바디는 비현실적. 대신 **articulated 근사** — rigid 패널 + compliant(스프링) 조인트. "변형"=조인트 각, "좌굴"=각 임계 초과.

---

## 설계

### A. Segmented-cup USD 자산 (가장 큰 리스크, 먼저)
신규 스크립트 `scripts/assets_tools/generate_deformable_cup.py` (pxr — 로컬 동작 확인):
- **base**: rigid 바닥 디스크 + 하단 링(articulation root). 컵 바닥 = bead가 얹히는 곳.
- **N 패널**(시작 12, 대안 8): 원통 벽을 각도분할한 rigid 세그먼트. 각 패널 = base 링에 **세로 hinge(revolute) 조인트**로 연결, **drive stiffness(스프링)+damping** 부여 → 안으로 눌리면 각 발생, 스프링이 복원.
- **collision**: 패널당 box/convex collider(얇은 벽). **contact sensor 활성**.
- **출력**: Articulation USD(articulation_enabled=True). scale/치수는 현 cup_big(반경 ~0.045)과 일치.
- **검증 게이트 A**: 1-env 스폰 → 안쪽 힘 인가 → 패널 각 발생·스프링 복원·리셋 시 각 0 복귀 확인(probe 스크립트).

### B. env 재작업 (단일-rigid 가정 지점 — 실측 완료)
파일: `grasp_right_env.py`, `grasp_right_env_cfg.py`.
| 지점 | 현재 | 변경 |
|---|---|---|
| cfg `cup_cfg` (485) | RigidObjectCfg, articulation off | ArticulationCfg, joints+drive |
| `self.cup=RigidObject` (527) | RigidObject | Articulation |
| `_CUP_FILTER` (536) | 단일 prim | per-panel glob + 패널별 접촉 집계 |
| `object_pos/rot` (904) | root_pos/quat | base 링 pose(또는 패널 centroid) |
| `cup_slip_speed` (1199) | root_lin_vel | base 링 lin_vel |
| `compute_radial_compression` (1243) | 힘 proxy | **실제 패널 각(deformation) 기반으로 대체/병용** |
| reset (1635) | write_root_state | articulation reset(조인트각 0=미변형) + stiffness 세팅 |
| friction (1654) | 단일 material | 패널별 material |
| beads | 단일 컵 안 | segmented 컵 base에 얹힘(collision 유지) |

### C. 변형 기반 damage 보상 (핵심)
- `deformation` = 패널 hinge 각의 함수(안쪽 편향 합 / 직경 감소량). **힘 proxy 아니라 실측 기하**.
- `r_damage = -w · relu(deformation - d_safe)` 연속(과파지=실제 편향=벌점).
- `buckle`: deformation > d_buckle → 종료 + 큰 벌점(컵 좌굴=실패).
- 성공조건: 기존 `damage_dose` 메커니즘을 **실제 deformation 누적**으로 재정의(`dose = Σ dt·relu((deform-d_safe)/d_safe)^q`), success에 `dose < max` 유지.
- 효과: 과파지→실제 편향→벌점 → **최소-힘 적응 강제**.

### D. Stiffness 커리큘럼 (ADR)
- 초기 stiffness **높음**(rigid-like, 거의 안 변형) → grasp/lift 먼저 안정 학습.
- 점진 **낮춤**(물렁, 과파지=crush) → gentle 적응 강제. 기존 `GraspADR` 스케줄러 재사용.
- rigid-stiffness 설정 시 현 base와 동등 거동 → 회귀 검증 기준.

---

## 단계 / 검증 게이트

1. **Gate A — 자산**: generate_deformable_cup.py + probe로 1-env 변형/복원/리셋 확인. (자산만, 학습 무관)
   - ✅ **완료(2026-08-03)**: `assets/cup/deformable_cup.usd`(12패널·base 1 = 13 body·12 revolute), `scripts/probes/probe_deformable_cup.py`.
   - 결과: rest 0.000° / 0.3N 안쪽 힘 12.621°(해석예측 torque/K=12.6°와 일치) / 복원 0.000° / 리셋 0.000° = **PASS**.
   - **핵심 발견**: 경량 패널(0.0004kg)은 **armature(1e-3 kg·m²) 없으면 즉시 NaN 폭주**. 학습에서도 컵 관절 actuator에 armature+solver 32iter 필수(Gate B/D 반영). 패널 관절은 정책이 제어 안 하는 수동 스프링(ImplicitActuatorCfg target=default 0).
2. **Gate B — env 통합(rigid-stiffness)**: segmented 컵으로 env 구성, obs/action 차원 불변, **높은 stiffness에서 현 base와 동등 파지** 재현(회귀). 정적 pytest + AST.
2b. ✅ **Gate B 완료(2026-08-03)**: `cup_is_articulated` 플래그 + `GraspRightEnvCfgDeformable`(cup_cfg=ArticulationCfg, actuator stiffness 5.0 rigid-like·armature 1e-3) + env `_setup_scene`/contact-filter/reset 분기. 태스크 등록 `open-tesol_r_grasp_adapt_deform-lstm`(+play). 검증: pytest 24 pass, 스모크(16env·60step) obs 133/act 27 불변·NaN 0·max 패널각 7.9°(고강성 near-rigid) = **PASS**. 기존 rigid 태스크 무손상.

3. **Gate C — deformation 보상 배선**: r_damage/dose를 실제 각 기반으로, reward-audit 통과. 로컬 smoke로 변형 지표 로깅 확인.
   - ✅ **완료(2026-08-03)**: `compute_panel_deformation_deg`(utils, max|힌지각|deg) → articulated일 때 `radial_compression`을 이 값으로 공급(env ~L1254). 하류 r_damage/dose/buckle/success 무변경. Deformable cfg에 deg 단위 override(f_safe 10°·f_buckle 35°·damage_penalty_weight 0.3). **reward-audit ACCEPT**(신규 태스크·플래그 게이트·penalty만·측정가능). pytest 29 pass(+5), py_compile OK.
   - 스모크(stiffness 0.15, 80step): radial=14.3°(>f_safe) 정상 유입·NaN 0. reward/damage·dose=0은 hold_gate=0(random action=실파지 아님)이라 **정상**(접근 스침 미벌점). buckle 24°<35° 정확. → **파이프 검증 PASS**. Gate D서 실파지+저강성이면 damage 활성 확실.
4. **Gate D — 학습**: stiffness 커리큘럼 fresh 재학습(obs 133 tactile-only). **GPU 대기**(server GPU0=사용자 left grasp_v1·GPU1=massshift_s2r2 점유 중). 판정:
   - **grip_force-vs-mass 곡선 평탄화**(ratio 무게 무관 ~일정, 빈 컵 3.0 → 적정) = **진짜 적응**.
   - deformation 낮음·buckle_rate~0·success 유지.
   - **정성**: play.py 영상 — 패널 안 접힘(intact) vs 과파지 시 접힘 대비.

---

## 리스크 / 폴백
- **Articulation 안정성**(패널×조인트×2048 env, 솔버 비용): 폴백 = 패널 8개, D6 대신 revolute, Phase2 학습 num_envs 축소.
- **조인트 스프링 튜닝**(현실 종이컵 좌굴력 정합): d_safe/stiffness를 실측 종이컵 좌굴하중 근거로 보정(로그 먼저). 초기엔 상대값.
- **패널-손가락 접촉**(per-panel sensor 다수): 접촉 집계 비용 → 패널 수로 조절.
- **bead-변형컵 상호작용**: 초기엔 bead 수 유지, 불안정 시 축소.
- **체크포인트**: 컵 물리 변경 = 동역학 변화 → **fresh 재학습 필수**(s2r_base1 미전이). obs/action 불변이라 코드 호환만.

## 성공 기준 (Phase 2 완료)
1. grip_force가 무게에 **비례**(ratio ~일정, 과파지 해소) = 진짜 적응.
2. deformation 낮음 + buckle~0 + success 유지(무게 견딤).
3. play.py 영상으로 "안 부숨(패널 intact)" 시연 가능.
4. 정량 eval: per-mass grip_force/deformation/slip/success 표 + grip-force-vs-mass 곡선.
