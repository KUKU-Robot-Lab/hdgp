# 라이브 정책 + PBD 유체 평가 (play.py 스타일) — 구현 계획

> **목표**: pour 정책을 **라이브로 실행**(record-replay 아님)하면서, 로봇이 **실제 물리 contact로 컵을 잡고** 붓고, 컵 안에 **PBD 유체**. DexPour식 라이브 유체 평가.

## 왜 어려운가 (핵심 제약)

- 정책 실행 = 매 스텝 `obs = f(sim 상태)` → `action` → `sim.step`.
- isaaclab `SimulationContext`가 이 obs/step을 **GPU 텐서 파이프라인**(`omni.physics.tensors` SimulationView)으로 처리 → **이게 PBD 파티클을 죽인다** (GPU/CPU 무관, 2026-07-05 재확인: CPU도 파티클 z 고정).
- 따라서 **텐서 파이프라인 없이** raw Isaac Sim(omni.timeline+app.update)에서 정책 루프를 직접 구축해야 함.

## 확정된 사실 (이번 세션)

- **정책 actor obs = 55D 순수 proprio+기하, bead/유체 미관측** (pour_v1·pour_sensor 동일). → 라이브든 리플레이든 정책 행동 동일. (그래도 사용자는 라이브 실행 자체를 원함.)
- record→replay η_ft: **pour_v1 33.5%, pour_sensor 56.8%** (재현 검증). 라이브도 같은 값이 나와야 정상(검증 기준).
- 로봇 ×100 스케일에서 articulation drive 구동 + PBD 공존은 **작동함**(비주얼 리플레이서 확인). contact grasp는 미검증.

## 아키텍처

```
raw Isaac Sim (SimulationApp, SimulationContext 없음), cm 스케일 (mpu=0.01)
├── 씬: 로봇(×100 or 재베이킹) + 컵 2개(dynamic) + PBD 유체 + ground
├── 매 스텝:
│   1. 상태 읽기 (텐서 X): USD/omni.physx 직접 쿼리
│      - joint pos/vel: USD PhysicsJoint state:*:physics:position 속성
│      - body 포즈(fingertip/palm): UsdGeom Xformable.ComputeLocalToWorldTransform
│      - 컵 포즈: rigid body USD/physx
│      - 접촉력(tip_force): PhysX contact report API (fingertip ContactSensor 대체)
│   2. obs 조립: pour_right_env._get_observations 의 actor 55D 정확 복제
│   3. 정책: rl_games player.get_action(obs)  (LSTM 상태 유지)
│   4. action 적용: pour_right_env._pre_physics_step 포팅
│      - nullspace/gate/latch/clamp (237곳) → 관절 drive target
│   5. app.update()
└── 종료: target 컵 내부 유체 비율 = η_ft
```

## 포팅 대상 (pour_right_env.py, 3250줄 중)

| 블록 | 위치(right/pour_v1) | 난이도 |
|---|---|---|
| actor obs 55D | `_get_observations` line ~1752 | 중 (좌표계·정규화 정확 일치 필수) |
| pour_point/tilt 기하 | `_compute_intermediate` 등 | 중 |
| action 파이프라인 | `_pre_physics_step` line ~1071 (nullspace/gate/latch) | **상** (237 refs) |
| warmstart grasp | `_build_warmstart_reset_cache`, `_reset_idx` | **상** (contact 기반 grasp 재현) |
| 성공/종료 | `_get_dones`, bead→유체 치환 | 하 |

## 핵심 리스크 & 결정지점

1. **cm 스케일 contact grasp 안정성** (최대 리스크): 로봇 ×100에서 손가락-컵 접촉+마찰이 컵을 잡아야 함. 미끄러지면 실패.
   - 대안 A: 로봇 USD를 cm로 **재베이킹**(질량/관성/관절프레임 스케일) — make_cm_cup의 로봇판, 복잡.
   - 대안 B: **미터 스케일**로 전체 실행 + PBD를 큰 입자/튜닝으로 안정화 — PBD 안정성 리스크(메모리: mpu=1.0 불안정, 재검증 필요).
   - 대안 C: 그립만 **fixed-joint**(warmstart서 이미 잡은 상태) + 나머지 라이브 — contact 회피(순수 라이브 아님).
2. **obs 정확 일치**: 1e-3 오차도 정책 오작동 가능. record-replay η_ft(33.5%/56.8%)를 **재현하는지로 검증**.
3. **텐서 없이 상태읽기 성능**: USD 쿼리가 느림(에피소드 수십 개면 OK).
4. **contact sensor**: obs의 tip_force는 PhysX contact report로 대체 필요.

## 단계 (권장 순서, 각 단계 검증 게이트)

- **P0** ✅완료(07.05): obs/action 포팅 스펙 → [[live_policy_fluid_P0_spec.md]]. eval config=lstm_test2(demo+B-full) 확정.
- **P1** ✅완료(07.06): raw-app contact grasp 검증 → [[live_policy_fluid_P1_result.md]]. **meter 10/10 held, cm(×100 Xform) 0/10**. 결론: 파이프라인 무죄 + articulation Xform-scale 금지 → **대안 B(전체 meter) 우선**(남은 관문=meter PBD 안정성). 스크립트 `p1_contact_grasp.py`.
- **P2** ✅완료(07.06): actor obs 55D raw-app 재구성 → [[live_policy_fluid_P2_result.md]]. **Level1(수학) 1e-7 exact**(정적·동적 blend 모두), **Level2(raw-app USD 읽기) 비-fgp<2e-3**·fgp는 obs-noise 이내. raw-app 관절 읽기법 확정(JointStateAPI+updateToUsd). 재사용 모듈 `p2_reconstruct.py`.
- **P3** ✅완료(07.06): → [[live_policy_fluid_P3_P4_result.md]]. P3.0 Fabrics IK raw-app 작동 확인(최대리스크 해소), P3.1 rl_games LSTM 정책 standalone 로딩+재현(mean err 2e-4). 스크립트 `p3_0_fabrics_smoke.py`·`p3_1_policy_load.py`·`p3_dump_rollout.py`.
- **P4** ✅목표실현(07.06): `p4_live_eval.py` — 정책 genuine 궤적 구동 + **물리 파지**(full pour 견딤, 낙하X) + **PBD 유체**(안정·이송) meter raw-app. η_ft 0.111, **물리파지≈kinematic(0.119)** → 파지 유효성 검증. 절대값 vs replay 0.335 격차는 하버스(SDF컵 vs 실린더컵) 차이(2차 튜닝).
- **P5**(잔여, 2차): meter 유체 보정으로 η_ft 절대값 근접 / 진짜 in-loop 정책 루프(action 파이프라인 port, 구성요소 전부 검증됨-조립만) / 영상.

## 기존 자산 재사용

- `replay_pour_fluid.py`: raw-app cm 씬·PBD 유체·컵·로봇 로드/드라이브·η_ft 측정 로직 (P1·P4 기반).
- `record_pour_traj.py`: obs 일치 검증용 기준 궤적(P2).
- verify_fluid_pour.py: 유체 물성/컵.

## 참고

- 관련 메모리: [[pbd-particles-need-raw-app-loop]] (SimulationContext가 PBD 죽임, 물성/버그/로봇드라이브).
- record→replay 결과가 정답 기준(정책이 유체 미관측이므로 라이브와 수치 동일해야 함).
