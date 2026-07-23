# P1 결과 — raw-app contact grasp 검증 (2026-07-06)

> 상위: [[live_policy_fluid_plan.md]] · 스펙: [[live_policy_fluid_P0_spec.md]].
> 스크립트: `hdgp/scripts/reinforcement_learning/probes/p1_contact_grasp.py` (로컬 RTX 5090, headless).

## 질문
SimulationContext(텐서 파이프라인) 없이 raw Isaac Sim(omni.timeline + app.update)에서 로봇 손가락이
**dynamic 컵을 실제 contact + friction 으로 잡고 유지/리프트**하는가. warm state(grasp_warm_tesollo.hdf5,
성공 파지 2048개)의 arm7+hand20+cup pose 를 그대로 재구성.

## 방법 (요점)
- raw-app, mpu=meter(1.0) 또는 cm(0.01, --cm). 로봇 USD 참조 + 고강성 USD DriveAPI.
- **arm+hand 를 warm 포즈로 초기화**(JointStateAPI) → t=0 부터 침투 없는 valid 파지. 컵 dynamic+gravity.
- 판독: **Fabric transform write-back 끄고**(`/app/useFabricSceneDelegate=False`,
  `/physics/fabricUpdateTransformations=False`, `/physics/updateToUsd=True`) → `ComputeLocalToWorldTransform`
  이 live physics pose 읽음. (이걸 안 하면 authored 초기값만 읽혀 이동이 0 으로 보임 — 함정.)
- **판독 검증 control**(`--control_drop`): 컵을 손 위 +20cm(접촉 없음) → 낙하해야 정상 → drop 378mm 확인 OK.

## 결과

| 스케일 | held | 대표 지표 | 판정 |
|---|---|---|---|
| **meter (네이티브)** | **10/10** | slip 0–18mm, drop −40~+23mm, lift_slip 14–39mm(컵이 손 추종) | ✅ 파지 유지 |
| **cm (×100 Xform-scale)** | **0/10** | slip/drop ~8000mm+ (컵 수 m 낙하) | ✗ 완전 실패 |

## 결론

1. **텐서 파이프라인 무죄**: contact grasp 는 SimulationContext 없이 meter raw-app 에서 그대로 유지된다
   (10/10, 리프트 시에도 컵 추종). SimulationContext 는 PBD 만 죽일 뿐 rigid contact 해석엔 무관하다는
   가설이 실증됨.
2. **articulation 을 Xform ×100 로 스케일하면 안 된다**: PhysX 가 contact offset/mass/inertia/solver 를
   Xform scale 에 맞추지 못해 파지가 완전히 붕괴(0/10). 계획서가 우려한 "cm 스케일 contact 리스크"의 정체 =
   **crude Xform-scale**. (주의: 이는 "proper cm 재베이킹(대안 A)"을 배제하는 게 아니라, **crude Xform ×100
   방식**을 배제한다.)

## 대안 결정 (계획서 리스크 1)

- ✅ **대안 B (전체 meter 스케일) 강력 우선**: 로봇을 네이티브 meter 로 두면 contact 가 신뢰성 있게 작동.
  남은 유일 관문 = **PBD 유체가 meter 스케일(mpu=1.0)에서 안정한가** (메모리 `pbd-particles-need-raw-app-loop`:
  "mpu=1.0 불안정, 재검증 필요"). → **P4 착수 전 meter PBD 스모크 필요**.
- ⚠️ **대안 A (로봇 cm 재베이킹)**: mass/inertia/collision offset/joint frame 를 cm 로 정식 오서링. Xform-scale
  과 달리 물리적으로 맞으나 작업량 큼. **B(meter PBD)가 실패할 때만** 검토.
- ✗ **crude Xform ×100**: 실증 실패 — 사용 금지.

## P1b — meter PBD 스케일 스모크 (2026-07-06, 대안 B 확정)

스크립트: `p1b_pbd_scale_smoke.py`. 정적 컵(kinematic)에 물 파티클 채우고 중력 안착 → 폭발/누수/동결
없이 안정하는지 측정. replay 파티클 시스템·컵·fill 을 스케일 파라미터화(_M) 재사용.

| 스케일 | 판정 | 지표 (155 파티클) |
|---|---|---|
| **meter (mpu=1.0)** | **STABLE** | finite 155/155, bbox 0.064×0.063×0.007m(컵~0.26m), frac_in_cup=1.00, tail_move 0.84mm |
| cm (mpu=0.01, replay 기존) | STABLE | finite 155/155, bbox 0.063×0.063×0.007m, frac_in_cup=1.00, tail_move 0.91mm |

**두 스케일 지표 거의 동일** → cm(기존 검증 스케일)과 meter 가 동등하게 안정. 메모리
`pbd-particles-need-raw-app-loop` 의 "mpu=1.0 불안정" 노트는 **정적 재검증에서 뒤집힘**.
⚠️ 단 이는 **정적 안착** 테스트 — 동적 pour(컵 기울임 중 유출/튐)는 P4 에서 확인.

## 결정 (확정)

✅ **대안 B (전체 meter 스케일) 확정**: 로봇 contact grasp(P1: 10/10) + PBD 유체(P1b: STABLE) 둘 다
meter raw-app 에서 작동. 라이브 루프를 meter 로 구축한다.

## 다음 단계
1. ✅ ~~meter PBD 스모크~~ 완료 — 대안 B 성립 확정.
2. **P2 (obs 일치)**: pour_right_env `_get_observations` actor 55D 를 raw-app USD 쿼리로 재구성 →
   record 궤적/lstm_test2 상태에서 동일 obs 나오는지 검증. (스케일 meter, config=P0 §4)
3. **P3 (action 라이브 루프)**: Fabrics IK + action 파이프라인 lift → 정책 라이브 실행(bead 먼저).
4. **P4**: PBD 유체 얹고 η_ft 측정, record-replay(33.5%)와 대조. 여기서 동적 pour PBD 안정성도 확인.

## 스크립트 사용법
```
./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p1_contact_grasp.py \
    --headless --states random:10 --lift          # meter
    [--cm]                                          # cm(×100, 실패 실증용)
    [--control_drop]                                # 판독 검증(낙하해야 정상)
    [--friction 1.2 --hand_stiffness 2e4 ...]       # 튜닝
```
