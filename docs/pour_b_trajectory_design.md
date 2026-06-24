# Pour B-trajectory 액션 구조 설계 (2026-06-24)

> **목표**: 정책이 "pour-point 위치 맞추고 → 유지하며 → demo pour 궤적을 따라 좋은 자세로 morph"를
> **액션으로 직접 발행**하게. 현재 3D tilt 액션(IK가 j6로 나쁘게 실현)을 **1-D pour-progress β**로 교체.

## 0. 배경 (왜 재설계)
- 현재 tilt 액션 = cup-local 3D 회전 → Fabrics IK가 임의 관절로 실현 → **j6 포화(나쁜 자세), j5 미roll**.
- posture를 reward로 강제(demo_pose_reward, w=9) → tilt 압도 local min 실패.
- posture를 joint clamp 강제(Stage3) → 실현은 강제하나 **정책이 "좋은 자세로 진행"을 명령할 수단 없음**.
- → 액션 구조에 **β(pour 진행도)** 채널을 도입, env가 demo 궤적으로 올바르게 실현.

## 1. Canonical 참조 궤적 R(β) — 검증됨
- 소스: `datasets/pour_v1_a11~a20.hdf5` (teleoperation). tilt 구간(j5: 0→min) 추출 → j5-진행도 정규화 → 데모 평균.
- R(β): β∈[0,1] → 7-DOF arm 자세. (β=0 pre-pour, β=1 deep pour)
- **검증**: j4 0.97→1.85, j5 0.00→-1.09, **j6 내내 0.02~0.13(≈0)**, 협응 부드러움. (7데모 일치, j6 std<0.14)
- 저장: 상수 테이블(11~21 knot) + 선형보간. init에 빌드 or 상수 하드코딩.

## 2. 액션 공간 (7-D → 5-D)
```
현재: [0:3] xyz | [3:6] tilt 3D(cup-local) | [6] α
신규: [0:3] xyz pour-point 위치 | [3] β pour-progress | [4] α nullspace
```
- `[3] β`: tanh→[0,1] (또는 β delta + EMA). **3D tilt를 대체.** demo 궤적 진행도.
- (옵션 +1) spin/fine-tilt: pour 방향 미세조정 필요 시 추가 → 6-D.
- obs/critic의 last_action 차원도 동기화. **action dim 변경 → fresh 재학습.**

## 3. β 실현 (핵심 선택지)
ready-latch(pour 단계)일 때만 적용. 미ready(approach)=β 무시, xy로 위치.

- **V2 (joint-drive, 추천)**: ready 시 **j5를 R(β)[j5]로 하드 구동**(β=tilt 깊이 직접제어), j6를 R(β)[j6]≈0로 클램프.
  j1-4는 palm position task(xy)로 자유. → β→j5 roll→deep tilt 보장, **j6 leak 구조 차단**, IK 회전실현 문제 제거.
  Stage3 클램프의 일반화(band 중심=R(β), β=액션).
- V1 (palm-orientation): palm pose 타겟 orientation = R(β)의 palm 자세 + Stage3 j6 클램프. Fabrics 실현. (회전 task 잔존)
- **선택**: V2 우선 (가장 직접적, β↔tilt 결정적). V1 fallback.

## 4. 관측 (obs)
- 현재 β(진행 상태) 추가 → 정책이 자기 위치 인지. obs +1, critic +1.
- (기존 source_up_dot 등 tilt 지표는 유지 — β와 실제 tilt 정합 확인용.)

## 5. 보상 단순화
- β가 tilt를 직접 제어 → **tilt 구성 보상(r_tilt_delta 등) 불필요**. demo_pose_reward 불필요(β=demo 자세).
- 핵심 보상 = `bead_in_target`(실목표) + corridor(위치) + (선택) over-target 시 β-진행 shaping.
- → 학습 단순화: "3D 회전으로 deep tilt 구성"(IK와 싸움) → "위치 맞추고 β 올리기".

## 6. 단계 구조 (사용자 비전 직접 구현)
```
approach(미ready): xy로 pour-point→target, β=0(upright 유지)
pour(ready):       pour-point 유지, β↑로 demo 궤적 따라 deep tilt
```
- ready-latch(corridor 진입)가 기존처럼 phase 전환. β는 ready 후 활성.

## 7. Stage3와의 관계
- Stage3(j5/j6 정적 band) = B의 특수형(β 고정 band). B는 band 중심을 R(β)로, β를 액션화.
- B 도입 시 Stage3 클램프는 흡수(또는 j6 클램프만 보조 유지).

## 8. 구현 단계 / 리스크
1. R(β) 빌드 유틸 + 상수 (offline 검증됨).
2. action dim 5-D 변경 (constants/cfg/obs/critic 동기화) — **차원 변경 주의**.
3. β 실현 V2: ready-gate j5 하드구동 + j6 클램프 (Stage3 코드 확장).
4. obs에 β 추가.
5. reward 단순화 (tilt 구성텀 제거, bead 중심).
6. 테스트(정적) + 런타임 스모크.
- **리스크**: action dim 변경→체크포인트 무효(fresh). β 하드구동이 palm position task와 충돌 가능(V2 검증 필요). spin 없으면 pour 방향 고정(aiming은 xy 의존).
- **대조군**: v5(rim)/v6(palm) 동일 적용. 단 β-joint-drive는 틸팅 방식 차이를 흡수할 수 있어 v5/v6 대조 의미 재검토 필요.
