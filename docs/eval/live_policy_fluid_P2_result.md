# P2 결과 — actor obs 55D raw-app 재구성 검증 (2026-07-06)

> 상위: [[live_policy_fluid_plan.md]] · 스펙: [[live_policy_fluid_P0_spec.md]] · 이전: [[live_policy_fluid_P1_result.md]].
> 스케일 = meter (대안 B, P1b 확정). config = lstm_test2(P0 §4).

## 질문
pour_v1 `_get_observations` 의 actor obs 55D 를 **텐서 파이프라인 없이 raw-app USD 쿼리로 재구성**해
같은 상태에서 같은 obs 가 나오는가. (P3 라이브 루프가 정책에 올바른 입력을 주는지의 전제)

## 산출물
- `p2_dump_ref.py`: isaaclab pour_v1 env 를 warmstart reset(noise off) → actor obs 55D + 재구성용
  raw 상태(관절/컵/palm) + geometry _w + cfg 상수를 npz 로 덤프. `--tilt_deg` 로 소스 컵을 기울여
  deep-tilt geometry(동적 blend) 분기도 생성.
- `p2_reconstruct.py`: **순수 numpy** actor obs 재구현(P0 §1 조립 + §2 geometry). **P3 라이브 루프의
  재사용 모듈.**
- `p2_obs_raw.py`: raw Isaac Sim(SimContext 없음)에서 덤프 상태로 로봇/컵 배치 → USD 로 상태 읽어
  재구성 → 덤프 obs 대조.

## 결과

### Level 1 — obs 조립 + geometry 수학 (재구현 vs env), GPU 불필요
| 상태 | max\|err\| | 판정 |
|---|---|---|
| upright (reset, dyn_w=0 정적 blend) | **2.6e-7** | ✅ PASS |
| **60° tilt (dyn_w≈0.74 동적 blend)** | **1.9e-7** | ✅ PASS |

채널 전부 float 정밀도(1e-7)로 일치 — arm/vel/fgp/left_arm/pour_point_to_opening/3축/last_palm.
**source_pour_point_w 의 정적·동적 blend 양쪽 분기 모두 검증**(deep-tilt 포함). → P0 §1/§2 스펙이 정확.

### Level 2 — raw-app USD 읽기 파이프라인 (raw Isaac Sim, meter)
8 envs, 관절 JointStateAPI init + 고강성 drive, 컵 kinematic, Fabric write-back off:
| 채널군 | max\|err\| | 비고 |
|---|---|---|
| arm_joint_pos | ~1e-3 | drive 정상상태 오차 |
| left_arm / cup pose / geometry(pp2open·3축) | **<2e-7 ~ 3e-8** | ✅ 정확 |
| finger_grasp_progress(fgp) | 0.004–0.011 | ⚠️ 물리 평형 잔차 |

- **porting 로직(비-fgp 전 채널): 모든 env <2e-3 PASS.** USD 관절 읽기(JointStateAPI + `updateToUsd`)가
  live physics 값을 주고, 컵 pose·geometry 재구성이 정확함을 확인.
- **fgp 잔차 0.004–0.011**: 손가락 관절이 raw-app drive(≠isaaclab ImplicitActuator)에서 ~0.02–0.04 rad
  다른 평형으로 안착(collision off 로도 지속 → 접촉 아닌 **drive 정상상태 오차**, 손가락 저관성). 이는
  정책 학습 obs_noise(`obs_noise_joint_pos=0.01`) 이내 → 정책 견딤. **라이브 루프에선 hand 를
  grasp_hold 로 drive-freeze 하며 실제 컵을 쥐므로(P1 파지 10/10) 무의미.**

## 핵심 발견 (P3 직결)
1. **raw-app 관절 읽기 방법 확정**: `PhysxSchema.JointStateAPI.GetPositionAttr()` + carb `updateToUsd=True`
   (+ Fabric write-back off) → live 관절값. P1 에서 확정한 rigid body pose 읽기(ComputeLocalToWorldTransform)와
   함께 **라이브 obs 조립에 필요한 모든 상태를 raw-app 에서 읽을 수 있음**.
2. **`p2_reconstruct.py` 가 P3 재사용 모듈**: 상태(관절+컵 pose) → 55D. env 와 exact 일치 검증됨.
3. obs 는 **컵 2개 pose + 로봇 관절만** 필요(contact force·palm_center 등은 actor obs 미참조) → 라이브 조립 단순.

## 판정
✅ **P2 PASS** — obs 재구성 로직 exact(1e-7, 정적·동적 blend 모두), raw-app 상태 읽기 검증(비-fgp <2e-3,
fgp 는 obs-noise 이내 물리 잔차). 라이브 루프 obs 입력 경로 확보.

## 다음 — P3
Fabrics IK(P0 §3, 텐서 독립 확인) + action 파이프라인(P0 §4) lift → 정책 라이브 실행(bead 먼저).
obs = `p2_reconstruct.assemble_actor_obs` + raw-app 관절/컵 읽기. LSTM 상태 유지. record-replay 궤적과 대조.
