# grasp_adapt — 촉각 기반 Fragile Object Grasping

> 상위 규칙: `hdgp/CLAUDE.md` (로그 먼저, reward-audit, 실험 루프).

## 태스크 정체성
- 목표: envelope(감싸쥐기)가 아니라 **엄지 + 대향 2지 이상 fingertip precision 파지**로
  fragile object(종이컵)를 미끄럼 하한과 파손 상한 사이 **안전 파지력**으로 든다.
- 설계 근거: Notion "촉각 기반 Fragile Object Grasping 설계".
- 계획: `hdgp/docs/superpowers/plans/2026-07-27-grasp-adapt-fingertip-fragile-rebuild.md`.

## 핵심 gate (Phase 1 이후)
- 안정 파지 = `compute_precision_grasp_mask` = 엄지(idx0) 접촉 AND 대향(idx1~4) 2개 이상 접촉.
- 5/5 hard gate(`num_contacts >= NUM_FINGERTIPS`)는 **금지** — envelope 강요라 제거됨.

## 살아있는 자산 (건드릴 때 주의)
- adaptive objective: `openarm/common/grasp_adaptive_core.py` (secure/efficient/drop).
- fingertip-local 3축 힘 15D: actor obs (sim2real 정합, world→body-local 회전).
- bead-mass 은닉: actor는 질량 모름(tactile 추론), critic/reward만 privileged
  (`GraspRightEnvCfgNoActorMass`가 실제 등록 태스크).
- real2sim actuator 보정: `real2sim_actuator_cfg.py`.

## 손가락 인덱스 규약
- tip contact 텐서 index 0 = 엄지(`rl_dg_1_tip`), index 1~4 = 검지/중지/약지/소지.

## 핵심 지표 (TFEvents)
- `reward/r_secure`, `reward/r_efficient`, `reward/r_drop`
- precision grasp 비율, 접촉 손가락 수 평균, slip speed, damage violation rate(Phase 2+)
- 단일 시점·단일 지표 판단 금지 — hdgp/CLAUDE.md 분석 원칙 준수.

## 정비 이력
- Phase 0 (2026-07-27): grasp_v10_3/v11에서 copy된 죽은 코드 제거
  (palm_action_utils, reward_utils 6종, finger_action_utils lift-retarget군, preset v4 블록),
  tests/ 스캐폴딩 신설, 이 CLAUDE.md 생성. **학습 동작 무변경.**
