# pour v1 — PALM 틸팅 (최종 버전)

@../pour_v5/CLAUDE.md

> 위 공통 규칙(목표·reward 구조·진단 지표·reward-audit 이력·코드 수정 규칙)을 그대로 따른다.
> 아래는 **v1 고유 사항만**.

---

## 틸팅 방식: PALM

- **v1 = palm 기준 틸팅** (v5 = rim-pivot과 대조군).
- **reward는 v5와 완전 동일** — 변경 시 양쪽 동기화. 변수는 틸팅 방식뿐.
- 비교 관점: 같은 epoch에서 `tilt_frac_110`·`source_up_dot`·`cup_rel_drift`로 rim vs palm 우열.

---

## v1 고유 구조

- **7-D α action** (잉여 1-DOF self-motion) + rim-pivot 3D + approach. critic obs **144**.
  - action 차원 7-D → v4/v5 체크포인트·warmstart 무효, fresh 재학습 필수.
- **ablation flag** (논문용, `pour_right_env_cfg.py`):
  - `nullspace_baseline: str` (`robot_start`=순수DRL / `demo`=hard prior)
  - `enable_demo_pose_reward: bool` (soft prior)
  - 현재 `deep_tilt_boot1`는 **기본값(`robot_start`, `False`) = 순수 DRL** 사용.
  - 계약 테스트: `tests/test_v1_ablation_flags.py` (정적, Isaac 불필요).

---

## v1 코드 수정 주의

- **ablation 무결성**: 세 셀 공통 코어(α action·reward 코어)는 한쪽만 고치면 안 됨. 코어 수정 시 모든 셀 동일.
- reward 변경은 **v5와 반드시 동기화**(대조군 무결성). 한쪽만 바꾸면 비교 무효.
