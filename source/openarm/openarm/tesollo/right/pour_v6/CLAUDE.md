# 5g_pour_right_v6 — 논문용 ablation 통합 태스크

## 목적 (2026-06-20 생성)

**v6 = v4(=v5 구조 + demo reward 기계 + demo nullspace)를 복사하되, demo prior 주입을
두 직교 cfg flag로 전환해 단일 env에서 4셀 ablation을 재현한다.**

| 셀 | `nullspace_baseline` | `enable_demo_pose_reward` | = 기존 |
|---|---|---|---|
| 순수 DRL (control) | `robot_start` | `False` | v5 |
| demo nullspace (hard prior) | `demo` | `False` | v4 |
| demo reward (soft prior) | `robot_start` | `True` | 신규 |
| both (interaction) | `demo` | `True` | — |

- **공통 불변(세 셀 모두)**: 7-D α action(잉여 1-DOF self-motion) + rim-pivot 3D + approach(06.18 복원).
  → "demo 없음"과 "제어 DOF 없음"이 섞이지 않도록 α action은 control에도 반드시 포함.
- **변수는 demo prior 주입 경로뿐**: nullspace baseline(hard) ⟂ demo_pose_reward(loss soft).
- 기본값 = (`robot_start`, `False`) = 순수 DRL control.

## 배선 위치

- flag: `pour_right_env_cfg.py` `nullspace_baseline: str`, `enable_demo_pose_reward: bool`.
- nullspace baseline 분기: `pour_right_env.py` `_pre_physics_step` nullspace 블록
  (`if self.cfg.nullspace_baseline == "demo": ...`). 분기 밖 offset·α·clamp는 공통.
- demo reward gate: `_get_demo_pose_reward_terms` — flag off 시 정확히 0 반환(ablation 청결).
- 계약 테스트: `tests/test_v6_ablation_flags.py` (정적 텍스트 검증, Isaac 불필요).

## 학습 (셀별 1 run)

```
python3 scripts/tools/record_test_snapshot.py --task open-tesol_r_pour_v6 --test <cell_name>
# cfg에서 nullspace_baseline / enable_demo_pose_reward 조합 지정 후 학습
```
- record_test_snapshot이 cfg 파라미터를 스냅샷 → ablation 셀이 기록에 보존됨.
- **action 차원 7-D = v4/v5 체크포인트·warmstart 무효, fresh 재학습 필수.**

---

## 상속 메커니즘: bead-count 커리큘럼 (v5/v4에서 상속, 그대로 유효)

- 물리 비드 `_DEFAULT_BEAD_COUNT=30` 고정 spawn. 커리큘럼이 **활성 N**만 사용(앞 N 슬라이스).
- 비활성 비드는 hide(z=-10). 모든 bead fraction은 **활성 N 정규화**.
- N 스케줄 `(1,5,8,10,20,30)`, stage-windowed success_rate ≥ 0.5 시 advance.

### 로그 지표
```
log/active_bead_count    ← 활성 비드 수 (커리큘럼 진행)
log/bead_in_target       ← 활성 N 정규화 채움 (0이면 pour 없음)
log/demo_arm_pose_w      ← demo reward weight (감쇠 이력)
joint_State/null_ref_j4/j5 ← α가 실제 팔꿈치/롤을 움직이나
```

---

## 코드 수정 규칙

- reward/gate/weight 변경 전 reward-audit 필수 (`~/.claude/skills/reward-audit/`)
- obs/action 차원 변경 금지 (명시적 요청 없이) — 7-D action·144 critic obs 고정.
- **ablation 무결성**: 세 셀 공통 코어(α action·rim-pivot·approach·reward 코어)는 한쪽만 고치면 안 됨.
  변수는 두 flag뿐. 코어 수정 시 모든 셀에 동일 적용.
- v3/v4/v5는 별개(보존) — v6만 수정.
