# pour 프로젝트 전용 규칙 — Tesollo hand pouring

> **v5/v6 공통 메인.** `pour_v6/CLAUDE.md`가 이 파일을 `@import`한다.
> v5/v6 차이는 **틸팅 방식뿐**(v5=RIM-PIVOT, v6=PALM) — reward는 동일(대조군).
> 저장소 공통 규칙(분산 학습 환경·도구·7축 분석 원칙·Notion 기록)은 `hdgp/CLAUDE.md` 참조.

---

## 프로젝트 목표

source cup → target cup로 bead(APA proxy)를 붓는다.
- arm 6D pose → Fabrics IK, joint freeze. hand는 PCA.
- **1차 목표**: 잘 못 넣더라도 **deep tilt를 계속 시도**하게 (정조준 무관).
- **2차 목표**: bead 실제 진입(`bead_in_target` 발화).

---

## 실험 현황 (2026-06-23)

- **deep_tilt_boot1 (B+C)** 학습 중 — server 두 GPU 대조:
  - `r_tilt`/`r_tilt_delta`에서 `latched_ready` 제거 → tilt를 pour_point corridor에서 분리.
  - `weight_tilt` 35→20 (유지 farming 완화, 증분 위주).
  - v5(rim) = GPU1, v6(palm) = GPU0. commit `bfa3f7b`.
  - 초기 효과: ep46에 `tilt_frac_110` 0.055 (test4는 2600ep 내내 0).
- **미해결 (다음 = A)**: pour_point를 실제 배출구로 재정의 → 2차 목표(bead 진입).
  - A 단독은 직립 wobble로 corridor 진동 → tilt-valid gate 필요(reward-audit REVISE).

---

## reward 구조 (현재)

| 항목 | 식 | 비고 |
|---|---|---|
| `r_tilt` | `weight_tilt · tilt_progress` | **deep_tilt_boot1: latched_ready 제거** |
| `r_tilt_delta` | `weight_tilt_delta · relu(Δtilt_amount)` | 더 깊어질 때만(증분) |
| `r_align` | `weight_align · (1+directional_tilt_cos_c)/2` | cup-center→target 방향 |
| `r_pour` | `weight_transport · tilt_progress · exp(-pour_z_scale·\|mouth_z_clearance - pour_z_target\|)` | z-clearance 보정 |
| `r_introt` | `weight_introt · internal_rot_gate` | always-on 내회전 |

- `tilt_amount = (1 - source_up_dot)/2` (직립 0 → 90° 0.5 → 뒤집힘 1.0)
- `tilt_progress = tilt_amount / tilt_target`

---

## pour_point / tilt 좌표계 (진단 핵심)

- **`source_up_dot` = cup up축 z성분.** 직립=1, 90°=0, 뒤집힘=-1. **정확한 자세 측정** (이걸로 deep tilt 판정).
- **`pour_point`(현재 근사)**: `rim_center ± outer_radius(4.5cm)`. xy 방향을 **target 고정(자세 무관)**으로 근사 → 직립 wobble 회피했으나 **방향성 죽임 + 도달불가**.
  - 두 컵 `cup_center_xy_dist` ≈ 0.19m 떨어져 있어 `mouth_xy_dist`가 구조적으로 0 불가.
  - **A에서 재정의 예정**: 실제 중력 최하단점(gravity_perp xy+z) + tilt-valid gate.
- **`directional_tilt_cos_c`**: 기우는 방향(up_axis xy) vs cup-center→target 방향. (pour_point 무관, 안정적)

---

## 진단 방법론 (표면 지표 ✗ → 가설 주도 인과)

⚠️ **reward·entropy·성공률·frac 같은 표면 지표만으론 RL 실험을 알 수 없다.**
고정 N축 체크리스트가 아니라, **가설마다 봐야 할 지표가 다르다.** 표면 지표에서 멈추면
local minimum·병목의 *원인*을 못 찾는다.

> **test4 교훈**: 표면 지표는 "frac_110=0, bead_in_target=0 정체"만 보였으나, 심화 진단
> (joint 포화 · gate 경로 · pour_point 기하 · action 추종)에서 진짜 원인 — pour_point가
> rim±4.5cm로 두 컵(19cm)에 도달불가 + `latched_ready`가 tilt를 차단 — 을 발견했다.

**진단 절차:**
1. **현상**: 무엇이 정체/붕괴하나 — 여러 지표 **추세**로 (단일값·단일시점 금지).
2. **가설 나열**: 가능한 원인 경로 (network 표현력 / reward 구조 / gate 차단 / 물리 좌표 / joint 한계 / 탐색 붕괴).
3. **각 가설을 직접 측정으로 검증** (가설에 필요한 것만):
   - **action 추종**: `raw_action` → `command_pre/post_gate` → `applied_action`, `cmd_minus_actual_tilt_deg` (명령이 실제로 실행되나)
   - **gate 경로**: `latched_ready`/`corridor_score`/`aim_score` — 무엇이 보상을 죽이나
   - **물리 좌표**: `pour_point_*` vs `target_open_*`, `source_up_dot`, `rim_facing_cos`, `spill_ratio` (실제 쏟김 vs 측정)
   - **joint**: `joint_State/jN_sat`(j4 높이·j6 tilt 포화), `palm_clamp_viol_*` (구조적 한계)
   - **reward 구성비**: 각 항 기여치 비교 — local min / farming 탐지
   - **건강**: `entropy` 추세(test1식 20 발산?), `kl`, `grasp_broken`, `cup_rel_drift_deg`
4. **하나의 primary bottleneck 특정** → 최소 수정 → reward-audit.

**표면 지표(출발점일 뿐, 여기서 멈추지 말 것):**
성공 `bead_in_target`(0이면 pour 없음) · deep tilt `tilt_frac_110`·`source_up_dot`(<0=90°+) · 위치 `mouth_xy_dist` · 건강 `entropy`·`kl`

---

## reward-audit 실패 이력 (pour)

| 변경 | 실패 |
|---|---|
| `weight_demo_arm=9.0` | tilt 13배 압도 → demo local min |
| `weight_align=10.0` | pour 탐색 없이 정렬 수렴 |
| `cup_collision_margin=0.12` | pour 중 두 컵 근접 불가 → 붕괴 |
| (test1) aim_score 곱셈 게이트 | deep tilt 억제 + entropy 붕괴(8.3→20) |
| (test4) tilt 증분만 | latched_ready에 묶여 110°벽 못 넘음(2600ep frac_110=0) |

---

## 틸팅 방식 (v5/v6 대조군)

- **v5 = RIM-PIVOT 틸팅** (rim 기준 pivot 회전)
- **v6 = PALM 틸팅** (palm 기준)
- reward 완전 동일. **어느 방식이 deep tilt에 유리한지** 같은 epoch에서 비교.

---

## 코드 수정 규칙

1. reward/gate/weight 변경 전 **reward-audit 필수** (`~/.claude/skills/reward-audit/`).
2. obs/action 차원 변경 금지 (명시 요청 없이).
3. v3/v4는 별개(보존) — v5/v6만 수정. **v5/v6는 대조군이므로 reward 변경은 양쪽 동일 적용**(틸팅 방식만 변수 유지).
4. 한 번에 하나의 가설 (사용자가 통합 명시 시 예외).
5. 변경 후 예상 지표 이동 방향 명시.
6. 학습 전 `record_test_snapshot.py` (train.sh가 자동 기록).

---

## bead-count 커리큘럼 (v3/v4 상속, 유효)

- 물리 bead `_DEFAULT_BEAD_COUNT=30` 고정 spawn. 커리큘럼이 **활성 N**만 사용(앞 N 슬라이스).
- N 스케줄 `(1,5,8,10,20,30)`, stage-windowed `success_rate ≥ 0.5` 시 advance.
- 모든 bead fraction은 활성 N 정규화. 모듈: `bead_curriculum.py` (`tests/test_bead_curriculum.py`).
