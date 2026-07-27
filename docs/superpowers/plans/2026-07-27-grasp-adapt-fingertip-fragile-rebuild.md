# grasp_adapt 팁-파지 · Fragile Object Grasping 전면 개편 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠️ 이 저장소는 RL 실험 루프다 (`hdgp/CLAUDE.md`).** 최종 검증은 정적 테스트가 아니라 **GPU 학습 + TFEvents 지표**다. 정적 pytest는 "코드가 깨지지 않았다"만 보장한다. 각 Phase 끝의 **학습 검증 게이트**를 통과해야 다음 Phase로 넘어간다. 모든 reward/gate/weight 변경 전 **reward-audit 스킬**을 통과한다.

**Goal:** 현 `tesollo/right/grasp_adapt`를 grasp_v1의 clean 구조 위에서 재정비하고, envelope(감싸쥐기)를 강제하는 5/5 손끝 접촉 hard gate를 **엄지 + 대향 2지 이상 fingertip precision 파지**로 교체한 뒤, Notion 설계 문서("촉각 기반 Fragile Object Grasping")의 fragile-damage / tactile-residual / damage-constraint / staged-curriculum 로드맵을 단계적으로 구현한다.

**Architecture:** 현 grasp_adapt의 살아있는 자산(20D joint-residual actuation, adaptive 최소-힘 objective `secure/efficient/drop`, fingertip-local 3축 힘 15D 관측, real2sim actuator 보정, bead-mass 은닉 비대칭)은 계승한다. grasp_v1에서 가져오는 것은 코드가 아니라 **구조 철학**이다: 죽은 코드 제거, `tests/` 스캐폴딩, 성공조건 단순화·문서화, 단일 gate 원칙. 핵심 물리 변경은 "모든 손끝 5접촉" 강제를 제거하고 "엄지 + 대향 2지 이상 + 낮은 slip + 변형/기울기 제한"으로 안정 파지를 재정의하는 것이다.

**Tech Stack:** Isaac Lab / Isaac Sim (DirectRLEnv), PyTorch, rl_games (PPO / PPO+LSTM), FABRICS geometric-fabrics IK (vendored), pytest (정적, GPU 불필요), TFEvents (`parse_tfevents.py`).

## Global Constraints

- **손끝-only (하드웨어 제약, 전 Phase 공통):** Tesollo 핸드는 **손끝에만 6축 F/T 센서**가 있다. 중간마디(`r_hl_*_3`)로 감싸는 envelope 파지는 그 접촉력을 센서가 못 읽어 tactile 파이프라인(Phase 2 damage, Phase 3 residual, Phase 5 constraint 전부)을 무효화한다. 따라서 **접촉은 손끝에서만 일어나야 하며, 이는 태스크 성공의 근본 정의**다. hold 구간 middle 접촉은 `envelope_penalty_weight`로 벌점(distal=`r_hl_*_4`는 손끝 인접 회색지대라 진단 로깅만). 이 제약은 Phase 1에서 도입하지만 모든 후속 Phase가 전제한다.

- **대상 폴더만 수정:** `hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/`. 공유 코어(`openarm/common/grasp_reward_core.py`, `grasp_adaptive_core.py`)는 **다른 태스크(RH56F1, grasp_v1/v2)가 함께 쓰므로** 기존 인자 기본값을 깨지 않는 방식(신규 optional 인자)으로만 확장한다. `hdgp/CLAUDE.md` 코드 수정 규칙 3.
- **손가락 인덱스 규약:** tip contact 텐서 index 0 = 엄지(`rl_dg_1_tip`), index 1~4 = 검지/중지/약지/소지. grasp_v1 `thumb_cup_grip = any_finger_contact[:, 0]` 근거.
- **접촉 임계값:** `CONTACT_FORCE_THRESHOLD = 0.1 N`, `CONTACT_FORCE_MAX = 10.0 N` (정규화 divisor). `grasp_right_constants.py`.
- **로그 먼저:** TFEvents 수치 근거 없이 reward 재조정 금지. Phase 검증 게이트는 `parse_tfevents.py`로 지표 추출 후 판단.
- **한 번에 하나의 가설:** 각 Phase는 독립 학습 실행으로 검증. reward 항을 여러 개 동시에 추가하지 않는다 (Phase 내 task 순서 준수).
- **차원 변경은 명시적으로만:** obs/action 차원을 바꾸는 task는 "cup 재학습 필수"를 반드시 명시하고 constants·cfg·docstring을 함께 갱신한다.
- **학습 실행 환경:** server(`oem@10.102.101.240`, RTX PRO 6000 ×2), conda `proj-hdgp-py311`, `CUDA_VISIBLE_DEVICES=N NOTE="" ./train.sh <task> <label> --num_envs 2048 --headless`. 로컬은 분석·수정만.

---

## Phase 구조 (전체 로드맵)

| Phase | 내용 | 차원 영향 | 진입 게이트 | 상세도 |
|---|---|---|---|---|
| **0** | Clean 베이스: 죽은 코드 제거, `tests/` 스캐폴딩, CLAUDE.md 생성 | 없음 | 즉시 | 상세 TDD |
| **1** | envelope 강제 제거 → 엄지+대향2지 fingertip precision 성공조건 | 없음 | Phase 0 | 상세 TDD |
| **2** | Stage A fragile: virtual damage 모델(F_yield/F_buckle/damage dose) + local damage penalty | critic obs +N | Phase 1 학습 게이트 | 스펙 개요 |
| **3** | tactile residual adapter (base freeze + 5D closure residual → staged 20D) | action 변경 | Phase 2 게이트 | 스펙 개요 |
| **4** | 시간 정보(4–8 frame stacking 또는 GRU) | obs/구조 | Phase 3 게이트 | 스펙 개요 |
| **5** | damage를 constraint로 분리 (Lagrangian `r' = r − λ·damage_cost`) | 없음 | Phase 4 게이트 | 스펙 개요 |
| **6** | Stage 0~5 커리큘럼 + segmented compliant shell (Stage B) | 다양 | Phase 5 게이트 | 스펙 개요 |

**Phase 2~6은 이 문서에서 스펙·진입 게이트·수용 기준만 정의한다.** 각 Phase는 착수 시점에 직전 Phase의 TFEvents 결과를 근거로 **별도의 상세 계획 문서(`docs/superpowers/plans/`)로 확장**한다. RL 특성상 다음 Phase의 reward 튜닝은 이전 Phase 학습 결과에 의존하므로, 지금 bite-sized로 못박으면 placeholder가 되기 때문이다.

---

# Phase 0 — Clean 베이스 정비 (학습 무영향)

**목표:** 학습 동작을 1비트도 바꾸지 않고, 죽은 코드를 제거하고 테스트·문서 골격을 세운다. Phase 0 완료 후 `open-tesol_r_grasp_adapt` 태스크의 학습 결과는 정비 전과 동일해야 한다(회귀 없음).

**Files:**
- Modify: `.../grasp_adapt/grasp_right_preset.py` (미사용 "Direct PD hand control (v4)" 블록 제거)
- Modify: `.../grasp_adapt/grasp_reward_utils.py` (6/7 미사용 함수 제거, `compute_upright_success_mask`만 유지)
- Modify: `.../grasp_adapt/finger_action_utils.py` (미사용 lift-retarget 함수군 제거, `compute_preset_residual_finger_targets`만 유지)
- Delete: `.../grasp_adapt/palm_action_utils.py` (호출 0곳)
- Modify: `.../grasp_adapt/grasp_right_constants.py` (미사용 `MIN_CONTACTS_FOR_SUCCESS` 제거 또는 실제 사용처와 정합)
- Create: `.../grasp_adapt/tests/__init__.py`, `.../grasp_adapt/tests/test_precision_grasp_mask.py`, `.../grasp_adapt/tests/test_adaptive_reward_contract.py`
- Create: `.../grasp_adapt/CLAUDE.md` (프로젝트 전용 도메인 규칙 — 현재 없음)

**Interfaces:**
- Produces: 정비된 모듈들. `compute_preset_residual_finger_targets(...)` 시그니처 불변. 테스트 러너 진입점 `tests/`.

- [ ] **Step 1: 죽은 코드 사용처 재확인 (grep 근거 수집)**

Run:
```bash
cd /home/user/rl_ws/hdgp/source/openarm
for f in compute_lift_stabilize_palm_targets compute_bounded_force_smooth_penalty compute_middle_contact_gate compute_thumb_pose_anchor_reward compute_thumb_downward_slide_penalty compute_thumb_tip_direction_reward compute_grasp_shape_consistency_reward compute_grasp_finger_targets compute_lift_finger_targets MIN_CONTACTS_FOR_SUCCESS; do
  echo "=== $f ==="; grep -rn "$f" openarm/tesollo/right/grasp_adapt/ | grep -v "def $f" | grep -v "\.pyc"
done
```
Expected: 각 심볼이 **정의부 외에 grasp_adapt 내부에서 import/호출되지 않음**을 확인(정의 라인만 출력되거나 무출력). 만약 예상외 사용처가 나오면 그 심볼은 제거 대상에서 제외하고 이 계획에 기록.

- [ ] **Step 2: 정적 테스트 — precision grasp mask 순수 함수 계약 (실패 확인용, Phase 1 미리 준비)**

이 테스트는 Phase 1에서 구현할 `compute_precision_grasp_mask`를 겨냥한다. Phase 0에서는 **테스트 파일만 만들고 skip 마크**로 두어 스캐폴딩을 완성한다(구현은 Phase 1).

Create `.../grasp_adapt/tests/test_precision_grasp_mask.py`:
```python
"""엄지+대향2지 precision grasp mask 계약 (Phase 1에서 구현)."""
import pytest
import torch

pytest.importorskip("openarm.tesollo.right.grasp_adapt.grasp_right_utils")


def _mask(tip_contact_bool):
    from openarm.tesollo.right.grasp_adapt.grasp_right_utils import (
        compute_precision_grasp_mask,
    )
    return compute_precision_grasp_mask(tip_contact_bool)


@pytest.mark.skip(reason="Phase 1에서 compute_precision_grasp_mask 구현 후 해제")
def test_thumb_plus_two_opposing_is_grasp():
    # 엄지(idx0) + 검지 + 중지 접촉 → True
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(_mask(tip)[0]) is True


@pytest.mark.skip(reason="Phase 1에서 구현 후 해제")
def test_no_thumb_is_not_grasp():
    # 엄지 없이 4지 접촉 → False (엄지 대향이 파지의 핵심)
    tip = torch.tensor([[False, True, True, True, True]])
    assert bool(_mask(tip)[0]) is False


@pytest.mark.skip(reason="Phase 1에서 구현 후 해제")
def test_thumb_plus_one_is_not_grasp():
    # 엄지 + 1지만 → False (대향 2지 미만)
    tip = torch.tensor([[True, True, False, False, False]])
    assert bool(_mask(tip)[0]) is False
```

- [ ] **Step 3: adaptive reward 계약 테스트 (현 동작 회귀 방지, 실제 실행)**

Create `.../grasp_adapt/tests/test_adaptive_reward_contract.py`:
```python
"""adaptive grip objective 회귀 방지: hold_gate=0이면 secure/efficient=0."""
import torch

from openarm.common.grasp_adaptive_core import compute_adaptive_grip_terms


class _Cfg:
    secure_weight = 10.0
    secure_slip_sharpness = 30.0
    force_efficiency_weight = 2.0
    drop_penalty_weight = 8.0
    force_ratio_cost_cap = 6.0


def test_hold_gate_zero_zeros_secure_and_efficient():
    n = 4
    total, terms = compute_adaptive_grip_terms(
        grip_normal_force=torch.ones(n) * 5.0,
        cup_weight=torch.ones(n) * 2.0,
        cup_speed=torch.zeros(n),
        tip_contact_frac=torch.ones(n),
        lift_gate=torch.zeros(n),      # lift 안 됨 → hold_gate=0
        lifted_gate=torch.zeros(n),
        full_tip_contact=torch.zeros(n),
        cfg=_Cfg(),
    )
    assert torch.allclose(terms["r_secure"], torch.zeros(n))
    assert torch.allclose(terms["r_efficient"], torch.zeros(n))


def test_no_slip_high_secure():
    n = 4
    _, terms = compute_adaptive_grip_terms(
        grip_normal_force=torch.zeros(n),
        cup_weight=torch.ones(n) * 2.0,
        cup_speed=torch.zeros(n),      # 정지 → secure_quality≈1
        tip_contact_frac=torch.ones(n),
        lift_gate=torch.ones(n),
        lifted_gate=torch.ones(n),
        full_tip_contact=torch.ones(n),
        cfg=_Cfg(),
    )
    assert torch.all(terms["secure_quality"] > 0.99)
```

- [ ] **Step 4: 테스트 실행 (스캐폴딩 검증)**

Run: `cd /home/user/rl_ws/hdgp && python -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -v`
Expected: `test_adaptive_reward_contract.py`의 2개 PASS, `test_precision_grasp_mask.py`의 3개 SKIP.

- [ ] **Step 5: `palm_action_utils.py` 삭제**

Run: `git rm hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/palm_action_utils.py`
Expected: 파일 삭제. (Step 1에서 호출 0곳 확인 완료 전제.)

- [ ] **Step 6: `grasp_reward_utils.py` 정리 — 미사용 6함수 제거**

`compute_upright_success_mask`(line 19) **만** 남기고 `compute_bounded_force_smooth_penalty`, `compute_middle_contact_gate`, `compute_thumb_pose_anchor_reward`, `compute_thumb_downward_slide_penalty`, `compute_thumb_tip_direction_reward`, `compute_grasp_shape_consistency_reward`를 제거한다. 파일 상단 docstring에 "Phase 0 정비: grasp_v10_3에서 copy된 미사용 헬퍼 제거, 활성 함수만 유지" 한 줄 기록.

- [ ] **Step 7: `finger_action_utils.py` 정리**

`compute_preset_residual_finger_targets`(line 12)만 남기고 `compute_grasp_finger_targets`/`compute_lift_finger_targets`/`_compute_reference_plus_delta_target`/`_clamp_indices_to_reference_delta`(line 28–142) 제거.

- [ ] **Step 8: `grasp_right_preset.py` 정리 — 미사용 "Direct PD hand control (v4)" 블록 제거**

`HAND_CURL_JOINT_NAMES`, `DISTAL_RATIO_PIP`, `DISTAL_RATIO_DIP` 등 grasp_adapt env가 import하지 않는 커플링 상수 블록 제거. 제거 전 `grep -n "HAND_CURL_JOINT_NAMES\|DISTAL_RATIO" grasp_right_env.py` 로 env 미사용 재확인.

- [ ] **Step 9: `grasp_right_constants.py` — `MIN_CONTACTS_FOR_SUCCESS` 정합**

이 상수는 docstring이 "성공 = 4접촉"이라 주장하지만 env는 import조차 안 하고 실제로는 `NUM_FINGERTIPS`(5)를 쓴다. Phase 0에서는 **제거하지 않고**(Phase 1에서 성공조건 자체를 바꿀 것이므로), 오해를 부르는 docstring만 정정: `# (미사용) 실제 성공 gate는 grasp_right_env._get_dones 참조 — Phase 1에서 precision mask로 교체 예정`.

- [ ] **Step 10: import 정합성 검증 (env 로드 가능 여부)**

Run: `cd /home/user/rl_ws/hdgp && python -c "import ast; [ast.parse(open(f).read()) for f in __import__('glob').glob('source/openarm/openarm/tesollo/right/grasp_adapt/*.py')]; print('AST OK')"`
Expected: `AST OK`. (Isaac Sim 없이 full import는 불가하므로 AST 파싱 + Step 1 grep으로 orphan import 부재를 확인.)
추가: `grep -rn "palm_action_utils\|compute_grasp_finger_targets\|compute_thumb_pose_anchor_reward" hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/ | grep -v "\.pyc"` → 무출력(제거한 심볼의 잔존 참조 없음).

- [ ] **Step 11: `grasp_adapt/CLAUDE.md` 생성 (프로젝트 전용 도메인 규칙)**

Create `.../grasp_adapt/CLAUDE.md`:
```markdown
# grasp_adapt — 촉각 기반 Fragile Object Grasping

> 상위 규칙: `hdgp/CLAUDE.md` (로그 먼저, reward-audit, 실험 루프).

## 태스크 정체성
- 목표: envelope(감싸쥐기)가 아니라 **엄지 + 대향 2지 이상 fingertip precision 파지**로
  fragile object(종이컵)를 미끄럼 하한과 파손 상한 사이 **안전 파지력**으로 든다.
- 설계 근거: Notion "촉각 기반 Fragile Object Grasping 설계".
- 계획: `hdgp/docs/superpowers/plans/2026-07-27-grasp-adapt-fingertip-fragile-rebuild.md`.

## 핵심 gate (Phase 1 이후)
- 안정 파지 = `compute_precision_grasp_mask` = 엄지 접촉 AND 대향(1~4지) 2개 이상 접촉.
- 5/5 hard gate(`num_contacts >= NUM_FINGERTIPS`)는 **금지** — envelope 강요라 제거됨.

## 살아있는 자산 (건드릴 때 주의)
- adaptive objective: `openarm/common/grasp_adaptive_core.py` (secure/efficient/drop).
- fingertip-local 3축 힘 15D: actor obs (sim2real 정합, world→body-local 회전).
- bead-mass 은닉: actor는 질량 모름(tactile 추론), critic/reward만 privileged.
- real2sim actuator 보정: `real2sim_actuator_cfg.py`.

## 핵심 지표 (TFEvents)
- `reward/r_secure`, `reward/r_efficient`, `reward/r_drop`
- precision grasp 비율, slip speed, damage violation rate(Phase 2+)
- 단일 시점 금지 — hdgp/CLAUDE.md 분석 원칙 준수.
```

- [ ] **Step 12: 테스트 재실행 + 커밋**

Run: `cd /home/user/rl_ws/hdgp && python -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -v`
Expected: 2 PASS, 3 SKIP (변화 없음).
```bash
cd /home/user/rl_ws
git add -A hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/ hdgp/docs/superpowers/plans/
git commit -m "refactor(grasp_adapt): Phase 0 clean base — 죽은 코드 제거, tests/·CLAUDE.md 골격"
```

### Phase 0 검증 게이트
- 정적: pytest 2 PASS / 3 SKIP, AST OK, orphan 참조 0.
- **학습 회귀 확인(선택, 권장):** 정비 전 커밋과 정비 후 커밋으로 각각 `--num_envs 512 --max_iterations 300` 단기 학습 → `reward/` 주요 스칼라 곡선이 통계적으로 동일한지 `parse_tfevents.py`로 비교. 죽은 코드만 지웠으므로 동일해야 한다. 다르면 "죽었다"는 판단이 틀린 것이므로 중단하고 조사.

---

# Phase 1 — Fingertip Precision 파지로 성공조건 교체 (핵심 요구)

**목표:** 학습 정책이 **감싸쥐기 대신 엄지+대향2지 팁으로** 컵을 들도록, 곳곳에 박힌 `full_tip_contact(5/5)` gate를 `precision_grasp_mask`로 교체한다. **차원 변경 없음 → 기존 체크포인트에서 warm-start 가능하나, 성공조건이 바뀌므로 fresh 재학습 권장**(메모리 "anchor 변경이면 재학습 필수" 교훈).

**Files:**
- Modify: `.../grasp_adapt/grasp_right_utils.py` (신규 `compute_precision_grasp_mask`, `compute_precision_grasp_frac`)
- Modify: `.../grasp_adapt/grasp_right_env.py` (`_get_dones` 1252/1261/1270/1274-1279, `_get_rewards` success_bonus 1140-1154, adaptive `hold_gate` 입력)
- Modify: `.../grasp_adapt/grasp_right_env_cfg.py` (`stage0_lift_start_min_contacts`, contact ADR, 신규 cfg 필드)
- Modify: `openarm/common/grasp_adaptive_core.py` (`full_tip_contact` 인자를 일반화 — **하위호환 유지**)
- Modify: `.../grasp_adapt/tests/test_precision_grasp_mask.py` (skip 해제)
- Test: `.../grasp_adapt/tests/test_precision_grasp_gate_wiring.py` (신규)

**Interfaces:**
- Produces:
  - `compute_precision_grasp_mask(tip_contact_bool: torch.Tensor) -> torch.Tensor` — 입력 `(N,5)` bool, 출력 `(N,)` bool. 정의: `tip_contact_bool[:,0] & (tip_contact_bool[:,1:].sum(dim=1) >= cfg.precision_min_opposing)`. Phase 1에서는 `precision_min_opposing` 기본 2를 상수로 하드코딩하지 않고 함수 인자 `min_opposing: int = 2`로 노출.
  - `compute_precision_grasp_frac(tip_contact_bool) -> torch.Tensor` — `(N,)` float, `1 + opposing_count`를 최대 3으로 정규화한 품질 `[0,1]`(엄지 없으면 0). reward graded 게이팅용.
- Consumes: Phase 0의 `tests/` 스캐폴딩, `compute_upright_success_mask`.

- [ ] **Step 1: skip 해제 — precision mask 테스트를 RED로**

`tests/test_precision_grasp_mask.py`의 `@pytest.mark.skip` 3개 데코레이터를 제거한다.

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `cd /home/user/rl_ws/hdgp && python -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/test_precision_grasp_mask.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_precision_grasp_mask'`.

- [ ] **Step 3: `compute_precision_grasp_mask` / `compute_precision_grasp_frac` 구현**

`grasp_right_utils.py`에 추가:
```python
def compute_precision_grasp_mask(
    tip_contact_bool: torch.Tensor, min_opposing: int = 2
) -> torch.Tensor:
    """엄지(idx0) 접촉 AND 대향(idx1~4) min_opposing개 이상 접촉.

    Args:
        tip_contact_bool: (N,5) bool — 손끝 접촉 여부(idx0=엄지).
        min_opposing: 엄지 대향으로 요구되는 최소 손가락 수(기본 2).
    Returns:
        (N,) bool — 안정 precision 파지 여부.
    """
    thumb = tip_contact_bool[:, 0]
    opposing = tip_contact_bool[:, 1:].sum(dim=1)
    return thumb & (opposing >= min_opposing)


def compute_precision_grasp_frac(tip_contact_bool: torch.Tensor) -> torch.Tensor:
    """precision 파지 품질 [0,1] (엄지 없으면 0). graded 게이팅용.

    엄지 + 대향 손가락 수를 최대 3접촉(엄지+2)으로 정규화.
    """
    thumb = tip_contact_bool[:, 0].float()
    opposing = tip_contact_bool[:, 1:].sum(dim=1).float()
    frac = (1.0 + opposing) / 3.0
    return (thumb * frac).clamp(0.0, 1.0)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `cd /home/user/rl_ws/hdgp && python -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/test_precision_grasp_mask.py -v`
Expected: 3 PASS.

- [ ] **Step 5: reward-audit 스킬 통과 (성공조건·gate 변경이므로 필수)**

성공조건 gate를 `full_tip_contact(5/5)` → `precision_grasp_mask(엄지+대향2지)`로 완화하는 변경에 대해 reward-audit 실행. 예상 지표 이동: (a) 성공 판정이 쉬워져 초기 success_rate↑, (b) envelope wrap 유인 소멸로 접촉 손가락 수 평균↓, (c) `r_secure`가 더 일찍 활성(hold_gate가 5/5 대신 3점에서 열림). ACCEPT 판정 기록 후 Step 6 진행.

- [ ] **Step 6: 공유 adaptive core의 `hold_gate` 일반화 (하위호환)**

`openarm/common/grasp_adaptive_core.py`의 `compute_adaptive_grip_terms`는 `full_tip_contact`로 `hold_gate`를 만든다(line 62). 이 인자를 **일반적인 "안정 파지 게이트"로 재해석**하되, 다른 태스크가 5/5 bool을 넘기던 것을 깨지 않는다. 인자 이름은 유지하고 docstring만 갱신:
```python
        full_tip_contact: (N,) 안정 파지 게이트 {0,1}. grasp_adapt Phase 1부터는
            precision_grasp_mask(엄지+대향2지)를 넘긴다. 다른 태스크는 5접촉 bool 유지.
```
**코드 로직은 변경하지 않는다** — 호출부(env)가 넘기는 텐서만 precision mask로 바뀐다.

- [ ] **Step 7: `_get_dones`에서 5/5 gate → precision mask 교체**

`grasp_right_env.py:1252`:
```python
# 변경 전: full_tip_contact = self.num_contacts_buf >= NUM_FINGERTIPS
full_tip_contact = compute_precision_grasp_mask(
    self.tip_contact_bool, int(self.cfg.precision_min_opposing)
)
```
- `self.tip_contact_bool` `(N,5)`가 env에 이미 있는지 확인(없으면 `_update_contact_forces`에서 `binary_contact` 계산 시점의 bool 텐서를 버퍼로 노출). 변수명 `full_tip_contact`는 하위 라인(1261/1270/1277)이 그대로 참조하므로 **유지**한다(의미만 precision으로 바뀜). 주석 추가: `# Phase 1: 5/5 envelope 강제 제거 → 엄지+대향2지 precision 파지`.
- `grasp_right_utils`에서 `compute_precision_grasp_mask` import 추가.

- [ ] **Step 8: `_get_rewards` success_bonus의 5/5 gate 교체**

`grasp_right_env.py:1140-1154`의 `height_hold_success_bonus`가 곱하는 `full_tip_contact`를 동일한 precision mask 결과로 교체(같은 변수 재사용 가능하도록 `_get_rewards`에서도 `compute_precision_grasp_mask` 계산 또는 `_get_dones`와 공유). adaptive core에 넘기는 `full_tip_contact=` 인자도 precision mask로 전달.

- [ ] **Step 9: cfg — lift latch·ADR을 precision 기준으로 조정**

`grasp_right_env_cfg.py`:
- `stage0_lift_start_min_contacts: int = 5` → `= 3` (엄지+2지 = 3접촉에서 lift latch 진입).
- 신규 필드 추가: `precision_min_opposing: int = 2  # 엄지 대향 최소 손가락 수`.
- contact ADR `contact_adr_custom_cfg`의 `"min_contacts": (2.0, 5.0)` → `(2.0, 3.0)` (5접촉으로 몰지 않음). 또는 설계 문서 §9대로 contact ADR을 제거하고 general difficulty ADR만 유지 — reward-audit에서 결정.

- [ ] **Step 10: gate wiring 테스트 (신규, 정적)**

Create `.../grasp_adapt/tests/test_precision_grasp_gate_wiring.py`:
```python
"""precision mask가 lift/success gate 로직과 정합하는지(순수 로직) 검증."""
import torch
from openarm.tesollo.right.grasp_adapt.grasp_right_utils import (
    compute_precision_grasp_mask,
    compute_precision_grasp_frac,
)


def test_frac_zero_without_thumb():
    tip = torch.tensor([[False, True, True, True, True]])
    assert float(compute_precision_grasp_frac(tip)[0]) == 0.0


def test_frac_full_at_three_contacts():
    tip = torch.tensor([[True, True, True, False, False]])
    assert abs(float(compute_precision_grasp_frac(tip)[0]) - 1.0) < 1e-6


def test_mask_and_frac_agree_on_boundary():
    # 엄지+대향2지: mask True, frac==1.0
    tip = torch.tensor([[True, True, True, False, False]])
    assert bool(compute_precision_grasp_mask(tip)[0]) is True
    assert float(compute_precision_grasp_frac(tip)[0]) >= 1.0 - 1e-6
    # 엄지+대향1지: mask False
    tip1 = torch.tensor([[True, True, False, False, False]])
    assert bool(compute_precision_grasp_mask(tip1)[0]) is False
```

- [ ] **Step 11: 전체 정적 테스트 + AST 검증**

Run: `cd /home/user/rl_ws/hdgp && python -m pytest source/openarm/openarm/tesollo/right/grasp_adapt/tests/ -v && python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('source/openarm/openarm/tesollo/right/grasp_adapt/*.py')]; print('AST OK')"`
Expected: 모두 PASS, AST OK.

- [ ] **Step 12: 커밋**

```bash
cd /home/user/rl_ws
git add -A hdgp/source/openarm/openarm/tesollo/right/grasp_adapt/ hdgp/source/openarm/openarm/common/grasp_adaptive_core.py
git commit -m "feat(grasp_adapt): Phase 1 — 5/5 envelope gate → 엄지+대향2지 fingertip precision 파지"
```

- [ ] **Step 13: 학습 스냅샷 기록 + GPU 학습 실행 (server)**

로컬에서 push, server에서 pull 후 학습:
```bash
# server (oem@10.102.101.240), conda proj-hdgp-py311
CUDA_VISIBLE_DEVICES=0 NOTE="Phase1 precision grasp fresh" \
  ./train.sh open-tesol_r_grasp_adapt-lstm precision_test1 --num_envs 2048 --headless
```
`train.sh`가 `record_test_snapshot.py`로 스냅샷 자동 기록.

### Phase 1 검증 게이트 (TFEvents 근거) — **Phase 2 진입 exit 기준**
`parse_tfevents.py`로 다음을 확인하고 `analysis.md`에 누적. 아래를 만족하면 **완벽한 정밀 파지까지 다듬지 말고 즉시 Phase 2로**(힘/최소접촉 미세조정은 tactile reward가 붙는 Phase 2 이후 자연히 다뤄짐):
- **손끝-only 확립(핵심):** `task/middle_contact_rate`가 낮게 수렴(감싸기 억제됨). envelope penalty 도입 전 대비 하락. — 손끝 파지의 직접 증거.
- **파지·리프트 성립:** `task/success_rate` 유의미(>0.5, 컵 난이도 의존), `cup/height_delta` 10cm 도달, `task/precision_grasp_rate` 상승.
- `reward/r_secure` 활성, `reward/envelope`가 0으로 수렴(감싸기 소멸).
- play.py 렌더링에서 **손끝으로 집는지**(중간마디로 감싸지 않는지) 육안 확인 — hdgp 증거 우선순위 2.
- **실패 시**: middle_contact_rate가 안 내려가면 `envelope_penalty_weight`↑(4.0→). 손끝만으로 못 들면(success↓) 컵 무게/마찰 재검토. TFEvents 근거로 조정, Phase 2로 넘어가지 않는다.

**주의(진행 이력):** 최초 학습(`lstm_test1`)은 USD 통일 네이밍 이관 누락으로 크래시 → 이관 후 `lstm_test2`는 학습됐으나 손끝-only 미반영(success 0.70·precision 0.82이나 five_tip 0.72로 감쌈 가능). envelope penalty 반영본으로 재학습이 실제 Phase 1 검증 대상.

---

# Phase 2 — Stage A Fragile: Virtual Damage 모델 (스펙 개요)

**진입 게이트:** Phase 1이 fingertip precision 파지로 컵 lift/hold에 성공(위 게이트 통과).

**스펙 (설계 문서 §3 Stage A, §4, §6 damage 항):**
- 기존 `cup_big_sdf.usd` rigid + ContactSensor 유지(128~512 envs, 재사용).
- 접촉력으로 **가상 변형량·peak pressure·누적 damage dose** 계산:
  - `p_i = Fn_i / (A_i + eps)` (contact area 알면) 또는 `Fn_i / F_safe_i` (모르면).
  - `D_{t+1} = D_t + dt·[relu((p_max − p_yield)/p_yield)]^q`.
  - 물성 초기값: `F_safe = 0.6~0.8 × F_yield` (실측 전 placeholder, randomization 대상).
- 신규 reward 항(설계 §6): `r_damage`(local 국부 힘 제한), `r_concentration`(힘 집중 억제), `r_delta_force`(힘 변화율). **손가락별 국부 최대 힘** 제한이 핵심 — 총 파지력만 제한하지 않는다.
- 신규 종료 조건: `p_max > p_buckle` 또는 `D > 1` 또는 permanent deformation > limit.
- **차원 영향:** damage/pressure 지표는 우선 critic obs(privileged)에만 추가. actor는 기존 tactile 힘 15D로 추론.

**수용 기준:** damage violation rate < 2%, peak force overshoot < 10%, precision 파지 성공률 유지. 상세 task는 Phase 1 결과를 근거로 `2026-XX-XX-grasp-adapt-phase2-fragile-damage.md`로 확장.

**리스크:** 새 reward 항 3개 동시 도입은 hdgp "한 번에 하나의 가설" 위반 → damage → concentration → delta_force 순차 도입, 각 reward-audit 통과.

---

# Phase 3 — Tactile Residual Adapter (스펙 개요)

**진입 게이트:** Phase 2 fragile 파지가 damage 제약 하에서 안정.

**스펙 (설계 문서 §2, §13):**
- 정책 구조: base grasp policy(palm pose + nominal finger pose) **freeze** + tactile residual adapter만 학습.
- action 단계화: 1단계 = 5D finger closure residual → 2단계 = 5D + global grip gain → 최종 = 20D joint residual.
- **차원 영향:** action 재정의 → 재학습 필수. base freeze 메커니즘은 rl_games 체크포인트 로드 + 부분 파라미터 동결 필요(인프라 조사 선행).

**수용 기준:** frozen-base 대비 residual이 force settling / damage rate를 개선. Ablation(§11): proprioception only vs binary vs 3-axis tactile vs +temporal vs +damage constraint.

**리스크:** rl_games에서 partial-freeze warm-start가 표준 지원되지 않을 수 있음 → 인프라 spike 필요. 별도 계획 첫 task로 편성.

---

# Phase 4 — 시간 정보 (스펙 개요)

**진입 게이트:** Phase 3 residual adapter 검증.

**스펙 (설계 문서 §5 시간 정보):** `history_length=1` → 4–8 frame stacking 또는 GRU 64–128 hidden. 60Hz 기준 6프레임 ≈ 100ms 접촉 변화. 메모리 "distillation RNN 윈도우 BPTT" 교훈 반영(BPTT=1 하드코딩 함정 주의, `seq_length=16`은 이미 LSTM cfg에 존재).

**수용 기준:** temporal 정보가 slip 예측/force settling을 개선(ablation §11의 4번). LSTM cfg 재사용 우선(신규 GRU보다).

---

# Phase 5 — Damage Constraint (Lagrangian) (스펙 개요)

**진입 게이트:** Phase 4 완료.

**스펙 (설계 문서 §6 누적 damage, §12 참고문헌 8 CPO):** damage를 penalty가 아니라 **constraint**로 분리. PPO 유지 시 Lagrangian `r' = r − λ·damage_cost`, λ 자동 조정. reward-audit에서 penalty vs constraint 트레이드오프 명시.

**수용 기준:** 같은 성공률에서 damage violation rate가 penalty-only 대비 감소.

---

# Phase 6 — Staged Curriculum + Segmented Shell (스펙 개요)

**진입 게이트:** Phase 5 완료.

**스펙 (설계 문서 §8 커리큘럼, §3 Stage B, §9 ADR):**
- Stage 0(force/contact calibration, palm 고정) → Stage 1(fragile contact, lift 없음) → Stage 2(lift·hold) → Stage 3(hidden mass/friction) → Stage 4(fragility randomization, `F_req < F_safe` 조합부터) → Stage 5(segmented compliant shell + 외란).
- Stage B segmented shell: 컵 벽 12~16 rigid panel + compliant joint. 변형 지표(radial displacement, hinge angle). 32~128 envs.
- ADR curriculum score: `score = min(worst_bin_success, 1 − damage_rate, 1 − drop_rate)`. 평균이 아니라 **worst-bin**으로 진행 판단.
- 학습량 시작점(설계 §10): Stage 0 3~5M, Stage 1 5~10M, Stage 2 10~20M, Stage 3 20~40M, Stage 4 20~50M.

**수용 기준(설계 §11):** unseen-object 성공률, damage violation rate, worst-bin success ≥ 80%, permanent deformation, force settling time. Stage C(FEM)는 최종 검증·영상용으로 제한적.

---

## Self-Review (spec 대비 커버리지)

- **설계 §1 진단 / §13 구현 우선순위:** Phase 0(정리) + Phase 1(5/5 gate 제거) + Phase 2(damage) + Phase 3(residual+bead-mass hidden — 이미 grasp_adapt에 `NoActorMass`로 존재) 커버.
- **설계 §2 정책 구조:** Phase 3.
- **설계 §3 환경(Stage A/B/C):** Stage A = Phase 2, Stage B = Phase 6, Stage C = Phase 6 비고.
- **설계 §5 촉각/시간:** 3축 힘 15D 이미 존재(Phase 0에서 확인), 시간 = Phase 4.
- **설계 §6 보상:** slip/under-force는 adaptive core에 이미 존재, damage/concentration/delta_force = Phase 2, damage dose constraint = Phase 5.
- **설계 §7 성공조건(엄지+대향2지):** Phase 1 ✅ (핵심).
- **설계 §8 커리큘럼 / §9 ADR / §10 학습량:** Phase 6.
- **설계 §11 평가/ablation:** 각 Phase 검증 게이트 + Phase 3/4 ablation.
- **사용자 명시 요구(grasp_v1 구조 베이스 + 팁 파지):** Phase 0(구조) + Phase 1(팁 파지) ✅.
- **미커버/의도적 지연:** 실제 종이컵 물성 측정(설계 §4)은 물리 실험 — 코드 계획 밖, Phase 2 placeholder를 실측으로 교체하는 후속 작업으로 표기.
- **Placeholder 스캔:** Phase 0~1은 실제 코드/명령/기대출력 포함. Phase 2~6은 의도적으로 스펙 수준 — RL 특성상 이전 Phase 결과 의존이라 지금 bite-sized화하면 허구가 되므로, 착수 시 별도 상세 계획으로 확장한다고 명시.
