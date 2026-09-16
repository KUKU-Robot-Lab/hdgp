# IKER 2단계 t2r 보상 생성 + 놓음 판정 구현 계획 (축소판)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2단계(신발을 선반에 놓기)가 "내려놓고 손을 떼는" 과제가 되도록 env 를 바꾸고, 그 보상을 t2r 로 생성한다.

**Architecture:** IKER 에서는 **VLM 목표 좌표만** 쓴다(루프·고정 5항 보상 없음). env 는 `IkerShoeEnv` 를 상속해 액션 1칸(그립)·관측 1칸을 더하고 성공 판정을 4조건으로 바꾸며, 보상은 생성 코드에서 읽는다. 라운드 반복은 자동 루프가 아니라 **사람이 손으로** 돈다.

**Tech Stack:** Isaac Lab (DirectRLEnv), rl_games PPO, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-iker-stage2-t2r-design.md` — **§0 범위 축소가 §2·§8·§12·§13 에 우선한다.**

## Global Constraints

- 로봇 값은 벤더·프로필에서만. 생산 코드에 `l_hj_`·`r_hj_` 관절명 리터럴 금지.
- 공유 프로필 `tesollo_left_short` 수정 금지.
- 1단계 자산 불변: `grasp_stage.py`, `iker_shoe_grasp_env.py`, `t2r/`(1단계 포크), 뱅크·수확, VLM 목표·게이트.
- **루프 파일 수정 금지**: `modules/iker/loop_state.py`·`loop_probe.py`·`scripts/iker/loop.py`·`LOOP_PROMPT.md`. 정책 키를 추가하지 않는다(추가하면 기존 트랙 상태 파일이 `validate_state` 에서 로드 불가가 된다).
- 새 트랙·크론을 만들지 않는다.
- `git add -A`·`git add .` 금지. 커밋은 경로 명시 + 별도 Bash 호출. `--no-verify` 금지.
- 커밋 트레일러: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- 학습 종료는 PID 로만. `pkill`·`killall` 금지. 로그는 `log/` 밑에만. 저장소 루트에 `.sh` 금지.
- Isaac 실행은 `/home/user/rl_ws/IsaacLab/_isaac_sim/python.sh` + `PYTHONUNBUFFERED=1` + `RUN_LABEL` + `PYTHONPATH=source/openarm`.
- 상수 출처: `EPISODE_STEPS=200`, `CONTROL_DT=0.1`, `DECIMATION=12`, `action_pos_scale=0.02`, `action_rot_scale=0.05`, `eval_success_distance_m=0.05`, `HOLD_RADIUS_M=0.15`, `layout.RACK_TOP_Z=0.325`, `layout.TABLE_TOP_Z=0.205`, `gs.HAND_EMA_ALPHA=0.468559`, `gs.SURFACE_POINT_COUNT=256`.

---

## File Structure

| 파일 | 책임 | 과제 |
|---|---|---|
| `scripts/iker/t2r2_smoke.py` | `--mode settle` 게이트 | 1 |
| `tasks/iker_shoe/place_stage.py` | 성공 판정(연속 4조건) 순수 함수 | 2 |
| `tasks/iker_shoe/iker_shoe_t2r_env.py` · `_cfg.py` | 그립 축·관측 39·판정·생성 보상 | 2 |
| `tasks/iker_shoe/config/__init__.py` | gym id 등록(수정) | 2 |
| `scripts/iker/eval_iker.py` | `placed`·`retreated` 보고 지표(수정) | 2 |
| `tasks/iker_shoe/t2r2/{__init__,context,loader,validator,prompts}.py` | 보상 생성 세트 | 3 |
| `scripts/iker/t2r2_reward.py` | `render` / `ingest` CLI | 3 |

**반복 절차(코드 아님, 사람이 실행):** `t2r2_reward.py render` → 생성기에게 프롬프트 전달 → `t2r2_reward.py ingest` → 학습 기동 → 평가 판독 → 다시 `render`(피드백 포함).

---

### Task 1: 정착 게이트 — 놓은 뒤 10초 서 있는가

**상태: 진행 중.** 초판(커밋 `6a3ee6a7`)은 측정 루프가 `env.step()` 을 써서 RL 종료·리셋 경로를 탔다 — 신발을 목표에 정확히 놓으면 `d_target ≈ 0.018 < success_tolerance 0.10` 이라 약 21스텝째 성공 종료 후 `_reset_idx` 가 신발을 뱅크 파지 자세로 되돌린다(뱅크 시작 자세의 키포인트 거리 중앙값 232.8 mm, 측정값 291.7 mm). fix round 1 진행 중.

**Files:** Modify `scripts/iker/t2r2_smoke.py`

- [ ] **Step 1: 물리만 진행하도록 고친다**

`env.step()` 대신 저장소의 기존 패턴(`scripts/iker/env_smoke.py:102-104`)을 쓴다:

```python
env.scene.write_data_to_sim()
env.sim.step(render=False)
env.scene.update(env.physics_dt)
```

창은 `100 * env.cfg.decimation`(=1200) 물리 스텝 = 10 s. 키포인트 거리는 `_keypoint_distance`(`_get_dones` 에서만 갱신) 대신 `transform_keypoints`(`openarm.agnostic.modules.iker.reward`)로 직접 계산한다. 시작값은 teleport 직후 1스텝, 끝값은 창 종료 후. z 낙하량도 같이 출력한다.

- [ ] **Step 2: 게이트를 돌린다**

```bash
cd /home/user/rl_ws/hdgp-iker && RUN_LABEL=iker_s2t2r_settle PYTHONUNBUFFERED=1 PYTHONPATH=source/openarm \
  /home/user/rl_ws/IsaacLab/_isaac_sim/python.sh scripts/iker/t2r2_smoke.py --mode settle --headless \
  2>&1 | tee log/iker_loop/settle_gate.log | tail -5
```
Expected: `T2R2 SMOKE settle: median <50 mm …` · `T2R2 SMOKE passed True`

- [ ] **Step 3: 실패하면 멈춘다**

`passed False` 면 **Task 2·3 을 시작하지 않는다.** 목표 자세가 신발을 못 받친다는 뜻이고, §3 판정이 달성 불가다. 측정값(중앙값·drift·z 낙하)을 사용자에게 보고하고 목표 자세 재선정 결정을 받는다. 문턱을 조정해 통과시키지 않는다.

- [ ] **Step 4: 커밋**

```bash
git add -- scripts/iker/t2r2_smoke.py
git commit -m "fix(iker): 정착 게이트가 리셋 대신 물리만 진행하도록"
```

---

### Task 2: env — 성공 판정, 그립 축, 관측, 생성 보상 훅, 평가 지표

**Files:**
- Create: `tasks/iker_shoe/place_stage.py`, `tasks/iker_shoe/iker_shoe_t2r_env.py`, `tasks/iker_shoe/iker_shoe_t2r_env_cfg.py`
- Modify: `tasks/iker_shoe/config/__init__.py`, `scripts/iker/eval_iker.py`
- Test: `tasks/iker_shoe/tests/test_place_stage.py`, `tasks/iker_shoe/tests/test_t2r2_env_contract.py`

**Interfaces:**
- Consumes: `IkerShoeEnv`, `IkerShoeEnvCfg`, `gs.hand_targets`, `gs.normalized_targets`, `gs.surface_subsample`, `t2r2.loader`(Task 3 — 없으면 부팅 시 영 보상 경로만 탄다)
- Produces: `PlaceRewardCfg`, `place_step(keypoint_dist, palm_shoe_dist, shoe_bottom_z, shoe_speed, stable_count, cfg) -> PlaceStep`, `IkerShoeT2rEnv`, `IkerShoeT2rEnvCfg(action_space=7, observation_space=39, reward_code_path="")`, gym id `open-sens_l_iker_shoe_t2r`(+`-play`)

- [ ] **Step 1: 판정 테스트를 쓴다**

`tests/test_place_stage.py`:

```python
import torch

from openarm.agnostic.tasks.iker_shoe import place_stage as ps


def test_all_four_conditions_must_hold():
    cfg = ps.PlaceRewardCfg()
    keypoint_dist = torch.tensor([0.01, 0.20, 0.01, 0.01])   # 2번만 멀다
    palm_shoe = torch.tensor([0.30, 0.30, 0.05, 0.30])       # 3번만 아직 손에 있다
    bottom_z = torch.full((4,), cfg.rack_top_z)
    speed = torch.tensor([0.0, 0.0, 0.0, 1.0])               # 4번만 움직인다
    step = ps.place_step(keypoint_dist, palm_shoe, bottom_z, speed, torch.zeros(4), cfg)
    assert step.ok.tolist() == [True, False, False, False]


def test_counter_is_consecutive_not_cumulative():
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(1)
    for _ in range(5):
        count = ps.place_step(*good, count, cfg).stable_count
    assert count.tolist() == [5.0]
    count = ps.place_step(*bad, count, cfg).stable_count
    assert count.tolist() == [0.0], "조건이 깨지면 0 으로 돌아가야 한다(기존 IKER 는 누적이었다)"


def test_success_needs_stable_steps_in_a_row():
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(1)
    for _ in range(cfg.stable_steps - 1):
        step = ps.place_step(*good, count, cfg)
        count = step.stable_count
        assert not bool(step.success)
    assert bool(ps.place_step(*good, count, cfg).success)


def test_resting_rejects_floating_shoes():
    cfg = ps.PlaceRewardCfg()
    floating = torch.tensor([cfg.rack_top_z + 10 * cfg.resting_tol, cfg.rack_top_z])
    step = ps.place_step(torch.tensor([0.01, 0.01]), torch.tensor([0.30, 0.30]),
                         floating, torch.zeros(2), torch.zeros(2), cfg)
    assert step.resting.tolist() == [False, True]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: place_stage`

- [ ] **Step 3: 판정을 구현한다**

`place_stage.py`:

```python
"""2단계 놓기 성공 판정 (스펙 2026-09-16-iker-stage2-t2r §3).

네 조건을 연속 stable_steps 스텝 만족하면 성공이다. 조건이 깨지면 카운터는 0 으로 돌아간다 —
기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import layout


@dataclass
class PlaceRewardCfg:
    place_tolerance: float = 0.05      # 키포인트 4개 평균 거리 [m]
    release_radius: float = 0.15       # palm-shoe 거리가 이 값을 넘어야 "손을 뗐다" [m]
    rack_top_z: float = layout.RACK_TOP_Z
    resting_tol: float = 0.01          # hull 최저점이 받침 상면에서 벗어날 수 있는 폭 [m]
    still_speed: float = 0.05          # 신발 속도 [m/s]
    stable_steps: int = 20             # 연속 만족 스텝 수


@dataclass(frozen=True)
class PlaceStep:
    placed: torch.Tensor
    released: torch.Tensor
    resting: torch.Tensor
    still: torch.Tensor
    ok: torch.Tensor
    stable_count: torch.Tensor
    success: torch.Tensor


def place_step(
    keypoint_dist: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    shoe_bottom_z: torch.Tensor,
    shoe_speed: torch.Tensor,
    stable_count: torch.Tensor,
    cfg: PlaceRewardCfg,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    ok = placed & released & resting & still
    count = torch.where(ok, stable_count + 1.0, torch.zeros_like(stable_count))
    return PlaceStep(placed, released, resting, still, ok, count, count >= float(cfg.stable_steps))
```

- [ ] **Step 4: 판정 테스트 통과를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py -q -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: env 계약 테스트를 쓴다**

`tests/test_t2r2_env_contract.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "iker_shoe_t2r_env.py").read_text(encoding="utf-8")
CFG = (ROOT / "iker_shoe_t2r_env_cfg.py").read_text(encoding="utf-8")
REG = (ROOT / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_subclasses_the_stage2_env_and_overrides_only_the_listed_hooks():
    assert "class IkerShoeT2rEnv(IkerShoeEnv)" in ENV
    allowed = {"__init__", "_pre_physics_step", "_get_observations", "_get_dones",
               "_get_rewards", "_build_context", "_reset_idx"}
    names = {line.split("def ")[1].split("(")[0] for line in ENV.splitlines() if line.strip().startswith("def ")}
    assert names <= allowed, names - allowed


def test_action_is_seven_and_observation_is_thirtynine():
    assert "action_space = 7" in CFG and "observation_space = 39" in CFG
    assert 'reward_code_path: str = ""' in CFG and "(IkerShoeEnvCfg)" in CFG


def test_grip_axis_uses_the_stage1_hand_law_not_a_new_one():
    assert "gs.hand_targets(" in ENV and "gs.normalized_targets(" in ENV and "actions[:, 6]" in ENV


def test_reward_comes_from_generated_code_and_place_flags_are_logged():
    block = ENV.split("def _get_rewards")[1]
    assert "call_reward_fn" in block and "nan_to_num" in block
    assert "t2r_reward/total" in ENV and "place/placed" in ENV and "place/released" in ENV


def test_context_tensors_are_copies():
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in ENV.split("def _build_context")[1]


def test_reset_zeroes_previous_actions_and_restores_the_bank_grip():
    block = ENV.split("def _reset_idx")[1]
    assert "_t2r_prev_actions" in block and "_grip_targets" in block and "_stable_count" in block


def test_registration_adds_the_t2r_ids():
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env:IkerShoeT2rEnv" in REG
    assert 'f"open-sens_l_iker_shoe_t2r{_suffix}"' in REG
```

- [ ] **Step 6: 실패를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_env_contract.py -q -p no:cacheprovider`
Expected: FAIL — `FileNotFoundError: iker_shoe_t2r_env.py`

- [ ] **Step 7: cfg 를 쓴다**

```python
"""cfg of the stage-2 t2r environment (spec 2026-09-16-iker-stage2-t2r §4, §6)."""
from __future__ import annotations

from isaaclab.utils import configclass

from .iker_shoe_env_cfg import IkerShoeEnvCfg
from .place_stage import PlaceRewardCfg


@configclass
class IkerShoeT2rEnvCfg(IkerShoeEnvCfg):
    action_space = 7        # 6 palm delta + 1 grip axis
    observation_space = 39  # the stage-2 38 + the filtered grip state
    reward_code_path: str = ""   # "" = zero reward (boot and random rollouts only)
    place: PlaceRewardCfg = PlaceRewardCfg()


@configclass
class IkerShoeT2rPlayEnvCfg(IkerShoeT2rEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
```

- [ ] **Step 8: env 를 쓴다**

`iker_shoe_t2r_env.py` — 오버라이드는 계약 테스트의 7개뿐이다.

- `__init__`: `load_reward_fn(cfg.reward_code_path)` → `self._reward_fn, self._reward_origin`; `gs.surface_subsample(meta[...]["hull_local"])` → `self._surface_local`; `self._hand_ids`(프로필의 손 관절 이름으로 조회), `self._grip_targets`(뱅크 손 목표), `self._stable_count`, `self._t2r_prev_actions`. 부팅 줄에 보상 코드 경로·sha256 을 찍는다.
- `_pre_physics_step`: 부모와 같은 팔 6축 IK 뒤, `a = actions[:, 6]` 으로
  `raw = grip_pose + (a[:, None] + 1) / 2 * (open_pose - grip_pose)` → `gs.hand_targets(...)`(α=`gs.HAND_EMA_ALPHA`) → `self._joint_targets[:, self._hand_ids]`.
- `_get_observations`: 부모 38칸 + `gs.normalized_targets(...)` 1칸.
- `_get_dones`: `place_step(...)` 으로 네 조건·연속 카운터·성공. 종료 = 성공 | 낙하 | 시간초과.
- `_get_rewards`: `_build_context()` → `call_reward_fn` → `torch.nan_to_num`. 로그 `t2r_reward/<항>`·`t2r_reward/total`·`place/placed`·`place/released`·`place/resting`.
- `_build_context`: 스펙 §5 필드, 모든 텐서 `clone`.
- `_reset_idx`: 부모 리셋 뒤 `_t2r_prev_actions[ids]=0`, `_grip_targets[ids]`=뱅크 손 목표, `_stable_count[ids]=0`.

- [ ] **Step 9: 등록을 더한다**

`config/__init__.py` 끝에 (기존 3블록과 같은 형태):

```python
# Stage-2 placement with a t2r-generated reward (spec 2026-09-16-iker-stage2-t2r).
_T2R_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env:IkerShoeT2rEnv"
_T2R_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeT2rEnvCfg"), ("-play", "IkerShoeT2rPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_t2r{_suffix}",
        entry_point=_T2R_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_T2R_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )
```

- [ ] **Step 10: 평가 지표를 더한다**

`scripts/iker/eval_iker.py`: `RETREAT_M = 0.15` 상수와 두 함수를 더하고, `ROW_KEYS` 에 `"palm_start_dist"` 를 넣고, `_record` 의 `values` 에 `"palm_start_dist": (palm - self.palm_start[finished]).norm(dim=-1)` 를 추가한다(`self.palm_start` 는 리셋 직후 팔바닥 위치).

```python
RETREAT_M = 0.15  # 종료 시 팔바닥이 시작 자세에서 이만큼 멀어지면 "물러났다"


def placed_mask(near: torch.Tensor, palm_shoe: torch.Tensor) -> torch.Tensor:
    return near & (palm_shoe > HOLD_RADIUS_M)


def retreated_mask(palm_start_dist: torch.Tensor) -> torch.Tensor:
    return palm_start_dist >= RETREAT_M
```

`summarize` 반환 dict 최상위에:

```python
"placed": frac(placed_mask(table["end_dist"] <= distance, table["palm_shoe"])),
"retreated": frac(retreated_mask(table["palm_start_dist"])),
```

- [ ] **Step 11: 전체 테스트 통과를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/ -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 12: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/place_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_t2r_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_t2r_env_cfg.py source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py scripts/iker/eval_iker.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/
git commit -m "feat(iker): 2단계 env — 놓음 판정·그립 축·관측 39·생성 보상 훅"
```

---

### Task 3: t2r 최소 세트 — 컨텍스트·검증기·프롬프트·CLI

**Files:**
- Create: `tasks/iker_shoe/t2r2/{__init__,context,loader,validator,prompts}.py`, `scripts/iker/t2r2_reward.py`
- Test: `tasks/iker_shoe/tests/test_t2r2_reward_set.py`

**Interfaces:**
- Consumes: `place_stage.PlaceRewardCfg`(Task 2), `layout`, `gs.SURFACE_POINT_COUNT`
- Produces: `RewardContext`·`TENSOR_FIELDS`·`SCALAR_FIELDS`·`context_stub_source()`; `ENTRY_NAME`·`zero_reward`·`load_reward_fn`·`call_reward_fn`; `static_check`·`make_fake_context`·`dry_run`·`validate`·`cuda_usable`; `render_prompt`·`task_text`·`render_feedback_table`·`FEEDBACK_TAG_PREFIXES`; CLI `render`·`ingest`

- [ ] **Step 1: 테스트를 쓴다**

`tests/test_t2r2_reward_set.py`:

```python
import torch

from openarm.agnostic.tasks.iker_shoe.place_stage import PlaceRewardCfg
from openarm.agnostic.tasks.iker_shoe.t2r2 import context as ctxmod
from openarm.agnostic.tasks.iker_shoe.t2r2 import loader, prompts, validator

GOOD = """
import torch

def compute_reward(ctx):
    near = 1.0 - torch.tanh(5.0 * ctx.keypoint_dist)
    let_go = ctx.released.float()
    return near + let_go, {"near": near, "let_go": let_go}
"""


def test_context_carries_the_stage2_fields_and_no_grasp_fields():
    required = {"grip_norm", "target_keypoints", "keypoints", "keypoint_err", "keypoint_dist",
                "shoe_bottom_z", "palm_shoe_dist", "placed", "released", "resting", "still",
                "stable_count", "success", "actions", "prev_actions"}
    assert required <= set(ctxmod.TENSOR_FIELDS)
    for gone in ("link_shoe_force", "thumb_curl", "dz_free", "hold_count", "latched", "hand_q"):
        assert gone not in ctxmod.TENSOR_FIELDS, gone


def test_good_code_passes_and_unknown_field_is_rejected():
    assert validator.static_check(GOOD).ok
    rep = validator.static_check(GOOD.replace("ctx.keypoint_dist", "ctx.distance_to_goal"))
    assert not rep.ok and any("distance_to_goal" in e for e in rep.errors)


def test_stage1_fields_are_unknown_here():
    rep = validator.static_check(GOOD.replace("ctx.released", "ctx.latched"))
    assert not rep.ok and any("latched" in e for e in rep.errors)


def test_in_place_mutation_and_bad_import_are_rejected():
    assert not validator.static_check(GOOD.replace("near = ", "ctx.keypoint_dist.clamp_(0.0)\n    near = ")).ok
    assert not validator.static_check("import os\n" + GOOD).ok


def test_fake_context_shapes_match_the_action_and_keypoint_counts():
    ctx = validator.make_fake_context(5)
    assert ctx.actions.shape == (5, 7) and ctx.prev_actions.shape == (5, 7)
    assert ctx.target_keypoints.shape == (5, 4, 3) and ctx.keypoint_err.shape == (5, 4)


def test_empty_path_loads_the_zero_reward():
    fn, origin = loader.load_reward_fn("")
    total, terms = loader.call_reward_fn(fn, validator.make_fake_context(3))
    assert total.shape == (3,) and terms == {} and "zero" in origin


def test_prompt_embeds_the_context_and_reads_numbers_from_cfg():
    cfg = PlaceRewardCfg()
    text = prompts.render_prompt(prompts.PromptSpec(task=prompts.task_text(cfg)))
    assert "class RewardContext" in text and "grip_norm" in text
    assert f"{cfg.place_tolerance}" in text and f"{cfg.release_radius}" in text and f"{cfg.stable_steps}" in text
    assert "actions[6]" in text and "actions[6:26]" not in text


def test_task_text_names_all_four_stages():
    for word in ("align", "set it down", "let go", "withdraw"):
        assert word in prompts.task_text(PlaceRewardCfg())
```

- [ ] **Step 2: 실패를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_reward_set.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: …t2r2`

- [ ] **Step 3: 로더를 복사한다**

```bash
mkdir -p source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2
printf '"""IKER 2단계 t2r 포크 (스펙 2026-09-16-iker-stage2-t2r)."""\n' > source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/__init__.py
cp source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/loader.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/loader.py
sed -i 's/stage-1 reward code/stage-2 reward code/' source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/loader.py
```

- [ ] **Step 4: 컨텍스트를 쓴다**

`t2r2/context.py` — 필드 주석이 곧 프롬프트의 환경 설명이므로 **영어로** 쓴다. 필드는 스펙 §5 표 그대로: 상수(`table_top_z`, `rack_x_min/max`, `rack_y_min/max`, `rack_top_z`, `episode_steps`, `control_dt`, `place_tolerance`, `release_radius`, `resting_tol`, `still_speed`, `stable_steps`), 팔·팔바닥(`palm_pos`, `palm_quat`, `palm_normal`, `arm_q`, `arm_qd`), 그립(`grip_norm`), 신발(`shoe_pos`, `shoe_quat`, `shoe_lin_vel`, `shoe_ang_vel`, `shoe_surface`, `shoe_bottom_z`, `palm_gap`, `palm_shoe_dist`), 목표(`target_keypoints`, `keypoints`, `init_keypoints`, `keypoint_err`, `keypoint_dist`), 상태(`placed`, `released`, `resting`, `still`, `stable_count`, `success`, `episode_progress`), 행동(`actions`, `prev_actions` — 둘 다 `(N,7)`). 끝에 1단계와 같은 `num_envs`·`device` property, `TENSOR_FIELDS`/`SCALAR_FIELDS`, `context_stub_source()`.

- [ ] **Step 5: 검증기를 복사하고 두 곳만 고친다**

```bash
cp source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/validator.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/validator.py
```

① 헤더 상수를 `NUM_ARM, NUM_ACTIONS, NUM_KEYPOINTS = 7, 7, 4` 로 바꾸고 손가락·링크 상수를 지운다(임포트에 `from ..place_stage import PlaceRewardCfg` 추가). ② `make_fake_context` 를 §5 필드로 교체하되 상수는 `PlaceRewardCfg()` 에서 읽는다. 나머지(`static_check`·`dry_run`·`validate`·`cuda_usable`)는 **수정하지 않는다** — 필드 무관이다.

- [ ] **Step 6: 프롬프트를 쓴다**

`t2r2/prompts.py` — 1단계 절 순서 유지(역할 → 로봇·행동 → 보상 구조 안내 → 컨텍스트 원문 → 추가 사실 → 과제문·출력 규칙 → 피드백). `REWARD_STRUCTURE`·`OUTPUT_RULES`·`FEEDBACK_HEADER`·`FEEDBACK_TAIL` 은 1단계에서 문구 그대로 가져온다. 바꾸는 것: 로봇·행동에 **그립 축 1칸**(−1 뱅크 grip, +1 프로필 open, EMA `gs.HAND_EMA_ALPHA`)을 넣고 20관절 표는 넣지 않는다; 추가 사실은 스펙 §7 목록만.

```python
FEEDBACK_TAG_PREFIXES = (
    "t2r_reward/", "place/placed", "place/released", "place/resting",
    "iker/success_5cm", "iker/keypoint_distance_m", "iker/dropped", "episode_lengths", "rewards",
)


def task_text(cfg: PlaceRewardCfg) -> str:
    return ("Place the shoe on the rack next to the other shoe: align it with the target keypoints, "
            "set it down, let go, and withdraw the hand.")
```

- [ ] **Step 7: CLI 를 쓴다**

`scripts/iker/t2r2_reward.py` — 두 하위 명령만.

- `render --out prompt.md [--previous compute_reward.py] [--feedback events.json]`: `render_prompt` 결과를 파일로.
- `ingest --response response.md --out compute_reward.py --report validation.json`: 응답의 마지막 python 코드 블록을 꺼내 `validate` 를 돌리고 둘 다 파일로. `ok=false` 면 종료코드 1.

- [ ] **Step 8: 테스트 통과를 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/ -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 9: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/ scripts/iker/t2r2_reward.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_reward_set.py
git commit -m "feat(iker): 2단계 t2r 보상 생성 세트"
```

---

## Self-Review

**1. 스펙 커버리지 (§0 축소 반영)**

| 스펙 절 | 과제 |
|---|---|
| §3 성공 판정 | 2 |
| §4 액션 6→7 · 관측 38→39 | 2 |
| §5 RewardContext | 3 |
| §6 env 훅 | 2 |
| §7 프롬프트 | 3 |
| §9.1 검증 | 3 |
| §9.2 settle 게이트 | 1 |
| §12 중 `eval_iker.py` 지표 | 2 |
| §0 이 뺀 것(§2 루프·§8 라운드·§12 루프파일·§13 전환) | 과제 없음 — 의도된 제외 |

**2. Placeholder 점검** — Task 2 Step 8(env 본문)과 Task 3 Step 4·6·7 은 필드·절 목록으로 적었다. 전부 스펙 §5·§6·§7 에 완전한 표가 있고 계획이 그 위치를 지목한다. 나머지 Step 은 실제 코드.

**3. 타입 일관성** — `place_step(...) -> PlaceStep` 이 Task 2 안에서 일관, `RewardContext` 필드가 Task 2(`_build_context`)·Task 3(context·validator·prompts)에서 동일, `load_reward_fn`/`call_reward_fn` 시그니처가 Task 2·3 에서 동일.
