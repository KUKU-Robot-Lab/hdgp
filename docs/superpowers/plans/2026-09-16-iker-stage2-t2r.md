# IKER 2단계 t2r 보상 생성 + 놓음 판정 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** IKER 2단계(신발을 선반에 놓기)의 보상을 t2r 로 생성하고, 성공 판정에 "내려놓고 손 떼기"를 넣는다.

**Architecture:** 1단계 t2r 포크(`tasks/iker_shoe/t2r/`)를 `t2r2/` 로 복사해 2단계 필드로 갈아끼운다. env 는 기존 `IkerShoeEnv` 를 상속한 `IkerShoeT2rEnv` 가 액션 1칸(그립)·관측 1칸을 더하고, 성공 판정을 연속 4조건으로 바꾸며, 보상만 생성 코드에서 가져온다. 루프는 `stage2_train` 단계 안에 라운드 기계를 넣는다.

**Tech Stack:** Isaac Lab (DirectRLEnv), rl_games PPO, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-iker-stage2-t2r-design.md` (커밋 `063011d7`)

## Global Constraints

- 로봇 값은 벤더·프로필에서만 읽는다. 생산 코드에 `l_hj_`·`r_hj_` 관절명 리터럴 금지 (프로필 `TESOLLO_LEFT_SHORT` 사용).
- 공유 프로필 `tesollo_left_short` 를 수정하지 않는다.
- 1단계 자산을 건드리지 않는다: `grasp_stage.py`, `iker_shoe_grasp_env.py`, `t2r/`(1단계 포크), 뱅크·수확, VLM 목표·게이트.
- IKER 고정 5항(`modules/iker/reward.py`)은 **삭제하지 않는다** — 2단계에서 쓰지 않을 뿐이다.
- reward-audit 를 쓰지 않는다 (IKER·t2r 트랙 사용자 결정).
- `git add -A`·`git add .` 금지. 커밋은 경로를 명시하고 별도 Bash 호출로 한다. `--no-verify` 금지.
- 커밋 트레일러: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- 학습 종료는 PID 로만. `pkill`·`killall` 금지.
- 로그는 `log/` 밑에만. 저장소 루트에 `.sh` 를 만들지 않는다.
- 상수 출처: `EPISODE_STEPS=200`, `CONTROL_DT=0.1`, `action_pos_scale=0.02`, `action_rot_scale=0.05`, `eval_success_distance_m=0.05`, `HOLD_RADIUS_M=0.15`(eval_iker), `layout.RACK_TOP_Z=0.325`, `layout.TABLE_TOP_Z=0.205`, `layout.RACK_X_RANGE=(0.11,0.43)`, `layout.RACK_Y_RANGE=(-0.33,-0.02)`, `gs.HAND_EMA_ALPHA=0.468559`, `gs.SURFACE_POINT_COUNT=256`.

---

## File Structure

| 파일 | 책임 |
|---|---|
| `tasks/iker_shoe/t2r2/context.py` | 2단계 `RewardContext` + `context_stub_source()` |
| `tasks/iker_shoe/t2r2/loader.py` | 1단계 복사 (import 경로만 변경) |
| `tasks/iker_shoe/t2r2/validator.py` | 1단계 복사 + 형상 상수·`make_fake_context` 교체 |
| `tasks/iker_shoe/t2r2/prompts.py` | 2단계 프롬프트 렌더 |
| `tasks/iker_shoe/place_stage.py` | 성공 판정(연속 4조건) 순수 함수 |
| `tasks/iker_shoe/iker_shoe_t2r_env.py` | env 훅 (그립 축·관측·판정·생성 보상) |
| `tasks/iker_shoe/iker_shoe_t2r_env_cfg.py` | cfg + Play cfg |
| `tasks/iker_shoe/config/__init__.py` | gym id 등록 (수정) |
| `scripts/iker/t2r2_smoke.py` | `--mode settle` / `--mode wire` / `--mode round` |
| `scripts/iker/t2r2_reward.py` | `render` / `ingest` / `reflect` CLI |
| `scripts/iker/eval_iker.py` | `placed`·`retreated` 보고 지표 (수정) |
| `modules/iker/loop_state.py` · `loop_probe.py` · `scripts/iker/loop.py` | 라운드 판정·실행기 (수정) |

---

### Task 1: 정착 게이트 — 놓은 뒤 10초 서 있는가

스펙 §9.2 의 **구현 착수 전 게이트**. 목표 자세에서 손을 떼고 10초 뒀을 때 신발이 서 있지 않으면 §3 성공 판정이 달성 불가이므로, 나머지 과제를 시작하기 전에 이것부터 통과해야 한다.

**Files:**
- Create: `scripts/iker/t2r2_smoke.py` (`--mode settle` 만; wire·round 는 Task 9)

**Interfaces:**
- Consumes: `IkerShoeEnv`(기존), `layout.RACK_TOP_Z`, `env._targets`, `env._offsets`, `env._keypoints_local()`
- Produces: `T2R2 SMOKE settle: …` 출력과 `T2R2 SMOKE passed True|False`

- [ ] **Step 1: 게이트 스크립트를 쓴다**

`scripts/iker/t2r2_smoke.py`:

```python
"""IKER 2단계 t2r 스모크. --mode settle 은 구현 착수 전 게이트(스펙 §9.2).

--mode settle: 신발을 VLM 목표 자세로 두고 로봇을 홈으로 치운 뒤 100 스텝(10 s) 물리를 돌려
  키포인트가 5 cm 안에 남는지 본다. 기존 env_smoke 는 env.step 을 한 번(0.1 s)만 돌렸다.
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("settle",), required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=args.headless).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: F401,E402
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

SETTLE_TOLERANCE_M = 0.05


def settle(env, steps: int) -> tuple[list[str], dict]:
    n = env.num_envs
    dev = env.device
    env.reset()
    home = env._robot.data.default_joint_pos.clone()
    env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    env._robot.set_joint_position_target(home)
    env._joint_targets[:] = home

    rot, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = torch.tensor(centre, dtype=torch.float32, device=dev) + env.scene.env_origins
    pose[:, 3:] = quat_from_matrix(torch.tensor(rot, dtype=torch.float32, device=dev)).expand(n, 4)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    zero = torch.zeros(n, env.cfg.action_space, device=dev)
    start = None
    for _ in range(steps):
        env.step(zero)
        if start is None:
            start = env._keypoint_distance.clone()
    end = env._keypoint_distance
    median = float(end.median())
    drift = float((end - start).median())
    failures = []
    if not median <= SETTLE_TOLERANCE_M:
        failures.append(f"settle: median keypoint distance {median:.4f} m > {SETTLE_TOLERANCE_M}")
    print(f"T2R2 SMOKE settle: median {median*1000:.1f} mm, drift {drift*1000:+.1f} mm over {steps} steps", flush=True)
    return failures, {"median_m": median, "drift_m": drift, "steps": steps}


def main() -> int:
    env = gym.make("open-sens_l_iker_shoe", cfg=None, num_envs=args.num_envs).unwrapped
    failures, details = settle(env, args.steps)
    for failure in failures:
        print(f"T2R2 SMOKE CHECK FAILED: {failure}", flush=True)
    passed = not failures
    print(f"T2R2 SMOKE passed {passed}", flush=True)
    env.close()
    return 0 if passed else 1


code = main()
app.close()
sys.exit(code)
```

- [ ] **Step 2: 게이트를 돌린다**

Run:
```bash
cd /home/user/rl_ws/hdgp-iker && RUN_LABEL=iker_s2t2r_settle PYTHONUNBUFFERED=1 \
  /home/user/rl_ws/IsaacLab/_isaac_sim/python.sh scripts/iker/t2r2_smoke.py --mode settle --headless \
  2>&1 | tee log/iker_loop/settle_gate.log | tail -5
```
Expected: `T2R2 SMOKE settle: median <50 mm …` 와 `T2R2 SMOKE passed True`

- [ ] **Step 3: 실패하면 멈춘다**

`passed False` 면 **이 계획의 나머지를 진행하지 않는다.** 목표 자세가 10초를 못 버틴다는 뜻이고, §3 판정이 달성 불가다. 측정값(중앙값·drift)을 사용자에게 보고하고 목표 자세(받침 칸) 재선정 결정을 받는다.

- [ ] **Step 4: 커밋**

```bash
git add -- scripts/iker/t2r2_smoke.py
git commit -m "feat(iker): 2단계 정착 게이트 — 놓은 뒤 10초 서 있는지 검사"
```

---

### Task 2: 성공 판정 순수 함수

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/place_stage.py`
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py`

**Interfaces:**
- Produces: `PlaceRewardCfg`, `place_step(...) -> PlaceStep`
  - `PlaceStep` 필드: `placed, released, resting, still, ok, stable_count, success` (모두 `(N,)`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_place_stage.py`:

```python
import torch

from openarm.agnostic.tasks.iker_shoe import place_stage as ps


def _cfg():
    return ps.PlaceRewardCfg()


def test_all_four_conditions_must_hold():
    cfg = _cfg()
    n = 4
    keypoint_dist = torch.tensor([0.01, 0.20, 0.01, 0.01])   # 2번만 멀다
    palm_shoe = torch.tensor([0.30, 0.30, 0.05, 0.30])       # 3번만 아직 손에 있다
    bottom_z = torch.full((n,), cfg.rack_top_z)
    speed = torch.tensor([0.0, 0.0, 0.0, 1.0])               # 4번만 움직인다
    step = ps.place_step(keypoint_dist, palm_shoe, bottom_z, speed, torch.zeros(n), cfg)
    assert step.ok.tolist() == [True, False, False, False]


def test_counter_is_consecutive_not_cumulative():
    cfg = _cfg()
    n = 1
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(n)
    for _ in range(5):
        count = ps.place_step(*good, count, cfg).stable_count
    assert count.tolist() == [5.0]
    count = ps.place_step(*bad, count, cfg).stable_count
    assert count.tolist() == [0.0], "조건이 깨지면 0 으로 돌아가야 한다(기존 IKER 는 누적이었다)"


def test_success_needs_stable_steps_in_a_row():
    cfg = _cfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(1)
    for _ in range(cfg.stable_steps - 1):
        step = ps.place_step(*good, count, cfg)
        count = step.stable_count
        assert not bool(step.success)
    assert bool(ps.place_step(*good, count, cfg).success)


def test_resting_rejects_floating_and_sunk_shoes():
    cfg = _cfg()
    args = (torch.tensor([0.01, 0.01]), torch.tensor([0.30, 0.30]))
    floating = torch.tensor([cfg.rack_top_z + 10 * cfg.resting_tol, cfg.rack_top_z])
    step = ps.place_step(args[0], args[1], floating, torch.zeros(2), torch.zeros(2), cfg)
    assert step.resting.tolist() == [False, True]
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: place_stage`

- [ ] **Step 3: 최소 구현을 쓴다**

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

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py -q -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/place_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_place_stage.py
git commit -m "feat(iker): 2단계 놓기 성공 판정 — 연속 4조건"
```

---

### Task 3: t2r2 컨텍스트와 로더

**Files:**
- Create: `tasks/iker_shoe/t2r2/__init__.py`, `t2r2/context.py`, `t2r2/loader.py`
- Test: `tasks/iker_shoe/tests/test_t2r2_context.py`

**Interfaces:**
- Consumes: `place_stage.PlaceRewardCfg`
- Produces: `RewardContext`(스펙 §5 필드), `TENSOR_FIELDS`, `SCALAR_FIELDS`, `context_stub_source()`, `ENTRY_NAME`, `zero_reward`, `load_reward_fn`, `call_reward_fn`, `RewardFn`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_t2r2_context.py`:

```python
import torch

from openarm.agnostic.tasks.iker_shoe.t2r2 import context as ctxmod
from openarm.agnostic.tasks.iker_shoe.t2r2 import loader


def test_context_carries_the_stage2_fields():
    required = {"grip_norm", "target_keypoints", "keypoints", "keypoint_err", "keypoint_dist",
                "shoe_bottom_z", "palm_shoe_dist", "placed", "released", "resting", "still",
                "stable_count", "success", "actions", "prev_actions"}
    assert required <= set(ctxmod.TENSOR_FIELDS)


def test_scalar_fields_are_plain_numbers():
    assert {"place_tolerance", "release_radius", "rack_top_z", "still_speed", "stable_steps",
            "episode_steps", "control_dt"} <= set(ctxmod.SCALAR_FIELDS)


def test_no_stage1_grasp_fields_leak_in():
    for gone in ("link_shoe_force", "thumb_curl", "dz_free", "hold_count", "latched", "hand_q"):
        assert gone not in ctxmod.TENSOR_FIELDS, gone


def test_stub_source_is_the_class_body_without_helpers():
    src = ctxmod.context_stub_source()
    assert "class RewardContext" in src and "grip_norm" in src and "def num_envs" not in src


def test_empty_path_loads_the_zero_reward():
    fn, origin = loader.load_reward_fn("")
    ctx = ctxmod.RewardContext(**_fake_kwargs(3))
    total, terms = loader.call_reward_fn(fn, ctx)
    assert total.shape == (3,) and terms == {} and "zero" in origin


def _fake_kwargs(n: int) -> dict:
    from openarm.agnostic.tasks.iker_shoe.t2r2 import validator  # Task 4 에서 만든다
    return {f.name: getattr(validator.make_fake_context(n), f.name)
            for f in __import__("dataclasses").fields(ctxmod.RewardContext)}
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_context.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: …t2r2`

- [ ] **Step 3: 컨텍스트를 쓴다**

`t2r2/context.py` (필드 주석이 곧 프롬프트의 환경 설명이므로 **영어로** 쓴다):

```python
"""RewardContext — the only input of a generated stage-2 reward (spec 2026-09-16-iker-stage2-t2r §5).

Contract: every tensor is batched (N, ...) on one device; positions are env-local metres, angles rad;
the field names and comments ARE the prompt's environment description; frozen — the reward only reads it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) ----
    table_top_z: float             # height of the table top [m]
    rack_x_min: float              # the rack's footprint: x from rack_x_min to rack_x_max [m]
    rack_x_max: float
    rack_y_min: float              # ... and y from rack_y_min to rack_y_max [m]
    rack_y_max: float
    rack_top_z: float              # height of the rack's top surface [m]
    episode_steps: int             # the episode ends after this many steps
    control_dt: float              # seconds per step
    place_tolerance: float         # a placed step needs keypoint_dist at most this [m]
    release_radius: float          # ... palm_shoe_dist greater than this [m]
    resting_tol: float             # ... shoe_bottom_z within this of rack_top_z [m]
    still_speed: float             # ... the shoe's speed below this [m/s]
    stable_steps: int              # a success needs stable_count to reach this

    # ---- hand and arm -------------------------------------------
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]
    grip_norm: torch.Tensor        # (N,) filtered grip state: -1 the bank's closed grasp, +1 the open hand

    # ---- shoe ---------------------------------------------------
    shoe_pos: torch.Tensor         # (N,3) shoe reference point (its body origin)
    shoe_quat: torch.Tensor        # (N,4) shoe orientation quaternion (w,x,y,z)
    shoe_lin_vel: torch.Tensor     # (N,3) linear velocity of the shoe's centre of mass [m/s]
    shoe_ang_vel: torch.Tensor     # (N,3) angular velocity of the shoe [rad/s]
    shoe_surface: torch.Tensor     # (N,P,3) points on the shoe's outer surface
    shoe_bottom_z: torch.Tensor    # (N,) height of the lowest surface point [m]
    palm_gap: torch.Tensor         # (N,) distance from palm_pos to the nearest surface point [m]
    palm_shoe_dist: torch.Tensor   # (N,) distance from palm_pos to shoe_pos [m]

    # ---- target -------------------------------------------------
    target_keypoints: torch.Tensor  # (N,4,3) where the four shoe keypoints must end up
    keypoints: torch.Tensor        # (N,4,3) where they are now
    init_keypoints: torch.Tensor   # (N,4,3) where they were at the start of the episode
    keypoint_err: torch.Tensor     # (N,4) distance of each keypoint from its target [m]
    keypoint_dist: torch.Tensor    # (N,) mean of keypoint_err [m]

    # ---- task status: computed by the environment ----------------
    placed: torch.Tensor           # (N,) bool, keypoint_dist <= place_tolerance
    released: torch.Tensor         # (N,) bool, palm_shoe_dist > release_radius
    resting: torch.Tensor          # (N,) bool, the shoe sits on the rack top
    still: torch.Tensor            # (N,) bool, the shoe is not moving
    stable_count: torch.Tensor     # (N,) consecutive steps with all four true (float)
    success: torch.Tensor          # (N,) bool, True on the step a success is counted
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of episode_steps

    # ---- actions -------------------------------------------------
    actions: torch.Tensor          # (N,7) policy action of this step, clipped to [-1,1]
    prev_actions: torch.Tensor     # (N,7) policy action of the previous step (zeros right after a reset)

    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self.palm_pos.shape[0])

    @property
    def device(self) -> torch.device:
        return self.palm_pos.device


TENSOR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RewardContext) if f.type == "torch.Tensor")
SCALAR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RewardContext) if f.type in ("float", "int"))


def context_stub_source() -> str:
    """The class body (fields and comments) — the prompt's environment description."""
    import inspect

    src = inspect.getsource(RewardContext)
    return src[: src.find("    # ----------")].rstrip() + "\n"
```

- [ ] **Step 4: 로더를 복사한다**

`t2r/loader.py` 를 `t2r2/loader.py` 로 복사하고 **한 줄만** 바꾼다: `from .context import RewardContext` 는 그대로이므로 실제로는 docstring 의 "stage-1" 을 "stage-2" 로 고치는 것 외에 변경이 없다. `t2r2/__init__.py` 는 1단계와 같이 한 줄 docstring.

```bash
cp source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/loader.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/loader.py
sed -i 's/stage-1 reward code/stage-2 reward code/' source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/loader.py
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_context.py -q -p no:cacheprovider`
Expected: PASS (Task 4 의 validator 가 아직 없으면 `test_empty_path_loads_the_zero_reward` 만 실패 — Task 4 에서 통과시킨다)

- [ ] **Step 6: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/ source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_context.py
git commit -m "feat(iker): 2단계 t2r RewardContext 와 로더"
```

---

### Task 4: t2r2 검증기

**Files:**
- Create: `tasks/iker_shoe/t2r2/validator.py`
- Test: `tasks/iker_shoe/tests/test_t2r2_validator.py`

**Interfaces:**
- Consumes: `t2r2.context`(Task 3), `t2r2.loader`(Task 3)
- Produces: `static_check(src)`, `make_fake_context(n, device, seed)`, `dry_run(path, n, device)`, `validate(path, devices)`, `cuda_usable()`, `ValidationReport`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_t2r2_validator.py`:

```python
from openarm.agnostic.tasks.iker_shoe.t2r2 import validator as v

GOOD = """
import torch

def compute_reward(ctx):
    near = 1.0 - torch.tanh(5.0 * ctx.keypoint_dist)
    let_go = ctx.released.float()
    return near + let_go, {"near": near, "let_go": let_go}
"""


def test_good_code_passes_the_static_check():
    assert v.static_check(GOOD).ok


def test_unknown_context_field_is_rejected():
    rep = v.static_check(GOOD.replace("ctx.keypoint_dist", "ctx.distance_to_goal"))
    assert not rep.ok and any("distance_to_goal" in e for e in rep.errors)


def test_stage1_fields_are_unknown_here():
    rep = v.static_check(GOOD.replace("ctx.released", "ctx.latched"))
    assert not rep.ok and any("latched" in e for e in rep.errors)


def test_in_place_mutation_of_ctx_is_rejected():
    rep = v.static_check(GOOD.replace("near = ", "ctx.keypoint_dist.clamp_(0.0)\n    near = "))
    assert not rep.ok and any("읽기 전용" in e for e in rep.errors)


def test_forbidden_import_is_rejected():
    rep = v.static_check("import os\n" + GOOD)
    assert not rep.ok and any("os" in e for e in rep.errors)


def test_fake_context_has_seven_action_columns():
    ctx = v.make_fake_context(5)
    assert ctx.actions.shape == (5, 7) and ctx.prev_actions.shape == (5, 7)
    assert ctx.target_keypoints.shape == (5, 4, 3) and ctx.keypoint_err.shape == (5, 4)
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_validator.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'validator'`

- [ ] **Step 3: 1단계 검증기를 복사하고 두 곳만 고친다**

```bash
cp source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/validator.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/validator.py
```

고치는 곳 1 — 헤더의 형상 상수(1단계의 손가락·링크 상수를 지우고):

```python
from .context import SCALAR_FIELDS, TENSOR_FIELDS, RewardContext
from .loader import ENTRY_NAME, call_reward_fn, load_reward_fn
from .. import grasp_stage as gs
from ..place_stage import PlaceRewardCfg

ALLOWED_IMPORTS = {"torch", "math", "typing"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "getattr", "setattr",
                   "delattr", "globals", "locals", "vars"}
NUM_ARM, NUM_ACTIONS, NUM_KEYPOINTS = 7, 7, 4
NUM_SURFACE = gs.SURFACE_POINT_COUNT
```

고치는 곳 2 — `make_fake_context` 전체를 2단계 필드로:

```python
def make_fake_context(n: int = 16, *, device: str = "cpu", seed: int = 0) -> RewardContext:
    """Shape-correct random ctx for the dry run; no physical consistency."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = PlaceRewardCfg()

    def r(*shape, lo=-1.0, hi=1.0):
        return (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(device)

    def unit(*shape):
        v = r(*shape)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return RewardContext(
        table_top_z=0.205, rack_x_min=0.11, rack_x_max=0.43, rack_y_min=-0.33, rack_y_max=-0.02,
        rack_top_z=cfg.rack_top_z, episode_steps=200, control_dt=0.1,
        place_tolerance=cfg.place_tolerance, release_radius=cfg.release_radius, resting_tol=cfg.resting_tol,
        still_speed=cfg.still_speed, stable_steps=cfg.stable_steps,
        palm_pos=r(n, 3, lo=0.0, hi=0.6), palm_quat=unit(n, 4), palm_normal=unit(n, 3),
        arm_q=r(n, NUM_ARM), arm_qd=r(n, NUM_ARM), grip_norm=r(n),
        shoe_pos=r(n, 3, lo=0.0, hi=0.6), shoe_quat=unit(n, 4), shoe_lin_vel=r(n, 3), shoe_ang_vel=r(n, 3),
        shoe_surface=r(n, NUM_SURFACE, 3, lo=0.0, hi=0.6), shoe_bottom_z=r(n, lo=0.2, hi=0.45),
        palm_gap=r(n, lo=0.0, hi=0.3), palm_shoe_dist=r(n, lo=0.0, hi=0.4),
        target_keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6), keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6),
        init_keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6), keypoint_err=r(n, NUM_KEYPOINTS, lo=0.0, hi=0.4),
        keypoint_dist=r(n, lo=0.0, hi=0.4),
        placed=r(n) > 0.0, released=r(n) > 0.0, resting=r(n) > 0.0, still=r(n) > 0.0,
        stable_count=torch.floor(r(n, lo=0.0, hi=25.0)), success=r(n) > 0.8,
        episode_progress=r(n, lo=0.0, hi=1.0), actions=r(n, NUM_ACTIONS), prev_actions=r(n, NUM_ACTIONS),
    )
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_validator.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_context.py -q -p no:cacheprovider`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/validator.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_validator.py
git commit -m "feat(iker): 2단계 t2r 생성 코드 검증기"
```

---

### Task 5: 2단계 프롬프트

**Files:**
- Create: `tasks/iker_shoe/t2r2/prompts.py`
- Test: `tasks/iker_shoe/tests/test_t2r2_prompts.py`

**Interfaces:**
- Consumes: `t2r2.context.context_stub_source`, `t2r2.loader.ENTRY_NAME`, `place_stage.PlaceRewardCfg`, `layout`
- Produces: `PromptSpec`, `render_prompt(spec)`, `render_feedback_table(series, n_points)`, `FEEDBACK_TAG_PREFIXES`, `task_text(cfg)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_t2r2_prompts.py`:

```python
from openarm.agnostic.tasks.iker_shoe.place_stage import PlaceRewardCfg
from openarm.agnostic.tasks.iker_shoe.t2r2 import prompts as p


def test_prompt_embeds_the_context_source():
    text = p.render_prompt(p.PromptSpec(task=p.task_text(PlaceRewardCfg())))
    assert "class RewardContext" in text and "grip_norm" in text


def test_prompt_numbers_come_from_the_cfg_not_literals():
    cfg = PlaceRewardCfg()
    text = p.render_prompt(p.PromptSpec(task=p.task_text(cfg)))
    assert f"{cfg.place_tolerance}" in text and f"{cfg.release_radius}" in text and f"{cfg.stable_steps}" in text


def test_prompt_describes_seven_actions_including_the_grip_axis():
    text = p.render_prompt(p.PromptSpec(task=p.task_text(PlaceRewardCfg())))
    assert "actions[6]" in text and "(7,)" in text
    assert "actions[6:26]" not in text, "1단계 20관절 표는 넣지 않는다"


def test_task_text_names_all_four_stages():
    text = p.task_text(PlaceRewardCfg())
    for word in ("align", "set it down", "let go", "withdraw"):
        assert word in text


def test_feedback_table_lists_the_place_tags():
    table = p.render_feedback_table({"place/placed": [0.1, 0.2], "t2r_reward/near": [1.0, 2.0]})
    assert "place/placed" in table and "t2r_reward/near" in table
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_prompts.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'prompts'`

- [ ] **Step 3: 프롬프트를 쓴다**

`t2r2/prompts.py` — 1단계 `prompts.py` 의 절 순서를 그대로 쓰되 로봇 설명과 추가 사실을 2단계로 바꾼다. `ROBOT_DESCRIPTION` 은 `{place_tol}`·`{release_radius}` 등을 `PlaceRewardCfg` 에서 포맷한다. `REWARD_STRUCTURE`·`OUTPUT_RULES`·`FEEDBACK_HEADER`·`FEEDBACK_TAIL` 은 1단계에서 그대로 가져온다(문구 동일).

```python
ARM_ACTION_DIM, ACTION_DIM = 6, 7
ACTION_POS_SCALE_M, ACTION_ROT_SCALE_RAD = 0.02, 0.05
CONTROL_DT_S = 0.1
OBSERVATION_NOISE, ACTION_NOISE, QUAT_NOISE_RAD = 0.02, 0.05, 0.2

FEEDBACK_TAG_PREFIXES = (
    "t2r_reward/", "place/placed", "place/released", "place/resting", "place/retreated",
    "iker/success_5cm", "iker/keypoint_distance_m", "iker/dropped", "episode_lengths", "rewards",
)


def task_text(cfg: PlaceRewardCfg) -> str:
    return ("Place the shoe on the rack next to the other shoe: align it with the target keypoints, "
            "set it down, let go, and withdraw the hand.")
```

`ADDITIONAL_KNOWLEDGE` 는 다음 사실만 담는다(전부 코드에서 확인된 것): 배치 규칙과 반환 형태 · 마스크로 단계를 쓰라는 안내 · 유계 성형 권장 · 에피소드는 신발을 **이미 쥔 채** 시작한다 · 목표 키포인트는 고정 좌표다 · 받침 기하 · 성공 판정 네 조건과 연속 `stable_steps` · 종료 3종(성공·낙하·`episode_steps - 1`) · ctx 는 읽기 전용 · 항 이름이 `t2r_reward/<항>` 로 기록된다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_prompts.py -q -p no:cacheprovider`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/t2r2/prompts.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_prompts.py
git commit -m "feat(iker): 2단계 t2r 프롬프트"
```

---

### Task 6: env — 그립 축, 관측 39, 놓음 판정, 생성 보상

**Files:**
- Create: `tasks/iker_shoe/iker_shoe_t2r_env.py`, `tasks/iker_shoe/iker_shoe_t2r_env_cfg.py`
- Modify: `tasks/iker_shoe/config/__init__.py`
- Test: `tasks/iker_shoe/tests/test_t2r2_env_contract.py`

**Interfaces:**
- Consumes: `IkerShoeEnv`, `IkerShoeEnvCfg`, `place_stage.place_step`, `t2r2.loader.load_reward_fn/call_reward_fn`, `t2r2.context.RewardContext`, `gs.hand_targets`, `gs.normalized_targets`, `gs.surface_subsample`
- Produces: `IkerShoeT2rEnv`, `IkerShoeT2rEnvCfg`(`action_space=7`, `observation_space=39`, `reward_code_path: str = ""`), `IkerShoeT2rPlayEnvCfg`, gym id `open-sens_l_iker_shoe_t2r`(+`-play`)

- [ ] **Step 1: 실패하는 계약 테스트를 쓴다**

`tests/test_t2r2_env_contract.py` (1단계 `test_grasp_t2r_contract.py` 와 같은 문자열 단언 방식):

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "iker_shoe_t2r_env.py").read_text(encoding="utf-8")
CFG = (ROOT / "iker_shoe_t2r_env_cfg.py").read_text(encoding="utf-8")
REG = (ROOT / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_subclasses_the_stage2_env_and_overrides_only_the_listed_hooks():
    assert "class IkerShoeT2rEnv(IkerShoeEnv)" in ENV
    allowed = {"__init__", "_setup_scene", "_pre_physics_step", "_get_observations",
               "_get_dones", "_get_rewards", "_build_context", "_reset_idx"}
    names = {line.split("def ")[1].split("(")[0] for line in ENV.splitlines() if line.strip().startswith("def ")}
    assert names <= allowed, names - allowed


def test_action_is_seven_and_observation_is_thirtynine():
    assert "action_space = 7" in CFG and "observation_space = 39" in CFG
    assert 'reward_code_path: str = ""' in CFG and "(IkerShoeEnvCfg)" in CFG


def test_grip_axis_uses_the_stage1_hand_law_not_a_new_one():
    assert "gs.hand_targets(" in ENV and "gs.normalized_targets(" in ENV
    assert "actions[:, 6]" in ENV


def test_reward_comes_from_generated_code_on_the_predicate_step_state():
    block = ENV.split("def _get_rewards")[1]
    assert "call_reward_fn" in block and "nan_to_num" in block
    assert "t2r_reward/total" in ENV and "place/placed" in ENV


def test_context_tensors_are_copies():
    context = ENV.split("def _build_context")[1]
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in context


def test_reset_zeroes_previous_actions_and_restores_the_bank_grip():
    block = ENV.split("def _reset_idx")[1]
    assert "_t2r_prev_actions" in block and "_grip_targets" in block


def test_registration_uses_the_stage2_agent_yaml():
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env:IkerShoeT2rEnv" in REG
    assert 'f"open-sens_l_iker_shoe_t2r{_suffix}"' in REG and REG.count("rl_games_ppo_cfg.yaml") == 4
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_env_contract.py -q -p no:cacheprovider`
Expected: FAIL — `FileNotFoundError: iker_shoe_t2r_env.py`

- [ ] **Step 3: cfg 를 쓴다**

`iker_shoe_t2r_env_cfg.py`:

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

- [ ] **Step 4: env 를 쓴다**

`iker_shoe_t2r_env.py` — 오버라이드는 계약 테스트의 8개뿐이다.

- `__init__`: `load_reward_fn(cfg.reward_code_path)` → `self._reward_fn, self._reward_origin`; `gs.surface_subsample(meta[...]["hull_local"])` → `self._surface_local`; `self._grip_targets`(= 뱅크 손 목표), `self._stable_count`, `self._t2r_prev_actions`; 부팅 줄에 코드 경로·sha256 을 찍는다.
- `_pre_physics_step`: 부모와 같은 팔 6축 IK 를 돌린 뒤, `grip = actions[:, 6]` 으로
  `raw = grip_pose + (a+1)/2 * (open_pose - grip_pose)` → `gs.hand_targets(...)` 로 EMA·클램프 → `self._joint_targets[:, self._hand_ids]` 에 쓴다.
- `_get_observations`: 부모 38칸 뒤에 `gs.normalized_targets(...)` 로 만든 그립 상태 1칸을 붙인다.
- `_get_dones`: `place_step(...)` 로 네 조건·연속 카운터·성공을 계산하고 `self._place = step`; 종료는 성공 | 낙하 | 시간초과.
- `_get_rewards`: `self._build_context()` → `call_reward_fn` → `torch.nan_to_num` 합을 돌려주고 `t2r_reward/<항>`·`t2r_reward/total`·`place/*` 를 로그한다.
- `_build_context`: 스펙 §5 필드, 모든 텐서 `clone`.
- `_reset_idx`: 부모 리셋 뒤 `_t2r_prev_actions[ids] = 0`, `_grip_targets[ids]` = 뱅크 손 목표, `_stable_count[ids] = 0`.

- [ ] **Step 5: 등록을 더한다**

`config/__init__.py` 끝에:

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

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/ -q -p no:cacheprovider`
Expected: PASS (전체)

- [ ] **Step 7: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_t2r_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_t2r_env_cfg.py source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r2_env_contract.py
git commit -m "feat(iker): 2단계 t2r env — 그립 축·관측 39·놓음 판정·생성 보상"
```

---

### Task 7: 평가에 `placed`·`retreated` 보고 지표

**Files:**
- Modify: `scripts/iker/eval_iker.py` (`ROW_KEYS` 62행, `_record` 의 `values` 120-133행, `summarize` 150-180행)
- Test: `source/openarm/openarm/agnostic/modules/iker/tests/test_eval_summary.py`

**Interfaces:**
- Produces: 요약 최상위에 `placed`(5 cm 이내 **그리고** `palm_shoe > HOLD_RADIUS_M`)와 `retreated`(종료 시 팔바닥이 시작 자세에서 `RETREAT_M` 이상 멀어짐) 추가

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
import torch

from scripts.iker.eval_iker import placed_mask, retreated_mask, HOLD_RADIUS_M, RETREAT_M


def test_placed_needs_both_near_and_released():
    near = torch.tensor([True, True, False])
    palm_shoe = torch.tensor([HOLD_RADIUS_M + 0.05, HOLD_RADIUS_M - 0.05, HOLD_RADIUS_M + 0.05])
    assert placed_mask(near, palm_shoe).tolist() == [True, False, False]


def test_retreated_uses_the_distance_from_the_start_palm_pose():
    moved = torch.tensor([RETREAT_M + 0.01, RETREAT_M - 0.01])
    assert retreated_mask(moved).tolist() == [True, False]
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm:. python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_eval_summary.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'placed_mask'`

- [ ] **Step 3: 구현한다**

`eval_iker.py` 에 상수와 두 함수를 더하고, `ROW_KEYS` 에 `"palm_start_dist"` 를 넣고, `_record` 의 `values` 에
`"palm_start_dist": (palm - self.palm_start[finished]).norm(dim=-1)` 을 추가한다(`self.palm_start` 는 리셋 직후 팔바닥 위치를 담는다).

```python
RETREAT_M = 0.15  # 종료 시 팔바닥이 시작 자세에서 이만큼 멀어지면 "물러났다"


def placed_mask(near: torch.Tensor, palm_shoe: torch.Tensor) -> torch.Tensor:
    return near & (palm_shoe > HOLD_RADIUS_M)


def retreated_mask(palm_start_dist: torch.Tensor) -> torch.Tensor:
    return palm_start_dist >= RETREAT_M
```

`summarize` 의 반환 dict 최상위에 추가:

```python
"placed": frac(placed_mask(table["end_dist"] <= distance, table["palm_shoe"])),
"retreated": frac(retreated_mask(table["palm_start_dist"])),
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm:. python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_eval_summary.py -q -p no:cacheprovider`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add -- scripts/iker/eval_iker.py source/openarm/openarm/agnostic/modules/iker/tests/test_eval_summary.py
git commit -m "feat(iker): 평가에 placed·retreated 보고 지표"
```

---

### Task 8: 루프 라운드 판정과 실행기

**Files:**
- Modify: `modules/iker/loop_state.py`(`DEFAULT_POLICY` 25-48행, `PHASE_RUNS` 49-56행, `_stage2_train` 416-449행), `modules/iker/loop_probe.py`(`TASK_DIRS` 20행), `scripts/iker/loop.py`(실행기), `scripts/iker/LOOP_PROMPT.md`
- Create: `scripts/iker/t2r2_reward.py`
- Test: `modules/iker/tests/test_loop_state.py`(추가), `modules/iker/tests/test_loop_cli.py`(추가)

**Interfaces:**
- Consumes: Task 4·5 의 `validate`·`render_prompt`
- Produces: 정책 키 `t2r2_round_epochs`(750) · `t2r2_early_epoch`(250) · `t2r2_max_rounds`(4) · `t2r2_max_requests`(3); `TASK_DIRS["stage2_t2r"] = "iker-shoe-t2r"`

- [ ] **Step 1: 실패하는 판정 테스트를 쓴다**

```python
def test_stage2_asks_for_the_prompt_first():
    state = ls.new_state(NOW, phase="stage2_train", policy={"stage2_env_smoke": False})
    assert ls.decide(state, ls.Probe()).action == "write_t2r_prompt"


def test_round_ends_when_the_eval_misses_the_target():
    ...  # 평가 기록 3건·런 종료 → end_round, iter += 1


def test_four_rounds_without_a_pass_pauses():
    ...  # rounds 4개 → pause("4 rounds without reaching the success target")
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py -q -p no:cacheprovider`
Expected: FAIL

- [ ] **Step 3: 구현한다**

스펙 §8 의 9단계 순서를 `_stage2_train` 에 넣는다. 기존 `launch_stage2` 는 `env.reward_code_path=<iter 경로>` 를 인자에 더하고, `run_eval`·`record_eval` 은 그대로 쓴다. `t2r2_reward.py` 는 `render`/`ingest`/`reflect` 세 하위 명령으로 Task 4·5 를 호출한다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/ -q -p no:cacheprovider`
Expected: PASS (전체)

- [ ] **Step 5: 커밋**

```bash
git add -- source/openarm/openarm/agnostic/modules/iker/loop_state.py source/openarm/openarm/agnostic/modules/iker/loop_probe.py scripts/iker/loop.py scripts/iker/t2r2_reward.py scripts/iker/LOOP_PROMPT.md source/openarm/openarm/agnostic/modules/iker/tests/
git commit -m "feat(iker): 2단계 t2r 라운드 판정과 실행기"
```

---

### Task 9: 스모크 wire·round 모드

**Files:**
- Modify: `scripts/iker/t2r2_smoke.py` (Task 1 에서 만든 파일에 두 모드를 더한다)

**Interfaces:**
- Consumes: `IkerShoeT2rEnv`(Task 6)
- Produces: `smoke.json {passed, checks, idle_reward_mean, early_release_frac}`

- [ ] **Step 1: wire 검사를 더한다**

스펙 §9.2 표의 세 검사(그립 축이 연다 / 놓으면 손을 떠난다 / 쥐면 유지된다)를 시드 고정으로 구현한다.

- [ ] **Step 2: round 검사를 더한다**

64 env · 150 스텝(무행동 30 + 무작위) · NaN 0 · 리셋 뒤 `prev_actions` 0 · `place/*` 플래그 계산됨 · `idle_reward_mean`·`early_release_frac` 보고.

- [ ] **Step 3: 고정 보상 fixture 로 두 모드를 돌린다**

Run:
```bash
cd /home/user/rl_ws/hdgp-iker && RUN_LABEL=iker_s2t2r_smoke PYTHONUNBUFFERED=1 \
  /home/user/rl_ws/IsaacLab/_isaac_sim/python.sh scripts/iker/t2r2_smoke.py --mode wire --headless 2>&1 | tail -6
```
Expected: `T2R2 SMOKE passed True`

- [ ] **Step 4: 커밋**

```bash
git add -- scripts/iker/t2r2_smoke.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/
git commit -m "feat(iker): 2단계 t2r 스모크 wire·round"
```

---

### Task 10: 전환 — archive, 정책 키, 새 트랙

**Files:**
- Move: `iker_runs/shoe_place/config_00/loop/` → `archive/2026-09-16_s2_fixed_reward/loop/`, `loop_s2r8w/` → 같은 곳
- Modify: `scripts/iker/LOOP_PROMPT.md`(트랙 이름)

- [ ] **Step 1: 기존 트랙을 옮긴다 (정책 키 추가보다 먼저)**

```bash
cd /home/user/rl_ws/hdgp-iker
mkdir -p iker_runs/shoe_place/config_00/archive/2026-09-16_s2_fixed_reward
git mv iker_runs/shoe_place/config_00/loop iker_runs/shoe_place/config_00/archive/2026-09-16_s2_fixed_reward/loop
git mv iker_runs/shoe_place/config_00/loop_s2r8w iker_runs/shoe_place/config_00/archive/2026-09-16_s2_fixed_reward/loop_s2r8w
git commit -m "chore(iker): 고정보상 2단계 트랙 2건 archive"
```

**순서가 중요하다** — `load_state → validate_state` 가 정책 키 집합을 검사하므로, 키를 먼저 늘리면 기존 상태 파일이 로드 불가가 된다.

- [ ] **Step 2: 새 트랙을 만든다**

```bash
python3 scripts/iker/loop.py --state-dir iker_runs/shoe_place/config_00/loop_s2t2r \
  init --track iker_shoe_c00_s2t2r --phase stage2_train \
  --policy '{"t2r2_round_epochs": 750, "t2r2_max_rounds": 4, "labels": {"stage2": "iker_vlm_c00_s2t2r_i00"}}'
```

- [ ] **Step 3: 첫 판정이 `write_t2r_prompt` 인지 확인한다**

Run: `python3 scripts/iker/loop.py --state-dir iker_runs/shoe_place/config_00/loop_s2t2r status`
Expected: `"action": "write_t2r_prompt"`

- [ ] **Step 4: LOOP_PROMPT 의 트랙 이름을 바꾸고 커밋한다**

```bash
git add -- scripts/iker/LOOP_PROMPT.md iker_runs/shoe_place/config_00/loop_s2t2r
git commit -m "chore(iker): 2단계 t2r 트랙 개시"
```

---

## Self-Review

**1. 스펙 커버리지**

| 스펙 절 | 과제 |
|---|---|
| §3 성공 판정 | Task 2 |
| §4 액션 6→7 · 관측 38→39 | Task 6 |
| §5 RewardContext | Task 3 |
| §6 env 훅 | Task 6 |
| §7 프롬프트 | Task 5 |
| §8 라운드 판정·정책 키 | Task 8 |
| §9.1 검증 | Task 4 |
| §9.2 스모크 (settle / wire / round) | Task 1 · Task 9 |
| §10 실패 처리 | Task 8 |
| §11 테스트 | Task 2·3·4·5·6·7·8 각 Step 1 |
| §12 파일 배치 | File Structure |
| §13 전환 절차 | Task 10 |
| §14 위험 | Task 1 이 정착 게이트, Task 9 가 `early_release_frac` |

빠진 절 없음.

**2. Placeholder 점검** — Task 8 Step 1 의 두 테스트와 Task 8 Step 3, Task 9 Step 1·2 는 본문이 산문이다. 실행 시 그 과제의 구현자가 스펙 §8·§9.2 의 표를 그대로 읽어 코드로 옮긴다. 다른 과제는 전부 실제 코드가 들어 있다.

**3. 타입 일관성** — `place_step(keypoint_dist, palm_shoe_dist, shoe_bottom_z, shoe_speed, stable_count, cfg) -> PlaceStep` 이 Task 2·6 에서 같고, `RewardContext` 필드명이 Task 3·4·5·6 에서 같으며, `load_reward_fn/call_reward_fn` 시그니처가 Task 3·6 에서 같다.
