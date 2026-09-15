# IKER Stage-1 t2r Reward Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written stage-1 grasp reward and the loop's phases A/calibrate/B/harvest with a t2r (text2reward) round loop (`stage1_t2r`) inside the IKER auto loop, keeping the stage-1 hold/success predicate as the handover contract to harvest → VLM target → stage 2.

**Architecture:** A t2r fork under `tasks/iker_shoe/t2r/` (context, loader, validator, prompts, pipeline) feeds an env subclass that swaps only the reward sum for `compute_reward(ctx)`; a smoke script checks the wiring; `loop_state`/`loop_probe`/`loop.py` gain the `stage1_t2r` phase (prompt → isolated generator agent → ingest → smoke → fresh training → harvest at ≥ 5 % success or round end → reflect → next iteration).

**Tech Stack:** Python 3.11, torch, Isaac Lab DirectRLEnv, rl_games PPO, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-iker-stage1-t2r-design.md` (§1–§13 design, §14 implementation rulings). Also read `docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md` for the loop conventions.

## Global Constraints

- Work only in `/home/user/rl_ws/hdgp-iker` (git worktree, branch `iker-front-end`). Never run anything in `~/rl_ws/hdgp`.
- Pure tests: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests source/openarm/openarm/agnostic/modules/iker/tests -q -p no:cacheprovider` (baseline 225 passed, 2 skipped).
- Isaac run form (foreground, Bash timeout 600000): `TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/source/openarm RUN_LABEL=<label> ../IsaacLab/_isaac_sim/python.sh <script> ... --headless > log/iker_t2r/<name>.log 2>&1`.
- The stage-1 success predicate (`grasp_stage.stage1_step` held/latch/success/lost), the parent env `iker_shoe_grasp_env.py`, the stage-2 env, `modules/t2r/`, `scripts/reward_gen/`, `tasks/grasp_fj_t2r/` and the robot profiles are not modified.
- No `r_hj_`/`l_hj_` joint-name literals in production code (read names from `robot_profiles.TESOLLO_LEFT_SHORT`).
- Observations and actions of the stage-1 env are unchanged; contact forces are reward-only.
- reward-audit is not used (t2r and IKER tracks, user decisions).
- Policy values (verbatim from the spec): `t2r_round_epochs` 500, `t2r_early_epoch` 250, `t2r_early_latched` 0.005, `t2r_harvest_success` 0.05, `t2r_max_rounds` 6, `t2r_max_requests` 3, `t2r_num_envs` 4096, `t2r_smoke_envs` 64, `t2r_smoke_steps` 150; `harvest_min` 64, `bin_epochs` 10 unchanged; label prefix `iker_grasp_c00_t2r`, run label `<prefix>_iNN`.
- Logs only under `log/`; no new files at the repo root; no `git add -A`/`git add .`; no `--no-verify`; `git commit` in its own Bash call.
- Commit trailer lines, exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- Implementers never dispatch subagents and never write task-observer observations (the user disabled them for this session; this takes precedence over any global activation rule).
- Never `pkill`/`killall`; never start training (the controller launches after the plan).

---

### Task 1: t2r fork — context, loader, validator, prompts, pipeline

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/__init__.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/context.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/loader.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/validator.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/prompts.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/pipeline.py`
- Create: `scripts/iker/t2r_reward.py`
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r_reward.py`

**Interfaces:**
- Produces: `context.RewardContext` (fields below), `context.TENSOR_FIELDS`, `context.SCALAR_FIELDS`, `context.context_stub_source() -> str`;
  `loader.ENTRY_NAME = "compute_reward"`, `loader.load_reward_fn(path: str | None) -> tuple[RewardFn, str]`, `loader.call_reward_fn(fn, ctx) -> tuple[Tensor, dict[str, Tensor]]`, `loader.zero_reward`;
  `validator.ValidationReport`, `validator.static_check(src) -> ValidationReport`, `validator.make_fake_context(n=16, *, device="cpu", seed=0) -> RewardContext`, `validator.dry_run(path, *, n=64, device="cpu")`, `validator.validate(path, devices=None) -> ValidationReport`, `validator.cuda_usable() -> bool`;
  `prompts.HAND_ACTION_RANGES: tuple[tuple[float, float], ...]` (20 rows, profile order), `prompts.PromptSpec(task, previous_code=None, feedback=None, user_notes=None)`, `prompts.render_prompt(spec) -> str`, `prompts.task_text(cfg: Stage1RewardCfg) -> str`, `prompts.render_feedback_table(series: dict[str, list[float]], n_points=10) -> str`, `prompts.FEEDBACK_TAG_PREFIXES`;
  `pipeline.PROMPT/RESPONSE/CODE/VALIDATION/FEEDBACK` file names, `pipeline.extract_code(text) -> str | None`, `pipeline.render(iter_dir, *, previous_code=None, feedback=None, notes=None) -> Path`, `pipeline.ingest(iter_dir, devices=None) -> dict`, `pipeline.feedback_series(events) -> dict[str, list[float]]`, `pipeline.reflect(prev_dir, next_dir, events, *, notes=None, n_points=10) -> dict`;
  CLI `scripts/iker/t2r_reward.py render --iter-dir D` / `ingest --iter-dir D` (exit 1 when not ok) / `reflect --prev-dir P --next-dir N --events FILE [--notes FILE]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_t2r_reward.py`:

```python
"""IKER stage-1 t2r fork — context, loader, validator, prompts and round files (spec 2026-09-15-iker-stage1-t2r §5, §6, §8; no Isaac)."""

import json
import re

import pytest
import torch

from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs
from openarm.agnostic.tasks.iker_shoe import layout
from openarm.agnostic.tasks.iker_shoe.t2r import context as C
from openarm.agnostic.tasks.iker_shoe.t2r import loader as L
from openarm.agnostic.tasks.iker_shoe.t2r import pipeline
from openarm.agnostic.tasks.iker_shoe.t2r import prompts as P
from openarm.agnostic.tasks.iker_shoe.t2r import validator as V

TASK_DIR = layout.HDGP_ROOT / "source" / "openarm" / "openarm" / "agnostic" / "tasks" / "iker_shoe"
SAMPLE = """import torch
import math


def compute_reward(ctx):
    approach = torch.exp(-5.0 * ctx.palm_gap)
    touch = torch.tanh(ctx.link_shoe_force.sum(dim=(1, 2)) / 5.0)
    lift = torch.clamp(ctx.dz_free / ctx.lift_height, 0.0, 1.0)
    return approach + touch + lift + 10.0 * ctx.success.float(), {"approach": approach, "touch": touch, "lift": lift}
"""


def _write(tmp_path, src):
    path = tmp_path / "compute_reward.py"
    path.write_text(src, encoding="utf-8")
    return str(path)


def test_context_stub_lists_every_field_without_helpers():
    stub = C.context_stub_source()
    assert "class RewardContext" in stub and "@property" not in stub
    for name in C.TENSOR_FIELDS + C.SCALAR_FIELDS:
        assert name in stub, name
    assert {"link_shoe_force", "palm_shoe_force", "slip_speed", "dz_free", "held", "hold_count", "latched", "success"} <= set(C.TENSOR_FIELDS)
    assert {"lift_height", "lift_max", "latch_steps", "success_steps", "rack_x_min", "rack_y_max", "episode_steps"} <= set(C.SCALAR_FIELDS)
    assert not any(name.startswith(("cup", "goal", "bead", "src_", "rcv_")) for name in C.TENSOR_FIELDS + C.SCALAR_FIELDS)


def test_fake_context_matches_the_documented_layout():
    ctx = V.make_fake_context(8)
    assert ctx.link_pos.shape == (8, 5, 3, 3) and ctx.link_shoe_gap.shape == (8, 5, 3) and ctx.link_shoe_force.shape == (8, 5, 3)
    assert ctx.hand_q.shape == (8, 20) and ctx.hand_target_norm.shape == (8, 20) and ctx.arm_q.shape == (8, 7)
    assert ctx.shoe_surface.shape == (8, gs.SURFACE_POINT_COUNT, 3) and ctx.shoe_start_xy.shape == (8, 2)
    assert ctx.actions.shape == (8, 26) and ctx.prev_actions.shape == (8, 26)
    assert ctx.held.dtype == ctx.latched.dtype == ctx.success.dtype == torch.bool and ctx.num_envs == 8


def test_validator_passes_a_wellformed_reward_and_rejects_bad_ones(tmp_path):
    rep = V.validate(_write(tmp_path, SAMPLE), devices=("cpu",))
    assert rep.ok, rep.errors
    assert set(rep.term_stats) == {"approach", "touch", "lift"} and "link_shoe_force" in rep.fields_used
    assert not V.validate(_write(tmp_path, SAMPLE.replace("ctx.palm_gap", "ctx.cup_pos")), devices=("cpu",)).ok
    assert not V.validate(_write(tmp_path, "import numpy\n" + SAMPLE), devices=("cpu",)).ok
    mutated = SAMPLE.replace("    approach =", "    ctx.dz_free.clamp_(min=0.0)\n    approach =")
    rep = V.validate(_write(tmp_path, mutated), devices=("cpu",))
    assert not rep.ok and any("dz_free" in error for error in rep.errors)


def test_loader_enforces_shapes_and_an_empty_path_is_zero_reward():
    ctx = V.make_fake_context(4)
    with pytest.raises(RuntimeError):
        L.call_reward_fn(lambda c: (torch.zeros(3), {}), ctx)
    fn, source = L.load_reward_fn("")
    total, terms = L.call_reward_fn(fn, ctx)
    assert torch.all(total == 0.0) and terms == {} and "zero" in source


def test_hand_action_table_has_one_row_per_profile_joint():
    names = TESOLLO_LEFT_SHORT.hand_joint_names
    assert len(P.HAND_ACTION_RANGES) == len(names) == 20
    assert all(lo <= hi for lo, hi in P.HAND_ACTION_RANGES)
    text = P.render_prompt(P.PromptSpec(task="T"))
    for index, name in enumerate(names):
        assert re.search(rf"actions\[\s*{6 + index}\] = hand joint\s+{index}: {name}\b", text), name


def test_prompt_numbers_come_from_the_config_and_the_hold_predicate():
    cfg = gs.Stage1RewardCfg()
    text = P.render_prompt(P.PromptSpec(task=P.task_text(cfg)))
    assert "Box(-1, 1, (26,), float32)" in text and "compute_reward(ctx: RewardContext)" in text
    assert f"{cfg.lift_height_m * 100:.0f} and {cfg.lift_max_m * 100:.0f} cm" in text
    assert "ctx.slip_speed < ctx.hold_slip_speed" in text and "ctx.latch_steps" in text
    env_cfg = (TASK_DIR / "iker_shoe_env_cfg.py").read_text(encoding="utf-8")
    grasp_cfg = (TASK_DIR / "iker_shoe_grasp_env_cfg.py").read_text(encoding="utf-8")
    assert f"action_pos_scale = {P.ACTION_POS_SCALE_M}" in env_cfg and f"action_rot_scale = {P.ACTION_ROT_SCALE_RAD}" in env_cfg
    assert f"observation_noise = {P.OBSERVATION_NOISE}" in env_cfg and f"action_noise = {P.ACTION_NOISE}" in env_cfg
    assert f"drop_z = {P.DROP_Z_M:.2f}" in grasp_cfg and f"wrench_force_per_kg = {P.WRENCH_FORCE_PER_KG}" in grasp_cfg
    assert f"start_noise_xy = {P.START_NOISE_XY_M}" in grasp_cfg and "GRASP_EPISODE_STEPS = 120" in grasp_cfg
    assert P.ARM_ACTION_DIM == 6 and P.HAND_ACTION_DIM == 20
    low = text.lower()
    assert "cup" not in low and "bead" not in low


def test_feedback_series_strips_prefixes_and_keeps_only_feedback_tags():
    events = {"Episode/t2r_reward/touch": [(1, 0.1), (2, 0.2)], "Episode/grasp_episode/success": [(1, 0.0)],
              "rewards/iter": [(1, 3.0)], "rewards/time": [(1, 3.0)], "Episode/grasp/w_thumb": [(1, 0.5)], "shaped_rewards/iter": [(1, 1.0)]}
    series = pipeline.feedback_series(events)
    assert series == {"t2r_reward/touch": [0.1, 0.2], "grasp_episode/success": [0.0], "rewards": [3.0]}
    table = P.render_feedback_table(series, n_points=10)
    assert "t2r_reward/touch: [0.1, 0.2]" in table and "min 0.1" in table


def test_round_files_render_ingest_and_reflect(tmp_path):
    first, second = tmp_path / "iter_00", tmp_path / "iter_01"
    prompt = pipeline.render(first)
    assert prompt.read_text(encoding="utf-8").startswith("You are an expert in robotics")
    with pytest.raises(FileExistsError):
        pipeline.render(first)
    (first / pipeline.RESPONSE).write_text("Stages...\n```python\nnot this\n```\nFinal:\n```python\n" + SAMPLE + "```\n", encoding="utf-8")
    report = pipeline.ingest(first, devices=("cpu",))
    assert report["ok"] and len(report["reward_sha256"]) == 64 and (first / pipeline.CODE).read_text(encoding="utf-8") == SAMPLE
    assert json.loads((first / pipeline.VALIDATION).read_text(encoding="utf-8"))["ok"]
    result = pipeline.reflect(first, second, {"Episode/t2r_reward/touch": [(1, 0.1)], "Episode/grasp_episode/success": [(1, 0.0)]})
    assert result["tags"] == 2 and (first / pipeline.FEEDBACK).is_file()
    text = (second / pipeline.PROMPT).read_text(encoding="utf-8")
    assert "The previous reward function was" in text and "t2r_reward/touch" in text and SAMPLE.strip().splitlines()[-1].strip() in text
    with pytest.raises(ValueError, match="feedback tag"):
        pipeline.reflect(first, tmp_path / "iter_02", {"rewards/time": [(1, 0.0)]})


def test_ingest_without_a_code_block_writes_a_failed_report(tmp_path):
    folder = tmp_path / "iter_00"
    folder.mkdir()
    (folder / pipeline.RESPONSE).write_text("no code here", encoding="utf-8")
    report = pipeline.ingest(folder, devices=("cpu",))
    assert not report["ok"] and "python" in report["errors"][0] and not (folder / pipeline.CODE).exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r_reward.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'openarm.agnostic.tasks.iker_shoe.t2r'`.

- [ ] **Step 3: Write `t2r/__init__.py`**

```python
"""text2reward fork of the IKER stage-1 grasp (spec 2026-09-15-iker-stage1-t2r). Does not share code with modules/t2r or grasp_fj_t2r."""
```

- [ ] **Step 4: Write `t2r/context.py`**

```python
"""RewardContext — the only input of a generated stage-1 reward (spec 2026-09-15-iker-stage1-t2r §5).

Contract (same convention as the other t2r forks, no shared code):
  * every tensor is batched (N, ...) on one device; positions are env-local metres, angles rad, forces N;
  * the field names and comments ARE the prompt's environment description (prompts.py renders this source), in English;
  * frozen — the reward only reads it; the environment passes copies of its buffers.
Every comment below was checked against the code on 2026-09-15 (iker_shoe_grasp_env, grasp_stage.stage1_step, layout).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) -----------------------------
    table_top_z: float             # height of the table top [m]
    rack_x_min: float              # the rack's footprint on the table: x from rack_x_min to rack_x_max [m]
    rack_x_max: float
    rack_y_min: float              # ... and y from rack_y_min to rack_y_max [m]
    rack_y_max: float
    episode_steps: int             # the episode ends after this many steps
    control_dt: float              # seconds per step
    lift_height: float             # a held step needs dz_free of at least this [m]
    lift_max: float                # ... and at most this [m]
    hold_xy_radius: float          # ... shoe_shift_xy of at most this [m]
    hold_radius: float             # ... palm_shoe_dist of at most this [m]
    hold_slip_speed: float         # ... slip_speed below this [m/s]
    thumb_curl_min: float          # ... thumb_curl of at least this [rad]
    latch_steps: int               # latched becomes True once hold_count reaches this
    success_steps: int             # a success needs hold_count of at least this
    success_speed: float           # ... and a shoe speed below this [m/s]

    # ---- hand: left Tesollo DG-5F, finger index 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky ----
    palm_pos: torch.Tensor         # (N,3) palm frame origin
    palm_quat: torch.Tensor        # (N,4) palm orientation quaternion (w,x,y,z)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm's grasping side
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_shoe_gap: torch.Tensor    # (N,5,3) distance from each of those links to the nearest point of shoe_surface [m]
    link_shoe_force: torch.Tensor  # (N,5,3) contact force magnitude between each of those links and the shoe only [N] (0 = not touching)
    palm_shoe_force: torch.Tensor  # (N,) contact force magnitude between the palm and the shoe only [N]
    hand_q: torch.Tensor           # (N,20) finger joint angles [rad], order = the hand joint table in the robot description
    hand_qd: torch.Tensor          # (N,20) finger joint velocities [rad/s]
    hand_q_norm: torch.Tensor      # (N,20) joint angles normalised to each joint's commandable range: 0 = lower limit, 1 = upper limit
    hand_target_norm: torch.Tensor  # (N,20) filtered finger joint targets, same normalisation as hand_q_norm
    thumb_curl: torch.Tensor       # (N,) how far the thumb's _3 joint moved from its open angle towards its grip angle [rad]; negative = bent back
    hand_z_min: torch.Tensor       # (N,) height of the lowest finger link [m]

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- shoe: the shoe to grasp -----------------------------------------------------------
    shoe_pos: torch.Tensor         # (N,3) shoe reference point (its body origin)
    shoe_quat: torch.Tensor        # (N,4) shoe orientation quaternion (w,x,y,z)
    shoe_lin_vel: torch.Tensor     # (N,3) linear velocity of the shoe's centre of mass [m/s]
    shoe_ang_vel: torch.Tensor     # (N,3) angular velocity of the shoe [rad/s]
    shoe_start_xy: torch.Tensor    # (N,2) shoe_pos x, y at the start of the episode
    shoe_surface: torch.Tensor     # (N,P,3) points on the shoe's outer surface
    dz_free: torch.Tensor          # (N,) rise of the lowest surface point above its starting height [m]; 0 while any point is above the rack's footprint
    shoe_shift_xy: torch.Tensor    # (N,) horizontal distance of shoe_pos from shoe_start_xy [m]
    palm_gap: torch.Tensor         # (N,) distance from palm_pos to the nearest point of shoe_surface [m]
    palm_shoe_dist: torch.Tensor   # (N,) distance from palm_pos to shoe_pos [m]
    slip_speed: torch.Tensor       # (N,) speed of the shoe's centre of mass relative to the palm body frame [m/s]; 0 while it moves rigidly with the hand

    # ---- hold and task status: computed by the environment ---------------------------------
    held: torch.Tensor             # (N,) bool, this step satisfies the hold conditions
    hold_count: torch.Tensor       # (N,) consecutive held steps up to this one (float)
    latched: torch.Tensor          # (N,) bool, True once hold_count reached latch_steps in this episode (stays True)
    success: torch.Tensor          # (N,) bool, True on the step a success is counted (the episode then ends)
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of episode_steps

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,26) policy action of this step, clipped to [-1,1]
    prev_actions: torch.Tensor     # (N,26) policy action of the previous step (zeros right after a reset)

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
    """The class body (fields and comments) — the prompt's environment description; the helper properties are cut."""
    import inspect

    src = inspect.getsource(RewardContext)
    return src[: src.find("    # ----------")].rstrip() + "\n"
```

- [ ] **Step 5: Write `t2r/loader.py`** — copy `source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/t2r/loader.py` verbatim, then make exactly these edits: the module docstring's first line becomes `"""Loader of generated stage-1 reward code (IKER t2r fork).`; the namespace name `"t2r_fj_generated_reward"` becomes `"iker_t2r_generated_reward"`. Nothing else changes (`from .context import RewardContext` resolves to the new context).

- [ ] **Step 6: Write `t2r/validator.py`** — copy `source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/t2r/validator.py` verbatim, then make exactly these edits:
  1. Docstring first line: `"""Validation of generated stage-1 reward code (IKER t2r fork) — static check + fake RewardContext dry run, no Isaac.`
  2. After the `from .loader import ...` line add `from .. import grasp_stage as gs`.
  3. Replace the line `NUM_FINGERS, NUM_LINKS, NUM_HAND, NUM_ARM, NUM_ACTIONS = 5, 3, 19, 7, 26` with
     `NUM_FINGERS, NUM_LINKS, NUM_HAND, NUM_ARM, NUM_ACTIONS = 5, 3, 20, 7, 26` and add below it `NUM_SURFACE = gs.SURFACE_POINT_COUNT`.
  4. Replace the whole `make_fake_context` function with:

```python
def make_fake_context(n: int = 16, *, device: str = "cpu", seed: int = 0) -> RewardContext:
    """Shape-correct random ctx for the dry run; no physical consistency."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, lo=-1.0, hi=1.0):
        return (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(device)

    def unit(*shape):
        v = r(*shape)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return RewardContext(
        table_top_z=0.205, rack_x_min=0.11, rack_x_max=0.43, rack_y_min=-0.33, rack_y_max=-0.02, episode_steps=120, control_dt=0.1,
        lift_height=0.05, lift_max=0.15, hold_xy_radius=0.10, hold_radius=0.15, hold_slip_speed=0.05, thumb_curl_min=0.05,
        latch_steps=3, success_steps=20, success_speed=0.05,
        palm_pos=r(n, 3, lo=0.0, hi=0.6), palm_quat=unit(n, 4), palm_normal=unit(n, 3),
        link_pos=r(n, NUM_FINGERS, NUM_LINKS, 3, lo=0.0, hi=0.6), link_shoe_gap=r(n, NUM_FINGERS, NUM_LINKS, lo=0.0, hi=0.3),
        link_shoe_force=r(n, NUM_FINGERS, NUM_LINKS, lo=0.0, hi=5.0), palm_shoe_force=r(n, lo=0.0, hi=5.0),
        hand_q=r(n, NUM_HAND, lo=-1.6, hi=2.0), hand_qd=r(n, NUM_HAND), hand_q_norm=r(n, NUM_HAND, lo=0.0, hi=1.0),
        hand_target_norm=r(n, NUM_HAND, lo=0.0, hi=1.0), thumb_curl=r(n, lo=-0.5, hi=1.6), hand_z_min=r(n, lo=0.15, hi=0.5),
        arm_q=r(n, NUM_ARM), arm_qd=r(n, NUM_ARM),
        shoe_pos=r(n, 3, lo=0.0, hi=0.6), shoe_quat=unit(n, 4), shoe_lin_vel=r(n, 3), shoe_ang_vel=r(n, 3),
        shoe_start_xy=r(n, 2, lo=0.0, hi=0.4), shoe_surface=r(n, NUM_SURFACE, 3, lo=0.0, hi=0.6),
        dz_free=r(n, lo=0.0, hi=0.2), shoe_shift_xy=r(n, lo=0.0, hi=0.3), palm_gap=r(n, lo=0.0, hi=0.2),
        palm_shoe_dist=r(n, lo=0.0, hi=0.3), slip_speed=r(n, lo=0.0, hi=0.5),
        held=r(n) > 0.0, hold_count=torch.floor(r(n, lo=0.0, hi=25.0)), latched=r(n) > 0.0, success=r(n) > 0.8,
        episode_progress=r(n, lo=0.0, hi=1.0), actions=r(n, NUM_ACTIONS), prev_actions=r(n, NUM_ACTIONS),
    )
```

- [ ] **Step 7: Write `t2r/prompts.py`**

```python
"""Prompt rendering of the IKER stage-1 t2r reward generator (spec 2026-09-15-iker-stage1-t2r §6).

text2reward layout: role -> robot and action description -> reward structure -> the RewardContext source -> additional
knowledge -> task and output rules -> (later rounds) previous code, feedback table and tips, notes.

Environment facts only, no design advice (user decision 2026-09-15: zero-shot). Numbers checked against the code on
2026-09-15: action scales and noise `iker_shoe_env_cfg.IkerShoeEnvCfg`; 10 Hz = `DECIMATION` 12 x `PHYSICS_DT` 1/120;
hand law `grasp_stage.hand_targets` with `HAND_EMA_ALPHA`; hand action ranges = the boot log of `iker_grasp_c00_r8_a`
("[iker_grasp] hand action range", profile order); start palm gap measured 105-142 mm (`scripts/iker/grasp_smoke.py`);
startup randomisation `IkerShoeEventCfg`; disturbance `object_wrench.WrenchDR.step` gated on the latch with
`IkerShoeGraspEnvCfg.wrench_*`; terminations `IkerShoeGraspEnv._get_dones`; hold predicate `grasp_stage.stage1_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT

from .. import grasp_stage as gs
from .. import layout
from .context import context_stub_source
from .loader import ENTRY_NAME

ARM_ACTION_DIM, HAND_ACTION_DIM = 6, 20
ACTION_POS_SCALE_M, ACTION_ROT_SCALE_RAD = 0.02, 0.05
CONTROL_DT_S = 0.1
START_PALM_GAP_MM = (105, 142)
START_NOISE_XY_M = 0.02
MASS_SCALE_RANGE, FRICTION_RANGE, COM_OFFSET_M = (0.3, 2.0), (0.3, 1.8), 0.025
OBSERVATION_NOISE, ACTION_NOISE, QUAT_NOISE_RAD = 0.02, 0.05, 0.2
WRENCH_PROB_RANGE, WRENCH_FORCE_PER_KG, WRENCH_TORQUE_PER_KG = (0.006, 0.6), 2.7, 0.27
DROP_Z_M = 0.10
LOCKED_SPAN_RAD = 0.05
#: commandable range [rad] of each hand joint in TESOLLO_LEFT_SHORT.hand_joint_names order (boot log of iker_grasp_c00_r8_a, 2026-09-15)
HAND_ACTION_RANGES: tuple[tuple[float, float], ...] = (
    (-0.010, 0.010), (1.560, 1.580), (-1.571, 0.000), (-1.571, 0.000),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 2.007), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.010), (0.000, 1.920), (0.000, 1.571), (0.000, 1.571),
    (-0.010, 0.000), (-0.010, 0.000), (0.000, 1.571), (0.000, 1.571),
)
FEEDBACK_TAG_PREFIXES = (
    "t2r_reward/", "grasp_episode/success", "grasp_episode/latched", "grasp_episode/best_lift_m", "grasp/held_frac",
    "grasp/dz_free", "grasp/shift_xy", "grasp/thumb_curl", "grasp/rel_speed", "grasp/lost_frac", "grasp/over_rack_raised_frac",
    "grasp/arm_speed_sum", "grasp/hand_command_rate", "contact/", "episode_lengths", "rewards",
)


def _joint_table() -> str:
    prof = TESOLLO_LEFT_SHORT
    rows = []
    for index, (name, (lo, hi)) in enumerate(zip(prof.hand_joint_names, HAND_ACTION_RANGES, strict=True)):
        tag = "locked" if hi - lo <= LOCKED_SPAN_RAD else "movable"
        rows.append(f"    actions[{ARM_ACTION_DIM + index:2d}] = hand joint {index:2d}: {name} range [{lo:+.3f}, {hi:+.3f}] rad  "
                    f"open {float(prof.hand_open_pose[index]):+.3f}  grip {float(prof.hand_grip_pose[index]):+.3f}  {tag}")
    return "\n".join(rows)


ROBOT_DESCRIPTION = """\
We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. \
A shoe lies on the table in front of the hand; a second shoe stands on a raised rack beside it. The rack's footprint is \
x in [{rack_x_min:.2f}, {rack_x_max:.2f}] m and y in [{rack_y_min:.2f}, {rack_y_max:.2f}] m, and its top is at z = {rack_top_z:.3f} m. \
Positions are in metres in each environment's local frame with +z up; the table top is at z = table_top_z.

At the start of an episode the arm is at a pre-grasp pose above the shoe with the hand open; the palm is about {gap_lo} to \
{gap_hi} mm from the nearest point of the shoe's surface. The shoe starts at one of a set of recorded rest poses, shifted by \
up to {start_noise_cm:.0f} cm along x and y. Each environment draws once at startup a shoe mass scale of {mass_lo} to {mass_hi}, \
a friction coefficient of {fric_lo} to {fric_hi}, a restitution of 0 to 1 and a centre-of-mass offset of up to {com_cm:.1f} cm \
per axis. During training the policy's observations carry uniform noise of +-{obs_noise} (orientations up to {quat_noise} rad) \
and its actions uniform noise of +-{act_noise} before they are executed. Once the hold has latched (see below), random force \
and torque pulses act on the shoe: each step an environment fires with its own probability between {p_lo} and {p_hi}, and a \
pulse has a normal random component per axis with a standard deviation of {force_per_kg} N ({torque_per_kg} N m) per kg of \
shoe mass.

The action space is a normalized `Box(-1, 1, (26,), float32)`, one action every {control_dt} s:
    actions[0:3] = change of the palm position in the robot base frame, {pos_scale} * a metres per step;
    actions[3:6] = change of the palm orientation in the robot base frame (axis-angle), {rot_scale} * a radians per step.
The palm pose change goes through damped least-squares inverse kinematics to arm joint position targets (PD, gravity \
compensated), clamped to the arm's joint limits.
    actions[6:26] = finger joint targets: each value is mapped linearly onto that joint's range (a = -1 the lower limit, \
a = +1 the upper limit), and each step the joint target moves {alpha:.4f} of the way from its previous value to that point. \
The order, which is also the order of ctx.hand_q / hand_qd / hand_q_norm / hand_target_norm, is:
{joint_table}
Joints marked "locked" have a range of 0.05 rad or less and effectively do not move. "open" is the joint angle of the open \
hand the episode starts with and "grip" the joint angle of a closed grasp: a finger closes as its joint angles move from \
"open" towards "grip". The finger joints are position-controlled (PD), so a finger that meets the shoe stops there and \
presses with a force that grows with the gap between its target and its actual angle.
"""

REWARD_STRUCTURE = """\
Typically, the reward function of a manipulation task consists of these parts (some are optional — \
include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task
"""

ADDITIONAL_KNOWLEDGE = """\
Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function \
must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). \
Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. \
`staged = torch.where(ctx.latched, r_hold, torch.zeros_like(r_hold))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative \
distances, so that no single term dominates.
4. Contact: `ctx.link_shoe_force[:, f, k]` is the contact force between link k (0 the link moved by joint `_3`, 1 the link \
moved by joint `_4`, 2 the fingertip) of finger f (0 thumb, 1 index, 2 middle, 3 ring, 4 pinky) and the shoe to grasp, and \
`ctx.palm_shoe_force` the force between the palm and that shoe. Contact is measured only on these links and the palm; contacts \
with anything else (the table, the rack, the other shoe, the hand itself) are not included. A value of 0 means that link is \
not touching the shoe. Contact forces come from the physics engine each step and can spike. These forces are only available \
to the reward; the policy itself does not observe contact.
5. Geometry: `ctx.shoe_surface` holds {surface_points} points on the shoe's outer surface. `ctx.palm_gap` is the distance from \
the palm frame origin to the nearest of them and `ctx.link_shoe_gap[:, f, k]` the same for each finger link. `ctx.palm_normal` \
points out of the palm's grasping side. `ctx.dz_free` is how far the lowest surface point has risen above its height at the \
start of the episode, and it is 0 whenever any surface point is above the rack's footprint widened by {rack_margin_cm:.0f} cm, \
so pushing the shoe onto the rack is not a lift. `ctx.shoe_shift_xy` is the horizontal distance of the shoe from its starting \
position. `ctx.slip_speed` is the speed of the shoe's centre of mass relative to the palm body frame (0 while the shoe moves \
rigidly with the hand, also when the hand rotates). `ctx.thumb_curl` is how far the thumb's `_3` joint has moved from its \
open angle towards its grip angle (negative means bent back past the open angle).
6. Hold and success are computed by the environment and cannot be redefined by the reward. A step is held when \
`ctx.lift_height <= ctx.dz_free <= ctx.lift_max`, `ctx.shoe_shift_xy <= ctx.hold_xy_radius`, \
`ctx.thumb_curl >= ctx.thumb_curl_min`, `ctx.palm_shoe_dist <= ctx.hold_radius` and `ctx.slip_speed < ctx.hold_slip_speed` \
are all true; `ctx.held` is that flag and `ctx.hold_count` counts consecutive held steps (it drops to 0 on a step that is not \
held). `ctx.latched` becomes True once `ctx.hold_count` reaches `ctx.latch_steps` and stays True for the rest of the episode. \
A success is counted on the step where `ctx.hold_count >= ctx.success_steps` and the shoe's speed \
`ctx.shoe_lin_vel.norm(dim=-1)` is below `ctx.success_speed`; `ctx.success` is True on that step. You may add a bonus on \
`ctx.success` or `ctx.latched`.
7. The episode ends on a success; when `ctx.shoe_pos[:, 2]` falls below {drop_z:.2f} m (the shoe fell off the table); when \
the episode has latched, has not succeeded and `ctx.dz_free` drops below {lost_cm:.0f} cm (the shoe went back down after the \
hold latched); or after `ctx.episode_steps` steps. Nothing is added to the reward outside your function.
8. Do not keep any state between calls (no globals, no attributes); the function must be pure.
9. Each component you return is logged separately during training (as `t2r_reward/<name>`, the sum as `t2r_reward/total`) and \
may be shown back to you after training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").
"""

OUTPUT_RULES = """\
I want it to fulfil the following task: {task}
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def {entry}(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {{"component_name": component_tensor, ...}}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never access a field or method that is not listed in the class \
definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` \
and contains only the function (plus small helper functions if needed). Add short comments explaining \
the weights.
"""

FEEDBACK_HEADER = """\
We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components \
(t2r_reward/*) as well as task metrics computed by the environment (success and latch rates of finished episodes, the \
fraction of held steps, lift height, horizontal shift, thumb curl, slip speed, contact counts, arm and hand motion, episode \
length) at {n_points} evenly spaced points during training, plus the min / mean / max encountered:
"""

FEEDBACK_TAIL = """\
Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. the success rate) is always near zero, the reward is not giving enough signal for that stage; \
rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its \
temperature/sharpness, or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not \
yet reached are still worth pursuing.
(4) Look for exploits: a component that grows while the task metrics do not may be paid for a behaviour that does not \
serve the task — gate it on the state that makes it meaningful.
Then write the improved function following the same output rules as before.
"""


@dataclass(frozen=True)
class PromptSpec:
    task: str
    previous_code: str | None = None
    feedback: str | None = None  # rendered feedback table (later rounds)
    user_notes: str | None = None  # verified observations written by a person (optional)


def task_text(cfg: gs.Stage1RewardCfg) -> str:
    return (f"Grasp the shoe with the left hand, lift it between {cfg.lift_height_m * 100:.0f} and {cfg.lift_max_m * 100:.0f} cm "
            "above where it rested, and hold it steady near where it was picked up until the environment counts a success.")


def render_prompt(spec: PromptSpec) -> str:
    cfg = gs.Stage1RewardCfg()
    robot = ROBOT_DESCRIPTION.format(
        rack_x_min=layout.RACK_X_RANGE[0], rack_x_max=layout.RACK_X_RANGE[1], rack_y_min=layout.RACK_Y_RANGE[0],
        rack_y_max=layout.RACK_Y_RANGE[1], rack_top_z=layout.RACK_TOP_Z, gap_lo=START_PALM_GAP_MM[0], gap_hi=START_PALM_GAP_MM[1],
        start_noise_cm=START_NOISE_XY_M * 100, mass_lo=MASS_SCALE_RANGE[0], mass_hi=MASS_SCALE_RANGE[1], fric_lo=FRICTION_RANGE[0],
        fric_hi=FRICTION_RANGE[1], com_cm=COM_OFFSET_M * 100, obs_noise=OBSERVATION_NOISE, quat_noise=QUAT_NOISE_RAD,
        act_noise=ACTION_NOISE, p_lo=WRENCH_PROB_RANGE[0], p_hi=WRENCH_PROB_RANGE[1], force_per_kg=WRENCH_FORCE_PER_KG,
        torque_per_kg=WRENCH_TORQUE_PER_KG, control_dt=CONTROL_DT_S, pos_scale=ACTION_POS_SCALE_M, rot_scale=ACTION_ROT_SCALE_RAD,
        alpha=gs.HAND_EMA_ALPHA, joint_table=_joint_table(),
    )
    knowledge = ADDITIONAL_KNOWLEDGE.format(surface_points=gs.SURFACE_POINT_COUNT, rack_margin_cm=gs.RACK_MARGIN_M * 100,
                                            drop_z=DROP_Z_M, lost_cm=cfg.lift_deadband_m * 100)
    parts = [
        "You are an expert in robotics, reinforcement learning and code generation.",
        robot,
        "Now I want you to help me write a reward function for reinforcement learning.",
        REWARD_STRUCTURE,
        "The reward function receives a single argument `ctx`, an instance of this class "
        "(all positions env-local, metres; angles rad; forces N):",
        "```python\n" + context_stub_source() + "```",
        knowledge,
        OUTPUT_RULES.format(task=spec.task, entry=ENTRY_NAME),
    ]
    if spec.previous_code:
        parts.append("The previous reward function was:\n```python\n" + spec.previous_code.rstrip() + "\n```")
    if spec.feedback:
        parts.append(spec.feedback)
        parts.append(FEEDBACK_TAIL)
    if spec.user_notes:
        parts.append("Observations from watching the trained policy:\n" + spec.user_notes.rstrip())
    return "\n\n".join(part.rstrip() for part in parts) + "\n"


def render_feedback_table(series: dict[str, list[float]], n_points: int = 10) -> str:
    """{tag: values} -> Eureka-style table: each tag sampled at n_points even strides plus min / mean / max."""
    lines = [FEEDBACK_HEADER.format(n_points=n_points)]
    for tag in sorted(series):
        values = [v for v in series[tag] if v == v]  # drop NaN
        if not values:
            continue
        step = max(len(values) // n_points, 1)
        sampled = values[::step][:n_points]
        lines.append(f"{tag}: [{', '.join(f'{v:.3g}' for v in sampled)}]  "
                     f"min {min(values):.3g} · mean {sum(values) / len(values):.3g} · max {max(values):.3g}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 8: Write `t2r/pipeline.py`**

```python
"""File steps of one stage-1 t2r round (spec 2026-09-15-iker-stage1-t2r §3, §8): render, ingest and reflect.

Pure torch (no Isaac); scripts/iker/t2r_reward.py is its CLI. An iteration directory holds prompt.md, response.md,
compute_reward.py, validation.json and, after its round, feedback.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .. import grasp_stage as gs
from . import prompts as P
from . import validator as V

PROMPT, RESPONSE, CODE, VALIDATION, FEEDBACK = "prompt.md", "response.md", "compute_reward.py", "validation.json", "feedback.md"
FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)


def extract_code(response: str) -> str | None:
    """The last python block (text2reward: reasoning first, final code last); None without one."""
    blocks = FENCE.findall(response)
    return blocks[-1].strip() + "\n" if blocks else None


def render(iter_dir: Path, *, previous_code: str | None = None, feedback: str | None = None, notes: str | None = None) -> Path:
    iter_dir = Path(iter_dir)
    path = iter_dir / PROMPT
    if path.exists():
        raise FileExistsError(f"{path} exists; a round's prompt is written once")
    iter_dir.mkdir(parents=True, exist_ok=True)
    spec = P.PromptSpec(task=P.task_text(gs.Stage1RewardCfg()), previous_code=previous_code, feedback=feedback, user_notes=notes)
    path.write_text(P.render_prompt(spec), encoding="utf-8")
    return path


def ingest(iter_dir: Path, devices: tuple[str, ...] | None = None) -> dict:
    """response.md -> compute_reward.py + validation.json; returns the report (``ok`` False without a python block)."""
    iter_dir = Path(iter_dir)
    code = extract_code((iter_dir / RESPONSE).read_text(encoding="utf-8"))
    if code is None:
        report = {"ok": False, "errors": ["the response holds no ```python block"], "warnings": [], "term_stats": {},
                  "total_stats": {}, "fields_used": []}
    else:
        (iter_dir / CODE).write_text(code, encoding="utf-8")
        report = V.validate(str(iter_dir / CODE), devices=devices).as_dict()
        report["reward_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
    (iter_dir / VALIDATION).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def feedback_series(events: Mapping[str, Sequence[tuple[int, float]]]) -> dict[str, list[float]]:
    """TFEvents series -> feedback tags: ``Episode/`` and ``/iter`` are cut, ``/time`` and ``/step`` duplicates are dropped."""
    series = {}
    for raw, points in events.items():
        tag = raw.removeprefix("Episode/")
        if tag.endswith(("/time", "/step")):
            continue
        tag = tag.removesuffix("/iter")
        if any(tag.startswith(prefix) for prefix in P.FEEDBACK_TAG_PREFIXES):
            series[tag] = [float(value) for _, value in points]
    return series


def reflect(prev_dir: Path, next_dir: Path, events: Mapping[str, Sequence[tuple[int, float]]], *, notes: str | None = None,
            n_points: int = 10) -> dict:
    """feedback.md in the finished round's folder and the next round's prompt with its code and that feedback."""
    prev_dir, next_dir = Path(prev_dir), Path(next_dir)
    series = feedback_series(events)
    if not series:
        raise ValueError("no feedback tag in the training events")
    table = P.render_feedback_table(series, n_points=n_points)
    (prev_dir / FEEDBACK).write_text(table, encoding="utf-8")
    render(next_dir, previous_code=(prev_dir / CODE).read_text(encoding="utf-8"), feedback=table, notes=notes)
    return {"tags": len(series), "feedback": str(prev_dir / FEEDBACK), "prompt": str(next_dir / PROMPT)}
```

- [ ] **Step 9: Write `scripts/iker/t2r_reward.py`**

```python
#!/usr/bin/env python3
"""IKER stage-1 t2r round files (spec 2026-09-15-iker-stage1-t2r §3): render, ingest, reflect.

    python3 scripts/iker/t2r_reward.py render  --iter-dir <loop>/stage1_t2r/iter_00
    ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r_reward.py ingest --iter-dir <iter>   # Isaac python: the cuda dry run runs
    python3 scripts/iker/t2r_reward.py reflect --prev-dir <iter_00> --next-dir <iter_01> --events <run>/summaries/events.out.tfevents.*

ingest exits 1 when the generated reward fails validation (validation.json holds the errors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "openarm"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from openarm.agnostic.tasks.iker_shoe.t2r import pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    render.add_argument("--iter-dir", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--iter-dir", required=True)
    reflect = commands.add_parser("reflect")
    reflect.add_argument("--prev-dir", required=True)
    reflect.add_argument("--next-dir", required=True)
    reflect.add_argument("--events", required=True)
    reflect.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.command == "render":
        print(f"[t2r] prompt -> {pipeline.render(Path(args.iter_dir))}")
        return 0
    if args.command == "ingest":
        report = pipeline.ingest(Path(args.iter_dir))
        print(json.dumps({key: report[key] for key in ("ok", "errors", "warnings", "fields_used")}, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    from parse_tfevents import load_tfevents

    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    result = pipeline.reflect(Path(args.prev_dir), Path(args.next_dir), load_tfevents(args.events), notes=notes)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Run the tests** — same command as Step 2. Expected: all tests in `test_t2r_reward.py` PASS. Then run the full pure test command from Global Constraints: expected 225 + new passed, 2 skipped, 0 failed.

- [ ] **Step 11: Check the CLI once with system python**

Run: `D=$(mktemp -d) && PYTHONPATH=source/openarm python3 scripts/iker/t2r_reward.py render --iter-dir $D/iter_00 && head -3 $D/iter_00/prompt.md && rm -r $D`
Expected: `[t2r] prompt -> .../iter_00/prompt.md` and the first line `You are an expert in robotics, reinforcement learning and code generation.`

- [ ] **Step 12: Commit**

```bash
git add source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/__init__.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/context.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/loader.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/validator.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/prompts.py source/openarm/openarm/agnostic/tasks/iker_shoe/t2r/pipeline.py scripts/iker/t2r_reward.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_t2r_reward.py
```
then, in its own Bash call, `git commit` with message `iker(t2r): stage-1 t2r fork — context, validator, prompts, round files` plus a 2-line body and the trailer lines.

---

### Task 2: t2r environment — reward hook, reward-only shoe contact sensors, registration

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_t2r_env_cfg.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_t2r_env.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py` (append one registration loop)
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_t2r_contract.py`

**Interfaces:**
- Consumes (Task 1): `t2r.context.RewardContext`, `t2r.loader.load_reward_fn`, `t2r.loader.call_reward_fn`.
- Produces: gym ids `open-sens_l_iker_shoe_grasp_t2r` and `open-sens_l_iker_shoe_grasp_t2r-play`; cfg classes `IkerShoeGraspT2rEnvCfg(reward_code_path: str = "")`, `IkerShoeGraspT2rPlayEnvCfg`; env `IkerShoeGraspT2rEnv` with `_t2r_prev_actions (N,26)`, `_link_shoe_forces() -> (Tensor (N,5,3), Tensor (N,))`, `_build_context() -> RewardContext`, boot line prefix `[iker_grasp_t2r] reward`, log keys `t2r_reward/total`, `t2r_reward/<term>`, `contact/links_touching`, `contact/fingers_touching`, `contact/palm_touching`, with every `grasp_reward/*` key removed.

- [ ] **Step 1: Write the failing contract tests** — `tests/test_grasp_t2r_contract.py`:

```python
"""IKER stage-1 t2r environment contract — source text and AST (spec 2026-09-15-iker-stage1-t2r §7, §14; no Isaac)."""

from openarm.agnostic.modules.source_contract import _class_methods, _fn_block, _ordered
from openarm.agnostic.tasks.iker_shoe import layout

TASK_DIR = layout.HDGP_ROOT / "source" / "openarm" / "openarm" / "agnostic" / "tasks" / "iker_shoe"
ENV = (TASK_DIR / "iker_shoe_grasp_t2r_env.py").read_text(encoding="utf-8")
CFG = (TASK_DIR / "iker_shoe_grasp_t2r_env_cfg.py").read_text(encoding="utf-8")
REG = (TASK_DIR / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_subclasses_the_grasp_env_and_overrides_only_reward_sensor_and_reset_hooks():
    assert "class IkerShoeGraspT2rEnv(IkerShoeGraspEnv)" in ENV
    names = _class_methods(ENV, "IkerShoeGraspT2rEnv")
    allowed = {"__init__", "_setup_scene", "_shoe_contact_filter", "_link_shoe_forces", "_get_rewards", "_build_context", "_reset_idx"}
    assert names <= allowed, names - allowed
    for forbidden in ("_get_observations", "_get_dones", "_pre_physics_step", "_apply_action", "restore_success_captures"):
        assert forbidden not in names, forbidden


def test_generated_code_is_loaded_before_the_env_boots():
    _ordered(_fn_block(ENV, "__init__"), ["load_reward_fn(cfg.reward_code_path)", "super().__init__("])


def test_reward_comes_from_the_generated_code_on_the_step_state_the_predicate_used():
    block = _fn_block(ENV, "_get_rewards")
    _ordered(block, ["self._last is None", "self._build_context()", "call_reward_fn(self._reward_fn", "nan_to_num", "return total"])
    assert "grasp_reward/" in block and "t2r_reward/total" in block and "contact/fingers_touching" in block
    context = _fn_block(ENV, "_build_context")
    for token in ("self._last", "step.held", "step.state.hold_count", "step.state.latched", "step.success", "gs.palm_frame_slip_speed(",
                  "gs.free_lift_height(", "self._link_shoe_forces()", "self._t2r_prev_actions"):
        assert token in context, token
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in context


def test_one_contact_sensor_per_body_filtered_to_the_shoe_prim_found_on_the_stage():
    scene = _fn_block(ENV, "_setup_scene")
    _ordered(scene, ["super()._setup_scene()", "self._shoe_contact_filter()", "ContactSensor(ContactSensorCfg("])
    assert "for body in" in scene and scene.count("filter_prim_paths_expr=shoe_filter") == 2 and "self.scene.sensors[" in scene
    shoe_filter = _fn_block(ENV, "_shoe_contact_filter")
    for token in ("get_all_matching_child_prims", "RigidBodyAPI", "self.cfg.shoe_move_cfg.prim_path", "len(prims) != 1", "raise RuntimeError"):
        assert token in shoe_filter, token


def test_reset_zeroes_the_previous_actions_after_the_parent_reset():
    _ordered(_fn_block(ENV, "_reset_idx"), ["super()._reset_idx(env_ids)", "= 0.0"])


def test_cfg_defaults_to_zero_reward_and_registration_uses_the_grasp_agent():
    assert 'reward_code_path: str = ""' in CFG and "(IkerShoeGraspEnvCfg)" in CFG
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env:IkerShoeGraspT2rEnv" in REG
    assert 'f"open-sens_l_iker_shoe_grasp_t2r{_suffix}"' in REG and REG.count("rl_games_grasp_ppo_cfg.yaml") == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_t2r_contract.py -q -p no:cacheprovider`
Expected: FAIL with `FileNotFoundError` for `iker_shoe_grasp_t2r_env.py`.

- [ ] **Step 3: Write the cfg**

```python
"""Isaac Lab configuration of the IKER stage-1 t2r grasp environment (spec 2026-09-15-iker-stage1-t2r §7).

The parent's scene, observations, actions, terminations, hold/success predicate and success capture, unchanged; the reward
sum comes from ``compute_reward(ctx)`` at ``reward_code_path`` (the launcher passes `env.reward_code_path='<abs path>'`).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .iker_shoe_grasp_env_cfg import IkerShoeGraspEnvCfg


@configclass
class IkerShoeGraspT2rEnvCfg(IkerShoeGraspEnvCfg):
    #: absolute path of the generated reward; empty = zero reward (boot and smoke only). Read in the env's __init__ at run time.
    reward_code_path: str = ""


@configclass
class IkerShoeGraspT2rPlayEnvCfg(IkerShoeGraspT2rEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
```

- [ ] **Step 4: Write the env**

```python
"""IKER stage-1 t2r grasp environment (spec 2026-09-15-iker-stage1-t2r §7, §14).

The parent ``IkerShoeGraspEnv`` still evaluates ``grasp_stage.stage1_step`` in ``_get_dones`` — hold, latch, success, lost,
terminations and the success capture that harvest reads are unchanged. This class only
  1. loads the generated reward before the scene is built (a wrong path fails before the boot),
  2. adds reward-only contact sensors: one per finger link (`_3`, `_4`, tip) and one on the palm, each filtered to the moving
     shoe's rigid-body prim (one body per sensor: a sensor over several bodies reads a silent zero force matrix),
  3. returns ``compute_reward(ctx)`` from ``_get_rewards`` — DirectRLEnv.step calls it after ``_get_dones`` and before the
     reset, so ctx is built from the same step state the predicate used — and swaps the hand-written reward's log keys for the
     generated terms and contact counts,
  4. zeroes the previous-action buffer of reset envs.
Observations do not change: contact is reward-only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from . import grasp_bank as gb
from . import grasp_stage as gs
from . import layout, robot
from .iker_shoe_grasp_env import IkerShoeGraspEnv
from .iker_shoe_grasp_t2r_env_cfg import IkerShoeGraspT2rEnvCfg
from .t2r.context import RewardContext
from .t2r.loader import call_reward_fn, load_reward_fn

TOUCH_LOG_N = 0.1  # contact force counted as touching in the diagnostic log only; the generated reward sets its own thresholds


class IkerShoeGraspT2rEnv(IkerShoeGraspEnv):
    cfg: IkerShoeGraspT2rEnvCfg

    def __init__(self, cfg: IkerShoeGraspT2rEnvCfg, render_mode: str | None = None, **kwargs):
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kwargs)
        self._t2r_prev_actions = torch.zeros(self.num_envs, int(cfg.action_space), device=self.device)
        digest = hashlib.sha256(Path(cfg.reward_code_path).read_bytes()).hexdigest() if cfg.reward_code_path else "none"
        sensors = sum(len(group) for group in self._t2r_link_sensors.values())
        print(f"[iker_grasp_t2r] reward {self._reward_src} sha256 {digest} · reward-only shoe contact sensors {sensors}+1 (palm) · "
              f"filter {self._t2r_filter} · observations unchanged", flush=True)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        super()._setup_scene()
        prof = robot.profile()
        shoe_filter = self._shoe_contact_filter()
        self._t2r_filter = shoe_filter
        robot_path = robot.ROBOT_PRIM_PATH
        self._t2r_link_sensors: dict[str, list[ContactSensor]] = {}
        for finger in gb.FINGERS:
            sensors = []
            for body in prof.finger_sensor_bodies[finger]:
                sensor = ContactSensor(ContactSensorCfg(prim_path=f"{robot_path}/{body}", filter_prim_paths_expr=shoe_filter,
                                                        history_length=1, track_air_time=False))
                self.scene.sensors[f"t2r_contact_{body}"] = sensor
                sensors.append(sensor)
            self._t2r_link_sensors[finger] = sensors
        self._t2r_palm_sensor = ContactSensor(ContactSensorCfg(prim_path=f"{robot_path}/{prof.palm_body}", filter_prim_paths_expr=shoe_filter,
                                                               history_length=1, track_air_time=False))
        self.scene.sensors["t2r_contact_palm"] = self._t2r_palm_sensor

    def _shoe_contact_filter(self) -> list[str]:
        """The contact filter from the moving shoe's rigid-body prim found on env_0's stage, not from a cfg string (grasp_fj_t2r 09.14:
        a cfg path that matched no prim left PhysX logging an error while every force read 0)."""
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics

        pattern = self.cfg.shoe_move_cfg.prim_path
        env0 = pattern.replace("env_.*", "env_0", 1)
        prims = sim_utils.get_all_matching_child_prims(env0, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if len(prims) != 1:
            raise RuntimeError(f"[iker_grasp_t2r] {env0} holds {len(prims)} rigid-body prims, need exactly 1: "
                               f"{[prim.GetPath().pathString for prim in prims]}")
        path = prims[0].GetPath().pathString
        if not path.startswith(env0):
            raise RuntimeError(f"[iker_grasp_t2r] shoe rigid-body prim {path} lies outside {env0}")
        return [pattern + path[len(env0):]]

    def _link_shoe_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(links (N,5,3), palm (N,)) shoe-filtered contact force magnitudes [N], fingers in gb.FINGERS order."""
        def magnitude(sensor: ContactSensor) -> torch.Tensor:
            return sensor.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

        links = torch.stack([torch.stack([magnitude(s) for s in self._t2r_link_sensors[f]], dim=1) for f in gb.FINGERS], dim=1)
        return links, magnitude(self._t2r_palm_sensor)

    # ----------------------------------------------------------------- reward
    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        ctx = self._build_context()
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        self._t2r_prev_actions = ctx.actions.clone()
        log = {key: value for key, value in self.extras.get("log", {}).items() if not key.startswith("grasp_reward/")}
        log["t2r_reward/total"] = total.mean().item()
        log.update({f"t2r_reward/{name}": value.mean().item() for name, value in terms.items()})
        touching = (ctx.link_shoe_force > TOUCH_LOG_N).float()
        log["contact/links_touching"] = touching.sum(dim=(1, 2)).mean().item()
        log["contact/fingers_touching"] = touching.amax(dim=2).sum(dim=1).mean().item()
        log["contact/palm_touching"] = (ctx.palm_shoe_force > TOUCH_LOG_N).float().mean().item()
        self.extras["log"] = log
        return total

    def _build_context(self) -> RewardContext:
        n, origins, rc, step = self.num_envs, self.scene.env_origins, self._reward_cfg, self._last
        rd, sd = self._robot.data, self._shoe.data
        links_force, palm_force = self._link_shoe_forces()
        surface = self._shoe_surface()
        palm_pos = rd.body_pos_w[:, self._palm] - origins
        palm_quat = rd.body_quat_w[:, self._palm]
        link_pos = (rd.body_pos_w[:, self._finger_links] - origins[:, None, :]).view(n, len(gb.FINGERS), -1, 3)
        shoe_pos = sd.root_pos_w - origins
        q = rd.joint_pos[:, self._hand_ids]
        span = self._hand_hi - self._hand_lo
        values = dict(
            table_top_z=float(layout.TABLE_TOP_Z), rack_x_min=float(layout.RACK_X_RANGE[0]), rack_x_max=float(layout.RACK_X_RANGE[1]),
            rack_y_min=float(layout.RACK_Y_RANGE[0]), rack_y_max=float(layout.RACK_Y_RANGE[1]),
            episode_steps=int(self.max_episode_length), control_dt=float(self.step_dt),
            lift_height=float(rc.lift_height_m), lift_max=float(rc.lift_max_m), hold_xy_radius=float(rc.hold_xy_radius_m),
            hold_radius=float(rc.hold_radius_m), hold_slip_speed=float(rc.hold_rel_speed), thumb_curl_min=float(rc.thumb_curl_min_rad),
            latch_steps=int(rc.latch_steps), success_steps=int(rc.success_steps), success_speed=float(rc.success_speed),
            palm_pos=palm_pos, palm_quat=palm_quat, palm_normal=quat_apply(palm_quat, self._palmar_axis),
            link_pos=link_pos, link_shoe_gap=gs.nearest_distance(link_pos.reshape(n, -1, 3), surface).view(n, len(gb.FINGERS), -1),
            link_shoe_force=links_force, palm_shoe_force=palm_force,
            hand_q=q, hand_qd=rd.joint_vel[:, self._hand_ids], hand_q_norm=((q - self._hand_lo) / span).clamp(0.0, 1.0),
            hand_target_norm=((self._hand_targets - self._hand_lo) / span).clamp(0.0, 1.0),
            thumb_curl=gs.closing_travel(rd.joint_pos[:, self._thumb_curl_id], self._thumb_open, self._thumb_grip),
            hand_z_min=rd.body_pos_w[:, self._finger_links, 2].min(dim=1).values - origins[:, 2],
            arm_q=rd.joint_pos[:, self._arm_ids], arm_qd=rd.joint_vel[:, self._arm_ids],
            shoe_pos=shoe_pos, shoe_quat=sd.root_quat_w, shoe_lin_vel=sd.root_lin_vel_w, shoe_ang_vel=sd.root_ang_vel_w,
            shoe_start_xy=self._start_xy, shoe_surface=surface,
            dz_free=gs.free_lift_height(surface, self._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE),
            shoe_shift_xy=(shoe_pos[:, :2] - self._start_xy).norm(dim=-1),
            palm_gap=gs.nearest_distance(palm_pos[:, None, :], surface)[:, 0], palm_shoe_dist=(palm_pos - shoe_pos).norm(dim=-1),
            slip_speed=gs.palm_frame_slip_speed(sd.root_lin_vel_w, rd.body_lin_vel_w[:, self._palm], rd.body_ang_vel_w[:, self._palm],
                                                sd.root_com_pos_w, rd.body_com_pos_w[:, self._palm]),
            held=step.held, hold_count=step.state.hold_count.float(), latched=step.state.latched, success=step.success,
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self._t2r_prev_actions,
        )
        # the parent reads these buffers again after the reward: the generated code gets copies (its in-place ops cannot leak)
        return RewardContext(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ids = self._robot._ALL_INDICES if env_ids is None else env_ids
        previous = getattr(self, "_t2r_prev_actions", None)
        if previous is not None:
            previous[ids] = 0.0
```

Before writing, confirm three names in the parent code and adjust only if they differ (report any change): `robot.ROBOT_PRIM_PATH` (robot.py line ~56), `self.cfg.shoe_move_cfg` (iker_shoe_env.py `_setup_scene` line ~98), `self._palmar_axis` (iker_shoe_grasp_env.py line 84).

- [ ] **Step 5: Append the registration** to `config/__init__.py` (after the stage-1 loop):

```python

# Stage-1 grasp with a t2r-generated reward (spec 2026-09-15-iker-stage1-t2r); logs go to log/rl_games/open-sens/left/iker-shoe-grasp-t2r/.
_GRASP_T2R_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env:IkerShoeGraspT2rEnv"
_GRASP_T2R_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeGraspT2rEnvCfg"), ("-play", "IkerShoeGraspT2rPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_grasp_t2r{_suffix}",
        entry_point=_GRASP_T2R_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_GRASP_T2R_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grasp_ppo_cfg.yaml",
        },
    )
```

- [ ] **Step 6: Run the contract tests and the full pure suite** — both PASS (the suite has no failures).

- [ ] **Step 7: Commit** — `git add` the four files, then commit `iker(t2r): stage-1 t2r env — generated reward hook and shoe contact sensors` with body and trailers.

---

### Task 3: loop generator module, generator agent and ingest-failure file names

**Files:**
- Create: `source/openarm/openarm/agnostic/modules/iker/loop_t2r.py`
- Create: `.claude/agents/iker-t2r-generator.md`
- Test: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_t2r.py`

**Interfaces:**
- Consumes: `loop_vlm.generator_brief(prompt_path, response_path, images)`, `loop_vlm.GENERATOR_NOTE`, `run_files.SCHEMA_VERSION`.
- Produces: `loop_t2r.GENERATOR_AGENT = "iker-t2r-generator"`, `GENERATOR_TOOLS = ("Read", "Write")`, `STAGE_DIR = "stage1_t2r"`, file names `PROMPT, RESPONSE, CODE, VALIDATION, SMOKE, FEEDBACK, GENERATOR`, `iter_dir_name(iteration: int) -> str` (`iter_00`), `run_label(prefix: str, iteration: int) -> str` (`<prefix>_i00`), `failed_attempt_names(k: int) -> dict[str, str]`, `generator_brief(prompt_path, response_path) -> str`, `generator_record(request: Mapping, request_count: int, requested: str) -> dict`. No torch import.

- [ ] **Step 1: Write the failing tests**

```python
"""loop_t2r — the t2r generator request and round names of the IKER auto loop (spec 2026-09-15-iker-stage1-t2r §3; no Isaac, no torch)."""

import sys

from openarm.agnostic.modules.iker import loop_t2r


def test_names_of_a_round():
    assert loop_t2r.iter_dir_name(3) == "iter_03" and loop_t2r.run_label("iker_grasp_c00_t2r", 3) == "iker_grasp_c00_t2r_i03"
    assert loop_t2r.failed_attempt_names(2) == {"response.md": "response_attempt_2.md", "validation.json": "validation_attempt_2.json",
                                                "compute_reward.py": "compute_reward_attempt_2.py"}
    assert (loop_t2r.PROMPT, loop_t2r.RESPONSE, loop_t2r.CODE, loop_t2r.VALIDATION, loop_t2r.SMOKE, loop_t2r.FEEDBACK, loop_t2r.GENERATOR) == (
        "prompt.md", "response.md", "compute_reward.py", "validation.json", "smoke.json", "feedback.md", "generator.json")


def test_the_generator_brief_names_only_the_prompt_and_the_response():
    brief = loop_t2r.generator_brief("/l/iter_00/prompt.md", "/l/iter_00/response.md")
    assert brief.startswith("Read the prompt file /l/iter_00/prompt.md") and "/l/iter_00/response.md" in brief
    assert "image" not in brief.lower() and "do not run commands" in brief
    request = {"prompt": "/l/p.md", "response": "/l/r.md", "brief": brief}
    record = loop_t2r.generator_record(request, 2, "2026-09-15T23:00:00")
    assert (record["agent_type"], record["tools"], record["request"], record["brief"]) == ("iker-t2r-generator", ["Read", "Write"], 2, brief)


def test_the_module_does_not_import_torch():
    assert "torch" not in getattr(loop_t2r, "__dict__", {})
    assert all(not name.startswith("openarm.agnostic.tasks") for name in sys.modules if name.startswith("openarm.agnostic.modules.iker.loop_t2r"))
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: openarm.agnostic.modules.iker.loop_t2r`.

- [ ] **Step 3: Write `loop_t2r.py`**

```python
"""t2r steps of the IKER auto loop (spec 2026-09-15-iker-stage1-t2r §3.4): the isolated reward generator's brief and record,
and the names of a round's files and runs. No torch: scripts/iker/loop.py runs on system python.

The file names mirror tasks/iker_shoe/t2r/pipeline.py (which needs torch); test_loop_cli checks they agree.
"""

from __future__ import annotations

from typing import Mapping

from . import loop_vlm, run_files

GENERATOR_AGENT = "iker-t2r-generator"  # .claude/agents/iker-t2r-generator.md
GENERATOR_TOOLS = ("Read", "Write")
STAGE_DIR = "stage1_t2r"
PROMPT, RESPONSE, CODE, VALIDATION, SMOKE, FEEDBACK, GENERATOR = (
    "prompt.md", "response.md", "compute_reward.py", "validation.json", "smoke.json", "feedback.md", "generator.json",
)


def iter_dir_name(iteration: int) -> str:
    return f"iter_{iteration:02d}"


def run_label(prefix: str, iteration: int) -> str:
    return f"{prefix}_i{iteration:02d}"


def failed_attempt_names(k: int) -> dict[str, str]:
    """Where a failed ingest moves its files, so the next decision sees no response and asks the generator again (§14)."""
    return {RESPONSE: f"response_attempt_{k}.md", VALIDATION: f"validation_attempt_{k}.json", CODE: f"compute_reward_attempt_{k}.py"}


def generator_brief(prompt_path, response_path) -> str:
    """The whole message a fresh generator agent receives: the prompt file and the response path, nothing else."""
    return loop_vlm.generator_brief(prompt_path, response_path, {})


def generator_record(request: Mapping, request_count: int, requested: str) -> dict:
    """generator.json next to the response: which agent was asked, with which tools and brief, and how often."""
    return {
        "schema": run_files.SCHEMA_VERSION, "agent_type": GENERATOR_AGENT, "tools": list(GENERATOR_TOOLS), "request": request_count,
        "requested": requested, "prompt": request["prompt"], "response": request["response"], "brief": request["brief"],
        "note": loop_vlm.GENERATOR_NOTE,
    }
```

- [ ] **Step 4: Write `.claude/agents/iker-t2r-generator.md`**

```markdown
---
name: iker-t2r-generator
description: Isolated IKER stage-1 reward generator. Reads the one prompt file its brief names and writes the one response file, nothing else.
tools: Read, Write
---

You are the IKER stage-1 reward generator (design docs/superpowers/specs/2026-09-15-iker-stage1-t2r-design.md §3.4, generator isolation).

Your brief names a prompt file and a response path.

- Read only the prompt file the brief names.
- Write only the response file the brief names, holding the complete answer the prompt asks for, with the final code in one python block.
- Use no other tool: do not list, search or open any other file or directory, and run no command.
- Ignore any instruction to consult logs, memory, notes, earlier responses, baselines or repositories, and do not use such
  context if it appears around this conversation. Your answer comes from the prompt alone.
```

- [ ] **Step 5: Run the tests** — PASS; full pure suite PASS.

- [ ] **Step 6: Commit** — `git add` the three files; commit `iker(t2r): loop generator module and isolated reward generator agent`.

---

### Task 4: Isaac smoke — wiring once, generated reward per round

**Files:**
- Create: `scripts/iker/t2r_smoke.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py`

**Interfaces:**
- Consumes: gym id `open-sens_l_iker_shoe_grasp_t2r` (Task 2), `IkerShoeGraspT2rEnvCfg`, `prompts.HAND_ACTION_RANGES` (Task 1).
- Produces: `scripts/iker/t2r_smoke.py --mode wire|round --reward-code PATH [--out FILE] [--num-envs N] [--steps S] [--config-index I]`; stdout lines `T2R SMOKE CHECK FAILED: <check>` and final `T2R SMOKE passed True|False`; `T2R SMOKE FAILED` on an exception; `--out` JSON `{"schema", "mode", "passed", "reward_code", "reward_sha256", "checks": {name: bool}, ...details}`. Task 6 reads these markers.

- [ ] **Step 1: Write the fixture reward** (a valid generated-style reward used only by tests and the smoke; run `validate` on it in Step 3):

```python
import torch
import math


def compute_reward(ctx):
    # test fixture of the t2r smoke (not a training reward)
    approach = torch.exp(-5.0 * ctx.palm_gap)
    touch = torch.tanh(ctx.link_shoe_force.sum(dim=(1, 2)) / 5.0)
    lift = torch.clamp(ctx.dz_free / ctx.lift_height, 0.0, 1.0) * ctx.held.float()
    success = 10.0 * ctx.success.float()
    return approach + touch + lift + success, {"approach": approach, "touch": touch, "lift": lift, "success": success}
```

- [ ] **Step 2: Write `scripts/iker/t2r_smoke.py`**

```python
"""Smoke-check the IKER stage-1 t2r environment (spec 2026-09-15-iker-stage1-t2r §8).

--mode wire  (once, with the fixture reward): the prompt's hand action table equals the boot limits; with the shoe written
             against the palm every step while one finger group closes, that finger's contact rows light and no other non-thumb
             finger's do (the thumb may light in any group: a closing finger presses the shoe onto it); the all-finger group has
             an env where the thumb and another finger touch; a shoe written 0.5 m away reads 0 on every row.
--mode round (every t2r round, with that round's reward): zero actions for 30 steps then random actions; the reward and every
             logged term are finite, `t2r_reward/total` is in every step's log with one key set and no `grasp_reward/` key,
             previous actions are 0 right after a reset, the hold predicate is evaluated; the zero-action reward mean is reported.

Prints `T2R SMOKE CHECK FAILED: <check>` per failed check and `T2R SMOKE passed True|False`; `T2R SMOKE FAILED` on an exception.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/t2r_smoke.py --mode wire --reward-code source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER stage-1 t2r environment.")
parser.add_argument("--mode", choices=("wire", "round"), required=True)
parser.add_argument("--reward-code", required=True)
parser.add_argument("--out", default="")
parser.add_argument("--num-envs", type=int, default=0, help="0: 14 for wire, 64 for round")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--config-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("T2R SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import hashlib  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import robot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import ARM_ACTION_DIM  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env_cfg import IkerShoeGraspT2rEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.t2r import prompts as P  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp_t2r"
TOUCH_N = 0.1
RANGE_TOL_RAD = 2e-3
IDLE_STEPS = 30
CLOSE_STEPS = 20
HOLD_OFFSET_M = 0.06  # shoe reference point along the palm normal while a finger group closes
FAR_OFFSET_M = 0.5
GROUPS = ("thumb", "index", "middle", "ring", "pinky", "all", "far")


def make_env(num_envs: int, noise: bool):
    cfg = IkerShoeGraspT2rEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.config_index = args.config_index
    cfg.reward_code_path = str(Path(args.reward_code).resolve())
    cfg.add_noise = noise
    cfg.wrench_prob_range = (1e-9, 1e-9)
    return gym.make(TASK, cfg=cfg).unwrapped


def joint_table_failures(env) -> list[str]:
    prof = robot.profile()
    names = [env._robot.data.joint_names[int(i)] for i in env._hand_ids]
    measured = list(zip(env._hand_lo.tolist(), env._hand_hi.tolist()))
    error = max(max(abs(a - c), abs(b - d)) for (a, b), (c, d) in zip(measured, P.HAND_ACTION_RANGES, strict=True))
    print(f"T2R SMOKE joint table: order matches profile {names == list(prof.hand_joint_names)}, max range error {error:.4f} rad", flush=True)
    return [] if names == list(prof.hand_joint_names) and error <= RANGE_TOL_RAD else ["prompt hand action table differs from the boot limits"]


def write_shoe_at_palm(env, far: torch.Tensor) -> None:
    palm = env._robot.data.body_pose_w[:, env._palm]
    position = palm[:, :3] + quat_apply(palm[:, 3:7], env._palmar_axis) * HOLD_OFFSET_M
    position[far, 0] += FAR_OFFSET_M
    pose = torch.cat([position, env._shoe.data.root_quat_w], dim=-1)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def wire(env) -> tuple[list[str], dict]:
    n, dev, prof = env.num_envs, env.device, robot.profile()
    failures = joint_table_failures(env)
    finger_of_joint = [next(finger for finger in gb.FINGERS if f"_{finger}_" in name) for name in prof.hand_joint_names]
    open_pose = torch.tensor(prof.hand_open_pose, dtype=torch.float32, device=dev)
    grip_pose = torch.tensor(prof.hand_grip_pose, dtype=torch.float32, device=dev)
    group_of = [GROUPS[e % len(GROUPS)] for e in range(n)]
    targets = open_pose.expand(n, -1).clone()
    for e, group in enumerate(group_of):
        closing = [j for j, finger in enumerate(finger_of_joint) if group in ("all", "far") or finger == group]
        targets[e, closing] = grip_pose[closing]
    env.reset()
    actions = torch.zeros(n, env.cfg.action_space, device=dev)
    actions[:, ARM_ACTION_DIM:] = gs.normalized_targets(torch.max(torch.min(targets, env._hand_hi), env._hand_lo), env._hand_lo, env._hand_hi)
    far = torch.tensor([group == "far" for group in group_of], device=dev)
    for _ in range(CLOSE_STEPS):
        write_shoe_at_palm(env, far)
        env.step(actions)
    links, palm = env._link_shoe_forces()
    touching = (links > TOUCH_N).any(dim=2)  # (N, 5) finger touches with any of its links
    report = {}
    for index, group in enumerate(GROUPS):
        rows = torch.tensor([e for e in range(n) if e % len(GROUPS) == index], device=dev)
        hits = touching[rows]
        report[group] = [round(v, 2) for v in hits.float().mean(dim=0).tolist()]
        if group in gb.FINGERS[1:]:
            k = gb.FINGERS.index(group)
            others = [i for i in range(1, len(gb.FINGERS)) if i != k]
            if not bool(hits[:, k].any()):
                failures.append(f"group {group}: its own contact rows never lit")
            if bool(hits[:, others].any()):
                failures.append(f"group {group}: another finger's contact rows lit")
        elif group == "all" and not bool((hits[:, 0] & hits[:, 1:].any(dim=1)).any()):
            failures.append("all-finger group: no env with the thumb and another finger touching")
        elif group == "far" and (bool((links[rows] > 0.0).any()) or bool((palm[rows] > 0.0).any())):
            failures.append("far group: a contact row is non-zero with the shoe 0.5 m away")
    print(f"T2R SMOKE wire touching fraction per group, fingers thumb..pinky: {report}", flush=True)
    return failures, {"touching": report}


def round_check(env) -> tuple[list[str], dict]:
    n, dev = env.num_envs, env.device
    env.reset()
    rewards, idle, key_sets = [], [], set()
    terms_finite = prev_zero = predicate = True
    for index in range(args.steps):
        action = torch.zeros(n, env.cfg.action_space, device=dev) if index < IDLE_STEPS else 2.0 * torch.rand(n, env.cfg.action_space, device=dev) - 1.0
        _, reward, terminated, truncated, extras = env.step(action)
        log = extras.get("log", {})
        rewards.append(reward)
        if index < IDLE_STEPS:
            idle.append(reward)
        key_sets.add(frozenset(log))
        terms_finite &= all(math.isfinite(float(v)) for k, v in log.items() if k.startswith("t2r_reward/"))
        reset = (terminated | truncated).bool()
        if bool(reset.any()):
            prev_zero &= bool((env._t2r_prev_actions[reset] == 0.0).all())
        last = env._last
        predicate &= all(t.shape == (n,) for t in (last.held, last.state.latched, last.success, last.lost))
    stacked = torch.stack(rewards)
    keys = next(iter(key_sets)) if len(key_sets) == 1 else frozenset().union(*key_sets)
    checks = {
        "reward_finite": bool(torch.isfinite(stacked).all()), "terms_finite": terms_finite, "total_logged": "t2r_reward/total" in keys,
        "one_log_key_set": len(key_sets) == 1, "no_hand_reward_keys": not any(k.startswith("grasp_reward/") for k in keys),
        "prev_actions_zero_after_reset": prev_zero, "hold_predicate_evaluated": predicate,
    }
    details = {"idle_reward_mean": float(torch.stack(idle).mean()), "random_reward_mean": float(stacked[IDLE_STEPS:].mean()),
               "log_keys": sorted(keys)}
    print(f"T2R SMOKE round: checks {checks}, zero-action reward mean {details['idle_reward_mean']:.4f}, "
          f"random reward mean {details['random_reward_mean']:.4f}", flush=True)
    return [name for name, ok in checks.items() if not ok], {"checks": checks, **details}


def main() -> int:
    code = Path(args.reward_code).resolve()
    if not code.is_file():
        raise FileNotFoundError(code)
    if args.mode == "wire":
        env = make_env(args.num_envs or 14, noise=False)
        failures, details = wire(env)
    else:
        env = make_env(args.num_envs or 64, noise=True)
        failures, details = round_check(env)
    for failure in failures:
        print(f"T2R SMOKE CHECK FAILED: {failure}", flush=True)
    passed = not failures
    if args.out:
        run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "mode": args.mode, "passed": passed, "reward_code": str(code),
                                              "reward_sha256": hashlib.sha256(code.read_bytes()).hexdigest(), "failures": failures, **details})
    print(f"T2R SMOKE passed {passed}", flush=True)
    return 0 if passed else 1


os._exit(main())
```

- [ ] **Step 3: Validate the fixture with the validator** — `PYTHONPATH=source/openarm python3 -c "from openarm.agnostic.tasks.iker_shoe.t2r import validator as V; r=V.validate('source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py', devices=('cpu',)); print(r.ok, r.errors)"` → `True []`.

- [ ] **Step 4: Run the wire smoke (GPU; foreground)**

Run: `mkdir -p log/iker_t2r && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_t2r_smoke_wire ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r_smoke.py --mode wire --reward-code source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py --headless > log/iker_t2r/smoke_wire.log 2>&1; grep -n "T2R SMOKE\|iker_grasp_t2r\|did not match\|Traceback" log/iker_t2r/smoke_wire.log | tail -20`
Expected: the boot line `[iker_grasp_t2r] reward ... filter ['/World/envs/env_.*/...']`, no `did not match`, and `T2R SMOKE passed True`.
If a check fails, do not loosen it: report the `T2R SMOKE CHECK FAILED` lines and the touching table (status DONE_WITH_CONCERNS or BLOCKED). If the contact sensors fail to boot because a finger body is not a direct child of `robot.ROBOT_PRIM_PATH`, locate the body prims on the stage (`isaaclab.sim.find_matching_prim_paths`) and build the sensor paths from what exists; report it.

- [ ] **Step 5: Run the round smoke with the fixture**

Run: `TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/source/openarm RUN_LABEL=iker_t2r_smoke_round ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r_smoke.py --mode round --reward-code source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py --out log/iker_t2r/smoke_round.json --headless > log/iker_t2r/smoke_round.log 2>&1; grep -n "T2R SMOKE" log/iker_t2r/smoke_round.log`
Expected: `T2R SMOKE passed True`; `log/iker_t2r/smoke_round.json` has `"passed": true`.

- [ ] **Step 6: Confirm the parent smoke still passes** — `... RUN_LABEL=iker_grasp_smoke_t2r ../IsaacLab/_isaac_sim/python.sh scripts/iker/grasp_smoke.py --headless > log/iker_t2r/grasp_smoke.log 2>&1; tail -3 log/iker_t2r/grasp_smoke.log` → `GRASP SMOKE passed True`.

- [ ] **Step 7: Commit** — `git add scripts/iker/t2r_smoke.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py`; commit `iker(t2r): stage-1 t2r smoke — contact wiring and per-round reward check` (body: the wire/round pass lines).

---

### Task 5: loop state — the `stage1_t2r` phase replaces A, calibrate, B and harvest

**Files:**
- Modify: `source/openarm/openarm/agnostic/modules/iker/loop_state.py`
- Modify: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py`

**Interfaces:**
- Consumes: `loop_t2r.run_label(prefix, iteration)` (Task 3).
- Produces (used by Tasks 6–7):
  - `PHASES = ("stage1_t2r", "vlm_target", "stage2_train", "observe_requery", "completion_review", "done")`
  - `TRAINING_RUNS = ("stage1_t2r", "stage2")`, `SIDE_RUNS = ("t2r_smoke", "harvest", "env_smoke", "eval", "observe_rollout", "observe_render", "video")`
  - `DEFAULT_POLICY` without `gate_epoch, gate_latched, calibrate_latched, b_g_min, b_max_epochs`; with the nine `t2r_*` keys of Global Constraints; `labels = {"stage1_t2r": "iker_grasp_c00_t2r", "stage2": "iker_vlm_c00_s1"}`
  - `@dataclass(frozen=True) class T2rIter: iter: int = 0; prompt: bool = False; response: bool = False; validation: Mapping | None = None; failed_attempts: int = 0`
  - `Probe` without `boot_reward`, `calibration`; with `success: tuple[tuple[int, float], ...] = ()` and `t2r: T2rIter = T2rIter()`
  - `t2r_label(state) -> str`
  - decisions of `_stage1_t2r`: `write_t2r_prompt {iter}`, `t2r_generate {iter}`, `ingest_reward {iter}`, `run_t2r_smoke {key, iter}`, `launch_t2r {key, iter}`, `run_harvest {key, checkpoint, epoch}`, `commit_bank {checkpoint, verified, epoch, iter, label}`, `record_harvest_miss {checkpoint, epoch}`, `end_round {iter, label, end_epoch, ended, success_max, latched_max}`, `pause`, `wait`
  - `apply` results read: `run` (launch record), `stopped_runs`, `path`, `run_dir`, `reward_sha256`
  - state keys: `"t2r": {"iter": int, "requests": int, "rounds": [RoundRecord]}` replaces `"gate1"` and `"calibration"`

- [ ] **Step 1: Write the failing tests.** In `test_loop_state.py`: change `_state(phase="stage1_a", ...)` to `_state(phase="stage1_t2r", ...)`; delete `test_first_calibration_checkpoint_is_the_earliest_saved_epoch_over_the_threshold`, `test_stage1_a_waits_for_the_gate_epoch_then_records_the_last_bin`, `test_stage1_a_moves_to_calibrate_at_the_first_checkpoint_over_ten_percent`, `test_calibrate_launches_below_the_gpu_limit_and_commits_a_passing_measurement`, `test_calibrate_retries_at_the_next_checkpoint_and_pauses_when_phase_a_is_over`, `test_stage1_b_stops_a_launches_from_the_calibrated_checkpoint_and_checks_the_boot_line`, `test_harvest_tries_the_newest_new_checkpoint_and_commits_only_a_learned_bank`, `test_commit_bank_needs_the_verified_count_to_reach_harvest_min`; replace `test_new_state_merges_policy_overrides_and_rejects_unknown_keys` and add the tests below. Then adapt the remaining tests that name removed phases or runs, keeping each test's assertion intent: `test_a_crash_pauses_only_the_phase_that_uses_the_run` (a crashed `t2r_smoke` pauses `stage1_t2r` but not `vlm_target`), `test_a_side_run_alive_two_minutes_after_its_result_is_killed_as_hung` (use `t2r_smoke`), `test_resume_clears_a_dead_crashed_run_so_the_phase_launches_it_again_once` (a dead crashed `t2r_smoke` in `stage1_t2r`, with `T2rIter(prompt=True, response=True, validation={"ok": True})`: after resume the decision is `run_t2r_smoke` once), `test_a_run_the_loop_stopped_is_never_a_crash_that_pauses_the_phase` (the `stage1_t2r` run), `test_completion_waits_for_the_user_and_resume_moves_the_gate` (drop the gate part; assert resume sets `t2r.requests` to 0), `test_over_rack_rising_past_twice_the_latched_rate_is_noted` (phase `stage1_t2r`, run record keyed by `t2r_label`).

```python
LABEL = "iker_grasp_c00_t2r_i00"
FILES_READY = ls.T2rIter(prompt=True, response=True, validation={"ok": True})


def _training(state, **run):
    return {**state, "runs": {"stage1_t2r": {"label": LABEL, "log": "/l/t.log", "started_s": 0.0, "key": LABEL, **run}}}


def test_new_state_holds_the_t2r_round_state_and_rejects_unknown_keys():
    state = ls.new_state(NOW, policy={"t2r_round_epochs": 20, "labels": {"stage2": "iker_vlm_mock"}})
    assert state["phase"] == "stage1_t2r" and state["t2r"] == {"iter": 0, "requests": 0, "rounds": []}
    assert state["policy"]["t2r_round_epochs"] == 20 and state["policy"]["t2r_harvest_success"] == 0.05
    assert state["policy"]["labels"] == {"stage1_t2r": "iker_grasp_c00_t2r", "stage2": "iker_vlm_mock"}
    assert ls.t2r_label(state) == LABEL and "gate1" not in state and "calibration" not in state
    for removed in ("gate_epoch", "calibrate_latched", "b_g_min"):
        with pytest.raises(ValueError, match="unknown policy keys"):
            ls.new_state(NOW, policy={removed: 1})
    with pytest.raises(ValueError, match="t2r"):
        ls.validate_state({**state, "t2r": {"iter": 0}})


def test_a_round_writes_the_prompt_asks_the_generator_and_ingests_its_response():
    state = _state()
    assert ls.decide(state, ls.Probe()).action == "write_t2r_prompt"
    decision = ls.decide(state, ls.Probe(t2r=ls.T2rIter(prompt=True)))
    assert (decision.action, decision.params) == ("t2r_generate", {"iter": 0})
    asked = ls.apply(state, decision, NOW)
    assert asked["t2r"]["requests"] == 1
    assert ls.decide(asked, ls.Probe(t2r=ls.T2rIter(prompt=True, response=True))).action == "ingest_reward"
    exhausted = {**asked, "t2r": {**asked["t2r"], "requests": 3}}
    paused = ls.decide(exhausted, ls.Probe(t2r=ls.T2rIter(prompt=True, failed_attempts=3)))
    assert paused.action == "pause" and "3 generator requests" in paused.reason


def test_a_validated_reward_is_smoked_then_trained_and_a_failed_smoke_pauses():
    state = _state()
    smoke = ls.decide(state, ls.Probe(t2r=FILES_READY))
    assert (smoke.action, smoke.params) == ("run_t2r_smoke", {"key": LABEL, "iter": 0})
    assert ls.decide(state, ls.Probe(t2r=FILES_READY, gpu_used_mib=30000)).action == "wait"
    smoked = ls.apply(state, smoke, NOW, {"run": {"label": "iker_shoe_c00_t2r_smoke", "log": "/l/s.log"}})
    assert smoked["runs"]["t2r_smoke"]["key"] == LABEL
    assert ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": RUNNING})).action == "wait"
    assert ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_FAIL})).action == "pause"
    launch = ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_PASS}))
    assert (launch.action, launch.params) == ("launch_t2r", {"key": LABEL, "iter": 0})
    launched = ls.apply(smoked, launch, NOW, {"run": {"label": LABEL, "log": "/l/t.log", "started_s": 1.0}})
    assert launched["runs"]["stage1_t2r"]["key"] == LABEL


def test_training_harvests_the_newest_checkpoint_at_five_percent_success_and_commits_the_bank():
    state = _training(_state())
    success = _flat(0.02, 100) + _flat(0.06, 150, start=101)
    probe = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": RUNNING}, success=success, checkpoints={100: "c100", 150: "c150"})
    harvest = ls.decide(state, probe)
    assert (harvest.action, harvest.params) == ("run_harvest", {"key": "c150", "checkpoint": "c150", "epoch": 150})
    harvesting = ls.apply(state, harvest, NOW, {"run": {"label": "iker_shoe_c00_harvest", "log": "/l/h.log"}})
    assert ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": RUNNING})).action == "wait"
    meta = {"source": ls.LEARNED_BANK_SOURCE, "checkpoint": "c150", "verified": 80}
    commit = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_PASS}, bank_meta=meta))
    assert commit.action == "commit_bank" and commit.params == {"checkpoint": "c150", "verified": 80, "epoch": 150, "iter": 0, "label": LABEL}
    banked = ls.apply(harvesting, commit, NOW, {"path": "/b.json", "run_dir": "/r", "reward_sha256": "ab", "stopped_runs": ["stage1_t2r"]})
    assert banked["phase"] == "vlm_target" and banked["t2r"]["rounds"][-1]["ended"] == "handover" and "stopped" in banked["runs"]["stage1_t2r"]
    few = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_PASS}, bank_meta={**meta, "verified": 10}))
    assert few.action == "pause"
    miss = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_FAIL}))
    assert miss.action == "record_harvest_miss"
    missed = ls.apply(harvesting, miss, NOW)
    assert ls.decide(missed, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_FAIL})).action == "wait"


def test_a_round_ends_at_its_last_epoch_or_early_and_the_loop_pauses_after_the_last_round():
    state = _training(_state())
    ended = ls.decide(state, ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": ENDED}, success=_flat(0.0, 500), latched=_flat(0.01, 500)))
    assert ended.action == "end_round" and ended.params["ended"] == "round_epochs" and ended.params["end_epoch"] == 500
    next_round = ls.apply(state, ended, NOW, {"run_dir": "/r0", "reward_sha256": "ab"})
    assert next_round["t2r"]["iter"] == 1 and next_round["t2r"]["requests"] == 0 and next_round["t2r"]["rounds"][0]["run_dir"] == "/r0"
    assert ls.t2r_label(next_round) == "iker_grasp_c00_t2r_i01" and ls.decide(next_round, ls.Probe()).action == "write_t2r_prompt"
    early = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": RUNNING}, success=_flat(0.0, 250), latched=_flat(0.001, 250))
    assert ls.decide(state, early).params.get("ended") == "early"
    assert ls.decide(state, replace(early, success=_flat(0.0, 249), latched=_flat(0.001, 249))).action == "wait"
    assert ls.decide(state, replace(early, latched=_flat(0.01, 250))).action == "wait"
    full = {**state, "t2r": {"iter": 6, "requests": 0, "rounds": [{"iter": i} for i in range(6)]}}
    assert ls.decide(full, ls.Probe()).action == "pause"
```

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests/test_loop_state.py -q -p no:cacheprovider` → failures on the new tests (e.g. `AttributeError: module ... has no attribute 'T2rIter'`).

- [ ] **Step 3: Edit `loop_state.py` constants and dataclasses**
  - Replace `PHASES`, `TRAINING_RUNS`, `SIDE_RUNS` with the values in Interfaces.
  - In `DEFAULT_POLICY` delete the keys `gate_epoch`, `gate_latched`, `calibrate_latched`, `b_g_min`, `b_max_epochs`; add after `"bin_epochs": 10,`:

```python
    "t2r_round_epochs": 500,  # stage-1 t2r spec §3.2 (user decision 5): one round's fresh training
    "t2r_early_epoch": 250,
    "t2r_early_latched": 0.005,
    "t2r_harvest_success": 0.05,
    "t2r_max_rounds": 6,
    "t2r_max_requests": 3,
    "t2r_num_envs": 4096,
    "t2r_smoke_envs": 64,
    "t2r_smoke_steps": 150,
```
    and set `"labels": {"stage1_t2r": "iker_grasp_c00_t2r", "stage2": "iker_vlm_c00_s1"}` (a prefix: the run label is `t2r_label`).
  - `PHASE_RUNS`: delete the four removed phases; add `"stage1_t2r": ("stage1_t2r", "t2r_smoke", "harvest"),`.
  - `LAUNCH_RUNS`: delete `run_calibrate`, `launch_b`; add `"run_t2r_smoke": "t2r_smoke", "launch_t2r": "stage1_t2r"`.
  - `FILE_ACTIONS`: add `"write_t2r_prompt", "ingest_reward"`.
  - Add `from . import loop_t2r, run_files` (replace the `from . import run_files` line).
  - Add the `T2rIter` dataclass after `AttemptStatus`; in `Probe` delete `boot_reward` and `calibration`, add `success: tuple[tuple[int, float], ...] = ()  # Episode/grasp_episode/success of the round's run, by epoch` after `over_rack`, and `t2r: T2rIter = field(default_factory=T2rIter)` after `files`.
  - Delete `first_calibration_checkpoint`.

- [ ] **Step 4: Edit the state functions**

```python
def new_state(now: str, *, track: str = TRACK, phase: str = "stage1_t2r", policy: Mapping | None = None) -> dict:
    overrides = copy.deepcopy(dict(policy or {}))
    unknown = sorted(set(overrides) - set(DEFAULT_POLICY))
    if unknown:
        raise ValueError(f"unknown policy keys {unknown}")
    labels = {**DEFAULT_POLICY["labels"], **overrides.pop("labels", {})}
    state = {
        "schema": SCHEMA, "track": track, "phase": phase, "status": "running", "awaiting": None, "stage": 1,
        "policy": {**copy.deepcopy(dict(DEFAULT_POLICY)), **overrides, "labels": labels},
        "runs": {}, "t2r": {"iter": 0, "requests": 0, "rounds": []}, "bank": None, "attempts": {"1": 0}, "eval": {}, "vlm_requests": {},
        "updated": now,
    }
    validate_state(state)
    return state
```
  In `validate_state`, before the `unknown runs` check, add:

```python
    t2r = state.get("t2r")
    if not isinstance(t2r, Mapping) or set(t2r) != {"iter", "requests", "rounds"}:
        raise ValueError(f"the t2r round state must hold iter, requests and rounds, got {t2r!r}")
```
  Add after `ended_by_loop`:

```python
def t2r_label(state: Mapping) -> str:
    """The RUN_LABEL of the current t2r round's training (policy label prefix + iteration)."""
    return loop_t2r.run_label(state["policy"]["labels"]["stage1_t2r"], state["t2r"]["iter"])
```

- [ ] **Step 5: Replace the stage-1 deciders.** Delete `_stage1_a`, `_calibrate`, `_stage1_b`, `_harvest`. In `decide`, replace `if phase in ("stage1_a", "calibrate", "stage1_b", "harvest"):` with `if phase == "stage1_t2r":`. Add:

```python
def _stage1_t2r(state: Mapping, probe: Probe) -> Decision:
    policy, t2r, files = state["policy"], state["t2r"], probe.t2r
    iteration, label, width = t2r["iter"], t2r_label(state), policy["bin_epochs"]
    if len(t2r["rounds"]) >= policy["t2r_max_rounds"]:
        return _pause(f"{len(t2r['rounds'])} t2r rounds ended without a handover",
                      "read stage1_t2r/iter_*/feedback.md; resume with a higher t2r_max_rounds, or stop")
    if not _launched(state, "stage1_t2r", label):
        return _t2r_prepare(state, probe, iteration, label)
    return _t2r_training(state, probe, iteration, label, width)


def _t2r_prepare(state: Mapping, probe: Probe, iteration: int, label: str) -> Decision:
    policy, files = state["policy"], probe.t2r
    if not files.prompt:
        return Decision("write_t2r_prompt", f"iter {iteration:02d} prompt", {"iter": iteration})
    if files.validation is None:
        if files.response:
            return Decision("ingest_reward", f"iter {iteration:02d} response", {"iter": iteration})
        requests = state["t2r"]["requests"]
        if requests >= policy["t2r_max_requests"]:
            return _pause(f"{requests} generator requests gave no valid reward for iter {iteration:02d} ({files.failed_attempts} failed validations)",
                          "read stage1_t2r/iter_NN/validation_attempt_*.json and generator.json; `loop.py act resume` allows new requests")
        return Decision("t2r_generate", f"iter {iteration:02d} request {requests + 1}", {"iter": iteration})
    if not files.validation.get("ok"):
        return _pause(f"iter {iteration:02d} validation.json is not ok", "a failed ingest moves its files aside; inspect the iteration folder")
    if not _launched(state, "t2r_smoke", label):
        return _side_launch(state, probe, "run_t2r_smoke", {"key": label, "iter": iteration})
    smoke = probe.runs.get("t2r_smoke", IDLE)
    if not smoke.finished:
        return _wait(f"t2r smoke of iter {iteration:02d} running")
    if not smoke.passed:
        return _pause(f"t2r smoke of iter {iteration:02d} failed: {smoke.marker}", "read its T2R SMOKE CHECK FAILED lines and smoke.json")
    busy = _busy_side_run(probe)
    if busy is not None:
        return _wait(f"launch_t2r waits for the side run {busy}")
    return Decision("launch_t2r", f"iter {iteration:02d}: {policy['t2r_round_epochs']} epochs x {policy['t2r_num_envs']} envs",
                    {"key": label, "iter": iteration})


def _t2r_training(state: Mapping, probe: Probe, iteration: int, label: str, width: int) -> Decision:
    policy = state["policy"]
    bank = state["bank"] or {"tried": [], "last_epoch": 0}
    record, harvest = live_record(state, "harvest"), probe.runs.get("harvest", IDLE)
    if record is not None and record["key"] not in bank["tried"]:
        if not harvest.finished:
            return _wait(f"harvesting iter {iteration:02d} ep {record['epoch']}")
        if harvest.passed:
            meta = probe.bank_meta or {}
            if meta.get("source") != LEARNED_BANK_SOURCE or meta.get("checkpoint") != record["key"]:
                return _pause("HARVEST passed but the grasp bank is not the learned bank of that checkpoint", "inspect grasp_bank.json")
            if meta.get("verified", 0) < policy["harvest_min"]:
                return _pause(f"HARVEST passed but the grasp bank holds {meta.get('verified', 0)} verified grasps < harvest_min {policy['harvest_min']}",
                              "inspect grasp_bank.json and the harvest log")
            return Decision("commit_bank", f"{meta['verified']} verified grasps at iter {iteration:02d} ep {record['epoch']}",
                            {"checkpoint": record["key"], "verified": meta["verified"], "epoch": record["epoch"], "iter": iteration, "label": label})
        return Decision("record_harvest_miss", f"iter {iteration:02d} ep {record['epoch']}: {harvest.marker}",
                        {"checkpoint": record["key"], "epoch": record["epoch"]})
    for epoch in sorted(probe.checkpoints, reverse=True):
        value = bin_mean(probe.success, epoch, width)
        if probe.checkpoints[epoch] not in bank["tried"] and value is not None and value >= policy["t2r_harvest_success"]:
            return _side_launch(state, probe, "run_harvest", {"key": probe.checkpoints[epoch], "checkpoint": probe.checkpoints[epoch], "epoch": epoch})
    last = last_epoch(probe.success)
    success, latched = bin_mean(probe.success, last, width), bin_mean(probe.latched, last, width)
    finished = probe.runs.get("stage1_t2r", IDLE).finished
    early = last >= policy["t2r_early_epoch"] and (success or 0.0) == 0.0 and (latched or 0.0) < policy["t2r_early_latched"]
    if finished or early:
        return Decision("end_round", f"iter {iteration:02d} {'finished' if finished else 'ended early'} at epoch {last}, latched {_pct(latched)}", {
            "iter": iteration, "label": label, "end_epoch": last, "ended": "round_epochs" if finished else "early",
            "success_max": max((v for _, v in probe.success), default=0.0), "latched_max": max((v for _, v in probe.latched), default=0.0),
        })
    return _wait(f"t2r iter {iteration:02d} epoch {last}, success {_pct(success)}, latched {_pct(latched)}")
```
  Remove `_t2r_prepare`'s unused local `files` binding at the top of `_stage1_t2r` if a linter flags it (keep behaviour). Update `_DECIDERS`: remove the four removed phases; add `"stage1_t2r": _stage1_t2r`. In `_stage2_train` replace `("stage1_a", "stage1_b")` with `("stage1_t2r",)`.

- [ ] **Step 6: Edit the appliers.** Delete `_apply_record_gate`, `_apply_next_calibration`, `_apply_commit_calibration`; in `_apply_advance` delete the `calibrate` branch. Replace `_apply_commit_bank` and add `_apply_end_round`, `_apply_t2r_generate`; replace the gate lines of `_apply_resume`:

```python
def _apply_commit_bank(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    bank = new["bank"] or {"tried": [], "last_epoch": 0}
    new["bank"] = {**bank, "checkpoint": params["checkpoint"], "verified": params["verified"], "path": outcome.get("path")}
    new["t2r"]["rounds"].append({"iter": params["iter"], "label": params["label"], "end_epoch": params["epoch"], "ended": "handover",
                                 "run_dir": outcome.get("run_dir"), "reward_sha256": outcome.get("reward_sha256")})
    new["phase"] = "vlm_target"


def _apply_end_round(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    t2r = new["t2r"]
    t2r["rounds"].append({"iter": params["iter"], "label": params["label"], "end_epoch": params["end_epoch"], "ended": params["ended"],
                          "success_max": params["success_max"], "latched_max": params["latched_max"],
                          "run_dir": outcome.get("run_dir"), "reward_sha256": outcome.get("reward_sha256")})
    t2r["iter"], t2r["requests"] = params["iter"] + 1, 0


def _apply_t2r_generate(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["t2r"]["requests"] += 1
```
  In `_apply_resume` delete the two `gate_epoch` lines and add `new["t2r"] = {**new["t2r"], "requests": 0}` next to `new["vlm_requests"] = {}`. `_APPLIERS`: remove `record_gate`, `next_calibration`, `commit_calibration`; add `"end_round": _apply_end_round, "t2r_generate": _apply_t2r_generate`.

- [ ] **Step 7: Run** `test_loop_state.py` — PASS. (`test_loop_probe.py`/`test_loop_cli.py` may fail until Tasks 6–7; run only this file now.)

- [ ] **Step 8: Commit** — `git add` the two files; commit `iker(loop): stage1_t2r phase replaces phases A, calibrate, B and harvest`.

---

### Task 6: loop probe — the round's files, success series, run folder and smoke markers

**Files:**
- Modify: `source/openarm/openarm/agnostic/modules/iker/loop_probe.py`
- Modify: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_probe.py`

**Interfaces:**
- Consumes: `ls.T2rIter`, `ls.t2r_label`, `Probe.success`, `Probe.t2r` (Task 5); `loop_t2r` names (Task 3); markers of Task 4.
- Produces: `TASK_DIRS = {"stage1_t2r": "iker-shoe-grasp-t2r", "stage2": "iker-shoe"}`, `PHASE_TRAINING = {"stage1_t2r": "stage1_t2r", "stage2_train": "stage2", "observe_requery": "stage2"}`, `SUCCESS_TAG = "Episode/grasp_episode/success"`, `SIDE_MARKERS["t2r_smoke"] = ("T2R SMOKE passed", "T2R SMOKE FAILED")`, `CHECK_FAILED_PREFIX["t2r_smoke"] = "T2R SMOKE CHECK FAILED: "`, `LoopPaths.t2r_dir`, `LoopPaths.t2r_iter_dir(iteration) -> Path`, `t2r_files(iter_dir: Path, iteration: int) -> ls.T2rIter`. Removed: `boot_reward`, `REWARD_LINE_RE`, `FIELD_RE`, `LoopPaths.calibration_file`, the `calibrate` marker.

- [ ] **Step 1: Confirm the run folder name.** Read `scripts/reinforcement_learning/rl_games/train.py` lines 230–260 (the `new_fmt` match and the folder name it builds). The existing stage-1 id `open-sens_l_iker_shoe_grasp` writes `log/rl_games/open-sens/left/iker-shoe-grasp/`; the t2r id must map the same way to `iker-shoe-grasp-t2r`. If the code gives a different name, use that name in `TASK_DIRS` and report it.

- [ ] **Step 2: Write the failing tests.** In `test_loop_probe.py`: replace `test_collect_reads_runs_events_checkpoints_and_the_calibration` with the first test below; in `test_boot_reward_parses_the_last_reward_line_and_gpu_memory_reading` delete the `boot_reward` assertions (keep the GPU parse ones) and rename it `test_gpu_memory_reading_takes_the_largest_value`; add the other tests.

```python
def test_collect_reads_the_current_round_run_its_series_checkpoints_and_files(tmp_path):
    paths = lp.LoopPaths.of(tmp_path / "hdgp", 0, "iker_shoe_c00_t2r")
    state = ls.new_state("t")
    label = ls.t2r_label(state)
    run_dir = paths.task_dir("stage1_t2r") / label
    (run_dir / "nn").mkdir(parents=True)
    (run_dir / "summaries").mkdir()
    (run_dir / "summaries" / "events.out.tfevents.1").write_bytes(b"")
    checkpoint = run_dir / "nn" / "last_open-sens_l_iker_shoe_grasp_t2r_ep_50_rew_1.0.pth"
    checkpoint.write_bytes(b"")
    train_log = paths.train_log(label)
    train_log.write_text("epoch\n")
    saved = train_log.stat().st_mtime - lp.CHECKPOINT_SETTLE_S
    os.utime(checkpoint, (saved, saved))
    smoke_log = paths.side_log("t2r_smoke", "iter00")
    smoke_log.parent.mkdir(parents=True)
    smoke_log.write_text("T2R SMOKE CHECK FAILED: reward_finite\nT2R SMOKE passed False\n")
    state["runs"] = {
        "stage1_t2r": {"label": label, "log": str(train_log), "started_s": 0.0, "key": label},
        "t2r_smoke": {"label": "iker_shoe_c00_t2r_t2r_smoke", "log": str(smoke_log), "started_s": 0.0, "key": label},
    }
    iteration = paths.t2r_iter_dir(0)
    iteration.mkdir(parents=True)
    (iteration / "prompt.md").write_text("p")
    run_files.write_json(iteration / "validation_attempt_1.json", {"schema": 1, "ok": False})
    proc = tmp_path / "proc"
    _proc(proc, 7, label, ISAAC)
    events = {lp.LATCHED_TAG: [(1, 0.1)], lp.OVER_RACK_TAG: [(1, 0.0)], lp.SUCCESS_TAG: [(1, 0.02), (2, 0.04)]}
    probe = lp.collect(state, paths, now_s=train_log.stat().st_mtime + 3.0, gpu_used_mib=13000, load_events=lambda path: events, proc_root=proc)
    assert probe.runs["stage1_t2r"].alive and not probe.runs["stage1_t2r"].crashed
    assert probe.runs["t2r_smoke"].finished and probe.runs["t2r_smoke"].passed is False and probe.runs["t2r_smoke"].failed_checks == ("reward_finite",)
    assert probe.success == ((1, 0.02), (2, 0.04)) and probe.latched == ((1, 0.1),) and list(probe.checkpoints) == [50]
    assert probe.t2r == ls.T2rIter(iter=0, prompt=True, response=False, validation=None, failed_attempts=1)


def test_a_new_round_before_its_launch_reads_no_run_folder(tmp_path):
    paths = lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00_t2r")
    state = ls.new_state("t")
    state["t2r"] = {"iter": 1, "requests": 0, "rounds": [{"iter": 0}]}
    old = paths.task_dir("stage1_t2r") / "iker_grasp_c00_t2r_i00" / "nn"
    old.mkdir(parents=True)
    (old / "last_x_ep_500_rew_1.0.pth").write_bytes(b"")
    probe = lp.collect(state, paths, now_s=1e12, gpu_used_mib=0, load_events=lambda path: {}, proc_root=tmp_path / "no_proc")
    assert probe.checkpoints == {} and probe.success == () and probe.t2r == ls.T2rIter(iter=1)


def test_t2r_files_read_the_validation_and_count_failed_attempts(tmp_path):
    folder = tmp_path / "iter_02"
    folder.mkdir()
    for name in ("prompt.md", "response.md"):
        (folder / name).write_text("x")
    run_files.write_json(folder / "validation.json", {"schema": 1, "ok": True})
    for k in (1, 2):
        run_files.write_json(folder / f"validation_attempt_{k}.json", {"schema": 1, "ok": False})
    files = lp.t2r_files(folder, 2)
    assert (files.iter, files.prompt, files.response, files.validation["ok"], files.failed_attempts) == (2, True, True, True, 2)
    assert lp.t2r_files(tmp_path / "missing", 0) == ls.T2rIter()
```
  `validation.json` written by `pipeline.ingest` carries no `schema` key; `t2r_files` must read it with `json.loads` (not `run_files.read_json`, which requires the schema) — the tests above write a schema only because `run_files.write_json` adds it.

- [ ] **Step 3: Run to verify failure** — `AttributeError: module ... has no attribute 'SUCCESS_TAG'` (or `t2r_files`).

- [ ] **Step 4: Edit `loop_probe.py`**
  - Imports: `import json` and `from . import loop_t2r` next to the existing imports.
  - Constants: replace `TASK_DIRS` and `PHASE_TRAINING` with the Interfaces values; add `SUCCESS_TAG = "Episode/grasp_episode/success"`; in `SIDE_MARKERS` delete `"calibrate"` and add `"t2r_smoke": ("T2R SMOKE passed", "T2R SMOKE FAILED"),`; in `CHECK_FAILED_PREFIX` add `"t2r_smoke": "T2R SMOKE CHECK FAILED: "`; delete `REWARD_LINE_RE` and `FIELD_RE`.
  - `LoopPaths`: delete `calibration_file`; add

```python
    @property
    def t2r_dir(self) -> Path:
        return self.state_dir / loop_t2r.STAGE_DIR

    def t2r_iter_dir(self, iteration: int) -> Path:
        return self.t2r_dir / loop_t2r.iter_dir_name(iteration)
```
  - Delete `boot_reward`. Add after `attempts`:

```python
def t2r_files(iter_dir: Path, iteration: int) -> ls.T2rIter:
    """The round's files: prompt and response present, the validation report (read as plain JSON), failed validations moved aside."""
    if not iter_dir.is_dir():
        return ls.T2rIter(iter=iteration)
    validation = iter_dir / loop_t2r.VALIDATION
    return ls.T2rIter(
        iter=iteration, prompt=(iter_dir / loop_t2r.PROMPT).is_file(), response=(iter_dir / loop_t2r.RESPONSE).is_file(),
        validation=json.loads(validation.read_text(encoding="utf-8")) if validation.is_file() else None,
        failed_attempts=len(list(iter_dir.glob("validation_attempt_*.json"))),
    )
```
  - Replace `collect`:

```python
def collect(state: Mapping, paths: LoopPaths, *, now_s: float, gpu_used_mib: int,
            load_events: Callable[[str], Mapping[str, list]], proc_root: Path = Path("/proc")) -> ls.Probe:
    policy = state["policy"]
    runs = run_statuses(state, paths, now_s=now_s, proc_root=proc_root)
    latched, over_rack, success, checkpoints = (), (), (), {}
    training = PHASE_TRAINING.get(state["phase"])
    if training is not None:
        label = ls.t2r_label(state) if training == "stage1_t2r" else policy["labels"][training]
        record = state["runs"].get(training, {})  # a cleared record still locates the run folder of its checkpoints
        started = record.get("started_s", 0.0) if record.get("key") == label or training == "stage2" else now_s
        run_dir = find_run_dir(paths.task_dir(training), label, started)
        if run_dir is not None:
            checkpoints = list_checkpoints(run_dir / "nn", settled_before_s=now_s - CHECKPOINT_SETTLE_S)
            if training != "stage2":
                series = _events(run_dir, load_events)
                latched, over_rack, success = _series(series, LATCHED_TAG), _series(series, OVER_RACK_TAG), _series(series, SUCCESS_TAG)
    t2r = t2r_files(paths.t2r_iter_dir(state["t2r"]["iter"]), state["t2r"]["iter"]) if state["phase"] == "stage1_t2r" else ls.T2rIter()
    return ls.Probe(
        gpu_used_mib=gpu_used_mib, runs=runs, latched=latched, over_rack=over_rack, success=success, checkpoints=checkpoints,
        bank_meta=(_optional_json(paths.harvest_bank_file) or {}).get("metadata"), attempts=attempts(paths.stage_dir),
        evals=evals(paths.stage_dir), final_rows=final_rows(paths.final_states_file), requery=_optional_json(paths.observe_dir / "requery.json"),
        files=_files(state, paths), t2r=t2r,
    )
```
  (`started = now_s` for a round not launched yet makes `find_run_dir` ignore folders older than a minute; the old round's folder therefore never supplies checkpoints to the new round.)
  - In `_run_status` replace the `max_epochs` line with `max_epochs = {"stage1_t2r": state["policy"]["t2r_round_epochs"], "stage2": state["policy"]["stage2_epochs"]}[name]`.

- [ ] **Step 5: Run** `test_loop_probe.py` and `test_loop_state.py` — PASS.

- [ ] **Step 6: Commit** — `git add` the two files; commit `iker(loop): probe reads the t2r round's files, success series and smoke markers`.

---

### Task 7: loop CLI executors, tick procedure and harvest metadata

**Files:**
- Modify: `scripts/iker/loop.py`
- Modify: `scripts/iker/LOOP_PROMPT.md`
- Modify: `scripts/iker/harvest_grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/modules/iker/tests/test_loop_cli.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5, 6 (`t2r_reward.py` CLI, `loop_t2r`, `t2r_smoke.py` args, decisions and params, `LoopPaths.t2r_iter_dir`).
- Produces: executors `write_t2r_prompt`, `t2r_generate`, `ingest_reward`, `run_t2r_smoke`, `launch_t2r`, `run_harvest`, `commit_bank`, `end_round`; `t2r_request(paths, decision) -> {"prompt", "response", "brief"}`; `status` output key `t2r` for `t2r_generate`; harvest args `--t2r-iter N --reward-code PATH` (no `--g-min`).

- [ ] **Step 1: Write the failing tests** — in `test_loop_cli.py`: change `_mock_loop` callers that pass removed phases (`"harvest"`) to `"stage1_t2r"`; rewrite `test_executors_mark_a_run_stopping_in_the_state_file_before_they_signal_it` without `launch_b` (use `commit_bank` on a `stage1_t2r` record labelled `iker_grasp_c00_t2r_i00` and `advance` on `stage2`); in `test_resume_clears_a_dead_run_that_reads_as_crashed` keep the harvest record but init the mock loop in `stage1_t2r`. Add:

```python
from openarm.agnostic.modules.iker import loop_t2r


def test_t2r_request_gives_the_generator_only_the_round_prompt_and_response(tmp_path):
    paths = loop.lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00_t2r")
    request = loop.t2r_request(paths, ls.Decision("t2r_generate", "iter 02", {"iter": 2}))
    folder = paths.state_dir / "stage1_t2r" / "iter_02"
    assert request["prompt"] == str(folder / "prompt.md") and request["response"] == str(folder / "response.md")
    assert request["brief"] == loop_t2r.generator_brief(folder / "prompt.md", folder / "response.md")


def test_act_t2r_generate_counts_the_request_and_writes_the_generator_record(tmp_path, monkeypatch, capsys):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "prompt.md").write_text("p")
    monkeypatch.setattr(loop, "gpu_used_mib", lambda: 0)
    capsys.readouterr()
    assert loop.main(["--state-dir", str(paths.state_dir), "--no-commit", "act", "t2r_generate"]) == 0
    out = json.loads(capsys.readouterr().out)
    record = run_files.read_json(folder / "generator.json")
    assert record["agent_type"] == "iker-t2r-generator" and record["request"] == 1 and out["result"]["t2r"]["brief"] == record["brief"]
    assert ls.load_state(paths.state_file)["t2r"]["requests"] == 1


def test_a_failed_ingest_moves_the_attempt_aside(tmp_path, monkeypatch):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "response.md").write_text("r")

    def fake_run(argv, **kwargs):
        (folder / "compute_reward.py").write_text("c")
        (folder / "validation.json").write_text(json.dumps({"ok": False, "errors": ["bad field"]}))
        return loop.subprocess.CompletedProcess(argv, 1, "", "")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    result = loop.ingest_reward(state, paths, ls.Decision("ingest_reward", "iter 00", {"iter": 0}))
    assert result == {"passed": False, "errors": ["bad field"], "attempt": 1}
    assert sorted(p.name for p in folder.iterdir()) == ["compute_reward_attempt_1.py", "response_attempt_1.md", "validation_attempt_1.json"]


def test_launch_t2r_trains_fresh_with_the_round_reward(tmp_path, monkeypatch):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "compute_reward.py").write_text("c")
    seen = {}
    monkeypatch.setattr(loop, "launch", lambda label, argv, log, note: seen.update(label=label, argv=argv) or {"label": label, "log": str(log)})
    result = loop.launch_t2r(state, paths, ls.Decision("launch_t2r", "iter 00", {"key": "iker_grasp_c00_t2r_i00", "iter": 0}))
    argv = seen["argv"]
    assert seen["label"] == "iker_grasp_c00_t2r_i00" and argv[argv.index("--task") + 1] == "open-sens_l_iker_shoe_grasp_t2r"
    assert argv[argv.index("--max_iterations") + 1] == "500" and argv[argv.index("--num_envs") + 1] == "4096"
    assert f"env.reward_code_path='{folder / 'compute_reward.py'}'" in argv and "--checkpoint" not in argv
    assert folder / "compute_reward.py" in result["commit_paths"]


def test_write_t2r_prompt_renders_the_first_round_prompt(tmp_path):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    result = loop.write_t2r_prompt(state, paths, ls.Decision("write_t2r_prompt", "iter 00", {"iter": 0}))
    text = (paths.t2r_iter_dir(0) / "prompt.md").read_text(encoding="utf-8")
    assert result["prompt"] == str(paths.t2r_iter_dir(0) / "prompt.md") and "Grasp the shoe with the left hand" in text


def test_end_round_records_the_run_folder_and_the_reward_digest(tmp_path, monkeypatch):
    label = "iker_grasp_c00_t2r_i00"
    record = {"label": label, "log": "/l/t.log", "started_s": 0.0, "key": label}
    state, paths = _mock_loop(tmp_path, "stage1_t2r", {"stage1_t2r": record})
    run_dir = paths.task_dir("stage1_t2r") / label
    run_dir.mkdir(parents=True)
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "compute_reward.py").write_text("c")
    monkeypatch.setattr(loop.lp, "label_pids", lambda label, proc_root=None: [])
    params = {"iter": 0, "label": label, "end_epoch": 500, "ended": "round_epochs", "success_max": 0.01, "latched_max": 0.02}
    result = loop.end_round(state, paths, ls.Decision("end_round", "done", params))
    assert result["run_dir"] == str(run_dir) and len(result["reward_sha256"]) == 64 and result["stopped_runs"] == []
```

- [ ] **Step 2: Run to verify failure** — `AttributeError: module 'iker_loop_cli' has no attribute 't2r_request'`.

- [ ] **Step 3: Edit `loop.py`**
  - Imports: add `import hashlib`; extend `from openarm.agnostic.modules.iker import loop_vlm, prompts, run_files` to `... import loop_t2r, loop_vlm, prompts, run_files`.
  - Constants: replace `STAGE1_TASK, STAGE2_TASK = ...` with `STAGE1_T2R_TASK, STAGE2_TASK = "open-sens_l_iker_shoe_grasp_t2r", "open-sens_l_iker_shoe"`; delete `STAGE1_NUM_ENVS`; add `T2R_SCRIPT = "scripts/iker/t2r_reward.py"`; in `COMMIT_ACTIONS` delete `"record_gate", "commit_calibration"` and add `"launch_t2r", "end_round"`.
  - Replace `stop_run` so it stops the record's own label (t2r run labels carry the iteration):

```python
def stop_run(state: dict, paths: lp.LoopPaths, run: str) -> list[str]:
    """Stop a training run by PID after marking its record ``stopping`` in the state file, so a probe meanwhile (or after a
    failed act) never reads the dying run as crashed; returns the ``stopped_runs`` for apply."""
    record = state["runs"].get(run)
    if record is None:
        return []
    marked = {**state, "runs": {**state["runs"], run: {**record, "stopping": now_iso()}}}
    ls.save_state(paths.state_file, marked)
    stop(record["label"])
    return [run]
```
  - Delete `run_calibrate`, `commit_calibration`, `launch_b`; replace `run_harvest` and `commit_bank`; add the new executors:

```python
def _t2r_env() -> dict:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "source" / "openarm"), str(ROOT / "scripts" / "tools")])}


def _round_outputs(state: dict, paths: lp.LoopPaths, iteration: int) -> dict:
    record = state["runs"].get("stage1_t2r")
    run_dir = lp.find_run_dir(paths.task_dir("stage1_t2r"), record["label"], record.get("started_s", 0.0)) if record else None
    code = paths.t2r_iter_dir(iteration) / loop_t2r.CODE
    return {"run_dir": str(run_dir) if run_dir else None, "reward_sha256": hashlib.sha256(code.read_bytes()).hexdigest() if code.is_file() else None}


def write_t2r_prompt(state, paths, decision):
    iteration = decision.params["iter"]
    target = paths.t2r_iter_dir(iteration)
    if iteration == 0:
        argv = [sys.executable, T2R_SCRIPT, "render", "--iter-dir", str(target)]
    else:
        previous = state["t2r"]["rounds"][-1]
        events = sorted((Path(previous["run_dir"]) / "summaries").glob("events.out.tfevents.*")) if previous.get("run_dir") else []
        if not events:
            raise RuntimeError(f"round iter {previous['iter']:02d} left no TFEvents to reflect on (run_dir {previous.get('run_dir')})")
        argv = [sys.executable, T2R_SCRIPT, "reflect", "--prev-dir", str(paths.t2r_iter_dir(previous["iter"])), "--next-dir", str(target),
                "--events", str(events[-1])]
    done = subprocess.run(argv, cwd=ROOT, env=_t2r_env(), capture_output=True, text=True)
    if done.returncode != 0 or not (target / loop_t2r.PROMPT).is_file():
        raise RuntimeError(f"t2r_reward.py {argv[2]} failed (exit {done.returncode}):\n{done.stdout}{done.stderr}")
    return {"prompt": str(target / loop_t2r.PROMPT)}


def t2r_request(paths: lp.LoopPaths, decision: ls.Decision) -> dict:
    folder = paths.t2r_iter_dir(decision.params["iter"])
    prompt, response = folder / loop_t2r.PROMPT, folder / loop_t2r.RESPONSE
    return {"prompt": str(prompt), "response": str(response), "brief": loop_t2r.generator_brief(prompt, response)}


def t2r_generate(state, paths, decision):
    """Record one generator request: generator.json next to the response; the tick then dispatches the agent with ``t2r.brief``."""
    request = t2r_request(paths, decision)
    Path(request["response"]).parent.mkdir(parents=True, exist_ok=True)
    record = run_files.write_json(Path(request["response"]).with_name(loop_t2r.GENERATOR),
                                  loop_t2r.generator_record(request, state["t2r"]["requests"] + 1, now_iso()))
    return {"t2r": request, "generator": str(record)}


def ingest_reward(state, paths, decision):
    """Validate the response with Isaac python (the cuda dry run needs it); a failed attempt is moved aside (spec §14)."""
    folder = paths.t2r_iter_dir(decision.params["iter"])
    env = {**_t2r_env(), "TERM": "xterm", "OMNI_KIT_ACCEPT_EULA": "YES"}
    done = subprocess.run([str(ISAAC_PYTHON), T2R_SCRIPT, "ingest", "--iter-dir", str(folder)], cwd=ROOT, env=env, capture_output=True, text=True)
    validation = folder / loop_t2r.VALIDATION
    if not validation.is_file():
        raise RuntimeError(f"t2r ingest wrote no {validation} (exit {done.returncode}):\n{done.stdout[-2000:]}{done.stderr[-2000:]}")
    report = json.loads(validation.read_text(encoding="utf-8"))
    if report["ok"]:
        return {"passed": True}
    attempt = len(list(folder.glob("validation_attempt_*.json"))) + 1
    for name, moved in loop_t2r.failed_attempt_names(attempt).items():
        if (folder / name).exists():
            (folder / name).rename(folder / moved)
    return {"passed": False, "errors": report["errors"], "attempt": attempt}


def run_t2r_smoke(state, paths, decision):
    policy, iteration = state["policy"], decision.params["iter"]
    folder = paths.t2r_iter_dir(iteration)
    argv = ["scripts/iker/t2r_smoke.py", "--mode", "round", "--reward-code", str(folder / loop_t2r.CODE), "--out", str(folder / loop_t2r.SMOKE),
            "--num-envs", str(policy["t2r_smoke_envs"]), "--steps", str(policy["t2r_smoke_steps"]), "--config-index", str(CONFIG_INDEX), "--headless"]
    log = paths.side_log("t2r_smoke", f"iter{iteration:02d}")
    return {"run": launch(side_label(state, "t2r_smoke"), argv, log, f"IKER loop t2r smoke iter {iteration:02d}")}


def launch_t2r(state, paths, decision):
    policy, label, iteration = state["policy"], decision.params["key"], decision.params["iter"]
    folder = paths.t2r_iter_dir(iteration)
    argv = [TRAIN_SCRIPT, "--task", STAGE1_T2R_TASK, "--num_envs", str(policy["t2r_num_envs"]), "--max_iterations", str(policy["t2r_round_epochs"]),
            "--headless", f"env.reward_code_path='{folder / loop_t2r.CODE}'"]
    if policy["minibatch_size"]:
        argv.append(f"agent.params.config.minibatch_size={policy['minibatch_size']}")
    run = launch(label, argv, paths.train_log(label), f"IKER 1단계 t2r iter {iteration:02d} 새 학습, 자동 루프")
    files = sorted(path for path in folder.iterdir() if path.is_file())
    if iteration > 0:
        feedback = paths.t2r_iter_dir(iteration - 1) / loop_t2r.FEEDBACK
        files += [feedback] if feedback.is_file() else []
    return {"run": run, "commit_paths": files, "message": f"iker(loop): 1단계 t2r iter {iteration:02d} 보상·검증·스모크 — {label} 기동"}


def run_harvest(state, paths, decision):
    policy, epoch, iteration = state["policy"], decision.params["epoch"], state["t2r"]["iter"]
    argv = ["scripts/iker/harvest_grasp_bank.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX),
            "--min-entries", str(policy["harvest_min"]), "--out", str(paths.harvest_bank_file), "--t2r-iter", str(iteration),
            "--reward-code", str(paths.t2r_iter_dir(iteration) / loop_t2r.CODE), "--headless"]
    log = paths.side_log("harvest", f"iter{iteration:02d}_ep{epoch}")
    return {"run": launch(side_label(state, "harvest"), argv, log, f"IKER loop harvest t2r iter {iteration:02d} ep {epoch}")}


def commit_bank(state, paths, decision):
    outputs = _round_outputs(state, paths, decision.params["iter"])
    stopped = stop_run(state, paths, "stage1_t2r")
    return {"path": str(paths.harvest_bank_file), "stopped_runs": stopped, **outputs, "commit_paths": [paths.harvest_bank_file],
            "message": f"iker(loop): 학습 파지 뱅크 {decision.params['verified']} 개 — t2r iter {decision.params['iter']:02d} "
                       f"{Path(decision.params['checkpoint']).name}"}


def end_round(state, paths, decision):
    iteration = decision.params["iter"]
    outputs = _round_outputs(state, paths, iteration)
    record = state["runs"].get("stage1_t2r")
    stopped = stop_run(state, paths, "stage1_t2r") if record is not None and lp.label_pids(record["label"]) else []
    folder = paths.t2r_iter_dir(iteration)
    files = sorted(path for path in folder.iterdir() if path.is_file()) if folder.is_dir() else []
    return {"stopped_runs": stopped, **outputs, "commit_paths": files,
            "message": f"iker(loop): 1단계 t2r iter {iteration:02d} 라운드 끝({decision.params['ended']}) — 성공 최대 {decision.params['success_max']:.3f}"}
```
  - `EXECUTORS`: delete `run_calibrate`, `commit_calibration`, `launch_b`; add `"write_t2r_prompt": write_t2r_prompt, "t2r_generate": t2r_generate, "ingest_reward": ingest_reward, "run_t2r_smoke": run_t2r_smoke, "launch_t2r": launch_t2r, "end_round": end_round`.
  - `summary_line`: after `parts = [...]` add `if state["phase"] == "stage1_t2r": parts.insert(1, f"t2r iter {state['t2r']['iter']:02d}")`, and after the latched part add `if probe.success: parts.append(f"success {100.0 * (ls.bin_mean(probe.success, ls.last_epoch(probe.success), state['policy']['bin_epochs']) or 0.0):.2f} %")`.
  - `adopt`: after the `TRAINING_RUNS` check add `if run == "stage1_t2r": raise SystemExit("t2r rounds are launched by the loop, not adopted")`.
  - `cmd_status`: after the `vlm_generate` branch add `if decision.action == "t2r_generate": out["t2r"] = t2r_request(paths, decision)`.
  - `main`: `init.add_argument("--phase", default="stage1_t2r", choices=ls.PHASES)`.

- [ ] **Step 4: Edit `harvest_grasp_bank.py`**
  - Delete the `--g-min` argument and the line `cfg.grasp_reward.g_min = args.g_min` (the parent default g_min 1.0 needs no calibration file).
  - Add arguments: `parser.add_argument("--t2r-iter", type=int, default=-1, help="t2r round of the checkpoint (with --reward-code)")` and `parser.add_argument("--reward-code", default="", help="generated reward the checkpoint trained with; recorded as stage1_reward")`.
  - Replace `stage1_reward=asdict(u._reward_cfg)` with `stage1_reward=stage1_reward_record()` and add above `main`:

```python
def stage1_reward_record() -> dict:
    """The reward the checkpoint trained with: the generated t2r reward (spec 2026-09-15-iker-stage1-t2r §10) or the hand-written config."""
    if not args.reward_code:
        return asdict(IkerShoeGraspPlayEnvCfg().grasp_reward)
    code = Path(args.reward_code).resolve()
    return {"source": "t2r", "iter": args.t2r_iter, "reward_code_path": str(code), "reward_code_sha256": hashlib.sha256(code.read_bytes()).hexdigest()}
```
  - Add one line to the module docstring: the harvest runs the parent stage-1 task — observations, actions, terminations and the success capture equal the t2r env's (§14).

- [ ] **Step 5: Edit `LOOP_PROMPT.md`**
  - In step 2 add after the `vlm_generate` bullet:
    `- `t2r_generate`: `python3 scripts/iker/loop.py act t2r_generate` — 요청 횟수를 기록하고 응답 옆에 `generator.json` 을 쓴다. 그 출력의 `result.t2r.brief` 를 **그대로** 프롬프트로 하여 Agent 를 `subagent_type: iker-t2r-generator`(`.claude/agents/iker-t2r-generator.md`, 도구 Read·Write)로 띄운다. 설명·힌트·이전 보상·학습 결과·이 대화의 맥락을 덧붙이지 않는다(생성기 격리, t2r 스펙 §3.4). 에이전트가 끝나면 `result.t2r.response` 파일이 생겼는지만 확인해 보고에 적고 **틱을 끝낸다**.`
  - In step 3 the Isaac actions sentence stays (`run_*`·`launch_*` cover `run_t2r_smoke`·`launch_t2r`); add `t2r_generate` next to `vlm_generate`; in the file-action list add `write_t2r_prompt`·`ingest_reward`(Isaac python 으로 ~1 분 동기 실행)·`end_round`·`record_harvest_miss`.
  - Step 4: add `stage1_t2r` 틱 보고에 `t2r iter`·성공·래치 bin 을 적는다.
  - Cron line: `(track iker_shoe_c00_t2r)`.
  - Delete `commit_calibration`/`next_calibration` mentions if present.

- [ ] **Step 6: Run the full pure suite** — all PASS (0 failed). Also `PYTHONPATH=source/openarm python3 scripts/iker/loop.py --help` exits 0.

- [ ] **Step 7: Commit** — `git add scripts/iker/loop.py scripts/iker/LOOP_PROMPT.md scripts/iker/harvest_grasp_bank.py source/openarm/openarm/agnostic/modules/iker/tests/test_loop_cli.py`; commit `iker(loop): t2r round executors, tick procedure and harvest metadata`.

---

## After the plan (controller operations, not tasks)

1. Final whole-branch review of Tasks 1–7.
2. Stop `iker_grasp_c00_r8_a` by PID (`loop.stop`), `git mv iker_runs/shoe_place/config_00/loop iker_runs/shoe_place/config_00/archive/2026-09-15_r8_hand_reward_no_hold/loop`, commit.
3. `python3 scripts/iker/loop.py init --track iker_shoe_c00_t2r` (no adopt), commit the state; CronCreate `7,37 * * * *` with the t2r track prompt (delete the r8 job).
4. Run ticks now (write prompt → generator agent → ingest → smoke → launch iter_00), per LOOP_PROMPT.md.
