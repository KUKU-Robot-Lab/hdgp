# VLM Pouring Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `vlm.pouring` extension that performs Qwen3-VL task grounding, deterministic high-level skill decisions, guarded hard routing, existing-policy/checkpoint references, and disk-backed grasp-to-pour handoff without modifying existing OpenArm code.

**Architecture:** `source/vlm/vlm/pouring` owns pure domain contracts and orchestration. It imports existing `openarm` policy contracts only through lazy adapters and resolves checkpoints/configuration from `hdgp/log/rl_games`; a same-machine Qwen process exposes localhost task grounding. The initial high-level policy is deterministic and implements the same interface reserved for a trained HRL policy.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, urllib, PyTorch at adapter boundaries, Isaac Lab/RL-Games at runtime boundaries, FastAPI/Uvicorn, Hugging Face Transformers Qwen3-VL, pytest, pytest-cov, Ruff, Pyright.

## Global Constraints

- Do not modify, move, rename, or stage any existing file outside `source/vlm` and this plan document.
- Do not modify `source/openarm`, `scripts`, `data`, or `log`.
- Runtime import root is `vlm.pouring`; do not create `source/vlm/openarm`.
- Existing `grasp_v1` and `pour_v1` observation, action, reset, task, and checkpoint contracts remain authoritative.
- Policy code, RL-Games YAML, checkpoints, and model weights are referenced in place and never copied.
- Qwen emits `TaskSpecification` only and never emits joint commands, raw actions, contact truth, or precise control poses.
- V1 uses deterministic high-level decisions and hard routing; no raw action blending.
- The default Qwen model ID is exactly `Qwen/Qwen3-VL-4B-Instruct`.
- The default warm-state file is exactly `data/grasp_warm_tesollo.hdf5`.
- Unit tests must not launch Isaac Sim, load Qwen weights, or allocate CUDA memory.
- New pure-Python code must reach at least 80% line coverage.

---

## File Map

```text
source/vlm/
├── config/extension.toml                  # Isaac extension metadata
├── pyproject.toml                         # build backend and Ruff/Pyright settings
├── setup.py                               # installs the `vlm` package
├── requirements-qwen.txt                  # optional Qwen server dependencies
└── vlm/
    ├── __init__.py                        # no eager Isaac/Transformers imports
    └── pouring/
        ├── __init__.py                    # exports stable domain interfaces only
        ├── contracts.py                   # task, decision, skill, and command types
        ├── task_grounding.py              # strict Qwen JSON extraction/validation
        ├── qwen_backend.py                # lazy Transformers backend
        ├── qwen_client.py                 # bounded localhost HTTP client
        ├── qwen_server.py                 # FastAPI app and CLI
        ├── state_provider.py              # SemanticState and provider protocol
        ├── high_level_policy.py           # protocol and deterministic implementation
        ├── skill_manager.py               # per-env guarded hard routing
        ├── skill_registry.py              # skill registration and lookup
        ├── checkpoint_resolver.py         # deterministic in-place log resolution
        ├── transitions.py                 # transition readiness and warm-state bridge
        ├── execution.py                   # skill/backend protocols and dispatch outputs
        ├── safety.py                      # final command veto
        ├── pipeline.py                    # one closed-loop high-level tick
        ├── skills/
        │   ├── __init__.py
        │   ├── approach.py                # rule-based task-space approach target
        │   ├── pre_grasp_bridge.py        # rule-based pregrasp readiness
        │   ├── grasp_lift.py              # existing grasp policy reference adapter
        │   ├── pre_pour_bridge.py         # existing warm-state loader adapter
        │   ├── bimanual_pour.py           # existing pour policy reference adapter
        │   └── recovery.py                # safe-stop recovery
        └── tests/
            ├── test_contracts.py
            ├── test_task_grounding.py
            ├── test_state_provider.py
            ├── test_high_level_policy.py
            ├── test_skill_manager.py
            ├── test_checkpoint_resolver.py
            ├── test_transitions.py
            ├── test_qwen_server.py
            ├── test_qwen_client.py
            └── test_pipeline.py
```

---

### Task 1: Independent Extension and Domain Contracts

**Files:**
- Create: `source/vlm/config/extension.toml`
- Create: `source/vlm/pyproject.toml`
- Create: `source/vlm/setup.py`
- Create: `source/vlm/vlm/__init__.py`
- Create: `source/vlm/vlm/pouring/__init__.py`
- Create: `source/vlm/vlm/pouring/contracts.py`
- Test: `source/vlm/vlm/pouring/tests/test_contracts.py`

**Interfaces:**
- Consumes: no project runtime imports.
- Produces: `SkillId`, `ControlMode`, `TaskSpecification`, `HighLevelDecision`, `SkillCommand`, and `TransitionRecord`.

- [ ] **Step 1: Write package/import and contract tests**

```python
# source/vlm/vlm/pouring/tests/test_contracts.py
import pytest

from vlm.pouring.contracts import (
    ControlMode,
    HighLevelDecision,
    SkillCommand,
    SkillId,
    TaskSpecification,
)


def test_task_specification_accepts_ordered_allowed_plan() -> None:
    spec = TaskSpecification(
        task="pour",
        source_id="right_cup",
        target_id="left_cup",
        nominal_plan=("grasp_lift", "pre_pour_bridge", "bimanual_pour"),
        allowed_skills=("grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"),
    )
    assert spec.nominal_plan[0] == "grasp_lift"


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "pick"},
        {"source_id": ""},
        {"target_id": ""},
        {"nominal_plan": ()},
        {"nominal_plan": ("joint_command",)},
    ],
)
def test_task_specification_rejects_invalid_or_disallowed_values(changes) -> None:
    values = {
        "task": "pour",
        "source_id": "source",
        "target_id": "target",
        "nominal_plan": ("grasp_lift",),
        "allowed_skills": ("grasp_lift", "recovery"),
    }
    values.update(changes)
    with pytest.raises(ValueError):
        TaskSpecification(**values)


def test_high_level_decision_rejects_conflicting_retry_and_recover() -> None:
    with pytest.raises(ValueError, match="retry and recover"):
        HighLevelDecision(SkillId.RECOVERY, retry=True, recover=True, reason="invalid")


def test_skill_command_keeps_control_mode_explicit() -> None:
    command = SkillCommand(ControlMode.TASK_SPACE_POSE, (0.1, 0.2, 0.3), "approach")
    assert command.control_mode is ControlMode.TASK_SPACE_POSE
```

- [ ] **Step 2: Run RED test**

Run:

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_contracts.py -q
```

Expected: collection fails because `vlm.pouring.contracts` does not exist.

- [ ] **Step 3: Commit the validated RED checkpoint**

```bash
git add source/vlm/vlm/pouring/tests/test_contracts.py
git commit -m "test: define vlm pouring domain contracts"
```

- [ ] **Step 4: Add extension metadata and minimal contract implementation**

```toml
# source/vlm/config/extension.toml
[package]
version = "0.1.0"
category = "isaaclab"
title = "HDGP VLM Pouring"
author = "HDGP"
maintainer = "HDGP"
description = "Vision-grounded hierarchical pouring orchestration"
repository = ""
keywords = ["vlm", "hrl", "pouring", "isaaclab"]

[dependencies]
"isaaclab" = {}
"isaaclab_rl" = {}
"openarm" = {}

[[python.module]]
name = "vlm"
```

```python
# source/vlm/setup.py
from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="hdgp-vlm",
    version="0.1.0",
    description="Vision-grounded hierarchical pouring orchestration",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
```

```python
# source/vlm/vlm/pouring/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class SkillId(str, Enum):
    WAIT_FOR_TASK = "wait_for_task"
    APPROACH = "approach"
    PRE_GRASP_BRIDGE = "pre_grasp_bridge"
    GRASP_LIFT = "grasp_lift"
    PRE_POUR_BRIDGE = "pre_pour_bridge"
    BIMANUAL_POUR = "bimanual_pour"
    RECOVERY = "recovery"
    ABORT = "abort"
    DONE = "done"


class ControlMode(str, Enum):
    TASK_SPACE_POSE = "task_space_pose"
    POLICY_ACTION = "policy_action"
    SAFE_STOP = "safe_stop"
    NO_OP = "no_op"


@dataclass(frozen=True)
class TaskSpecification:
    task: str
    source_id: str
    target_id: str
    nominal_plan: tuple[str, ...]
    allowed_skills: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task != "pour":
            raise ValueError("v1 supports only task='pour'")
        if not self.source_id or not self.target_id:
            raise ValueError("source_id and target_id must be non-empty")
        if not self.nominal_plan:
            raise ValueError("nominal_plan must be non-empty")
        valid = {item.value for item in SkillId}
        if not set(self.allowed_skills) <= valid:
            raise ValueError("allowed_skills contains an unknown skill")
        if not set(self.nominal_plan) <= set(self.allowed_skills):
            raise ValueError("nominal_plan must be a subset of allowed_skills")


@dataclass(frozen=True)
class HighLevelDecision:
    skill_id: SkillId
    terminate_current_skill: bool = False
    retry: bool = False
    recover: bool = False
    transition_parameters: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.retry and self.recover:
            raise ValueError("retry and recover cannot both be true")
        object.__setattr__(self, "transition_parameters", MappingProxyType(dict(self.transition_parameters)))


@dataclass(frozen=True)
class SkillCommand:
    control_mode: ControlMode
    values: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class TransitionRecord:
    env_id: int
    previous_skill: SkillId
    requested_skill: SkillId
    accepted_skill: SkillId
    accepted: bool
    reason: str
    step_index: int
```

`source/vlm/pyproject.toml` uses `setuptools.build_meta`, Python `>=3.10`, Ruff line length 120, and Pyright with `typeCheckingMode = "standard"`. Both `__init__.py` files contain only version/export statements and no Isaac or Transformers imports.

- [ ] **Step 5: Run GREEN tests and import check**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_contracts.py -q
PYTHONPATH=source/vlm python -c "import sys, vlm.pouring; assert 'transformers' not in sys.modules; assert 'isaaclab' not in sys.modules"
```

Expected: tests pass and the import check exits 0.

- [ ] **Step 6: Commit GREEN checkpoint**

```bash
git add source/vlm
git commit -m "feat: add standalone vlm pouring extension"
```

---

### Task 2: Strict Task Grounding Parser

**Files:**
- Create: `source/vlm/vlm/pouring/task_grounding.py`
- Test: `source/vlm/vlm/pouring/tests/test_task_grounding.py`

**Interfaces:**
- Consumes: `TaskSpecification` from Task 1.
- Produces: `extract_json_object(text: str) -> dict[str, object]` and `parse_task_specification(text: str) -> TaskSpecification`.

- [ ] **Step 1: Write failing parser tests**

```python
import pytest

from vlm.pouring.task_grounding import parse_task_specification


VALID = """```json
{"task":"pour","source_id":"cup_2","target_id":"cup_5",\
"nominal_plan":["grasp_lift","pre_pour_bridge","bimanual_pour"],\
"allowed_skills":["grasp_lift","pre_pour_bridge","bimanual_pour","recovery"]}
```"""


def test_parse_task_specification_accepts_one_fenced_json_object() -> None:
    result = parse_task_specification(VALID)
    assert result.source_id == "cup_2"


@pytest.mark.parametrize(
    "text",
    [
        "no json",
        "{} {}",
        '{"task":"pour","source_id":"a","target_id":"b",'
        '"nominal_plan":["grasp_lift"],"allowed_skills":["grasp_lift"],"joint_command":[1]}',
    ],
)
def test_parse_task_specification_rejects_missing_multiple_or_extra_control_fields(text: str) -> None:
    with pytest.raises(ValueError):
        parse_task_specification(text)
```

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_task_grounding.py -q
git add source/vlm/vlm/pouring/tests/test_task_grounding.py
git commit -m "test: define strict qwen task grounding schema"
```

Expected: missing-module failure, then RED checkpoint commit.

- [ ] **Step 3: Implement exact-key JSON extraction**

```python
# source/vlm/vlm/pouring/task_grounding.py
from __future__ import annotations

import json

from .contracts import TaskSpecification

_EXPECTED_KEYS = {"task", "source_id", "target_id", "nominal_plan", "allowed_skills"}


def extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Qwen response does not contain a JSON object")
    value, end = decoder.raw_decode(cleaned, start)
    if cleaned[end:].strip():
        raise ValueError("Qwen response must contain exactly one JSON object")
    if not isinstance(value, dict):
        raise ValueError("Qwen response root must be an object")
    return value


def parse_task_specification(text: str) -> TaskSpecification:
    data = extract_json_object(text)
    if set(data) != _EXPECTED_KEYS:
        raise ValueError(f"TaskSpecification keys must be exactly {sorted(_EXPECTED_KEYS)}")
    try:
        return TaskSpecification(
            task=str(data["task"]),
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            nominal_plan=tuple(str(item) for item in data["nominal_plan"]),
            allowed_skills=tuple(str(item) for item in data["allowed_skills"]),
        )
    except TypeError as exc:
        raise ValueError("nominal_plan and allowed_skills must be arrays") from exc
```

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_task_grounding.py -q
git add source/vlm/vlm/pouring/task_grounding.py
git commit -m "feat: validate qwen task grounding output"
```

---

### Task 3: Simulator-Neutral Semantic State

**Files:**
- Create: `source/vlm/vlm/pouring/state_provider.py`
- Test: `source/vlm/vlm/pouring/tests/test_state_provider.py`

**Interfaces:**
- Consumes: `SkillId`.
- Produces: frozen `SemanticState` and `StateProvider` protocol.

- [ ] **Step 1: Write shape, finite-value, and provider tests**

```python
import math
import pytest

from vlm.pouring.contracts import SkillId
from vlm.pouring.state_provider import SemanticState


def valid_state(**changes) -> SemanticState:
    values = dict(
        source_pose=(0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0),
        target_pose=(0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0),
        source_velocity=(0.0,) * 6,
        target_velocity=(0.0,) * 6,
        left_arm_joint_pos=(0.0,) * 7,
        left_arm_joint_vel=(0.0,) * 7,
        right_arm_joint_pos=(0.0,) * 7,
        right_arm_joint_vel=(0.0,) * 7,
        left_hand_joint_pos=(0.0,) * 20,
        left_hand_joint_vel=(0.0,) * 20,
        right_hand_joint_pos=(0.0,) * 20,
        right_hand_joint_vel=(0.0,) * 20,
        left_ee_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    values.update(changes)
    return SemanticState(**values)


def test_semantic_state_accepts_fixed_sim_neutral_contract() -> None:
    assert valid_state().current_skill is SkillId.WAIT_FOR_TASK


def test_semantic_state_rejects_bad_shape_and_non_finite_value() -> None:
    with pytest.raises(ValueError, match="source_pose"):
        valid_state(source_pose=(0.0,) * 6)
    with pytest.raises(ValueError, match="finite"):
        valid_state(right_arm_joint_pos=(math.nan,) + (0.0,) * 6)
```

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_state_provider.py -q
git add source/vlm/vlm/pouring/tests/test_state_provider.py
git commit -m "test: define semantic state contract"
```

- [ ] **Step 3: Implement the frozen state and provider protocol**

Implement `SemanticState` with the vectors exercised above plus these scalar fields and defaults:

```python
source_confidence: float = 1.0
target_confidence: float = 1.0
contact_count: int = 0
tactile_summary: tuple[float, ...] = ()
source_grasped: bool = False
source_lifted: bool = False
source_upright_score: float = 1.0
cup_drop: bool = False
pregrasp_ready: bool = False
warm_state_valid: bool = False
pour_complete: bool = False
workspace_valid: bool = True
joint_limit_margin: float = 1.0
current_skill: SkillId = SkillId.WAIT_FOR_TASK
skill_elapsed_steps: int = 0
current_skill_success: bool = False
current_skill_failed: bool = False
```

Use this validation helper from `__post_init__`:

```python
def _validate_vector(name: str, values: tuple[float, ...], size: int) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain only finite values")
```

Define:

```python
class StateProvider(Protocol):
    def get_states(self) -> tuple[SemanticState, ...]: ...
```

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_state_provider.py -q
git add source/vlm/vlm/pouring/state_provider.py
git commit -m "feat: add simulator-neutral semantic state"
```

---

### Task 4: Deterministic High-Level Policy

**Files:**
- Create: `source/vlm/vlm/pouring/high_level_policy.py`
- Test: `source/vlm/vlm/pouring/tests/test_high_level_policy.py`

**Interfaces:**
- Consumes: `TaskSpecification`, batches of `SemanticState`.
- Produces: `HighLevelPolicy.decide(...)` and `DeterministicHighLevelPolicy`.

- [ ] **Step 1: Write nominal, maintain, completion, and recovery tests**

```python
from dataclasses import replace

from vlm.pouring.contracts import SkillId, TaskSpecification
from vlm.pouring.high_level_policy import DeterministicHighLevelPolicy
from vlm.pouring.tests.test_state_provider import valid_state


TASK = TaskSpecification(
    "pour",
    "source",
    "target",
    ("grasp_lift", "pre_pour_bridge", "bimanual_pour"),
    ("grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"),
)


def test_policy_enters_first_nominal_skill() -> None:
    decision = DeterministicHighLevelPolicy().decide(TASK, (valid_state(),))[0]
    assert decision.skill_id is SkillId.GRASP_LIFT


def test_policy_keeps_current_skill_until_success() -> None:
    state = valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=3)
    assert DeterministicHighLevelPolicy().decide(TASK, (state,))[0].skill_id is SkillId.GRASP_LIFT


def test_policy_advances_and_finishes_nominal_plan() -> None:
    policy = DeterministicHighLevelPolicy()
    grasp_done = valid_state(current_skill=SkillId.GRASP_LIFT, current_skill_success=True)
    pour_done = valid_state(current_skill=SkillId.BIMANUAL_POUR, current_skill_success=True)
    assert policy.decide(TASK, (grasp_done,))[0].skill_id is SkillId.PRE_POUR_BRIDGE
    assert policy.decide(TASK, (pour_done,))[0].skill_id is SkillId.DONE


def test_policy_recovers_on_skill_failure_and_aborts_on_safety_failure() -> None:
    policy = DeterministicHighLevelPolicy()
    failed = valid_state(current_skill=SkillId.GRASP_LIFT, current_skill_failed=True)
    dropped = replace(failed, cup_drop=True)
    assert policy.decide(TASK, (failed,))[0].skill_id is SkillId.RECOVERY
    assert policy.decide(TASK, (dropped,))[0].skill_id is SkillId.ABORT
```

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_high_level_policy.py -q
git add source/vlm/vlm/pouring/tests/test_high_level_policy.py
git commit -m "test: define deterministic high-level decisions"
```

- [ ] **Step 3: Implement the replaceable policy interface**

```python
class HighLevelPolicy(Protocol):
    def decide(
        self,
        task: TaskSpecification,
        states: tuple[SemanticState, ...],
    ) -> tuple[HighLevelDecision, ...]: ...


class DeterministicHighLevelPolicy:
    def decide(self, task, states):
        return tuple(self._decide_one(task, state) for state in states)

    def _decide_one(self, task, state):
        if state.cup_drop or not state.workspace_valid or state.joint_limit_margin <= 0.0:
            return HighLevelDecision(SkillId.ABORT, terminate_current_skill=True, reason="safety_violation")
        if state.current_skill_failed:
            recovery_allowed = SkillId.RECOVERY.value in task.allowed_skills
            next_skill = SkillId.RECOVERY if recovery_allowed else SkillId.ABORT
            return HighLevelDecision(next_skill, terminate_current_skill=True, recover=True, reason="skill_failed")
        plan = tuple(SkillId(item) for item in task.nominal_plan)
        if state.current_skill is SkillId.WAIT_FOR_TASK:
            return HighLevelDecision(plan[0], reason="task_started")
        if not state.current_skill_success:
            return HighLevelDecision(state.current_skill, reason="continue_current_skill")
        if state.current_skill not in plan:
            return HighLevelDecision(SkillId.ABORT, terminate_current_skill=True, reason="skill_not_in_plan")
        index = plan.index(state.current_skill)
        if index + 1 == len(plan):
            return HighLevelDecision(SkillId.DONE, terminate_current_skill=True, reason="plan_complete")
        return HighLevelDecision(plan[index + 1], terminate_current_skill=True, reason="skill_complete")
```

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_high_level_policy.py -q
git add source/vlm/vlm/pouring/high_level_policy.py
git commit -m "feat: add replaceable deterministic high-level policy"
```

---

### Task 5: Guarded Per-Environment Hard Routing

**Files:**
- Create: `source/vlm/vlm/pouring/execution.py`
- Create: `source/vlm/vlm/pouring/safety.py`
- Create: `source/vlm/vlm/pouring/skill_registry.py`
- Create: `source/vlm/vlm/pouring/skill_manager.py`
- Test: `source/vlm/vlm/pouring/tests/test_skill_manager.py`

**Interfaces:**
- Consumes: decisions and semantic states.
- Produces: `Skill` protocol, `SkillRegistry`, `SafetySupervisor`, `SkillManager.step`, commands, and transition records.

- [ ] **Step 1: Write fake-skill routing tests**

The tests create two fake skills that return distinct command lengths and verify:

```python
commands, records = manager.step(
    states=(grasp_state, pour_state),
    decisions=(
        HighLevelDecision(SkillId.GRASP_LIFT, reason="grasp"),
        HighLevelDecision(SkillId.BIMANUAL_POUR, reason="pour"),
    ),
)
assert commands[0].values == (1.0,) * 11
assert commands[1].values == (2.0,) * 12
assert grasp_skill.calls == [(0,)]
assert pour_skill.calls == [(1,)]
assert all(record.accepted for record in records)
```

Use these additional assertions in the same test file:

```python
def test_manager_rejects_disallowed_and_early_transitions() -> None:
    manager, grasp_skill, pour_skill = make_manager(minimum_steps={SkillId.BIMANUAL_POUR: 5})
    task = make_task(allowed=("grasp_lift", "recovery"))
    state = valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=1)
    commands, records = manager.step(
        task,
        (state,),
        (HighLevelDecision(SkillId.BIMANUAL_POUR, reason="early"),),
    )
    assert records[0].accepted is False
    assert records[0].accepted_skill is SkillId.GRASP_LIFT
    assert pour_skill.calls == []


def test_manager_resets_only_switched_environment_and_never_blends() -> None:
    manager, grasp_skill, pour_skill = make_manager()
    task = make_task()
    states = (
        valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=8),
        valid_state(current_skill=SkillId.BIMANUAL_POUR, skill_elapsed_steps=8),
    )
    commands, _ = manager.step(
        task,
        states,
        (
            HighLevelDecision(SkillId.BIMANUAL_POUR, terminate_current_skill=True, reason="switch"),
            HighLevelDecision(SkillId.BIMANUAL_POUR, reason="continue"),
        ),
    )
    assert pour_skill.resets == [(0,)]
    assert tuple(len(command.values) for command in commands) == (12, 12)


def test_safety_supervisor_replaces_non_finite_command() -> None:
    unsafe = SkillCommand(ControlMode.POLICY_ACTION, (float("nan"),), "bad_skill")
    safe = SafetySupervisor().validate(unsafe)
    assert safe.control_mode is ControlMode.SAFE_STOP
    assert safe.values == ()
```

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_skill_manager.py -q
git add source/vlm/vlm/pouring/tests/test_skill_manager.py
git commit -m "test: define guarded per-environment hard routing"
```

- [ ] **Step 3: Implement execution and registry protocols**

```python
# execution.py
class Skill(Protocol):
    skill_id: SkillId
    def reset(self, env_ids: tuple[int, ...]) -> None: ...
    def infer(self, env_ids: tuple[int, ...], states: tuple[SemanticState, ...]) -> tuple[SkillCommand, ...]: ...


# skill_registry.py
class SkillRegistry:
    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills = {skill.skill_id: skill for skill in skills}

    def get(self, skill_id: SkillId) -> Skill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"skill is not registered: {skill_id.value}") from exc
```

`SkillManager` stores `current_skill` and elapsed steps as Python lists indexed by environment. For each tick it:

1. validates the decision against the task allowlist and minimum-duration map;
2. records accepted/rejected transitions;
3. resets only switched environment IDs;
4. groups environment IDs by accepted skill;
5. invokes one skill per group;
6. restores results to environment order;
7. applies `SafetySupervisor.validate(command)` independently.

The `SafetySupervisor` returns `SkillCommand(ControlMode.SAFE_STOP, (), "safety_supervisor")` when any command value is non-finite.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_skill_manager.py -q
git add source/vlm/vlm/pouring/{execution.py,safety.py,skill_registry.py,skill_manager.py}
git commit -m "feat: add guarded hard-routed skill manager"
```

---

### Task 6: Existing Checkpoint and Warm-State References

**Files:**
- Create: `source/vlm/vlm/pouring/checkpoint_resolver.py`
- Create: `source/vlm/vlm/pouring/transitions.py`
- Test: `source/vlm/vlm/pouring/tests/test_checkpoint_resolver.py`
- Test: `source/vlm/vlm/pouring/tests/test_transitions.py`

**Interfaces:**
- Consumes: existing `hdgp/log/rl_games` and `data/grasp_warm_tesollo.hdf5` paths.
- Produces: `PolicyArtifacts`, `CheckpointResolver.resolve`, and `PrePourWarmStateBridge.load`.

- [ ] **Step 1: Write resolver RED tests using a temporary log tree**

```python
def test_resolver_returns_checkpoint_and_neighbor_params(tmp_path: Path) -> None:
    run = tmp_path / "log/rl_games/open-tesol/right/grasp-v1/lstm_test1"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir()
    (run / "nn/open-tesol_r_grasp_v1-lstm.pth").touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    result = CheckpointResolver(tmp_path).resolve(
        task_id="open-tesol_r_grasp_v1-lstm",
        run_dir="lstm_test1",
    )
    assert result.checkpoint.name == "open-tesol_r_grasp_v1-lstm.pth"
    assert result.agent_yaml == run / "params/agent.yaml"


def test_resolver_rejects_missing_or_ambiguous_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CheckpointResolver(tmp_path).resolve("open-tesol_r_grasp_v1-lstm", "lstm_test1")
```

Add an ambiguity case with `run_dir="lstm_test*"` matching two folders; it must raise `ValueError` listing both paths.

- [ ] **Step 2: Write real warm-state compatibility RED test**

```python
def test_pre_pour_bridge_loads_existing_grasp_warm_state() -> None:
    root = Path(__file__).resolve().parents[5]
    bridge = PrePourWarmStateBridge(root)
    result = bridge.load(
        root / "data/grasp_warm_tesollo.hdf5",
        expected_object_spawn_z=0.297,
    )
    assert result.num_states > 0
    assert result.arm_joint_pos.shape[1] == 7
    assert result.hand_joint_pos.shape[1] == 20
```

The test imports `PourWarmStateBank` from its existing source file without copying it. If the actual logged `object_spawn_z` differs, use the numeric value read from the HDF5 metadata in the assertion fixture rather than weakening the production check.

- [ ] **Step 3: Run RED and commit**

```bash
PYTHONPATH=source/vlm:source/openarm python -m pytest \
  source/vlm/vlm/pouring/tests/test_checkpoint_resolver.py \
  source/vlm/vlm/pouring/tests/test_transitions.py -q
git add source/vlm/vlm/pouring/tests/test_checkpoint_resolver.py source/vlm/vlm/pouring/tests/test_transitions.py
git commit -m "test: define existing policy artifact references"
```

- [ ] **Step 4: Implement deterministic checkpoint resolution**

```python
@dataclass(frozen=True)
class PolicyArtifacts:
    task_id: str
    run_dir: Path
    checkpoint: Path
    agent_yaml: Path
    env_yaml: Path


class CheckpointResolver:
    def __init__(self, hdgp_root: Path) -> None:
        self.hdgp_root = hdgp_root.resolve()

    def resolve(self, task_id: str, run_dir: str, checkpoint: Path | None = None) -> PolicyArtifacts:
        side, task_folder = _task_log_components(task_id)
        log_root = self.hdgp_root / "log/rl_games" / side / task_folder
        runs = tuple(sorted(path for path in log_root.glob(run_dir) if path.is_dir()))
        if len(runs) != 1:
            raise ValueError(f"run selector must resolve exactly once: {run_dir}: {runs}") if runs else FileNotFoundError(log_root / run_dir)
        selected_run = runs[0]
        selected_checkpoint = checkpoint or selected_run / "nn" / f"{task_id}.pth"
        required = (selected_checkpoint, selected_run / "params/agent.yaml", selected_run / "params/env.yaml")
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(f"missing policy artifacts: {missing}")
        return PolicyArtifacts(task_id, selected_run, *required)
```

`_task_log_components` handles the approved task IDs explicitly:

```python
_TASK_LOGS = {
    "open-tesol_r_grasp_v1-lstm": ("open-tesol/right", "grasp-v1"),
    "open-tesol_r_pour_v1-lstm": ("open-tesol/right", "pour-v1"),
}
```

Unknown task IDs raise `KeyError`; v1 does not add speculative path parsing.

- [ ] **Step 5: Implement the delegating warm-state bridge**

`PrePourWarmStateBridge.load` lazily imports the existing `PourWarmStateBank`, calls `from_hdf5_paths`, and returns that bank unchanged. The constructor accepts an injected loader for unit tests. It does not restate HDF5 dataset names or copy tensors.

- [ ] **Step 6: Run GREEN, real-data check, and commit**

```bash
PYTHONPATH=source/vlm:source/openarm python -m pytest \
  source/vlm/vlm/pouring/tests/test_checkpoint_resolver.py \
  source/vlm/vlm/pouring/tests/test_transitions.py -q
git add source/vlm/vlm/pouring/checkpoint_resolver.py source/vlm/vlm/pouring/transitions.py
git commit -m "feat: reference existing policy and warm-state artifacts"
```

---

### Task 7: Concrete Skill Adapters

**Files:**
- Create: `source/vlm/vlm/pouring/skills/__init__.py`
- Create: `source/vlm/vlm/pouring/skills/approach.py`
- Create: `source/vlm/vlm/pouring/skills/pre_grasp_bridge.py`
- Create: `source/vlm/vlm/pouring/skills/grasp_lift.py`
- Create: `source/vlm/vlm/pouring/skills/pre_pour_bridge.py`
- Create: `source/vlm/vlm/pouring/skills/bimanual_pour.py`
- Create: `source/vlm/vlm/pouring/skills/recovery.py`
- Test: extend `source/vlm/vlm/pouring/tests/test_skill_manager.py`

**Interfaces:**
- Consumes: `PolicyArtifacts`, `PrePourWarmStateBridge`, and an injected `PolicyInferenceBackend`.
- Produces: six registered v1 skills without copying existing policy code.

- [ ] **Step 1: Add adapter behavior tests**

Test exact behavior:

- `ApproachSkill` returns a `TASK_SPACE_POSE` target equal to source position plus configured `(x, y, z)` offset and a configured `wxyz` orientation.
- `PreGraspBridgeSkill` returns `NO_OP` only when `pregrasp_ready` is true, otherwise a bounded task-space correction.
- `GraspLiftSkill` passes the existing 106D actor observation from its observation adapter to the injected backend and requires an 11D action.
- `BimanualPourSkill` passes the existing 55D actor observation to the injected backend and requires a 12D action.
- either policy adapter raises `ValueError` on a wrong observation or action dimension.
- `PrePourBridgeSkill` emits `NO_OP` only after the existing warm-state bank validates.
- `RecoverySkill` always emits `SAFE_STOP`.

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm:source/openarm python -m pytest source/vlm/vlm/pouring/tests/test_skill_manager.py -q
git add source/vlm/vlm/pouring/tests/test_skill_manager.py
git commit -m "test: define existing-policy skill adapters"
```

- [ ] **Step 3: Implement injected inference adapters**

Define in `execution.py`:

```python
class PolicyInferenceBackend(Protocol):
    def infer(self, observations: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]: ...
    def reset(self, env_ids: tuple[int, ...]) -> None: ...
```

`GraspLiftSkill` and `BimanualPourSkill` receive a `PolicyArtifacts`, observation-builder callable, backend, expected observation dimension, and expected action dimension. They validate dimensions around the backend call. Actual RL-Games player construction remains at the Isaac application boundary because it requires the launched simulator and registered environment; this source package stores no alternate policy implementation.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm:source/openarm python -m pytest source/vlm/vlm/pouring/tests/test_skill_manager.py -q
git add source/vlm/vlm/pouring/skills source/vlm/vlm/pouring/execution.py
git commit -m "feat: add reference-based pouring skill adapters"
```

---

### Task 8: Same-Machine Qwen Server and Client

**Files:**
- Create: `source/vlm/requirements-qwen.txt`
- Create: `source/vlm/vlm/pouring/qwen_backend.py`
- Create: `source/vlm/vlm/pouring/qwen_server.py`
- Create: `source/vlm/vlm/pouring/qwen_client.py`
- Test: `source/vlm/vlm/pouring/tests/test_qwen_server.py`
- Test: `source/vlm/vlm/pouring/tests/test_qwen_client.py`

**Interfaces:**
- Consumes: user command and base64 RGB image.
- Produces: `POST /v1/task-grounding`, `GET /health`, and `QwenTaskClient.ground`.

- [ ] **Step 1: Write server tests with a fake generation backend**

```python
class FakeBackend:
    model_id = "fake/qwen"
    loaded = True

    def generate(self, command: str, image: bytes) -> str:
        return json.dumps({
            "task": "pour",
            "source_id": "right_cup",
            "target_id": "left_cup",
            "nominal_plan": ["grasp_lift", "pre_pour_bridge", "bimanual_pour"],
            "allowed_skills": ["grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"],
        })


def test_task_grounding_endpoint_returns_validated_specification() -> None:
    client = TestClient(create_app(FakeBackend()))
    response = client.post("/v1/task-grounding", json={
        "command": "Pour from the right cup into the left cup",
        "image_base64": base64.b64encode(b"image").decode("ascii"),
    })
    assert response.status_code == 200
    assert response.json()["source_id"] == "right_cup"
```

Add the concrete failure and health tests:

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"command": "", "image_base64": base64.b64encode(b"image").decode("ascii")},
        {"command": "pour", "image_base64": "%%%not-base64%%%"},
    ],
)
def test_task_grounding_endpoint_rejects_invalid_request(payload) -> None:
    response = TestClient(create_app(FakeBackend())).post("/v1/task-grounding", json=payload)
    assert response.status_code == 422


def test_task_grounding_endpoint_rejects_malformed_or_control_output() -> None:
    for output in ("not-json", '{"task":"pour","joint_command":[1]}'):
        backend = FakeBackend()
        backend.generate = lambda command, image, value=output: value
        response = TestClient(create_app(backend)).post(
            "/v1/task-grounding",
            json={
                "command": "pour",
                "image_base64": base64.b64encode(b"image").decode("ascii"),
            },
        )
        assert response.status_code == 502


def test_health_reports_backend_model_and_load_state() -> None:
    response = TestClient(create_app(FakeBackend())).get("/health")
    assert response.json() == {"model_id": "fake/qwen", "loaded": True}
```

- [ ] **Step 2: Write client timeout and HTTP error tests**

Inject a transport callable into `QwenTaskClient` and use these tests:

```python
def test_client_turns_timeout_into_explicit_unavailable_error() -> None:
    def timeout_transport(request, timeout):
        raise TimeoutError("timed out")

    client = QwenTaskClient(transport=timeout_transport)
    with pytest.raises(TaskGroundingUnavailable, match="timed out"):
        client.ground("pour", b"image")


def test_client_rejects_non_200_without_guessing_task() -> None:
    def failed_transport(request, timeout):
        return 503, b'{"detail":"model unavailable"}'

    client = QwenTaskClient(transport=failed_transport)
    with pytest.raises(TaskGroundingUnavailable, match="503"):
        client.ground("pour", b"image")
```

- [ ] **Step 3: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest \
  source/vlm/vlm/pouring/tests/test_qwen_server.py \
  source/vlm/vlm/pouring/tests/test_qwen_client.py -q
git add source/vlm/vlm/pouring/tests/test_qwen_server.py source/vlm/vlm/pouring/tests/test_qwen_client.py
git commit -m "test: define localhost qwen task grounding"
```

- [ ] **Step 4: Add bounded optional dependencies**

```text
# source/vlm/requirements-qwen.txt
fastapi>=0.115,<1
uvicorn>=0.30,<1
transformers>=4.57,<6
accelerate>=1,<2
pillow>=10,<13
```

- [ ] **Step 5: Implement lazy Qwen backend**

`QwenBackend.__init__` stores only configuration. `load()` imports Transformers and Pillow, then constructs `AutoProcessor` and `AutoModelForImageTextToText` with `device_map="auto"` and `torch_dtype="auto"`. `generate()`:

1. decodes the RGB image with Pillow;
2. builds a system prompt containing the exact five allowed keys and skill allowlist;
3. calls `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")`;
4. calls `model.generate(..., do_sample=False, max_new_tokens=256)`;
5. trims prompt tokens and decodes only generated tokens;
6. returns raw text to `parse_task_specification` at the server boundary.

No model is loaded during module import.

- [ ] **Step 6: Implement FastAPI app and stdlib client**

`create_app(backend)` owns two Pydantic request/response models. The client sends JSON with `urllib.request.urlopen(..., timeout=self.timeout_seconds)`. Default URL is `http://127.0.0.1:8100/v1/task-grounding`, and default timeout is 30 seconds. `python -m vlm.pouring.qwen_server --host 127.0.0.1 --port 8100 --model Qwen/Qwen3-VL-4B-Instruct` starts Uvicorn.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=source/vlm python -m pytest \
  source/vlm/vlm/pouring/tests/test_qwen_server.py \
  source/vlm/vlm/pouring/tests/test_qwen_client.py -q
git add source/vlm/requirements-qwen.txt source/vlm/vlm/pouring/qwen_*.py
git commit -m "feat: add same-machine qwen task grounding service"
```

---

### Task 9: Closed-Loop Tick and Full Verification

**Files:**
- Create: `source/vlm/vlm/pouring/pipeline.py`
- Test: `source/vlm/vlm/pouring/tests/test_pipeline.py`
- Modify: `source/vlm/vlm/pouring/__init__.py`

**Interfaces:**
- Consumes: `TaskSpecification`, `StateProvider`, `HighLevelPolicy`, and `SkillManager`.
- Produces: `PouringPipeline.tick() -> TickResult`.

- [ ] **Step 1: Write end-to-end fake pipeline RED test**

```python
def test_pipeline_routes_grounded_task_without_loading_qwen_or_isaac() -> None:
    pipeline = PouringPipeline(
        task=TASK,
        state_provider=FakeStateProvider((valid_state(),)),
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=manager_with_fake_grasp_skill(),
    )
    result = pipeline.tick()
    assert result.decisions[0].skill_id is SkillId.GRASP_LIFT
    assert result.commands[0].source == "fake_grasp"
    assert result.transitions[0].accepted
```

Add a second test with two environments in different skills and a third test proving a safety failure yields `ABORT`/`SAFE_STOP`.

- [ ] **Step 2: Run RED and commit**

```bash
PYTHONPATH=source/vlm python -m pytest source/vlm/vlm/pouring/tests/test_pipeline.py -q
git add source/vlm/vlm/pouring/tests/test_pipeline.py
git commit -m "test: define closed-loop pouring tick"
```

- [ ] **Step 3: Implement the orchestration tick**

```python
@dataclass(frozen=True)
class TickResult:
    states: tuple[SemanticState, ...]
    decisions: tuple[HighLevelDecision, ...]
    commands: tuple[SkillCommand, ...]
    transitions: tuple[TransitionRecord, ...]


class PouringPipeline:
    def __init__(self, task, state_provider, high_level_policy, skill_manager):
        self.task = task
        self.state_provider = state_provider
        self.high_level_policy = high_level_policy
        self.skill_manager = skill_manager

    def tick(self) -> TickResult:
        states = self.state_provider.get_states()
        decisions = self.high_level_policy.decide(self.task, states)
        commands, transitions = self.skill_manager.step(self.task, states, decisions)
        return TickResult(states, decisions, commands, transitions)
```

- [ ] **Step 4: Run full unit/integration suite and coverage**

```bash
PYTHONPATH=source/vlm:source/openarm python -m pytest source/vlm/vlm/pouring/tests \
  --cov=vlm.pouring --cov-report=term-missing --cov-fail-under=80 -q
```

Expected: all tests pass, no skips, and coverage is at least 80%.

- [ ] **Step 5: Run existing contract regressions**

```bash
python -m pytest \
  source/openarm/openarm/tesollo/right/grasp_v1/tests/test_warm_state_cache.py \
  source/openarm/openarm/tesollo/right/pour_v1/tests/test_actor_observation_layout.py -q
```

Expected: existing tests pass without modified OpenArm files.

- [ ] **Step 6: Run static verification**

```bash
ruff check source/vlm
pyright source/vlm/vlm/pouring
git diff --check
git diff --name-only f110ea9..HEAD -- source/openarm scripts data log
```

Expected: Ruff, Pyright, and whitespace checks pass; the final command prints nothing.

- [ ] **Step 7: Document and run the optional real-Qwen smoke check**

Start the server only when the model can be downloaded or is already cached and sufficient GPU memory is available:

```bash
PYTHONPATH=source/vlm python -m vlm.pouring.qwen_server \
  --host 127.0.0.1 --port 8100 \
  --model Qwen/Qwen3-VL-4B-Instruct
```

In another shell, call `/health`, then send one real RGB frame and command through `QwenTaskClient`. Success is a validated five-field `TaskSpecification`; GPU/model unavailability is reported as an unverified optional smoke condition and does not weaken the unit/integration gates.

- [ ] **Step 8: Commit GREEN integration checkpoint**

```bash
git add source/vlm
git commit -m "feat: integrate vision-grounded hierarchical pouring pipeline"
git status --short
```

Expected: only the user's pre-existing unrelated working-tree changes remain.

---

## Plan Self-Review Results

- Spec coverage: additive extension, task grounding, semantic state, deterministic HRL interface, per-env hard routing, two bridges, policy/log references, Qwen service, safety, and closed-loop tick are mapped to Tasks 1–9.
- Existing-code safety: every production edit is under `source/vlm`; OpenArm, scripts, datasets, logs, and user changes are read-only inputs.
- Type consistency: `SkillId`, `TaskSpecification`, `SemanticState`, `HighLevelDecision`, `SkillCommand`, and `TransitionRecord` are defined once and reused by every later task.
- Runtime boundary: unit tests inject Qwen and policy inference backends; Transformers and Isaac imports stay lazy.
- Deferred training: no task creates a trainable high-level policy, learned approach, learned recovery, FoundationPose, ROS 2 node, or raw action blending.
