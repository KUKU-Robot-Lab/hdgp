# Vision-Grounded Hierarchical Pouring Design

## 1. Purpose

Build an additive, closed-loop hierarchical pouring architecture around the existing OpenArm + TESOLLO tasks. The first implementation runs Qwen3-VL task grounding, deterministic high-level decisions, hard skill routing, rule-based transitions, and adapters for the existing grasp and pour policies. Training a high-level RL policy is explicitly deferred.

The truthful initial capability remains:

```text
pregrasp-ready or rule-based approach-ready state
→ grasp_v1
→ verified pre-pour warm state
→ pour_v1
```

The code structure must support the later target without claiming that untrained skills already work:

```text
RGB-D + command
→ approach
→ grasp-and-lift
→ bimanual pouring
→ recovery or completion
```

## 2. Non-Negotiable Constraints

- Do not move, rename, reorganize, or rewrite existing `hdgp` folders.
- Do not change the observation, action, reset, checkpoint, or task-registration contracts of `grasp_v1` and `pour_v1`.
- Add new packages and adapters around the existing tasks.
- Keep Qwen out of the low-level control loop.
- Qwen may emit task metadata only; it must never emit joint commands, policy actions, contact truth, or precise control poses.
- Use hard skill routing in v1. Do not blend raw actions from heterogeneous policies.
- Use the existing disk-backed `data/grasp_warm_tesollo.hdf5` contract before adding an in-memory transition.
- Keep Isaac/OpenArm imports lazy at integration boundaries so domain tests run without launching Isaac Sim.
- Keep model weights and generated artifacts out of Git.
- Actual policy training, ROS 2 deployment, FoundationPose integration, and learned recovery are outside this implementation.

## 3. Additive Repository Layout

Existing folders remain unchanged:

```text
source/openarm/openarm/tesollo/right/grasp_v1/
source/openarm/openarm/tesollo/right/pour_v1/
```

Add the hierarchical package beside them:

```text
source/openarm/openarm/tesollo/right/hierarchical_pouring/
├── __init__.py
├── task_grounding/
│   ├── __init__.py
│   ├── task_specification.py
│   ├── task_schema.py
│   └── qwen_vlm_client.py
├── state_provider/
│   ├── __init__.py
│   ├── semantic_state.py
│   ├── provider.py
│   └── sim_state_provider.py
├── high_level_rl/
│   ├── __init__.py
│   ├── actions.py
│   ├── policy.py
│   └── deterministic_policy.py
├── skill_manager/
│   ├── __init__.py
│   ├── skill_manager.py
│   ├── skill_registry.py
│   ├── transition_guard.py
│   └── readiness_estimator.py
├── skills/
│   ├── __init__.py
│   ├── base_skill.py
│   ├── approach/
│   ├── pre_grasp_bridge/
│   ├── grasp_lift/
│   ├── pre_pour_bridge/
│   ├── bimanual_pour/
│   └── recovery/
├── execution/
│   ├── __init__.py
│   ├── low_level_executor.py
│   ├── fabric_ik_adapter.py
│   ├── joint_pd_adapter.py
│   └── safety_supervisor.py
└── tests/
```

Add Qwen as a separate same-machine process:

```text
vlm_server/
├── __init__.py
├── app.py
├── config.py
├── qwen_backend.py
├── schemas.py
├── requirements.txt
└── tests/
```

ROS 2 packages remain future additions under `robot_ws/`; they are not placed inside the Isaac Lab extension.

## 4. Closed-Loop Architecture

```text
User command + RGB/RGB-D frame
              │
              ▼
Qwen3-VL server (episode start or replanning only)
              │ validated JSON
              ▼
TaskSpecification
              │
              ├───────────────┐
              ▼               ▼
SemanticState          HighLevelPolicy
                              │
                              ▼
                     HighLevelDecision
                              │
                              ▼
                       SkillManager
                              │ hard route
                              ▼
                      LowLevelExecutor
                              │
                              ▼
                  Fabric IK / Joint PD / Robot
                              │
                              └──── feedback ────→ StateProvider
```

Control rates are separated by responsibility:

- Qwen: episode start, command change, ambiguity, object loss, or explicit replanning.
- High-level decision: approximately 2–10 Hz.
- Low-level skill: approximately 50–200 Hz.
- Robot controller: the hardware or simulator control period.

## 5. Stable Domain Contracts

### 5.1 TaskSpecification

`TaskSpecification` is immutable after validation and contains:

```python
task: str
source_id: str
target_id: str
nominal_plan: tuple[str, ...]
allowed_skills: tuple[str, ...]
```

V1 accepts only `task == "pour"`. Skill names must come from the registry allowlist. The nominal plan must be a non-empty ordered subset of allowed skills.

### 5.2 SemanticState

`SemanticState` is the same domain object for simulation and real deployment. It contains no Isaac Lab objects and includes:

```text
source/target pose and velocity
source/target confidence
left/right arm joint position and velocity
left/right hand joint position and velocity
left/right end-effector pose
contact and tactile summaries
grasped/lifted/upright/drop indicators
joint-limit margin and workspace validity
current skill and elapsed steps
previous skill result
```

All fixed-size vectors are validated for shape and finite values. Quaternions are documented explicitly at each adapter boundary; the domain pose convention is `xyz + quat_wxyz`.

### 5.3 HighLevelDecision

```python
skill_id: SkillId
terminate_current_skill: bool
retry: bool
recover: bool
transition_parameters: Mapping[str, float]
reason: str
```

Contradictory flags such as `retry=True` and `recover=True` are invalid. Transition parameters use an allowlist per transition and cannot contain joint or raw action targets.

### 5.4 HighLevelPolicy

The policy interface consumes one `TaskSpecification` and a batch of `SemanticState` values and produces one `HighLevelDecision` per environment.

V1 provides `DeterministicHighLevelPolicy`. A later `RLHighLevelPolicy` implements the same interface without changing the manager or low-level skills.

## 6. Skill Model

The initial registry contains:

```text
APPROACH
PRE_GRASP_BRIDGE
GRASP_LIFT
PRE_POUR_BRIDGE
BIMANUAL_POUR
RECOVERY
ABORT
DONE
```

Each skill implements a small common contract:

```text
reset(env_ids)
build_observation(semantic_state_batch)
infer(observation_batch)
status(semantic_state_batch)
```

The registry maps generic skills to current implementations:

- `GRASP_LIFT` → adapter around existing `tesollo/right/grasp_v1` checkpoint and observation contract.
- `BIMANUAL_POUR` → adapter around existing `tesollo/right/pour_v1` checkpoint and observation contract.
- `APPROACH` → rule-based task-space target generation through Fabric IK; no trained approach policy is claimed.
- `PRE_GRASP_BRIDGE` → rule-based readiness alignment for the start distribution expected by `grasp_v1`.
- `PRE_POUR_BRIDGE` → validation and selection through the existing grasp warm-state HDF5 schema and pour warm-state loader.
- `RECOVERY` → safe stop/abort behavior only in v1.

## 7. High-Level State Flow

The deterministic policy follows this nominal sequence:

```text
WAIT_FOR_TASK
→ APPROACH
→ PRE_GRASP_BRIDGE
→ GRASP_LIFT
→ PRE_POUR_BRIDGE
→ BIMANUAL_POUR
→ DONE
```

For deployments that begin pregrasp-ready, `APPROACH` and `PRE_GRASP_BRIDGE` may be omitted by the validated nominal plan.

Failures transition to `RECOVERY` or `ABORT`. V1 recovery performs a safe stop and reports the reason; it does not invent an untrained recovery motion.

The `TransitionGuard` enforces minimum execution duration, readiness predicates, task allowlists, workspace safety, confidence thresholds, grasp/lift checks, and timeout rules. The `SkillManager` may reject an unsafe high-level decision but may not silently replace it without logging the reason.

## 8. Parallel Environment Behavior

All high-level state is indexed by environment:

```text
current_skill.shape == [num_envs]
skill_elapsed_steps.shape == [num_envs]
previous_skill_success.shape == [num_envs]
```

The manager groups environment indices by selected skill, calls each skill once per group, and writes the outputs back to their original batch positions. Policy reset is applied only to environments that switch skills.

Because existing low-level skills use different action meanings and dimensions, their outputs are not averaged. The executor dispatches only the selected skill output and converts it through that skill's own adapter.

## 9. Pre-Pour Warm-State Contract

The existing `grasp_v1/warm_state_cache.py` and `pour_v1/warm_state_bank.py` remain authoritative. The new bridge delegates HDF5 compatibility checks to `PourWarmStateBank` rather than duplicating its schema.

The bridge validates at least:

- arm joint shape `(N, 7)` and ordering;
- hand joint shape `(N, 20)` and ordering;
- palm pose and quaternion convention;
- source cup pose relative to the environment origin;
- finite values;
- `object_spawn_z` compatibility;
- palm workspace compatibility;
- non-empty state bank;
- sufficient contact metadata.

The default source is `data/grasp_warm_tesollo.hdf5`. An in-memory bridge may be added only after the disk path passes integration tests.

## 10. Qwen Server and Client

The same machine runs Qwen in a process separate from Isaac Sim. The default model is configurable and initially set to `Qwen/Qwen3-VL-4B-Instruct`.

The server exposes:

```text
GET  /health
POST /v1/task-grounding
```

The request contains a command and one RGB image. RGB-D depth may be supplied later as additional grounding context but is not converted into a control pose by Qwen.

The server prompt requires one JSON object matching `TaskSpecification`. Generation is deterministic by default. The server extracts exactly one JSON object, validates it, and returns a structured error for malformed or disallowed output.

The client uses bounded connect/read timeouts and never retries inside the control loop. On timeout, malformed output, unavailable model, or schema failure, the current safe state is retained and the high-level policy receives a replanning failure signal. It does not guess source/target identities.

Model loading is lazy so importing the server package does not allocate GPU memory. Unit tests use a fake generation backend and never download model weights.

## 11. Safety and Error Handling

The safety supervisor has final veto authority over all skill outputs. V1 detects and reports:

- grasp timeout or insufficient contact;
- source cup not lifted or dropped;
- excessive pre-pour tilt or slip;
- joint-limit margin violation;
- workspace violation;
- missing target pose or low confidence;
- pour timeout;
- non-finite observation or action;
- unavailable or invalid VLM response.

Every transition records environment ID, previous skill, requested skill, accepted skill, reason, and step index. Failures are explicit; no implicit fallback policy or raw action blending is allowed.

## 12. Testing Strategy

Development follows test-first implementation with at least 80% coverage for the new pure-Python packages.

### Unit tests

- Task schema acceptance and rejection cases.
- SemanticState vector shapes, finite values, and quaternion convention.
- Deterministic high-level sequence, retry, recovery, timeout, and abort decisions.
- Transition guards and minimum-duration behavior.
- Per-environment routing and targeted reset behavior.
- Heterogeneous action outputs are never blended.
- Qwen JSON extraction, allowlist enforcement, timeout, and malformed-response behavior.

### Integration tests

- Load the real `data/grasp_warm_tesollo.hdf5` through the existing `PourWarmStateBank` path.
- Map registry aliases to existing `grasp_v1` and `pour_v1` adapters without modifying either task.
- Exercise `command + image → fake Qwen backend → TaskSpecification → deterministic decision`.
- Exercise manager routing with fake skill implementations across multiple environments.

### Static and regression checks

- `ruff` on all new Python files.
- `pyright` on pure-Python domain modules where external Isaac typing is not required.
- Existing focused `grasp_v1` and `pour_v1` contract tests remain green.
- Importing the core hierarchical package does not import Transformers, load Qwen, or launch Isaac Sim.

Actual Qwen inference is a manual smoke test because model weights and GPU availability are runtime resources. The smoke test must call `/health`, send a real image and command, and receive a valid `TaskSpecification`.

## 13. Acceptance Criteria

The initial architecture is complete when:

1. No existing `hdgp` task folder or contract has been moved, renamed, or reorganized.
2. Existing grasp and pour contract tests still pass.
3. The real grasp warm-state HDF5 loads through the existing pour loader.
4. A validated Qwen response produces `TaskSpecification`, including `allowed_skills`.
5. Qwen cannot express low-level actions through the accepted schema.
6. The deterministic high-level policy emits one decision per environment.
7. The manager hard-routes environments independently and logs every accepted or rejected transition.
8. Approach and both bridges exist behind explicit interfaces; v1 approach uses rule-based Fabric IK.
9. Existing grasp and pour policies are reached only through adapters that preserve their contracts.
10. New unit and integration tests pass with at least 80% coverage.
11. A real Qwen smoke command is documented and works when model weights and sufficient GPU memory are available.
12. The implementation does not claim that high-level RL, learned approach, learned recovery, FoundationPose, ROS 2, or real-robot deployment has already been trained or validated.

## 14. Deferred Work

- Train `RLHighLevelPolicy` while low-level policies are frozen.
- Add transition residual policies after rule-based bridges are stable.
- Add perception noise, delay, confidence fluctuation, and temporary loss.
- Connect FoundationPose and RealSense to the same `SemanticState` contract.
- Add in-memory grasp-to-pour handoff after disk compatibility is proven.
- Add ROS 2 skill executor, robot bridge, and safety supervisor nodes.
- Add learned recovery and optional low-level fine-tuning only after deterministic end-to-end validation.
