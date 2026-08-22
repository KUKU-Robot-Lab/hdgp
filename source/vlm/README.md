# HDGP VLM Pouring

This extension adds vision-grounded hierarchical orchestration without changing
the existing `source/openarm` tasks. It lives in the shallow `vlm.pouring`
package and references trained HDGP artifacts in place.

## Included CPU-side structure

- strict `TaskSpecification` parsing for Qwen output
- semantic state aggregation and deterministic high-level routing
- guarded, per-environment hard skill selection and safety validation
- approach, grasp/lift, bridge, bimanual-pour, and recovery adapters
- lazy Qwen3-VL backend plus a same-machine HTTP service/client contract

The learned grasp and pour policies are injected through adapter backends at the
Isaac Lab integration boundary. This package does not copy checkpoints or alter
the existing task definitions.

## Existing artifacts

`CheckpointResolver` resolves runs relative to the HDGP repository root. The
default task map points at the runs that exist on this machine today:

- `open-tesol_r_grasp_v1-lstm` → `log/rl_games/open-tesol/right/grasp-v1/<run>`
- `open-tesol_b_pour_v1-lstm` → `log/rl_games/open-tesol/both/pour-v1/<run>`

Retrained runs (e.g. the `agnostic` tracks) are injected via the
`task_logs=` constructor argument or a direct `checkpoint=` path without
editing this package. The warm grasp state is referenced at
`data/grasp_warm_tesollo_right.hdf5`, and grasp-to-pour compatibility is
delegated to the bimanual loader in
`source/openarm/openarm/tesollo/both/pour_v1/warm_state_bank.py`.

## Isaac integration boundary (fabric tasks)

Two modules realize the boundary for the `agnostic` fabric tasks
(`grasp_lift_fabric` / `pour_fabric`) whose action space is an absolute palm
6D pose plus absolute hand joint targets:

- `fabric_bridge.py` — pure math, no Isaac/torch imports. `PalmActionSpace`
  is the exact inverse of the env's symmetric action decode, and
  `command_to_action` maps `TASK_SPACE_POSE` (rule-based skills such as
  approach), `POLICY_ACTION` (RL skills), and `SAFE_STOP`/`NO_OP` (hold the
  current pose) onto one action row.
- `isaac_state.py` — `FabricStateProvider` assembles validated
  `SemanticState` batches from a narrow `FabricSceneView` protocol (a thin
  adapter over the env buffers) plus the skill manager's routing state.

`scripts/vlm/run_pouring_demo.py` wires both into a live Isaac scene with a
stubbed task specification (the Qwen GPU boundary stays closed): the approach
skill drives the arm to a pregrasp point through Fabrics, the deterministic
HRL advances to DONE on semantic success, and a scene camera saves the RGB
frames that later become the Qwen input.

```bash
PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/vlm/run_pouring_demo.py \
  --num_envs 2 --steps 600 --out outputs/vlm_demo --headless
```

## CPU verification

From the HDGP repository root:

```bash
PYTHONPATH=source/vlm:source/openarm python3 -m pytest \
  source/vlm/vlm/pouring/tests \
  --cov=vlm.pouring --cov-report=term-missing --cov-fail-under=80 -q
ruff check source/vlm
pyright --project source/vlm
```

## Qwen service and GPU boundary

Install the extension and the optional model-server dependencies before the
first real model run:

```bash
python3 -m pip install -e source/vlm
python3 -m pip install -r source/vlm/requirements-qwen.txt
```

The following command starts the localhost service without loading model
weights. The first valid `POST /v1/task-grounding` request calls
`QwenBackend.load()`, which loads `Qwen/Qwen3-VL-4B-Instruct` with
`device_map="auto"`. That request is the GPU-use boundary and is intentionally
outside the CPU-only setup and verification stage.

```bash
PYTHONPATH=source/vlm python3 -m vlm.pouring.qwen_server \
  --host 127.0.0.1 --port 8100 \
  --model Qwen/Qwen3-VL-4B-Instruct
```

The endpoint accepts JSON with a non-empty command and a base64-encoded RGB
image:

```json
{
  "command": "Pour from the right cup into the left cup",
  "image_base64": "..."
}
```

Qwen may only produce the validated task contract. Joint commands, actions,
contact state, and control poses are rejected; those remain owned by the
deterministic HRL and skill layers.
