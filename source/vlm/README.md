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

`CheckpointResolver` resolves these runs relative to the HDGP repository root:

- `log/rl_games/open-tesol/right/grasp-v1/lstm_test1`
- `log/rl_games/open-tesol/right/pour-v1/lstm_test2`

The warm grasp state remains referenced at `data/grasp_warm_tesollo.hdf5`.

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
