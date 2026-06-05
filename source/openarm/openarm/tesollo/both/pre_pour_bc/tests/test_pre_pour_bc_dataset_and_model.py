from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch

_TASK_ROOT = Path(__file__).resolve().parents[1]
_HDGP_ROOT = Path(__file__).resolve().parents[7]
_SCRIPT_PATH = _HDGP_ROOT / "scripts" / "build_pre_pour_bc_dataset.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_demo(path: Path, *, name: str, action_dim: int = 18, obs_dim: int = 91, align_index: int | None = 3) -> None:
    with h5py.File(path, "w") as f:
        demo = f.create_group(f"data/{name}")
        n = 6
        demo.create_dataset("actions", data=np.ones((n, action_dim), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("actor_obs", data=np.ones((n, obs_dim), dtype=np.float32))
        obs.create_dataset("prev_actions", data=np.zeros((n, 18), dtype=np.float32))
        obs.create_dataset("tip_force_norm", data=np.full((n, 5), 0.2, dtype=np.float32))
        info = obs.create_group("datagen_info")
        terms = info.create_group("subtask_term_signals")
        if align_index is not None:
            align = np.zeros((n,), dtype=bool)
            align[align_index:] = True
            terms.create_dataset("align_done", data=align)


def test_dataset_builder_truncates_at_first_align_done(tmp_path: Path) -> None:
    module = _load_module(_SCRIPT_PATH, "build_pre_pour_bc_dataset")
    src = tmp_path / "src"
    src.mkdir()
    _write_demo(src / "good.hdf5", name="demo_0", align_index=3)
    _write_demo(src / "bad_obs.hdf5", name="demo_0", obs_dim=90)
    output = tmp_path / "out.hdf5"

    report = module.build_dataset(src, output, force_threshold=0.05, curl_threshold=0.05)

    assert report.kept == 1
    assert report.skipped >= 1
    with h5py.File(output, "r") as f:
        demo = f["data/demo_0"]
        assert demo["actions"].shape == (4, 18)
        assert demo["obs/actor_obs"].shape == (4, 91)
        assert demo.attrs["source_file"] == "good.hdf5"


def test_bc_rnn_forward_and_group_weighted_loss() -> None:
    module = _load_module(_TASK_ROOT / "train_bc_rnn.py", "pre_pour_train_bc_rnn")
    model = module.BCRNN(obs_dim=91, action_dim=18, hidden_dim=16, num_layers=1)
    obs = torch.randn(2, 5, 91)
    target = torch.randn(2, 5, 18)
    pred = model(obs)
    loss = module.group_weighted_mse(pred, target)

    assert pred.shape == (2, 5, 18)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
