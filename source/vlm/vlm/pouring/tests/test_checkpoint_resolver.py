from __future__ import annotations

from pathlib import Path

import pytest

from vlm.pouring.checkpoint_resolver import CheckpointResolver


def _make_run(root: Path, task: str, run_name: str) -> Path:
    run = root / "log/rl_games/open-tesol/right" / task / run_name
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir()
    return run


def test_resolver_returns_checkpoint_and_neighbor_params(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "grasp-v1", "lstm_test1")
    (run / "nn/open-tesol_r_grasp_v1-lstm.pth").touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    result = CheckpointResolver(tmp_path).resolve(
        task_id="open-tesol_r_grasp_v1-lstm",
        run_dir="lstm_test1",
    )

    assert result.checkpoint.name == "open-tesol_r_grasp_v1-lstm.pth"
    assert result.agent_yaml == run / "params/agent.yaml"
    assert result.env_yaml == run / "params/env.yaml"


def test_resolver_rejects_missing_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lstm_test1"):
        CheckpointResolver(tmp_path).resolve("open-tesol_r_grasp_v1-lstm", "lstm_test1")


def test_resolver_rejects_ambiguous_run_selector(tmp_path: Path) -> None:
    _make_run(tmp_path, "grasp-v1", "lstm_test1")
    _make_run(tmp_path, "grasp-v1", "lstm_test2")

    with pytest.raises(ValueError, match="exactly one"):
        CheckpointResolver(tmp_path).resolve("open-tesol_r_grasp_v1-lstm", "lstm_test*")


def test_resolver_accepts_explicit_checkpoint_but_still_requires_neighbor_params(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "pour-v1", "manual")
    checkpoint = run / "nn/custom.pth"
    checkpoint.touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    result = CheckpointResolver(tmp_path).resolve(
        "open-tesol_b_pour_v1-lstm",
        "ignored",
        checkpoint=checkpoint,
    )

    assert result.checkpoint == checkpoint


def test_resolver_rejects_unknown_task_without_explicit_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unsupported task"):
        CheckpointResolver(tmp_path).resolve("unknown-task", "test1")


def test_resolver_accepts_injected_task_map_for_retrained_runs(tmp_path: Path) -> None:
    run = tmp_path / "log/rl_games/pipeline/right/grasp-fabric/agn_test1"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir()
    (run / "nn/agn_grasp-fabric.pth").touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    resolver = CheckpointResolver(
        tmp_path,
        task_logs={"agn_grasp-fabric": ("pipeline/right", "grasp-fabric")},
    )
    result = resolver.resolve("agn_grasp-fabric", "agn_test1")

    assert result.checkpoint == run / "nn/agn_grasp-fabric.pth"
