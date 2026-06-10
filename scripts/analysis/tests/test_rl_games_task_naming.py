from __future__ import annotations

from pathlib import Path


TRAIN_PATH = Path("/home/user/rl_ws/hdgp/scripts/reinforcement_learning/rl_games/train.py")
PLAY_PATH = Path("/home/user/rl_ws/hdgp/scripts/reinforcement_learning/rl_games/play.py")


def test_train_uses_distinct_lstm_run_prefix_for_auto_generated_run_names():
    source = TRAIN_PATH.read_text(encoding="utf-8")

    assert "def _resolve_run_dir_prefix(task_name: str) -> str:" in source
    assert 'return "lstm_test" if task_key.endswith("-lstm") else "test"' in source
    assert 'run_prefix = _resolve_run_dir_prefix(task_name)' in source
    assert 'if name.startswith(run_prefix):' in source
    assert 'log_dir = f"{run_prefix}{next_idx}"' in source


def test_play_searches_lstm_test_runs_by_default_for_lstm_tasks():
    source = PLAY_PATH.read_text(encoding="utf-8")

    assert "def _resolve_run_dir_prefix(task_name: str) -> str:" in source
    assert 'return "lstm_test" if task_key.endswith("-lstm") else "test"' in source
    assert 'run_prefix = _resolve_run_dir_prefix(train_task_name)' in source
    assert 'run_dir = agent_cfg["params"]["config"].get("full_experiment_name", f"{run_prefix}.*")' in source
