from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock


MODULE_PATH = Path(__file__).parents[1] / "wandb_tb_watcher.py"
SPEC = importlib.util.spec_from_file_location("wandb_tb_watcher", MODULE_PATH)
assert SPEC is not None
watcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watcher)


def test_discover_runs_finds_only_summaries_with_tfevents(tmp_path: Path) -> None:
    valid = tmp_path / "fj_b9" / "summaries"
    valid.mkdir(parents=True)
    (valid / "events.out.tfevents.1.host").touch()
    (tmp_path / "empty" / "summaries").mkdir(parents=True)

    assert watcher.discover_runs([tmp_path]) == [valid.parent]


def test_run_id_is_stable_and_distinguishes_parent_paths() -> None:
    fj = Path("/logs/grasp-fj/run1")
    kp = Path("/logs/grasp-kp/run1")

    assert watcher.stable_run_id(fj) == watcher.stable_run_id(fj)
    assert watcher.stable_run_id(fj) != watcher.stable_run_id(kp)
    assert watcher.stable_run_id(fj).startswith("tb-")


def test_select_new_scalars_returns_only_unseen_events() -> None:
    snapshots = {
        "events.1": {
            "reward": [(1, 1.5, 10.0), (2, 2.5, 11.0)],
            "loss": [(1, 0.8, 10.0)],
        }
    }
    offsets = {"events.1": {"reward": 1}}

    rows, new_offsets = watcher.select_new_scalars(snapshots, offsets)

    assert rows == [
        {"tb_step": 1, "_timestamp": 10.0, "loss": 0.8},
        {"tb_step": 2, "_timestamp": 11.0, "reward": 2.5},
    ]
    assert new_offsets == {"events.1": {"reward": 2, "loss": 1}}


def test_select_new_scalars_recovers_when_event_file_is_replaced() -> None:
    snapshots = {"events.1": {"reward": [(0, 3.0, 20.0)]}}
    stale_offsets = {"events.1": {"reward": 9}}

    rows, offsets = watcher.select_new_scalars(snapshots, stale_offsets)

    assert rows == [{"tb_step": 0, "_timestamp": 20.0, "reward": 3.0}]
    assert offsets["events.1"]["reward"] == 1


def test_state_round_trip_and_invalid_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    expected = {"events.1": {"reward": 3}}

    watcher._write_state(state_path, expected)
    assert watcher._read_state(state_path) == expected

    state_path.write_text("not-json")
    assert watcher._read_state(state_path) == {}


def test_watch_run_uploads_new_rows_and_finishes(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "grasp-fj" / "fj_b9"
    summaries = run_dir / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "events.out.tfevents.1").write_bytes(b"event")

    fake_run = SimpleNamespace(
        url="https://wandb.invalid/run",
        define_metric=Mock(),
        log=Mock(),
        finish=Mock(),
    )
    fake_wandb = SimpleNamespace(init=Mock(return_value=fake_run))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        watcher,
        "_load_snapshots",
        lambda _: {"events.out.tfevents.1": {"reward": [(4, 9.0, 12.0)]}},
    )
    monkeypatch.setattr(watcher.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    args = SimpleNamespace(
        run_dir=run_dir,
        entity="team",
        project="project",
        poll_interval=1,
        idle_timeout=30,
    )

    assert watcher.watch_run(args) == 0
    fake_run.log.assert_called_once_with(
        {"tb_step": 4, "_timestamp": 12.0, "reward": 9.0}
    )
    fake_run.finish.assert_called_once()
    assert (run_dir / watcher.STATE_FILE).exists()


def test_supervisor_starts_recent_run(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "grasp-kp" / "kp_a12"
    summaries = run_dir / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "events.out.tfevents.1").write_bytes(b"event")
    child = SimpleNamespace(poll=Mock(return_value=0), terminate=Mock(), wait=Mock())
    popen = Mock(return_value=child)
    monkeypatch.setattr(watcher.subprocess, "Popen", popen)
    monkeypatch.setattr(watcher.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    args = SimpleNamespace(
        root=[run_dir.parent],
        entity="team",
        project="project",
        poll_interval=10,
        scan_interval=10,
        idle_timeout=30,
        active_within=30,
    )

    assert watcher.supervise(args) == 0
    command = popen.call_args.args[0]
    assert command[command.index("--run-dir") + 1] == str(run_dir)
    assert command[command.index("--entity") + 1] == "team"
    assert command[command.index("--project") + 1] == "project"


def test_load_snapshots_reads_scalar_events(tmp_path: Path, monkeypatch) -> None:
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "events.out.tfevents.1").touch()

    class FakeAccumulator:
        def __init__(self, path, size_guidance):
            assert path.endswith("events.out.tfevents.1")
            assert size_guidance == {"scalars": 0}

        def Reload(self):
            return self

        def Tags(self):
            return {"scalars": ["reward"]}

        def Scalars(self, tag):
            assert tag == "reward"
            return [SimpleNamespace(step=2, value=3.5, wall_time=4.5)]

    module = SimpleNamespace(EventAccumulator=FakeAccumulator)
    monkeypatch.setitem(
        sys.modules,
        "tensorboard.backend.event_processing.event_accumulator",
        module,
    )

    assert watcher._load_snapshots(summaries) == {
        "events.out.tfevents.1": {"reward": [(2, 3.5, 4.5)]}
    }


def test_parse_args_accepts_multiple_roots(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watcher",
            "--root",
            "/logs/fj",
            "--root",
            "/logs/kp",
            "--entity",
            "team",
            "--project",
            "rl",
        ],
    )

    args = watcher.parse_args()

    assert args.root == [Path("/logs/fj"), Path("/logs/kp")]
    assert args.entity == "team"
    assert args.project == "rl"
