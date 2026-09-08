#!/usr/bin/env python3
"""Continuously mirror rl_games TensorBoard runs to separate W&B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable


EVENT_GLOB = "events.out.tfevents.*"
STATE_FILE = ".wandb-tb-watcher-state.json"


def discover_runs(roots: Iterable[Path]) -> list[Path]:
    """Return run directories whose summaries directory contains tfevents."""
    runs: set[Path] = set()
    for root in roots:
        for event_file in root.glob(f"*/summaries/{EVENT_GLOB}"):
            runs.add(event_file.parent.parent)
    return sorted(runs)


def stable_run_id(run_dir: Path) -> str:
    """Return a deterministic W&B-compatible ID for a local run path."""
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:12]
    return f"tb-{digest}"


def select_new_scalars(
    snapshots: dict[str, dict[str, list[tuple[int, float, float]]]],
    offsets: dict[str, dict[str, int]],
) -> tuple[list[dict[str, float | int]], dict[str, dict[str, int]]]:
    """Convert unseen scalar events into W&B rows and updated offsets."""
    rows_by_step: dict[int, dict[str, float | int]] = {}
    updated = {source: dict(tags) for source, tags in offsets.items()}
    for source, tags in snapshots.items():
        source_offsets = updated.setdefault(source, {})
        for tag, events in tags.items():
            start = source_offsets.get(tag, 0)
            if start > len(events):
                start = 0
            for step, value, wall_time in events[start:]:
                row = rows_by_step.setdefault(
                    step, {"tb_step": step, "_timestamp": wall_time}
                )
                row[tag] = value
                row["_timestamp"] = max(float(row["_timestamp"]), wall_time)
            source_offsets[tag] = len(events)
    return [rows_by_step[step] for step in sorted(rows_by_step)], updated


def _read_state(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, offsets: dict[str, dict[str, int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(offsets, sort_keys=True))
    temporary.replace(path)


def _load_snapshots(summaries: Path) -> dict[str, dict[str, list[tuple[int, float, float]]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore[reportMissingImports]

    snapshots: dict[str, dict[str, list[tuple[int, float, float]]]] = {}
    for event_file in sorted(summaries.glob(EVENT_GLOB)):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:  # a partially-written tail is retried next poll
            print(f"[WARN] Cannot read {event_file}: {exc}", flush=True)
            continue
        tags: dict[str, list[tuple[int, float, float]]] = {}
        for tag in accumulator.Tags().get("scalars", []):
            tags[tag] = [
                (event.step, float(event.value), float(event.wall_time))
                for event in accumulator.Scalars(tag)
            ]
        snapshots[event_file.name] = tags
    return snapshots


def watch_run(args: argparse.Namespace) -> int:
    import wandb  # type: ignore[reportMissingImports]

    run_dir = args.run_dir.resolve()
    summaries = run_dir / "summaries"
    state_path = run_dir / STATE_FILE
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=stable_run_id(run_dir),
        resume="allow",
        name=run_dir.name,
        group=run_dir.parent.name,
        job_type="tensorboard-watch",
        tags=["tensorboard", run_dir.parent.name],
        config={"local_run_dir": str(run_dir)},
    )
    run.define_metric("tb_step")
    run.define_metric("*", step_metric="tb_step")
    print(f"[INFO] Watching {summaries} -> {run.url}", flush=True)

    offsets = _read_state(state_path)
    last_change = time.monotonic()
    last_signature: tuple[tuple[str, int, int], ...] | None = None
    try:
        while True:
            signature = tuple(
                (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                for path in sorted(summaries.glob(EVENT_GLOB))
            )
            if signature != last_signature:
                last_change = time.monotonic()
                last_signature = signature
                rows, offsets = select_new_scalars(_load_snapshots(summaries), offsets)
                for row in rows:
                    run.log(row)
                if rows:
                    _write_state(state_path, offsets)
                    print(f"[INFO] {run_dir.name}: uploaded {len(rows)} new rows", flush=True)
            elif time.monotonic() - last_change >= args.idle_timeout:
                print(f"[INFO] {run_dir.name}: idle timeout; watcher exiting", flush=True)
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        run.finish()
    return 0


def supervise(args: argparse.Namespace) -> int:
    children: dict[Path, subprocess.Popen[bytes]] = {}
    print(f"[INFO] Scanning {', '.join(map(str, args.root))}", flush=True)
    try:
        while True:
            now = time.time()
            for run_dir in discover_runs(args.root):
                child = children.get(run_dir)
                if child is not None and child.poll() is None:
                    continue
                event_files = list((run_dir / "summaries").glob(EVENT_GLOB))
                newest_mtime = max(path.stat().st_mtime for path in event_files)
                if now - newest_mtime > args.active_within:
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-dir",
                    str(run_dir),
                    "--entity",
                    args.entity,
                    "--project",
                    args.project,
                    "--poll-interval",
                    str(args.poll_interval),
                    "--idle-timeout",
                    str(args.idle_timeout),
                ]
                children[run_dir] = subprocess.Popen(command)
                print(f"[INFO] Started watcher for {run_dir}", flush=True)
            time.sleep(args.scan_interval)
    except KeyboardInterrupt:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            if child.poll() is None:
                child.wait(timeout=30)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path, action="append", help="Parent containing run directories")
    mode.add_argument("--run-dir", type=Path, help="Watch one run (used by the supervisor)")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--poll-interval", type=float, default=60)
    parser.add_argument("--scan-interval", type=float, default=60)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    parser.add_argument("--active-within", type=float, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    raise SystemExit(watch_run(cli_args) if cli_args.run_dir else supervise(cli_args))
