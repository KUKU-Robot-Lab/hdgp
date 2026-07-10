#!/usr/bin/env python3
"""Build a compact analysis dataset from an IsaacLab RSL-RL run directory.

Outputs:
- scalars.csv: all TensorBoard scalars (tag, step, value, wall_time)
- summary.json: tag stats + config pointers + basic aggregates
- grasp_debug.csv (optional): parsed GRASP_DEBUG lines from a stdout log
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime


def _read_yaml(path: str):
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _find_event_files(log_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))


def _read_scalars(event_files: list[str]) -> list[dict]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"[WARN] TensorBoard not available: {exc}")
        return []

    rows = []
    for path in event_files:
        try:
            acc = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
            acc.Reload()
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        tags = acc.Tags().get("scalars", [])
        for tag in tags:
            try:
                for e in acc.Scalars(tag):
                    rows.append(
                        {
                            "tag": tag,
                            "step": int(e.step),
                            "value": float(e.value),
                            "wall_time": float(e.wall_time),
                            "source": os.path.basename(path),
                        }
                    )
            except Exception:
                continue
    return rows


def _write_scalars_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["tag"], r["step"], r["wall_time"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "step", "value", "wall_time", "source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarize_tags(rows: list[dict], last_n: int) -> dict:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)

    stats = {}
    for tag, items in by_tag.items():
        items = sorted(items, key=lambda r: (r["step"], r["wall_time"]))
        values = [r["value"] for r in items]
        last = items[-1]
        if last_n > 0:
            tail = values[-last_n:]
            last_n_mean = sum(tail) / len(tail)
        else:
            last_n_mean = None
        stats[tag] = {
            "count": len(items),
            "last_step": last["step"],
            "last_value": last["value"],
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": sum(values) / len(values),
            "last_n_mean": last_n_mean,
        }
    return stats


def _parse_grasp_debug(log_path: str) -> list[dict]:
    if not log_path or not os.path.isfile(log_path):
        return []

    # Example:
    # [GRASP_DEBUG] step=1000 side=left close_mean=0.123 dist_mean=0.045 action_mean=4.1202
    # joint_mean=0.0501 open_mean=0.0440 close_mean=0.0000 delta_mean=-4.0762
    pattern = re.compile(
        r"\[GRASP_DEBUG\]\s+step=(\d+)\s+side=(left|right)\s+"
        r"close_mean=([-0-9.]+)\s+dist_mean=([-0-9.]+)\s+"
        r"action_mean=([-0-9.]+)\s+joint_mean=([-0-9.]+)\s+"
        r"open_mean=([-0-9.]+)\s+close_mean=([-0-9.]+)\s+delta_mean=([-0-9.]+)"
    )

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            rows.append(
                {
                    "step": int(m.group(1)),
                    "side": m.group(2),
                    "close_mean": float(m.group(3)),
                    "dist_mean": float(m.group(4)),
                    "action_mean": float(m.group(5)),
                    "joint_mean": float(m.group(6)),
                    "open_mean": float(m.group(7)),
                    "close_limit_mean": float(m.group(8)),
                    "delta_mean": float(m.group(9)),
                }
            )
    return rows


def _write_grasp_debug_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["side"], r["step"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "side",
                "close_mean",
                "dist_mean",
                "action_mean",
                "joint_mean",
                "open_mean",
                "close_limit_mean",
                "delta_mean",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build analysis dataset from a run directory.")
    ap.add_argument("--log_dir", required=True, help="Path to run directory (e.g., .../log/rsl_rl/grasp2g/test3)")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <log_dir>/analysis)")
    ap.add_argument("--last_n", type=int, default=200, help="Window for last_n_mean stats")
    ap.add_argument("--stdout_log", default=None, help="Optional stdout log path to parse GRASP_DEBUG lines")
    args = ap.parse_args()

    log_dir = os.path.abspath(args.log_dir)

    # Auto-detect true log directory if event files are in a subdirectory
    if not _find_event_files(log_dir):
        print(f"[INFO] No event files found in {log_dir}. Checking subdirectories...")
        possible_dirs = []
        for item in os.listdir(log_dir):
            sub_dir = os.path.join(log_dir, item)
            if os.path.isdir(sub_dir) and _find_event_files(sub_dir):
                possible_dirs.append(sub_dir)

        if len(possible_dirs) == 1:
            new_log_dir = possible_dirs[0]
            print(f"[INFO] Found a single log directory: {new_log_dir}. Using it.")
            log_dir = new_log_dir
        elif len(possible_dirs) > 1:
            print("[ERROR] Found multiple possible log directories. Please specify one explicitly:")
            for d in sorted(possible_dirs):
                print(f"  - {d}")
            return 1  # Exit with error
        else:
            print("[WARN] Could not find any subdirectory with event files.")

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(log_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    event_files = _find_event_files(log_dir)
    scalars = _read_scalars(event_files)
    scalars_csv = os.path.join(out_dir, "scalars.csv")
    _write_scalars_csv(scalars, scalars_csv)

    tag_stats = _summarize_tags(scalars, args.last_n) if scalars else {}

    env_yaml = os.path.join(log_dir, "params", "env.yaml")
    agent_yaml = os.path.join(log_dir, "params", "agent.yaml")
    env_cfg = _read_yaml(env_yaml) if os.path.isfile(env_yaml) else None
    agent_cfg = _read_yaml(agent_yaml) if os.path.isfile(agent_yaml) else None

    grasp_rows = _parse_grasp_debug(args.stdout_log) if args.stdout_log else []
    grasp_csv = os.path.join(out_dir, "grasp_debug.csv")
    _write_grasp_debug_csv(grasp_rows, grasp_csv)

    summary = {
        "log_dir": log_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "event_files": [os.path.basename(p) for p in event_files],
        "scalar_tags": sorted(tag_stats.keys()),
        "scalar_count": len(scalars),
        "scalars_csv": os.path.relpath(scalars_csv, log_dir),
        "grasp_debug_csv": os.path.relpath(grasp_csv, log_dir) if grasp_rows else None,
        "env_yaml": os.path.relpath(env_yaml, log_dir) if os.path.isfile(env_yaml) else None,
        "agent_yaml": os.path.relpath(agent_yaml, log_dir) if os.path.isfile(agent_yaml) else None,
        "tag_stats": tag_stats,
        "env_cfg": env_cfg,
        "agent_cfg": agent_cfg,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"[INFO] Wrote: {summary_path}")
    if scalars:
        print(f"[INFO] Wrote: {scalars_csv}")
    if grasp_rows:
        print(f"[INFO] Wrote: {grasp_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
