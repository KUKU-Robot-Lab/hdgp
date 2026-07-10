#!/usr/bin/env python3
"""Report lift-task KPI summary from TensorBoard event file.

Usage:
  ./isaaclab.sh -p ../hdgp/scripts/reports/report_lift_kpi.py \
      --log_dir ../hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


KPI_TAGS = [
    "rewards/iter",
    "episode_lengths/iter",
    "Episode/Episode_Reward/strict_grasp_success",
    "Episode/Episode_Reward/lifting_object",
    "Episode/Episode_Reward/object_goal_tracking",
    "Episode/Episode_Reward/object_goal_tracking_fine_grained",
    "Episode/Episode_Reward/finger_contact_coverage",
    "Episode/Episode_Reward/contact_persistence",
]


def _find_event_file(log_dir: Path) -> Path:
    summary_dir = log_dir / "summaries"
    if not summary_dir.exists():
        raise FileNotFoundError(f"summaries directory not found: {summary_dir}")
    files = sorted(summary_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"no TensorBoard event file under: {summary_dir}")
    return files[-1]


def _window_mean(arr: np.ndarray, frac: float) -> float:
    n = len(arr)
    k = max(1, int(n * frac))
    return float(arr[-k:].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lift-task KPI from event logs.")
    parser.add_argument("--log_dir", type=Path, required=True, help="Run directory path (contains summaries/).")
    args = parser.parse_args()

    event_file = _find_event_file(args.log_dir)
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalar_tags = set(ea.Tags().get("scalars", []))

    print(f"[INFO] event_file: {event_file}")
    print("[INFO] KPI summary (last 10% window + last value)")
    print("-" * 110)
    print(f"{'tag':58} {'count':>8} {'last10%':>14} {'last':>14} {'max':>14}")
    print("-" * 110)

    for tag in KPI_TAGS:
        if tag not in scalar_tags:
            print(f"{tag:58} {'-':>8} {'(missing)':>14} {'(missing)':>14} {'(missing)':>14}")
            continue
        events = ea.Scalars(tag)
        vals = np.array([e.value for e in events], dtype=np.float64)
        print(
            f"{tag:58} {len(vals):8d} {_window_mean(vals, 0.10):14.6f} {vals[-1]:14.6f} {vals.max():14.6f}"
        )

    print("-" * 110)
    print("[NOTE] Qualitative video judgment is manual. This report is numeric-only KPI tracking.")


if __name__ == "__main__":
    main()

