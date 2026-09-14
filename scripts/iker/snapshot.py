"""Head-camera snapshot of one IKER configuration (design spec §5).

Spawns the scene, holds the head at the snapshot angles while the shoes settle, renders RGB and
depth, and writes iker_runs/shoe_place/config_XX/{snapshot_raw.png, snapshot.png, keypoints.json}.
The files are written even when a check fails (``checks.passed`` is false) and the exit code is 1.
The scene, settling and checks live in tasks/iker_shoe/head_snapshot.py (shared with observe.py).

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/snapshot.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of one IKER configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
parser.add_argument("--out-dir", default="", help="output directory (default: iker_runs/shoe_place/config_XX)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("SNAPSHOT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import head_snapshot as hs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402


def main() -> int:
    if not args.settle_steps > max(hs.SETTLE_WINDOW_STEPS, args.render_steps):
        raise ValueError("--settle-steps must exceed both the settle window and --render-steps")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    scene = hs.build_scene(layout.shoe_start_poses(config, meta), args.device)

    home = scene.arm.data.default_joint_pos.clone()
    scene.arm.write_joint_state_to_sim(home, torch.zeros_like(home))
    target, aim = hs.head_target(scene.arm, home)
    traces = hs.settle(scene, target, args.settle_steps, args.render_steps)
    snapshot = hs.capture(scene, traces, meta, aim)

    out_dir = Path(args.out_dir) if args.out_dir else layout.RUNS_DIR / f"config_{config.index:02d}"
    passed = hs.write(snapshot, out_dir, config)
    annotated, checks = snapshot.annotated, snapshot.checks
    print(
        f"SNAPSHOT config {config.index:02d} derotation {annotated.derotation_deg:+.1f} deg margin {annotated.min_margin_px:+.1f} px "
        f"gap {annotated.min_gap_px} px head_err {checks['head_error_deg']:.2f} deg settle {checks['settle_motion_m'] * 1000:.2f} mm "
        f"rack_depth_err {checks['rack_depth_error_max_m']} m passed {passed} -> {out_dir}",
        flush=True,
    )
    for failure in snapshot.failures.values():
        print(f"SNAPSHOT CHECK FAILED: {failure}", flush=True)
    return 0 if passed else 1


os._exit(main())
