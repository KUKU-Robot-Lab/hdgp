#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect 5g_grasp_right_v11 states for pour v3/v5 warm starts."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


_RL_WS = Path(__file__).resolve().parents[2]
_HDGP_ROOT = _RL_WS / "hdgp"
_ISAACLAB_SH = _RL_WS / "IsaacLab" / "isaaclab.sh"
_PLAY_PY = _HDGP_ROOT / "scripts/reinforcement_learning/rl_games/play.py"
_DEFAULT_OUT = _RL_WS / "datasets/grasp_warm_v11_pour.hdf5"
_DEFAULT_TASK = "5g_grasp_right-v11-lstm"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="trained grasp v11 checkpoint")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help="output HDF5 path")
    p.add_argument("--task", default=_DEFAULT_TASK, help="registered grasp v11 task id")
    p.add_argument("--num_envs", type=int, default=256, help="parallel env count")
    p.add_argument("--target_count", type=int, default=2048, help="states to collect")
    p.add_argument("--goal_x", type=float, required=True, help="fixed transport goal x")
    p.add_argument("--goal_y", type=float, required=True, help="fixed transport goal y")
    p.add_argument("--goal_z", type=float, required=True, help="fixed transport goal z")
    p.add_argument(
        "--bead_count",
        type=int,
        default=30,
        help="fixed active bead count; use a negative value to keep task cfg",
    )
    p.add_argument("--poll_sec", type=float, default=5.0, help="output polling interval")
    p.add_argument("--timeout_sec", type=float, default=3600.0, help="max wait time")
    p.add_argument(
        "--keep_adr",
        action="store_true",
        help="keep ADR enabled (default disables ADR for deterministic collection)",
    )
    return p.parse_args()


def _validate(args: argparse.Namespace) -> None:
    for path, label in ((_ISAACLAB_SH, "isaaclab.sh"), (_PLAY_PY, "play.py")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if args.target_count <= 0:
        raise ValueError(f"--target_count must be positive, got {args.target_count}")
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}")


def _axis_range_override(name: str, value: float) -> str:
    return f"env.{name}=[{value},{value}]"


def _build_command(args: argparse.Namespace) -> list[str]:
    out_path = Path(args.out).resolve()
    cmd = [
        str(_ISAACLAB_SH),
        "-p",
        str(_PLAY_PY),
        "--task",
        args.task,
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--num_envs",
        str(args.num_envs),
        "--headless",
    ]
    if not args.keep_adr:
        cmd.append("--disable_adr")

    cmd += [
        "env.enable_warm_state_export=true",
        f"env.warm_state_export_path={out_path}",
        f"env.warm_state_target_count={args.target_count}",
        "env.warm_state_success_source=transport",
        "env.enable_phase_curriculum=false",
        _axis_range_override("transport_goal_x_range", args.goal_x),
        _axis_range_override("transport_goal_y_range", args.goal_y),
        _axis_range_override("transport_goal_z_range", args.goal_z),
    ]
    if args.bead_count >= 0:
        cmd += [
            "env.dynamic_bead_spawn_enabled=false",
            f"env.bead_count_min={args.bead_count}",
            f"env.bead_count_max={args.bead_count}",
        ]
    return cmd


def _file_signature(path: Path) -> tuple[bool, float]:
    if not path.is_file():
        return (False, 0.0)
    return (True, path.stat().st_mtime)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    for sig, wait_s in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    args = _parse_args()
    _validate(args)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    before_exists, before_mtime = _file_signature(out_path)

    cmd = _build_command(args)
    print("[collect_grasp_v11_pour_warm_states] launching:\n  " + " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, cwd=str(_HDGP_ROOT), start_new_session=True)
    deadline = time.monotonic() + args.timeout_sec
    saved = False
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(
                    f"[collect_grasp_v11_pour_warm_states] play.py exited early (code={ret}).",
                    flush=True,
                )
                break

            exists, mtime = _file_signature(out_path)
            if exists and (not before_exists or mtime > before_mtime):
                saved = True
                print(
                    f"[collect_grasp_v11_pour_warm_states] output written: {out_path}; "
                    "stopping rollout.",
                    flush=True,
                )
                break

            if time.monotonic() > deadline:
                print(
                    f"[collect_grasp_v11_pour_warm_states] timeout after {args.timeout_sec}s.",
                    flush=True,
                )
                break

            time.sleep(args.poll_sec)
    finally:
        _terminate_process_group(proc)

    if saved and out_path.is_file():
        print(
            f"[collect_grasp_v11_pour_warm_states] DONE. Warm-state cache: {out_path}",
            flush=True,
        )
        return 0

    print(
        "[collect_grasp_v11_pour_warm_states] FAILED: no completed warm-state cache produced.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
