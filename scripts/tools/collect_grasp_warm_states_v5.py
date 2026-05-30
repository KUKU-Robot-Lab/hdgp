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

"""Collect grasp warm states for 5g_pour_right_v5 warmstart.

5g_grasp_right_v7_2 체크포인트(test4)를 이용해 pour v5 warmstart 용
HDF5 를 수집한다. v3 대비 --warm_min_contacts / --warm_stable_steps 를
높여 품질 기준을 강화하는 것이 v5 전용 목적이다.

사용 예:
    python3 scripts/tools/collect_grasp_warm_states_v5.py \\
        --checkpoint /home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7_2/test4/nn/5g_grasp_right-v7-2.pth \\
        --num_envs 2048 \\
        --target_count 2048 \\
        --out /home/user/rl_ws/datasets/grasp_warm_v7_2_contact4_raw.hdf5 \\
        --warm_min_contacts 4 \\
        --warm_stable_steps 12 \\
        --timeout_sec 10800 \\
        --status_sec 30
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HDGP_ROOT = Path(__file__).resolve().parents[2]   # hdgp/scripts/tools → hdgp/
_RL_WS = _HDGP_ROOT.parent                          # hdgp/ → rl_ws/
_ISAACLAB_SH = _RL_WS / "IsaacLab" / "isaaclab.sh"
_PLAY_PY = _HDGP_ROOT / "scripts/reinforcement_learning/rl_games/play.py"
_DEFAULT_CKPT = (
    _HDGP_ROOT
    / "log/rl_games/pipeline/right/5g_grasp_right_v7_2/test4/nn/5g_grasp_right-v7-2.pth"
)
_DEFAULT_OUT = _RL_WS / "datasets/grasp_warm_v7_2_contact4_raw.hdf5"
_TASK = "5g_grasp_right-v7-2"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--target_count", type=int, default=2048)
    p.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
    p.add_argument("--out", type=str, default=str(_DEFAULT_OUT))
    p.add_argument(
        "--warm_min_contacts",
        type=int,
        default=4,
        help="grasp 성공으로 인정할 최소 접촉 손가락 수 (env.warm_min_contacts 오버라이드)",
    )
    p.add_argument(
        "--warm_stable_steps",
        type=int,
        default=12,
        help="안정 접촉 유지 스텝 수 (env.warm_contact_stable_steps 오버라이드)",
    )
    p.add_argument(
        "--status_sec",
        type=float,
        default=30.0,
        help="출력 파일 폴링 간격(초)",
    )
    p.add_argument("--timeout_sec", type=float, default=3600.0)
    p.add_argument(
        "--keep_adr",
        action="store_true",
        help="ADR 노이즈 유지 (기본: 비활성, 깨끗한 성공 분포 수집)",
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
    if args.warm_min_contacts < 1:
        raise ValueError(f"--warm_min_contacts must be >= 1, got {args.warm_min_contacts}")
    if args.warm_stable_steps < 1:
        raise ValueError(f"--warm_stable_steps must be >= 1, got {args.warm_stable_steps}")


def _build_command(args: argparse.Namespace) -> list[str]:
    out_path = Path(args.out).resolve()
    cmd = [
        str(_ISAACLAB_SH),
        "-p",
        str(_PLAY_PY),
        "--task",
        _TASK,
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
        f"env.warm_min_contacts={args.warm_min_contacts}",
        f"env.warm_contact_stable_steps={args.warm_stable_steps}",
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
    print(
        "[collect_grasp_warm_states_v5] launching:\n  " + " ".join(cmd),
        flush=True,
    )
    print(
        f"[collect_grasp_warm_states_v5] "
        f"min_contacts={args.warm_min_contacts} stable_steps={args.warm_stable_steps} "
        f"target={args.target_count} out={out_path}",
        flush=True,
    )

    proc = subprocess.Popen(cmd, cwd=str(_HDGP_ROOT), start_new_session=True)
    deadline = time.monotonic() + args.timeout_sec
    saved = False
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(
                    f"[collect_grasp_warm_states_v5] play.py exited early (code={ret}).",
                    flush=True,
                )
                break

            exists, mtime = _file_signature(out_path)
            if exists and (not before_exists or mtime > before_mtime):
                saved = True
                print(
                    f"[collect_grasp_warm_states_v5] output written: {out_path} → stopping.",
                    flush=True,
                )
                break

            elapsed = time.monotonic() - (deadline - args.timeout_sec)
            if time.monotonic() > deadline:
                print(
                    f"[collect_grasp_warm_states_v5] timeout after {args.timeout_sec}s.",
                    flush=True,
                )
                break

            print(
                f"[collect_grasp_warm_states_v5] waiting... elapsed={elapsed:.0f}s",
                flush=True,
            )
            time.sleep(args.status_sec)
    finally:
        _terminate_process_group(proc)

    if saved and out_path.is_file():
        import h5py
        try:
            with h5py.File(out_path, "r") as h5:
                n = int(h5["warm_states"]["arm_joint_pos"].shape[0])
            print(
                f"[collect_grasp_warm_states_v5] DONE. {n} states → {out_path}",
                flush=True,
            )
        except Exception:
            print(
                f"[collect_grasp_warm_states_v5] DONE. → {out_path}",
                flush=True,
            )
        return 0

    print(
        "[collect_grasp_warm_states_v5] FAILED: no warm-state cache produced.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
