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

"""Collect grasp-success warm states for 5g_pour_right_v3 warmstart.

검증된 play.py 체크포인트 로딩/AppLauncher 기계를 그대로 재사용하는 얇은
래퍼. play.py 를 서브프로세스로 실행하되 grasp env 의 warm-state export 를
hydra override 로 켠다. grasp env 가 목표 개수만큼 모으면 HDF5 를 원자적으로
저장(tmp→replace)하므로, 출력 파일이 새로 나타나는 즉시 서브프로세스를
종료한다.

사용 예:
    python3 scripts/warm_states/collect_grasp_warm_states.py \
        --num_envs 256 --target_count 2048

    # 산출물: /home/user/rl_ws/datasets/grasp_warm_v7_2.hdf5
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HDGP_ROOT = Path("/home/user/rl_ws/hdgp")
_ISAACLAB_SH = Path("/home/user/rl_ws/IsaacLab/isaaclab.sh")
_PLAY_PY = _HDGP_ROOT / "scripts/reinforcement_learning/rl_games/play.py"
_DEFAULT_CKPT = (
    _HDGP_ROOT
    / "log/rl_games/pipeline/right/5g_grasp_right_v7_2/test3/nn/5g_grasp_right-v7-2.pth"
)
_DEFAULT_OUT = Path("/home/user/rl_ws/datasets/grasp_warm_v7_2.hdf5")
_TASK = "5g_grasp_right-v7-2"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=256, help="병렬 환경 수")
    p.add_argument("--target_count", type=int, default=2048, help="수집할 성공 상태 개수")
    p.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT), help="grasp v7-2 체크포인트")
    p.add_argument("--out", type=str, default=str(_DEFAULT_OUT), help="출력 HDF5 경로")
    p.add_argument("--poll_sec", type=float, default=5.0, help="출력 파일 폴링 간격(초)")
    p.add_argument("--timeout_sec", type=float, default=3600.0, help="최대 수집 대기(초)")
    p.add_argument(
        "--keep_adr",
        action="store_true",
        help="ADR 노이즈 유지 (기본: 비활성, 깨끗한 정책 성공 분포 수집)",
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


def _build_command(args: argparse.Namespace) -> list[str]:
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
    # hydra env_cfg overrides (play.py 는 @hydra_task_config 사용)
    cmd += [
        "env.enable_warm_state_export=true",
        f"env.warm_state_export_path={Path(args.out).resolve()}",
        f"env.warm_state_target_count={args.target_count}",
    ]
    return cmd


def _file_signature(path: Path) -> tuple[bool, float]:
    if not path.is_file():
        return (False, 0.0)
    return (True, path.stat().st_mtime)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """프로세스 그룹 전체를 SIGINT→SIGTERM→SIGKILL 로 단계적 종료.

    isaaclab.sh 만 죽이면 자식 Isaac Sim(python.sh→kit python)이 고아로
    남아 GPU 를 점유한다. 그룹 전체에 신호를 보내 누수를 막는다.
    """
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
            return  # 그룹이 이미 사라짐
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue

    # 최종 스윕: shell 리더가 SIGINT 로 먼저 죽고 Isaac 손자 프로세스가
    # 그룹에 남는 케이스를 확실히 정리한다 (이번 결함의 정확한 시나리오).
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
    print("[collect_grasp_warm_states] launching:\n  " + " ".join(cmd), flush=True)

    # 자체 프로세스 그룹으로 띄운다. isaaclab.sh → python.sh → kit python 의
    # 자식 트리 전체를 한 번에 신호 보낼 수 있어야 Isaac Sim 누수가 안 생김.
    proc = subprocess.Popen(cmd, cwd=str(_HDGP_ROOT), start_new_session=True)
    deadline = time.monotonic() + args.timeout_sec
    saved = False
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(
                    f"[collect_grasp_warm_states] play.py exited early (code={ret}).",
                    flush=True,
                )
                break

            exists, mtime = _file_signature(out_path)
            if exists and (not before_exists or mtime > before_mtime):
                saved = True
                print(
                    f"[collect_grasp_warm_states] output written: {out_path} "
                    "→ stopping rollout.",
                    flush=True,
                )
                break

            if time.monotonic() > deadline:
                print(
                    f"[collect_grasp_warm_states] timeout after {args.timeout_sec}s "
                    "without a completed cache. Stopping rollout.",
                    flush=True,
                )
                break

            time.sleep(args.poll_sec)
    finally:
        _terminate_process_group(proc)

    if saved and out_path.is_file():
        print(
            f"[collect_grasp_warm_states] DONE. Warm-state cache: {out_path}",
            flush=True,
        )
        return 0

    print(
        "[collect_grasp_warm_states] FAILED: no completed warm-state cache produced.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
