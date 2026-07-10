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

"""grasp_v1 (rh56f1 / tesollo) 성공 상태 → pour warmstart HDF5 수집기.

각 로봇 환경의 grasp_v1 최신 LSTM 체크포인트를 **전용 수집 스크립트**
(collect_warm_states.py)로 rollout 하여 grasp 성공(종료) 시점 상태를
HDF5(``warm_states`` 그룹)로 export 한다. 산출물은 해당 로봇 pour 태스크의
``warm_state_paths`` (PourWarmStateBank) 입력으로 그대로 쓰인다.

수집은 render/eval 용 play.py 가 아니라 전용 진입점을 쓴다 — play.py 의
logged-cfg 복원이 export 설정을 덮어써 수집을 무력화한 사고(2026-06-30) 이후
관심사를 분리했다. export 는 전용 스크립트의 1급 CLI 인자로 강제된다:
    --warm_export_path <out>  --warm_target_count <N>

검증된 collect_grasp_warm_states.py 의 서브프로세스 폴링/프로세스그룹 정리
로직(_file_signature, _terminate_process_group)을 재사용하는 얇은 래퍼다.

사용 예:
    # rh56f1 (기본 체크포인트 lstm_test1, 출력 hdgp/data/grasp_warm_rh56f1.hdf5)
    python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot rh56f1

    # tesollo (기본 체크포인트 lstm_test1, 출력 hdgp/data/grasp_warm_tesollo.hdf5)
    python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo

    # 체크포인트/출력 직접 지정
    python3 scripts/warm_states/collect_grasp_v1_warm_states.py \\
        --robot tesollo --checkpoint /abs/path.pth --out /abs/out.hdf5 \\
        --num_envs 256 --target_count 2048
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 검증된 서브프로세스 폴링/프로세스그룹 종료 로직을 그대로 재사용한다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_grasp_warm_states import (  # noqa: E402
    _file_signature,
    _terminate_process_group,
)

# __file__ 기준 유도 → 로컬(/home/user)·server(/home/oem) 어디서든 동작.
# .../hdgp/scripts/warm_states/collect_grasp_v1_warm_states.py → parents[2]=hdgp, parents[3]=rl_ws
_HDGP_ROOT = Path(__file__).resolve().parents[2]
_RL_WS = _HDGP_ROOT.parent
_ISAACLAB_SH = _RL_WS / "IsaacLab" / "isaaclab.sh"
_COLLECT_PY = _HDGP_ROOT / "scripts/reinforcement_learning/rl_games/collect_warm_states.py"
_DATA_DIR = _HDGP_ROOT / "data"
_LOG_ROOT = _HDGP_ROOT / "log/rl_games"


@dataclass(frozen=True)
class RobotPreset:
    """로봇별 play task / 기본 체크포인트 / 기본 출력 경로."""

    task: str
    default_checkpoint: Path
    default_out: Path


# 최신 test 기준(사용자 확정):
#   rh56f1 → grasp-v1/lstm_test1 (ep_2000)
#   tesollo → grasp-v1/lstm_test1 (ep_3000, stiffness 수정 수렴본)
_PRESETS: dict[str, RobotPreset] = {
    "rh56f1": RobotPreset(
        # MLP 전환: envelope 파이프라인은 MLP(open-rh56f1_r_grasp_v1)로 학습됨(test6).
        task="open-rh56f1_r_grasp_v1-play",
        default_checkpoint=(
            _LOG_ROOT
            / "open-rh56f1/right/grasp-v1/test6/nn"
            / "open-rh56f1_r_grasp_v1.pth"
        ),
        default_out=_DATA_DIR / "grasp_warm_rh56f1.hdf5",
    ),
    "tesollo": RobotPreset(
        task="open-tesol_r_grasp_v1-play-lstm",
        default_checkpoint=(
            _LOG_ROOT
            / "open-tesol/right/grasp-v1/lstm_test1/nn"
            / "last_open-tesol_r_grasp_v1-lstm_ep_3000_rew_9431.901.pth"
        ),
        default_out=_DATA_DIR / "grasp_warm_tesollo.hdf5",
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--robot",
        required=True,
        choices=sorted(_PRESETS.keys()),
        help="대상 로봇 환경 (task/기본 체크포인트/기본 출력 프리셋 선택)",
    )
    p.add_argument("--num_envs", type=int, default=256, help="병렬 환경 수")
    p.add_argument("--target_count", type=int, default=2048, help="수집할 성공 상태 개수")
    p.add_argument("--checkpoint", type=str, default=None, help="grasp_v1 LSTM 체크포인트 (생략 시 프리셋 기본값)")
    p.add_argument("--out", type=str, default=None, help="출력 HDF5 경로 (생략 시 hdgp/data/grasp_warm_<robot>.hdf5)")
    p.add_argument("--poll_sec", type=float, default=5.0, help="출력 파일 폴링 간격(초)")
    p.add_argument("--timeout_sec", type=float, default=3600.0, help="최대 수집 대기(초)")
    p.add_argument(
        "--keep_adr",
        action="store_true",
        help="ADR 노이즈 유지 (기본: 비활성, 깨끗한 정책 성공 분포 수집)",
    )
    return p.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[RobotPreset, Path, Path]:
    preset = _PRESETS[args.robot]
    checkpoint = Path(args.checkpoint) if args.checkpoint else preset.default_checkpoint
    out = Path(args.out) if args.out else preset.default_out
    return preset, checkpoint, out


def _validate(checkpoint: Path, target_count: int) -> None:
    for path, label in ((_ISAACLAB_SH, "isaaclab.sh"), (_COLLECT_PY, "collect_warm_states.py")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if target_count <= 0:
        raise ValueError(f"--target_count must be positive, got {target_count}")


def _build_command(
    *, task: str, checkpoint: Path, out: Path, num_envs: int, target_count: int, keep_adr: bool
) -> list[str]:
    cmd = [
        str(_ISAACLAB_SH),
        "-p",
        str(_COLLECT_PY),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--num_envs",
        str(num_envs),
        "--headless",
        # warm-state export 는 전용 스크립트의 1급 CLI 인자(복원이 덮어쓰지 못함).
        "--warm_export_path",
        str(out.resolve()),
        "--warm_target_count",
        str(target_count),
    ]
    if not keep_adr:
        cmd.append("--disable_adr")
    return cmd


def main() -> int:
    args = _parse_args()
    preset, checkpoint, out = _resolve(args)
    _validate(checkpoint, args.target_count)

    out_path = out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    before_exists, before_mtime = _file_signature(out_path)

    cmd = _build_command(
        task=preset.task,
        checkpoint=checkpoint,
        out=out_path,
        num_envs=args.num_envs,
        target_count=args.target_count,
        keep_adr=args.keep_adr,
    )
    print(
        f"[collect_grasp_v1_warm_states] robot={args.robot} task={preset.task}\n"
        f"  checkpoint={checkpoint}\n  out={out_path}\n  launching:\n  " + " ".join(cmd),
        flush=True,
    )

    # 자체 프로세스 그룹으로 띄워 isaaclab.sh→python.sh→kit python 자식 트리를
    # 한 번에 신호 보낼 수 있게 한다 (Isaac Sim GPU 누수 방지).
    proc = subprocess.Popen(cmd, cwd=str(_HDGP_ROOT), start_new_session=True)
    deadline = time.monotonic() + args.timeout_sec
    saved = False
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(
                    f"[collect_grasp_v1_warm_states] play.py exited early (code={ret}).",
                    flush=True,
                )
                break

            exists, mtime = _file_signature(out_path)
            if exists and (not before_exists or mtime > before_mtime):
                saved = True
                print(
                    f"[collect_grasp_v1_warm_states] output written: {out_path} "
                    "→ stopping rollout.",
                    flush=True,
                )
                break

            if time.monotonic() > deadline:
                print(
                    f"[collect_grasp_v1_warm_states] timeout after {args.timeout_sec}s "
                    "without a completed cache. Stopping rollout.",
                    flush=True,
                )
                break

            time.sleep(args.poll_sec)
    finally:
        _terminate_process_group(proc)

    if saved and out_path.is_file():
        print(
            f"[collect_grasp_v1_warm_states] DONE. Warm-state cache: {out_path}",
            flush=True,
        )
        return 0

    print(
        "[collect_grasp_v1_warm_states] FAILED: no completed warm-state cache produced.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
