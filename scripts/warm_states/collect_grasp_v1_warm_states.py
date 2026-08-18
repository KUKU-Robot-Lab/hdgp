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

    # [both/pour_v1] 양손 파지 pour 용 — 좌/우 순차 수집.
    #   source(우팔)는 비드를 채운 상태로, receiver(좌팔)는 빈 컵으로 수집한다.
    python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_right --with_beads
    python3 scripts/warm_states/collect_grasp_v1_warm_states.py --robot tesollo_left

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
    # ★[both/pour_v1] 양손 파지 pour 용 좌/우 분리 프리셋.
    #   pour_v1 은 왼팔(receiver 컵)·오른팔(source 컵) **두 뱅크**를 각각 로드하므로
    #   출력 파일명을 좌우로 구분한다. 기존 `tesollo` 프리셋은 구 단일뱅크 호환용으로 남긴다.
    #   ⚠ 체크포인트는 신 USD(bi_s_rl) 로 학습한 런을 반드시 지정할 것 —
    #     구 USD 체크포인트로 수집하면 pour 에서 조용히 어긋난 초기상태가 된다.
    "tesollo_right": RobotPreset(
        task="open-tesol_r_grasp_v1-play-lstm",
        default_checkpoint=(
            _LOG_ROOT
            / "open-tesol/right/grasp-v1/lstm_test1/nn"
            / "last_open-tesol_r_grasp_v1-lstm_ep_3000_rew_9431.901.pth"
        ),
        default_out=_DATA_DIR / "grasp_warm_tesollo_right.hdf5",
    ),
    "tesollo_left": RobotPreset(
        task="open-tesol_l_grasp_v1-play-lstm",
        default_checkpoint=(
            _LOG_ROOT
            / "open-tesol/left/grasp-v1/lstm_test1/nn"
            / "last_open-tesol_l_grasp_v1-lstm.pth"
        ),
        default_out=_DATA_DIR / "grasp_warm_tesollo_left.hdf5",
    ),
    # ★08.18 sim2real(a1) 트랙: right/grasp_sensor(openarm_tesollo_sensor_rl, DG-5F)
    #   → right/pour_v1(→pour_sensor) 단일 뱅크. 출력 파일명을 bi_s 계열
    #   (grasp_warm_tesollo*.hdf5)과 반드시 분리한다 — 두 자산은 텐서 차원이 같아
    #   파일이 섞이면 로더가 조용히 성공한다(뱅크 meta/robot_usd 가드가 최후 방어선).
    "tesollo_sensor": RobotPreset(
        task="open-tesol_r_grasp_sensor-play-lstm",
        default_checkpoint=(
            _LOG_ROOT
            / "open-tesol/right/grasp-sensor/lstm_test1/nn"
            / "last_open-tesol_r_grasp_sensor-lstm.pth"
        ),
        default_out=_DATA_DIR / "grasp_warm_tesollo_sensor.hdf5",
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
    p.add_argument(
        "--latest",
        action="store_true",
        help=(
            "프리셋 기본값 대신 해당 로봇의 grasp-v1 로그에서 **가장 최근 수정된** "
            "ep 체크포인트를 고른다. 프리셋 경로는 학습 런이 바뀔 때마다 낡으므로"
            "(2026-08-18 lstm_test1→lstm_test2) 재수집 자동화에는 이쪽을 쓴다. "
            "무엇을 골랐는지 항상 출력하고, 뱅크 attrs 에도 기록된다."
        ),
    )
    p.add_argument("--out", type=str, default=None, help="출력 HDF5 경로 (생략 시 hdgp/data/grasp_warm_<robot>.hdf5)")
    p.add_argument("--poll_sec", type=float, default=5.0, help="출력 파일 폴링 간격(초)")
    p.add_argument("--timeout_sec", type=float, default=3600.0, help="최대 수집 대기(초)")
    p.add_argument(
        "--keep_adr",
        action="store_true",
        help="ADR 노이즈 유지 (기본: 비활성, 깨끗한 정책 성공 분포 수집)",
    )
    p.add_argument(
        "--with_beads",
        action="store_true",
        help=(
            "source 컵에 비드를 채운 상태로 파지를 형성/수집한다 (both/pour_v1 용). "
            "pour 는 손을 warm 자세로 동결하므로, 빈 컵 파지에 나중에 비드를 넣으면 "
            "하중을 흡수하지 못해 컵을 놓칠 수 있다. receiver(왼팔)는 빈 컵이라 붙이지 않는다."
        ),
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=None,
        help="자식 수집 프로세스에 그대로 전달할 hydra override (예: env.collect_sdf_cup_assets=true)",
    )
    return p.parse_args()


def _latest_checkpoint(preset: RobotPreset) -> Path:
    """프리셋 체크포인트가 속한 grasp-v1 트리에서 최신 ep 체크포인트를 고른다.

    프리셋의 `default_checkpoint` 는 특정 런 폴더(lstm_test1 등)를 하드코딩하므로
    재학습으로 런이 늘어나면 조용히 낡는다(실측: 2026-08-18 재학습은 lstm_test2 에
    저장됐는데 프리셋은 lstm_test1 의 삭제된 파일을 가리키고 있었다).
    여기서는 `<...>/grasp-v1/*/nn/*ep_*.pth` 를 모아 mtime 최신을 고른다.
    """
    # .../grasp-v1/<run>/nn/<file>.pth → parents[2] = grasp-v1
    grasp_root = preset.default_checkpoint.parents[2]
    # ★`rew__1234_` 처럼 밑줄이 겹친 변형은 제외한다. 같은 ep 에 정상 이름과 이 변형이
    #   함께 있는 런이 실재하고(lstm_test1), **파일 크기가 서로 다르다**(89743036 vs
    #   89743270). mtime 이 같은 초에 찍히면 어느 쪽이 뽑힐지 불안정해진다.
    cands = sorted(
        (q for q in grasp_root.glob("*/nn/*ep_*.pth") if "__" not in q.name),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(
            f"--latest: {grasp_root} 아래에서 ep 체크포인트를 찾지 못했다.\n"
            "  학습 로그가 이 머신에 없으면 --checkpoint 로 경로를 직접 줄 것."
        )
    return cands[0]


def _resolve(args: argparse.Namespace) -> tuple[RobotPreset, Path, Path]:
    preset = _PRESETS[args.robot]
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif getattr(args, "latest", False):
        checkpoint = _latest_checkpoint(preset)
        print(f"[collect_grasp_v1_warm_states] --latest → {checkpoint}", flush=True)
    else:
        checkpoint = preset.default_checkpoint
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
    *, task: str, checkpoint: Path, out: Path, num_envs: int, target_count: int, keep_adr: bool,
    with_beads: bool = False,
    extra: list[str] | None = None,
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
    if with_beads:
        # ★[both/pour_v1] source 컵을 비드로 채운 상태에서 파지를 형성/수집한다.
        #   hydra override 로 넣으므로 학습 기본값(False)은 건드리지 않는다.
        cmd.append("env.collect_with_beads=true")
    if extra:
        cmd.extend(extra)  # hydra env.* override (예: env.collect_sdf_cup_assets=true)
    return cmd


def _stamp_provenance(out_path: Path, checkpoint: Path, task: str, args) -> None:
    """수집이 끝난 HDF5 에 **어떤 체크포인트로 만들었는지**를 기록한다.

    왜 필요한가 (2026-08-18): 뱅크에는 상태만 있고 출처가 없었다. 그래서
      · 파일 크기가 구 뱅크와 같아(고정 shape) 재수집이 됐는지 눈으로 구분할 수 없었고
      · 임시 체크포인트(ep_8000)로 만든 뱅크로 잰 게이트 수치를 최종 정책의 것으로
        오해할 여지가 있었다.
    상태 데이터는 건드리지 않고 파일 attrs 만 덧붙인다(수집기 본체 수정 없이 안전).
    실패해도 수집 자체는 성공이므로 경고만 남긴다.
    """
    try:
        import hashlib

        import h5py
    except ImportError as exc:  # h5py 없는 환경에서도 수집은 성공으로 둔다
        print(f"[collect_grasp_v1_warm_states][WARN] provenance 기록 생략: {exc}", flush=True)
        return

    try:
        h = hashlib.sha256()
        with open(checkpoint, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        with h5py.File(out_path, "a") as f:
            f.attrs["prov/checkpoint"] = str(checkpoint.resolve())
            f.attrs["prov/checkpoint_sha256"] = h.hexdigest()
            f.attrs["prov/task"] = task
            f.attrs["prov/robot"] = args.robot
            f.attrs["prov/with_beads"] = bool(args.with_beads)
            f.attrs["prov/keep_adr"] = bool(args.keep_adr)
            f.attrs["prov/target_count"] = int(args.target_count)
        print(
            f"[collect_grasp_v1_warm_states] provenance 기록: "
            f"ckpt={checkpoint.name} sha256={h.hexdigest()[:12]}…",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — 기록 실패가 수집을 무효로 만들지는 않는다
        print(f"[collect_grasp_v1_warm_states][WARN] provenance 기록 실패: {exc}", flush=True)


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
        with_beads=args.with_beads,
        extra=args.extra,
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
                # 정상 종료 race 방지(08.15 실증): 자식이 캐시를 쓰고 같은 폴링 윈도우
                # 안에 종료하면 파일 확인 전에 이 분기로 빠져 FAILED 오판정. 종료 후
                # 최종 파일 확인으로 완료 여부를 판정한다.
                exists, mtime = _file_signature(out_path)
                if ret == 0 and exists and (not before_exists or mtime > before_mtime):
                    saved = True
                print(
                    f"[collect_grasp_v1_warm_states] play.py exited early "
                    f"(code={ret}, cache_saved={saved}).",
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
        _stamp_provenance(out_path, checkpoint, preset.task, args)
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
