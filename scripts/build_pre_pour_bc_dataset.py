#!/usr/bin/env python3
"""Build the 91D align-truncated pre_pour_bc behavior cloning dataset."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_INPUT_DIR = Path("/home/user/rl_ws/teleopration_openarm_tesollo/datasets")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "pre_pour_bc_91d_align_trunc.hdf5"


@dataclass
class DatasetBuildReport:
    kept: int = 0
    skipped: int = 0
    force_excluded: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, source: Path, demo: str, status: str, reason: str, length: int = 0) -> None:
        if status == "kept":
            self.kept += 1
        else:
            self.skipped += 1
            if reason == "low_force_or_curl":
                self.force_excluded += 1
        self.entries.append(
            {"source": source.name, "demo": demo, "status": status, "reason": reason, "length": int(length)}
        )


def _first_align_done(demo: h5py.Group) -> int | None:
    path = "obs/datagen_info/subtask_term_signals/align_done"
    if path not in demo:
        return None
    align = np.asarray(demo[path], dtype=bool)
    idx = np.flatnonzero(align)
    if idx.size == 0:
        return None
    return int(idx[0])


def _valid_demo(demo: h5py.Group) -> tuple[bool, str]:
    if "actions" not in demo or "obs/actor_obs" not in demo:
        return False, "missing_actions_or_actor_obs"
    if demo["actions"].shape[-1] != 18:
        return False, "bad_action_dim"
    if demo["obs/actor_obs"].shape[-1] != 91:
        return False, "bad_actor_obs_dim"
    if _first_align_done(demo) is None:
        return False, "missing_align_done"
    n = demo["actions"].shape[0]
    for path in ("obs/actor_obs", "obs/prev_actions", "obs/tip_force_norm"):
        if path not in demo:
            return False, f"missing_{path}"
        if demo[path].shape[0] != n:
            return False, f"length_mismatch_{path}"
    return True, "ok"


def _copy_attrs(src: h5py.Group | h5py.Dataset, dst: h5py.Group | h5py.Dataset) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_truncated(src: h5py.Group, dst: h5py.Group, stop: int) -> None:
    _copy_attrs(src, dst)
    for key, value in src.items():
        if isinstance(value, h5py.Dataset):
            data = value[: stop + 1] if value.shape and value.shape[0] >= stop + 1 else value[()]
            out = dst.create_dataset(key, data=data, compression="gzip")
            _copy_attrs(value, out)
        else:
            child = dst.create_group(key)
            _copy_truncated(value, child, stop)


def _low_force_or_curl(demo: h5py.Group, stop: int, force_threshold: float, curl_threshold: float) -> bool:
    tip_force = np.asarray(demo["obs/tip_force_norm"][: stop + 1])
    action_curl = np.asarray(demo["actions"][: stop + 1, 6:11])
    return float(np.max(tip_force)) < force_threshold or float(np.max(np.abs(action_curl))) < curl_threshold


def build_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    force_threshold: float = 0.05,
    curl_threshold: float = 0.05,
    include_low_force: bool = False,
) -> DatasetBuildReport:
    report = DatasetBuildReport()
    files = sorted(input_dir.glob("*.hdf5"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as out:
        data_out = out.create_group("data")
        out.attrs["source_dir"] = str(input_dir)
        out.attrs["obs_dim"] = 91
        out.attrs["action_dim"] = 18
        out.attrs["truncate_signal"] = "align_done"
        demo_idx = 0

        for path in files:
            if path.resolve() == output_path.resolve():
                continue
            with h5py.File(path, "r") as src:
                for demo_name, demo in src.get("data", {}).items():
                    ok, reason = _valid_demo(demo)
                    if not ok:
                        report.add(source=path, demo=demo_name, status="skipped", reason=reason)
                        continue
                    stop = _first_align_done(demo)
                    assert stop is not None
                    if not include_low_force and _low_force_or_curl(demo, stop, force_threshold, curl_threshold):
                        report.add(source=path, demo=demo_name, status="skipped", reason="low_force_or_curl", length=stop + 1)
                        continue
                    dst = data_out.create_group(f"demo_{demo_idx}")
                    _copy_truncated(demo, dst, stop)
                    dst.attrs["source_file"] = path.name
                    dst.attrs["source_demo"] = demo_name
                    dst.attrs["align_done_index"] = stop
                    report.add(source=path, demo=demo_name, status="kept", reason="ok", length=stop + 1)
                    demo_idx += 1

        out.attrs["num_demos"] = report.kept
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-threshold", type=float, default=0.05)
    parser.add_argument("--curl-threshold", type=float, default=0.05)
    parser.add_argument("--include-low-force", action="store_true")
    args = parser.parse_args()
    report = build_dataset(
        args.input_dir,
        args.output,
        force_threshold=args.force_threshold,
        curl_threshold=args.curl_threshold,
        include_low_force=args.include_low_force,
    )
    print(f"kept={report.kept} skipped={report.skipped} force_excluded={report.force_excluded}")
    for entry in report.entries:
        print("{source}:{demo} {status} {reason} length={length}".format(**entry))


if __name__ == "__main__":
    main()
