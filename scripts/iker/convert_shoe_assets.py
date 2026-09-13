"""Convert the IKER shoe URDFs to USD under assets/iker_shoe/<shoe>/ (design spec §4).

The URDF and mesh hashes must match assets/iker_shoe/shoe_meta.json, so the USD and the keypoint
offsets always come from the same mesh.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/convert_shoe_assets.py --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_IKER_DIR = Path("/home/user/rl_ws/repo/skill_gen/IKER/envs/shoe_place")

parser = argparse.ArgumentParser(description="Convert the IKER shoe URDFs to USD.")
parser.add_argument("--iker-dir", type=Path, default=DEFAULT_IKER_DIR)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("CONVERT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, scene_cfg  # noqa: E402

SHOE_DENSITY = 100.0  # IKER URDF <density value="100.0"/>


def _check_hash(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{path} changed since shoe_meta.json was built ({actual} != {expected})")


meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
for name, obj in meta["objects"].items():
    urdf = args.iker_dir / f"{obj['source']}_object.urdf"
    mesh = args.iker_dir / f"{obj['source']}_mesh.obj"
    for path in (urdf, mesh):
        _check_hash(path, meta["source"]["sha256"][path.name])
    target = scene_cfg.shoe_usd_path(name)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(target.parent),
            usd_file_name=target.name,
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            joint_drive=None,
            link_density=SHOE_DENSITY,
            collider_type="convex_hull",
        )
    )
    print(f"CONVERT {name}: {urdf.name} -> {converter.usd_path}", flush=True)
print("CONVERT DONE", flush=True)
os._exit(0)
