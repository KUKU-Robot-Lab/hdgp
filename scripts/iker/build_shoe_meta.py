#!/usr/bin/env python3
"""Build assets/iker_shoe/shoe_meta.json from the IKER shoe meshes (design spec §4). No Isaac needed.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/build_shoe_meta.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from openarm.agnostic.modules.iker import keypoints, run_files
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz, rpy_xyz_to_matrix
from openarm.agnostic.tasks.iker_shoe import layout

DEFAULT_IKER_DIR = Path("/home/user/rl_ws/repo/skill_gen/IKER/envs/shoe_place")
SOURCES = {layout.MOVING_SHOE: "object_0", layout.OTHER_SHOE: "object_1"}
# The legacy task spawns object_pose.json 1.0 m higher (shoe_place.py:650 `sampled_obj_state[:, 2] += 1`);
# target.json is already in that lifted frame.
LEGACY_Z_LIFT = 1.0
IDENTITY_ORIGIN = '<origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>'
DECIMALS = 6


def parse_obj_vertices(path: Path) -> np.ndarray:
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    if len(rows) < 4:
        raise ValueError(f"{path}: fewer than 4 vertices")
    return np.asarray(rows, dtype=float)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_urdf_frame(urdf: Path, mesh_name: str) -> None:
    """Keypoint offsets are measured in the mesh frame, so the mesh must sit at the link origin."""
    text = urdf.read_text()
    if f'filename="{mesh_name}"' not in text or text.count(IDENTITY_ORIGIN) != 2:
        raise ValueError(f"{urdf}: expected visual and collision {mesh_name} at the link origin")


def object_meta(iker_dir: Path, source: str, legacy_rpy) -> dict:
    mesh = iker_dir / f"{source}_mesh.obj"
    check_urdf_frame(iker_dir / f"{source}_object.urdf", mesh.name)
    vertices = parse_obj_vertices(mesh)
    hull = vertices[ConvexHull(vertices).vertices]
    rest = keypoints.snap_rotation(rpy_xyz_to_matrix(legacy_rpy))
    axes = keypoints.horizontal_axes(rest, vertices)
    return {
        "source": source,
        "rest_rotation": rest.tolist(),
        "rest_quat_wxyz": matrix_to_quat_wxyz(rest).round(DECIMALS).tolist(),
        "horizontal_axes": list(axes),
        "keypoint_offsets": keypoints.extremity_offsets(vertices, axes).round(DECIMALS).tolist(),
        "rest_height": round(-float((hull @ rest.T)[:, 2].min()), DECIMALS),
        "hull_local": hull.round(DECIMALS).tolist(),
    }


def build(iker_dir: Path) -> dict:
    poses = json.loads((iker_dir / "object_pose.json").read_text())
    target = json.loads((iker_dir / "target.json").read_text())["target_position"]
    inputs = ["object_pose.json", "target.json"]
    inputs += [f"{source}_{kind}" for source in SOURCES.values() for kind in ("mesh.obj", "object.urdf")]
    return {
        "schema": run_files.SCHEMA_VERSION,
        "source": {"iker_dir": str(iker_dir), "sha256": {name: sha256(iker_dir / name) for name in inputs}},
        "legacy": {"z_lift": LEGACY_Z_LIFT, "object_pose": poses, "target_position": target},
        "objects": {name: object_meta(iker_dir, source, poses[source]["rpy"]) for name, source in SOURCES.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the IKER shoe keypoint meta file.")
    parser.add_argument("--iker-dir", type=Path, default=DEFAULT_IKER_DIR)
    parser.add_argument("--out", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    required = [args.iker_dir / name for name in ("object_pose.json", "target.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(f"missing IKER inputs: {missing}", file=sys.stderr)
        return 2
    meta = build(args.iker_dir)
    run_files.write_json(args.out, meta)
    for name, obj in meta["objects"].items():
        print(f"{name}: offsets {obj['keypoint_offsets']} rest_height {obj['rest_height']} hull {len(obj['hull_local'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
