#!/usr/bin/env python3
"""Build object bank JSON/YAML + URDF wrappers from demo datasets.

Sources:
- DexYCB: /.../dex-ycb/models/*/*.stl
- DexGraspNet: /.../DexGraspNet/data/meshdata/*/coacd/decomposed.obj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _sanitize(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _write_urdf(urdf_path: Path, mesh_path: Path, obj_id: str, mass: float, scale: tuple[float, float, float]) -> None:
    sx, sy, sz = scale
    content = f"""<?xml version="1.0" ?>
<robot name="{obj_id}">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(content)


def main() -> None:
    p = argparse.ArgumentParser(description="Build 5g_grasp_v1 object bank from demo assets.")
    p.add_argument("--dex-ycb-dir", type=Path, default=Path("/home/user/rl_ws/demo/dex-ycb"))
    p.add_argument("--dexgraspnet-dir", type=Path, default=Path("/home/user/rl_ws/demo/DexGraspNet"))
    p.add_argument("--out-dir", type=Path, required=True, help=".../5g_grasp_v1/assets")
    p.add_argument("--max-dex-ycb", type=int, default=8)
    p.add_argument("--max-dexgraspnet", type=int, default=8)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    urdf_dir = out_dir / "generated_urdf"
    urdf_dir.mkdir(parents=True, exist_ok=True)

    objects: list[dict] = []

    # DexYCB STLs
    dxycb_models = args.dex_ycb_dir / "models"
    stls = sorted(dxycb_models.glob("*/*.stl"))[: args.max_dex_ycb]
    for stl in stls:
        obj_id = _sanitize(f"dxycb_{stl.parent.name}")
        rel_urdf = f"generated_urdf/{obj_id}.urdf"
        _write_urdf(urdf_dir / f"{obj_id}.urdf", stl.resolve(), obj_id, mass=0.35, scale=(1.0, 1.0, 1.0))
        objects.append(
            {
                "id": obj_id,
                "source": "dex-ycb",
                "mesh_path": str(stl.resolve()),
                "urdf_path": rel_urdf,
                "scale": [1.0, 1.0, 1.0],
                "mass": 0.35,
            }
        )

    # DexGraspNet decomposed meshes
    dgn_mesh = args.dexgraspnet_dir / "data" / "meshdata"
    decomp = sorted(dgn_mesh.glob("*/coacd/decomposed.obj"))[: args.max_dexgraspnet]
    for obj in decomp:
        obj_id = _sanitize(f"dexgn_{obj.parent.parent.name}")
        rel_urdf = f"generated_urdf/{obj_id}.urdf"
        _write_urdf(urdf_dir / f"{obj_id}.urdf", obj.resolve(), obj_id, mass=0.30, scale=(1.0, 1.0, 1.0))
        objects.append(
            {
                "id": obj_id,
                "source": "dexgraspnet",
                "mesh_path": str(obj.resolve()),
                "urdf_path": rel_urdf,
                "scale": [1.0, 1.0, 1.0],
                "mass": 0.30,
            }
        )

    bank = {"objects": objects}
    (out_dir / "object_bank.json").write_text(json.dumps(bank, indent=2))
    (out_dir / "object_bank.yaml").write_text(yaml.safe_dump(bank, sort_keys=False))

    print(f"saved: {out_dir / 'object_bank.json'}")
    print(f"saved: {out_dir / 'object_bank.yaml'}")
    print(f"objects={len(objects)}")


if __name__ == "__main__":
    main()

