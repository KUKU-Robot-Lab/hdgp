#!/usr/bin/env python3
"""
URDF geometry spec reporter.

Examples:
  python3 urdf_spec_report.py /home/user/Desktop/dd.urdf
  python3 urdf_spec_report.py /home/user/Desktop/dd.urdf --from visual
  python3 urdf_spec_report.py /home/user/Desktop/dd.urdf --top-frac 0.02
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GeometryRecord:
    link_name: str
    source_kind: str  # visual or collision
    geom_type: str  # mesh/box/cylinder/sphere
    mesh_file: Optional[pathlib.Path] = None
    scale: Optional[np.ndarray] = None
    origin_xyz: Optional[np.ndarray] = None
    box_size: Optional[np.ndarray] = None
    cylinder_radius: Optional[float] = None
    cylinder_length: Optional[float] = None
    sphere_radius: Optional[float] = None


def _parse_vec(text: Optional[str], n: int, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    vals = [float(x) for x in text.replace(",", " ").split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)} from '{text}'")
    return np.array(vals, dtype=float)


def _resolve_mesh_path(filename: str, urdf_path: pathlib.Path) -> pathlib.Path:
    if filename.startswith("file://"):
        return pathlib.Path(filename[len("file://") :]).expanduser().resolve()
    p = pathlib.Path(filename).expanduser()
    if not p.is_absolute():
        p = (urdf_path.parent / p).resolve()
    return p


def _load_obj_vertices(path: pathlib.Path) -> np.ndarray:
    verts: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError(f"No vertices found in OBJ: {path}")
    return np.array(verts, dtype=float)


def _bbox_from_vertices(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    ext = maxs - mins
    return mins, maxs, ext


def _top_diameter_estimates(v: np.ndarray, top_frac: float) -> dict:
    mins, maxs, _ = _bbox_from_vertices(v)
    zmin, zmax = mins[2], maxs[2]
    h = zmax - zmin
    if h <= 0:
        return {}
    s = v[v[:, 2] >= (zmax - h * top_frac)]
    if len(s) == 0:
        return {}
    cx, cy = np.median(s[:, 0]), np.median(s[:, 1])
    r = np.sqrt((s[:, 0] - cx) ** 2 + (s[:, 1] - cy) ** 2)
    out = {
        "sample_count": int(len(s)),
        "center_xy": (float(cx), float(cy)),
        "outer_diam_p95": float(2.0 * np.percentile(r, 95)),
        "outer_diam_p99": float(2.0 * np.percentile(r, 99)),
    }

    rs = np.sort(r)
    if len(rs) > 4:
        gaps = np.diff(rs)
        split_i = int(np.argmax(gaps))
        split_thr = float((rs[split_i] + rs[split_i + 1]) * 0.5)
        inner = rs[rs <= split_thr]
        outer = rs[rs > split_thr]
        # Only accept the split when both groups are meaningful.
        if len(inner) >= 0.2 * len(rs) and len(outer) >= 0.2 * len(rs):
            out["inner_diam_median"] = float(2.0 * np.median(inner))
            out["outer_diam_median"] = float(2.0 * np.median(outer))
            out["split_threshold_radius"] = split_thr
    return out


def parse_urdf(urdf_path: pathlib.Path, source_kind: str) -> List[GeometryRecord]:
    root = ET.parse(urdf_path).getroot()
    out: List[GeometryRecord] = []
    for link in root.findall("link"):
        link_name = link.get("name", "<unnamed_link>")
        for src in link.findall(source_kind):
            origin_el = src.find("origin")
            origin_xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, 3, [0, 0, 0])
            geom = src.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if not filename:
                    continue
                mesh_path = _resolve_mesh_path(filename, urdf_path)
                scale = _parse_vec(mesh.get("scale"), 3, [1, 1, 1])
                out.append(
                    GeometryRecord(
                        link_name=link_name,
                        source_kind=source_kind,
                        geom_type="mesh",
                        mesh_file=mesh_path,
                        scale=scale,
                        origin_xyz=origin_xyz,
                    )
                )
                continue

            box = geom.find("box")
            if box is not None:
                out.append(
                    GeometryRecord(
                        link_name=link_name,
                        source_kind=source_kind,
                        geom_type="box",
                        box_size=_parse_vec(box.get("size"), 3, [0, 0, 0]),
                        origin_xyz=origin_xyz,
                    )
                )
                continue

            cyl = geom.find("cylinder")
            if cyl is not None:
                out.append(
                    GeometryRecord(
                        link_name=link_name,
                        source_kind=source_kind,
                        geom_type="cylinder",
                        cylinder_radius=float(cyl.get("radius", "0")),
                        cylinder_length=float(cyl.get("length", "0")),
                        origin_xyz=origin_xyz,
                    )
                )
                continue

            sph = geom.find("sphere")
            if sph is not None:
                out.append(
                    GeometryRecord(
                        link_name=link_name,
                        source_kind=source_kind,
                        geom_type="sphere",
                        sphere_radius=float(sph.get("radius", "0")),
                        origin_xyz=origin_xyz,
                    )
                )
    return out


def _fmt_m(v: float) -> str:
    return f"{v:.6f} m ({v * 100.0:.2f} cm)"


def _print_mesh_report(rec: GeometryRecord, top_frac: float) -> None:
    assert rec.mesh_file is not None and rec.scale is not None
    print(f"[{rec.source_kind}] link='{rec.link_name}' type=mesh")
    print(f"  mesh_file: {rec.mesh_file}")
    print(f"  scale: {rec.scale.tolist()}")
    print(f"  origin_xyz: {rec.origin_xyz.tolist() if rec.origin_xyz is not None else [0, 0, 0]}")
    if not rec.mesh_file.exists():
        print("  ERROR: mesh file does not exist")
        return
    if rec.mesh_file.suffix.lower() != ".obj":
        print("  NOTE: currently OBJ only. Convert mesh to OBJ to analyze.")
        return

    v = _load_obj_vertices(rec.mesh_file)
    v_scaled = v * rec.scale.reshape(1, 3)
    mins, maxs, ext = _bbox_from_vertices(v_scaled)

    print(f"  vertices: {len(v_scaled)}")
    print(f"  bbox_min_xyz: {mins.tolist()}")
    print(f"  bbox_max_xyz: {maxs.tolist()}")
    print(f"  bbox_extents_xyz: {_fmt_m(ext[0])}, {_fmt_m(ext[1])}, {_fmt_m(ext[2])}")
    print(f"  max_xy_span (handle may be included): {_fmt_m(max(ext[0], ext[1]))}")

    top = _top_diameter_estimates(v_scaled, top_frac=top_frac)
    if top:
        print(f"  top_slice_frac: {top_frac}")
        print(f"  top_slice_vertices: {top['sample_count']}")
        print(f"  top_center_xy: {top['center_xy']}")
        print(f"  top_outer_diam_p95: {_fmt_m(top['outer_diam_p95'])}")
        print(f"  top_outer_diam_p99: {_fmt_m(top['outer_diam_p99'])}")
        if "inner_diam_median" in top:
            print(f"  top_inner_diam_est: {_fmt_m(top['inner_diam_median'])}")
            print(f"  top_outer_diam_est: {_fmt_m(top['outer_diam_median'])}")


def _print_primitive_report(rec: GeometryRecord) -> None:
    print(f"[{rec.source_kind}] link='{rec.link_name}' type={rec.geom_type}")
    print(f"  origin_xyz: {rec.origin_xyz.tolist() if rec.origin_xyz is not None else [0, 0, 0]}")
    if rec.geom_type == "box":
        assert rec.box_size is not None
        sx, sy, sz = rec.box_size.tolist()
        print(f"  box_size_xyz: {_fmt_m(sx)}, {_fmt_m(sy)}, {_fmt_m(sz)}")
    elif rec.geom_type == "cylinder":
        r = float(rec.cylinder_radius or 0.0)
        l = float(rec.cylinder_length or 0.0)
        print(f"  radius: {_fmt_m(r)}")
        print(f"  diameter: {_fmt_m(2.0 * r)}")
        print(f"  length: {_fmt_m(l)}")
    elif rec.geom_type == "sphere":
        r = float(rec.sphere_radius or 0.0)
        print(f"  radius: {_fmt_m(r)}")
        print(f"  diameter: {_fmt_m(2.0 * r)}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Report geometric specs from URDF")
    ap.add_argument("urdf", type=pathlib.Path, help="Path to URDF file")
    ap.add_argument(
        "--from",
        dest="source_kind",
        default="collision",
        choices=["collision", "visual"],
        help="Use geometry from collision or visual elements",
    )
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.02,
        help="Top slice ratio for mouth diameter estimates on mesh (default: 0.02)",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    urdf_path: pathlib.Path = args.urdf.expanduser().resolve()
    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        return 1
    if args.top_frac <= 0 or args.top_frac >= 1:
        print("--top-frac must be in (0, 1)")
        return 1

    records = parse_urdf(urdf_path, source_kind=args.source_kind)
    if not records:
        print(f"No '{args.source_kind}' geometry found in {urdf_path}")
        return 1

    print(f"URDF: {urdf_path}")
    print(f"geometry_source: {args.source_kind}")
    print(f"num_geometry_items: {len(records)}")
    print("")

    for i, rec in enumerate(records, start=1):
        print(f"=== Geometry {i} ===")
        if rec.geom_type == "mesh":
            _print_mesh_report(rec, top_frac=args.top_frac)
        else:
            _print_primitive_report(rec)
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
