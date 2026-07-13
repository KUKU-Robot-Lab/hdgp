#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grasp_v2 물체군의 half-extent(bbox) 산출 → assets/object_bbox.json.

env reset 이 "이 물체가 납작한가"(→ top-down 접근 분기)를 판정하려면 물체별
치수가 필요하다. Isaac/USD 를 띄우지 않고 원본 메시에서 직접 계산한다.
  - visdex_objects: urdf/<name>/visual_model.obj 의 정점 bbox
  - primitives:     urdf/<name>/<name>.stl (binary STL) 의 정점 bbox

출력 JSON: {"<name>": [hx, hy, hz], ...}  (half-extent, meter, 물체 로컬 프레임)

실행: python3 scripts/tools/compute_object_bbox.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VISDEX_URDF = REPO / "assets/visdex_objects/urdf"
PRIMITIVE_URDF = REPO / "assets/primitives/urdf"
OUT_JSON = REPO / "assets/object_bbox.json"


def _bbox_from_obj(path: Path) -> tuple[float, float, float]:
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            for i in range(3):
                v = float(parts[i + 1])
                mins[i] = min(mins[i], v)
                maxs[i] = max(maxs[i], v)
    if mins[0] == float("inf"):
        raise ValueError(f"정점 없음: {path}")
    return tuple((maxs[i] - mins[i]) * 0.5 for i in range(3))  # type: ignore[return-value]


def _bbox_from_stl(path: Path) -> tuple[float, float, float]:
    with path.open("rb") as f:
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for _ in range(n):
            data = f.read(50)
            for v in range(3):
                for i, c in enumerate(struct.unpack_from("<3f", data, 12 + v * 12)):
                    mins[i] = min(mins[i], c)
                    maxs[i] = max(maxs[i], c)
    return tuple((maxs[i] - mins[i]) * 0.5 for i in range(3))  # type: ignore[return-value]


def main() -> None:
    out: dict[str, list[float]] = {}

    for d in sorted(p for p in VISDEX_URDF.iterdir() if p.is_dir()):
        obj = d / "visual_model.obj"
        if not obj.is_file():
            print(f"[SKIP] mesh 없음: {d.name}")
            continue
        out[d.name] = list(_bbox_from_obj(obj))

    for d in sorted(p for p in PRIMITIVE_URDF.iterdir() if p.is_dir()):
        stl = d / f"{d.name}.stl"
        if not stl.is_file():
            print(f"[SKIP] mesh 없음: {d.name}")
            continue
        out[d.name] = list(_bbox_from_stl(stl))

    # cup 은 visdex/primitives 어디에도 urdf 소스가 없다(별도 자산).
    # cup_radius_approx=0.045(폭 9cm) 와 일치하는 cup_big_2.stl 을 쓴다.
    cup_stl = REPO / "assets/cup/cup_big_2.stl"
    if cup_stl.is_file():
        out["cup"] = list(_bbox_from_stl(cup_stl))

    OUT_JSON.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")

    heights = sorted(((2.0 * v[2], k) for k, v in out.items()))
    print(f"[OK] {len(out)}종 → {OUT_JSON}")
    print("가장 납작한 12종 (물체 로컬 z 높이):")
    for h, k in heights[:12]:
        print(f"  {k:18s} {h * 100:5.1f} cm")


if __name__ == "__main__":
    main()
