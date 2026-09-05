#!/usr/bin/env python3
"""Generate deterministic object-code -> feature mapping (.pt) for Phase C 4.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _feature_from_code(code: str, dim: int) -> torch.Tensor:
    """Deterministic pseudo-random feature from code string."""
    vals = []
    ctr = 0
    while len(vals) < dim:
        h = hashlib.sha256(f"{code}:{ctr}".encode("utf-8")).digest()
        for b in h:
            vals.append((b / 255.0) * 2.0 - 1.0)
            if len(vals) >= dim:
                break
        ctr += 1
    x = torch.tensor(vals[:dim], dtype=torch.float32)
    n = x.norm(p=2).clamp(min=1e-6)
    return x / n


def build_codes(assets_dir: Path) -> list[str]:
    codes: list[str] = []
    # cup object used by 5g_grasp_right_v2
    codes.append("cup")
    # primitive bank codes (if present)
    primitive_root = assets_dir / "multi_obj" / "primitives" / "USD"
    if primitive_root.exists():
        subdirs = sorted([p.name for p in primitive_root.iterdir() if p.is_dir()])
        for name in subdirs:
            codes.append(f"primitive:{name}")
    # fallback primitive key
    codes.append("primitive:default")
    # visdex object bank codes (if present)
    visdex_root = assets_dir / "multi_obj" / "visdex_objects" / "USD"
    if visdex_root.exists():
        for name in sorted(p.name for p in visdex_root.iterdir() if p.is_dir()):
            codes.append(f"visdex:{name}")
    # de-dup while preserving order
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate object-code feature map.")
    parser.add_argument("--assets_dir", type=Path, default=Path("/home/user/rl_ws/hdgp/assets"))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/user/rl_ws/hdgp/assets/multi_obj/object_pc_features/openarm_right_object_code_feat_dim64.pt"),
    )
    args = parser.parse_args()

    codes = build_codes(args.assets_dir)
    features = torch.stack([_feature_from_code(code, args.dim) for code in codes], dim=0)
    code_to_index = {code: i for i, code in enumerate(codes)}

    payload = {
        "feature_dim": int(args.dim),
        "codes": codes,
        "features": features,
        "code_to_index": code_to_index,
        "meta": {
            "generator": "generate_object_pc_feature_map.py",
            "scheme": "sha256_deterministic_unit_l2",
            "assets_dir": str(args.assets_dir),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    summary = {
        "output": str(args.output),
        "feature_dim": int(args.dim),
        "num_codes": len(codes),
        "codes_preview": codes[:10],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
