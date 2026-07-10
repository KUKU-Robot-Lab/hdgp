#!/usr/bin/env python3
"""Summarize Phase D matrix reports from report_phaseb3_metrics outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Phase D matrix JSON reports.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    rows: list[dict] = []
    for path in sorted(input_dir.glob("A?B?C?.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "case": path.stem,
            "obs_dim_match": bool(data.get("obs_dim_match", False)),
            "obs_dim": data.get("obs_dim"),
            "expected_obs_dim": data.get("expected_obs_dim"),
            "object_pc_feature_norm": data.get("tracked_metrics_last", {}).get("object_pc_feature_norm"),
            "object_pc_clip_ratio": data.get("tracked_metrics_last", {}).get("object_pc_clip_ratio"),
            "object_pc_invalid_ratio": data.get("tracked_metrics_last", {}).get("object_pc_invalid_ratio"),
            "reference_palm_tracking_error": data.get("tracked_metrics_last", {}).get("reference_palm_tracking_error"),
        }
        rows.append(row)

    summary = {
        "input_dir": str(input_dir),
        "num_cases_found": len(rows),
        "all_obs_dim_match": all(r["obs_dim_match"] for r in rows) if rows else False,
        "rows": rows,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
