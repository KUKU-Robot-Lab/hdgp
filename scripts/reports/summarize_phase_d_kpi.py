#!/usr/bin/env python3
"""Summarize KPI matrix reports from report_phase32_success_metrics outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Phase D KPI matrix JSON reports.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def _pick(metrics: dict, key: str):
    return metrics.get(key, None)


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    rows: list[dict] = []
    for path in sorted(input_dir.glob("A?B?C?.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        mean_metrics = data.get("metrics_mean", {})
        row = {
            "case": path.stem,
            "final_success": _pick(mean_metrics, "final_success"),
            "grasp_success_contact_rate": _pick(mean_metrics, "grasp_success_contact_rate"),
            "tip_table_contact_rate": _pick(mean_metrics, "tip_table_contact_rate"),
            "table_contact_penalty": _pick(mean_metrics, "table_contact_penalty"),
            "object_impact_penalty": _pick(mean_metrics, "object_impact_penalty"),
            "tip_object_contact_rate": _pick(mean_metrics, "tip_object_contact_rate"),
        }
        rows.append(row)

    # Simple ranking: prioritize high final_success/contact success and low table contact penalty.
    def _score(r: dict) -> float:
        fs = float(r["final_success"] or 0.0)
        cs = float(r["grasp_success_contact_rate"] or 0.0)
        tr = float(r["tip_table_contact_rate"] or 0.0)
        tp = float(r["table_contact_penalty"] or 0.0)
        return 1.5 * fs + 1.0 * cs - 0.5 * tr - 0.5 * tp

    ranked = sorted(rows, key=_score, reverse=True)
    tuning_notes = []
    if ranked:
        best = ranked[0]["case"]
        tuning_notes.append(f"추천 기본 케이스: {best}")
    for r in ranked:
        case = r["case"]
        tr = float(r["tip_table_contact_rate"] or 0.0)
        if tr > 0.3:
            tuning_notes.append(f"{case}: table 접촉률이 높음 -> table_contact_penalty_weight 상향 검토")
        fs = float(r["final_success"] or 0.0)
        if fs < 0.2:
            tuning_notes.append(f"{case}: final_success 낮음 -> replay/contact gate 조합 재검토")

    summary = {
        "input_dir": str(input_dir),
        "num_cases_found": len(rows),
        "rows": rows,
        "ranked_cases": [r["case"] for r in ranked],
        "tuning_notes": tuning_notes,
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
