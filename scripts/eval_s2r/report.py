"""셀 집계·CSV/JSON/히트맵 출력 (순수 함수, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.4
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # 서버 headless 대비
import matplotlib.pyplot as plt

_HEATMAP_METRICS = ("success_rate", "lifted_rate", "grip_finger_count", "displacement_mean")


@dataclass(frozen=True)
class EpisodeResult:
    cell_idx: int
    success: bool
    lifted: bool
    grip_count: float
    displacement: float
    obj_idx: int
    invalid: bool


def aggregate(results: list[EpisodeResult], cells: list[tuple[float, float]]) -> list[dict]:
    rows = []
    for ci, (x, y) in enumerate(cells):
        eps = [r for r in results if r.cell_idx == ci and not r.invalid]
        n_invalid = sum(1 for r in results if r.cell_idx == ci and r.invalid)
        n = len(eps)
        per_obj: dict[int, float] = {}
        for oi in sorted({r.obj_idx for r in eps}):
            sub = [r for r in eps if r.obj_idx == oi]
            per_obj[oi] = sum(r.success for r in sub) / len(sub)
        rows.append({
            "cell_idx": ci, "x": x, "y": y,
            "n_episodes": n, "n_invalid": n_invalid,
            "success_rate": (sum(r.success for r in eps) / n) if n else None,
            "lifted_rate": (sum(r.lifted for r in eps) / n) if n else None,
            "grip_finger_count": (sum(r.grip_count for r in eps) / n) if n else None,
            "displacement_mean": (sum(r.displacement for r in eps) / n) if n else None,
            "per_obj_success": per_obj,
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["cell_idx", "x", "y", "n_episodes", "n_invalid",
              "success_rate", "lifted_rate", "grip_finger_count",
              "displacement_mean", "per_obj_success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["per_obj_success"] = json.dumps(r["per_obj_success"])
            w.writerow(out)


def write_summary(rows: list[dict], meta: dict, path: str) -> None:
    valid = [r for r in rows if r["n_episodes"] > 0]
    total = sum(r["n_episodes"] for r in valid)
    overall = (
        sum(r["success_rate"] * r["n_episodes"] for r in valid) / total if total else None
    )
    payload = {
        "meta": meta,
        "total_episodes": total,
        "total_invalid": sum(r["n_invalid"] for r in rows),
        "overall_success_rate": overall,
        "cells": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_heatmap(rows: list[dict], nx: int, ny: int, metric: str, path: str) -> None:
    if metric not in _HEATMAP_METRICS:
        raise ValueError(f"unknown metric {metric!r} (choose from {_HEATMAP_METRICS})")
    # 셀 idx = xi * ny + yi (grid.build_cells x-major와 동일 규약)
    grid = [[None] * nx for _ in range(ny)]
    for r in rows:
        xi, yi = r["cell_idx"] // ny, r["cell_idx"] % ny
        grid[yi][xi] = r[metric]
    data = [[(v if v is not None else float("nan")) for v in row] for row in grid]
    fig, ax = plt.subplots(figsize=(1.2 * nx + 2, 1.0 * ny + 2))
    im = ax.imshow(data, origin="lower", cmap="RdYlGn", vmin=0.0,
                   vmax=1.0 if metric.endswith("_rate") else None)
    xs = sorted({r["x"] for r in rows})
    ys = sorted({r["y"] for r in rows})
    ax.set_xticks(range(nx), [f"{v:.3f}" for v in xs])
    ax.set_yticks(range(ny), [f"{v:.3f}" for v in ys])
    ax.set_xlabel("object x [m]")
    ax.set_ylabel("object y [m]")
    ax.set_title(metric)
    for yi in range(ny):
        for xi in range(nx):
            v = grid[yi][xi]
            if v is not None:
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
