# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run adaptive grasp comparison as isolated IsaacLab processes.

The per-bin evaluator creates one Isaac env at a time. This wrapper launches a
fresh Isaac process for every version/bead-count pair, then merges the child
CSV files into the comparison outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HDGP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(HDGP_ROOT, ".."))
DEFAULT_EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_grasp_adaptive_compare.py")
DEFAULT_OUTPUT_DIR = os.path.join(HDGP_ROOT, "grasping_project", "eval_policy_compare")
DEFAULT_ISAACLAB_DIR = os.path.join(WORKSPACE_ROOT, "IsaacLab")

VERSIONS = ["v7", "v8", "v9", "v10"]
BEAD_COUNTS = [0, 10, 20, 30]
COMMON_FIELDS = [
    "version",
    "policy_label",
    "task",
    "checkpoint",
    "bead_count",
    "mass_kg",
    "success",
    "grip",
    "contact_count",
    "lift_height_m",
    "total_force_n",
    "force_ratio",
    "slip_proxy",
    "grasp_action_mean",
    "grasp_action_std",
    "grasp_action_min",
    "grasp_action_max",
    "thumb_action",
    "index_action",
    "middle_action",
    "ring_action",
    "pinky_action",
    "action_schema",
    "notes",
]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Isolated-process batch runner for adaptive grasp comparison.")
    parser.add_argument("--versions", nargs="+", default=VERSIONS, choices=VERSIONS)
    parser.add_argument("--bead_counts", nargs="+", type=int, default=BEAD_COUNTS)
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--episodes_per_bin", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--isaaclab_dir", type=str, default=DEFAULT_ISAACLAB_DIR)
    parser.add_argument("--eval_script", type=str, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--max_steps_per_bin", type=int, default=200000)
    parser.add_argument("--timeout_sec", type=int, default=0, help="Per child timeout. 0 means no timeout.")
    parser.add_argument("--merge_only", action="store_true", help="Only merge existing child outputs.")
    parser.add_argument("--rerun_existing", action="store_true", help="Rerun bins even when enough rows already exist.")
    parser.add_argument("--no_headless", action="store_true", help="Do not pass --headless to child Isaac processes.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip merged PNG generation.")
    return parser.parse_known_args()


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    if len(values) < 2:
        return 0.0 if values else math.nan
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _part_dir(output_dir: str, version: str, bead_count: int) -> str:
    return os.path.join(output_dir, "parts", f"{version}_b{bead_count}")


def _candidate_part_dirs(output_dir: str, version: str, bead_count: int) -> list[str]:
    name = f"{version}_b{bead_count}"
    return [
        os.path.join(output_dir, "parts", name),
        os.path.join(output_dir, name),
    ]


def _read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _has_enough_rows(path: str, version: str, bead_count: int, expected: int) -> bool:
    rows = _read_csv(path)
    matching = [
        row for row in rows
        if row.get("version") == version and int(_float(row.get("bead_count"), -1)) == bead_count
    ]
    return len(matching) >= expected


def _run_child(args: argparse.Namespace, extra_args: list[str], version: str, bead_count: int) -> None:
    child_out = _part_dir(args.output_dir, version, bead_count)
    if not args.rerun_existing:
        for candidate_dir in _candidate_part_dirs(args.output_dir, version, bead_count):
            child_csv = os.path.join(candidate_dir, "per_episode.csv")
            if _has_enough_rows(child_csv, version, bead_count, args.episodes_per_bin):
                print(f"[SKIP] {version} bead={bead_count}: existing output has enough rows at {candidate_dir}")
                return

    os.makedirs(child_out, exist_ok=True)
    command = [
        "./isaaclab.sh",
        "-p",
        os.path.abspath(args.eval_script),
        "--versions",
        version,
        "--bead_counts",
        str(bead_count),
        "--num_envs",
        str(args.num_envs),
        "--episodes_per_bin",
        str(args.episodes_per_bin),
        "--seed",
        str(args.seed),
        "--output_dir",
        child_out,
        "--max_steps_per_bin",
        str(args.max_steps_per_bin),
        "--skip_plots",
    ]
    if not args.no_headless:
        command.append("--headless")
    command.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("TERM", "xterm")
    print(f"[RUN] {version} bead={bead_count} -> {child_out}")
    subprocess.run(
        command,
        cwd=os.path.abspath(args.isaaclab_dir),
        env=env,
        check=True,
        timeout=None if args.timeout_sec <= 0 else args.timeout_sec,
    )


def _collect_records(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    missing: list[str] = []
    for version in args.versions:
        for bead_count in args.bead_counts:
            path = ""
            rows: list[dict] = []
            for candidate_dir in _candidate_part_dirs(args.output_dir, version, bead_count):
                candidate_path = os.path.join(candidate_dir, "per_episode.csv")
                rows = _read_csv(candidate_path)
                if rows:
                    path = candidate_path
                    break
            if not rows:
                missing.append(os.path.join(_part_dir(args.output_dir, version, bead_count), "per_episode.csv"))
                continue
            records.extend(rows[: args.episodes_per_bin])
    if missing:
        print("[WARN] missing child outputs:")
        for path in missing:
            print(f"  {path}")
    return records


def _write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(COMMON_FIELDS)
    extra_keys = sorted({key for row in records for key in row.keys()} - set(keys))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra_keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"[INFO] wrote {path}")


def _summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["version"], int(_float(row["bead_count"], -1)))].append(row)

    metrics = ["success", "grip", "contact_count", "lift_height_m", "total_force_n", "force_ratio", "slip_proxy"]
    summary: list[dict] = []
    for (version, bead_count), rows in sorted(groups.items(), key=lambda item: (VERSIONS.index(item[0][0]), item[0][1])):
        item = {
            "version": version,
            "policy_label": rows[0].get("policy_label", ""),
            "bead_count": bead_count,
            "mass_kg": _float(rows[0].get("mass_kg")),
            "n": len(rows),
            "success_rate": _mean([_float(row.get("success")) for row in rows]),
        }
        for metric in metrics:
            vals = [_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = _mean(vals)
            item[f"{metric}_std"] = _std(vals)
        summary.append(item)
    return summary


def _write_summary(summary: list[dict], path: str) -> None:
    if not summary:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[INFO] wrote {path}")


def _plot(summary: list[dict], out_dir: str, versions: list[str], skip_plots: bool) -> None:
    if skip_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    def series(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(8, 5))
        for version in versions:
            rows = [row for row in summary if row["version"] == version]
            rows.sort(key=lambda row: row["bead_count"])
            if rows:
                plt.plot([row["mass_kg"] for row in rows], [row[metric] for row in rows], marker="o", label=version)
        plt.xlabel("mass kg")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"[INFO] wrote {path}")

    series("success_rate", "success rate", "success_by_mass.png")
    series("total_force_n_mean", "total fingertip force N", "force_by_mass.png")
    series("force_ratio_mean", "F_total / (mass*g)", "force_ratio_by_mass.png")
    series("contact_count_mean", "contact count", "contact_by_mass.png")
    series("slip_proxy_mean", "slip proxy", "slip_proxy_by_mass.png")
    series("grip_mean", "grip action intensity", "grip_action_adaptation_by_mass.png")


def _write_report(summary: list[dict], out_dir: str) -> None:
    path = os.path.join(out_dir, "report_snippet.md")
    lines = [
        "# Adaptive Grasp Checkpoint Comparison",
        "",
        "Protocol: each version/bead-count bin was evaluated in an isolated Isaac process, then merged.",
        "",
        "Interpretation notes:",
        "- v7 is a `synergy_forced_mass_baseline`: the cup mass is forced to 0.17/0.27/0.37/0.47kg, but the policy was not trained with bead or mass observations.",
        "- v8 uses the same 5D synergy grip interface as v7, but is mass-aware through bead-conditioned training.",
        "- v9 and v10 use 20D full-joint grip control; compare their grip metric as a curl-joint/full-hand intensity summary rather than a direct action-space equivalent to v7/v8.",
        "- v10 should be judged by combined evidence: success across mass, sufficient force ratio, stable contact, lower slip proxy, and full-joint adaptive grip behavior.",
        "",
        "| version | label | bead | mass_kg | n | success_rate | grip | contacts | force_ratio | slip_proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['version']} | {row['policy_label']} | {row['bead_count']} | "
            f"{row['mass_kg']:.3f} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['grip_mean']:.3f} | {row['contact_count_mean']:.3f} | "
            f"{row['force_ratio_mean']:.3f} | {row['slip_proxy_mean']:.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] wrote {path}")


def _print_summary(summary: list[dict]) -> None:
    print("\nversion bead n success grip contacts force_ratio")
    for row in summary:
        print(
            f"{row['version']:>7} {row['bead_count']:>4} {row['n']:>4} "
            f"{row['success_rate']:.3f} {row['grip_mean']:.3f} "
            f"{row['contact_count_mean']:.3f} {row['force_ratio_mean']:.3f}"
        )


def main() -> int:
    args, extra_args = _parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.isaaclab_dir = os.path.abspath(args.isaaclab_dir)
    args.eval_script = os.path.abspath(args.eval_script)

    if not os.path.isfile(os.path.join(args.isaaclab_dir, "isaaclab.sh")):
        raise FileNotFoundError(f"isaaclab.sh not found under --isaaclab_dir: {args.isaaclab_dir}")
    if not os.path.isfile(args.eval_script):
        raise FileNotFoundError(f"eval script not found: {args.eval_script}")

    if not args.merge_only:
        for version in args.versions:
            for bead_count in args.bead_counts:
                _run_child(args, extra_args, version, bead_count)

    records = _collect_records(args)
    if not records:
        raise RuntimeError("No per-episode records found to merge.")

    per_episode_csv = os.path.join(args.output_dir, "per_episode.csv")
    summary_csv = os.path.join(args.output_dir, "bin_summary.csv")
    _write_csv(records, per_episode_csv)
    summary = _summarize(records)
    _write_summary(summary, summary_csv)
    _plot(summary, args.output_dir, args.versions, args.skip_plots)
    _write_report(summary, args.output_dir)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
