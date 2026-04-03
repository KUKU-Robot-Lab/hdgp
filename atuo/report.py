from __future__ import annotations

import json
from pathlib import Path


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _mean_pair(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def build_report_md(report: dict, metrics: dict) -> str:
    train = metrics.get("train", {})
    eval_metrics = metrics.get("eval", {})
    decision = report.get("decision", {})
    analysis = report.get("analysis", {})
    scalars = train.get("scalars", {})

    mean_reward = scalars.get("mean_reward", {})
    entropy = scalars.get("entropy", {})

    lift_mean = _mean_pair(
        eval_metrics.get("lift_success_left"), eval_metrics.get("lift_success_right")
    )
    goal_track_mean = _mean_pair(
        eval_metrics.get("goal_track_success_left"), eval_metrics.get("goal_track_success_right")
    )
    success_rate_mean = _mean_pair(
        eval_metrics.get("success_rate_left"), eval_metrics.get("success_rate_right")
    )
    goal_dist_mean = _mean_pair(
        eval_metrics.get("goal_dist_min_left_mean"), eval_metrics.get("goal_dist_min_right_mean")
    )

    lines = []
    lines.append(f"# Experiment Report: {report.get('run_id', 'unknown')}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- status: {decision.get('status', 'unknown')}")
    lines.append(f"- task: {report.get('task', 'grasp2g-v1')}")
    lines.append(f"- agent: {report.get('agent', 'rsl_rl_dual_cfg_entry_point')}")
    lines.append("")

    lines.append("## Training")
    lines.append(f"- log_dir: {report.get('train', {}).get('log_dir', 'n/a')}")
    lines.append(f"- seed: {report.get('train', {}).get('seed', 'n/a')}")
    lines.append(f"- mean_reward_last_100: {_fmt_float(mean_reward.get('mean_last_100'))}")
    lines.append(f"- mean_reward_max: {_fmt_float(mean_reward.get('max'))}")
    lines.append(f"- entropy_last: {_fmt_float(entropy.get('last'))}")
    lines.append(f"- total_iterations: {report.get('train', {}).get('total_iterations', 'n/a')}")
    lines.append("")

    lines.append("## Evaluation")
    lines.append(f"- success_rate_mean: {_fmt_float(success_rate_mean)}")
    lines.append(f"- lift_success_mean: {_fmt_float(lift_mean)}")
    lines.append(f"- goal_track_success_mean: {_fmt_float(goal_track_mean)}")
    lines.append(f"- goal_dist_mean: {_fmt_float(goal_dist_mean)}")
    lines.append("")

    pour_diag_keys = [
        ("success_rate", "success_rate"),
        ("bead_cross_fraction", "bead_cross_fraction"),
        ("bead_cross_count", "bead_cross_count"),
        ("bead_in_source_rate", "bead_in_source_rate"),
        ("bead_in_target_rate", "bead_in_target_rate"),
        ("mouth_xy_distance", "mouth_xy_distance"),
        ("mouth_z_clearance", "mouth_z_clearance"),
        ("reward_transport", "reward_transport"),
        ("reward_transport_progress", "reward_transport_progress"),
        ("reward_tilt", "reward_tilt"),
        ("reward_tilt_raw", "reward_tilt_raw"),
        ("reward_aligned_tilt", "reward_aligned_tilt"),
        ("directional_tilt_cos", "directional_tilt_cos"),
        ("pct_correct_tilt_dir", "pct_correct_tilt_dir"),
        ("pour_gate", "pour_gate"),
        ("pour_gate_xy", "pour_gate_xy"),
        ("pour_gate_z", "pour_gate_z"),
        ("grasp_maintain", "grasp_maintain"),
        ("contact_maintain", "contact_maintain"),
        ("force_balance", "force_balance"),
        ("force_balance_err", "force_balance_err"),
        ("finger_curl_min", "finger_curl_min"),
        ("spill_ratio", "spill_ratio"),
    ]
    has_pour_diag = any(
        isinstance(scalars.get(key), dict) and scalars.get(key, {}).get("mean_last_100") is not None
        for _, key in pour_diag_keys
    )
    if has_pour_diag:
        lines.append("## Pour Diagnostics")
        lines.append("| Metric | Mean | Last | Min | Max |")
        lines.append("|--------|------|------|-----|-----|")
        for label, key in pour_diag_keys:
            v = scalars.get(key, {})
            if isinstance(v, dict) and v.get("mean_last_100") is not None:
                lines.append(
                    f"| {label} | {_fmt_float(v.get('mean_last_100'))} "
                    f"| {_fmt_float(v.get('last'))} "
                    f"| {_fmt_float(v.get('min'))} "
                    f"| {_fmt_float(v.get('max'))} |"
                )
        lines.append("")

    # ── Key Diagnostic Metrics ──
    diag_keys = [
        ("left_eef_dist", "reward_left_eef_dist_diag"),
        ("right_eef_dist", "reward_right_eef_dist_diag"),
        ("left_eef_dist_xy", "reward_left_eef_dist_xy_diag"),
        ("right_eef_dist_xy", "reward_right_eef_dist_xy_diag"),
        ("left_eef_dist_z", "reward_left_eef_dist_z_diag"),
        ("right_eef_dist_z", "reward_right_eef_dist_z_diag"),
        ("left_eef_dist_delta", "reward_left_eef_dist_delta_diag"),
        ("right_eef_dist_delta", "reward_right_eef_dist_delta_diag"),
        ("left_hand_closure", "reward_left_hand_closure_diag"),
        ("right_hand_closure", "reward_right_hand_closure_diag"),
        ("left_object_height", "reward_left_object_height_diag"),
        ("right_object_height", "reward_right_object_height_diag"),
        ("left_object_displacement", "reward_left_object_displacement_diag"),
        ("right_object_displacement", "reward_right_object_displacement_diag"),
        ("left_phase", "reward_left_grasp2g_phase"),
        ("right_phase", "reward_right_grasp2g_phase"),
    ]
    has_diag = False
    for label, key in diag_keys:
        v = scalars.get(key, {})
        if isinstance(v, dict) and v.get("mean_last_100") is not None:
            has_diag = True
            break

    if has_diag:
        lines.append("## Diagnostic Metrics")
        lines.append("| Metric | Mean | Last | Min | Max |")
        lines.append("|--------|------|------|-----|-----|")
        for label, key in diag_keys:
            v = scalars.get(key, {})
            if isinstance(v, dict) and v.get("mean_last_100") is not None:
                lines.append(
                    f"| {label} | {_fmt_float(v.get('mean_last_100'))} "
                    f"| {_fmt_float(v.get('last'))} "
                    f"| {_fmt_float(v.get('min'))} "
                    f"| {_fmt_float(v.get('max'))} |"
                )
        lines.append("")

    # ── Analysis ──
    if analysis:
        lines.append("## Detected Issues")
        if analysis.get("issues"):
            for issue in analysis.get("issues", []):
                lines.append(f"- {issue}")
        else:
            lines.append("- (none)")
        lines.append("")

        if analysis.get("observations"):
            lines.append("## Observations")
            for obs in analysis.get("observations", []):
                lines.append(f"- {obs}")
            lines.append("")

        if analysis.get("llm_summary"):
            lines.append("## LLM Analysis (Step-by-Step)")
            lines.append("```")
            lines.append(analysis.get("llm_summary", ""))
            lines.append("```")
            lines.append("")

    if analysis.get("applied_overrides"):
        lines.append("## Applied Overrides")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for item in analysis.get("applied_overrides", []):
            if "=" in item:
                k, v = item.split("=", 1)
                lines.append(f"| `{k.strip()}` | `{v.strip()}` |")
            else:
                lines.append(f"| `{item}` | |")
        lines.append("")
        # Override source
        override_source = report.get("train", {}).get("override_source", "")
        if not override_source:
            override_source = "llm" if analysis.get("llm_summary") else "rule_based"
        lines.append(f"- override source: **{override_source}**")
        lines.append("")

    lines.append("## Files")
    lines.append(f"- report.json: {report.get('path_report_json', 'report.json')}")
    lines.append(f"- metrics.json: {report.get('path_metrics_json', 'metrics.json')}")
    if analysis.get("llm_summary"):
        lines.append(f"- analysis_prompt.txt: {report.get('path_analysis_prompt', 'analysis_prompt.txt')}")
        lines.append(f"- analysis_response.txt: {report.get('path_analysis_response', 'analysis_response.txt')}")
    lines.append("")

    return "\n".join(lines)


def write_report_md(report_path: str, metrics_path: str, out_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    metrics_file = Path(metrics_path)
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    else:
        metrics = {}

    report["path_report_json"] = str(Path(report_path).resolve())
    report["path_metrics_json"] = str(metrics_file.resolve())
    report["path_analysis_prompt"] = str(Path(report_path).parent.joinpath("analysis_prompt.txt").resolve())
    report["path_analysis_response"] = str(Path(report_path).parent.joinpath("analysis_response.txt").resolve())

    md = build_report_md(report, metrics)
    Path(out_path).write_text(md, encoding="utf-8")
