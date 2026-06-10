#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import struct
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from tensorboardX.proto import event_pb2


DEFAULT_ROOT = Path("/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v10")
DEFAULT_RUN_A = DEFAULT_ROOT / "test7"
DEFAULT_RUN_B = DEFAULT_ROOT / "test8"

CHECKPOINT_RE = re.compile(r"last_5g_grasp_right-v10_ep_(\d+)_rew_(-?\d+(?:\.\d+)?)\.pth$")

CORE_METRICS = [
    "rewards",
    "object_stat/success_rate",
    "hand_contact/contact_count",
    "hand_contact/middle_contacts",
    "hand_force/f_ratio",
    "hand_force/f_ratio_delta",
    "reward/adaptive_grip",
    "reward/slip",
    "reward/multi_phalanx",
    "reward/full_contact",
    "reward/lift",
    "reward/success_bonus",
    "reward/thumb_pose_anchor",
    "reward/thumb_slide_penalty",
    "reward/grasp_shape_consistency",
    "hand_joint/rj_1/thumb_anchor_error",
    "hand_joint/rj_1/thumb_downward_delta",
    "hand_joint/rj_1/thumb_tip_direction_cos",
    "hand_joint/rj_1/grasp_shape_error",
    "adr_difficulty_progress",
    "adr_success_min",
]

BIN_METRICS = [
    "mass_bin/0b/sr",
    "mass_bin/10b/sr",
    "mass_bin/20b/sr",
    "mass_bin/30b/sr",
    "mass_bin/0b/f_ratio",
    "mass_bin/10b/f_ratio",
    "mass_bin/20b/f_ratio",
    "mass_bin/30b/f_ratio",
    "mass_bin/0b/contacts",
    "mass_bin/10b/contacts",
    "mass_bin/20b/contacts",
    "mass_bin/30b/contacts",
    "mass_bin/0b/lift",
    "mass_bin/10b/lift",
    "mass_bin/20b/lift",
    "mass_bin/30b/lift",
    "mass_bin/0b/adaptive_grip",
    "mass_bin/10b/adaptive_grip",
    "mass_bin/20b/adaptive_grip",
    "mass_bin/30b/adaptive_grip",
    "mass_bin/0b/full_contact",
    "mass_bin/10b/full_contact",
    "mass_bin/20b/full_contact",
    "mass_bin/30b/full_contact",
    "mass_bin/0b/multi_phalanx",
    "mass_bin/10b/multi_phalanx",
    "mass_bin/20b/multi_phalanx",
    "mass_bin/30b/multi_phalanx",
]

LOSS_TAGS = [
    "losses/a_loss",
    "losses/c_loss",
    "losses/cval_loss",
    "losses/entropy",
    "losses/bounds_loss",
    "info/kl",
    "info/last_lr",
    "info/cval_lr",
]


@dataclass
class RunArtifacts:
    name: str
    run_dir: Path
    event_file: Path
    data: dict[str, pd.DataFrame]
    iter_df: pd.DataFrame
    checkpoint_df: pd.DataFrame
    env_cfg: dict[str, Any]
    agent_cfg: dict[str, Any]


def parse_checkpoint_filename(filename: str) -> dict[str, float | int] | None:
    match = CHECKPOINT_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "episode": int(match.group(1)),
        "reward": float(match.group(2)),
    }


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        try:
            loaded = yaml.safe_load(text)
        except Exception:
            loaded = yaml.unsafe_load(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def iter_scalar_events(event_file: Path):
    with event_file.open("rb") as fh:
        while True:
            header = fh.read(8)
            if not header:
                break
            if len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            fh.read(4)  # masked crc of length
            payload = fh.read(length)
            fh.read(4)  # masked crc of data

            event = event_pb2.Event()
            event.ParseFromString(payload)
            if not event.summary or not event.summary.value:
                continue

            for value in event.summary.value:
                if value.HasField("simple_value"):
                    yield value.tag, int(event.step), float(value.simple_value), float(event.wall_time)


def load_events(events_path: str) -> dict[str, pd.DataFrame]:
    rows_by_tag: dict[str, list[dict[str, float | int]]] = {}
    for tag, step, value, wall_time in iter_scalar_events(Path(events_path)):
        rows_by_tag.setdefault(tag, []).append(
            {"step": step, "value": value, "wall_time": wall_time}
        )

    data: dict[str, pd.DataFrame] = {}
    for tag, rows in rows_by_tag.items():
        df = pd.DataFrame(rows).sort_values(["step", "wall_time"]).reset_index(drop=True)
        data[tag] = df
    return data


def build_master_df(data: dict[str, pd.DataFrame], suffix: str = "/iter") -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for tag, df in data.items():
        if tag.endswith(suffix):
            name = tag[: -len(suffix)]
            frames[name] = df.set_index("step")["value"]
    if not frames:
        return pd.DataFrame()
    master = pd.DataFrame(frames).sort_index()
    master.index.name = "iter"
    return master


def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def summarize_checkpoints(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted((run_dir / "nn").glob("last_5g_grasp_right-v10_ep_*_rew_*.pth")):
        parsed = parse_checkpoint_filename(path.name)
        if parsed is None:
            continue
        rows.append(
            {
                "run": run_dir.name,
                "episode": parsed["episode"],
                "reward": parsed["reward"],
                "file": path.name,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["run", "episode", "reward", "file"])
    return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)


def build_window_summary(
    df: pd.DataFrame,
    metrics: list[str],
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    ranked = df.reset_index(drop=False).copy()
    index_name = ranked.columns[0]

    for start, end in windows:
        subset = ranked[(ranked[index_name] >= start) & (ranked[index_name] <= end)]
        row: dict[str, Any] = {"window": f"idx {start}-{end}"}
        for metric in metrics:
            row[metric] = float(subset[metric].dropna().mean()) if metric in subset.columns and not subset.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_metric_table(run_artifacts: list[RunArtifacts], metrics: list[str], tail_fraction: float = 0.1) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        row: dict[str, Any] = {"metric": metric}
        for artifact in run_artifacts:
            if metric not in artifact.iter_df.columns or artifact.iter_df.empty:
                row[artifact.name] = math.nan
                continue
            tail_count = max(1, int(len(artifact.iter_df) * tail_fraction))
            row[artifact.name] = float(artifact.iter_df[metric].tail(tail_count).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No data_"
    rendered = df.copy()
    for column in rendered.columns:
        if pd.api.types.is_numeric_dtype(rendered[column]):
            rendered[column] = rendered[column].map(
                lambda value: f"{value:.{digits}f}" if pd.notna(value) else "nan"
            )
    try:
        return rendered.to_markdown(index=False)
    except ImportError:
        headers = list(rendered.columns)
        separator = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rendered.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)


def make_default_windows(index: pd.Index) -> list[tuple[int, int]]:
    if len(index) == 0:
        return []
    start_value = int(index.min())
    end_value = int(index.max())
    span = max(1, end_value - start_value + 1)
    chunk = max(1, math.ceil(span / 4))
    windows = []
    start = start_value
    while start <= end_value:
        end = min(end_value, start + chunk - 1)
        windows.append((start, end))
        start = end + 1
    while len(windows) < 4:
        windows.append((end_value, end_value))
    return windows[:4]


def plot_run_series(ax, artifact: RunArtifacts, metric: str, label: str, color: str, smooth_window: int) -> bool:
    if metric not in artifact.iter_df.columns:
        return False
    series = artifact.iter_df[metric].dropna()
    if series.empty:
        return False
    x = series.index.to_numpy()
    ax.plot(x, series.values, alpha=0.22, linewidth=0.8, color=color)
    ax.plot(x, smooth(series, smooth_window).values, linewidth=1.8, color=color, label=label)
    return True


def finalize_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("iter", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def plot_overview(run_artifacts: list[RunArtifacts], out_path: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["steelblue", "darkorange"]
    labels = [artifact.name for artifact in run_artifacts]

    metrics = [
        ("rewards", "Episode Reward", "reward"),
        ("object_stat/success_rate", "Success Rate", "ratio"),
        ("hand_force/f_ratio", "Force Ratio", "ratio"),
        ("hand_contact/contact_count", "Num Contacts", "count"),
    ]

    for ax, (metric, title, ylabel) in zip(axes.flatten(), metrics):
        drawn = False
        for artifact, label, color in zip(run_artifacts, labels, colors):
            drawn = plot_run_series(ax, artifact, metric, label, color, smooth_window) or drawn
        if drawn:
            finalize_axis(ax, title, ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adaptive_grip(run_artifacts: list[RunArtifacts], out_path: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["steelblue", "darkorange"]
    labels = [artifact.name for artifact in run_artifacts]
    metric_groups = [
        (["hand_force/f_ratio", "f_ratio_light", "f_ratio_heavy"], "Force Ratio Mean/Light/Heavy", "ratio"),
        (["hand_force/f_ratio_delta", "reward/adaptive_grip"], "Adaptivity Gap + Reward", "value"),
        (["mass_bin/0b/f_ratio", "mass_bin/10b/f_ratio", "mass_bin/20b/f_ratio", "mass_bin/30b/f_ratio"], "Per-Mass Force Ratio", "ratio"),
        (["mass_bin/0b/sr", "mass_bin/10b/sr", "mass_bin/20b/sr", "mass_bin/30b/sr"], "Per-Mass Success Rate", "ratio"),
    ]

    for ax, (metrics, title, ylabel) in zip(axes.flatten(), metric_groups):
        drawn = False
        for artifact, run_label, color in zip(run_artifacts, labels, colors):
            for idx, metric in enumerate(metrics):
                metric_label = f"{run_label}:{metric}"
                line_color = color if idx == 0 else None
                if metric not in artifact.iter_df.columns:
                    continue
                series = artifact.iter_df[metric].dropna()
                if series.empty:
                    continue
                ax.plot(series.index.to_numpy(), smooth(series, smooth_window).values, linewidth=1.4, alpha=0.95, label=metric_label, color=line_color)
                drawn = True
        if drawn:
            finalize_axis(ax, title, ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_contact_success(run_artifacts: list[RunArtifacts], out_path: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["steelblue", "darkorange"]
    labels = [artifact.name for artifact in run_artifacts]
    metric_groups = [
        (["hand_contact/contact_count", "reward/multi_phalanx", "reward/full_contact"], "Contact Quality", "value"),
        (["reward/slip", "reward/preload", "reward/force_smooth"], "Lift Stability", "value"),
        (["reward/lift", "reward/success_bonus"], "Lift/Success Bonus", "value"),
        (["reward/thumb_pose_anchor", "reward/thumb_slide_penalty", "reward/grasp_shape_consistency"], "Thumb / Shape Control", "value"),
    ]

    for ax, (metrics, title, ylabel) in zip(axes.flatten(), metric_groups):
        drawn = False
        for artifact, run_label, color in zip(run_artifacts, labels, colors):
            for idx, metric in enumerate(metrics):
                if metric not in artifact.iter_df.columns:
                    continue
                series = artifact.iter_df[metric].dropna()
                if series.empty:
                    continue
                line_color = color if idx == 0 else None
                ax.plot(series.index.to_numpy(), smooth(series, smooth_window).values, linewidth=1.4, alpha=0.95, label=f"{run_label}:{metric}", color=line_color)
                drawn = True
        if drawn:
            finalize_axis(ax, title, ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adr_diagnostics(run_artifacts: list[RunArtifacts], out_path: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["steelblue", "darkorange"]
    labels = [artifact.name for artifact in run_artifacts]
    metric_groups = [
        (["adr_difficulty_progress", "adr_success_min"], "ADR Progress / Success Min", "value"),
        (["episode_lengths", "shaped_rewards"], "Episode Length / Shaped Reward", "value"),
        (["mass_bin/0b/lift", "mass_bin/10b/lift", "mass_bin/20b/lift", "mass_bin/30b/lift"], "Per-Mass Lift", "value"),
        (["mass_bin/0b/adaptive_grip", "mass_bin/10b/adaptive_grip", "mass_bin/20b/adaptive_grip", "mass_bin/30b/adaptive_grip"], "Per-Mass Adaptive Grip", "value"),
    ]

    for ax, (metrics, title, ylabel) in zip(axes.flatten(), metric_groups):
        drawn = False
        for artifact, run_label, color in zip(run_artifacts, labels, colors):
            for idx, metric in enumerate(metrics):
                if metric not in artifact.iter_df.columns:
                    continue
                series = artifact.iter_df[metric].dropna()
                if series.empty:
                    continue
                line_color = color if idx == 0 else None
                ax.plot(series.index.to_numpy(), smooth(series, smooth_window).values, linewidth=1.4, alpha=0.95, label=f"{run_label}:{metric}", color=line_color)
                drawn = True
        if drawn:
            finalize_axis(ax, title, ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_losses(run_artifacts: list[RunArtifacts], out_path: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["steelblue", "darkorange"]
    labels = [artifact.name for artifact in run_artifacts]
    groups = [
        (["losses/a_loss", "losses/c_loss", "losses/cval_loss"], "Actor/Critic Losses", "loss"),
        (["losses/entropy", "losses/bounds_loss"], "Entropy / Bounds", "loss"),
        (["info/kl"], "KL", "value"),
        (["info/last_lr", "info/cval_lr"], "Learning Rate", "lr"),
    ]

    for ax, (metrics, title, ylabel) in zip(axes.flatten(), groups):
        drawn = False
        for artifact, run_label, color in zip(run_artifacts, labels, colors):
            for idx, metric in enumerate(metrics):
                if metric not in artifact.data:
                    continue
                series = artifact.data[metric].set_index("step")["value"].dropna()
                if series.empty:
                    continue
                line_color = color if idx == 0 else None
                ax.plot(series.index.to_numpy(), smooth(series, min(smooth_window, 10)).values, linewidth=1.4, alpha=0.95, label=f"{run_label}:{metric}", color=line_color)
                drawn = True
        if drawn:
            finalize_axis(ax, title, ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_config(run_a: RunArtifacts, run_b: RunArtifacts) -> dict[str, Any]:
    env_a = run_a.env_cfg
    env_b = run_b.env_cfg
    keys = [
        "seed",
        "enable_contact_adr",
        "enable_adr",
        "finger_delta_scale",
        "obs_noise_cup_pos",
        "adaptive_force_weight",
        "af_target_ratio",
        "af_sharpness",
    ]
    key_diffs = []
    for key in keys:
        if env_a.get(key) != env_b.get(key):
            key_diffs.append(f"{key}: {env_a.get(key)} vs {env_b.get(key)}")
    if not key_diffs:
        key_diffs.append("env.yaml 기준 의미 있는 차이는 seed뿐")
    return {
        "seed_a": env_a.get("seed"),
        "seed_b": env_b.get("seed"),
        "enable_contact_adr": env_a.get("enable_contact_adr"),
        "enable_adr": env_a.get("enable_adr"),
        "adr_success_min_final_a": get_last_value(run_a.iter_df, "adr_success_min"),
        "adr_success_min_final_b": get_last_value(run_b.iter_df, "adr_success_min"),
        "key_diffs": key_diffs,
    }


def get_last_value(df: pd.DataFrame, metric: str) -> float | None:
    if metric not in df.columns:
        return None
    series = df[metric].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def build_checkpoint_summary_table(run_artifacts: list[RunArtifacts]) -> pd.DataFrame:
    rows = []
    for artifact in run_artifacts:
        ckpt = artifact.checkpoint_df
        if ckpt.empty:
            rows.append(
                {
                    "run": artifact.name,
                    "num_checkpoints": 0,
                    "peak_reward": math.nan,
                    "worst_reward": math.nan,
                    "final_reward": math.nan,
                    "final_episode": math.nan,
                    "negative_count": 0,
                }
            )
            continue
        rows.append(
            {
                "run": artifact.name,
                "num_checkpoints": int(len(ckpt)),
                "peak_reward": float(ckpt["reward"].max()),
                "worst_reward": float(ckpt["reward"].min()),
                "final_reward": float(ckpt["reward"].iloc[-1]),
                "final_episode": int(ckpt["episode"].iloc[-1]),
                "negative_count": int((ckpt["reward"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def generate_findings(
    run_a: RunArtifacts,
    run_b: RunArtifacts,
    final_comparison: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    config_summary: dict[str, Any],
) -> list[str]:
    findings = []
    findings.append(
        f"{run_a.name} seed={config_summary.get('seed_a')}, {run_b.name} seed={config_summary.get('seed_b')}이며 env.yaml 기준 핵심 설정은 동일하다."
    )
    findings.append(
        f"이번 run에서는 enable_contact_adr={config_summary.get('enable_contact_adr')}라서, 해석 축은 contact ADR mismatch가 아니라 difficulty ADR와 seed sensitivity다."
    )

    ckpt_map = checkpoint_summary.set_index("run").to_dict("index")
    if ckpt_map.get(run_a.name, {}).get("final_reward", math.nan) > ckpt_map.get(run_b.name, {}).get("final_reward", math.nan):
        findings.append(
            f"체크포인트 기준 최종 reward는 {run_a.name}이 더 높다 ({ckpt_map[run_a.name]['final_reward']:.1f} vs {ckpt_map[run_b.name]['final_reward']:.1f})."
        )
    else:
        findings.append(
            f"체크포인트 기준 최종 reward는 {run_b.name}이 더 높다 ({ckpt_map[run_b.name]['final_reward']:.1f} vs {ckpt_map[run_a.name]['final_reward']:.1f})."
        )

    metric_map = final_comparison.set_index("metric")
    for metric, label in [
        ("object_stat/success_rate", "final success rate"),
        ("hand_contact/contact_count", "final contact count"),
        ("reward/lift", "final lift reward"),
        ("reward/success_bonus", "final success bonus"),
    ]:
        if metric in metric_map.index:
            a_val = metric_map.at[metric, run_a.name]
            b_val = metric_map.at[metric, run_b.name]
            better = run_a.name if a_val > b_val else run_b.name
            findings.append(f"{label}는 {better} 쪽이 더 높다 ({a_val:.3f} vs {b_val:.3f}).")

    if "hand_force/f_ratio_delta" in metric_map.index:
        a_val = metric_map.at["hand_force/f_ratio_delta", run_a.name]
        b_val = metric_map.at["hand_force/f_ratio_delta", run_b.name]
        findings.append(
            f"heavy-light force ratio gap(hand_force/f_ratio_delta)은 {run_a.name}={a_val:.3f}, {run_b.name}={b_val:.3f}다."
        )

    for metric, label in [("mass_bin/30b/sr", "30b success"), ("mass_bin/0b/sr", "0b success")]:
        if metric in metric_map.index:
            a_val = metric_map.at[metric, run_a.name]
            b_val = metric_map.at[metric, run_b.name]
            findings.append(f"{label}는 {run_a.name}={a_val:.3f}, {run_b.name}={b_val:.3f}다.")

    return findings


def render_markdown_report(
    report_date: str,
    run_a_name: str,
    run_b_name: str,
    run_a_dir: Path,
    run_b_dir: Path,
    config_summary: dict[str, Any],
    checkpoint_summary: pd.DataFrame,
    window_tables: dict[str, pd.DataFrame],
    final_comparison: pd.DataFrame,
    findings: list[str],
    missing_metrics: list[str],
) -> str:
    lines = [
        f"# 5g_grasp_right_v10 {run_a_name}/{run_b_name} adaptive grasp analysis",
        "",
        f"작성일: {report_date}",
        "",
        "분석 대상:",
        "",
        f"- `{run_a_dir}`",
        f"- `{run_b_dir}`",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend([f"{idx}. {finding}" for idx, finding in enumerate(findings, start=1)])
    lines.extend(
        [
            "",
            "## 1. 실험 설정 비교",
            "",
            f"- `{run_a_name}` seed: {config_summary.get('seed_a')}",
            f"- `{run_b_name}` seed: {config_summary.get('seed_b')}",
            f"- `enable_contact_adr={config_summary.get('enable_contact_adr')}`",
            f"- `enable_adr={config_summary.get('enable_adr')}`",
            "",
            "설정 차이:",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in config_summary.get("key_diffs", [])])
    lines.extend(
        [
            "",
            "## 2. 체크포인트 reward 비교",
            "",
            dataframe_to_markdown(checkpoint_summary, digits=3),
            "",
            "## 3. Window Summary",
            "",
        ]
    )
    for name, table in window_tables.items():
        lines.extend([f"### {name}", "", dataframe_to_markdown(table, digits=3), ""])
    lines.extend(
        [
            "## 4. Final Tail Comparison",
            "",
            dataframe_to_markdown(final_comparison, digits=3),
            "",
            "## 5. 해석 포인트",
            "",
            f"- 이번 비교에서는 `adr_success_min` 최종값이 {config_summary.get('adr_success_min_final_a')} / {config_summary.get('adr_success_min_final_b')}인지 함께 본다.",
            "- `hand_force/f_ratio`, `f_ratio_light/heavy`, `mass_bin/*/f_ratio`는 질량 적응 성향을 직접 보여준다.",
            "- `hand_contact/contact_count`, `reward/multi_phalanx`, `reward/full_contact`는 deep-envelope grasp 유지력을 보여준다.",
            "- `reward/slip`, `reward/preload`, `reward/force_smooth`, `reward/lift`는 lift 단계 failure mode를 분해한다.",
            "- `reward/thumb_pose_anchor`, `reward/thumb_slide_penalty`, `reward/grasp_shape_consistency`와 `hand_joint/rj_1/*`는 grasp form 유지 여부를 보여준다.",
            "- mass-bin 로깅은 `mass_bin/*/f_ratio`와 bin별 success/contact/lift/adaptive_grip/full_contact/multi_phalanx만 유지한다.",
            "",
        ]
    )
    if missing_metrics:
        lines.extend(
            [
                "## Missing Metrics",
                "",
                "다음 메트릭은 현재 run에서 관측되지 않았거나 비활성화 상태였다.",
                "",
            ]
        )
        lines.extend([f"- `{metric}`" for metric in missing_metrics])
        lines.append("")
    return "\n".join(lines)


def collect_run_artifacts(run_dir: Path) -> RunArtifacts:
    event_files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no event file found under {run_dir / 'summaries'}")

    event_file = event_files[-1]
    data = load_events(str(event_file))
    iter_df = build_master_df(data, suffix="/iter")
    checkpoint_df = summarize_checkpoints(run_dir)
    env_cfg = read_yaml(run_dir / "params" / "env.yaml")
    agent_cfg = read_yaml(run_dir / "params" / "agent.yaml")
    return RunArtifacts(
        name=run_dir.name,
        run_dir=run_dir,
        event_file=event_file,
        data=data,
        iter_df=iter_df,
        checkpoint_df=checkpoint_df,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
    )


def save_run_metrics_csv(artifact: RunArtifacts, out_dir: Path) -> Path:
    out_path = out_dir / f"metrics_{artifact.name}.csv"
    artifact.iter_df.to_csv(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 5g_grasp_right_v10 TensorBoard runs")
    parser.add_argument("--run-a", type=Path, default=DEFAULT_RUN_A)
    parser.add_argument("--run-b", type=Path, default=DEFAULT_RUN_B)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--report-date", type=str, default=date.today().isoformat())
    parser.add_argument("--title-suffix", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_a = collect_run_artifacts(args.run_a)
    run_b = collect_run_artifacts(args.run_b)
    run_artifacts = [run_a, run_b]

    output_dir = args.output_dir or (DEFAULT_ROOT / f"analysis_{run_a.name}_{run_b.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for artifact in run_artifacts:
        save_run_metrics_csv(artifact, output_dir)

    checkpoint_summary = build_checkpoint_summary_table(run_artifacts)
    checkpoint_summary.to_csv(output_dir / "checkpoint_summary.csv", index=False)

    missing_metrics = sorted(
        metric
        for metric in (CORE_METRICS + BIN_METRICS)
        if any(metric not in artifact.iter_df.columns for artifact in run_artifacts)
    )

    core_windows = {
        artifact.name: build_window_summary(
            artifact.iter_df,
            metrics=[metric for metric in CORE_METRICS if metric in artifact.iter_df.columns],
            windows=make_default_windows(artifact.iter_df.index),
        )
        for artifact in run_artifacts
    }

    final_metrics = [metric for metric in (CORE_METRICS + BIN_METRICS) if all(metric in artifact.iter_df.columns for artifact in run_artifacts)]
    final_comparison = build_final_metric_table(run_artifacts, final_metrics, tail_fraction=0.1)
    final_comparison.to_csv(output_dir / "run_comparison.csv", index=False)

    config_summary = summarize_config(run_a, run_b)
    findings = generate_findings(run_a, run_b, final_comparison, checkpoint_summary, config_summary)

    plot_overview(run_artifacts, output_dir / "overview.png", args.smooth_window)
    plot_adaptive_grip(run_artifacts, output_dir / "adaptive_grip.png", args.smooth_window)
    plot_contact_success(run_artifacts, output_dir / "contact_success.png", args.smooth_window)
    plot_adr_diagnostics(run_artifacts, output_dir / "adr_diagnostics.png", args.smooth_window)
    plot_losses(run_artifacts, output_dir / "losses.png", args.smooth_window)

    window_tables = {
        f"{run_name} core metrics": table
        for run_name, table in core_windows.items()
    }
    report = render_markdown_report(
        report_date=args.report_date,
        run_a_name=run_a.name,
        run_b_name=run_b.name,
        run_a_dir=run_a.run_dir,
        run_b_dir=run_b.run_dir,
        config_summary=config_summary,
        checkpoint_summary=checkpoint_summary,
        window_tables=window_tables,
        final_comparison=final_comparison,
        findings=findings,
        missing_metrics=missing_metrics,
    )

    report_name = f"{args.report_date}_v10_{run_a.name}_{run_b.name}_adaptive_grasp_analysis.md"
    report_path = DEFAULT_ROOT / report_name
    report_path.write_text(report, encoding="utf-8")

    print(f"saved output dir: {output_dir}")
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
