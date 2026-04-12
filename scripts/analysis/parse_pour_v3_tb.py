"""
TensorBoard 이벤트 파일 파싱 및 시각화 스크립트
대상: 5g_pour_right_v3 학습 실험

사용법:
    python parse_pour_v3_tb.py --events PATH/TO/events.out.tfevents.*
    python parse_pour_v3_tb.py --events PATH/TO/events.out.tfevents.* --outdir ./analysis_output

출력:
    - metrics.csv           : 전체 스칼라 메트릭 DataFrame
    - overview.png          : reward / episode_length / bead / ADR 4패널
    - reward_components.png : 각 reward 항목 시계열
    - stability.png         : cost 항목 + action_rate 시계열
    - pour_quality.png      : pouring 품질 메트릭 (bead, spill, dist)
    - losses.png            : PPO 손실 / KL / lr
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing import event_accumulator


# ──────────────────────────────────────────────────────────────
# 1. 파싱
# ──────────────────────────────────────────────────────────────

def load_events(events_path: str) -> dict[str, pd.DataFrame]:
    """events 파일을 읽어 {tag: DataFrame(step, value)} 딕셔너리로 반환."""
    ea = event_accumulator.EventAccumulator(events_path)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    data: dict[str, pd.DataFrame] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        )
    return data


def build_master_df(data: dict[str, pd.DataFrame], suffix: str = "/iter") -> pd.DataFrame:
    """suffix로 끝나는 태그만 모아 step을 인덱스로 하는 단일 DataFrame 구성."""
    frames = {}
    for tag, df in data.items():
        if tag.endswith(suffix):
            name = tag[: -len(suffix)]
            frames[name] = df.set_index("step")["value"]
    if not frames:
        return pd.DataFrame()
    master = pd.DataFrame(frames)
    master.index.name = "iter"
    return master


# ──────────────────────────────────────────────────────────────
# 2. 시각화 헬퍼
# ──────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _plot_series(ax, df, cols, labels=None, colors=None, smooth_w=20, title="", ylabel="", legend=True):
    """여러 컬럼을 하나의 ax에 그리는 헬퍼."""
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        label = labels[i] if labels else col
        color = colors[i] if colors else None
        ax.plot(s.index, s.values, alpha=0.25, linewidth=0.8, color=color)
        ax.plot(s.index, smooth(s, smooth_w).values, linewidth=1.6, label=label, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("iter", fontsize=8)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────
# 3. 그림 생성
# ──────────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("학습 개요 (5g_pour_right_v3)", fontsize=12)

    # (0,0) Episode reward
    ax = axes[0, 0]
    _plot_series(ax, df, ["rewards"], title="Episode Reward", ylabel="reward", legend=False)

    # (0,1) Episode length
    ax = axes[0, 1]
    _plot_series(ax, df, ["episode_lengths"], title="Episode Length", ylabel="steps", legend=False)

    # (1,0) Bead metrics
    ax = axes[1, 0]
    _plot_series(
        ax, df,
        ["bead_in_target", "bead_cross", "spill_ratio"],
        labels=["bead_in_target", "bead_cross", "spill_ratio"],
        colors=["steelblue", "orange", "red"],
        title="Bead 메트릭 (비율)", ylabel="fraction",
    )

    # (1,1) ADR progress
    ax = axes[1, 1]
    _plot_series(
        ax, df,
        ["adr_success_progress", "adr_noise_progress", "success_fill_ratio"],
        labels=["success_ADR", "noise_ADR", "fill_ratio_target"],
        colors=["green", "purple", "gray"],
        title="ADR Progress", ylabel="progress / ratio",
    )

    plt.tight_layout()
    out = outdir / "overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}")


def plot_reward_components(df: pd.DataFrame, outdir: Path) -> None:
    reward_cols = ["r_hold", "r_lift", "r_approach", "r_prepour", "r_pour",
                   "r_pour_align", "r_success_weighted", "r_tilt_onset"]
    present = [c for c in reward_cols if c in df.columns]

    n = len(present)
    cols_g = 4
    rows_g = (n + cols_g - 1) // cols_g
    fig, axes = plt.subplots(rows_g, cols_g, figsize=(16, rows_g * 3))
    axes_flat = axes.flatten() if n > 1 else [axes]
    fig.suptitle("Reward 항목별 시계열", fontsize=12)

    for i, col in enumerate(present):
        ax = axes_flat[i]
        _plot_series(ax, df, [col], title=col, ylabel="value", legend=False)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = outdir / "reward_components.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}")


def plot_stability(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Pose Stability Cost Metrics", fontsize=12)

    _plot_series(axes[0, 0], df, ["cost_premature_tilt"],
                 title="Premature Tilt Cost", ylabel="cost/step", legend=False)
    _plot_series(axes[0, 1], df, ["cost_grasp_loss"],
                 title="Grasp Loss Cost", ylabel="cost/step", legend=False)
    _plot_series(axes[0, 2], df, ["cost_spill"],
                 title="Spill Cost", ylabel="cost/step", legend=False)
    _plot_series(axes[1, 0], df, ["g_ready", "g_pour"],
                 labels=["g_ready", "g_pour"],
                 colors=["green", "blue"],
                 title="Gate (g_ready / g_pour)", ylabel="gate value")
    _plot_series(axes[1, 1], df, ["mouth_xy_dist", "cup_center_xy_dist"],
                 labels=["mouth_xy_dist", "cup_center_xy_dist"],
                 colors=["navy", "darkorange"],
                 title="Distance Metrics (m)", ylabel="m")
    _plot_series(axes[1, 2], df, ["source_empty_steps"],
                 title="Source Empty Steps", ylabel="steps/ep", legend=False)

    plt.tight_layout()
    out = outdir / "stability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_arm_velocity(df: pd.DataFrame, outdir: Path) -> None:
    """Phase-0 진단: arm joint velocity/acceleration 시계열."""
    vel_cols = ["arm_joint_vel_l2_mean", "arm_joint_vel_max_mean",
                "arm_joint_acc_l2_mean", "tilt_phase_arm_vel"]
    present = [c for c in vel_cols if c in df.columns]
    if not present:
        print("  arm vel 메트릭 없음 (새 학습 후 재실행 필요)")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    fig.suptitle("Arm Joint Velocity/Acceleration (Phase-0 Diagnostics)", fontsize=12)

    colors = ["steelblue", "darkorange", "green", "red"]
    for i, col in enumerate(present):
        _plot_series(axes[i], df, [col], title=col, ylabel="rad/s or rad/s^2",
                     colors=[colors[i]], legend=False)

    plt.tight_layout()
    out = outdir / "arm_velocity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_pour_quality(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pouring 품질 메트릭", fontsize=12)

    _plot_series(axes[0], df, ["bead_in_target"],
                 title="bead_in_target (fraction)", ylabel="fraction", legend=False)
    _plot_series(axes[1], df, ["spill_ratio", "cost_spill"],
                 labels=["spill_ratio", "cost_spill"],
                 colors=["red", "firebrick"],
                 title="Spill 메트릭", ylabel="value")
    _plot_series(axes[2], df, ["mouth_xy_dist"],
                 title="mouth_xy_dist (m)", ylabel="m", legend=False)

    plt.tight_layout()
    out = outdir / "pour_quality.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}")


def plot_losses(df: pd.DataFrame, outdir: Path, data: dict) -> None:
    """PPO losses는 /iter suffix가 없으므로 별도 처리."""
    loss_tags = ["losses/a_loss", "losses/c_loss", "losses/entropy",
                 "losses/bounds_loss", "info/kl", "info/last_lr"]

    frames = {}
    for tag in loss_tags:
        if tag in data:
            frames[tag.replace("/", "_")] = data[tag].set_index("step")["value"]

    if not frames:
        print("  losses 데이터 없음, 스킵")
        return

    loss_df = pd.DataFrame(frames)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("PPO 학습 손실 / 하이퍼파라미터", fontsize=12)
    axes_flat = axes.flatten()

    for i, col in enumerate(loss_df.columns):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        s = loss_df[col].dropna()
        ax.plot(s.index, s.values, alpha=0.3, linewidth=0.8)
        ax.plot(s.index, smooth(s, 10).values, linewidth=1.6, label=col)
        ax.set_title(col.replace("_", "/"), fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = outdir / "losses.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}")


# ──────────────────────────────────────────────────────────────
# 4. 통계 요약 출력
# ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("학습 결과 요약 (마지막 500 iter 평균)")
    print("=" * 60)
    tail = df.tail(500)

    metrics = {
        "Episode Reward": "rewards",
        "Episode Length": "episode_lengths",
        "bead_in_target": "bead_in_target",
        "bead_cross": "bead_cross",
        "spill_ratio": "spill_ratio",
        "mouth_xy_dist (m)": "mouth_xy_dist",
        "cup_center_xy_dist (m)": "cup_center_xy_dist",
        "g_ready": "g_ready",
        "g_pour": "g_pour",
        "cost_premature_tilt": "cost_premature_tilt",
        "cost_grasp_loss": "cost_grasp_loss",
        "ADR success_progress": "adr_success_progress",
        "success_fill_ratio": "success_fill_ratio",
        "arm_joint_vel_l2_mean (rad/s)": "arm_joint_vel_l2_mean",
        "arm_joint_vel_max_mean (rad/s)": "arm_joint_vel_max_mean",
        "arm_joint_acc_l2_mean": "arm_joint_acc_l2_mean",
        "tilt_phase_arm_vel (rad/s)": "tilt_phase_arm_vel",
    }

    for label, col in metrics.items():
        if col in tail.columns:
            vals = tail[col].dropna()
            if len(vals) > 0:
                print(f"  {label:<30s}: {vals.mean():.4f}  (std={vals.std():.4f})")

    print("\n--- 전체 학습 기간 최고값 ---")
    high_metrics = ["rewards", "bead_in_target", "g_ready", "g_pour"]
    for col in high_metrics:
        if col in df.columns:
            peak_iter = df[col].idxmax()
            peak_val = df[col].max()
            print(f"  {col:<25s}: {peak_val:.4f}  @ iter {peak_iter}")

    print("\n--- 거리 메트릭 4분위 (mouth_xy_dist) ---")
    if "mouth_xy_dist" in df.columns:
        q = df["mouth_xy_dist"].dropna().quantile([0.25, 0.5, 0.75])
        print(f"  Q1={q[0.25]:.4f}  Q2={q[0.50]:.4f}  Q3={q[0.75]:.4f}")

    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TensorBoard 이벤트 파싱 및 시각화")
    parser.add_argument(
        "--events",
        type=str,
        default=(
            "/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/"
            "5g_pour_right_v3/test2/summaries/events.out.tfevents.1775803000.user"
        ),
        help="events.out.tfevents 파일 경로",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_pour_right_v3/test2/analysis",
        help="출력 디렉토리",
    )
    parser.add_argument("--smooth", type=int, default=20, help="이동평균 윈도우 크기")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 이벤트 파일 로드: {args.events}")
    data = load_events(args.events)
    print(f"     총 {len(data)}개 태그 로드 완료")

    print("[2/4] DataFrame 구성 (/iter suffix 기준)")
    df = build_master_df(data, suffix="/iter")
    print(f"     컬럼 수: {len(df.columns)}, 행 수: {len(df)}")

    # CSV 저장
    csv_path = outdir / "metrics.csv"
    df.to_csv(csv_path)
    print(f"     CSV 저장: {csv_path}")

    print("[3/4] 그래프 생성")
    plot_overview(df, outdir)
    plot_reward_components(df, outdir)
    plot_stability(df, outdir)
    plot_pour_quality(df, outdir)
    plot_losses(df, outdir, data)
    plot_arm_velocity(df, outdir)

    print("[4/4] 통계 요약")
    print_summary(df)

    print(f"\n완료. 출력 디렉토리: {outdir}")


if __name__ == "__main__":
    main()
