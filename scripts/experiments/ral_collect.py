#!/usr/bin/env python3
"""RA-L Active-Receiver ablation — 결과 수집기.

log/rl_games/open-tesol/both/pour-sensor/<label>/ 아래 각 run의 최신 TFEvents에서
핵심 지표의 최종값을 뽑아 method(M0/M2/M4)별 비교표로 출력한다.

사용:
    python3 scripts/experiments/ral_collect.py
    python3 scripts/experiments/ral_collect.py --filter C0        # label 부분일치
    python3 scripts/experiments/ral_collect.py --metrics success spill_ratio tilt_frac_110

지표 이름은 substring 매칭이라 로깅 tag 접두어가 바뀌어도 적응한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HDGP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HDGP_ROOT / "scripts" / "tools"))
from parse_tfevents import load_tfevents  # noqa: E402

LOG_ROOT = HDGP_ROOT / "log" / "rl_games" / "open-tesol" / "both" / "pour-sensor"
# 실제 로깅 tag 기준 정밀 substring (test6 확인). success=ADR 에피소드 성공률.
DEFAULT_METRICS = [
    "shaped_rewards/iter",     # shaped reward 합
    "adr_ep_success_rate",     # ADR 에피소드 성공률 (핵심)
    "success_fill_ratio",      # 수용컵 fill 비율 (transfer 대용)
    "spill_ratio",             # spill 비율
    "tilt_frac_110",           # deep tilt 도달률
]


def _latest_events(run_dir: Path) -> Path | None:
    files = sorted(run_dir.rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _final_value(rows: list) -> float | None:
    return rows[-1][1] if rows else None


def collect(label_filter: str, metrics: list[str]) -> None:
    if not LOG_ROOT.exists():
        print(f"[!] 로그 루트 없음: {LOG_ROOT}")
        return

    runs = sorted(d for d in LOG_ROOT.iterdir() if d.is_dir() and label_filter in d.name)
    if not runs:
        print(f"[!] '{label_filter}' 일치 run 없음 in {LOG_ROOT}")
        return

    print(f"# RA-L 결과 수집  ({LOG_ROOT})\n")
    # 헤더: 지표의 마지막 경로 조각만 짧게 표시
    def _short(m: str) -> str:
        return m.split("/")[0][:18]
    header = f"{'run':<20} {'step':>12}  " + "  ".join(f"{_short(m):>18}" for m in metrics)
    print(header)
    print("-" * len(header))

    for run in runs:
        ev = _latest_events(run)
        if ev is None:
            print(f"{run.name:<20} {'(no events)':>6}")
            continue
        data = load_tfevents(str(ev))
        max_step = max((rows[-1][0] for rows in data.values() if rows), default=0)
        cells = []
        for m in metrics:
            # substring 매칭: 해당 지표를 포함하는 tag 중 첫 매칭의 최종값
            match = next((t for t in sorted(data) if m in t), None)
            val = _final_value(data[match]) if match else None
            cells.append(f"{val:>18.4f}" if val is not None else f"{'n/a':>18}")
        print(f"{run.name:<20} {max_step:>12}  " + "  ".join(cells))

    print("\n(값은 최신 events의 마지막 step 값. 태그는 substring 첫 매칭.)")


def main() -> None:
    p = argparse.ArgumentParser(description="RA-L ablation 결과 수집")
    p.add_argument("--filter", default="", help="run label 부분일치 필터 (예: C0, M4)")
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                   help=f"지표 substring 목록 (기본: {DEFAULT_METRICS})")
    p.add_argument("--list-tags", action="store_true",
                   help="첫 run의 전체 tag 목록만 출력하고 종료 (지표명 확인용)")
    args = p.parse_args()

    if args.list_tags:
        runs = sorted(d for d in LOG_ROOT.iterdir() if d.is_dir()) if LOG_ROOT.exists() else []
        for run in runs:
            ev = _latest_events(run)
            if ev is None:
                continue
            print(f"# tags in {run.name}:")
            for t in sorted(load_tfevents(str(ev))):
                print(f"  {t}")
            return
        print(f"[!] run 없음 in {LOG_ROOT}")
        return

    collect(args.filter, args.metrics)


if __name__ == "__main__":
    main()
