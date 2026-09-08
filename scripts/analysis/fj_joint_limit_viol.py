#!/usr/bin/env python3
"""손 관절이 **하드 한계를 벗어난 양**을 재는 단일 판정 도구.

왜 이 도구가 있나 (09.08). `r_hj_thumb_1` 이 접촉 중 하한 −0.384 를 넘어 −4.075 rad 까지
밀려나는 것이 관측됐다. 지령은 늘 한계 안이므로 액션·정책 문제가 아니라 물리 문제다.
그런데 이걸 고치려고 돌린 첫 스윕은 **조건당 1런**이었고, 나중에 같은 baseline 을 다시
돌리자 0.12 → 0.44 로 움직였다 — 보고했던 "효과"가 전부 편차 안이었다. 그래서 이 도구는
두 가지를 강제한다:

  1. 판정은 **스칼라 3개**뿐이다 — `max_viol`(rad) · `viol_frac` · 관절별 분해.
     한 시점의 위반율(가장 시끄러운 읽기)로 판정하지 않는다.
  2. `--baseline` 에 **같은 조건의 런 2개 이상**을 주면 편차를 먼저 출력하고,
     조건 간 차이가 그 편차를 넘는지를 같이 찍는다. 넘지 못하면 "효과 미검출"이다.

한계표는 npz 안에 있는 그 런의 `lim`(= `robot.data.joint_pos_limits`)을 쓴다.
분석 쪽에 한계를 손으로 박으면 자산이 바뀔 때 조용히 어긋난다.

입력은 `play.py --dump_hand_q` 가 남긴 npz.

    python3 scripts/analysis/fj_joint_limit_viol.py \
        --baseline base_s1.npz base_s2.npz --cond hull_s1.npz hull_s2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PASS_MAX_VIOL_RAD = 0.01     # 계획서 합격선


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    for key in ("q", "names", "lim"):
        if key not in d:
            raise SystemExit(
                f"{path}: '{key}' 가 없다 — play.py --dump_hand_q 를 한계 기록 버전으로 다시 돌릴 것")
    return {"q": d["q"], "names": [str(x) for x in d["names"]], "lim": d["lim"],
            "tgt": d["tgt"] if "tgt" in d else None}


def violation(rec: dict) -> np.ndarray:
    """(T, E, J) 각 표본이 하드 한계를 벗어난 양[rad]. 한계 안이면 0."""
    q, lim = rec["q"], rec["lim"]
    lo, hi = lim[:, 0][None, None, :], lim[:, 1][None, None, :]
    return np.maximum(np.maximum(lo - q, q - hi), 0.0)


def summarize(paths: list[Path]) -> dict:
    per_run = []
    names = None
    for p in paths:
        rec = load(p)
        names = names or rec["names"]
        if rec["names"] != names:
            raise SystemExit(f"{p}: 관절 이름 순서가 다르다 — 같은 자산의 런끼리만 비교할 것")
        v = violation(rec)
        per_run.append({
            "path": p,
            "max_viol": float(v.max()),
            "viol_frac": float((v > 1e-3).mean()),
            "by_joint": v.reshape(-1, v.shape[-1]).max(axis=0),
        })
    return {"names": names, "runs": per_run,
            "max_viol": np.array([r["max_viol"] for r in per_run]),
            "viol_frac": np.array([r["viol_frac"] for r in per_run])}


def _spread(a: np.ndarray) -> float:
    return float(a.max() - a.min()) if a.size > 1 else float("nan")


def report(tag: str, s: dict) -> None:
    print(f"\n=== {tag} ({len(s['runs'])}런)")
    for r in s["runs"]:
        print(f"  {r['path'].name:34s} max_viol={r['max_viol']:.4f} rad  viol_frac={r['viol_frac']:.4f}")
    print(f"  평균 max_viol={s['max_viol'].mean():.4f}  런간 폭={_spread(s['max_viol']):.4f}")
    worst = np.max([r["by_joint"] for r in s["runs"]], axis=0)
    order = np.argsort(-worst)[:6]
    print("  관절별 최대 이탈(상위 6):",
          "  ".join(f"{s['names'][i]}={worst[i]:.3f}" for i in order if worst[i] > 1e-4) or "없음")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", type=Path, required=True,
                    help="같은 조건의 런 2개 이상. 편차를 먼저 박기 위한 것")
    ap.add_argument("--cond", nargs="*", type=Path, default=[],
                    help="비교 조건의 런들(생략 가능)")
    ap.add_argument("--label", default="조건")
    args = ap.parse_args()

    base = summarize(args.baseline)
    report("baseline", base)
    if len(args.baseline) < 2:
        print("  ⚠ baseline 이 1런뿐이라 편차를 모른다 — 조건 비교는 근거가 되지 못한다.")

    ok = base["max_viol"].max() < PASS_MAX_VIOL_RAD
    if not args.cond:
        print(f"\n판정: max_viol {base['max_viol'].max():.4f} rad "
              f"{'<' if ok else '≥'} 합격선 {PASS_MAX_VIOL_RAD} → {'통과' if ok else '미달'}")
        return 0

    cond = summarize(args.cond)
    report(args.label, cond)
    delta = base["max_viol"].mean() - cond["max_viol"].mean()
    noise = max(_spread(base["max_viol"]), _spread(cond["max_viol"]))
    print(f"\n차이(baseline − {args.label}) = {delta:+.4f} rad · 런간 편차 = {noise:.4f} rad")
    if not np.isfinite(noise):
        print("판정: 편차를 모른다(각 조건 1런) → **효과 판정 불가**")
    elif abs(delta) <= noise:
        print("판정: 차이가 편차 안 → **효과 미검출**")
    else:
        print(f"판정: 차이가 편차보다 큼 → 효과 있음({'개선' if delta > 0 else '악화'})")
    print(f"합격선 {PASS_MAX_VIOL_RAD} rad: {args.label} max_viol "
          f"{cond['max_viol'].max():.4f} → {'통과' if cond['max_viol'].max() < PASS_MAX_VIOL_RAD else '미달'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
