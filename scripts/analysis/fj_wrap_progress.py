#!/usr/bin/env python3
"""09.09 손 게인·감쌈 실험(A/B/C)의 판정 지표를 한 번에 읽는다.

왜 이 도구가 있나. `task/syn_close` 는 **지령**이라(grasp_fj_env.py:295) "정책이 안 닫는다"와
"손이 못 닫는다"를 3200 epoch 동안 구분하지 못했다. 그 둘은 처방이 정반대다(보상 vs 게인).
09.09 에 실측 태그를 넣었고, 이 도구는 그 둘을 **항상 나란히** 찍어 다시 헷갈리지 않게 한다.

    실현율 = 실측 / 지령   ← A(게인)가 미는 축
    지령    = syn_close_cmd ← B(감쌈 보상)가 미는 축

⚠로컬 인터프리터는 protobuf descriptor 중복 등록으로 죽는다. 이 스크립트는
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 을 스스로 세팅한다(import 전에 해야 한다).

    python3 scripts/analysis/fj_wrap_progress.py <run_dir> [run_dir ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfevents_read import read_scalars              # noqa: E402

# (태그, 표시명, 이 지표를 미는 실험) — 실측·지령을 반드시 짝으로 둔다.
ROWS = [
    ("task/syn_close_actual/iter",      "폐쇄도 실측",       "A"),
    ("task/syn_close/iter",             "폐쇄도 지령",       "B"),
    ("task/syn_realized_frac/iter",     "실현율(실측/지령)", "A"),
    ("task/syn_close_actual_seg2/iter", "  _2 뿌리 실측",    "A"),
    ("task/syn_close_cmd_seg2/iter",    "  _2 뿌리 지령",    "B"),
    ("task/syn_close_actual_seg3/iter", "  _3 중간 실측",    "A"),
    ("task/syn_close_cmd_seg3/iter",    "  _3 중간 지령",    "B"),
    ("reward/wrap/iter",                "reward/wrap",       "B"),
    ("ctrl/hand_blocked_frac/iter",     "막힌 관절 비율",    "A"),
    ("ctrl/hand_joint_err_max/iter",    "손 추종오차 최대",  "A"),
    ("task/ft_dist_mean/iter",          "손끝-물체 거리",    "-"),
    ("task/successes_mean/iter",        "성공률",            "-"),
    ("stage/lifted/iter",               "들어올림",          "-"),
    ("task/tol/iter",                   "허용오차(과녁)",    "-"),
    ("diag/act_sat_hand/iter",          "손 액션 포화",      "-"),
]


def read(run: Path) -> dict[str, list[float]]:
    """★tensorboard 를 안 쓴다 — 이 환경에서 그건 인터프리터를 죽인다(tfevents_read 주석)."""
    d = run / "summaries" if (run / "summaries").is_dir() else run
    raw = read_scalars(d)
    return {t: [v for _, v in raw[t]] for t, _, _ in ROWS if t in raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    data = {r.name: read(r) for r in args.runs}
    names = list(data)
    width = max(len(n) for n in names) + 2

    print(f"{'지표':<22}{'미는쪽':<7}" + "".join(f"{n:>{width + 12}}" for n in names))
    print("-" * (29 + (width + 12) * len(names)))
    for tag, label, pusher in ROWS:
        cells = []
        for n in names:
            v = data[n].get(tag)
            cells.append("(없음)" if not v else f"{v[0]:.4f} → {v[-1]:.4f}")
        print(f"{label:<22}{pusher:<7}" + "".join(f"{c:>{width + 12}}" for c in cells))
    print()
    for n in names:
        v = data[n].get("task/successes_mean/iter") or []
        print(f"  {n}: {len(v)} epoch")
    print("\n⚠단일 시점·단일 런 차이를 효과로 읽지 말 것 — 조건당 시드 2개가 판정의 최소 단위다.")


if __name__ == "__main__":
    main()
