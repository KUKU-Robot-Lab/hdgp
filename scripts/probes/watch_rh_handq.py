#!/usr/bin/env python3
"""rh_e1 판정선 감시 — 관절별 handq 기준.

왜 syn_close 가 아닌가: `syn_close` 는 프로필의 open→grip 기준자세에 대한 진행도인데,
그 grip(4지 1.08 · 엄지 1.20)이 09.03 컵 한 점에서 잡은 값이고 09.08 손안 판정에서
실제로 버틴 자세는 4지 0.90 이었다. 엄지는 기준이 1.57→1.20 이라 **대향할수록 값이
떨어져** 정책이 엄지를 2.094 로 몰았을 때 0 을 찍었다. 기준자세 없는 관절 실측으로 본다.

판정선(09.08 사용자 합의):
  1. done/tipped == 0            — 변경이 먹었는가
  2. handq/q_index_1 등 4지가 0 에서 올라오는가 — 닫기 시도가 생겼는가
  3. task/lifted_frac > 0        — 최종 판정
  4. done/fell · out_xy          — 넘어진 물체가 굴러 나가는 부작용
"""
import glob
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1] if len(sys.argv) > 1 else "rh_e1"
BASE = "/home/user/rl_ws/hdgp/log/rl_games/open-rh/right/grasp-fj-rh"

FLEX = ["index_1", "middle_1", "ring_1", "pinky_1"]
KEYS = (["done/tipped", "done/fell", "done/out_xy", "episode_lengths",
         "task/tilt_deg", "task/ft_dist_mean", "task/dz", "task/lifted_frac",
         "reward/total", "reward/lift"]
        + [f"handq/q_{j}" for j in ["thumb_1", "thumb_2"] + FLEX]
        + [f"handq/cmd_{j}" for j in FLEX])


def load(run):
    fs = sorted(glob.glob(f"{BASE}/{run}/summaries/*"))
    if not fs:
        raise SystemExit(f"summaries 없음: {run}")
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def band(ea, key, a, b):
    try:
        e = ea.Scalars(f"{key}/iter")
    except Exception:
        return float("nan")
    st = np.array([x.step for x in e])
    v = np.array([x.value for x in e])
    m = (st >= a) & (st <= b)
    return float(v[m].mean()) if m.any() else float("nan")


def last_step(ea):
    return max(x.step for x in ea.Scalars("reward/total/iter"))


def main():
    ea = load(RUN)
    n = last_step(ea)
    edges = [1, 200, 700, 1500, 2500, 4000, 6000, 10000, 20000]
    bands = [(a, b) for a, b in zip(edges, edges[1:]) if a <= n]
    if not bands:
        bands = [(1, n)]
    bands[-1] = (bands[-1][0], min(bands[-1][1], n))

    print(f"=== {RUN} · epoch {n} ===")
    hdr = f"{'지표':26s}" + "".join(f"{a}-{b}".rjust(12) for a, b in bands)
    print(hdr)
    for k in KEYS:
        row = "".join(f"{band(ea, k, a, b):12.4f}" for a, b in bands)
        print(f"{k:26s}{row}")

    # 판정
    tip = band(ea, "done/tipped", bands[-1][0], bands[-1][1])
    flex = np.mean([band(ea, f"handq/q_{j}", bands[-1][0], bands[-1][1]) for j in FLEX])
    lift = band(ea, "task/lifted_frac", bands[-1][0], bands[-1][1])
    fell = band(ea, "done/fell", bands[-1][0], bands[-1][1])
    outxy = band(ea, "done/out_xy", bands[-1][0], bands[-1][1])
    print("\n판정선")
    print(f"  1 전도종료 제거   done/tipped {tip:.5f}      {'OK' if tip < 1e-5 else '★미적용'}")
    print(f"  2 닫기 시도       4지 실측 평균 {flex:.4f} rad  "
          f"{'OK 닫는다' if flex > 0.05 else '★여전히 편 채'}")
    print(f"  3 최종            lifted_frac {lift:.4f}      {'OK' if lift > 0.0 else '아직'}")
    print(f"  4 부작용          fell {fell:.5f} · out_xy {outxy:.5f}  "
          f"{'OK' if (fell + outxy) < 1e-4 else '★굴러 나감'}")


main()
