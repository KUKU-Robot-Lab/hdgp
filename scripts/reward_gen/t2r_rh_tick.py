#!/usr/bin/env python3
"""t2r_rh 틱 요약 — status.json → 사용자 지정 보고 형식의 숫자 한 줄 + 상위 보상항 + mimic 폭주 검사."""
import glob, json, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parents[2]
st = json.load(open(H / "reward_gen/pour_bi_rh/LOOP_STATE.json"))
label, it = st["label"], f"reward_gen/pour_bi_rh/iter_{st['iter']:02d}"
subprocess.run([sys.executable, str(H / "scripts/reward_gen/t2r_rh_round.py"), "status", "--label", label, "--iter", it],
               capture_output=True, timeout=400)
d = json.load(open(H / it / "status.json")); m = d["metrics"]
g = lambda k, s=1: (m[k]["last"] * s, m[k]["max"] * s) if k in m else (float("nan"),) * 2
print("verdict", d["verdict"], "epoch", d["epoch"], "hours", d["hours"], "alive", d["alive"])
for k, s, u in [("task/episode_success", 100, "%"), ("task/src_grasped", 100, "%"), ("task/rcv_grasped", 100, "%"),
                ("task/src_cup_lift", 100, "cm"), ("task/rcv_cup_lift", 100, "cm"), ("task/src_tilt_deg", 1, "°"),
                ("task/aim_dist", 1, "m"), ("task/nested_rate", 100, "%"), ("task/cup_collision_rate", 100, "%"),
                ("task/src_hand_foreign_rate", 100, "%"), ("task/rcv_hand_foreign_rate", 100, "%"), ("adr/progress", 1, ""),
                ("bead/in_target", 100, "%"), ("bead/spill", 100, "%"), ("done/drop", 100, "%"), ("reward/total", 1, ""),
                ("ctrl/mimic_err_max", 1, "rad")]:
    l, x = g(k, s); print(f"{k:28s} {l:9.3f} (max {x:.3f}) {u}")
sys.path.insert(0, str(H / "scripts/tools")); from parse_tfevents import load_tfevents  # noqa: E402
f = sorted(glob.glob(str(H / f"log/pour_fabric_mimic/mirror/{label}*/summaries/events.out.tfevents.*")))[-1]; dd = load_tfevents(f)
v = [x for _, x in (dd.get("ctrl/mimic_err_max") or dd.get("ctrl/mimic_err_max/iter"))]
print("mimic>10 epochs after 30:", [i for i, x in enumerate(v) if i >= 30 and x > 10][-8:], "n", len(v))
top = sorted(((m[k]["last"], k) for k in m if k.startswith("reward/") and k != "reward/total"), reverse=True)[:6]
print("top terms", [(k[7:], round(v, 3)) for v, k in top])
