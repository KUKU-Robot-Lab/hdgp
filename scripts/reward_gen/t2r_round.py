#!/usr/bin/env python3
"""자동 루프의 라운드 판정기 — 서버 런 상태를 읽어 "continue / advance / crashed" 를 낸다.

  status  : 서버 로그·TFEvents 를 동기화해 진행 epoch·핵심 지표·항별 보상을 요약(JSON+표)
  advance : 현재 iter 의 events 로 `t2r.py reflect` 를 돌려 다음 iter 프롬프트를 만든다

    python3 scripts/reward_gen/t2r_round.py status  --label t2r_i00e1k --iter reward_gen/pour_bi/iter_00
    python3 scripts/reward_gen/t2r_round.py advance --label t2r_i00e1k --iter reward_gen/pour_bi/iter_00

판정 규칙(ROUND_POLICY — 루프 프롬프트와 같은 값을 여기 한 곳에 둔다):
  · 라운드 길이 = ROUND_EPOCHS 이상 진행 또는 ROUND_HOURS 경과 → advance
  · 단 task/episode_success 최근 평균 ≥ KEEP_SUCCESS 면 advance 하지 않고 계속(수렴 대기)
  · 로그에 Traceback/Killed → crashed
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

_HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP / "scripts" / "tools"))
from parse_tfevents import load_tfevents   # noqa: E402

ROUND_POLICY = {"ROUND_EPOCHS": 600, "ROUND_HOURS": 4.0, "KEEP_SUCCESS": 0.30, "MAX_ROUNDS": 8}
SERVER = "server"
SERVER_LOGDIR = "~/rl_ws/hdgp/log/rl_games/open-short/both/pour-fab"
LOCAL_MIRROR = _HDGP / "log" / "server_mirror" / "pour-fab"
KEY_TAGS = ("task/episode_success", "task/success_now", "task/src_grasped", "task/rcv_grasped",
            "task/src_cup_lift", "task/rcv_cup_lift", "task/src_tilt_deg", "task/aim_dist",
            "bead/in_target", "bead/spill", "done/drop", "reward/total")


def _ssh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", SERVER, cmd], capture_output=True,
                       text=True, timeout=timeout)
    return r.stdout


def run_dir_for(label: str) -> str:
    """auto-increment(-r2…) 를 포함해 가장 최근 디렉터리."""
    out = _ssh(f"ls -dt {SERVER_LOGDIR}/{label}* 2>/dev/null | head -1").strip()
    if not out:
        raise SystemExit(f"[round] 서버에 런 디렉터리가 없다: {label}")
    return out


def sync(label: str) -> Path:
    rd = run_dir_for(label)
    dst = LOCAL_MIRROR / Path(rd).name
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-aq", "--include=summaries/***", "--include=test_history.md",
                    "--exclude=*", f"{SERVER}:{rd}/", str(dst) + "/"], check=False, timeout=300)
    return dst


def tail_state(label: str) -> dict:
    log = f"~/logs/t2r/{label}.log"
    out = _ssh(f"grep -E 'epoch:|Traceback|Killed|overflow' {log} 2>/dev/null | tail -3; "
               f"echo PROCS=$(pgrep -fc 'rl_games/train[.]py'); "
               f"for p in $(pgrep -f 'rl_games/train[.]py'); do tr '\\0' '\\n' < /proc/$p/environ "
               f"| grep -q '^RUN_LABEL={label}$' && echo PID=$p; done; "
               f"stat -c %Y {log} 2>/dev/null")
    lines = out.strip().splitlines()
    st = {"epoch": None, "crashed": False, "pids": [], "log_mtime": None, "alive": False}
    for ln in lines:
        if "epoch:" in ln:
            try:
                st["epoch"] = int(ln.split("epoch:")[1].split("/")[0])
            except ValueError:
                pass
        if any(k in ln for k in ("Traceback", "Killed", "overflow")):
            st["crashed"] = True
        if ln.startswith("PID="):
            st["pids"].append(int(ln[4:]))
        if ln.strip().isdigit():
            st["log_mtime"] = int(ln.strip())
    st["alive"] = bool(st["pids"])
    return st


def summarize(events_dir: Path, last_n: int = 50) -> dict:
    files = sorted(glob.glob(str(events_dir / "summaries" / "events.out.tfevents.*")))
    if not files:
        return {}
    data = load_tfevents(files[-1])
    out = {}
    for raw, pts in data.items():
        tag = raw[:-5] if raw.endswith("/iter") else raw     # rl_games 는 env extras 를 <tag>/iter 로 쓴다
        if tag in KEY_TAGS or tag.startswith("reward/"):
            vals = [v for _, v in pts]
            if vals:
                out[tag] = {"n": len(vals), "last": round(sum(vals[-last_n:]) / len(vals[-last_n:]), 4),
                            "max": round(max(vals), 4), "first": round(vals[0], 4)}
    return out


def cmd_status(a) -> int:
    st = tail_state(a.label)
    mirror = sync(a.label)
    summ = summarize(mirror)
    meta_p = Path(a.iter) / "launch.json"
    started = json.loads(meta_p.read_text())["started"] if meta_p.exists() else None
    hours = (time.time() - started) / 3600 if started else None
    succ = summ.get("task/episode_success", {}).get("last", 0.0)
    if st["crashed"] or (not st["alive"] and (st["epoch"] or 0) < 10):
        verdict = "crashed"
    elif succ >= ROUND_POLICY["KEEP_SUCCESS"]:
        verdict = "continue(success)"
    elif (st["epoch"] or 0) >= ROUND_POLICY["ROUND_EPOCHS"] or (hours or 0) >= ROUND_POLICY["ROUND_HOURS"]:
        verdict = "advance"
    elif not st["alive"]:
        verdict = "dead"
    else:
        verdict = "continue"
    rep = {"label": a.label, "verdict": verdict, "epoch": st["epoch"], "alive": st["alive"],
           "pids": st["pids"], "hours": round(hours, 2) if hours else None, "mirror": str(mirror),
           "metrics": summ, "policy": ROUND_POLICY}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    (Path(a.iter) / "status.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


def cmd_advance(a) -> int:
    mirror = sync(a.label)
    files = sorted(glob.glob(str(mirror / "summaries" / "events.out.tfevents.*")))
    if not files:
        raise SystemExit("[round] events 없음")
    cmd = [sys.executable, str(_HDGP / "scripts" / "reward_gen" / "t2r.py"), "reflect",
           "--iter", a.iter, "--events", files[-1]]
    if a.notes:
        cmd += ["--notes", a.notes]
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("--label", required=True); s.add_argument("--iter", required=True)
    s.set_defaults(fn=cmd_status)
    v = sub.add_parser("advance"); v.add_argument("--label", required=True); v.add_argument("--iter", required=True)
    v.add_argument("--notes", default=None); v.set_defaults(fn=cmd_advance)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
