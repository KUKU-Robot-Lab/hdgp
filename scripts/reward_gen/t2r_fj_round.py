#!/usr/bin/env python3
"""grasp_fj t2r 자동 루프 — 라운드 판정 · 다음 iter 프롬프트 · 서버 기동 (붓기 `t2r_round.py` 와 코드 공유 없음).

  status  : 서버 콘솔·프로세스(RUN_LABEL 대조)·TFEvents → verdict → iter_NN/status.json
  advance : 현재 iter 의 events 로 `t2r_fj.py reflect` → iter_NN/feedback.md + iter_(NN+1)/prompt.md
  launch  : 검증 PASS·push 확인 → 서버 git 동기화·HEAD 대조 → (선택) 이전 런을 **PID 로만** 종료 → run_fj.sh →
            새 PID 를 RUN_LABEL·CUDA 로 확인 → iter_NN/launch.json

    python3 scripts/reward_gen/t2r_fj_round.py status  --label fj_t2r_i00 --iter reward_gen/grasp_fj_envelope/iter_00
    python3 scripts/reward_gen/t2r_fj_round.py advance --label fj_t2r_i00 --iter reward_gen/grasp_fj_envelope/iter_00
    python3 scripts/reward_gen/t2r_fj_round.py launch  --label fj_t2r_i01 --iter reward_gen/grasp_fj_envelope/iter_01 \
        --kill-label fj_t2r_i00

판정 규칙(ROUND_POLICY — 루프 프롬프트는 이 값을 인용만 한다):
  · 로그 Traceback/Killed/overflow, 또는 죽었는데 epoch < 10 → crashed / 그 밖에 죽음 → dead
  · 성공(ctrl/prev_ep_successes_mean 최근 평균) ≥ DONE_SUCCESSES 이고 인벨롭 판정이 실패가 아님 → done_candidate
      인벨롭 = 성공 순간 손가락 ≥ DONE_FINGERS 그리고 손바닥 ≥ DONE_PALM. 지표가 없거나 아직 성공이 없으면 판정 보류(None)
      → 종료 게이트의 영상이 가른다.
  · 성공 ≥ KEEP_SUCCESSES → continue(success). 단 epoch ≥ 2×ROUND_EPOCHS 인데 인벨롭 판정이 실패면 advance(envelope)
  · epoch ≥ ROUND_EPOCHS 또는 ROUND_HOURS 경과 → advance
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

#: ★09.14 근거: 12,288 env 에서 ~340 epoch/h. fj_i2(사람 보상)는 e51 에 들기, e500 에 성공 3.8~4.0 —
#:   ROUND 1000 epoch(~3h) 안에 들기·성공이 안 서면 보상이 신호를 못 주는 것으로 본다.
#:   KEEP 2.0 = 공차 커리큘럼 게이트(tol_success_threshold)와 같은 40%. 인벨롭 = 사용자 09.13 "5손가락 개입".
ROUND_POLICY = {"ROUND_EPOCHS": 1000, "ROUND_HOURS": 4.0, "KEEP_SUCCESSES": 2.0, "DONE_SUCCESSES": 4.0,
                "DONE_FINGERS": 4.0, "DONE_PALM": 0.5, "MAX_ROUNDS": 8, "LAST_N": 50}
SERVER = "server"
SERVER_HDGP = "/home/oem/rl_ws/hdgp"
SERVER_LOGDIR = f"{SERVER_HDGP}/log/rl_games/open-short/right/grasp-fj-t2r"
SERVER_CONSOLE = "/home/oem/rl_ws/our_source/fj_t2r_runs"
TASK = "open-short_r_grasp_fj_t2r-lstm-sapg"
GPU = "0"                     # ★사용자 09.13: fj 실험은 GPU0 만(GPU1 = 붓기 루프)
SAPG_BLOCKS = 6               # run_fj.sh 상류 규약: num_envs ÷ 6 = expl_coef_block_size
LOCAL_MIRROR = _HDGP / "log" / "server_mirror" / "grasp-fj-t2r"
SUCCESS_TAG = "ctrl/prev_ep_successes_mean"
KEY_TAGS = (SUCCESS_TAG, "task/successes_mean", "task/lifted_frac", "task/tol", "task/tilt_deg",
            "task/grasp_q_at_success", "done/tipped", "done/fell", "done/out_xy", "done/hand_floor",
            "done/abnormal", "done/max_goals")
_CRASH_WORDS = ("Traceback", "Killed", "overflow")


def _ssh(cmd: str, timeout: int = 60, allow_timeout: bool = False) -> str:
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", SERVER, cmd], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        if allow_timeout:
            return ""
        raise SystemExit(f"[round_fj] ssh {timeout}s 초과: {cmd[:80]}")
    return r.stdout


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_HDGP), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


# ---------------------------------------------------------------------------- 순수 함수(테스트 대상)
def procs_cmd(label: str | None = None) -> str:
    """oem 의 train.py 프로세스를 `PID= LABEL= CUDA_VISIBLE_DEVICES=` 로 찍는 서버 명령. label 이면 그 런만.

    ★`train[.]py` — 이 명령을 실행하는 셸 자신과 매칭되지 않게(pgrep -f 는 자기 명령줄도 본다).
    """
    only = f"[ \"$l\" = '{label}' ] && " if label else ""
    return ("for p in $(pgrep -u oem -f 'rl_games/train[.]py'); do "
            "e=$(tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null); "
            "l=$(echo \"$e\" | grep '^RUN_LABEL=' | cut -d= -f2); "
            "c=$(echo \"$e\" | grep '^CUDA_VISIBLE_DEVICES='); "
            f"{only}echo \"PID=$p LABEL=$l $c\"; done")


def parse_procs(text: str) -> list[tuple[int, str, str]]:
    rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("PID="):
            kv = dict(p.split("=", 1) for p in s.split() if "=" in p)
            rows.append((int(kv["PID"]), kv.get("LABEL", ""), kv.get("CUDA_VISIBLE_DEVICES", "")))
    return rows


def parse_tail(text: str) -> dict:
    """콘솔 grep + procs_cmd 출력 → {epoch, crashed, procs, alive}."""
    st = {"epoch": None, "crashed": False, "procs": parse_procs(text), "alive": False}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("PID="):
            continue
        if s.startswith("epoch") and ":" in s:
            try:
                st["epoch"] = int(s.split(":", 1)[1].split("/")[0].replace(",", "").strip())
            except ValueError:
                pass
        elif any(w in s for w in _CRASH_WORDS):
            st["crashed"] = True
    st["alive"] = bool(st["procs"])
    return st


def summarize(data: dict, last_n: int) -> dict:
    """`load_tfevents` 결과 → 판정 태그 {n, last(최근 last_n 평균), now(마지막 점), max, first}."""
    out = {}
    for raw, pts in data.items():
        if not raw.endswith("/iter"):
            continue
        tag = raw[:-5]
        if tag in KEY_TAGS or tag.startswith(("reward/", "contact/")):
            vals = [v for _, v in pts if v == v]
            if vals:
                tail = vals[-last_n:]
                out[tag] = {"n": len(vals), "last": round(sum(tail) / len(tail), 4), "now": round(vals[-1], 4),
                            "max": round(max(vals), 4), "first": round(vals[0], 4)}
    return out


def envelope_ok(summary: dict, policy: dict = ROUND_POLICY) -> bool | None:
    """성공 순간 접촉으로 본 인벨롭 여부. 지표가 없거나(구 코드) 아직 성공이 없으면(−1) None = 영상으로."""
    f = summary.get("contact/fingers_touching_at_success", {}).get("now")
    p = summary.get("contact/palm_touching_at_success", {}).get("now")
    if f is None or p is None or f < 0 or p < 0:
        return None
    return f >= policy["DONE_FINGERS"] and p >= policy["DONE_PALM"]


def judge(summary: dict, st: dict, hours: float | None, policy: dict = ROUND_POLICY) -> tuple[str, dict]:
    epoch = st.get("epoch") or 0
    succ = summary.get(SUCCESS_TAG, {}).get("last", 0.0)
    env_ok = envelope_ok(summary, policy)
    info = {"epoch": epoch, "successes": succ, "envelope_ok": env_ok}
    if st.get("crashed") or (not st.get("alive") and epoch < 10):
        return "crashed", info
    if not st.get("alive"):
        return "dead", info
    if succ >= policy["DONE_SUCCESSES"] and env_ok is not False:
        return "done_candidate", info
    if succ >= policy["KEEP_SUCCESSES"]:
        if env_ok is False and epoch >= 2 * policy["ROUND_EPOCHS"]:
            return "advance(envelope)", info
        return "continue(success)", info
    if epoch >= policy["ROUND_EPOCHS"] or (hours or 0.0) >= policy["ROUND_HOURS"]:
        return "advance", info
    return "continue", info


def launch_command(label: str, rel_iter: str, num_envs: int, seed: int) -> str:
    """서버에서 run_fj.sh 를 백그라운드로 띄우는 한 줄.

    ★i1/i2·i00 과 같은 런처·조건. `SERVER=1` 이 없으면 서버의 ../IsaacLab 으로 가서 conda 를 안 탄다.
    ★백그라운드 detach 는 `nohup bash <파일> > log 2>&1 < /dev/null &` 만 된다(서버 conda 함정).
    """
    if num_envs % SAPG_BLOCKS:
        raise ValueError(f"num_envs {num_envs} 가 SAPG {SAPG_BLOCKS}블록으로 안 나뉜다")
    blk = num_envs // SAPG_BLOCKS
    code = f"{SERVER_HDGP}/{rel_iter}/compute_reward.py"
    return (f"mkdir -p {SERVER_CONSOLE} && cd {SERVER_HDGP} && TASK={TASK} RUN={label} GPU={GPU} "
            f"ENVS={num_envs} BLK={blk} SEED={seed} SERVER=1 NOTE='t2r {rel_iter}' "
            f"EXTRA='agent.params.config.expl_coef_block_size={blk} env.reward_code_path={code}' "
            f"nohup bash ./run_fj.sh > {SERVER_CONSOLE}/{label}.out 2>&1 < /dev/null &")


# ---------------------------------------------------------------------------- 서버 I/O
def run_dir_for(label: str) -> str:
    out = _ssh(f"ls -dt {SERVER_LOGDIR}/{label} {SERVER_LOGDIR}/{label}-* 2>/dev/null | head -1").strip()
    if not out:
        raise SystemExit(f"[round_fj] 서버에 런 디렉터리가 없다: {label}")
    return out


def sync(label: str) -> Path:
    rd = run_dir_for(label)
    dst = LOCAL_MIRROR / Path(rd).name
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-aq", "--include=summaries/***", "--exclude=*", f"{SERVER}:{rd}/", str(dst) + "/"],
                   check=False, timeout=300)
    return dst


def cmd_status(a) -> int:
    log = f"{SERVER_CONSOLE}/{a.label}.out"
    st = parse_tail(_ssh(f"grep -a -E 'epoch  |Traceback|Killed|overflow' {log} 2>/dev/null | tail -3; "
                         + procs_cmd(a.label)))
    mirror = sync(a.label)
    files = sorted(glob.glob(str(mirror / "summaries" / "events.out.tfevents.*")))
    summ = summarize(load_tfevents(files[-1]), ROUND_POLICY["LAST_N"]) if files else {}
    it = Path(a.iter)
    started = json.loads((it / "launch.json").read_text())["started"] if (it / "launch.json").exists() else None
    hours = (time.time() - started) / 3600.0 if started else None
    n_tb = max((v["n"] for v in summ.values()), default=0)
    st["epoch"] = max(st["epoch"] or 0, n_tb)      # ★콘솔은 블록 버퍼라 뒤처진다 — TB 점 개수와 큰 쪽
    verdict, info = judge(summ, st, hours)
    rep = {"label": a.label, "verdict": verdict, **info, "alive": st["alive"], "procs": st["procs"],
           "crashed_log": st["crashed"], "hours": round(hours, 2) if hours else None, "mirror": str(mirror),
           "policy": ROUND_POLICY, "metrics": summ}
    (it / "status.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("label", "verdict", "epoch", "successes", "envelope_ok", "alive", "hours")},
                     ensure_ascii=False))
    for tag in (SUCCESS_TAG, "task/lifted_frac", "task/tol", "contact/fingers_touching_at_success",
                "contact/palm_touching_at_success", "contact/fingers_touching", "contact/palm_touching",
                "reward/envelope", "reward/total", "done/tipped"):
        if tag in summ:
            m = summ[tag]
            print(f"  {tag:38s} last {m['last']:+.3f}  now {m['now']:+.3f}  max {m['max']:+.3f}")
    return 0


def cmd_advance(a) -> int:
    mirror = sync(a.label)
    files = sorted(glob.glob(str(mirror / "summaries" / "events.out.tfevents.*")))
    if not files:
        raise SystemExit("[round_fj] events 없음")
    cmd = [sys.executable, str(_HDGP / "scripts" / "reward_gen" / "t2r_fj.py"), "reflect",
           "--iter", a.iter, "--events", files[-1]]
    if a.notes:
        cmd += ["--notes", a.notes]
    return subprocess.call(cmd)


def cmd_launch(a) -> int:
    it = Path(a.iter).resolve()
    rel = it.relative_to(_HDGP).as_posix()
    v = json.loads((it / "validation.json").read_text())
    if not v.get("ok"):
        raise SystemExit(f"[round_fj] validation FAIL — 기동 안 함: {v.get('errors')}")
    _git("fetch", "-q", "origin")
    if subprocess.run(["git", "-C", str(_HDGP), "merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode:
        raise SystemExit("[round_fj] 로컬 HEAD 가 origin/main 에 없다 — 먼저 git push origin main")
    target = _git("rev-parse", "origin/main")
    if not _git("ls-tree", "--name-only", "origin/main", f"{rel}/compute_reward.py"):
        raise SystemExit(f"[round_fj] {rel}/compute_reward.py 가 origin/main 에 없다")
    dirty = _ssh(f"cd {SERVER_HDGP} && git status --porcelain --untracked-files=no").strip()
    if dirty:
        raise SystemExit(f"[round_fj] 서버 tracked dirty — 고유 작업물 확인 전 reset 하지 않는다:\n{dirty}")
    srv = _ssh(f"cd {SERVER_HDGP} && git fetch -q origin && git reset -q --hard origin/main && git rev-parse HEAD",
               timeout=180).strip().splitlines()
    if srv[-1:] != [target]:
        raise SystemExit(f"[round_fj] 서버 HEAD {srv[-1:]} ≠ origin/main {target}")

    procs = parse_procs(_ssh(procs_cmd()))
    if any(lab == a.label for _, lab, _ in procs):
        raise SystemExit(f"[round_fj] {a.label} 가 이미 돌고 있다 — 중복 기동 금지")
    on_gpu = [p for p in procs if p[2] == GPU]
    victims = [p for p in on_gpu if a.kill_label and p[1] == a.kill_label]
    strangers = [p for p in on_gpu if p not in victims]
    if strangers:
        raise SystemExit(f"[round_fj] GPU{GPU} 에 다른 학습이 있다(건드리지 않는다): {strangers}")
    if a.kill_label and not victims:
        print(f"[round_fj] 경고: 종료할 {a.kill_label} 가 GPU{GPU} 에 없다(이미 끝났을 수 있다)")
    for pid, lab, _ in victims:
        print(f"[round_fj] 종료: pid {pid} (RUN_LABEL={lab}, CUDA {GPU})")
        _ssh(f"kill {pid}")
    gone = {pid for pid, _, _ in victims}
    for _ in range(30):
        if not gone & {pid for pid, _, _ in parse_procs(_ssh(procs_cmd()))}:
            break
        time.sleep(2)
    else:
        raise SystemExit(f"[round_fj] 이전 런 {sorted(gone)} 이 60초 안에 안 죽었다 — 확인 필요")

    _ssh(launch_command(a.label, rel, a.num_envs, a.seed), timeout=20, allow_timeout=True)
    # ★기동은 되는데 ssh 가 안 돌아온다(09.14 실측) — 성공 여부는 PID 로만 판단한다.
    pid = None
    for _ in range(45):
        rows = parse_procs(_ssh(procs_cmd(a.label)))
        if rows:
            pid, _, cuda = rows[0]
            if cuda != GPU:
                raise SystemExit(f"[round_fj] 새 런이 GPU{cuda} 에 떴다 — 즉시 확인 필요 (pid {pid})")
            break
        time.sleep(2)
    if pid is None:
        raise SystemExit(f"[round_fj] 90초 안에 {a.label} 프로세스가 안 보인다 — {SERVER_CONSOLE}/{a.label}.out 확인")
    launch = {"label": a.label, "num_envs": a.num_envs, "started": time.time(), "commit": target, "gpu": int(GPU),
              "pid": pid, "console": f"{SERVER_CONSOLE}/{a.label}.out", "killed": sorted(gone)}
    (it / "launch.json").write_text(json.dumps(launch, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(launch, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--label", required=True)
    s.add_argument("--iter", required=True)
    s.set_defaults(fn=cmd_status)
    v = sub.add_parser("advance")
    v.add_argument("--label", required=True)
    v.add_argument("--iter", required=True)
    v.add_argument("--notes", default=None)
    v.set_defaults(fn=cmd_advance)
    la = sub.add_parser("launch")
    la.add_argument("--label", required=True)
    la.add_argument("--iter", required=True)
    la.add_argument("--kill-label", default=None)
    la.add_argument("--num-envs", type=int, default=12288)
    la.add_argument("--seed", type=int, default=42)
    la.set_defaults(fn=cmd_launch)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
