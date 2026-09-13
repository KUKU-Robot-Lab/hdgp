#!/usr/bin/env python3
"""text2reward 파이프라인 CLI (Isaac 불요).

  render  : 프롬프트 파일 생성(첫 라운드 또는 reflect 결과에서)
  ingest  : LLM 응답(response.md)에서 ```python 블록 추출 → compute_reward.py + 검증 보고
  reflect : 학습 로그(TFEvents) → 지표 표 + 이전 코드로 다음 라운드 프롬프트 생성

디렉터리 규약:  <root>/<track>/iter_NN/{prompt.md, response.md, compute_reward.py,
                validation.json, feedback.md, meta.json}

    python3 scripts/reward_gen/t2r.py render  --track pour_bi --task-file scripts/reward_gen/tasks/pour_bi.txt
    python3 scripts/reward_gen/t2r.py ingest  --iter reward_gen/pour_bi/iter_00
    python3 scripts/reward_gen/t2r.py reflect --iter reward_gen/pour_bi/iter_00 \
        --events <log>/summaries/events.out.tfevents.* [--notes notes.md]

★백엔드: 지금은 사람/Claude 세션이 prompt.md 를 읽고 response.md 를 쓴다. openai 백엔드는
  `generate` 서브커맨드로 나중에 붙인다(프롬프트·ingest 는 그대로).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP / "source" / "openarm"))
sys.path.insert(0, str(_HDGP / "scripts" / "tools"))

from openarm.agnostic.modules.t2r import prompts as P          # noqa: E402
from openarm.agnostic.modules.t2r import validator as V        # noqa: E402

DEFAULT_ROOT = _HDGP / "reward_gen"
# 피드백에 넣을 TFEvents 태그(접두사). reward/* 는 생성 코드의 항 이름을 그대로 탄다.
FEEDBACK_TAG_PREFIXES = ("reward/", "task/success_now", "task/episode_success",
                         "task/src_grasped", "task/rcv_grasped", "task/src_cup_lift",
                         "task/rcv_cup_lift", "task/src_tilt_deg", "task/aim_dist",
                         "bead/in_target", "bead/spill", "done/drop", "episode_lengths/step", "rewards/step")


def _iter_dir(root: Path, track: str, n: int) -> Path:
    return root / track / f"iter_{n:02d}"


def _next_iter(root: Path, track: str) -> int:
    d = root / track
    if not d.exists():
        return 0
    ns = [int(p.name[5:]) for p in d.glob("iter_*") if p.name[5:].isdigit()]
    return (max(ns) + 1) if ns else 0


def cmd_render(a) -> int:
    task = Path(a.task_file).read_text().strip()
    n = a.iter if a.iter is not None else _next_iter(a.root, a.track)
    d = _iter_dir(a.root, a.track, n)
    d.mkdir(parents=True, exist_ok=True)
    spec = P.PromptSpec(task=task, num_actions=a.num_actions, num_beads=a.num_beads,
                        previous_code=Path(a.prev_code).read_text() if a.prev_code else None,
                        feedback=Path(a.feedback).read_text() if a.feedback else None,
                        user_notes=Path(a.notes).read_text() if a.notes else None)
    (d / "prompt.md").write_text(P.render_prompt(spec))
    meta = {"track": a.track, "iter": n, "task": task, "created": datetime.now().isoformat(),
            "prev_code": a.prev_code, "feedback": a.feedback, "notes": a.notes}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[t2r] prompt → {d / 'prompt.md'}  (다음: response.md 를 여기 쓰고 `ingest --iter {d}`)")
    return 0


_FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)


def extract_code(response: str) -> str:
    """마지막 ```python 블록 — text2reward 와 같은 규약(사고 과정 뒤에 최종 코드)."""
    blocks = _FENCE.findall(response)
    if not blocks:
        raise SystemExit("[t2r] 응답에 ```python 블록이 없다")
    return blocks[-1].strip() + "\n"


def cmd_ingest(a) -> int:
    d = Path(a.iter)
    resp = d / (a.response or "response.md")
    if not resp.exists():
        raise SystemExit(f"[t2r] {resp} 가 없다")
    code = extract_code(resp.read_text())
    out = d / "compute_reward.py"
    out.write_text(code)
    rep = V.validate(str(out), num_fingers=a.num_fingers, num_arm=a.num_arm, num_actions=a.num_actions)
    (d / "validation.json").write_text(json.dumps(rep.as_dict(), indent=1, ensure_ascii=False))
    print(f"[t2r] code → {out}")
    print(f"[t2r] 검증 {'PASS' if rep.ok else 'FAIL'}")
    for e in rep.errors:
        print(f"   ✗ {e}")
    for w in rep.warnings:
        print(f"   ⚠ {w}")
    if rep.term_stats:
        print("   항별 드라이런(가짜 ctx, 물리적 의미 없음 — 크기 비교용):")
        for k, st in rep.term_stats.items():
            print(f"     {k:24s} mean {st['mean']:+.3f}  |max| {st['abs_max']:.3f}")
        print(f"     {'TOTAL':24s} mean {rep.total_stats['mean']:+.3f}  |max| {rep.total_stats['abs_max']:.3f}")
    print(f"   ctx 필드 사용: {', '.join(rep.fields_used)}")
    return 0 if rep.ok else 1


def cmd_reflect(a) -> int:
    from parse_tfevents import load_tfevents   # scripts/tools

    d = Path(a.iter)
    code = (d / "compute_reward.py").read_text()
    series: dict[str, list[float]] = {}
    for ev in a.events:
        data = load_tfevents(ev)
        for raw, pts in data.items():
            tag = raw[:-5] if raw.endswith("/iter") else raw   # rl_games: env extras → <tag>/iter
            if any(tag.startswith(p) for p in FEEDBACK_TAG_PREFIXES):
                series.setdefault(tag, []).extend(v for _, v in pts)
    if not series:
        raise SystemExit("[t2r] 피드백 태그가 하나도 없다 — events 경로/태그 접두사 확인")
    fb = P.render_feedback_table(series, n_points=a.points)
    (d / "feedback.md").write_text(fb)
    meta = json.loads((d / "meta.json").read_text())
    n_next = a.next_iter if a.next_iter is not None else meta["iter"] + 1
    nd = _iter_dir(a.root, meta["track"], n_next)
    nd.mkdir(parents=True, exist_ok=True)
    spec = P.PromptSpec(task=meta["task"], num_actions=a.num_actions, num_beads=a.num_beads,
                        previous_code=code, feedback=fb,
                        user_notes=Path(a.notes).read_text() if a.notes else None)
    (nd / "prompt.md").write_text(P.render_prompt(spec))
    (nd / "meta.json").write_text(json.dumps({
        "track": meta["track"], "iter": n_next, "task": meta["task"],
        "created": datetime.now().isoformat(), "prev_iter": str(d), "events": a.events,
        "notes": a.notes}, indent=1, ensure_ascii=False))
    print(f"[t2r] feedback → {d / 'feedback.md'}  ({len(series)} tags)")
    print(f"[t2r] next prompt → {nd / 'prompt.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--num-actions", type=int, default=42)
    ap.add_argument("--num-beads", type=int, default=20)
    ap.add_argument("--num-fingers", type=int, default=5)
    ap.add_argument("--num-arm", type=int, default=7)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--track", required=True)
    r.add_argument("--task-file", required=True)
    r.add_argument("--iter", type=int, default=None)
    r.add_argument("--prev-code", default=None)
    r.add_argument("--feedback", default=None)
    r.add_argument("--notes", default=None)
    r.set_defaults(fn=cmd_render)
    i = sub.add_parser("ingest")
    i.add_argument("--iter", required=True)
    i.add_argument("--response", default=None)
    i.set_defaults(fn=cmd_ingest)
    f = sub.add_parser("reflect")
    f.add_argument("--iter", required=True)
    f.add_argument("--events", nargs="+", required=True)
    f.add_argument("--points", type=int, default=10)
    f.add_argument("--next-iter", type=int, default=None)
    f.add_argument("--notes", default=None)
    f.set_defaults(fn=cmd_reflect)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
