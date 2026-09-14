#!/usr/bin/env python3
"""text2reward CLI — grasp_fj_t2r 트랙(단일 팔 · DG-5F full-joint 손 · 인벨롭 파지). Isaac 불요.

  render : 과제 문장(+선택: 이전 코드·피드백·메모) → reward_gen/<track>/iter_NN/prompt.md
  ingest : response.md 의 마지막 ```python 블록 → compute_reward.py + validation.json

디렉터리 규약:  reward_gen/<track>/iter_NN/{prompt.md, response.md, compute_reward.py, validation.json, meta.json}

    python3 scripts/reward_gen/t2r_fj.py render --track grasp_fj_envelope \
        --task-file scripts/reward_gen/tasks/grasp_fj_envelope.txt
    python3 scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/grasp_fj_envelope/iter_00

★붓기 트랙 CLI(`t2r.py`)와 코드를 공유하지 않는다 — 컨텍스트가 다르고 그쪽은 다른 세션의 루프가 쓴다.
★생성기: prompt.md 경로만 받은 새 에이전트가 response.md 를 쓴다(이 세션의 설계 의견을 섞지 않는다).
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

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P      # noqa: E402
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import validator as V    # noqa: E402

DEFAULT_ROOT = _HDGP / "reward_gen"
_FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)


def _next_iter(root: Path, track: str) -> int:
    d = root / track
    ns = [int(p.name[5:]) for p in d.glob("iter_*") if p.name[5:].isdigit()] if d.exists() else []
    return (max(ns) + 1) if ns else 0


def cmd_render(a) -> int:
    task = Path(a.task_file).read_text(encoding="utf-8").strip()
    n = a.iter if a.iter is not None else _next_iter(a.root, a.track)
    d = a.root / a.track / f"iter_{n:02d}"
    if (d / "prompt.md").exists() and not a.force:
        raise SystemExit(f"[t2r_fj] {d / 'prompt.md'} 가 이미 있다 — 덮으려면 --force")
    d.mkdir(parents=True, exist_ok=True)
    spec = P.PromptSpec(task=task,
                        previous_code=Path(a.prev_code).read_text(encoding="utf-8") if a.prev_code else None,
                        feedback=Path(a.feedback).read_text(encoding="utf-8") if a.feedback else None,
                        user_notes=Path(a.notes).read_text(encoding="utf-8") if a.notes else None)
    (d / "prompt.md").write_text(P.render_prompt(spec), encoding="utf-8")
    meta = {"track": a.track, "iter": n, "task": task, "created": datetime.now().isoformat(),
            "context": "openarm.agnostic.tasks.grasp_fj_t2r.t2r.context.RewardContext",
            "prev_code": a.prev_code, "feedback": a.feedback, "notes": a.notes}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[t2r_fj] prompt → {d / 'prompt.md'}  (다음: response.md 를 여기 쓰고 `ingest --iter {d}`)")
    return 0


def extract_code(response: str) -> str:
    """마지막 ```python 블록 — text2reward 규약(사고 과정 뒤에 최종 코드)."""
    blocks = _FENCE.findall(response)
    if not blocks:
        raise SystemExit("[t2r_fj] 응답에 ```python 블록이 없다")
    return blocks[-1].strip() + "\n"


def cmd_ingest(a) -> int:
    d = Path(a.iter)
    resp = d / (a.response or "response.md")
    if not resp.exists():
        raise SystemExit(f"[t2r_fj] {resp} 가 없다")
    out = d / "compute_reward.py"
    out.write_text(extract_code(resp.read_text(encoding="utf-8")), encoding="utf-8")
    rep = V.validate(str(out))
    (d / "validation.json").write_text(json.dumps(rep.as_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[t2r_fj] code → {out}")
    print(f"[t2r_fj] 검증 {'PASS' if rep.ok else 'FAIL'}")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--track", required=True)
    r.add_argument("--task-file", required=True)
    r.add_argument("--iter", type=int, default=None)
    r.add_argument("--prev-code", default=None)
    r.add_argument("--feedback", default=None)
    r.add_argument("--notes", default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_render)
    i = sub.add_parser("ingest")
    i.add_argument("--iter", required=True)
    i.add_argument("--response", default=None)
    i.set_defaults(fn=cmd_ingest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
