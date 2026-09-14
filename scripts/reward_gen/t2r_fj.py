#!/usr/bin/env python3
"""text2reward CLI — grasp_fj_t2r 트랙(단일 팔 · DG-5F full-joint 손 · 인벨롭 파지). Isaac 불요.

  render  : 과제 문장(+선택: 이전 코드·피드백·메모) → reward_gen/<track>/iter_NN/prompt.md
  ingest  : response.md 의 마지막 ```python 블록 → compute_reward.py + validation.json
  reflect : 원본 t2r interactive — (코드 · 영상으로 본 로봇 관찰 · 개선 피드백) 을 history.jsonl 에 더하고
            전 이력(+선택: 참고 지표 표)으로 iter_(NN+1)/prompt.md

디렉터리 규약:  reward_gen/<track>/iter_NN/{prompt.md, response.md, compute_reward.py, validation.json, meta.json,
                feedback.md, launch.json, status.json}

    python3 scripts/reward_gen/t2r_fj.py render --track grasp_fj_envelope \
        --task-file scripts/reward_gen/tasks/grasp_fj_envelope.txt
    ../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_fj.py ingest --iter reward_gen/grasp_fj_envelope/iter_00
    python3 scripts/reward_gen/t2r_fj.py reflect --iter reward_gen/grasp_fj_envelope/iter_00 \
        --description iter_00/observation.md --feedback iter_00/improvement.md [--events <tfevents>]

★붓기 트랙 CLI(`t2r.py`)와 코드를 공유하지 않는다 — 컨텍스트가 다르고 그쪽은 다른 세션의 루프가 쓴다.
★생성기: prompt.md 경로만 받은 새 에이전트가 response.md 를 쓴다(이 세션의 설계 의견을 섞지 않는다).
★루프 판정·기동은 `t2r_fj_round.py`, 틱 절차는 `LOOP_PROMPT_fj.md`.
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

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P      # noqa: E402
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import validator as V    # noqa: E402

DEFAULT_ROOT = _HDGP / "reward_gen"
_FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)
#: 피드백 표에 넣을 태그. reward/* = 생성 코드 항, contact/* = 보상 전용 센서 진단(머리말이 뜻을 설명한다).
#:   env extras 는 rl_games 가 `<tag>/iter` 로 쓴다. ★B 의 설계 계측(task/grasp_q* 등)은 넣지 않는다 — 생성기에 사람 설계를 흘리지 않게.
FEEDBACK_TAG_PREFIXES = ("reward/", "contact/")
FEEDBACK_TAGS_EXACT = ("ctrl/prev_ep_successes_mean", "task/successes_mean", "task/lifted_frac", "task/tol",
                       "task/tilt_deg", "done/fell", "done/tipped", "done/out_xy", "done/hand_floor",
                       "done/abnormal", "done/max_goals", "episode_lengths/step", "rewards/step")


def collect_feedback_series(data: dict[str, list[tuple[int, float]]]) -> dict[str, list[float]]:
    """`load_tfevents` 결과 → 피드백 태그만 {tag: 값열}. 순수 함수."""
    series: dict[str, list[float]] = {}
    for raw, pts in data.items():
        if raw.endswith("/iter"):
            tag = raw[:-5]
            wanted = tag.startswith(FEEDBACK_TAG_PREFIXES) or tag in FEEDBACK_TAGS_EXACT
        else:
            tag = raw
            wanted = tag in FEEDBACK_TAGS_EXACT
        if wanted:
            series.setdefault(tag, []).extend(v for _, v in pts)
    return series


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
    spec = P.PromptSpec(task=task, variant=a.variant,
                        previous_code=Path(a.prev_code).read_text(encoding="utf-8") if a.prev_code else None,
                        feedback=Path(a.feedback).read_text(encoding="utf-8") if a.feedback else None,
                        user_notes=Path(a.notes).read_text(encoding="utf-8") if a.notes else None)
    (d / "prompt.md").write_text(P.render_prompt(spec), encoding="utf-8")
    meta = {"track": a.track, "iter": n, "task": task, "variant": a.variant, "created": datetime.now().isoformat(),
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


def load_history(root: Path, track: str) -> list[dict]:
    p = root / track / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def cmd_reflect(a) -> int:
    """원본 text2reward interactive: 이번 (코드 · 로봇 관찰 · 개선 피드백) 을 이력에 더하고 전 이력으로 다음 프롬프트.

    ★관찰·피드백은 학습한 로봇을 영상으로 보고 쓴 **사용자 승인본**이다(09.14). 지표 표는 `--events` 가 있을 때 참고로만 붙는다.
    """
    d = Path(a.iter)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    track, n_cur = meta["track"], int(meta["iter"])
    entry = {"iter": n_cur, "code": (d / "compute_reward.py").read_text(encoding="utf-8"),
             "description": Path(a.description).read_text(encoding="utf-8").strip(),
             "feedback": Path(a.feedback).read_text(encoding="utf-8").strip(),
             "created": datetime.now().isoformat()}
    if not entry["description"] or not entry["feedback"]:
        raise SystemExit("[t2r_fj] 관찰·개선 피드백이 비었다 — 영상을 보고 쓴 사용자 승인본이 필요하다")
    history = sorted([h for h in load_history(a.root, track) if int(h["iter"]) != n_cur] + [entry],
                     key=lambda h: int(h["iter"]))
    metrics, n_tags = None, 0
    if a.events:
        from parse_tfevents import load_tfevents   # scripts/tools

        series: dict[str, list[float]] = {}
        for ev in a.events:
            for tag, vals in collect_feedback_series(load_tfevents(ev)).items():
                series.setdefault(tag, []).extend(vals)
        if not any(t.startswith("reward/") for t in series):
            raise SystemExit("[t2r_fj] reward/* 태그가 없다 — events 경로 확인")
        metrics, n_tags = P.render_feedback_table(series, n_points=a.points), len(series)
    n_next = a.next_iter if a.next_iter is not None else n_cur + 1
    nd = a.root / track / f"iter_{n_next:02d}"
    if (nd / "prompt.md").exists() and not a.force:
        raise SystemExit(f"[t2r_fj] {nd / 'prompt.md'} 가 이미 있다 — 덮으려면 --force")
    (a.root / track / "history.jsonl").write_text(
        "".join(json.dumps(h, ensure_ascii=False) + "\n" for h in history), encoding="utf-8")
    (d / "feedback.md").write_text(
        "## I can see from the robot that\n" + entry["description"] + "\n\n## Feedback for improvement\n"
        + entry["feedback"] + "\n" + ("\n" + metrics if metrics else ""), encoding="utf-8")
    nd.mkdir(parents=True, exist_ok=True)
    variant = meta.get("variant", "envelope")
    spec = P.PromptSpec(task=meta["task"], history=tuple(history), metrics=metrics, variant=variant)
    (nd / "prompt.md").write_text(P.render_prompt(spec), encoding="utf-8")
    (nd / "meta.json").write_text(json.dumps({
        "track": track, "iter": n_next, "task": meta["task"], "variant": variant, "created": datetime.now().isoformat(),
        "context": meta.get("context"), "prev_iter": str(d), "history_iters": [int(h["iter"]) for h in history],
        "description": a.description, "feedback": a.feedback, "events": a.events},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[t2r_fj] history → {a.root / track / 'history.jsonl'}  ({len(history)} rounds)")
    print(f"[t2r_fj] feedback → {d / 'feedback.md'}  (metrics {n_tags} tags)")
    print(f"[t2r_fj] next prompt → {nd / 'prompt.md'}")
    return 0


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
    r.add_argument("--variant", default="envelope", choices=sorted(P.VARIANTS),
                   help="환경 사실 묶음 — 트랙의 env 와 맞춘다(grasp_fj_reach → reach)")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_render)
    i = sub.add_parser("ingest")
    i.add_argument("--iter", required=True)
    i.add_argument("--response", default=None)
    i.set_defaults(fn=cmd_ingest)
    f = sub.add_parser("reflect")
    f.add_argument("--iter", required=True)
    f.add_argument("--description", required=True, help="영상에서 본 로봇 동작(사용자 승인본)")
    f.add_argument("--feedback", required=True, help="개선 피드백(사용자 승인본)")
    f.add_argument("--events", nargs="+", default=None, help="(선택) 참고 지표 표를 붙인다")
    f.add_argument("--points", type=int, default=10)
    f.add_argument("--next-iter", type=int, default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_reflect)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
