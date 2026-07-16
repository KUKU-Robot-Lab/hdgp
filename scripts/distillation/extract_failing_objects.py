#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""teacher(grasp_v2) 최종 로그에서 실패 물체를 추출해 distillation 제외 목록을 만든다.

teacher 가 잘 못 잡는 물체는 distillation(모방)에서도 student 가 배울 수 없다.
teacher TFEvents 의 per-object 성공률(`episode_success_rate/{name}/iter`)에서 임계 미만
물체를 뽑고, right·left **합집합**(좌우 대칭)으로 `DISTILL_EXCLUDED_OBJECT_NAMES` 를 만든다.

onehot 차원(153)은 그대로 유지되고 스폰만 빠지므로 teacher 체크포인트와 호환된다
(env 의 kept_object_names_and_indices 참고).

사용:
  # right/left teacher run 디렉토리(또는 events 파일) 지정 → 제외 튜플 출력
  python3 scripts/distillation/extract_failing_objects.py \
      --right log/rl_games/open-tesol/right/grasp-v2/<run> \
      --left  log/rl_games/open-tesol/left/grasp-v2/<run> \
      --threshold 0.3

  # 확인 후 두 DISTILL cfg 에 바로 주입
  python3 scripts/distillation/extract_failing_objects.py --right <run> --left <run> --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HDGP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP_ROOT / "scripts" / "tools"))
from parse_tfevents import load_tfevents  # noqa: E402

# per-object 성공률 태그: episode_success_rate/{name}/iter  (전체는 episode_success_rate/iter → 제외)
_PEROBJ_RE = re.compile(r"^episode_success_rate/(?P<name>.+)/iter$")

_CFG_REL = "source/openarm/openarm/tesollo/{side}/grasp_v2/config/__init__.py"
_FIELD_RE = re.compile(
    r"(DISTILL_EXCLUDED_OBJECT_NAMES:\s*tuple\[str, \.\.\.\]\s*=\s*)\([^)]*\)"
)


def _resolve_events_file(path: Path) -> Path:
    """디렉토리면 summaries 하위 최신 events 파일, 파일이면 그대로."""
    if path.is_file():
        return path
    cands = sorted(path.glob("**/events.out.tfevents.*"))
    if not cands:
        raise FileNotFoundError(f"events 파일 없음: {path}")
    return cands[-1]


def per_object_final_success(
    events_file: Path, last_n: int
) -> dict[str, float]:
    """물체별 최종 성공률 (마지막 last_n 포인트 평균으로 노이즈 완화)."""
    data = load_tfevents(str(events_file))
    out: dict[str, float] = {}
    for tag, rows in data.items():
        m = _PEROBJ_RE.match(tag)
        if not m or not rows:
            continue
        vals = [v for _, v in rows[-last_n:]]
        out[m.group("name")] = sum(vals) / len(vals)
    if not out:
        raise ValueError(
            f"per-object 성공률 태그가 없다: {events_file}\n"
            "  (episode_success_rate/{name}/iter 형식 — 다물체 grasp_v2 run 인지 확인)"
        )
    return out


def failing_objects(final: dict[str, float], threshold: float) -> dict[str, float]:
    """임계 미만 물체 {name: 최종성공률}."""
    return {n: v for n, v in final.items() if v < threshold}


def _patch_cfg(side: str, excluded: tuple[str, ...]) -> Path:
    """해당 side 의 DISTILL cfg 에 제외 튜플을 주입한다."""
    cfg_path = _HDGP_ROOT / _CFG_REL.format(side=side)
    src = cfg_path.read_text(encoding="utf-8")
    if not _FIELD_RE.search(src):
        raise RuntimeError(f"DISTILL_EXCLUDED_OBJECT_NAMES 필드를 못 찾음: {cfg_path}")
    tup = "(" + ", ".join(f'"{n}"' for n in excluded) + (")" if excluded else ")")
    new = _FIELD_RE.sub(lambda m: m.group(1) + tup, src, count=1)
    cfg_path.write_text(new, encoding="utf-8")
    return cfg_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--right", type=Path, help="right teacher run 디렉토리 또는 events 파일")
    ap.add_argument("--left", type=Path, help="left teacher run 디렉토리 또는 events 파일")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="이 성공률 미만이면 제외 (기본 0.3)")
    ap.add_argument("--last-n", type=int, default=10,
                    help="최종 성공률 = 마지막 N 포인트 평균 (기본 10)")
    ap.add_argument("--write", action="store_true",
                    help="합집합을 right/left DISTILL cfg 에 바로 주입")
    args = ap.parse_args()

    if not args.right and not args.left:
        ap.error("--right 또는 --left 중 최소 하나 필요")

    per_arm: dict[str, dict[str, float]] = {}
    fails: dict[str, dict[str, float]] = {}
    for side, path in (("right", args.right), ("left", args.left)):
        if not path:
            continue
        ev = _resolve_events_file(path)
        final = per_object_final_success(ev, args.last_n)
        fail = failing_objects(final, args.threshold)
        per_arm[side] = final
        fails[side] = fail
        print(f"[{side}] {ev}")
        print(f"  물체 {len(final)}종, 실패(<{args.threshold}) {len(fail)}종")
        for n, v in sorted(fail.items(), key=lambda kv: kv[1]):
            print(f"    {n:<12} {v:.3f}")

    # 제공된 arm 들의 합집합(양쪽 주면 좌우 대칭, 한쪽만 주면 그 arm 것).
    union = tuple(sorted(set().union(*[set(f) for f in fails.values()])))
    scope = "+".join(fails.keys())
    print(f"\n제외 대상 {len(union)}종 ({scope}):")
    print("DISTILL_EXCLUDED_OBJECT_NAMES = (")
    for n in union:
        print(f'    "{n}",')
    print(")")

    if args.write:
        if not union:
            print("\n제외 대상이 없어 주입 생략.")
            return
        # 제공된 arm 의 cfg 만 패치한다(단일 arm 실행이 반대쪽을 덮어쓰지 않게).
        for side in fails:
            p = _patch_cfg(side, union)
            print(f"주입 완료: {p}")
        print("※ py_compile/테스트로 확인 후 커밋하라.")


if __name__ == "__main__":
    main()
