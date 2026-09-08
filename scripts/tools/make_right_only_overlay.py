#!/usr/bin/env python3
"""오른팔 전용 USD 오버레이 생성기 — 왼팔 subtree 를 합성 단계에서 끈다.

왜 필요한가 (09.07 실측, dg5f-m)
--------------------------------
`replicate_physics=False`(다물체 소환의 전제)는 env 마다 articulation 을 통째로 PhysX 가
파싱한다. 우리는 오른팔만 학습하는데 왼팔 링크·관절이 env 마다 파싱된다. 왼쪽을 끄면
2,048 env 에서 처리량 24,155 → 39,448 fps(+63%)였고 **물리 지표는 비트 단위로 동일**했다.

방법
----
base USD 를 `subLayer` 로 물고, 왼쪽 링크·관절 prim 을 `active = false` 로 덮는다.
`active=false` 는 prim 을 **합성에서 제외**하므로 PhysX 가 아예 보지 못한다. 메시를 다시
만들지 않고 base 도 안 건드린다(양팔 트랙은 그대로 돈다).

★링크만 끄고 관절을 남기면 PhysX 가 없는 body 를 참조해 죽는다 — 관절도 같이 끈다.
★자산에서만 빼면 IsaacLab 이 `Not all regular expressions are matched!` 로 부팅에서 죽는다.
  프로필 쪽은 `robot_profiles._drop_left` 가 `init_joint_pos`·`actuator_specs` 를 함께 거른다.

사용
----
    python3 scripts/tools/make_right_only_overlay.py openarm_rh56f1_bi_rl

이름 목록은 자산 manifest(`link_order` · `kinematic_joint_order`)에서 읽는다 — 손으로 적으면
자산이 바뀔 때 조용히 갈린다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robot"
_HEADER = """#usda 1.0
(
    defaultPrim = "{default_prim}"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @./{base}@
    ]
)

# ★오른팔 전용 오버레이 — 왼팔 subtree 를 합성 단계에서 끈다(생성기:
#   scripts/tools/make_right_only_overlay.py, 이름 목록은 자산 manifest 에서 온다).
#   왜: replicate_physics=False 는 env 마다 articulation 을 통째로 PhysX 가 파싱한다.
#       왼쪽을 끄면 dg5f-m 2,048 env 실측에서 처리량 +63%, 물리 지표는 완전 동일했다.
#   ★관절도 함께 꺼야 한다 — 링크만 끄면 관절이 없는 body 를 참조해 PhysX 가 죽는다.
#   남는 것: body_root · body_link · head_* · r_al_* · r_hl_* · root_joint · body_j_base

over "{default_prim}"
{{
"""


def _yaml_list(text: str, key: str) -> list[str]:
    """manifest 의 `key:` 아래 `- 이름` 목록만 읽는다(pyyaml 없이도 돌게)."""
    out, on = [], False
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            on = True
            continue
        if on:
            if line.startswith("  - ") or line.startswith("- "):
                out.append(line.split("- ", 1)[1].strip())
            elif line.strip() and not line.startswith(" "):
                break
    if not out:
        raise SystemExit(f"manifest 에 {key} 가 없다")
    return out


def build(asset: str) -> Path:
    d = _ASSETS / asset
    manifest = (d / f"{asset}_manifest.yaml").read_text(encoding="utf-8")
    links = [n for n in _yaml_list(manifest, "link_order") if n.startswith(("l_al_", "l_hl_"))]
    joints = [n for n in _yaml_list(manifest, "kinematic_joint_order")
              if n.startswith(("l_aj_", "l_hj_"))]
    if not links or not joints:
        raise SystemExit(f"{asset}: 왼쪽 링크/관절을 못 찾았다")

    body = _HEADER.format(default_prim=asset.replace("-", "_"), base=f"{asset}.usd")
    body += f"    # --- 왼팔·왼손 링크 {len(links)}개 ---\n"
    for n in links:
        body += f'    over "{n}" (\n        active = false\n    ) {{\n    }}\n'
    body += '\n    over "joints"\n    {\n'
    body += f"        # --- 왼팔·왼손 관절 {len(joints)}개 ---\n"
    for n in joints:
        body += f'        over "{n}" (\n            active = false\n        ) {{\n        }}\n'
    body += "    }\n}\n"

    out = d / f"{asset}_right.usda"
    out.write_text(body, encoding="utf-8")
    print(f"[overlay] {out} — 링크 {len(links)} · 관절 {len(joints)} 비활성화")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("asset", help="예: openarm_rh56f1_bi_rl")
    build(ap.parse_args().asset)
    sys.exit(0)
