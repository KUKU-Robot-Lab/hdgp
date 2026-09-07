"""로봇 자산의 링크별 충돌 근사를 매니페스트로 기록한다.

왜 필요한가: physics 레이어가 바이너리(usdc)라 정적 계약이 직접 읽을 수 없다. 자산을
바꿀 때마다 이 스크립트를 한 번 돌려 매니페스트를 갱신하면, Isaac 없이 도는 계약
테스트가 "파지 부위가 hull 로 바뀌지 않았는가"를 검증할 수 있다.

★좌 그리퍼 3개가 convexDecomposition 이어야 하는 이유는 그것이 **파지 도구**이기 때문이다.
  hull 이 되면 개구(84.5 mm 실측)와 파지 대역(10~85 mm)이 무효가 된다.

실행:
    ./isaaclab.sh -p scripts/assets_tools/write_collider_manifest.py <asset_dir_name>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("asset", help="assets/robot/ 아래 자산 디렉토리명")
parser.add_argument("--assets-root", default="assets/robot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from pxr import Usd, UsdPhysics  # noqa: E402

root = Path(args.assets_root) / args.asset
stage_path = next(root.glob("*.usd"), None)
if stage_path is None:
    raise SystemExit(f"루트 usd 를 못 찾았다: {root}")

stage = Usd.Stage.Open(str(stage_path))
links: dict[str, str] = {}
# ★instance proxy 를 순회해야 한다 — 자산이 instanceable 이라 기본 Traverse 로는 안 보인다.
for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        continue
    approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "(none)"
    parts = str(prim.GetPath()).split("/")
    link = parts[2] if len(parts) > 2 else str(prim.GetPath())
    links[link] = approx

out = root / "collider_manifest.json"
counts: dict[str, int] = {}
for v in links.values():
    counts[v] = counts.get(v, 0) + 1
out.write_text(json.dumps({"asset": args.asset, "counts": counts,
                           "links": dict(sorted(links.items()))},
                          indent=2, ensure_ascii=False) + "\n")
print(f"매니페스트 기록: {out}")
print(f"  {counts}")
app.close()
