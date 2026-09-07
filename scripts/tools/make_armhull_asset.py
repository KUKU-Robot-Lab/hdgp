"""손을 제외한 링크만 convexHull 로 바꾼 얇은 자산 변형을 만든다.

근거(arm5080 A/B 실측, 2026-08-23):
  base32   2,005 env-steps/s  force_max 36.23N  envelope_frac 0.242
  armhull  2,280 env-steps/s  force_max 32.84N  envelope_frac 0.236
  → 처리량 +13.7%(런간 편차 1.6% 의 8배 = 실재), 접촉력·감쌈은 편차 안(변화 없음).

왜 손만 남기는가:
  · 컵에 닿는 건 손뿐 → 팔 형상이 거칠어져도 촉각 obs 에 영향이 없다.
  · 팔 자기충돌은 Fabrics `body_repulsion` 이 계획 단계에서 이미 회피한다.
  · ★손까지 hull 로 하면 접촉력이 4배(133N) 로 뛴다 — 촉각 s2r 이 깨진다. 금지.
  · ★조각수만 줄이는 것(maxConvexHulls 8/4)도 접촉력 2.2~2.3배라 부적합.

산출물은 physics 레이어만 교체한 **얇은 변형**이다. 108MB 짜리 base 는 원본을
심볼릭 링크로 가리키므로 디스크 비용이 수십 KB 에 그친다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pxr import Sdf

# 손 링크 표식. 이 부분문자열을 가진 링크만 convexDecomposition 을 유지한다.
HAND_LINK_MARKER = "_hl_"
APPROXIMATION_ATTR = "physics:approximation"
HULL = "convexHull"
DECOMPOSITION = "convexDecomposition"


def _approximation_specs(layer: Sdf.Layer) -> list[tuple[Sdf.AttributeSpec, str]]:
    """레이어 안의 모든 physics:approximation 속성과 그 소유 링크명을 모은다."""
    found: list[tuple[Sdf.AttributeSpec, str]] = []

    def walk(spec: Sdf.PrimSpec) -> None:
        for child in spec.nameChildren:
            walk(child)
        for attr in spec.properties:
            if attr.name == APPROXIMATION_ATTR:
                # 경로는 /colliders/<link>/<mesh>/... 형태 — 두 번째 요소가 링크명이다.
                parts = spec.path.pathString.split("/")
                found.append((attr, parts[2] if len(parts) > 2 else ""))

    for root in layer.rootPrims:
        walk(root)
    return found


def build(asset_dir: Path, out_dir: Path, keep_marker: str) -> int:
    """out_dir 에 얇은 armhull 변형을 만들고 hull 로 바꾼 링크 수를 돌려준다."""
    config = asset_dir / "configuration"
    if not config.is_dir():
        raise FileNotFoundError(f"configuration 디렉토리가 없다: {config}")
    physics = next(config.glob("*_physics.usd"), None)
    base = next(config.glob("*_base.usd"), None)
    if physics is None or base is None:
        raise FileNotFoundError(f"physics/base 레이어를 못 찾았다: {config}")

    out_config = out_dir / "configuration"
    out_config.mkdir(parents=True, exist_ok=True)

    # base 는 원본을 가리키는 심볼릭 링크 — 108MB 를 복제하지 않는다.
    link = out_config / base.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("../../") / asset_dir.name / "configuration" / base.name)

    # physics 외 레이어와 루트 usd 는 그대로 복사한다(참조명이 원본과 같아야 한다).
    for src in list(config.glob("*.usd")) + list(asset_dir.glob("*.usd")):
        if src.name in (base.name, physics.name):
            continue
        shutil.copy2(src, (out_config if src.parent == config else out_dir) / src.name)

    shutil.copy2(physics, out_config / physics.name)
    layer = Sdf.Layer.FindOrOpen(str(out_config / physics.name))
    if layer is None:
        raise RuntimeError(f"physics 레이어를 열지 못했다: {out_config / physics.name}")

    changed = 0
    kept = 0
    for attr, link_name in _approximation_specs(layer):
        if keep_marker in link_name:
            kept += 1
            continue
        if attr.default != HULL:
            attr.default = HULL
            changed += 1
    layer.Save()

    print(f"  {out_dir.name}: hull {changed}개 · 손 보존 {kept}개")
    if changed == 0:
        print("  ⚠ 바뀐 링크가 없다 — keep_marker 가 전부를 잡았거나 이미 hull 이다.")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", help="자산 디렉토리명 (assets/robot/ 아래)")
    ap.add_argument("--assets-root", default="assets/robot")
    ap.add_argument("--suffix", default="_armhull")
    ap.add_argument("--keep-marker", default=HAND_LINK_MARKER,
                    help="이 부분문자열을 가진 링크는 decomposition 유지 (기본 '_hl_')")
    args = ap.parse_args()

    root = Path(args.assets_root)
    asset_dir = root / args.asset
    if not asset_dir.is_dir():
        print(f"자산이 없다: {asset_dir}", file=sys.stderr)
        return 1
    out_dir = root / f"{args.asset}{args.suffix}"
    print(f"armhull 변형 생성: {asset_dir} → {out_dir}")
    build(asset_dir, out_dir, args.keep_marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
