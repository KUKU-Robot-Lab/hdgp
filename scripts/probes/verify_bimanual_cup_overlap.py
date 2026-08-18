#!/usr/bin/env python3
"""[both/pour_v1] 좌/우 warm 뱅크의 **실제 페어 겹침**을 측정한다 (Isaac 불필요).

pour_v1 은 좌/우 뱅크를 **독립 샘플링**하므로, 가능한 모든 조합이 실제 초기 분포다.
좌우 대칭을 가정하고 한쪽만 보면 꼬리를 놓친다 — 실제 값을 교차해서 잰다.

겹침 기준은 pour_v1 리셋의 재추첨 판정과 같은 규약을 쓴다:
    중심거리(xy) < 소스내경 + 타겟내경 + 2×벽두께 + 최소여유

왜 필요한가 (2026-08-18)
------------------------
grasp_v1 이 스폰 박스를 재설정하면서(2f33e2c) y 폭이 ±0.06 → ±0.10 으로 1.67배
넓어졌다. warm 컵 분포가 따라 넓어지므로 좌우 겹침이 증가한다. 뱅크를 새로 만들
때마다 이 스크립트로 실측하고, 필요하면 `left_right_cup_min_gap_m` /
`object_spawn_y_center` 를 조정할 것. 추정으로 값을 바꾸지 말 것.

사용:
    python3 scripts/probes/verify_bimanual_cup_overlap.py
    python3 scripts/probes/verify_bimanual_cup_overlap.py --pour_specs 0 1 2 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

_HDGP_ROOT = Path(__file__).resolve().parents[2]

# pour_v1 cfg 기본값과 일치시킨다(바꿀 때 양쪽을 같이 볼 것).
DEFAULT_INNER_R = 0.041   # source/target_inner_radius
DEFAULT_WALL = 0.007      # cup_wall_thickness_m
DEFAULT_GAP = 0.005       # left_right_cup_min_gap_m
# grasp _ACTIVE_OBJECT_SPECS 의 컵 스케일 (pour 가 쓰는 spec 0-3)
SPEC_SCALE = {0: 1.00, 1: 1.00, 2: 0.90, 3: 1.05}


def _load(path: Path):
    with h5py.File(path, "r") as h:
        g = h["warm_states"]
        xy = np.asarray(g["cup_pos_local"][:, :2])
        spec = np.asarray(g["object_spec_idx"]) if "object_spec_idx" in g else None
        prov = {k: v for k, v in h.attrs.items() if str(k).startswith("prov/")}
    return xy, spec, prov


def _radii(spec, n: int, inner_r: float, wall: float) -> np.ndarray:
    if spec is None:
        return np.full(n, inner_r + wall)
    scale = np.array([SPEC_SCALE.get(int(v), 1.0) for v in spec])
    return inner_r * scale + wall


def _report(rxy, rr, lxy, lr, gap: float, tag: str) -> None:
    """전 조합을 블록으로 훑어 겹침률·최소거리를 낸다(메모리 절약)."""
    total = bad = 0
    d_min = np.inf
    slack_min = np.inf
    for i in range(0, len(rxy), 256):
        blk = rxy[i : i + 256]
        blk_r = rr[i : i + 256]
        d = np.linalg.norm(blk[:, None, :] - lxy[None, :, :], axis=-1)
        thr = blk_r[:, None] + lr[None, :] + gap
        bad += int((d < thr).sum())
        total += d.size
        d_min = min(d_min, float(d.min()))
        slack_min = min(slack_min, float((d - thr).min()))
    pct = 100.0 * bad / max(total, 1)
    print(
        f"  [{tag}] 조합 {total:,}  겹침 {pct:.3f}%  "
        f"최소 중심거리 {d_min * 1000:.1f}mm  최소 여유 {slack_min * 1000:+.1f}mm"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--right", type=str, default=str(_HDGP_ROOT / "data/grasp_warm_tesollo_right.hdf5"))
    ap.add_argument("--left", type=str, default=str(_HDGP_ROOT / "data/grasp_warm_tesollo_left.hdf5"))
    ap.add_argument("--inner_r", type=float, default=DEFAULT_INNER_R)
    ap.add_argument("--wall", type=float, default=DEFAULT_WALL)
    ap.add_argument("--min_gap", type=float, default=DEFAULT_GAP)
    ap.add_argument("--pour_specs", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="pour 가 실제로 쓰는 컵 spec (이 부분집합만 따로 한 번 더 잰다)")
    args = ap.parse_args()

    rxy, rspec, rprov = _load(Path(args.right))
    lxy, lspec, lprov = _load(Path(args.left))

    for tag, prov in (("우", rprov), ("좌", lprov)):
        if prov:
            ck = str(prov.get("prov/checkpoint", "?"))
            sha = str(prov.get("prov/checkpoint_sha256", ""))[:12]
            print(f"[{tag}] ckpt={Path(ck).name}  sha={sha}…")
        else:
            print(f"[{tag}] ⚠ provenance 없음 — 어떤 체크포인트로 만든 뱅크인지 알 수 없다")

    rr = _radii(rspec, len(rxy), args.inner_r, args.wall)
    lr = _radii(lspec, len(lxy), args.inner_r, args.wall)

    print(f"우팔 n={len(rxy)}  y {rxy[:, 1].mean():+.4f}±{rxy[:, 1].std():.4f}"
          f"  범위 [{rxy[:, 1].min():+.3f}, {rxy[:, 1].max():+.3f}]")
    print(f"좌팔 n={len(lxy)}  y {lxy[:, 1].mean():+.4f}±{lxy[:, 1].std():.4f}"
          f"  범위 [{lxy[:, 1].min():+.3f}, {lxy[:, 1].max():+.3f}]")
    print(f"x  우 {rxy[:, 0].mean():+.4f}  좌 {lxy[:, 0].mean():+.4f}")
    print(f"좌우 간격(평균) {abs(lxy[:, 1].mean() - rxy[:, 1].mean()) * 1000:.1f} mm")
    print("=== 실제 페어 겹침 (대칭 가정 아님) ===")
    _report(rxy, rr, lxy, lr, args.min_gap, "전체")

    if args.pour_specs and rspec is not None and lspec is not None:
        keep = set(args.pour_specs)
        rm = np.array([int(v) in keep for v in rspec])
        lm = np.array([int(v) in keep for v in lspec])
        if rm.any() and lm.any():
            _report(rxy[rm], rr[rm], lxy[lm], lr[lm], args.min_gap,
                    f"pour spec {sorted(keep)}")
            print(f"  (우 {int(rm.sum())}개 · 좌 {int(lm.sum())}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
