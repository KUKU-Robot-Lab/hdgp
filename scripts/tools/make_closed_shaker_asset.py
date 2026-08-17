#!/usr/bin/env python3
"""shaker_body 에 **바닥 플러그**를 추가한 사본 생성 — 비드를 담을 수 있게 한다.

문제(2026-08-17 실측)
--------------------
`assets/cup/shaker_body_rl.usd` 는 **양쪽이 뚫린 관**이다. 축 근처 정점이 하나도 없어
닫힌 바닥 면이 존재하지 않는다(이름 그대로 칵테일 셰이커의 *몸통*만 있고 바닥은 별개 파트):

    cup_big_rl     : r<5mm 정점 44개 (z −0.076~−0.065)   → 닫힌 바닥 있음
    shaker_body_rl : r<5mm 정점  0개, r<20mm 정점 0개     → 바닥 없음
                     최하단 5% 구간 반경 min 28.2mm

그래서 warm 수집 시 컵에 비드를 채우면 셰이커에서만 **유지율 0.000** 이었다
(cup_big 6종은 0.975~1.000). 공동 반경은 오히려 셰이커가 더 넓어서(28~38mm vs
비드 배치 최대 도달 24mm) **비드를 줄여도 해결되지 않는다** — 바닥으로 그냥 빠진다.

해결
----
메시 토폴로지를 수술하지 않고, `baseLink` 아래에 **얇은 원기둥 콜라이더**를 하나 더
붙여 바닥을 막는다. 원기둥은 convex 라 SDF 가 필요 없고, 관의 SDF 콜라이더는 그대로 둔다.

왜 안전한가
-----------
* 변환: 원본은 `object_shaker_body / baseLink / {collisions,visuals}` 전부 xformOp 없음
  (identity) → 메시 좌표가 곧 baseLink 좌표라 플러그를 메시 z 최저점에 바로 놓을 수 있다.
* 접촉 판정: grasp 의 `object_contact_filter` 가 `.../Cup/baseLink` 이므로 그 **하위**에
  두면 필터에 그대로 포함된다 → `num_contacts` 규약 불변.
* 파지 형상: 플러그 반경 29mm 는 셰이커 최대 반경 44mm 보다 작아 외곽 프로파일을
  바꾸지 않는다 → grasp 파지 학습에 영향 없음.
* 원본을 제자리 수정하지 않는다(`make_sdf_grasp_assets.py` 와 같은 원칙).

실행:
  python3 scripts/tools/make_closed_shaker_asset.py
  (pxr 만 쓴다. Isaac 없이 시스템 python 으로도 동작)

산출:
  assets/cup/shaker_closed_rl.usd
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CUP_DIR = os.path.join(_HDGP_ROOT, "assets", "cup")
_SRC = os.path.join(_CUP_DIR, "shaker_body_rl.usd")
_DST = os.path.join(_CUP_DIR, "shaker_closed_rl.usd")

_BASE_LINK = "/object_shaker_body/baseLink"
_COLLISIONS = f"{_BASE_LINK}/collisions"
_PLUG = f"{_BASE_LINK}/bottom_plug"

# 플러그 두께[m]. 얇으면 비드가 관통하고, 두꺼우면 내부 용적을 잠식한다.
_PLUG_THICKNESS = 0.004
# 플러그 반경 여유[m]. 관 최하단 외경까지 덮어 완전히 막는다.
_PLUG_RADIUS_MARGIN = 0.0005


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim 없음: {prim_path}")
    attr = prim.GetAttribute("points")
    if not attr or attr.Get() is None:
        raise RuntimeError(f"points 없음: {prim_path}")
    return np.asarray(attr.Get(), dtype=float)


def _assert_identity_transforms(stage: Usd.Stage) -> None:
    """메시 좌표 == baseLink 좌표 가정을 검증한다 (틀리면 플러그가 엉뚱한 곳에 놓인다)."""
    for path in ("/object_shaker_body", _BASE_LINK, _COLLISIONS):
        prim = stage.GetPrimAtPath(path)
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if ops:
            raise RuntimeError(
                f"{path} 에 xformOp {[o.GetOpName() for o in ops]} 가 있다 — "
                "플러그 배치 좌표 가정이 깨진다. 스크립트를 프레임 변환까지 처리하도록 고칠 것."
            )


def _has_closed_bottom(points: np.ndarray) -> bool:
    """축 근처(r<20mm) 정점 존재 여부로 닫힌 바닥을 판정한다."""
    r = np.linalg.norm(points[:, :2], axis=1)
    return bool((r < 0.020).any())


def main() -> None:
    if not os.path.isfile(_SRC):
        raise FileNotFoundError(
            f"{_SRC} 없음 — 먼저 scripts/tools/make_sdf_grasp_assets.py 를 실행할 것."
        )

    # 원본 진단 (바닥이 정말 없는지 확인 — 있으면 이 스크립트는 불필요)
    src_stage = Usd.Stage.Open(_SRC)
    _assert_identity_transforms(src_stage)
    pts = _mesh_points(src_stage, _COLLISIONS)
    z_min = float(pts[:, 2].min())
    r = np.linalg.norm(pts[:, :2], axis=1)
    bottom_band = pts[:, 2] <= z_min + 0.05 * (pts[:, 2].max() - z_min)
    r_bottom_max = float(r[bottom_band].max())
    print(
        f"[원본] z_min={z_min:.4f}  최하단 외경 {r_bottom_max*1000:.1f}mm  "
        f"닫힌바닥={_has_closed_bottom(pts)}"
    )
    if _has_closed_bottom(pts):
        raise SystemExit(
            "원본에 이미 닫힌 바닥이 있다 — 플러그가 불필요하다. 진단을 다시 볼 것."
        )

    shutil.copyfile(_SRC, _DST)
    stage = Usd.Stage.Open(_DST)

    plug_r = r_bottom_max + _PLUG_RADIUS_MARGIN
    plug_h = _PLUG_THICKNESS
    # 관의 최하단 개구를 위에서 덮는다 (플러그 상면이 z_min + h).
    plug_center_z = z_min + 0.5 * plug_h

    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(_PLUG))
    cyl.CreateRadiusAttr(plug_r)
    cyl.CreateHeightAttr(plug_h)
    cyl.CreateAxisAttr("Z")
    # extent 를 명시해 bbox 계산이 어긋나지 않게 한다.
    cyl.CreateExtentAttr(
        [(-plug_r, -plug_r, -0.5 * plug_h), (plug_r, plug_r, 0.5 * plug_h)]
    )
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set((0.0, 0.0, plug_center_z))

    # 콜라이더로 등록. 원기둥은 convex 라 SDF 가 필요 없다(관 쪽 SDF 는 건드리지 않는다).
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    cyl.GetPrim().CreateAttribute(
        "physics:approximation", Sdf.ValueTypeNames.Token
    ).Set("convexHull")

    stage.GetRootLayer().Save()
    print(
        f"[산출] {_DST}\n"
        f"  bottom_plug: r={plug_r*1000:.1f}mm  h={plug_h*1000:.1f}mm  "
        f"center_z={plug_center_z:+.4f}"
    )

    # ---- 검증: 되읽어 플러그가 실제로 바닥을 덮는지 ----
    chk = Usd.Stage.Open(_DST)
    plug = chk.GetPrimAtPath(_PLUG)
    if not plug.IsValid():
        raise RuntimeError("플러그 prim 생성 실패")
    if not plug.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("플러그에 CollisionAPI 미적용")
    got_r = float(UsdGeom.Cylinder(plug).GetRadiusAttr().Get())
    if got_r < r_bottom_max:
        raise RuntimeError(f"플러그 반경 {got_r*1000:.1f}mm < 관 외경 {r_bottom_max*1000:.1f}mm")
    # 관 쪽 SDF 가 그대로인지도 확인 (사본 생성 중 유실 방지)
    col = chk.GetPrimAtPath(_COLLISIONS)
    approx = col.GetAttribute("physics:approximation")
    if not approx or approx.Get() != "sdf":
        raise RuntimeError("관 콜라이더의 SDF 설정이 사라졌다")
    print("[검증] 플러그 CollisionAPI OK · 반경 충분 · 관 SDF 유지 OK")


if __name__ == "__main__":
    main()
