"""grasp_v1 컵 자산에 실제 SDF 콜라이더를 활성화한 사본 생성 (pour 기하 정합).

문제(2026-08-16 실측):
  grasp_v1 은 visdex cup_big/shaker_body 를, pour 는 assets/cup/cup_big_sdf.usd 를 쓴다.
  **메시는 완전히 동일**(cup_big: 1765 pts / 3526 faces / bbox 동일)한데 물리 authoring 만
  다르다 — visdex 쪽은 `physics:approximation = "sdf"` 를 적어놓고도 apiSchemas 에
  `PhysxSDFMeshCollisionAPI` 가 없고 `physxSDFMeshCollision:sdfResolution` 도 없어서
  PhysX 가 SDF 를 만들지 못하고 **convexHull 로 폴백**한다(런타임 경고 실증:
  "triangle mesh collision ... falling back to convexHull approximation").

영향:
  convexHull 은 컵의 오목한 내부를 메운다 → grasp 는 "속이 찬 원통"을 잡는 법을 배우고
  pour 는 "속이 빈 컵"을 쓴다. 그 자세를 pour 에 넣으면 관통·공극으로 컵이 빠진다
  (08.16 실측 pour 통과율 4.0%).

해결:
  메시는 그대로 두고 apiSchemas + sdfResolution 만 채운 **사본**을 만든다.
  ⚠️원본을 제자리 수정하지 않는 이유: grasp_v2 가 visdex_objects/USD 를 sorted-glob 해
  148 종을 구성하므로(obs 차원이 물체 수에 종속) 그 디렉토리는 건드리면 안 된다.
  그래서 산출물을 assets/cup/ 에 둔다 — glob 대상 밖이라 grasp_v2 무영향.

결과:
  assets/cup/cup_big_rl.usd
  assets/cup/shaker_body_rl.usd
실행:
  IsaacLab/isaaclab.sh -p scripts/tools/make_sdf_grasp_assets.py
  (pxr 만 쓰므로 시스템 python 으로도 동작한다 — PhysxSchema 미사용)
"""
import os
import shutil

from pxr import Sdf, Usd

_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VISDEX = os.path.join(_HDGP_ROOT, "assets", "multi_obj", "visdex_objects", "USD")
_OUT_DIR = os.path.join(_HDGP_ROOT, "assets", "cup")

# 참조 자산(assets/cup/cup_big_sdf.usd, pour 가 실제로 쓰는 것)의 apiSchemas 를 그대로 맞춘다.
# PhysxSDFMeshCollisionAPI 가 핵심이고 나머지는 참조 자산과의 동등성 확보용.
_SDF_API_SCHEMAS = (
    "PhysxCollisionAPI",
    "PhysxConvexHullCollisionAPI",
    "PhysxConvexDecompositionCollisionAPI",
    "PhysxTriangleMeshSimplificationCollisionAPI",
    "PhysxTriangleMeshCollisionAPI",
    "PhysxSDFMeshCollisionAPI",
    "PhysxSphereFillCollisionAPI",
)
_SDF_RESOLUTION = 64  # assets/cup/cup_big_sdf.usd 와 동일

# (원본 상대경로, 콜라이더 prim 경로, 산출 파일명)
_TARGETS = (
    (os.path.join("cup_big", "cup_big.usd"), "/object_cup_big/baseLink/collisions", "cup_big_rl.usd"),
    (os.path.join("shaker_body", "shaker_body.usd"), "/object_shaker_body/baseLink/collisions", "shaker_body_rl.usd"),
)


def _enable_sdf(dst_path: str, collider_prim_path: str) -> None:
    """콜라이더 prim 에 SDF 관련 apiSchemas + sdfResolution 을 채운다(메시 불변)."""
    stage = Usd.Stage.Open(dst_path)
    prim = stage.GetPrimAtPath(collider_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"콜라이더 prim 없음: {collider_prim_path} in {dst_path}")

    existing = prim.GetMetadata("apiSchemas")
    names = list(existing.GetAddedOrExplicitItems()) if existing else []
    for schema in _SDF_API_SCHEMAS:
        if schema not in names:
            names.append(schema)
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(names))

    prim.CreateAttribute(
        "physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int
    ).Set(_SDF_RESOLUTION)
    # approximation 은 이미 "sdf" 지만 사본이 원본과 갈릴 수 있으니 명시적으로 못 박는다.
    prim.CreateAttribute(
        "physics:approximation", Sdf.ValueTypeNames.Token
    ).Set("sdf")
    stage.GetRootLayer().Save()


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    for rel_src, collider, out_name in _TARGETS:
        src = os.path.join(_VISDEX, rel_src)
        dst = os.path.join(_OUT_DIR, out_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)
        _enable_sdf(dst, collider)

        # 검증: 산출물이 참조 자산과 동일한 SDF 설정을 갖는지 되읽기.
        # ★stage 를 변수로 붙들어야 한다 — 임시 Stage 는 즉시 GC 되어 prim 이 expired 된다.
        chk_stage = Usd.Stage.Open(dst)
        chk = chk_stage.GetPrimAtPath(collider)
        schemas = list(chk.GetMetadata("apiSchemas").GetAddedOrExplicitItems())
        res = chk.GetAttribute("physxSDFMeshCollision:sdfResolution").Get()
        approx = chk.GetAttribute("physics:approximation").Get()
        assert "PhysxSDFMeshCollisionAPI" in schemas, dst
        assert res == _SDF_RESOLUTION, (dst, res)
        assert approx == "sdf", (dst, approx)
        print(f"[make_sdf] {out_name}: approximation={approx} sdfResolution={res} "
              f"apiSchemas={len(schemas)}개 → {dst}", flush=True)
    print("[make_sdf] 완료. grasp_v1 cfg 의 cup_big/shaker usd_path 를 이 파일로 교체할 것.",
          flush=True)


if __name__ == "__main__":
    main()
