"""cocktail shaker_body 를 visdex 표준 grasp 구조로 이식.

문제(2026-07-26 GPU 실측):
  cocktail/usd/shaker_body.usda 는 defaultPrim "ShakerBody"(Xform 루트)에 직접
  PhysicsRigidBodyAPI 를 건 비표준 구조 → MultiAsset 스폰 시 rigid body 가 "/Cup" 자체가
  되어, visdex 자산(cup_big/large_*_cyl)의 "/Cup/baseLink" 와 prim 경로가 어긋난다.
  ContactSensor filter "/Cup/baseLink" 가 8종 중 shaker(1종)만 못 잡아
  "expected 8, found 7" 에러 + Xform 루트("/Cup") filter 는 GPU contact 미지원.

해결:
  fix_cup_assets.py 와 동일한 검증된 visdex 표준(defaultPrim Xform → baseLink[RB+MASS]
  → collisions[COL]+visuals)으로 shaker mesh 를 재작성 → 8종 전부 "/Cup/baseLink" 단일
  filter 로 통일.

결과: visdex_objects/USD/shaker_body/shaker_body.usd (신규). 원본 usda 는 보존.
실행: 서버 isaac 환경에서
  python scripts/tools/fix_shaker_asset.py
"""
import os

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_HDGP_ROOT, "assets", "cocktail", "usd", "shaker_body.usda")
_DST_DIR = os.path.join(_HDGP_ROOT, "assets", "visdex_objects", "USD", "shaker_body")
_DST = os.path.join(_DST_DIR, "shaker_body.usd")

_SHAKER_MASS = 0.2633  # 원본 usda physics:mass

src = Usd.Stage.Open(_SRC)
src_mesh = UsdGeom.Mesh(src.GetPrimAtPath("/ShakerBody/Geom"))
pts = src_mesh.GetPointsAttr().Get()
fvc = src_mesh.GetFaceVertexCountsAttr().Get()
fvi = src_mesh.GetFaceVertexIndicesAttr().Get()
nrm = src_mesh.GetNormalsAttr().Get()
ext = src_mesh.GetExtentAttr().Get()
src_mc = UsdPhysics.MeshCollisionAPI(src_mesh.GetPrim())
approx = (
    src_mc.GetApproximationAttr().Get()
    if src_mesh.GetPrim().HasAPI(UsdPhysics.MeshCollisionAPI)
    else "sdf"
)

os.makedirs(_DST_DIR, exist_ok=True)
if os.path.exists(_DST):
    os.remove(_DST)
dst = Usd.Stage.CreateNew(_DST)
UsdGeom.SetStageMetersPerUnit(dst, 1.0)
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)

root = UsdGeom.Xform.Define(dst, "/object_shaker_body")
dst.SetDefaultPrim(root.GetPrim())
base = UsdGeom.Xform.Define(dst, "/object_shaker_body/baseLink")
UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
massapi = UsdPhysics.MassAPI.Apply(base.GetPrim())
massapi.CreateMassAttr(_SHAKER_MASS)


def _mk_mesh(path: str, collision: bool):
    m = UsdGeom.Mesh.Define(dst, path)
    m.GetPointsAttr().Set(pts)
    m.GetFaceVertexCountsAttr().Set(fvc)
    m.GetFaceVertexIndicesAttr().Set(fvi)
    if nrm:
        m.GetNormalsAttr().Set(nrm)
    if ext:
        m.GetExtentAttr().Set(ext)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr(approx if approx else "sdf")
    return m


_mk_mesh("/object_shaker_body/baseLink/collisions", True)
_mk_mesh("/object_shaker_body/baseLink/visuals", False)
dst.GetRootLayer().Save()

# ---- 검증 출력 ----
d2 = Usd.Stage.Open(_DST)
dp2 = d2.GetDefaultPrim()
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
rng = bc.ComputeWorldBound(dp2).ComputeAlignedRange()
sz = [rng.GetMax()[i] - rng.GetMin()[i] for i in range(3)]
print(f"[shaker_body] 신규 {_DST}")
print(f"  defaultPrim = {dp2.GetPath()}, metersPerUnit = {UsdGeom.GetStageMetersPerUnit(d2)}")
print(f"  collision approx = {approx}, mass = {_SHAKER_MASS}")
print(f"  bbox (meters) = ({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})")
for p in d2.Traverse():
    apis = []
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        apis.append("RB")
    if p.HasAPI(UsdPhysics.CollisionAPI):
        apis.append("COL")
    if p.HasAPI(UsdPhysics.MassAPI):
        apis.append("MASS")
    print(f"    {p.GetPath()} <{p.GetTypeName()}>{' [' + '+'.join(apis) + ']' if apis else ''}")
print("DONE")
_app.close()
