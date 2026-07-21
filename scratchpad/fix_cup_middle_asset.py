"""cup_middle → visdex 표준구조 이식 (rh56f1 파지 상한 대표, 실측 직경 7cm).

grasp_v1 다형상 envelope 리빌드(07.21): cup_middle 은 rh56f1 이 "꽉 끼게" 간신히
감싸는 파지 상한 크기. 물체군에 넣어 상한을 명시적으로 학습시킨다.

fix_cup_assets.py(cup_big) 방식 재사용:
  - metersPerUnit 정상화(1.0) — DEXTRAH cup 처럼 0.01 이면 100배 작게 스폰됨.
  - visdex 표준 구조(defaultPrim Xform → baseLink[RB+MASS] → collisions[COL]+visuals)로 재작성.

결과: visdex_objects/USD/cup_middle/cup_middle.usd (신규).
검증: bbox 가 (0.07, 0.07, 0.14) 근처여야 정상. 다르면 mesh 좌표 스케일 재확인.

★ 서버 isaacsim(pxr) 환경에서 실행:
    conda activate proj-hdgp-py311 && python scratchpad/fix_cup_middle_asset.py
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os
from pxr import Usd, UsdGeom, UsdPhysics

_VISDEX = "/home/user/rl_ws/hdgp/assets/visdex_objects/USD"
_SRC = "/home/user/rl_ws/hdgp/assets/cup/cup_middle.usd"
_DST_DIR = f"{_VISDEX}/cup_middle"
_DST = f"{_DST_DIR}/cup_middle.usd"
_EXPECT = (0.07, 0.07, 0.14)  # cup_middle_2.stl 실측 (직경 7cm, 높이 14cm)

log = []
def out(s=""):
    log.append(s)
    print(s)

# ---- 1) src mesh + metersPerUnit ----
src = Usd.Stage.Open(_SRC)
mpu = UsdGeom.GetStageMetersPerUnit(src)
out(f"[src] {_SRC}")
out(f"  metersPerUnit = {mpu}")

mesh = None
for p in src.Traverse():
    if p.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(p)
        break
assert mesh is not None, "mesh prim 을 찾지 못함"
out(f"  mesh prim = {mesh.GetPath()}")

pts = mesh.GetPointsAttr().Get()
fvc = mesh.GetFaceVertexCountsAttr().Get()
fvi = mesh.GetFaceVertexIndicesAttr().Get()
nrm = mesh.GetNormalsAttr().Get()
ext = mesh.GetExtentAttr().Get()

src_mc = UsdPhysics.MeshCollisionAPI(mesh.GetPrim())
approx = (
    src_mc.GetApproximationAttr().Get()
    if mesh.GetPrim().HasAPI(UsdPhysics.MeshCollisionAPI) else None
)

# src bbox (metersPerUnit 반영 실제 미터)
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
srng = bc.ComputeWorldBound(src.GetDefaultPrim() or mesh.GetPrim()).ComputeAlignedRange()
ssz = [(srng.GetMax()[i] - srng.GetMin()[i]) * mpu for i in range(3)]
out(f"  src bbox (meters, mpu 반영) = ({ssz[0]:.4f}, {ssz[1]:.4f}, {ssz[2]:.4f})")

# ---- 2) visdex 표준 구조로 재작성 (mpu=1.0) ----
os.makedirs(_DST_DIR, exist_ok=True)
if os.path.exists(_DST):
    os.remove(_DST)
dst = Usd.Stage.CreateNew(_DST)
UsdGeom.SetStageMetersPerUnit(dst, 1.0)
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)

root = UsdGeom.Xform.Define(dst, "/object_cup_middle")
dst.SetDefaultPrim(root.GetPrim())
base = UsdGeom.Xform.Define(dst, "/object_cup_middle/baseLink")
UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
massapi = UsdPhysics.MassAPI.Apply(base.GetPrim())
massapi.CreateMassAttr(0.15)  # visdex 물체 균일 질량 0.15kg (grasp_v2 _primitive_usd_cfg 정합)

# mesh 좌표를 실제 미터로: src mpu != 1.0 이면 pts × mpu 로 미터 정규화(dst mpu=1.0).
_sc = float(mpu)
_pts = [(p[0] * _sc, p[1] * _sc, p[2] * _sc) for p in pts] if abs(_sc - 1.0) > 1e-9 else pts
_ext = [(e[0] * _sc, e[1] * _sc, e[2] * _sc) for e in ext] if (ext and abs(_sc - 1.0) > 1e-9) else ext

def _mk_mesh(path, collision):
    m = UsdGeom.Mesh.Define(dst, path)
    m.GetPointsAttr().Set(_pts)
    m.GetFaceVertexCountsAttr().Set(fvc)
    m.GetFaceVertexIndicesAttr().Set(fvi)
    if nrm:
        m.GetNormalsAttr().Set(nrm)
    if _ext:
        m.GetExtentAttr().Set(_ext)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr(approx if approx else "convexHull")
    return m

_mk_mesh("/object_cup_middle/baseLink/collisions", True)
_mk_mesh("/object_cup_middle/baseLink/visuals", False)
dst.GetRootLayer().Save()

# ---- 3) 검증 ----
d2 = Usd.Stage.Open(_DST)
dp2 = d2.GetDefaultPrim()
bc2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
rng2 = bc2.ComputeWorldBound(dp2).ComputeAlignedRange()
sz2 = [rng2.GetMax()[i] - rng2.GetMin()[i] for i in range(3)]
out(f"\n[dst] {_DST}")
out(f"  defaultPrim = {dp2.GetPath()}, metersPerUnit = {UsdGeom.GetStageMetersPerUnit(d2)}")
out(f"  collision approx = {approx}")
out(f"  bbox (meters) = ({sz2[0]:.4f}, {sz2[1]:.4f}, {sz2[2]:.4f})")
_ok = all(abs(sz2[i] - _EXPECT[i]) < 0.02 for i in range(3))
out(f"  기대 {_EXPECT} 근처? {'OK' if _ok else '✗ 재확인 필요 (mesh 좌표 스케일)'}")
for p in d2.Traverse():
    apis = []
    if p.HasAPI(UsdPhysics.RigidBodyAPI): apis.append("RB")
    if p.HasAPI(UsdPhysics.CollisionAPI): apis.append("COL")
    if p.HasAPI(UsdPhysics.MassAPI): apis.append("MASS")
    out(f"    {p.GetPath()} <{p.GetTypeName()}>{' [' + '+'.join(apis) + ']' if apis else ''}")

with open("/home/user/rl_ws/hdgp/scratchpad/fix_cup_middle_result.txt", "w") as f:
    f.write("\n".join(log) + "\n")
print("DONE")
_app.close()
