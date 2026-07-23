"""cup 자산 2종 grasp 호환 정상화.

probe 발견:
  - DEXTRAH cup(visdex/cup/cup.usd): metersPerUnit=0.01 → 씬(1.0)에서 100배 작게(0.9mm) 스폰.
    mesh 좌표는 이미 9cm급(bbox 0.09 stage unit) → metersPerUnit 만 1.0 으로 고치면 9cm 정상.
  - pour cup_big(cup/cup_big_sdf.usd): 스케일 정상(9×9×17.8cm)이나 구조 비표준
    (/Render 잔재 + defaultPrim=/cup_big(RB 직접), baseLink·MASS·visuals 없음).
    검증된 visdex 표준(defaultPrim Xform → baseLink[RB+MASS] → collisions[COL]+visuals)으로 재작성.

결과: visdex_objects/USD/cup/cup.usd(수정) + visdex_objects/USD/cup_big/cup_big.usd(신규).
원본 백업 .bak.
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os
import shutil
from pxr import Usd, UsdGeom, UsdPhysics

_VISDEX = "/home/user/rl_ws/hdgp/assets/visdex_objects/USD"
_CUP_DEXTRAH = f"{_VISDEX}/cup/cup.usd"
_CUP_BIG_SRC = "/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf.usd"
_CUP_BIG_DST_DIR = f"{_VISDEX}/cup_big"
_CUP_BIG_DST = f"{_CUP_BIG_DST_DIR}/cup_big.usd"

log = []
def out(s=""): log.append(s);

# ---- 1) DEXTRAH cup: metersPerUnit 0.01 → 1.0 ----
if not os.path.exists(_CUP_DEXTRAH + ".bak"):
    shutil.copy(_CUP_DEXTRAH, _CUP_DEXTRAH + ".bak")
s = Usd.Stage.Open(_CUP_DEXTRAH)
old_mpu = UsdGeom.GetStageMetersPerUnit(s)
UsdGeom.SetStageMetersPerUnit(s, 1.0)
s.GetRootLayer().Save()
# 검증
s2 = Usd.Stage.Open(_CUP_DEXTRAH)
dp = s2.GetDefaultPrim()
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
rng = bc.ComputeWorldBound(dp).ComputeAlignedRange()
sz = [rng.GetMax()[i] - rng.GetMin()[i] for i in range(3)]
out(f"[DEXTRAH cup] metersPerUnit {old_mpu} → {UsdGeom.GetStageMetersPerUnit(s2)}")
out(f"  bbox (meters) = ({sz[0]*1.0:.4f}, {sz[1]*1.0:.4f}, {sz[2]*1.0:.4f})")

# ---- 2) cup_big: 검증된 visdex 표준 구조로 mesh 이식 ----
src = Usd.Stage.Open(_CUP_BIG_SRC)
src_mesh = UsdGeom.Mesh(src.GetPrimAtPath("/cup_big/cup_big"))
pts = src_mesh.GetPointsAttr().Get()
fvc = src_mesh.GetFaceVertexCountsAttr().Get()
fvi = src_mesh.GetFaceVertexIndicesAttr().Get()
nrm = src_mesh.GetNormalsAttr().Get()
ext = src_mesh.GetExtentAttr().Get()
# 원본 collision approximation
src_mc = UsdPhysics.MeshCollisionAPI(src_mesh.GetPrim())
approx = src_mc.GetApproximationAttr().Get() if src_mc.GetPrim().HasAPI(UsdPhysics.MeshCollisionAPI) else None

os.makedirs(_CUP_BIG_DST_DIR, exist_ok=True)
if os.path.exists(_CUP_BIG_DST):
    os.remove(_CUP_BIG_DST)
dst = Usd.Stage.CreateNew(_CUP_BIG_DST)
UsdGeom.SetStageMetersPerUnit(dst, 1.0)
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)

root = UsdGeom.Xform.Define(dst, "/object_cup_big")
dst.SetDefaultPrim(root.GetPrim())
base = UsdGeom.Xform.Define(dst, "/object_cup_big/baseLink")
UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
massapi = UsdPhysics.MassAPI.Apply(base.GetPrim())
massapi.CreateMassAttr(0.1)  # 컵 100g

def _mk_mesh(path, collision):
    m = UsdGeom.Mesh.Define(dst, path)
    m.GetPointsAttr().Set(pts)
    m.GetFaceVertexCountsAttr().Set(fvc)
    m.GetFaceVertexIndicesAttr().Set(fvi)
    if nrm: m.GetNormalsAttr().Set(nrm)
    if ext: m.GetExtentAttr().Set(ext)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr(approx if approx else "sdf")
    return m

_mk_mesh("/object_cup_big/baseLink/collisions", True)
_mk_mesh("/object_cup_big/baseLink/visuals", False)
dst.GetRootLayer().Save()

# 검증
d2 = Usd.Stage.Open(_CUP_BIG_DST)
dp2 = d2.GetDefaultPrim()
bc2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
rng2 = bc2.ComputeWorldBound(dp2).ComputeAlignedRange()
sz2 = [rng2.GetMax()[i] - rng2.GetMin()[i] for i in range(3)]
out(f"\n[cup_big] 신규 {_CUP_BIG_DST}")
out(f"  defaultPrim = {dp2.GetPath()}, metersPerUnit = {UsdGeom.GetStageMetersPerUnit(d2)}")
out(f"  collision approx = {approx}")
out(f"  bbox (meters) = ({sz2[0]:.4f}, {sz2[1]:.4f}, {sz2[2]:.4f})")
for p in d2.Traverse():
    apis = []
    if p.HasAPI(UsdPhysics.RigidBodyAPI): apis.append("RB")
    if p.HasAPI(UsdPhysics.CollisionAPI): apis.append("COL")
    if p.HasAPI(UsdPhysics.MassAPI): apis.append("MASS")
    out(f"    {p.GetPath()} <{p.GetTypeName()}>{' ['+'+'.join(apis)+']' if apis else ''}")

with open("/home/user/rl_ws/scratchpad_fix_cup_result.txt", "w") as f:
    f.write("\n".join(log) + "\n")
print("DONE")
_app.close()
