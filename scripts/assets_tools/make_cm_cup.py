"""cup_big_sdf.usd (미터 저작) → cup_big_sdf_cm.usd (정점 ×100, cm 스케일).

xform 스케일이 아니라 메시 정점 자체를 ×100 구워, metersPerUnit=0.01 씬에서
참조해도 SDF 충돌이 유효하게 만든다(스케일된 xform 은 SDF 를 깨뜨림).
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import os
import shutil

from pxr import Gf, Usd, UsdGeom, Vt

_SRC = "/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf.usd"
_DST = "/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf_cm.usd"
_S = 100.0

shutil.copy(_SRC, _DST)
stage = Usd.Stage.Open(_DST)

n_mesh = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(p[0] * _S, p[1] * _S, p[2] * _S) for p in pts]))
            n_mesh += 1
        ext = mesh.GetExtentAttr().Get()
        if ext:
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(e[0] * _S, e[1] * _S, e[2] * _S) for e in ext]))

stage.GetRootLayer().Save()
zs_lo, zs_hi = None, None
for prim in Usd.Stage.Open(_DST).Traverse():
    if prim.IsA(UsdGeom.Mesh):
        pa = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pa:
            zs = [p[2] for p in pa]
            zs_lo, zs_hi = min(zs), max(zs)
print(f"[make_cm_cup] 저장: {_DST}  (mesh {n_mesh}개, z[{zs_lo:.2f},{zs_hi:.2f}] cm)", flush=True)
os._exit(0)
