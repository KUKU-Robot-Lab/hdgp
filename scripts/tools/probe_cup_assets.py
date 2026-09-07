"""cup 자산 정합 probe — grasp(visdex) cup vs pour(cup_big) 구조·스케일 비교.

SimulationApp 없이 pxr 단독으로 usd 스테이지를 읽어 defaultPrim / prim tree(RB·COL) /
mesh world bbox / metersPerUnit 을 덤프한다. 결과는 파일에 직접 기록(stdout 삼킴 방지).
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics

_OUT = "/home/user/rl_ws/scratchpad_probe_cup_result.txt"
_TARGETS = {
    "grasp visdex cup": "/home/user/rl_ws/hdgp/assets/multi_obj/visdex_objects/USD/cup/cup.usd",
    "pour cup_big_sdf": "/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf.usd",
    "pour cup_big_sdf_cm": "/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf_cm.usd",
    "grasp visdex large_5_cyl": "/home/user/rl_ws/hdgp/assets/multi_obj/visdex_objects/USD/large_5_cyl/large_5_cyl.usd",
}

lines = []
def out(s=""):
    lines.append(s)

def dump(label, path):
    out(f"\n===== {label} =====")
    out(f"  {path}")
    stage = Usd.Stage.Open(path)
    if stage is None:
        out("  [열기 실패]")
        return
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    dp = stage.GetDefaultPrim()
    out(f"  metersPerUnit = {mpu}")
    out(f"  defaultPrim   = {dp.GetPath() if dp else '(없음)'}")
    for prim in stage.Traverse():
        apis = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): apis.append("RB")
        if prim.HasAPI(UsdPhysics.CollisionAPI): apis.append("COL")
        if prim.HasAPI(UsdPhysics.MassAPI): apis.append("MASS")
        tag = f" [{'+'.join(apis)}]" if apis else ""
        out(f"    {prim.GetPath()}  <{prim.GetTypeName()}>{tag}")
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    if dp:
        rng = bcache.ComputeWorldBound(dp).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        size = [mx[i] - mn[i] for i in range(3)]
        size_m = [s * mpu for s in size]
        out(f"  bbox (stage unit) = ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
        out(f"  bbox (meters)     = ({size_m[0]:.4f}, {size_m[1]:.4f}, {size_m[2]:.4f})")

for label, path in _TARGETS.items():
    try:
        dump(label, path)
    except Exception as e:
        out(f"\n===== {label} =====\n  [예외] {type(e).__name__}: {e}")

with open(_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"WROTE {_OUT}")
_app.close()
