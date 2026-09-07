"""전체 grasp 물체군 스케일 정합 검사 — cup 같은 metersPerUnit 버그가 다른 물체(특히
small 계열)에도 있는지 확인.

각 visdex_objects/USD/<name>/<name>.usd 의 metersPerUnit + world bbox(미터) 를 덤프.
정상 grasp 물체는 metersPerUnit=1.0, bbox 수 cm~십수 cm. 이상치(metersPerUnit≠1.0
또는 bbox 극소/극대)를 플래그.
"""
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import os
from pxr import Usd, UsdGeom

_ROOT = "/home/user/rl_ws/hdgp/assets/multi_obj/visdex_objects/USD"
_OUT = "/home/user/rl_ws/scratchpad_allscale_result.txt"

names = sorted(
    n for n in os.listdir(_ROOT)
    if os.path.isfile(os.path.join(_ROOT, n, f"{n}.usd"))
)

rows = []  # (name, mpu, bbox_m tuple, flag)
for n in names:
    path = os.path.join(_ROOT, n, f"{n}.usd")
    try:
        s = Usd.Stage.Open(path)
        mpu = UsdGeom.GetStageMetersPerUnit(s)
        dp = s.GetDefaultPrim()
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = bc.ComputeWorldBound(dp).ComputeAlignedRange()
        sz = [(rng.GetMax()[i] - rng.GetMin()[i]) * mpu for i in range(3)]
        maxdim = max(sz)
        # 이상치: metersPerUnit≠1.0, 또는 최대변 <1cm(극소) 또는 >50cm(극대)
        flag = []
        if abs(mpu - 1.0) > 1e-6: flag.append(f"MPU={mpu}")
        if maxdim < 0.01: flag.append("TINY")
        if maxdim > 0.5: flag.append("HUGE")
        rows.append((n, mpu, sz, "  ".join(flag)))
    except Exception as e:
        rows.append((n, None, None, f"ERR:{type(e).__name__}"))

lines = [f"총 {len(names)}종 스케일 검사\n"]
# 이상치 먼저
bad = [r for r in rows if r[3]]
lines.append(f"=== 이상치 {len(bad)}종 ===")
for n, mpu, sz, fl in bad:
    szs = f"({sz[0]*100:.1f},{sz[1]*100:.1f},{sz[2]*100:.1f})cm" if sz else "?"
    lines.append(f"  {n:20s} mpu={mpu} bbox={szs}  <<< {fl}")
# small 계열 전용 확인
lines.append("\n=== small 계열(제거 대상) 상세 ===")
for n, mpu, sz, fl in rows:
    if n.startswith("small_"):
        szs = f"({sz[0]*100:.1f},{sz[1]*100:.1f},{sz[2]*100:.1f})cm" if sz else "?"
        lines.append(f"  {n:20s} mpu={mpu} bbox={szs}  {fl}")
# cup 계열
lines.append("\n=== cup 계열 ===")
for n, mpu, sz, fl in rows:
    if "cup" in n:
        szs = f"({sz[0]*100:.1f},{sz[1]*100:.1f},{sz[2]*100:.1f})cm" if sz else "?"
        lines.append(f"  {n:20s} mpu={mpu} bbox={szs}  {fl}")
# 전체 크기 분포 요약
oks = [max(r[2]) for r in rows if r[2] and not r[3]]
if oks:
    oks.sort()
    lines.append(f"\n=== 정상 {len(oks)}종 최대변(cm) 분포 ===")
    lines.append(f"  min={oks[0]*100:.1f}  median={oks[len(oks)//2]*100:.1f}  max={oks[-1]*100:.1f}")

with open(_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("DONE")
_app.close()
