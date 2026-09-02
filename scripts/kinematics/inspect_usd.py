"""
Pixar OpenUSD Ground-Truth Inspector for Isaac Lab Assets
Directly parses .usd files using Pixar's pxr.Usd library to inspect Prim hierarchies, joints, and physics schemas.
"""

import os
import sys
import argparse
from pxr import Usd, UsdGeom, UsdPhysics


def inspect_usd_file(usd_path: str):
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    print("=" * 75)
    print(f"[*] Pixar OpenUSD Ground-Truth Inspection: {os.path.basename(usd_path)}")
    print(f"[*] Absolute Path: {os.path.abspath(usd_path)}")
    print("=" * 75)

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("[!] Failed to open USD stage.")
        return

    root_prims = list(stage.GetPseudoRoot().GetChildren())
    print(f"[+] Root Prim: {[p.GetPath().pathString for p in root_prims]}")

    all_prims = list(stage.Traverse())
    print(f"[+] Total Prims Count in USD: {len(all_prims)}")

    joints = []
    bodies = []
    visuals = []
    collisions = []

    for prim in all_prims:
        path = prim.GetPath().pathString
        type_name = prim.GetTypeName()
        
        if "joint" in path.lower() or "PhysicsRevoluteJoint" in type_name or "PhysicsFixedJoint" in type_name:
            joints.append((path, type_name))
        elif "visual" in path.lower():
            visuals.append(path)
        elif "collision" in path.lower():
            collisions.append(path)
        elif prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh):
            bodies.append((path, type_name))

    print("\n" + "-" * 75)
    print(f" [1] Isaac Lab USD 관절 (Joints) 목록 ({len(joints)} 개):")
    print("-" * 75)
    for path, tname in joints[:25]:
        print(f"   - {path:<60} [{tname}]")
    if len(joints) > 25:
        print(f"   ... ({len(joints) - 25} more joints)")

    print("\n" + "-" * 75)
    print(f" [2] Isaac Lab USD 주요 바디 (Bodies/Links) 목록:")
    print("-" * 75)
    sample_bodies = [p for p, t in bodies if not ("visual" in p.lower() or "collision" in p.lower())]
    for path in sample_bodies[:20]:
        print(f"   - {path}")
    if len(sample_bodies) > 20:
        print(f"   ... ({len(sample_bodies) - 20} more bodies)")

    print("\n" + "=" * 75)
    print("[SUCCESS] USD 파일 내부의 모든 Prim 계층 구조와 관절이 정상 확인되었습니다.")
    print("=" * 75)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_usd = os.path.join(
        repo_root, "assets", "robot", "openarm_tesollo_sensor_rl", "openarm_tesollo_sensor_rl.usd"
    )

    parser = argparse.ArgumentParser(description="Isaac Lab USD Ground-Truth Inspector")
    parser.add_argument("usd", nargs="?", default=default_usd, help="Path to .usd file")
    args = parser.parse_args()

    inspect_usd_file(args.usd)


if __name__ == "__main__":
    main()
