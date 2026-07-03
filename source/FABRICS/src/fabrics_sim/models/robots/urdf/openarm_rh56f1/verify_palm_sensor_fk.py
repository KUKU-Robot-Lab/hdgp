#!/usr/bin/env python3
"""palm_sensor FK 정합 검증기 (rh56f1 fabric URDF ↔ 참고 URDF).

palm_sensor 는 손목(arm 마지막 링크)에서 fixed joint 만으로 연결되므로,
"손목 기준 palm_sensor 상대변환"은 관절각과 무관한 상수다. 두 URDF에서 이 상수
변환(위치·자세)이 일치하면 fabric IK 가 실제 palm_sensor 를 정확히 제어한다.

사용:
  python3 verify_palm_sensor_fk.py --urdf FABRIC.urdf --ref REF.urdf --side right|left|both
  # 자기검증: --urdf 와 --ref 를 동일 파일로 → 오차 0

PASS 기준: 위치 < 2mm, 자세 < 1°.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np

# side → (arm 손목 링크 후보들, palm_sensor 링크). fabric/참고 URDF 모두 커버.
WRIST_CANDIDATES = {
    "right": ["openarm_right_link7", "r_al_7"],
    "left": ["openarm_left_link7", "l_al_7"],
}
PALM_SENSOR = {"right": "r_hl_palm_sensor", "left": "l_hl_palm_sensor"}


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _load_joints(urdf_path: str) -> dict:
    """child_link 이름 → joint(origin, parent) dict. fixed 체인 FK 용."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        joints[child] = {
            "parent": j.find("parent").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "type": j.get("type"),
        }
    return joints


def _find_wrist(side: str, all_links: set) -> str:
    for cand in WRIST_CANDIDATES[side]:
        if cand in all_links:
            return cand
    raise ValueError(f"{side} 손목 링크를 찾지 못함(후보: {WRIST_CANDIDATES[side]}).")


def _relative_wrist_to_palm(urdf_path: str, side: str) -> np.ndarray:
    """손목→palm_sensor 상대 4x4 변환. palm_sensor 에서 손목까지 부모를 추적하며,
    이 구간이 전부 fixed(관절각 무관 상수)임을 검증한다."""
    joints = _load_joints(urdf_path)
    all_links = set(joints.keys()) | {j["parent"] for j in joints.values()}
    wrist = _find_wrist(side, all_links)
    palm = PALM_SENSOR[side]
    if palm not in all_links:
        raise ValueError(f"{palm} 링크가 {urdf_path} 에 없음.")

    chain = []
    cur = palm
    while cur != wrist:
        if cur not in joints:
            raise ValueError(f"{palm}→{wrist} 경로 단절(cur={cur}).")
        chain.append(joints[cur])
        cur = joints[cur]["parent"]

    T = np.eye(4)
    for j in reversed(chain):
        if j["type"] not in ("fixed", None):
            raise ValueError(
                f"손목→palm_sensor 체인에 non-fixed joint: {j['type']} "
                f"(parent={j['parent']}). 상수 변환 가정 위배."
            )
        M = np.eye(4)
        M[:3, :3] = _rpy_to_R(*j["rpy"])
        M[:3, 3] = j["xyz"]
        T = T @ M
    return T


def _compare(side: str, fabric_urdf: str, ref_urdf: str) -> bool:
    T_fab = _relative_wrist_to_palm(fabric_urdf, side)
    T_ref = _relative_wrist_to_palm(ref_urdf, side)
    pos_err_mm = np.linalg.norm(T_fab[:3, 3] - T_ref[:3, 3]) * 1000.0
    R_err = T_fab[:3, :3].T @ T_ref[:3, :3]
    cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cos))
    ok = bool(pos_err_mm < 2.0 and ang_err_deg < 1.0)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{side:>5}] pos_err={pos_err_mm:7.3f} mm  "
        f"ang_err={ang_err_deg:7.3f} deg  -> {status}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="fabric URDF 경로")
    ap.add_argument("--ref", required=True, help="참고(ground-truth) URDF 경로")
    ap.add_argument("--side", choices=["right", "left", "both"], default="right")
    args = ap.parse_args()

    sides = ["right", "left"] if args.side == "both" else [args.side]
    all_ok = True
    for side in sides:
        all_ok &= _compare(side, args.urdf, args.ref)
    print("=" * 48)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
