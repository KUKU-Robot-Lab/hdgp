#!/usr/bin/env python3
"""좌팔 tesollo fabric URDF 생성 + FK 교차검증 (numpy only).

우측 fabric URDF(openarm_tesollo/openarm_tesollo.urdf)를 기준으로:
  - 모든 joint origin 에 XZ평면 반사 M-conjugation (xyz y-flip, R -> M R M)
  - revolute axis/limits 는 bi USD URDF(openarm_tesollo_bi_rl.urdf) 좌측 체인 값 그대로
    (bi 좌측 규약: y축 curl 관절 q 유지, x/z축 관절 한계 스왑, aj_7 축 반전)
  - 링크/조인트 이름은 우측 fabric URDF 와 동일하게 유지
    -> OpenArmTeoslloPoseFabric 이 robot_dir_name 만 바꿔 그대로 로드 가능
  - mesh filename 은 ../openarm_tesollo/meshes/ 상대참조 (fabric 은 FK/충돌구만 사용)

검증:
  (a) 우측 fabric URDF vs bi 우측 체인 — palm/5tip FK 일치 (기존 정합 증명)
  (b) 생성 좌측 URDF vs bi 좌측 체인 — 동일 q 에서 palm/5tip FK 일치

사용: python3 scripts/assets_tools/generate_left_fabric_urdf.py [--check-only]
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HDGP = Path(__file__).resolve().parents[2]
RIGHT_FABRIC_URDF = HDGP / "source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf"
BI_URDF = HDGP / "assets/robot/openarm_tesollo_bi_rl/openarm_tesollo_bi_rl.urdf"
LEFT_DIR = HDGP / "source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo_left"
LEFT_FABRIC_URDF = LEFT_DIR / "openarm_tesollo_left.urdf"

M = np.diag([1.0, -1.0, 1.0])

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# fabric revolute joint 이름 -> bi URDF 좌/우 joint 이름
FABRIC_TO_BI = {f"openarm_right_joint{i}": f"aj_{i}" for i in range(1, 8)}
FABRIC_TO_BI.update({
    f"rj_dg_{fi+1}_{j}": f"hj_{f}_{j}" for fi, f in enumerate(FINGERS) for j in range(1, 5)
})


# ── 기본 회전 유틸 ──────────────────────────────────────────────

def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


def axis_angle(a: np.ndarray, q: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def make_T(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# ── URDF 파싱/FK ────────────────────────────────────────────────

def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.iter("joint"):
        name = j.get("name")
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()])
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None
        lim = j.find("limit")
        limits = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None
        joints[name] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy, "axis": axis, "limits": limits,
        }
    return joints


def fk_link(joints: dict, link: str, q: dict[str, float]) -> np.ndarray:
    """루트까지 부모 체인 누적 FK. q: revolute joint name -> 값."""
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    T = np.eye(4)
    cur = link
    chain = []
    while cur in by_child:
        chain.append(by_child[cur])
        cur = by_child[cur][1]["parent"]
    for name, j in reversed(chain):
        T = T @ make_T(j["xyz"], rpy_to_mat(*j["rpy"]))
        if j["type"] == "revolute":
            T = T @ make_T(np.zeros(3), axis_angle(j["axis"], q.get(name, 0.0)))
    return T


# ── 좌측 URDF 생성 ──────────────────────────────────────────────

def mirror_origin(xyz: np.ndarray, rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_m = M @ xyz
    R_m = M @ rpy_to_mat(*rpy) @ M
    return xyz_m, np.array(mat_to_rpy(R_m))


def fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.9g}" if abs(x) > 1e-12 else "0" for x in v)


# palm 6-DOF attractor 의 7점 패턴·get_palm_pose 축 유도는 fabric 코드에
# +0.25/-0.25 로컬 오프셋으로 하드코딩 → helper 프레임은 지오메트리가 아니라
# 코드 규약이므로 미러하지 않는다 (미러 시 y/y_neg 스왑 → 자세 불일치, IK 잔차 5.7cm 실증).
PALM_HELPER_JOINTS = {
    "palm_x_joint", "palm_x_neg_joint",
    "palm_y_joint", "palm_y_neg_joint",
    "palm_z_joint", "palm_z_neg_joint",
}


def generate_left(right_joints: dict, bi_joints: dict) -> None:
    tree = ET.parse(RIGHT_FABRIC_URDF)
    root = tree.getroot()
    root.set("name", "openarm_tesollo_left")

    for j in root.iter("joint"):
        name = j.get("name")
        info = right_joints[name]
        if name in PALM_HELPER_JOINTS:
            continue  # 코드 규약 프레임: 우측 로컬 오프셋 그대로 유지
        xyz_m, rpy_m = mirror_origin(info["xyz"], info["rpy"])
        o = j.find("origin")
        if o is None:
            o = ET.SubElement(j, "origin")
        o.set("xyz", fmt(xyz_m))
        o.set("rpy", fmt(rpy_m))
        # revolute: fabric 자신의 축을 미러. 부호 규약 s 는 bi 좌우에서 도출:
        #   s=+1 iff a_bi_left == -M a_bi_right (q 유지형), 아니면 s=-1 (q 반전형).
        #   fabric-left axis = -s * M * a_fabric_right, limits: s=-1 이면 (-u,-l).
        # (bi 축을 직접 꽂으면 fabric 의 상이한 분해 규약과 충돌 — joint2 실증)
        if info["type"] == "revolute" and name in FABRIC_TO_BI:
            bi_r = bi_joints[f"r_{FABRIC_TO_BI[name]}"]
            bi_l = bi_joints[f"l_{FABRIC_TO_BI[name]}"]
            s = 1.0 if np.allclose(bi_l["axis"], -(M @ bi_r["axis"])) else -1.0
            axis_left = -s * (M @ info["axis"])
            j.find("axis").set("xyz", fmt(axis_left))
            lo, hi = info["limits"]
            if s < 0:
                lo, hi = -hi, -lo
            lim = j.find("limit")
            lim.set("lower", f"{lo:.9g}")
            lim.set("upper", f"{hi:.9g}")

    # mesh 경로: 우측 폴더 상대참조 (미러 시각화는 부정확하나 fabric 은 FK 만 사용)
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("meshes/"):
            mesh.set("filename", f"../openarm_tesollo/{fn}")

    LEFT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "<!-- generated by scripts/assets_tools/generate_left_fabric_urdf.py -->\n"
        "<!-- 좌팔 tesollo fabric IK URDF: 우측 fabric URDF 의 M-conjugation 미러 + bi USD 좌측 axis/limits. -->\n"
        "<!-- 링크/조인트 이름은 우측과 동일 유지 (OpenArmTeoslloPoseFabric 재사용 목적). -->\n"
    )
    body = ET.tostring(root, encoding="unicode")
    LEFT_FABRIC_URDF.write_text('<?xml version="1.0" ?>\n' + header + body, encoding="utf-8")
    print(f"생성: {LEFT_FABRIC_URDF.relative_to(HDGP)}")


# ── 검증 ────────────────────────────────────────────────────────

def bi_chain_fk(bi_joints: dict, side: str, q27: dict[str, float], link: str) -> np.ndarray:
    """bi URDF 체인 FK. q27: bi joint 이름(aj_1 등 side 미포함) -> 값."""
    q = {f"{side}_{k}": v for k, v in q27.items()}
    return fk_link(bi_joints, link, q)


def validate(side: str, fabric_urdf: Path, bi_joints: dict, seed: int) -> float:
    fj = parse_urdf(fabric_urdf)
    rng = np.random.default_rng(seed)
    # fabric revolute 관절의 한계 내 랜덤 q
    max_err = 0.0
    for _ in range(20):
        qf, qb = {}, {}
        for name, bi_name in FABRIC_TO_BI.items():
            lo, hi = fj[name]["limits"]
            v = float(rng.uniform(lo, hi))
            qf[name] = v
            qb[bi_name] = v
        # palm: fabric palm_link vs bi {side}_hl_palm_alias (위치 동일 검증됨, 방향은 상이)
        pf = fk_link(fj, "palm_link", qf)[:3, 3]
        pb = bi_chain_fk(bi_joints, side, qb, f"{side}_hl_palm_alias")[:3, 3]
        max_err = max(max_err, float(np.linalg.norm(pf - pb)))
        # 5 tips
        for i, f in enumerate(FINGERS):
            tf = fk_link(fj, f"rl_dg_{i+1}_tip", qf)[:3, 3]
            tb = bi_chain_fk(bi_joints, side, qb, f"{side}_hl_{f}_tip")[:3, 3]
            max_err = max(max_err, float(np.linalg.norm(tf - tb)))
    return max_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true", help="생성 없이 기존 좌측 URDF 만 검증")
    args = p.parse_args()

    for f in (RIGHT_FABRIC_URDF, BI_URDF):
        if not f.exists():
            sys.exit(f"missing: {f}")

    right_joints = parse_urdf(RIGHT_FABRIC_URDF)
    bi_raw = parse_urdf(BI_URDF)

    if not args.check_only:
        generate_left(right_joints, bi_raw)

    err_r = validate("r", RIGHT_FABRIC_URDF, bi_raw, seed=7)
    print(f"(a) 우측 fabric vs bi 우측 palm+tip FK 최대오차: {err_r:.3e} m")
    err_l = validate("l", LEFT_FABRIC_URDF, bi_raw, seed=11)
    print(f"(b) 좌측 fabric vs bi 좌측 palm+tip FK 최대오차: {err_l:.3e} m")

    tol = 2e-4  # palm alias 미세 오프셋(3e-7) + 부동소수 여유
    if err_r > tol or err_l > tol:
        sys.exit(f"FAIL: FK 오차 tol({tol}) 초과")
    print("PASS: 좌우 fabric URDF 모두 bi 체인과 FK 일치")


if __name__ == "__main__":
    main()
