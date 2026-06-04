"""openarm_rh56f1.urdf FK sanity (numpy only, 외부 의존성/GPU 불필요).

검증:
  - URDF 파싱, kinematic tree 구성 가능
  - default config 에서 palm_link 가 base 보다 전방(+팔 방향)에 위치
  - palm_x/y/z 가 palm_link 기준 직교 단위축
  - fingertip 5개가 palm_link 보다 손가락 말단 방향에 위치 (palm 으로부터 일정 거리)

실행:
  python3 test_fk_sanity.py
  python3 -m pytest test_fk_sanity.py
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openarm_rh56f1.urdf")

DEFAULT_Q = {
    "openarm_right_joint1": 1.0, "openarm_right_joint2": -0.1,
    "openarm_right_joint3": -0.6, "openarm_right_joint4": 0.5,
    "openarm_right_joint5": 0.0, "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
    "rh56f1_right_right_thumb_1_joint": 0.6,
    "rh56f1_right_right_thumb_2_joint": 0.40,
    "rh56f1_right_right_index_1_joint": 0.90,
    "rh56f1_right_right_middle_1_joint": 0.90,
    "rh56f1_right_right_ring_1_joint": 0.90,
    "rh56f1_right_right_little_1_joint": 0.90,
}


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_angle(axis, ang):
    a = np.array(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def fk():
    root = ET.parse(URDF).getroot()
    joints = root.findall("joint")
    children = {}  # parent_link -> list of (child_link, T_local_fn)
    for j in joints:
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()]
        T_origin = _T(_rpy(*rpy), np.array(xyz))
        jtype = j.get("type")
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
        name = j.get("name")

        def local(jt=jtype, To=T_origin, axs=axis, nm=name):
            if jt == "revolute":
                ang = DEFAULT_Q.get(nm, 0.0)
                return To @ _T(_axis_angle(axs, ang), np.zeros(3))
            return To
        children.setdefault(parent, []).append((child, local))

    # find root link (a parent that is never a child)
    all_children = {c for lst in children.values() for c, _ in lst}
    roots = [p for p in children if p not in all_children]
    base = roots[0]
    world = {base: np.eye(4)}
    stack = [base]
    while stack:
        p = stack.pop()
        for c, local in children.get(p, []):
            world[c] = world[p] @ local()
            stack.append(c)
    return {k: v[:3, 3] for k, v in world.items()}, world


def report():
    pos, world = fk()
    print("base link world origin frames computed:", len(pos))
    for k in ["openarm_right_link7", "rh56f1_right_base_link", "palm_link",
              "palm_x", "palm_y", "palm_z",
              "rh56f1_tip_thumb", "rh56f1_tip_index", "rh56f1_tip_middle",
              "rh56f1_tip_ring", "rh56f1_tip_little"]:
        if k in pos:
            print(f"  {k:32s} {np.round(pos[k], 4)}")
    palm = pos["palm_link"]
    for t in ["rh56f1_tip_thumb", "rh56f1_tip_index", "rh56f1_tip_middle",
              "rh56f1_tip_ring", "rh56f1_tip_little"]:
        d = np.linalg.norm(pos[t] - palm)
        print(f"  dist(palm, {t}) = {d:.4f} m")
    return pos


def test_palm_axes_orthonormal():
    pos, _ = fk()
    palm = pos["palm_link"]
    ux = pos["palm_x"] - palm
    uy = pos["palm_y"] - palm
    uz = pos["palm_z"] - palm
    for u in (ux, uy, uz):
        assert abs(np.linalg.norm(u) - 0.25) < 1e-3, "palm axis 길이 0.25 이어야"
    ux, uy, uz = (u / np.linalg.norm(u) for u in (ux, uy, uz))
    assert abs(ux @ uy) < 1e-3 and abs(uy @ uz) < 1e-3 and abs(ux @ uz) < 1e-3


def test_fingertips_distal_to_palm():
    pos, _ = fk()
    palm = pos["palm_link"]
    for t in ["rh56f1_tip_thumb", "rh56f1_tip_index", "rh56f1_tip_middle",
              "rh56f1_tip_ring", "rh56f1_tip_little"]:
        d = np.linalg.norm(pos[t] - palm)
        assert 0.02 < d < 0.25, f"{t} palm 거리 비정상: {d:.3f}m"


def test_palm_forward_of_base():
    pos, _ = fk()
    # palm_link 는 link7(손목) 보다 손 방향(거리>0)에 있어야
    d = np.linalg.norm(pos["palm_link"] - pos["openarm_right_link7"])
    assert 0.05 < d < 0.25, f"palm-link7 거리 비정상: {d:.3f}m"


if __name__ == "__main__":
    report()
