# pyright: reportOptionalMemberAccess=false
"""openarm_rh56f1 fabrics URDF 생성기 — 양팔, _rl 소스 단일 기준.

Isaac USD(openarm_bi_rh56f1_rl)와 동일한 _rl URDF 를 유일 소스로 삼아 양팔
(각 arm 7 + hand drive 6 = 26 DOF) fabric URDF 를 생성한다. (구: Tesollo URDF 에서
팔을 복사하던 방식 폐기 — 팔 base 마운트가 Isaac 과 6.4cm 어긋났음.)

- 팔:  {side}_aj_base(fixed, body_link 마운트) + {side}_aj_1~7(revolute).
- 손:  {side}_hj_mount + palm_1/2/3 + palm_sensor(fixed), drive 6(revolute),
       mimic 6(fixed 근사; PhysxMimic 은 sim 이 처리), fingertip *_tip(fixed FK).
- palm_sensor 자식 IK 축점 6개(ps_{side}_*) — fabric palm attractor 제어점.
- 메시 불필요(taskmap=프레임 FK + sphere) → 링크는 massless, visual/collision 없음.

cspace 순서(out append 순) = [r_al 1~7, r_hand drive 6, l_al 1~7, l_hand drive 6].
  drive 6 = URDF 정의순의 revolute = [thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1].

실행: python3 generate_openarm_rh56f1_urdf.py → openarm_rh56f1.urdf
"""

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_URDF = "/home/user/rl_ws/hdgp/assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.urdf"
OUT_URDF = os.path.join(HERE, "openarm_rh56f1.urdf")

DRIVE_SUFFIXES = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
MIMIC_SUFFIXES = ("thumb_3", "thumb_4", "index_2", "middle_2", "ring_2", "pinky_2")
TIP_SUFFIXES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
PALM_FIXED_SUFFIXES = ("mount", "palm_1", "palm_2", "palm_3", "palm_sensor")

# palm IK 축점: convert_transform_to_points 가 원점 + ±0.25m 축점으로 6D pose 실현.
PALM_AXIS = {
    "x": "0.25 0 0", "x_neg": "-0.25 0 0",
    "y": "0 0.25 0", "y_neg": "0 -0.25 0",
    "z": "0 0 0.25", "z_neg": "0 0 -0.25",
}
# 손 충돌구: 손가락 proximal 링크 1개씩 (thumb 은 thumb_2).
SPHERE_PARENT_SUFFIX = ("thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")


def _massless(link):
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "1e-6", "ixy": "0", "ixz": "0",
        "iyy": "1e-6", "iyz": "0", "izz": "1e-6",
    })


def _add_link(out, name):
    _massless(ET.SubElement(out, "link", {"name": name}))


def _copy_joint(out, src_j, force_type):
    j = ET.SubElement(out, "joint", {"name": src_j.get("name"), "type": force_type})
    o = src_j.find("origin")
    ET.SubElement(j, "origin", {
        "xyz": o.get("xyz", "0 0 0") if o is not None else "0 0 0",
        "rpy": o.get("rpy", "0 0 0") if o is not None else "0 0 0",
    })
    ET.SubElement(j, "parent", {"link": src_j.find("parent").get("link")})
    ET.SubElement(j, "child", {"link": src_j.find("child").get("link")})
    if force_type == "revolute":
        ax = src_j.find("axis")
        ET.SubElement(j, "axis", {"xyz": ax.get("xyz") if ax is not None else "0 0 1"})
        lim = src_j.find("limit")
        if lim is not None:
            ET.SubElement(j, "limit", {
                "lower": lim.get("lower"), "upper": lim.get("upper"),
                "effort": "5", "velocity": "2",
            })
        else:
            ET.SubElement(j, "limit", {"lower": "0", "upper": "1.5", "effort": "5", "velocity": "2"})


def _add_palm_axis_points(out, side):
    for key, off in PALM_AXIS.items():
        name = f"ps_{side}_{key}"
        _add_link(out, name)
        j = ET.SubElement(out, "joint", {"name": f"{name}_joint", "type": "fixed"})
        ET.SubElement(j, "origin", {"xyz": off, "rpy": "0 0 0"})
        ET.SubElement(j, "parent", {"link": f"{side}_hl_palm_sensor"})
        ET.SubElement(j, "child", {"link": name})


def _add_collision_spheres(out, side):
    for suf in SPHERE_PARENT_SUFFIX:
        name = f"{side}_sphere_{suf.split('_')[0]}"
        _add_link(out, name)
        j = ET.SubElement(out, "joint", {"name": f"{name}_joint", "type": "fixed"})
        ET.SubElement(j, "origin", {"xyz": "0 0.015 0", "rpy": "0 0 0"})
        ET.SubElement(j, "parent", {"link": f"{side}_hl_{suf}"})
        ET.SubElement(j, "child", {"link": name})


def build():
    src = ET.parse(SRC_URDF).getroot()
    src_joints = list(src.findall("joint"))
    out = ET.Element("robot", {"name": "openarm_rh56f1"})
    _add_link(out, "body_link")

    for side in ("r", "l"):
        pre = f"{side}_"
        # --- arm: aj_base(fixed) + aj_1~7(revolute), 정의 순서 ---
        for j in src_joints:
            n = j.get("name") or ""
            if n == f"{pre}aj_base":
                _add_link(out, j.find("child").get("link"))
                _copy_joint(out, j, "fixed")
            elif n.startswith(f"{pre}aj_") and n[len(pre) + 3:].isdigit():
                _add_link(out, j.find("child").get("link"))
                _copy_joint(out, j, "revolute")
        # --- hand: 정의 순서로 순회(drive→revolute, mimic/palm/tip→fixed) ---
        for j in src_joints:
            n = j.get("name") or ""
            if not n.startswith(f"{pre}hj_"):
                continue
            suf = n[len(pre) + 3:]
            child = j.find("child").get("link")
            if suf in PALM_FIXED_SUFFIXES:
                _add_link(out, child); _copy_joint(out, j, "fixed")
            elif suf in DRIVE_SUFFIXES:
                _add_link(out, child); _copy_joint(out, j, "revolute")
            elif suf in MIMIC_SUFFIXES:
                _add_link(out, child); _copy_joint(out, j, "fixed")
            elif suf in TIP_SUFFIXES:
                _add_link(out, child); _copy_joint(out, j, "fixed")
            # {side}_hj_*_sensor (힘센서 링크): FK 불필요 → skip
        _add_palm_axis_points(out, side)
        _add_collision_spheres(out, side)

    _indent(out)
    ET.ElementTree(out).write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {OUT_URDF}")


def _indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


if __name__ == "__main__":
    build()
