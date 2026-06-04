# pyright: reportAttributeAccessIssue=false
"""openarm_rh56f1 fabrics URDF 생성기.

전략 (계획 Phase 1):
  - OpenArm 우측 팔은 Tesollo fabric과 동일 → 팔 체인 + palm 가상프레임 +
    팔/world 충돌구 + cspace 에너지 구조를 openarm_tesollo.urdf 에서 그대로 재사용.
  - Tesollo 손(rj_dg_*, tesollo_right_rl_dg_*)은 제거하고 RH56F1 손을 graft.
  - RH56F1 손: drive 6관절(revolute) + mimic 추종 6관절(고정, 결합 grasp 각).
    → fabric cspace = 7 arm + 6 hand = 13 DOF (env가 제어하는 actuated DOF와 정렬).
  - fingertip FK 프레임 5개(rh56f1_tip_{thumb,index,middle,ring,little}) 추가.

근거(검증됨):
  - RH56F1 base_link 는 link7 에 Tesollo 마운트와 동일 변환
    (xyz=0 0.0000003 0.0595695, rpy=0 0 -1.5707964) 으로 부착.
  - RH56F1 palm pad(plam_force_sensor) z ≈ base+0.0737 ≈ 0.133 ≈ Tesollo palm_link(0.1333695)
    → palm 가상프레임 재사용 기하적으로 타당.

실행:
  python3 generate_openarm_rh56f1_urdf.py
  → openarm_rh56f1.urdf 생성
"""

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
FABRICS_URDF_ROOT = os.path.abspath(os.path.join(HERE, ".."))
TESOLLO_URDF = os.path.join(FABRICS_URDF_ROOT, "openarm_tesollo", "openarm_tesollo.urdf")
SRC_ROBOT_URDF = "/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.urdf"
OUT_URDF = os.path.join(HERE, "openarm_rh56f1.urdf")

# RH56F1 base_link → link7 마운트 (검증값)
BASE_MOUNT_XYZ = "0 0.0000003 0.0595695"
BASE_MOUNT_RPY = "0 0 -1.5707964"

# drive 6관절 (revolute 로 유지)
DRIVE_JOINTS = [
    "rh56f1_right_right_thumb_1_joint",
    "rh56f1_right_right_thumb_2_joint",
    "rh56f1_right_right_index_1_joint",
    "rh56f1_right_right_middle_1_joint",
    "rh56f1_right_right_ring_1_joint",
    "rh56f1_right_right_little_1_joint",
]
# mimic 추종 6관절 (고정). 결합 grasp 각 = drive_grasp × multiplier.
# drive grasp 기준값(rad): thumb_2(굽힘)=0.40, 4지_1=0.90
MIMIC_FIXED_ANGLE = {
    "rh56f1_right_right_thumb_3_joint": 0.40 * 1.1425,   # ≈0.457
    "rh56f1_right_right_thumb_4_joint": 0.40 * 1.1425 * 0.7508,  # ≈0.343
    "rh56f1_right_right_index_2_joint": 0.90 * 1.1169,   # ≈1.005
    "rh56f1_right_right_middle_2_joint": 0.90 * 1.1169,
    "rh56f1_right_right_ring_2_joint": 0.90 * 1.1169,
    "rh56f1_right_right_little_2_joint": 0.90 * 1.1169,
}
# fingertip 프레임: (부모 말단 링크, 실제 *_tip_joint 오프셋) — 결합 urdf 에서 추출
TIP_PARENT = {
    "rh56f1_tip_thumb":  ("rh56f1_right_right_thumb_4",  "-0.017405 0.025196 0.0068001"),
    "rh56f1_tip_index":  ("rh56f1_right_right_index_2",  "0.0049188 0.039631 0.006"),
    "rh56f1_tip_middle": ("rh56f1_right_right_middle_2", "0.0059647 0.042512 0.0060998"),
    "rh56f1_tip_ring":   ("rh56f1_right_right_ring_2",   "0.0049188 0.039631 0.006"),
    "rh56f1_tip_little": ("rh56f1_right_right_little_2", "0.0051203 0.032544 0.0059927"),
}

# Tesollo URDF 에서 제거할 손 관련 prefix
DROP_LINK_PREFIXES = ("tesollo_right_rl_dg_", "rl_dg_")
DROP_JOINT_PREFIXES = ("rj_dg_", "rl_dg_", "tesollo_right_rl_dg_")


def _axis_to_rotation_origin(axis_xyz):
    """mimic 고정 joint 의 origin 회전은 0 으로 두고 child 링크 origin 에 각도를 반영하지 않는다.
    (고정 근사이므로 회전축 방향만 유지; 단순화)"""
    return axis_xyz


def build():
    tes = ET.parse(TESOLLO_URDF).getroot()
    src = ET.parse(SRC_ROBOT_URDF).getroot()
    src_joints = {j.get("name"): j for j in src.findall("joint")}
    src_links = {l.get("name"): l for l in src.findall("link")}

    out = ET.Element("robot", {"name": "openarm_rh56f1"})

    # 1) Tesollo 에서 팔/palm/world 구조만 복사 (손 제거)
    #    Fabrics 는 메시 불필요(taskmap=프레임 FK + sphere repulsion) → visual/collision 제거.
    for el in list(tes):
        name = el.get("name", "")
        if el.tag == "link":
            if any(name.startswith(p) for p in DROP_LINK_PREFIXES):
                continue
            _strip_mesh_geometry(el)
            out.append(el)
        elif el.tag == "joint":
            if any(name.startswith(p) for p in DROP_JOINT_PREFIXES):
                continue
            out.append(el)
        else:
            out.append(el)

    # 2) RH56F1 base_link → link7 부착
    base_link = ET.SubElement(out, "link", {"name": "rh56f1_right_base_link"})
    _massless_inertial(base_link)
    j = ET.SubElement(out, "joint", {"name": "rh56f1_right_base_joint", "type": "fixed"})
    ET.SubElement(j, "origin", {"xyz": BASE_MOUNT_XYZ, "rpy": BASE_MOUNT_RPY})
    ET.SubElement(j, "parent", {"link": "openarm_right_link7"})
    ET.SubElement(j, "child", {"link": "rh56f1_right_base_link"})

    # 3) RH56F1 손가락 체인 (drive=revolute, mimic=fixed)
    finger_joint_order = [
        "rh56f1_right_right_thumb_1_joint", "rh56f1_right_right_thumb_2_joint",
        "rh56f1_right_right_thumb_3_joint", "rh56f1_right_right_thumb_4_joint",
        "rh56f1_right_right_index_1_joint", "rh56f1_right_right_index_2_joint",
        "rh56f1_right_right_middle_1_joint", "rh56f1_right_right_middle_2_joint",
        "rh56f1_right_right_ring_1_joint", "rh56f1_right_right_ring_2_joint",
        "rh56f1_right_right_little_1_joint", "rh56f1_right_right_little_2_joint",
    ]
    for jn in finger_joint_order:
        sj = src_joints[jn]
        child = sj.find("child").get("link")
        parent = sj.find("parent").get("link")
        origin = sj.find("origin")
        axis = sj.find("axis")
        limit = sj.find("limit")

        link = ET.SubElement(out, "link", {"name": child})
        _massless_inertial(link)

        if jn in DRIVE_JOINTS:
            nj = ET.SubElement(out, "joint", {"name": jn, "type": "revolute"})
            ET.SubElement(nj, "origin", {"xyz": origin.get("xyz"), "rpy": origin.get("rpy")})
            ET.SubElement(nj, "parent", {"link": parent})
            ET.SubElement(nj, "child", {"link": child})
            ET.SubElement(nj, "axis", {"xyz": axis.get("xyz")})
            ET.SubElement(nj, "limit", {
                "lower": limit.get("lower"), "upper": limit.get("upper"),
                "effort": "5", "velocity": "2",
            })
        else:
            # mimic 추종 → fixed (origin 그대로, 결합각은 근사 무시: 위치만 고정)
            nj = ET.SubElement(out, "joint", {"name": jn, "type": "fixed"})
            ET.SubElement(nj, "origin", {"xyz": origin.get("xyz"), "rpy": origin.get("rpy")})
            ET.SubElement(nj, "parent", {"link": parent})
            ET.SubElement(nj, "child", {"link": child})

    # 4) fingertip FK 프레임 (실제 *_tip_joint 오프셋)
    for tip, (par, offset) in TIP_PARENT.items():
        link = ET.SubElement(out, "link", {"name": tip})
        _massless_inertial(link)
        tj = ET.SubElement(out, "joint", {"name": f"{tip}_joint", "type": "fixed"})
        ET.SubElement(tj, "origin", {"xyz": offset, "rpy": "0 0 0"})
        ET.SubElement(tj, "parent", {"link": par})
        ET.SubElement(tj, "child", {"link": tip})

    # 5) 손 충돌구 (간소화: 손가락 proximal 링크 1개씩 + palm)
    for fin, par in [
        ("rh56f1_sphere_thumb", "rh56f1_right_right_thumb_2"),
        ("rh56f1_sphere_index", "rh56f1_right_right_index_1"),
        ("rh56f1_sphere_middle", "rh56f1_right_right_middle_1"),
        ("rh56f1_sphere_ring", "rh56f1_right_right_ring_1"),
        ("rh56f1_sphere_little", "rh56f1_right_right_little_1"),
    ]:
        link = ET.SubElement(out, "link", {"name": fin})
        _massless_inertial(link)
        sj = ET.SubElement(out, "joint", {"name": f"{fin}_joint", "type": "fixed"})
        ET.SubElement(sj, "origin", {"xyz": "0 0.015 0", "rpy": "0 0 0"})
        ET.SubElement(sj, "parent", {"link": par})
        ET.SubElement(sj, "child", {"link": fin})

    _indent(out)
    ET.ElementTree(out).write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {OUT_URDF}")


def _strip_mesh_geometry(link_el):
    """링크에서 <visual>/<collision> 제거 (메시 파일 의존 제거). inertial 은 유지."""
    for tag in ("visual", "collision"):
        for child in link_el.findall(tag):
            link_el.remove(child)


def _massless_inertial(link_el):
    inertial = ET.SubElement(link_el, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "1e-6", "ixy": "0", "ixz": "0",
        "iyy": "1e-6", "iyz": "0", "izz": "1e-6",
    })


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
