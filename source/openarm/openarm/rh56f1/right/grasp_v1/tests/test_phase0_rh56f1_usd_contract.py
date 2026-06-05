# pyright: reportAttributeAccessIssue=false
"""Phase 0 검증: RH56F1 USD articulation 계약 확인 (pxr 직접 검사, Isaac Sim 불필요).

목적 (계획 Phase 0 BLOCKER 2건):
  1) mimic 처리: 손이 6 actuated DOF(드라이브만) 인지, 12 독립 revolute 인지.
     - URDF mimic이 USD에서 PhysxMimicJointAPI로 보존됐는지 확인.
  2) force_sensor / tip 링크 생존: fixed joint 병합으로 body가 사라졌는지 확인.
     → 사라졌으면 ContactSensor remap 또는 USD 재생성 필요.

실행:
  python3 tests/test_phase0_rh56f1_usd_contract.py            # 사실 리포트
  python3 -m pytest tests/test_phase0_rh56f1_usd_contract.py  # 계약 단언(RED→GREEN)
"""

import os

from pxr import Usd, UsdPhysics

USD_PATH = "/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.usd"

# 계획에서 가정한 6 actuated 손 관절 (드라이브)
EXPECTED_ACTUATED_HAND_JOINTS = [
    "rh56f1_right_right_thumb_1_joint",
    "rh56f1_right_right_thumb_2_joint",
    "rh56f1_right_right_index_1_joint",
    "rh56f1_right_right_middle_1_joint",
    "rh56f1_right_right_ring_1_joint",
    "rh56f1_right_right_little_1_joint",
]
# mimic 추종 관절 (URDF 기준)
MIMIC_HAND_JOINTS = [
    "rh56f1_right_right_thumb_3_joint",
    "rh56f1_right_right_thumb_4_joint",
    "rh56f1_right_right_index_2_joint",
    "rh56f1_right_right_middle_2_joint",
    "rh56f1_right_right_ring_2_joint",
    "rh56f1_right_right_little_2_joint",
]
# Phase 0 발견: fingertip *_force_sensor / *_tip 링크는 USD에서 fixed-joint 병합으로 소멸.
# → 생존하는 말단 손가락 링크에 ContactSensor를 매핑한다 (palm은 plam_force_sensor 생존).
PALM_SENSOR_BODY = "rh56f1_right_plam_force_sensor"
FINGERTIP_SENSOR_BODIES = [
    "rh56f1_right_right_thumb_4",
    "rh56f1_right_right_index_2",
    "rh56f1_right_right_middle_2",
    "rh56f1_right_right_ring_2",
    "rh56f1_right_right_little_2",
]
EXPECTED_SENSOR_BODIES = [PALM_SENSOR_BODY] + FINGERTIP_SENSOR_BODIES

# 참고: 아래 force_sensor 링크들은 병합으로 소멸함이 확인됨 (재생성 없이 위 매핑 사용)
MERGED_FORCE_SENSOR_LINKS = [
    "rh56f1_right_thumb_force_sensor",
    "rh56f1_right_index_force_sensor",
    "rh56f1_right_middle_force_sensor",
    "rh56f1_right_ring_force_sensor",
    "rh56f1_right_little_force_sensor",
]


def _scan():
    stage = Usd.Stage.Open(USD_PATH)
    revolute_right_hand = {}   # name -> {has_drive, has_mimic}
    rigid_bodies = set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.add(name)
        tn = prim.GetTypeName()
        if tn == "PhysicsRevoluteJoint" and "rh56f1_right" in str(prim.GetPath()):
            has_drive = bool(prim.HasAPI(UsdPhysics.DriveAPI)) or any(
                a.GetName().startswith("drive:") for a in prim.GetAttributes()
            )
            has_mimic = any(
                "mimic" in a.GetName().lower() for a in prim.GetAttributes()
            ) or any("Mimic" in s for s in prim.GetAppliedSchemas())
            revolute_right_hand[name] = {"drive": has_drive, "mimic": has_mimic}
    return revolute_right_hand, rigid_bodies


def report():
    rj, bodies = _scan()
    print(f"\n[USD] {USD_PATH}")
    print(f"\n=== rh56f1_right revolute joints ({len(rj)}) ===")
    for n in sorted(rj):
        print(f"  {n:42s} drive={rj[n]['drive']} mimic={rj[n]['mimic']}")
    drive_joints = [n for n, v in rj.items() if v["drive"]]
    print(f"\n  >> drive(actuated) joint count = {len(drive_joints)}")
    print(f"\n=== sensor body 생존 여부 ===")
    for b in EXPECTED_SENSOR_BODIES:
        print(f"  {b:42s} {'OK(body 존재)' if b in bodies else 'MISSING(병합됨)'}")
    print(f"\n=== 전체 rh56f1_right rigid body ({sum('rh56f1_right' in b for b in bodies)}) ===")
    for b in sorted(b for b in bodies if "rh56f1_right" in b):
        print(f"  {b}")
    return rj, bodies


# --- pytest 계약 (Phase 0 GREEN 기준) ---
def test_usd_exists():
    assert os.path.exists(USD_PATH), USD_PATH


def test_six_actuated_hand_dof():
    rj, _ = _scan()
    drive = sorted(n for n, v in rj.items() if v["drive"])
    assert drive == sorted(EXPECTED_ACTUATED_HAND_JOINTS), (
        f"actuated 손 관절 불일치. 실제 drive joints={drive}"
    )


def test_sensor_bodies_survive():
    """ContactSensor에 사용할 (palm + 말단 손가락) body가 모두 생존하는지."""
    _, bodies = _scan()
    missing = [b for b in EXPECTED_SENSOR_BODIES if b not in bodies]
    assert not missing, f"센서용 body 누락: {missing}"


def test_fingertip_force_sensor_links_are_merged():
    """fingertip force_sensor 링크는 병합 소멸함을 명시적으로 기록 (전략 근거)."""
    _, bodies = _scan()
    survived = [b for b in MERGED_FORCE_SENSOR_LINKS if b in bodies]
    assert not survived, f"예상과 달리 생존: {survived} (USD 변경 가능성 재확인)"


if __name__ == "__main__":
    report()
