"""RH56F1 로봇 articulation 로드 스모크 (Phase 3 사전 확정).

목적: Isaac Lab 에서 USD 를 articulation 으로 로드해
  1) joint_names / 개수 (mimic 추종관절이 DOF로 잡히는지 → hand 6 vs 12)
  2) body_names (센서용 body 생존 확인)
  3) drive 6관절 + 센서 body 존재
를 출력. → env_cfg actuator/sensor 설계 확정.

실행:
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p \
    /home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/inspire_r/grasp_r_v1/tests/smoke_rh56f1_robot.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

USD = "/home/user/rl_ws/hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.usd"

DRIVE_JOINTS = [
    "rh56f1_right_right_thumb_1_joint", "rh56f1_right_right_thumb_2_joint",
    "rh56f1_right_right_index_1_joint", "rh56f1_right_right_middle_1_joint",
    "rh56f1_right_right_ring_1_joint", "rh56f1_right_right_little_1_joint",
]
MIMIC_JOINTS = [
    "rh56f1_right_right_thumb_3_joint", "rh56f1_right_right_thumb_4_joint",
    "rh56f1_right_right_index_2_joint", "rh56f1_right_right_middle_2_joint",
    "rh56f1_right_right_ring_2_joint", "rh56f1_right_right_little_2_joint",
]
SENSOR_BODIES = [
    "rh56f1_right_plam_force_sensor",
    "rh56f1_right_right_thumb_4", "rh56f1_right_right_index_2",
    "rh56f1_right_right_middle_2", "rh56f1_right_right_ring_2",
    "rh56f1_right_right_little_2",
]


def main():
    sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))

    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            ),
        },
    )
    robot = Articulation(cfg)
    sim.reset()

    jn = list(robot.joint_names)
    bn = list(robot.body_names)
    print("\n=== ARTICULATION SUMMARY ===")
    print(f"총 DOF(joint) 수 = {len(jn)}")
    print(f"총 body 수 = {len(bn)}")

    rh_joints = [j for j in jn if "rh56f1_right" in j]
    print(f"\n우측 손 joint ({len(rh_joints)}):")
    for j in rh_joints:
        tag = "DRIVE" if j in DRIVE_JOINTS else ("MIMIC?" if j in MIMIC_JOINTS else "")
        print(f"   {j:42s} {tag}")

    n_mimic_as_dof = sum(1 for j in MIMIC_JOINTS if j in jn)
    print(f"\n>> mimic 추종관절이 DOF로 잡힌 수 = {n_mimic_as_dof}")
    if n_mimic_as_dof == 0:
        print("   => hand 6 DOF. actuator 그룹 6개(drive)만 설정하면 됨.")
    else:
        print(f"   => hand {6 + n_mimic_as_dof} DOF. mimic 그룹(passive, stiffness=0) 추가 필요.")

    print(f"\n센서 body 생존:")
    for b in SENSOR_BODIES:
        print(f"   {b:42s} {'OK' if b in bn else 'MISSING'}")

    print(f"\n우측 손 body 전체:")
    for b in bn:
        if "rh56f1_right" in b:
            print(f"   {b}")

    print("\n[DONE] env_cfg 설계용 정보 출력 완료.")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    main()
