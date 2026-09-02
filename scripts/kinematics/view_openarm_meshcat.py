"""
OpenArm-Tesollo Kinematics Studio & Meshcat 3D Visualizer
Renders OpenArm 7-DoF Arm, Tesollo 5-Finger Hand, Frame TFs, and Kinematic Chains in real-time.
"""

import os
import sys
import time
import argparse
import numpy as np

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf


def create_openarm_visualizer(urdf_path: str, open_browser: bool = True):
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    mesh_dirs = [os.path.dirname(urdf_path)]

    print("=" * 65)
    print(f"[*] OpenArm URDF: {os.path.basename(urdf_path)}")
    print(f"[*] 경로: {urdf_path}")

    robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dirs)
    print(f"[+] 로봇 모델명: '{robot.model.name}'")
    print(f"[+] 총 관절 자유도(nq): {robot.model.nq} | 총 프레임 수: {robot.model.nframes}")

    # Initialize Meshcat
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=open_browser)
    viz.loadViewerModel()

    return robot, viz


def setup_skeleton_visuals(viz: MeshcatVisualizer, robot: RobotWrapper):
    """Adds visual link cylinders and coordinate frames for kinematic inspection."""
    # Add coordinate frame markers on palm and fingertips
    key_frames = ["palm_link", "rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
    
    # Palm target sphere
    viz.viewer["palm_indicator"].set_object(
        g.Sphere(radius=0.025),
        g.MeshLambertMaterial(color=0x3B82F6, opacity=0.85)  # Blue
    )
    
    # Fingertip spheres
    colors = [0xEF4444, 0xF59E0B, 0x10B981, 0x6366F1, 0xEC4899]  # 5 colors for 5 fingers
    for idx in range(1, 6):
        viz.viewer[f"tip_{idx}"].set_object(
            g.Sphere(radius=0.012),
            g.MeshLambertMaterial(color=colors[idx - 1], opacity=0.9)
        )

    # Arm joint link spheres (7 joints)
    for j_idx in range(1, 8):
        viz.viewer[f"arm_joint_{j_idx}"].set_object(
            g.Sphere(radius=0.028),
            g.MeshLambertMaterial(color=0xF97316, opacity=0.9)  # Orange joints
        )

    # Target cup
    viz.viewer["target_cup"].set_object(
        g.Cylinder(height=0.12, radius=0.04),
        g.MeshLambertMaterial(color=0x22C55E, opacity=0.8)  # Green cup
    )
    viz.viewer["target_cup"].set_transform(tf.translation_matrix([0.35, -0.15, 0.45]))


def update_skeleton_transforms(viz: MeshcatVisualizer, robot: RobotWrapper):
    """Updates the positions of joints, palm, and fingertips from Forward Kinematics."""
    # Update arm joint spheres
    for j_idx in range(1, 8):
        fname = f"openarm_right_link{j_idx}"
        if robot.model.existFrame(fname):
            fid = robot.model.getFrameId(fname)
            T = robot.data.oMf[fid].homogeneous
            viz.viewer[f"arm_joint_{j_idx}"].set_transform(T)

    # Update palm
    if robot.model.existFrame("palm_link"):
        fid = robot.model.getFrameId("palm_link")
        T = robot.data.oMf[fid].homogeneous
        viz.viewer["palm_indicator"].set_transform(T)

    # Update 5 fingertips
    for idx in range(1, 6):
        fname = f"rl_dg_{idx}_tip"
        if robot.model.existFrame(fname):
            fid = robot.model.getFrameId(fname)
            T = robot.data.oMf[fid].homogeneous
            viz.viewer[f"tip_{idx}"].set_transform(T)


def get_ready_pose(robot: RobotWrapper):
    q = pin.neutral(robot.model)
    # OpenArm Right Arm (7-DoF)
    # Joint 1: shoulder pan, Joint 2: shoulder lift, Joint 3: arm roll,
    # Joint 4: elbow pitch, Joint 5: wrist roll, Joint 6: wrist pitch, Joint 7: wrist yaw
    if robot.model.nq >= 7:
        q[0] = 0.3    # shoulder pan
        q[1] = 0.8    # shoulder lift
        q[2] = 0.0    # arm roll
        q[3] = 1.4    # elbow pitch (bending forward)
        q[4] = 0.0    # wrist roll
        q[5] = 0.5    # wrist pitch
        q[6] = 0.0    # wrist yaw

    # Tesollo 5-Finger Hand (joints 7..26)
    if robot.model.nq >= 27:
        for i in range(7, robot.model.nq):
            q[i] = 0.35  # Ready grasp pose

    return q


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_path = os.path.join(
        repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf",
        "openarm_tesollo", "openarm_tesollo.urdf"
    )

    parser = argparse.ArgumentParser(description="OpenArm-Tesollo Meshcat Visualizer")
    parser.add_argument("--urdf", type=str, default=urdf_path, help="Path to URDF file")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open browser")

    args = parser.parse_args()

    robot, viz = create_openarm_visualizer(args.urdf, open_browser=not args.no_open)
    setup_skeleton_visuals(viz, robot)

    # Initial ready configuration
    q_ready = get_ready_pose(robot)
    viz.display(q_ready)

    # Compute Forward Kinematics
    pin.forwardKinematics(robot.model, robot.data, q_ready)
    pin.updateFramePlacements(robot.model, robot.data)
    update_skeleton_transforms(viz, robot)

    print("\n" + "=" * 65)
    print(f"[SUCCESS] OpenArm-Tesollo 3D 뷰어가 LIVE 실행되었습니다:")
    print(f"          {viz.viewer.url()}")
    print("-" * 65)
    print("  화면 구성:")
    print("    - 하단: OpenArm 베이스 스탠드 (Body Link)")
    print("    - 주황색 구체: 7자유도 로봇 팔 관절 (Joints 1~7)")
    print("    - 파란색 구체: Tesollo 핸드 손바닥 (Palm Center)")
    print("    - 오색 구체: 5개 손가락 끝점 (Fingertips 1~5)")
    print("    - 초록색 원통: 작업 대상 목표 컵 (Target Cup)")
    print("-" * 65)
    print("  조작법: 좌클릭(회전), 우클릭(이동), 휠(줌), 종료(Ctrl+C)")
    print("=" * 65 + "\n")

    print("[*] 실시간 관절 모션 & 기구학 연산 진행 중... (종료: Ctrl+C)")

    import signal
    running = True

    def sig_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, sig_handler)

    t = 0.0
    dt = 0.03
    try:
        while running:
            q = q_ready.copy()
            # Smooth arm reaching / waving motion
            q[0] += 0.25 * np.sin(1.2 * t)
            q[1] += 0.15 * np.sin(1.5 * t)
            q[3] += 0.20 * np.cos(1.0 * t)
            q[5] += 0.18 * np.sin(2.0 * t)

            # Smooth 5-finger articulation (Grasping / Releasing)
            if robot.model.nq >= 27:
                for f_idx in range(7, robot.model.nq):
                    q[f_idx] += 0.25 * np.sin(2.5 * t + 0.2 * f_idx)

            # Update Pinocchio Kinematics & Meshcat
            viz.display(q)
            pin.forwardKinematics(robot.model, robot.data, q)
            pin.updateFramePlacements(robot.model, robot.data)
            update_skeleton_transforms(viz, robot)

            time.sleep(dt)
            t += dt
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n[*] OpenArm 뷰어를 정상 종료했습니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
