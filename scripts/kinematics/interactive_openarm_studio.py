"""
OpenArm-Tesollo Interactive Kinematics Studio
- Real-time 3D RGB Coordinate Axes (TF Triads) for Palm, Joints, and 5 Fingertips
- Live Joint Angle (deg/rad) and 6D Pose (X, Y, Z, Roll, Pitch, Yaw) display
- Interactive Poses (Ready, Reach, Grasp, Pour, Wave)
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
from scipy.spatial.transform import Rotation


def create_openarm_visualizer(urdf_path: str, open_browser: bool = True):
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    mesh_dirs = [os.path.dirname(urdf_path)]

    print("=" * 70)
    print(f"[*] OpenArm-Tesollo 로봇 모델 로드: {os.path.basename(urdf_path)}")

    robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dirs)
    print(f"[+] 로봇 모델명: '{robot.model.name}'")
    print(f"[+] 총 자유도(nq): {robot.model.nq} | 총 프레임 수: {robot.model.nframes}")

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=open_browser)
    viz.loadViewerModel()

    return robot, viz


def setup_coordinate_frames(viz: MeshcatVisualizer):
    """Adds 3D RGB coordinate axes (TF Triads) on key frames."""
    # Palm TF Frame (Large)
    viz.viewer["tf/palm"].set_object(g.triad(scale=0.12))
    
    # World Base TF Frame
    viz.viewer["tf/world"].set_object(g.triad(scale=0.15))
    viz.viewer["tf/world"].set_transform(tf.translation_matrix([0, 0, 0]))

    # Elbow and Wrist TF Frames
    viz.viewer["tf/elbow"].set_object(g.triad(scale=0.08))
    viz.viewer["tf/wrist"].set_object(g.triad(scale=0.08))

    # 5 Fingertips TF Frames
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for fname in finger_names:
        viz.viewer[f"tf/tip_{fname}"].set_object(g.triad(scale=0.04))

    # Palm indicator sphere
    viz.viewer["indicators/palm"].set_object(
        g.Sphere(radius=0.02),
        g.MeshLambertMaterial(color=0x3B82F6, opacity=0.85)  # Blue
    )

    # 5 Fingertips spheres
    colors = [0xEF4444, 0xF59E0B, 0x10B981, 0x6366F1, 0xEC4899]
    for idx, fname in enumerate(finger_names):
        viz.viewer[f"indicators/{fname}"].set_object(
            g.Sphere(radius=0.012),
            g.MeshLambertMaterial(color=colors[idx], opacity=0.95)
        )

    # Orange Arm Joint Spheres
    for j in range(1, 8):
        viz.viewer[f"indicators/joint_{j}"].set_object(
            g.Sphere(radius=0.028),
            g.MeshLambertMaterial(color=0xF97316, opacity=0.9)
        )

    # Target Green Cup
    viz.viewer["target_cup"].set_object(
        g.Cylinder(height=0.12, radius=0.038),
        g.MeshLambertMaterial(color=0x22C55E, opacity=0.85)
    )
    viz.viewer["target_cup"].set_transform(tf.translation_matrix([0.35, -0.15, 0.45]))


def update_tf_and_visuals(viz: MeshcatVisualizer, robot: RobotWrapper):
    """Updates coordinate axes transforms and prints live FK."""
    # Joint positions
    for j in range(1, 8):
        fname = f"openarm_right_link{j}"
        if robot.model.existFrame(fname):
            fid = robot.model.getFrameId(fname)
            T = robot.data.oMf[fid].homogeneous
            viz.viewer[f"indicators/joint_{j}"].set_transform(T)
            if j == 4:
                viz.viewer["tf/elbow"].set_transform(T)
            elif j == 7:
                viz.viewer["tf/wrist"].set_transform(T)

    # Palm Frame
    if robot.model.existFrame("palm_link"):
        fid = robot.model.getFrameId("palm_link")
        T_palm = robot.data.oMf[fid].homogeneous
        viz.viewer["tf/palm"].set_transform(T_palm)
        viz.viewer["indicators/palm"].set_transform(T_palm)

    # 5 Fingertips
    finger_keys = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for key, name in zip(finger_keys, finger_names):
        if robot.model.existFrame(key):
            fid = robot.model.getFrameId(key)
            T = robot.data.oMf[fid].homogeneous
            viz.viewer[f"tf/tip_{name}"].set_transform(T)
            viz.viewer[f"indicators/{name}"].set_transform(T)


def get_pose_preset(robot: RobotWrapper, preset_name: str):
    q = pin.neutral(robot.model)
    if preset_name == "ready":
        # 1. Ready Posture (Arm lifted forward, hand open)
        q[0] = 0.25   # shoulder pan
        q[1] = 0.75   # shoulder lift
        q[2] = 0.0    # arm roll
        q[3] = 1.35   # elbow pitch
        q[4] = 0.0    # wrist roll
        q[5] = 0.50   # wrist pitch
        q[6] = 0.0    # wrist yaw
        for j in range(7, robot.model.nq):
            q[j] = 0.30
    elif preset_name == "reach":
        # 2. Reach towards Cup
        q[0] = 0.15
        q[1] = 0.95
        q[2] = 0.0
        q[3] = 1.65
        q[4] = 0.0
        q[5] = 0.65
        q[6] = 0.0
        for j in range(7, robot.model.nq):
            q[j] = 0.15
    elif preset_name == "grasp":
        # 3. Grasping Cup
        q[0] = 0.15
        q[1] = 0.95
        q[2] = 0.0
        q[3] = 1.65
        q[4] = 0.0
        q[5] = 0.65
        q[6] = 0.0
        for j in range(7, robot.model.nq):
            q[j] = 0.85
    elif preset_name == "pour":
        # 4. Pouring (Wrist tilted)
        q[0] = 0.20
        q[1] = 0.85
        q[2] = 0.80   # Roll tilt
        q[3] = 1.45
        q[4] = 0.0
        q[5] = 0.70
        q[6] = -0.50
        for j in range(7, robot.model.nq):
            q[j] = 0.85
    return q


def print_hud(robot: RobotWrapper, q: np.ndarray, mode_name: str):
    # Calculate Palm 6D Pose
    fid = robot.model.getFrameId("palm_link")
    T = robot.data.oMf[fid]
    pos = T.translation
    rot = Rotation.from_matrix(T.rotation).as_euler('xyz', degrees=True)

    # Arm joint angles in degrees
    arm_deg = np.round(np.degrees(q[:7]), 1)

    print("\033[H\033[J", end="")  # Clear screen for smooth HUD
    print("=" * 72)
    print(f"       🦾 OpenArm-Tesollo Kinematics & TF Studio [모드: {mode_name}]")
    print("=" * 72)
    print(f" [1] 팔 관절 회전각도 (Joints 1 ~ 7):")
    print(f"     J1(Pan): {arm_deg[0]:+6.1f}° | J2(Lift): {arm_deg[1]:+6.1f}° | J3(Roll): {arm_deg[2]:+6.1f}°")
    print(f"     J4(Elb): {arm_deg[3]:+6.1f}° | J5(Roll): {arm_deg[4]:+6.1f}° | J6(Ptch): {arm_deg[5]:+6.1f}° | J7(Yaw): {arm_deg[6]:+6.1f}°")
    print("-" * 72)
    print(f" [2] 손바닥(Palm) 6D 위치 & 회전각 (Forward Kinematics):")
    print(f"     위치(X, Y, Z)  : [{pos[0]:+0.3f}, {pos[1]:+0.3f}, {pos[2]:+0.3f}] m")
    print(f"     회전(Roll,Pitch,Yaw): [{rot[0]:+0.1f}°, {rot[1]:+0.1f}°, {rot[2]:+0.1f}°]")
    print("-" * 72)
    print(" [3] 3D 좌표축(TF) 안내:")
    print("     🔴 Red Axis = X축 | 🟢 Green Axis = Y축 | 🔵 Blue Axis = Z축")
    print("=" * 72)
    print(" [단축키 안내]")
    print("   [1] Ready 자세    | [2] Reach (도달)    | [3] Grasp (쥐기)")
    print("   [4] Pour (기울임) | [5] Live Wave 모션  | [Ctrl+C] 종료")
    print("=" * 72)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_path = os.path.join(
        repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf",
        "openarm_tesollo", "openarm_tesollo.urdf"
    )

    robot, viz = create_openarm_visualizer(urdf_path, open_browser=True)
    setup_coordinate_frames(viz)

    # Start with live wave motion
    q_base = get_pose_preset(robot, "ready")
    viz.display(q_base)

    pin.forwardKinematics(robot.model, robot.data, q_base)
    pin.updateFramePlacements(robot.model, robot.data)
    update_tf_and_visuals(viz, robot)

    print("\n[+] Meshcat 3D Studio 활성화 완료: " + viz.viewer.url())
    time.sleep(1.0)

    t = 0.0
    dt = 0.03
    last_print = 0.0
    mode = "Live Wave 모션 (실시간 기구학)"

    try:
        while True:
            # Live dynamic kinematics wave
            q = q_base.copy()
            q[0] += 0.25 * np.sin(1.2 * t)
            q[1] += 0.18 * np.sin(1.5 * t)
            q[3] += 0.22 * np.cos(1.0 * t)
            q[5] += 0.20 * np.sin(2.0 * t)
            q[6] += 0.15 * np.cos(1.8 * t)

            for f_idx in range(7, robot.model.nq):
                q[f_idx] += 0.28 * np.sin(2.2 * t + 0.25 * f_idx)

            # Update FK and 3D visualizer
            viz.display(q)
            pin.forwardKinematics(robot.model, robot.data, q)
            pin.updateFramePlacements(robot.model, robot.data)
            update_tf_and_visuals(viz, robot)

            # Print Real-time HUD every 0.15 sec
            if time.time() - last_print > 0.15:
                print_hud(robot, q, mode)
                last_print = time.time()

            time.sleep(dt)
            t += dt
    except KeyboardInterrupt:
        print("\n\n[*] OpenArm Studio를 정상 종료했습니다.")


if __name__ == "__main__":
    main()
