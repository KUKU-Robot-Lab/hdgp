"""
Pinocchio + Meshcat 1-Second Verification Script
Verifies that Pinocchio and Meshcat work together seamlessly.
"""

import os
import sys
import time
import numpy as np

try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat
except ImportError as e:
    print(f"[Error] Required packages not found: {e}")
    sys.exit(1)


def main():
    print("=" * 60)
    print(f"[*] Pinocchio Version: {pin.__version__}")
    print(f"[*] Meshcat Visualizer: Available")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_path = os.path.join(
        repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf",
        "openarm_tesollo", "openarm_tesollo.urdf"
    )

    if not os.path.exists(urdf_path):
        print(f"[!] Warning: URDF not found at {urdf_path}, building simple test model...")
        model = pin.buildSampleModelManipulator()
        geom_model = pin.GeometryModel()
        robot = None
    else:
        mesh_dirs = [
            os.path.join(repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf"),
            os.path.join(repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "openarm_tesollo"),
        ]
        robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dirs)
        model = robot.model
        geom_model = robot.visual_model
        print(f"[+] Loaded Model: '{model.name}' (DoF: {model.nq})")

    # Neutral joint configuration
    q0 = pin.neutral(model)

    # Compute Forward Kinematics
    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    print(f"[+] Computed FK for {model.nframes} frames successfully.")

    # Initialize Meshcat Visualizer
    print("\n[*] Initializing Meshcat Visualizer (Opening Browser at localhost:7001)...")
    if robot is not None:
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    else:
        viz = MeshcatVisualizer(model, geom_model, geom_model)

    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(q0)

    print("\n" + "=" * 60)
    print("[SUCCESS] Meshcat is running at:")
    print(f"          {viz.viewer.url()}")
    print("=" * 60)

    # Gentle motion test
    print("[*] Running 3-second gentle joint test motion...")
    dt = 0.03
    t_end = time.time() + 3.0
    t = 0.0
    while time.time() < t_end:
        q = q0.copy()
        for j in range(min(4, model.nq)):
            q[j] += 0.3 * np.sin(2.0 * t + j)
        viz.display(q)
        time.sleep(dt)
        t += dt

    # Return to neutral
    viz.display(q0)
    print("[+] Test completed successfully!")


if __name__ == "__main__":
    main()
