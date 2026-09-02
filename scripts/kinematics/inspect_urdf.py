"""
Pure URDF Inspector & Kinematics Ground Truth Studio (Zero-Touch / Isaac Lab Style)
- 100% pure parsing from URDF file (No hardcoding, no manual offsets).
- Automatically iterates through ALL URDF frames and attaches exact RGB coordinate axes.
- Automatically extracts exact joint limits (lower/upper) from the URDF.
- Displays Forward Kinematics (FK) matching ground-truth physics engines.
"""

import os
import sys
import argparse
import numpy as np

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
from scipy.spatial.transform import Rotation


def inspect_urdf(urdf_path: str, show_all_tfs: bool = True, tf_scale: float = 0.05):
    """
    Zero-touch pure URDF loader.
    Parses exact kinematics directly from the URDF file using Pinocchio's C++ urdfdom parser.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    # Standard search directories for meshes
    mesh_search_dirs = [
        urdf_dir,
        os.path.join(urdf_dir, "meshes"),
        os.path.join(os.path.dirname(urdf_dir), "openarm_tesollo", "meshes"),
        os.path.join(os.path.dirname(urdf_dir), "kuka_allegro", "meshes"),
    ]
    mesh_search_dirs = [d for d in mesh_search_dirs if os.path.isdir(d)]

    print("=" * 75)
    print(f"[*] Pure URDF Ground-Truth Parsing: {os.path.basename(urdf_path)}")
    print(f"[*] URDF Absolute Path: {os.path.abspath(urdf_path)}")
    print("=" * 75)

    # 1. Pure build from URDF without any modifications
    robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_search_dirs)
    model = robot.model

    print(f"\n[URDF 속성 파싱 결과]")
    print(f" - Robot Name        : {model.name}")
    print(f" - Config DoF (nq)   : {model.nq}")
    print(f" - Velocity DoF (nv) : {model.nv}")
    print(f" - Total Joints      : {model.njoints} (including universe)")
    print(f" - Total Frames      : {model.nframes}")

    # 2. Extract joint limits directly from URDF
    print("\n[URDF 원본 관절 정보 및 한계값(Limits)]")
    for j_id in range(1, model.njoints):
        j_name = model.names[j_id]
        idx_q = model.joints[j_id].idx_q
        idx_v = model.joints[j_id].idx_v
        nq_j = model.joints[j_id].nq
        if nq_j > 0:
            lower = model.lowerPositionLimit[idx_q:idx_q+nq_j]
            upper = model.upperPositionLimit[idx_q:idx_q+nq_j]
            print(f"  [{j_id:2d}] Joint '{j_name:<30}': Limit = [{lower[0]:+.3f}, {upper[0]:+.3f}] rad ({np.degrees(lower[0]):+6.1f}° ~ {np.degrees(upper[0]):+6.1f}°)")

    # 3. Choose visual or collision geometry
    has_visual = robot.visual_model.ngeoms > 0
    display_model = robot.visual_model if has_visual else robot.collision_model

    # 4. Initialize Meshcat Visualizer
    viz = MeshcatVisualizer(robot.model, robot.collision_model, display_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # 5. Compute Ground Truth Forward Kinematics at Neutral (Zero) Configuration
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, robot.data, q0)
    pin.updateFramePlacements(model, robot.data)
    viz.display(q0)

    # 6. Automatic Ground-Truth TF Axes (Isaac Lab / RViz style)
    # Automatically attaches exact RGB coordinate triad to every single frame defined in the URDF
    if show_all_tfs:
        print(f"\n[*] URDF에 정의된 모든 {model.nframes}개 프레임에 3D 좌표축(TF Triad)을 자동 부착합니다...")
        for frame_id, frame in enumerate(model.frames):
            # Skip universe/root if identical to world
            tf_node_name = f"tf_frames/{frame.name}"
            viz.viewer[tf_node_name].set_object(g.triad(scale=tf_scale))
            
            # Set exact transform from FK
            T = robot.data.oMf[frame_id].homogeneous
            viz.viewer[tf_node_name].set_transform(T)

    print("\n" + "=" * 75)
    print(f"[성공] Isaac Lab 스타일 순수 URDF 뷰어가 구동되었습니다:")
    print(f"       {viz.viewer.url()}")
    print("-" * 75)
    print("  특징:")
    print("   1. URDF 원본 XML에 명시된 관절 축과 위치만 100% 그대로 반영합니다.")
    print("   2. URDF의 모든 프레임에 🔴X 🟢Y 🔵Z 좌표축이 원본 그대로 표시됩니다.")
    print("   3. 종료: 터미널에서 Ctrl + C")
    print("=" * 75 + "\n")

    # Print summary of first 15 key frame FK positions
    print("[주요 프레임 원점 위치 (Forward Kinematics Ground-Truth)]")
    for frame_id in range(min(20, model.nframes)):
        frame = model.frames[frame_id]
        pos = np.round(robot.data.oMf[frame_id].translation, 4)
        rot = Rotation.from_matrix(robot.data.oMf[frame_id].rotation).as_euler('xyz', degrees=True)
        rot = np.round(rot, 1)
        print(f"  Frame [{frame_id:2d}] {frame.name:<32} -> Pos: {pos} m | Rot(rpy): {rot}°")

    try:
        import signal
        running = True
        def sig_handler(s, f):
            nonlocal running
            running = False
        signal.signal(signal.SIGINT, sig_handler)

        while running:
            import time
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n[*] URDF 인스펙터를 종료합니다.")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_urdf = os.path.join(
        repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf",
        "openarm_tesollo", "openarm_tesollo.urdf"
    )

    parser = argparse.ArgumentParser(description="Pure URDF Ground-Truth Kinematics Inspector")
    parser.add_argument("urdf", nargs="?", default=default_urdf, help="Path to ANY robot .urdf file")
    parser.add_argument("--scale", type=float, default=0.06, help="Scale of RGB coordinate axis triads")
    parser.add_argument("--no-tf", action="store_true", help="Disable automatic TF axis rendering")

    args = parser.parse_args()

    inspect_urdf(args.urdf, show_all_tfs=not args.no_tf, tf_scale=args.scale)


if __name__ == "__main__":
    main()
