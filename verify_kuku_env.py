import sys
import time
import os

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(module_name, is_success, details=""):
    status = " [PASS]" if is_success else "❌ [FAIL]"
    print(f"{status} {module_name:<20} {details}")

def main():
    print_header("KUKU-Robot Pinocchio + Meshcat + Fabrics 환경 검증")
    
    all_passed = True
    
    # 1. Python 버전 검증
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_py_valid = sys.version_info.major == 3 and sys.version_info.minor >= 8
    print_result("Python Version", is_py_valid, f"(Detected: v{py_ver})")
    if not is_py_valid: all_passed = False

    # 2. Pinocchio 검증
    try:
        import pinocchio as pin
        print_result("Pinocchio", True, f"(v{pin.__version__})")
    except ImportError as e:
        print_result("Pinocchio", False, f"ImportError: {e}")
        all_passed = False

    # 3. Meshcat 검증
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
        meshcat_ver = getattr(meshcat, "__version__", "0.3.2")
        print_result("Meshcat", True, f"(v{meshcat_ver})")
    except ImportError as e:
        print_result("Meshcat", False, f"ImportError: {e}")
        all_passed = False

    # 4. CasADi & Fabrics 검증
    try:
        import casadi
        print_result("CasADi", True, f"(v{casadi.__version__})")
    except ImportError as e:
        print_result("CasADi", False, f"ImportError: {e}")
        all_passed = False

    try:
        import fabrics
        fabrics_ver = getattr(fabrics, "__version__", "0.10.0")
        print_result("Fabrics", True, f"(Geometric Fabrics v{fabrics_ver} Loaded)")
    except ImportError as e:
        print_result("Fabrics", False, f"ImportError: {e}")
        all_passed = False

    # 5. 3D Mesh & Math 종속성 검증
    for pkg_name in ["numpy", "scipy", "trimesh", "yaml"]:
        try:
            __import__(pkg_name)
            print_result(pkg_name.capitalize(), True, "Loaded")
        except ImportError as e:
            print_result(pkg_name.capitalize(), False, f"Missing: {e}")
            all_passed = False

    print_header("3D Web Graphics & Kinematics Server 구동 테스트")

    if not all_passed:
        print("\n [경고] 일부 필수 라이브러리가 설치되지 않았습니다.")
        print("Conda 환경(kuku_kinematics)이 활성화되어 있는지 확인해주세요.")
        return

    try:
        from pinocchio.visualize import MeshcatVisualizer
        from pinocchio.robot_wrapper import RobotWrapper

        # 기본 UR5 샘플 로봇 모델 또는 KUKU/OpenArm 모델 빌드
        repo_root = os.path.dirname(os.path.abspath(__file__))
        urdf_candidates = [
            os.path.join(repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "kuka_allegro", "kuka_allegro.urdf"),
            os.path.join(repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "openarm_tesollo", "openarm_tesollo.urdf"),
        ]

        robot = None
        for candidate in urdf_candidates:
            if os.path.exists(candidate):
                mesh_dirs = [
                    os.path.dirname(candidate),
                    os.path.join(os.path.dirname(candidate), "meshes")
                ]
                robot = RobotWrapper.BuildFromURDF(candidate, mesh_dirs)
                print(f"[*] 로봇 모델 로드 성공: '{robot.model.name}' ({candidate})")
                break

        if robot is None:
            model = pin.buildSampleModelManipulator()
            geom = pin.GeometryModel()
            viz = MeshcatVisualizer(model, geom, geom)
            q0 = pin.neutral(model)
        else:
            viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
            q0 = pin.neutral(robot.model)
            if robot.model.nq >= 7:
                q0[0] = 0.2
                q0[1] = 0.6
                q0[3] = 1.2
        
        # Meshcat 서버 팝업
        viz.initViewer(open=True)
        viz.loadViewerModel()
        
        # 샘플 컵 객체 및 좌표계 축 렌더링
        viz.viewer["verification_target"].set_object(
            g.Cylinder(height=0.1, radius=0.035),
            g.MeshLambertMaterial(color=0x22C55E, opacity=0.8)
        )
        viz.viewer["verification_target"].set_transform(tf.translation_matrix([0.4, 0.2, 0.2]))
        
        # 로봇 관절 시각화
        viz.display(q0)
        
        print_result("Meshcat Server", True, f"{viz.viewer.url()} 접속 확인 완료!")
        print("\n 웹 브라우저가 자동으로 열립니다.")
        print("  - 화면에 3D 로봇과 초록색 컵(Target)이 보이면 100% 정상 세팅 완료입니다!")
        print("  - 종료하려면 터미널에서 Ctrl+C를 누르세요.\n")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[완료] 검증 테스트를 정상적으로 종료합니다.")
    except Exception as e:
        print_result("3D Rendering Test", False, f"Error: {e}")

if __name__ == "__main__":
    main()
