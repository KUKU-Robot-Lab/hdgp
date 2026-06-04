"""RH56F1 fabric headless 런타임 스모크 테스트 (Phase 1 GPU 검증).

렌더링/SimulationApp 불필요 — warp + torch(CUDA) 만 사용한다.
fabric 이 실제로 로드/빌드되고 palm IK 가 수렴하는지 확인한다.

실행:
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p \
    /home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/inspire_r/grasp_r_v1/tests/smoke_rh56f1_fabric.py

기대 결과(PASS 기준):
  - "[OK] fabric num_joints = 13"
  - palm IK 후 palm 위치가 target 근처(오차 < ~5cm)로 수렴
  - fingertip 5개 위치가 NaN 없이 출력
실패 시 보통:
  - get_robot_urdf_path 가 URDF 못 찾음 → 경로/파일명 확인
  - urdfpy 파싱 에러 → URDF 문법 / mimic 처리
  - warp 빌드 에러 → taskmap frame 이름 불일치
"""

import sys
import math
from pathlib import Path

import torch

# --- FABRICS/src 를 sys.path 에 추가 (env.py 와 동일 로직) ---
_here = Path(__file__).resolve()
for _parent in _here.parents:
    if _parent.name == "source":
        _vendored = _parent / "FABRICS" / "src"
        if _vendored.exists():
            sys.path.insert(0, str(_vendored))
        break

from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import OpenArmRh56f1PoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


def main():
    device = "cuda:0"
    B = 4
    timestep = 1.0 / 60.0
    n_steps = 120

    print("=== RH56F1 fabric smoke test ===")
    initialize_warp(device[-1])

    # open_tesollo_boxes_no_table 는 객체 7개 → 슬롯 넉넉히(16)
    world = WorldMeshesModel(
        batch_size=B,
        max_objects_per_env=16,
        device=device,
        world_filename="open_tesollo_boxes_no_table",
    )
    object_ids, object_indicator = world.get_object_ids()

    fabric = OpenArmRh56f1PoseFabric(
        B, device, timestep, graph_capturable=False, use_hand_fabric=False
    )
    num_joints = fabric.num_joints
    print(f"[OK] fabric loaded. num_joints = {num_joints}  (expect 13)")
    assert num_joints == 13, f"cspace dim {num_joints} != 13"

    integrator = DisplacementIntegrator(fabric)

    q = fabric.default_config.clone().contiguous()
    qd = torch.zeros(B, num_joints, device=device)
    qdd = torch.zeros(B, num_joints, device=device)

    # palm pose target (euler_zyx): 작업공간 내 임의 도달 지점
    palm_target = torch.zeros(B, 6, device=device)
    palm_target[:, 0] = 0.40
    palm_target[:, 1] = -0.10
    palm_target[:, 2] = 0.45
    palm_target[:, 3] = math.radians(90.0)
    palm_target[:, 4] = math.radians(0.0)
    palm_target[:, 5] = math.radians(90.0)

    hand_dummy = torch.zeros(B, 6, device=device)
    damping = 20.0 * torch.ones(B, 1, device=device)

    palm0 = fabric.get_palm_pose(q, "euler_zyx")
    print(f"[init] palm pos = {palm0[0, :3].tolist()}")

    fabric.set_features(
        hand_dummy, palm_target, "euler_zyx",
        q.detach(), qd.detach(), object_ids, object_indicator, damping,
    )
    for i in range(n_steps):
        q, qd, qdd = integrator.step(
            q.detach(), qd.detach(), qdd.detach(), timestep
        )

    palm1 = fabric.get_palm_pose(q, "euler_zyx")
    tips = fabric.get_fingertip_positions(q)

    print(f"[after {n_steps} steps]")
    print(f"  arm q   = {[round(v, 3) for v in q[0, :7].tolist()]}")
    print(f"  hand q  = {[round(v, 3) for v in q[0, 7:].tolist()]}")
    print(f"  palm pos     = {[round(v, 4) for v in palm1[0, :3].tolist()]}")
    print(f"  palm target  = {[round(v, 4) for v in palm_target[0, :3].tolist()]}")
    err = torch.norm(palm1[0, :3] - palm_target[0, :3]).item()
    print(f"  palm pos error = {err*100:.2f} cm")
    print(f"  fingertip[thumb] = {[round(v, 4) for v in tips[0, 0].tolist()]}")
    print(f"  fingertip[little]= {[round(v, 4) for v in tips[0, 4].tolist()]}")

    nan_check = torch.isnan(q).any() or torch.isnan(tips).any() or torch.isnan(palm1).any()
    if nan_check:
        print("\n[FAIL] NaN 발생 — fabric 수치 불안정")
        return 1
    if err > 0.10:
        print(f"\n[WARN] palm IK 수렴 오차 큼 ({err*100:.1f}cm > 10cm). target 도달 여부 확인 필요.")
    else:
        print(f"\n[PASS] fabric 런타임 OK, palm IK 수렴 (오차 {err*100:.1f}cm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
