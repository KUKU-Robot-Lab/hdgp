"""P3.0 — Fabrics IK raw-app 스모크 (live_policy_fluid_plan P3 기반 리스크).

질문: OpenArmTeoslloPoseFabric + DisplacementIntegrator 가 SimulationContext 없이(raw SimulationApp)
생성·step 되는가. P0 §3 에서 "warp init + Fabrics 가 SimContext 없이 되는지 미검증" 으로 플래그.
이게 되어야 action 파이프라인(P0 §4)을 raw-app 라이브 루프로 lift 가능(안 되면 전략 변경).

방법: raw-app 에서 Fabrics 생성 → palm pose target 주고 integrator 를 120 step → fabric_q(관절)가
finite 하게 target 향해 evolve 하는지 확인. 로봇 USD 불필요(Fabrics 는 관절 텐서만 다루는 순수 kinematic).

실행: ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p3_0_fabrics_smoke.py --headless
"""

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="P3.0 Fabrics raw-app smoke")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import torch  # noqa: E402

# FABRICS 경로 (env.py 와 동일 로직)
_FAB = "/home/user/rl_ws/hdgp/source/FABRICS/src"
if _FAB not in sys.path:
    sys.path.insert(0, _FAB)

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp  # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel  # noqa: E402

DEV = "cuda:0"
N = args.num_envs
DT = 1.0 / 60.0


def main():
    print("[P3.0] warp init (raw-app, SimulationContext 없음)...", flush=True)
    initialize_warp(DEV[-1])

    world = WorldMeshesModel(batch_size=N, max_objects_per_env=8, device=DEV,
                             world_filename="open_tesollo_boxes_pour_v5")
    object_ids, object_indicator = world.get_object_ids()
    print("[P3.0] WorldMeshesModel OK", flush=True)

    fabric = OpenArmTeoslloPoseFabric(N, DEV, DT, graph_capturable=False,
                                      use_hand_fabric=False, palm_position_only=False)
    _csa = fabric.fabric_params["cspace_attractor"]
    _csa["min_isotropic_mass"] = 3.0; _csa["max_isotropic_mass"] = 3.0
    nj = fabric.num_joints
    integrator = DisplacementIntegrator(fabric)
    print(f"[P3.0] Fabric OK, num_joints={nj}", flush=True)

    # 초기 관절: arm start(cfg init) + hand 0
    q0 = torch.zeros(N, nj, device=DEV)
    arm_start = torch.tensor([0.5, 0.1, 0.4, 0.6, -0.2, 0.0, 0.0], device=DEV)
    q0[:, :7] = arm_start.unsqueeze(0)
    fabric_q = q0.clone().contiguous()
    fabric_qd = torch.zeros(N, nj, device=DEV)
    fabric_qdd = torch.zeros(N, nj, device=DEV)
    hand_pca = torch.zeros(N, 5, device=DEV)
    damping = 20.0 * torch.ones(N, 1, device=DEV)

    # palm pose target (xyzw quat) — 시작에서 +x 로 8cm 이동 목표(도달성 무관, evolve 확인용)
    palm_target = torch.zeros(N, 7, device=DEV)
    palm_target[:, :3] = torch.tensor([0.45, -0.10, 0.45], device=DEV)  # 대략 workspace 내
    palm_target[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEV)  # identity xyzw

    q_hist = [fabric_q[0, :7].detach().cpu().numpy().copy()]
    for k in range(args.steps):
        fabric.set_features(hand_pca, palm_target, "quaternion",
                            fabric_q.detach(), fabric_qd.detach(),
                            object_ids, object_indicator, damping)
        for _ in range(2):  # fabric_decimation
            fabric_q, fabric_qd, fabric_qdd = integrator.step(
                fabric_q.detach(), fabric_qd.detach(), fabric_qdd.detach(), DT)
        if k in (0, args.steps // 2, args.steps - 1):
            q_hist.append(fabric_q[0, :7].detach().cpu().numpy().copy())

    finite = bool(torch.isfinite(fabric_q).all().item())
    moved = float(np.linalg.norm(q_hist[-1] - q_hist[0]))
    palm_j = fabric.get_taskmap_jacobian("palm")  # taskmap 접근 확인(B-full 에 필요)
    j_ok = bool(torch.isfinite(palm_j).all().item())

    print(f"[P3.0] step {args.steps} 완료 | fabric_q finite={finite} | "
          f"arm |Δq|={moved:.4f} rad | palm Jacobian shape={tuple(palm_j.shape)} finite={j_ok}", flush=True)
    print(f"[P3.0] arm q 궤적(j1-7) start→end:", flush=True)
    print(f"[P3.0]   start={np.array2string(q_hist[0], precision=3)}", flush=True)
    print(f"[P3.0]   end  ={np.array2string(q_hist[-1], precision=3)}", flush=True)

    ok = finite and moved > 1e-3 and j_ok
    print(f"[P3.0] ===== 판정: {'PASS — Fabrics raw-app 작동, 라이브 루프 가능' if ok else 'FAIL — 전략 변경 필요'} =====",
          flush=True)
    app.close()


if __name__ == "__main__":
    main()
