"""openarm_rh56f1 양팔 fabric 서버 스모크: 로드 + 오른손 palm IK 왕복 수렴.

warp 런타임(서버 전용)에서 fabric 을 단독 인스턴스화하여 검증한다(env 불필요):
  1) num_joints == 26 (양팔 cspace)
  2) 오른손 palm(r_hl_palm_sensor) IK 왕복: 목표를 초기 palm pose +3cm 로 주고
     적분 → 최종 palm 위치가 목표에 수렴(< pos_tol).
  3) 왼팔/왼손은 능동 IK 없이 default_config 중립 유지(< hold_tol).

로컬(warp 없음)에서는 실행 불가 — 서버에서:
  python verify_fabric_load_ik.py --batch_size 4
PASS 시 exit 0, 실패 시 assert 로 비정상 종료.
"""

import argparse

import torch

from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import (
    OpenArmRh56f1PoseFabric,
    NUM_DOF,
    NUM_SIDE_DOF,
)
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp

import warp as wp


def main():
    parser = argparse.ArgumentParser(description="rh56f1 bimanual fabric load+IK smoke.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device_int", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--pos_tol", type=float, default=0.015, help="IK 수렴 위치 허용(m)")
    parser.add_argument("--hold_tol", type=float, default=0.05, help="왼쪽 중립 유지 허용(rad)")
    parser.add_argument("--max_objects", type=int, default=20)
    args = parser.parse_args()

    device = f"cuda:{args.device_int}"
    initialize_warp(str(args.device_int))

    control_rate = 60.0
    timestep = 1.0 / control_rate
    b = args.batch_size

    fabric = OpenArmRh56f1PoseFabric(b, device, timestep, graph_capturable=False)
    integrator = DisplacementIntegrator(fabric)

    # --- Gate 1: DOF ---
    assert fabric.num_joints == NUM_DOF == 26, \
        f"num_joints={fabric.num_joints}, expected 26"
    print(f"[Gate1] num_joints == {fabric.num_joints} (26) OK")

    # 초기 상태 = default_config
    q = fabric.default_config.clone().contiguous()
    qd = torch.zeros(b, fabric.num_joints, device=device)
    qdd = torch.zeros(b, fabric.num_joints, device=device)
    q_default = q.clone()

    # 외부 오브젝트 없음(자기충돌만): indicator 전부 0
    zeros_i = torch.zeros(b, args.max_objects, dtype=torch.int64, device=device)
    object_ids = wp.from_torch(zeros_i, dtype=wp.uint64)
    object_indicator = wp.from_torch(zeros_i.clone(), dtype=wp.uint64)

    hand_target = torch.zeros(b, 6, device=device)  # use_hand_fabric=False → 무시

    # 초기 오른손 palm pose(euler_zyx) → 목표 = +3cm x
    palm0 = fabric.get_palm_pose(q.detach(), "euler_zyx")  # (b, 6)
    palm_target = palm0.clone()
    palm_target[:, 0] += 0.03

    damping_gain = torch.full((b, 1), 10.0, device=device)

    for _ in range(args.steps):
        fabric.set_features(
            hand_target, palm_target, "euler_zyx",
            q.detach(), qd.detach(),
            object_ids, object_indicator,
            damping_gain,
        )
        q, qd, qdd = integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)

    # --- Gate 2: IK 왕복 수렴 ---
    palm_final = fabric.get_palm_pose(q.detach(), "euler_zyx")
    pos_err = torch.norm(palm_final[:, :3] - palm_target[:, :3], dim=1)
    max_pos_err = pos_err.max().item()
    print(f"[Gate2] IK palm pos err max={max_pos_err*1000:.2f}mm (tol={args.pos_tol*1000:.0f}mm)")
    assert max_pos_err < args.pos_tol, f"IK 미수렴: {max_pos_err*1000:.2f}mm"

    # --- Gate 3: 왼쪽 중립 유지 ---
    left = slice(NUM_SIDE_DOF, NUM_DOF)  # [13:26]
    left_drift = (q[:, left] - q_default[:, left]).abs().max().item()
    print(f"[Gate3] left(13:26) drift max={left_drift:.4f}rad (tol={args.hold_tol})")
    assert left_drift < args.hold_tol, f"왼쪽 중립 이탈: {left_drift:.4f}rad"

    print("[PASS] rh56f1 bimanual fabric load + right palm IK + left hold OK")


if __name__ == "__main__":
    main()
