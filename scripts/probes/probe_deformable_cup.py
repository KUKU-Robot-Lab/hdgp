"""Gate A 물리 검증: segmented-shell deformable cup 스폰 → 안쪽 힘 → 패널각 발생 →
스프링 복원 → 리셋 시 각 0 복귀.

articulation(자유부유 base + 12 접선 힌지 스프링 패널)이 실제로 시뮬레이트되고,
반경 방향 외력에 패널이 안쪽으로 눌렸다가 스프링으로 복원되는지 확인한다.

사용:
  cd /home/user/rl_ws/IsaacLab
  ./isaaclab.sh -p ../hdgp/scripts/probes/probe_deformable_cup.py
  (또는 hdgp 루트에서 절대경로)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=str,
                    default="/home/user/rl_ws/hdgp/assets/cup/deformable_cup.usd")
parser.add_argument("--push-force", type=float, default=0.3,
                    help="패널에 인가할 반경 안쪽 힘(N). 경량 패널이라 작게.")
parser.add_argument("--stiffness", type=float, default=0.1)
parser.add_argument("--damping", type=float, default=0.02)
parser.add_argument("--armature", type=float, default=1.0e-3,
                    help="관절 로터 관성 — 경량 패널 안정화(kg·m²)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def main() -> None:
    # 중력 0: 자유부유 base + 대칭 안쪽 힘(합력 0) → base 정지, 스프링 컴플라이언스만 측정.
    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 120.0, device=args.device, gravity=(0.0, 0.0, 0.0)))
    sim.set_camera_view([0.4, 0.4, 0.3], [0.0, 0.0, 0.0])

    # 지면 + 조명
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light",
                                                  sim_utils.DomeLightCfg(intensity=2000.0))

    cup_cfg = ArticulationCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
        # 패널 관절 = 정책이 제어하지 않는 수동 스프링(target=default 0). armature로 안정화.
        actuators={
            "panels": ImplicitActuatorCfg(
                joint_names_expr=["revolute_.*"],
                stiffness=args.stiffness,
                damping=args.damping,
                armature=args.armature,
                effort_limit=1.0e6,
            ),
        },
    )
    cup = Articulation(cup_cfg)

    sim.reset()
    print("[probe] joint_names:", cup.joint_names, flush=True)
    n_joints = cup.num_joints
    print(f"[probe] num_joints={n_joints} num_bodies={cup.num_bodies}", flush=True)

    def settle(steps: int) -> None:
        for _ in range(steps):
            cup.write_data_to_sim()
            sim.step()
            cup.update(sim.get_physics_dt())

    def max_abs_angle_deg() -> float:
        q = cup.data.joint_pos  # (1, n_joints) rad
        return float(q.abs().max()) * 180.0 / 3.14159265

    # 1) 무외력 정착 → 각 ~0 유지(스프링이 중력에 버팀)
    settle(120)
    rest_deg = max_abs_angle_deg()
    print(f"[probe] (1) 무외력 정착: max|angle|={rest_deg:.3f} deg", flush=True)

    # 2) 패널에 반경 안쪽 힘 인가 → 각 발생
    #    body_names에서 panel 인덱스 찾기, 각 패널 중심에 -radial 방향 힘.
    body_names = cup.body_names
    panel_ids = [i for i, n in enumerate(body_names) if n.startswith("panel_")]
    print(f"[probe] panel body count={len(panel_ids)}", flush=True)
    pos_w = cup.data.body_pos_w[0]  # (num_bodies, 3)
    root_xy = cup.data.root_pos_w[0, :2]
    forces = torch.zeros((1, cup.num_bodies, 3), device=cup.device)
    torques = torch.zeros((1, cup.num_bodies, 3), device=cup.device)
    for i in panel_ids:
        radial = pos_w[i, :2] - root_xy
        n = torch.norm(radial)
        if n > 1e-6:
            inward = -radial / n
            forces[0, i, 0] = inward[0] * args.push_force
            forces[0, i, 1] = inward[1] * args.push_force
    cup.set_external_force_and_torque(forces, torques)
    settle(120)
    pushed_deg = max_abs_angle_deg()
    print(f"[probe] (2) 안쪽 힘 {args.push_force}N: max|angle|={pushed_deg:.3f} deg", flush=True)

    # 3) 힘 제거 → 스프링 복원
    cup.set_external_force_and_torque(
        torch.zeros_like(forces), torch.zeros_like(torques))
    settle(180)
    restored_deg = max_abs_angle_deg()
    print(f"[probe] (3) 힘 제거 복원: max|angle|={restored_deg:.3f} deg", flush=True)

    # 4) 리셋 → 각 0
    cup.reset()
    default_q = cup.data.default_joint_pos.clone()
    cup.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))
    settle(5)
    reset_deg = max_abs_angle_deg()
    print(f"[probe] (4) 리셋: max|angle|={reset_deg:.3f} deg", flush=True)

    ok_rest = rest_deg < 3.0
    ok_push = pushed_deg > rest_deg + 3.0
    ok_restore = restored_deg < pushed_deg * 0.5 + 2.0
    ok_reset = reset_deg < 1.0
    verdict = "PASS" if (ok_rest and ok_push and ok_restore and ok_reset) else "FAIL"
    print(f"[probe] GATE A: rest={ok_rest} push={ok_push} restore={ok_restore} "
          f"reset={ok_reset} => {verdict}", flush=True)

    app.close()


if __name__ == "__main__":
    main()
