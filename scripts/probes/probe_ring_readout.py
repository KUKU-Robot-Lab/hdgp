"""Feasibility: maximal-coord 링 패널을 RigidObjectCollection으로 감싸 변형(회전)을 읽는다.

닫힌 링은 articulation 불가(maximal-coord) → joint_pos를 못 읽음. 대신 패널(rigid body)을
RigidObjectCollection으로 감싸 object_quat_w를 읽고, rest 대비 회전각으로 변형 신호를 만든다.
USD 조인트(edge+bottom 스프링)는 physx가 물리로 담당, collection은 read-only view.

이게 되면 학습 env 통합(변형 신호=이 방식)이 가능. paddle로 눌러 변형이 추적되는지 확인.

  cd /home/user/rl_ws/hdgp
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ring_readout.py
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=str,
                    default="/home/user/rl_ws/hdgp/assets/cup/deformable_cup_ring.usd")
parser.add_argument("--panels", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import (RigidObject, RigidObjectCfg,  # noqa: E402
                             RigidObjectCollection, RigidObjectCollectionCfg)
from isaaclab.sim import SimulationContext  # noqa: E402


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # 링 컵 raw 스폰(base 정적, 패널 동적, edge+bottom 조인트)
    cup_cfg = sim_utils.UsdFileCfg(usd_path=args.usd)
    cup_cfg.func("/World/Cup", cup_cfg, translation=(0.0, 0.0, 0.10))

    # 패널들을 RigidObjectCollection으로 래핑(spawn=None=기존 prim view)
    objs = {}
    for i in range(args.panels):
        objs[f"panel_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/Cup/panel_{i:02d}", spawn=None)
    panels = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=objs))

    # paddle 2개(kinematic)로 양쪽 누름
    def paddle(prim, x):
        return RigidObject(RigidObjectCfg(
            prim_path=prim,
            spawn=sim_utils.CuboidCfg(
                size=(0.012, 0.06, 0.12),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 0.0885))))
    left = paddle("/World/PL", -0.085)
    right = paddle("/World/PR", 0.085)

    sim.reset()
    print(f"[readout] collection num_objects={panels.num_objects} "
          f"quat shape={tuple(panels.data.object_quat_w.shape)}", flush=True)

    rest = panels.data.object_quat_w.clone()  # (1, 12, 4) 초기(rest) 방향

    def deform_deg():
        cur = panels.data.object_quat_w  # (1,12,4)
        dot = (cur * rest).sum(-1).abs().clamp(max=1.0)  # (1,12)
        ang = 2.0 * torch.acos(dot) * 180.0 / math.pi
        return ang

    x0, xc = 0.085, 0.032
    for f in range(160):
        t = f / 160
        xr = x0 - (x0 - xc) * (t / 0.4) if t < 0.4 else (
            xc if t < 0.65 else xc + (x0 - xc) * ((t - 0.65) / 0.35))
        left.write_root_pose_to_sim(torch.tensor([[-xr, 0, 0.0885, 1, 0, 0, 0]], device=args.device))
        right.write_root_pose_to_sim(torch.tensor([[xr, 0, 0.0885, 1, 0, 0, 0]], device=args.device))
        left.write_data_to_sim(); right.write_data_to_sim()
        sim.step()
        left.update(sim.get_physics_dt()); right.update(sim.get_physics_dt())
        panels.update(sim.get_physics_dt())
        if f % 20 == 0:
            d = deform_deg()
            nan = bool(torch.isnan(panels.data.object_quat_w).any())
            print(f"[readout] f{f} xr={xr:.3f} deform max={float(d.max()):.1f} "
                  f"mean={float(d.mean()):.1f} deg  nan={nan}", flush=True)

    d = deform_deg()
    ok = (panels.num_objects == args.panels) and (float(d.max()) > 5.0)
    print(f"[readout] VERDICT: read OK={panels.num_objects==args.panels} "
          f"deform_tracked={float(d.max())>5.0} => {'PASS' if ok else 'CHECK'}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
