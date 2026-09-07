#!/usr/bin/env python3
"""좌팔 그리퍼 **유효 개구**를 PhysX 안에서 직접 잰다 — 콜라이더 근사별로.

왜 필요한가
-----------
`GRIPPER_MAX_OPENING = 84.5 mm` 는 `probe_gripper_opening.py` 가 STL 에서 계산한 값인데,
그 프로브는 docstring 에 이렇게 적어 뒀다:

    "충돌 근사가 **convexHull** 이므로 핑거 안쪽 오목부는 메워지고,
     통과 가능 폭은 가장 안쪽 점(핑거 팁)이 지배한다."

그런데 지금 쓰는 자산(`_lgrip`)은 그 세 링크만 **convexDecomposition** 이다.
즉 **상수와 자산이 서로 다른 근사를 전제하고 있다.** 어느 쪽이 맞는지는 재야 안다.

기하 계산으로는 못 가른다 — 핑거 STL 의 볼록껍질은 안쪽 면의 홈을 최대 22.4 mm 메운다
(메시 부피 32.8 vs 껍질 71.9 cm³, 54% 채움). 그 홈이 컵이 지나는 구역에 있는지 아닌지가
개구를 정하는데, 그건 PhysX 가 실제로 만든 볼록 조각에 달렸다.

방법
----
env 마다 **다른 개구**로 턱을 벌려 놓고 가운데에 지름 `--diameter` 원기둥을 놓는다.
한 스텝 굴리고 접촉력을 읽는다. 접촉력이 0 이 되는 **최소 개구**가 곧
"이 지름이 통과하는 개구" 이므로, 유효 개구 = 그 지점의 턱 간격이다.

⚠ 접촉 판정은 `force_matrix_w` 로 한다. 다중 body 단일 센서는 조용히 0 을 준다 —
  이 저장소가 이미 당한 함정이라 body 마다 개별 센서를 만든다.

사용:
  TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH=<hdgp>/source/openarm \\
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_gripper_opening_sim.py \\
      --asset openarm_tesollo_sensor_rl_lgrip --diameter 0.058
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", default="openarm_tesollo_sensor_rl_lgrip",
                    help="assets/robot/ 아래 자산 디렉토리")
parser.add_argument("--diameter", type=float, default=0.058,
                    help="시험 원기둥 지름 (m). 기본 = shaker 파지대역 단면")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0e9
    for t in ("time_out", "object_dropping", "object_out_of_workspace"):
        setattr(cfg.terminations, t, None)
    cfg.curriculum.adr = None
    cfg.scene.robot.spawn.usd_path = os.path.join(
        ASSETS, "robot", args.asset, "openarm_tesollo_sensor_rl.usd")
    # 아래에서 핑거 body 마다 ContactSensor 를 붙인다 — 스폰에서 리포터 API 를
    # 켜 두지 않으면 센서 초기화가 "contact reporter API 없음" 으로 죽는다.
    cfg.scene.robot.spawn.activate_contact_sensors = True
    # 시험 원기둥 — 컵 대신 이걸 턱 사이에 놓는다.
    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CylinderCfg(
            radius=args.diameter / 2.0, height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.2, 0.3)),
    )
    for b in P.GRIPPER_FINGER_BODIES:
        setattr(cfg.scene, f"contact_{b}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{b}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            history_length=1, track_air_time=False))

    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev = env.device
    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = env.num_envs

    jaw_ids = [robot.joint_names.index(j) for j in P.GRIPPER_JOINT_NAMES]
    body_ids = [robot.body_names.index(b) for b in P.GRIPPER_FINGER_BODIES]

    # env 마다 다른 개구로 턱을 고정한다(관절 지령이 아니라 상태를 직접 쓴다 —
    # PD 가 못 이기는 구간이 생기면 지령과 실제가 갈린다).
    q = torch.linspace(0.0, P.GRIPPER_OPEN_POS, n, device=dev)
    jp = robot.data.default_joint_pos.clone()
    jp[:, jaw_ids[0]] = q
    jp[:, jaw_ids[1]] = q
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))

    for _ in range(2):
        env.step(torch.zeros(n, env.action_manager.total_action_dim, device=dev))

    # 원기둥을 두 턱의 정확한 중점에, 파지 대역 높이로 옮긴다.
    fp = robot.data.body_pos_w[:, body_ids, :]
    mid = fp.mean(dim=1)
    root = obj.data.default_root_state.clone()
    root[:, :3] = mid
    root[:, 2] = P.TABLE_SURFACE_Z + 0.5 * (P.GRASP_HEIGHT_BAND[0] + P.GRASP_HEIGHT_BAND[1])
    obj.write_root_pose_to_sim(root[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))

    a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    for _ in range(args.settle):
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))   # 턱을 계속 고정
        env.step(a)

    force = torch.zeros(n, device=dev)
    for b in P.GRIPPER_FINGER_BODIES:
        fm = env.scene.sensors[f"contact_{b}"].data.force_matrix_w
        force = force + fm.view(n, -1, 3).sum(dim=1).norm(dim=-1)
    gap = (robot.data.body_pos_w[:, body_ids[0], :]
           - robot.data.body_pos_w[:, body_ids[1], :]).norm(dim=-1)

    free = force < 1e-3
    print(f"\n자산 {args.asset}  ·  시험 지름 {args.diameter*1000:.1f} mm")
    print(f"{'q(m)':>8}{'body 간격(mm)':>16}{'접촉력(N)':>12}{'통과':>6}")
    for i in range(0, n, max(1, n // 16)):
        print(f"{q[i]:8.4f}{gap[i]*1000:16.2f}{force[i]:12.3f}{'  O' if free[i] else '  X':>6}")
    if int(free.sum()) == 0:
        print("\n★ 어느 개구에서도 통과하지 못했다 — 이 지름은 못 문다.")
    else:
        i = int(torch.nonzero(free)[0])
        print(f"\n★ 통과 최소 관절 개구 q = {q[i]:.4f} m  ·  그때 body 간격 {gap[i]*1000:.2f} mm")
        print(f"★ 유효 개구 ≈ 시험 지름 {args.diameter*1000:.1f} mm 가 "
              f"q={q[i]:.4f} 에서 통과 → 최대 q({P.GRIPPER_OPEN_POS}) 에서의 여유 = "
              f"{(P.GRIPPER_OPEN_POS - q[i])*2*1000:.1f} mm")
        print(f"★ 추정 최대 개구 = {args.diameter*1000 + (P.GRIPPER_OPEN_POS - q[i])*2*1000:.1f} mm "
              f"(현재 상수 {P.GRIPPER_MAX_OPENING*1000:.1f} mm)")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
