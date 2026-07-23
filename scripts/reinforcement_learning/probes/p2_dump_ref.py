"""P2 reference dump — pour_v1 isaaclab env 의 actor obs 55D + 재구성용 raw 상태를 덤프.

목적: raw-app obs 재구성(p2_obs_raw.py)의 ground truth 생성. warmstart reset 상태에서
noise off 로 결정적 actor obs 를 뽑고, 이를 재현하는 데 필요한 관절/컵/palm 상태 + geometry
_w + cfg 상수를 npz 로 저장.

실행:
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p2_dump_ref.py \
      --task open-tesol_r_pour_v1-lstm --num_envs 16 --headless \
      --out hdgp/docs/eval/p2_ref.npz
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="P2 reference obs/state dump")
parser.add_argument("--task", type=str, default="open-tesol_r_pour_v1-lstm")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--tilt_deg", type=float, default=0.0,
                    help=">0 이면 소스 컵을 world x축 기준 tilt_deg 회전 → source_pour_point 동적 blend(deep tilt) 분기 검증.")
parser.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "eval", "p2_ref.npz"))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
# Hydra 가 남은 인자(하이드라 오버라이드)만 보도록 알려진 CLI 인자를 sys.argv 에서 제거.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import openarm.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    # 결정적 obs: 모든 noise/ADR off (base σ=0 강제).
    for attr in ("obs_noise_joint_pos", "obs_noise_joint_vel", "obs_noise_body_pos", "obs_noise_cup_pos"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, 0.0)
    for attr in ("enable_adr", "enable_noise_adr", "enable_bead_count_adr",
                 "enable_success_adr", "enable_spill_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped

    env.reset()
    if args_cli.tilt_deg > 0.0:
        # 소스 컵을 world x축 기준 회전시켜 deep-tilt geometry(동적 blend) 분기 exercise.
        th = math.radians(args_cli.tilt_deg)
        tilt_q = torch.tensor([math.cos(th / 2), math.sin(th / 2), 0.0, 0.0],
                              device=e.device)  # wxyz, x축
        cur = e.cup.data.root_quat_w.clone()
        w1, x1, y1, z1 = tilt_q
        w2, x2, y2, z2 = cur[:, 0], cur[:, 1], cur[:, 2], cur[:, 3]
        new_q = torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)
        pose = torch.cat([e.cup.data.root_pos_w, new_q], dim=-1)
        e.cup.write_root_pose_to_sim(pose)
        for _ in range(2):
            e.sim.step(render=False)
            e.scene.update(dt=e.physics_dt)
        print(f"[P2-dump] 소스 컵 {args_cli.tilt_deg}° tilt 적용 (deep-tilt geometry 검증)", flush=True)
    e._compute_intermediate_values()
    obs = e._get_observations()
    actor = obs["policy"].detach().cpu().numpy()   # (N, 55)

    def cpu(x):
        return x.detach().cpu().numpy()

    data = dict(
        actor_obs=actor,
        # ---- 재구성용 raw 상태 ----
        arm_joint_pos=cpu(e.robot.data.joint_pos[:, e.arm_dof_indices]),
        arm_joint_vel=cpu(e.robot.data.joint_vel[:, e.arm_dof_indices]),
        finger_joint_pos=cpu(e.robot.data.joint_pos[:, e.hand_dof_indices]),
        finger_joint_vel=cpu(e.robot.data.joint_vel[:, e.hand_dof_indices]),
        left_arm_joint_pos=cpu(e.robot.data.joint_pos[:, e.left_arm_dof_indices]),
        left_arm_joint_vel=cpu(e.robot.data.joint_vel[:, e.left_arm_dof_indices]),
        cup_pos_w=cpu(e.cup.data.root_pos_w),
        cup_quat_w=cpu(e.cup.data.root_quat_w),
        left_cup_pos_w=cpu(e.left_target_cup.data.root_pos_w),
        left_cup_quat_w=cpu(e.left_target_cup.data.root_quat_w),
        palm_pos_w=cpu(e.robot.data.body_pos_w[:, e.palm_body_index]),
        palm_quat_w=cpu(e.robot.data.body_quat_w[:, e.palm_body_index]),
        palm_center_pos=cpu(e.palm_center_pos),
        env_origins=cpu(e.scene.env_origins),
        # ---- obs-support 상수 ----
        hand_open_pose=cpu(e.hand_open_pose),
        hand_grasp_pose=cpu(e.hand_grasp_pose),
        # ---- geometry _w (교차검증) ----
        source_pour_point_w=cpu(e._source_pour_point_w),
        target_opening_w=cpu(e._target_opening_w),
        source_pour_axis_w=cpu(e._source_pour_axis_w),
        source_up_axis_w=cpu(e._source_up_axis_w),
        target_up_axis_w=cpu(e._target_up_axis_w),
        # ---- geometry cfg 상수(라이브 재현용) ----
        source_cup_pour_point_pos_b=cpu(e._source_cup_pour_point_pos_b),
        source_cup_pour_axis_b=cpu(e._source_cup_pour_axis_b),
        source_cup_up_axis_b=cpu(e._source_cup_up_axis_b),
        target_cup_opening_pos_b=cpu(e._target_cup_opening_pos_b),
        target_cup_up_axis_b=cpu(e._target_cup_up_axis_b),
        palm_ee_offset_local=cpu(e._palm_ee_offset_local),
        source_outer_radius=np.float64(e.cfg.source_outer_radius),
        pour_point_dyn_lo=np.float64(e.cfg.pour_point_dyn_lo),
        pour_point_dyn_hi=np.float64(e.cfg.pour_point_dyn_hi),
    )

    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **data)
    print(f"[P2-dump] saved {actor.shape} actor obs + state → {out}", flush=True)
    print(f"[P2-dump] actor obs[0][:12] = {np.array2string(actor[0][:12], precision=4)}", flush=True)
    print(f"[P2-dump] pour_point_to_opening[0] = "
          f"{np.array2string((data['target_opening_w'][0]-data['source_pour_point_w'][0]), precision=4)}",
          flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
