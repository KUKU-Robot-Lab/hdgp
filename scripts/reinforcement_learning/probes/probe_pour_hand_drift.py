"""pour_fabric: fabric 내부 손 자세 vs 실제 손 자세 괴리 실측 (진단 전용)."""
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch             # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import openarm.tasks     # noqa: E402,F401

cfg = parse_env_cfg("open-bis_b_pour_fab", device=args.device, num_envs=8)
cfg.require_warm_bank = False
# VRAM 경합 회피(진단 전용 축소 — 학습 cfg 를 바꾸는 것이 아니다)
px = cfg.sim.physx
px.gpu_temp_buffer_capacity = 8 * 1024 * 1024
px.gpu_heap_capacity = 8 * 1024 * 1024
px.gpu_found_lost_pairs_capacity = 2 ** 20
px.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
px.gpu_total_aggregate_pairs_capacity = 2 ** 20
px.gpu_max_rigid_contact_count = 2 ** 20
px.gpu_max_rigid_patch_count = 2 ** 18
px.gpu_collision_stack_size = 2 ** 26
cfg.bead_count = 4
env = gym.make("open-bis_b_pour_fab", cfg=cfg).unwrapped
env.reset()
a = torch.zeros(8, env.cfg.action_space, device=env.device)

rig = env.src
n_arm = rig.profile.num_arm_joints
for t in (0, 30, 120, 200, 400):
    while int(env.episode_length_buf[0]) < t:
        env.step(a)
    fab_hand = rig.fabric_q[:, n_arm:]                    # fabric 내부 손 (fabric 순서)
    real_hand = env.robot.data.joint_pos[:, rig.fab_t[n_arm:]]   # 같은 순서 실제
    cmd = rig.fabric_hand_cmd                             # 넘긴 목표 (fabric 순서)
    d = (fab_hand - real_hand).abs()          # 전체 괴리
    c = (fab_hand - cmd).abs()                # 1층: fabric 이 목표를 실현했나
    k = (real_hand - cmd).abs()               # 2층: PhysX PD 가 목표를 실현했나
    print(f"[drift] step {t:4d}  전체 mean {float(d.mean()):.4f} max {float(d.max()):.4f}"
          f" | 1층 fabric cmd_err mean {float(c.mean()):.4f} max {float(c.max()):.4f}"
          f" | 2층 PD track_err mean {float(k.mean()):.4f} max {float(k.max()):.4f} [rad]",
          flush=True)
print("[drift] fabric hand cmd shape:", tuple(rig.fabric_hand_cmd.shape),
      "(20 = hand_mode='direct' 관절 경로)")
env.close()
app.close()
