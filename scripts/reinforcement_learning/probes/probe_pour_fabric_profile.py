"""pour_fabric 스텝 구간별 시간 프로파일 (cuda sync 타이머). env 는 바꾸지 않는다.

    python probe_pour_fabric_profile.py --num_envs 512 --steps 200 --headless
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_b_pour_fab")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401
import openarm.agnostic.tasks.pour_fabric.config  # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
T: dict[str, float] = defaultdict(float)
on = {"v": False}


def wrap(obj: object, name: str, tag: str) -> None:
    fn = getattr(obj, name)

    def timed(*a, **k):
        if not on["v"]:
            return fn(*a, **k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[tag] += time.perf_counter() - t0
        return r

    setattr(obj, name, timed)


for n in ("_pre_physics_step", "_apply_action", "_get_dones", "_get_rewards", "_get_observations", "_reset_idx"):
    wrap(env, n, n)
wrap(env.sim, "step", "sim.step")
wrap(env.scene, "update", "scene.update")
wrap(env.scene, "write_data_to_sim", "scene.write")
for rig in env.rigs:
    for n in dir(rig):
        if n.startswith("_") or not callable(getattr(rig, n)):
            continue
        wrap(rig, n, f"rig.{n}")

N, A = env.num_envs, env.cfg.action_space
for t in range(args.steps):
    if t == args.warmup:
        on["v"] = True
        torch.cuda.synchronize()
        t_all = time.perf_counter()
    env.step(torch.rand(N, A, device=env.device) * 2 - 1)
torch.cuda.synchronize()
total = time.perf_counter() - t_all
n = args.steps - args.warmup
print(f"[profile] envs={N} step={1000 * total / n:.1f} ms fps={N * n / total:.0f}")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"[profile] {k:28s} {1000 * v / n:8.2f} ms/step {100 * v / total:5.1f}%")
env.close()
app.close()
