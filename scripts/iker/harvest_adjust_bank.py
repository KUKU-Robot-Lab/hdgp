"""Harvest the adjust-start bank (adjust_bank.py) from a trained stage-2 policy's own rollouts.

Every step, each env whose shoe qualifies (not placed, resting, still, grip open, not retracting, keypoint_dist within
--max-kp) is recorded once per episode: joint state and targets, both shoes' env-local poses, and the filtered grip
state. Training noise stays ON so the collected states are spread out like the training distribution.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/source/openarm \\
        ../IsaacLab/_isaac_sim/python.sh scripts/iker/harvest_adjust_bank.py \\
        --checkpoint <t2r2 .pth> --out <bank.pt> --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IKER stage-2 adjust-start bank harvest.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--target", type=int, default=2000, help="stop after this many states")
parser.add_argument("--max-steps", type=int, default=6000)
parser.add_argument("--max-kp", type=float, default=0.10, help="largest keypoint_dist kept [m]")
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("HARVEST FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (gym id registration)
from openarm.agnostic.tasks.iker_shoe import adjust_bank as ab  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env_cfg import IkerShoeT2rEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe_t2r"


class Harvester:
    """Wraps `_get_rewards` (after `_get_dones` filled `_t2r_last`, before the reset) and keeps qualifying states."""

    def __init__(self, u):
        self.u = u
        self.taken = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)  # already recorded this episode
        self.rows: dict[str, list[torch.Tensor]] = {key: [] for key in ab.KEYS}
        self.count = 0
        self._get_rewards = u._get_rewards
        u._get_rewards = self.get_rewards

    def get_rewards(self):
        out = self._get_rewards()
        u, last = self.u, self.u._t2r_last
        self.taken &= u.episode_length_buf > 1  # a fresh episode may be recorded again
        open_frac = ((u._grip_scalar.reshape(-1) + 1.0) * 0.5).clamp(0.0, 1.0)
        ok = ab.qualifies(last["placed"], last["resting"], last["still"], open_frac, last["retracting"],
                          last["keypoint_dist"], u.cfg.place, max_keypoint_dist=args.max_kp) & ~self.taken
        ids = ok.nonzero(as_tuple=True)[0]
        if len(ids):
            origins = u.scene.env_origins[ids]
            shoe = torch.cat([u._shoe.data.root_pos_w[ids] - origins, u._shoe.data.root_quat_w[ids]], dim=-1)
            other = torch.cat([u._other.data.root_pos_w[ids] - origins, u._other.data.root_quat_w[ids]], dim=-1)
            values = {"joint_pos": u._robot.data.joint_pos[ids], "joint_target": u._joint_targets[ids],
                      "shoe_pose": shoe, "other_pose": other, "grip_scalar": u._grip_scalar[ids]}
            for key in ab.KEYS:
                self.rows[key].append(values[key].detach().clone())
            self.taken[ids] = True
            self.count += len(ids)
        return out


def main() -> int:
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = IkerShoeT2rEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.add_noise = True
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    harvester = Harvester(u)
    wrapped, agent = load_player(env, TASK, Path(args.checkpoint))
    steps = 0
    with torch.inference_mode():
        obs = reset_player(wrapped, agent)
        while harvester.count < args.target and steps < args.max_steps:
            obs, _ = step_player(wrapped, agent, obs)
            steps += 1
    if harvester.count == 0:
        raise RuntimeError(f"no qualifying state in {steps} steps")
    entries = {key: torch.cat(harvester.rows[key])[: args.target] for key in ab.KEYS}
    kp_meta = {"checkpoint": str(Path(args.checkpoint).resolve()), "max_keypoint_dist": args.max_kp,
               "steps": steps, "num_envs": args.num_envs, "seed": args.seed}
    path = ab.save_bank(args.out, entries, list(u._robot.data.joint_names), kp_meta)
    print(f"HARVEST {len(entries['joint_pos'])} states in {steps} steps -> {path}", flush=True)
    return 0


os._exit(main())
