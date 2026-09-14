"""Measure the grasp quality q of a phase-A stage-1 checkpoint and write q_lo/q_hi (design 2026-09-14 §5).

Runs the checkpoint deterministically (no noise, no disturbance), takes q at every lift-latch and success moment of
each env's first episode, and writes q_lo = p25 and q_hi = p90 of those moments. Refuses to write with fewer than
``--min-events`` moments: a checkpoint that does not lift yet cannot calibrate anything.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/measure_grasp_quality.py \
        --checkpoint <phase-A .pth> --headless
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure grasp quality q on a phase-A stage-1 checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--min-events", type=int, default=64)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("QUALITY FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env import QUALITY_CALIBRATION_FILE  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import IkerShoeGraspPlayEnvCfg  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp"
Q_LO_PERCENTILE, Q_HI_PERCENTILE = 0.25, 0.90


def main() -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = IkerShoeGraspPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.grasp_reward.g_min = 1.0  # measuring must not depend on a calibration
    cfg.wrench_prob_range = (1e-9, 1e-9)
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
    params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"].get("device", "cuda:0"), params.get("clip_observations", math.inf),
                                   params.get("clip_actions", math.inf), params.get("obs_groups"), params.get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = n
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    done_first = torch.zeros(n, dtype=torch.bool, device=dev)
    latch_q, success_q, w_at_latch = [], [], []
    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    with torch.inference_mode():
        for _ in range(u.max_episode_length + 5):
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            live = ~done_first
            obs, _, dones, _ = wrapped.step(actions)
            obs = obs["obs"] if isinstance(obs, dict) else obs
            step = u._last
            latched_now = step.just_latched & live
            if bool(latched_now.any()):
                latch_q.append(u._q_at_latch[latched_now].cpu())
                w_at_latch.append(u.grasp_quality_now()[1][latched_now].cpu())
            succeeded_now = step.success & live
            if bool(succeeded_now.any()):
                success_q.append(u._q_at_success[succeeded_now].cpu())
            done_first |= dones.bool()
            if bool(done_first.all()):
                break

    moments = torch.cat(latch_q + success_q) if (latch_q or success_q) else torch.zeros(0)
    events = int(moments.numel())
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "envs": n,
        "latch_events": int(sum(t.numel() for t in latch_q)),
        "success_events": int(sum(t.numel() for t in success_q)),
        "percentiles": {"lo": Q_LO_PERCENTILE, "hi": Q_HI_PERCENTILE},
    }
    if events < args.min_events:
        print(f"QUALITY config {args.config_index:02d} {result} passed False: {events} latch/success moments < {args.min_events}", flush=True)
        return 1
    result.update({
        "q_lo": float(torch.quantile(moments, Q_LO_PERCENTILE)),
        "q_hi": float(torch.quantile(moments, Q_HI_PERCENTILE)),
        "q_quantiles_10_25_50_75_90": [round(float(torch.quantile(moments, p)), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
        "w_f_median_at_latch": {f: round(float(torch.cat(w_at_latch)[:, i].median()), 4) for i, f in enumerate(gb.FINGERS)} if w_at_latch else {},
    })
    passed = result["q_lo"] < result["q_hi"]
    out = layout.RUNS_DIR / f"config_{args.config_index:02d}" / QUALITY_CALIBRATION_FILE
    if passed:
        run_files.write_json(out, result)
    print(f"QUALITY config {args.config_index:02d} {result} passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    os._exit(code)
