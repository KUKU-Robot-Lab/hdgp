"""Unified task: where in the pick-and-place the episodes stop (2026-09-18).

The training log carries per-step means only (`grasp/held_frac`, `grasp/q`, ...), which cannot say whether a failing
episode never reached the shoe, closed on it without lifting, or lifted and lost it. This probe records the first
episode of every env (deterministic policy, no noise, same convention as eval_iker.py) and classifies its end state
with the environment's own quantities:

    never_reached     the palm never came within --touch-gap of the shoe surface
    no_lift           the palm reached it, but the shoe never rose past the reward's lift dead band
    lift_no_hold      the shoe rose past the dead band but `held` was never true (thumb, slip or the hold zone)
    lost              the grasp latched and the shoe then fell back below the dead band
    dropped           the shoe fell below the env's drop height
    timeout_held      `held` happened but the 20-step hold never completed before the time limit

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/source/openarm \\
        ../IsaacLab/_isaac_sim/python.sh scripts/iker/grasp_failure_probe.py --checkpoint <stage1 .pth> \\
        --num-envs 256 --out <probe.json> --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="1단계 파지 실패 원인 계측 프로브.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--touch-gap", type=float, default=0.02, help="이 거리 안에 들어오면 신발에 닿은 것으로 본다 [m]")
parser.add_argument("--reward-code-path", required=True)
parser.add_argument("--grasp-bank", default="")
parser.add_argument("--held-start-frac", type=float, default=0.0)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("PROBE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (gym id registration)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_unified_env_cfg import IkerShoeUnifiedEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe_unified"


class Recorder:
    """Wraps `_get_rewards` (runs right after `_get_dones` filled `_last`) and keeps each env's first episode."""

    def __init__(self, u):
        n, dev = u.num_envs, u.device
        self.u = u
        self.live = torch.ones(n, dtype=torch.bool, device=dev)
        self.steps = torch.zeros(n, device=dev)
        self.min_gap = torch.full((n,), float("inf"), device=dev)
        self.max_lift = torch.zeros(n, device=dev)
        self.max_curl = torch.zeros(n, device=dev)
        self.ever_held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_lost = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hold_max = torch.zeros(n, device=dev)
        self.q_max = torch.zeros(n, device=dev)
        self.max_contact_links = torch.zeros(n, device=dev)   # how many finger links touched the shoe at once
        self.max_close = torch.zeros(n, device=dev)           # how far the fingers closed from their open pose
        self.min_kp = torch.full((n,), float("inf"), device=dev)
        self.ever_placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.slip_at_hold: list[float] = []
        self._get_rewards, self._log_episode_end = u._get_rewards, u._log_episode_end
        u._get_rewards, u._log_episode_end = self.get_rewards, self.log_episode_end

    def get_rewards(self):
        out = self._get_rewards()
        u, step = self.u, self.u._last
        live = self.live
        origins = u.scene.env_origins
        palm_pos = u._robot.data.body_pos_w[:, u._palm] - origins
        shoe_pos = u._shoe.data.root_pos_w - origins
        import openarm.agnostic.tasks.iker_shoe.grasp_stage as gs

        gap = gs.nearest_distance(palm_pos[:, None, :], u._shoe_surface())[:, 0]
        curl = gs.closing_travel(u._robot.data.joint_pos[:, u._thumb_curl_id], u._thumb_open, u._thumb_grip)
        self.min_gap = torch.where(live, torch.minimum(self.min_gap, gap), self.min_gap)
        self.max_curl = torch.where(live, torch.maximum(self.max_curl, curl), self.max_curl)
        self.max_lift = torch.where(live, torch.maximum(self.max_lift, step.state.best_lift), self.max_lift)
        self.hold_max = torch.where(live, torch.maximum(self.hold_max, step.state.hold_count.float()), self.hold_max)
        self.q_max = torch.where(live, torch.maximum(self.q_max, u._q_step), self.q_max)
        self.ever_held |= step.held & live
        self.ever_latched |= step.state.latched & live
        self.ever_lost |= step.lost & live
        self.ever_dropped |= (shoe_pos[:, 2] < u.cfg.drop_z) & live
        self.succeeded |= step.success & live
        links, _ = u._link_shoe_forces()
        touching = (links > 0.1).float().sum(dim=(1, 2))
        self.max_contact_links = torch.where(live, torch.maximum(self.max_contact_links, touching), self.max_contact_links)
        closed = (u._hand_targets - u._hand_reset).abs().mean(dim=-1)
        self.max_close = torch.where(live, torch.maximum(self.max_close, closed), self.max_close)
        kp = u._unified_last["keypoint_dist"]
        self.min_kp = torch.where(live, torch.minimum(self.min_kp, kp), self.min_kp)
        self.ever_placed |= u._unified_last["placed"] & live
        self.steps += live.float()
        return out

    def log_episode_end(self, env_ids):
        ended = env_ids[self.u.episode_length_buf[env_ids] > 0]
        self.live[ended] = False
        self._log_episode_end(env_ids)


def classify(rec: Recorder, touch_gap: float, deadband: float) -> dict:
    n = rec.u.num_envs
    reached = rec.min_gap <= touch_gap
    lifted = rec.max_lift > 0.0                      # best_lift counts only past the dead band
    classes = {
        "success": rec.succeeded,
        "dropped": ~rec.succeeded & rec.ever_dropped,
        "lost": ~rec.succeeded & ~rec.ever_dropped & rec.ever_lost,
        "timeout_held": ~rec.succeeded & ~rec.ever_dropped & ~rec.ever_lost & rec.ever_held,
        "lift_no_hold": ~rec.succeeded & ~rec.ever_dropped & ~rec.ever_lost & ~rec.ever_held & lifted,
        "no_lift": ~rec.succeeded & ~rec.ever_dropped & ~rec.ever_lost & ~rec.ever_held & ~lifted & reached,
        "never_reached": ~rec.succeeded & ~rec.ever_dropped & ~rec.ever_lost & ~rec.ever_held & ~lifted & ~reached,
    }

    def q(values: torch.Tensor) -> list[float] | None:
        if values.numel() == 0:
            return None
        return [round(float(v), 4) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))]

    out = {"episodes": n, "touch_gap_m": touch_gap, "lift_deadband_m": deadband}
    for name, mask in classes.items():
        out[name] = {
            "frac": round(float(mask.float().mean()), 4),
            "min_palm_gap_q10_50_90": q(rec.min_gap[mask]),
            "best_lift_q10_50_90": q(rec.max_lift[mask]),
            "max_thumb_curl_q10_50_90": q(rec.max_curl[mask]),
            "hold_count_max_q10_50_90": q(rec.hold_max[mask]),
            "q_max_q10_50_90": q(rec.q_max[mask]),
            "max_contact_links_q10_50_90": q(rec.max_contact_links[mask]),
            "max_finger_close_q10_50_90": q(rec.max_close[mask]),
            "min_keypoint_dist_q10_50_90": q(rec.min_kp[mask]),
            "ever_placed_frac": round(float(rec.ever_placed[mask].float().mean()), 4) if bool(mask.any()) else None,
            "steps_mean": round(float(rec.steps[mask].mean()), 1) if bool(mask.any()) else None,
        }
    return out


def main() -> int:
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = IkerShoeUnifiedEnvCfg()
    cfg.reward_code_path = str(Path(args.reward_code_path).resolve())
    cfg.grasp_bank_start_path = str(Path(args.grasp_bank).resolve()) if args.grasp_bank else ""
    cfg.held_start_frac = args.held_start_frac
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.add_noise = False
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    rec = Recorder(u)
    wrapped, agent = load_player(env, TASK, Path(args.checkpoint))
    with torch.inference_mode():
        obs = reset_player(wrapped, agent)
        steps = 0
        while bool(rec.live.any()) and steps < 3 * u.max_episode_length:
            obs, _ = step_player(wrapped, agent, obs)
            steps += 1
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_envs": u.num_envs, "noise": False, "steps": steps,
        "breakdown": classify(rec, args.touch_gap, float(u.cfg.grasp_reward.lift_deadband_m)),
    }
    run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "summary": summary})
    print("PROBE " + json.dumps(summary), flush=True)
    return 0


os._exit(main())
