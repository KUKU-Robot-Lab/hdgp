"""Deterministic first-episode evaluation of a stage-2 IKER checkpoint (design 2026-09-14-iker-auto-loop §4, §7).

Every env's FIRST episode is recorded (all envs start at the same reset, so there is no short-episode bias). Failures
split into drop / timeout-far / timeout-near and the end error into centroid translation vs rotation (Kabsch on the 4
keypoints). ``--final-states`` also records every env's final joints, joint targets and both shoe poses: the scene
observe.py renders for the re-query. Moved from the scratch probe that measured the right-arm K1 runs (2026-09-14).

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/eval_iker.py --checkpoint <stage-2 .pth> --out <summary.json> --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic first-episode evaluation of a stage-2 IKER checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no-noise", action="store_true", help="turn the training observation and action noise off")
parser.add_argument("--interaction", default="", help="interaction file (default: the configuration's interaction_human.json)")
parser.add_argument("--grasp-bank", default="", help="grasp bank file (default: the configuration's grasp_bank.json)")
parser.add_argument("--out", required=True, help="summary JSON")
parser.add_argument("--rows-out", default="", help="per-env rows JSON")
parser.add_argument("--final-states", default="", help="per-env final robot and shoe states JSON")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("EVAL FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe"
HOLD_RADIUS_M = 0.15  # palm-to-shoe-root distance treated as "still in hand"
NEAR_M = 0.10
ROW_KEYS = ("env", "length", "end_dist", "min_dist", "first_5cm", "sustained", "dropped", "centroid_err", "rot_deg", "shoe_z", "palm_shoe")


def kabsch_angle_deg(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N,) rotation angle in degrees of the best rigid fit of ``current`` (N, K, 3) onto ``target`` (N, K, 3)."""
    a = current - current.mean(dim=1, keepdim=True)
    b = target - target.mean(dim=1, keepdim=True)
    u, _, vh = torch.linalg.svd(a.transpose(1, 2) @ b)
    d = torch.sign(torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    rotation = vh.transpose(1, 2) @ fix @ u.transpose(1, 2)
    cos = ((rotation.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def make_env():
    cfg = IkerShoeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.add_noise = not args.no_noise
    cfg.interaction_path = args.interaction
    cfg.grasp_bank_path = args.grasp_bank
    return gym.make(TASK, cfg=cfg)


class FirstEpisodeRecorder:
    """Wraps the env's ``_get_dones`` and ``_log_episode_end`` to record each env's first episode before its reset."""

    def __init__(self, u, keep_states: bool):
        n, dev = u.num_envs, u.device
        self.u, self.keep_states = u, keep_states
        self.running_min = torch.full((n,), math.inf, device=dev)
        self.first_hit = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.recorded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rows = {key: [] for key in ROW_KEYS}
        self.states = []
        self._get_dones, self._log_episode_end = u._get_dones, u._log_episode_end
        u._get_dones, u._log_episode_end = self.get_dones, self.log_episode_end

    def get_dones(self):
        out = self._get_dones()
        u, distance = self.u, self.u._keypoint_distance
        hit = (distance <= u.cfg.eval_success_distance_m) & (self.first_hit < 0)
        self.first_hit[hit] = u.episode_length_buf[hit]
        torch.minimum(self.running_min, distance, out=self.running_min)
        return out

    def log_episode_end(self, env_ids):
        u = self.u
        finished = env_ids[(u.episode_length_buf[env_ids] > 0) & ~self.recorded[env_ids]]
        if len(finished):
            self._record(finished)
            self.recorded[finished] = True
        self.running_min[env_ids] = math.inf
        self.first_hit[env_ids] = -1
        self._log_episode_end(env_ids)

    def _record(self, finished: torch.Tensor) -> None:
        u = self.u
        origins = u.scene.env_origins[finished]
        current, target = u._keypoints_local()[finished], u._targets.expand(len(finished), 4, 3)
        shoe, palm = u._shoe.data.root_pos_w[finished], u._robot.data.body_pos_w[finished, u._palm]
        sustain = u.cfg.reward.sustain_steps
        values = {
            "env": finished, "length": u.episode_length_buf[finished], "end_dist": u._keypoint_distance[finished],
            "min_dist": self.running_min[finished], "first_5cm": self.first_hit[finished],
            "sustained": u._success_count[finished] > sustain, "dropped": u._failure_count[finished] > sustain,
            "centroid_err": (current.mean(1) - target.mean(1)).norm(dim=-1), "rot_deg": kabsch_angle_deg(current, target),
            "shoe_z": shoe[:, 2] - origins[:, 2], "palm_shoe": (palm - shoe).norm(dim=-1),
        }
        for key, value in values.items():
            self.rows[key].extend(value.cpu().tolist() if value.dtype == torch.bool else value.float().cpu().tolist())
        if self.keep_states:
            self._record_states(finished, origins)

    def _record_states(self, finished: torch.Tensor, origins: torch.Tensor) -> None:
        u = self.u
        shoe = torch.cat([u._shoe.data.root_pos_w[finished] - origins, u._shoe.data.root_quat_w[finished]], dim=-1)
        other = torch.cat([u._other.data.root_pos_w[finished] - origins, u._other.data.root_quat_w[finished]], dim=-1)
        distance = u._keypoint_distance[finished]
        for i, env in enumerate(finished.tolist()):
            self.states.append({
                "env": env, "success": bool(distance[i] <= u.cfg.eval_success_distance_m), "end_dist": float(distance[i]),
                "joint_pos": u._robot.data.joint_pos[env].tolist(), "joint_target": u._joint_targets[env].tolist(),
                "shoe_move": shoe[i].tolist(), "shoe_other": other[i].tolist(),
            })


def summarize(rows: dict, steps: int, u) -> dict:
    table = {key: torch.tensor(values) for key, values in rows.items()}
    distance = u.cfg.eval_success_distance_m
    succeeded = table["end_dist"] <= distance
    timeout = ~table["sustained"] & ~table["dropped"]
    holding = table["palm_shoe"] < HOLD_RADIUS_M
    failed = ~succeeded
    kept = failed & ~table["dropped"]

    def frac(mask):
        return round(float(mask.float().mean()), 4) if mask.numel() else None

    def q(values):
        return [round(float(v), 4) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9]))] if values.numel() else None

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()), "num_envs": u.num_envs, "episodes": len(rows["env"]), "steps": steps,
        "noise": not args.no_noise, "success_5cm_end": frac(succeeded), "sustained": frac(table["sustained"]),
        "dropped": frac(table["dropped"]), "timeout": frac(timeout), "reached_5cm_ever": frac(table["min_dist"] <= distance),
        "failures": {
            "n": int(failed.sum()), "dropped": frac(table["dropped"][failed]), "timeout_holding": frac((timeout & holding)[failed]),
            "timeout_released": frac((timeout & ~holding)[failed]), "ever_reached_5cm": frac((table["min_dist"] <= distance)[failed]),
            "min_dist_q10_50_90": q(table["min_dist"][kept]), "end_dist_q10_50_90": q(table["end_dist"][kept]),
            "centroid_err_q10_50_90": q(table["centroid_err"][kept]), "rot_deg_q10_50_90": q(table["rot_deg"][kept]),
            "near_but_rotated": frac(((table["centroid_err"] < NEAR_M) & (table["rot_deg"] > 30))[kept]),
        },
        "successes": {
            "rot_deg_q10_50_90": q(table["rot_deg"][succeeded]), "length_q10_50_90": q(table["length"][succeeded]),
            "first_5cm_q10_50_90": q(table["first_5cm"][succeeded]),
        },
    }


def main() -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env = make_env()
    u = env.unwrapped
    recorder = FirstEpisodeRecorder(u, keep_states=bool(args.final_states))
    wrapped, agent = load_player(env, TASK, checkpoint)
    steps = 0
    with torch.inference_mode():
        obs = reset_player(wrapped, agent)
        while not bool(recorder.recorded.all()) and steps < 3 * u.max_episode_length:
            obs, _ = step_player(wrapped, agent, obs)
            steps += 1
    if not recorder.rows["env"]:
        raise RuntimeError(f"no episode finished in {steps} steps")
    summary = summarize(recorder.rows, steps, u)
    run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "summary": summary})
    if args.rows_out:
        run_files.write_json(Path(args.rows_out), {"schema": run_files.SCHEMA_VERSION, "rows": recorder.rows})
    if args.final_states:
        run_files.write_json(Path(args.final_states), {
            "schema": run_files.SCHEMA_VERSION, "checkpoint": summary["checkpoint"], "noise": summary["noise"],
            "joint_names": list(u._robot.data.joint_names), "rows": sorted(recorder.states, key=lambda row: row["env"]),
        })
    print("EVAL " + json.dumps(summary), flush=True)
    return 0


os._exit(main())
