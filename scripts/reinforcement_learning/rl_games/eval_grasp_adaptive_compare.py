# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare adaptive grasp policies across 5g_grasp_right v7/v8/v9/v10.

The evaluator runs deterministic RL-Games policies for fixed bead-count bins
0/10/20/30 and writes common per-episode metrics, bin summaries, plots, and a
short markdown report snippet.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from types import MethodType
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Adaptive grasp policy comparison for v7/v8/v9/v10.")
parser.add_argument("--versions", nargs="+", default=["v7", "v8", "v9", "v10"], choices=["v7", "v8", "v9", "v10"])
parser.add_argument("--bead_counts", nargs="+", type=int, default=[0, 10, 20, 30])
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--episodes_per_bin", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="grasping_project/eval_policy_compare")
parser.add_argument("--check_only", action="store_true", help="Verify task registrations and checkpoint paths, then exit.")
parser.add_argument("--skip_plots", action="store_true", help="Skip PNG generation.")
parser.add_argument("--max_steps_per_bin", type=int, default=200000, help="Safety cap for rollout steps per version/bin.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401


def _force_local_openarm_path() -> str:
    hdgp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    openarm_src = os.path.join(hdgp_root, "source", "openarm")
    openarm_pkg = os.path.join(openarm_src, "openarm")
    if not os.path.isdir(openarm_pkg):
        raise RuntimeError(f"Local openarm package not found: {openarm_pkg}")
    if openarm_src in sys.path:
        sys.path.remove(openarm_src)
    sys.path.insert(0, openarm_src)
    return os.path.abspath(openarm_pkg)


_EXPECTED_OPENARM_DIR = _force_local_openarm_path()
import openarm  # noqa: E402

if not os.path.abspath(getattr(openarm, "__file__", "")).startswith(_EXPECTED_OPENARM_DIR + os.sep):
    raise RuntimeError(f"openarm resolved to unexpected location: {getattr(openarm, '__file__', None)}")
import openarm.tasks  # noqa: F401, E402


HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(HDGP_ROOT, "grasping_project", "eval_policy_compare")


@dataclass(frozen=True)
class PolicyPreset:
    version: str
    task: str
    checkpoint: str
    label: str
    action_schema: str
    grip_dof: int
    notes: str


PRESETS: dict[str, PolicyPreset] = {
    "v7": PolicyPreset(
        version="v7",
        task="5g_grasp_right-v7",
        checkpoint=os.path.join(
            HDGP_ROOT, "log", "rl_games", "pipeline", "right", "5g_grasp_right_v7", "test2", "nn",
            "5g_grasp_right-v7.pth",
        ),
        label="synergy_forced_mass_baseline",
        action_schema="6D palm + 5D synergy",
        grip_dof=5,
        notes="forced-mass baseline; policy has no bead/mass observation",
    ),
    "v8": PolicyPreset(
        version="v8",
        task="5g_grasp_right-v8",
        checkpoint=os.path.join(
            HDGP_ROOT, "log", "rl_games", "pipeline", "right", "5g_grasp_right_v8", "test2", "nn",
            "5g_grasp_right-v8.pth",
        ),
        label="synergy_mass_aware",
        action_schema="6D palm + 5D synergy",
        grip_dof=5,
        notes="mass-aware synergy policy",
    ),
    "v9": PolicyPreset(
        version="v9",
        task="5g_grasp_right-v9",
        checkpoint=os.path.join(
            HDGP_ROOT, "log", "rl_games", "pipeline", "right", "5g_grasp_right_v9", "test2", "nn",
            "5g_grasp_right-v9.pth",
        ),
        label="full_joint",
        action_schema="6D palm + 20D full-joint",
        grip_dof=20,
        notes="mass-aware full-joint policy with slip/adaptive force rewards",
    ),
    "v10": PolicyPreset(
        version="v10",
        task="5g_grasp_right-v10",
        checkpoint=os.path.join(
            HDGP_ROOT, "log", "rl_games", "pipeline", "right", "5g_grasp_right_v10", "test2", "nn",
            "5g_grasp_right-v10.pth",
        ),
        label="full_joint",
        action_schema="6D palm + 20D full-joint",
        grip_dof=20,
        notes="mass-aware full-joint policy with bin KPI rewards",
    ),
}

COMMON_FIELDS = [
    "version",
    "policy_label",
    "task",
    "checkpoint",
    "bead_count",
    "mass_kg",
    "success",
    "grip",
    "contact_count",
    "lift_height_m",
    "total_force_n",
    "force_ratio",
    "slip_proxy",
    "grasp_action_mean",
    "grasp_action_std",
    "grasp_action_min",
    "grasp_action_max",
    "thumb_action",
    "index_action",
    "middle_action",
    "ring_action",
    "pinky_action",
    "action_schema",
    "notes",
]

EXTRA_SNAPSHOT_KEYS = [
    "grip_adaptive_delta",
    "grip_norm_light",
    "grip_norm_heavy",
    "force_target_err",
    "force_target_value",
    "force_ratio_delta",
    "f_ratio_delta",
    "slip_reward",
    "adaptive_force_reward",
    "full_contact_bonus",
    "r_slip",
    "r_adaptive_grip",
    "r_full_contact",
]


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().flatten()[0].cpu().item())
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().flatten()[0].cpu().item())
    return bool(value)


def _mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    if len(values) < 2:
        return 0.0 if values else math.nan
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _mass_kg(env: Any, bead_count: int) -> float:
    base = float(getattr(env.cfg, "cup_base_mass", 0.170))
    bead_mass = float(getattr(env.cfg, "bead_single_mass", 0.010))
    return base + bead_count * bead_mass


def _install_eval_recorder(env: Any, preset: PolicyPreset, bead_count: int, records: list[dict]) -> None:
    original_reset_idx = env._reset_idx

    def wrapped_reset_idx(self, env_ids):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        ids = ids.to(device=self.device, dtype=torch.long)
        if ids.numel() > 0:
            _capture_reset_records(self, ids, preset, bead_count, records)
        if preset.version == "v7":
            return original_reset_idx(env_ids)
        return _with_forced_bead_randint(self, bead_count, lambda: original_reset_idx(env_ids))

    env._reset_idx = MethodType(wrapped_reset_idx, env)


def _with_forced_bead_randint(env: Any, bead_count: int, fn):
    """Force task reset bead-level sampling while leaving other randint calls intact."""
    module = sys.modules[env.__class__.__module__]
    original_randint = module.torch.randint
    forced_level = int(bead_count // 10)

    def forced_randint(low, high, size, *args, **kwargs):
        if low == 0 and high == 4 and len(size) == 1:
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", torch.long) or torch.long
            return torch.full(size, forced_level, device=device, dtype=dtype)
        return original_randint(low, high, size, *args, **kwargs)

    module.torch.randint = forced_randint
    try:
        return fn()
    finally:
        module.torch.randint = original_randint


def _capture_reset_records(env: Any, env_ids: torch.Tensor, preset: PolicyPreset, bead_count: int, records: list[dict]) -> None:
    if hasattr(env, "_eval_episode_started"):
        started = env._eval_episode_started[env_ids].bool()
    else:
        started = env.episode_length_buf[env_ids] > 0
    if not bool(started.any().item()):
        return

    mass_kg = _mass_kg(env, bead_count)
    actions = getattr(env, "actions", None)
    if actions is None:
        actions = torch.zeros((env.num_envs, 6 + preset.grip_dof), device=env.device)
    finger_actions = actions[:, 6 : 6 + preset.grip_dof]
    force = getattr(env, "contact_force_raw", torch.full((env.num_envs, 5), math.nan, device=env.device))
    total_force = force.sum(dim=-1)
    contacts = getattr(env, "num_contacts_buf", torch.full((env.num_envs,), math.nan, device=env.device))
    success = getattr(env, "episode_success_buf", getattr(env, "success_flag", torch.zeros(env.num_envs, device=env.device)))
    object_pos = getattr(env, "object_pos", torch.full((env.num_envs, 3), math.nan, device=env.device))
    object_init_pos = getattr(env, "object_init_pos", torch.full((env.num_envs, 3), math.nan, device=env.device))
    lift_height = object_pos[:, 2] - object_init_pos[:, 2]
    slip_proxy = _compute_slip_proxy(env)

    for env_id in env_ids[started]:
        idx = int(env_id.item())
        fa = finger_actions[idx]
        action_fingers = _finger_summary_actions(env, preset, idx, fa)
        record = {
            "version": preset.version,
            "policy_label": preset.label,
            "task": preset.task,
            "checkpoint": preset.checkpoint,
            "bead_count": bead_count,
            "mass_kg": mass_kg,
            "success": int(_to_bool(success[idx])),
            "grip": _grip_at_lift(env, preset, idx, fa),
            "contact_count": _to_float(contacts[idx]),
            "lift_height_m": _to_float(lift_height[idx]),
            "total_force_n": _to_float(total_force[idx]),
            "force_ratio": _to_float(total_force[idx]) / (mass_kg * 9.81 + 1e-6),
            "slip_proxy": _to_float(slip_proxy[idx]) if isinstance(slip_proxy, torch.Tensor) else math.nan,
            "grasp_action_mean": _to_float(fa.mean()),
            "grasp_action_std": _to_float(fa.std(unbiased=False)),
            "grasp_action_min": _to_float(fa.min()),
            "grasp_action_max": _to_float(fa.max()),
            "thumb_action": action_fingers[0],
            "index_action": action_fingers[1],
            "middle_action": action_fingers[2],
            "ring_action": action_fingers[3],
            "pinky_action": action_fingers[4],
            "action_schema": preset.action_schema,
            "notes": preset.notes,
        }
        record.update(_extras_snapshot(env))
        records.append(record)


def _compute_slip_proxy(env: Any) -> torch.Tensor | float:
    if hasattr(env, "contact_friction_xyz_raw"):
        return env.contact_friction_xyz_raw.norm(dim=-1).mean(dim=-1)
    if hasattr(env, "_prev_object_pos"):
        return (env.object_pos - env._prev_object_pos).norm(dim=-1)
    return math.nan


def _extras_snapshot(env: Any) -> dict[str, float]:
    extras = getattr(env, "extras", {})
    snapshot = {}
    for key in EXTRA_SNAPSHOT_KEYS:
        if key in extras:
            snapshot[key] = _to_float(extras[key])
    for key, value in extras.items():
        if isinstance(key, str) and key.startswith("bin_"):
            snapshot[key] = _to_float(value)
    return snapshot


def _grip_at_lift(env: Any, preset: PolicyPreset, idx: int, fallback: torch.Tensor) -> float:
    if hasattr(env, "_eval_grip_at_lift"):
        return _to_float(env._eval_grip_at_lift[idx])
    if preset.grip_dof == 5:
        return _to_float((fallback.mean() + 1.0) / 2.0)
    return _to_float(fallback.abs().mean())


def _finger_summary_actions(env: Any, preset: PolicyPreset, idx: int, fallback: torch.Tensor) -> list[float]:
    if hasattr(env, "_eval_finger_actions_at_lift") and hasattr(env, "_eval_lift_snapshot_valid"):
        if bool(env._eval_lift_snapshot_valid[idx].item()):
            fallback = env._eval_finger_actions_at_lift[idx]
    if preset.grip_dof == 5:
        selected = fallback[:5]
    else:
        selected = fallback[[1, 5, 9, 13, 17]]
    return [_to_float(x) for x in selected]


def _set_fixed_bead_or_mass(env_cfg: Any, preset: PolicyPreset, bead_count: int) -> None:
    if preset.version == "v7":
        return
    if hasattr(env_cfg, "bead_count_min"):
        env_cfg.bead_count_min = bead_count
    # The task normalizes observations by cfg.bead_count_max. Keep the training
    # denominator while reset-time sampling is forced by _with_forced_bead_randint.
    if hasattr(env_cfg, "bead_count_max"):
        env_cfg.bead_count_max = 30


def _apply_v7_forced_mass(env: Any, bead_count: int) -> str:
    mass = _mass_kg(env, bead_count)
    if bead_count == 0:
        return "v7 baseline mass left at default 0.170kg"
    if not hasattr(env, "cup") or not hasattr(env.cup, "root_physx_view"):
        raise RuntimeError("v7 forced-mass mode is unsupported: env.cup.root_physx_view is unavailable.")
    view = env.cup.root_physx_view
    if not hasattr(view, "get_masses") or not hasattr(view, "set_masses"):
        raise RuntimeError("v7 forced-mass mode is unsupported: PhysX mass get/set API is unavailable.")
    masses = view.get_masses().clone()
    masses[:] = mass
    view.set_masses(masses, torch.arange(masses.shape[0], device=masses.device, dtype=torch.int64))
    return f"v7 cup mass forced to {mass:.3f}kg"


def _make_player(env: Any, agent_cfg: dict, checkpoint: str):
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_acts = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    wrapped_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_acts)

    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n_actors, **kw: RlGamesGpuEnv(cfg_name, n_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped_env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = wrapped_env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    return wrapped_env, agent


def _rollout_bin(preset: PolicyPreset, bead_count: int) -> list[dict]:
    env_cfg = load_cfg_from_registry(preset.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(preset.task, "rl_games_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed + bead_count
    if "params" in agent_cfg:
        agent_cfg["params"]["seed"] = args_cli.seed + bead_count
    if hasattr(env_cfg, "enable_adr"):
        env_cfg.enable_adr = False
    _set_fixed_bead_or_mass(env_cfg, preset, bead_count)

    checkpoint = retrieve_file_path(preset.checkpoint)
    bin_records: list[dict] = []
    raw_env = gym.make(preset.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    base_env = raw_env.unwrapped
    _install_eval_recorder(base_env, preset, bead_count, bin_records)
    if preset.version == "v7":
        print(f"[INFO] {preset.version} bead={bead_count}: {_apply_v7_forced_mass(base_env, bead_count)}")

    env, agent = _make_player(raw_env, copy.deepcopy(agent_cfg), checkpoint)
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    if hasattr(base_env, "_total_episodes"):
        base_env._total_episodes = 0
    if hasattr(base_env, "_successful_episodes"):
        base_env._successful_episodes = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    target = args_cli.episodes_per_bin
    steps = 0
    last_report = 0
    while simulation_app.is_running() and len(bin_records) < target and steps < args_cli.max_steps_per_bin:
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            if agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0
        steps += 1
        progress = (len(bin_records) // 25) * 25
        if progress > last_report:
            print(f"[INFO] {preset.version} bead={bead_count}: {min(len(bin_records), target)}/{target} episodes")
            last_report = progress

    env.close()
    if len(bin_records) < target:
        raise RuntimeError(
            f"Collected only {len(bin_records)}/{target} episodes for {preset.version} bead={bead_count} "
            f"within {args_cli.max_steps_per_bin} steps."
        )
    return bin_records[:target]


def _check_inputs() -> None:
    missing: list[str] = []
    for version in args_cli.versions:
        preset = PRESETS[version]
        try:
            gym.spec(preset.task)
        except Exception as exc:
            missing.append(f"task {preset.task}: {exc}")
        if not os.path.isfile(preset.checkpoint):
            missing.append(f"checkpoint missing: {preset.checkpoint}")
    if missing:
        raise FileNotFoundError("\n".join(missing))
    print("[INFO] All requested task ids are registered and checkpoint paths exist.")


def _write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(COMMON_FIELDS)
    extra_keys = sorted({k for r in records for k in r.keys()} - set(keys))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra_keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"[INFO] wrote {path}")


def _summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["version"], int(record["bead_count"]))].append(record)

    summary = []
    metrics = ["success", "grip", "contact_count", "lift_height_m", "total_force_n", "force_ratio", "slip_proxy"]
    for (version, bead_count), rows in sorted(groups.items()):
        item = {
            "version": version,
            "policy_label": rows[0]["policy_label"],
            "bead_count": bead_count,
            "mass_kg": rows[0]["mass_kg"],
            "n": len(rows),
            "success_rate": _mean([float(r["success"]) for r in rows]),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rows]
            item[f"{metric}_mean"] = _mean(vals)
            item[f"{metric}_std"] = _std(vals)
        summary.append(item)
    return summary


def _write_summary(summary: list[dict], path: str) -> None:
    if not summary:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[INFO] wrote {path}")


def _plot(summary: list[dict], out_dir: str) -> None:
    if args_cli.skip_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    def series(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(8, 5))
        for version in args_cli.versions:
            rows = [r for r in summary if r["version"] == version]
            rows.sort(key=lambda r: r["bead_count"])
            if not rows:
                continue
            plt.plot([r["mass_kg"] for r in rows], [r[metric] for r in rows], marker="o", label=version)
        plt.xlabel("mass kg")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"[INFO] wrote {path}")

    series("success_rate", "success rate", "success_by_mass.png")
    series("total_force_n_mean", "total fingertip force N", "force_by_mass.png")
    series("force_ratio_mean", "F_total / (mass*g)", "force_ratio_by_mass.png")
    series("contact_count_mean", "contact count", "contact_by_mass.png")
    series("slip_proxy_mean", "slip proxy", "slip_proxy_by_mass.png")
    series("grip_mean", "grip action intensity", "grip_action_adaptation_by_mass.png")


def _write_report(summary: list[dict], out_dir: str) -> None:
    path = os.path.join(out_dir, "report_snippet.md")
    lines = [
        "# Adaptive Grasp Checkpoint Comparison",
        "",
        "Protocol: deterministic RL-Games rollout with fixed bead-count bins 0/10/20/30 and equal episode count per bin.",
        "",
        "Interpretation notes:",
        "- v7 is a `synergy_forced_mass_baseline`: the cup mass is forced to 0.17/0.27/0.37/0.47kg, but the policy was not trained with bead or mass observations.",
        "- v8 uses the same 5D synergy grip interface as v7, but is mass-aware through bead-conditioned training.",
        "- v9 and v10 use 20D full-joint grip control; compare their grip metric as a curl-joint/full-hand intensity summary rather than a direct action-space equivalent to v7/v8.",
        "- v10 should be judged by combined evidence: success across mass, sufficient force ratio, stable contact, lower slip proxy, and full-joint adaptive grip behavior.",
        "",
        "| version | label | bead | mass_kg | n | success_rate | grip | contacts | force_ratio | slip_proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['version']} | {row['policy_label']} | {row['bead_count']} | "
            f"{row['mass_kg']:.3f} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['grip_mean']:.3f} | {row['contact_count_mean']:.3f} | "
            f"{row['force_ratio_mean']:.3f} | {row['slip_proxy_mean']:.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] wrote {path}")


def _print_summary(summary: list[dict]) -> None:
    print("\nversion bead n success grip contacts force_ratio")
    for row in summary:
        print(
            f"{row['version']:>7} {row['bead_count']:>4} {row['n']:>4} "
            f"{row['success_rate']:.3f} {row['grip_mean']:.3f} "
            f"{row['contact_count_mean']:.3f} {row['force_ratio_mean']:.3f}"
        )


def main() -> None:
    if args_cli.output_dir == parser.get_default("output_dir"):
        args_cli.output_dir = DEFAULT_OUTPUT_DIR
    else:
        args_cli.output_dir = os.path.abspath(args_cli.output_dir)
    _check_inputs()
    if args_cli.check_only:
        return

    all_records: list[dict] = []
    for version in args_cli.versions:
        preset = PRESETS[version]
        for bead_count in args_cli.bead_counts:
            print(f"[INFO] evaluating {version} bead={bead_count} task={preset.task}")
            all_records.extend(_rollout_bin(preset, bead_count))

    os.makedirs(args_cli.output_dir, exist_ok=True)
    per_episode_csv = os.path.join(args_cli.output_dir, "per_episode.csv")
    summary_csv = os.path.join(args_cli.output_dir, "bin_summary.csv")
    _write_csv(all_records, per_episode_csv)
    summary = _summarize(all_records)
    _write_summary(summary, summary_csv)
    _plot(summary, args_cli.output_dir)
    _write_report(summary, args_cli.output_dir)
    _print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
