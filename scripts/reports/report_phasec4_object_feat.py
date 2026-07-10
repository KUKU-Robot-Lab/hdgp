#!/usr/bin/env python3
"""Phase C4 validation for object point-cloud feature observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase C object feature binding/observation shape.")
    parser.add_argument("--task", type=str, default="5g_grasp_right-v2")
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Optional output json path.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _to_float(v):
    if hasattr(v, "detach"):
        t = v.detach().float().mean()
        return float(t.cpu().item())
    return float(v)


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "source"))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import openarm.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, _agent_cfg):
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = args.num_envs

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        obs, _ = env.reset()
        core_env = env.unwrapped if hasattr(env, "unwrapped") else env

        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"]
        else:
            policy_obs = obs

        obs_dim = int(policy_obs.shape[-1])
        expected_obs = int(core_env.cfg.observation_space)
        obs_dim_match = bool(obs_dim == expected_obs)

        # Step a few frames to populate extras and stabilize buffers.
        extras_last = {}
        for _ in range(args.steps):
            zero_action = torch.zeros((core_env.num_envs, core_env.cfg.num_actions), device=core_env.device)
            env.step(zero_action)
            extras_last = getattr(core_env, "extras", {})

        feat0 = core_env.object_pc_feat.detach().clone()
        idx0 = core_env.active_object_feat_index.detach().clone()
        type0 = core_env.active_object_type.detach().clone()

        # Second reset to verify feature can change with object change.
        env.reset()
        feat1 = core_env.object_pc_feat.detach().clone()
        idx1 = core_env.active_object_feat_index.detach().clone()
        type1 = core_env.active_object_type.detach().clone()

        eps = 1e-6
        feat_changed = ((feat0 - feat1).abs().max(dim=-1)[0] > eps)
        type_changed = type0 != type1
        idx_changed = idx0 != idx1

        # Mapping integrity check: bound feature index follows object code expectation.
        cup_idx = int(core_env._object_pc_feat_code_to_index.get("cup", -1))
        default_idx = int(core_env._object_pc_feat_default_index)
        map_ok = True
        expected_idx_list = []
        for env_i in range(core_env.num_envs):
            obj_type = int(type1[env_i].item())
            if obj_type == 0:
                expected_idx = cup_idx if cup_idx >= 0 else default_idx
            else:
                primitive_name = core_env._primitive_name_per_env[env_i]
                code = f"primitive:{primitive_name}" if primitive_name else "primitive:default"
                expected_idx = int(core_env._object_pc_feat_code_to_index.get(code, default_idx))
            expected_idx_list.append(expected_idx)
            if int(idx1[env_i].item()) != expected_idx:
                map_ok = False

        # Env-level diversity: how many unique indices/features are present.
        unique_idx_count = int(torch.unique(idx1).numel())
        unique_feat_count = int(torch.unique((feat1 * 1e6).round().to(torch.int64), dim=0).shape[0])

        summary = {
            "task": args.task,
            "num_envs": int(core_env.num_envs),
            "steps": int(args.steps),
            "obs_dim": obs_dim,
            "expected_obs_dim": expected_obs,
            "obs_dim_match": obs_dim_match,
            "feature_enabled_cfg": bool(core_env.cfg.use_object_pc_feat),
            "feature_dim_cfg": int(core_env.cfg.object_pc_feat_dim),
            "feature_map_path": str(getattr(core_env, "_object_pc_feat_path", "")),
            "feature_map_num_codes": int(len(getattr(core_env, "_object_pc_feat_codes", []))),
            "active_type_counts_reset1": {
                "cup": int((type0 == 0).sum().item()),
                "primitive": int((type0 == 1).sum().item()),
            },
            "active_type_counts_reset2": {
                "cup": int((type1 == 0).sum().item()),
                "primitive": int((type1 == 1).sum().item()),
            },
            "reset_change_counts": {
                "type_changed": int(type_changed.sum().item()),
                "feat_index_changed": int(idx_changed.sum().item()),
                "feat_vector_changed": int(feat_changed.sum().item()),
            },
            "env_diversity_reset2": {
                "unique_feature_indices": unique_idx_count,
                "unique_feature_vectors": unique_feat_count,
            },
            "mapping_integrity_ok": map_ok,
            "metrics_last": {
                "object_pc_feature_norm": _to_float(extras_last.get("object_pc_feature_norm", 0.0)),
                "object_pc_feat_norm": _to_float(extras_last.get("object_pc_feat_norm", 0.0)),
                "object_pc_clip_ratio": _to_float(extras_last.get("object_pc_clip_ratio", 0.0)),
                "object_pc_invalid_ratio": _to_float(extras_last.get("object_pc_invalid_ratio", 0.0)),
            },
        }

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.output:
            out = Path(args.output).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] Saved report: {out}")

        env.close()

    try:
        _run()
    finally:
        if hasattr(simulation_app, "close"):
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
