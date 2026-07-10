#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Patch IsaacLab's RslRlVecEnvWrapper to support swap_lr."""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys


DEFAULT_RELATIVE_VECENV = os.path.join(
    "source", "isaaclab_rl", "isaaclab_rl", "rsl_rl", "vecenv_wrapper.py"
)
DEFAULT_RELATIVE_REWARD = os.path.join(
    "source", "isaaclab", "isaaclab", "managers", "reward_manager.py"
)


def _find_isaaclab_dir(cli_dir: str | None) -> str | None:
    if cli_dir:
        return cli_dir
    env_dir = os.environ.get("ISAACLAB_DIR")
    if env_dir:
        return env_dir
    # Try relative to this script: hdgp/scripts/tools -> ../../IsaacLab
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "IsaacLab"))
    if os.path.isfile(os.path.join(candidate, DEFAULT_RELATIVE_VECENV)):
        return candidate
    # Try CWD/IsaacLab
    candidate = os.path.abspath(os.path.join(os.getcwd(), "IsaacLab"))
    if os.path.isfile(os.path.join(candidate, DEFAULT_RELATIVE_VECENV)):
        return candidate
    return None


def _insert_after(text: str, marker: str, insert: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"marker not found: {marker!r}")
    idx += len(marker)
    return text[:idx] + insert + text[idx:]


def _replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError("expected snippet not found")
    return text.replace(old, new, 1)


def _has_helpers(content: str) -> bool:
    required = [
        "def _build_swap_helpers",
        "def _sample_swap_mask",
        "def _swap_actions_inplace",
        "def _swap_obs_inplace",
        "def _swap_reward_terms_inplace",
    ]
    return all(r in content for r in required)


def _patch_vecenv_wrapper(content: str, force: bool = False) -> str:
    if not force and "swap_lr" in content and _has_helpers(content):
        return content

    content = _replace(
        content,
        "    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):\n",
        "    def __init__(\n"
        "        self,\n"
        "        env: ManagerBasedRLEnv | DirectRLEnv,\n"
        "        clip_actions: float | None = None,\n"
        "        swap_lr: bool = False,\n"
        "        swap_prob: float = 0.5,\n"
        "        swap_obs_term_pairs: list[tuple[str, str]] | None = None,\n"
        "        swap_action_term_pairs: list[tuple[str, str]] | None = None,\n"
        "    ):\n",
    )

    content = _insert_after(
        content,
        "        self.env = env\n        self.clip_actions = clip_actions\n",
        "        self._swap_lr = swap_lr\n"
        "        self._swap_prob = float(swap_prob)\n",
    )

    content = _replace(
        content,
        "        # reset at the start since the RSL-RL runner does not call reset\n"
        "        self.env.reset()\n",
        "        # prepare swap helpers (if enabled)\n"
        "        self._swap_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)\n"
        "        self._obs_term_slices: dict[str, dict[str, slice]] = {}\n"
        "        self._obs_group_concat: dict[str, bool] = {}\n"
        "        self._action_term_slices: dict[str, slice] = {}\n"
        "        self._reward_term_pairs: list[tuple[str, str]] = []\n"
        "        self._reward_term_indices: list[tuple[int, int]] = []\n"
        "\n"
        "        self._swap_obs_term_pairs = swap_obs_term_pairs or [\n"
        "            (\"target_object_position\", \"target_object2_position\"),\n"
        "            (\"object_position\", \"object2_position\"),\n"
        "            (\"object_obs\", \"object2_obs\"),\n"
        "        ]\n"
        "        self._swap_action_term_pairs = swap_action_term_pairs or [\n"
        "            (\"left_arm_action\", \"right_arm_action\"),\n"
        "            (\"left_hand_action\", \"right_hand_action\"),\n"
        "        ]\n"
        "\n"
        "        if self._swap_lr:\n"
        "            self._build_swap_helpers()\n"
        "            self._sample_swap_mask()\n"
        "\n"
        "        # reset at the start since the RSL-RL runner does not call reset\n"
        "        self.env.reset()\n",
    )

    content = _replace(
        content,
        "        obs_dict, extras = self.env.reset()\n"
        "        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras\n",
        "        obs_dict, extras = self.env.reset()\n"
        "        if self._swap_lr:\n"
        "            self._sample_swap_mask()\n"
        "            self._swap_obs_inplace(obs_dict, self._swap_mask)\n"
        "        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras\n",
    )

    content = _replace(
        content,
        "        # clip actions\n"
        "        if self.clip_actions is not None:\n"
        "            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)\n"
        "        # record step information\n"
        "        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)\n"
        "        # compute dones for compatibility with RSL-RL\n"
        "        dones = (terminated | truncated).to(dtype=torch.long)\n"
        "        # move time out information to the extras dict\n"
        "        # this is only needed for infinite horizon tasks\n"
        "        if not self.unwrapped.cfg.is_finite_horizon:\n"
        "            extras[\"time_outs\"] = truncated\n"
        "        # return the step information\n"
        "        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras\n",
        "        # clip actions\n"
        "        if self.clip_actions is not None:\n"
        "            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)\n"
        "        # swap actions for mirrored environments\n"
        "        if self._swap_lr:\n"
        "            actions = self._swap_actions_inplace(actions, self._swap_mask)\n"
        "        # record step information\n"
        "        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)\n"
        "        # compute dones for compatibility with RSL-RL\n"
        "        dones = (terminated | truncated).to(dtype=torch.long)\n"
        "        if self._swap_lr:\n"
        "            # swap reward terms/logs using current mask (pre-reset)\n"
        "            self._swap_reward_terms_inplace(self._swap_mask)\n"
        "            # resample swap mask for envs that just ended\n"
        "            if torch.any(dones.bool()):\n"
        "                self._sample_swap_mask(dones.bool())\n"
        "            # swap observations for current mask (post-reset for done envs)\n"
        "            self._swap_obs_inplace(obs_dict, self._swap_mask)\n"
        "        # move time out information to the extras dict\n"
        "        # this is only needed for infinite horizon tasks\n"
        "        if not self.unwrapped.cfg.is_finite_horizon:\n"
        "            extras[\"time_outs\"] = truncated\n"
        "        # return the step information\n"
        "        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras\n",
    )

    helper_block = (
        "\n"
        "    def _build_swap_helpers(self):\n"
        "        \"\"\"Pre-compute term slices for swapping.\"\"\"\n"
        "        if not hasattr(self.unwrapped, \"observation_manager\"):\n"
        "            return\n"
        "\n"
        "        obs_mgr = self.unwrapped.observation_manager\n"
        "        for group_name, term_names in obs_mgr.active_terms.items():\n"
        "            self._obs_group_concat[group_name] = obs_mgr.group_obs_concatenate[group_name]\n"
        "            if not obs_mgr.group_obs_concatenate[group_name]:\n"
        "                continue\n"
        "            term_dims = obs_mgr.group_obs_term_dim[group_name]\n"
        "            term_slices: dict[str, slice] = {}\n"
        "            idx = 0\n"
        "            for name, dims in zip(term_names, term_dims):\n"
        "                length = int(torch.prod(torch.tensor(dims)).item())\n"
        "                term_slices[name] = slice(idx, idx + length)\n"
        "                idx += length\n"
        "            self._obs_term_slices[group_name] = term_slices\n"
        "\n"
        "        if hasattr(self.unwrapped, \"action_manager\"):\n"
        "            names = self.unwrapped.action_manager.active_terms\n"
        "            dims = self.unwrapped.action_manager.action_term_dim\n"
        "            idx = 0\n"
        "            for name, dim in zip(names, dims):\n"
        "                self._action_term_slices[name] = slice(idx, idx + int(dim))\n"
        "                idx += int(dim)\n"
        "\n"
        "        if hasattr(self.unwrapped, \"reward_manager\"):\n"
        "            reward_terms = set(self.unwrapped.reward_manager.active_terms)\n"
        "            for name in reward_terms:\n"
        "                if \"left_\" in name:\n"
        "                    counterpart = name.replace(\"left_\", \"right_\", 1)\n"
        "                    if counterpart in reward_terms:\n"
        "                        self._reward_term_pairs.append((name, counterpart))\n"
        "            if self._reward_term_pairs:\n"
        "                name_to_idx = {n: i for i, n in enumerate(self.unwrapped.reward_manager.active_terms)}\n"
        "                for left, right in self._reward_term_pairs:\n"
        "                    self._reward_term_indices.append((name_to_idx[left], name_to_idx[right]))\n"
        "\n"
        "    def _sample_swap_mask(self, env_ids: torch.Tensor | None = None):\n"
        "        \"\"\"Sample swap mask per environment (per-episode).\"\"\"\n"
        "        if env_ids is None:\n"
        "            env_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)\n"
        "        count = int(env_ids.sum().item())\n"
        "        if count == 0:\n"
        "            return\n"
        "        rand = torch.rand((count,), device=self.device)\n"
        "        self._swap_mask[env_ids] = rand < self._swap_prob\n"
        "\n"
        "    def _swap_actions_inplace(self, actions: torch.Tensor, swap_mask: torch.Tensor) -> torch.Tensor:\n"
        "        if not torch.any(swap_mask):\n"
        "            return actions\n"
        "        if not self._action_term_slices:\n"
        "            return actions\n"
        "        for left, right in self._swap_action_term_pairs:\n"
        "            if left not in self._action_term_slices or right not in self._action_term_slices:\n"
        "                continue\n"
        "            left_slice = self._action_term_slices[left]\n"
        "            right_slice = self._action_term_slices[right]\n"
        "            tmp = actions[swap_mask, left_slice].clone()\n"
        "            actions[swap_mask, left_slice] = actions[swap_mask, right_slice]\n"
        "            actions[swap_mask, right_slice] = tmp\n"
        "        return actions\n"
        "\n"
        "    def _swap_obs_inplace(self, obs_dict: dict, swap_mask: torch.Tensor):\n"
        "        if not torch.any(swap_mask):\n"
        "            return\n"
        "        for group_name, group_obs in obs_dict.items():\n"
        "            if group_name not in self._obs_group_concat:\n"
        "                continue\n"
        "            if self._obs_group_concat[group_name]:\n"
        "                if group_name not in self._obs_term_slices:\n"
        "                    continue\n"
        "                term_slices = self._obs_term_slices[group_name]\n"
        "                for left, right in self._swap_obs_term_pairs:\n"
        "                    if left not in term_slices or right not in term_slices:\n"
        "                        continue\n"
        "                    left_slice = term_slices[left]\n"
        "                    right_slice = term_slices[right]\n"
        "                    tmp = group_obs[swap_mask, left_slice].clone()\n"
        "                    group_obs[swap_mask, left_slice] = group_obs[swap_mask, right_slice]\n"
        "                    group_obs[swap_mask, right_slice] = tmp\n"
        "            else:\n"
        "                if not isinstance(group_obs, dict):\n"
        "                    continue\n"
        "                for left, right in self._swap_obs_term_pairs:\n"
        "                    if left not in group_obs or right not in group_obs:\n"
        "                        continue\n"
        "                    tmp = group_obs[left][swap_mask].clone()\n"
        "                    group_obs[left][swap_mask] = group_obs[right][swap_mask]\n"
        "                    group_obs[right][swap_mask] = tmp\n"
        "\n"
        "    def _swap_reward_terms_inplace(self, swap_mask: torch.Tensor):\n"
        "        if not torch.any(swap_mask):\n"
        "            return\n"
        "        if not hasattr(self.unwrapped, \"reward_manager\"):\n"
        "            return\n"
        "        reward_manager = self.unwrapped.reward_manager\n"
        "        if self._reward_term_indices and hasattr(reward_manager, \"_step_reward\"):\n"
        "            for left_idx, right_idx in self._reward_term_indices:\n"
        "                tmp = reward_manager._step_reward[swap_mask, left_idx].clone()\n"
        "                reward_manager._step_reward[swap_mask, left_idx] = reward_manager._step_reward[\n"
        "                    swap_mask, right_idx\n"
        "                ]\n"
        "                reward_manager._step_reward[swap_mask, right_idx] = tmp\n"
        "        if self._reward_term_pairs and hasattr(reward_manager, \"_episode_sums\"):\n"
        "            for left_name, right_name in self._reward_term_pairs:\n"
        "                if left_name not in reward_manager._episode_sums or right_name not in reward_manager._episode_sums:\n"
        "                    continue\n"
        "                tmp = reward_manager._episode_sums[left_name][swap_mask].clone()\n"
        "                reward_manager._episode_sums[left_name][swap_mask] = reward_manager._episode_sums[right_name][\n"
        "                    swap_mask\n"
        "                ]\n"
        "                reward_manager._episode_sums[right_name][swap_mask] = tmp\n"
    )

    if "_build_swap_helpers" not in content:
        insert_marker = "\n    def _modify_action_space"
        if insert_marker in content:
            content = _insert_after(content, insert_marker, helper_block)
        else:
            content = content.rstrip() + helper_block + "\n"

    return content


def _has_rewardmanager_patch(content: str) -> bool:
    return "log-only term: record raw values without affecting total reward" in content


def _patch_reward_manager(content: str) -> str:
    if _has_rewardmanager_patch(content):
        return content

    old = (
        "            if term_cfg.weight == 0.0:\n"
        "                continue\n"
        "            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt\n"
    )
    new = (
        "            if term_cfg.weight == 0.0:\n"
        "                # log-only term: record raw values without affecting total reward\n"
        "                raw_value = term_cfg.func(self._env, **term_cfg.params)\n"
        "                self._episode_sums[name] += raw_value * dt\n"
        "                self._step_reward[:, term_idx] = raw_value\n"
        "                continue\n"
        "            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt\n"
    )
    return _replace(content, old, new)


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch IsaacLab vecenv_wrapper.py with swap_lr support.")
    parser.add_argument("--isaaclab-dir", type=str, default=None, help="Path to IsaacLab root directory.")
    parser.add_argument("--target", type=str, default=None, help="Explicit path to vecenv_wrapper.py.")
    parser.add_argument(
        "--patch-reward", action="store_true", help="Also patch RewardManager for weight==0 logging."
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")
    parser.add_argument(
        "--force", action="store_true", help="Force patch even if swap_lr is already present."
    )
    args = parser.parse_args()

    if args.target:
        target_path = os.path.abspath(args.target)
    else:
        isaaclab_dir = _find_isaaclab_dir(args.isaaclab_dir)
        if not isaaclab_dir:
            print("[ERROR] IsaacLab directory not found. Set --isaaclab-dir or ISAACLAB_DIR.", file=sys.stderr)
            return 1
        target_path = os.path.join(isaaclab_dir, DEFAULT_RELATIVE_VECENV)

    if not os.path.isfile(target_path):
        print(f"[ERROR] Target file not found: {target_path}", file=sys.stderr)
        return 1

    with open(target_path, "r", encoding="utf-8") as f:
        original = f.read()

    patched = _patch_vecenv_wrapper(original, force=args.force)
    if patched == original:
        if _has_helpers(original):
            print("[INFO] swap_lr already present; no changes made.")
        else:
            print("[WARN] swap_lr present but helpers missing; re-run with --force.")
        return 0

    if args.dry_run:
        print("[INFO] Patch would be applied. (dry-run)")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{target_path}.bak_{stamp}"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(original)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"[INFO] Patched: {target_path}")
    print(f"[INFO] Backup: {backup_path}")

    if args.patch_reward:
        reward_path = os.path.join(isaaclab_dir, DEFAULT_RELATIVE_REWARD)
        if not os.path.isfile(reward_path):
            print(f"[WARN] RewardManager not found: {reward_path}")
            return 0
        with open(reward_path, "r", encoding="utf-8") as f:
            reward_original = f.read()
        reward_patched = _patch_reward_manager(reward_original)
        if reward_patched == reward_original:
            if _has_rewardmanager_patch(reward_original):
                print("[INFO] RewardManager already patched; no changes made.")
            else:
                print("[WARN] RewardManager pattern not found; no changes made.")
            return 0
        if args.dry_run:
            print("[INFO] RewardManager patch would be applied. (dry-run)")
            return 0
        reward_backup = f"{reward_path}.bak_{stamp}"
        with open(reward_backup, "w", encoding="utf-8") as f:
            f.write(reward_original)
        with open(reward_path, "w", encoding="utf-8") as f:
            f.write(reward_patched)
        print(f"[INFO] Patched: {reward_path}")
        print(f"[INFO] Backup: {reward_backup}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
