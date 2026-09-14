"""rl_games player on an IKER gym env for evaluation scripts (auto-loop §5, §7). Import after the Isaac app launcher.

The wiring is the one measure_grasp_quality.py uses: the task's registered PPO config, the Isaac Lab rl_games wrapper
and ``Runner.create_player``, with deterministic actions.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def load_player(env, task: str, checkpoint: Path):
    """(wrapped env, player) with ``checkpoint`` restored."""
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(
        env, agent_cfg["params"]["config"].get("device", "cuda:0"), params.get("clip_observations", math.inf),
        params.get("clip_actions", math.inf), params.get("obs_groups"), params.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(checkpoint))
    return wrapped, agent


def reset_player(wrapped, agent) -> torch.Tensor:
    """Reset every env and the player's state; returns the first observation batch."""
    agent.reset()
    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    return obs


def step_player(wrapped, agent, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic policy step; returns (next observation batch, dones)."""
    obs, _, dones, _ = wrapped.step(agent.get_action(agent.obs_to_torch(obs), is_deterministic=True))
    return (obs["obs"] if isinstance(obs, dict) else obs), dones
