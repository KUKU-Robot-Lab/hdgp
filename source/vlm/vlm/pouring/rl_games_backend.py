"""Standalone rl_games inference backend for referenced policy skills.

Loads one run's checkpoint with the exact logged `params/agent.yaml` — the
same Runner/create_player pattern as
`scripts/warm_states/collect_pour_fab_warm_states.py` — but without an
rl_games-wrapped simulator: `env_info` is injected from the run's policy
contract, so creating the player never boots an environment.

Conventions carried over from the rest of this package:
- Construction never touches CUDA or heavy imports; `load()` is the boundary
  (the QwenBackend convention). Tests inject a fake player instead.
- A recurrent policy keeps one hidden state per environment while the hard
  router only sends the subset of envs currently assigned to this skill.
  The backend therefore always runs the full batch — inactive rows receive
  zero observations — and gathers the requested rows. `reset(env_ids)`
  zeroes those envs' hidden state, which the skill manager calls on entry.
- Observation/action clipping mirrors the training-time wrapper using the
  logged `params.env.clip_observations` / `clip_actions`.
"""

from __future__ import annotations

# torch's stub re-export quirk trips reportPrivateImportUsage on zeros/tensor/etc.
# pyright: reportPrivateImportUsage=false
import math
from typing import Any

from .checkpoint_resolver import PolicyArtifacts, PolicyContract, read_policy_contract


class RlGamesPolicyBackend:
    """PolicyInferenceBackend over one resolved rl_games run."""

    def __init__(
        self,
        artifacts: PolicyArtifacts,
        *,
        num_envs: int,
        device: str = "cuda:0",
        deterministic: bool = True,
        contract: PolicyContract | None = None,
        player: Any | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.artifacts = artifacts
        self.num_envs = int(num_envs)
        self.device = device
        self.deterministic = bool(deterministic)
        self.contract = contract if contract is not None else read_policy_contract(artifacts.env_yaml)
        self._player = player
        self._clip_obs = math.inf
        self._clip_act = math.inf

    @property
    def loaded(self) -> bool:
        return self._player is not None

    def load(self) -> None:
        """Build and restore the rl_games player. First call is the GPU boundary."""
        if self.loaded:
            return
        import torch
        import yaml
        from rl_games.torch_runner import Runner

        try:
            from gym import spaces
        except ImportError:  # pragma: no cover - depends on installed rl_games extras
            from gymnasium import spaces

        agent_cfg = yaml.safe_load(self.artifacts.agent_yaml.read_text())
        if "params" not in agent_cfg:
            raise ValueError(f"{self.artifacts.agent_yaml} is not an rl_games agent config")
        params = agent_cfg["params"]
        env_params = params.get("env", {})
        self._clip_obs = float(env_params.get("clip_observations", math.inf))
        self._clip_act = float(env_params.get("clip_actions", math.inf))

        params["load_checkpoint"] = True
        params["load_path"] = str(self.artifacts.checkpoint)
        config = params["config"]
        config["num_actors"] = self.num_envs
        config["device"] = self.device
        config["device_name"] = self.device
        config["env_info"] = {
            "observation_space": spaces.Box(
                -math.inf, math.inf, (self.contract.observation_dim,)
            ),
            "action_space": spaces.Box(-1.0, 1.0, (self.contract.action_dim,)),
            "agents": 1,
        }

        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(self.artifacts.checkpoint))
        player.reset()
        dummy = torch.zeros(self.num_envs, self.contract.observation_dim, device=self.device)
        player.get_batch_size(dummy, 1)
        if player.is_rnn:
            player.init_rnn()
        self._player = player

    # ---- PolicyInferenceBackend ----------------------------------------------
    def infer(
        self,
        env_ids: tuple[int, ...],
        observations: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        import torch

        self._validate_batch(env_ids, observations)
        if not self.loaded:
            self.load()
        player = self._player
        assert player is not None

        full = torch.zeros(
            self.num_envs, self.contract.observation_dim,
            dtype=torch.float32, device=self.device,
        )
        rows = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if math.isfinite(self._clip_obs):
            rows = rows.clamp(-self._clip_obs, self._clip_obs)
        index = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        full[index] = rows

        with torch.inference_mode():
            actions = player.get_action(
                player.obs_to_torch(full), is_deterministic=self.deterministic
            )
        if math.isfinite(self._clip_act):
            actions = actions.clamp(-self._clip_act, self._clip_act)
        picked = actions[index].detach().cpu()
        return tuple(tuple(float(v) for v in row) for row in picked.tolist())

    def reset(self, env_ids: tuple[int, ...]) -> None:
        """Zero the recurrent state of the given envs (skill entry / episode reset)."""
        for env_id in env_ids:
            if not 0 <= env_id < self.num_envs:
                raise IndexError(f"env_id {env_id} out of range 0..{self.num_envs - 1}")
        player = self._player
        if player is None or not getattr(player, "is_rnn", False):
            return
        states = getattr(player, "states", None)
        if states is None:
            return
        index = list(env_ids)
        for state in states:
            state[:, index, :] = 0.0

    # --------------------------------------------------------------------------
    def _validate_batch(
        self,
        env_ids: tuple[int, ...],
        observations: tuple[tuple[float, ...], ...],
    ) -> None:
        if len(env_ids) != len(observations):
            raise ValueError(
                f"env_ids ({len(env_ids)}) and observations ({len(observations)}) must align"
            )
        if not env_ids:
            raise ValueError("infer requires at least one environment")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"env_ids must be unique: {env_ids}")
        for env_id in env_ids:
            if not 0 <= env_id < self.num_envs:
                raise IndexError(f"env_id {env_id} out of range 0..{self.num_envs - 1}")
        for row in observations:
            if len(row) != self.contract.observation_dim:
                raise ValueError(
                    f"observation must be {self.contract.observation_dim}D, got {len(row)}"
                )
