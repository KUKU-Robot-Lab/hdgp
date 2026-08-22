from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_TASK_LOGS: Mapping[str, tuple[str, str]] = {
    "open-tesol_r_grasp_v1-lstm": ("open-tesol/right", "grasp-v1"),
    "open-tesol_b_pour_v1-lstm": ("open-tesol/both", "pour-v1"),
}


@dataclass(frozen=True)
class PolicyContract:
    """The observation/action dimensions a run was actually trained with."""

    observation_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError(
                f"policy contract dims must be positive, got "
                f"obs={self.observation_dim} act={self.action_dim}"
            )


_CONTRACT_LINE = re.compile(r"^(observation_space|action_space):\s*(\d+)\s*$")


def read_policy_contract(env_yaml: Path) -> PolicyContract:
    """Read the logged env dimensions from a run's `params/env.yaml`.

    Isaac's dumped env.yaml embeds python object tags that break
    `yaml.safe_load`, so only the two flat top-level integer lines are
    matched. Hardcoding dims in skill adapters is what this replaces —
    retrained tracks ship different contracts (e.g. bimanual pour is
    51/15 where the legacy right-arm pour was 55/12).
    """
    found: dict[str, int] = {}
    for line in env_yaml.read_text().splitlines():
        match = _CONTRACT_LINE.match(line)
        if match:
            found.setdefault(match.group(1), int(match.group(2)))
    missing = {"observation_space", "action_space"} - set(found)
    if missing:
        raise ValueError(f"{env_yaml} lacks contract keys: {sorted(missing)}")
    return PolicyContract(
        observation_dim=found["observation_space"],
        action_dim=found["action_space"],
    )


@dataclass(frozen=True)
class PolicyArtifacts:
    """Existing policy checkpoint and the exact logged training parameters."""

    task_id: str
    run_dir: Path
    checkpoint: Path
    agent_yaml: Path
    env_yaml: Path


class CheckpointResolver:
    """Resolve policy artifacts without copying or guessing across runs."""

    def __init__(
        self,
        hdgp_root: Path,
        *,
        task_logs: Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        self.hdgp_root = hdgp_root.resolve()
        self.task_logs = dict(task_logs if task_logs is not None else _DEFAULT_TASK_LOGS)

    def resolve(
        self,
        task_id: str,
        run_dir: str,
        *,
        checkpoint: Path | None = None,
    ) -> PolicyArtifacts:
        if checkpoint is not None:
            selected_checkpoint = checkpoint.expanduser()
            if not selected_checkpoint.is_absolute():
                selected_checkpoint = self.hdgp_root / selected_checkpoint
            selected_checkpoint = selected_checkpoint.resolve()
            selected_run = selected_checkpoint.parent.parent
        else:
            try:
                side, task_folder = self.task_logs[task_id]
            except KeyError as exc:
                raise KeyError(f"unsupported task: {task_id}") from exc
            log_root = self.hdgp_root / "log/rl_games" / side / task_folder
            runs = tuple(sorted(path for path in log_root.glob(run_dir) if path.is_dir()))
            if not runs:
                raise FileNotFoundError(log_root / run_dir)
            if len(runs) != 1:
                raise ValueError(f"run selector must resolve exactly one directory: {run_dir}: {runs}")
            selected_run = runs[0]
            selected_checkpoint = selected_run / "nn" / f"{task_id}.pth"

        agent_yaml = selected_run / "params/agent.yaml"
        env_yaml = selected_run / "params/env.yaml"
        missing = tuple(path for path in (selected_checkpoint, agent_yaml, env_yaml) if not path.is_file())
        if missing:
            raise FileNotFoundError(f"missing policy artifacts: {missing}")
        return PolicyArtifacts(
            task_id=task_id,
            run_dir=selected_run,
            checkpoint=selected_checkpoint,
            agent_yaml=agent_yaml,
            env_yaml=env_yaml,
        )
