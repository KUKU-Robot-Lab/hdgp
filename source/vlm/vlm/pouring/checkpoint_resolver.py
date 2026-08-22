from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TASK_LOGS = {
    "open-tesol_r_grasp_v1-lstm": ("open-tesol/right", "grasp-v1"),
    "open-tesol_r_pour_v1-lstm": ("open-tesol/right", "pour-v1"),
}


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

    def __init__(self, hdgp_root: Path) -> None:
        self.hdgp_root = hdgp_root.resolve()

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
                side, task_folder = _TASK_LOGS[task_id]
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
