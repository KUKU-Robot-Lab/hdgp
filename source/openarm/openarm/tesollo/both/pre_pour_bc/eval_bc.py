#!/usr/bin/env python3
"""Roll out a trained BC-RNN checkpoint in pre_pour_bc-play."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import gymnasium as gym

try:
    from . import PrePourBCPlayEnvCfg
    from .train_bc_rnn import BCRNN, DEFAULT_CHECKPOINT
except ImportError:
    from pathlib import Path as _Path
    import sys as _sys

    _pkg_file = _Path(__file__).resolve()
    _sys.path.insert(0, str(_pkg_file.parents[4]))
    from openarm.tesollo.both import pre_pour_bc as _task
    from openarm.tesollo.both.pre_pour_bc import PrePourBCPlayEnvCfg
    from openarm.tesollo.both.pre_pour_bc.train_bc_rnn import (
        BCRNN,
        DEFAULT_CHECKPOINT,
    )


def load_policy(path: Path, device: torch.device) -> BCRNN:
    ckpt = torch.load(path, map_location=device)
    model = BCRNN(obs_dim=int(ckpt.get("obs_dim", 91)), action_dim=int(ckpt.get("action_dim", 18))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate(checkpoint: Path, *, rollouts: int = 50, max_steps: int = 1300) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("pre_pour_bc-play", cfg=PrePourBCPlayEnvCfg())
    model = load_policy(checkpoint, device)
    successes = 0
    final_errors = []
    smoothness = []
    force_means = []
    curl_means = []

    for _ in range(rollouts):
        obs, _ = env.reset()
        hidden = None
        prev = torch.zeros(1, 18, device=device)
        for _step in range(max_steps):
            policy_obs = torch.as_tensor(obs["policy"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out, hidden = model.rnn(policy_obs, hidden)
                action = model.head(out[:, -1]).clamp(-1.0, 1.0)
            obs, _reward, terminated, truncated, info = env.step(action.cpu().numpy())
            smoothness.append(float(torch.linalg.norm(action - prev).item()))
            prev = action
            if bool(terminated[0] or truncated[0]):
                successes += int(bool(terminated[0]))
                break
        unwrapped = env.unwrapped
        final_errors.append(float(unwrapped.final_joint_error[0].item()))
        policy = obs["policy"][0] if isinstance(obs["policy"], torch.Tensor) else obs["policy"][0]
        force_means.append(float(policy[68:73].mean()))
        curl_means.append(float(prev[:, 6:11].mean().item()))

    env.close()
    return {
        "success_rate": successes / max(1, rollouts),
        "final_joint_error": sum(final_errors) / max(1, len(final_errors)),
        "action_smoothness": sum(smoothness) / max(1, len(smoothness)),
        "force_mean": sum(force_means) / max(1, len(force_means)),
        "curl_mean": sum(curl_means) / max(1, len(curl_means)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rollouts", type=int, default=50)
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, rollouts=args.rollouts)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
