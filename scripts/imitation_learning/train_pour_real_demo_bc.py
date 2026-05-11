#!/usr/bin/env python3
"""Offline BC pretraining for 5g_pour_right_v4 real pour HDF5 demos.

This script does not create an IsaacLab environment.  It loads the real HDF5
teleoperation demos, converts them to the v4 11D policy action contract, trains
the RL-Games LSTM actor network on `(actor_obs, action)` sequences, and writes a
checkpoint that RL-Games PPO can load with `--checkpoint`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import yaml


HDGP_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = (
    HDGP_ROOT
    / "source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_pour_right_v4"
)
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from real_demo_bc import RealDemoBCBuffer, load_real_demo_episodes  # noqa: E402
from recurrent_gate import install_recurrent_gate  # noqa: E402


DEFAULT_AGENT_CFG = TASK_DIR / "config/agents/rl_games_ppo_lstm_bc_cfg.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_glob",
        required=True,
        action="append",
        help="Glob for real pour HDF5 demos. Repeat the flag to include multiple patterns.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output RL-Games checkpoint path.")
    parser.add_argument("--agent_cfg", type=Path, default=DEFAULT_AGENT_CFG)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--demo_stride", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--policy_obs_dim", type=int, default=110)
    parser.add_argument("--critic_obs_dim", type=int, default=143)
    parser.add_argument("--action_dim", type=int, default=11)
    parser.add_argument("--pour_sample_ratio", type=float, default=0.6)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def _load_agent_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["params"]


def _build_rlgames_model(params: dict, *, obs_dim: int, action_dim: int, device: torch.device):
    from rl_games.algos_torch import model_builder

    params = dict(params)
    params["config"] = dict(params.get("config", {}))
    params["config"]["normalize_input"] = True
    params["config"]["normalize_value"] = True
    builder = model_builder.ModelBuilder()
    model = builder.load(params).build(
        {
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
        }
    )
    return model.to(device)


def _build_central_value_state(params: dict, *, critic_obs_dim: int, action_dim: int, device: torch.device) -> dict | None:
    """Build an untrained asymmetric critic state so PPO restore succeeds."""
    if not bool(params.get("config", {}).get("has_central_value", False)):
        return None

    from rl_games.algos_torch import central_value, model_builder

    cfg = params["config"]
    cv_cfg = dict(cfg["central_value_config"])
    if "model" not in cv_cfg:
        cv_cfg["model"] = {"name": "central_value"}
    network = model_builder.ModelBuilder().load(cv_cfg)

    horizon = int(cfg.get("horizon_length", 64))
    minibatch = int(cv_cfg.get("minibatch_size", 2048))
    num_actors = max(1, minibatch // max(1, horizon))
    cv = central_value.CentralValueTrain(
        state_shape=(critic_obs_dim,),
        value_size=1,
        ppo_device=device,
        num_agents=1,
        horizon_length=horizon,
        num_actors=num_actors,
        num_actions=action_dim,
        seq_length=int(cfg.get("seq_length", 16)),
        normalize_value=bool(cv_cfg.get("normalize_value", True)),
        network=network,
        config=cv_cfg,
        writter=None,
        max_epochs=int(cfg.get("max_epochs", 10000)),
        multi_gpu=False,
        zero_rnn_on_done=bool(cfg.get("zero_rnn_on_done", False)),
    ).to(device)
    if bool(cfg.get("recurrent_gate_enable", True)):
        install_recurrent_gate(cv.model.a2c_network, obs_dim=critic_obs_dim)
    return cv.state_dict()


def _adapt_obs(obs: torch.Tensor, obs_dim: int) -> torch.Tensor:
    # real_demo_bc._remap_hdf5_obs_to_sim_layout()에서 이미 110D sim layout으로 변환됨.
    # obs_dim과 일치하면 그대로, 초과하면 자름. 부족한 경우는 없어야 함.
    current_dim = int(obs.shape[-1])
    if current_dim == obs_dim:
        return obs
    if current_dim > obs_dim:
        return obs[..., :obs_dim]
    # 이 경로는 더 이상 발생하지 않아야 함 (remap 후 110D = obs_dim)
    pad = torch.zeros(*obs.shape[:-1], obs_dim - current_dim, dtype=obs.dtype, device=obs.device)
    return torch.cat([obs, pad], dim=-1)


def _zero_rnn_states(model, batch_size: int, device: torch.device):
    rnn = getattr(model.a2c_network, "rnn", None)
    if rnn is None:
        return None
    layers = int(getattr(rnn, "num_layers", 1))
    hidden = int(getattr(rnn, "hidden_size", 256))
    state = torch.zeros(layers, batch_size, hidden, device=device)
    return [state.clone(), state.clone()]


def _bc_loss(model, batch: dict, *, obs_dim: int, seq_len: int, device: torch.device) -> torch.Tensor:
    obs = _adapt_obs(batch["obs"].to(device), obs_dim)
    actions = batch["actions"].to(device)
    mask = batch["mask"].to(device).float()
    batch_size = int(obs.shape[0])
    flat_obs = obs.reshape(batch_size * seq_len, obs_dim)
    flat_actions = actions.reshape(batch_size * seq_len, -1)
    result = model(
        {
            "is_train": True,
            "obs": flat_obs,
            "prev_actions": flat_actions,
            "seq_length": seq_len,
            "rnn_states": _zero_rnn_states(model, batch_size, device),
        }
    )
    mu = result["mus"].reshape(batch_size, seq_len, -1)
    per_step = (mu - actions).pow(2).sum(dim=-1)
    return (per_step * mask).sum() / mask.sum().clamp(min=1.0)


def _make_checkpoint(model, optimizer, *, epoch: int, frame: int, central_value_state: dict | None) -> dict:
    checkpoint = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "frame": int(frame),
        "optimizer": optimizer.state_dict(),
        "last_mean_rewards": torch.tensor(-1.0e9).cpu().numpy(),
        "env_state": None,
    }
    if central_value_state is not None:
        checkpoint["assymetric_vf_nets"] = central_value_state
        checkpoint["central_val_stats"] = {}
    return checkpoint


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    paths = []
    for pattern in args.dataset_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched --dataset_glob={args.dataset_glob!r}")

    print(f"[INFO] Loading {len(paths)} demos, stride={args.demo_stride}", flush=True)
    episodes = load_real_demo_episodes(paths, demo_stride=args.demo_stride, device=device)
    buffer = RealDemoBCBuffer(episodes, device=device, pour_sample_ratio=args.pour_sample_ratio)

    params = _load_agent_params(args.agent_cfg)
    model = _build_rlgames_model(params, obs_dim=args.policy_obs_dim, action_dim=args.action_dim, device=device)
    if bool(params.get("config", {}).get("recurrent_gate_enable", True)):
        install_recurrent_gate(model.a2c_network, obs_dim=args.policy_obs_dim)
    central_value_state = _build_central_value_state(
        params,
        critic_obs_dim=args.critic_obs_dim,
        action_dim=args.action_dim,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1.0e-8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = 0
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            batch = buffer.sample(args.batch_size, args.seq_len)
            loss = _bc_loss(model, batch, obs_dim=args.policy_obs_dim, seq_len=args.seq_len, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            total += float(loss.detach().item())
            frame += args.batch_size * args.seq_len
        avg = total / float(args.steps_per_epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[BC] epoch={epoch:04d}/{args.epochs} loss={avg:.6f} "
                f"pour_sample_ratio={buffer.last_pour_phase_ratio:.3f}",
                flush=True,
            )
        if avg < best_loss or epoch % args.save_every == 0 or epoch == args.epochs:
            best_loss = min(best_loss, avg)
            torch.save(
                _make_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    frame=frame,
                    central_value_state=central_value_state,
                ),
                args.output,
            )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                print(f"[INFO] Saved checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
