#!/usr/bin/env python3
"""Train a small BC-RNN policy for the 91D pre_pour_bc dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATASET = Path("/home/user/rl_ws/teleopration_openarm_tesollo/datasets/pre_pour_bc_91d_align_trunc.hdf5")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "logs" / "bc_rnn" / "best.pt"


class BCRNN(nn.Module):
    def __init__(self, obs_dim: int = 91, action_dim: int = 18, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(obs_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(obs)
        return self.head(out)


def group_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target)
    weights[..., 6:11] = 0.5
    return torch.mean(weights * (pred - target).square())


class PrePourSequenceDataset(Dataset):
    def __init__(self, path: Path, seq_len: int = 64):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        with h5py.File(path, "r") as f:
            for demo in f["data"].values():
                obs = torch.as_tensor(demo["obs/actor_obs"][:], dtype=torch.float32)
                actions = torch.as_tensor(demo["actions"][:], dtype=torch.float32)
                n = min(obs.shape[0], actions.shape[0])
                for start in range(0, max(1, n - seq_len + 1), seq_len):
                    end = min(start + seq_len, n)
                    if end - start < 2:
                        continue
                    self.samples.append((obs[start:end], actions[start:end]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def _pad_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    obs, actions = zip(*batch)
    return nn.utils.rnn.pad_sequence(obs, batch_first=True), nn.utils.rnn.pad_sequence(actions, batch_first=True)


def train(dataset_path: Path, checkpoint_path: Path, *, epochs: int = 50, batch_size: int = 16, lr: float = 1e-3) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PrePourSequenceDataset(dataset_path)
    if not data:
        raise ValueError(f"dataset has no trainable sequences: {dataset_path}")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=_pad_batch)
    model = BCRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total = 0.0
        for obs, target in loader:
            obs = obs.to(device)
            target = target.to(device)
            loss = group_weighted_mse(model(obs), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        mean_loss = total / max(1, len(loader))
        if mean_loss < best:
            best = mean_loss
            torch.save({"model": model.state_dict(), "loss": best, "obs_dim": 91, "action_dim": 18}, checkpoint_path)
        print(f"epoch={epoch + 1} loss={mean_loss:.6f} best={best:.6f}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.dataset, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

