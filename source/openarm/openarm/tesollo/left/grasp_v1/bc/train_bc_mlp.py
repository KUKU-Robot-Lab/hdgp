#!/usr/bin/env python3
"""Simple behavior cloning trainer (MLP regressor).

Input dataset (.npz):
- required: actions [N, action_dim]
- preferred: observations [N, obs_dim]
- fallback: tesollo_joints [N, 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class BCActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = np.load(args.dataset, allow_pickle=True)

    if "observations" in ds:
        obs = ds["observations"].astype(np.float32)
    elif "tesollo_joints" in ds:
        obs = ds["tesollo_joints"].astype(np.float32)
        print("[WARN] observations missing, using tesollo_joints as fallback input")
    else:
        raise KeyError("Dataset needs 'observations' or 'tesollo_joints'")

    if "actions" not in ds:
        raise KeyError("Dataset needs 'actions'")
    act = ds["actions"].astype(np.float32)

    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"length mismatch obs={obs.shape[0]} act={act.shape[0]}")

    x = torch.from_numpy(obs)
    y = torch.from_numpy(act)

    dataset = TensorDataset(x, y)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCActor(obs_dim=obs.shape[1], act_dim=act.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += float(loss.item()) * xb.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs.shape[1],
                "act_dim": act.shape[1],
                "best_val_mse": best_val,
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Saved checkpoint: {args.out}")
    print(f"Best val MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
