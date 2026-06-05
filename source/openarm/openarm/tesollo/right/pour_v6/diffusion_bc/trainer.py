"""Phase 1: Diffusion BC 학습 루프.

Isaac Sim / diffusers / diffusion_policy 레포 불필요. 순수 PyTorch.
v6 폴더만 복사해도 실행 가능.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .net import DiffusionBCNet, EMAModel
from .buffer import DemoBCDatasetV2


def train_bc(
    demo_paths: Sequence[Path],
    output_dir: str | Path,
    num_epochs: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    chunk_size: int = 16,
    stride: int = 1,
    save_every: int = 500,
    device: str = "cuda",
) -> None:
    """Diffusion BC Phase 1 학습.

    Args:
        demo_paths: HDF5 demo 파일 목록 (pour_v1_a11~a20.hdf5)
        output_dir: checkpoint 저장 디렉터리
        num_epochs: 학습 epoch 수
        batch_size: mini-batch 크기
        lr: Adam learning rate
        chunk_size: action chunk 길이
        stride: HDF5 서브샘플링 stride
        save_every: checkpoint 저장 주기
        device: "cuda" or "cpu"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DemoBCDatasetV2(demo_paths, stride=stride, chunk_size=chunk_size)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty — demo files not found: {list(demo_paths)[:3]}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    model = DiffusionBCNet(chunk_size=chunk_size).to(device)
    ema   = EMAModel(model, decay=0.999)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=lr * 0.1)

    print(f"[train_bc] dataset size: {len(dataset):,} samples, {len(loader):,} batches/epoch")
    print(f"[train_bc] output_dir: {output_dir}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for obs, actions, mask in loader:
            obs     = obs.to(device)
            actions = actions.to(device)
            mask    = mask.to(device)

            loss = model.compute_loss(obs, actions, mask)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ema.step(model)
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[train_bc] epoch {epoch+1:5d}/{num_epochs} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            ckpt_path = output_dir / f"bc_{epoch+1}.pt"
            torch.save({
                "model": ema.averaged_model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, ckpt_path)
            print(f"[train_bc] saved checkpoint: {ckpt_path}")

    print("[train_bc] Phase 1 complete.")
