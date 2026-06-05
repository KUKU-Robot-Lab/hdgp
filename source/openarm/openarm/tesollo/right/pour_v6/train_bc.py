#!/usr/bin/env python3
"""Phase 1: Diffusion BC pre-training. Isaac Sim 불필요.

Usage:
    cd <v6_dir>
    python3 train_bc.py --epochs 5000

    # smoke test (10 epoch)
    python3 train_bc.py --epochs 10 --batch_size 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# v6 디렉터리를 sys.path에 추가 (diffusion_bc 패키지 import용)
_V6_DIR = Path(__file__).parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from diffusion_bc.trainer import train_bc

_DEFAULT_DEMO_DIR    = "/home/user/rl_ws/datasets"
_DEFAULT_OUTPUT_DIR  = "/home/user/rl_ws/hdgp/log/diffusion_bc/v6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Diffusion BC pre-training")
    parser.add_argument("--demo_dir",   default=_DEFAULT_DEMO_DIR)
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs",     type=int,   default=5000)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--stride",     type=int,   default=1)
    parser.add_argument("--save_every", type=int,   default=500)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    demo_paths = [demo_dir / f"pour_v1_a{i}.hdf5" for i in range(11, 21)]
    existing = [p for p in demo_paths if p.exists()]
    if not existing:
        print(f"[train_bc] ERROR: no demo files found in {demo_dir}")
        sys.exit(1)
    print(f"[train_bc] found {len(existing)}/{len(demo_paths)} demo files")

    train_bc(
        demo_paths=existing,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stride=args.stride,
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
