#!/usr/bin/env python3
"""Convert DemoGrasp reference pkls to normalized palm(6)+pca(5) .pt files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


def quat_xyzw_to_euler_zyx(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternion sequence to intrinsic ZYX Euler [ez, ey, ex]."""
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    ex = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    ey = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    ez = torch.atan2(t3, t4)
    return torch.stack([ez, ey, ex], dim=-1)


def teosollo_pca_basis() -> torch.Tensor:
    """Return the same 5x20 PCA basis used by OpenArm Teosollo fabric."""
    return torch.tensor(
        [
            [
                -1.4790e-02, -9.8163e-02, 4.3551e-02, 3.1699e-01,
                -3.8872e-02, 3.7917e-01, 4.4703e-01, 7.1016e-03,
                2.1159e-03, 3.2014e-01, 4.4660e-01, 5.2108e-02,
                5.6869e-05, 2.9845e-01, 3.8575e-01, 7.5774e-03,
                0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00,
            ],
            [
                2.9753e-02, -2.6149e-02, 6.6994e-02, 1.8117e-01,
                -5.1148e-02, -1.3007e-01, 5.7727e-02, 5.7914e-01,
                1.0156e-02, -1.8469e-01, 5.3809e-02, 5.4888e-01,
                1.3351e-04, -1.7747e-01, 2.7809e-02, 4.8187e-01,
                0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00,
            ],
            [
                2.3925e-03, -3.7238e-02, -1.0124e-01, -1.7442e-02,
                -5.7137e-02, -3.4707e-01, 3.3365e-01, -1.8029e-01,
                -4.3560e-02, -4.7666e-01, 3.2517e-01, -1.5208e-01,
                -5.9691e-05, -4.5790e-01, 3.6536e-01, -1.3916e-01,
                0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00,
            ],
            [
                2.2661e-01, 5.9911e-01, 7.0257e-01, -2.4525e-01,
                2.2795e-02, -3.4090e-02, 3.4366e-02, -2.6531e-02,
                2.3471e-02, 4.6123e-02, 9.8059e-02, -1.2619e-03,
                -1.6452e-04, -1.3741e-02, 1.3813e-01, 2.8677e-02,
                0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00,
            ],
            [
                -4.7617e-01, 2.7734e-01, -2.3989e-01, -3.1222e-01,
                -4.4911e-02, -4.7156e-01, 9.3124e-02, 2.3135e-01,
                -2.4607e-03, 9.5564e-02, 1.2470e-01, 3.6613e-02,
                1.3821e-04, 4.6072e-01, 9.9315e-02, -8.1080e-02,
                0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00,
            ],
        ],
        dtype=torch.float32,
    )


def project_hand_qpos_to_pca5(hand_qpos: torch.Tensor, basis_5x20: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Project hand_qpos(T,D) to PCA(T,5) with pseudo-inverse."""
    src = hand_qpos
    src_dim = int(src.shape[1])

    if src_dim == 27:
        src = src[:, -20:]
        src_dim = 20

    if src_dim == 20:
        pca = src @ torch.linalg.pinv(basis_5x20)
        return pca, "pinv_full_20d"

    if src_dim < 20:
        basis_sub = basis_5x20[:, :src_dim]
        pca = src @ torch.linalg.pinv(basis_sub)
        return pca, f"pinv_subset_{src_dim}d"

    pca = src[:, :20] @ torch.linalg.pinv(basis_5x20)
    return pca, f"pinv_truncate_{src_dim}to20d"


def fit_pca_to_bounds(pca_seq: torch.Tensor, pca_mins: torch.Tensor, pca_maxs: torch.Tensor, margin: float) -> torch.Tensor:
    """Affine-map per-dimension PCA sequence into target bounds with optional inner margin."""
    margin = max(0.1, min(1.0, float(margin)))
    span = pca_maxs - pca_mins
    inner_min = pca_mins + 0.5 * (1.0 - margin) * span
    inner_max = pca_maxs - 0.5 * (1.0 - margin) * span

    src_min = pca_seq.min(dim=0).values
    src_max = pca_seq.max(dim=0).values
    src_span = (src_max - src_min).clamp(min=1e-6)
    norm = (pca_seq - src_min) / src_span
    fitted = inner_min + norm * (inner_max - inner_min)

    # Degenerate dims (almost constant): place at inner range midpoint.
    degenerate = (src_max - src_min) < 1e-6
    if degenerate.any():
        mid = 0.5 * (inner_min + inner_max)
        fitted[:, degenerate] = mid[degenerate].unsqueeze(0).expand(pca_seq.shape[0], -1)
    return fitted


def convert_one(
    src_path: Path,
    dst_path: Path,
    basis_5x20: torch.Tensor,
    fit_to_bounds: bool,
    pca_mins: torch.Tensor,
    pca_maxs: torch.Tensor,
    fit_margin: float,
) -> None:
    with src_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{src_path}: expected dict payload.")
    if "wrist_initobj_pos" not in payload or "wrist_quat" not in payload or "hand_qpos" not in payload:
        raise ValueError(f"{src_path}: missing required keys.")

    wrist_pos = torch.as_tensor(payload["wrist_initobj_pos"], dtype=torch.float32)
    wrist_quat = torch.as_tensor(payload["wrist_quat"], dtype=torch.float32)
    hand_qpos = torch.as_tensor(payload["hand_qpos"], dtype=torch.float32)

    if wrist_pos.ndim != 2 or wrist_pos.shape[1] != 3:
        raise ValueError(f"{src_path}: wrist_initobj_pos must be (T,3).")
    if wrist_quat.ndim != 2 or wrist_quat.shape[1] != 4:
        raise ValueError(f"{src_path}: wrist_quat must be (T,4) in xyzw.")
    if hand_qpos.ndim != 2:
        raise ValueError(f"{src_path}: hand_qpos must be (T,D).")

    steps = min(wrist_pos.shape[0], wrist_quat.shape[0], hand_qpos.shape[0])
    wrist_pos = wrist_pos[:steps]
    wrist_quat = wrist_quat[:steps]
    hand_qpos = hand_qpos[:steps]

    wrist_euler = quat_xyzw_to_euler_zyx(wrist_quat)
    palm_pose = torch.cat([wrist_pos, wrist_euler], dim=-1)  # (T,6)
    hand_pca, proj_mode = project_hand_qpos_to_pca5(hand_qpos, basis_5x20)

    projection_mode = proj_mode
    if fit_to_bounds:
        hand_pca = fit_pca_to_bounds(hand_pca, pca_mins, pca_maxs, margin=fit_margin)
        projection_mode = f"{proj_mode}+fit_to_bounds(margin={fit_margin:.2f})"

    out = {
        "palm_pose": palm_pose,
        "hand_pca": hand_pca,
        "meta": {
            "source_file": str(src_path),
            "steps": int(steps),
            "source_hand_qpos_dim": int(hand_qpos.shape[1]),
            "projection_mode": projection_mode,
        },
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_path)

    print(
        f"[OK] {src_path.name} -> {dst_path.name} | "
        f"steps={steps}, hand_dim={hand_qpos.shape[1]}, mode={proj_mode}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DemoGrasp pkl refs to normalized .pt refs")
    parser.add_argument(
        "--src_dir",
        type=Path,
        default=Path("/home/user/rl_ws/hdgp/assets/demograsp_references"),
    )
    parser.add_argument(
        "--dst_dir",
        type=Path,
        default=Path("/home/user/rl_ws/hdgp/assets/demograsp_references/normalized"),
    )
    parser.add_argument("--fit_to_bounds", action="store_true", help="Map PCA sequence into configured PCA bounds.")
    parser.add_argument("--fit_margin", type=float, default=0.95, help="Inner-range occupancy for fit_to_bounds.")
    args = parser.parse_args()

    src_dir = args.src_dir.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()
    basis = teosollo_pca_basis()
    pca_mins = torch.tensor([0.0, -0.5, -1.0, -1.2, -0.5], dtype=torch.float32)
    pca_maxs = torch.tensor([3.5, 2.0, 1.0, 2.0, 2.0], dtype=torch.float32)

    src_files = sorted(src_dir.glob("grasp_ref_*.pkl"))
    if not src_files:
        raise FileNotFoundError(f"No grasp_ref_*.pkl files found in: {src_dir}")

    for src in src_files:
        dst = dst_dir / f"{src.stem}_teosollo_pca5.pt"
        convert_one(
            src,
            dst,
            basis,
            fit_to_bounds=bool(args.fit_to_bounds),
            pca_mins=pca_mins,
            pca_maxs=pca_maxs,
            fit_margin=float(args.fit_margin),
        )
    print(f"[DONE] Converted {len(src_files)} files into: {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
