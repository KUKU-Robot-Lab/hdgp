#!/usr/bin/env python3
"""Convert Allegro-style hand demo sequence into Tesollo joints and v3 actions.

Expected input format (GraspXL-like):
- np.load(path, allow_pickle=True).item()
- data["right_hand"]["pose"]: [T, 22] (first 6 dims can be zeros)

Output .npz fields:
- actions: [T, action_dim] for 5g_lift_left_v3 action order
- tesollo_joints: [T, 20] in config["target"]["joint_order"]
- synergy: [T]
- thumb_action: [T, 8]
- meta_json: JSON metadata string
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _nested_get(d: dict, dotted_key: str):
    cur = d
    for part in dotted_key.split("."):
        cur = cur[part]
    return cur


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _denormalize(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


def _clip_joint(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _build_source_matrix(pose: np.ndarray, cfg: dict) -> Tuple[np.ndarray, List[str]]:
    source_cfg = cfg["source"]
    skip_first = int(source_cfg.get("skip_first", 0))
    pose_dim = int(source_cfg["pose_dim"])
    joint_order = list(source_cfg["joint_order"])

    if pose.shape[1] < skip_first + pose_dim:
        raise ValueError(
            f"Input pose dim is too small: got {pose.shape[1]}, need >= {skip_first + pose_dim}"
        )

    src = pose[:, skip_first : skip_first + pose_dim]
    if src.shape[1] != len(joint_order):
        raise ValueError(
            f"source pose dim mismatch: src={src.shape[1]} joint_order={len(joint_order)}"
        )
    return src, joint_order


def _retarget_to_tesollo(src: np.ndarray, src_joint_order: List[str], cfg: dict) -> Tuple[np.ndarray, List[str]]:
    target_cfg = cfg["target"]
    map_s2t: Dict[str, str] = target_cfg["map_source_to_target"]
    src_limits: Dict[str, List[float]] = cfg["source"]["joint_limits"]
    tgt_limits: Dict[str, List[float]] = target_cfg["joint_limits"]
    tgt_order: List[str] = target_cfg["joint_order"]
    defaults: Dict[str, float] = target_cfg["defaults"]

    T = src.shape[0]
    out = np.zeros((T, len(tgt_order)), dtype=np.float32)

    for j, tgt_name in enumerate(tgt_order):
        out[:, j] = float(defaults.get(tgt_name, 0.0))

    src_index = {name: i for i, name in enumerate(src_joint_order)}
    tgt_index = {name: i for i, name in enumerate(tgt_order)}

    for src_name, tgt_name in map_s2t.items():
        if src_name not in src_index:
            continue
        if tgt_name not in tgt_index:
            continue

        sidx = src_index[src_name]
        tidx = tgt_index[tgt_name]

        s_lo, s_hi = src_limits[src_name]
        t_lo, t_hi = tgt_limits[tgt_name]

        u = _normalize(src[:, sidx], float(s_lo), float(s_hi))
        u = np.clip(u, 0.0, 1.0)
        q_tgt = _denormalize(u, float(t_lo), float(t_hi))
        out[:, tidx] = _clip_joint(q_tgt, float(t_lo), float(t_hi))

    return out, tgt_order


def _compute_left_synergy(tesollo: np.ndarray, cfg: dict, tgt_order: List[str]) -> np.ndarray:
    v3 = cfg["v3_action"]
    open_pose = v3["left_synergy_open"]
    close_pose = v3["left_synergy_close"]

    idx = {name: i for i, name in enumerate(tgt_order)}
    keys = [k for k in open_pose.keys() if k in idx and k in close_pose]
    if not keys:
        return np.full((tesollo.shape[0],), -1.0, dtype=np.float32)

    t_values = []
    for k in keys:
        q = tesollo[:, idx[k]]
        q0 = float(open_pose[k])
        q1 = float(close_pose[k])
        if abs(q1 - q0) < 1e-6:
            continue
        t = (q - q0) / (q1 - q0)
        t_values.append(np.clip(t, 0.0, 1.0))

    if not t_values:
        return np.full((tesollo.shape[0],), -1.0, dtype=np.float32)

    t_avg = np.mean(np.stack(t_values, axis=1), axis=1)
    synergy = 2.0 * t_avg - 1.0
    return np.clip(synergy, -1.0, 1.0).astype(np.float32)


def _compute_left_thumb_action(tesollo: np.ndarray, cfg: dict, tgt_order: List[str]) -> np.ndarray:
    v3 = cfg["v3_action"]
    joint_order = list(v3["left_thumb_joint_order"])
    offset = np.asarray(v3["left_thumb_offset"], dtype=np.float32)
    scale = float(v3["left_thumb_scale"])

    idx = {name: i for i, name in enumerate(tgt_order)}
    T = tesollo.shape[0]
    out = np.zeros((T, len(joint_order)), dtype=np.float32)

    for j, name in enumerate(joint_order):
        q = tesollo[:, idx[name]] if name in idx else np.zeros((T,), dtype=np.float32)
        a = (q - offset[j]) / scale
        out[:, j] = np.clip(a, -1.0, 1.0)

    return out


def _build_full_action(synergy: np.ndarray, thumb_action: np.ndarray, cfg: dict) -> np.ndarray:
    v3 = cfg["v3_action"]
    total_dim = int(v3["full_action_dim"])

    left_arm_dim = int(v3["left_arm_dim"])
    left_hand_dim = int(v3["left_hand_dim"])
    left_thumb_dim = int(v3["left_thumb_dim"])
    right_arm_dim = int(v3["right_arm_dim"])
    right_thumb_dim = int(v3["right_thumb_dim"])
    right_hand_default = float(v3["right_hand_default"])

    T = synergy.shape[0]
    act = np.zeros((T, total_dim), dtype=np.float32)

    i = 0
    act[:, i : i + left_arm_dim] = 0.0
    i += left_arm_dim

    act[:, i : i + left_hand_dim] = synergy.reshape(-1, 1)
    i += left_hand_dim

    act[:, i : i + left_thumb_dim] = thumb_action
    i += left_thumb_dim

    act[:, i : i + right_arm_dim] = 0.0
    i += right_arm_dim

    act[:, i : i + 1] = right_hand_default
    i += 1

    act[:, i : i + right_thumb_dim] = 0.0

    return act


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="Input Allegro demo .npy")
    p.add_argument("--config", required=True, type=Path, help="Retarget config JSON")
    p.add_argument("--output", required=True, type=Path, help="Output .npz path")
    p.add_argument("--stride", type=int, default=1, help="Temporal subsample stride")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data = np.load(args.input, allow_pickle=True).item()
    pose_key = cfg["source"]["pose_key"]
    pose = np.asarray(_nested_get(data, pose_key), dtype=np.float32)

    if pose.ndim != 2:
        raise ValueError(f"Expected 2D pose array, got shape={pose.shape}")

    if args.stride > 1:
        pose = pose[:: args.stride]

    src, src_joint_order = _build_source_matrix(pose, cfg)
    tesollo, tgt_order = _retarget_to_tesollo(src, src_joint_order, cfg)

    synergy = _compute_left_synergy(tesollo, cfg, tgt_order)
    thumb_action = _compute_left_thumb_action(tesollo, cfg, tgt_order)
    actions = _build_full_action(synergy, thumb_action, cfg)

    meta = {
        "input": str(args.input),
        "config": str(args.config),
        "num_frames": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "target_joint_order": tgt_order,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        actions=actions,
        tesollo_joints=tesollo,
        synergy=synergy,
        thumb_action=thumb_action,
        meta_json=json.dumps(meta),
    )

    print(f"Saved: {args.output}")
    print(f"frames={actions.shape[0]} action_dim={actions.shape[1]}")


if __name__ == "__main__":
    main()
