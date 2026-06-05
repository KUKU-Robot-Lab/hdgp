#!/usr/bin/env python3
"""Convert DexYCB MANO joints into Tesollo joints and grasp actions.

Input:
- DexYCB root directory containing labels_*.npz files
- label npz must include joint_3d with shape [1, 21, 3]

Output (.npz):
- actions: [T, action_dim] for 5g_grasp_v1 action order
- tesollo_joints: [T, 20] in config["target"]["joint_order"]
- synergy: [T]
- thumb_action: [T, 8]
- meta_json: JSON metadata string
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize_vec(a)
    bb = _normalize_vec(b)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.acos(c))


def _signed_angle_around_axis(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    uu = _normalize_vec(u)
    vv = _normalize_vec(v)
    ax = _normalize_vec(axis)
    cross = np.cross(uu, vv)
    return float(math.atan2(np.dot(cross, ax), np.dot(uu, vv)))


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _denormalize(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


def _clip_joint(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _set_from_ratio(src_limits: Dict[str, List[float]], name: str, ratio_01: float) -> float:
    lo, hi = src_limits[name]
    r = float(np.clip(ratio_01, 0.0, 1.0))
    return float(lo + r * (hi - lo))


def _set_from_signed(src_limits: Dict[str, List[float]], name: str, value_m11: float) -> float:
    lo, hi = src_limits[name]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    v = float(np.clip(value_m11, -1.0, 1.0))
    return float(np.clip(mid + v * half, lo, hi))


def _flex_triplet(j: np.ndarray, a: int, b: int, c: int, d: int, e: int) -> Tuple[float, float, float]:
    # flex ~= pi - inner_angle, so straight finger -> near 0, bent -> larger.
    f1 = math.pi - _angle(j[a] - j[b], j[c] - j[b])
    f2 = math.pi - _angle(j[b] - j[c], j[d] - j[c])
    f3 = math.pi - _angle(j[c] - j[d], j[e] - j[d])
    n1 = float(np.clip(f1 / (0.70 * math.pi), 0.0, 1.0))
    n2 = float(np.clip(f2 / (0.60 * math.pi), 0.0, 1.0))
    n3 = float(np.clip(f3 / (0.55 * math.pi), 0.0, 1.0))
    return n1, n2, n3


def _dexycb_joint3d_to_source(
    j: np.ndarray, src_limits: Dict[str, List[float]], source_order: List[str]
) -> np.ndarray:
    # DexYCB/MANO joint indexing used here:
    # 0:wrist, 1..4:thumb, 5..8:index, 9..12:middle, 13..16:ring, 17..20:little
    w = j[0]
    idx_mcp, mid_mcp, rng_mcp = j[5], j[9], j[13]
    palm_forward = _normalize_vec((idx_mcp - w) + (mid_mcp - w) + (rng_mcp - w))
    palm_side = _normalize_vec(rng_mcp - idx_mcp)
    palm_normal = _normalize_vec(np.cross(palm_forward, palm_side))
    if np.linalg.norm(palm_forward) < 1e-6:
        palm_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if np.linalg.norm(palm_normal) < 1e-6:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    src = {}

    def _maybe_set_finger(prefix: str, mcp: int, pip: int, dip: int, tip: int) -> None:
        f2, f3, f4 = _flex_triplet(j, 0, mcp, pip, dip, tip)
        f_dir = _normalize_vec(j[pip] - j[mcp])
        f_abd = _signed_angle_around_axis(palm_forward, f_dir, palm_normal) / 0.70
        if f"{prefix}_1" in src_limits:
            src[f"{prefix}_1"] = _set_from_signed(src_limits, f"{prefix}_1", f_abd)
        if f"{prefix}_2" in src_limits:
            src[f"{prefix}_2"] = _set_from_ratio(src_limits, f"{prefix}_2", f2)
        if f"{prefix}_3" in src_limits:
            src[f"{prefix}_3"] = _set_from_ratio(src_limits, f"{prefix}_3", f3)
        if f"{prefix}_4" in src_limits:
            src[f"{prefix}_4"] = _set_from_ratio(src_limits, f"{prefix}_4", f4)

    _maybe_set_finger("index", 5, 6, 7, 8)
    _maybe_set_finger("middle", 9, 10, 11, 12)
    _maybe_set_finger("ring", 13, 14, 15, 16)
    _maybe_set_finger("little", 17, 18, 19, 20)

    # Thumb
    t2, t3, t4 = _flex_triplet(j, 0, 1, 2, 3, 4)
    t_base_dir = _normalize_vec(j[2] - j[1])
    t_opp = _signed_angle_around_axis(palm_forward, t_base_dir, palm_normal) / 0.90
    if "thumb_1" in src_limits:
        src["thumb_1"] = _set_from_signed(src_limits, "thumb_1", t_opp)
    if "thumb_2" in src_limits:
        src["thumb_2"] = _set_from_ratio(src_limits, "thumb_2", t2)
    if "thumb_3" in src_limits:
        src["thumb_3"] = _set_from_ratio(src_limits, "thumb_3", t3)
    if "thumb_4" in src_limits:
        src["thumb_4"] = _set_from_ratio(src_limits, "thumb_4", t4)

    src_out = []
    for k in source_order:
        if k in src:
            src_out.append(src[k])
            continue
        if k in src_limits:
            lo, hi = src_limits[k]
            src_out.append(0.5 * (float(lo) + float(hi)))
        else:
            src_out.append(0.0)
    return np.asarray(src_out, dtype=np.float32)


def _retarget_to_tesollo(src: np.ndarray, src_joint_order: List[str], cfg: dict) -> Tuple[np.ndarray, List[str]]:
    target_cfg = cfg["target"]
    map_s2t: Dict[str, str] = target_cfg["map_source_to_target"]
    src_limits: Dict[str, List[float]] = cfg["source"]["joint_limits"]
    tgt_limits: Dict[str, List[float]] = target_cfg["joint_limits"]
    tgt_order: List[str] = target_cfg["joint_order"]
    defaults: Dict[str, float] = target_cfg["defaults"]

    t_steps = src.shape[0]
    out = np.zeros((t_steps, len(tgt_order)), dtype=np.float32)

    for j, tgt_name in enumerate(tgt_order):
        out[:, j] = float(defaults.get(tgt_name, 0.0))

    src_index = {name: i for i, name in enumerate(src_joint_order)}
    tgt_index = {name: i for i, name in enumerate(tgt_order)}

    for src_name, tgt_name in map_s2t.items():
        if src_name not in src_index or tgt_name not in tgt_index:
            continue
        sidx = src_index[src_name]
        tidx = tgt_index[tgt_name]

        s_lo, s_hi = src_limits[src_name]
        t_lo, t_hi = tgt_limits[tgt_name]
        u = _normalize(src[:, sidx], float(s_lo), float(s_hi))
        q_tgt = _denormalize(np.clip(u, 0.0, 1.0), float(t_lo), float(t_hi))
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
        t_values.append(np.clip((q - q0) / (q1 - q0), 0.0, 1.0))

    if not t_values:
        return np.full((tesollo.shape[0],), -1.0, dtype=np.float32)

    t_avg = np.mean(np.stack(t_values, axis=1), axis=1)
    return np.clip(2.0 * t_avg - 1.0, -1.0, 1.0).astype(np.float32)


def _compute_left_thumb_action(tesollo: np.ndarray, cfg: dict, tgt_order: List[str]) -> np.ndarray:
    v3 = cfg["v3_action"]
    joint_order = list(v3["left_thumb_joint_order"])
    offset = np.asarray(v3["left_thumb_offset"], dtype=np.float32)
    scale = float(v3.get("left_thumb_scale", 1.0))
    if len(joint_order) == 0:
        return np.zeros((tesollo.shape[0], 0), dtype=np.float32)
    if len(offset) != len(joint_order):
        raise ValueError(
            f"left_thumb_offset length ({len(offset)}) does not match left_thumb_joint_order ({len(joint_order)})"
        )
    if abs(scale) < 1e-8:
        scale = 1.0

    idx = {name: i for i, name in enumerate(tgt_order)}
    t_steps = tesollo.shape[0]
    out = np.zeros((t_steps, len(joint_order)), dtype=np.float32)
    for j, name in enumerate(joint_order):
        q = tesollo[:, idx[name]] if name in idx else np.zeros((t_steps,), dtype=np.float32)
        out[:, j] = np.clip((q - offset[j]) / scale, -1.0, 1.0)
    return out


def _build_full_action(synergy: np.ndarray, thumb_action: np.ndarray, tesollo: np.ndarray, tgt_order: List[str], cfg: dict) -> np.ndarray:
    v3 = cfg["v3_action"]
    mode = str(v3.get("mode", "synergy_thumb"))
    total_dim = int(v3["full_action_dim"])
    left_arm_dim = int(v3["left_arm_dim"])
    left_hand_dim = int(v3["left_hand_dim"])
    left_thumb_dim = int(v3["left_thumb_dim"])
    right_arm_dim = int(v3["right_arm_dim"])
    right_thumb_dim = int(v3["right_thumb_dim"])
    right_hand_default = float(v3["right_hand_default"])

    t_steps = synergy.shape[0]
    act = np.zeros((t_steps, total_dim), dtype=np.float32)

    i = 0
    act[:, i : i + left_arm_dim] = 0.0
    i += left_arm_dim

    if mode == "left_hand_20d":
        hand_order = list(v3.get("left_hand_joint_order", []))
        if len(hand_order) != left_hand_dim:
            raise ValueError(f"left_hand_joint_order length ({len(hand_order)}) != left_hand_dim ({left_hand_dim})")
        tgt_idx = {name: j for j, name in enumerate(tgt_order)}
        t_limits = cfg["target"]["joint_limits"]
        for j, name in enumerate(hand_order):
            if name not in tgt_idx:
                raise KeyError(f"left_hand_joint_order contains unknown joint: {name}")
            lo, hi = t_limits[name]
            q = tesollo[:, tgt_idx[name]]
            if hi <= lo:
                u = np.zeros_like(q)
            else:
                u = (q - float(lo)) / (float(hi) - float(lo))
            act[:, i + j] = np.clip(2.0 * u - 1.0, -1.0, 1.0)
    else:
        act[:, i : i + left_hand_dim] = synergy.reshape(-1, 1)
    i += left_hand_dim

    if left_thumb_dim > 0:
        if thumb_action.shape[1] != left_thumb_dim:
            raise ValueError(
                f"thumb_action dim ({thumb_action.shape[1]}) != configured left_thumb_dim ({left_thumb_dim})"
            )
        act[:, i : i + left_thumb_dim] = thumb_action
    i += left_thumb_dim
    act[:, i : i + right_arm_dim] = 0.0
    i += right_arm_dim
    act[:, i : i + 1] = right_hand_default
    i += 1
    act[:, i : i + right_thumb_dim] = 0.0

    return act


def _collect_label_files(dex_ycb_dir: Path, max_frames: int, stride: int) -> List[Path]:
    label_files = sorted(dex_ycb_dir.glob("*/*/*/labels_*.npz"))
    if stride > 1:
        label_files = label_files[::stride]
    if max_frames > 0:
        label_files = label_files[:max_frames]
    return label_files


def _read_mano_side_from_meta(label_file: Path, cache: Dict[Path, str]) -> str:
    seq_dir = label_file.parents[1]
    meta_path = seq_dir / "meta.yml"
    if meta_path in cache:
        return cache[meta_path]
    side = "unknown"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        mano_sides = meta.get("mano_sides", [])
        if isinstance(mano_sides, list) and mano_sides:
            side = str(mano_sides[0]).lower()
    cache[meta_path] = side
    return side


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dex-ycb-dir", type=Path, default=None, help="DexYCB root dir")
    p.add_argument("--config", required=True, type=Path, help="Retarget config JSON")
    p.add_argument("--output", required=True, type=Path, help="Output .npz path")
    p.add_argument("--max-frames", type=int, default=10000, help="Max number of labels to use")
    p.add_argument("--stride", type=int, default=1, help="Label subsample stride")
    p.add_argument("--start-index", type=int, default=0, help="Start label index in sorted file list")
    p.add_argument(
        "--mano-side",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="Filter DexYCB sequences by mano side. Default is left for v3-left pipeline.",
    )
    p.add_argument(
        "--debug-one-frame",
        type=int,
        default=-1,
        help="Print source->target mapping values for one frame index (0-based after filtering).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dex_ycb_dir = args.dex_ycb_dir
    if dex_ycb_dir is None:
        dex_ycb_env = os.environ.get("DEX_YCB_DIR", "")
        dex_ycb_dir = Path(dex_ycb_env) if dex_ycb_env else Path.cwd()
    if not dex_ycb_dir.exists():
        raise FileNotFoundError(f"DexYCB dir not found: {dex_ycb_dir}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    src_limits = cfg["source"]["joint_limits"]
    src_joint_order = list(cfg["source"]["joint_order"])

    label_files = _collect_label_files(dex_ycb_dir, max_frames=-1, stride=args.stride)
    if args.start_index > 0:
        label_files = label_files[args.start_index :]
    if args.max_frames > 0:
        label_files = label_files[: args.max_frames]
    if not label_files:
        raise RuntimeError("No label files found")

    src_rows = []
    used_files = []
    side_cache: Dict[Path, str] = {}
    side_kept = 0
    side_skipped = 0
    for p in label_files:
        if args.mano_side != "both":
            side = _read_mano_side_from_meta(p, side_cache)
            if side != args.mano_side:
                side_skipped += 1
                continue
        side_kept += 1
        d = np.load(p, allow_pickle=True)
        if "joint_3d" not in d.files:
            continue
        j3d = np.asarray(d["joint_3d"], dtype=np.float32)
        if j3d.ndim != 3 or j3d.shape[1:] != (21, 3):
            continue
        src_frame = _dexycb_joint3d_to_source(j3d[0], src_limits, src_joint_order)
        src_rows.append(src_frame)
        used_files.append(str(p))

    if not src_rows:
        raise RuntimeError("No valid joint_3d frames found")

    src = np.stack(src_rows, axis=0).astype(np.float32)
    if src.shape[1] != len(src_joint_order):
        raise ValueError(f"source dim mismatch: {src.shape[1]} vs {len(src_joint_order)}")

    tesollo, tgt_order = _retarget_to_tesollo(src, src_joint_order, cfg)
    synergy = _compute_left_synergy(tesollo, cfg, tgt_order)
    thumb_action = _compute_left_thumb_action(tesollo, cfg, tgt_order)
    actions = _build_full_action(synergy, thumb_action, tesollo, tgt_order, cfg)

    if args.debug_one_frame >= 0:
        fi = int(np.clip(args.debug_one_frame, 0, src.shape[0] - 1))
        map_s2t: Dict[str, str] = cfg["target"]["map_source_to_target"]
        src_idx = {n: i for i, n in enumerate(src_joint_order)}
        tgt_idx = {n: i for i, n in enumerate(tgt_order)}
        print("")
        print("=== DEBUG ONE FRAME ===")
        print(f"frame_index={fi}")
        print(f"label_file={used_files[fi]}")
        print("--- source joints ---")
        for n in src_joint_order:
            print(f"{n:>10s}: {src[fi, src_idx[n]]:+.6f}")
        print("--- mapped target joints ---")
        for s_name, t_name in map_s2t.items():
            if s_name in src_idx and t_name in tgt_idx:
                print(
                    f"{s_name:>10s} -> {t_name:<10s}: "
                    f"{src[fi, src_idx[s_name]]:+.6f} -> {tesollo[fi, tgt_idx[t_name]]:+.6f}"
                )
        print("--- full target joints ---")
        for n in tgt_order:
            print(f"{n:>10s}: {tesollo[fi, tgt_idx[n]]:+.6f}")
        print("=======================")
        print("")

    meta = {
        "dex_ycb_dir": str(dex_ycb_dir),
        "config": str(args.config),
        "num_frames": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "target_joint_order": tgt_order,
        "source_joint_order": src_joint_order,
        "start_index": int(args.start_index),
        "stride": int(args.stride),
        "mano_side_filter": args.mano_side,
        "side_kept_files": int(side_kept),
        "side_skipped_files": int(side_skipped),
        "used_label_files_head": used_files[:20],
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
    print(f"mano_side={args.mano_side} kept={side_kept} skipped={side_skipped}")


if __name__ == "__main__":
    main()
