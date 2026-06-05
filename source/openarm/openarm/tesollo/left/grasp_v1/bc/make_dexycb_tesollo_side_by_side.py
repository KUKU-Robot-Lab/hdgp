#!/usr/bin/env python3
"""Build side-by-side video: DexYCB RGB frame (left) vs Tesollo replay video (right)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml


def _collect_label_files(dex_ycb_dir: Path, stride: int, start_index: int, max_frames: int) -> List[Path]:
    labels = sorted(dex_ycb_dir.glob("*/*/*/labels_*.npz"))
    if stride > 1:
        labels = labels[::stride]
    if start_index > 0:
        labels = labels[start_index:]
    if max_frames > 0:
        labels = labels[:max_frames]
    return labels


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


def _label_to_color_file(label_file: Path) -> Path:
    stem = label_file.stem.replace("labels_", "")
    return label_file.parent / f"color_{stem}.jpg"


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    new_w = int(w * (height / h))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dex-ycb-dir", type=Path, required=True)
    p.add_argument("--tesollo-video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mano-side", type=str, default="left", choices=["left", "right", "both"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all available")
    p.add_argument("--video-start", type=int, default=0, help="Start frame index in tesollo video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = _collect_label_files(args.dex_ycb_dir, args.stride, args.start_index, args.max_frames)
    if not labels:
        raise RuntimeError("No DexYCB label files found")

    side_cache: Dict[Path, str] = {}
    filtered = []
    for p in labels:
        if args.mano_side == "both":
            filtered.append(p)
            continue
        side = _read_mano_side_from_meta(p, side_cache)
        if side == args.mano_side:
            filtered.append(p)

    if not filtered:
        raise RuntimeError(f"No label files after mano_side filter: {args.mano_side}")

    cap = cv2.VideoCapture(str(args.tesollo_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open tesollo video: {args.tesollo_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.video_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.video_start)

    writer = None
    written = 0

    try:
        for i, label_file in enumerate(filtered):
            ok, tesollo_frame = cap.read()
            if not ok:
                break

            color_file = _label_to_color_file(label_file)
            if not color_file.exists():
                continue
            dexycb_frame = cv2.imread(str(color_file), cv2.IMREAD_COLOR)
            if dexycb_frame is None:
                continue

            target_h = max(dexycb_frame.shape[0], tesollo_frame.shape[0])
            left = _resize_to_height(dexycb_frame, target_h)
            right = _resize_to_height(tesollo_frame, target_h)
            canvas = np.hstack([left, right])

            cv2.putText(canvas, f"DexYCB frame {i}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, f"Tesollo replay frame {i + args.video_start}", (left.shape[1] + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                h, w = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

            writer.write(canvas)
            written += 1
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    print(f"Saved: {args.output}")
    print(f"frames_written={written}")
    print(f"dexycb_frames_after_filter={len(filtered)}")
    print(f"tesollo_video_frames_total={total_video_frames}")


if __name__ == "__main__":
    main()

