#!/usr/bin/env python3
"""Replay grasp action sequence and record rendered frames to an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay actions in grasp env and record video.")
    parser.add_argument("--task", type=str, default="5g_grasp_left-v1")
    parser.add_argument("--input-actions", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="If > 0, cap replay steps.")
    parser.add_argument("--reset-on-done", action="store_true", default=False)
    parser.add_argument("--disable-obs-noise", action="store_true", default=True)
    parser.add_argument("--fps", type=float, default=0.0, help="Override output fps. 0 means use env dt.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _to_uint8_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    data = np.load(args_cli.input_actions, allow_pickle=True)
    if "actions" not in data:
        raise KeyError("Input dataset must contain 'actions'")
    actions_np = data["actions"].astype(np.float32)

    if args_cli.max_steps > 0:
        actions_np = actions_np[: args_cli.max_steps]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if getattr(env_cfg, "observations", None) is not None and getattr(env_cfg.observations, "policy", None) is not None:
        env_cfg.observations.policy.concatenate_terms = True
        if args_cli.disable_obs_noise:
            env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    obs, _ = env.reset()

    frame_rgb = env.render()
    if frame_rgb is None:
        raise RuntimeError("env.render() returned None. Try running without --headless.")

    frame_bgr = _to_uint8_bgr(np.asarray(frame_rgb))
    h, w = frame_bgr.shape[:2]
    fps = float(args_cli.fps) if args_cli.fps > 0 else float(1.0 / env.step_dt)
    args_cli.output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args_cli.output_video), fourcc, fps, (w, h))

    try:
        with torch.inference_mode():
            for t in range(actions_np.shape[0]):
                writer.write(frame_bgr)
                action_tensor = torch.from_numpy(actions_np[t]).to(env.device).unsqueeze(0)
                obs, _, terminated, truncated, _ = env.step(action_tensor)
                frame_rgb = env.render()
                if frame_rgb is None:
                    break
                frame_bgr = _to_uint8_bgr(np.asarray(frame_rgb))

                done = bool((terminated | truncated).any().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
                if done:
                    if args_cli.reset_on_done:
                        obs, _ = env.reset()
                        frame_rgb = env.render()
                        if frame_rgb is None:
                            break
                        frame_bgr = _to_uint8_bgr(np.asarray(frame_rgb))
                    else:
                        print(f"[INFO] Episode terminated at step {t}. Stopping replay.")
                        break
    finally:
        writer.release()
        env.close()

    print(f"Saved video: {args_cli.output_video}")


if __name__ == "__main__":
    main()
    simulation_app.close()

