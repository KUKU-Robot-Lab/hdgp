#!/usr/bin/env python3
"""Interactive thumb/pinky joint control.

Type joint values in the terminal and see the result in the viewport.

Usage:
    ./isaaclab.sh -p ../hdgp/scripts/probes/test_finger_joints.py \
        --task 5g_lift_left-v1 --num_envs 1

Commands:
    <joint_name> <value>       Set a single joint (e.g., lj_dg_1_2 1.5)
    all <value>                Set all thumb/pinky joints to same value
    reset                      Reset all to default
    show                       Show current joint values
    pose <name>                Apply a preset pose (open / half / close)
    help                       Show this help
    quit                       Exit
"""

import argparse
import sys
import threading
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive thumb/pinky joint control.")
    parser.add_argument("--task", type=str, default="5g_lift_left-v1")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)

    import gymnasium as gym
    import torch

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "source"))

    import openarm.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric if hasattr(args, "disable_fabric") else True,
    )
    env_cfg.episode_length_s = 9999.0

    env = gym.make(args.task, cfg=env_cfg)
    core_env = env.unwrapped
    robot = core_env.scene["robot"]

    # Determine which hand
    if "left" in args.task.lower():
        thumb_pinky_joints = [
            "lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4",
            "lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4",
        ]
        synergy_joints = [
            f"lj_dg_{f}_{j}" for f in range(2, 5) for j in range(1, 5)
        ]
        side = "LEFT"
    else:
        thumb_pinky_joints = [
            "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
            "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
        ]
        synergy_joints = [
            f"rj_dg_{f}_{j}" for f in range(2, 5) for j in range(1, 5)
        ]
        side = "RIGHT"

    all_finger_joints = thumb_pinky_joints + synergy_joints

    # Build joint name -> index map
    all_joint_names = list(robot.data.joint_names)
    joint_idx_map = {}
    for jname in all_finger_joints:
        if jname in all_joint_names:
            joint_idx_map[jname] = all_joint_names.index(jname)

    # Reset
    obs, _ = env.reset()

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    current_pos = default_joint_pos.clone()

    physics_dt = core_env.physics_dt

    # Shared state for interactive input
    command_queue = []
    lock = threading.Lock()
    running = True

    def show_joints():
        print(f"\n  {'─'*55}")
        print(f"  {'Joint':<15} {'Current':>10} {'Default':>10} {'Limits'}")
        print(f"  {'─'*55}")
        for jname in thumb_pinky_joints:
            idx = joint_idx_map[jname]
            cur = current_pos[0, idx].item()
            default = default_joint_pos[0, idx].item()
            finger = "THUMB" if "_1_" in jname else "PINKY"
            jnum = jname.split("_")[-1]
            print(f"  {jname:<15} {cur:>10.4f} {default:>10.4f}   {finger} j{jnum}")
        print(f"  {'─'*55}")
        print(f"\n  Synergy joints (2,3,4):")
        for jname in synergy_joints:
            idx = joint_idx_map[jname]
            cur = current_pos[0, idx].item()
            print(f"  {jname:<15} {cur:>10.4f}")
        print()

    def show_help():
        print(f"""
  {'='*55}
  Commands:
  {'─'*55}
  <joint> <value>    Set joint (e.g., lj_dg_1_2 1.5)
  <joint> <v1> <v2> <v3> <v4>
                     Set finger 4 joints at once
                     (e.g., lj_dg_1 0.0 1.0 0.5 0.3)
  syn <value>        Set all synergy joints (2,3,4) close amount [0~1]
  all <value>        Set all thumb/pinky joints
  reset              Reset all to default
  show               Show current values
  save               Print current pose as Python dict
  help               Show this help
  quit               Exit
  {'='*55}
""")

    def input_thread():
        nonlocal running
        show_help()
        show_joints()
        while running:
            try:
                line = input("  >>> ").strip()
                if not line:
                    continue
                with lock:
                    command_queue.append(line)
            except (EOFError, KeyboardInterrupt):
                running = False
                break

    # Start input thread
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    print(f"\n  {side} hand interactive joint control ready.")
    print(f"  Type 'help' for commands.\n")

    while app_launcher.app.is_running() and running:
        # Process commands
        with lock:
            commands = list(command_queue)
            command_queue.clear()

        for cmd in commands:
            parts = cmd.split()
            if not parts:
                continue

            if parts[0] == "quit":
                running = False
                break

            elif parts[0] == "reset":
                current_pos = default_joint_pos.clone()
                print("  Reset to default.")
                show_joints()

            elif parts[0] == "show":
                show_joints()

            elif parts[0] == "help":
                show_help()

            elif parts[0] == "save":
                print("\n  # Current thumb/pinky pose:")
                for jname in thumb_pinky_joints:
                    idx = joint_idx_map[jname]
                    val = current_pos[0, idx].item()
                    print(f'  "{jname}": {val:.4f},')
                print()

            elif parts[0] == "syn" and len(parts) == 2:
                try:
                    t_val = float(parts[1])
                    t_val = max(0.0, min(1.0, t_val))
                    # Synergy close pose values for fingers 2-4
                    close_vals = {1: 0.0, 2: 0.8, 3: 0.8, 4: 0.4}
                    for jname in synergy_joints:
                        idx = joint_idx_map[jname]
                        jnum = int(jname.split("_")[-1])
                        target = close_vals.get(jnum, 0.0) * t_val
                        current_pos[:, idx] = target
                    print(f"  Synergy set to {t_val:.2f} (0=open, 1=close)")
                except ValueError:
                    print("  Usage: syn <0.0~1.0>")

            elif parts[0] == "all" and len(parts) == 2:
                try:
                    val = float(parts[1])
                    for jname in thumb_pinky_joints:
                        idx = joint_idx_map[jname]
                        current_pos[:, idx] = val
                    print(f"  All thumb/pinky joints set to {val:.4f}")
                except ValueError:
                    print("  Usage: all <value>")

            elif parts[0] in joint_idx_map and len(parts) == 2:
                # Single joint set: lj_dg_1_2 1.5
                try:
                    val = float(parts[1])
                    idx = joint_idx_map[parts[0]]
                    current_pos[:, idx] = val
                    print(f"  {parts[0]} = {val:.4f}")
                except ValueError:
                    print(f"  Usage: {parts[0]} <value>")

            elif len(parts) == 5:
                # Finger shorthand: lj_dg_1 v1 v2 v3 v4
                prefix = parts[0]
                try:
                    vals = [float(v) for v in parts[1:5]]
                    found = False
                    for j in range(1, 5):
                        jname = f"{prefix}_{j}"
                        if jname in joint_idx_map:
                            idx = joint_idx_map[jname]
                            current_pos[:, idx] = vals[j - 1]
                            found = True
                    if found:
                        print(f"  {prefix}_[1-4] = {vals}")
                    else:
                        print(f"  Unknown joint prefix: {prefix}")
                except ValueError:
                    print(f"  Usage: {prefix} <v1> <v2> <v3> <v4>")

            else:
                print(f"  Unknown command: {cmd}")
                print(f"  Type 'help' for available commands.")

        # Write current joint positions to sim
        robot.write_joint_state_to_sim(current_pos, default_joint_vel)
        robot.set_joint_position_target(current_pos)
        core_env.sim.step()
        core_env.scene.update(dt=physics_dt)

    env.close()


if __name__ == "__main__":
    main()
