"""Smoke-check the IKER shoe environment before training (design spec §10, simulator checks).

1. boots with the grasp bank and the gated interaction, observations (N, 38) are finite;
2. with zero actions for 5 s the shoe stays in the hand: its position in the palm frame moves < 3 cm (the arm's zero-action sag is reported, not judged — relative IK does not undo it, auto-loop spec §13);
3. random actions for 25 s keep rewards finite and log episode-end metrics;
4. a shoe placed at the interaction's target keypoints (robot moved home, out of the way) counts as a success;
5. the grasping arm reaches the bank's grasp palm pose above the interaction's target slot (the mirrored slot on the
   other side of the other shoe is measured and reported, not required).

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/env_smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER shoe environment.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--interaction", default="", help="interaction file (default: the configuration's interaction_human.json)")
parser.add_argument("--grasp-bank", default="", help="grasp bank file (default: the configuration's grasp_bank.json)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_from_matrix  # noqa: E402

MAX_DROP_FRACTION_ZERO_ACTION = 0.25  # startup DR (friction down to 0.3, mass up to x2) loosens some grasps
MAX_SLIP_M = 0.03
MAX_SLOT_REACH_ERROR_M = 0.02
PARKED_SHOE = (0.42, 0.40, 0.26)
IK_REACH_STEPS = 360


def shoe_in_palm(env) -> tuple[torch.Tensor, torch.Tensor]:
    """((N, 3) shoe position in the palm frame, (N,) env-local palm height)."""
    palm = env._robot.data.body_pose_w[:, env._palm]
    return quat_apply_inverse(palm[:, 3:7], env._shoe.data.root_pos_w - palm[:, :3]), palm[:, 2] - env.scene.env_origins[:, 2]


def quantiles_mm(values: torch.Tensor) -> list[int]:
    return [round(float(v) * 1000) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))]


def reach_slots(env) -> dict[str, float]:
    """Worst palm position error (m) when IK drives every env's bank grasp pose above the target slot and its mirror."""
    n, dev = env.num_envs, env.device
    robot, origins, palm = env._robot, env.scene.env_origins, env._palm
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    entry = torch.arange(n, device=dev) % env._bank.size
    grasp_offset = env._bank.palm_pose[entry, :3] - env._bank.shoe_pose[entry, :3]
    palm_quat = env._bank.palm_pose[entry, 3:]
    parked = torch.zeros(n, 7, device=dev)
    parked[:, :3] = torch.tensor(PARKED_SHOE, device=dev) + origins
    parked[:, 3] = 1.0
    _, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    target_slot = torch.tensor(centre, dtype=torch.float32, device=dev)
    mirrored_slot = target_slot.clone()
    mirrored_slot[1] = 2.0 * env._other_pos[1] - target_slot[1]
    errors = {}
    for name, slot in (("target_slot", target_slot), ("mirrored_slot", mirrored_slot)):
        env.reset()
        ik.reset()
        goal = slot + grasp_offset
        for _ in range(IK_REACH_STEPS):
            env._shoe.write_root_pose_to_sim(parked)
            env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
            ee_pos, ee_quat = robot.data.body_pos_w[:, palm] - origins, robot.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jacobian = robot.root_physx_view.get_jacobians()[:, env._palm_jacobian, :, env._arm_ids]
            target = robot.data.joint_pos_target.clone()
            target[:, env._arm_ids] = ik.compute(ee_pos, ee_quat, jacobian, robot.data.joint_pos[:, env._arm_ids])
            robot.set_joint_position_target(target)
            tau = robot.root_physx_view.get_gravity_compensation_forces()
            robot.set_joint_effort_target(tau[:, env._gravity_ids], joint_ids=env._gravity_ids)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        errors[name] = float((robot.data.body_pos_w[:, palm] - origins - goal).norm(dim=-1).max())
    return errors


def main() -> int:
    cfg = IkerShoeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.interaction_path = args.interaction
    cfg.grasp_bank_path = args.grasp_bank
    env = gym.make("open-sens_l_iker_shoe", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    failures = []

    obs, _ = env.reset()
    policy = obs["policy"]
    print(f"SMOKE obs {tuple(policy.shape)} finite {bool(torch.isfinite(policy).all())} bank {env._bank.size}", flush=True)
    if policy.shape != (n, 38) or not torch.isfinite(policy).all():
        failures.append("observation shape or finiteness")

    env.step(torch.zeros(n, 6, device=dev))  # a pose written at reset is read back only after a physics step
    rel0, palm_z0 = shoe_in_palm(env)
    for _ in range(49):
        env.step(torch.zeros(n, 6, device=dev))
    rel, palm_z = shoe_in_palm(env)
    slip = (rel - rel0).norm(dim=-1)
    lost = float((slip > MAX_SLIP_M).float().mean())
    print(f"SMOKE zero action 5 s: shoe slip in palm mm q10/50/90 {quantiles_mm(slip)}, lost fraction {lost:.2f}; "
          f"palm dz mm q10/50/90 {quantiles_mm(palm_z - palm_z0)} (arm sag, reported only)", flush=True)
    if not lost <= MAX_DROP_FRACTION_ZERO_ACTION:
        failures.append(f"zero-action lost fraction {lost:.2f}")

    rewards, logs = [], []
    for _ in range(250):
        _, reward, terminated, truncated, extras = env.step(2.0 * torch.rand(n, 6, device=dev) - 1.0)
        rewards.append(reward)
        if "log" in extras:
            logs.append(dict(extras["log"]))
    stacked = torch.stack(rewards)
    print(f"SMOKE random 25 s: reward finite {bool(torch.isfinite(stacked).all())} mean {float(stacked.mean()):.3f} min {float(stacked.min()):.1f} max {float(stacked.max()):.1f} logs {len(logs)} last {logs[-1] if logs else None}", flush=True)
    if not torch.isfinite(stacked).all() or not logs:
        failures.append("random-action rewards or episode logs")

    # Move the robot home first: at a bank state the closed hand sits where the thumb would push a shoe placed
    # at the target. Then place every shoe at the target pose (rigid fit of its keypoints).
    env.reset()
    home = env._robot.data.default_joint_pos.clone()
    env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    env._robot.set_joint_position_target(home)
    env._joint_targets[:] = home
    rot, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = torch.tensor(centre, dtype=torch.float32, device=dev) + env.scene.env_origins
    pose[:, 3:] = quat_from_matrix(torch.tensor(rot, dtype=torch.float32, device=dev)).expand(n, 4)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    env.step(torch.zeros(n, 6, device=dev))
    counted = float((env._success_count > 0).float().mean())
    distance = float(env._keypoint_distance.mean())
    print(f"SMOKE target placement: success counted {counted:.2f}, mean keypoint distance {distance * 1000:.1f} mm", flush=True)
    if not counted == 1.0:
        failures.append(f"target placement counted as success in {counted:.2f} of envs")

    errors = reach_slots(env)
    print(f"SMOKE rack slot reach: worst palm error mm {({k: round(v * 1000, 1) for k, v in errors.items()})}", flush=True)
    if not errors["target_slot"] <= MAX_SLOT_REACH_ERROR_M:
        failures.append(f"target slot: palm error {errors['target_slot'] * 1000:.1f} mm")

    for failure in failures:
        print(f"SMOKE CHECK FAILED: {failure}", flush=True)
    print(f"SMOKE passed {not failures}", flush=True)
    return 0 if not failures else 1


os._exit(main())
