"""Build the grasp bank of one IKER configuration (design spec §8).

Every round, all envs approach a sampled pre-grasp above the shoe (the shoe parked out of the way), the shoe
is placed at the configuration's start pose, the synergy hand closes, the state is recorded, and the palm
lifts 0.10 m and holds 1 s. Recorded states that held are replayed from a fresh reset and lifted again;
only states that hold twice enter the bank.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/make_grasp_bank.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the IKER grasp bank of one configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--min-entries", type=int, default=64)
parser.add_argument("--max-rounds", type=int, default=12)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("BANK FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
APPROACH_STEPS, DESCEND_STEPS, SETTLE_STEPS, CLOSE_STEPS = 360, 240, 60, 240
LIFT_STEPS, HOLD_STEPS, LIFT_HEIGHT_M = 120, 120, 0.10
PARK_POS = (0.42, 0.40)  # table corner away from the grasp area
FRICTION = 1.0  # same contact material as the training environment


def build(config, meta):
    poses = layout.shoe_start_poses(config, meta)
    move_pos, move_quat = poses[layout.MOVING_SHOE]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        table = scene_cfg.table_cfg()
        rack = scene_cfg.rack_cfg()
        light = scene_cfg.light_cfg()
        robot = robot.robot_cfg()
        shoe = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (PARK_POS[0], PARK_POS[1], move_pos[2]), move_quat)

    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
            device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        )
    )
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=True))
    sim.reset()
    return sim, scene, poses


def main() -> int:
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    sim, scene, poses = build(config, meta)
    arm, shoe, dev, n = scene["robot"], scene["shoe"], sim.device, args.num_envs
    prof = robot.profile()
    origins = scene.env_origins
    arm_ids, _ = arm.find_joints(prof.arm_joint_regex)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    hand_ids = [arm.data.joint_names.index(name) for name in prof.hand_joint_names]
    palm = arm.find_bodies(prof.palm_body)[0][0]
    lower = arm.data.soft_joint_pos_limits[0, hand_ids, 0]
    upper = arm.data.soft_joint_pos_limits[0, hand_ids, 1]
    open_pose = torch.tensor(prof.hand_open_pose, device=dev)
    grip_pose = torch.tensor(prof.hand_grip_pose, device=dev)
    thumb3 = list(prof.hand_joint_names).index(gb.hand_joint_name(prof.hand_joint_names, "thumb", 3))
    side_sign = gb.side_sign(prof.hand_joint_names)  # +1 right arm, -1 left arm (y-mirrored pre-grasp)
    finger_ids = gb.finger_index(prof.hand_joint_names).to(dev)
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    home = arm.data.default_joint_pos.clone()
    shoe_obj = meta["objects"][layout.MOVING_SHOE]
    top_z = layout.TABLE_TOP_Z + 2.0 * float(shoe_obj["rest_height"])
    half_width = abs(float(shoe_obj["keypoint_offsets"][2][0]))
    start = torch.tensor([*poses[layout.MOVING_SHOE][0], *poses[layout.MOVING_SHOE][1]], dtype=torch.float32, device=dev)
    park = start.clone()
    park[:2] = torch.tensor(PARK_POS, device=dev)
    generator = torch.Generator().manual_seed(args.seed)
    zero_vel = torch.zeros(n, 6, device=dev)

    def write_shoe(local_pose):
        pose = local_pose.clone()
        pose[:, :3] += origins
        shoe.write_root_pose_to_sim(pose)
        shoe.write_root_velocity_to_sim(zero_vel)

    def run(steps, palm_goal, palm_quat, hand_target, parked=False):
        for _ in range(steps):
            if parked:
                write_shoe(park.expand(n, 7))
            ee_pos, ee_quat = arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([palm_goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jac = arm.root_physx_view.get_jacobians()[:, palm - 1, :, arm_ids]
            target = arm.data.joint_pos_target.clone()
            target[:, arm_ids] = ik.compute(ee_pos, ee_quat, jac, arm.data.joint_pos[:, arm_ids])
            target[:, hand_ids] = hand_target() if callable(hand_target) else hand_target
            arm.set_joint_position_target(target)
            tau = arm.root_physx_view.get_gravity_compensation_forces()
            arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

    def lift_test(palm_goal, palm_quat, hand_target):
        z0 = shoe.data.root_pos_w[:, 2].clone()
        for k in range(1, LIFT_STEPS + 1):
            run(1, palm_goal + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M * k / LIFT_STEPS], device=dev), palm_quat, hand_target)
        rel_lift = shoe.data.root_pos_w - arm.data.body_pos_w[:, palm]
        run(HOLD_STEPS, palm_goal + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M], device=dev), palm_quat, hand_target)
        rel_hold = shoe.data.root_pos_w - arm.data.body_pos_w[:, palm]
        return gb.lift_held(z0, shoe.data.root_pos_w[:, 2], rel_lift, rel_hold)

    kept = {key: [] for key in ("joint_pos", "joint_target", "shoe_pose", "palm_pose")}
    stats = []
    t0 = time.time()
    for round_index in range(args.max_rounds):
        arm.write_joint_state_to_sim(home, torch.zeros_like(home))
        arm.set_joint_position_target(home)
        ik.reset()
        pre = gb.sample_pregrasp(n, generator, dev)
        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg, side_sign))
        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z, side_sign)
        start_hand = open_pose.expand(n, -1).clone()
        start_hand[:, thumb3] = side_sign * pre.thumb3
        start_hand = torch.max(torch.min(start_hand, upper), lower)
        above = goal + torch.tensor([0.0, 0.0, 0.12], device=dev)
        run(APPROACH_STEPS, above, palm_quat, start_hand, parked=True)
        run(DESCEND_STEPS, goal, palm_quat, start_hand, parked=True)
        write_shoe(start.expand(n, 7))
        run(SETTLE_STEPS, goal, palm_quat, start_hand)
        # Diagnostic: fingers already bent back before closing means the teleport overlap is the cause.
        settle_bad = ~gb.grasp_acceptable(arm.data.joint_pos[:, hand_ids], start_hand, grip_pose, lower, upper)
        state = gb.FingerStopState.start(start_hand)

        def closing():
            nonlocal state
            state = gb.finger_stop_step(
                state, arm.data.joint_pos[:, hand_ids], start_hand, grip_pose, lower, upper, finger_ids
            )
            return state.target

        run(CLOSE_STEPS, goal, palm_quat, closing)
        record = {
            "joint_pos": arm.data.joint_pos.clone(),
            "joint_target": arm.data.joint_pos_target.clone(),
            "shoe_pose": torch.cat([shoe.data.root_pos_w - origins, shoe.data.root_quat_w], dim=-1),
            "palm_pose": torch.cat([arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]], dim=-1),
        }
        hand_ok = gb.grasp_acceptable(record["joint_pos"][:, hand_ids], start_hand, grip_pose, lower, upper)
        held = lift_test(goal, palm_quat, state.target)

        # Replay: restore the recorded state as an environment reset would, then lift again.
        arm.write_joint_state_to_sim(record["joint_pos"], torch.zeros_like(record["joint_pos"]))
        arm.set_joint_position_target(record["joint_target"])
        write_shoe(record["shoe_pose"])
        ik.reset()
        run(SETTLE_STEPS, record["palm_pose"][:, :3], record["palm_pose"][:, 3:], record["joint_target"][:, hand_ids])
        replay_held = lift_test(record["palm_pose"][:, :3], record["palm_pose"][:, 3:], record["joint_target"][:, hand_ids])
        ok = held & replay_held & hand_ok
        for key in kept:
            kept[key].append(record[key][ok])
        total = sum(int(t.shape[0]) for t in kept["joint_pos"])
        stats.append(
            {
                "round": round_index,
                "held": int(held.sum()),
                "replay_held": int(replay_held.sum()),
                "hand_ok": int(hand_ok.sum()),
                "settle_bad": int(settle_bad.sum()),
                "kept": int(ok.sum()),
                "total": total,
            }
        )
        print(
            f"BANK round {round_index} held {int(held.sum())}/{n} replay {int(replay_held.sum())}/{n} "
            f"hand_ok {int(hand_ok.sum())}/{n} settle_bad {int(settle_bad.sum())}/{n} kept {int(ok.sum())} "
            f"total {total} ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if total >= args.min_entries:
            break

    entries = {key: torch.cat(parts) for key, parts in kept.items()}
    total = int(entries["joint_pos"].shape[0])
    metadata = {
        "config_index": config.index,
        "scene_config": vars(config),
        "robot_usd": str(robot.profile().usd_relpath),
        "physics_dt": PHYSICS_DT,
        "friction": FRICTION,
        "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
        "gains": gb.gains_metadata(arm.data.joint_names, arm.data.joint_stiffness[0], arm.data.joint_damping[0]),
        "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rounds": stats,
        "side_sign": side_sign,
        "closing": {
            "rule": "finger_stop",
            "trigger_backbend_rad": gb.FINGER_TRIGGER_BACKBEND_RAD,
            "trigger_error_rad": gb.BLOCKED_ERR_RAD,
            "squeeze_rad": gb.FINGER_SQUEEZE_RAD,
            "close_rate_per_step": gb.CLOSE_RATE_PER_STEP,
            "max_backbend_rad": gb.MAX_BACKBEND_RAD,
        },
    }
    out = layout.RUNS_DIR / f"config_{config.index:02d}" / "grasp_bank.json"
    run_files.write_json(out, gb.bank_document(entries, arm.data.joint_names, metadata))
    passed = total >= args.min_entries
    print(f"BANK config {config.index:02d} entries {total} (min {args.min_entries}) passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


os._exit(main())
