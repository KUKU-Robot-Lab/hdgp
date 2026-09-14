"""Build the pre-grasp arm pose bank of one IKER configuration for the stage-1 grasp policy (design 2026-09-14 §4).

Every round, all envs drive the palm by IK from home to a sampled pose 8-12 cm above the shoe top (the K1 pre-grasp
geometry lifted, wrist near level, mirrored for the left arm) with the hand open and the shoe parked out of the way;
then the shoe is placed at its settled snapshot pose and the scene settles. States whose palm converged (position
< 5 mm, orientation < 3 deg), whose finger links all stay >= 1 cm from the shoe surface and whose shoe did not move
are kept.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/make_pregrasp_bank.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the IKER stage-1 pre-grasp pose bank of one configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--min-entries", type=int, default=256)
parser.add_argument("--max-rounds", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("PREGRASP FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_matrix  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
APPROACH_STEPS, SETTLE_STEPS = 480, 60
# design §4: the palm starts 8-12 cm above the shoe top with the hand open and the wrist near level. The K1 tilt
# (-40..-15 deg) points the open fingers into the shoe. With the palm facing down the thumb hangs below the palm,
# so "no contact" is the distance from every finger link to the shoe's surface points, not height above the top.
START_HEIGHT_ABOVE_TOP_RANGE = (0.08, 0.12)
START_TILT_DEG_RANGE = (-10.0, 0.0)
MAX_PALM_ERROR_M = 0.005
MAX_PALM_ANGLE_DEG = 3.0
MIN_FINGER_SURFACE_GAP_M = 0.01
MAX_SHOE_SHIFT_M = 0.002  # the shoe starts at its settled snapshot pose, so it must not move
PARK_POS = (0.42, 0.40)
FRICTION = 1.0  # same contact material as the training environment


def build(start_pos, start_quat):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        table = scene_cfg.table_cfg()
        rack = scene_cfg.rack_cfg()
        light = scene_cfg.light_cfg()
        robot = robot.robot_cfg()
        shoe = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (PARK_POS[0], PARK_POS[1], start_pos[2]), start_quat)

    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
            device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        )
    )
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=True))
    sim.reset()
    return sim, scene


def quat_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(2.0 * torch.acos((a * b).sum(dim=-1).abs().clamp(max=1.0)))


def main() -> int:
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    run_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    settled = run_files.read_json(run_dir / "keypoints.json")["objects"][layout.MOVING_SHOE]
    sim, scene = build(settled["position"], settled["quat_wxyz"])
    arm, shoe, dev, n = scene["robot"], scene["shoe"], sim.device, args.num_envs
    prof = robot.profile()
    origins = scene.env_origins
    arm_ids, _ = arm.find_joints(prof.arm_joint_regex)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    hand_ids = [arm.data.joint_names.index(name) for name in prof.hand_joint_names]
    body_names = list(arm.data.body_names)
    finger_links = [body_names.index(b) for f in gb.FINGERS for b in prof.finger_sensor_bodies[f]]
    palm = arm.find_bodies(prof.palm_body)[0][0]
    side_sign = gb.side_sign(prof.hand_joint_names)
    lower = arm.data.soft_joint_pos_limits[0, hand_ids, 0]
    upper = arm.data.soft_joint_pos_limits[0, hand_ids, 1]
    open_hand = torch.max(torch.min(torch.tensor(prof.hand_open_pose, device=dev), upper), lower)
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    home = arm.data.default_joint_pos.clone()
    shoe_obj = meta["objects"][layout.MOVING_SHOE]
    top_z = layout.TABLE_TOP_Z + 2.0 * float(shoe_obj["rest_height"])
    half_width = abs(float(shoe_obj["keypoint_offsets"][2][0]))
    start = torch.tensor([*settled["position"], *settled["quat_wxyz"]], dtype=torch.float32, device=dev)
    surface = gs.surface_subsample(shoe_obj["hull_local"]).to(dev)
    start_surface = quat_apply(start[3:].expand(surface.shape[0], 4), surface) + start[:3]
    park = start.clone()
    park[:2] = torch.tensor(PARK_POS, device=dev)
    generator = torch.Generator().manual_seed(args.seed)
    zero_vel = torch.zeros(n, 6, device=dev)

    def write_shoe(local_pose):
        pose = local_pose.clone()
        pose[:, :3] += origins
        shoe.write_root_pose_to_sim(pose)
        shoe.write_root_velocity_to_sim(zero_vel)

    def run(steps, palm_goal, palm_quat, parked):
        for _ in range(steps):
            if parked:
                write_shoe(park.expand(n, 7))
            ee_pos, ee_quat = arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([palm_goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jac = arm.root_physx_view.get_jacobians()[:, palm - 1, :, arm_ids]
            target = arm.data.joint_pos_target.clone()
            target[:, arm_ids] = ik.compute(ee_pos, ee_quat, jac, arm.data.joint_pos[:, arm_ids])
            target[:, hand_ids] = open_hand
            arm.set_joint_position_target(target)
            tau = arm.root_physx_view.get_gravity_compensation_forces()
            arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, generator=generator).to(dev)

    kept = {key: [] for key in ("joint_pos", "joint_target", "shoe_pose", "palm_pose")}
    stats, t0 = [], time.time()
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=dev)
    for round_index in range(args.max_rounds):
        start_pose = home.clone()
        start_pose[:, hand_ids] = open_hand
        arm.write_joint_state_to_sim(start_pose, torch.zeros_like(start_pose))
        arm.set_joint_position_target(start_pose)
        ik.reset()
        pre = dataclasses.replace(
            gb.sample_pregrasp(n, generator, dev),
            height_above_top=uniform(START_HEIGHT_ABOVE_TOP_RANGE),
            tilt_deg=uniform(START_TILT_DEG_RANGE),
        )
        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg, side_sign))
        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z, side_sign)
        run(APPROACH_STEPS, goal, palm_quat, parked=True)
        write_shoe(start.expand(n, 7))
        run(SETTLE_STEPS, goal, palm_quat, parked=False)
        palm_pos = arm.data.body_pos_w[:, palm] - origins
        position_error = (palm_pos - goal).norm(dim=-1)
        angle_error = quat_angle_deg(arm.data.body_quat_w[:, palm], palm_quat)
        links = arm.data.body_pos_w[:, finger_links] - origins[:, None, :]
        finger_gap = gs.nearest_distance(links, start_surface.expand(n, -1, 3)).min(dim=1).values
        shoe_shift = (shoe.data.root_pos_w - origins - start[:3]).norm(dim=-1)
        ok = (position_error < MAX_PALM_ERROR_M) & (angle_error < MAX_PALM_ANGLE_DEG) & (finger_gap >= MIN_FINGER_SURFACE_GAP_M) & (shoe_shift < MAX_SHOE_SHIFT_M)
        record = {
            "joint_pos": arm.data.joint_pos.clone(),
            "joint_target": arm.data.joint_pos_target.clone(),
            "shoe_pose": torch.cat([shoe.data.root_pos_w - origins, shoe.data.root_quat_w], dim=-1),
            "palm_pose": torch.cat([palm_pos, arm.data.body_quat_w[:, palm]], dim=-1),
        }
        for key in kept:
            kept[key].append(record[key][ok])
        total = sum(int(t.shape[0]) for t in kept["joint_pos"])
        stats.append({"round": round_index, "kept": int(ok.sum()), "total": total})
        print(
            f"PREGRASP round {round_index} kept {int(ok.sum())}/{n} total {total} "
            f"pos err mm median {float(position_error.median() * 1000):.1f} angle deg median {float(angle_error.median()):.2f} "
            f"finger gap mm q10/50/90 {[round(float(v) * 1000, 1) for v in torch.quantile(finger_gap, quantiles)]} "
            f"shoe shift mm q90 {float(torch.quantile(shoe_shift, 0.9) * 1000):.2f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if total >= args.min_entries:
            break

    entries = {key: torch.cat(parts) for key, parts in kept.items()}
    total = int(entries["joint_pos"].shape[0])
    metadata = {
        "config_index": config.index,
        "scene_config": vars(config),
        "robot_usd": str(prof.usd_relpath),
        "physics_dt": PHYSICS_DT,
        "friction": FRICTION,
        "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
        "gains": gb.gains_metadata(arm.data.joint_names, arm.data.joint_stiffness[0], arm.data.joint_damping[0]),
        "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rounds": stats,
        "source": "pregrasp",
        "start_height_above_top_m": list(START_HEIGHT_ABOVE_TOP_RANGE),
        "start_tilt_deg": list(START_TILT_DEG_RANGE),
        "side_sign": side_sign,
    }
    out = run_dir / "pregrasp_bank.json"
    if total == 0:
        print(f"PREGRASP config {config.index:02d} entries 0 (min {args.min_entries}) passed False", flush=True)
        return 1
    run_files.write_json(out, gb.bank_document(entries, arm.data.joint_names, metadata))
    passed = total >= args.min_entries
    print(f"PREGRASP config {config.index:02d} entries {total} (min {args.min_entries}) passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


os._exit(main())
