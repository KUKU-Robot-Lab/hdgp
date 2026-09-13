"""Head-camera snapshot of one IKER configuration (design spec §5).

Spawns the scene, holds the head at the snapshot angles while the shoes settle, renders RGB and
depth, and writes iker_runs/shoe_place/config_XX/{snapshot_raw.png, snapshot.png, keypoints.json}.
The files are written even when a check fails (``checks.passed`` is false) and the exit code is 1.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/snapshot.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of one IKER configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    # Isaac Sim can hang on shutdown after an exception; report and leave immediately.
    traceback.print_exception(exc_type, exc, tb)
    print("SNAPSHOT FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.modules.iker.projection import camera_pose_from_link  # noqa: E402
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix  # noqa: E402
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402
from openarm.sensors.head_camera import head_camera_cfg, load_spec, urdf_head_angles  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
ENV_PRIM = "/World/envs/env_0"
SETTLE_WINDOW_STEPS = 30
MAX_SETTLE_MOTION_M = 0.005
MAX_HEAD_ERROR_DEG = 0.5
MAX_RACK_DEPTH_ERROR_M = 0.005


def build_scene(poses):
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, render_interval=1, device=args.device))
    prim_utils.create_prim(ENV_PRIM, "Xform")
    for cfg, prim in ((scene_cfg.table_cfg(), f"{ENV_PRIM}/Table"), (scene_cfg.rack_cfg(), f"{ENV_PRIM}/Rack")):
        cfg.spawn.func(prim, cfg.spawn, translation=cfg.init_state.pos)
    light = scene_cfg.light_cfg()
    light.spawn.func(light.prim_path, light.spawn)
    arm = Articulation(robot.robot_cfg().replace(prim_path=f"{ENV_PRIM}/Robot"))
    shoes = {
        name: RigidObject(scene_cfg.shoe_cfg(name, pos, quat).replace(prim_path=f"{ENV_PRIM}/{name}"))
        for name, (pos, quat) in poses.items()
    }
    spec = load_spec()
    camera = Camera(
        head_camera_cfg(spec, data_types=("rgb", "distance_to_image_plane")).replace(
            prim_path=f"{ENV_PRIM}/Robot/{spec.link}/head_cam_real"
        )
    )
    sim.reset()
    if not camera.is_initialized:  # the camera is created after the robot; see openarm.sensors.head_camera
        camera._initialize_impl()
        camera._is_initialized = True
    return sim, arm, shoes, camera, spec


def settle(sim, arm, shoes, camera, target, gravity_ids):
    """Hold the head (re-commanded every step) with arm gravity compensation; return shoe position traces."""
    traces = {name: [] for name in shoes}
    for step in range(args.settle_steps):
        arm.set_joint_position_target(target)
        tau = arm.root_physx_view.get_gravity_compensation_forces()
        arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
        arm.write_data_to_sim()
        render = step >= args.settle_steps - args.render_steps
        sim.step(render=render)
        arm.update(PHYSICS_DT)
        for name, shoe in shoes.items():
            shoe.update(PHYSICS_DT)
            traces[name].append(shoe.data.root_pos_w[0].cpu().numpy().copy())
        if render:
            camera.update(PHYSICS_DT)
    return {name: np.asarray(trace) for name, trace in traces.items()}


def main() -> int:
    if not args.settle_steps > max(SETTLE_WINDOW_STEPS, args.render_steps):
        raise ValueError("--settle-steps must exceed both the settle window and --render-steps")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    sim, arm, shoes, camera, spec = build_scene(layout.shoe_start_poses(config, meta))

    home = arm.data.default_joint_pos.clone()
    arm.write_joint_state_to_sim(home, torch.zeros_like(home))
    pan_id = arm.find_joints(robot.HEAD_PAN_JOINT)[0][0]
    tilt_id = arm.find_joints(robot.HEAD_TILT_JOINT)[0][0]
    pan_deg, tilt_deg = urdf_head_angles(layout.HEAD_PAN_ENCODER_DEG, layout.HEAD_TILT_ENCODER_DEG)
    target = home.clone()
    target[:, pan_id], target[:, tilt_id] = math.radians(pan_deg), math.radians(tilt_deg)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    traces = settle(sim, arm, shoes, camera, target, gravity_ids)

    link = arm.find_bodies(spec.link)[0][0]
    camera_pose = camera_pose_from_link(
        arm.data.body_pos_w[0, link].cpu().numpy(), arm.data.body_quat_w[0, link].cpu().numpy(), spec.pos, spec.quat_wxyz
    )
    intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()
    rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
    poses = {name: (s.data.root_pos_w[0].cpu().numpy(), s.data.root_quat_w[0].cpu().numpy()) for name, s in shoes.items()}

    groups = [
        PointGroup(name, False, pos + np.asarray(meta["objects"][name]["keypoint_offsets"]) @ quat_wxyz_to_matrix(quat).T)
        for name, (pos, quat) in poses.items()
    ]
    groups.append(PointGroup(layout.RACK, True, layout.rack_keypoints()))
    anchor = np.concatenate([group.points_w for group in groups]).mean(axis=0)
    snapshot = annotate_snapshot(rgb, depth, camera_pose, intrinsic, groups, anchor, layout.OVERLAP_MIN_PX)

    head_error = max(
        abs(math.degrees(arm.data.joint_pos[0, pan_id].item()) - pan_deg),
        abs(math.degrees(arm.data.joint_pos[0, tilt_id].item()) - tilt_deg),
    )
    settle_motion = max(
        float(np.linalg.norm(trace[-SETTLE_WINDOW_STEPS:] - trace[-1], axis=1).max()) for trace in traces.values()
    )
    rack_errors = [abs(r.render_depth - r.depth) for r in snapshot.records if r.is_static and r.visible and r.kept]
    rack_depth_error = max(rack_errors) if rack_errors else None
    failures = []
    if not snapshot.min_margin_px >= layout.MIN_BORDER_MARGIN_PX:
        failures.append(f"keypoint border margin {snapshot.min_margin_px:.1f} px < {layout.MIN_BORDER_MARGIN_PX}")
    if not head_error <= MAX_HEAD_ERROR_DEG:
        failures.append(f"head angle error {head_error:.2f} deg > {MAX_HEAD_ERROR_DEG}")
    if not settle_motion <= MAX_SETTLE_MOTION_M:
        failures.append(f"shoes still moving {settle_motion * 1000:.1f} mm over the last {SETTLE_WINDOW_STEPS} steps")
    if rack_depth_error is None or not rack_depth_error <= MAX_RACK_DEPTH_ERROR_M:
        failures.append(f"rack keypoint depth vs render depth {rack_depth_error} m > {MAX_RACK_DEPTH_ERROR_M}")
    try:
        run_files.movable_objects(
            {"keypoints": [vars(r) for r in snapshot.records], "objects": {n: {"position": p.tolist()} for n, (p, _) in poses.items()}},
            meta,
            layout.supports(),
        )
    except ValueError as exc:
        failures.append(str(exc))

    run_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(run_dir / "snapshot_raw.png")
    snapshot.image.save(run_dir / "snapshot.png")
    checks = {
        "passed": not failures,
        "failures": failures,
        "head_error_deg": head_error,
        "settle_motion_m": settle_motion,
        "rack_depth_error_max_m": rack_depth_error,
    }
    head = {"pan_encoder_deg": layout.HEAD_PAN_ENCODER_DEG, "tilt_encoder_deg": layout.HEAD_TILT_ENCODER_DEG}
    doc = run_files.keypoints_document(
        scene_config=vars(config), camera=camera_pose, intrinsic=intrinsic, snapshot=snapshot,
        image_size=(rgb.shape[1], rgb.shape[0]), object_poses=poses, head=head, checks=checks,
    )
    run_files.write_json(run_dir / "keypoints.json", doc)
    print(
        f"SNAPSHOT config {config.index:02d} derotation {snapshot.derotation_deg:+.1f} deg margin {snapshot.min_margin_px:+.1f} px "
        f"gap {snapshot.min_gap_px} px head_err {head_error:.2f} deg settle {settle_motion * 1000:.2f} mm "
        f"rack_depth_err {rack_depth_error} m passed {not failures} -> {run_dir}",
        flush=True,
    )
    for failure in failures:
        print(f"SNAPSHOT CHECK FAILED: {failure}", flush=True)
    return 0 if not failures else 1


os._exit(main())
