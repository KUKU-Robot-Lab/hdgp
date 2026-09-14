"""Head-camera scene, settling and keypoint snapshot shared by snapshot.py and observe.py (design spec §5, auto-loop §4).

Moved out of scripts/iker/snapshot.py unchanged in behaviour. Import after the Isaac app launcher has started with
cameras enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch
from PIL import Image

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg, SimulationContext

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.projection import CameraPose, camera_pose_from_link
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix
from openarm.agnostic.modules.iker.scene_image import AnnotatedSnapshot, PointGroup, annotate_snapshot
from openarm.sensors.head_camera import head_camera_cfg, load_spec, urdf_head_angles

from . import layout, robot, scene_cfg

PHYSICS_DT = 1.0 / 120.0
ENV_PRIM = "/World/envs/env_0"
SETTLE_WINDOW_STEPS = 30
MAX_SETTLE_MOTION_M = 0.005
MAX_HEAD_ERROR_DEG = 0.5
MAX_RACK_DEPTH_ERROR_M = 0.005


@dataclass(frozen=True)
class HeadScene:
    sim: SimulationContext
    arm: Articulation
    shoes: Mapping[str, RigidObject]
    camera: Camera
    spec: object


@dataclass(frozen=True)
class HeadAim:
    pan_id: int
    tilt_id: int
    pan_deg: float
    tilt_deg: float


@dataclass(frozen=True)
class Snapshot:
    rgb: np.ndarray
    annotated: AnnotatedSnapshot
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]]
    camera_pose: CameraPose
    intrinsic: np.ndarray
    checks: Mapping[str, object]  # head_error_deg, settle_motion_m, rack_depth_error_max_m
    failures: Mapping[str, str]  # check name (margin, head, settle, rack_depth, support) -> message


def build_scene(poses: Mapping[str, tuple], device: str) -> HeadScene:
    """Table, rack, light, the robot and the shoes at ``poses`` under env_0, with the head camera."""
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, render_interval=1, device=device))
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
    return HeadScene(sim, arm, shoes, camera, spec)


def head_target(arm: Articulation, joint_target: torch.Tensor) -> tuple[torch.Tensor, HeadAim]:
    """``joint_target`` (1, J) with the head joints at the snapshot angles."""
    aim = HeadAim(
        arm.find_joints(robot.HEAD_PAN_JOINT)[0][0],
        arm.find_joints(robot.HEAD_TILT_JOINT)[0][0],
        *urdf_head_angles(layout.HEAD_PAN_ENCODER_DEG, layout.HEAD_TILT_ENCODER_DEG),
    )
    target = joint_target.clone()
    target[:, aim.pan_id], target[:, aim.tilt_id] = math.radians(aim.pan_deg), math.radians(aim.tilt_deg)
    return target, aim


def settle(scene: HeadScene, target: torch.Tensor, settle_steps: int, render_steps: int) -> dict[str, np.ndarray]:
    """Hold ``target`` (re-commanded every step) with arm gravity compensation; returns each shoe's position trace."""
    gravity_ids, _ = scene.arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    traces = {name: [] for name in scene.shoes}
    for step in range(settle_steps):
        scene.arm.set_joint_position_target(target)
        tau = scene.arm.root_physx_view.get_gravity_compensation_forces()
        scene.arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
        scene.arm.write_data_to_sim()
        render = step >= settle_steps - render_steps
        scene.sim.step(render=render)
        scene.arm.update(PHYSICS_DT)
        for name, shoe in scene.shoes.items():
            shoe.update(PHYSICS_DT)
            traces[name].append(shoe.data.root_pos_w[0].cpu().numpy().copy())
        if render:
            scene.camera.update(PHYSICS_DT)
    return {name: np.asarray(trace) for name, trace in traces.items()}


def capture(scene: HeadScene, traces: Mapping[str, np.ndarray], meta: Mapping, aim: HeadAim) -> Snapshot:
    """Render, annotate the keypoints and run the snapshot checks."""
    arm = scene.arm
    link = arm.find_bodies(scene.spec.link)[0][0]
    camera_pose = camera_pose_from_link(
        arm.data.body_pos_w[0, link].cpu().numpy(), arm.data.body_quat_w[0, link].cpu().numpy(), scene.spec.pos, scene.spec.quat_wxyz
    )
    intrinsic = scene.camera.data.intrinsic_matrices[0].cpu().numpy()
    rgb = scene.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    depth = scene.camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
    poses = {name: (s.data.root_pos_w[0].cpu().numpy(), s.data.root_quat_w[0].cpu().numpy()) for name, s in scene.shoes.items()}

    groups = [
        PointGroup(name, False, pos + np.asarray(meta["objects"][name]["keypoint_offsets"]) @ quat_wxyz_to_matrix(quat).T)
        for name, (pos, quat) in poses.items()
    ]
    groups.append(PointGroup(layout.RACK, True, layout.rack_keypoints()))
    anchor = np.concatenate([group.points_w for group in groups]).mean(axis=0)
    annotated = annotate_snapshot(rgb, depth, camera_pose, intrinsic, groups, anchor, layout.OVERLAP_MIN_PX)

    head_error = max(
        abs(math.degrees(arm.data.joint_pos[0, aim.pan_id].item()) - aim.pan_deg),
        abs(math.degrees(arm.data.joint_pos[0, aim.tilt_id].item()) - aim.tilt_deg),
    )
    settle_motion = max(
        float(np.linalg.norm(trace[-SETTLE_WINDOW_STEPS:] - trace[-1], axis=1).max()) for trace in traces.values()
    )
    rack_errors = [abs(r.render_depth - r.depth) for r in annotated.records if r.is_static and r.visible and r.kept]
    rack_depth_error = max(rack_errors) if rack_errors else None
    failures = {}
    if not annotated.min_margin_px >= layout.MIN_BORDER_MARGIN_PX:
        failures["margin"] = f"keypoint border margin {annotated.min_margin_px:.1f} px < {layout.MIN_BORDER_MARGIN_PX}"
    if not head_error <= MAX_HEAD_ERROR_DEG:
        failures["head"] = f"head angle error {head_error:.2f} deg > {MAX_HEAD_ERROR_DEG}"
    if not settle_motion <= MAX_SETTLE_MOTION_M:
        failures["settle"] = f"shoes still moving {settle_motion * 1000:.1f} mm over the last {SETTLE_WINDOW_STEPS} steps"
    if rack_depth_error is None or not rack_depth_error <= MAX_RACK_DEPTH_ERROR_M:
        failures["rack_depth"] = f"rack keypoint depth vs render depth {rack_depth_error} m > {MAX_RACK_DEPTH_ERROR_M}"
    try:
        run_files.movable_objects(
            {"keypoints": [vars(r) for r in annotated.records], "objects": {n: {"position": p.tolist()} for n, (p, _) in poses.items()}},
            meta,
            layout.supports(),
        )
    except ValueError as exc:
        failures["support"] = str(exc)
    checks = {"head_error_deg": head_error, "settle_motion_m": settle_motion, "rack_depth_error_max_m": rack_depth_error}
    return Snapshot(rgb, annotated, poses, camera_pose, intrinsic, checks, failures)


def write(snapshot: Snapshot, out_dir: Path, scene_config, ignore: tuple[str, ...] = ()) -> bool:
    """snapshot_raw.png, snapshot.png and keypoints.json; True when every check outside ``ignore`` passed."""
    failures = [message for name, message in snapshot.failures.items() if name not in ignore]
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot.rgb).save(out_dir / "snapshot_raw.png")
    snapshot.annotated.image.save(out_dir / "snapshot.png")
    head = {"pan_encoder_deg": layout.HEAD_PAN_ENCODER_DEG, "tilt_encoder_deg": layout.HEAD_TILT_ENCODER_DEG}
    doc = run_files.keypoints_document(
        scene_config=vars(scene_config), camera=snapshot.camera_pose, intrinsic=snapshot.intrinsic, snapshot=snapshot.annotated,
        image_size=(snapshot.rgb.shape[1], snapshot.rgb.shape[0]), object_poses=snapshot.poses, head=head,
        checks={"passed": not failures, "failures": failures, **snapshot.checks},
    )
    run_files.write_json(out_dir / "keypoints.json", doc)
    return not failures
