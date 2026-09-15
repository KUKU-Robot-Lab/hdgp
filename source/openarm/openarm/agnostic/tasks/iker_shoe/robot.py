"""Robot articulation for the IKER shoe task, assembled from the hdgp robot profile (design spec §3, §7).

Every robot value comes from ``modules/robot_profiles.py`` and ``modules/vendor_gains.py``; grasp_s2r
is not imported. Conventions kept from the validated hdgp tracks: gravity on and compensated by the
caller on both arms (``GRAVITY_COMPENSATION_JOINTS``), robot solver iterations 32/1 as in the vendor
DG-5F Isaac USD, and DG-5F hand gains proportional to joint inertia (the vendor Isaac rule).
"""

from __future__ import annotations

import re
from typing import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from openarm.agnostic.modules import robot_profiles, vendor_gains

from . import grasp_stage, layout

ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 1
GRAVITY_COMPENSATION_JOINTS = "[rl]_aj_[1-7]"
HEAD_PAN_JOINT = "head_j_pan"
HEAD_TILT_JOINT = "head_j_tilt"


def profile() -> robot_profiles.RobotProfile:
    return robot_profiles.PROFILES[layout.PROFILE_NAME]


def _actuators(prof: robot_profiles.RobotProfile, asset_dir: str) -> dict[str, ImplicitActuatorCfg]:
    inertia = vendor_gains.load_joint_inertia(asset_dir)
    actuators = {}
    for name, spec in prof.actuator_specs.items():
        kwargs = dict(spec)
        if name.endswith("hand"):
            patterns = [re.compile(pattern) for pattern in kwargs["joint_names_expr"]]
            joints = tuple(joint for joint in sorted(inertia) if any(p.fullmatch(joint) for p in patterns))
            if not joints:
                raise RuntimeError(f"{asset_dir}: no joint in the inertia table matches actuator {name!r}")
            kwargs["stiffness"], kwargs["damping"] = vendor_gains.hand_gains_by_joint(asset_dir, joints)
        actuators[name] = ImplicitActuatorCfg(**kwargs)
    return actuators


def robot_cfg() -> ArticulationCfg:
    prof = profile()
    usd = layout.ASSETS_DIR / prof.usd_relpath
    if not usd.is_file():
        raise FileNotFoundError(f"robot USD for profile {prof.name!r} not found: {usd}")
    return ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=dict(prof.init_joint_pos)
        ),
        actuators=_actuators(prof, str(usd.parent)),
        soft_joint_pos_limit_factor=1.0,
    )


def apply_hand_backstop(articulation, roles: Sequence[str]) -> dict[str, list[float]]:
    """Write the learned-grasp spec §16 backstop (``grasp_stage.backstop_limits``) into the simulator and return it as boot
    metadata, ``{joint: [lo, hi]}`` rounded to 4 decimals. A runtime-written limit holds against the joint's own drive
    (probe 2026-09-15: thumb_3 target 0.8 rad past the open pose, measured -0.0003 rad)."""
    prof = profile()
    names = list(articulation.data.joint_names)
    hand = [names.index(name) for name in prof.hand_joint_names]
    hard = articulation.data.joint_pos_limits[0, hand]
    limits = grasp_stage.backstop_limits(prof.hand_joint_names, hard[:, 0], hard[:, 1], prof.hand_open_pose, prof.hand_grip_pose, roles)
    for name, bounds in limits.items():
        value = torch.tensor(bounds, device=articulation.device).repeat(articulation.num_instances, 1, 1)
        articulation.write_joint_position_limit_to_sim(value, joint_ids=[names.index(name)])
    return {name: [round(lo, 4), round(hi, 4)] for name, (lo, hi) in limits.items()}
