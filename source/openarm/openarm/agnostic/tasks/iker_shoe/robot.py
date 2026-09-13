"""Robot articulation for the IKER shoe task, assembled from the hdgp robot profile (design spec §3, §7).

Every robot value comes from ``modules/robot_profiles.py`` and ``modules/vendor_gains.py``; grasp_s2r
is not imported. Conventions kept from the validated hdgp tracks: gravity on and compensated by the
caller on both arms (``GRAVITY_COMPENSATION_JOINTS``), robot solver iterations 32/1 as in the vendor
DG-5F Isaac USD, and DG-5F hand gains proportional to joint inertia (the vendor Isaac rule).
"""

from __future__ import annotations

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from openarm.agnostic.modules import robot_profiles, vendor_gains

from . import layout

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
