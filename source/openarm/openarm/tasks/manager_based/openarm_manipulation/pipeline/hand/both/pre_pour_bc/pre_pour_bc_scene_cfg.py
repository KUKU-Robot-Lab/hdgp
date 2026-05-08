"""InteractiveSceneCfg for pre_pour_bc.

Reuses the same assets as pour_v1 (openarm_tesollo_sensor.usd, cup_big_sdf.usd)
and wires up the 5 per-fingertip ContactSensors that provide real-compatible
force observations.
"""

from __future__ import annotations

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")

# HDF5 pour_v1_a1 demo-0 step-0 joint positions (100 Hz teleop data).
_HDF5_LEFT_ARM_JOINT_POS = {
    "openarm_left_joint1":  0.093,
    "openarm_left_joint2": -0.236,
    "openarm_left_joint3": -0.017,
    "openarm_left_joint4":  0.594,
    "openarm_left_joint5":  0.447,
    "openarm_left_joint6": -0.311,
    "openarm_left_joint7": -1.415,
    "openarm_left_finger_joint1": 0.044,
    "openarm_left_finger_joint2": 0.044,
}


@configclass
class PrePourBCSceneCfg(InteractiveSceneCfg):
    """Scene for bimanual pre-pour BC rollout/evaluation.

    Robot: openarm_tesollo_sensor.usd (activate_contact_sensors=True)
    Objects: source cup (dynamic), target cup (kinematic)
    Sensors: 5 × fingertip ContactSensor (rl_dg_*_tip)
    """

    # ------------------------------------------------------------------ #
    # Ground + lighting                                                    #
    # ------------------------------------------------------------------ #
    ground = sim_utils.GroundPlaneCfg() if False else None  # set via spawn

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ------------------------------------------------------------------ #
    # Table (kinematic — matches pour_v1 table position)                  #
    # ------------------------------------------------------------------ #
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # ------------------------------------------------------------------ #
    # Robot                                                                #
    # ------------------------------------------------------------------ #
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_tesollo_sensor/openarm_tesollo_sensor.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                # Right arm — HDF5 pour_v1_a1 demo-0 step-0
                "openarm_right_joint1": -0.072,
                "openarm_right_joint2":  0.007,
                "openarm_right_joint3": -0.004,
                "openarm_right_joint4":  0.599,
                "openarm_right_joint5": -0.168,
                "openarm_right_joint6":  0.041,
                "openarm_right_joint7":  0.869,
                # Right hand — HDF5 pour_v1_a1 demo-0 step-0.
                "rj_dg_1_1": -0.019, "rj_dg_1_2": -1.578, "rj_dg_1_3": -0.471, "rj_dg_1_4":  0.065,
                "rj_dg_2_1":  0.016, "rj_dg_2_2":  0.040, "rj_dg_2_3":  0.033, "rj_dg_2_4":  0.051,
                "rj_dg_3_1": -0.003, "rj_dg_3_2":  0.026, "rj_dg_3_3":  0.031, "rj_dg_3_4":  0.024,
                "rj_dg_4_1": -0.033, "rj_dg_4_2":  0.072, "rj_dg_4_3":  0.044, "rj_dg_4_4":  0.031,
                "rj_dg_5_1":  0.028, "rj_dg_5_2": -0.052, "rj_dg_5_3":  0.136, "rj_dg_5_4":  0.148,
                # Left arm — HDF5 step-0
                **_HDF5_LEFT_ARM_JOINT_POS,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=2000.0,
                damping=200.0,
            ),
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=90.0,
                damping=15.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=100.0,
                damping=18.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=100.0,
                damping=18.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=100.0,
                damping=18.0,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # ------------------------------------------------------------------ #
    # Source cup (dynamic)                                                 #
    # ------------------------------------------------------------------ #
    # Cup z = real-world bottom-center z + USD-origin-to-bottom offset
    # Real bottom z: 0.20 m (measured, hardcoded in teleop preprocessing)
    # cup_big.usd: origin is 0.077 m above the bottom (bottom at -0.077 in mesh)
    # → spawn origin at 0.20 + 0.077 = 0.277 m
    source_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.27, -0.10, 0.277],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_sdf.usd"),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # ------------------------------------------------------------------ #
    # Target cup (kinematic — held by left arm)                           #
    # ------------------------------------------------------------------ #
    target_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.27, 0.10, 0.277],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_sdf.usd"),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # ------------------------------------------------------------------ #
    # Fingertip ContactSensors (actor-level, real-compatible)             #
    # Each sensor covers exactly one tip link → force_matrix_w shape     #
    # (N, 1, num_filter_shapes, 3).  Extract [:, 0, 0, :] for 3D force.  #
    # ------------------------------------------------------------------ #
    tip1_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_1_tip",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/SourceCup"],
    )
    tip2_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_2_tip",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/SourceCup"],
    )
    tip3_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_3_tip",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/SourceCup"],
    )
    tip4_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_4_tip",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/SourceCup"],
    )
    tip5_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_5_tip",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/SourceCup"],
    )
