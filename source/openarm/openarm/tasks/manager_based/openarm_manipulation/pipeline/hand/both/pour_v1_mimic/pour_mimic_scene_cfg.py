"""InteractiveSceneCfg for pour_v1_mimic.

Reuses the same assets as pour_v1 (openarm_tesollo_sensor.usd, cup_big_sdf.usd)
and wires up the 5 per-fingertip ContactSensors that provide real-compatible
force observations.
"""

from __future__ import annotations

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from ..pour_v1.pour_preset import LEFT_ARM_REST_JOINT_POS

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class PourMimicSceneCfg(InteractiveSceneCfg):
    """Scene for bimanual pour Mimic data collection.

    Robot: openarm_tesollo_sensor.usd (activate_contact_sensors=True)
    Objects: source cup (dynamic), target cup (kinematic)
    Sensors: 5 × fingertip ContactSensor (rl_dg_*_tip)
    """

    # ------------------------------------------------------------------ #
    # Ground + lighting                                                    #
    # ------------------------------------------------------------------ #
    ground = sim_utils.GroundPlaneCfg() if False else None  # set via spawn

    light = sim_utils.AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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
                "openarm_right_joint1":  0.5,
                "openarm_right_joint2":  0.1,
                "openarm_right_joint3":  0.0,
                "openarm_right_joint4":  0.60,
                "openarm_right_joint5": -0.2,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                "rj_dg_1_1": 0.0, "rj_dg_1_2": -1.57, "rj_dg_1_3": -0.5, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.0,  "rj_dg_2_3":  0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.0,  "rj_dg_3_3":  0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.0,  "rj_dg_4_3":  0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0,  "rj_dg_5_3":  0.0, "rj_dg_5_4": 0.0,
                **{k: v for k, v in LEFT_ARM_REST_JOINT_POS.items()},
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
    source_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.50, 0.00, 0.30],
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
            pos=[0.42, 0.18, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big_sdf.usd"),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=-0.1,
                rest_offset=-0.1,
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
