# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from openarm import OPENARM_ROOT_DIR

# Hand joint names
# Thumb (finger 1): 4 joints
LEFT_THUMB_JOINTS = [f"lj_dg_1_{joint}" for joint in range(1, 5)]
RIGHT_THUMB_JOINTS = [f"rj_dg_1_{joint}" for joint in range(1, 5)]

# Pinky (finger 5): 4 joints - separated for independent control
LEFT_PINKY_JOINTS = [f"lj_dg_5_{joint}" for joint in range(1, 5)]
RIGHT_PINKY_JOINTS = [f"rj_dg_5_{joint}" for joint in range(1, 5)]

# Fingers 2-4: controlled via synergy (defined in mdp/actions.py)
# Full hand joints (for reference only)
LEFT_HAND_JOINTS = [f"lj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]
RIGHT_HAND_JOINTS = [f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]

from .. import mdp
from ..lift_left_env_cfg import Lift5gLeftEnvCfg


@configclass
class OpenArmLift5gLeftEnvCfg(Lift5gLeftEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/openarm_tesollo_t2.usd",
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
                pos=[0.0, 0.0, -0.25],
                rot=[1.0, 0.0, 0.0, 0.0],
                joint_pos={
                    "openarm_left_joint1": -0.5,
                    "openarm_left_joint2": -0.5,
                    "openarm_left_joint3": 0.6,
                    "openarm_left_joint4": 0.7,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": -1.0,
                    "openarm_right_joint1": 0.5,
                    "openarm_right_joint2": 0.5,
                    "openarm_right_joint3": -0.6,
                    "openarm_right_joint4": 0.7,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 1.0,
                    # Left synergy fingers (2,3,4) - controlled by synergy action
                    "lj_dg_[2-4]_.*": 0.0,
                    # Left thumb (1)
                    "lj_dg_1_1": 0.0,
                    "lj_dg_1_2": 1.571,   # [0.0, 1.571] max open
                    "lj_dg_1_3": 0.0,
                    "lj_dg_1_4": 0.0,
                    # Left pinky (5)
                    "lj_dg_5_1": 0.0,
                    "lj_dg_5_2": 0.0,
                    "lj_dg_5_3": 0.0,
                    "lj_dg_5_4": 0.0,
                    # Right synergy fingers (2,3,4)
                    "rj_dg_[2-4]_.*": 0.0,
                    # Right thumb (1)
                    "rj_dg_1_1": 0.0,
                    "rj_dg_1_2": -1.571,  # [-1.571, 0.0] max open
                    "rj_dg_1_3": 0.0,
                    "rj_dg_1_4": 0.0,
                    # Right pinky (5)
                    "rj_dg_5_1": 0.0,
                    "rj_dg_5_2": 0.0,
                    "rj_dg_5_3": 0.0,
                    "rj_dg_5_4": 0.0,
                },
            ),
            actuators={
                "openarm_arm": ImplicitActuatorCfg(
                    joint_names_expr=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
                    stiffness=400.0,
                    damping=80.0,
                ),
                "openarm_gripper": ImplicitActuatorCfg(
                    joint_names_expr=["lj_dg_.*", "rj_dg_.*"],
                    stiffness=2e3,
                    damping=2e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        cup_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/cup.usd"
        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0.1, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(1.0, 1.0, 1.2),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.cup2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, -0.1, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(1.0, 1.0, 1.2),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Action order: left_arm/hand/thumb, right_arm/hand/thumb
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            scale=0.2,
            use_default_offset=True,
        )
        # Left hand: Synergy for fingers 2-4 (1D) + Individual control for thumb+pinky (8D)
        self.actions.left_hand_action = mdp.FingerSynergyActionLeftCfg(
            asset_name="robot",
            # Uses default: NON_THUMB_JOINTS_LEFT (fingers 2-4)
        )
        # Thumb action (finger 1 only): 4 DOF
        self.actions.left_thumb_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_THUMB_JOINTS,
            scale={
                "lj_dg_1_1": 0.8901179,
                "lj_dg_1_2": 1.5705927,
                "lj_dg_1_3": 1.5707963,
                "lj_dg_1_4": 1.5707963,
            },
            clip={
                "lj_dg_1_1": (-0.8901179, 0.3839724),
                "lj_dg_1_2": (0.0, 3.1415927),
                "lj_dg_1_3": (-1.5707963, 1.5707963),
                "lj_dg_1_4": (-1.5707963, 1.5707963),
            },
            use_default_offset=True,
        )
        # Pinky action (finger 5 only): 4 DOF - separated for independent control
        self.actions.left_pinky_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_PINKY_JOINTS,
            scale={
                "lj_dg_5_1": 1.0471976,
                "lj_dg_5_2": 0.6108652,
                "lj_dg_5_3": 1.5707963,
                "lj_dg_5_4": 1.5707963,
            },
            clip={
                "lj_dg_5_1": (-1.0471976, 0.0174533),
                "lj_dg_5_2": (-0.6108652, 0.418879),
                "lj_dg_5_3": (-1.5707963, 1.5707963),
                "lj_dg_5_4": (-1.5707963, 1.5707963),
            },
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ],
            scale=0.2,
            use_default_offset=True,
        )
        # Right hand: Synergy for fingers 2-4 (1D) + Individual control for thumb+pinky (8D)
        self.actions.right_hand_action = mdp.FingerSynergyActionCfg(
            asset_name="robot",
        )
        # Right thumb action (finger 1 only): 4 DOF
        self.actions.right_thumb_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_THUMB_JOINTS,
            scale={
                "rj_dg_1_1": 0.8901179,
                "rj_dg_1_2": 1.5705927,
                "rj_dg_1_3": 1.5707963,
                "rj_dg_1_4": 1.5707963,
            },
            clip={
                "rj_dg_1_1": (-0.3839724, 0.8901179),
                "rj_dg_1_2": (-3.1415927, 0.0),
                "rj_dg_1_3": (-1.5707963, 1.5707963),
                "rj_dg_1_4": (-1.5707963, 1.5707963),
            },
            use_default_offset=True,
        )
        # Right pinky action (finger 5 only): 4 DOF
        self.actions.right_pinky_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_PINKY_JOINTS,
            scale={
                "rj_dg_5_1": 1.0471976,
                "rj_dg_5_2": 0.6108652,
                "rj_dg_5_3": 1.5707963,
                "rj_dg_5_4": 1.5707963,
            },
            clip={
                "rj_dg_5_1": (-0.0174533, 1.0471976),
                "rj_dg_5_2": (-0.418879, 0.6108652),
                "rj_dg_5_3": (-1.5707963, 1.5707963),
                "rj_dg_5_4": (-1.5707963, 1.5707963),
            },
            use_default_offset=True,
        )

        self.commands.object_pose.body_name = "ll_dg_ee"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        # Match test1 regularization strengths.


@configclass
class OpenArmLift5gLeftEnvCfg_PLAY(OpenArmLift5gLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = True
