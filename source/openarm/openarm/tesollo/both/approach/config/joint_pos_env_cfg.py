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

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from .. import mdp
from ..approach_env_cfg import (
    ApproachEnvCfg,
)

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)
from openarm import OPENARM_ROOT_DIR
from isaaclab.assets.articulation import ArticulationCfg

##
# Environment configuration
##


@configclass
class OpenArmApproachEnvCfg(ApproachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to tesollo robot
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
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "lj_dg_.*": 0.0,
                    "rj_dg_.*": 0.0,
                },  # Close the gripper
            ),
            actuators={
                "openarm_arm": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "openarm_left_joint[1-7]",
                        "openarm_right_joint[1-7]",
                    ],
                    stiffness=400.0,
                    damping=80.0,
                ),
                "openarm_gripper": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "lj_dg_.*",
                        "rj_dg_.*",
                    ],
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        # override rewards
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["ll_dg_ee"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ll_dg_ee"]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ll_dg_ee"]
        # self.rewards.left_end_effector_x_axis_tracking.params["asset_cfg"].body_names = ["ll_dg_ee"]

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["rl_dg_ee"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["rl_dg_ee"]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["rl_dg_ee"]
        # self.rewards.right_end_effector_x_axis_tracking.params["asset_cfg"].body_names = ["rl_dg_ee"]

        # override actions
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
            scale=0.5,
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
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lj_dg_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["rj_dg_.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.left_ee_pose.body_name = "ll_dg_ee"
        self.commands.right_ee_pose.body_name = "rl_dg_ee"


@configclass
class OpenArmApproachEnvCfg_PLAY(OpenArmApproachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
