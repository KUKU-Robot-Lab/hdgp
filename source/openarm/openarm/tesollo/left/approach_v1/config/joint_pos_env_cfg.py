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

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from openarm import OPENARM_ROOT_DIR

# Thumb action controls finger 1 (thumb) + finger 5 (pinky): 8 joints.
LEFT_THUMB_JOINTS = [f"lj_dg_{finger}_{joint}" for finger in (1, 5) for joint in range(1, 5)]
RIGHT_THUMB_JOINTS = [f"rj_dg_{finger}_{joint}" for finger in (1, 5) for joint in range(1, 5)]

from .. import mdp
from ..approach_left_env_cfg import ApproachLeftEnvCfg


@configclass
class OpenArmApproachLeftEnvCfg(ApproachLeftEnvCfg):

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
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, 0.0, -0.25],
                rot=[1.0, 0.0, 0.0, 0.0],
                joint_pos={
                    # Match non-hand initial joints with 2g_grasp_left_v1.
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
                    # Keep 5g fingers neutral/open for non-grasp approach mode.
                    "lj_dg_.*": 0.0,
                    "rj_dg_.*": 0.0,
                },
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

        cup_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/cup.usd"
        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.15, 0.1, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(0.8, 0.8, 1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
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
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.15, -0.1, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(0.8, 0.8, 1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
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

        # Left hand: finger 2-4 synergy (1D) + thumb/pinky individual (8D).
        self.actions.left_hand_action = mdp.FingerSynergyActionLeftCfg(
            asset_name="robot",
        )
        self.actions.left_thumb_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_THUMB_JOINTS,
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

        self.actions.right_hand_action = mdp.FingerSynergyActionCfg(
            asset_name="robot",
        )
        self.actions.right_thumb_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_THUMB_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )


@configclass
class OpenArmApproachLeftEnvCfg_PLAY(OpenArmApproachLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = True
