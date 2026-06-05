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
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import math

from openarm import OPENARM_ROOT_DIR

from .. import mdp
from ..grasp_right_v1_env_cfg import Grasp2gRightV2EnvCfg


@configclass
class OpenArmGrasp2gRightV2EnvCfg(Grasp2gRightV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/openarm_bimanual.usd",
                activate_contact_sensors=False,
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
                    # Pre-grasp pose for left arm (similar to reach end pose)10deg=0.1745rad
                    "openarm_left_joint1": -0.5,
                    "openarm_left_joint2": -0.5,
                    "openarm_left_joint3": 0.6,
                    "openarm_left_joint4": 0.7,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": -1.0,
                    # Pre-grasp pose for right arm (mirrored)
                    "openarm_right_joint1": 0.5,
                    "openarm_right_joint2": 0.5,
                    "openarm_right_joint3": -0.6,
                    "openarm_right_joint4": 0.7,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 1.0,
                    # Grippers fully open
                    "openarm_left_finger_joint1": 0.044,
                    "openarm_left_finger_joint2": 0.052,
                    "openarm_right_finger_joint1": 0.044,
                    "openarm_right_finger_joint2": 0.052,
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
                        "openarm_left_finger_joint.*",
                        "openarm_right_finger_joint.*",
                    ],
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        cup_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/cup.usd"
        # Cup for left arm - positioned to match reach task reset
        # Fixed position: same as reach reset (x=0.15, y=0.1, yaw=-pi/2)
        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.15, 0.1, 0.0],  # Fixed: match reach task
                rot=[1.0, 0.0, 0.0, 0.0],  # yaw = -pi/2 (match reach task)
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

        # Cup for right arm - positioned to match reach task reset
        # Fixed position: same as reach reset (x=0.15, y=-0.1, yaw=-pi/2)
        self.scene.cup2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.15, -0.1, 0.0],  # Fixed: match reach task
                rot=[1.0, 0.0, 0.0, 0.0],  # yaw = -pi/2 (match reach task)
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
        self.actions.left_hand_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={
                "openarm_left_finger_joint1": 0.044,
                "openarm_left_finger_joint2": 0.052,
            },
            close_command_expr={
                "openarm_left_finger_joint1": 0.0,
                "openarm_left_finger_joint2": 0.0,
            },
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

        self.actions.right_hand_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={
                "openarm_right_finger_joint1": 0.044,
                "openarm_right_finger_joint2": 0.052,
            },
            close_command_expr={
                "openarm_right_finger_joint1": 0.0,
                "openarm_right_finger_joint2": 0.0,
            },
        )

        # Lift-style command target: sampled TCP pose for right hand.
        self.commands.object_pose.body_name = "openarm_right_hand"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # add frame transformer for visualization
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

        # End-effector frame for reach/lift rewards (prim_path must be a rigid body)
        marker_cfg.prim_path = "/Visuals/LeftEEFrameTransformer"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_end_effector",
                ),
            ],
        )
        marker_cfg.prim_path = "/Visuals/RightEEFrameTransformer"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_end_effector",
                ),
            ],
        )


@configclass
class OpenArmGrasp2gRightV2EnvCfg_PLAY(OpenArmGrasp2gRightV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # RSL-RL dual-head policy expects concatenated 1D observations.
        self.observations.policy.concatenate_terms = True
