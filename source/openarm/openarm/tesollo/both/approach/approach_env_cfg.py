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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class ApproachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.3)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.5),
            pos_y=(0.15, 0.4),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 2, -math.pi / 2),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.5),
            pos_y=(-0.4, -0.15),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(math.pi / 2, math.pi / 2),
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                    "openarm_left_joint2",
                                                                    "openarm_left_joint3",
                                                                    "openarm_left_joint4",
                                                                    "openarm_left_joint5",
                                                                    "openarm_left_joint6",
                                                                    "openarm_left_joint7",
                                                                  ])
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                    "openarm_right_joint2",
                                                                    "openarm_right_joint3",
                                                                    "openarm_right_joint4",
                                                                    "openarm_right_joint5",
                                                                    "openarm_right_joint6",
                                                                    "openarm_right_joint7"
                                                                  ])
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                    "openarm_left_joint2",
                                                                    "openarm_left_joint3",
                                                                    "openarm_left_joint4",
                                                                    "openarm_left_joint5",
                                                                    "openarm_left_joint6",
                                                                    "openarm_left_joint7",
                                                                  ])
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                    "openarm_right_joint2",
                                                                    "openarm_right_joint3",
                                                                    "openarm_right_joint4",
                                                                    "openarm_right_joint5",
                                                                    "openarm_right_joint6",
                                                                    "openarm_right_joint7"
                                                                  ])
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )
        left_actions = ObsTerm(func=mdp.last_action,
                params={
                "action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action,
                params={
                "action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_with_deadzone,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "threshold": 0.01,
        },
    )

    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_with_deadzone,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "threshold": 0.01,
        },
    )

    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_z_axis_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_z_axis_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )
    # left_end_effector_x_axis_tracking = RewTerm(
    #     func=mdp.orientation_x_axis_error,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
    #         "command_name": "left_ee_pose",
    #     },
    # )
    # right_end_effector_x_axis_tracking = RewTerm(
    #     func=mdp.orientation_x_axis_error,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
    #         "command_name": "right_ee_pose",
    #     },
    # )

    # action penalty
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.01,
    )
    near_goal_joint_vel = RewTerm(
        func=mdp.near_goal_joint_vel_l2,
        weight=-0.005,
        params={
            "left_command_name": "left_ee_pose",
            "right_command_name": "right_ee_pose",
            "pos_threshold": 0.01,
            "ori_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "left_ee_cfg": SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
            "right_ee_cfg": SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
        },
    )
    left_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                "openarm_left_joint2",
                                                                "openarm_left_joint3",
                                                                "openarm_left_joint4",
                                                                "openarm_left_joint5",
                                                                "openarm_left_joint6",
                                                                "openarm_left_joint7",
                                                              ]),
        },
    )
    right_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                "openarm_right_joint2",
                                                                "openarm_right_joint3",
                                                                "openarm_right_joint4",
                                                                "openarm_right_joint5",
                                                                "openarm_right_joint6",
                                                                "openarm_right_joint7"
                                                              ]),
        },
    )
    left_hand_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
    )
    right_hand_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
    )
    left_hand_joint_target = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"]),
            "target": 0.0,
        },
    )
    right_hand_joint_target = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"]),
            "target": 0.0,
        },
    )
    left_hand_joint_open = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"]),
            "target": 0.0,
        },
    )
    right_hand_joint_open = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"]),
            "target": 0.0,
        },
    )
    left_hand_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
    )
    right_hand_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    success = DoneTerm(
        func=mdp.ee_reached_and_stopped,
        params={
            "left_command_name": "left_ee_pose",
            "right_command_name": "right_ee_pose",
            "pos_threshold": 0.005,
            "ori_threshold": 0.0175,  # 0.0175 rad, about 1 degree
            "joint_vel_threshold": 0.03, # rad/s
            "robot_cfg": SceneEntityCfg("robot"),
            "left_ee_cfg": SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
            "right_ee_cfg": SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # No curriculum terms to restore previous behavior
    pass


##
# Environment configuration 
##16384 20480 4096 8192 


@configclass
class ApproachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the approach end-effector pose tracking environment."""

    # Scene settings
    scene: ApproachSceneCfg = ApproachSceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
