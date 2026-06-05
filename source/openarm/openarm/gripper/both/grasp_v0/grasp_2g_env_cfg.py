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
from isaaclab.sim import PhysxCfg
#from isaaclab.sim.schemas.schemas_cfg import RigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class Grasp2gSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and a cube to be grasped."""

    # robots
    robot: ArticulationCfg = MISSING

    # target object
    object: RigidObjectCfg = MISSING
    object2: RigidObjectCfg = MISSING

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_object_pose = mdp.UniformPoseCommandCfg(
          asset_name="robot",
          body_name=MISSING,
          resampling_time_range=(5.0, 5.0),
          debug_vis=True,
          ranges=mdp.UniformPoseCommandCfg.Ranges(
              pos_x=(0.3, 0.5),
              pos_y=(0.1, 0.3),
              pos_z=(0.25, 0.55),
              roll=(0.0, 0.0),
              pitch=(0.0, 0.0),
              yaw=(0.0, 0.0),
          ),
      )

    right_object_pose = mdp.UniformPoseCommandCfg(
          asset_name="robot",
          body_name=MISSING,
          resampling_time_range=(5.0, 5.0),
          debug_vis=True,
          ranges=mdp.UniformPoseCommandCfg.Ranges(
              pos_x=(0.3, 0.5),
              pos_y=(-0.3, -0.1),
              pos_z=(0.25, 0.55),
              roll=(0.0, 0.0),
              pitch=(0.0, 0.0),
              yaw=(0.0, 0.0),
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

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PolicyLowCfg(ObsGroup):
        """Observations for low-level skills (legacy shape)."""

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_low: PolicyLowCfg = PolicyLowCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # world-frame ranges
                "x": (0.2, 0.2),
                "y": (0.1, 0.1),
                "z": (0.05, 0.05),
            },#left object
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_object2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.2, 0.2),
                "y": (-0.1, -0.1),
                "z": (0.05, 0.05),
            },#right object
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    left_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        weight=1.0,
    )
    right_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object2"), "ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        weight=1.0,
    )

    left_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=15.0,
    )
    right_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=15.0,
    )

    left_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=16.0,
    )
    right_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=16.0,
    )

    left_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )
    right_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )

    left_grasp2g_phase = RewTerm(
        func=mdp.grasp2g_phase_value,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,
    )
    right_grasp2g_phase = RewTerm(
        func=mdp.grasp2g_phase_value,
        params={
            "object_cfg": SceneEntityCfg("object2"),
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    object2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object2")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class Grasp2gEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual grasping environment."""

    scene: Grasp2gSceneCfg = Grasp2gSceneCfg(num_envs=2048*1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)

        # assign a default physx material to all scene geometries
        # we can also do this per-asset in the scene definition
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # increase buffers to prevent overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**24,
            # set default material properties
            # default_material=RigidBodyMaterialCfg(
            #     static_friction=1.0,
            #     dynamic_friction=1.0,
            #     restitution=0.0,
            # ),
        )
