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

"""Lift-style left grasp environment configuration for 2g cup task.

This task keeps the original 2g_grasp_left_v1 scene spawn layout (robot/table/cup placement),
while using a simplified lift-style training setup and rewards.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
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

from . import mdp


@configclass
class Grasp2gSceneCfg(InteractiveSceneCfg):
    """Scene with bimanual robot, table and two cups.

    Spawn structure and placement are intentionally kept aligned with 2g_grasp_left_v1.
    """

    robot: ArticulationCfg = MISSING
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications.

    Right actions are preserved for compatibility with the hold-env wrapper,
    but are masked by curriculum_stage=0 so only left side moves.
    """

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class CommandsCfg:
    """Command terms for object-lift target."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set in robot-specific env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(0.0, 0.4),
            pos_z=(0.3, 0.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Lift-style policy observations focused on left-hand grasp/lift."""

        joint_pos = ObsTerm(func=mdp.left_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.left_joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cup")},
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset/randomization events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (0.2, 0.2),
                "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    reset_cup2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (-0.2, -0.2),
                "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )


@configclass
class RewardsCfg:
    """Lift-style reward terms focused on left cup."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "openarm_left_hand"},
        weight=3.0,
    )

    # Orientation alignment: encourage hand +X axis to align with object +Z axis.
    end_effector_orientation = RewTerm(
        func=mdp.hand_x_align_object_z_reward,
        params={"command_name": "object_pose", "asset_cfg": SceneEntityCfg("robot", body_names=["openarm_left_hand"])},
        weight=0.1,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cup")},
        weight=10.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.4, "minimal_height": 0.04, "command_name": "object_pose", "object_cfg": SceneEntityCfg("cup")},
        weight=20.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.04, "command_name": "object_pose", "object_cfg": SceneEntityCfg("cup")},
        weight=10.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint.*",
                    "openarm_left_finger_joint.*",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")},
    )

    cup_tipping = DoneTerm(
        func=mdp.cup_tipped,
        params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 30.0},
    )


@configclass
class CurriculumCfg:
    """Lift-style curriculum on regularization terms only."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-1, "num_steps": 50000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-1, "num_steps": 50000},
    )


@configclass
class Grasp2gV2EnvCfg(ManagerBasedRLEnvCfg):
    """Left-hand cup grasp/lift environment using lift-style learning design."""

    task_name: str = "grasp_2g_left_lift_style"
    debug_enabled: bool = True

    # Keep stage-masking behavior in Grasp2gHoldEnv:
    # stage 0 masks right arm/hand actions -> only left side is trainable.
    curriculum_stage: int = 0
    mask_inactive_arm_actions: bool = True

    # Target offset from cup root (0, 0, 0.05) = 5cm above cup bottom
    # This guides the gripper to approach from above the cup rim
    grasp2g_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.05)

    scene: Grasp2gSceneCfg = Grasp2gSceneCfg(num_envs=2048*1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum = None

    def __post_init__(self):
        # lift_openarm-like timing
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.observations.policy.concatenate_terms = True

        if not self.debug_enabled:
            self.commands.object_pose.debug_vis = False

        self.sim.physx.bounce_threshold_velocity = 0.01
        # Aggressive high-capacity setting for 24 GiB GPUs (e.g., RTX 4090).
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 64 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 1024
        # Required for dense contact scenes: avoid "Patch buffer overflow" warnings.
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**24
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.friction_correlation_distance = 0.00625
