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

"""Bimanual pouring environment configuration.

This task trains both arms to coordinate for pouring liquid (beads) from
the left cup to the right cup. It uses a rollout-based approach where
Phase 1 policies are executed first to grasp cups, then pouring is trained.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class PouringSceneCfg(InteractiveSceneCfg):
    """Scene with bimanual robot, table, two cups, and beads."""

    robot: ArticulationCfg = MISSING

    # Left cup (source - contains beads)
    cup: RigidObjectCfg = MISSING
    # Right cup (target - empty)
    cup2: RigidObjectCfg = MISSING
    # Beads (liquid proxy)
    bead: RigidObjectCfg = MISSING

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
    """Action specifications for bimanual pouring."""

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class CommandsCfg:
    """Command terms for the MDP (placeholder for goal positions)."""
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for bimanual pouring."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint states (both arms)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Cup positions relative to robot
        left_cup_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cup")},
        )
        right_cup_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cup2")},
        )

        # Bead position relative to robot
        bead_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bead")},
        )

        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset left cup (source) position
    reset_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (0.15, 0.15),
                "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    # Reset right cup (target) position
    reset_cup2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (-0.15, -0.15),
                "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )

    # Reset bead inside left cup
    reset_bead_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (0.15, 0.15),
                "z": (0.05, 0.05),  # Above cup bottom
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bead"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for bimanual pouring.

    Reward design based on leapiper_r_sens_env.py:
    - Phase-based rewards (rho=0: approach, rho=1: pour)
    - Distance rewards with tanh-based shaping (XY, Z separately)
    - Orientation rewards (upright vs tilt towards target)
    - Bead transfer rewards (high weight for success)
    """

    # === Phase-based Combined Distance Reward ===
    # XY alignment + Z height, changes behavior based on phase
    phase_distance = RewTerm(
        func=mdp.phase_distance_reward,
        params={
            "source_cup_cfg": SceneEntityCfg("cup"),
            "target_cup_cfg": SceneEntityCfg("cup2"),
            "approach_xy_min": 0.0,
            "approach_xy_max": 0.03,
            "approach_z_min": 0.14,
            "approach_z_max": 0.17,
            "pour_z_min": 0.07,
            "pour_z_max": 0.09,
        },
        weight=4.0,
    )

    # === Phase-based Orientation Reward ===
    # Approach: keep upright, Pour: tilt towards target
    phase_orientation = RewTerm(
        func=mdp.phase_orientation_reward,
        params={
            "source_cup_cfg": SceneEntityCfg("cup"),
            "target_cup_cfg": SceneEntityCfg("cup2"),
        },
        weight=18.0,  # High weight like leapiper (drop_reward * 18.5)
    )

    # === Target Cup Stability ===
    # Right cup (target) should stay upright always
    right_cup_upright = RewTerm(
        func=mdp.cup_upright_reward,
        params={"cup_cfg": SceneEntityCfg("cup2")},
        weight=5.0,
    )

    # === Phase Bonus ===
    # Bonus when reaching pour phase (cups aligned)
    phase_bonus = RewTerm(
        func=mdp.phase_reached_bonus,
        params={
            "source_cup_cfg": SceneEntityCfg("cup"),
            "target_cup_cfg": SceneEntityCfg("cup2"),
        },
        weight=18.5,  # From leapiper: rewards += 18.5 when rho=1
    )

    # === Bead Rewards ===
    # Keep bead above ground
    bead_height = RewTerm(
        func=mdp.bead_height_reward,
        params={"bead_cfg": SceneEntityCfg("bead"), "min_height": 0.05},
        weight=5.0,
    )

    # Smooth reward for bead approaching target
    bead_approaching = RewTerm(
        func=mdp.bead_approaching_target_reward,
        params={
            "bead_cfg": SceneEntityCfg("bead"),
            "target_cup_cfg": SceneEntityCfg("cup2"),
            "std": 0.1,
        },
        weight=10.0,
    )

    # Big bonus for bead in target cup
    bead_in_target = RewTerm(
        func=mdp.bead_in_target_cup_reward,
        params={
            "bead_cfg": SceneEntityCfg("bead"),
            "target_cup_cfg": SceneEntityCfg("cup2"),
            "xy_radius": 0.05,
            "z_offset_min": -0.02,
            "z_offset_max": 0.1,
        },
        weight=180.0,  # From leapiper: +180 when bead transferred
    )

    # === Regularization ===
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for bimanual pouring."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Cup dropping
    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")},
    )
    cup2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup2")},
    )

    # Cup tipping (target cup should stay upright)
    cup2_tipping = DoneTerm(
        func=mdp.cup_tipped,
        params={"asset_cfg": SceneEntityCfg("cup2"), "max_tilt_deg": 45.0},
    )

    # Bead spill (falls to ground)
    bead_spill = DoneTerm(
        func=mdp.bead_on_ground,
        params={"bead_name": "bead", "ground_height": 0.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum for gradually increasing regularization penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class PouringV1EnvCfg(ManagerBasedRLEnvCfg):
    """Bimanual pouring environment configuration."""

    task_name: str = "2g_pouring_v1"
    debug_enabled: bool = True

    scene: PouringSceneCfg = PouringSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Timing
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation

        self.observations.policy.concatenate_terms = True

        # PhysX settings for stable bead simulation
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=1000000000,
        )
