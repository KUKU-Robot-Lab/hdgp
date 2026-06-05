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

"""Left-only approach environment configuration for 5g hand task.

Scene layout (table/cups) and physics buffers are kept aligned with 2g_grasp_left_v1.
Grasp/lift-specific terms are disabled for a pure approach task.
"""

from dataclasses import MISSING
import math

import isaaclab.envs.mdp as base_mdp
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class ApproachLeftSceneCfg(InteractiveSceneCfg):
    """Scene with bimanual robot, table and two cups."""

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
    """Action specifications."""

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    left_thumb_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING
    right_thumb_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations focused on left approach behavior."""

        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cup")},
        )
        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        left_thumb_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_thumb_action"})

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
                "yaw": (math.pi, math.pi),
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
                "yaw": (0, 0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )


@configclass
class RewardsCfg:
    """Approach-style rewards (grasp/lift terms intentionally disabled)."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.12, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=2.0,
    )

    end_effector_orientation = RewTerm(
        func=mdp.eef_to_object_orientation,
        params={"std": 0.4, "eef_link_name": "ll_dg_ee", "object_cfg": SceneEntityCfg("cup")},
        weight=1.0,
    )

    cup_displacement_penalty = RewTerm(
        func=mdp.object_root_displacement_penalty,
        params={"object_cfg": SceneEntityCfg("cup"), "scale": 1.0},
        weight=-5.0,
    )

    left_hand_joint_target = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"]),
            "target": 0.0,
        },
    )

    left_hand_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lj_dg_.*"])},
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint.*",
                    "lj_dg_.*",
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
        params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 45.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-1, "num_steps": 20000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-1, "num_steps": 20000},
    )


@configclass
class ApproachLeftEnvCfg(ManagerBasedRLEnvCfg):
    """Left-hand cup approach environment with 5g hand."""

    scene: ApproachLeftSceneCfg = ApproachLeftSceneCfg(num_envs=2048 * 1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Cup-local target offset for side approach (x: cup forward, z: up).
    # (0,0,0.05) was near center-top; this pushes the hand to the cup side.
    grasp2g_target_offset: tuple[float, float, float] = (0.0, -0.05, 0.08)
    reach_dynamic_z_high: float = 0.25
    reach_dynamic_xy_hi: float = 0.10
    reach_dynamic_xy_lo: float = 0.03
    reach_dynamic_xy_gate: float = 0.07
    reach_dynamic_z_descent_rate: float = 0.01
    reach_stage_xy_threshold: float = 0.07
    reach_stage_lowz_std_scale: float = 0.7
    reach_displacement_free_threshold: float = 0.015
    reach_displacement_suppress_scale: float = 0.03
    reach_switch_threshold: float = 0.01
    reach_switch_hold_steps: int = 5
    curriculum_stage: int = 0
    mask_inactive_arm_actions: bool = True
    debug_ll_dg_ee_vis: bool = True
    debug_approach_target_vis: bool = True
    debug_approach_target_vis_interval: int = 1
    debug_approach_target_vis_env_id: int = 0

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.observations.policy.concatenate_terms = True

        # Keep physics buffer sizing same as 2g_grasp_left_v1.
        self.sim.physx.bounce_threshold_velocity = 0.01
        # Aggressive high-capacity setting for 24 GiB GPUs (e.g., RTX 4090).
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 64 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 2**25
        self.sim.physx.gpu_max_rigid_contact_count = 2**25
        self.sim.physx.gpu_collision_stack_size = 2**25
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.friction_correlation_distance = 0.00625
