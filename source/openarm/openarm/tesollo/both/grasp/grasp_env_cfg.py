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
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Scene with a single robot, table, and grasp objects."""

    # robots
    robot: ArticulationCfg = MISSING

    # object collection
    objects: RigidObjectCollectionCfg = MISSING

    # contact sensor on the robot bodies
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_.*"],
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object = ObsTerm(
            func=mdp.object_obs,
            params={
                "eef_link_name": MISSING,
                "object_cfg": SceneEntityCfg("objects"),
            },
        )
        object_pc_feat = ObsTerm(func=mdp.object_pc_feat)
        arm_actions = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})
        hand_actions = ObsTerm(func=mdp.last_action, params={"action_name": "hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_objects = EventTerm(
        func=mdp.reset_unidex_objects,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "object_cfg": SceneEntityCfg("objects"),
            "parking_pos": (0.0, 0.0, -2.0),
        },
    )

    load_pc_feat = EventTerm(func=mdp.load_unidex_pc_feat, mode="startup", params={})
    load_grasp_prior = EventTerm(func=mdp.load_unidex_grasp_prior, mode="startup", params={})


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_distance = RewTerm(
        func=mdp.object_eef_distance_tanh,
        weight=1.0,
        params={"std": 0.15, "eef_link_name": MISSING, "object_cfg": SceneEntityCfg("objects")},
    )
    object_lifted = RewTerm(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={"minimal_height": 0.05, "object_cfg": SceneEntityCfg("objects")},
    )
    object_contact = RewTerm(
        func=mdp.object_contact_reward,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_grasp"),
            "threshold": 1.0,
            "eef_link_name": MISSING,
            "max_dist": 0.12,
            "object_cfg": SceneEntityCfg("objects"),
            "body_name_pattern": ".*dg.*",
        },
    )
    grasp_success = RewTerm(
        func=mdp.object_grasp_success,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_grasp"),
            "threshold": 1.0,
            "minimal_height": 0.05,
            "eef_link_name": MISSING,
            "max_dist": 0.12,
            "object_cfg": SceneEntityCfg("objects"),
            "body_name_pattern": ".*dg.*",
        },
    )
    object_stability = RewTerm(
        func=mdp.object_stability_reward,
        weight=0.2,
        params={"std": 0.2, "minimal_height": 0.05, "object_cfg": SceneEntityCfg("objects")},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_right_joint1",
                    "openarm_right_joint2",
                    "openarm_right_joint3",
                    "openarm_right_joint4",
                    "openarm_right_joint5",
                    "openarm_right_joint6",
                    "openarm_right_joint7",
                ],
            )
        },
    )
    hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rj_dg_.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.object_below_minimum,
        params={"minimum_height": 0.0, "object_cfg": SceneEntityCfg("objects")},
    )


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the grasp environment."""

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        if self.scene.contact_grasp is not None:
            self.scene.contact_grasp.update_period = self.sim.dt
