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

"""Multi-object grasp-focused left 5g environment."""

from dataclasses import MISSING
import math

import isaaclab.envs.mdp as base_mdp
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
class Grasp5gLeftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    objects: RigidObjectCollectionCfg = MISSING

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )

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
    # Order: left_arm/hand/thumb, right_arm/hand/thumb
    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    left_thumb_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING
    right_thumb_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_info = ObsTerm(
            func=mdp.selected_object_obs_left,
            params={
                "left_eef_link_name": "ll_dg_ee",
                "object_cfg": SceneEntityCfg("objects"),
            },
        )
        object_id_onehot = ObsTerm(func=mdp.selected_object_id_onehot)
        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        left_thumb_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_thumb_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_objects = EventTerm(
        func=mdp.reset_object_collection,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.22, 0.30),
                "y": (0.10, 0.20),
                "z": (0.02, 0.04),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "object_cfg": SceneEntityCfg("objects"),
            "parking_pos": (0.0, 0.0, -2.0),
        },
    )


@configclass
class RewardsCfg:
    object_distance = RewTerm(
        func=mdp.selected_object_eef_distance_tanh,
        weight=8.0,
        params={"std": 0.12, "left_eef_link_name": "ll_dg_ee", "object_cfg": SceneEntityCfg("objects")},
    )

    object_contact = RewTerm(
        func=mdp.selected_object_contact_reward_left,
        weight=4.0,
        params={
            "threshold": 1.0,
            "left_eef_link_name": "ll_dg_ee",
            "max_dist": 0.12,
            "object_cfg": SceneEntityCfg("objects"),
            "sensor_cfg": SceneEntityCfg("contact_grasp"),
            "body_name_pattern": ".*lj_dg.*",
        },
    )

    object_lifted = RewTerm(
        func=mdp.selected_object_is_lifted,
        weight=4.0,
        params={"minimal_height": 0.05, "object_cfg": SceneEntityCfg("objects")},
    )

    grasp_success = RewTerm(
        func=mdp.selected_object_grasp_success_left,
        weight=8.0,
        params={
            "threshold": 1.0,
            "minimal_height": 0.05,
            "left_eef_link_name": "ll_dg_ee",
            "max_dist": 0.12,
            "object_cfg": SceneEntityCfg("objects"),
            "sensor_cfg": SceneEntityCfg("contact_grasp"),
            "body_name_pattern": ".*lj_dg.*",
        },
    )

    object_stability = RewTerm(
        func=mdp.selected_object_stability_reward,
        weight=1.0,
        params={"std": 0.2, "minimal_height": 0.05, "object_cfg": SceneEntityCfg("objects")},
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.selected_object_below_minimum,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("objects")},
    )


@configclass
class Grasp5gLeftEnvCfg(ManagerBasedRLEnvCfg):
    task_name: str = "grasp_5g_left_v1"
    curriculum_stage: int = 0
    mask_inactive_arm_actions: bool = True

    scene: Grasp5gLeftSceneCfg = Grasp5gLeftSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.observations.policy.concatenate_terms = True

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 64 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 2**25
        self.sim.physx.gpu_max_rigid_contact_count = 2**25
        self.sim.physx.gpu_collision_stack_size = 2**25
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.friction_correlation_distance = 0.00625

        if self.scene.contact_grasp is not None:
            self.scene.contact_grasp.update_period = self.sim.dt
