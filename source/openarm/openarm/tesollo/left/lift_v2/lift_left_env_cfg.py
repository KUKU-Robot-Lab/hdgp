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

"""Lift-style left 5g environment v2: v1 + DexPour with contact sensor support.

Identical reward structure/order to v1, but uses T3 hand contact sensors
for grasp detection (DexPour style). Geometry-based fallback is kept for
functions that do not receive sensor_cfg.
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp
from .config.joint_pos_env_cfg import LEFT_CONTACT_LINKS


@configclass
class Lift5gLeftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING
    left_contact_sensor: ContactSensorCfg = MISSING

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
    # Order: left_arm/hand/thumb/pinky, right_arm/hand/thumb/pinky
    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    left_thumb_action: ActionTerm = MISSING
    left_pinky_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING
    right_thumb_action: ActionTerm = MISSING
    right_pinky_action: ActionTerm = MISSING


# Create larger marker config for better visibility
GOAL_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
GOAL_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.3),
            pos_y=(0.1, 0.2),
            pos_z=(0.3, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=GOAL_MARKER_CFG,
    )


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
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cup")})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        left_thumb_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_thumb_action"})
        left_pinky_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_pinky_action"})

        left_contact_flags = ObsTerm(
            func=mdp.contact_flags_multi,
            params={"sensor_cfg": SceneEntityCfg("left_contact_sensor")},
        )
        left_normal_forces = ObsTerm(
            func=mdp.normal_force_magnitude_multi,
            params={"sensor_cfg": SceneEntityCfg("left_contact_sensor")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        left_slip_velocity = ObsTerm(
            func=mdp.slip_velocity,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=[
                    f"tesollo_left_{ln}" for ln in LEFT_CONTACT_LINKS
                ]),
                "object_cfg": SceneEntityCfg("cup"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 2.0),  # 관절 폭발 시 극단값이 policy에 피드백되지 않도록 클리핑
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
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


# ---------------------------------------------------------------------------
# Rewards: Identical structure/order to v1 + sensor_cfg where applicable
# ---------------------------------------------------------------------------
_SENSOR_CFG = SceneEntityCfg("left_contact_sensor")


@configclass
class RewardsCfg:
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=8.0,
    )
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance_fine,
        params={"std": 0.065, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=10.0,
    )

    end_effector_orientation = RewTerm(
        func=mdp.eef_z_perpendicular_object_z,
        params={"std": 0.3, "eef_link_name": "ll_dg_ee", "object_cfg": SceneEntityCfg("cup")},
        weight=4.0,
    )

    # Force-based grasp rewards (replacing geometry-based thumb_grasp/pinky_grasp/synergy_grip)
    contact_persistence = RewTerm(
        func=mdp.contact_persistence_reward_multi,
        weight=10.0,
        params={
            "sensor_cfg": _SENSOR_CFG,
            "min_contacts": 4,
            "contact_threshold": 0.05,
            "use_filtered": False,
            "object_cfg": SceneEntityCfg("cup"),        # λ gate: approach 전 조기 손가락 폐쇄 방지
            "eef_link_name": "ll_dg_ee",
        },
    )
    slip_penalty = RewTerm(
        func=mdp.slip_magnitude_penalty,
        weight=0.0,  # stability phase: disabled
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=[
                f"tesollo_left_{ln}" for ln in LEFT_CONTACT_LINKS
            ]),
            "object_cfg": SceneEntityCfg("cup"),
            "max_slip": 0.1,
            "contact_sensor_cfg": _SENSOR_CFG,
            "contact_threshold": 0.05,
        },
    )
    force_spike = RewTerm(
        func=mdp.force_spike_penalty_multi,
        weight=0.0,  # stability phase: disabled
        params={
            "sensor_cfg": _SENSOR_CFG,
            "spike_threshold": 10.0,
            "contact_threshold": 0.05,
        },
    )
    overgrip = RewTerm(
        func=mdp.overgrip_penalty_multi,
        weight=0.0,  # stability phase: disabled
        params={
            "sensor_cfg": _SENSOR_CFG,
            "max_force": 15.0,
            "contact_threshold": 0.05,
        },
    )

    finger_tip_to_cup = RewTerm(
        func=mdp.finger_wrap_cylinder_reward,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "ll_dg_ee",
            "target_radius": 0.045,
            "radial_std": 0.015,
            "opposition_weight": 0.3,
        },
        weight=0.0,  # 비활성화: opposition이 엄지-시너지 길항 유발
    )

    finger_wrap_coverage = RewTerm(
        func=mdp.finger_wrap_coverage_reward,
        params={"object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # 비활성화: 시너지 단일 DOF로 각도 분산 불가, oscillation 유발
    )

    finger_tip_orientation = RewTerm(
        func=mdp.finger_tip_orientation_reward,
        params={"std": 0.5, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.04, "object_cfg": SceneEntityCfg("cup"),
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
        },
        weight=50.0,  # 10.0 → 50.0: sparse reward 돌파를 위한 강력한 인센티브
    )

    # μ=1이면 컵 Z 상승에 연속적 gradient 제공 (tanh: delta=0에서 최대 gradient)
    # ee_descent가 (1-μ)로 비활성되므로 리프트 방향 힘을 이 보상이 담당
    cup_lift_progress = RewTerm(
        func=mdp.cup_lift_progress_reward,
        params={
            "std": 0.05, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee",
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
        },
        weight=20.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3, "minimal_height": 0.04, "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cup"),
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
        },
        weight=0.0,  # stability phase: disabled
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.1, "minimal_height": 0.04, "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cup"),
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
        },
        weight=0.0,  # stability phase: disabled
    )

    object_displacement = RewTerm(
        func=mdp.object_displacement_penalty,
        params={"object_cfg": SceneEntityCfg("cup"), "threshold": 0.01},
        weight=0.0,  # stability phase: disabled
    )

    finger_normal_range = RewTerm(
        func=mdp.finger_normal_range_penalty,
        params={},
        weight=0.0,  # stability phase: disabled
    )

    thumb_reaching_pose = RewTerm(
        func=mdp.thumb_reaching_pose_reward,
        params={"std": 1.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    pinky_reaching_pose = RewTerm(
        func=mdp.pinky_reaching_pose_reward,
        params={"std": 1.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    synergy_reaching_pose = RewTerm(
        func=mdp.synergy_reaching_pose_reward,
        params={"std": 5.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    thumb_tip_z = RewTerm(
        func=mdp.thumb_tip_z_reward,
        params={"std": 0.03, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    synergy_tip_z = RewTerm(
        func=mdp.synergy_tip_z_reward,
        params={"std": 0.06, "cup_height": 0.09, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.0,  # stability phase: disabled
    )

    ee_descent = RewTerm(
        func=mdp.ee_descent_reward,
        params={
            "std": 0.04, "target_z_offset": 0.04,
            "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee",
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),  # 구조적 수정: μ 판정을 sensor 기준으로 통일
        },
        weight=0.0,  # stability phase: disabled
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-5e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cup_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")})
    cup_tipping = DoneTerm(func=mdp.cup_tipped, params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 90.0})
    cup_xy_out_of_bounds = DoneTerm(
        func=mdp.cup_xy_displacement_exceeded,
        params={"asset_cfg": SceneEntityCfg("cup"), "max_xy_displacement": 0.10},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class Lift5gLeftEnvCfg(ManagerBasedRLEnvCfg):
    task_name: str = "lift_5g_left_v2"
    curriculum_stage: int = 0
    mask_inactive_arm_actions: bool = True
    grasp2g_target_offset: tuple[float, float, float] = (0.01, -0.06, 0.08)
    reach_dynamic_z_high: float = 0.25
    reach_dynamic_xy_hi: float = 0.10
    reach_dynamic_xy_lo: float = 0.03
    reach_dynamic_xy_gate: float = 0.03
    reach_dynamic_z_descent_rate: float = 0.001
    reach_displacement_free_threshold: float = 0.015
    reach_displacement_suppress_scale: float = 0.03
    reach_switch_threshold: float = 0.05
    reach_switch_hold_steps: int = 2
    reach_soft_gate_near: float = 0.02
    reach_soft_gate_far: float = 0.10
    grasp_switch_threshold: float = 0.025
    grasp_switch_hold_steps: int = 4
    grasp_soft_gate_near: float = 0.012
    grasp_soft_gate_far: float = 0.05
    displacement_penalty_scale: float = 0.02
    displacement_penalty_power: float = 2.0
    displacement_penalty_gate_mix: float = 0.5
    # Cap raw displacement penalty before reward weight is applied.
    # Note: object_displacement is disabled (weight=0.0) in the current stability phase.
    displacement_penalty_max: float = 2.0
    require_filtered_contact_matrix: bool = True
    # Debug controls (default OFF for training performance)
    debug_approach_target_vis: bool = False
    debug_fingertip_vis: bool = False
    debug_fingertip_vis_interval: int = 500
    debug_grasp_quality: bool = False
    debug_grasp_quality_interval: int = 500
    debug_triggers: bool = False
    debug_triggers_interval: int = 500
    debug_reaching: bool = False
    debug_reaching_interval: int = 500

    scene: Lift5gLeftSceneCfg = Lift5gLeftSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.commands.object_pose.debug_vis = False

        self.observations.policy.concatenate_terms = True

        self.sim.physx.bounce_threshold_velocity = 0.01
        # 2048 envs
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 128 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 2**25
        self.sim.physx.gpu_max_rigid_contact_count = 2**25
        self.sim.physx.gpu_collision_stack_size = 2**25
        self.sim.physx.gpu_max_num_partitions = 16
        self.sim.physx.friction_correlation_distance = 0.01
