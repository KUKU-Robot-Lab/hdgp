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

"""Environment config for bi_pouring_v1.

Training-facing v1 scope:
  - direct env patterned after 5g_grasp_right_v5
  - right-arm-only policy control
  - fixed left-side target holder
  - rigid cup-to-EE attachment on both sides
  - single-bead transfer proxy

Future left-holder randomization is intentionally kept as explicit config hooks
instead of adding a broader curriculum abstraction in v1.
"""

from dataclasses import field
import os as _os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .bi_pouring_constants import NUM_ACTIONS, NUM_OBSERVATIONS, NUM_STATES
from .bi_pouring_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_TARGET_CUP_ATTACH_FRAME_NAME,
    LEFT_TARGET_CUP_ATTACH_POS_B,
    LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B,
    LEFT_HOLDER_FIXED_JOINT_POS,
    LEFT_HOLDER_JOINT_NAMES,
    RIGHT_HAND_GRASP_JOINT_POS,
    SOURCE_CUP_POUR_AXIS_B,
    SOURCE_CUP_POUR_POINT_POS_B,
    SOURCE_CUP_UP_AXIS_B,
    TARGET_CUP_OPENING_POS_B,
    TARGET_CUP_UP_AXIS_B,
    RIGHT_SOURCE_CUP_ATTACH_FRAME_NAME,
    RIGHT_SOURCE_CUP_ATTACH_POS_B,
    RIGHT_SOURCE_CUP_ATTACH_QUAT_WXYZ_B,
    RIGHT_ARM_POUR_READY_POSE,
    RIGHT_POLICY_ARM_JOINT_NAMES,
)

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class BiPouringEnvCfg(DirectRLEnvCfg):
    """Initial direct RL config for bi_pouring_v1."""

    episode_length_s: float = 6.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_max_objects_per_env: int = 4
    use_cuda_graph: bool = False

    observation_space: int = NUM_OBSERVATIONS
    action_space: int = NUM_ACTIONS
    state_space: int = NUM_STATES

    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_STATES

    action_scale: float = 0.25

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**22,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_tesollo_sensor/openarm_tesollo_sensor.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                "openarm_right_joint1": RIGHT_ARM_POUR_READY_POSE[0],
                "openarm_right_joint2": RIGHT_ARM_POUR_READY_POSE[1],
                "openarm_right_joint3": RIGHT_ARM_POUR_READY_POSE[2],
                "openarm_right_joint4": RIGHT_ARM_POUR_READY_POSE[3],
                "openarm_right_joint5": RIGHT_ARM_POUR_READY_POSE[4],
                "openarm_right_joint6": RIGHT_ARM_POUR_READY_POSE[5],
                "openarm_right_joint7": RIGHT_ARM_POUR_READY_POSE[6],
                **RIGHT_HAND_GRASP_JOINT_POS,
                **LEFT_HOLDER_FIXED_JOINT_POS,
            },
        ),
        actuators={
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=400.0,
                damping=80.0,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=400.0,
                damping=80.0,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=400.0,
                damping=80.0,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    right_source_cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RightSourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42, -0.18, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big.usd"),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    left_target_cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/LeftTargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42, 0.18, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup/cup_big.usd"),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    bead_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bead",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42, -0.18, 0.38],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "bead", "bead.usd"),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=5.0,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        ),
    )

    policy_arm_joint_names: list[str] = field(default_factory=lambda: list(RIGHT_POLICY_ARM_JOINT_NAMES))
    left_holder_joint_names: list[str] = field(default_factory=lambda: list(LEFT_HOLDER_JOINT_NAMES))
    right_source_cup_attach_frame_name: str = RIGHT_SOURCE_CUP_ATTACH_FRAME_NAME
    right_source_cup_attach_pos_b: tuple[float, float, float] = tuple(RIGHT_SOURCE_CUP_ATTACH_POS_B)
    right_source_cup_attach_quat_wxyz_b: tuple[float, float, float, float] = tuple(
        RIGHT_SOURCE_CUP_ATTACH_QUAT_WXYZ_B
    )
    left_target_cup_attach_frame_name: str = LEFT_TARGET_CUP_ATTACH_FRAME_NAME
    left_target_cup_attach_pos_b: tuple[float, float, float] = tuple(LEFT_TARGET_CUP_ATTACH_POS_B)
    left_target_cup_attach_quat_wxyz_b: tuple[float, float, float, float] = tuple(
        LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B
    )
    bead_count: int = 1
    bead_spawn_pos_source_cup_b: tuple[float, float, float] = tuple(BEAD_SPAWN_POS_SOURCE_CUP_B)
    bead_spawn_quat_source_cup_wxyz: tuple[float, float, float, float] = tuple(BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ)
    enable_right_arm_init_noise: bool = True
    right_arm_init_joint_noise_abs: tuple[float, ...] = (0.015, 0.020, 0.020, 0.025, 0.015, 0.010, 0.010)
    source_cup_pour_point_pos_b: tuple[float, float, float] = tuple(SOURCE_CUP_POUR_POINT_POS_B)
    target_cup_opening_pos_b: tuple[float, float, float] = tuple(TARGET_CUP_OPENING_POS_B)
    source_cup_pour_axis_b: tuple[float, float, float] = tuple(SOURCE_CUP_POUR_AXIS_B)
    source_cup_up_axis_b: tuple[float, float, float] = tuple(SOURCE_CUP_UP_AXIS_B)
    target_cup_up_axis_b: tuple[float, float, float] = tuple(TARGET_CUP_UP_AXIS_B)
    stable_alignment_xy_threshold: float = 0.040
    stable_alignment_z_min: float = 0.010
    stable_alignment_z_max: float = 0.120
    approach_xy_scale: float = 0.25
    alignment_xy_scale: float = 0.12
    alignment_z_target: float = 0.040
    alignment_z_scale: float = 0.035
    controlled_tilt_target_cos: float = 0.55
    controlled_tilt_cos_scale: float = 0.18
    approach_upright_min_cos: float = 0.92
    target_opening_capture_radius: float = 0.050
    source_cup_capture_radius: float = 0.060
    target_inner_radius: float = 0.050
    target_inside_z_min: float = -0.015
    target_inside_z_max: float = 0.095
    source_inner_radius: float = 0.055
    source_inside_z_min: float = -0.020
    source_inside_z_max: float = 0.110
    target_entry_z_max: float = 0.000
    stable_retention_speed_threshold: float = 0.35
    success_retention_steps: int = 100
    bead_spill_z_threshold: float = 0.220
    major_spill_xy_radius: float = 0.100
    major_spill_z_margin: float = -0.040
    rim_clearance_threshold: float = 0.035
    ee_clearance_threshold: float = 0.120
    cup_to_opposite_ee_clearance_threshold: float = 0.110
    invalid_cup_xy_threshold: float = 0.300
    invalid_cup_z_threshold: float = 0.250
    invalid_bead_drop_z_threshold: float = 0.230
    invalid_bead_floor_z_threshold: float = 0.050
    invalid_bead_xy_threshold: float = 0.800
    reward_approach_weight: float = 1.5
    reward_alignment_weight: float = 1.5
    reward_controlled_tilt_weight: float = 0.75
    reward_bead_entry_weight: float = 6.0
    reward_stable_retention_weight: float = 2.5
    penalty_spill_weight: float = 4.0
    penalty_collision_weight: float = 1.0
    penalty_action_smoothness_weight: float = 0.05

    # Left-holder reset seam aligned with 5g_grasp_right_v5 reset-fabric usage.
    # bi_pouring_v1 still defaults to the fixed holder path because this codebase
    # currently does not provide a reusable left-side FABRICS pose sampler.
    use_left_holder_reset_fabric: bool = False
    pregrasp_fabric_steps: int = 60
    reset_fabric_chunk_size: int = 128
    left_holder_init_pose_sampler: str = "fixed"
    left_holder_init_pos_noise_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_holder_init_rot_noise_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


class BiPouringEnvCfg_PLAY(BiPouringEnvCfg):
    """Play config with fewer environments."""

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
