from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.utils import configclass

from ..pour_v1.pour_constants import NUM_ACTIONS, PREGRASP_FABRICS_STEPS
from ..pour_v1.pour_env_cfg import PourEnvCfg
from ..pour_v1.pour_utils import ACTOR_OBSERVATION_DIM


@configclass
class PourMimicEnvCfg(PourEnvCfg):
    """Evaluation config for the pour_v1 environment exposed through Mimic helpers."""

    episode_length_s: float = 10.0
    scene_num_envs: int = 20

    num_actions: int = NUM_ACTIONS
    num_observations: int = ACTOR_OBSERVATION_DIM
    observation_space: int = ACTOR_OBSERVATION_DIM
    action_space: int = NUM_ACTIONS
    state_space: int = ACTOR_OBSERVATION_DIM
    num_states: int = ACTOR_OBSERVATION_DIM

    pregrasp_fabric_steps: int = PREGRASP_FABRICS_STEPS
    max_pose_angle: float = 30.0
    palm_delta_xyz: float = 0.5
    palm_delta_rot_deg: float = 30.0

    lift_threshold_z: float = 0.45
    align_threshold_xy: float = 0.03
    pour_threshold_tilt_deg: float = 70.0
    grasp_force_threshold: float = 0.5

    enable_bead_count_adr: bool = False
    enable_trajectory_capture: bool = False
    enable_warmstart_reset: bool = False

    def __post_init__(self):
        self.scene.num_envs = self.scene_num_envs
        self.scene.env_spacing = 2.5


@configclass
class PourMimicMimicEnvCfg(PourMimicEnvCfg, MimicEnvCfg):
    """IsaacLab Mimic data-generation config for bimanual pouring."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "Pour-Mimic"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_select_src_per_arm = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 250
        self.datagen_config.seed = 1

        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref="source_cup",
                subtask_term_signal="grasp_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=10,
                apply_noise_during_interpolation=False,
                description="Grasp source cup",
                next_subtask_description="Lift source cup",
            ),
            SubTaskConfig(
                object_ref="source_cup",
                subtask_term_signal="lift_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                apply_noise_during_interpolation=False,
                next_subtask_description="Align source cup above target cup",
            ),
            SubTaskConfig(
                object_ref="target_cup",
                subtask_term_signal="align_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.015,
                num_interpolation_steps=15,
                apply_noise_during_interpolation=False,
                next_subtask_description="Tilt source cup to pour beads",
            ),
            SubTaskConfig(
                object_ref="source_cup",
                subtask_term_signal="pour_done",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                apply_noise_during_interpolation=False,
                description="Pour beads into target cup",
            ),
        ]
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="target_cup",
                subtask_term_signal="grasp_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=10,
                apply_noise_during_interpolation=False,
                description="Hold target cup",
            )
        ]
        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 0), ("right", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            )
        ]
