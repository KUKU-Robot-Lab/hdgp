"""ManagerBased + MimicEnv configuration for pour_v1_mimic.

Two configs are exported:
  PourMimicManagedEnvCfg       — plain RL env (for eval / single-step rollout)
  PourMimicManagedMimicEnvCfg  — Mimic data-generation env (subtask_configs wired)
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionGroupCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .fabrics_action_term import FabricsRightArmActionTermCfg
from .left_arm_action_term import LeftArmDeltaActionTermCfg
from .pour_mimic_obs_cfg import PourMimicObservationsCfg
from .pour_mimic_scene_cfg import PourMimicSceneCfg


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class PourMimicActionsCfg:
    """18D action space split into two ActionTerms."""

    right_arm_hand: FabricsRightArmActionTermCfg = FabricsRightArmActionTermCfg(
        asset_name="robot",
    )
    left_arm: LeftArmDeltaActionTermCfg = LeftArmDeltaActionTermCfg(
        asset_name="robot",
    )


# ---------------------------------------------------------------------------
# Events (reset)
# ---------------------------------------------------------------------------


@configclass
class PourMimicEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class PourMimicTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Base env cfg
# ---------------------------------------------------------------------------


@configclass
class PourMimicManagedEnvCfg(ManagerBasedRLEnvCfg):
    """ManagerBased eval config for bimanual pour Mimic env."""

    scene: PourMimicSceneCfg = PourMimicSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PourMimicObservationsCfg = PourMimicObservationsCfg()
    actions: PourMimicActionsCfg = PourMimicActionsCfg()
    terminations: PourMimicTerminationsCfg = PourMimicTerminationsCfg()
    events: PourMimicEventsCfg = PourMimicEventsCfg()

    commands = None
    rewards = None
    curriculum = None

    # Subtask thresholds (used by PourMimicManagedEnv.get_subtask_term_signals)
    grasp_force_threshold: float = 0.5   # N
    lift_threshold_z: float = 0.45       # m (world z)
    align_threshold_xy: float = 0.03     # m (mouth XY distance)
    pour_threshold_tilt_deg: float = 70.0

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 300.0   # matches Fabrics fabric_dt × decimation
        self.sim.render_interval = self.decimation


# ---------------------------------------------------------------------------
# Mimic data-generation cfg
# ---------------------------------------------------------------------------


@configclass
class PourMimicManagedMimicEnvCfg(PourMimicManagedEnvCfg, MimicEnvCfg):
    """IsaacLab Mimic data-generation config for bimanual pouring."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20

        self.datagen_config.name = "Pour-Mimic-V1"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 250
        self.datagen_config.seed = 1

        # Right arm: grasp → lift → align → pour
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

        # Left arm: hold target cup
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
            ),
        ]

        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 0), ("right", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            )
        ]
