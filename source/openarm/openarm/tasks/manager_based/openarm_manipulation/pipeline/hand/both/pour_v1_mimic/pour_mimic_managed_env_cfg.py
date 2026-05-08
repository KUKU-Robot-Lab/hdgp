"""ManagerBased + MimicEnv configuration for pour_v1_mimic.

Two configs are exported:
  PourMimicManagedEnvCfg       — plain RL env (for eval / single-step rollout)
  PourMimicManagedMimicEnvCfg  — Mimic data-generation env (subtask_configs wired)
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .fabrics_action_term import FabricsRightArmActionTerm, FabricsRightArmActionTermCfg
from .left_arm_action_term import LeftArmDeltaActionTerm, LeftArmDeltaActionTermCfg
from .pour_mimic_obs_cfg import PourMimicObservationsCfg
from .pour_mimic_scene_cfg import PourMimicSceneCfg


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class PourMimicActionsCfg:
    """18D action space split into two ActionTerms."""

    right_arm_hand: FabricsRightArmActionTermCfg = FabricsRightArmActionTermCfg(
        class_type=FabricsRightArmActionTerm,
        asset_name="robot",
    )
    left_arm: LeftArmDeltaActionTermCfg = LeftArmDeltaActionTermCfg(
        class_type=LeftArmDeltaActionTerm,
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


def _check_prepour_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Annotation-quality check using right palm kinematics (sim replay compatible).

    Sim replay does not physically grasp the cup, so cup position stays on the
    table. Instead we check that the right palm ended above the table AND moved
    toward the target cup — both are achievable from action replay alone.
    """
    robot = env.scene["robot"]
    palm_idx = None
    for candidate in ("rl_dg_palm", "right_hand", "openarm_right_hand"):
        try:
            palm_idx = robot.data.body_names.index(candidate)
            break
        except ValueError:
            continue
    if palm_idx is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    palm_pos = robot.data.body_pos_w[:, palm_idx, :]
    tgt = env.scene["target_cup"].data.root_pos_w
    xy_dist = (palm_pos[:, :2] - tgt[:, :2]).norm(dim=-1)
    palm_lifted = palm_pos[:, 2] > 0.28  # arm raised above table top
    return palm_lifted & (xy_dist <= 0.30)  # arm moved toward target cup


@configclass
class PourMimicTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_check_prepour_success)


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
    grasp_force_threshold: float = 0.05  # N  (HDF5 shows max ~0.146 at grasp)
    left_lift_threshold_z: float = 0.325 # m  (left EEF: starts ~0.309, peaks ~0.345 during lift)
    lift_threshold_z: float = 0.30       # m  (right palm; cup spawns at z=0.277)
    align_threshold_xy: float = 0.05     # m  (kept for backward compat; not used by align_done)
    align_threshold_z: float = 0.43      # m  (right palm Z; fires after lift, in pre-pour phase)
    pour_threshold_tilt_deg: float = 70.0

    def __post_init__(self) -> None:
        self.decimation = 3              # 1/300 × 3 = 100 Hz — matches HDF5
        self.episode_length_s = 13.0    # HDF5 demo is 11.67 s; add buffer
        self.sim.dt = 1.0 / 300.0       # Fabrics fabric_dt unchanged
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

        # Left arm: grasp target cup → lift target cup
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="target_cup",
                subtask_term_signal="left_grasp_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm grasp target cup",
            ),
            SubTaskConfig(
                object_ref="target_cup",
                subtask_term_signal="left_lift_done",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                apply_noise_during_interpolation=False,
                description="Left arm lift target cup",
            ),
        ]

        # Right arm: grasp source cup → lift source cup → align over target cup
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
                description="Right arm grasp source cup",
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
                description="Right arm lift source cup",
            ),
            SubTaskConfig(
                object_ref="target_cup",
                subtask_term_signal="align_done",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.015,
                num_interpolation_steps=15,
                apply_noise_during_interpolation=False,
                description="Align source cup over target cup",
            ),
        ]

        # left[1] (left_lift) must complete before right[0] (grasp) starts
        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 1), ("right", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            )
        ]
