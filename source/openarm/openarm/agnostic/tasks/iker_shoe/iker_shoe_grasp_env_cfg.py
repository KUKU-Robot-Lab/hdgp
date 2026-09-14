"""Isaac Lab configuration of the IKER shoe stage-1 grasp environment (design 2026-09-14 §4-§6)."""

from __future__ import annotations

from isaaclab.utils import configclass

from .grasp_stage import Stage1RewardCfg
from .iker_shoe_env_cfg import CONTROL_DT, IkerShoeEnvCfg

GRASP_EPISODE_STEPS = 120  # 12 s at 10 Hz: approach from 8-12 cm, close, lift and hold take 5-8 s
ARM_ACTION_DIM = 6
HAND_ACTION_DIM = 20


@configclass
class IkerShoeGraspEnvCfg(IkerShoeEnvCfg):
    """Same scene, physics, noise and startup randomisation as the stage-2 environment; a different task."""

    episode_length_s = GRASP_EPISODE_STEPS * CONTROL_DT
    action_space = ARM_ACTION_DIM + HAND_ACTION_DIM
    observation_space = 38 + 2 * HAND_ACTION_DIM

    # Phase A (user decision 2026-09-14): train with the grasp factor off (g_min 1.0), measure q at the latch and
    # success moments of a checkpoint that lifts (scripts/iker/measure_grasp_quality.py), then resume with
    # `env.grasp_reward.g_min=0.5`. Any g_min < 1 requires the calibration file.
    grasp_reward: Stage1RewardCfg = Stage1RewardCfg(g_min=1.0)

    # empty paths resolve to iker_runs/shoe_place/config_XX/
    pregrasp_bank_path = ""
    quality_calibration_path = ""

    start_noise_xy = 0.02  # legacy IKER interact-object position noise
    drop_z = 0.10  # env-local shoe height below which the shoe has fallen off the table
    hand_floor_offset = 0.01  # the hand-below-table penalty starts this far above the table top
    hand_reset_clamp_max_rad = 0.6  # the profile's open pose may sit this far outside the hand action range
    # Hand joint roles pinned at the profile's open pose: grasp_fj (2026-09-11) saw a free thumb_2 drop the thumb's
    # opposition, and stage-1 phase A (2026-09-14) closed the thumb from the same side as the fingers.
    frozen_hand_joints = ("thumb_2", "pinky_2")

    # grasp_fj disturbance (2.7 N/kg, 0.27 N m/kg) at 10 Hz: the per-step firing probability is x6 of its 60 Hz range
    wrench_force_per_kg = 2.7
    wrench_torque_per_kg = 0.27
    wrench_prob_range = (0.006, 0.6)


@configclass
class IkerShoeGraspPlayEnvCfg(IkerShoeGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
