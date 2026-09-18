"""cfg of the IKER unified environment: one policy picks the shoe up and places it on the rack (2026-09-18).

Same scene, physics, noise and domain randomisation as stage 1 and stage 2; one longer episode that contains both
halves, and the generated reward of the t2r3 fork.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .iker_shoe_env_cfg import CONTROL_DT
from .iker_shoe_grasp_t2r_env_cfg import IkerShoeGraspT2rEnvCfg
from .place_stage import PlaceRewardCfg

#: 36 s at 10 Hz. Stage 1 needed 120 steps for the grasp alone and stage 2 200 for the placement; the unified episode
#: holds both plus the scripted return.
UNIFIED_EPISODE_STEPS = 360


@configclass
class IkerShoeUnifiedEnvCfg(IkerShoeGraspT2rEnvCfg):
    episode_length_s = UNIFIED_EPISODE_STEPS * CONTROL_DT
    place: PlaceRewardCfg = PlaceRewardCfg()

    # Every episode starts from the robot's rest posture with the hand open and the shoe lying on the table; only its
    # position is drawn (user decision 2026-09-18: orientation randomisation comes later). The offset is applied to the
    # recorded rest pose of the pre-grasp bank entry the episode drew.
    spawn_noise_xy = 0.06

    # A fraction of episodes starts from a recorded later state instead, so the second half of the task is practised
    # before the first half works: `held_start_frac` from the stage-1 grasp bank (shoe already in the hand),
    # `adjust_start_frac` from the stage-2 adjust bank (shoe already set down off target). "" disables a bank.
    grasp_bank_start_path: str = ""
    held_start_frac: float = 0.0
    adjust_bank_path: str = ""
    adjust_start_frac: float = 0.0


@configclass
class IkerShoeUnifiedPlayEnvCfg(IkerShoeUnifiedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 25
        self.add_noise = False
