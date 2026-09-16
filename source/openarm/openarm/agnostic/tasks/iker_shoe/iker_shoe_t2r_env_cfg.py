"""cfg of the stage-2 t2r environment (spec 2026-09-16-iker-stage2-t2r §4, §6)."""
from __future__ import annotations

from isaaclab.utils import configclass

from .iker_shoe_env_cfg import IkerShoeEnvCfg
from .place_stage import PlaceRewardCfg


@configclass
class IkerShoeT2rEnvCfg(IkerShoeEnvCfg):
    action_space = 7        # 6 palm delta + 1 grip axis
    observation_space = 39  # the stage-2 38 + the filtered grip state
    reward_code_path: str = ""   # "" = zero reward (boot and random rollouts only)
    place: PlaceRewardCfg = PlaceRewardCfg()
    # fix round 1: spec §3's "낙하(fall_height 아래)" is not a new threshold — this env already inherits one
    # from the base cfg's `reward: IkerRewardCfg = reward_cfg_for_start_support(layout.TABLE_TOP_Z, ...)`
    # (`cfg.reward.fall_height`, env-local frame). A separate `drop_z` field here would disagree with it (0.10
    # vs 0.168) and break the promised baseline comparison, so the env reads `cfg.reward.fall_height` directly
    # instead of a field on this cfg — one source of truth.


@configclass
class IkerShoeT2rPlayEnvCfg(IkerShoeT2rEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
