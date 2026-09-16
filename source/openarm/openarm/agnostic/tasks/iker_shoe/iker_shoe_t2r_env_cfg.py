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
    # spec §3 "종료는 성공·낙하(fall_height 아래)·시간초과 셋" names no value; the base cfg has none either
    # (only IkerShoeGraspEnvCfg does). Reused verbatim from that sibling cfg (drop_z = 0.10, same semantics:
    # env-local shoe height below which the shoe has fallen off the table) rather than inventing a new constant.
    drop_z: float = 0.10


@configclass
class IkerShoeT2rPlayEnvCfg(IkerShoeT2rEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
