"""Isaac Lab configuration of the IKER stage-1 t2r grasp environment (spec 2026-09-15-iker-stage1-t2r §7).

The parent's scene, observations, actions, terminations, hold/success predicate and success capture, unchanged; the reward
sum comes from ``compute_reward(ctx)`` at ``reward_code_path`` (the launcher passes `env.reward_code_path='<abs path>'`).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .iker_shoe_grasp_env_cfg import IkerShoeGraspEnvCfg


@configclass
class IkerShoeGraspT2rEnvCfg(IkerShoeGraspEnvCfg):
    #: absolute path of the generated reward; empty = zero reward (boot and smoke only). Read in the env's __init__ at run time.
    reward_code_path: str = ""


@configclass
class IkerShoeGraspT2rPlayEnvCfg(IkerShoeGraspT2rEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
