"""Direct RL config for the pre_pour_bc warm-start task."""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .pre_pour_bc_scene_cfg import PrePourBCSceneCfg


@configclass
class PrePourBCEnvCfg(DirectRLEnvCfg):
    decimation = 3
    episode_length_s = 13.0
    action_space = 18
    observation_space = 91
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 300.0, render_interval=decimation)
    scene: PrePourBCSceneCfg = PrePourBCSceneCfg(num_envs=64, env_spacing=2.5)

    success_joint_error_threshold: float = 0.20
    target_joint_pos: tuple = (
        -0.042, -0.083, -0.162, 0.862, -0.226, 0.109, 1.009,
        -0.012, -1.012, -0.305, 0.059, 0.010, 0.522, 0.580, 0.612, 0.014,
        -0.001, 0.540, 0.607, 0.640, -0.011, -0.035, 0.568, 0.640, 0.676,
        0.000, -0.041,
        0.154, -0.184, 0.038, 0.759, 0.336, -0.232, -1.206,
    )
    target_force_norm: tuple = (0.08, 0.08, 0.08, 0.08, 0.08)
    target_hand_curl: tuple = (0.65, 0.65, 0.65, 0.65, 0.65)

    rew_joint_target: float = 3.0
    rew_success: float = 10.0
    rew_force: float = 0.4
    rew_curl: float = 0.2
    rew_joint_vel: float = -0.02
    rew_action_rate: float = -0.02

    def __post_init__(self) -> None:
        self.decimation = 3
        self.episode_length_s = 13.0
        self.sim.dt = 1.0 / 300.0
        self.sim.render_interval = self.decimation

        if len(self.target_joint_pos) != 34:
            raise ValueError("target_joint_pos must contain 27 right + 7 left joints")


@configclass
class PrePourBCPlayEnvCfg(PrePourBCEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1

