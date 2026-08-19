"""gym 등록 — robot-agnostic grasp-lift. 로봇당 id 4개(train/play × mlp/lstm).

로봇 추가 = robot_profiles.py 프로필 + 여기 _register() 한 줄.
"""

import gymnasium as gym

from . import agents
from ..grasp_lift_env_cfg import (
    GraspLiftGripperLeftEnvCfg,
    GraspLiftTesolloRightEnvCfg,
)

_ENTRY = "openarm.agnostic.tasks.grasp_lift.grasp_lift_env:GraspLiftEnv"


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


_CFGS = {
    "tesollo_r": GraspLiftTesolloRightEnvCfg,
    "gripper_l": GraspLiftGripperLeftEnvCfg,
}

for _tag, _cls in _CFGS.items():
    _play_cls = _play(_cls)
    # config entry point 는 "모듈:속성" 문자열 — 동적 클래스를 모듈 네임스페이스에 노출
    globals()[_cls.__name__] = _cls
    globals()[_play_cls.__name__] = _play_cls
    for _suffix, _cfg_name, _agent in (
        ("", _cls.__name__, "rl_games_ppo_cfg.yaml"),
        ("-play", _play_cls.__name__, "rl_games_ppo_cfg.yaml"),
        ("-lstm", _cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        ("-play-lstm", _play_cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
    ):
        gym.register(
            id=f"open-agn_grasp_lift_{_tag}{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
