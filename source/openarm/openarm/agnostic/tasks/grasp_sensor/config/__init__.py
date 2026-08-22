"""gym 등록 — robot-agnostic grasp-sensor. 로봇당 id 4개(train/play × mlp/lstm).

로봇 추가 = robot_profiles.py 프로필 + 여기 _register() 한 줄.
"""

import gymnasium as gym

from . import agents
from ..grasp_sensor_env_cfg import (
    GraspSensorGripperLeftEnvCfg,
    GraspSensorTesolloRightEnvCfg,
)

_ENTRY = "openarm.agnostic.tasks.grasp_sensor.grasp_sensor_env:GraspSensorEnv"


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


# ★gym id 규약: train.py 의 run_naming 정규식 ^(open-\w+)_([rbl])_(.+)$ 에 걸려야
#   로그가 log/rl_games/<robot>/<side>/<task>/ 로 분리된다. 구 id
#   (open-agn_grasp_sensor_tesollo_r)는 두 번째 슬롯이 r/l/b 가 아니라 매칭에 실패해
#   로그가 pipeline/left/... 로 오분류됐다.
_CFGS = {
    "sens_r": GraspSensorTesolloRightEnvCfg,
    "sens_l": GraspSensorGripperLeftEnvCfg,
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
            id=f"open-{_tag}_grasp_sensor{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
