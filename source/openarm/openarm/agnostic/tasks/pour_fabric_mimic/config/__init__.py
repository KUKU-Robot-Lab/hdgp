"""gym 등록 — 자산 쌍마다 id 4개 (train/play × mlp/lstm). 원본 pour_fabric 규약 그대로.

★id: `open-<short>_b_pour_fab_mimic` (예: open-rh_b_pour_fab_mimic)
  → 로그 `log/rl_games/open-rh/both/pour-fab-mimic/`.
"""

import gymnasium as gym

from .. import bimanual as _bm
from ..pour_fabric_env_cfg import PourFabricMimicEnvCfg

from . import agents

_ENTRY = "openarm.agnostic.tasks.pour_fabric_mimic.pour_fabric_env:PourFabricMimicEnv"

SKIPPED: dict[str, str] = dict(_bm.SKIPPED)
REGISTERED: dict[str, str] = {}


def _cfg_class(pair_name: str, play: bool):
    name = f"PourFabricMimic_{pair_name}{'_PLAY' if play else ''}_Cfg"
    # ★클래스 속성으로 두면 configclass 상속 __init__ 이 베이스 기본값으로 덮는다(원본 주석 참조)
    #   → __post_init__ 에서 인스턴스 속성으로 강제.
    _play = play

    def __post_init__(self, _pn=pair_name, _play=_play):
        self.pair_name = _pn
        if _play:
            self.scene.num_envs = 50
        PourFabricMimicEnvCfg.__post_init__(self)

    cls = type(name, (PourFabricMimicEnvCfg,), {"__post_init__": __post_init__})
    globals()[name] = cls
    return cls


for _short in sorted(_bm.PAIRS):
    _train_cls = _cfg_class(_short, play=False)
    _play_cls = _cfg_class(_short, play=True)
    _base = f"open-{_short}_b_pour_fab_mimic"
    REGISTERED[_short] = _base
    for _suffix, _cls, _yaml in (
        ("", _train_cls, "rl_games_ppo_cfg.yaml"),
        ("-play", _play_cls, "rl_games_ppo_cfg.yaml"),
        ("-lstm", _train_cls, "rl_games_ppo_lstm_cfg.yaml"),
        ("-play-lstm", _play_cls, "rl_games_ppo_lstm_cfg.yaml"),
    ):
        gym.register(
            id=f"{_base}{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cls.__name__}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_yaml}",
            },
        )
