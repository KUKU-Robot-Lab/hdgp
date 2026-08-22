"""gym 등록 — 자산 쌍마다 id 4개 (train/play × mlp/lstm).

★id 의 로봇 슬롯에 자산 short, side 슬롯에 `b`(bimanual):
      open-<short>_b_pour_fab
  train.py 의 `open-<robot>_<side>_<task>` 규약을 그대로 타므로 수정 없이
  `log/rl_games/open-<short>/b/pour-fab/` 로 로그가 갈린다.

Fabrics 가 없는 쪽 팔이 있는 자산은 등록하지 않는다 — SKIPPED 에 사유를 남긴다
(조용히 빠뜨리지 않음, grasp_lift_fabric 규약).
"""

import gymnasium as gym

from .. import bimanual as _bm
from ..pour_fabric_env_cfg import PourFabricEnvCfg

from . import agents

_ENTRY = "openarm.agnostic.tasks.pour_fabric.pour_fabric_env:PourFabricEnv"

SKIPPED: dict[str, str] = dict(_bm.SKIPPED)
REGISTERED: dict[str, str] = {}


def _cfg_class(pair_name: str, play: bool):
    name = f"PourFabric_{pair_name}{'_PLAY' if play else ''}_Cfg"
    ns = {"pair_name": pair_name}
    if play:
        def __post_init__(self):
            self.scene.num_envs = 50
            PourFabricEnvCfg.__post_init__(self)
        ns["__post_init__"] = __post_init__
    cls = type(name, (PourFabricEnvCfg,), ns)
    globals()[name] = cls        # entry point 가 "모듈:속성" 문자열이라 노출 필요
    return cls


for _short in sorted(_bm.PAIRS):
    _train_cls = _cfg_class(_short, play=False)
    _play_cls = _cfg_class(_short, play=True)
    _base = f"open-{_short}_b_pour_fab"
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
