"""gym registration — IKER shoe placement (design spec §3, §7).

★gym id 규약: train.py 의 run_naming 정규식 `^(open-[A-Za-z0-9]+)_([rbl])_(.+?)...` 에 걸려야 로그가
  `log/rl_games/<robot>/<side>/<task>/` 로 분리된다. `-play` id 도 같이 등록한다.
"""

import gymnasium as gym

from . import agents

_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env:IkerShoeEnv"
_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeEnvCfg"), ("-play", "IkerShoePlayEnvCfg")):
    gym.register(
        id=f"open-sens_r_iker_shoe{_suffix}",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )
