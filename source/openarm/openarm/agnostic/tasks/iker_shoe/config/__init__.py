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

# Stage-1 grasp policy (design 2026-09-14). The side letter is the left arm that grasps; logs go to
# log/rl_games/open-sens/left/iker-shoe-grasp/.
_GRASP_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env:IkerShoeGraspEnv"
_GRASP_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeGraspEnvCfg"), ("-play", "IkerShoeGraspPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_grasp{_suffix}",
        entry_point=_GRASP_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_GRASP_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grasp_ppo_cfg.yaml",
        },
    )
