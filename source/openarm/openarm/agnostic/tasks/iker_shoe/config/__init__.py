"""gym registration — IKER shoe placement (design spec §3, §7).

★gym id 규약: train.py 의 run_naming 정규식 `^(open-[A-Za-z0-9]+)_([rbl])_(.+?)...` 에 걸려야 로그가
  `log/rl_games/<robot>/<side>/<task>/` 로 분리된다. `-play` id 도 같이 등록한다.
"""

import gymnasium as gym

from . import agents

_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env:IkerShoeEnv"
_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg"

# Stage 2 moves the shoe with the left arm since 2026-09-14 (logs log/rl_games/open-sens/left/iker-shoe/); the right-arm
# K1 runs under right/iker-shoe/ stay as the comparison.
for _suffix, _cfg_name in (("", "IkerShoeEnvCfg"), ("-play", "IkerShoePlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe{_suffix}",
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

# Stage-1 grasp with a t2r-generated reward (spec 2026-09-15-iker-stage1-t2r); logs go to log/rl_games/open-sens/left/iker-shoe-grasp-t2r/.
_GRASP_T2R_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env:IkerShoeGraspT2rEnv"
_GRASP_T2R_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeGraspT2rEnvCfg"), ("-play", "IkerShoeGraspT2rPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_grasp_t2r{_suffix}",
        entry_point=_GRASP_T2R_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_GRASP_T2R_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grasp_ppo_cfg.yaml",
        },
    )

# Stage-2 placement with a t2r-generated reward (spec 2026-09-16-iker-stage2-t2r).
_T2R_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env:IkerShoeT2rEnv"
_T2R_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeT2rEnvCfg"), ("-play", "IkerShoeT2rPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_t2r{_suffix}",
        entry_point=_T2R_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_T2R_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )

# Unified task: one policy picks the shoe up and places it (2026-09-18). 26 actions / 78 observations like stage 1, so
# the stage-1 agent cfg applies; the reward comes from the t2r3 fork.
_UNIFIED_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_unified_env:IkerShoeUnifiedEnv"
_UNIFIED_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_unified_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeUnifiedEnvCfg"), ("-play", "IkerShoeUnifiedPlayEnvCfg")):
    gym.register(
        id=f"open-sens_l_iker_shoe_unified{_suffix}",
        entry_point=_UNIFIED_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_UNIFIED_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grasp_ppo_cfg.yaml",
        },
    )
