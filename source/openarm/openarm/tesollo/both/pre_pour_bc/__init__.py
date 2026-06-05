from __future__ import annotations

import gymnasium as gym

_MODULE = "openarm.tesollo.both.pre_pour_bc"

gym.register(
    id="pre_pour_bc",
    entry_point=f"{_MODULE}.pre_pour_bc_env:PrePourBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PrePourBCEnvCfg",
    },
)

gym.register(
    id="pre_pour_bc-play",
    entry_point=f"{_MODULE}.pre_pour_bc_env:PrePourBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PrePourBCPlayEnvCfg",
    },
)


def __getattr__(name: str):
    if name in {"PrePourBCEnvCfg", "PrePourBCPlayEnvCfg"}:
        from .pre_pour_bc_env_cfg import PrePourBCEnvCfg, PrePourBCPlayEnvCfg

        return {
            "PrePourBCEnvCfg": PrePourBCEnvCfg,
            "PrePourBCPlayEnvCfg": PrePourBCPlayEnvCfg,
        }[name]
    raise AttributeError(name)


__all__ = ["PrePourBCEnvCfg", "PrePourBCPlayEnvCfg"]

