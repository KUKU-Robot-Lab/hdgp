from __future__ import annotations

import gymnasium as gym

_MODULE = (
    "openarm.tasks.manager_based.openarm_manipulation"
    ".pipeline.hand.both.pour_v1_mimic"
)

gym.register(
    id="Pour-Mimic-V1-v0",
    entry_point=f"{_MODULE}.pour_mimic_managed_env:PourMimicManagedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourMimicManagedEnvCfg",
    },
)

gym.register(
    id="Pour-Mimic-V1-Mimic-v0",
    entry_point=f"{_MODULE}.pour_mimic_managed_env:PourMimicManagedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourMimicManagedMimicEnvCfg",
    },
)


def __getattr__(name: str):
    _managed = {"PourMimicManagedEnvCfg", "PourMimicManagedMimicEnvCfg"}
    _legacy = {"PourMimicEnvCfg", "PourMimicMimicEnvCfg"}

    if name in _managed:
        from .pour_mimic_managed_env_cfg import (
            PourMimicManagedEnvCfg,
            PourMimicManagedMimicEnvCfg,
        )
        return {
            "PourMimicManagedEnvCfg": PourMimicManagedEnvCfg,
            "PourMimicManagedMimicEnvCfg": PourMimicManagedMimicEnvCfg,
        }[name]

    if name in _legacy:
        from .pour_mimic_env_cfg import PourMimicEnvCfg, PourMimicMimicEnvCfg
        return {
            "PourMimicEnvCfg": PourMimicEnvCfg,
            "PourMimicMimicEnvCfg": PourMimicMimicEnvCfg,
        }[name]

    raise AttributeError(name)


__all__ = [
    "PourMimicManagedEnvCfg",
    "PourMimicManagedMimicEnvCfg",
    # legacy — kept for backward compat
    "PourMimicEnvCfg",
    "PourMimicMimicEnvCfg",
]
