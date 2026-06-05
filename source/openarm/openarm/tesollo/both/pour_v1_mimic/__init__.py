from __future__ import annotations

import gymnasium as gym

_MODULE = "openarm.tesollo.both.pour_v1_mimic"

_ROBOMIMIC_BC_CFG = f"{__name__}.config.agents:robomimic/bc_rnn_policy.json"


def _register(id: str, env_cfg_entry_point: str) -> None:
    gym.register(
        id=id,
        entry_point=f"{_MODULE}.pour_mimic_managed_env:PourMimicManagedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "robomimic_bc_cfg_entry_point": _ROBOMIMIC_BC_CFG,
        },
    )


_register("Pour-Mimic", f"{__name__}:PourMimicManagedEnvCfg")
_register("Pour-Mimic-Mimic", f"{__name__}:PourMimicManagedMimicEnvCfg")

# Backward-compatible aliases for old datasets and notes.
_register("Pour-Mimic-V1-v0", f"{__name__}:PourMimicManagedEnvCfg")
_register("Pour-Mimic-V1-Mimic-v0", f"{__name__}:PourMimicManagedMimicEnvCfg")


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
