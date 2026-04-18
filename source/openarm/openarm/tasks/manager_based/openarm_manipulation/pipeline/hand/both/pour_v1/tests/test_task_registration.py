from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_INIT = _ROOT / "config" / "__init__.py"
_ENV_CFG = _ROOT / "pour_env_cfg.py"
_ENV = _ROOT / "pour_env.py"


def test_pour_v1_task_ids_use_pour_v1_names() -> None:
    config_text = _CONFIG_INIT.read_text(encoding="utf-8")

    for task_id in (
        'id="pour_v1"',
        'id="pour_v1-play"',
        'id="pour_v1-lstm-bc"',
        'id="pour_v1-play-lstm-bc"',
    ):
        assert task_id in config_text, f"missing task registration {task_id}"


def test_pour_v1_entry_points_target_pour_v1_env_module() -> None:
    config_text = _CONFIG_INIT.read_text(encoding="utf-8")

    assert ".pipeline.hand.both.pour_v1" in config_text
    assert ".pour_env:PourEnv" in config_text
    assert 'env_cfg_entry_point": f"{__name__}:PourEnvCfg"' in config_text
    assert 'env_cfg_entry_point": f"{__name__}:PourEnvCfgPlay"' in config_text


def test_pour_v1_keeps_backward_compatible_aliases() -> None:
    env_cfg_text = _ENV_CFG.read_text(encoding="utf-8")
    env_text = _ENV.read_text(encoding="utf-8")

    assert "class PourEnvCfg(" in env_cfg_text
    assert "PourRightEnvCfg = PourEnvCfg" in env_cfg_text
    assert "class PourEnv(" in env_text
    assert "PourRightEnv = PourEnv" in env_text
