from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _load_preset():
    spec = importlib.util.spec_from_file_location("_rh56f1_grasp_preset", _ROOT / "grasp_right_preset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_constants_and_cfg_text() -> tuple[types.ModuleType, str]:
    pkg = types.ModuleType("_rh56f1_grasp_pkg")
    pkg.__path__ = [str(_ROOT)]
    sys.modules[pkg.__name__] = pkg

    preset_spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.grasp_right_preset",
        _ROOT / "grasp_right_preset.py",
    )
    preset = importlib.util.module_from_spec(preset_spec)
    sys.modules[preset_spec.name] = preset
    assert preset_spec.loader is not None
    preset_spec.loader.exec_module(preset)

    const_spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.grasp_right_constants",
        _ROOT / "grasp_right_constants.py",
    )
    constants = importlib.util.module_from_spec(const_spec)
    sys.modules[const_spec.name] = constants
    assert const_spec.loader is not None
    const_spec.loader.exec_module(constants)

    return constants, (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")


def test_rh56f1_grasp_pose_biases_non_thumb_closure_before_lift() -> None:
    preset = _load_preset()

    thumb_2 = preset.HAND_GRASP_POSE[1]
    non_thumb_curl = preset.HAND_GRASP_POSE[2:]

    assert thumb_2 <= 0.25
    assert min(non_thumb_curl) >= 0.85
    assert max(non_thumb_curl) <= 1.05


def test_rh56f1_procedural_pregrasp_starts_closer_and_has_recovery_range() -> None:
    _, cfg = _load_constants_and_cfg_text()

    assert "pregrasp_offset_x:     float = -0.045" in cfg
    assert "pregrasp_offset_y:     float = -0.055" in cfg
    assert "pregrasp_offset_z:     float = 0.015" in cfg
    assert "palm_delta_xyz:     float = 0.08" in cfg


def test_rh56f1_lift_start_requires_success_contact_count() -> None:
    constants, cfg = _load_constants_and_cfg_text()

    assert constants.MIN_CONTACTS_FOR_SUCCESS == 3
    assert "stage0_lift_start_min_contacts: int = 3" in cfg
