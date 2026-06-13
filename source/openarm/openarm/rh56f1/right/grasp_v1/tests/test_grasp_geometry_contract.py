from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _load_preset():
    spec = importlib.util.spec_from_file_location("_rh56f1_grasp_preset", _ROOT / "grasp_right_preset.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    assert preset_spec is not None
    assert preset_spec.loader is not None
    preset = importlib.util.module_from_spec(preset_spec)
    sys.modules[preset_spec.name] = preset
    preset_spec.loader.exec_module(preset)

    const_spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.grasp_right_constants",
        _ROOT / "grasp_right_constants.py",
    )
    assert const_spec is not None
    assert const_spec.loader is not None
    constants = importlib.util.module_from_spec(const_spec)
    sys.modules[const_spec.name] = constants
    const_spec.loader.exec_module(constants)

    return constants, (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")


def test_rh56f1_grasp_pose_biases_non_thumb_closure_before_lift() -> None:
    preset = _load_preset()

    thumb_2 = preset.HAND_GRASP_POSE[1]
    index_curl = preset.HAND_GRASP_POSE[2]
    middle_curl = preset.HAND_GRASP_POSE[3]
    ring_curl = preset.HAND_GRASP_POSE[4]
    little_curl = preset.HAND_GRASP_POSE[5]

    assert thumb_2 <= 0.25
    assert index_curl > ring_curl
    assert middle_curl > little_curl
    assert 1.05 <= index_curl <= 1.10
    assert 1.05 <= middle_curl <= 1.10
    assert 0.80 <= ring_curl <= 0.90
    assert 0.80 <= little_curl <= 0.90


def test_rh56f1_procedural_pregrasp_starts_closer_and_has_recovery_range() -> None:
    _, cfg = _load_constants_and_cfg_text()

    assert "pregrasp_offset_x:     float = -0.045" in cfg
    assert "pregrasp_offset_y:     float = -0.055" in cfg
    assert "pregrasp_offset_z:     float = 0.015" in cfg
    assert "palm_delta_xyz:     float = 0.04" in cfg
    assert "grasp_palm_delta_scale: float = 1.0" in cfg
    assert "grasp_palm_inward_offset: float = 0.025" in cfg


def test_rh56f1_lift_start_requires_success_contact_count() -> None:
    constants, cfg = _load_constants_and_cfg_text()

    assert constants.NUM_FINGERTIPS == 5
    assert "stage0_lift_start_min_contacts: int = 4" in cfg
    assert "grasp_phase_full_grip_contact_threshold: int = 4" in cfg
    assert "grasp_phase_full_grip_progress_threshold: float = 0.65" in cfg
