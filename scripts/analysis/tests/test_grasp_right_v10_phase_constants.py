from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CONSTANTS_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "grasp_right_constants.py"
)
CFG_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "grasp_right_env_cfg.py"
)


def load_constants():
    spec = importlib.util.spec_from_file_location(
        "openarm.tesollo.right.grasp_v10_3.grasp_right_constants",
        CONSTANTS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_episode_splits_raise_and_stabilize_after_grasp():
    constants = load_constants()

    assert constants.GRASP_PHASE_STEPS == 480
    assert constants.LIFT_RAISE_PHASE_STEPS == 120
    assert constants.STABILIZE_PHASE_STEPS == 240
    assert constants.LIFT_START_STEP == 480
    assert constants.STABILIZE_START_STEP == 600
    assert constants.EPISODE_STEPS == 840
    assert constants.LIFT_Z_DELTA == 0.05


def test_env_config_documents_fourteen_second_episode_and_stabilize_bounds():
    source = CFG_PATH.read_text(encoding="utf-8")

    assert "episode_length_s: float = 14.0" in source
    assert "stabilize_palm_delta_xyz: float = 0.01" in source
    assert "stabilize_palm_delta_rot_deg: float = 10.0" in source
