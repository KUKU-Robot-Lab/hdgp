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


def test_episode_uses_state_latched_lift_and_stabilize():
    constants = load_constants()

    assert constants.GRASP_PHASE_STEPS == 600
    assert constants.LIFT_RAISE_PHASE_STEPS == 0
    assert constants.STABILIZE_PHASE_STEPS == 0
    assert constants.EPISODE_STEPS == 600


def test_env_config_documents_ten_second_incremental_control():
    source = CFG_PATH.read_text(encoding="utf-8")

    assert "episode_length_s: float = 10.0" in source
    assert "palm_delta_xyz: float = 0.03" in source
    assert "palm_delta_rot_deg: float = 15.0" in source
    assert "ema_action_alpha: float = 0.7" in source
    assert "transport" not in source
