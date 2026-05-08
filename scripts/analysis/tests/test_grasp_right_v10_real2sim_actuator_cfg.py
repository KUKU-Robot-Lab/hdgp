import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_grasp_right_v10/real2sim_actuator_cfg.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("real2sim_actuator_cfg", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_real2sim_calibration_accepts_group_values(tmp_path):
    module = _load_module()
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "groups": {
                    "openarm_right_arm": {
                        "stiffness": 650.0,
                        "damping": 95.0,
                        "effort_limit": 40.0,
                        "velocity_limit": 5.0,
                        "joint_friction": 0.4,
                    }
                },
            }
        )
    )

    calibration = module.load_real2sim_calibration(path)

    assert calibration["openarm_right_arm"].stiffness == 650.0
    assert calibration["openarm_right_arm"].damping == 95.0
    assert calibration["openarm_right_arm"].effort_limit == 40.0
    assert calibration["openarm_right_arm"].velocity_limit == 5.0
    assert calibration["openarm_right_arm"].joint_friction == 0.4


def test_get_actuator_params_falls_back_to_defaults():
    module = _load_module()

    params = module.get_actuator_params(
        "tesollo_hand_curl",
        calibration={},
        default_stiffness=30.0,
        default_damping=5.0,
    )

    assert params == {"stiffness": 30.0, "damping": 5.0}
