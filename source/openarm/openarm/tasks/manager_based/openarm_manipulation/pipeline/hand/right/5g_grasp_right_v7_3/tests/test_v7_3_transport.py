from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import torch
import yaml


TASK_DIR = Path(__file__).resolve().parents[1]
OPENARM_SOURCE = TASK_DIR.parents[7]
PACKAGE = (
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right."
    "5g_grasp_right_v7_3"
)

if str(OPENARM_SOURCE) not in sys.path:
    sys.path.insert(0, str(OPENARM_SOURCE))


def test_constants_match_v7_3_contract() -> None:
    constants = importlib.import_module(f"{PACKAGE}.grasp_right_constants")

    assert constants.NUM_ACTIONS == 11
    assert constants.NUM_OBSERVATIONS == 110
    assert constants.NUM_CRITIC_OBSERVATIONS == 146
    assert constants.GRASP_PHASE_STEPS == 480
    assert constants.LIFT_START_STEP == 480
    assert constants.STABILIZE_START_STEP == 720
    assert constants.TRANSPORT_START_STEP == 840
    assert constants.EPISODE_STEPS == 1080


def test_transport_success_requires_all_conditions() -> None:
    utils = importlib.import_module(f"{PACKAGE}.grasp_right_utils")

    lifted = torch.tensor([True, False, True, True, True])
    grasped = torch.tensor([True, True, False, True, True])
    upright = torch.tensor([True, True, True, False, True])
    goal_dist = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.08])

    success = utils.compute_transport_success_mask(
        lifted=lifted,
        grasped=grasped,
        upright=upright,
        goal_dist=goal_dist,
        goal_dist_threshold=0.04,
    )

    assert success.tolist() == [True, False, False, False, False]


def test_lstm_yaml_uses_v7_3_entry_and_no_warm_start() -> None:
    path = TASK_DIR / "config/agents/rl_games_ppo_lstm_cfg.yaml"
    data = yaml.safe_load(path.read_text())

    assert data["params"]["config"]["name"] == "5g_grasp_right-v7-3-lstm"
    assert data["params"]["load_checkpoint"] is False
    assert data["params"]["load_path"] == ""
    assert data["params"]["network"]["rnn"]["name"] == "lstm"
    central_network = data["params"]["config"]["central_value_config"]["network"]
    assert central_network["mlp"]["units"] == [1024, 512]
    assert "rnn" not in central_network


def test_config_registers_expected_task_ids_static() -> None:
    source = (TASK_DIR / "config/__init__.py").read_text()

    for task_id in (
        "5g_grasp_right-v7-3",
        "5g_grasp_right-play-v7-3",
        "5g_grasp_right-v7-3-lstm",
        "5g_grasp_right-play-v7-3-lstm",
    ):
        assert f'id="{task_id}"' in source
    assert "rl_games_ppo_lstm_cfg.yaml" in source


def _float_assignment(source: str, name: str) -> float:
    match = re.search(rf"^\s*{name}:\s*float\s*=\s*([-+]?[0-9]*\.?[0-9]+)", source, re.MULTILINE)
    assert match is not None, name
    return float(match.group(1))


def test_transport_goal_defaults_are_inside_workspace_by_static_config() -> None:
    source = (TASK_DIR / "grasp_right_env_cfg.py").read_text()

    transport_goal_y_center = _float_assignment(source, "transport_goal_y_center")
    object_spawn_y_center = _float_assignment(source, "object_spawn_y_center")
    transport_goal_x_center = _float_assignment(source, "transport_goal_x_center")
    obj_out_x_min = _float_assignment(source, "obj_out_x_min")
    obj_out_x_max = _float_assignment(source, "obj_out_x_max")
    obj_out_y_min = _float_assignment(source, "obj_out_y_min")
    obj_out_y_max = _float_assignment(source, "obj_out_y_max")
    lift_success_height = _float_assignment(source, "lift_success_height")
    transport_goal_dist_threshold = _float_assignment(source, "transport_goal_dist_threshold")
    constants = importlib.import_module(f"{PACKAGE}.grasp_right_constants")

    assert transport_goal_y_center > object_spawn_y_center
    assert obj_out_x_min < transport_goal_x_center < obj_out_x_max
    assert obj_out_y_min < transport_goal_y_center < obj_out_y_max
    assert "transport_goal_z_offset: float = LIFT_Z_DELTA" in source
    assert constants.LIFT_Z_DELTA >= lift_success_height
    assert transport_goal_dist_threshold == 0.04
