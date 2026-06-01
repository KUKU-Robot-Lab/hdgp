#!/usr/bin/env python3
"""Register Pour-Mimic and delegate to IsaacLab's Robomimic trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


HDGP_ROOT = Path(__file__).resolve().parents[2]
RL_WS_ROOT = HDGP_ROOT.parent
ISAACLAB_ROOT = RL_WS_ROOT / "IsaacLab"
DEFAULT_DATASET = RL_WS_ROOT / "datasets/pour_v1_from_bags_slave_right_robot_object_bc_robomimic.hdf5"
DEFAULT_TASK = "Pour-Mimic"
DEFAULT_ALGO = "bc"
HDGP_LOG_DIR = HDGP_ROOT / "log"


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main() -> None:
    openarm_source = HDGP_ROOT / "source/openarm"
    if str(openarm_source) not in sys.path:
        sys.path.insert(0, str(openarm_source))

    # Importing this module registers Pour-Mimic with gymnasium before
    # IsaacLab's generic Robomimic trainer calls gym.spec(task).
    import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.pour_v1_mimic  # noqa: F401

    passthrough = sys.argv[1:]
    defaults: list[str] = []
    if not _has_option(passthrough, "--task"):
        defaults.extend(["--task", DEFAULT_TASK])
    if not _has_option(passthrough, "--algo"):
        defaults.extend(["--algo", DEFAULT_ALGO])
    if not _has_option(passthrough, "--dataset"):
        defaults.extend(["--dataset", str(DEFAULT_DATASET)])
    if not _has_option(passthrough, "--log_dir"):
        defaults.extend(["--log_dir", str(HDGP_LOG_DIR)])

    trainer = ISAACLAB_ROOT / "scripts/imitation_learning/robomimic/train.py"
    sys.argv = [str(trainer), *defaults, *passthrough]
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
