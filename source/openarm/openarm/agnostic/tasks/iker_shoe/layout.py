"""IKER shoe-placement scene, snapshot settings and evaluation configurations (design spec §4, §5, §9).

Pure Python: importable without Isaac Sim. Positions are in the robot frame (origin on the mount
plate top), metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker.gate import GateConfig, Support
from openarm.agnostic.modules.iker.keypoints import surface_grid
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz, yaw_matrix

HDGP_ROOT = Path(__file__).resolve().parents[6]
ASSETS_DIR = HDGP_ROOT / "assets"
SHOE_ASSET_DIR = ASSETS_DIR / "iker_shoe"
SHOE_META_PATH = SHOE_ASSET_DIR / "shoe_meta.json"
TABLE_USD_PATH = ASSETS_DIR / "simulation_setting" / "env_v1" / "usd" / "env_v1.usda"
RUNS_DIR = HDGP_ROOT / "iker_runs" / "shoe_place"

PROFILE_NAME = "tesollo_right_short"
TASK_INSTRUCTION = "Place the shoe on the rack next to the other shoe."

TABLE_TOP_Z = 0.205
RACK_X_RANGE = (0.11, 0.43)
RACK_Y_RANGE = (-0.33, 0.03)
RACK_TOP_Z = 0.325
RACK_KEYPOINT_INSET = 0.05
RACK_GRID_SHAPE = (3, 4)

MOVING_SHOE = "shoe_move"
OTHER_SHOE = "shoe_other"
RACK = "rack"
OTHER_SHOE_Y = -0.15
OTHER_SHOE_YAW_DEG = 0.0
SPAWN_CLEARANCE_M = 0.002  # drop the shoes from just above their support; the snapshot uses the settled pose

CONFIG_SEED = 20260913
MOVE_X_RANGE = (0.22, 0.32)
MOVE_Y_RANGE = (0.12, 0.16)
MOVE_YAW_DEG_RANGE = (-10.0, 10.0)
OTHER_X_RANGE = (0.25, 0.29)

HEAD_PAN_ENCODER_DEG = -20.0
HEAD_TILT_ENCODER_DEG = -20.0
OVERLAP_MIN_PX = 15.0
MIN_BORDER_MARGIN_PX = 15.0


@dataclass(frozen=True)
class SceneConfig:
    index: int
    move_x: float
    move_y: float
    move_yaw_deg: float
    other_x: float


def config_stream(seed: int = CONFIG_SEED) -> Iterator[SceneConfig]:
    """Configurations drawn in a fixed order; a rejected one is replaced by the next draw."""
    rng = np.random.default_rng(seed)
    index = 0
    while True:
        yield SceneConfig(
            index=index,
            move_x=float(rng.uniform(*MOVE_X_RANGE)),
            move_y=float(rng.uniform(*MOVE_Y_RANGE)),
            move_yaw_deg=float(rng.uniform(*MOVE_YAW_DEG_RANGE)),
            other_x=float(rng.uniform(*OTHER_X_RANGE)),
        )
        index += 1


def sample_configs(count: int, seed: int = CONFIG_SEED) -> tuple[SceneConfig, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    stream = config_stream(seed)
    return tuple(next(stream) for _ in range(count))


def shoe_start_poses(config: SceneConfig, meta: Mapping) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Spawn position and ``wxyz`` quaternion of both shoes for one configuration (before settling)."""
    placements = {
        MOVING_SHOE: ((config.move_x, config.move_y), config.move_yaw_deg, TABLE_TOP_Z),
        OTHER_SHOE: ((config.other_x, OTHER_SHOE_Y), OTHER_SHOE_YAW_DEG, RACK_TOP_Z),
    }
    poses = {}
    for name, ((x, y), yaw_deg, support_z) in placements.items():
        obj = meta["objects"][name]
        rotation = yaw_matrix(math.radians(yaw_deg)) @ np.asarray(obj["rest_rotation"], dtype=float)
        position = np.array([x, y, support_z + float(obj["rest_height"]) + SPAWN_CLEARANCE_M])
        poses[name] = (position, matrix_to_quat_wxyz(rotation))
    return poses


def supports() -> tuple[Support, ...]:
    return (Support(RACK, RACK_TOP_Z, RACK_X_RANGE, RACK_Y_RANGE), Support("table", TABLE_TOP_Z))


def rack_keypoints() -> np.ndarray:
    return surface_grid(RACK_X_RANGE, RACK_Y_RANGE, RACK_TOP_Z, RACK_KEYPOINT_INSET, *RACK_GRID_SHAPE)


def gate_config() -> GateConfig:
    profile = robot_profiles.PROFILES[PROFILE_NAME]
    return GateConfig(
        supports=supports(),
        palm_box_min=tuple(float(v) for v in profile.palm_box_min),
        palm_box_max=tuple(float(v) for v in profile.palm_box_max),
    )
