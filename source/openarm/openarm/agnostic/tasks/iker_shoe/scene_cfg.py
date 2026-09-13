"""Spawn configurations for the IKER shoe scene (design spec §4): env_v1 table, kinematic rack, two shoes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from . import layout

RACK_COLOR = (0.62, 0.45, 0.28)
RACK_MASS_KG = 1.0  # kinematic; the value only satisfies the mass API
LIGHT_INTENSITY = 2500.0
SHOE_MAX_DEPENETRATION_VELOCITY = 1.0


def shoe_usd_path(name: str) -> Path:
    return layout.SHOE_ASSET_DIR / name / f"{name}.usd"


def table_cfg() -> AssetBaseCfg:
    if not layout.TABLE_USD_PATH.is_file():
        raise FileNotFoundError(f"env_v1 table USD not found: {layout.TABLE_USD_PATH}")
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table", spawn=sim_utils.UsdFileCfg(usd_path=str(layout.TABLE_USD_PATH))
    )


def rack_cfg() -> AssetBaseCfg:
    (x0, x1), (y0, y1) = layout.RACK_X_RANGE, layout.RACK_Y_RANGE
    height = layout.RACK_TOP_Z - layout.TABLE_TOP_Z
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/Rack",
        spawn=sim_utils.CuboidCfg(
            size=(x1 - x0, y1 - y0, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=RACK_MASS_KG),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RACK_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=((x0 + x1) / 2, (y0 + y1) / 2, layout.TABLE_TOP_Z + height / 2)),
    )


def shoe_cfg(name: str, position: Sequence[float], quat_wxyz: Sequence[float]) -> RigidObjectCfg:
    usd = shoe_usd_path(name)
    if not usd.is_file():
        raise FileNotFoundError(f"shoe USD not found: {usd} (run scripts/iker/convert_shoe_assets.py)")
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=SHOE_MAX_DEPENETRATION_VELOCITY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(v) for v in position), rot=tuple(float(v) for v in quat_wxyz)
        ),
    )


def light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=LIGHT_INTENSITY))
