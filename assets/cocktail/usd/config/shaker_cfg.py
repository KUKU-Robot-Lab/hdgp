"""Isaac Lab asset configuration for the cocktail-shaker USD assets.

Usage:
    from shaker_cfg import SHAKER_BODY_CFG, SHAKER_ASSEMBLED_CFG
    cup = RigidObject(SHAKER_BODY_CFG.replace(prim_path="/World/Cup"))

The referenced USDs already contain rigid-body + SDF-mesh collision + mass/inertia,
so these Cfgs mostly point at the file and (optionally) restate solver-friendly props.
Import paths target Isaac Lab (isaaclab.*, formerly omni.isaac.lab.*).
"""
import os

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
except ImportError:  # older Isaac Lab / Orbit
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import RigidObjectCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_USD_DIR = os.path.abspath(os.path.join(_HERE, ".."))

SHAKER_BODY_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ShakerBody",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_USD_DIR, "shaker_body.usda"),
        # USD already declares mass/inertia + SDF collision; override here only if needed.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

SHAKER_LID_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ShakerLid",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_USD_DIR, "shaker_lid.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    # staging pose; move to (0,0,0.157) in the body frame to close.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.0)),
)

SHAKER_CAP_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ShakerCap",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_USD_DIR, "shaker_cap.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.0)),
)

# Pre-welded closed shaker (references the 3 parts + fixed joints). Use for the
# SHAKING phase when you want the sealed shaker to move as one body.
SHAKER_ASSEMBLED_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shaker",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_USD_DIR, "shaker_assembled.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

# --- assembly / mate transforms (body frame, meters) ---
# All origins are at COM, so a mate pose = (part_COM - body_COM).
LID_MATE_POS = (0.0, 0.0, 0.09126)   # place lid here (in body frame) to close
CAP_MATE_POS = (0.0, 0.0, 0.12815)   # place cap here (in body frame)
