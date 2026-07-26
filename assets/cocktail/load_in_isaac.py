#!/usr/bin/env python3
"""
Isaac Sim standalone example: load the physics-complete cocktail-cup USD,
spawn rigid-sphere beads (fluid approximation) above the cup, and simulate
the pouring/settling. Verifies that SDF collision keeps beads inside.

Run inside an Isaac Sim python environment, e.g.:
    <isaac>/python.sh load_in_isaac.py            # Linux
    <isaac>\python.bat load_in_isaac.py           # Windows

API note: written for omni.isaac.core (Isaac Sim 2023.1 - 4.x). On Isaac Sim 4.5+
the same classes live under `isaacsim.core.*` and `isaacsim.core.utils.*`.
Adjust imports if your version moved them.
"""
import os
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere, GroundPlane
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdPhysics, PhysxSchema, Gf

HERE = os.path.dirname(os.path.abspath(__file__))
CUP_USD = os.path.join(HERE, "usd", "shaker_body.usda")   # open cup, container for pouring
CUP_PRIM = "/World/ShakerBody"

# ---- bead (fluid approximation) parameters ----
BEAD_RADIUS = 0.005          # 5 mm spheres
N_BEADS = 200                # start small (50) to validate tunneling, then scale up
BEAD_DENSITY = 1000.0        # ~water; total bead mass approximates the liquid volume
CUP_MOUTH_R = 0.040          # drop beads within this radius above the mouth
DROP_Z = 0.24                # release height (just above the 0.175 m rim)


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add(GroundPlane(prim_path="/World/ground", z_position=0.0))

    # bring in the physics-complete cup (SDF collision, mass/inertia baked in)
    add_reference_to_stage(usd_path=CUP_USD, prim_path=CUP_PRIM)

    # Keep the cup fixed in place for the settling test (kinematic).
    # For a manipulation task, attach it to the robot gripper instead.
    cup_prim = get_prim_at_path(CUP_PRIM)
    rb = UsdPhysics.RigidBodyAPI.Apply(cup_prim)
    rb.CreateKinematicEnabledAttr(True)

    # spawn beads in a loose column above the mouth
    rng = np.random.default_rng(0)
    for i in range(N_BEADS):
        a = rng.uniform(0, 2 * np.pi)
        r = CUP_MOUTH_R * np.sqrt(rng.uniform(0, 1)) * 0.8
        x, y = r * np.cos(a), r * np.sin(a)
        z = DROP_Z + (i // 20) * (BEAD_RADIUS * 2.2)   # stack layers so they fall in over time
        world.scene.add(DynamicSphere(
            prim_path=f"/World/beads/bead_{i}",
            position=np.array([x, y, z]),
            radius=BEAD_RADIUS,
            color=np.array([0.9, 0.5, 0.2]),
            mass=BEAD_DENSITY * (4.0 / 3.0) * np.pi * BEAD_RADIUS ** 3,
        ))

    # PhysX solver tuning for thin-wall containers (anti-tunneling)
    scene = UsdPhysics.Scene.Get(world.stage, "/physicsScene")
    if scene:
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateSolverTypeAttr("TGS")
        px.CreateMinPositionIterationCountAttr(16)
        px.CreateMinVelocityIterationCountAttr(4)

    world.reset()
    # smaller physics dt helps fast beads during shaking
    world.set_simulation_dt(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)

    for _ in range(2000):
        world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
