"""Isaac Lab configuration of the IKER shoe-placement training environment (design spec §7)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules.iker.reward import IkerRewardCfg, reward_cfg_for_start_support

from . import events, layout, robot, scene_cfg

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 12  # 10 Hz policy, as the paper
EPISODE_STEPS = 200
CONTROL_DT = PHYSICS_DT * DECIMATION
FRICTION = 1.0  # contact material shared with make_grasp_bank.py (checked against the bank at boot)
SHOE_REST_QUAT_WXYZ = (0.5, -0.5, 0.5, -0.5)  # spawn only; every reset overwrites the shoe poses


@configclass
class IkerShoeEventCfg:
    """Legacy IKER object randomization (spec §7), sampled once per environment at startup."""

    shoe_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE),
            "mass_distribution_params": (0.3, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    shoe_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE, body_names=".*"),
            # absolute values; with the nominal friction 1.0 this is the legacy x0.3-1.8 scale
            "static_friction_range": (0.3 * FRICTION, 1.8 * FRICTION),
            "dynamic_friction_range": (0.3 * FRICTION, 1.8 * FRICTION),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 250,
        },
    )
    shoe_com = EventTermCfg(
        func=events.randomize_rigid_object_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)},
        },
    )


@configclass
class IkerShoeEnvCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = EPISODE_STEPS * CONTROL_DT
    action_space = 6
    observation_space = 38
    state_space = 0

    # action: normalized delta palm pose per policy step
    action_pos_scale = 0.02
    action_rot_scale = 0.05

    # noise (legacy IKER, training only)
    add_noise = True
    observation_noise = 0.02
    action_noise = 0.05
    quat_noise_rad = 0.2
    other_shoe_pos_noise = 0.01
    other_shoe_yaw_noise = 0.05

    # run artifacts; empty paths resolve to iker_runs/shoe_place/config_XX/
    config_index = 0
    interaction_source = "human"
    interaction_path = ""
    keypoints_path = ""
    grasp_bank_path = ""
    eval_success_distance_m = 0.05

    reward: IkerRewardCfg = reward_cfg_for_start_support(layout.TABLE_TOP_Z, max_episode_length=EPISODE_STEPS)
    events: IkerShoeEventCfg = IkerShoeEventCfg()

    viewer: ViewerCfg = ViewerCfg(eye=(1.1, 0.9, 0.85), lookat=(0.25, -0.02, 0.3), origin_type="env", env_index=0)
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        # 4096 envs overflow the default PhysX buffers and drop contacts (measured 2026-09-13)
        physx=PhysxCfg(gpu_collision_stack_size=2**29, gpu_max_rigid_patch_count=2**23, gpu_max_rigid_contact_count=2**24),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = robot.robot_cfg()
    shoe_move_cfg: RigidObjectCfg = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (0.27, 0.14, 0.26), SHOE_REST_QUAT_WXYZ)
    shoe_other_cfg: RigidObjectCfg = scene_cfg.shoe_cfg(layout.OTHER_SHOE, (0.27, -0.15, 0.38), SHOE_REST_QUAT_WXYZ)
    table_cfg: AssetBaseCfg = scene_cfg.table_cfg()
    rack_cfg: AssetBaseCfg = scene_cfg.rack_cfg()
    light_cfg: AssetBaseCfg = scene_cfg.light_cfg()


@configclass
class IkerShoePlayEnvCfg(IkerShoeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
