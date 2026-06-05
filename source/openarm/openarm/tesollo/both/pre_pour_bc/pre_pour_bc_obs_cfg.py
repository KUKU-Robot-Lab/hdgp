"""Observation utilities for pre_pour_bc.

Policy obs is fixed to the recorded 91D actor_obs contract:
right joint pos/vel (54D), left joint pos/vel (14D), fingertip force norm
(5D), and previous action (18D).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.observations import joint_pos, joint_vel, last_action
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


RIGHT_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "openarm_right_joint[1-7]",
        "rj_dg_[1-5]_[1-4]",
    ],
)
LEFT_JOINT_CFG = SceneEntityCfg("robot", joint_names=["openarm_left_joint[1-7]"])

_CONTACT_FORCE_MAX = 10.0
_TIP_SENSOR_NAMES = ("tip1_sensor", "tip2_sensor", "tip3_sensor", "tip4_sensor", "tip5_sensor")


def tip_force_norm(env: "ManagerBasedEnv") -> torch.Tensor:
    forces = []
    for name in _TIP_SENSOR_NAMES:
        sensor: ContactSensor = env.scene[name]
        force = sensor.data.force_matrix_w[:, 0, 0, :].norm(dim=-1, keepdim=True)
        forces.append(force)
    return (torch.cat(forces, dim=-1) / _CONTACT_FORCE_MAX).clamp(0.0, 1.0)


def pre_pour_actor_obs(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return the 91D actor obs in the same order as the HDF5 dataset."""
    robot = env.scene["robot"]
    right_ids, _ = robot.find_joints(RIGHT_JOINT_CFG.joint_names)
    left_ids, _ = robot.find_joints(LEFT_JOINT_CFG.joint_names)
    prev = getattr(env, "actions", None)
    if prev is None:
        prev = torch.zeros(env.num_envs, 18, device=env.device)
    return torch.cat(
        (
            robot.data.joint_pos[:, right_ids],
            robot.data.joint_vel[:, right_ids],
            robot.data.joint_pos[:, left_ids],
            robot.data.joint_vel[:, left_ids],
            tip_force_norm(env),
            prev[:, :18],
        ),
        dim=-1,
    )


@configclass
class PrePourBCPolicyObsGroupCfg(ObsGroup):
    right_joint_pos = ObsTerm(func=joint_pos, params={"asset_cfg": RIGHT_JOINT_CFG})
    right_joint_vel = ObsTerm(func=joint_vel, params={"asset_cfg": RIGHT_JOINT_CFG})
    left_joint_pos = ObsTerm(func=joint_pos, params={"asset_cfg": LEFT_JOINT_CFG})
    left_joint_vel = ObsTerm(func=joint_vel, params={"asset_cfg": LEFT_JOINT_CFG})
    tip_force_norm = ObsTerm(func=tip_force_norm)
    prev_actions = ObsTerm(func=last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PrePourBCObservationsCfg:
    policy: PrePourBCPolicyObsGroupCfg = PrePourBCPolicyObsGroupCfg()

