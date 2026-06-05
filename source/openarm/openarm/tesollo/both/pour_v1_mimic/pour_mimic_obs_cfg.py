"""Observation configuration for pour_v1_mimic (ManagerBased).

Policy obs (105D, sim-aware):
  right_joint_pos  27  arm7 + hand20
  right_joint_vel  27
  left_joint_pos    7
  left_joint_vel    7
  tip_force_norm    5  norm of 3D force per fingertip, clamped [0,1]
  prev_actions     18
  source_cup_pose   7  xyz + quat(wxyz), relative to env origin
  target_cup_pose   7  xyz + quat(wxyz), relative to env origin
  ─────────────────────
  total           105

subtask_terms group (4 bool): grasp_done, lift_done, align_done, pour_done
  Used by IsaacLab Mimic annotate_demos.py; not fed to the policy.
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


# ---------------------------------------------------------------------------
# Custom observation functions
# ---------------------------------------------------------------------------

_RIGHT_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "openarm_right_joint[1-7]",
        "rj_dg_[1-5]_[1-4]",
    ],
)
_LEFT_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["openarm_left_joint[1-7]"],
)

_CONTACT_FORCE_MAX = 10.0  # N — same as pour_right_v4


def tip_force_norm(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return per-fingertip contact force norms normalised to [0, 1].

    Reads the 5 individual ContactSensors wired in PourMimicSceneCfg and
    returns a (N, 5) tensor.  Equivalent to pour_right_v4's tip_force_norm
    obs, which is real-robot compatible.
    """
    sensor_names = ("tip1_sensor", "tip2_sensor", "tip3_sensor", "tip4_sensor", "tip5_sensor")
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        f = sensor.data.force_matrix_w[:, 0, 0, :].norm(dim=-1, keepdim=True)  # (N,1)
        forces.append(f)
    raw = torch.cat(forces, dim=-1)  # (N, 5)
    return (raw / _CONTACT_FORCE_MAX).clamp(0.0, 1.0)


def _cup_pose(env: "ManagerBasedEnv", asset_name: str) -> torch.Tensor:
    cup = env.scene[asset_name]
    pos = cup.data.root_pos_w
    if hasattr(env.scene, "env_origins"):
        pos = pos - env.scene.env_origins
    return torch.cat([pos, cup.data.root_quat_w], dim=-1)


def source_cup_pose(env: "ManagerBasedEnv") -> torch.Tensor:
    return _cup_pose(env, "source_cup")


def target_cup_pose(env: "ManagerBasedEnv") -> torch.Tensor:
    return _cup_pose(env, "target_cup")


def grasp_done_signal(env: "ManagerBasedEnv") -> torch.Tensor:
    """Cup is in contact with fingertips and above init z."""
    return _get_term_signal(env, "grasp_done")


def lift_done_signal(env: "ManagerBasedEnv") -> torch.Tensor:
    return _get_term_signal(env, "lift_done")


def align_done_signal(env: "ManagerBasedEnv") -> torch.Tensor:
    return _get_term_signal(env, "align_done")


def pour_done_signal(env: "ManagerBasedEnv") -> torch.Tensor:
    return _get_term_signal(env, "pour_done")


def _get_term_signal(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Delegate to env.get_subtask_term_signals() if available."""
    if hasattr(env, "get_subtask_term_signals"):
        signals = env.get_subtask_term_signals()
        val = signals.get(name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        return val.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, device=env.device)


# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------


@configclass
class PolicyObsGroupCfg(ObsGroup):
    """Sim-aware 105D policy observation."""

    right_joint_pos = ObsTerm(func=joint_pos, params={"asset_cfg": _RIGHT_JOINT_CFG})
    right_joint_vel = ObsTerm(func=joint_vel, params={"asset_cfg": _RIGHT_JOINT_CFG})
    left_joint_pos = ObsTerm(func=joint_pos, params={"asset_cfg": _LEFT_JOINT_CFG})
    left_joint_vel = ObsTerm(func=joint_vel, params={"asset_cfg": _LEFT_JOINT_CFG})
    tip_force_norm = ObsTerm(func=tip_force_norm)
    prev_actions = ObsTerm(func=last_action)
    source_cup_pose = ObsTerm(func=source_cup_pose)
    target_cup_pose = ObsTerm(func=target_cup_pose)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SubtaskTermsGroupCfg(ObsGroup):
    """Subtask termination signals for Mimic annotation (not concatenated)."""

    grasp_done = ObsTerm(func=grasp_done_signal)
    lift_done = ObsTerm(func=lift_done_signal)
    align_done = ObsTerm(func=align_done_signal)
    pour_done = ObsTerm(func=pour_done_signal)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PourMimicObservationsCfg:
    """Full observation specification for PourMimicManagedEnv."""

    policy: PolicyObsGroupCfg = PolicyObsGroupCfg()
    subtask_terms: SubtaskTermsGroupCfg = SubtaskTermsGroupCfg()
