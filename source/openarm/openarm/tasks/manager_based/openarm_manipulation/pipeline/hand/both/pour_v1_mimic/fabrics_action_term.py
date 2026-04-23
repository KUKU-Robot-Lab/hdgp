"""Fabrics-based right arm + right hand ActionTerm for pour_v1_mimic.

Wraps OpenArmTeoslloPoseFabric so it fits the IsaacLab ActionTerm contract:
  process_actions  – called once per env step; decodes 18D action into
                     palm pose target and hand joint target, then sets
                     Fabrics features.
  apply_actions    – called once per sim step (decimation); runs one
                     Fabrics integrator step and writes joint targets to
                     the robot articulation.

Action slice layout (18D total):
  [0:6]   right palm pose delta  (xyz m + axis-angle rad), Fabrics IK
  [6:11]  right hand curl        (5D per-finger lerp, [-1,1] → open/grasp)
  [11:18] left arm joint delta   (handled by LeftArmDeltaActionTerm)
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ---------------------------------------------------------------------------
# Fabrics import (vendored path: hdgp/source/FABRICS/src)
# ---------------------------------------------------------------------------
for _p in Path(__file__).resolve().parents:
    if _p.name == "source":
        _fab = _p / "FABRICS" / "src"
        if _fab.exists():
            _s = str(_fab)
            if _s not in sys.path:
                sys.path.insert(0, _s)
        break

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp  # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel  # noqa: E402


_NUM_ARM_DOF = 7
_NUM_HAND_DOF = 20
_NUM_ROBOT_DOF = _NUM_ARM_DOF + _NUM_HAND_DOF


@configclass
class FabricsRightArmActionTermCfg(ActionTermCfg):
    """Configuration for the Fabrics right arm + hand action term."""

    class_type: type = None  # set below after class definition

    asset_name: str = "robot"

    # Fabric hyper-parameters
    fabric_decimation: int = 5
    fabric_dt: float = 1.0 / 300.0
    fabric_num_obstacles: int = 0
    fabric_damping_gain: float = 5.0

    # Palm workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]
    palm_xyz_min: tuple = (0.20, -0.40, 0.30)
    palm_xyz_max: tuple = (1.00,  0.10, 0.95)

    # Per-step delta scales (un-normalised world units)
    delta_xyz_scale: float = 0.30     # m  per action unit
    delta_rot_scale: float = 0.30     # rad per action unit

    # Open / grasp hand poses (20D HAND joint order from preset)
    hand_open_pose: tuple | None = None   # filled from pour_preset at init
    hand_grasp_pose: tuple | None = None

    # arm start pose (7D)
    arm_start_pose: tuple | None = None


class FabricsRightArmActionTerm(ActionTerm):
    """Fabrics-based right arm IK + per-finger curl action term.

    Manages its own fabric_q / palm_pose_buf state and resets them
    per-env-id on episode reset.
    """

    cfg: FabricsRightArmActionTermCfg

    def __init__(self, cfg: FabricsRightArmActionTermCfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)

        self._robot = self._asset
        self._device = env.device
        num_envs = env.num_envs

        # ---------------- Fabrics setup ----------------
        initialize_warp()
        world_model = WorldMeshesModel(
            num_envs=num_envs,
            max_objects_per_env=cfg.fabric_num_obstacles,
            device=self._device,
        )
        self._fabric = OpenArmTeoslloPoseFabric(
            num_envs=num_envs,
            world=world_model,
            use_hand_fabric=False,
            device=self._device,
        )
        self._integrator = DisplacementIntegrator(self._fabric)
        self._timestep = cfg.fabric_dt

        # Fabric state buffers
        _zero27 = torch.zeros(num_envs, _NUM_ROBOT_DOF, device=self._device)
        self._fabric_q = _zero27.clone().contiguous()
        self._fabric_qd = _zero27.clone().contiguous()
        self._fabric_qdd = _zero27.clone().contiguous()

        # Damping gain (per env, shape (N,1))
        self._damping = cfg.fabric_damping_gain * torch.ones(num_envs, 1, device=self._device)

        # Dummy PCA (zeros = no hand fabric target)
        self._hand_pca = torch.zeros(num_envs, 5, device=self._device)

        # Obstacle ids
        self._obj_ids = torch.zeros(num_envs, 1, dtype=torch.long, device=self._device)
        self._obj_ind = torch.zeros(num_envs, 1, dtype=torch.long, device=self._device)

        # Palm pose buffer (absolute) [x,y,z,qx,qy,qz,qw]
        self._palm_pose_buf = torch.zeros(num_envs, 7, device=self._device)
        self._palm_pose_buf[:, 6] = 1.0  # identity quaternion

        # Hand joint buffers (from pour_v1 preset; cfg overrides allowed)
        from ..pour_v1.pour_preset import HAND_APPROACH_POSE, HAND_GRASP_POSE
        from ..pour_v1.pour_constants import ARM_START_POSE

        _open_src = list(cfg.hand_open_pose) if cfg.hand_open_pose else list(HAND_APPROACH_POSE)
        _grasp_src = list(cfg.hand_grasp_pose) if cfg.hand_grasp_pose else list(HAND_GRASP_POSE)
        _arm_src = list(cfg.arm_start_pose) if cfg.arm_start_pose else list(ARM_START_POSE)

        self._hand_open = torch.tensor(_open_src, dtype=torch.float32, device=self._device).unsqueeze(0).expand(num_envs, -1).clone()
        self._hand_grasp = torch.tensor(_grasp_src, dtype=torch.float32, device=self._device).unsqueeze(0).expand(num_envs, -1).clone()
        self._hand_targets = self._hand_open.clone()
        self._arm_start = torch.tensor(_arm_src, dtype=torch.float32, device=self._device)

        # Workspace clamps
        self._palm_min = torch.tensor(cfg.palm_xyz_min, device=self._device)
        self._palm_max = torch.tensor(cfg.palm_xyz_max, device=self._device)
        self._delta_xyz = cfg.delta_xyz_scale
        self._delta_rot = cfg.delta_rot_scale

        # DOF indices (resolved after articulation is loaded)
        self._arm_ids: torch.Tensor | None = None
        self._hand_ids: torch.Tensor | None = None

        # Raw / processed buffers (required by ActionTerm contract)
        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=self._device)
        self._processed = torch.zeros(num_envs, _NUM_ROBOT_DOF, device=self._device)

    # ------------------------------------------------------------------
    # ActionTerm properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 11  # [0:6] palm delta, [6:11] finger curl

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dof_ids(self) -> None:
        if self._arm_ids is not None:
            return
        names = self._robot.joint_names
        arm_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
        self._arm_ids = torch.tensor(
            [names.index(n) for n in arm_names], device=self._device, dtype=torch.long
        )
        hand_names = [
            f"rj_dg_{f}_{j}"
            for f in range(1, 6)
            for j in range(1, 5)
        ]
        self._hand_ids = torch.tensor(
            [names.index(n) for n in hand_names], device=self._device, dtype=torch.long
        )

    @staticmethod
    def _compose_quat(q_base: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
        """Apply axis-angle rotvec on top of q_base (xyzw)."""
        angle = torch.linalg.norm(rotvec, dim=-1)
        safe_angle = angle.clamp(min=1e-8)
        axis = rotvec / safe_angle.unsqueeze(-1)
        # convert to wxyz for quat_from_angle_axis
        q_delta_wxyz = quat_from_angle_axis(angle, axis)  # (N,4) wxyz
        q_base_wxyz = q_base[:, [3, 0, 1, 2]]             # xyzw → wxyz
        q_out_wxyz = quat_mul(q_delta_wxyz, q_base_wxyz)
        return q_out_wxyz[:, [1, 2, 3, 0]]                # back to xyzw

    # ------------------------------------------------------------------
    # process_actions  (once per env step)
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """Decode 18D action slice [0:11]; update palm_pose_buf; set Fabrics features."""
        self._get_dof_ids()

        palm_delta = actions[:, :6]
        finger_action = actions[:, 6:11]

        # ---- palm target (simple additive integration) ----
        self._palm_pose_buf[:, :3] = torch.clamp(
            self._palm_pose_buf[:, :3] + palm_delta[:, :3] * self._delta_xyz,
            self._palm_min.unsqueeze(0),
            self._palm_max.unsqueeze(0),
        )
        if torch.any(palm_delta[:, 3:6].abs() > 1e-6):
            self._palm_pose_buf[:, 3:7] = self._compose_quat(
                self._palm_pose_buf[:, 3:7], palm_delta[:, 3:6] * self._delta_rot
            )

        # ---- hand joint targets (per-finger lerp) ----
        t = (finger_action + 1.0) / 2.0           # [0,1]
        t_exp = t.repeat_interleave(4, dim=1)      # (N,20)
        self._hand_targets = (
            self._hand_open + t_exp * (self._hand_grasp - self._hand_open)
        )

        # ---- update fabric_q hand slice ----
        self._fabric_q[:, _NUM_ARM_DOF:] = self._hand_targets
        self._fabric_qd[:, _NUM_ARM_DOF:].zero_()

        # ---- Fabrics: null-space tracking + set_features ----
        self._fabric.default_config.copy_(self._fabric_q.detach())
        self._fabric.set_features(
            self._hand_pca,
            self._palm_pose_buf,
            "quaternion",
            self._fabric_q.detach(),
            self._fabric_qd.detach(),
            self._obj_ids,
            self._obj_ind,
            self._damping,
        )

        self._raw_actions.copy_(actions[:, :11])

    # ------------------------------------------------------------------
    # apply_actions  (once per sim step, i.e., decimation times per env step)
    # ------------------------------------------------------------------

    def apply_actions(self) -> None:
        """Run one Fabrics integrator step and write joint targets to robot."""
        self._fabric_q, self._fabric_qd, self._fabric_qdd = self._integrator.step(
            self._fabric_q.detach(),
            self._fabric_qd.detach(),
            self._fabric_qdd.detach(),
            self._timestep,
        )
        self._fabric_q[:, _NUM_ARM_DOF:] = self._hand_targets

        arm_target = self._fabric_q[:, :_NUM_ARM_DOF]
        self._robot.set_joint_position_target(arm_target, joint_ids=self._arm_ids)
        self._robot.set_joint_velocity_target(torch.zeros_like(arm_target), joint_ids=self._arm_ids)
        self._robot.set_joint_position_target(self._hand_targets, joint_ids=self._hand_ids)
        self._robot.set_joint_velocity_target(torch.zeros_like(self._hand_targets), joint_ids=self._hand_ids)

        self._processed.copy_(self._fabric_q)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids

        n = self._fabric_q[ids].shape[0]
        self._fabric_q[ids, :_NUM_ARM_DOF] = self._arm_start.unsqueeze(0).expand(n, -1)
        self._fabric_q[ids, _NUM_ARM_DOF:] = self._hand_open[ids]
        self._fabric_qd[ids].zero_()
        self._fabric_qdd[ids].zero_()
        self._fabric.default_config[ids].copy_(self._fabric_q[ids].detach())

        # Reset palm pose buffer from articulation body pose if available
        if hasattr(self._robot, "data") and self._robot.data.body_names:
            try:
                palm_idx = self._robot.data.body_names.index("rl_dg_palm")
                pos = self._robot.data.body_pos_w[env_ids, palm_idx, :]
                quat_wxyz = self._robot.data.body_quat_w[env_ids, palm_idx, :]
                self._palm_pose_buf[env_ids, :3] = pos
                self._palm_pose_buf[env_ids, 3:7] = quat_wxyz[:, [1, 2, 3, 0]]
            except (ValueError, AttributeError):
                pass

        self._hand_targets[env_ids] = self._hand_open[env_ids]


FabricsRightArmActionTermCfg.class_type = FabricsRightArmActionTerm
