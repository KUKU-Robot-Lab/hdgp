"""PourMimicManagedEnv — ManagerBasedRLMimicEnv for bimanual pour.

Directly inherits ManagerBasedRLMimicEnv (no DirectRLEnv in the chain)
so the full IsaacLab Mimic pipeline works:
  annotate_demos.py  →  scene.get_state() / reset_to()
  generate_dataset.py → env.step() / obs_buf["policy"]

Subtask term signals are derived from the scene state (no bead physics
in the Mimic env — those are privileged PPO-only quantities).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as PoseUtils

from .pour_mimic_math import pose7_xyzw_to_matrix


class PourMimicManagedEnv(ManagerBasedRLMimicEnv):
    """ManagerBased Mimic env for bimanual pouring."""

    # ------------------------------------------------------------------ #
    # ManagerBasedRLMimicEnv interface                                    #
    # ------------------------------------------------------------------ #

    def get_robot_eef_pose(
        self,
        eef_name: str = "right",
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return right palm pose as (N, 4, 4) from obs_buf."""
        if eef_name not in ("right", "palm", "right_palm"):
            raise KeyError(f"unsupported eef_name {eef_name!r}")

        ids = slice(None) if env_ids is None else env_ids

        # Prefer obs_buf if already computed
        if hasattr(self, "obs_buf") and "policy" in self.obs_buf:
            # obs_buf["policy"] is the concatenated 91D vector;
            # we need the raw body pose from the articulation instead.
            pass

        robot = self.scene["robot"]
        try:
            palm_idx = robot.data.body_names.index("rl_dg_palm")
        except ValueError:
            # fallback: return identity
            n = len(env_ids) if env_ids is not None else self.num_envs
            return torch.eye(4, device=self.device).unsqueeze(0).expand(n, -1, -1)

        pos = robot.data.body_pos_w[ids, palm_idx, :]
        if hasattr(self.scene, "env_origins"):
            pos = pos - self.scene.env_origins[ids]
        quat_wxyz = robot.data.body_quat_w[ids, palm_idx, :]
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        pose7 = torch.cat([pos, quat_xyzw], dim=-1)
        return pose7_xyzw_to_matrix(pose7)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert right palm target pose + gripper → 18D action."""
        from .pour_mimic_math import target_pose_to_action

        target_pose = target_eef_pose_dict.get("right", next(iter(target_eef_pose_dict.values())))
        gripper_action = gripper_action_dict.get("right", next(iter(gripper_action_dict.values())))

        if target_pose.ndim == 2:
            target_pose = target_pose.unsqueeze(0)
        if gripper_action.ndim == 1:
            gripper_action = gripper_action.unsqueeze(0)

        current_pose = self.get_robot_eef_pose("right", env_ids=[env_id])
        action = target_pose_to_action(
            current_pose, target_pose[:1].to(current_pose.device), gripper_action[:1].to(current_pose.device)
        )

        if action_noise_dict is not None:
            noise_scale = action_noise_dict.get("right", next(iter(action_noise_dict.values())))
            action[:, :6] = torch.clamp(
                action[:, :6] + noise_scale * torch.randn_like(action[:, :6]), -1.0, 1.0
            )
        return action[0]

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert 18D pour actions → right palm target pose matrices."""
        from .pour_mimic_math import action_to_target_pose

        current_pose = self.get_robot_eef_pose("right", env_ids=None)
        return {"right": action_to_target_pose(current_pose, action.to(current_pose.device))}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract 5D hand curl from 18D pour actions."""
        from .pour_mimic_math import actions_to_gripper_actions

        return {"right": actions_to_gripper_actions(actions)}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return source and target cup pose matrices for Mimic."""
        ids = slice(None) if env_ids is None else env_ids

        source = self.scene["source_cup"]
        target = self.scene["target_cup"]

        src_pose7 = torch.cat(
            [source.data.root_pos_w[ids], source.data.root_quat_w[ids][:, [1, 2, 3, 0]]], dim=-1
        )
        tgt_pose7 = torch.cat(
            [target.data.root_pos_w[ids], target.data.root_quat_w[ids][:, [1, 2, 3, 0]]], dim=-1
        )
        return {
            "source_cup": pose7_xyzw_to_matrix(src_pose7),
            "target_cup": pose7_xyzw_to_matrix(tgt_pose7),
        }

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Derive phase completion flags from scene state (no bead physics)."""
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)

        source = self.scene["source_cup"]
        cfg = self.cfg  # type: PourMimicManagedEnvCfg

        # --- grasp_done: at least one fingertip has force > threshold ---
        tip_names = ("tip1_sensor", "tip2_sensor", "tip3_sensor", "tip4_sensor", "tip5_sensor")
        tip_forces = []
        for name in tip_names:
            sensor: ContactSensor = self.scene[name]
            f = sensor.data.force_matrix_w[ids, 0, 0, :].norm(dim=-1)  # (N,)
            tip_forces.append(f)
        max_tip_force = torch.stack(tip_forces, dim=-1).max(dim=-1).values
        contact_ok = max_tip_force >= float(cfg.grasp_force_threshold)
        cup_z = source.data.root_pos_w[ids, 2]
        grasp_done = contact_ok & (cup_z >= self._source_cup_init_z[ids] + 0.01)

        # --- lift_done ---
        lift_done = cup_z >= float(cfg.lift_threshold_z)

        # --- align_done (source cup mouth XY distance to target cup opening) ---
        align_done = self._compute_mouth_xy_distance(ids) <= float(cfg.align_threshold_xy)

        # --- pour_done (cup tilt angle) ---
        tilt_threshold = math.cos(math.radians(float(cfg.pour_threshold_tilt_deg)))
        up_dot = self._compute_source_up_dot(ids)
        pour_done = up_dot <= tilt_threshold

        return {
            "grasp_done": grasp_done,
            "lift_done": lift_done,
            "align_done": align_done,
            "pour_done": pour_done,
        }

    def get_subtask_start_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        terms = self.get_subtask_term_signals(env_ids)
        n = self.num_envs if env_ids is None else len(env_ids)
        return {
            "grasp_start": torch.zeros(n, dtype=torch.bool, device=self.device),
            "lift_start": terms["grasp_done"],
            "align_start": terms["lift_done"],
            "pour_start": terms["align_done"],
        }

    # ------------------------------------------------------------------ #
    # ManagerBasedRLEnv lifecycle hooks                                   #
    # ------------------------------------------------------------------ #

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._source_cup_init_z = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        source = self.scene["source_cup"]
        self._source_cup_init_z[env_ids] = source.data.root_pos_w[env_ids, 2]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _compute_mouth_xy_distance(self, ids) -> torch.Tensor:
        """XY distance from source cup mouth to target cup opening.

        Approximated as horizontal distance between cup roots.
        """
        src_pos = self.scene["source_cup"].data.root_pos_w[ids]
        tgt_pos = self.scene["target_cup"].data.root_pos_w[ids]
        return torch.norm(src_pos[:, :2] - tgt_pos[:, :2], dim=-1)

    def _compute_source_up_dot(self, ids) -> torch.Tensor:
        """Dot product of source cup +Z axis with world +Z.

        Returns ~1.0 when upright, ~0 or negative when tilted/inverted.
        Uses cup root quaternion (wxyz from Isaac Lab).
        """
        quat_wxyz = self.scene["source_cup"].data.root_quat_w[ids]
        # cup body Z in world frame = rotate [0,0,1] by quat
        w, x, y, z = quat_wxyz.unbind(-1)
        cup_z_world = torch.stack(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dim=-1,
        )  # (N, 3)
        return cup_z_world[:, 2]  # dot with world Z = cup_z_world.z component
