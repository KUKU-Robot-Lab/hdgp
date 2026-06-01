"""PourMimicManagedEnv — ManagerBasedRLMimicEnv for bimanual pour.

Directly inherits ManagerBasedRLMimicEnv (no DirectRLEnv in the chain)
so the full IsaacLab Mimic pipeline works:
  annotate_demos.py  →  scene.get_state() / reset_to()
  generate_dataset.py → env.step() / obs_buf["policy"]

Subtask term signals are derived from the scene state (no bead physics
in the Mimic env — those are privileged PPO-only quantities).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
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
        """Return requested end-effector pose as (N, 4, 4)."""
        if eef_name not in ("right", "palm", "right_palm", "left", "left_hand"):
            raise KeyError(f"unsupported eef_name {eef_name!r}")

        ids = slice(None) if env_ids is None else env_ids

        # Prefer obs_buf if already computed
        if hasattr(self, "obs_buf") and "policy" in self.obs_buf:
            # obs_buf["policy"] is the concatenated 91D vector;
            # we need the raw body pose from the articulation instead.
            pass

        body_candidates = (
            ("rl_dg_palm", "right_hand", "openarm_right_hand")
            if eef_name in ("right", "palm", "right_palm")
            else ("openarm_left_hand", "left_hand", "openarm_left_link7")
        )
        return self._get_body_pose_matrix(body_candidates, ids, env_ids)

    def _get_body_pose_matrix(self, body_candidates: tuple[str, ...], ids, env_ids) -> torch.Tensor:
        robot = self.scene["robot"]
        body_idx = None
        for body_name in body_candidates:
            try:
                body_idx = robot.data.body_names.index(body_name)
                break
            except ValueError:
                continue

        if body_idx is None:
            n = len(env_ids) if env_ids is not None else self.num_envs
            return torch.eye(4, device=self.device).unsqueeze(0).expand(n, -1, -1)

        pos = robot.data.body_pos_w[ids, body_idx, :]
        if hasattr(self.scene, "env_origins"):
            pos = pos - self.scene.env_origins[ids]
        quat_wxyz = robot.data.body_quat_w[ids, body_idx, :]
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
        """Derive phase completion flags from scene state.

        The term signals are intentionally based on robot/cup state available
        during sim replay.  Final success is stricter than annotation: cups must
        physically lift, while subtask boundaries still include palm proximity so
        old teleop demonstrations can be segmented for diagnosis.
        """
        ids = slice(None) if env_ids is None else env_ids

        source = self.scene["source_cup"]
        target = self.scene["target_cup"]
        cfg = self.cfg  # type: PourMimicManagedEnvCfg
        robot = self.scene["robot"]

        # --- right palm position ---
        palm_idx = None
        for candidate in ("rl_dg_palm", "right_hand", "openarm_right_hand"):
            try:
                palm_idx = robot.data.body_names.index(candidate)
                break
            except ValueError:
                continue
        if palm_idx is not None:
            palm_pos = robot.data.body_pos_w[ids, palm_idx, :]
        else:
            palm_pos = source.data.root_pos_w[ids]

        # --- left palm position ---
        left_palm_idx = None
        for candidate in ("openarm_left_hand", "left_hand", "openarm_left_link7"):
            try:
                left_palm_idx = robot.data.body_names.index(candidate)
                break
            except ValueError:
                continue
        if left_palm_idx is not None:
            left_palm_pos = robot.data.body_pos_w[ids, left_palm_idx, :]
        else:
            left_palm_pos = target.data.root_pos_w[ids]

        # --- left_grasp_done: left palm within 0.12 m of target cup ---
        tgt_pos = target.data.root_pos_w[ids]
        left_grasp_done = torch.norm(left_palm_pos - tgt_pos, dim=-1) <= 0.12

        source_init_z = getattr(self, "_source_cup_init_z", source.data.root_pos_w[:, 2])
        target_init_z = getattr(self, "_target_cup_init_z", target.data.root_pos_w[:, 2])
        lift_delta = float(getattr(cfg, "physical_lift_success_height", 0.04))

        source_lifted = source.data.root_pos_w[ids, 2] >= source_init_z[ids] + lift_delta
        target_lifted = target.data.root_pos_w[ids, 2] >= target_init_z[ids] + lift_delta

        # --- left_lift_done: physical lift, with palm-height fallback for annotation diagnostics ---
        left_lift_done = target_lifted | (left_palm_pos[:, 2] >= float(cfg.left_lift_threshold_z))

        # --- grasp_done: right palm within 0.12 m of source cup ---
        src_pos = source.data.root_pos_w[ids]
        grasp_done = torch.norm(palm_pos - src_pos, dim=-1) <= 0.12

        # --- lift_done: physical lift, with palm-height fallback for annotation diagnostics ---
        lift_done = source_lifted | (palm_pos[:, 2] >= float(cfg.lift_threshold_z))

        # --- align_done: right palm raised to pre-pour height ---
        # Using palm Z threshold is more reliable than cup-to-cup XY distance,
        # which requires physical grasping. Threshold 0.43 m fires after lift_done
        # and in the second half of the demo across all recorded demonstrations.
        align_done = palm_pos[:, 2] >= float(cfg.align_threshold_z)

        # --- pour_done: pre-pour terminal gate ---
        # This Mimic env does not model liquid/beads.  The terminal "pour" phase
        # means the right palm reached the pre-pour pose while the source cup is
        # physically lifted, or the replay reached the stricter pose-only gate
        # used for dataset triage.
        pour_done = align_done & (
            source_lifted | (palm_pos[:, 2] >= float(cfg.pour_ready_threshold_z))
        )

        return {
            "left_grasp_done": left_grasp_done,
            "left_lift_done": left_lift_done,
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
            "left_grasp_start": torch.zeros(n, dtype=torch.bool, device=self.device),
            "left_lift_start": terms["left_grasp_done"],
            "grasp_start": terms["left_lift_done"],
            "lift_start": terms["grasp_done"],
            "align_start": terms["lift_done"],
            "pour_start": terms["align_done"],
        }

    # ------------------------------------------------------------------ #
    # ManagerBasedRLEnv lifecycle hooks                                   #
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        source = self.scene["source_cup"]
        target = self.scene["target_cup"]
        if not hasattr(self, "_source_cup_init_z"):
            self._source_cup_init_z = source.data.root_pos_w[:, 2].clone()
            self._target_cup_init_z = target.data.root_pos_w[:, 2].clone()
        self._source_cup_init_z[env_ids] = source.data.root_pos_w[env_ids, 2]
        self._target_cup_init_z[env_ids] = target.data.root_pos_w[env_ids, 2]
