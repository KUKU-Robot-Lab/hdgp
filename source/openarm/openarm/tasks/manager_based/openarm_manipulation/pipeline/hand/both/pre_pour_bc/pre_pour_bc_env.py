"""Direct IsaacLab env for pre-pour BC rollout and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from .pre_pour_bc_env_cfg import PrePourBCEnvCfg
from .pre_pour_bc_obs_cfg import pre_pour_actor_obs


class PrePourBCEnv(DirectRLEnv):
    cfg: PrePourBCEnvCfg

    def __init__(self, cfg: PrePourBCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 18, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.final_joint_error = torch.zeros(self.num_envs, device=self.device)
        self._resolve_joint_ids()
        self._target_joint_pos = torch.tensor(cfg.target_joint_pos, dtype=torch.float32, device=self.device)
        self._target_force_norm = torch.tensor(cfg.target_force_norm, dtype=torch.float32, device=self.device)
        self._target_hand_curl = torch.tensor(cfg.target_hand_curl, dtype=torch.float32, device=self.device)

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.source_cup = self.scene["source_cup"]
        self.target_cup = self.scene["target_cup"]
        self.table = self.scene["table"]
        self.tip_sensors = {f"tip{i}_sensor": self.scene[f"tip{i}_sensor"] for i in range(1, 6)}
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def _resolve_joint_ids(self) -> None:
        names = self.robot.joint_names
        right_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
        right_names += [f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]
        left_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self._right_joint_ids = torch.tensor([names.index(name) for name in right_names], device=self.device)
        self._left_joint_ids = torch.tensor([names.index(name) for name in left_names], device=self.device)
        self._policy_joint_ids = torch.cat((self._right_joint_ids, self._left_joint_ids), dim=0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        pos = self.robot.data.joint_pos
        target = pos.clone()
        target[:, self._right_joint_ids[:7]] = pos[:, self._right_joint_ids[:7]] + self.actions[:, :6].mean(dim=-1, keepdim=True) * 0.01
        hand = self.actions[:, 6:11].repeat_interleave(4, dim=1) * 0.15
        target[:, self._right_joint_ids[7:]] = pos[:, self._right_joint_ids[7:]] + hand
        target[:, self._left_joint_ids] = pos[:, self._left_joint_ids] + self.actions[:, 11:18] * 0.01
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": pre_pour_actor_obs(self)}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self._policy_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._policy_joint_ids]
        target_joint_pos = self._target_joint_pos.unsqueeze(0)
        joint_error = torch.linalg.norm(joint_pos - target_joint_pos, dim=-1)
        self.final_joint_error = joint_error

        tip_force = pre_pour_actor_obs(self)[:, 68:73]
        force_reward = torch.exp(-torch.linalg.norm(tip_force - self._target_force_norm.unsqueeze(0), dim=-1))
        curl = self.actions[:, 6:11]
        curl_reward = torch.exp(-torch.linalg.norm(curl - self._target_hand_curl.unsqueeze(0), dim=-1))
        vel_penalty = torch.linalg.norm(joint_vel, dim=-1)
        action_rate_penalty = torch.linalg.norm(self.actions - self.prev_actions, dim=-1)
        success = joint_error <= self.cfg.success_joint_error_threshold
        return (
            self.cfg.rew_joint_target * torch.exp(-joint_error)
            + self.cfg.rew_force * force_reward
            + self.cfg.rew_curl * curl_reward
            + self.cfg.rew_joint_vel * vel_penalty
            + self.cfg.rew_action_rate * action_rate_penalty
            + self.cfg.rew_success * success.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.final_joint_error <= self.cfg.success_joint_error_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.final_joint_error[env_ids] = float("inf")

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
