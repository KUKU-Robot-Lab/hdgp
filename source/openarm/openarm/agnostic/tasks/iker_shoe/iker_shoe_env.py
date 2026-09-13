"""IKER shoe-placement environment (design spec §7).

Episodes start from a grasp-bank state (shoe already in the closed hand on the table). The policy moves the
palm with a 6-D delta pose through damped least-squares IK at 10 Hz; the hand keeps the bank's commanded
targets. The reward is the fixed IKER reward toward the target keypoints of one interaction file.
"""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS, RewardOutput, compute_reward_and_termination, transform_keypoints

from . import grasp_bank as gb
from . import layout, robot
from .iker_shoe_env_cfg import FRICTION, PHYSICS_DT, IkerShoeEnvCfg


def _artifact(path_text: str, default: Path) -> Path:
    return Path(path_text) if path_text else default


class IkerShoeEnv(DirectRLEnv):
    cfg: IkerShoeEnvCfg

    def __init__(self, cfg: IkerShoeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        if self.max_episode_length != cfg.reward.max_episode_length:
            raise ValueError(
                f"episode is {self.max_episode_length} steps but the reward expects {cfg.reward.max_episode_length}"
            )
        run_dir = layout.RUNS_DIR / f"config_{cfg.config_index:02d}"
        meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
        interaction = run_files.read_json(_artifact(cfg.interaction_path, run_dir / f"interaction_{cfg.interaction_source}.json"))
        if not interaction["gate"]["passed"] or interaction["object"] != layout.MOVING_SHOE:
            raise ValueError(f"interaction for {interaction.get('object')} did not pass the gate: {interaction['gate']['failures']}")
        snapshot = run_files.read_json(_artifact(cfg.keypoints_path, run_dir / "keypoints.json"))

        self._offsets = torch.tensor(meta["objects"][layout.MOVING_SHOE]["keypoint_offsets"], dtype=torch.float32, device=dev)
        self._targets = torch.tensor(interaction["target_keypoints"], dtype=torch.float32, device=dev)
        if self._targets.shape != (NUM_KEYPOINTS, 3) or self._offsets.shape != (NUM_KEYPOINTS, 3):
            raise ValueError("interaction targets and shoe keypoint offsets must both be (4, 3)")
        other = snapshot["objects"][layout.OTHER_SHOE]
        self._other_pos = torch.tensor(other["position"], dtype=torch.float32, device=dev)
        self._other_quat = torch.tensor(other["quat_wxyz"], dtype=torch.float32, device=dev)

        prof = robot.profile()
        self._arm_ids, _ = self._robot.find_joints(prof.arm_joint_regex)
        self._gravity_ids, _ = self._robot.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
        self._palm = self._robot.find_bodies(prof.palm_body)[0][0]
        self._palm_jacobian = self._palm - 1  # fixed-base Jacobians omit the root body
        expected = {
            "config_index": cfg.config_index,
            "physics_dt": PHYSICS_DT,
            "friction": FRICTION,
            "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
            "gains": gb.gains_metadata(self._robot.data.joint_names, self._robot.data.joint_stiffness[0], self._robot.data.joint_damping[0]),
        }
        bank_doc = run_files.read_json(_artifact(cfg.grasp_bank_path, run_dir / "grasp_bank.json"))
        self._bank = gb.load_bank(bank_doc, self._robot.data.joint_names, expected, dev)

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), num_envs=n, device=dev
        )
        self._action_scale = torch.tensor([cfg.action_pos_scale] * 3 + [cfg.action_rot_scale] * 3, device=dev)
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._init_keypoints = torch.zeros(n, NUM_KEYPOINTS, 3, device=dev)
        self._keypoint_distance = torch.zeros(n, device=dev)
        self._success_count = torch.zeros(n, device=dev)
        self._failure_count = torch.zeros(n, device=dev)
        self._last: RewardOutput | None = None
        self.actions = torch.zeros(n, cfg.action_space, device=dev)

    # ---------------------------------------------------------------- scene

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._shoe = RigidObject(self.cfg.shoe_move_cfg)
        self._other = RigidObject(self.cfg.shoe_other_cfg)
        for asset in (self.cfg.table_cfg, self.cfg.rack_cfg):
            asset.spawn.func(asset.prim_path, asset.spawn, translation=asset.init_state.pos)
        self.cfg.light_cfg.spawn.func(self.cfg.light_cfg.prim_path, self.cfg.light_cfg.spawn)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects[layout.MOVING_SHOE] = self._shoe
        self.scene.rigid_objects[layout.OTHER_SHOE] = self._other

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        command = self.actions
        if self.cfg.add_noise:
            command = (command + sample_uniform(-self.cfg.action_noise, self.cfg.action_noise, command.shape, self.device)).clamp(-1.0, 1.0)
        root = self._robot.data.root_pose_w
        palm = self._robot.data.body_pose_w[:, self._palm]
        palm_pos_b, palm_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])
        self._ik.set_command(command * self._action_scale, ee_pos=palm_pos_b, ee_quat=palm_quat_b)
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._palm_jacobian, :, self._arm_ids]
        arm_targets = self._ik.compute(palm_pos_b, palm_quat_b, jacobian, self._robot.data.joint_pos[:, self._arm_ids])
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids]
        self._joint_targets[:, self._arm_ids] = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_targets)
        tau = self._robot.root_physx_view.get_gravity_compensation_forces()
        self._robot.set_joint_effort_target(tau[:, self._gravity_ids], joint_ids=self._gravity_ids)

    # ---------------------------------------------------------- observation

    def _keypoints_local(self) -> torch.Tensor:
        world = transform_keypoints(self._shoe.data.root_pos_w, self._shoe.data.root_quat_w, self._offsets)
        return world - self.scene.env_origins[:, None, :]

    def _noisy_quat(self, quat: torch.Tensor) -> torch.Tensor:
        n = quat.shape[0]
        noisy = quat
        for axis in torch.eye(3, device=self.device):
            angle = sample_uniform(-self.cfg.quat_noise_rad, self.cfg.quat_noise_rad, (n,), self.device)
            noisy = quat_mul(noisy, quat_from_angle_axis(angle, axis.expand(n, 3)))
        # q and -q are the same rotation: flip the whole quaternion (the legacy task flipped components one by one)
        sign = torch.where(torch.rand(n, 1, device=self.device) < 0.5, -1.0, 1.0)
        return noisy * sign

    def _get_observations(self) -> dict:
        origins = self.scene.env_origins
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        palm_quat = self._robot.data.body_quat_w[:, self._palm]
        shoe_pos = self._shoe.data.root_pos_w - origins
        shoe_quat = self._shoe.data.root_quat_w
        if self.cfg.add_noise:
            palm_quat, shoe_quat = self._noisy_quat(palm_quat), self._noisy_quat(shoe_quat)
        n = self.num_envs
        obs = torch.cat(
            [
                palm_pos,
                palm_quat[:, [1, 2, 3, 0]],
                shoe_pos,
                shoe_quat[:, [1, 2, 3, 0]],
                self._keypoints_local().reshape(n, -1),
                self._targets.expand(n, NUM_KEYPOINTS, 3).reshape(n, -1),
            ],
            dim=-1,
        )
        if self.cfg.add_noise:
            obs = obs + sample_uniform(-self.cfg.observation_noise, self.cfg.observation_noise, obs.shape, self.device)
        return {"policy": obs}

    # ------------------------------------------------------- reward / dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        current = self._keypoints_local()
        targets = self._targets.expand(self.num_envs, NUM_KEYPOINTS, 3)
        out = compute_reward_and_termination(
            actions=self.actions,
            eef_pos=self._robot.data.body_pos_w[:, self._palm] - origins,
            object_pos=self._shoe.data.root_pos_w - origins,
            current_keypoints=current,
            init_keypoints=self._init_keypoints,
            target_keypoints=targets,
            progress=self.episode_length_buf,
            success_count=self._success_count,
            failure_count=self._failure_count,
            cfg=self.cfg.reward,
        )
        self._success_count, self._failure_count, self._last = out.success_count, out.failure_count, out
        self._keypoint_distance = (current - targets).norm(dim=-1).mean(dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return out.terminated & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        return self._last.reward

    # ----------------------------------------------------------------- reset

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(finished) == 0:
            return
        distance = self._keypoint_distance[finished]
        sustain = self.cfg.reward.sustain_steps
        self.extras["log"] = {
            "iker/keypoint_distance_m": distance.mean().item(),
            "iker/success_5cm": (distance <= self.cfg.eval_success_distance_m).float().mean().item(),
            "iker/sustained_success": (self._success_count[finished] > sustain).float().mean().item(),
            "iker/dropped": (self._failure_count[finished] > sustain).float().mean().item(),
        }

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._log_episode_end(env_ids)
        super()._reset_idx(env_ids)
        count, dev = len(env_ids), self.device
        pick = torch.randint(0, self._bank.size, (count,), device=dev)
        joint_pos = self._bank.joint_pos[pick]
        joint_target = self._bank.joint_target[pick]
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_target

        origins = self.scene.env_origins[env_ids]
        zero_velocity = torch.zeros(count, 6, device=dev)
        shoe_pose = self._bank.shoe_pose[pick].clone()
        shoe_pose[:, :3] += origins
        self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
        self._shoe.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        other_pose = torch.zeros(count, 7, device=dev)
        xy_noise = sample_uniform(-self.cfg.other_shoe_pos_noise, self.cfg.other_shoe_pos_noise, (count, 2), dev)
        other_pose[:, :3] = self._other_pos + origins
        other_pose[:, :2] += xy_noise
        yaw = sample_uniform(-self.cfg.other_shoe_yaw_noise, self.cfg.other_shoe_yaw_noise, (count,), dev)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(count, 3)
        other_pose[:, 3:] = quat_mul(quat_from_angle_axis(yaw, z_axis), self._other_quat.expand(count, 4))
        self._other.write_root_pose_to_sim(other_pose, env_ids=env_ids)
        self._other.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self._init_keypoints[env_ids] = (
            transform_keypoints(shoe_pose[:, :3], shoe_pose[:, 3:], self._offsets) - origins[:, None, :]
        )
        self._keypoint_distance[env_ids] = 0.0
        self._success_count[env_ids] = 0.0
        self._failure_count[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._ik.reset(env_ids)
