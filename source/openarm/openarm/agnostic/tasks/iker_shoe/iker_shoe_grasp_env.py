"""IKER shoe stage-1 grasp environment (design 2026-09-14 §4-§6).

Episodes start from a pre-grasp arm pose 8-12 cm above the shoe with the hand open. The policy moves the palm with a
6-D delta pose through damped least-squares IK (as in stage 2) and commands all 20 hand joints through the grasp_fj
full-joint law. The reward is ``grasp_stage.stage1_step``: approach and lift progress, a lift bonus and a success
bonus scaled by the five-finger grasp quality. Success = the shoe held 5-15 cm above and within 10 cm of its start, the
thumb closing, for 20 steps (§6, §16), with motion penalties while lifted and a lost grasp ending the episode
(revision 3-5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS
from openarm.agnostic.modules.object_wrench import WrenchDR

from . import grasp_bank as gb
from . import grasp_stage as gs
from . import layout, robot
from .iker_shoe_env import IkerShoeEnv
from .iker_shoe_env_cfg import FRICTION, PHYSICS_DT
from .iker_shoe_grasp_env_cfg import ARM_ACTION_DIM, IkerShoeGraspEnvCfg

PREGRASP_BANK_FILE = "pregrasp_bank.json"
QUALITY_CALIBRATION_FILE = "grasp_quality_calibration.json"
# Published on every step (0.0 until the first episode ends): rl_games' observer reads each step's log with the key set of
# the epoch's first step, so a key that appears only on reset steps raises KeyError.
EPISODE_LOG_KEYS = ("grasp_episode/success", "grasp_episode/latched", "grasp_episode/best_lift_m",
                    "grasp_episode/q_at_latch", "grasp_episode/q_at_success")
PALMAR_AXIS_LOCAL = (1.0, 0.0, 0.0)  # palm link +x is the grasping side (grasp_bank.palm_rotations, both hands)


def _artifact(path_text: str, default: Path) -> Path:
    return Path(path_text) if path_text else default


class IkerShoeGraspEnv(DirectRLEnv):
    cfg: IkerShoeGraspEnvCfg

    # Scene, keypoints and quaternion noise are the stage-2 environment's, unchanged.
    _setup_scene = IkerShoeEnv._setup_scene
    _keypoints_local = IkerShoeEnv._keypoints_local
    _noisy_quat = IkerShoeEnv._noisy_quat

    def __init__(self, cfg: IkerShoeGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
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
        self._surface_local = gs.surface_subsample(meta["objects"][layout.MOVING_SHOE]["hull_local"]).to(dev)

        prof = robot.profile()
        joint_names, body_names = list(self._robot.data.joint_names), list(self._robot.data.body_names)
        self._arm_ids, _ = self._robot.find_joints(prof.arm_joint_regex)
        self._gravity_ids, _ = self._robot.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
        self._hand_ids = torch.tensor([joint_names.index(name) for name in prof.hand_joint_names], device=dev)
        self._palm = self._robot.find_bodies(prof.palm_body)[0][0]
        self._palm_jacobian = self._palm - 1  # fixed-base Jacobians omit the root body
        link_names = [body for finger in gb.FINGERS for body in prof.finger_sensor_bodies[finger]]
        self._finger_links = torch.tensor([body_names.index(body) for body in link_names], device=dev)
        self._finger_sizes = tuple(len(prof.finger_sensor_bodies[finger]) for finger in gb.FINGERS)
        self._palmar_axis = torch.tensor(PALMAR_AXIS_LOCAL, device=dev).expand(n, 3)
        curl = gs.role_joint_index(prof.hand_joint_names, cfg.thumb_curl_role)
        self._thumb_curl_id = int(self._hand_ids[curl])
        self._thumb_open = torch.tensor(float(prof.hand_open_pose[curl]), device=dev)
        self._thumb_grip = torch.tensor(float(prof.hand_grip_pose[curl]), device=dev)
        gs.closing_travel(self._thumb_open, self._thumb_open, self._thumb_grip)  # boot error when the role has no closing direction

        backstop = robot.apply_hand_backstop(self._robot, cfg.hand_backstop_joints)  # before the action limits read the hard limits
        hard = self._robot.data.joint_pos_limits[0, self._hand_ids]
        self._hand_lo, self._hand_hi = gs.stage1_hand_limits(prof.hand_joint_names, hard[:, 0], hard[:, 1], prof.hand_action_limit_override,
                                                             prof.hand_open_pose, cfg.frozen_hand_joints)
        default_hand = self._robot.data.default_joint_pos[0, self._hand_ids]
        self._hand_reset = torch.max(torch.min(default_hand, self._hand_hi), self._hand_lo)
        moved = (self._hand_reset - default_hand).abs()
        print("[iker_grasp] hand action range (joint: lo..hi, open pose -> reset)", flush=True)
        for name, lo, hi, q0, q1 in zip(prof.hand_joint_names, self._hand_lo.tolist(), self._hand_hi.tolist(), default_hand.tolist(), self._hand_reset.tolist()):
            print(f"[iker_grasp]   {name:16s} {lo:+.3f}..{hi:+.3f}  {q0:+.3f} -> {q1:+.3f}", flush=True)
        if float(moved.max()) > cfg.hand_reset_clamp_max_rad:
            raise ValueError(f"the open hand pose sits {float(moved.max()):.3f} rad outside the action range (max {cfg.hand_reset_clamp_max_rad})")

        scene_config = layout.sample_configs(cfg.config_index + 1)[cfg.config_index]
        expected = json.loads(json.dumps({
            "config_index": cfg.config_index,
            "physics_dt": PHYSICS_DT,
            "friction": FRICTION,
            "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
            "gains": gb.gains_metadata(joint_names, self._robot.data.joint_stiffness[0], self._robot.data.joint_damping[0]),
            "robot_usd": str(prof.usd_relpath),
            "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
            "scene_config": vars(scene_config),
        }))
        bank_doc = run_files.read_json(_artifact(cfg.pregrasp_bank_path, run_dir / PREGRASP_BANK_FILE))
        self._bank = gb.load_bank(bank_doc, joint_names, expected, dev)
        # harvest_grasp_bank.py writes these comparison keys into the learned bank; the pre-grasp bank above is compared
        # without the backstop (grasp_bank.LEARNED_BOOT_KEYS)
        self._boot_metadata = {**expected, "hand_backstop": backstop}

        reward_cfg = replace(cfg.grasp_reward)
        calibration_path = _artifact(cfg.quality_calibration_path, run_dir / QUALITY_CALIBRATION_FILE)
        if reward_cfg.g_min < 1.0:
            # The grasp factor acts only with q_lo/q_hi measured on this shoe (design §5, audit digest row 12).
            if not calibration_path.is_file():
                raise FileNotFoundError(f"g_min {reward_cfg.g_min} < 1 needs the grasp quality calibration {calibration_path} "
                                        "(scripts/iker/measure_grasp_quality.py on a phase-A checkpoint)")
            q_lo, q_hi = gs.read_quality_calibration(calibration_path)
            reward_cfg = replace(reward_cfg, q_lo=q_lo, q_hi=q_hi)
        reward_cfg.validate()  # re-validate after the hydra round trip and the calibration
        self._reward_cfg = reward_cfg
        idle = gs.stage1_step(gs.Stage1State.start(1, dev), reward_cfg, **{k: torch.zeros(1, device=dev) for k in (
            "palm_gap", "dz_free", "shoe_shift_xy", "thumb_curl", "rel_speed", "shoe_speed", "q", "hand_floor_depth",
            "arm_speed_sum", "hand_speed_sum", "hand_command_rate")},
            palm_shoe_dist=torch.ones(1, device=dev))
        print(f"[iker_grasp] reward {reward_cfg} · idle income per step {float(idle.reward):.3f} · "
              f"calibration {calibration_path if calibration_path.is_file() else 'none'}", flush=True)
        if float(idle.reward) != 0.0:
            raise RuntimeError("the stage-1 reward pays an idle policy")

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), num_envs=n, device=dev
        )
        self._action_scale = torch.tensor([cfg.action_pos_scale] * 3 + [cfg.action_rot_scale] * 3, device=dev)
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._hand_targets = self._hand_reset.expand(n, -1).clone()
        self._stage = gs.Stage1State.start(n, dev)
        self._start_bottom_z = torch.zeros(n, device=dev)
        self._start_xy = torch.zeros(n, 2, device=dev)  # env-local shoe centre at the episode start (§16)
        self._q_at_latch = torch.zeros(n, device=dev)
        self._q_at_success = torch.zeros(n, device=dev)
        # q and w_f of the latest _get_dones; _reset_idx leaves them alone, so a caller can read the step that ended an episode
        self._q_step, self._w_f_step = torch.zeros(n, device=dev), None
        self._episode_log = dict.fromkeys(EPISODE_LOG_KEYS, 0.0)  # statistics of the most recently finished episodes
        self._capture = gs.SuccessCapture.empty(n, self._robot.num_joints, dev)  # written only with cfg.capture_success_states
        self._shoe_mass = self._shoe.root_physx_view.get_masses()[:, 0].to(dev)
        self._wrench = WrenchDR(n, dev, force_scale=cfg.wrench_force_per_kg, torque_scale=cfg.wrench_torque_per_kg,
                                prob_range=cfg.wrench_prob_range)
        self._last: gs.Stage1Step | None = None
        self.actions = torch.zeros(n, cfg.action_space, device=dev)
        # revision 3-5: hand command rate |a_hand(t) - a_hand(t-1)| of the policy's own actions; 0 on the first step of an episode
        self._prev_hand_action = torch.zeros(n, cfg.action_space - ARM_ACTION_DIM, device=dev)
        self._prev_hand_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hand_command_rate = torch.zeros(n, device=dev)

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        hand_command = self.actions[:, ARM_ACTION_DIM:]
        self._hand_command_rate = torch.where(
            self._prev_hand_valid, (hand_command - self._prev_hand_action).abs().sum(dim=-1), torch.zeros_like(self._hand_command_rate)
        )
        self._prev_hand_action = hand_command.clone()
        self._prev_hand_valid = torch.ones_like(self._prev_hand_valid)
        command = self.actions
        if self.cfg.add_noise:
            command = (command + sample_uniform(-self.cfg.action_noise, self.cfg.action_noise, command.shape, self.device)).clamp(-1.0, 1.0)
        root = self._robot.data.root_pose_w
        palm = self._robot.data.body_pose_w[:, self._palm]
        palm_pos_b, palm_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])
        self._ik.set_command(command[:, :ARM_ACTION_DIM] * self._action_scale, ee_pos=palm_pos_b, ee_quat=palm_quat_b)
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._palm_jacobian, :, self._arm_ids]
        arm_targets = self._ik.compute(palm_pos_b, palm_quat_b, jacobian, self._robot.data.joint_pos[:, self._arm_ids])
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids]
        self._joint_targets[:, self._arm_ids] = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])
        self._hand_targets = gs.hand_targets(command[:, ARM_ACTION_DIM:], self._hand_lo, self._hand_hi, self._hand_targets)
        self._joint_targets[:, self._hand_ids] = self._hand_targets
        forces, torques = self._wrench.step(self._shoe_mass, self._stage.latched)
        self._shoe.set_external_force_and_torque(forces, torques, is_global=True)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_targets)
        tau = self._robot.root_physx_view.get_gravity_compensation_forces()
        self._robot.set_joint_effort_target(tau[:, self._gravity_ids], joint_ids=self._gravity_ids)

    # ---------------------------------------------------------- observation

    def _shoe_surface(self) -> torch.Tensor:
        """(N, P, 3) env-local surface points of the moving shoe."""
        n, p = self.num_envs, self._surface_local.shape[0]
        quat = self._shoe.data.root_quat_w[:, None, :].expand(n, p, 4)
        points = quat_apply(quat.reshape(-1, 4), self._surface_local.expand(n, p, 3).reshape(-1, 3)).view(n, p, 3)
        return points + (self._shoe.data.root_pos_w - self.scene.env_origins)[:, None, :]

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
                self._robot.data.joint_pos[:, self._hand_ids],
                gs.normalized_targets(self._hand_targets, self._hand_lo, self._hand_hi),
            ],
            dim=-1,
        )
        if self.cfg.add_noise:
            obs = obs + sample_uniform(-self.cfg.observation_noise, self.cfg.observation_noise, obs.shape, self.device)
        return {"policy": obs}

    # ------------------------------------------------------- reward / dones

    def grasp_quality_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(q, w_f, palm_cos) of the current simulator state."""
        origins = self.scene.env_origins
        links = self._robot.data.body_pos_w[:, self._finger_links] - origins[:, None, :]
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        palm_normal = quat_apply(self._robot.data.body_quat_w[:, self._palm], self._palmar_axis)
        shoe_center = self._shoe.data.root_pos_w - origins
        return gs.grasp_quality(links, self._finger_sizes, self._shoe_surface(), palm_pos, palm_normal, shoe_center)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        surface = self._shoe_surface()
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        shoe_pos = self._shoe.data.root_pos_w - origins
        q, w_f, palm_cos = self.grasp_quality_now()
        self._q_step, self._w_f_step = q, w_f
        hand_z = torch.cat([self._robot.data.body_pos_w[:, self._finger_links, 2], palm_pos[:, 2:3]], dim=1).min(dim=1).values
        dz_free = gs.free_lift_height(surface, self._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
        shoe_vel = self._shoe.data.root_lin_vel_w
        shift_xy = (shoe_pos[:, :2] - self._start_xy).norm(dim=-1)
        thumb_curl = gs.closing_travel(self._robot.data.joint_pos[:, self._thumb_curl_id], self._thumb_open, self._thumb_grip)
        # revision 3-6: judge slip in the palm frame — the plain |v_shoe - v_palm| counted a shoe fixed in a rotating hand as
        # slip (root_lin_vel_w / body_lin_vel_w / body_ang_vel_w are centre-of-mass quantities, hence the COM positions).
        rel_speed = gs.palm_frame_slip_speed(
            shoe_vel,
            self._robot.data.body_lin_vel_w[:, self._palm],
            self._robot.data.body_ang_vel_w[:, self._palm],
            self._shoe.data.root_com_pos_w,
            self._robot.data.body_com_pos_w[:, self._palm],
        )
        step = gs.stage1_step(
            self._stage,
            self._reward_cfg,
            palm_gap=gs.nearest_distance(palm_pos[:, None, :], surface)[:, 0],
            dz_free=dz_free,
            palm_shoe_dist=(palm_pos - shoe_pos).norm(dim=-1),
            shoe_shift_xy=shift_xy,
            thumb_curl=thumb_curl,
            rel_speed=rel_speed,
            shoe_speed=shoe_vel.norm(dim=-1),
            q=q,
            hand_floor_depth=(layout.TABLE_TOP_Z + self.cfg.hand_floor_offset - hand_z).clamp(min=0.0),
            arm_speed_sum=self._robot.data.joint_vel[:, self._arm_ids].abs().sum(dim=-1),
            hand_speed_sum=self._robot.data.joint_vel[:, self._hand_ids].abs().sum(dim=-1),
            hand_command_rate=self._hand_command_rate,
        )
        self._q_at_latch = torch.where(step.just_latched, q, self._q_at_latch)
        self._q_at_success = torch.where(step.success, q, self._q_at_success)
        if self.cfg.capture_success_states:
            # the same step's reset overwrites the joints, their targets and the shoe pose (auto-loop design §5)
            self._capture = gs.capture_rows(
                self._capture,
                step.success,
                joint_pos=self._robot.data.joint_pos,
                joint_target=gs.holding_targets(self._joint_targets, self._robot.data.joint_pos, self._arm_ids, self._hand_ids, self._hand_targets),
                shoe_pose=torch.cat([shoe_pos, self._shoe.data.root_quat_w], dim=-1),
                palm_pose=torch.cat([palm_pos, self._robot.data.body_quat_w[:, self._palm]], dim=-1),
                step=self.common_step_counter,
            )
        self._stage, self._last = step.state, step
        over_rack = (dz_free == 0.0) & (surface[..., 2].min(dim=-1).values - self._start_bottom_z > 0.02)
        log = {f"grasp_reward/{name}": value.mean().item() for name, value in step.terms.items()}
        log.update({
            "grasp/q": q.mean().item(),
            "grasp/palm_cos": palm_cos.mean().item(),
            "grasp/dz_free": dz_free.mean().item(),
            "grasp/shift_xy": shift_xy.mean().item(),
            "grasp/thumb_curl": thumb_curl.mean().item(),
            "grasp/rel_speed": rel_speed.mean().item(),
            "grasp/held_frac": step.held.float().mean().item(),
            "grasp/latched_frac": step.state.latched.float().mean().item(),
            "grasp/over_rack_raised_frac": over_rack.float().mean().item(),
            "grasp/arm_speed_sum": self._robot.data.joint_vel[:, self._arm_ids].abs().sum(dim=-1).mean().item(),
            "grasp/hand_speed_sum": self._robot.data.joint_vel[:, self._hand_ids].abs().sum(dim=-1).mean().item(),
            "grasp/hand_command_rate": self._hand_command_rate.mean().item(),
            "grasp/lost_frac": step.lost.float().mean().item(),
            **{f"grasp/w_{finger}": w_f[:, i].mean().item() for i, finger in enumerate(gb.FINGERS)},
        })
        log.update(self._episode_log)
        self.extras["log"] = log
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        dropped = shoe_pos[:, 2] < self.cfg.drop_z
        return (step.success | dropped | step.lost) & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        return self._last.reward

    # ------------------------------------------------------ success capture

    def clear_success_captures(self) -> None:
        self._capture = gs.SuccessCapture.empty(self.num_envs, self._robot.num_joints, self.device)

    def restore_success_captures(self, env_ids: torch.Tensor) -> None:
        """Write the captured success states of ``env_ids`` back as the start of a fresh episode (auto-loop design §5)."""
        if not bool(self._capture.valid[env_ids].all()):
            raise ValueError("restore_success_captures: some of the envs hold no capture")
        count, dev = len(env_ids), self.device
        joint_pos, joint_target = self._capture.joint_pos[env_ids], self._capture.joint_target[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_target
        self._hand_targets[env_ids] = joint_target[:, self._hand_ids]
        shoe_pose = self._capture.shoe_pose[env_ids].clone()
        self._start_xy[env_ids] = shoe_pose[:, :2]
        self._start_bottom_z[env_ids] = self._bottom_z(shoe_pose)
        shoe_pose[:, :3] += self.scene.env_origins[env_ids]
        self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
        self._shoe.write_root_velocity_to_sim(torch.zeros(count, 6, device=dev), env_ids=env_ids)
        self._stage = self._stage.reset_rows(env_ids)
        self._q_at_latch[env_ids] = 0.0
        self._q_at_success[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self._wrench.reset(env_ids)
        self.actions[env_ids] = 0.0
        self._prev_hand_valid[env_ids] = False
        self._ik.reset(env_ids)

    def _bottom_z(self, shoe_pose: torch.Tensor) -> torch.Tensor:
        """(K,) height of the lowest surface point of env-local shoe poses (K, 7)."""
        count, points = shoe_pose.shape[0], self._surface_local.shape[0]
        surface = quat_apply(
            shoe_pose[:, None, 3:].expand(count, points, 4).reshape(-1, 4), self._surface_local.expand(count, -1, 3).reshape(-1, 3)
        ).view(count, points, 3)
        return (surface[..., 2] + shoe_pose[:, None, 2]).min(dim=-1).values

    # ----------------------------------------------------------------- reset

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(finished) == 0:
            return
        latched, succeeded = self._stage.latched[finished], self._stage.succeeded[finished]
        self._episode_log = dict(zip(EPISODE_LOG_KEYS, (
            succeeded.float().mean().item(),
            latched.float().mean().item(),
            self._stage.best_lift[finished].mean().item(),
            self._q_at_latch[finished][latched].mean().item() if bool(latched.any()) else 0.0,
            self._q_at_success[finished][succeeded].mean().item() if bool(succeeded.any()) else 0.0,
        )))
        self.extras.setdefault("log", {}).update(self._episode_log)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._log_episode_end(env_ids)
        super()._reset_idx(env_ids)
        count, dev = len(env_ids), self.device
        pick = torch.randint(0, self._bank.size, (count,), device=dev)
        joint_pos = self._bank.joint_pos[pick].clone()
        joint_target = self._bank.joint_target[pick].clone()
        joint_pos[:, self._hand_ids] = self._hand_reset
        joint_target[:, self._hand_ids] = self._hand_reset
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_target
        self._hand_targets[env_ids] = self._hand_reset

        origins = self.scene.env_origins[env_ids]
        zero_velocity = torch.zeros(count, 6, device=dev)
        shoe_pose = self._bank.shoe_pose[pick].clone()
        shoe_pose[:, :2] += sample_uniform(-self.cfg.start_noise_xy, self.cfg.start_noise_xy, (count, 2), dev)
        self._start_xy[env_ids] = shoe_pose[:, :2]
        self._start_bottom_z[env_ids] = self._bottom_z(shoe_pose)
        shoe_pose[:, :3] += origins
        self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
        self._shoe.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        other_pose = torch.zeros(count, 7, device=dev)
        other_pose[:, :3] = self._other_pos + origins
        other_pose[:, :2] += sample_uniform(-self.cfg.other_shoe_pos_noise, self.cfg.other_shoe_pos_noise, (count, 2), dev)
        yaw = sample_uniform(-self.cfg.other_shoe_yaw_noise, self.cfg.other_shoe_yaw_noise, (count,), dev)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(count, 3)
        other_pose[:, 3:] = quat_mul(quat_from_angle_axis(yaw, z_axis), self._other_quat.expand(count, 4))
        self._other.write_root_pose_to_sim(other_pose, env_ids=env_ids)
        self._other.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self._stage = self._stage.reset_rows(env_ids)
        self._q_at_latch[env_ids] = 0.0
        self._q_at_success[env_ids] = 0.0
        self._wrench.reset(env_ids)
        self.actions[env_ids] = 0.0
        self._prev_hand_valid[env_ids] = False
        self._ik.reset(env_ids)
