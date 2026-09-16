"""IKER stage-2 t2r placement environment (spec 2026-09-16-iker-stage2-t2r §3-§6).

The parent ``IkerShoeEnv`` still owns the bank, IK, gravity compensation, scene, events and domain
randomisation. This class replaces the fixed-reward path with a generated one and adds the grip axis the
fixed policy never had:
  1. the action grows from 6 (palm delta) to 7 — ``actions[:, 6]`` is a single grip axis that interpolates the
     20 hand joints between the grasp bank's grip pose (-1) and the profile's open pose (+1), through the
     stage-1 EMA law (``grasp_stage.hand_targets``, not a new filter);
  2. the observation grows from 38 to 39 with the EMA-filtered grip state, so the (non-recurrent) policy can
     perceive its own grip;
  3. ``_get_dones`` replaces the old end-of-episode keypoint check with ``place_stage.place_step``'s four-way,
     consecutive-steps predicate (placed & released & resting & still), so a hovering hand cannot "succeed";
  4. ``_get_rewards`` builds a ``RewardContext`` from the same step state the predicate used and calls
     generated code (``t2r2.loader``) instead of the fixed IKER reward.

``grasp_lo``/``grasp_hi`` note: the profile's per-joint (grip_pose, open_pose) pair is not consistently
ordered (e.g. left thumb_3 grip=-1.8 < open=0, but left index_2 grip=1.9 > open=0 — see
``robot_profiles.TESOLLO_LEFT_SHORT``). ``grasp_stage.hand_targets``'s final clamp assumes ``lo <= hi``; passing
the two poses in as literally lo=grip/hi=open would freeze every joint where grip > open at the grip pose
forever (the clamp's ``min(x, hi)`` then ``max(., lo)`` collapses to ``lo`` whenever ``hi < lo`` — verified by
hand: index_2 asked to open would compute a correct EMA state then get clamped straight back to 1.9). The
fix keeps every one of ``hand_targets``'s own EMA/clamp lines untouched (still "reuse, don't invent"): sort
each joint's bounds (``lo=min(grip,open)``, ``hi=max(grip,open)``) and flip the action's sign per joint with
``sign(open_pose - grip_pose)`` so ``raw`` still lands on grip at action=-1 and open at action=+1 regardless of
which pose is numerically larger.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from isaaclab.utils.math import quat_apply, sample_uniform, subtract_frame_transforms

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS

from . import grasp_stage as gs
from . import layout, robot
from . import place_stage as ps
from .iker_shoe_env import IkerShoeEnv
from .iker_shoe_t2r_env_cfg import IkerShoeT2rEnvCfg
from .t2r2.context import RewardContext
from .t2r2.loader import call_reward_fn, load_reward_fn

PALMAR_AXIS_LOCAL = (1.0, 0.0, 0.0)  # palm link +x is the grasping side (grasp_bank.palm_rotations, both hands)
RETREAT_M = 0.15  # the palm within this of its start-of-episode pose counts as "returned" (spec: "원래 자리로 돌아오게")


class IkerShoeT2rEnv(IkerShoeEnv):
    cfg: IkerShoeT2rEnvCfg

    def __init__(self, cfg: IkerShoeT2rEnvCfg, render_mode: str | None = None, **kwargs):
        self._reward_fn, self._reward_origin = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        prof = robot.profile()
        joint_names = list(self._robot.data.joint_names)
        self._hand_ids = torch.tensor([joint_names.index(name) for name in prof.hand_joint_names], device=dev)
        meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
        self._surface_local = gs.surface_subsample(meta["objects"][layout.MOVING_SHOE]["hull_local"]).to(dev)
        self._palmar_axis = torch.tensor(PALMAR_AXIS_LOCAL, device=dev).expand(n, 3)

        grip_pose = torch.tensor(prof.hand_grip_pose, device=dev)
        open_pose = torch.tensor(prof.hand_open_pose, device=dev)
        self._grip_lo = torch.minimum(grip_pose, open_pose)
        self._grip_hi = torch.maximum(grip_pose, open_pose)
        self._grip_dir = torch.sign(open_pose - grip_pose)  # 0 for the frozen joints (grip == open, harmless)

        self._grip_targets = self._joint_targets[:, self._hand_ids].clone()  # placeholder; _reset_idx sets the real value
        self._grip_scalar = -torch.ones(n, 1, device=dev)
        self._stable_count = torch.zeros(n, device=dev)
        self._t2r_prev_actions = torch.zeros(n, int(cfg.action_space), device=dev)
        # fix round 1 (finding 3): env-local palm position captured at reset, for place/retreated ("did the
        # hand come back near where it started"). Placeholder here; _reset_idx sets the real value.
        self._palm_start = self._robot.data.body_pos_w[:, self._palm] - self.scene.env_origins
        self._t2r_last: dict[str, torch.Tensor] | None = None
        # fix round 2: this step's t2r_reward/*+place/* log, so _log_episode_end can merge it back into
        # self.extras["log"] after the parent's own _log_episode_end replaces that dict wholesale.
        self._t2r_log: dict[str, float] | None = None

        digest = hashlib.sha256(Path(cfg.reward_code_path).read_bytes()).hexdigest() if cfg.reward_code_path else "none"
        print(f"[iker_t2r] reward {self._reward_origin} sha256 {digest} · hand joints {len(self._hand_ids)} · "
              f"observations 39 (38 + grip) · action 7 (6 palm + 1 grip)", flush=True)

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        command = self.actions[:, :6]
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

        a = self.actions[:, 6]  # the grip axis: -1 bank grip pose, +1 profile open pose
        self._grip_targets = gs.hand_targets(a[:, None] * self._grip_dir, self._grip_lo, self._grip_hi,
                                             self._grip_targets, alpha=gs.HAND_EMA_ALPHA)
        self._joint_targets[:, self._hand_ids] = self._grip_targets
        # a second, scalar-only EMA in the same [-1,1] convention as the action itself: the per-joint state above
        # is not a single number (20 joints, two of which are direction-flipped), so this is the true "one grip
        # axis" state observed below and by the generated reward (grip_norm), mathematically the same EMA of a=-1
        # bank -> +1 open, expressed once instead of reconstructed (and sign-corrected) from an arbitrary joint.
        self._grip_scalar = gs.hand_targets(a[:, None], -torch.ones_like(self._grip_scalar), torch.ones_like(self._grip_scalar),
                                            self._grip_scalar, alpha=gs.HAND_EMA_ALPHA)

    # ---------------------------------------------------------- observation

    def _get_observations(self) -> dict:
        origins = self.scene.env_origins
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        palm_quat = self._robot.data.body_quat_w[:, self._palm]
        shoe_pos = self._shoe.data.root_pos_w - origins
        shoe_quat = self._shoe.data.root_quat_w
        if self.cfg.add_noise:
            palm_quat, shoe_quat = self._noisy_quat(palm_quat), self._noisy_quat(shoe_quat)
        n = self.num_envs
        grip_norm = gs.normalized_targets(self._grip_scalar, -torch.ones_like(self._grip_scalar), torch.ones_like(self._grip_scalar))
        obs = torch.cat(
            [
                palm_pos,
                palm_quat[:, [1, 2, 3, 0]],
                shoe_pos,
                shoe_quat[:, [1, 2, 3, 0]],
                self._keypoints_local().reshape(n, -1),
                self._targets.expand(n, NUM_KEYPOINTS, 3).reshape(n, -1),
                grip_norm,
            ],
            dim=-1,
        )
        if self.cfg.add_noise:
            obs = obs + sample_uniform(-self.cfg.observation_noise, self.cfg.observation_noise, obs.shape, self.device)
        return {"policy": obs}

    # ------------------------------------------------------- reward / dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        rd, sd = self._robot.data, self._shoe.data
        n, p = self.num_envs, self._surface_local.shape[0]
        palm_pos = rd.body_pos_w[:, self._palm] - origins
        palm_quat = rd.body_quat_w[:, self._palm]
        shoe_pos = sd.root_pos_w - origins
        shoe_quat = sd.root_quat_w
        quat = shoe_quat[:, None, :].expand(n, p, 4).reshape(-1, 4)
        surface = quat_apply(quat, self._surface_local.expand(n, p, 3).reshape(-1, 3)).view(n, p, 3) + shoe_pos[:, None, :]
        shoe_bottom_z = surface[..., 2].min(dim=-1).values
        palm_gap = gs.nearest_distance(palm_pos[:, None, :], surface)[:, 0]
        palm_shoe_dist = (palm_pos - shoe_pos).norm(dim=-1)
        current = self._keypoints_local()
        targets = self._targets.expand(n, NUM_KEYPOINTS, 3)
        keypoint_err = (current - targets).norm(dim=-1)
        keypoint_dist = keypoint_err.mean(dim=-1)
        shoe_speed = sd.root_lin_vel_w.norm(dim=-1)

        step = ps.place_step(keypoint_dist, palm_shoe_dist, shoe_bottom_z, shoe_speed, self._stable_count, self.cfg.place)
        self._stable_count = step.stable_count
        self._keypoint_distance = keypoint_dist  # keep the parent's field valid for _log_episode_end / eval_iker.py

        # fix round 1 (finding 3): the parent's own _log_episode_end (which this class cannot override — it is
        # not one of the 7 allowed hooks) reads _success_count/_failure_count to log iker/sustained_success and
        # iker/dropped. Nothing in this class's _get_dones updated them before, so both were always 0. Feed them
        # from the new predicate instead: success from the consecutive stable count (place_step's own counter —
        # its sustain_steps default (20) already matches cfg.place.stable_steps), failure by accumulating the
        # fall condition below, the same "+= condition" shape the old compute_reward_and_termination used.
        self._success_count = step.stable_count
        dropped = shoe_pos[:, 2] < self.cfg.reward.fall_height  # fix round 1 (finding 1): inherited threshold, not a new drop_z
        self._failure_count = self._failure_count + dropped.float()

        self._t2r_last = dict(
            palm_pos=palm_pos, palm_quat=palm_quat, arm_q=rd.joint_pos[:, self._arm_ids], arm_qd=rd.joint_vel[:, self._arm_ids],
            shoe_pos=shoe_pos, shoe_quat=shoe_quat, shoe_lin_vel=sd.root_lin_vel_w, shoe_ang_vel=sd.root_ang_vel_w,
            shoe_surface=surface, shoe_bottom_z=shoe_bottom_z, palm_gap=palm_gap, palm_shoe_dist=palm_shoe_dist,
            target_keypoints=targets, keypoints=current, init_keypoints=self._init_keypoints,
            keypoint_err=keypoint_err, keypoint_dist=keypoint_dist,
            placed=step.placed, released=step.released, resting=step.resting, still=step.still,
            stable_count=step.stable_count, success=step.success,
        )

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return (step.success | dropped) & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._t2r_last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        ctx = self._build_context()
        total, terms = call_reward_fn(self._reward_fn, ctx)
        raw = torch.cat([total.reshape(-1)] + [value.reshape(-1) for value in terms.values()])
        nonfinite_frac = (~torch.isfinite(raw)).float().mean()
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        self._t2r_prev_actions = self.actions.clone()

        log = dict(self.extras.get("log", {}))
        log["t2r_reward/total"] = total.mean().item()
        log.update({f"t2r_reward/{name}": value.mean().item() for name, value in terms.items()})
        log["t2r_reward/nonfinite_frac"] = float(nonfinite_frac.item())
        log["place/placed"] = ctx.placed.float().mean().item()
        log["place/released"] = ctx.released.float().mean().item()
        log["place/resting"] = ctx.resting.float().mean().item()
        log["place/still"] = ctx.still.float().mean().item()
        retreated = (self._t2r_last["palm_pos"] - self._palm_start).norm(dim=-1) <= RETREAT_M
        log["place/retreated"] = retreated.float().mean().item()
        self.extras["log"] = log
        self._t2r_log = log  # fix round 2: _log_episode_end re-merges this after the parent's own log write
        return total

    def _build_context(self) -> RewardContext:
        n, rc, last = self.num_envs, self.cfg.place, self._t2r_last
        values = dict(
            table_top_z=float(layout.TABLE_TOP_Z), rack_x_min=float(layout.RACK_X_RANGE[0]), rack_x_max=float(layout.RACK_X_RANGE[1]),
            rack_y_min=float(layout.RACK_Y_RANGE[0]), rack_y_max=float(layout.RACK_Y_RANGE[1]), rack_top_z=float(rc.rack_top_z),
            episode_steps=int(self.max_episode_length), control_dt=float(self.step_dt),
            place_tolerance=float(rc.place_tolerance), release_radius=float(rc.release_radius),
            resting_tol=float(rc.resting_tol), still_speed=float(rc.still_speed), stable_steps=int(rc.stable_steps),
            palm_pos=last["palm_pos"], palm_quat=last["palm_quat"], palm_normal=quat_apply(last["palm_quat"], self._palmar_axis),
            arm_q=last["arm_q"], arm_qd=last["arm_qd"],
            grip_norm=self._grip_scalar.reshape(n),
            shoe_pos=last["shoe_pos"], shoe_quat=last["shoe_quat"], shoe_lin_vel=last["shoe_lin_vel"], shoe_ang_vel=last["shoe_ang_vel"],
            shoe_surface=last["shoe_surface"], shoe_bottom_z=last["shoe_bottom_z"], palm_gap=last["palm_gap"], palm_shoe_dist=last["palm_shoe_dist"],
            target_keypoints=last["target_keypoints"], keypoints=last["keypoints"], init_keypoints=last["init_keypoints"],
            keypoint_err=last["keypoint_err"], keypoint_dist=last["keypoint_dist"],
            placed=last["placed"], released=last["released"], resting=last["resting"], still=last["still"],
            stable_count=last["stable_count"], success=last["success"],
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self._t2r_prev_actions,
        )
        # the parent reads several of these buffers again after the reward: the generated code gets copies (its
        # in-place ops cannot leak into live buffers)
        return RewardContext(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    # ------------------------------------------------------------------ log

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        """fix round 2: the parent's own version (called from its _reset_idx, which our _reset_idx calls via
        super()) does ``self.extras["log"] = {...iker/* keys...}`` — a full replace, not an update. Since
        _reset_idx runs after _get_rewards within the same step(), that replace was silently dropping this
        step's t2r_reward/*/place/* keys on every step that resets any env — with num_envs in the thousands
        and 200-step episodes, that is nearly every step. Call the parent first so iker/* is produced exactly
        as before, then merge our own log back in rather than letting the parent's replace stand alone."""
        super()._log_episode_end(env_ids)
        if self._t2r_log is not None:
            self.extras.setdefault("log", {}).update(self._t2r_log)

    # ----------------------------------------------------------------- reset

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._t2r_prev_actions[env_ids] = 0.0
        self._grip_targets[env_ids] = self._joint_targets[env_ids][:, self._hand_ids]  # the bank's hand target
        self._grip_scalar[env_ids] = -1.0
        self._stable_count[env_ids] = 0.0
        # fix round 1 (finding 3): the parent's bank restore above already wrote the new joint state via
        # write_joint_state_to_sim, so body_pos_w already reflects it (same idiom eval_iker.py's FirstEpisodeRecorder
        # uses right after reset_player, no extra physics step needed).
        self._palm_start[env_ids] = self._robot.data.body_pos_w[env_ids, self._palm] - self.scene.env_origins[env_ids]
