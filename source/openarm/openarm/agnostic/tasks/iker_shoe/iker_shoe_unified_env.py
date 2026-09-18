"""IKER unified environment: ONE policy picks the shoe up off the table and places it on the rack (2026-09-18).

Why this class exists. Stage 1 (grasp, 26 actions / 78 observations) and stage 2 (place, 7 / 39) were trained apart and
joined only through a bank file. Measured on 2026-09-18: stage 1 succeeds in 0.43 of episodes, and the stage-2 policy
started from those handover states succeeds in 0.285, so the chained task ends around 0.12. The user's decision is one
policy over the whole task, with the stage-1 action space (all 20 finger joints) kept for both halves.

What this class adds to ``IkerShoeGraspT2rEnv`` (whose hold/latch predicate, contact sensors, hand law, IK and
disturbances are unchanged):
  * a longer episode that starts from the robot's rest posture with the shoe lying on the table at a drawn position,
  * the stage-2 placement predicate (``place_stage.place_step``) and its scripted return of the arm to the rest posture,
  * terminations of the whole task: the placement success, the shoe falling, or the time limit — NOT the stage-1 success,
  * the unified reward context (t2r3) and the generated reward that reads it,
  * optional shortened starts drawn from the stage-1 grasp bank (shoe already held) and the stage-2 adjust bank (shoe
    already set down off target), so the later halves are practised before the first half works.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.utils.math import quat_apply, sample_uniform

from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT

from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS, transform_keypoints

from . import adjust_bank as ab
from . import grasp_bank as gb
from . import grasp_stage as gs
from . import layout
from . import place_stage as ps
from .iker_shoe_grasp_env import ARM_ACTION_DIM
from .iker_shoe_grasp_t2r_env import IkerShoeGraspT2rEnv
from .iker_shoe_unified_env_cfg import IkerShoeUnifiedEnvCfg
from .t2r3.context import RewardContext
from .t2r3.loader import call_reward_fn


class IkerShoeUnifiedEnv(IkerShoeGraspT2rEnv):
    cfg: IkerShoeUnifiedEnvCfg

    def __init__(self, cfg: IkerShoeUnifiedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._place_window = ps.new_window(n, cfg.place, dev)
        self._carried = torch.zeros(n, dtype=torch.bool, device=dev)
        self._start_held = torch.zeros(n, dtype=torch.bool, device=dev)
        self._retracting = torch.zeros(n, dtype=torch.bool, device=dev)
        self._retract_k = torch.zeros(n, dtype=torch.long, device=dev)
        self._arm_ids_t = torch.tensor(self._arm_ids, device=dev)  # find_joints gives a list; advanced indexing needs a tensor
        self._retract_q0 = torch.zeros(n, self._arm_ids_t.numel(), device=dev)
        self._home_arm_q = self._robot.data.default_joint_pos[0, self._arm_ids].clone()
        self._unified_last: dict[str, torch.Tensor] | None = None
        self._unified_log: dict[str, float] | None = None
        # success per start kind, written every step: rl_games' observer indexes every step's log with the keys of the
        # first one it saw, so a key that appears only on reset steps raises KeyError (stage-2, 2026-09-18).
        self._start_log = {"place/success_table_start": 0.0, "place/success_held_start": 0.0}

        # The hand's open fraction: 0 at the profile's grip pose, 1 at the hand pose a reset puts the hand in (the
        # open pose clamped into the action range — the profile's raw open pose is NOT commandable on every joint, and
        # measuring against it read 0.55 for a freshly reset open hand, which the 0.9 takeover threshold never clears).
        self._grip_pose = torch.tensor(TESOLLO_LEFT_SHORT.hand_grip_pose, device=dev)
        self._open_pose = self._hand_reset.clone()
        self._grip_span_mask = (self._open_pose - self._grip_pose).abs() > 1e-3
        if not bool(self._grip_span_mask.any()):
            raise RuntimeError("the hand profile's grip and open poses are identical; the open fraction is undefined")
        # the placement keypoints of the episode start; the stage-1 parent does not track them
        self._init_keypoints = torch.zeros(n, NUM_KEYPOINTS, 3, device=dev)

        # `home` of the placement predicate: the arm back in its rest posture. The palm position it corresponds to is
        # measured once by writing the default joint state and taking a single physics step (a written pose reads back
        # only after a step), holding position targets plus gravity compensation exactly as _apply_action does.
        home_q = self._robot.data.default_joint_pos.clone()
        self._robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q))
        self._robot.set_joint_position_target(home_q)
        tau = self._robot.root_physx_view.get_gravity_compensation_forces()
        self._robot.set_joint_effort_target(tau[:, self._gravity_ids], joint_ids=self._gravity_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.physics_dt)
        home_palm = self._robot.data.body_pos_w[:, self._palm] - self.scene.env_origins
        self._home_palm_pos = home_palm.mean(dim=0)
        spread = float((home_palm - self._home_palm_pos).norm(dim=-1).max())
        if spread > 1e-3:
            raise RuntimeError(f"home palm position disagrees across envs by {spread:.4f} m; measurement is wrong")

        self._held_bank = self._load_held_bank()
        self._adjust_bank = (ab.load_bank(cfg.adjust_bank_path, list(self._robot.data.joint_names), dev)
                             if cfg.adjust_bank_path and cfg.adjust_start_frac > 0.0 else None)
        home = ", ".join(f"{v:.3f}" for v in self._home_palm_pos.tolist())
        print(f"[iker_unified] episode {self.max_episode_length} steps · spawn noise {cfg.spawn_noise_xy:.3f} m · "
              f"held starts {cfg.held_start_frac:.2f} ({self._held_bank.size if self._held_bank else 0}) · "
              f"adjust starts {cfg.adjust_start_frac:.2f} ({self._adjust_bank.size if self._adjust_bank else 0}) · "
              f"home palm ({home})", flush=True)

    def _load_held_bank(self):
        """The stage-1 grasp bank, validated against this run exactly as the stage-2 environment validates it."""
        cfg = self.cfg
        if not cfg.grasp_bank_start_path or cfg.held_start_frac <= 0.0:
            return None
        from openarm.agnostic.modules.iker import run_files

        doc = run_files.read_json(Path(cfg.grasp_bank_start_path))
        expected = json.loads(json.dumps({key: self._boot_metadata[key] for key in gb.LEARNED_BOOT_KEYS}))
        return gb.load_bank(doc, list(self._robot.data.joint_names), expected, self.device)

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        if bool(self._retracting.any()):
            # scripted return: the environment drives the arm to the rest posture and holds the hand open; the policy's
            # actions have no effect from the takeover on (same law as the stage-2 environment)
            ids = self._retracting.nonzero(as_tuple=True)[0]
            self._joint_targets[ids[:, None], self._arm_ids_t[None, :]] = ps.retract_targets(
                self._retract_q0[ids], self._home_arm_q, self._retract_k[ids], self.cfg.place)
            self._retract_k[ids] += 1
            self._hand_targets[ids] = self._open_pose
            self._joint_targets[ids[:, None], self._hand_ids[None, :]] = self._open_pose

    def open_fraction(self) -> torch.Tensor:
        """(N,) how far the filtered finger targets have travelled from the grip pose towards the open pose."""
        span = (self._open_pose - self._grip_pose)[self._grip_span_mask]
        moved = (self._hand_targets[:, self._grip_span_mask] - self._grip_pose[self._grip_span_mask])
        return (moved / span).mean(dim=-1).clamp(0.0, 1.0)

    # ------------------------------------------------------- reward / dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        super()._get_dones()  # stage-1 hold/latch/success bookkeeping, contact logs; its terminations are replaced below
        step1 = self._last
        origins = self.scene.env_origins
        surface = self._shoe_surface()
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        shoe_pos = self._shoe.data.root_pos_w - origins
        shoe_quat = self._shoe.data.root_quat_w
        keypoints = self._keypoints_local()
        targets = self._targets.expand(self.num_envs, NUM_KEYPOINTS, 3)
        keypoint_err = (keypoints - targets).norm(dim=-1)
        keypoint_dist = keypoint_err.mean(dim=-1)
        shoe_bottom_z = surface[..., 2].min(dim=-1).values
        palm_shoe_dist = (palm_pos - shoe_pos).norm(dim=-1)
        arm_q = self._robot.data.joint_pos[:, self._arm_ids]
        arm_home_err = (arm_q - self._home_arm_q).abs().max(dim=-1).values
        shoe_vel = self._shoe.data.root_lin_vel_w

        step = ps.place_step(keypoint_dist, palm_shoe_dist, shoe_bottom_z, shoe_vel.norm(dim=-1), self._place_window,
                             self.cfg.place, shoe_ang_speed=self._shoe.data.root_ang_vel_w.norm(dim=-1),
                             arm_home_err=arm_home_err)
        self._place_window = step.window
        self._keypoint_distance = keypoint_dist
        self._carried |= step1.state.latched

        open_frac = self.open_fraction()
        start = ps.retract_trigger(step.placed, step.resting, step.still, open_frac, self._retracting, self.cfg.place)
        if bool(start.any()):
            self._retracting |= start
            self._retract_k[start] = 0
            self._retract_q0[start] = arm_q[start]

        self._unified_last = dict(
            palm_pos=palm_pos, palm_quat=self._robot.data.body_quat_w[:, self._palm], arm_q=arm_q,
            arm_qd=self._robot.data.joint_vel[:, self._arm_ids], shoe_pos=shoe_pos, shoe_quat=shoe_quat,
            shoe_lin_vel=shoe_vel, shoe_ang_vel=self._shoe.data.root_ang_vel_w, shoe_surface=surface,
            shoe_bottom_z=shoe_bottom_z, palm_shoe_dist=palm_shoe_dist, keypoints=keypoints,
            target_keypoints=targets, keypoint_err=keypoint_err, keypoint_dist=keypoint_dist,
            arm_home_err=arm_home_err, placed=step.placed, released=step.released, resting=step.resting,
            still=step.still, home=step.home, stable_count=step.stable_count, success=step.success,
            retracting=self._retracting.clone(), open_frac=open_frac,
        )
        dropped = shoe_pos[:, 2] < self.cfg.drop_z
        # the parent's counters drive iker/* logging: jump past its strict `> sustain_steps` comparison exactly on the
        # step the predicate is true (same fix as the stage-2 environment)
        sustain = float(self.cfg.reward.sustain_steps)
        self._success_count = torch.where(step.success, sustain + 1.0, torch.zeros_like(self._keypoint_distance))
        self._failure_count = torch.where(dropped, sustain + 1.0, torch.zeros_like(self._keypoint_distance))
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return (step.success | dropped) & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._unified_last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        ctx = self._build_unified_context()
        total, terms = call_reward_fn(self._reward_fn, ctx)
        raw = torch.cat([total.reshape(-1)] + [value.reshape(-1) for value in terms.values()])
        nonfinite_frac = (~torch.isfinite(raw)).float().mean()
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        self._t2r_prev_actions = self.actions.clone()

        last = self._unified_last
        log: dict[str, float] = {"t2r_reward/total": total.mean().item()}
        log.update({f"t2r_reward/{name}": value.mean().item() for name, value in terms.items()})
        log["t2r_reward/nonfinite_frac"] = float(nonfinite_frac.item())
        for key in ("placed", "released", "resting", "still", "home", "retracting"):
            log[f"place/{key}"] = last[key].float().mean().item()
        log["place/carried"] = self._carried.float().mean().item()
        log["place/open_frac"] = last["open_frac"].mean().item()
        log.update(self._start_log)
        self.extras.setdefault("log", {}).update(log)
        self._unified_log = log
        return total

    def _build_unified_context(self) -> RewardContext:
        n, origins, rc = self.num_envs, self.scene.env_origins, self._reward_cfg
        rd, sd, last, step1 = self._robot.data, self._shoe.data, self._unified_last, self._last
        links_force, palm_force = self._link_shoe_forces()
        surface = last["shoe_surface"]
        link_pos = (rd.body_pos_w[:, self._finger_links] - origins[:, None, :]).view(n, len(gb.FINGERS), -1, 3)
        q = rd.joint_pos[:, self._hand_ids]
        span = self._hand_hi - self._hand_lo
        values = dict(
            table_top_z=float(layout.TABLE_TOP_Z), rack_x_min=float(layout.RACK_X_RANGE[0]),
            rack_x_max=float(layout.RACK_X_RANGE[1]), rack_y_min=float(layout.RACK_Y_RANGE[0]),
            rack_y_max=float(layout.RACK_Y_RANGE[1]), rack_top_z=float(layout.RACK_TOP_Z),
            episode_steps=int(self.max_episode_length), control_dt=float(self.step_dt),
            lift_height=float(rc.lift_height_m), hold_radius=float(rc.hold_radius_m),
            hold_slip_speed=float(rc.hold_rel_speed), thumb_curl_min=float(rc.thumb_curl_min_rad),
            latch_steps=int(rc.latch_steps),
            place_tolerance=float(self.cfg.place.place_tolerance), release_radius=float(self.cfg.place.release_radius),
            resting_tol=float(self.cfg.place.resting_tol), still_speed=float(self.cfg.place.still_speed),
            stable_steps=int(self.cfg.place.stable_steps), window_steps=int(self.cfg.place.window_steps),
            home_joint_tol=float(self.cfg.place.home_joint_tol), retract_steps=int(self.cfg.place.retract_steps),
            retract_open_min=float(self.cfg.place.retract_open_min),
            palm_pos=last["palm_pos"], palm_quat=last["palm_quat"],
            palm_normal=quat_apply(last["palm_quat"], self._palmar_axis),
            link_pos=link_pos,
            link_shoe_gap=gs.nearest_distance(link_pos.reshape(n, -1, 3), surface).view(n, len(gb.FINGERS), -1),
            link_shoe_force=links_force, palm_shoe_force=palm_force,
            hand_q=q, hand_qd=rd.joint_vel[:, self._hand_ids], hand_q_norm=((q - self._hand_lo) / span).clamp(0.0, 1.0),
            hand_target_norm=((self._hand_targets - self._hand_lo) / span).clamp(0.0, 1.0),
            thumb_curl=gs.closing_travel(rd.joint_pos[:, self._thumb_curl_id], self._thumb_open, self._thumb_grip),
            hand_z_min=rd.body_pos_w[:, self._finger_links, 2].min(dim=1).values - origins[:, 2],
            arm_q=last["arm_q"], arm_qd=last["arm_qd"],
            home_palm_pos=self._home_palm_pos.expand(n, 3),
            palm_home_dist=(last["palm_pos"] - self._home_palm_pos).norm(dim=-1),
            arm_home_err=last["arm_home_err"], retracting=last["retracting"], open_frac=last["open_frac"],
            shoe_pos=last["shoe_pos"], shoe_quat=last["shoe_quat"], shoe_lin_vel=last["shoe_lin_vel"],
            shoe_ang_vel=last["shoe_ang_vel"], shoe_start_xy=self._start_xy, shoe_surface=surface,
            shoe_bottom_z=last["shoe_bottom_z"],
            dz_free=gs.free_lift_height(surface, self._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE),
            shoe_shift_xy=(last["shoe_pos"][:, :2] - self._start_xy).norm(dim=-1),
            palm_gap=gs.nearest_distance(last["palm_pos"][:, None, :], surface)[:, 0],
            palm_shoe_dist=last["palm_shoe_dist"],
            slip_speed=gs.palm_frame_slip_speed(sd.root_lin_vel_w, rd.body_lin_vel_w[:, self._palm],
                                                rd.body_ang_vel_w[:, self._palm], sd.root_com_pos_w,
                                                rd.body_com_pos_w[:, self._palm]),
            target_keypoints=last["target_keypoints"], keypoints=last["keypoints"], init_keypoints=self._init_keypoints,
            keypoint_err=last["keypoint_err"], keypoint_dist=last["keypoint_dist"],
            held=step1.held, hold_count=step1.state.hold_count.float(), latched=step1.state.latched,
            carried=self._carried.clone(), placed=last["placed"], released=last["released"], resting=last["resting"],
            still=last["still"], home=last["home"], stable_count=last["stable_count"], success=last["success"],
            start_held=self._start_held.clone(),
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self._t2r_prev_actions,
        )
        # the parent reads these buffers again after the reward: the generated code gets copies
        return RewardContext(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    # ----------------------------------------------------------------- log

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        super()._log_episode_end(env_ids)
        if self._unified_log is not None:
            self.extras.setdefault("log", {}).update(self._unified_log)
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(finished):
            won = self._success_count[finished] > self.cfg.reward.sustain_steps
            held_start = self._start_held[finished]
            for name, group in (("held", held_start), ("table", ~held_start)):
                if bool(group.any()):
                    self._start_log[f"place/success_{name}_start"] = won[group].float().mean().item()
        self.extras.setdefault("log", {}).update(self._start_log)

    # ---------------------------------------------------------------- reset

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)          # pre-grasp bank start, shoe rest pose, other shoe, stage-1 state
        count, dev = len(env_ids), self.device
        origins = self.scene.env_origins[env_ids]
        self._place_window[env_ids] = 0.0
        self._carried[env_ids] = False
        self._start_held[env_ids] = False
        self._retracting[env_ids] = False
        self._retract_k[env_ids] = 0
        self._t2r_prev_actions[env_ids] = 0.0

        # the whole task starts from the robot's rest posture with the hand open, not from the pre-grasp pose
        home_q = self._robot.data.default_joint_pos[env_ids].clone()
        home_q[:, self._hand_ids] = self._hand_reset
        self._robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q), env_ids=env_ids)
        self._robot.set_joint_position_target(home_q, env_ids=env_ids)
        self._joint_targets[env_ids] = home_q
        self._hand_targets[env_ids] = self._hand_reset

        # the shoe lies on the table at a drawn position (its recorded resting orientation is kept)
        shoe_pose = torch.cat([self._shoe.data.root_pos_w[env_ids] - origins, self._shoe.data.root_quat_w[env_ids]], dim=-1)
        shoe_pose[:, :2] += sample_uniform(-self.cfg.spawn_noise_xy, self.cfg.spawn_noise_xy, (count, 2), dev)
        self._write_start_state(env_ids, shoe_pose)

        if self._held_bank is not None:
            ids = env_ids[ab.start_mask(count, self.cfg.held_start_frac, dev)]
            if len(ids):
                pick = torch.randint(0, self._held_bank.size, (len(ids),), device=dev)
                self._restore(ids, self._held_bank.joint_pos[pick], self._held_bank.joint_target[pick],
                              self._held_bank.shoe_pose[pick])
                self._start_held[ids] = True
        if self._adjust_bank is not None:
            ids = env_ids[ab.start_mask(count, self.cfg.adjust_start_frac, dev)]
            if len(ids):
                pick = torch.randint(0, self._adjust_bank.size, (len(ids),), device=dev)
                self._restore(ids, self._adjust_bank.joint_pos[pick], self._adjust_bank.joint_target[pick],
                              self._adjust_bank.shoe_pose[pick])
                self._start_held[ids] = True

    def _restore(self, ids: torch.Tensor, joint_pos: torch.Tensor, joint_target: torch.Tensor, shoe_pose: torch.Tensor):
        """Put these envs into a recorded later state of the task (shoe held, or already set down)."""
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)
        self._robot.set_joint_position_target(joint_target, env_ids=ids)
        self._joint_targets[ids] = joint_target
        self._hand_targets[ids] = joint_target[:, self._hand_ids]
        self._write_start_state(ids, shoe_pose.clone())

    def _write_start_state(self, ids: torch.Tensor, shoe_pose: torch.Tensor):
        """Write the shoe pose (env-local) and refresh everything the episode measures against its start."""
        origins = self.scene.env_origins[ids]
        self._start_xy[ids] = shoe_pose[:, :2]
        # dz_free measures the rise above where the shoe LAY; a start that restores it already in the hand or already on
        # the rack must keep the table as that reference, or the grasp predicate can never read a lift in those episodes
        bottom = self._bottom_z(shoe_pose)
        self._start_bottom_z[ids] = torch.where(bottom > layout.TABLE_TOP_Z + self.cfg.place.resting_tol,
                                                torch.full_like(bottom, layout.TABLE_TOP_Z), bottom)
        self._init_keypoints[ids] = transform_keypoints(shoe_pose[:, :3], shoe_pose[:, 3:], self._offsets)
        world = shoe_pose.clone()
        world[:, :3] += origins
        self._shoe.write_root_pose_to_sim(world, env_ids=ids)
        self._shoe.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
