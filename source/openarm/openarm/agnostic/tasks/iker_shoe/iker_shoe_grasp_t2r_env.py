"""IKER stage-1 t2r grasp environment (spec 2026-09-15-iker-stage1-t2r §7, §14).

The parent ``IkerShoeGraspEnv`` still evaluates ``grasp_stage.stage1_step`` in ``_get_dones`` — hold, latch, success, lost,
terminations and the success capture that harvest reads are unchanged. This class only
  1. loads the generated reward before the scene is built (a wrong path fails before the boot),
  2. adds reward-only contact sensors: one per finger link (`_3`, `_4`, tip) and one on the palm, each filtered to the moving
     shoe's rigid-body prim (one body per sensor: a sensor over several bodies reads a silent zero force matrix),
  3. returns ``compute_reward(ctx)`` from ``_get_rewards`` — DirectRLEnv.step calls it after ``_get_dones`` and before the
     reset, so ctx is built from the same step state the predicate used — and swaps the hand-written reward's log keys for the
     generated terms and contact counts,
  4. zeroes the previous-action buffer of reset envs.
Observations do not change: contact is reward-only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from . import grasp_bank as gb
from . import grasp_stage as gs
from . import layout, robot
from .iker_shoe_grasp_env import IkerShoeGraspEnv
from .iker_shoe_grasp_t2r_env_cfg import IkerShoeGraspT2rEnvCfg
from .t2r.context import RewardContext
from .t2r.loader import call_reward_fn, load_reward_fn

TOUCH_LOG_N = 0.1  # contact force counted as touching in the diagnostic log only; the generated reward sets its own thresholds


class IkerShoeGraspT2rEnv(IkerShoeGraspEnv):
    cfg: IkerShoeGraspT2rEnvCfg

    def __init__(self, cfg: IkerShoeGraspT2rEnvCfg, render_mode: str | None = None, **kwargs):
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kwargs)
        self._t2r_prev_actions = torch.zeros(self.num_envs, int(cfg.action_space), device=self.device)
        digest = hashlib.sha256(Path(cfg.reward_code_path).read_bytes()).hexdigest() if cfg.reward_code_path else "none"
        sensors = sum(len(group) for group in self._t2r_link_sensors.values())
        print(f"[iker_grasp_t2r] reward {self._reward_src} sha256 {digest} · reward-only shoe contact sensors {sensors}+1 (palm) · "
              f"filter {self._t2r_filter} · observations unchanged", flush=True)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        super()._setup_scene()
        prof = robot.profile()
        shoe_filter = self._shoe_contact_filter()
        self._t2r_filter = shoe_filter
        robot_path = robot.ROBOT_PRIM_PATH
        self._t2r_link_sensors: dict[str, list[ContactSensor]] = {}
        for finger in gb.FINGERS:
            sensors = []
            for body in prof.finger_sensor_bodies[finger]:
                sensor = ContactSensor(ContactSensorCfg(prim_path=f"{robot_path}/{body}", filter_prim_paths_expr=shoe_filter,
                                                        history_length=1, track_air_time=False))
                self.scene.sensors[f"t2r_contact_{body}"] = sensor
                sensors.append(sensor)
            self._t2r_link_sensors[finger] = sensors
        self._t2r_palm_sensor = ContactSensor(ContactSensorCfg(prim_path=f"{robot_path}/{prof.palm_body}", filter_prim_paths_expr=shoe_filter,
                                                               history_length=1, track_air_time=False))
        self.scene.sensors["t2r_contact_palm"] = self._t2r_palm_sensor

    def _shoe_contact_filter(self) -> list[str]:
        """The contact filter from the moving shoe's rigid-body prim found on env_0's stage, not from a cfg string (grasp_fj_t2r 09.14:
        a cfg path that matched no prim left PhysX logging an error while every force read 0)."""
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics

        pattern = self.cfg.shoe_move_cfg.prim_path
        env0 = pattern.replace("env_.*", "env_0", 1)
        prims = sim_utils.get_all_matching_child_prims(env0, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if len(prims) != 1:
            raise RuntimeError(f"[iker_grasp_t2r] {env0} holds {len(prims)} rigid-body prims, need exactly 1: "
                               f"{[prim.GetPath().pathString for prim in prims]}")
        path = prims[0].GetPath().pathString
        if not path.startswith(env0):
            raise RuntimeError(f"[iker_grasp_t2r] shoe rigid-body prim {path} lies outside {env0}")
        return [pattern + path[len(env0):]]

    def _link_shoe_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(links (N,5,3), palm (N,)) shoe-filtered contact force magnitudes [N], fingers in gb.FINGERS order."""
        def magnitude(sensor: ContactSensor) -> torch.Tensor:
            return sensor.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

        links = torch.stack([torch.stack([magnitude(s) for s in self._t2r_link_sensors[f]], dim=1) for f in gb.FINGERS], dim=1)
        return links, magnitude(self._t2r_palm_sensor)

    # ----------------------------------------------------------------- reward
    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        ctx = self._build_context()
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        self._t2r_prev_actions = self.actions.clone()
        log = {key: value for key, value in self.extras.get("log", {}).items() if not key.startswith("grasp_reward/")}
        log["t2r_reward/total"] = total.mean().item()
        log.update({f"t2r_reward/{name}": value.mean().item() for name, value in terms.items()})
        touching = (ctx.link_shoe_force > TOUCH_LOG_N).float()
        log["contact/links_touching"] = touching.sum(dim=(1, 2)).mean().item()
        log["contact/fingers_touching"] = touching.amax(dim=2).sum(dim=1).mean().item()
        log["contact/palm_touching"] = (ctx.palm_shoe_force > TOUCH_LOG_N).float().mean().item()
        self.extras["log"] = log
        return total

    def _build_context(self) -> RewardContext:
        n, origins, rc, step = self.num_envs, self.scene.env_origins, self._reward_cfg, self._last
        rd, sd = self._robot.data, self._shoe.data
        links_force, palm_force = self._link_shoe_forces()
        surface = self._shoe_surface()
        palm_pos = rd.body_pos_w[:, self._palm] - origins
        palm_quat = rd.body_quat_w[:, self._palm]
        link_pos = (rd.body_pos_w[:, self._finger_links] - origins[:, None, :]).view(n, len(gb.FINGERS), -1, 3)
        shoe_pos = sd.root_pos_w - origins
        q = rd.joint_pos[:, self._hand_ids]
        span = self._hand_hi - self._hand_lo
        values = dict(
            table_top_z=float(layout.TABLE_TOP_Z), rack_x_min=float(layout.RACK_X_RANGE[0]), rack_x_max=float(layout.RACK_X_RANGE[1]),
            rack_y_min=float(layout.RACK_Y_RANGE[0]), rack_y_max=float(layout.RACK_Y_RANGE[1]),
            episode_steps=int(self.max_episode_length), control_dt=float(self.step_dt),
            lift_height=float(rc.lift_height_m), lift_max=float(rc.lift_max_m), hold_xy_radius=float(rc.hold_xy_radius_m),
            hold_radius=float(rc.hold_radius_m), hold_slip_speed=float(rc.hold_rel_speed), thumb_curl_min=float(rc.thumb_curl_min_rad),
            latch_steps=int(rc.latch_steps), success_steps=int(rc.success_steps), success_speed=float(rc.success_speed),
            palm_pos=palm_pos, palm_quat=palm_quat, palm_normal=quat_apply(palm_quat, self._palmar_axis),
            link_pos=link_pos, link_shoe_gap=gs.nearest_distance(link_pos.reshape(n, -1, 3), surface).view(n, len(gb.FINGERS), -1),
            link_shoe_force=links_force, palm_shoe_force=palm_force,
            hand_q=q, hand_qd=rd.joint_vel[:, self._hand_ids], hand_q_norm=((q - self._hand_lo) / span).clamp(0.0, 1.0),
            hand_target_norm=((self._hand_targets - self._hand_lo) / span).clamp(0.0, 1.0),
            thumb_curl=gs.closing_travel(rd.joint_pos[:, self._thumb_curl_id], self._thumb_open, self._thumb_grip),
            hand_z_min=rd.body_pos_w[:, self._finger_links, 2].min(dim=1).values - origins[:, 2],
            arm_q=rd.joint_pos[:, self._arm_ids], arm_qd=rd.joint_vel[:, self._arm_ids],
            shoe_pos=shoe_pos, shoe_quat=sd.root_quat_w, shoe_lin_vel=sd.root_lin_vel_w, shoe_ang_vel=sd.root_ang_vel_w,
            shoe_start_xy=self._start_xy, shoe_surface=surface,
            dz_free=gs.free_lift_height(surface, self._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE),
            shoe_shift_xy=(shoe_pos[:, :2] - self._start_xy).norm(dim=-1),
            palm_gap=gs.nearest_distance(palm_pos[:, None, :], surface)[:, 0], palm_shoe_dist=(palm_pos - shoe_pos).norm(dim=-1),
            slip_speed=gs.palm_frame_slip_speed(sd.root_lin_vel_w, rd.body_lin_vel_w[:, self._palm], rd.body_ang_vel_w[:, self._palm],
                                                sd.root_com_pos_w, rd.body_com_pos_w[:, self._palm]),
            held=step.held, hold_count=step.state.hold_count.float(), latched=step.state.latched, success=step.success,
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self._t2r_prev_actions,
        )
        # the parent reads these buffers again after the reward: the generated code gets copies (its in-place ops cannot leak)
        return RewardContext(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ids = self._robot._ALL_INDICES if env_ids is None else env_ids
        previous = getattr(self, "_t2r_prev_actions", None)
        if previous is not None:
            previous[ids] = 0.0
