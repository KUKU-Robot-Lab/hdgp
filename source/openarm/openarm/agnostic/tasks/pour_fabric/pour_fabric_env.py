"""pour_fabric — 양팔 잡기→들기→붓기 (Fabrics ×2 + 시너지 손 ×2, robot-agnostic, direct).

★09.13 재작성. 보상은 여기 없다 — `RewardContext` 를 채워 `cfg.reward_code_path` 의
  생성 코드(`modules/t2r`)에 넘긴다. 성공 판정(fill·spill·xy)은 env 가 계산하고 ctx 에
  넣는다: 보상이 바꿀 수 없는 **기준 지표**다.

에피소드: reset(테이블 위 컵 2 + 소스 컵 안 비드) → hold(비드 정착, 팔 고정) → 정책.
제어: 팔 = palm 6D(앵커+델타) → Fabrics → 관절 PD. 손 = 시너지 폐쇄도(접촉 동결).
"""

from __future__ import annotations

import math
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from openarm.agnostic.modules.t2r.context import RewardContext
from openarm.agnostic.modules.t2r.loader import call_reward_fn, load_reward_fn
from openarm.common.bead_assets import bead_offsets_in_cup

from . import bimanual as _bm
from . import pour_fabric_env_cfg as _cfg
from .bead_flags import BeadGeometry, compute_bead_flags
from .pour_fabric_env_cfg import PourFabricEnvCfg
from .side_rig import SideRig

_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "FABRICS", "src"))
if os.path.isdir(_FABRICS_SRC) and _FABRICS_SRC not in sys.path:
    sys.path.insert(0, _FABRICS_SRC)

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh   # noqa: E402
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tes  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp                   # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel      # noqa: E402


def _fabric_class(name: str):
    for mod in (_fab_tes, _fab_rh):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Fabrics 클래스 '{name}' 를 찾을 수 없다")


class PourFabricEnv(DirectRLEnv):
    cfg: PourFabricEnvCfg

    # ==================================================================
    def __init__(self, cfg: PourFabricEnvCfg, render_mode: str | None = None, **kw):
        _cfg.resolve_cfg(cfg)
        self.pair = _bm.get_pair(cfg.pair_name)
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        sp = cfg.robot_cfg.spawn
        if (bool(sp.rigid_props.disable_gravity) == bool(cfg.enable_gravity)
                or bool(sp.articulation_props.enabled_self_collisions) != bool(cfg.enable_self_collisions)):
            raise RuntimeError("물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로 확인")
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kw)

        N, dev = self.num_envs, self.device
        # 소스=우(붓기)·리시버=좌. 대향 관절 미러 부호: thumb_2 축 Z → 좌측 −1.
        self.src = SideRig(self, self.pair.source, role="src",
                           sensors=self._sensor_store["src"], palm_sensor=self._palm_store["src"],
                           delta_lo=cfg.src_palm_delta_lo, delta_hi=cfg.src_palm_delta_hi,
                           oppose_sign=1.0)
        self.rcv = SideRig(self, self.pair.receiver, role="rcv",
                           sensors=self._sensor_store["rcv"], palm_sensor=self._palm_store["rcv"],
                           delta_lo=cfg.rcv_palm_delta_lo, delta_hi=cfg.rcv_palm_delta_hi,
                           oppose_sign=-1.0)
        self.rigs = (self.src, self.rcv)
        w_src, w_rcv = self.src.hand_action_width, self.rcv.hand_action_width
        if 6 + w_src != cfg.num_actions_per_side or 6 + w_rcv != cfg.num_actions_per_side:
            raise RuntimeError(f"손 액션 폭 불일치: cfg {cfg.num_actions_per_side - 6} vs "
                               f"src {w_src} / rcv {w_rcv}")

        # ---- 시작 자세 전체(두 팔 + 두 손 + head) ---------------------------------------------
        self._reset_q = self.robot.data.default_joint_pos[0].clone()
        for rig in self.rigs:
            self._reset_q[rig.arm_t] = rig.reset_q[rig.arm_t]
            self._reset_q[rig.syn_t] = rig.reset_q[rig.syn_t]
        for rig in self.rigs:
            rig.reset_q = self._reset_q.clone()

        # ---- Fabrics ×2 --------------------------------------------------------------------
        initialize_warp(str(dev)[-1])
        self._world = WorldMeshesModel(
            batch_size=N, device=dev, max_objects_per_env=int(cfg.fabrics_max_objects_per_env),
            world_dict=self._build_fabric_world())
        self._world_ids, self._world_indicator = self._world.get_object_ids()
        for rig in self.rigs:
            rig.setup_fabric(_fabric_class(rig.profile.fabric_class), DisplacementIntegrator)

        # ---- 시작 자세 실측(앵커·fabric 게이트) — 리셋 직후 버퍼는 stale, 물리 2스텝 뒤 읽는다 ----
        q0 = self._reset_q.unsqueeze(0).expand(N, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        for rig in self.rigs:
            rig.init_anchor()

        A = cfg.action_space
        self._half = cfg.num_actions_per_side
        self.actions = torch.zeros(N, A, device=dev)
        self.prev_actions = torch.zeros(N, A, device=dev)
        self._policy_dt = float(cfg.sim.dt) * int(cfg.decimation)

        # ---- 비드 판정 상태 ------------------------------------------------------------------
        k = int(cfg.bead_count)
        self._geom = BeadGeometry(
            inner_radius=float(cfg.cup_inner_radius), inside_z_min=float(cfg.cup_inside_z_min),
            inside_z_max=float(cfg.cup_inside_z_max), mouth_z=float(cfg.cup_mouth_z))
        self._prev_tgt_z = torch.full((N, k), -1e6, device=dev)
        self._crossed = torch.zeros(N, k, dtype=torch.bool, device=dev)
        self._prev_in_tgt = torch.zeros(N, device=dev)
        self._prev_in_src = torch.ones(N, device=dev)
        self._prev_spill = torch.zeros(N, device=dev)
        self._flags_fresh = torch.ones(N, dtype=torch.bool, device=dev)
        self._bead_offs = torch.tensor(bead_offsets_in_cup(k), device=dev)

        self._success_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._src_spawn = torch.zeros(N, 3, device=dev)
        self._rcv_spawn = torch.zeros(N, 3, device=dev)
        self._src_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._rcv_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._last_terms: dict = {}
        self._log_tick = 0

        print(f"[pour_fabric] pair={self.pair.name} usd={self.pair.usd_relpath} "
              f"src={self.src.profile.name} rcv={self.rcv.profile.name} "
              f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
              f"beads={k} reward={self._reward_src}", flush=True)

    # ==================================================================
    def _build_fabric_world(self) -> dict | None:
        """fabric 장애물 = 테이블 박스 1개(두 프로필 palm 박스 합집합에서 파생)."""
        cfg = self.cfg
        if not bool(cfg.fabric_table_obstacle):
            print("[pour_fabric] ⚠fabric 테이블 장애물 OFF", flush=True)
            return None
        los = [p.palm_box_min for p in (self.pair.source, self.pair.receiver)]
        his = [p.palm_box_max for p in (self.pair.source, self.pair.receiver)]
        lo = [min(v[i] for v in los) for i in range(2)]
        hi = [max(v[i] for v in his) for i in range(2)]
        m = float(cfg.fabric_table_margin_xy)
        sx, sy = (hi[0] - lo[0]) + 2 * m, (hi[1] - lo[1]) + 2 * m
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        th = float(cfg.fabric_table_thickness)
        cz = float(cfg.table_surface_z) - 0.5 * th
        return {"table": {"env_index": "all", "type": "box", "scaling": f"{sx} {sy} {th}",
                          "transform": f"{cx} {cy} {cz} 0. 0. 0. 1."}}

    # ==================================================================
    def _setup_scene(self) -> None:
        cfg = self.cfg
        self.robot = Articulation(cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 작업면: env_0 에 정적 프림 → clone 이 복제(grasp_s2r 규약). 마찰 재질 바인딩.
        cfg.table_spawn.func("/World/envs/env_0/Table", cfg.table_spawn, translation=(0.0, 0.0, 0.0))
        mu = float(cfg.surface_friction)
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, restitution=0.0)
        mat.func("/World/Materials/taskSurface", mat)
        tables = find_matching_prim_paths(_cfg.TABLE_PRIM)
        if not tables:
            raise RuntimeError(f"[pour_fabric] 테이블 프림이 없다: {_cfg.TABLE_PRIM}")
        for tp in tables:
            bind_physics_material(tp, "/World/Materials/taskSurface")

        # 손가락 마디별 접촉 센서 — body 하나당 하나, **자기 컵만** 필터.
        self._sensor_store: dict = {}
        self._palm_store: dict = {}
        for role, prof, flt in (("src", self.pair.source, list(cfg.source_contact_filter)),
                                ("rcv", self.pair.receiver, list(cfg.receiver_contact_filter))):
            store: dict = {}
            for finger, bodies in prof.finger_sensor_bodies.items():
                ss = []
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
                    ss.append(s)
                    self.scene.sensors[f"contact_{role}_{finger}_{body}"] = s
                store[finger] = ss
            self._sensor_store[role] = store
            ps = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{prof.palm_body}",
                filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
            self.scene.sensors[f"contact_{role}_palm"] = ps
            self._palm_store[role] = ps

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, float(cfg.ground_plane_z)))
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        self.scene.clone_environments(copy_from_source=True)
        self.source_cup = RigidObject(cfg.source_cup_cfg)
        self.scene.rigid_objects["source_cup"] = self.source_cup
        self.receiver_cup = RigidObject(cfg.receiver_cup_cfg)
        self.scene.rigid_objects["receiver_cup"] = self.receiver_cup
        self.beads = RigidObjectCollection(cfg.beads_cfg)
        self.scene.rigid_object_collections["beads"] = self.beads
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ==================================================================
    def _hold_mask(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.hold_steps)

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _cup_up(self, cup: RigidObject) -> torch.Tensor:
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(cup.data.root_quat_w, z)

    def _mouth(self, cup: RigidObject) -> torch.Tensor:
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return self._local(cup.data.root_pos_w + quat_apply(cup.data.root_quat_w, off))

    def _close_gate(self, rig: SideRig, cup: RigidObject, grasped: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not bool(cfg.close_gate_enabled):
            return torch.ones(self.num_envs, device=self.device)
        d = (rig.palm_pos() - self._local(cup.data.root_pos_w)).norm(dim=-1)
        r = float(cfg.close_gate_radius)
        ramp = max(float(cfg.close_gate_ramp) * r, 1e-6)
        g = ((r - d) / ramp).clamp(0.0, 1.0)
        return torch.where(grasped, torch.ones_like(g), g)    # 파지 후 해제(들고 갈 때 잠기지 않게)

    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        active = ~self._hold_mask()
        h = self._half
        for i, (rig, cup, grasped) in enumerate(
                ((self.src, self.source_cup, self._src_grasped),
                 (self.rcv, self.receiver_cup, self._rcv_grasped))):
            a = self.actions[:, i * h:(i + 1) * h]
            rig.compose_palm_target(a[:, :6], active)
            prev = rig.syn_target
            gate = self._close_gate(rig, cup, grasped) * active.float()
            rig.syn_target = rig.synergy_targets(a[:, 6:], gate)
            rig.syn_vel = (rig.syn_target - prev) / self._policy_dt
            rig.sync_fabric_hand()
            q_pin = rig.fabric_q
            rig.step_fabric(self._world_ids, self._world_indicator)
            rig.pin_fabric(~active, q_pin)

    def _apply_action(self) -> None:
        for rig in self.rigs:
            rig.apply_action()
        if self._grav_comp > 0.0:
            tau = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    # ==================================================================
    def _side_obs(self, rig: SideRig, cup: RigidObject) -> list[torch.Tensor]:
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        palm = rig.palm_pos()
        R = rig.palm_R()
        tips = rig.tips_pos()
        cup_p = self._local(cup.data.root_pos_w)
        return [
            q[:, rig.arm_t], qd[:, rig.arm_t], q[:, rig.hand_t], qd[:, rig.hand_t],
            palm, torch.cat([R[:, :, 0], R[:, :, 1]], dim=1),
            (tips - palm.unsqueeze(1)).reshape(self.num_envs, -1),
            cup_p - palm,
            (tips - cup_p.unsqueeze(1)).reshape(self.num_envs, -1),
            rig.joint_err(),
            self._cup_up(cup),
        ]

    def _get_observations(self) -> dict:
        parts = self._side_obs(self.src, self.source_cup) + self._side_obs(self.rcv, self.receiver_cup)
        src_p = self._local(self.source_cup.data.root_pos_w)
        rcv_p = self._local(self.receiver_cup.data.root_pos_w)
        parts += [rcv_p - src_p, self._mouth(self.receiver_cup) - self._mouth(self.source_cup),
                  self.prev_actions]
        obs = torch.cat(parts, dim=1)
        bead_fracs = torch.stack([self._prev_in_src, self._prev_in_tgt, self._prev_spill,
                                  self._crossed.float().mean(dim=-1)], dim=1)
        centroid_rel = self.beads.data.object_pos_w.mean(dim=1) - self.receiver_cup.data.root_pos_w
        centroid_local = quat_apply_inverse(self.receiver_cup.data.root_quat_w, centroid_rel)
        state = torch.cat([
            obs, bead_fracs, centroid_local,
            self.source_cup.data.root_lin_vel_w, self.source_cup.data.root_ang_vel_w,
            self.receiver_cup.data.root_lin_vel_w, self.receiver_cup.data.root_ang_vel_w,
            (self.episode_length_buf.float() / float(self.max_episode_length)).unsqueeze(1),
            self.src.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
            self.rcv.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _build_context(self, flags, d_in_target, d_spill) -> RewardContext:
        cfg = self.cfg
        src_f, rcv_f = self.src.finger_forces(), self.rcv.finger_forces()
        self._src_grasped = self.src.grasped(src_f)
        self._rcv_grasped = self.rcv.grasped(rcv_f)
        scup, rcup = self.source_cup, self.receiver_cup
        s_up, r_up = self._cup_up(scup), self._cup_up(rcup)

        def _tilt(up):
            return torch.acos(up[:, 2].clamp(-1.0, 1.0))
        return RewardContext(
            table_z=float(cfg.table_surface_z), cup_radius=float(cfg.cup_inner_radius),
            cup_mouth_z=float(cfg.cup_mouth_z), cup_bottom_z=-float(cfg.object_origin_offset_z),
            num_beads=int(cfg.bead_count),
            src_palm_pos=self.src.palm_pos(),
            src_palm_axes=torch.cat([self.src.palm_R()[:, :, 0], self.src.palm_R()[:, :, 1]], dim=1),
            src_tips_pos=self.src.tips_pos(), src_hand_closure=self.src.closure(),
            src_finger_force=src_f, src_palm_force=self.src.palm_force(),
            src_grasped=self._src_grasped, src_arm_qd=self.robot.data.joint_vel[:, self.src.arm_t],
            rcv_palm_pos=self.rcv.palm_pos(),
            rcv_palm_axes=torch.cat([self.rcv.palm_R()[:, :, 0], self.rcv.palm_R()[:, :, 1]], dim=1),
            rcv_tips_pos=self.rcv.tips_pos(), rcv_hand_closure=self.rcv.closure(),
            rcv_finger_force=rcv_f, rcv_palm_force=self.rcv.palm_force(),
            rcv_grasped=self._rcv_grasped, rcv_arm_qd=self.robot.data.joint_vel[:, self.rcv.arm_t],
            src_cup_pos=self._local(scup.data.root_pos_w), src_cup_quat=scup.data.root_quat_w,
            src_cup_up=s_up, src_cup_tilt=_tilt(s_up), src_cup_mouth_pos=self._mouth(scup),
            src_cup_lin_vel=scup.data.root_lin_vel_w, src_cup_ang_vel=scup.data.root_ang_vel_w,
            src_cup_spawn_pos=self._src_spawn,
            rcv_cup_pos=self._local(rcup.data.root_pos_w), rcv_cup_quat=rcup.data.root_quat_w,
            rcv_cup_up=r_up, rcv_cup_tilt=_tilt(r_up), rcv_cup_mouth_pos=self._mouth(rcup),
            rcv_cup_lin_vel=rcup.data.root_lin_vel_w, rcv_cup_ang_vel=rcup.data.root_ang_vel_w,
            rcv_cup_spawn_pos=self._rcv_spawn,
            bead_in_source_frac=flags.in_source_frac, bead_in_target_frac=flags.in_target_frac,
            bead_spill_frac=flags.spill_frac, bead_centroid=self._local(flags.centroid_w),
            d_in_target=d_in_target, d_spill=d_spill,
            success=self._success_now,
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self.prev_actions,
        )

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        flags = compute_bead_flags(
            bead_pos_w=self.beads.data.object_pos_w,
            source_pos_w=self.source_cup.data.root_pos_w, source_quat_w=self.source_cup.data.root_quat_w,
            target_pos_w=self.receiver_cup.data.root_pos_w, target_quat_w=self.receiver_cup.data.root_quat_w,
            geom_source=self._geom, geom_target=self._geom,
            prev_target_local_z=self._prev_tgt_z, crossed_mask=self._crossed)
        fresh = self._flags_fresh.float()
        d_in_target = (1.0 - fresh) * (flags.in_target_frac - self._prev_in_tgt)
        d_spill = (1.0 - fresh) * (flags.spill_frac - self._prev_spill)
        self._prev_in_tgt, self._prev_in_src, self._prev_spill = (
            flags.in_target_frac, flags.in_source_frac, flags.spill_frac)
        self._prev_tgt_z, self._crossed = flags.target_local_z, flags.crossed_mask
        self._flags_fresh[:] = False

        xy = (self.source_cup.data.root_pos_w[:, :2] - self.receiver_cup.data.root_pos_w[:, :2]).norm(dim=-1)
        self._success_now = ((flags.in_target_frac >= float(cfg.success_fill_ratio))
                             & (flags.spill_frac <= float(cfg.success_spill_max))
                             & (xy < float(cfg.success_xy_thresh)))
        self._success_streak = torch.where(self._success_now, self._success_streak + 1,
                                           torch.zeros_like(self._success_streak))

        ctx = self._build_context(flags, d_in_target, d_spill)
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        # hold 중엔 보상 0(팔이 고정된 구간의 보상은 정책과 무관하다).
        total = total * (~self._hold_mask()).float()
        self._last_terms = terms

        self.extras["action/step_delta"] = (self.actions - self.prev_actions).abs().mean()
        self.prev_actions.copy_(self.actions)
        self._log(total, terms, flags, ctx)
        return total

    def _log(self, total, terms, flags, ctx: RewardContext) -> None:
        cfg = self.cfg
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/success_now"] = self._success_now.float().mean()
        self.extras["task/episode_success"] = (
            self._success_streak >= int(cfg.success_hold_steps)).float().mean()
        self.extras["task/src_grasped"] = self._src_grasped.float().mean()
        self.extras["task/rcv_grasped"] = self._rcv_grasped.float().mean()
        self.extras["task/src_closure"] = ctx.src_hand_closure.mean()
        self.extras["task/rcv_closure"] = ctx.rcv_hand_closure.mean()
        self.extras["task/src_cup_lift"] = (ctx.src_cup_pos[:, 2] - self._src_spawn[:, 2]).mean()
        self.extras["task/rcv_cup_lift"] = (ctx.rcv_cup_pos[:, 2] - self._rcv_spawn[:, 2]).mean()
        self.extras["task/src_tilt_deg"] = torch.rad2deg(ctx.src_cup_tilt).mean()
        self.extras["task/rcv_tilt_deg"] = torch.rad2deg(ctx.rcv_cup_tilt).mean()
        self.extras["task/aim_dist"] = (ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos).norm(dim=-1).mean()
        self.extras["task/src_palm_to_cup"] = (ctx.src_palm_pos - ctx.src_cup_pos).norm(dim=-1).mean()
        self.extras["task/rcv_palm_to_cup"] = (ctx.rcv_palm_pos - ctx.rcv_cup_pos).norm(dim=-1).mean()
        self.extras["contact/src_max"] = ctx.src_finger_force.max(dim=1).values.mean()
        self.extras["contact/rcv_max"] = ctx.rcv_finger_force.max(dim=1).values.mean()
        self.extras["bead/in_source"] = flags.in_source_frac.mean()
        self.extras["bead/in_target"] = flags.in_target_frac.mean()
        self.extras["bead/spill"] = flags.spill_frac.mean()
        self.extras["bead/crossed"] = flags.crossed_frac.mean()
        self.extras["done/drop"] = self._dropped.float().mean()
        for rig, tag in ((self.src, "src"), (self.rcv, "rcv")):
            perr = (rig.palm_targets[:, :3] + rig.fab_to_env - rig.palm_pos()).norm(dim=-1)
            self.extras[f"fabric/{tag}_palm_err"] = perr.mean()
            # 회전 추종(각 슬롯 wrap 후 최대 절대 오차) — 붓기 tilt 지령이 실제로 도달하는지의 근거
            rerr = rig.palm_targets[:, 3:] - rig.palm_pose_6d()[:, 3:]
            rerr = torch.remainder(rerr + math.pi, 2 * math.pi) - math.pi
            self.extras[f"fabric/{tag}_rot_err_deg"] = torch.rad2deg(rerr.abs().max(dim=1).values).mean()
        self._log_tick += 1
        every = int(cfg.console_log_interval)
        if every > 0 and self._log_tick % every == 0:
            print(f"[METRICS] step={self._log_tick:>8d} rew={total.mean():+.3f} "
                  f"gS={self._src_grasped.float().mean():.2f} gR={self._rcv_grasped.float().mean():.2f} "
                  f"liftS={float(self.extras['task/src_cup_lift']):+.3f} "
                  f"tiltS={float(self.extras['task/src_tilt_deg']):.1f} "
                  f"aim={float(self.extras['task/aim_dist']):.3f} "
                  f"inT={flags.in_target_frac.mean():.3f} inS={flags.in_source_frac.mean():.3f} "
                  f"spill={flags.spill_frac.mean():.3f} succ={self._success_now.float().mean():.3f}",
                  flush=True)

    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        floor = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z) - float(cfg.drop_below_table_m)
        s_z = self._local(self.source_cup.data.root_pos_w)[:, 2]
        r_z = self._local(self.receiver_cup.data.root_pos_w)[:, 2]
        self._dropped = ((s_z < floor) | (r_z < floor)) & (~self._hold_mask())
        qd = torch.cat([self.robot.data.joint_vel[:, self.src.arm_t],
                        self.robot.data.joint_vel[:, self.rcv.arm_t]], dim=1)
        runaway = (qd.abs() > float(cfg.runaway_joint_vel)).any(dim=-1)
        terminated = runaway | self._dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        return terminated, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n, dev, cfg = len(env_ids), self.device, self.cfg

        q0 = self._reset_q.unsqueeze(0).expand(n, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        for rig in self.rigs:
            rig.reset(env_ids)

        rest_z = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z)
        rng = float(cfg.object_spawn_range)
        for rig, cup, store in ((self.src, self.source_cup, self._src_spawn),
                                (self.rcv, self.receiver_cup, self._rcv_spawn)):
            offs = (torch.rand(n, 2, device=dev) - 0.5) * 2.0 * rng
            pos = torch.zeros(n, 3, device=dev)
            pos[:, 0] = rig.profile.object_spawn_center[0] + offs[:, 0]
            pos[:, 1] = rig.profile.object_spawn_center[1] + offs[:, 1]
            pos[:, 2] = rest_z
            store[env_ids] = pos
            root = torch.zeros(n, 13, device=dev)
            root[:, :3] = pos + self.scene.env_origins[env_ids]
            root[:, 2] += float(cfg.object_spawn_pad)
            root[:, 3] = 1.0
            cup.write_root_state_to_sim(root, env_ids=env_ids)

        k = int(cfg.bead_count)
        bead = torch.zeros(n, k, 13, device=dev)
        bead[:, :, :3] = (self._src_spawn[env_ids].unsqueeze(1) + self._bead_offs.unsqueeze(0)
                          + self.scene.env_origins[env_ids].unsqueeze(1))
        bead[:, :, 2] += float(cfg.object_spawn_pad)
        bead[:, :, 3] = 1.0
        self.beads.write_object_state_to_sim(bead, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._success_now[env_ids] = False
        self._success_streak[env_ids] = 0
        self._dropped[env_ids] = False
        self._src_grasped[env_ids] = False
        self._rcv_grasped[env_ids] = False
        self._crossed[env_ids] = False
        self._prev_tgt_z[env_ids] = -1e6
        self._prev_in_tgt[env_ids] = 0.0
        self._prev_in_src[env_ids] = 1.0
        self._prev_spill[env_ids] = 0.0
        self._flags_fresh[env_ids] = True
