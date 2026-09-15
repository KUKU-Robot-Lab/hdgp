"""grasp_fj_t2r — Track B(`grasp_fj`) 위에 text2reward 생성 보상을 얹는다.

덮는 것은 넷뿐이다.
  1. `__init__`         — 생성 보상 코드를 **부팅 전에** 읽는다(경로가 틀리면 씬을 만들기 전에 죽는다).
  2. `_setup_scene`     — **보상 전용** 접촉 센서: 손가락 마디(중간·원위·손끝) body 하나당 센서 하나 +
                          손바닥, 전부 컵에만 필터. 관측에는 넣지 않는다(09.14 사용자 결정).
  3. `_progress_reward` — B 모듈을 그대로 돌려 에피소드 상태(lifted 래치·추적기)를 갱신한 뒤,
                          총보상과 항은 `compute_reward(ctx)` 결과로 바꿔 돌려준다.
  4. `_log_fabric_metrics`·`_reset_idx` — 접촉 진단 · 에피소드 단계 퍼널(접근→파지→인벨롭→리프트→성공, 루프 틱 재료)
                          로그 · 직전 액션 버퍼 리셋.

★관측·액션·종료·성공 판정·공차 커리큘럼은 B 그대로다 — 보상만 바뀐 같은 과제다.
★09.15 사용자 3단계("기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트"): 보상 게이트용 에피소드 래치
  (`grasp_gates.py`: 접근 완료·인벨롭 완료)를 `_build_context` 가 매 스텝 갱신해 ctx 로 넘긴다. 보상 입력일 뿐 관측이 아니다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from ..grasp_fj.grasp_fj_env import GraspFJEnv
from ..grasp_fj.robot_profiles import PROFILES
from .grasp_fj_t2r_env_cfg import GraspFJT2RRightShortEnvCfg
from .grasp_gates import TOUCH_N as _GATE_TOUCH_N
from .grasp_gates import (APPROACH_CONDITIONS, approach_conditions, c_pregrasp_geometry, hand_orientation,
                          pose_deviation, update_gates)
from .palm_frame import palm_center_offset
from .stage_funnel import STAGES, palm_band_gap, step_flags
from .t2r.context import RewardContext
from .t2r.loader import call_reward_fn, load_reward_fn
from .t2r.prompts import LOCKED_SPAN_RAD

#: 진단 로그에서 "닿았다"로 셀 컵 접촉력 [N]. 보상에는 쓰지 않는다(보상의 임계는 생성 코드 몫).
_TOUCH_LOG_N = 0.1
#: 보상 게이트 래치 이름 — 로그 `stage/<이름>_gate_ep`, 순서 = `_t2r_gate_ema` 열.
GATE_NAMES = ("approach", "envelope")
#: 출발 그룹 — 로그 `stage/<그룹>_<래치|단계>_ep`, 순서 = `_t2r_gate_ema_grp`·`_t2r_stage_ema_grp` 행(09.15 시작 상태 커리큘럼).
START_GROUPS = ("far", "near")


class GraspFJT2REnv(GraspFJEnv):
    cfg: GraspFJT2RRightShortEnvCfg

    def __init__(self, cfg: GraspFJT2RRightShortEnvCfg, render_mode: str | None = None, **kw):
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kw)
        self._t2r_prev_actions = torch.zeros(self.num_envs, int(self.cfg.action_space), device=self.device)
        self._t2r_ctx = None
        # 성공 순간 접촉 이벤트 EMA — [손가락 수 · 마디 수 · 손바닥] 과 손가락별. 음수 = 아직 성공 없음(센티널).
        self._t2r_succ_ema = torch.full((3,), -1.0, device=self.device)
        self._t2r_finger_succ_ema = torch.full((len(self._finger_names),), -1.0, device=self.device)
        # 에피소드 단계 퍼널 — 에피소드 동안 한 번이라도 닿았나(래치) → 에피소드가 끝날 때 이벤트 EMA. 음수 = 끝난 에피소드 없음.
        self._t2r_stage_latch = torch.zeros(self.num_envs, len(STAGES), dtype=torch.bool, device=self.device)
        self._t2r_stage_ema = torch.full((len(STAGES),), -1.0, device=self.device)
        # ★09.15 보상 게이트 래치(`grasp_gates.py`) — ctx.approach_done·envelope_done. 기본 손 자세 = B 리셋이 심는 자세.
        self._t2r_gate_approach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._t2r_gate_env_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._t2r_gate_envelope = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._t2r_gate_ema = torch.full((len(GATE_NAMES),), -1.0, device=self.device)
        self._t2r_default_q_norm = ((self._hand_reset_q - self._act_lo) / self._act_span).clamp(0.0, 1.0)
        self._t2r_movable = (self._act_hi - self._act_lo) > LOCKED_SPAN_RAD
        self._t2r_approach_ok = torch.zeros(len(APPROACH_CONDITIONS), device=self.device)
        # ★09.15 사용자 "시작 상태 커리큘럼 + env 고정" — 이번 에피소드의 출발 그룹(가까운 출발 = True)과 그룹별 래치·퍼널 EMA,
        #   컵 종류별 가까운 출발 팔 자세(reach leaf cfg 의 IK 표, 행 = 뱅크 순서 = env_id % N).
        self._t2r_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._t2r_gate_ema_grp = torch.full((len(START_GROUPS), len(GATE_NAMES)), -1.0, device=self.device)
        self._t2r_stage_ema_grp = torch.full((len(START_GROUPS), len(STAGES)), -1.0, device=self.device)
        self._t2r_near_q = None
        if float(self.cfg.near_start_frac) > 0.0:
            if tuple(self.cfg.near_start_species) != tuple(self._species_names):
                raise RuntimeError(f"[grasp_fj_t2r] 가까운 출발 IK 표의 컵 순서 {tuple(self.cfg.near_start_species)} ≠ 뱅크 "
                                   f"{tuple(self._species_names)} — env_id % N 배정과 어긋난다")
            _rows = tuple(self.cfg.near_start_arm_q)
            if len(_rows) != len(self._species_names) or any(len(r) != len(self._arm_ids_t) for r in _rows):
                raise RuntimeError(f"[grasp_fj_t2r] near_start_arm_q 는 컵 {len(self._species_names)}종 × 팔 "
                                   f"{len(self._arm_ids_t)}관절이어야 한다: {[len(r) for r in _rows]}")
            self._t2r_near_q = torch.tensor(_rows, device=self.device, dtype=torch.float32)
        # ★09.14 사용자 "손바닥의 중심쪽은 palm_ee xform" — palm_idx 는 프로필 palm_body(손바닥 링크 원점, 손목 쪽)다.
        #   ctx.palm_pos 는 자산 URDF 의 palm_ee fixed 조인트 오프셋만큼 옮긴 손바닥 중심을 넘긴다(회전은 같아 법선은 그대로).
        _prof = PROFILES[self.cfg.profile_name]
        _off = palm_center_offset(Path(self.cfg.robot_cfg.spawn.usd_path).with_suffix(".urdf"), _prof.palm_body)
        self._t2r_palm_center_off = torch.tensor(_off, device=self.device)
        _n = sum(len(v) for v in self._t2r_link_sensors.values())
        print(f"[grasp_fj_t2r] 보상 = {self._reward_src} · 보상 전용 컵 접촉 센서 {_n}+1(손바닥) · "
              f"필터 {self._t2r_filter} · 손바닥 중심 = {_prof.palm_body} + {tuple(round(v, 4) for v in _off)} m(palm_ee) · "
              f"게이트 래치 {GATE_NAMES} · 기본 손 자세(움직이는 관절 {int(self._t2r_movable.sum())}개) · 관측 불변", flush=True)
        print(f"[grasp_fj_t2r] 접근 래치 = C자 사전파지 {APPROACH_CONDITIONS} · 가까운 출발 "
              + (f"{float(self.cfg.near_start_frac):.0%} (공통 스텝 > {int(self.cfg.near_start_after_common_steps)}, "
                 f"컵 {len(self._species_names)}종 IK)" if self._t2r_near_q is not None else "끔"), flush=True)

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """B 씬 + **보상 전용** 컵 접촉 센서. body 하나당 센서 하나(묶으면 force_matrix_w 가 조용히 0)."""
        super()._setup_scene()
        prof = PROFILES[self.cfg.profile_name]
        _filter = self._cup_contact_filter()
        self._t2r_filter = _filter
        self._t2r_link_sensors: dict[str, list[ContactSensor]] = {}
        for finger, bodies in prof.finger_sensor_bodies.items():
            sensors = []
            for body in bodies:
                s = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body}",
                    filter_prim_paths_expr=_filter,
                    history_length=1, track_air_time=False))
                self.scene.sensors[f"t2r_contact_{body}"] = s
                sensors.append(s)
            self._t2r_link_sensors[finger] = sensors
        self._t2r_palm_sensor = ContactSensor(ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{prof.palm_body}",
            filter_prim_paths_expr=_filter,
            history_length=1, track_air_time=False))
        self.scene.sensors["t2r_contact_palm"] = self._t2r_palm_sensor

    def _cup_contact_filter(self) -> list[str]:
        """컵 **강체 프림의 실제 경로**로 접촉 필터를 만든다 — cfg `object_contact_filter` 는 쓰지 않는다.

        ★09.14 스모크에서 드러났다: cfg 값 `Object/{bank.rigid_body_name}` 이 shaker_sweep 에서
          `/World/envs/env_*/Object/ShakerFDM5mm` 로 풀려 **0개 매칭**이었다. USD 루트 Xform
          (`ShakerFDM5mm`, RigidBodyAPI)이 참조되면서 `Object` 프림 **자체**가 되기 때문이다.
          PhysX 는 에러 로그만 남기고 `force_matrix_w` 는 조용히 0 이었다(컵이 25° 기울었는데 힘 0).
          RigidObject 초기화와 같은 방식으로 env_0 스테이지에서 RigidBodyAPI 프림을 찾는다.
        """
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics

        pattern = self.cfg.object_cfg.prim_path
        env0 = pattern.replace("env_.*", "env_0", 1)
        prims = sim_utils.get_all_matching_child_prims(
            env0, predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
        if len(prims) != 1:
            raise RuntimeError(
                f"[grasp_fj_t2r] {env0} 아래 컵 강체 프림이 {len(prims)}개다(1개여야 필터를 만든다): "
                f"{[p.GetPath().pathString for p in prims]}")
        path = prims[0].GetPath().pathString
        if not path.startswith(env0):
            raise RuntimeError(f"[grasp_fj_t2r] 컵 강체 프림 {path} 가 {env0} 밖이다")
        return [pattern + path[len(env0):]]

    def _link_cup_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(마디 (N,F,L), 손바닥 (N,)) 컵 필터 접촉력 크기 [N]. 손가락 순서 = `_finger_names`."""
        def _mag(sensor: ContactSensor) -> torch.Tensor:
            return sensor.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

        links = torch.stack([torch.stack([_mag(s) for s in self._t2r_link_sensors[f]], dim=1)
                             for f in self._finger_names], dim=1)
        return links, _mag(self._t2r_palm_sensor)

    # ------------------------------------------------------------------
    def _progress_reward(self, **kw):
        """B 모듈로 에피소드 상태를 갱신하고, 총보상·항은 생성 코드 결과로 바꾼다."""
        _fj_total, _fj_terms, out = super()._progress_reward(**kw)
        ctx = self._build_context(kw, out)
        total, terms = call_reward_fn(self._reward_fn, ctx)
        # ★09.14 reach 스모크(random 64env): 팔 runaway(>20 rad/s) env 에서 생성 코드의 속도 제곱 항이 터져
        #   `reward/joint_vel_penalty` 평균이 −67,185 였다. 물리가 깨진 스텝(`_get_dones` 가 이번 스텝에 `_abnormal` 로 표시)의
        #   상태는 보상 근거가 못 된다 — 그 env 는 생성 보상 0 + B 의 abnormal_penalty 만 받고, 로그 항도 같이 가린다.
        ok = ~self._abnormal
        total = torch.where(ok, torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(total))
        terms = {k: torch.where(ok, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(v))
                 for k, v in terms.items()}
        self._t2r_ctx = ctx
        self._t2r_prev_actions = ctx.actions.clone()
        return total, terms, out

    def _build_context(self, kw: dict, out: dict) -> RewardContext:
        n = self.num_envs
        links_f, palm_f = self._link_cup_forces()
        if links_f.shape[1:] != (5, 3):
            raise RuntimeError(f"[grasp_fj_t2r] 마디 접촉력 {tuple(links_f.shape)} — 프롬프트 계약은 (N,5,3)")
        R = self._palm_ee_R()
        # 손바닥 중심(palm_ee) = 손바닥 링크 원점 + R·오프셋 — `__init__` 주석 참조
        palm_link = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        palm_center = palm_link + torch.einsum("nij,j->ni", R, self._t2r_palm_center_off)
        link_pos = (self.robot.data.body_pos_w[:, self._hull_all_t]
                    - self.scene.env_origins[:, None, :]).view(n, len(self._finger_names), -1, 3)
        q = self.robot.data.joint_pos[:, self._syn_ids]
        lo, span = self._act_lo.unsqueeze(0), self._act_span.unsqueeze(0)
        q_norm = ((q - lo) / span).clamp(0.0, 1.0)
        cup_quat = self.object.data.root_quat_w
        cup_local = self._env_local(self.object.data.root_pos_w)
        axis = quat_apply(cup_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3))
        raw = getattr(self, "_act_raw", self.actions).clamp(-1.0, 1.0)
        # ★09.15 보상 게이트 래치 — 이 스텝 상태로 접근 완료·인벨롭 완료를 갱신한다(한 번 서면 리셋까지 유지).
        #   ★09.15 23:2x 사용자 "C자 사전파지로 재정의" — 기본 자세 엄지가 손바닥면 118 mm 앞으로 뻗어, 접근 완료는 컵이 엄지와 네
        #   손가락 사이에 든 자리다: 손바닥면↔옆면 · 컵 축의 손가락 방향 오프셋 · 띠 높이 · 시작 방향(palm_ee 프레임 x 법선·z 손가락) ·
        #   기본 손 자세 · 손가락·엄지 무접촉. 인벨롭 = 접근 뒤 손바닥+엄지+손가락 수 연속. 수치는 grasp_gates.
        finger_touch = (links_f > _GATE_TOUCH_N).any(dim=2)
        plane_gap, along_offset, height = c_pregrasp_geometry(palm_center, R[:, :, 0], R[:, :, 2], cup_local, axis, self._obj_grasp_r)
        cond = approach_conditions(
            plane_gap=plane_gap, along_offset=along_offset, height=height, half_height=self._obj_grasp_h,
            orient=hand_orientation(R[:, :, 0], R[:, :, 2]),
            pose_dev=pose_deviation(q_norm, self._t2r_default_q_norm, self._t2r_movable),
            digit_touch=finger_touch.any(dim=1))
        self._t2r_approach_ok = cond.float().mean(dim=0)
        self._t2r_gate_approach, self._t2r_gate_env_count, self._t2r_gate_envelope = update_gates(
            self._t2r_gate_approach, self._t2r_gate_env_count, self._t2r_gate_envelope, approach_ok=cond.all(dim=1),
            palm_touch=palm_f > _GATE_TOUCH_N, finger_touch=finger_touch)
        vals = dict(
            table_z=float(self.cfg.table_surface_z),
            lift_latch_height=float(self._rw_cfg.lift_latch_height),
            success_hold_steps=int(self.cfg.goal_success_steps),
            max_successes=int(self.cfg.goal_max),
            palm_pos=palm_center,
            palm_normal=R[:, :, 0], palm_side=R[:, :, 1], palm_finger_dir=R[:, :, 2],
            link_pos=link_pos, link_cup_force=links_f, palm_cup_force=palm_f,
            hand_q=q,
            hand_q_norm=q_norm,
            hand_target_norm=((self._syn_target - lo) / span).clamp(0.0, 1.0),
            hand_default_q_norm=self._t2r_default_q_norm.unsqueeze(0).expand(n, -1),
            hand_qd=self.robot.data.joint_vel[:, self._syn_ids],
            hand_z_min=self._hand_z_min,
            arm_q=self.robot.data.joint_pos[:, self._arm_ids_t],
            arm_qd=self.robot.data.joint_vel[:, self._arm_ids_t],
            cup_pos=cup_local, cup_quat=cup_quat, cup_axis=axis,
            cup_tilt=torch.acos(axis[:, 2].clamp(-1.0, 1.0)),
            cup_lin_vel=self.object.data.root_lin_vel_w, cup_ang_vel=self.object.data.root_ang_vel_w,
            cup_spawn_pos=self.object_spawn_pos,
            cup_radius=self._obj_grasp_r, cup_half_height=self._obj_grasp_h,
            goal_pos=self.goal_pos, goal_dist=kw["kp_dist"],
            success_tol=torch.full((n,), float(self._tol.tol), device=self.device),
            lifted=out["lifted"], success=kw["is_success"],
            num_successes=self._trk.successes.float(),
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            approach_done=self._t2r_gate_approach, envelope_done=self._t2r_gate_envelope,
            actions=raw, prev_actions=self._t2r_prev_actions,
        )
        # ★09.14 리뷰: goal_pos·lifted(→_latched)·is_success(→_advance_goals)·kp_dist(→_log_step)·_obj_grasp_r/h·
        #   root_quat_w 등은 env 가 이 보상 **뒤에** 다시 읽는 살아 있는 버퍼다. 생성 코드의 제자리 연산이 학습을
        #   조용히 오염시키지 않게 ctx 에는 복사본만 넣는다(검증기도 ctx 제자리 연산을 막지만 지역변수 별칭은 못 잡는다).
        return RewardContext(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in vals.items()})

    # ------------------------------------------------------------------
    def _log_fabric_metrics(self) -> None:
        """B 진단 + 컵 접촉 진단(보상 전용 센서) + 에피소드 단계 퍼널 — 생성 보상의 피드백 재료. host 동기화 0."""
        super()._log_fabric_metrics()
        ctx = self._t2r_ctx
        if ctx is None:
            return
        ex = self.extras
        touching = (ctx.link_cup_force > _TOUCH_LOG_N).to(ctx.link_cup_force.dtype)    # (N,F,L)
        ex["contact/links_touching"] = touching.sum(dim=(1, 2)).mean()
        ex["contact/fingers_touching"] = touching.amax(dim=2).sum(dim=1).mean()
        ex["contact/palm_touching"] = (ctx.palm_cup_force > _TOUCH_LOG_N).to(touching.dtype).mean()
        ex["contact/link_force_mean"] = ctx.link_cup_force.mean()
        for k, finger in enumerate(self._finger_names):
            ex[f"contact/finger_{finger}"] = touching[:, k].amax(dim=1).mean()
        # ★성공 **순간**의 접촉(09.14 t2r 루프) — "성공이 인벨롭이었나"는 스텝 평균으로 못 가른다(접근 중 env 가 뭉갠다).
        #   루프 판정(`t2r_fj_round.py`)과 생성기 피드백 표가 읽는다. B `_log_grasp_quality` 와 같은 이벤트 EMA.
        succ = self._success_now.to(touching.dtype)
        per_env = torch.stack([touching.amax(dim=2).sum(dim=1), touching.sum(dim=(1, 2)),
                               (ctx.palm_cup_force > _TOUCH_LOG_N).to(touching.dtype)], dim=1)   # (N,3)
        self._t2r_succ_ema = self._event_ema(self._t2r_succ_ema, per_env, succ)
        self._t2r_finger_succ_ema = self._event_ema(self._t2r_finger_succ_ema, touching.amax(dim=2), succ)
        for k, name in enumerate(("fingers_touching", "links_touching", "palm_touching")):
            ex[f"contact/{name}_at_success"] = self._t2r_succ_ema[k]
        for k, finger in enumerate(self._finger_names):
            ex[f"contact/finger_{finger}_at_success"] = self._t2r_finger_succ_ema[k]
        # ★에피소드 단계 퍼널(09.14 사용자 "컵에 접근, 파지, 리프트가 잘 되는지 틱을 확인") — 스텝 평균은 접근 중 스텝이 섞인다.
        #   이 스텝 ctx 로 래치를 켠다(`_get_rewards`→`_log_step` 경로라 리셋 전). 에피소드가 끝날 때 `_reset_idx` 가 EMA 로 민다.
        gap = palm_band_gap(ctx.palm_pos, ctx.cup_pos, ctx.cup_axis, ctx.cup_radius, ctx.cup_half_height)
        self._t2r_stage_latch |= step_flags(gap, ctx.link_cup_force > _TOUCH_LOG_N, ctx.palm_cup_force > _TOUCH_LOG_N,
                                            ctx.lifted, ctx.num_successes)
        ex["stage/palm_cup_gap"] = gap.mean()
        for k, name in enumerate(STAGES):
            ex[f"stage/{name}_ep"] = self._t2r_stage_ema[k]
        # ★09.15 보상 게이트 래치 — 끝난 에피소드 중 그 래치가 선 비율(루프 체크포인트 판정 재료) + 지금 서 있는 env 비율.
        for k, name in enumerate(GATE_NAMES):
            ex[f"stage/{name}_gate_ep"] = self._t2r_gate_ema[k]
        ex["stage/approach_gate_now"] = ctx.approach_done.to(touching.dtype).mean()
        ex["stage/envelope_gate_now"] = ctx.envelope_done.to(touching.dtype).mean()
        # 접근 래치가 안 설 때 네 조건 중 무엇이 막는지 — 이 스텝에 조건별로 통과한 env 비율(09.15 틱: 퍼널 접근 0.56 · 래치 0).
        for k, name in enumerate(APPROACH_CONDITIONS):
            ex[f"stage/approach_ok_{name}_now"] = self._t2r_approach_ok[k]
        # ★09.15 시작 상태 커리큘럼 — 끝난 에피소드의 출발 그룹별 래치·퍼널 비율(가까운 출발이 먼 출발 접근률을 부풀리지 않게).
        for g, grp in enumerate(START_GROUPS):
            for k, name in enumerate(GATE_NAMES):
                ex[f"stage/{grp}_{name}_gate_ep"] = self._t2r_gate_ema_grp[g, k]
            for k, name in enumerate(STAGES):
                ex[f"stage/{grp}_{name}_ep"] = self._t2r_stage_ema_grp[g, k]
        ex["stage/near_start_frac_now"] = self._t2r_near.to(touching.dtype).mean()

    def _reset_idx(self, env_ids) -> None:
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        latch = getattr(self, "_t2r_stage_latch", None)
        if latch is not None:
            # 에피소드 결과는 부모 리셋이 길이·추적기를 지우기 **전에** 읽는다. 길이 0 = 첫 reset() — 에피소드 끝이 아니다.
            ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            ended[ids] = self.episode_length_buf[ids] > 0
            self._t2r_stage_ema = self._event_ema(self._t2r_stage_ema, latch.float(), ended)
            gates = torch.stack([self._t2r_gate_approach, self._t2r_gate_envelope], dim=1).float()
            self._t2r_gate_ema = self._event_ema(self._t2r_gate_ema, gates, ended)
            # ★09.15 시작 상태 커리큘럼 — 끝난 에피소드를 그 출발 그룹(먼 · 가까운)으로 따로 민다(그룹은 아래에서 다시 뽑는다).
            groups = (~self._t2r_near, self._t2r_near)
            self._t2r_gate_ema_grp = torch.stack([self._event_ema(self._t2r_gate_ema_grp[g], gates, ended & m)
                                                  for g, m in enumerate(groups)])
            self._t2r_stage_ema_grp = torch.stack([self._event_ema(self._t2r_stage_ema_grp[g], latch.float(), ended & m)
                                                   for g, m in enumerate(groups)])
            latch[ids] = False
            self._t2r_gate_approach[ids] = False
            self._t2r_gate_env_count[ids] = 0
            self._t2r_gate_envelope[ids] = False
        super()._reset_idx(env_ids)
        prev = getattr(self, "_t2r_prev_actions", None)
        if prev is not None:
            prev[ids] = 0.0
        near_q = getattr(self, "_t2r_near_q", None)
        if near_q is not None:
            # ★09.15 사용자 "시작 상태 커리큘럼 + env 고정" — B 리셋(홈 → 고정 시작 자세 → 손 기본 자세)이 끝난 **뒤**, 두 번째 에피소드부터
            #   (B 부팅 시작 거리 가드는 common_step_counter ≤ 4 에서 먼 출발 평균만 본다) `near_start_frac` 을 컵 종류별 IK 자세
            #   (C자 완료 조금 앞 · 시작 방향 · 기본 손 자세)에서 시작한다. 팔 관절 상태와 q*·이전 q* 를 함께 덮어 실측 q = 지령 q* 다
            #   (B 의 고정 시작 자세와 같은 규약). 컵 스폰 ±2 cm 는 그대로 — 자세 여유(손바닥면 3 cm · 컵 축 R+2.5 cm)가 겹침을 막는다.
            idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            self._t2r_near[idx] = False
            if self.common_step_counter > int(self.cfg.near_start_after_common_steps):
                pick = idx[torch.rand(len(idx), device=self.device) < float(self.cfg.near_start_frac)]
                if len(pick):
                    q = self.robot.data.joint_pos[pick].clone()
                    q[:, self._arm_ids_t] = torch.clamp(near_q[self._species_ids[pick]],
                                                        self._arm_lo[pick], self._arm_hi[pick])
                    self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=pick)
                    self._arm_q_target[pick] = q[:, self._arm_ids_t]
                    self._prev_arm_q_target[pick] = self._arm_q_target[pick]
                    self._t2r_near[pick] = True
                    # ★09.16 사용자 "가까운 출발 = 접근 완료로 시작" — 컵 옆 기본 자세 출발은 접근이 끝난 것으로 보고 2단계(자리 맞추기·
                    #   닫기·접촉)부터 시작한다. 4.5 cm 판을 1단계로 두자 컵이 손 앞인데도 닫기 보상이 켜지지 않았다(i02 e240 래치 0).
                    self._t2r_gate_approach[pick] = True
