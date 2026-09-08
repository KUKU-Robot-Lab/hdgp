"""grasp_kp — SimToolReal 식 목표열·progress 보상, Track A (fabric palm 6D + 시너지 15D).

`GraspS2REnv` 를 상속해 DESIGN.md §8 의 훅만 덮어쓴다. **접촉 센서를 만들지 않는다** —
보상·성공·관측·종료 어디에도 접촉이 없다(사용자 확정 09.06). 부모의 접촉 헬퍼
(`_tip_force_local`·`_contact_forces*`·`_log_diagnostics`·`_palmar_mask`)는 부르지 않는다.

DirectRLEnv 스텝 순서: `_pre_physics_step` → 물리 → `_get_dones` → `_get_rewards` →
`_reset_idx` → `_get_observations`. 성공·목표 전진은 `_get_rewards` 에서 일어나고,
`_get_dones` 는 **직전 스텝**의 successes 로 max_goals truncation 을 판정한다.

★`self._latched` 는 부모의 접촉 래치가 아니라 **높이 래치(lifted)** 다 — 닫기 게이트·앵커·
  종별 로깅이 그대로 쓴다. `_hold_count/_wrap_at_latch/_disp_at_latch` 는 쓰지 않는다.
★모듈(`modules/keypoint_goal·progress_reward·perception_delay·object_wrench`)은 무상태다 —
  래치·최소거리·목표·큐의 되먹임은 전부 이 env 의 버퍼다.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material

from ...modules.keypoint_goal import (
    GoalTrackers,
    ToleranceCurriculum,
    keypoint_max_dist,
    keypoint_offsets,
    keypoints_world,
    sample_delta_goal,
    sample_first_goal,
    update_near_goal,
)
from ...modules.object_wrench import WrenchDR
from ...modules.perception_delay import DelayQueue, noisy_pose
from ...modules.progress_reward import compute_progress_reward
from ..grasp_s2r.grasp_s2r_env import GraspS2REnv
from .grasp_kp_env_cfg import GraspKPEnvCfg

_OBJ_POSE_DIM = 7   # pos(3) + quat wxyz(4) — 물체 지연 큐의 폭
def _clamp_norm(v: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """행별 노름을 `cap` 이하로 — 방향은 보존한다(축별 클램프는 방향을 왜곡한다)."""
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return v * (cap / n).clamp(max=1.0)


# 목표 박스 코너까지 걸어가는 데 쓸 수 있는 에피소드 비율(A-i 증분 매핑 도달성 가드).
_TRAVERSE_BUDGET_FRAC = 0.25
# 대각 걸음의 노름이 리미터에 **정확히** 닿으면 float32 오차로 리미터가 nm 단위로 걸려
# `palm_cmd_rate_sat` 진단이 오염된다(스모크 실측 0.0625 = 16 env 중 1). 0.1% 여유를 둬
# 어떤 액션에서도 구조적으로 0 이 되게 한다 — 이 지표는 A-i 의 성공 판정선이다.
_STEP_GAIN_MARGIN = 0.999
#: 진단용 낙하 판정 — 래치는 섰는데 정착고 대비 이만큼도 안 뜬 상태(=놓쳤다).
#: 보상·종료에는 쓰지 않는다(관측 전용).
_DROP_DZ = 0.03


class GraspKPEnv(GraspS2REnv):
    cfg: GraspKPEnvCfg

    # ------------------------------------------------------------------
    # 씬 — mixin `_setup_scene` 과 동일하되 **ContactSensor 생성부만 없다**
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        from openarm.agnostic.modules import object_bank as _ob

        _bank = _ob.get(self.cfg.object_bank)
        _multi = _bank.needs_multi_asset
        self._spawn_table(_multi)
        _sensors = self._spawn_debug_camera()
        # env.usd 의 platform 상면이 정확히 z=0 이라 기본 지면과 겹친다 — 지면은 내린다.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, -0.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ★clone → 씬 등록 → 물체 순서는 부모(DEXTRAH 규약) 그대로 — 바꾸면 replicate_physics=False
        #   에서 리셋이 안 먹는 폭주(관절 편차 18 rad)가 재발한다.
        _replicate = bool(self.cfg.scene.replicate_physics)
        if not _multi:
            _ob.assert_spawned_after_clone(_bank, cloned=not _replicate)
            self.object = RigidObject(self.cfg.object_cfg)
        if _replicate:
            self.scene.clone_environments(copy_from_source=True)
        if _multi:
            _ob.assert_spawned_after_clone(_bank, cloned=True)
            self.object = RigidObject(self.cfg.object_cfg)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        for _k, _v in _sensors.items():
            self.scene.sensors[_k] = _v
        if _multi:
            self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        print(f"[grasp_kp] 물체 뱅크 '{_bank.name}' {len(_bank)}종 · "
              f"replicate_physics={self.cfg.scene.replicate_physics} · 접촉 센서 0개(설계)",
              flush=True)

    def _spawn_table(self, multi: bool) -> None:
        """작업면 + 마찰 재질 — 부모와 동일(단일: 정적 프림, 다물체: kinematic 씬 자산)."""
        tbl = self.cfg.table_cfg
        if multi:
            self.table = RigidObject(tbl)
        else:
            tbl.spawn.func("/World/envs/env_0/Table", tbl.spawn,
                           translation=tuple(tbl.init_state.pos),
                           orientation=tuple(tbl.init_state.rot))
        _mu = float(self.cfg.surface_friction)
        _mat = sim_utils.RigidBodyMaterialCfg(static_friction=_mu, dynamic_friction=_mu, restitution=0.0)
        _mat.func("/World/Materials/taskSurface", _mat)
        from isaaclab.sim.utils import find_matching_prim_paths

        _tables = find_matching_prim_paths(tbl.prim_path)     # regex 가 아니라 실제 프림만 바인딩된다
        if not _tables:
            raise RuntimeError(f"[grasp_kp] 테이블 프림이 없다: {tbl.prim_path}")
        for _tp in _tables:
            bind_physics_material(_tp, "/World/Materials/taskSurface")

    def _spawn_debug_camera(self) -> dict:
        """진단 카메라(선택) — 접촉 센서가 아니므로 유지."""
        if not bool(self.cfg.debug_camera):
            return {}
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        return {"debug_cam": TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/DebugCam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(self.cfg.debug_camera_pos), rot=tuple(self.cfg.debug_camera_rot),
                convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=26.0, focus_distance=0.6, horizontal_aperture=20.955,
                clipping_range=(0.05, 6.0)),
            width=640, height=480,
        ))}

    # ------------------------------------------------------------------
    # 부트스트랩
    # ------------------------------------------------------------------
    def _init_task_state(self) -> None:
        super()._init_task_state()             # 인덱스·fabric·시너지·홈·앵커·박스·ADR(OFF)
        self._assert_kp_contract()
        # 손 액션 슬라이스 시작 = 팔 액션 폭(A palm 6 / B n_arm) — cfg 훅 하나에서 온다.
        self._hand_action_offset = int(self.cfg._arm_action_dim(self.profile))
        self._apply_palm_floor_override()
        n, dev, c = self.num_envs, self.device, self.cfg
        self._setup_palm_step_gain()
        self._kp_offsets = keypoint_offsets(c.keypoint_half_height(), dev)     # (4,3)
        self._goal_cfg = c.goal_seq_cfg()
        self._rw_cfg = c.progress_reward_cfg()
        # 목표 자세 + 정착 자세(직립 단위 쿼터니언, 에피소드 상수). goal_pos 는 부모 버퍼 재사용.
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.goal_quat[:, 0] = 1.0
        self._settled_quat = self.goal_quat.clone()
        self._trk = GoalTrackers(n, len(self.tip_ids), dev)
        _tol_kw = c.tolerance_curriculum_kwargs()
        if float(c.tol_eval) > 0.0:
            # 왜: 평가는 고정 tol — start=floor 면 `update` 가 항상 False 라 스텝 경로에 분기가 없다.
            _tol_kw.update(start=float(c.tol_eval), floor=float(c.tol_eval))
        self._tol = ToleranceCurriculum(**_tol_kw)
        self._assert_goal_box_in_arm_reach()
        self._obs_delay = DelayQueue(n, int(c.obs_delay_steps), int(c.observation_space), dev)
        self._act_delay = DelayQueue(n, int(c.action_delay_steps), int(c.action_space), dev)
        self._obj_delay = DelayQueue(n, int(c.object_delay_steps), _OBJ_POSE_DIM, dev)
        self._wrench = WrenchDR(
            n, dev, force_scale=float(c.wrench_force_scale), torque_scale=float(c.wrench_torque_scale),
            prob_range=tuple(float(v) for v in c.wrench_prob_range))
        self._obj_mass = self._read_object_mass()
        # 스텝 간 되먹임 버퍼 — `_get_dones` 가 쓰고 `_get_rewards`·`_get_observations` 가 읽는다.
        self._hand_z_min = torch.zeros(n, device=dev)
        self._hand_floor_depth_max = torch.zeros((), device=dev)
        self._last_reward = torch.zeros(n, device=dev)
        # ★09.07 A-v: 리프트 후 지령 변화 벌점의 측도(정규화 지령 변화율, `_arm_command` 가 매 스텝 채움)
        #   + 회전 원지령 변화 노름(위치 쪽 `_palm_cmd_step_raw` 는 부모 버퍼).
        self._cmd_rate = torch.zeros(n, device=dev)
        self._palm_cmd_step_raw_rot = torch.zeros(n, device=dev)
        # ★A-vi 전역 arm 래치 상태(0-dim 텐서 — 분기 없이 `|=` 로 래치, per-step 동기화 금지). 프로세스 로컬.
        self._lift_ema = torch.zeros((), device=dev)
        self._cmd_rate_armed = torch.zeros((), dtype=torch.bool, device=dev)
        self._obs_shape_checked = False
        # 단계 사다리 재정의: 높이 래치 → 목표 1·2·3 (부모 `_reset_idx` 가 이 이름으로 평균 기록).
        self._stage_names = ("lifted", "goal1", "goal2", "goal3")
        self._stage_hit = torch.zeros(n, len(self._stage_names), dtype=torch.bool, device=dev)
        self._seed_palm_integrator(slice(None))       # 첫 리셋 전에도 지령 출발점이 있어야 한다
        print(f"[grasp_kp] 키포인트 s={c.keypoint_half_height():.3f}m · 목표 박스 "
              f"{[round(v, 3) for v in self._goal_cfg.box_min]}~"
              f"{[round(v, 3) for v in self._goal_cfg.box_max]} · "
              f"tol {'고정 ' + str(c.tol_eval) if float(c.tol_eval) > 0.0 else f'{c.tol_start}→{c.tol_floor}'} · "
              f"지연 obs/act/obj {c.obs_delay_steps}/{c.action_delay_steps}/{c.object_delay_steps} · "
              f"외란 {c.wrench_force_scale}N/kg·{c.wrench_torque_scale}N·m/kg · "
              f"질량 {float(self._obj_mass.min()):.3f}~{float(self._obj_mass.max()):.3f}kg", flush=True)

    def _setup_palm_step_gain(self) -> None:
        """★09.07 A-i: 팔 지령을 절대(앵커+델타) → **증분**으로 바꾸며 쓰는 걸음 크기.

        왜 바꾸나. 구식은 `a` 가 **위치**를 뜻했다 — a=−1 이 박스 맨 아래, +1 이 맨 위다.
        그런데 스텝당 실제 이동은 리미터가 `palm_cmd_rate_limit_m`(0.02 m)로 자른다.
        z 액션 범위가 0.70 m 이니 **한 스텝에 쓸 수 있는 건 범위의 2.9%** 뿐이고, 남는 몫은
        리미터가 다음 스텝에도 계속 같은 방향으로 밀어준다. 즉 레일에 붙어 있는 것이 공짜로
        이득이라 정책이 그리 수렴한다 — kp_a1/kp_a2 실측 `palm_cmd_rate_sat` 0.94/0.98,
        `palm_cmd_box_sat_z` 0.89/0.87. bounds_loss·entropy 로는 못 고친다(둘 다 mu 를
        벌할 뿐 게인 불일치를 못 건드린다. a2 에서 σ 만 24% 줄고 지령은 오히려 커졌다).

        증분식에서는 `a=±1` 이 곧 **허용된 최대 걸음**이라 더 크게 낼 이득이 사라진다.
        ★게인은 리미터에서 **파생**시킨다 — 상수로 적으면 리미터를 바꿀 때 조용히 어긋난다.
          정육면체 a∈[-1,1]³ 의 최악(대각) 노름이 리미터와 **정확히** 같도록 /√3 한다.
          그래야 어떤 액션에서도 리미터가 걸리지 않아 방향 왜곡이 생기지 않는다.
        """
        c, dev = self.cfg, self.device
        _lm = float(c.palm_cmd_rate_limit_m)
        _lr = math.radians(float(c.palm_cmd_rate_limit_rot_deg))
        if _lm <= 0.0 or _lr <= 0.0:
            raise RuntimeError(
                f"[grasp_kp] 증분 매핑은 걸음 크기를 리미터에서 파생시킨다 — "
                f"palm_cmd_rate_limit_m({c.palm_cmd_rate_limit_m}) 와 "
                f"palm_cmd_rate_limit_rot_deg({c.palm_cmd_rate_limit_rot_deg}) 는 둘 다 > 0 이어야 한다")
        # ★A-ii: 리미터 예산을 복원(pull)과 액션(step)이 **나눠 쓴다**. 안 나누면 둘이 겹칠 때
        #   합이 리미터를 넘어 A-i 가 없애려던 포화가 되돌아온다.
        _res = float(c.palm_cmd_leak_reserve)
        _k = (1.0 - _res) * _STEP_GAIN_MARGIN / math.sqrt(3.0)
        self._palm_step_gain = torch.cat([
            torch.full((3,), _lm * _k, device=dev),
            torch.full((3,), _lr * _k, device=dev)])
        self._palm_pull = float(c.palm_cmd_anchor_pull)
        self._palm_pull_cap = torch.tensor([_lm * _res, _lr * _res], device=dev)   # (xyz, rot)
        print(f"[grasp_kp] 팔 지령 = 증분+복원 · 걸음 {_lm * _k:.5f} m / "
              f"{math.degrees(_lr * _k):.3f}° · 복원 {self._palm_pull} (상한 {_lm * _res:.5f} m) · "
              f"리미터 {_lm} m / {c.palm_cmd_rate_limit_rot_deg}° 를 {1 - _res:.0%}/{_res:.0%} 로 "
              f"나눠 쓴다 → 합이 리미터를 못 넘는다", flush=True)

    def _seed_palm_integrator(self, env_ids) -> None:
        """증분 적분기의 출발점 = 액션 앵커(스폰 추종). 부팅 1회 + 리셋마다.

        부모는 리셋에서 `palm_targets` 를 **홈**으로 되돌리고 primed 를 내린다. 증분 매핑에서
        그대로 두면 `a=0` 이 홈 유지가 되어 컵에서 25 cm 떨어진 곳에서 출발한다 — Track B 를
        막은 바로 그 구조(긴 무보상 이동)를 A 가 물려받는다. 앵커로 씨딩하면 에피소드 첫 지령이
        구식과 같은 자리라 발견 조건이 보존되고, 바뀌는 것은 그 뒤의 **이동 규칙**뿐이다.
        ★부팅에서도 불러야 한다 — 씨딩 전에 `_arm_command` 가 돌면 이전 지령이 0 이라
          지령이 박스 구석으로 튄다(첫 리셋 전에는 `_palm_anchor` 가 홈을 준다).
        """
        if not bool(self.cfg.palm_cmd_incremental):
            return      # 절대 매핑은 매 스텝 앵커에서 새로 계산한다 — 유지할 적분기 상태가 없다.
        _anc = self._palm_anchor()[env_ids]
        self.palm_targets[env_ids] = _anc
        self._prev_palm_cmd[env_ids] = _anc[:, :3]
        self._prev_palm_cmd_rot[env_ids] = _anc[:, 3:6]
        self._palm_cmd_primed[env_ids] = True      # 출발점이 실제 지령이라 첫 스텝도 리미터 대상

    def _assert_kp_contract(self) -> None:
        """CLI 오버라이드가 접촉 의존 분기를 되살리면 부팅에서 죽는다(조용한 무시 금지)."""
        c = self.cfg
        if str(c.synergy_hold_mode) != "blocked" or bool(c.synergy_contact_freeze):
            raise RuntimeError(
                "[grasp_kp] 시너지 홀드는 'blocked' 만 허용(접촉 센서가 없다): "
                f"synergy_hold_mode={c.synergy_hold_mode!r} synergy_contact_freeze={c.synergy_contact_freeze}")
        if bool(c.respawn_on_fail):
            raise RuntimeError("[grasp_kp] respawn_on_fail 은 접촉 래치 규약이다 — 낙하는 리셋(False)")
        if bool(c.enable_adr):
            raise RuntimeError("[grasp_kp] ADR 금지 — 커리큘럼은 허용오차(tol_*) 하나뿐")
        if bool(getattr(c, "obs_object_rigid_after_latch", False)):
            raise RuntimeError("[grasp_kp] obs_object_rigid_after_latch 는 이 트랙에서 소비되지 않는다")
        _arm_dim = int(c._arm_action_dim(self.profile))      # A palm 6 / B n_arm
        if int(c.arm_cmd_dim) != _arm_dim:
            raise RuntimeError(
                f"[grasp_kp] cmd_state 폭 arm_cmd_dim={c.arm_cmd_dim} ≠ 팔 액션 폭 {_arm_dim}")

    def _apply_palm_floor_override(self) -> None:
        """palm 지령 박스 z 하한을 **올린다**(낮추지 않음) — 상판 관통 방지. 앵커가 밖이면 죽는다."""
        z = float(self.cfg.palm_box_min_z_override)
        if z <= 0.0:
            return
        old_palm, old_box = float(self._palm_lo[2]), float(self._box_lo[2])
        if z >= float(self._palm_hi[2]) or z > float(self._home_palm[2]):
            raise RuntimeError(
                f"[grasp_kp] palm_box_min_z_override={z} 가 박스 상한 {float(self._palm_hi[2]):.3f} "
                f"또는 홈 z {float(self._home_palm[2]):.3f} 를 넘는다")
        self._palm_lo[2] = max(old_palm, z)
        self._box_lo[2] = max(old_box, z)
        if self._anchor_mode == "spawn":
            _anchor_z = (float(self.cfg.table_surface_z) + float(self._obj_origin_off.min())
                         + float(self._anchor_off[2]))
            if _anchor_z < z:
                raise RuntimeError(
                    f"[grasp_kp] 스폰 앵커 z {_anchor_z:.3f} 가 새 하한 {z} 아래다 — a=0 이 잘린다. "
                    "palm_anchor_offset_xyz[2] 를 올리거나 override 를 낮춰라")
        print(f"[grasp_kp] palm 박스 z 하한 {old_palm:.3f}→{float(self._palm_lo[2]):.3f} · "
              f"최종 클램프 {old_box:.3f}→{float(self._box_lo[2]):.3f}", flush=True)

    def _assert_goal_box_in_arm_reach(self) -> None:
        """목표 박스 전 코너가 팔 지령 범위(앵커±델타 ∩ 클램프 박스) 안인지 부팅에서 대조(Track A).

        왜: 부모 `_assert_goal_reachable` 은 구 `goal_offset_xyz` 한 점만 본다. 목표 박스가 지령
        범위를 넘으면 목표열·tol 커리큘럼이 **조용히** 멈춘다(09.06 리뷰). 팔이 물체를 앵커
        오프셋으로 쥔 채 옮긴다고 보고, 목표+오프셋 ∈ 클램프 박스와 **이동 시간 예산**을 본다.
        ★09.07 A-i 로 매핑이 증분이 되어 "이동량 ∈ 델타" 전제가 사라졌다 — 적분기는 박스 전체를
          덮으므로 남는 제약은 걸음 수뿐이다.
        스폰 xy ±spawn_range·뱅크 정착고 최저/최고 극단, 허용오차 하한(tol_floor)만큼 여유.
        """
        if getattr(self, "fabric", None) is None:
            return                           # Track B: 관절공간 증분 — palm 박스가 지령 한계가 아니다
        c, p = self.cfg, self.profile
        if self._anchor_mode != "spawn":
            raise RuntimeError(f"[grasp_kp] palm_anchor_mode={self._anchor_mode!r} — 목표 박스 도달성은 spawn 앵커 전제다")
        tol, r = float(c.tol_floor), float(c.spawn_range)
        cx, cy = (float(v) for v in p.object_spawn_center)
        _tz = float(c.table_surface_z)
        s_lo = (cx - r, cy - r, _tz + float(self._obj_origin_off.min()))
        s_hi = (cx + r, cy + r, _tz + float(self._obj_origin_off.max()))
        g_lo, g_hi = self._goal_cfg.box_min, self._goal_cfg.box_max
        off = (self._anchor_off - self._fab_to_env).tolist()      # 목표(env-local) → palm 지령(fabric 프레임)
        b_lo, b_hi = self._box_lo[:3].tolist(), self._box_hi[:3].tolist()
        gain = self._palm_step_gain[:3].tolist()
        budget = _TRAVERSE_BUDGET_FRAC * float(self.max_episode_length)
        bad, worst = [], 0.0
        for i, ax in enumerate("xyz"):
            need_hi, need_lo = g_hi[i] - s_lo[i], g_lo[i] - s_hi[i]          # 최악 이동량
            p_hi, p_lo = g_hi[i] + off[i], g_lo[i] + off[i]                  # 필요한 palm 지령
            if p_hi > b_hi[i] + tol or p_lo < b_lo[i] - tol:
                bad.append(f"{ax}: palm [{p_lo:.3f},{p_hi:.3f}] ⊄ 클램프 박스 [{b_lo[i]:.3f},{b_hi[i]:.3f}]")
            # ★A-i(증분): 도달성은 "델타 안인가"가 아니라 "에피소드 안에 걸어갈 수 있는가"다.
            #   적분기는 박스 전체를 덮으므로 남는 제약은 **시간**뿐 — 걸음 수가 예산을 넘으면
            #   목표열이 조용히 멈춘다(구식의 델타 부족과 증상이 같다).
            if not bool(c.palm_cmd_incremental):
                continue        # 절대 매핑은 한 스텝에 박스 어디로든 지령한다 — 시간 제약이 없다
            steps = (max(abs(need_hi), abs(need_lo)) + tol) / max(gain[i], 1e-9)
            worst = max(worst, steps)
            if steps > budget:
                bad.append(f"{ax}: 이동 {max(abs(need_hi), abs(need_lo)):.3f} m 에 {steps:.0f} 스텝 "
                           f"> 예산 {budget:.0f} (에피소드 {int(self.max_episode_length)} 의 "
                           f"{_TRAVERSE_BUDGET_FRAC:.0%})")
        if bad:
            raise RuntimeError(
                "[grasp_kp] 목표 박스가 팔 지령 범위를 넘는다(±tol_floor 여유) — "
                "palm_cmd_rate_limit_m 을 키우거나 goal_box_* 를 줄여라: " + " · ".join(bad))
        _mode = ("증분 · 최악 이동 %.0f 스텝 / 예산 %.0f" % (worst, budget)
                 if bool(c.palm_cmd_incremental) else "절대 · 시간 제약 없음")
        print(f"[grasp_kp] 목표 박스 ⊂ 팔 지령 범위 ✓ ({_mode} · "
              f"클램프 z [{b_lo[2]:.3f},{b_hi[2]:.3f}] · tol 여유 {tol})", flush=True)

    def _read_object_mass(self) -> torch.Tensor:
        """물체 공칭 질량 (N,) — 외란 크기의 기준. 질량 DR 은 기본 항등이라 공칭값을 쓴다."""
        m = self.object.data.default_mass
        if m is None or m.ndim != 2 or m.shape[0] != self.num_envs:
            raise RuntimeError(
                f"[grasp_kp] object default_mass 형상 이상: {None if m is None else tuple(m.shape)} "
                f"(기대 ({self.num_envs}, 1))")
        return m[:, 0].to(self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 액션: 지연 큐 → self.actions → 팔 → 손 → fabric → 외란
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        flush = self.episode_length_buf == 0                # 리셋 직후 첫 지령은 큐 전 슬롯을 채운다
        # ★09.07 진단: 정책이 액션 상한에 붙어 있는 비율. sigma 팽창의 **직접** 증거다
        #   (지금까지는 losses/entropy 로 역산했다). 래퍼가 이미 ±1 로 자르므로
        #   |a| ≥ 0.99 는 "잘린 값"으로 읽는다.
        self._act_sat = (actions.abs() >= 0.99).float().mean()
        self._act_absmean = actions.abs().mean()
        self._act_raw = actions.detach()      # 축별 분해(로깅 전용)
        self.actions = self._act_delay.push(actions.clamp(-1.0, 1.0), flush)
        self._arm_command()
        self._hand_command()
        self._post_command()
        self._apply_wrench()

    def _arm_command(self) -> None:
        """팔 palm 6D 지령 → 박스 클램프 → 리미터(안전망) → 마커. 매핑은 cfg 로 고른다.

        **절대**(기본, `palm_cmd_incremental=False`) — `앵커 + 델타(a)`. `a` 가 위치를 뜻한다.
        구식 grasp_s2r 와 같은 식이고 kp_a1/kp_a2 가 이걸로 컵을 들었다(a2 e476 lifted 0.658).
        대가는 포화다 — 지령 변화의 2.9% 만 리미터를 통과해 `rate_sat` 0.99 · `step_raw` 0.17 m
        (리미터 0.02 m 의 10.8배)로 굳는다. 즉 액션의 크기 정보가 버려지고 방향만 남는다.

        **증분**(`palm_cmd_incremental=True`) — `이전지령 + 복원 + 게인·a`. `a=±1` 이 허용된 최대
        걸음이라 "레일에 붙이면 리미터가 대신 밀어준다"가 구조적으로 불가능하다. 실측으로
        `rate_sat` 0.000 · `step_raw` 0.0067 m · `arm_qd_p99` 2.30 → 1.30 을 얻었다.
        ★그런데 과제를 못 배웠다 — a3/a4/a5 세 번 모두 lifted 0.0000(사유는 cfg 주석).
          제어 품질만 보면 이쪽이 맞으므로 경로를 지우지 않고 스위치 뒤에 남긴다.

        ★적분기는 **클램프된 값**을 저장한다 — 원값을 저장하면 박스 밖에서 와인드업이 생겨
          정책이 방향을 바꿔도 한동안 지령이 안 움직인다.
        """
        if bool(self.cfg.palm_cmd_incremental):
            # a=0 → 앵커로 서서히 복귀. 복원 노름 상한이 있어 액션 몫과 합쳐도 리미터를 못 넘는다.
            step = self._palm_step_gain * self.actions[:, :6]
            _prev6 = torch.cat([self._prev_palm_cmd, self._prev_palm_cmd_rot], dim=1)
            _pull = self._palm_pull * (self._palm_anchor() - _prev6)
            _pull[:, :3] = _clamp_norm(_pull[:, :3], self._palm_pull_cap[0])
            _pull[:, 3:6] = _clamp_norm(_pull[:, 3:6], self._palm_pull_cap[1])
            _raw_targets = _prev6 + _pull + step
        else:
            # a=0 → 앵커. 탐색이 앵커 주변 유계 오프셋으로 묶인다(grasp_s2r 와 동일 식).
            delta = 0.5 * (self.actions[:, :6] + 1.0) * (self._delta_hi - self._delta_lo) \
                + self._delta_lo
            _raw_targets = self._palm_anchor() + delta
        self.palm_targets = _raw_targets.clamp(self._box_lo, self._box_hi)
        self._palm_cmd_box_sat = (self.palm_targets[:, :3] != _raw_targets[:, :3]).float()
        self._palm_delta_cmd = self.palm_targets - self._palm_anchor()   # 축별 로깅(앵커 기준 변위)

        _lim = float(self.cfg.palm_cmd_rate_limit_m)
        _lim_r = math.radians(float(self.cfg.palm_cmd_rate_limit_rot_deg))
        _step3 = self.palm_targets[:, :3] - self._prev_palm_cmd
        _dr = self.palm_targets[:, 3:6] - self._prev_palm_cmd_rot
        self._palm_cmd_step_raw = torch.where(
            self._palm_cmd_primed, _step3.norm(dim=-1), torch.zeros_like(self._palm_cmd_step_raw))
        self._palm_cmd_step_raw_rot = torch.where(
            self._palm_cmd_primed, _dr.norm(dim=-1), torch.zeros_like(self._palm_cmd_step_raw_rot))
        # ★09.07 A-v: 벌점 측도 = 리미터 **전** 원지령 변화를 리미터 상한으로 정규화(위치·회전 평균).
        #   1.0 = "리미터에 딱 맞게 지령", 10 = "리미터의 10배를 요구"(a2/a6 작동점 step_raw 0.20 m).
        #   리미터 **후** 값은 포화(rate_sat 0.96)에서 상수라 μ 에 기울기가 없다 — 그래서 원값이다.
        #   상한도 없다(작동점에서 clamp 되면 항이 상수가 된다). 리셋 첫 스텝(primed False)은 0.
        self._cmd_rate = torch.where(
            self._palm_cmd_primed,
            0.5 * (self._palm_cmd_step_raw / max(_lim, 1e-9)
                   + self._palm_cmd_step_raw_rot / max(_lim_r, 1e-9)),
            torch.zeros_like(self._cmd_rate))
        if _lim > 0.0:
            _scale = (_lim / _step3.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self._palm_cmd_rate_sat = ((_scale.squeeze(-1) < 1.0) & self._palm_cmd_primed).float()
            self.palm_targets[:, :3] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1), self._prev_palm_cmd + _step3 * _scale,
                self.palm_targets[:, :3])
        self._prev_palm_cmd = self.palm_targets[:, :3].clone()

        if _lim_r > 0.0:
            _sr = (_lim_r / _dr.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self.palm_targets[:, 3:6] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1), self._prev_palm_cmd_rot + _dr * _sr,
                self.palm_targets[:, 3:6])
        self._prev_palm_cmd_rot = self.palm_targets[:, 3:6].clone()
        self._palm_cmd_primed |= True
        self._update_cmd_markers()          # 시각화 전용(goal_pos 마커 포함) — 물리·보상 영향 없음

    def _hand_command(self) -> None:
        """grasp_s2r 손 구간 그대로: 케이지 닫기 게이트(높이 래치 뒤 해제) → 시너지 목표."""
        _prev = self._syn_target
        _obj = self._env_local(self.object.data.root_pos_w)
        _palm = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _cage = _palm + (self._palm_ee_R() @ self._cage_offset_palm)   # palm 강체 오프셋(홈 실측)
        self._cage_ctr_dist = self._banded_dist(_cage - _obj)
        if bool(self.cfg.close_gate_enabled):
            _ramp = max(float(self.cfg.close_gate_ramp) * self._r_cage, 1e-6)
            _g = ((self._r_cage - self._cage_ctr_dist) / _ramp).clamp(0.0, 1.0)
            # 들고 가는 중에 컵이 흔들려 게이트가 닫히면 다시 쥘 길이 막힌다 — 래치(lifted) 뒤 해제.
            self._close_gate = torch.where(self._latched, torch.ones_like(_g), _g)
        else:
            self._close_gate = torch.ones(self.num_envs, device=self.device)
        self._syn_target = self._hand_targets(self.actions[:, self._hand_action_offset:])
        self._syn_vel = (self._syn_target - _prev) / self._policy_dt

    def _hand_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """액션 손 구간 → 관절 목표 (N, n_hand). **A 는 시너지 그대로다.**

        ★09.08 이음매를 여기 만든 이유: 시너지 본체(`_synergy_targets`)는 grasp_s2r 불변 트랙
          안에 있어 Track B 가 덮을 수 없다. 호출 지점을 A 에 두면 B 는 이 한 메서드만 덮어
          손 제어법을 바꾸고, A 의 거동은 한 글자도 안 변한다.
        """
        return self._synergy_targets(a_hand)

    def _post_command(self) -> None:
        """fabric 손 상태를 실제 지령으로 동기화 → 적분 한 번(정책 스텝당)."""
        # 끊으면 fabric 이 없는 자기충돌을 피하려 팔을 민다(실측 palm_err 475mm).
        self.fabric_q[:, self.profile.num_arm_joints:] = self._syn_to_fab(self._syn_target)
        self._step_fabric()

    def _apply_wrench(self) -> None:
        """리프트 후 질량정규화 외란 — 매 스텝 새로 뽑고(decay 0) lifted 게이트, world 프레임."""
        forces, torques = self._wrench.step(self._obj_mass, self._latched)
        self.object.set_external_force_and_torque(forces, torques, is_global=True)

    # ------------------------------------------------------------------
    # 관측 — `_derive_spaces` 와 정확히 같은 순서·차원
    # ------------------------------------------------------------------
    def _action_obs(self) -> torch.Tensor:
        """관측의 액션 블록(action_space 폭). ★A 는 지연 큐를 통과한 액션 **그대로** — 산술 불변.

        09.08 이음매를 여기 만든 이유: SimToolReal 은 raw action 이 아니라 post-EMA `prev_action_targets`
        (팔+손)를 관측한다. 하위 트랙(B)이 손 관절 목표 q*_{t-1}(EMA 상태)을 정책에 보이게 하려면 이 한
        메서드만 덮는다 — 폭은 바뀌지 않고(계약 136), A 의 관측은 한 글자도 안 변한다.
        """
        return self.actions

    def _get_observations(self) -> dict:
        flush = self.episode_length_buf == 0
        pr = self._proprio_blocks()
        ob = self._object_blocks(pr["palm_pos"], flush)
        _nb = float(self.cfg.obs_noise_body)
        _act = self._action_obs()
        # actor 에만 노이즈 — 관절은 ADR 스칼라(OFF 면 base), 물체는 `_object_blocks` 의 코히런트 자세.
        _noisy = torch.cat([
            pr["arm_q"] + torch.randn_like(pr["arm_q"]) * self._adr_obs_noise_qpos,
            pr["arm_qd"] + torch.randn_like(pr["arm_qd"]) * self._adr_obs_noise_qvel,
            pr["hand_q"] + torch.randn_like(pr["hand_q"]) * self._adr_obs_noise_qpos,
            pr["hand_qd"] + torch.randn_like(pr["hand_qd"]) * self._adr_obs_noise_qvel,
            pr["palm_pos"] + torch.randn_like(pr["palm_pos"]) * _nb,
            pr["palm_ax"],
            pr["tips_rel_palm"] + torch.randn_like(pr["tips_rel_palm"]) * _nb,
            pr["cmd_state"],
            ob["n_kp_rel_palm"], ob["n_kp_rel_goal"],
            _act,
        ], dim=1)
        clean = torch.cat([
            pr["arm_q"], pr["arm_qd"], pr["hand_q"], pr["hand_qd"], pr["palm_pos"], pr["palm_ax"],
            pr["tips_rel_palm"], pr["cmd_state"], ob["kp_rel_palm"], ob["kp_rel_goal"],
            _act,
        ], dim=1)
        state = torch.cat([clean] + self._privileged_blocks(ob), dim=1)
        self._check_obs_shapes_once(_noisy, state)
        # 전체 actor 벡터에 obs 지연 큐(≤obs_delay_steps). NaN 은 큐에 들어가기 **전에** 지운다.
        policy = self._obs_delay.push(torch.nan_to_num(_noisy), flush)
        return {"policy": policy, "critic": torch.nan_to_num(state)}

    def _proprio_blocks(self) -> dict:
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        palm_w = self.robot.data.body_pos_w[:, self.palm_idx]
        _R = self._palm_ee_R()
        tips_w = self.robot.data.body_pos_w[:, self._tip_ids_t]
        return dict(
            arm_q=q[:, self._arm_ids_t], arm_qd=qd[:, self._arm_ids_t],
            hand_q=q[:, self._hand_ids_t], hand_qd=qd[:, self._hand_ids_t],
            palm_pos=self._env_local(palm_w),
            palm_ax=torch.cat([_R[:, :, 0], _R[:, :, 1]], dim=1),   # q ≡ −q 이중성 회피(부모 규약)
            tips_rel_palm=(tips_w - palm_w.unsqueeze(1)).reshape(self.num_envs, -1),
            cmd_state=self._cmd_state(),
        )

    def _cmd_state(self) -> torch.Tensor:
        """정책의 마지막 팔 지령 상태 (N, arm_cmd_dim). A = palm_targets − 앵커(6, 프레임 무관).

        Track B 는 이 훅만 덮어써 `q*_prev`(7) 를 준다.
        """
        return self.palm_targets - self._palm_anchor()

    def _object_blocks(self, palm_pos: torch.Tensor, flush: torch.Tensor) -> dict:
        """참값 파생(critic·지표) + 지각 파생(actor). 지각은 지연 → 노이즈 → **한 자세**에서 2항.

        물체 쿼터니언은 어느 벡터에도 넣지 않는다 — 키포인트가 위치·기울기를 담고 남는 건 yaw·부호뿐
        (축대칭 물체의 실기 yaw 는 임의라 배포 시 분포 밖 채널).
        """
        n, c = self.num_envs, self.cfg
        obj_pos = self._env_local(self.object.data.root_pos_w)
        obj_quat = self.object.data.root_quat_w
        kp_obj = keypoints_world(obj_pos, obj_quat, self._kp_offsets)          # (N,4,3)
        kp_goal = keypoints_world(self.goal_pos, self.goal_quat, self._kp_offsets)
        _pose = self._obj_delay.push(torch.cat([obj_pos, obj_quat], dim=1), flush)
        _p, _q = noisy_pose(_pose[:, :3], _pose[:, 3:], float(c.obs_object_xyz_std),
                            float(c.obs_object_rot_deg))
        kp_obs = keypoints_world(_p, _q, self._kp_offsets)
        return dict(
            kp_rel_palm=(kp_obj - palm_pos.unsqueeze(1)).reshape(n, -1),
            kp_rel_goal=(kp_obj - kp_goal).reshape(n, -1),
            n_kp_rel_palm=(kp_obs - palm_pos.unsqueeze(1)).reshape(n, -1),
            n_kp_rel_goal=(kp_obs - kp_goal).reshape(n, -1),
            kp_dist=keypoint_max_dist(kp_obj, kp_goal),
            dz=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
        )

    def _privileged_blocks(self, ob: dict) -> list[torch.Tensor]:
        """critic 전용 24차원(DESIGN §4): 물체·palm 속도, 진행 기준선, 래치, 진행률, 성공수, 보상, dz, d_kp."""
        return [
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.robot.data.body_lin_vel_w[:, self.palm_idx],
            self.robot.data.body_ang_vel_w[:, self.palm_idx],
            self._trk.closest_kp.unsqueeze(1),            # −1 = 새 목표(센티널) — critic 에겐 정보
            self._trk.closest_ft,
            self._latched.float().unsqueeze(1),
            (self.episode_length_buf.float() / float(self.max_episode_length)).unsqueeze(1),
            self._trk.successes.float().unsqueeze(1),
            (self._last_reward * 0.01).unsqueeze(1),
            ob["dz"].unsqueeze(1),
            ob["kp_dist"].unsqueeze(1),
        ]

    def _check_obs_shapes_once(self, policy: torch.Tensor, state: torch.Tensor) -> None:
        """첫 호출에서 조립 폭과 cfg 공식을 대조 — 어긋나면 두 숫자를 들고 죽는다."""
        if self._obs_shape_checked:
            return
        _o, _s = int(self.cfg.observation_space), int(self.cfg.state_space)
        if policy.shape[1] != _o or state.shape[1] != _s:
            raise RuntimeError(
                f"[grasp_kp] obs 조립 폭 ≠ cfg 공식: policy {policy.shape[1]} vs "
                f"observation_space {_o} · critic {state.shape[1]} vs state_space {_s}")
        self._obs_shape_checked = True

    # ------------------------------------------------------------------
    # 보상 — progress-only 8항 (접촉 0)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        qd = self.robot.data.joint_vel
        obj_pos = self._env_local(self.object.data.root_pos_w)
        tips_l = self.robot.data.body_pos_w[:, self._tip_ids_t] - self.scene.env_origins[:, None, :]
        ft_dist = (tips_l - obj_pos.unsqueeze(1)).norm(dim=-1)                 # (N, nt)
        kp_obj = keypoints_world(obj_pos, self.object.data.root_quat_w, self._kp_offsets)
        kp_goal = keypoints_world(self.goal_pos, self.goal_quat, self._kp_offsets)
        kp_dist = keypoint_max_dist(kp_obj, kp_goal)
        dz = obj_pos[:, 2] - self.object_spawn_pos[:, 2]
        near_goal, is_success = update_near_goal(kp_dist, self._tol.tol, self._trk, self._goal_cfg)
        # ★09.07 A-vi: 벌점은 **전역** 래치 뒤에만 — lifted_frac EMA ≥ 임계(a6 는 e130 에 0.30)면 sticky 로 arm.
        #   per-env lifted 만으로 켠 a7 은 e25 의 우연한 리프트(튕겨 올라갔다 상판에 놓인 컵, sticky 래치)에
        #   −1.35/step 이 500 스텝 붙어 접근 자체가 죽었다(e25 close 0.37 → e50 0.007). 래치는 되돌리지 않는다.
        _a = float(c.rw_cmd_rate_arm_ema)
        self._lift_ema = (1.0 - _a) * self._lift_ema + _a * self._latched.float().mean()
        self._cmd_rate_armed = self._cmd_rate_armed | (self._lift_ema >= float(c.rw_cmd_rate_arm_lifted_frac))
        total, terms, out = self._progress_reward(
            obj_z=obj_pos[:, 2], settled_z=self.object_spawn_pos[:, 2], lifted_prev=self._latched,
            ft_dist=ft_dist, closest_ft=self._trk.closest_ft,
            kp_dist=kp_dist, closest_kp=self._trk.closest_kp, near_goal=near_goal,
            is_success=is_success,
            arm_qd=qd[:, self._arm_ids_t], hand_qd=qd[:, self._hand_ids_t],
            hand_z_min=self._hand_z_min, cmd_rate=self._cmd_rate * self._cmd_rate_armed.float(), cfg=self._rw_cfg,
        )
        # 상태 되먹임 — 모듈은 무상태. 래치는 에피소드 리셋에서만 풀린다(sticky).
        self._latched = out["lifted"]
        self._trk.closest_ft = out["closest_ft"]
        self._trk.closest_kp = out["closest_kp"]
        self._advance_goals(is_success)
        self._success_now = is_success
        total = total + float(c.abnormal_penalty) * self._abnormal.float()
        self._last_reward = total
        if self._tol.update(self._trk.prev_episode_successes):
            print(f"[grasp_kp] 허용오차 커리큘럼 → tol {self._tol.tol:.4f}", flush=True)
        # arm 시점 기록 — 2000 스텝마다 한 번만 동기화(bool)한다.
        if self.common_step_counter % 2000 == 0 and bool(self._cmd_rate_armed):
            print(f"[grasp_kp] cmd_rate 벌점 ARMED · lift_ema {float(self._lift_ema):.3f} · step {self.common_step_counter}", flush=True)
        self._stage_hit[:, 0] |= self._latched
        for k in (1, 2, 3):
            self._stage_hit[:, k] |= self._trk.successes >= k
        self._log_step(terms, total, kp_dist, ft_dist, near_goal, out, dz)
        return total

    def _progress_reward(self, *, is_success: torch.Tensor, **kw):
        """보상 계산 이음매. ★A 는 공유 모듈 그대로 — `is_success` 를 받아서 **버린다**.

        A 의 `goal_bonus` 는 near_goal 스텝마다 `goal_bonus/success_steps` 를 나눠 주므로
        성공 순간이 필요 없다. 그래도 여기서 인자를 받는 이유는 `modules/progress_reward.py` 를
        **한 글자도 안 건드리기** 위해서다 — 여분 인자를 이음매가 흡수한다.

        B(`grasp_fj`)는 제어 방식이 달라 보상 모듈을 포크했고(09.08 사용자 확정) 이 메서드만
        덮는다. `_get_rewards` 본체는 양쪽이 공유한다.
        """
        return compute_progress_reward(**kw)

    def _advance_goals(self, is_success: torch.Tensor) -> None:
        """성공 env: successes+1 · 추적기 초기화 · **이전 목표** 기준 델타 목표(박스 클램프).

        분기 없이 where 로 쓴다 — per-step `.nonzero()` 동기화(util killer) 회피.
        """
        self._trk.successes += is_success.long()
        self._trk.clear_goal(is_success)
        new_pos, new_quat = sample_delta_goal(self.goal_pos, self.goal_quat, self._goal_cfg)
        m = is_success.unsqueeze(1)
        self.goal_pos = torch.where(m, new_pos, self.goal_pos)
        self.goal_quat = torch.where(m, new_quat, self.goal_quat)

    def _log_step(self, terms, total, kp_dist, ft_dist, near_goal, out, dz) -> None:
        ex = self.extras
        # ★09.08 모듈 전역 상수가 아니라 **반환된 terms 자체**를 순회한다.
        #   하위 트랙(B)이 보상 모듈을 포크했는데, 고정 이름표를 쓰면 포크한 항을 못 따라간다.
        #   ★인스턴스 이름표(`_rw_terms`)로 했다가 되돌렸다: B 는 그것을 `_setup_fabrics` 에서
        #   세팅하는데, 그 훅이 부모 `_init_task_state` **본문 중간**에서 불리고 그 뒤 이 클래스가
        #   다시 덮어써서 **B 에서 이음매가 아예 안 먹었다**(감사 실행 재현). 두 이름표가 우연히
        #   같아 무증상이었고, "그 줄이 있는가"만 보는 소스 테스트는 영원히 초록이었다.
        #   dict 는 삽입 순서를 보존하고 두 모듈 모두 순서 가드를 갖고 있으므로 이게 안전하다.
        for k, v in terms.items():
            ex[f"reward/{k}"] = v.mean()
        ex["reward/total"] = total.mean()
        _ck = self._trk.closest_kp
        ex["task/kp_dist"] = kp_dist.mean()
        ex["task/kp_dist_min_mean"] = torch.where(_ck >= 0.0, _ck, kp_dist).mean()
        ex["task/near_goal"] = near_goal.float().mean()
        ex["task/lifted_frac"] = self._latched.float().mean()
        ex["task/just_lifted"] = out["just_lifted"].float().mean()
        ex["task/dz"] = dz.mean()
        ex["task/ft_dist_mean"] = ft_dist.mean()
        ex["task/successes_mean"] = self._trk.successes.float().mean()
        ex["task/tol"] = self._tol.tol
        ex["task/hand_z_min"] = self._hand_z_min.mean()
        ex["task/hand_z_min_worst"] = self._hand_z_min.min()
        ex["task/hand_floor_depth_max"] = self._hand_floor_depth_max
        ex["task/abnormal_rate"] = self._abnormal.float().mean()
        ex["task/tilt_deg"] = self._tilt_deg.mean()
        ex["task/syn_close"] = self._syn_close.mean()
        ex["task/close_gate"] = self._close_gate.mean()
        # ★09.07 리프트 후 정지 판정의 1차 지표(정책 몫만) — 전체 평균은 리프트 전 env 가 뭉갠다.
        ex["task/cmd_rate_lifted"] = self._lifted_mean(self._cmd_rate)
        ex["task/cmd_rate_armed"] = self._cmd_rate_armed.float()
        ex["task/lift_ema"] = self._lift_ema
        self._log_probe_metrics(dz)
        self._log_fabric_metrics()

    def _log_probe_metrics(self, dz: torch.Tensor) -> None:
        """09.07 신설 진단 — 재학습 없이 probe 로 판정하기 위한 관측 전용 지표.

        ①액션 포화율: sigma 팽창이 실제로 액션을 벽에 붙이고 있나
        ②palm 실측 속도: 리미터 1.2 m/s 가 과한지 부족한지의 근거
        ③낙하율: `done/fell` 은 판정선 0.15 < 상판 0.205 라 **상판에 떨어뜨리는 것을
          못 잡는다**. 리프트 래치가 선 채 높이가 무너진 상태를 직접 센다
        ④팔 관절속도 분위수: `arm_qd_max` 는 4096×7 중 최댓값이라 이상치다
        """
        ex = self.extras
        ex["diag/action_sat"] = getattr(self, "_act_sat", torch.zeros((), device=self.device))
        ex["diag/action_absmean"] = getattr(self, "_act_absmean", torch.zeros((), device=self.device))
        _pv = self.robot.data.body_lin_vel_w[:, self.palm_idx].norm(dim=-1)
        ex["diag/palm_speed_mean"] = _pv.mean()
        ex["diag/palm_speed_p95"] = torch.quantile(_pv, 0.95)
        # 래치는 유지되는데 높이가 무너졌다 = 리프트 후 놓쳤다
        ex["diag/drop_frac"] = (self._latched & (dz < _DROP_DZ)).float().mean()
        # ★09.07 액션 매핑 판정 — 21/22 차원 평균은 팔 탓인지 손 탓인지 못 가른다.
        #   팔 구간(A: palm 6D / B: 관절 7D)과 손 시너지 15D 를 나눠 본다.
        _ar = getattr(self, "_act_raw", None)
        if _ar is not None:
            _o = self._hand_action_offset
            _sat = (_ar.abs() >= 0.99).float()
            ex["diag/act_sat_arm"] = _sat[:, :_o].mean()
            ex["diag/act_sat_hand"] = _sat[:, _o:].mean()
            for _i in range(_o):                    # 팔 축별 — 어느 축이 벽에 붙었나
                ex[f"diag/act_sat_arm{_i}"] = _sat[:, _i].mean()
        # 실현 델타 |목표−앵커| 를 박스 반폭과 나란히 본다(A 전용).
        _pd = getattr(self, "_palm_delta_cmd", None)
        if _pd is not None:
            for _i, _ax in enumerate("xyz"):
                ex[f"diag/palm_delta_{_ax}"] = _pd[:, _i].abs().mean()
        _qd = self.robot.data.joint_vel[:, self._arm_ids_t].abs()
        ex["diag/arm_qd_p50"] = torch.quantile(_qd, 0.50)
        ex["diag/arm_qd_p95"] = torch.quantile(_qd, 0.95)
        ex["diag/arm_qd_p99"] = torch.quantile(_qd, 0.99)
        # ★09.07 리프트 후 정지 판정 — lifted env 만. 물체 속도는 외란(≈2 g 킥, Δv 0.33 m/s)과 정책 진동이
        #   섞인 값이라 `task/cmd_rate_lifted`(정책 몫만)와 나란히 읽는다. nanquantile 이라 per-step 동기화가 없다.
        _m = self._latched
        ex["diag/arm_qd_p99_lifted"] = torch.nan_to_num(torch.nanquantile(
            torch.where(_m.unsqueeze(1), _qd, torch.full_like(_qd, float("nan"))), 0.99), nan=0.0)
        ex["diag/obj_speed_lifted"] = self._lifted_mean(self.object.data.root_lin_vel_w.norm(dim=-1))

    def _lifted_mean(self, x: torch.Tensor) -> torch.Tensor:
        """lifted env 만의 평균 (N,)→(). 빈 마스크는 0(nan 금지) · 분기 없음(per-step 동기화 금지)."""
        m = self._latched.float()
        return (x * m).sum() / m.sum().clamp(min=1.0)

    def _log_fabric_metrics(self) -> None:
        """부모 공식 그대로(joint_err 평균·최대, palm_err, 지령 원값). Track B(fabric 없음)는 건너뛴다."""
        if getattr(self, "fabric", None) is None:
            return
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        self.extras["fabric/palm_cmd_step_raw"] = self._palm_cmd_step_raw.mean()
        # ★`_arm_command` 가 산출만 하고 로깅은 안 하던 값. 리미터가 실제로 자르는 비율이다.
        self.extras["fabric/palm_cmd_rate_sat"] = self._palm_cmd_rate_sat.mean()
        # ★`_arm_command` 가 산출만 하던 축별 박스 포화. 클램프 박스가 실제로 무는지.
        for _i, _ax in enumerate("xyz"):
            self.extras[f"fabric/palm_cmd_box_sat_{_ax}"] = self._palm_cmd_box_sat[:, _i].mean()
        _jerr = (self.fabric_q[:, : self.profile.num_arm_joints]
                 - self.robot.data.joint_pos[:, self._arm_ids_t]).abs()
        self.extras["fabric/joint_err_mean"] = _jerr.mean()
        self.extras["fabric/joint_err_max"] = _jerr.max()     # 평균은 막힘 구간을 묻는다
        self.extras["fabric/palm_err_mean"] = (
            self.palm_targets[:, :3] + self._fab_to_env - palm_pos).norm(dim=-1).mean()

    # ------------------------------------------------------------------
    # 종료 — 부모 기하(tilt·out·fell·abnormal) + 손 바닥 관통 + max_goals truncation
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()      # respawn_on_fail=False 경로(부팅 가드)
        self._hand_z_min = self._compute_hand_z_min()
        floor_hit = self._hand_z_min < (float(self.cfg.table_surface_z)
                                        - float(self.cfg.hand_floor_terminate_depth))
        goals_done = self._trk.successes >= int(self.cfg.goal_max)   # 직전 스텝까지의 성공수
        terminated = terminated | floor_hit
        truncated = truncated | goals_done
        self.extras["done/hand_floor"] = floor_hit.float().mean()
        self.extras["done/max_goals"] = goals_done.float().mean()
        self.extras["done/truncated"] = truncated.float().mean()
        return terminated, truncated

    def _compute_hand_z_min(self) -> torch.Tensor:
        """손 **전 링크**(palm 제외) 최저 z, env-local (N,). 바닥 깊이 최댓값도 같이 잰다(진단)."""
        _z = (self.robot.data.body_pos_w[:, self._hand_body_ids_t, 2]
              - self.scene.env_origins[:, 2].unsqueeze(1))
        self._hand_floor_depth_max = torch.relu(float(self.cfg.rw_hand_floor_z) - _z).max()
        return _z.min(dim=1).values

    # ------------------------------------------------------------------
    # 리셋 — 부모(홈·시너지·fabric·스폰·정착고) 뒤에 목표·추적기·큐·외란
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)       # stage/* 기록 · 홈 · fabric 씨딩 · object_spawn_pos(정착고)
        # ★09.07 A-i: 부모가 palm_targets 를 홈으로 되돌린 **뒤** 앵커로 다시 씌운다
        #   (사유는 `_seed_palm_integrator`). `object_spawn_pos` 는 위 super() 가 갱신했다.
        self._seed_palm_integrator(env_ids)
        settled = self.object_spawn_pos[env_ids]
        self.goal_pos[env_ids], self.goal_quat[env_ids] = sample_first_goal(
            settled, self._settled_quat[env_ids], self._goal_cfg)
        self._trk.full_reset(env_ids)     # prev_episode_successes ← successes(커리큘럼 지표)
        self._obs_delay.reset(env_ids)
        self._act_delay.reset(env_ids)
        self._obj_delay.reset(env_ids)
        self._wrench.reset(env_ids)
        self._latched[env_ids] = False
        self._last_reward[env_ids] = 0.0
        self._success_now[env_ids] = False
