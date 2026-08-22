"""robot-agnostic grasp-lift 환경 (direct).

로봇 종속 정보는 전부 RobotProfile 이 공급한다 — 이 파일에 조인트/바디 이름 하드코딩 금지.

제어 스택 (사용자 결정, 2026-08-22 — diff-IK 에서 전환):
  팔  = Fabrics(geometric fabrics) 절대 palm 6D pose attractor.
        액션은 내부 절대 목표 누산기를 움직인다(a=0 = 목표 고정 = 유지).
        ★전환 근거: diff-IK relative 모드는 매 스텝 목표를 실측 pose 로 재기준화해
          action=0 이면 PD 오차가 0 → **복원력 0**. 만중력 zero-action 실측에서
          240스텝에 palm −22.3mm 처짐·무명령 기울기 18.7°·480스텝에 컵 낙하였다.
        ⚠대가: Fabrics 는 로봇당 fabric URDF 1벌이 필요하다 — 프로필만 추가하면 되던
          agnosticism 이 "프로필 + 자산 1벌"로 후퇴한다. 자산 생성은
          urdf/tools/gen_fabric_urdfs.py (FK 게이트 0.2mm).
  손  = relative joint position (dexsuite 방식) — 동결 게이트·커플링·PCA·latch·
        스크립트 램프 전부 없음. 외전 3축만 홈 고정.

중력: **만중력 고정**(반중력 커리큘럼 제거, 08.22). 난이도는 스폰 반경만 담당한다.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

# Fabrics (벤더링: hdgp/source/FABRICS/src — openarm/tasks/__init__ 가 sys.path 주입)
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tesollo
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

_FABRIC_MODULES = (_fab_tesollo,)


def _fabric_class(name: str):
    """프로필의 클래스 **이름 문자열** → 클래스. env 에 로봇 이름 하드코딩 금지 계약."""
    for mod in _FABRIC_MODULES:
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(f"fabric 클래스 '{name}' 를 찾을 수 없다: {[m.__name__ for m in _FABRIC_MODULES]}")

from .grasp_lift_env_cfg import GraspLiftEnvCfg
from .rewards import compute_grasp_lift_rewards
from .robot_profiles import PROFILES

_GRAVITY = 9.81


class GraspLiftEnv(DirectRLEnv):
    cfg: GraspLiftEnvCfg

    def __init__(self, cfg: GraspLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        p = PROFILES[self.cfg.profile_name]
        self.profile = p

        # ---- 조인트/바디 해석 (fail-loud: 프로필 선언 수와 대조) -----------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}!={p.num_arm_joints} "
                f"({arm_names}), hand {len(self.hand_ids)}!={p.num_hand_joints} ({hand_names})"
            )
        self._arm_ids_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self._hand_ids_t = torch.tensor(self.hand_ids, device=self.device, dtype=torch.long)

        # 정책이 건드리지 않는 손 관절(홈 고정) — hand_ids 안에서의 지역 인덱스로 분리.
        locked_ids: list[int] = []
        if p.hand_locked_joint_regex:
            locked_ids, locked_names = self.robot.find_joints(p.hand_locked_joint_regex)
            if len(locked_ids) != p.num_locked_hand_joints:
                raise RuntimeError(
                    f"[{p.name}] 고정 손관절 수 불일치: {len(locked_ids)}!="
                    f"{p.num_locked_hand_joints} ({locked_names})"
                )
            if not set(locked_ids) <= set(self.hand_ids):
                raise RuntimeError(f"[{p.name}] 고정 손관절이 hand_joint_regex 밖이다: {locked_names}")
        _locked_set = set(locked_ids)
        self._hand_free_local = torch.tensor(
            [i for i, j in enumerate(self.hand_ids) if j not in _locked_set],
            device=self.device, dtype=torch.long,
        )

        palm_ids, _ = self.robot.find_bodies(p.palm_body)
        if len(palm_ids) != 1:
            raise RuntimeError(f"[{p.name}] palm_body '{p.palm_body}' 해석 실패: {palm_ids}")
        self.palm_idx = palm_ids[0]
        self.tip_ids = []
        for n in p.fingertip_bodies:
            ids, _ = self.robot.find_bodies(n)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] fingertip body '{n}' 해석 실패: {ids}")
            self.tip_ids.append(ids[0])
        self._tip_ids_t = torch.tensor(self.tip_ids, device=self.device, dtype=torch.long)

        # ---- 팔 제어: Fabrics -------------------------------------------------------
        self._setup_fabrics()

        # ---- 목표/버퍼 -------------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits  # (N, J, 2)
        self._hand_lo = jl[:, self._hand_ids_t, 0]
        self._hand_hi = jl[:, self._hand_ids_t, 1]
        self._arm_lo = jl[:, self._arm_ids_t, 0]
        self._arm_hi = jl[:, self._arm_ids_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()
        self.hand_targets = self._default_q[:, self._hand_ids_t].clone()
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._abnormal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)       # env-local
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # ---- 커리큘럼 ---------------------------------------------------------------
        self.difficulty = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # _get_dones 가 매 스텝 갱신, _get_rewards 가 같은 스텝에 재사용(dones 가 먼저다)
        self._tilt_deg_buf = torch.zeros(self.num_envs, device=self.device)

        # ---- 접촉 그룹 인덱스 ---------------------------------------------------------
        fingers = list(p.finger_sensor_bodies.keys())
        self._finger_names = fingers
        self._group_a_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_a], device=self.device, dtype=torch.long)
        self._group_b_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_b], device=self.device, dtype=torch.long)
        # 인벨롭 손가락(감쌈 판정 분모·d_side wrap 그룹) — 프로필이 정의(pinky 제외 등)
        if not p.envelope_fingers:
            raise RuntimeError(f"[{p.name}] envelope_fingers 미정의 — 인벨롭 보상 성립 불가")
        self._env_finger_idx = torch.tensor(
            [fingers.index(f) for f in p.envelope_fingers], device=self.device, dtype=torch.long)
        self._group_b_env_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_b if f in p.envelope_fingers],
            device=self.device, dtype=torch.long)
        if len(self._group_b_env_idx) == 0:
            raise RuntimeError(f"[{p.name}] contact_group_b ∩ envelope_fingers 가 비었다")
        # 그룹 인덱스를 fingertip_bodies 에도 그대로 쓴다(접근 보상) — 두 목록의
        # 손가락 순서가 같아야 성립하므로 fail-loud 로 강제한다.
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"finger_sensor_bodies({len(fingers)}) 의 손가락 수가 달라 그룹 인덱스를 공유할 수 없다"
            )

        self._init_home_palm()

        print(f"[grasp_lift] profile={p.name} arm={len(self.arm_ids)} hand={len(self.hand_ids)} "
              f"tips={len(self.tip_ids)} action={self.cfg.action_space} obs={self.cfg.observation_space} "
              f"fabric={p.fabric_robot_dir}",
              flush=True)

    # ------------------------------------------------------------------
    def _init_home_palm(self) -> None:
        """홈 palm pose 실측 + fabric FK 정합 검사.

        ★`__init__` 시점의 body_pos_w 는 stale 이다(로봇이 아직 홈에 안 놓임). 반드시
          관절을 써넣고 물리를 2스텝 돌린 뒤 읽는다 — 병행 트랙이 이 함정에 당했다.
        """
        q0 = self.robot.data.default_joint_pos
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        home = self._palm_pose_6d()[0]
        self._home_palm = home.clone()
        self.palm_targets[:] = home.unsqueeze(0)

        out = (home < self._palm_lo) | (home > self._palm_hi)
        if bool(out.any()):
            raise RuntimeError(
                f"[{self.profile.name}] 홈 palm 이 워크스페이스 박스 밖이다: "
                f"home={[round(v, 3) for v in home.tolist()]} "
                f"lo={[round(v, 3) for v in self._palm_lo.tolist()]} "
                f"hi={[round(v, 3) for v in self._palm_hi.tolist()]}"
            )

        # ★fabric FK ↔ USD palm 정합. 이 한 줄이 (fabric URDF 오선택 / joint_order 오류 /
        #   palm_body 오지정) 3대 배선 사고를 부팅에서 전부 잡는다.
        fab = self.fabric.get_palm_pose(self.fabric_q.detach(), "euler_zyx")[0]
        dp = float(torch.norm(fab[:3] - home[:3]))
        dr = float(torch.max(torch.abs(fab[3:] - home[3:])))
        print(f"[grasp_lift] 홈 palm={[round(v, 4) for v in home.tolist()]} | "
              f"fabric FK 정합 pos {dp * 1000:.2f}mm rot {math.degrees(dr):.2f}°", flush=True)
        if dp > 0.005 or dr > math.radians(2.0):
            raise RuntimeError(
                f"[{self.profile.name}] fabric FK 가 USD palm 과 어긋난다: "
                f"{dp * 1000:.1f}mm / {math.degrees(dr):.1f}° (허용 5mm/2°). "
                "fabric_robot_dir·fabric_joint_order·palm_body 를 확인하라."
            )

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        # 정적 환경 USD (env.usd: RigidBodyAPI 없음, 전 메시 충돌체) —
        # env_0 에 spawn 하면 clone_environments 가 복제한다.
        tbl = self.cfg.table_cfg
        tbl.spawn.func(
            "/World/envs/env_0/Table", tbl.spawn,
            translation=tuple(tbl.init_state.pos), orientation=tuple(tbl.init_state.rot),
        )
        # ★테이블은 scene 자산이 아니라 정적 프림이라 EventTerm 이 못 건다. 직접 바인딩한다.
        #   씬 기본 재질만 믿으면 안 된다 — PhysX 결합이 average 라 한쪽만 낮아도 실효 μ 가
        #   중간값이 되고, 컵-테이블 마찰은 접근·안정에 직접 영향을 준다.
        _mu = float(self.cfg.surface_friction)
        _mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=_mu, dynamic_friction=_mu, restitution=0.0)
        _mat.func("/World/Materials/taskSurface", _mat)
        bind_physics_material("/World/envs/env_0/Table", "/World/Materials/taskSurface")

        # 손가락별 접촉 센서 — body 마다 개별 생성(다중 body 단일 센서는 force_matrix_w=0,
        # grasp_sensor 실측 함정). Object-only 필터.
        p = PROFILES[self.cfg.profile_name]
        _filter = list(self.cfg.object_contact_filter)
        self._finger_sensors: dict[str, list[ContactSensor]] = {}
        for finger, bodies in p.finger_sensor_bodies.items():
            sensors = []
            for body in bodies:
                s = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body}",
                    filter_prim_paths_expr=_filter,
                    history_length=1,
                    track_air_time=False,
                ))
                sensors.append(s)
                self.scene.sensors[f"contact_{finger}_{body}"] = s
            self._finger_sensors[finger] = sensors

        # env.usd 의 platform 상면이 정확히 z=0 이라 기본 지면(z=0)과 겹친다 —
        # 바닥은 env.usd base_plate(z -0.025~-0.015)가 담당, 지면은 시각 배경으로 내린다.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, -0.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # Fabrics — 절대 palm pose attractor. 목표는 **실측을 참조하지 않는** 자체 상태다.
    # ------------------------------------------------------------------
    def _palm_pose_6d(self) -> torch.Tensor:
        """현재 palm pose (env-local xyz + euler_zyx) — fabric 명령과 같은 규약."""
        pos = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([pos, torch.stack([y, pi, r], dim=1)], dim=1)

    def _build_fabric_index(self) -> torch.Tensor:
        """프로필 `fabric_joint_order` → articulation 인덱스. 순서가 유일한 방어선이다."""
        order = self.profile.fabric_joint_order
        if len(order) != self.fabric.num_joints:
            raise RuntimeError(
                f"[{self.profile.name}] fabric_joint_order 길이 {len(order)} != "
                f"fabric num_joints {self.fabric.num_joints}"
            )
        idx = []
        for name in order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{self.profile.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def _setup_fabrics(self) -> None:
        p = self.profile
        if not p.fabric_class or not p.fabric_robot_dir:
            raise RuntimeError(
                f"[{p.name}] fabric_class/fabric_robot_dir 가 없다. 이 태스크는 Fabrics 로만 돈다 "
                "— 자산을 만들거나(urdf/tools/gen_fabric_urdfs.py) 다른 프로필을 쓰라."
            )
        initialize_warp(str(self.device)[-1])          # 멀티 GPU 캐시 분리(grasp_v1 규약)
        self._world = WorldMeshesModel(
            batch_size=self.num_envs, device=self.device,
            max_objects_per_env=int(self.cfg.fabrics_max_objects_per_env),
        )
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        self.fabric = _fabric_class(p.fabric_class)(
            batch_size=self.num_envs, device=self.device,
            timestep=float(self.cfg.fabrics_dt),
            graph_capturable=bool(self.cfg.fabric_use_cuda_graph),
            use_hand_fabric=False,                     # 손은 Fabrics 밖(직접 관절 PD)
            robot_dir_name=p.fabric_robot_dir,
            robot_name=p.fabric_robot_dir,
        )
        self.integrator = DisplacementIntegrator(self.fabric)

        expect = p.num_arm_joints + p.num_hand_joints
        if self.fabric.num_joints != expect:
            raise RuntimeError(
                f"[{p.name}] fabric num_joints={self.fabric.num_joints} != 프로필 {expect}. "
                "fabric URDF 와 USD 자산이 어긋났다."
            )
        self._fab_t = self._build_fabric_index()
        # 손 구간만: fabric 순서의 손 관절이 hand_ids 안에서 몇 번째인가
        _hand_pos = {j: i for i, j in enumerate(self.hand_ids)}
        self._hand_to_fab_local = torch.tensor(
            [_hand_pos[int(j)] for j in self._fab_t[p.num_arm_joints:].tolist()],
            device=self.device, dtype=torch.long,
        )
        self.fabric_q = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, self.fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        # use_hand_fabric=False 라 무시되지만 원본 계약(B,5 PCA)은 지킨다.
        self._fabric_hand_cmd = torch.zeros(self.num_envs, 5, device=self.device)
        # cspace attractor(널스페이스) rest 자세를 프로필 홈으로.
        self.fabric.default_config.copy_(self.fabric_q)
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)

        # palm 목표 박스(env-local 절대) + 회전 박스
        d = math.pi / 180.0
        c = torch.tensor(p.palm_rot_center_deg, device=self.device) * d
        h = float(p.palm_rot_half_deg) * d
        self._palm_lo = torch.cat([torch.tensor(p.palm_box_min, device=self.device), c - h])
        self._palm_hi = torch.cat([torch.tensor(p.palm_box_max, device=self.device), c + h])
        self.palm_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self._home_palm = torch.zeros(6, device=self.device)   # _init_home_palm 에서 실측
        if not p.palm_box_verified:
            print(f"[grasp_lift] ⚠ palm_box 미검증({p.name}) — P-2 로 도달성 확인 후 승격할 것",
                  flush=True)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

        # ---- 팔: 내부 절대 목표 누산 -------------------------------------------------
        # ★diff-IK relative 모드는 매 스텝 목표를 **실측 pose 로 재기준화**해서 action=0 이면
        #   PD 오차가 0 이 됐다 → 복원력 0. 만중력 zero-action 실측: 240스텝에 palm −22.3mm,
        #   무명령 기울기 18.7°, 480스텝에 컵 낙하. 여기서는 목표가 자체 상태라 실측이
        #   흔들려도 되돌아온다. a=0 = "목표 고정 = 유지".
        step = torch.zeros_like(self.palm_targets)
        step[:, :3] = self.actions[:, :3] * float(self.cfg.arm_pos_scale)
        step[:, 3:] = self.actions[:, 3:6] * float(self.cfg.arm_rot_scale)
        # ★leash(목표를 실측±5cm 로 재클램프) 제거(08.22). 자유공간 추종오차는 mm 대라
        #   "막혔을 때의 와인드업 방지"가 명목이었지만, 정책이 상시 최대속도를 명령하는
        #   실제 학습에서는 **rate lag** 가 leash 폭을 상시 채웠다(lstm_test2 실측:
        #   palm_err_mean 0.051~0.065 vs 축별 한계 0.05 → 노름 천장 0.0866,
        #   leash_active_frac 0.43~0.90). 걸린 축의 액션 성분은 효과도 gradient 도 0 —
        #   즉 팔 액션의 절반이 버려지고 있었다. 목표 상한은 워크스페이스 박스가 맡는다.
        self.palm_targets = (self.palm_targets + step).clamp(self._palm_lo, self._palm_hi)

        # ---- 손: 절대 목표 누산(자유 관절만) + 관절한계 clamp -------------------------
        _delta = torch.zeros_like(self.hand_targets)
        _delta[:, self._hand_free_local] = self.actions[:, 6:] * float(self.cfg.hand_joint_scale)
        self.hand_targets = (self.hand_targets + _delta).clamp(self._hand_lo, self._hand_hi)

        # ---- Fabrics: 목표 주입 + 적분 (정책 스텝당 **한 번**) ------------------------
        # 손 목표를 fabric 순서로 넣어야 충돌구 FK 가 실제 손 자세를 본다(안 넣으면
        # 손이 접힌 줄 모르고 없는 충돌을 피하려 팔을 민다).
        self.fabric_q[:, self.profile.num_arm_joints:] = self._hand_to_fabric(self.hand_targets)
        self.fabric.set_features(
            self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self._world_ids, self._world_indicator, self._fabric_damping,
        )
        # ★`_apply_action` 은 decimation 만큼 불리므로 거기서 적분하면 fabric 시간이 2배로 흐른다.
        for _ in range(int(self.cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(self.cfg.fabrics_dt),
            )

    def _hand_to_fabric(self, hand_q: torch.Tensor) -> torch.Tensor:
        """손 목표(articulation hand 순서) → fabric 손 구간 순서."""
        return hand_q[:, self._hand_to_fab_local]

    def _apply_action(self) -> None:
        # Fabrics 가 만든 참조 궤적을 팔 PD 목표로. fabric_q 는 **오픈루프 plant** —
        # 실측 관절로 되돌려 동기화하면 팔이 명령을 못 따라간다(E1 사고 2건).
        arm_target = self.fabric_q[:, : self.profile.num_arm_joints]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.hand_targets, joint_ids=self.hand_ids)
        # ★만중력 고정(08.22): 반중력 보상력 제거. 게이트가 유효 중력을 토글하던 결합도
        #   함께 사라졌다 — 접촉이 순간 끊길 때 컵 무게가 급변하던 절벽이 없다.

    # ------------------------------------------------------------------
    def _contact_forces(self) -> torch.Tensor:
        """손가락별 물체 접촉력 크기 (N, F). body 별 센서 합산, Object-필터."""
        mags = []
        for finger in self._finger_names:
            total = torch.zeros(self.num_envs, device=self.device)
            for s in self._finger_sensors[finger]:
                fm = s.data.force_matrix_w  # (N, B, M, 3)
                total = total + fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
            mags.append(total)
        return torch.stack(mags, dim=1)

    def _contact_forces_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(mid, dist) 마디별 접촉력 (N, F) — envelope_frac 용. 같은 센서, 합산만 분리.

        finger_sensor_bodies 규약: 마지막 원소 = 팁, 그 앞이 (중간, 원위) 순.
        body 가 하나뿐인 손가락(2지 그리퍼 jaw)은 그 접촉 자체가 감쌈 — mid=dist=그 body.
        """
        mids, dists = [], []
        for finger in self._finger_names:
            sensors = self._finger_sensors[finger]
            n = len(sensors)
            mid_i, dist_i = (0, 1) if n >= 3 else (0, 0)

            def _mag(s):
                fm = s.data.force_matrix_w
                return fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

            mids.append(_mag(sensors[mid_i]))
            dists.append(_mag(sensors[dist_i]))
        return torch.stack(mids, dim=1), torch.stack(dists, dim=1)

    def _env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        joint_pos = torch.cat([q[:, self._arm_ids_t], q[:, self._hand_ids_t]], dim=1)
        joint_vel = torch.cat([qd[:, self._arm_ids_t], qd[:, self._hand_ids_t]], dim=1)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        tips = (
            self.robot.data.body_pos_w[:, self._tip_ids_t]
            - self.scene.env_origins[:, None, :]
        ).reshape(self.num_envs, -1)
        obj_pos = self._env_local(self.object.data.root_pos_w)
        obj_quat = self.object.data.root_quat_w
        # ★clip_observations=5.0 이 raw 접촉력을 5N 에서 자른다(env 의 clamp(20) 은 무효였다).
        #   5로 나눠 20N 에서 4.0 으로 포화시키면 의미(포화점 20N)는 유지하고 클립에 안 걸린다.
        contact = (self._contact_forces() / 5.0).clamp(max=4.0)
        obs = torch.cat([
            joint_pos, joint_vel, palm_pos, palm_quat, tips,
            obj_pos, obj_quat, self.goal_pos, contact, self.actions,
        ], dim=1)
        state = torch.cat([
            obs,
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.difficulty.float().unsqueeze(1) / float(self.cfg.curriculum_max_level),
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        tips = (
            self.robot.data.body_pos_w[:, self._tip_ids_t]
            - self.scene.env_origins[:, None, :]
        )
        contact = self._contact_forces()
        mid_f, dist_f = self._contact_forces_split()
        # 물체 기울기 — 같은 스텝에 _get_dones 가 계산·캐시한 값(dones 가 rewards 보다 먼저)
        tilt_deg = self._tilt_deg_buf
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        total, terms, gate, env_frac = compute_grasp_lift_rewards(
            object_tilt_deg=tilt_deg,
            height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
            palm_pos=palm_pos,
            fingertip_pos=tips,
            object_pos=obj_pos,
            goal_pos=self.goal_pos,
            group_a_tip_idx=self._group_a_idx,
            group_b_env_tip_idx=self._group_b_env_idx,
            group_a_force=contact[:, self._group_a_idx],
            group_b_force=contact[:, self._group_b_idx],
            env_mid_force=mid_f[:, self._env_finger_idx],
            env_dist_force=dist_f[:, self._env_finger_idx],
            actions=self.actions,
            prev_actions=self.prev_actions,
            cfg=self.cfg,
        )
        # abnormal 종료 페널티(관절한계) — _get_dones 가 같은 스텝에 계산한 플래그 사용
        total = total + float(self.cfg.abnormal_penalty) * self._abnormal_buf.float()
        self.prev_actions.copy_(self.actions)

        # ★성공 판정 3조건(08.22): goal 근접 AND 감쌈 AND 직립 — "인벨롭으로 세워 든
        #   것"만 성공. 커리큘럼 승급도 이 기준(리셋 시 마지막 값 사용).
        goal_dist = torch.norm(obj_pos - self.goal_pos, dim=-1)
        self._goal_reached_now = (
            (goal_dist < float(self.cfg.success_pos_tolerance))
            & (env_frac >= float(self.cfg.success_envelope_min))
            & (tilt_deg < float(self.cfg.success_tilt_max_deg))
        )

        # ★컵 단독 리스폰은 폐기됐다(08.22, 사용자 지시). 텔레포트 전이(s→s′ 불연속)가
        #   학습 데이터에 그대로 들어가 value/LSTM 이 비마르코프 점프를 학습했고, 손이
        #   스폰 지대에 있으면 컵이 손가락 위로 겹쳐 소환됐다("순간이동 관통").
        #   전도/낙하/이탈은 이제 _get_dones 의 **truncation** 이 env 전체 리셋으로 처리한다.
        self.extras["task/object_tilt_deg"] = tilt_deg.mean()
        self.extras["task/tipped_rate"] = (
            tilt_deg > float(self.cfg.tilt_reset_deg)).float().mean()

        # ---- 로깅 ----
        # 접근 기하 진단(가중 전 원거리) — terms 에 _접두로 실려 온다
        self.extras["task/d_palm"] = terms.pop("_d_palm").mean()
        self.extras["task/d_side"] = terms.pop("_d_side").mean()
        self.extras["task/envelope_frac"] = env_frac.mean()
        # 게이트 참인 env 의 감쌈 — "접촉은 됐는데 감쌈이 안 되는" 상태의 직접 지표
        _gf = gate.float()
        self.extras["task/envelope_at_gate"] = (
            (env_frac * _gf).sum() / _gf.sum().clamp(min=1.0))
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/contact_gate"] = gate.float().mean()
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/goal_reached"] = self._goal_reached_now.float().mean()
        self.extras["task/object_height_delta"] = (obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean()
        # 이전 12,000ep 런과의 연속성 비교 전용(보상·커리큘럼 미사용)
        self.extras["task/goal_reached_loose"] = (
            goal_dist < float(self.cfg.success_pos_tolerance_loose)).float().mean()
        # 게이트가 참인 env 의 평균 기울기 — 이번 런의 1차 판정 지표
        _g = gate.float()
        self.extras["task/tilt_at_gate"] = (
            (tilt_deg * _g).sum() / _g.sum().clamp(min=1.0))
        # Fabrics 추종(정상상태 오차 포함) — 배선 사고의 조기 신호
        _perr = torch.norm(self.palm_targets[:, :3] - self._palm_pose_6d()[:, :3], dim=-1)
        self.extras["fabric/palm_err_mean"] = _perr.mean()
        self.extras["fabric/palm_err_p95"] = torch.quantile(_perr, 0.95)
        self.extras["fabric/joint_err_max"] = (
            self.fabric_q[:, : self.profile.num_arm_joints]
            - self.robot.data.joint_pos[:, self._arm_ids_t]).abs().max()
        # leash 제거 후 목표-실측 격차는 위 palm_err_{mean,p95} 가 그대로 와인드업
        # 감시기 역할을 한다(상한이 없어졌으므로 발산하면 여기서 먼저 보인다).
        self.extras["fabric/palm_err_max"] = _perr.max()
        self.extras["curriculum/difficulty_mean"] = self.difficulty.float().mean()
        self.extras["curriculum/difficulty_max_frac"] = (
            self.difficulty == int(self.cfg.curriculum_max_level)).float().mean()
        _unused_gravity_frac = self.difficulty.float().mean() / float(
            self.cfg.curriculum_max_level)
        _hand_tau = self.robot.data.applied_torque[:, self._hand_ids_t].abs()
        self.extras["debug/hand/torque_mean"] = _hand_tau.mean()
        self.extras["debug/hand/torque_max"] = _hand_tau.max()
        for gi, gname in ((self._group_a_idx, "group_a"), (self._group_b_idx, "group_b")):
            self.extras[f"contact/{gname}_force"] = contact[:, gi].mean()
        # 관통 프록시: 정상 파지 접촉은 수 N 대. 수십~수백 N 은 깊은 상호침투를
        # 솔버가 밀어내는 신호다(08.20 사용자 지적 — 손가락이 컵을 뚫음).
        _d = torch.norm(tips - obj_pos[:, None, :], dim=-1)
        self.extras["task/dist_group_a"] = _d[:, self._group_a_idx].min(dim=-1).values.mean()
        self.extras["task/dist_group_b"] = _d[:, self._group_b_idx].min(dim=-1).values.mean()
        _raw = self._contact_forces()
        self.extras["contact/force_max"] = _raw.max()
        self.extras["contact/force_p95"] = torch.quantile(_raw.reshape(-1), 0.95)
        return total

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        spawn_xy = self.object_spawn_pos[:, :2]
        out_xy = (obj_pos[:, :2] - spawn_xy).norm(dim=-1) > float(self.cfg.object_out_of_bounds_xy)
        fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        # abnormal = 물리 위반만: 관절이 하드 한계를 실제로 넘었거나 속도 폭주.
        # ★근접도(범위의 99%) 기준은 오판정이었다 — j4 하한(0)은 "팔 뻗기"라는 정상
        #   동작의 종점이고 j2 는 init 자세부터 하한 근처다(probe 실측: +x 계단에서
        #   16/16 전멸). IK 해 clamp(위)가 명령을 한계 안에 가두므로, 실제 초과는
        #   접촉으로 밀렸을 때뿐이다.
        q_arm = self.robot.data.joint_pos[:, self._arm_ids_t]
        qd_arm = self.robot.data.joint_vel[:, self._arm_ids_t]
        beyond = (q_arm < self._arm_lo - 0.05) | (q_arm > self._arm_hi + 0.05)
        runaway = qd_arm.abs() > 20.0
        self._abnormal_buf = (beyond | runaway).any(dim=-1)
        # ---- 전도/낙하/이탈 = env 전체 리셋 (08.22, 사용자 지시) ----------------------
        # 두 실패한 선행 설계의 교훈을 모두 반영한 형태:
        #   · termination(agn_test2): done 이 미래 보상 전부를 끊어 **암묵적 대형 페널티**
        #     → "접근하지 않는" 회피 학습 (reaching 0.056→0.005 붕괴).
        #   · 컵 단독 리스폰(agn_test4~lstm_test4): 텔레포트 전이가 학습 데이터에 들어가
        #     value/LSTM 오염 + 손 위 겹침 소환.
        # → **truncation + value_bootstrap**(yaml): env 는 리셋되지만 V(s′) 부트스트랩이
        #   미래 보상을 대신해 return 절벽이 없다. 넘어뜨림의 대가는 tilt_penalty 와
        #   "낮은 V(쓰러진 상태)" 만큼만 — 보상과 무관한 순수 리셋에 가장 가깝다.
        _obj_z = quat_apply(
            self.object.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3),
        )
        self._tilt_deg_buf = torch.rad2deg(torch.acos(_obj_z[:, 2].clamp(-1.0, 1.0)))
        tipped = self._tilt_deg_buf > float(self.cfg.tilt_reset_deg)
        terminated = self._abnormal_buf
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        truncated = timeout | fell | tipped | out_xy
        self.extras["task/abnormal_rate"] = self._abnormal_buf.float().mean()
        self.extras["task/fell_rate"] = fell.float().mean()
        self.extras["task/out_xy_rate"] = out_xy.float().mean()
        self.extras["task/object_reset_rate"] = (fell | tipped | out_xy).float().mean()
        return terminated, truncated

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ---- 커리큘럼 갱신: 에피소드 종료 시점의 goal 근접 여부로 ±1 ----------------
        succ = self._goal_reached_now[env_ids]
        self.difficulty[env_ids] = (
            self.difficulty[env_ids] + torch.where(succ, 1, -1)
        ).clamp(0, int(self.cfg.curriculum_max_level))
        self._goal_reached_now[env_ids] = False

        # ---- 로봇: 프로필 init 자세 ---------------------------------------------------
        q0 = self._default_q[env_ids].clone()
        qd0 = torch.zeros_like(q0)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[:, self._hand_ids_t]
        # Fabrics 상태 씨딩 — 리셋 **외**에는 실측으로 동기화하지 않는다(오픈루프 plant)
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.palm_targets[env_ids] = self._home_palm.unsqueeze(0)
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # ---- 물체 스폰(난이도 비례 범위) + goal = 스폰 + z offset ----------------------
        p = self.profile
        frac = self.difficulty[env_ids].float() / float(self.cfg.curriculum_max_level)
        rng = float(self.cfg.spawn_range_initial) + frac * (
            float(self.cfg.spawn_range_final) - float(self.cfg.spawn_range_initial))
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng.unsqueeze(1)
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        # 높이는 cfg 단일 소스(table_surface_z + origin_offset + pad). 이중 패딩 금지.
        spawn[:, 2] = self.cfg.object_spawn_z
        # ★보상 기준선은 스폰점이 아니라 **정착고**다(패딩만큼 가라앉는다). 스폰점을
        #   기준으로 잡으면 그 패딩이 lift 보상의 데드존이 되고 goal 도 그만큼 멀어진다.
        settled = spawn.clone()
        settled[:, 2] = self.cfg.table_surface_z + self.cfg.object_origin_offset_z
        self.object_spawn_pos[env_ids] = settled
        self.goal_pos[env_ids] = settled + torch.tensor(
            [0.0, 0.0, float(self.cfg.goal_height_offset)], device=self.device)

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
