#!/usr/bin/env python3
"""통합 pour 씬 위에 두 파지 정책의 **배포 사슬**을 세우는 어댑터 — 재구현 금지.

실기 배포 구조(사용자 확정 09.02): 정책이 (로봇 상태 + FD++ 컵 pose)를 관측하고
액션을 내면 그게 로봇을 **직접** 제어한다. sim 리허설에서 '현실' 역할은 통합 pour
씬이고, 이 모듈은 그 씬 위에 각 학습 env 의 **원본 코드**로 사슬을 세운다:

  우(E1, grasp_s2r Direct):
      `GraspS2REnv._init_task_state`(추출 메서드)를 shim 위에서 그대로 실행 —
      인덱스·fabric·시너지·앵커·리미터·부팅 게이트(FK 5mm/2°)까지 전부 원본.
      관측도 원본 `_get_observations` 바인딩(래치 후 강체 지각 포함).
      latch 갱신만 `_get_rewards` 의 접촉 구간을 발췌한다(보상·respawn 없이).
  좌(v2B25, ManagerBased):
      학습 cfg 의 액션 텀 2개(`FabricPalmAction` + `GatedBinaryJointPositionAction`)
      를 얇은 env-shim 으로 인스턴스화. 관측은 배포 빌더(sim2real
      `left_obs_builder`) — 실기에 나갈 바로 그 코드다.

★미러(probe_bimanual_mirror)가 폭발한 3가지 — 매 프레임 관절 텔레포트(물리 우회로
  관통), 컵 root-pose 못박기(무한질량 임펄스 펌프 → 반대팔 360° 회전), 부착 프레임
  순간 스냅 — 는 여기 없다. 초기화 1회 뒤 상태 쓰기 금지, 컵은 자유 강체,
  파지는 접촉이 만든다. 여기서 실패하면 실기에서도 실패한다는 예측 신호다.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_control import GraspS2RControlMixin
from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env import GraspS2REnv
from openarm.agnostic.tasks.grasp_s2r.robot_profiles import PROFILES

LEFT9 = [f"l_aj_{i}" for i in range(1, 8)] + ["l_hj_gripper_1", "l_hj_gripper_2"]


def _v(field, joint: str) -> float:
    """액추에이터 cfg 필드(스칼라 또는 관절별 dict) → 해당 관절 값."""
    return float(field[joint] if isinstance(field, dict) else field)


# ───────────────────────────── 우팔: grasp_s2r shim ─────────────────────────
class RightChainShim(GraspS2RControlMixin):
    """grasp_s2r 의 제어·관측 사슬을 pour 씬 로봇 위에 세운다.

    로직은 전부 env 클래스에서 **바인딩**한다(바이트 동일). shim 이 소유하는 것은
    핸들(robot·scene·sim)과 버퍼 초기화뿐이다. `_init_task_state` 는 부팅 게이트
    3종(fabric FK 정합·palm 박스·joint 수)을 그대로 수행하므로 배선 사고는
    여기서 죽는다 — 조용히 틀릴 수 없다.
    """

    # ── 원본 바인딩 ──
    _init_task_state = GraspS2REnv._init_task_state
    _pre_physics_step = GraspS2REnv._pre_physics_step      # 리미터+시너지+fabric 적분
    _get_observations = GraspS2REnv._get_observations
    _perceived_object = GraspS2REnv._perceived_object
    _tip_force_local = GraspS2REnv._tip_force_local
    _joint_pos_err = GraspS2REnv._joint_pos_err
    _palm_anchor = GraspS2REnv._palm_anchor
    _setup_palm_anchor = GraspS2REnv._setup_palm_anchor
    _report_home_cage = GraspS2REnv._report_home_cage
    _assert_goal_reachable = GraspS2REnv._assert_goal_reachable
    _assert_adr_monotonic = GraspS2REnv._assert_adr_monotonic
    _adr_apply = GraspS2REnv._adr_apply
    _adr_apply_physics = GraspS2REnv._adr_apply_physics   # event_manager=None → no-op
    _lerp_range = staticmethod(GraspS2REnv._lerp_range)
    _palmar_mask = GraspS2REnv._palmar_mask

    def _setup_cmd_markers(self) -> None:                  # 시각화 전용 — shim 은 끈다
        pass

    def _update_cmd_markers(self) -> None:
        pass

    def __init__(self, host, cfg, finger_sensors: dict, palm_sensor) -> None:
        self.cfg = cfg
        self.robot = host.robot
        self.scene = host.scene
        self.sim = host.sim
        self.device = host.device
        self.num_envs = int(host.num_envs)
        self.physics_dt = float(host.physics_dt)
        self.object = host.cup                             # cup_big_s100 (러너가 검증)
        self._finger_sensors = finger_sensors
        self._palm_sensor = palm_sensor
        self.extras: dict = {}
        self.event_manager = None
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_length = int(round(
            float(cfg.episode_length_s) / (float(cfg.sim.dt) * int(cfg.decimation))))
        # ★_init_task_state 는 `_init_home_palm` 에서 로봇을 default 홈으로 놓고
        #   물리 2스텝을 밟는다 — 러너의 preset 배치가 여기서 같이 일어난다.
        self._init_task_state()

    # ------------------------------------------------------------------
    def update_latch(self) -> None:
        """`_get_rewards` 의 접촉→래치 구간 발췌 (grasp_s2r_env.py L830~L935).

        보상·respawn·커리큘럼 없이 래치와 `_obj_off_palm`(래치 후 강체 지각의
        기준 스냅샷)만 갱신한다. 발췌 드리프트는 러너의 재현 검증
        (`--verify`: 래치 프레임 대조)이 잡는다.
        """
        cfgn = self.cfg
        obj_pos = self._env_local(self.object.data.root_pos_w)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _thr = float(cfgn.contact_force_threshold)
        tip_c = self._tip_contact_forces() > _thr
        mid_f, dist_f = self._contact_forces_split()
        mid_c, dist_c = mid_f > _thr, dist_f > _thr
        if self._palmar_axes is not None:
            _pm = self._palmar_mask()
            mid_c = mid_c & _pm[:, :, 0]
            dist_c = dist_c & _pm[:, :, 1 if _pm.shape[2] >= 3 else 0]
            tip_c = tip_c & _pm[:, :, -1]
        grip_c = tip_c | mid_c | dist_c
        n_grip = grip_c.float().sum(dim=1)
        if str(getattr(cfgn, "latch_mode", "count")) == "opposition":
            _a_c = grip_c[:, self._group_a_idx].any(dim=1)
            _b_c = grip_c[:, self._group_b_idx].any(dim=1)
            _p_c = self._palm_contact_force() > _thr
            _ready = _a_c & (_b_c | _p_c)
        else:
            _ready = n_grip >= int(cfgn.lift_start_min_grip_fingers)
        self._hold_count = torch.where(
            _ready & ~self._latched, self._hold_count + 1,
            torch.where(self._latched, self._hold_count,
                        torch.zeros_like(self._hold_count)))
        _just = (~self._latched) & (
            self._hold_count >= int(cfgn.grasp_ready_hold_steps))
        self._latched = self._latched | _just
        _R = self._palm_ee_R()
        _off = torch.einsum("nji,nj->ni", _R, obj_pos - palm_pos)
        self._obj_off_palm = torch.where(_just.unsqueeze(1), _off, self._obj_off_palm)

    # ------------------------------------------------------------------
    def step_policy(self, action: torch.Tensor, *, render: bool,
                    on_substep=None) -> torch.Tensor:
        """DirectRLEnv.step 의 물리 경로만 — 보상·리셋 없음. actor obs 를 돌려준다.

        on_substep: 물리스텝마다 부르는 훅(실기로 관절목표 송출) — 좌 체인과 동일 규약.
        """
        self.prev_actions.copy_(self.actions)
        self._pre_physics_step(action)
        for _ in range(int(self.cfg.decimation)):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.physics_dt)
            if on_substep is not None:
                on_substep()
        self.episode_length_buf += 1
        self.update_latch()
        return self._get_observations()["policy"]

    def observe(self) -> torch.Tensor:
        return self._get_observations()["policy"]

    def zero_obs_noise(self) -> None:
        """결정론 리허설·재현 검증용 — DR 관측 노이즈를 끈다."""
        self._adr_obs_noise_object = 0.0
        self._adr_obs_noise_qpos = 0.0
        self._adr_obs_noise_qvel = 0.0

    def freeze_targets(self) -> None:
        """국면 종료 — 위치 목표는 유지, 속도 FF 는 0 (안 하면 상시 드리프트)."""
        za = torch.zeros(self.num_envs, len(self.arm_ids), device=self.device)
        self.robot.set_joint_velocity_target(za, joint_ids=self.arm_ids)
        zh = torch.zeros(self.num_envs, len(self._syn_ids), device=self.device)
        self.robot.set_joint_velocity_target(zh, joint_ids=self._syn_ids)


# ───────────────────────────── 좌팔: v2B25 텀 사슬 ──────────────────────────
class LeftChain:
    """v2B25 좌팔 사슬 — 학습 cfg 의 액션 텀 2개를 pour 씬 위에 인스턴스화.

    텀이 요구하는 env 표면은 (num_envs·device·scene·step_dt) 뿐이라
    `SimpleNamespace` 로 충분하다. 물체 참조 이름만 pour 씬의 것으로 바꾼다
    (`fine_latch_object_name`/`object_name` → left_target_cup) — 값이 아니라
    **이름**의 오버라이드라 학습 파라미터는 건드리지 않는다.
    """

    def __init__(self, host, left_env_cfg, *, object_entity: str = "left_target_cup",
                 step_dt: float = 0.02) -> None:
        shim = SimpleNamespace(num_envs=int(host.num_envs), device=host.device,
                               scene=host.scene, step_dt=float(step_dt))
        acfg = left_env_cfg.actions.arm_action
        acfg.debug_vis = False
        acfg.fine_latch_object_name = object_entity
        gcfg = left_env_cfg.actions.gripper_action
        gcfg.object_name = object_entity
        self.arm = acfg.class_type(acfg, shim)
        self.grip = gcfg.class_type(gcfg, shim)
        n = int(self.arm.action_dim) + int(self.grip.action_dim)
        if n != 7:
            raise RuntimeError(f"좌팔 액션 차원 {n} ≠ 7 — 텀 구성이 학습과 다르다")
        self.host = host
        self.decimation = int(left_env_cfg.decimation)
        self.step_dt = float(step_dt)

    def reset(self) -> None:
        self.arm.reset(None)
        self.grip.reset(None)

    @property
    def gate_open(self) -> torch.Tensor:
        return self.grip.gate_open

    def step_policy(self, action: torch.Tensor, *, render: bool,
                    on_substep=None) -> None:
        """액션 1회 → fabric 이 **물리스텝마다** 관절목표를 재생성 → PD.

        ★on_substep 은 그 관절목표를 실기로 흘리기 위한 훅이다. 정책 주기(50Hz)로만
        솎아 보내면 fabric 이 만들어 둔 보간이 버려져 실기엔 4~6° 씩 점프해 도착한다
        (09.02 bag 실측). 학습·s2r 모두 fabric 을 쓰므로 실기 PD 도 같은 신호를 받아야
        한다: action → fabric IK → 관절목표 → PD.
        """
        na = int(self.arm.action_dim)
        self.arm.process_actions(action[:, :na])
        self.grip.process_actions(action[:, na:])
        for _ in range(self.decimation):
            self.arm.apply_actions()
            self.grip.apply_actions()
            self.host.scene.write_data_to_sim()
            self.host.sim.step(render=render)
            self.host.scene.update(self.host.physics_dt)
            if on_substep is not None:
                on_substep()

    def freeze_targets(self) -> None:
        robot = self.host.robot
        z = torch.zeros(int(self.host.num_envs), 7, device=self.host.device)
        robot.set_joint_velocity_target(z, joint_ids=self.arm._arm_joint_ids)


# ───────────────────────────── 씬: 센서 확장 ────────────────────────────────
def make_show_env(pour_env_cls):
    """pour env 서브클래스 — 우손 접촉 센서를 mixin 규약대로 추가한다.

    ★body **하나당 센서 하나** + 컵 필터. 다중 body 를 한 센서에 묶으면
      `force_matrix_w` 가 무증상 0 을 반환한다(grasp_s2r 실측 함정 그대로).
      pour 자체 센서(distal/middle)는 필터가 없어 래치·obs 에 못 쓴다.
    """

    class BimanualShowEnv(pour_env_cls):
        def _setup_scene(self):
            super()._setup_scene()
            p = PROFILES["tesollo_right"]
            filt = ["/World/envs/env_.*/Cup/baseLink"]
            self.bi_finger_sensors: dict[str, list] = {}
            for finger, bodies in p.finger_sensor_bodies.items():
                arr = []
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=filt,
                        history_length=1, track_air_time=False))
                    self.scene.sensors[f"bi_contact_{finger}_{body}"] = s
                    arr.append(s)
                self.bi_finger_sensors[finger] = arr
            ps = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{p.palm_body}",
                filter_prim_paths_expr=filt,
                history_length=1, track_air_time=False))
            self.scene.sensors["bi_contact_palm"] = ps
            self.bi_palm_sensor = ps

            # ★마찰 명시 바인딩 — 학습에선 매 리셋 DR 이 재질을 덮었으므로 USD 기본값은
            #   **학습이 본 적 없는 값**이다. 기본값(저마찰)으로 두면 살짝만 스쳐도 컵이
            #   1m 를 미끄러져 날아간다(09.02 diagL4 실측: shaker → (0.99, 0.97)).
            #   좌 DR 중앙값(컵 0.8/0.65 · 턱 1.0/0.85), 우는 E1 고정값 1.0.
            import isaaclab.sim as sim_utils  # noqa: PLC0415
            from isaaclab.sim.utils import bind_physics_material  # noqa: PLC0415
            from isaaclab.sim.utils import find_matching_prim_paths  # noqa: PLC0415
            mats = (
                ("shakerMat", 0.8, 0.65, ("/World/envs/env_.*/LeftTargetCup",)),
                # 턱 마찰은 DR 상단쪽(범위 0.6~1.4) — 09.02 전환 램프에서 중앙값(1.0)으로는
                # 쥔 컵이 이송 중 미끄러져 빠졌다. 이송 여유가 필요하고 범위 안이다.
                ("jawMat", 1.3, 1.1,
                 ("/World/envs/env_.*/Robot/l_hl_gripper_left_finger",
                  "/World/envs/env_.*/Robot/l_hl_gripper_right_finger")),
                ("cupMat", 1.0, 1.0, ("/World/envs/env_.*/Cup",)),
                # ★테이블 = 좌 학습 씬 기본 재질 0.5 (좌 env 는 테이블에 명시 재질이
                #   없어 sim 기본 0.5 가 적용됐다). pour 기본 1.0 이면 컵-테이블 결합
                #   μ가 0.65→0.9 로 올라 스침이 미끄럼 대신 **전도**가 된다(diagL5).
                #   우측 파워그립은 0.75 로 내려가도 견딘다(+10mm 테이블에서도 성공).
                ("tableMat", 0.5, 0.5, ("/World/envs/env_.*/Table",)),
            )
            for name, s, d, pats in mats:
                mcfg = sim_utils.RigidBodyMaterialCfg(
                    static_friction=s, dynamic_friction=d, restitution=0.0)
                mcfg.func(f"/World/Materials/{name}", mcfg)
                bound = 0
                for pat in pats:
                    for tp in find_matching_prim_paths(pat):
                        bind_physics_material(tp, f"/World/Materials/{name}")
                        bound += 1
                if bound == 0:
                    raise RuntimeError(f"[마찰] {name}: 바인딩 대상 프림이 없다 {pats}")
                print(f"[마찰] {name} μs={s} μd={d} → {bound}프림", flush=True)

            # ★★좌팔 링크만 중력 ON — 로봇 전체는 E1 규약(disable_gravity=True)인데
            #   v2B25 는 **중력을 켠 채** 학습했다(fab env "실기에는 중력이 있다").
            #   중력이 꺼진 좌팔은 첫 스텝부터 속도가 0.2 rad/s 어긋나 접근 궤적이
            #   갈라졌다(09.02 diagL7: obs1 joint_vel Δ0.198 — 접촉 전 발산의 진범).
            #   disable_gravity 는 링크 단위 속성이라 팔별로 갈라칠 수 있다.
            from isaaclab.sim.schemas import modify_rigid_body_properties  # noqa: PLC0415
            _gon = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
            _n = 0
            for tp in find_matching_prim_paths(
                    "/World/envs/env_.*/Robot/l_(al|hl)_.*"):
                modify_rigid_body_properties(tp, _gon)
                _n += 1
            if _n == 0:
                raise RuntimeError("[중력] 좌팔 링크 프림을 못 찾았다")
            print(f"[중력] 좌팔 링크 {_n}개 중력 ON (v2B25 학습 조건)", flush=True)

    return BimanualShowEnv


# ───────────────────────────── cfg 정합 ─────────────────────────────────────
def _eff(actuator: dict) -> float:
    v = actuator.get("effort_limit_sim")
    if v is None:
        v = actuator.get("effort_limit")
    return float(v)


def _expand_right_gains(dump_actuators: dict) -> dict:
    """우팔 dump 액추에이터(그룹 구성이 KUKA/real 분기마다 다름) → 관절별 (kp,kd).

    real 분기는 관절별 그룹(right_arm_j1..j7), KUKA 분기는 [1-4]/5/6/7 그룹이다.
    joint_names_expr 를 r_aj_1..7 에 정규식 매칭해 관절별 맵으로 편다.
    """
    import re
    names = [f"r_aj_{i}" for i in range(1, 8)]
    out: dict = {}
    for grp in dump_actuators.values():
        exprs = grp.get("joint_names_expr") or []
        for n in names:
            if any(re.fullmatch(e, n) for e in exprs):
                kp, kd = grp.get("stiffness"), grp.get("damping")
                out[n] = (float(kp[n] if isinstance(kp, dict) else kp),
                          float(kd[n] if isinstance(kd, dict) else kd))
    if len(out) != 7:
        raise RuntimeError(f"우팔 게인 전개 실패 — {sorted(out)} (7관절이어야 한다)")
    return out


def align_pour_cfg(env_cfg, *, left_scene: dict, right_actuators: dict,
                   lgrip_usd: str, left_spawn, physics_dt: float) -> list[str]:
    """pour 씬을 두 학습 env 의 물리와 정합 (09.02 감사 결과). 적용 로그를 반환.

    우측은 pour 가 이미 정합돼 있다(자산 hull 동일·게인 300/45…·hand 5/2·dt 1/120)
    — warm 인계를 위해 의도적으로 맞춰져 있었기 때문이다. 좌측만 4개를 바꾼다.
    ★정책 홈 주입은 여기가 아니라 러너가 `default_joint_pos` 에 직접 한다 —
      cfg 의 init_state 는 정규식 키라 구체 관절명을 섞으면 다중 매치로 죽는다.
    """
    log: list[str] = []
    env_cfg.scene.num_envs = 1
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    # 컵 = cup_big_s100 단일 (E1 지정 컵). 뱅크 교체는 finalize 가 파생까지 다시 만든다.
    env_cfg.object_bank = "single_cup"
    log.append("object_bank → single_cup (cup_big_s100)")
    # warm 텔레포트 차단 — 이 리허설의 시작은 preset 이지 warm 자세가 아니다.
    # ★paths=() 만으로는 부팅이 거부된다(source='disk' 가드) — source 도 preset 으로.
    if hasattr(env_cfg, "warm_state_paths"):
        env_cfg.warm_state_paths = ()
        if hasattr(env_cfg, "warm_state_source"):
            env_cfg.warm_state_source = "preset"
        log.append("warm_state_paths → () · source → preset (리셋 텔레포트 차단)")

    # 로봇 자산: lgrip (좌 그리퍼 3링크 convexDecomposition — v2B25 학습 자산.
    # 우측 헐은 hull 판과 파일 동일(base·robot·sensor cmp 일치)이라 E1 도 충실).
    env_cfg.robot_cfg.spawn.usd_path = lgrip_usd
    log.append(f"robot usd → lgrip ({lgrip_usd.rsplit('/', 2)[-2]})")

    # 좌팔 게인 = v2B25 dump 벤더값 (pour 는 2000/200·400/80 — 그대로면 사슬이 다른 팔이 된다)
    acts = env_cfg.robot_cfg.actuators
    la, lg = left_scene["robot"]["actuators"]["left_arm"], \
        left_scene["robot"]["actuators"]["left_gripper"]
    acts["openarm_left_arm"].stiffness = dict(la["stiffness"])
    acts["openarm_left_arm"].damping = dict(la["damping"])
    acts["openarm_left_arm"].effort_limit_sim = _eff(la)
    acts["openarm_left_gripper"].stiffness = float(lg["stiffness"])
    acts["openarm_left_gripper"].damping = float(lg["damping"])
    acts["openarm_left_gripper"].effort_limit_sim = _eff(lg)
    # ★★속도한계 — 게인만 복사하면 팔이 학습보다 **빨라진다**. v2B25 는 팔 2.175/2.61
    #   rad/s · 그리퍼 0.2 로 묶여 학습됐는데 pour 는 무제한이라, 같은 지령에도 과속
    #   접근이 컵을 쳐냈다(09.02 diagL3: 첫 지령은 기록과 동일한데 step 10~20 발산).
    vl = la.get("velocity_limit_sim")
    if vl is not None:
        acts["openarm_left_arm"].velocity_limit_sim = (
            dict(vl) if isinstance(vl, dict) else float(vl))
    vg = lg.get("velocity_limit_sim")
    if vg is not None:
        acts["openarm_left_gripper"].velocity_limit_sim = float(vg)
    log.append("좌 게인 → 벤더 (kp 70/…/10 · 그리퍼 2000/100 · ★속도한계 2.175/2.61 · 그리퍼 0.2)")

    # ★우팔 게인 — `arm_gain_profile` 지정이 정공법. pour cfg 의 finalize 가
    #   `_apply_arm_gain_profile()` 로 액추에이터를 **재구성**하므로, 값 직접 주입은
    #   env 부팅에서 조용히 덮인다(09.02 실측 — cl_g1 의 "우 게인 → dump" 는 무효였다).
    #   "r2s" 프로필은 g1 dump(HDGP_S2R_REAL_GAINS) 실측값과 동일함이 cfg 주석에 명시.
    _real = any(k.startswith("right_arm_j") for k in right_actuators)
    env_cfg.arm_gain_profile = "r2s" if _real else "kuka"
    log.append(f"우 게인 프로필 → {env_cfg.arm_gain_profile} (우 dump 분기 자동)")

    # shaker: pour 의 kinematic 마커 → **자유 강체** (여기가 미러 시대의 유령이었다)
    lt = env_cfg.left_target_cup_cfg
    lt.spawn.rigid_props.kinematic_enabled = False
    lt.spawn.rigid_props.disable_gravity = False
    lt.spawn.rigid_props.max_depenetration_velocity = 5.0
    lt.spawn.rigid_props.max_linear_velocity = 1000.0
    lt.spawn.rigid_props.max_angular_velocity = 1000.0
    lt.spawn.rigid_props.solver_position_iteration_count = 16
    lt.spawn.rigid_props.solver_velocity_iteration_count = 1
    lt.spawn.collision_props = None          # contact_offset -0.1(충돌 무력화) 제거
    import isaaclab.sim as sim_utils
    lt.spawn.mass_props = sim_utils.MassPropertiesCfg(
        mass=float(left_scene["object"]["spawn"]["mass_props"]["mass"]))
    lt.init_state.pos = tuple(float(v) for v in left_spawn)
    log.append(f"shaker → 동적 강체 · mass {lt.spawn.mass_props.mass} · "
               f"스폰 {tuple(round(float(v), 3) for v in left_spawn)}")

    # ★테이블 상면 정합 — pour 원판 테이블(scene_objects/table.usd)은 학습 테이블보다
    #   15mm 높다(09.02 컵 정착 실측: 좌 +15mm·우 +10mm — 좌 게이트 along 대역의 절반).
    #   단, e1_pour1 dump 복원이 테이블을 env_rigid.usd(E1 테이블, 상면 정확히 0.2)로
    #   바꾸면 보정이 오히려 어긋난다 — 그 테이블은 그대로가 정합이다.
    tb = env_cfg.table_cfg.init_state
    if "scene_objects/table.usd" in str(env_cfg.table_cfg.spawn.usd_path):
        tb.pos = (float(tb.pos[0]), float(tb.pos[1]), float(tb.pos[2]) - 0.015)
        log.append("테이블 z −15mm (pour 원판 테이블 → 학습 상면 정합)")
    else:
        log.append(f"테이블 {str(env_cfg.table_cfg.spawn.usd_path).rsplit('/', 1)[-1]}"
                   " — 보정 없음 (학습 테이블 그대로)")

    # 물리 dt: 좌 국면(먼저)의 100 Hz 로 부팅. 우 국면 전에 러너가 1/120 로 전환.
    env_cfg.sim.dt = float(physics_dt)
    log.append(f"sim.dt → {physics_dt} (좌 국면 기준, 우 국면 전 1/120 전환)")
    return log


# ───────────────────────────── pour 국면 (e1_pour1) ─────────────────────────
def init_pour_from_live(env, right_shim, pour_actuators: dict) -> None:
    """pour 에피소드 상태를 **현재 물리 상태**로 초기화 — warm 텔레포트의 라이브 대응물.

    `_reset_from_warmstart_cache`(pour_right_env.py L3301~)가 뱅크에서 세우는 것과
    같은 버퍼들을, 텔레포트 없이 지금 씬의 실측/지령으로 채운다. 항목별 대응은
    그 함수 본문과 1:1 이다 — 여기 로직이 아니라 **값의 출처**만 다르다.

    ★뱅크 규약 그대로: 물리 상태는 실측, hold 목표는 **지령**(파지력의 원천).
      palm 도 지령(`right_shim.palm_targets` — 파지 종료 시 팔이 유지 중인 목표)을
      pour 형식(pos3+quat_xyzw)으로 변환해 넣는다. 실측을 넣으면 파지력이 사라진다.
    ★좌팔: pour 는 매 스텝 `left_arm_zero_pos` 를 좌 9관절(그리퍼 포함!)에 명령한다.
      그리퍼 원본값은 0.044(열림) — 그대로 두면 **쥔 컵을 놓는다**. 현재 홀드
      지령으로 교체한다.
    """
    robot = env.robot
    q = robot.data.joint_pos
    arm = q[:, env.arm_dof_indices].clone()
    hand_meas = q[:, env.hand_dof_indices].clone()
    hand_cmd = robot.data.joint_pos_target[:, env.hand_dof_indices].clone()

    n_arm = arm.shape[1]
    env.fabric_q.zero_()
    env.fabric_q[:, :n_arm] = arm
    env.fabric_q[:, n_arm:] = hand_meas
    env.fabric_qd.zero_()
    env.fabric_qdd.zero_()
    env.pregrasp_arm_pos_buf[:] = arm
    env.grasp_hold_hand_pos_buf[:] = hand_cmd
    env.hand_joint_targets[:] = hand_cmd
    env.open_tesollo_fabric.default_config[:, :n_arm] = arm

    # palm 기준: **램프 후 실측** — ⑤′ 우 FK 세팅 램프가 파지 종료 지령을 무의미하게
    # 만들었다(팔이 뱅크 mean 자세로 이동). 정지 정착 상태라 실측≈지령이고, pour 의
    # palm_pose_targets 는 회전 액션의 기준점이므로 실제 자세가 정확한 원점이다.
    p6 = right_shim._palm_pose_6d().clone()
    pose7 = torch.zeros(env.num_envs, 7, device=env.device)
    pose7[:, :3] = torch.max(
        torch.min(p6[:, :3], env.palm_maxs[:3].unsqueeze(0)),
        env.palm_mins[:3].unsqueeze(0))
    pose7[:, 3:7] = env._quat_xyzw_from_euler_zyx(p6[:, 3:6])
    env.palm_pose_targets[:] = pose7
    env.pregrasp_palm_pose_buf[:] = pose7

    cup_local = env.cup.data.root_pos_w - env.scene.env_origins
    palm_local = pose7[:, :3]
    env.object_init_pos[:] = cup_local
    env.object_init_pos[:, 2] = float(env.cfg.object_spawn_z)
    env._grasp_rel_palm_to_cup_init[:] = cup_local - palm_local
    env._grasp_cup_height_init[:] = cup_local[:, 2]

    # 받는 컵: 훈련의 FK 상수 대신 **진짜 쥔 컵의 현재 자세** — obs·조준이 이걸 본다
    live_left = torch.cat([env.left_target_cup.data.root_pos_w,
                           env.left_target_cup.data.root_quat_w], dim=-1)
    env._left_target_cup_fixed_pose_w[:] = live_left

    # 좌팔 상시 명령 = 현재 홀드 지령 (REST7 도착 뒤라 팔은 동일, 그리퍼만 교체 효과)
    left_hold = robot.data.joint_pos_target[:, env.left_arm_dof_indices].clone()
    env.left_arm_zero_pos[:] = left_hold

    # ★국면별 플랜트 게인 — pour 정책은 자기 학습 게인의 팔을 전제한다
    #   (각 국면의 플랜트 = 그 국면 정책의 학습 플랜트). 파지 국면은 우 dump 프로필
    #   (g1=r2s)로 돌았으므로, 여기서 **pour dump 의 우팔 게인**(e1_pour1=kuka
    #   300/45…25/15)으로 전환해 sim 에 즉시 기입한다.
    _rg = _expand_right_gains(pour_actuators)
    _kp = {n: _rg[n][0] for n in _rg}
    _kd = {n: _rg[n][1] for n in _rg}
    _names = [f"r_aj_{i}" for i in range(1, 8)]
    _rids = [robot.joint_names.index(n) for n in _names]
    robot.write_joint_stiffness_to_sim(
        torch.tensor([[float(_kp[n]) for n in _names]], device=env.device), joint_ids=_rids)
    robot.write_joint_damping_to_sim(
        torch.tensor([[float(_kd[n]) for n in _names]], device=env.device), joint_ids=_rids)
    print(f"[pour init] 우팔 게인 → pour 학습값 kp {[round(float(_kp[n])) for n in _names]}",
          flush=True)

    ids = torch.arange(env.num_envs, device=env.device)
    # ★기하 버퍼 즉시 갱신 — 원본은 rewards/dones 첫머리에서만 갱신하므로, 여기서 안
    #   부르면 **첫 관측이 부팅 시점의 낡은 pour_point/opening/축**을 본다(LSTM 오염).
    env._compute_intermediate_values()
    env._hide_beads(ids)
    env._beads_spawned[:] = False          # → _pre_physics_step 이 hold 뒤 자동 소환
    env._clear_episode_buffers(ids)
    env.episode_length_buf[:] = 0
    env.actions.zero_()
    env.prev_actions.zero_()
    print(f"[pour init] palm 지령 {[round(float(v), 3) for v in pose7[0, :3]]} · "
          f"컵 {[round(float(v), 3) for v in cup_local[0]]} · "
          f"받는컵 {[round(float(v), 3) for v in live_left[0, :3]]} · "
          f"구슬 {int(env.num_beads)}개 {int(env.cfg.episode_hold_steps)}스텝 후 자동 소환",
          flush=True)


def disarm_receiver_pin(env) -> None:
    """pour `_apply_action` 의 받는 컵 고정핀을 **원천 무효화** — ⑥에서 1회 호출.

    원본은 받는 컵(훈련에선 kinematic 마커)에 매 스텝 (고정 pose, 속도 0)을 박는다.
    여기선 진짜 쥔 컵이다. 1차 시도(라이브 상태 되쓰기)는 반스텝 지연 상태 주입이
    돼 컵이 그리퍼에서 떨려 빠져나갔다(09.02 렌더 실측 — 미러 사고의 축소판).
    write 메서드 자체를 인스턴스 수준 no-op 으로 갈아끼우면 컵은 **순수 물리**다.
    """
    cup = env.left_target_cup
    cup.write_root_pose_to_sim = lambda *a, **k: None
    cup.write_root_velocity_to_sim = lambda *a, **k: None
    print("[pour] 받는컵 고정핀 무장해제 — 순수 물리 (좌손이 쥔다)", flush=True)


def refresh_receiver_buffer(env) -> None:
    """obs·조준이 읽는 받는컵 버퍼를 라이브 실측으로 — 매 정책스텝 호출."""
    cup = env.left_target_cup
    env._left_target_cup_fixed_pose_w[:] = torch.cat(
        [cup.data.root_pos_w, cup.data.root_quat_w], dim=-1)
