"""grasp_lift_fabric — Fabrics 팔 + 절대 손 액션 (direct).

로봇 종속 정보는 전부 `modules.robots.RobotProfile` 이 공급한다 — 이 파일에
조인트/바디 이름 하드코딩 금지(계약 테스트가 소스 grep 으로 강제한다).

제어 스택:
  팔  = Geometric Fabrics. 정책이 **절대 palm 6D pose**(워크스페이스 박스 안)를 내면
        Fabrics 가 충돌회피·관절한계·매끄러운 궤적을 보장하며 관절 목표를 만든다.
        같은 Fabrics 를 실기에서도 돌리므로 sim/real 의 팔 거동이 같은 함수가 된다.
  손  = 관절 **절대** 목표(전범위). 적분기·래치·커플링·시너지·PCA 없음.

★obs 에 `fabric_q`/`fabric_qd` 를 넣지 않는다 — 실기에 없는 내부 상태이고,
  grasp_v1 이 s2r 불가였던 직접 원인이다.
"""

from __future__ import annotations

import math
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from openarm.agnostic.modules import adr as _adr
from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robots as _rb

from . import grasp_lift_fabric_env_cfg as _cfg
from .grasp_lift_fabric_env_cfg import GraspLiftFabricEnvCfg
# ★08.23 보상을 자매 트랙 grasp_sensor 와 **공유**한다(사용자 결정: "리워드 구조는
#   grasp_sensor 쪽으로 모두 바꾸기"). 복사가 아니라 import 다 — 복사하면 두 트랙이
#   다시 갈라지고, 한쪽에서 고친 결함이 다른 쪽에 남는다(lift 분모 0.10 이 그랬다).
from ..grasp_sensor.rewards import compute_grasp_sensor_rewards, contact_gate
from .rewards_tip import envelope_fraction_graded

# Fabrics 경로 (hdgp/source/FABRICS/src 우선)
_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                 "FABRICS", "src")
)
if os.path.isdir(_FABRICS_SRC) and _FABRICS_SRC not in sys.path:
    sys.path.insert(0, _FABRICS_SRC)

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh   # noqa: E402
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tes  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp                   # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel      # noqa: E402

_DEG = math.pi / 180.0


def _fabric_class(name: str):
    for mod in (_fab_tes, _fab_rh):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Fabrics 클래스 '{name}' 를 찾을 수 없다")


def _kuka_absolute(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """DEXTRAH `compute_absolute_action` 과 동일한 절대 박스 매핑.

    ``scale(a, lo, hi) = 0.5·(hi−lo)·a + 0.5·(hi+lo)`` 뒤 박스로 clamp.
    a=−1 → lo · a=0 → **박스 중앙** · a=+1 → hi. 방향 대칭이라 같은 액션 크기가
    부호와 무관하게 같은 이동을 낸다(구 "홈 기준 구간별 선형"은 비대칭이었다).
    """
    return (0.5 * (hi - lo) * a + 0.5 * (hi + lo)).clamp(lo, hi)


class GraspLiftFabricEnv(DirectRLEnv):
    cfg: GraspLiftFabricEnvCfg

    # ==================================================================
    def __init__(self, cfg: GraspLiftFabricEnvCfg, render_mode: str | None = None, **kw):
        # ★hydra CLI 오버라이드(`env.object_bank=...` 등)는 cfg 필드만 덮어쓰고
        #   __post_init__ 을 다시 돌리지 않는다. 파생값(자산 cfg·차원·replicate_physics)을
        #   여기서 맞추지 않으면 "cup_family 라고 적혀 있는데 컵 하나만 스폰" 같은
        #   조용히 틀린 조합으로 학습이 돈다. resolve_cfg 는 멱등이다.
        _cfg.resolve_cfg(cfg)
        self.profile = _rb.get(cfg.profile_name)
        self.bank = _ob.get(cfg.object_bank, expected_size=cfg.object_bank_expected_size)
        # ★super().__init__ 안에서 _apply_action 이 불릴 수 있으므로 **먼저** 정한다.
        #   중력이 꺼져 있으면 보상은 무의미하므로 0 으로 잠근다(이중 부정 방지).
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        # ★★부팅 가드: cfg **필드**와 파생 robot_cfg 가 실제로 일치하는지 확인한다.
        #   `params/env.yaml` 덤프는 resolve_cfg **이전** 상태라 필드만 보고 판단하면
        #   조용히 틀린 물리로 학습이 돈다(08.22 실측: probe 가 중력 False 를 찍었는데
        #   USD 는 True 였다. 같은 계열 결함으로 fab_test1~4 를 통째로 날린 적 있다).
        _sp = cfg.robot_cfg.spawn
        _gr_off = bool(_sp.rigid_props.disable_gravity)
        _sc_on = bool(_sp.articulation_props.enabled_self_collisions)
        if _gr_off == bool(cfg.enable_gravity) or _sc_on != bool(cfg.enable_self_collisions):
            raise RuntimeError(
                "물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로를 확인할 것.\n"
                f"  enable_gravity={cfg.enable_gravity} 인데 spawn.disable_gravity={_gr_off}\n"
                f"  enable_self_collisions={cfg.enable_self_collisions} 인데 "
                f"spawn.enabled_self_collisions={_sc_on}")
        print(f"[grasp_lift_fabric] 물리: self_collisions={_sc_on} · gravity={not _gr_off}"
              f" · grav_comp={self._grav_comp}", flush=True)
        super().__init__(cfg, render_mode, **kw)
        p = self.profile

        # ---- 조인트/바디 해석 (fail-loud) --------------------------------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: "
                f"arm {len(self.arm_ids)}!={p.num_arm_joints} ({arm_names}), "
                f"hand {len(self.hand_ids)}!={p.num_hand_joints} ({hand_names})"
            )
        self._arm_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self._hand_t = torch.tensor(self.hand_ids, device=self.device, dtype=torch.long)

        # ★외전 관절은 정책이 제어하지 않고 init 값에 고정한다 — 손가락 교차를
        #   자유도 수준에서 없앤다(self-collision 을 끈 채 두는 대신).
        # ★트랙 전용 고정 관절 오버라이드(08.26 사용자 지시: _1 전부 + 소지 _2).
        #   프로필 상수는 자매 공유라 여기서만 덮는다. "{side}" 는 프로필 관절
        #   이름 규약을 그대로 따라 치환한다(로봇 종속 이름을 태스크에 안 박는다).
        _fr_ovr = getattr(self.cfg, "frozen_hand_joints_override", None)
        if _fr_ovr is not None:
            _side = p.name.split("_")[-1][0]          # bis_right → 'r'
            _frozen_names = tuple(j.replace("{side}", _side) for j in _fr_ovr)
            print(f"[grasp_lift_fabric] 고정 관절 오버라이드 {len(_frozen_names)}개: "
                  f"{[n.split('hj_')[-1] for n in _frozen_names]}", flush=True)
        else:
            _frozen_names = p.frozen_hand_joints
        _frozen = []
        for jn in _frozen_names:
            ids, _ = self.robot.find_joints(jn)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] 고정 관절 '{jn}' 해석 실패: {ids}")
            _frozen.append(ids[0])
        _fset = set(_frozen)
        self._frozen_t = torch.tensor(_frozen, device=self.device, dtype=torch.long)
        self._hand_free_t = torch.tensor(
            [i for i in self.hand_ids if i not in _fset], device=self.device, dtype=torch.long)
        n_free = len(self._hand_free_t)
        # ★tip IK 모드에서는 손 액션이 관절이 아니라 손끝 5점 × xyz 다. frozen_hand_joints
        #   는 이 모드에서 **적용되지 않는다** — fabric 이 손 20-DOF 를 전부 소유하므로
        #   일부만 얼리면 fabric 이 아는 자세와 실제가 어긋난다(그게 바로 고치려는 결함).
        # 손 제어 = 풀 관절(fabric direct) 단일 경로. tip IK 는 08.26 폐기 —
        # 5지 손끝 15D 독립 지시는 기구학적으로 성립하지 않았다(지령↔실제 111mm).
        self._tip_ik = False
        # 손 20-DOF 를 Fabrics 가 소유하되 **액션 의미는 관절 그대로**.
        # fabric 이 손을 알아야 body_repulsion 에 손가락↔손가락 쌍을 넣을 수 있고,
        # 그래야 PhysX self-collision 을 끌 근거가 생긴다(스텝 시간의 55~64%).
        self._hand_fabric = True
        _n_hand_act = n_free
        if 6 + _n_hand_act != self.cfg.action_space:
            raise RuntimeError(
                f"[{p.name}] 액션 차원 불일치: 6+{_n_hand_act} != {self.cfg.action_space}. "
                "use_tip_fabric / frozen_hand_joints 와 cfg 파생이 어긋났다."
            )
        # ★손가락 상호 겹침을 **학습 중에** 잰다. 인위적 probe(컵과 무관하게 강제 폐합)는
        #   정책 거동이 아니라 100% 겹침으로 나온다 — 실제로 그랬다. in-situ 로 봐야 한다.
        _lnk, _own = [], []
        for _fi, _f in enumerate(p.fingers):
            for _b in p.finger_tip_bodies[_f] + p.finger_wrap_bodies.get(_f, ()):
                _ids, _ = self.robot.find_bodies(_b)
                _lnk.append(_ids[0]); _own.append(_fi)
        self._fl_t = torch.tensor(_lnk, device=self.device, dtype=torch.long)
        _o = torch.tensor(_own, device=self.device)
        self._fl_diff = (_o[:, None] != _o[None, :])      # 다른 손가락 쌍만

        print(f"[grasp_lift_fabric] 손 관절 {len(self.hand_ids)} 중 "
              f"{len(_frozen)}개 고정 → 정책 제어 {n_free}", flush=True)

        self.palm_idx = self._one_body(p.palm_body)
        # 가용 손가락(hand_unusable_fingers 제외) — 자매 `_usable_idx` 규약.
        # ★자매는 pinky 를 뺀다(자세표 lerp 에서 상수가 되기 때문). 우리는 손끝 IK 로
        #   pinky 를 직접 지시하고 실측 wrap 0.37 로 5 지 중 가장 높아 빈 집합이 기본이다.
        _unusable = set(getattr(self.cfg, "hand_unusable_fingers", ()) or ())
        _bad = _unusable - set(p.fingers)
        if _bad:
            raise RuntimeError(
                f"[{p.name}] hand_unusable_fingers 에 없는 손가락: {sorted(_bad)}")
        self._usable_t = torch.tensor(
            [i for i, f in enumerate(p.fingers) if f not in _unusable],
            device=self.device, dtype=torch.long)
        if len(self._usable_t) < 2:
            raise RuntimeError(
                f"[{p.name}] 가용 손가락이 {len(self._usable_t)}개 — 대향 파지 불가")
        # 자매와 같은 분모 집합 둘 추가(08.26 동일 세팅):
        #   _grp_b_env_t — wrap4 분모 = group_b ∩ envelope_fingers
        #   _wrap_t      — deep 분모  = group_b ∩ envelope ∩ 가용
        _env_f = set(p.envelope_fingers or p.fingers)
        self._grp_b_env_t = torch.tensor(
            [i for i, f in enumerate(p.fingers)
             if f in p.contact_group_b and f in _env_f],
            device=self.device, dtype=torch.long)
        # ★이름 주의 — `self._wrap_t` 는 이 env 에 **이미 있다**(마디 바디 인덱스
        #   (F,P), 값 59~73). 같은 이름을 썼다가 `_deep_all[:, (5,2)텐서]` 가 CUDA
        #   index OOB 로 터졌다(첫 스텝 즉사, 비동기 어서트라 엉뚱한 줄이 찍힘).
        #   손가락 인덱스 집합은 `_wrap_f_t` 로 가른다.
        self._wrap_f_t = torch.tensor(
            [i for i, f in enumerate(p.fingers)
             if f in p.contact_group_b and f in _env_f and f not in _unusable],
            device=self.device, dtype=torch.long)
        if len(self._grp_b_env_t) == 0 or len(self._wrap_f_t) == 0:
            raise RuntimeError(
                f"[{p.name}] wrap 분모가 비었다 — contact_group_b ∩ envelope_fingers")
        # 자매 보상 상태 버퍼(08.26): deep 연속 지속 · 코리더 위반 이력(래치).
        self._persist_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._corridor_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        # 단계 hit(에피소드 누적, 리셋 때 기록) + 정지 유지 연속 카운터 — 자매 규약.
        # ★순간 게이트 평균과 다르다: "이 에피소드에서 그 단계가 **한 번이라도**
        #   열렸는가"라서 자매 hier_test2 의 stage/approach 1.0 과 직접 비교된다.
        self._stage_hit = torch.zeros(
            self.num_envs, 5, dtype=torch.bool, device=self.device)
        self._stay_run = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        if True and (
                float(self.cfg.stage_gate_contact_n) > len(self._usable_t)):
            raise RuntimeError(
                f"[{p.name}] μ 게이트 임계 {self.cfg.stage_gate_contact_n} > 가용 손가락 "
                f"{len(self._usable_t)} — 계층이 영원히 안 열린다")

        # 자매 부팅 fail-loud(08.26 동일 세팅) — 코리더·stay 상수 정합.
        if True:
            _c = self.cfg
            if float(_c.stage_corridor_xy_m[0]) < float(_c.stage_corridor_xy_m[1]):
                raise RuntimeError(f"corridor_xy {_c.stage_corridor_xy_m} — 조여지는 방향이어야 한다")
            if float(_c.stage_corridor_tilt_deg[0]) < float(_c.stage_corridor_tilt_deg[1]):
                raise RuntimeError(f"corridor_tilt {_c.stage_corridor_tilt_deg} — 조여지는 방향이어야 한다")
            if float(_c.stage_corridor_tilt_deg[1]) < float(_c.stage_succ_tilt_band_deg[0]):
                raise RuntimeError(
                    f"만렙 tilt 코리더({_c.stage_corridor_tilt_deg[1]}°)가 성공 tilt 밴드 "
                    f"하한({_c.stage_succ_tilt_band_deg[0]}°)보다 좁다 — 성공 자체가 몰수된다")
            if float(_c.stage_corridor_xy_m[1]) < float(_c.stage_stay_pos_tol_m):
                raise RuntimeError(
                    f"만렙 xy 코리더({_c.stage_corridor_xy_m[1]})가 stay 판정 반경"
                    f"({_c.stage_stay_pos_tol_m})보다 좁다")        # ── 손끝 목표 마커(반지름 1cm 구) — 정책이 **어디를 지시하는지** 본다 ─────
        # ★headless 에서는 만들지 않는다(렌더 대상이 없고 prim 만 늘어난다).
        # ★손가락별 색을 달리해 어느 손가락 목표인지 구분한다. 이건 fabric 에 넘기는
        #   **지령**이지 실제 손끝 위치가 아니다 — 둘이 벌어지면 그게 곧 추종오차다
        #   (`tip/target_err_mm` 이 같은 것을 수치로 잰다).
        self._tip_markers = None
        # ★★08.26 사용자 요청 — 영상(play --video)에도 액션 cmd 마커가 보여야 한다.
        #   구 조건은 has_gui() 뿐이라 headless 녹화에서 마커가 빠졌다. 카메라 렌더
        #   여부는 carb 설정(/isaaclab/cameras_enabled — AppLauncher 가 세팅)으로 본다.
        try:
            import carb
            _cams = bool(carb.settings.get_settings().get("/isaaclab/cameras_enabled"))
        except Exception:
            _cams = False
        if bool(getattr(self.cfg, "enable_tip_markers", False)) and (
                self.sim.has_gui() or _cams):
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            _r = float(getattr(self.cfg, "tip_marker_radius", 0.01))
            _cols = ((1.0, 0.25, 0.25), (1.0, 0.65, 0.1), (0.3, 0.9, 0.3),
                     (0.3, 0.5, 1.0), (0.8, 0.3, 0.9))
            _mk = {
                f: sim_utils.SphereCfg(
                    radius=_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_cols[i % len(_cols)]))
                for i, f in enumerate(p.fingers)
            }
            # palm **지령**(slew 후 palm_cmd) — 흰색 큰 구. 팔이 어디로 가라고
            # 지시받는지와 실제 palm 의 간격이 곧 추종 상태다.
            _mk["palm_cmd"] = sim_utils.SphereCfg(
                radius=_r * 1.5,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0)))
            self._tip_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/GraspLiftFabTipTargets", markers=_mk))
            # env0 만 그리므로 손가락 수 + palm 1 이면 된다.
            self._tip_marker_idx = torch.arange(
                len(p.fingers) + 1, device=self.device)
            print(f"[grasp_lift_fabric] 액션 cmd 마커 {len(p.fingers)}색+palm(흰) · "
                  f"반지름 {_r*1000:.0f}mm · env0 전용 · "
                  f"{'GUI' if self.sim.has_gui() else '카메라 녹화'}", flush=True)

        # ── GUI 카메라를 env0 정면으로 ──────────────────────────────────────
        # ★기본 뷰는 씬 전체(2048 env)를 잡아 확인용으로 쓸모가 없고 렌더 비용만 크다.
        #   env0 를 정면에서 보게 두면 나머지 env 가 frustum 밖으로 빠진다.
        if bool(getattr(self.cfg, "gui_focus_env0", False)) and self.sim.has_gui():
            _o0 = self.scene.env_origins[0].tolist()
            _eye = [a + b for a, b in zip(self.cfg.gui_camera_eye, _o0)]
            _tgt = [a + b for a, b in zip(self.cfg.gui_camera_target, _o0)]
            self.sim.set_camera_view(eye=_eye, target=_tgt)
            print(f"[grasp_lift_fabric] GUI 카메라 → env0 정면 "
                  f"eye{tuple(round(v, 2) for v in _eye)} "
                  f"target{tuple(round(v, 2) for v in _tgt)}", flush=True)

        # 접근 중 손가락 고정 — env 별 래치(히스테리시스). 리셋에서 False 로 돌린다.
        self._tip_t = torch.tensor(
            [self._one_body(b) for b in p.fingertip_bodies],
            device=self.device, dtype=torch.long,
        )
        # ★★손의 TCP = `palm_ee`(사용자 확인: **+x 축이 손바닥 법선**). palm 원점은
        #   손목 쪽이라 접근 기준점이 아니다. 자산 URDF 실측 palm+(28,0,40)mm.
        #   ★위치만으로는 정렬을 못 본다 — palm_ee 의 rpy 가 0 이라 회전이 palm 과 같고,
        #     오프셋 방향 (0.028,0,0.04) 은 법선과 다르다. 그래서 obs 에 **자세**를
        #     함께 넣는다(회전행렬 x·z 열 = 6D 연속 표현, y 는 외적으로 복원 가능).
        _ee = p.palm_ee_body or p.palm_body
        self._tcp_idx = self._one_body(_ee)
        if p.palm_ee_body is None:
            print(f"[grasp_lift_fabric] ⚠ 프로필에 palm_ee_body 가 없어 palm 원점을 "
                  f"TCP 로 쓴다 — 접근 정렬 관측이 부정확해진다", flush=True)

        # ---- 감쌈 판정 손바닥면 필터 + grip 항 입력 (08.24) --------------------------
        # wrap 마디 body 인덱스 (F, P). 손가락마다 마디 수가 같아야 텐서가 성립한다.
        _wraps = [p.finger_wrap_bodies.get(f, ()) for f in p.fingers]
        _np = len(_wraps[0])
        if any(len(w) != _np for w in _wraps):
            raise RuntimeError(f"[{p.name}] finger_wrap_bodies 마디 수 불일치: "
                               f"{[len(w) for w in _wraps]}")
        self._wrap_t = torch.tensor(
            [[self._one_body(b) for b in w] for w in _wraps],
            device=self.device, dtype=torch.long,
        )                                                   # (F, P)
        self._palmar_ax = None    # palmar 필터는 tip IK 폐기와 함께 제거

        # ---- palm 워크스페이스 박스 (액션 [-1,1] → 절대 pose) --------------------
        # euler_zyx 중심: 우팔 [90,0,90]°, 좌팔은 부호 반전. ±max_pose_angle.
        a = float(cfg.max_pose_angle_deg)
        sign = 1.0 if p.side == "r" else -1.0
        centre = (sign * 90.0, 0.0, sign * 90.0)
        # ★박스는 **프로필**이 준다(팔 도달범위 = 로봇 종속 정보).
        #   태스크 cfg 상수로 두면 자산이 바뀔 때 조용히 틀린다 — 실제로 그랬다.
        #   좌우 미러는 프로필이 이미 반영하고 있으므로 여기서 뒤집지 않는다.
        lo = list(p.palm_box_min) + [(c - a) * _DEG for c in centre]
        hi = list(p.palm_box_max) + [(c + a) * _DEG for c in centre]
        if not p.palm_box_verified:
            print(f"[grasp_lift_fabric] ⚠ 프로필 '{p.name}' 의 palm 박스는 **미실측**이다"
                  " — probe_workspace_reach.py 로 확인할 것(다른 로봇 값을 물려받았을 수 있다).",
                  flush=True)
        # ★트랙 전용 z바닥 완화(08.26) — 프로필 박스는 자매 공유라 여기서만 낮춘다.
        _zmin_ovr = getattr(self.cfg, "palm_box_z_min_override", None)
        if _zmin_ovr is not None:
            if float(_zmin_ovr) > lo[2]:
                raise RuntimeError(
                    f"palm_box_z_min_override({_zmin_ovr}) 가 프로필 바닥({lo[2]})보다 "
                    "높다 — 이 오버라이드는 **완화** 전용이다")
            print(f"[grasp_lift_fabric] palm 박스 z바닥 {lo[2]:.3f} → "
                  f"{float(_zmin_ovr):.3f} (트랙 오버라이드)", flush=True)
            lo[2] = float(_zmin_ovr)
        self.palm_lo = torch.tensor(lo, device=self.device).unsqueeze(0)
        self.palm_hi = torch.tensor(hi, device=self.device).unsqueeze(0)

        # ---- 홈 자세 트랙 오버라이드 (fabric 구성 **전**에 적용) --------------------
        # ★순서가 중요하다: `_setup_fabrics` 가 fabric.default_config 를 프로필 홈으로
        #   교체하므로, 그 전에 덮어야 articulation·fabric cspace rest·리셋 q0 가
        #   한 값으로 일치한다. 뒤에서 덮으면 fabric 만 옛 홈을 쥔 채 어긋난다.
        _home_ovr = getattr(self.cfg, "hand_home_override", None)
        if _home_ovr:
            _jn = self.robot.data.joint_names
            _dj = self.robot.data.default_joint_pos
            for _nm_t, _val in _home_ovr:
                _nm = _nm_t.replace("{side}", p.side)
                if _nm not in _jn:
                    raise RuntimeError(
                        f"hand_home_override 관절 '{_nm}' 이 로봇에 없다 "
                        f"— 자산/프로필이 바뀌었으면 이름을 재확인할 것")
                _j = _jn.index(_nm)
                _before = float(_dj[0, _j])
                _dj[:, _j] = float(_val)
                print(f"[grasp_lift_fabric] 홈 오버라이드 {_nm} "
                      f"{_before:+.3f} → {float(_val):+.3f} rad (트랙 전용)", flush=True)

        # ---- Fabrics -------------------------------------------------------------
        self._setup_fabrics()

        # ---- 홈 palm pose 실측 (액션 매핑의 기준점) --------------------------------
        # ★액션이 절대값이라 "가만히 있기"에 해당하는 값이 필요하다. 박스 중심으로
        #   두면 정책 초기 출력(≈0)이 곧 홈에서 0.29m 떨어진 곳으로의 돌진이 된다
        #   (probe 실측 0.29m). a=0 → 홈 이 되도록 구간별 선형으로 매핑한다.
        # ★★__init__ 시점의 body_pos_w 는 **stale** 이다(로봇이 아직 홈에 배치되기 전).
        #   그대로 읽으면 [0.002, -0.139, 0.187] 같은 미배치 자세가 나온다.
        #   홈을 물리로 확정한 뒤 읽는다 — 이 저장소에서 반복된 버퍼 stale 함정.
        _q_home = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(_q_home, torch.zeros_like(_q_home))
        self.robot.set_joint_position_target(_q_home)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.home_palm = self._palm_pose_6d()
        _in_box = ((self.home_palm >= self.palm_lo) & (self.home_palm <= self.palm_hi)).all()
        if not bool(_in_box):
            raise RuntimeError(
                f"[{p.name}] 홈 palm pose 가 워크스페이스 박스 밖이다.\n"
                f"  홈   {[round(v, 3) for v in self.home_palm[0].tolist()]}\n"
                f"  박스 lo {[round(v, 3) for v in self.palm_lo[0].tolist()]}\n"
                f"       hi {[round(v, 3) for v in self.palm_hi[0].tolist()]}"
            )

        # ---- 관절 한계 / 버퍼 ------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits          # (N, J, 2)
        m = float(cfg.hand_limit_margin)
        self._hand_lo = jl[:, self._hand_free_t, 0] + m
        self._hand_hi = jl[:, self._hand_free_t, 1] - m
        self._arm_lo = jl[:, self._arm_t, 0]
        self._arm_hi = jl[:, self._arm_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()


        # ---- A: 지령 속도 제한 ----------------------------------------------------
        _sp = float(cfg.palm_slew_pos)
        _sr = float(cfg.palm_slew_rot_deg) * math.pi / 180.0
        self._slew = torch.tensor([_sp, _sp, _sp, _sr, _sr, _sr], device=self.device)
        self._slew_on = _sp > 0.0 or _sr > 0.0
        # 지령 버퍼. 리셋 시 홈으로 되돌린다(안 되돌리면 이전 에피소드 지령이 샌다).
        self.palm_cmd = self.home_palm.clone()
        self._prev_cmd = self.home_palm.clone()

        A = cfg.action_space
        self.actions = torch.zeros(self.num_envs, A, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, A, device=self.device)
        # 자유 관절만 정책이 움직인다. 고정 관절은 default(init) 값 그대로 명령된다.
        # tip IK 모드에서는 손 **전체**가 fabric 산출물이라 크기가 다르다.
        self.hand_targets = self._default_q[
            :, self._hand_t].clone()
        # 손 PD 속도 피드포워드용 — 지령 램프의 도함수(자매 트랙 `_syn_vel` 규약).
        self._policy_dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        self._hand_vel = torch.zeros_like(self.hand_targets)
        self.palm_targets = torch.zeros(self.num_envs, 6, device=self.device)

        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- 접촉 그룹 인덱스 -------------------------------------------------------
        fingers = list(p.fingers)
        self._fingers = fingers
        # ★대향 파지점 approach 기하는 08.23 부터 grasp_sensor.rewards 안에서 계산한다.
        #   그쪽은 손끝 인덱스로 접촉 그룹 인덱스(_grp_a/_grp_b_env)를 **그대로** 쓰므로
        #   fingertip_bodies 와 fingers 의 순서가 같아야 성립한다 — fail-loud 로 강제한다.
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"fingers({len(fingers)}) 길이가 다르다 — 접근 보상의 그룹 인덱스가 어긋난다")
        # ★B. persistence — 대향 게이트 연속 유지 스텝 수
        self._gate_hold = torch.zeros(self.num_envs, device=self.device)
        self._grp_a = torch.tensor([fingers.index(f) for f in p.contact_group_a],
                                   device=self.device, dtype=torch.long)
        self._grp_b = torch.tensor([fingers.index(f) for f in p.contact_group_b],
                                   device=self.device, dtype=torch.long)
        # 인벨롭 손가락(감쌈 분모·d_side 의 wrap 그룹) — 프로필이 정의한다.
        # ★pinky 는 밑동 굴곡이 기본 자세에 없다 — _1 을 60° 로 열어 _2 를 굴곡축으로
        #   돌려놔야 분모 5 가 성립한다(robots.py envelope_fingers 주석).
        if not p.envelope_fingers:
            raise RuntimeError(f"[{p.name}] envelope_fingers 미정의 — 인벨롭 보상 성립 불가")
        self._env_f = torch.tensor([fingers.index(f) for f in p.envelope_fingers],
                                   device=self.device, dtype=torch.long)
        self._grp_b_env = torch.tensor(
            [fingers.index(f) for f in p.contact_group_b if f in p.envelope_fingers],
            device=self.device, dtype=torch.long)
        if len(self._grp_b_env) == 0:
            raise RuntimeError(f"[{p.name}] contact_group_b ∩ envelope_fingers 가 비었다")

        # ---- 물체 배정 / onehot ------------------------------------------------------
        self.object_idx = torch.tensor(self.bank.assign_indices(self.num_envs),
                                       device=self.device, dtype=torch.long)
        # ★물체별 "작업면에 놓인" z = 작업면 상면 + 그 물체의 원점 오프셋.
        #   합쳐서 상수 하나로 두면 물체가 바뀔 때 조용히 틀린 높이가 되고,
        #   그 값이 곧 lift 보상의 **기준선**이라 보상이 통째로 오염된다
        #   (실측: 구 설정은 스폰 0.302 vs 안착 0.2773 → height_delta 가 -24.7mm 에서 시작).
        _rest = torch.tensor(
            [s.origin_offset_z for s in self.bank.specs], device=self.device
        ) + float(p.surface_z)
        self._object_rest_z = _rest[self.object_idx]
        self._onehot = None
        if cfg.enable_object_onehot:
            self._onehot = torch.nn.functional.one_hot(
                self.object_idx, num_classes=self.bank.onehot_dim).float()
        # ★★KUKA 고정(08.25) 물체 크기 obs — 원본 teacher 관측의 `object_scale`.
        #   onehot 은 "몇 번 물체인가"만 알려 주고 **크기 자체는 못 알려 준다**.
        #   같은 형상을 스케일만 바꿔 쓰는 뱅크에서는 onehot 이 커질수록 낭비이고,
        #   미학습 크기로 일반화도 안 된다. 스케일은 실기에서도 비전으로 추정 가능하다.
        _scl = torch.tensor([list(s.scale) for s in self.bank.specs],
                            device=self.device, dtype=torch.float32)   # (S,3)
        self._object_scale = _scl[self.object_idx]
        # per-object 파지 표면(측면 원통 띠) — tip_bridge 목표면. 스칼라 하드코딩은
        # 다물체(스케일 0.85~1.30 + shaker)에서 최대 ±30% 어긋나 폐기(08.26).
        _gr = torch.tensor([sp.grasp_radius_m for sp in self.bank.specs],
                           device=self.device)
        _gh = torch.tensor([sp.grasp_halfheight_m for sp in self.bank.specs],
                           device=self.device)
        self._grasp_radius = _gr[self.object_idx]          # (N,)
        self._grasp_halfheight = _gh[self.object_idx]      # (N,)

        # ── 역순 커리큘럼(08.27 재구현): **팔이 아니라 컵을 옮긴다**.
        #   ★★grasp_v1 이 하는 방식이 이것이다(사용자가 두 번 근거로 든 선례):
        #       obj_xy = palm_xy − pregrasp_offset + noise ,  obj_z = 테이블 높이
        #     (tesollo/right/grasp_v1/demo_grasp_reset.py: compute_demo_cup_spawn_local)
        #   이전 판은 반대로 **팔을 컵 옆으로 IK 텔레포트**했고, 그 IK 하나가
        #   h7 우팔 데드락(단일 해 → 나쁜 시드 → 2172ep 동결)과 play 경로 IK 실패
        #   (잔차 300mm)를 둘 다 만들었다. 컵을 옮기면 IK 가 통째로 사라진다.
        #   시작 자세는 항상 홈이므로 시작 다양성이 성공 지표에 게이팅되지 않는다.
        self._cup_near_xy = self._home_grasp_center_xy()

        # ---- ADR (축 둘: 스폰 반경 · goal 오프셋 반경) --------------------------------
        # goal 축은 initial 0 이라 반경 0 = 구 고정 goal 과 동치로 시작한다.
        # ★★KUKA 고정(08.25): 원본 ADR 축을 전부 연결한다. 시작값만 맞추고 끝으로 가는
        #   경로가 없으면 커리큘럼이 아니라 고정값이다(보상 가중치 축은 지시로 제외).
        self.adr = _adr.TaskADR(
            {"reset_near": {"frac": (0.0, 1.0)},   # 0=컵 옆 시작 → 1=홈(만렙=배포 분포)
             "spawn": {"xy_range": (cfg.spawn_range_initial, cfg.spawn_range_final),
                       "rotation": (0.0, cfg.adr_object_rotation_final)},
             "goal": {"xy_radius": (cfg.goal_xy_radius_initial, cfg.goal_xy_radius_final),
                      "z_radius": (cfg.goal_z_radius_initial, cfg.goal_z_radius_final)},
             "pd_targets": {"velocity_target_factor": (
                 cfg.velocity_target_factor, cfg.adr_velocity_target_factor_final)},
             "fabric_damping": {"gain": (
                 cfg.fabrics_damping_gain, cfg.adr_fabric_damping_final)},
             "robot_spawn": {"joint_pos_noise": (0.0, cfg.adr_robot_joint_pos_noise_final),
                             "joint_vel_noise": (0.0, cfg.adr_robot_joint_vel_noise_final)},
             "object_state_noise": {"scale": (
                 cfg.obs_noise_scale, cfg.adr_obs_noise_scale_final)},
             "object_wrench": {"max_linear_accel": (
                 0.0, cfg.adr_object_wrench_accel_final)}},
            num_increments=cfg.adr_num_increments,
            increment_interval=cfg.adr_increment_interval,
            trigger_threshold=cfg.adr_trigger_threshold,
            enabled=bool(cfg.enable_adr),
            event_manager=getattr(self, "event_manager", None) if cfg.enable_physics_dr else None,
            physics_cfg=self._physics_terminal() if cfg.enable_physics_dr else None,
        )

        # ★★접촉 필터 prim 이 rigid body 를 가리키는지 부팅 시 확인한다.
        #   루트 Xform 을 가리키면 PhysX 가 "GPU contact filter … is not supported" 를
        #   내고 force_matrix_w 가 **항상 0** 이 된다. 보상 7항 중 6항이 접촉 게이트에
        #   걸려 있으므로 그 상태로 학습하면 approach 하나만 남는다 —
        #   fab_test1~4 를 그렇게 날렸다(force_max 가 전 구간 정확히 0.0000).
        _flt = list(self.cfg.object_contact_filter)
        _want = self.bank.rigid_body_name
        if not _flt or not all(f.rstrip("/").endswith("/" + _want) for f in _flt):
            raise RuntimeError(
                f"object_contact_filter 가 rigid body prim('{_want}')을 안 가리킨다: {_flt}\n"
                "  루트 Xform 을 가리키면 PhysX 가 force_matrix_w 를 항상 0 으로 준다 —\n"
                "  보상 7항 중 6항이 접촉 게이트라 학습이 approach 하나로 줄어든다."
            )

        print(
            f"[grasp_lift_fabric] profile={p.name} asset={p.asset.name} "
            f"fabric={p.fabric_robot_dir} bank={self.bank.name}({len(self.bank)}) "
            f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
            f"dr={cfg.enable_physics_dr} adr={cfg.enable_adr}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _one_body(self, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"[{self.profile.name}] body '{name}' 해석 실패: {ids}")
        return ids[0]

    def _physics_terminal(self):
        from openarm.agnostic.modules import physics_dr as _pdr
        return _pdr.PHYSICS_ADR_TERMINAL

    def _palm_pose_6d(self) -> torch.Tensor:
        """현재 palm pose (env-local xyz + euler_zyx) — fabric 명령과 같은 규약."""
        from isaaclab.utils.math import euler_xyz_from_quat

        pos = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([pos, torch.stack([y, pi, r], dim=1)], dim=1)

    def _build_fabric_index(self) -> torch.Tensor:
        """프로필의 `fabric_joint_order` → articulation 인덱스 텐서.

        ★articulation 은 depth-major (index_1, middle_1, pinky_1, ring_1, thumb_1, …)
          인데 fabric URDF 는 **finger-major** (thumb_1..4, index_1..4, …) 다.
          `cat([q[:, arm_t], q[:, hand_t]])` 로 만들면 손 20관절이 통째로 어긋나고,
          fabric 이 엉뚱한 손 자세로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다.
        """
        order = self.profile.fabric_joint_order
        if len(order) != self.fabric.num_joints:
            raise RuntimeError(
                f"[{self.profile.name}] fabric_joint_order 길이 {len(order)} != "
                f"fabric num_joints {self.fabric.num_joints}. 프로필을 고쳐라."
            )
        idx = []
        for name in order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{self.profile.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def _fabric_order(self, q: torch.Tensor) -> torch.Tensor:
        """로봇 articulation 순서 → fabric 순서."""
        return q[:, self._fab_t]

    # ==================================================================
    def _setup_fabrics(self) -> None:
        p = self.profile
        initialize_warp(str(self.device)[-1])   # 멀티 GPU 캐시 분리(grasp_v1 규약)
        self._world = WorldMeshesModel(
            batch_size=self.num_envs, device=self.device,
            max_objects_per_env=int(self.cfg.fabrics_max_objects_per_env),
        )
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        cls = _fabric_class(p.fabric_class)
        self.fabric = cls(
            batch_size=self.num_envs, device=self.device,
            timestep=float(self.cfg.fabrics_dt),
            graph_capturable=bool(self.cfg.fabric_use_cuda_graph),
            # PCA(5D)는 감쌈을 제약하므로 쓰지 않는다 — direct 는 20-DOF 관절 그대로다.
            use_hand_fabric=self._hand_fabric,
            hand_mode="direct",
            hand_attractor_gain=getattr(self.cfg, "hand_attractor_gain", None),
            use_hand_repulsion=bool(getattr(self.cfg, "use_hand_repulsion", False)),
            # ★★KUKA식 손↔팔 repulsion(13 쌍 상당). 손가락 쌍은 여기 포함되지 않는다.
            use_body_repulsion_pairs=bool(
                getattr(self.cfg, "use_body_repulsion_pairs", False)),
            use_tip_fabric=False,            # tip IK 폐기(08.26) — 풀 관절 단일 경로
            tip_attractor_gain=None,
            tip_per_finger=False,
            robot_dir_name=p.fabric_robot_dir,
            robot_name=p.fabric_robot_dir,
        )
        self.integrator = DisplacementIntegrator(self.fabric)

        # ★fabric 의 관절 순서는 [팔 N, 손 M] 이다. 로봇 articulation 순서와 다르다 —
        #   양팔 로봇에서 `default_joint_pos[:, :27]` 로 자르면 좌우가 섞인 전혀 다른
        #   관절이 들어간다(probe 실측: palm 이 2초에 61mm 밖에 못 움직이면서 관절속도는
        #   20 rad/s 로 포화). 반드시 프로필 인덱스로 재조립한다.
        n_j = self.fabric.num_joints
        n_arm = self.profile.num_arm_joints
        expect = self.profile.num_arm_joints + self.profile.num_hand_joints
        if n_j != expect:
            raise RuntimeError(
                f"[{self.profile.name}] fabric num_joints={n_j} 인데 프로필은 "
                f"팔{self.profile.num_arm_joints}+손{self.profile.num_hand_joints}={expect}. "
                "fabric URDF 와 자산이 어긋났다."
            )
        self._fab_t = self._build_fabric_index()
        self.fabric_q = self._fabric_order(self.robot.data.default_joint_pos).contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, n_j, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, n_j, device=self.device)
        # hand_mode="direct" — (B, 손DOF) 관절 목표. ★0 이 아니라 **홈**으로 초기화:
        # obs 의 joint_pos_err = (지령−실측)이 리셋 직후 스텝에서 0−q 로 오염되지
        # 않게 한다(지령을 아직 안 낸 상태의 자연값은 "현재 자세 유지").
        self._fabric_hand_cmd = torch.zeros(
            self.num_envs, p.num_hand_joints, device=self.device)
        # ★단계형 보상이 파지중심을 쓰고, 그 실측(`_measure_tip_workspace`)이 이
        #   인덱스/한계를 쓴다 — 손 제어 방식과 무관하게 만들어야 pd 대조군이 산다.
        if True:
            # fabric 손 순서 → robot 손 관절 순서(적용 시 되돌리는 역매핑).
            _fab_hand_ids = self._fab_t[n_arm:].tolist()
            self._hand_from_fab = torch.tensor(
                [_fab_hand_ids.index(int(j)) for j in self._hand_t.tolist()],
                device=self.device, dtype=torch.long)
            # robot 손 관절 순서 → fabric 손 구간 순서(목표를 넘길 때 쓴다).
            _robot_hand_ids = self._hand_t.tolist()
            self._fab_from_hand = torch.tensor(
                [_robot_hand_ids.index(int(j)) for j in _fab_hand_ids],
                device=self.device, dtype=torch.long)

        # cspace attractor(널스페이스)의 rest 자세를 **프로필 홈**으로 맞춘다.
        #   fabric URDF 내장 기본값은 [1.0, -0.1, -0.6, 0.5, 0, 0, 0] 로 다른 셋업용이다.
        #   robot-agnostic 에서는 rest 자세도 프로필이 정하는 게 맞다.
        #   ※정직한 단서: 이 변경을 "추종 불량의 원인"으로 지목했다가 **철회**했다.
        #     그때 쓴 probe 가 scene.write_data_to_sim() 을 빠뜨려 관절 목표가 PhysX 에
        #     도달하지 않은 상태였다. 이 설정은 설계상 옳아서 남겨둘 뿐,
        #     추종 성능에 대한 근거는 아직 없다.
        # ★★palm attractor 오버라이드 — 공유 yaml 을 건드리지 않는다.
        #   Attractor 는 params dict 를 **참조**로 들고(fabric_term.py: self.params =
        #   params) 매 스텝 self.params['conical_gain'] 을 읽으므로, 생성 후 하위
        #   dict 를 덮으면 그대로 반영된다. 사본 yaml 도 임시파일도 필요 없다.
        _prior = self.fabric.default_config.clone()
        self.fabric.default_config.copy_(self.fabric_q)
        # ★파지중심(손끝 홈 FK 평균)은 **제어 방식과 무관한 기하 상수**인데 산출이
        #   tip 모드에 묶여 있었다 — pd 대조군이 `_grasp_center_local` 없음으로 죽었다.
        #   단계형 보상이 이 값을 쓰므로 손 제어와 무관하게 항상 실측한다.
        if True:
            self._fabric_hand_cmd[:] = self.fabric_q[:, p.num_arm_joints:]
            self._measure_tip_workspace()
        print(
            "[grasp_lift_fabric] fabric.default_config 를 프로필 홈으로 교체\n"
            f"  이전(fabric 내장) 팔 : {[round(v, 3) for v in _prior[0, :n_arm].tolist()]}\n"
            f"  이후(프로필 홈) 팔  : "
            f"{[round(v, 3) for v in self.fabric_q[0, :n_arm].tolist()]}",
            flush=True,
        )
        # ★로봇 적응(08.26): palm_y 미러 부호. 자매 수식 roll_q=clamp(cos(palm_y,up))^4
        #   는 좌손(미러, palm_y 가 아래)에서 ≡0 이 되어 자세 항이 통째로 죽는다 —
        #   h4 좌팔 실측. 수식은 자매와 동일하게 두고 **입력 부호**를 부팅 홈 자세에서
        #   실측해 곱한다(자산이 바뀌어도 스스로 맞는다). fabric 홈 팔레트 R 의 y열
        #   z성분 부호 = cos(palm_y, up) 부호.
        _o0, _R0 = self._palm_frame(self.fabric_q)
        _ys = float(torch.sign(_R0[0, 2, 1]))
        self._palm_y_sign = _ys if _ys != 0.0 else 1.0
        print(f"[grasp_lift_fabric] palm_y 미러 부호 = {self._palm_y_sign:+.0f} "
              f"(cos(palm_y,up) 홈 실측 {float(_R0[0, 2, 1]):+.3f})", flush=True)
        # ★★08.27 관절별 **굴곡 부호** 부팅 실측 — 액션 매핑의 기준.
        self._measure_flex_signs()
        # close_bridge 용 폐쇄도 기준 — fabric 손 관절 홈과 **닫힘 방향**. 닫힘은
        #   hi 가 아니라 `_flex_limit`(굴곡 부호 실측) 이다 — 엄지는 좌우가 반대라
        #   hi 로 고정하면 좌팔 폐쇄도 부호가 뒤집힌다(08.27 URDF FK 실측).
        _n_arm0 = self.profile.num_arm_joints
        self._fab_hand_home = self.fabric_q[0, _n_arm0:].clone()
        _den = self._fab_flex_limit - self._fab_hand_home
        self._close_valid = _den.abs() > 1e-6
        self._close_den = torch.where(
            self._close_valid, _den, torch.ones_like(_den))
        # ── close_bridge **v2**(자매 3ac85a9) — 엄지/4지 분리 폐쇄도용 슬롯 매핑.
        #   fabric 손 슬롯 i ↔ 로봇 관절 이름 joint_names[_fab_t[n_arm+i]] 이고,
        #   관절 이름에 손가락 키가 들어 있다(테솔로 규약: *_thumb_* 등).
        #   min(엄지, 4지) 합성은 항상 **뒤처진 그룹**에 gradient 를 준다("모두
        #   잡히게" — 사용자 지시). 이름 매칭 실패 슬롯은 fail-loud.
        _jn = list(self.robot.data.joint_names)
        _thumb_key = self.profile.contact_group_a[0]
        _fingers_all = list(self.profile.fingers)
        _slot_finger = []
        for _i in range(len(self._fab_hand_home)):
            _nm = _jn[int(self._fab_t[_n_arm0 + _i])]
            _hit = [f for f in _fingers_all if f in _nm]
            if len(_hit) != 1:
                raise RuntimeError(
                    f"fabric 손 슬롯 {_i}({_nm}) 의 손가락 매칭이 모호하다: {_hit}")
            _slot_finger.append(_hit[0])
        _usable_names = {_fingers_all[int(i)] for i in self._usable_t}
        self._close_thumb_m = torch.tensor(
            [f == _thumb_key for f in _slot_finger], device=self.device)
        self._close_fingers_m = torch.tensor(
            [(f != _thumb_key and f in _usable_names) for f in _slot_finger],
            device=self.device)
        self._close_thumb_m &= self._close_valid
        self._close_fingers_m &= self._close_valid
        if not bool(self._close_thumb_m.any()) or not bool(self._close_fingers_m.any()):
            raise RuntimeError("close_bridge v2 슬롯 매핑이 비었다(엄지 또는 4지)")
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)
        self._wrench_tick = 0        # 외란 렌치 주기 카운터 (KUKA wrench_trigger_every)

    # ==================================================================
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 환경 픽스처: rigid body 없는 **정적** 삼각메시 콜라이더라 RigidObject 로
        # 올리지 않는다. 원점은 로봇 base link 원점에 붙인다(사용자 규약).
        _cfg_fix = self.cfg.env_fixture_spawn
        _cfg_fix.func(_cfg.ENV_FIXTURE_PRIM, _cfg_fix, translation=(0.0, 0.0, 0.0))

        # 손가락 마디별 접촉 센서 — body 마다 **개별** 생성한다.
        # (다중 body 단일 센서는 force_matrix_w 가 0 을 반환한다 — 실측 함정)
        p = self.profile
        flt = list(self.cfg.object_contact_filter)
        self._sensors: dict[str, dict[str, list]] = {}
        for finger in p.fingers:
            roles = {"tip": [], "wrap": []}
            for role, bodies in (("tip", p.finger_tip_bodies[finger]),
                                 ("wrap", p.finger_wrap_bodies.get(finger, ()))):
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=flt,
                        history_length=1, track_air_time=False,
                    ))
                    roles[role].append(s)
                    self.scene.sensors[f"contact_{finger}_{body}"] = s
            self._sensors[finger] = roles

        # ★기본 ground plane 을 z=0 에 두지 않는다 — env.usd 의 platform 상면이 정확히
        #   z=0 이라 지면과 겹쳐 상시 접촉/관통이 생긴다. 바닥은 env.usd 의 base_plate
        #   (z -0.025~-0.015)가 담당하므로, 지면은 그 아래로 내려 시각적 배경으로만 둔다.
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, float(self.cfg.ground_plane_z)),
        )
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        # ★clone → 물체 순서. 반대로 하면 env_0 만 존재하는 시점에 MultiAssetSpawner 가
        #   assets_cfg[0] 하나만 스폰하고 clone 이 그걸 전 env 에 복제한다(배정 붕괴).
        self.scene.clone_environments(copy_from_source=True)
        _ob.assert_spawned_after_clone(self.bank, cloned=True)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object
        if not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self.actions = actions.clamp(-1.0, 1.0)

        # ★★KUKA 고정(08.25): 원본은 매 스텝 ADR 에서 cspace 감쇠를 읽어 갱신한다
        #   (`fabric_damping.gain` 10 → 20). 부팅 시 한 번만 채우면 커리큘럼이 죽는다.
        self._fabric_damping.fill_(float(self.adr.get_param("fabric_damping", "gain")))

        # ★★KUKA 고정(08.25) — 원본 `compute_absolute_action` 과 **같은 식**이다:
        #     scale(a, lo, hi) = 0.5·(hi−lo)·a + 0.5·(hi+lo)   →  a=−1 lo · a=0 **중앙** · a=+1 hi
        #     그다음 tensor_clamp(lo, hi).
        #   구 규약은 a=0 이 **홈**인 구간별 선형이었다 — 홈이 박스 중앙이 아니면 두 반쪽의
        #   기울기가 달라 같은 액션 크기가 방향에 따라 다른 이동을 낸다.
        a_arm = self.actions[:, :6]
        desired = _kuka_absolute(a_arm, self.palm_lo, self.palm_hi)
        if self._slew_on:
            # A: 지령을 목표 쪽으로 **스텝당 상한만큼만** 움직인다.
            d = (desired - self.palm_cmd).clamp(-self._slew, self._slew)
            self.palm_cmd = self.palm_cmd + d
        else:
            self.palm_cmd = desired
        self.palm_targets = self.palm_cmd
        # 손: **절대** 관절 목표(자유 관절만 정책이 소유).
        # ★★08.27 `a=−1 → 홈(펴짐)`, `a=+1 → 굴곡 한계`. 구 매핑은 `[lo, hi]` 선형
        #   이었는데 `_3`/`_4` 한계가 대칭 ±90° 라 홈이 **중앙**이었다 — 액션 절반이
        #   손등 쪽 역굴곡을 지시했다. 이 매핑에서는 역굴곡이 **액션 공간 밖**이라
        #   클램프도 벌점도 필요 없다. `_flex_limit` 은 부팅 FK 실측 부호가 정한다
        #   (엄지는 좌우가 반대라 상수로 박으면 좌팔이 뒤집힌다).
        u_hand = 0.5 * (self.actions[:, 6:] + 1.0)
        _free_targets = self._hand_home_free + u_hand * (
            self._flex_limit - self._hand_home_free)
        # 손 전체(고정 관절 포함)를 fabric 에 넘긴다 — 고정 관절은 init 값을 목표로
        # 줘서 fabric 이 유지. 일부만 넘기면 fabric 이 아는 손과 실제가 어긋난다.
        _full = self._default_q[:, self._hand_t].clone()
        _free_cols = self._hand_from_fab.new_tensor(
            [self._hand_t.tolist().index(int(j)) for j in self._hand_free_t.tolist()])
        _full[:, _free_cols] = _free_targets
        self._fabric_hand_cmd = _full[:, self._fab_from_hand]
        # ★★08.27 손을 fabric 밖으로 뺐다(사용자 지시 · 자매 트랙 배선과 동일).
        #   h7 실측: |fabric_q_hand − 정책 지령| 이 우 0.956rad(55°)·좌 0.645rad 였다.
        #   PD 는 fabric 해를 0.12~0.25rad 로 잘 따라갔으므로 범인은 PD 가 아니라
        #   **fabric attractor 가 지령을 재해석한 것**이다. 정책이 "쥐어라" 해도 손은
        #   55° 덜 쥐었다. fabric 은 실행 보조지 지령 해석기가 아니다(사용자 확정).
        #   부수 피해도 사라진다 — fabric 이 손 20개를 소유하면 "고정" 외전까지
        #   body_repulsion·joint_limit 가 밀어 손가락이 벌어졌다(fmin 27~28 vs PD 23~24mm).
        _prev_hand = self.hand_targets
        self.hand_targets = _full
        self._hand_vel = (self.hand_targets - _prev_hand) / self._policy_dt
        # ★fabric 의 손 **상태**는 실제 지령으로 덮는다. 안 그러면 fabric 이 다른 손으로
        #   충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다(자매 트랙 경고).
        self.fabric_q[:, self.profile.num_arm_joints:] = self._fabric_hand_cmd
        # **지령** 손끝 = 지령 관절의 FK — tip_bridge 가 지령을 평가해야 정책 액션과
        # 보상의 인과가 직결된다(실 손끝 기준은 IK 추종오차만큼 인과가 끊겼었다).
        _q_cmd = self.fabric_q.detach().clone()
        _q_cmd[:, self.profile.num_arm_joints:] = self._fabric_hand_cmd
        _tw, _ = self.fabric._fingertip_taskmap(_q_cmd, None)
        self._tip_cmd_local = _tw.reshape(self.num_envs, -1, 3)
        # 액션 cmd 마커(env0) — 손끝 지령 5색 + palm 지령 흰색. GUI·영상 공용.
        if self._tip_markers is not None:
            self._update_tip_markers(self._tip_cmd_local)
        self.fabric.set_features(
            self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self._world_ids, self._world_indicator, self._fabric_damping,
        )
        self._step_fabric()
        # ★★KUKA 고정(08.25): 원본은 fabric 적분 **직후** 물체 외란 렌치를 넣는다.
        self._apply_object_wrench()

    # ------------------------------------------------------------------
    def _palm_frame(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fabric FK 로 (palm 원점 (N,3), palm 회전 (N,3,3) — 열이 x/y/z 축).

        프레임 이름을 여기서 알 필요가 없다: fabric 의 "palm" taskmap 이 원점과 세 축
        보조점을 이미 자기 규약으로 뽑는다(`get_palm_pose` 와 같은 구성). 로봇마다
        다른 링크명을 태스크 코드가 알면 robot-agnostic 이 깨진다.
        """
        pts, _ = self.fabric.get_taskmap("palm")(q, None)
        if pts.shape[1] < 18:
            raise RuntimeError(
                "palm taskmap 이 1점 모드다(palm_position_only) — 손끝 목표를 palm "
                "좌표계로 변환할 회전이 없다. tip IK 는 6-DOF palm 규약을 요구한다.")
        o = pts[:, :3]
        ax = torch.stack([
            torch.nn.functional.normalize(pts[:, 3:6] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 9:12] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 15:18] - o, dim=1),
        ], dim=-1)                                    # (N,3,3) 열 = 축
        return o, ax

    def _measure_flex_signs(self) -> None:
        """부팅 1회 FK — 자유 손관절마다 **굴곡(말림) 방향 부호**를 실측한다.

        ★왜 필요한가(08.27 실측): 액션이 `a∈[-1,1] → [관절 lo, hi]` 선형이었는데
          `_3`/`_4` 한계가 **좌우 모두 대칭 ±90°** 라 홈(0)이 한계 **중앙**이다.
          즉 액션 범위의 절반이 손등 쪽 **역굴곡**을 지시하고 있었다("손가락이
          난리" 의 직접 원인). 게다가 엄지 `_3`/`_4` 는 우 `+q`·좌 `−q` 가 굴곡인데
          (URDF `thumb_3` origin rpy 가 좌우 뒤집힘, axis 는 둘 다 (0,0,1) 이라
          한계에는 안 드러난다) 액션 매핑에 미러가 없어 좌손 엄지는 `a=+1` 이
          완전 개방이었다.

        판정 기준: **대향 그룹 쪽으로 가면 굴곡**이다. 손끝이 `contact_group_a/b`
        의 반대 그룹 중심에 가까워지는 부호를 굴곡으로 잡는다. palm 로컬 +x(법선)
        성분으로 재는 `probe_curl_local` 규약은 4지에는 맞지만 **엄지는 대향
        운동이라 −x 로 움직여** 오판한다(FK 실측). 대향 그룹 거리는 프레임 불변이고
        로봇 종속 이름도 안 쓴다.
        """
        p = self.profile
        n_arm = p.num_arm_joints
        _fg = list(p.fingers)
        _ia = [_fg.index(f) for f in p.contact_group_a]
        _ib = [_fg.index(f) for f in p.contact_group_b]
        _jn = list(self.robot.data.joint_names)
        # 자유 관절(로봇 순서) → fabric 손 슬롯 · 손가락 인덱스
        _free_fab, _free_fing = [], []
        for _j in self._hand_free_t.tolist():
            _slot = int((self._fab_t[n_arm:] == _j).nonzero()[0, 0])
            _nm = _jn[_j]
            _hit = [k for k, f in enumerate(_fg) if f in _nm]
            if len(_hit) != 1:
                raise RuntimeError(
                    f"관절 '{_nm}' 의 손가락을 특정할 수 없다(매칭 {_hit}) "
                    f"— fingers={_fg}. 프로필 이름 규약을 확인할 것")
            _free_fab.append(_slot)
            _free_fing.append(_hit[0])

        def _opp_dist(q: torch.Tensor) -> torch.Tensor:
            """(B,) — 각 배치의 '해당 손가락 손끝 ↔ 대향 그룹 손끝 중심' 거리."""
            _t, _ = self.fabric._fingertip_taskmap(q, None)
            _t = _t.reshape(q.shape[0], -1, 3)
            return _t

        _q0 = self.fabric_q[:1].repeat(self.num_envs, 1)
        _tips0 = _opp_dist(_q0)[0]                       # (F,3) 홈 손끝
        _delta = 0.30                                    # [rad] 부호만 보므로 크게
        signs, worst = [], 1e9
        _n = len(_free_fab)
        for _s in range(0, _n, self.num_envs):
            _chunk = list(range(_s, min(_s + self.num_envs, _n)))
            _qb = _q0.clone()
            for _r, _i in enumerate(_chunk):
                _qb[_r, n_arm + _free_fab[_i]] += _delta
            _tips = _opp_dist(_qb)
            for _r, _i in enumerate(_chunk):
                _f = _free_fing[_i]
                _opp = _ib if _f in _ia else _ia
                _c0 = _tips0[_opp].mean(dim=0)
                _d0 = float(torch.norm(_tips0[_f] - _c0))
                _c1 = _tips[_r][_opp].mean(dim=0)
                _d1 = float(torch.norm(_tips[_r][_f] - _c1))
                _chg = _d0 - _d1                          # >0 = 가까워짐 = 굴곡
                worst = min(worst, abs(_chg))
                signs.append(1.0 if _chg > 0.0 else -1.0)
        if worst < 1e-4:
            raise RuntimeError(
                f"굴곡 부호 실측 실패: 대향거리 변화 최소 {worst*1000:.3f}mm 로 "
                "판별 불가 — taskmap/한계/자산을 확인할 것")
        self._flex_sign = torch.tensor(signs, device=self.device)   # (n_free,)
        # 굴곡 쪽 한계. a=+1 이 여기로 간다.
        self._flex_limit = torch.where(
            self._flex_sign > 0, self._hand_hi[0], self._hand_lo[0])   # (n_free,)
        self._hand_home_free = self._default_q[0, self._hand_free_t].clone()
        # fabric 손 슬롯 순서의 굴곡 한계 — close_bridge 분모용.
        _fl_fab = self.fabric_q[0, n_arm:].clone()
        for _i, _slot in enumerate(_free_fab):
            _fl_fab[_slot] = self._flex_limit[_i]
        self._fab_flex_limit = _fl_fab
        _neg = [_jn[int(j)].split("hj_")[-1]
                for j, s in zip(self._hand_free_t.tolist(), signs) if s < 0]
        print(f"[grasp_lift_fabric] 굴곡 부호 실측 {len(signs)}관절 · "
              f"음수(−q 가 굴곡) {_neg if _neg else '없음'} · "
              f"최소 판별폭 {worst*1000:.1f}mm", flush=True)

    def _home_grasp_center_xy(self) -> torch.Tensor:
        """홈 자세 파지중심의 env-local XY (2,) — 컵을 "palm 바로 앞"에 놓을 좌표.

        파지중심(`_grasp_center_local`)은 palm 프레임에 붙은 점이고, d_gc=0 이
        곧 "컵이 손 안에 있다"이다. 그 점을 홈 palm 프레임으로 world 변환한 XY 에
        컵을 놓으면 정책은 **Z 만 내리면** 되는 상태에서 시작한다(grasp_v1 규약:
        XY 는 palm 을 따라가고 Z 는 테이블 높이 고정).

        ★IK 가 아니다 — 순방향 FK 한 번이다. 실패할 자유도가 없다.
        """
        with torch.no_grad():
            o, R = self._palm_frame(self.fabric_q)
            gc_w = o + torch.einsum(
                "bij,j->bi", R, self._grasp_center_local.to(o.dtype))
        xy = gc_w[0, :2].clone()
        print(f"[grasp_lift_fabric] 컵 근접 스폰 XY = 홈 파지중심 "
              f"[{float(xy[0]):.3f}, {float(xy[1]):.3f}] "
              f"(ADR reset_near 로 테이블 스폰까지 후퇴)", flush=True)
        return xy

    def _measure_tip_workspace(self) -> None:
        """부팅 1회 FK — 홈 손끝(`_tip_home`)과 파지중심(`_grasp_center_local`) 유도.

        파지중심은 대향 두 그룹 손끝 중점(홈 자세)으로 유도한 뒤, cfg 의
        **자유 컵 실측 오버라이드**(stage_gc_local_override)로 덮는다 — 홈 유도값은
        y 부호(좌우 미러)와 상식범위 검사에만 쓰인다. 물리·Isaac 불필요(FK 전용).
        """
        n_arm = self.profile.num_arm_joints
        q0 = self.fabric_q.clone()
        o, R = self._palm_frame(q0)
        tips, _ = self.fabric._fingertip_taskmap(q0, None)
        tips = tips.reshape(self.num_envs, -1, 3)
        rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2), tips - o[:, None, :])
        self._tip_home = rel[0].clone()               # (T,3) 홈 손끝 (palm 상대)
        _fg = list(self.profile.fingers)
        _ia = [_fg.index(f) for f in self.profile.contact_group_a]
        _ib = [_fg.index(f) for f in self.profile.contact_group_b]
        self._grasp_center_local = 0.5 * (
            rel[0][_ia].mean(dim=0) + rel[0][_ib].mean(dim=0))
        _ovr = getattr(self.cfg, "stage_gc_local_override", None)
        if _ovr is not None:
            _home_mid = self._grasp_center_local.clone()
            _ys_gc = 1.0 if float(_home_mid[1]) >= 0.0 else -1.0
            self._grasp_center_local = torch.tensor(
                [float(_ovr[0]), _ys_gc * abs(float(_ovr[1])), float(_ovr[2])],
                device=self.device)
            _shift = float(torch.norm(self._grasp_center_local - _home_mid))
            if not (0.010 <= _shift <= 0.120):
                raise RuntimeError(
                    f"파지중심 오버라이드 이동량 {_shift*1000:.0f}mm 이 상식 범위"
                    "(10~120mm) 밖 — 자산이 바뀌었으면 재실측할 것")
            print(f"[grasp_lift_fabric] 파지중심 "
                  f"{[round(float(v)*1000) for v in self._grasp_center_local]}mm "
                  f"(자유 컵 실측 오버라이드 · 홈 유도 대비 {_shift*1000:.0f}mm)",
                  flush=True)
    def _tip_cmd_surface_dist(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """지령 손끝 → 물체 **측면 표면 띠** 까지의 거리 (N, 가용손가락).

        컵 축은 물체 +z(obj_up). 축 기준 반경 radial 과 높이 h 로 분해해 반경은
        R 로, 높이는 +-H 띠 안으로 오게 만드는 거리를 낸다. 손끝이 띠 위에
        정확히 있으면 0.
        ★수식(공유 파일)은 거리만 받는다 — 형상 가정은 여기 env 에만 둔다.
        """
        from isaaclab.utils.math import matrix_from_quat
        axis = matrix_from_quat(self.object.data.root_quat_w)[:, :, 2]
        d = self._tip_cmd_local - obj_pos[:, None, :]
        h = (d * axis[:, None, :]).sum(dim=-1)
        radial = (d - h[..., None] * axis[:, None, :]).norm(dim=-1)
        R = self._grasp_radius[:, None]                    # (N,1) per-object 실측
        H = self._grasp_halfheight[:, None]
        out = ((radial - R) ** 2 + torch.relu(h.abs() - H) ** 2).sqrt()
        return out[:, self._usable_t]

    def _update_tip_markers(self, tip_cmd: torch.Tensor) -> None:
        """손끝 **목표**(fabric 지령)에 구 마커를 놓는다 — **env0 만**.

        ★★08.26 두 가지를 고쳤다(사용자 관찰: "마커가 world 중심에 몰려 있다").
          ① `tip_cmd` 는 `_palm_frame`(fabric FK) 기준이라 **world 절대좌표가 아니다**.
             `tip/target_err_mm` 은 목표·실제를 **둘 다 palm 상대로** 바꿔 프레임이
             상쇄되므로 정상값이 나왔고, 절대좌표를 요구하는 마커만 어긋났다.
             → `env_origins` 를 더해 world 로 올린다.
          ② 전 env(2048×5=10,240 개)를 그리던 것을 **env0 5 개**로 줄인다.
             프림 수가 렌더 병목의 주원인이고, 확인용으로는 한 env 면 충분하다.
        """
        tw = tip_cmd[0] + self.scene.env_origins[0]           # (T,3) world
        # palm **지령** 위치(slew 후) — env-local 6D 의 위치부를 world 로.
        _pcmd = (self.palm_cmd[0, :3] + self.scene.env_origins[0]).unsqueeze(0)
        self._tip_markers.visualize(
            translations=torch.cat([tw, _pcmd], dim=0),
            marker_indices=self._tip_marker_idx,
        )

    def _step_fabric(self) -> None:
        """★정책 스텝마다 **한 번**만 적분한다.

        `_apply_action` 은 decimation(=2) 번 호출되므로 그 안에서 적분하면
        fabric 시간이 2배로 빨리 흐른다(grasp_v1 은 _pre_physics_step 에서 적분한다).
        """
        for _ in range(int(self.cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(self.cfg.fabrics_dt),
            )

    def _apply_object_wrench(self) -> None:
        """★★KUKA 고정(08.25) — 원본 `apply_object_wrench`.

        1 초(=`wrench_trigger_every` 정책 스텝)마다 물체에 **랜덤 외란 가속도**를 준다.
        파지가 성립한 뒤 흔들려도 놓치지 않게 하는 강건화 축이고, ADR 로 0 → 10 m/s²
        까지 올라간다. 손이 물체에서 멀면(`wrench_hand_distance_threshold`) 넣지 않는다
        — 아직 안 쥔 물체를 흔드는 것은 과제와 무관한 교란이다.

        가속도로 주는 이유: 힘으로 주면 질량 DR 과 곱해져 무거운 물체만 덜 흔들린다.
        토크는 같은 가속도에 `torsional_radius` 를 곱한 모멘트 팔로 환산한다(원본 동일).
        """
        accel = float(self.adr.get_param("object_wrench", "max_linear_accel"))
        if accel <= 0.0:
            return
        self._wrench_tick += 1
        if self._wrench_tick % max(1, int(self.cfg.wrench_trigger_every)) != 0:
            return
        # 손 ↔ 물체 거리 게이트 (원본 hand_to_object_dist_threshold)
        obj_w = self.object.data.root_pos_w
        palm_w = self.robot.data.body_pos_w[:, self.palm_idx]
        near = (obj_w - palm_w).norm(dim=-1) < float(self.cfg.wrench_hand_distance_threshold)
        mass = self.object.data.default_mass.to(self.device).view(-1, 1)
        a = accel * 2.0 * (torch.rand(self.num_envs, 3, device=self.device) - 0.5)
        forces = a * mass * near.unsqueeze(-1)
        torques = forces * float(self.cfg.torsional_radius)
        self.object.set_external_force_and_torque(
            forces.unsqueeze(1), torques.unsqueeze(1))

    def _apply_action(self) -> None:
        """fabric 계획을 관절 PD 목표로 내린다 — **위치와 속도를 모두** 준다.

        ★★08.25 속도 피드포워드를 되살렸다. 이전에는 `fabric_qd` 를 계산해 놓고
        버린 채 0 을 넣었는데, 그러면 PD 의 감쇠항이 움직임 자체를 되민다:

            속도목표 0   : kp·err = kd·v + τ_마찰  →  err ≈ (kd/kp)·v = 0.2·v [rad]
            속도목표 qd  : kp·err = τ_마찰만       →  err ≈ τ_f/kp

        우리 팔은 kp=400 · kd=80 이라 kd/kp = 0.2 — **속도에 정비례해 뒤처지도록**
        배선돼 있었다. DEXTRAH 원본은 `dof_vel_targets = clone(fabric_qd)` 를 그대로
        내리고, `velocity_target_factor` 를 ADR 로 1.0 → 0.0 으로 **줄여 간다**
        (실기의 불완전한 속도 목표에 대비한 커리큘럼). 우리는 그 최종 단계에서
        시작한 셈이었다.
        """
        n_arm = self.profile.num_arm_joints
        # ★★KUKA 고정(08.25): 원본은 이 계수를 ADR 로 1.0 → 0.0 으로 줄여 간다.
        _vf = float(self.adr.get_param("pd_targets", "velocity_target_factor"))
        arm_target = self.fabric_q[:, :n_arm]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(
            self.fabric_qd[:, :n_arm] * _vf, joint_ids=self.arm_ids)
        # ★★손은 fabric 밖이다 — 정책 관절 목표를 PD 에 **그대로** 내린다.
        #   고정 관절은 `hand_targets` 안에서 init 값 그대로라 별도 처리가 필요 없다.
        self.robot.set_joint_position_target(self.hand_targets, joint_ids=self.hand_ids)
        # 속도 피드포워드 — 없으면 감쇠항 kd·(0 − q̇) 이 닫는 동작을 상시 반대로 밀어
        # err ≈ (kd/kp)·q̇ 의 지연이 생긴다. fabric 밖이라 `fabric_qd` 를 쓸 수 없고
        # 지령 램프의 도함수(`_hand_vel`)를 쓴다(자매 트랙 `_syn_vel` 과 동일).
        self.robot.set_joint_velocity_target(
            float(self.cfg.hand_velocity_ff_scale) * self._hand_vel,
            joint_ids=self.hand_ids)

    # ------------------------------------------------------------------
    def _contact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(손가락별 총 접촉력, wrap 전마디 동시접촉 여부, 중간마디 힘, 원위마디 힘) 각 (N,F).

        · tot  — 참여 임계(0.1N) 소비자용 원값.
        · mid/dist — 마디별 원값. 보상의 `envelope_fraction`(grasp_sensor 규약: 마디
          **하나라도** 접촉하면 그 손가락은 감쌈)이 이 둘을 받아 판정한다.
        · wrapped — 우리 트랙이 08.22 에 도입한 **엄격** 판정(전 마디가 동시에 0.5N 초과).
          ★08.23 부터 보상에서 빠지고 **진단 지표**로만 남는다. 사용자 결정으로 보상
            규약을 grasp_sensor 와 통일했기 때문이다. 버리지 않는 이유: 같은 정책을 두
            판정으로 재면 0.503 vs 0.069 로 **7배**가 벌어진 실측이 있다. 느슨한 판정이
            오르는데 엄격 판정이 안 오르면 "받치기"를 감쌈으로 세고 있다는 신호다.
        """
        w_thr = float(getattr(self.cfg, "envelope_force_threshold", 0.5))
        tot, wrapped, mids, dists, tipfs, tipvs = [], [], [], [], [], []
        zero = torch.zeros(self.num_envs, device=self.device)
        for f in self._fingers:
            roles = self._sensors[f]
            t = torch.zeros(self.num_envs, device=self.device)
            w = None
            mags = []
            _tf = torch.zeros(self.num_envs, device=self.device)
            _tv = torch.zeros(self.num_envs, 3, device=self.device)
            for s_ in roles["tip"]:
                _v = s_.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1)
                _tv = _tv + _v
                _m = _v.norm(dim=-1)
                _tf = _tf + _m
                t = t + _m
            for s_ in roles["wrap"]:
                m = s_.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1).norm(dim=-1)
                t = t + m
                mags.append(m)
                w = m if w is None else torch.minimum(w, m)   # ★전 마디 동시 접촉
            if w is None:                                     # wrap 센서가 없는 프로필
                w = zero
            # 프로필 규약: wrap_bodies = (중간 _3, 원위 _4). 마디가 하나뿐이면 둘을 같게
            # 둔다(2지 그리퍼 jaw — 그 접촉 자체가 감쌈). grasp_sensor 와 같은 규약.
            mids.append(mags[0] if mags else zero)
            dists.append(mags[1] if len(mags) > 1 else (mags[0] if mags else zero))
            tot.append(t)
            tipfs.append(_tf)
            tipvs.append(_tv)
            wrapped.append((w > w_thr).float())
        # 자매 tip_c 용 — 팁 마디만의 힘. 반환 계약(4-tuple)을 안 바꾸려고 속성에 둔다.
        self._tip_force = torch.stack(tipfs, 1)
        # 팁 힘 **world 벡터** (N,T,3) — obs 의 tip-local F/T 원료.
        self._tip_force_vec_w = torch.stack(tipvs, 1)
        return (torch.stack(tot, 1), torch.stack(wrapped, 1),
                torch.stack(mids, 1), torch.stack(dists, 1))

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _object_axis_w(self) -> torch.Tensor:
        """물체 local +z 를 world 로 (N,3) — 컵 축. 파지 자세 보상의 기준축이다."""
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(self.object.data.root_quat_w, z)

    def _object_up_and_tilt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(물체 로컬 +z · world +z = cos(기울기), 기울기[deg]).

        ★보상은 cos 을 **직접** 쓴다(자매 트랙 규약). deg 로 바꿨다가 다시 cos 을
          취하면 acos/cos 왕복이라 계산만 늘고 소각에서 수치오차가 붙는다.
          deg 는 종료 판정·로깅용으로만 남긴다.
        """
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        up = quat_apply(self.object.data.root_quat_w, z)[:, 2].clamp(-1.0, 1.0)
        return up, torch.rad2deg(torch.acos(up))

    def _object_tilt_deg(self) -> torch.Tensor:
        """물체 로컬 +z 와 world +z 사이 각도[deg]."""
        return self._object_up_and_tilt()[1]

    # ==================================================================
    def _get_observations(self) -> dict:
        from isaaclab.utils.math import matrix_from_quat

        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        idx = torch.cat([self._arm_t, self._hand_t])
        joint_pos, joint_vel = q[:, idx], qd[:, idx]

        # ★★KUKA 고정(08.25) 프레임 규약 — 원본 teacher 관측은 전부 로봇 베이스
        #   (env-local) 기준이다(hand_pos / object_pos / object_goal).
        object_pos = self._local(self.object.data.root_pos_w)          # (N,3)
        object_rot = self.object.data.root_quat_w                      # (N,4) critic 전용
        object_goal = self.goal_pos                                    # (N,3) env-local

        contact, _, _, _ = self._contact()                             # ★critic 전용
        # ★★KUKA 고정(08.25) 관측 노이즈. 원본은 전 항목이 *_noisy 이되 **ADR 로 0 에서
        #   시작**한다 — object_state_noise 축이 그 계수다.
        _ns = float(self.adr.get_param("object_state_noise", "scale"))
        if _ns > 0.0:
            def _u(x, w):        # 균일 ±w
                return x + (_ns * w) * 2.0 * (torch.rand_like(x) - 0.5)
            joint_pos = _u(joint_pos, float(self.cfg.obs_noise_joint_pos))
            joint_vel = _u(joint_vel, float(self.cfg.obs_noise_joint_vel))
            object_pos = _u(object_pos, float(self.cfg.obs_noise_object_pos))

        # ---- 손 TCP(palm_ee) 위치·자세 + 손끝 위치 ----------------------------------
        # ★TCP 는 `palm_ee` — 사용자 확인으로 **+x 축이 손바닥 법선**이다. 정렬(align)을
        #   보려면 위치가 아니라 이 축이 필요하다. 자세는 회전행렬의 x·z 열(6D)로 준다:
        #   quaternion 은 부호 이중성이 있고 euler 는 불연속이라 회귀 입력에 부적합하다.
        tcp_pos = (self.robot.data.body_pos_w[:, self._tcp_idx]
                   - self.scene.env_origins)                            # (N,3)
        _R = matrix_from_quat(self.robot.data.body_quat_w[:, self._tcp_idx])
        tcp_axes = torch.cat([_R[:, :, 0], _R[:, :, 2]], dim=-1)        # (N,6) x축·z축
        tip_pos = (self.robot.data.body_pos_w[:, self._tip_t]
                   - self.scene.env_origins[:, None, :])                # (N,T,3)

        # ★★KUKA 고정(08.25) fabric 계획 상태. 정책이 지시한 목표가 계획으로 어떻게
        #   풀렸는지 보지 못하면 "액션 → 실제 이동"이 부분관측이 된다.
        #   ※원본의 fabric_qd/qdd 는 `observation_annealing` 계수가 (0,0) 이라 **항상 0**
        #     이다(같은 계수가 원본의 joint_vel·hand_vel 까지 죽인다). 상수 0 을 LSTM 에
        #     통과시킬 이유가 없어 policy 에서 빼고 **critic 에만 실값**으로 준다.
        fabric_q = self.fabric_q

        # ── sim2real obs (08.26 재설계) ────────────────────────────────────────
        # tip-local F/T — 실기 손끝 wrench 토픽의 force 3축과 동형. ★world 프레임
        # 금지: 배포 시 변환이 어긋나면 조용히 틀린 obs 가 된다("손 obs zeros" 동형).
        _R_tips = matrix_from_quat(
            self.robot.data.body_quat_w[:, self._tip_t])              # (N,T,3,3)
        tip_ft = torch.einsum(
            "ntij,nti->ntj", _R_tips, self._tip_force_vec_w
        ) / float(self.cfg.contact_force_max)
        tip_ft = tip_ft.clamp(-1.0, 1.0).reshape(self.num_envs, -1)
        # 손 joint_pos_err = (지령 − 실측)/max — 인벨롭에서 팁 F/T 가 0 을 읽는
        # 구간의 주 힘 관측. 실기 = (보낸 지령 − joint_states), 추가 센서 불필요.
        _hand_cmd_robot = self._fabric_hand_cmd[:, self._hand_from_fab]
        hand_err = ((_hand_cmd_robot
                     - self.robot.data.joint_pos[:, self._hand_t])
                    / float(self.cfg.joint_pos_err_max)).clamp(-1.0, 1.0)
        parts = [joint_pos, joint_vel,
                 object_pos, object_goal, self.actions,
                 tcp_pos, tcp_axes, tip_pos.reshape(self.num_envs, -1),
                 fabric_q, self.palm_cmd, tip_ft, hand_err]
        if self._onehot is not None:
            parts.append(self._onehot)
        obs = torch.cat(parts, dim=1)
        # critic 은 배포하지 않으므로 실기에서 못 얻는 것까지 본다:
        #   물체 회전·접촉력 스칼라·물체 6D 속도·fabric qd/qdd·물체 scale 참값.
        state = torch.cat([obs, object_rot, contact.clamp(max=20.0),
                           self.object.data.root_lin_vel_w,
                           self.object.data.root_ang_vel_w,
                           self.fabric_qd, self.fabric_qdd,
                           self._object_scale], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _get_rewards(self) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat

        obj_pos = self._local(self.object.data.root_pos_w)
        palm_pos = self._local(self.robot.data.body_pos_w[:, self.palm_idx])
        tips = self.robot.data.body_pos_w[:, self._tip_t] - self.scene.env_origins[:, None, :]

        contact, wrapped, mid_f, dist_f = self._contact()
        # ★E. 임계 분리: 게이트(파지 성립)는 1.0N, **참여 판정**은 0.1N (v1/v2 동일).
        p_thr = float(self.cfg.participation_force_threshold)
        grip_frac = (contact > p_thr).float().mean(dim=1)
        # 진단 전용 — 08.22 엄격 감쌈(전 마디 동시 0.5N). 보상은 grasp_sensor 규약
        # (마디 하나라도 접촉)이므로 이 값은 **대조 지표**다: 같은 정책을 두 판정으로
        # 재면 0.503 vs 0.069 로 7배가 벌어졌다. 느슨한 쪽만 오르면 "받치기"다.
        strict_env = (wrapped[:, self._env_f].mean(dim=1)
                      if self.profile.has_wrap_sensors else grip_frac)

        # persistence — 보상에서는 빠졌지만(자매 트랙 규약) 축퇴 진단으로 계속 잰다.
        _gate_now = contact_gate(
            contact[:, self._grp_a], contact[:, self._grp_b],
            float(self.cfg.contact_force_threshold)).float()
        self._gate_hold = (self._gate_hold + 1.0) * _gate_now
        persistence = (self._gate_hold
                       / float(self.cfg.persistence_ref_steps)).clamp(0.0, 1.0)

        xy_disp = (obj_pos[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1)
        up_cos, tilt = self._object_up_and_tilt()
        goal_dist = (obj_pos - self.goal_pos).norm(dim=-1)

        # ★★08.23 보상 전면 교체 — 자매 트랙 grasp_sensor 와 **같은 함수**를 쓴다.
        #   구 7항이 실측으로 드러낸 결함 셋(전부 이 교체로 사라진다):
        #     · `envelope_mul_floor` 0.3 → 감쌈 없이도 이송 보상의 30% 가 흘렀다.
        #       좌팔이 감쌈 0.21 로 이송(goal 0.58)만 학습한 직접 원인 — 우선순위 ① 붕괴.
        #     · `grasp_quality` 의 grip/persist 30% 몫 → 감쌈이 아닌 것으로 채워졌다
        #       (우팔 실측 1.3 중 0.43).
        #     · `lift_success_height` 0.10 ≠ goal 0.15 → dz 10cm 에서 포화. 우팔이
        #       한때 18cm 까지 들었다가 6cm 로 되돌아와 고착했다(되돌려도 손해가 없다).
        #   grasp_sensor 는 셋 다 없다: g_eff 하한 없음 · envelope 순수항 · 분모=goal 높이.
        #   대향 파지점 approach 기하도 그쪽 `approach_reward` 가 내부에서 계산한다
        #   (A 그룹 min · B∩envelope mean — 우리 구판은 A 도 mean 이었고 pinky 를
        #    분모에 넣고 있었다).
        if True:
            # 단계형 보상 — 손 제어 방식과 **무관**하게 적용한다. "pd" 대조군도 같은
            # 보상으로 돌려야 손 제어 방식만의 차이를 본다(보상까지 바뀌면 원인 분리 불가).
            # 구 10 항은 관절 액션 전제라 세 곳이 어긋난다
            # (rewards_tip.py docstring: action_l2 부호·approach 커널·2 층 게이트).
            # 파지중심(world): 실제 palm 자세로 회전시킨 palm-local 상수.
            #   fabric 계획이 아니라 **실제** 자세를 쓴다 — 추종오차만큼 어긋나면
            #   보상이 계획 기준으로 후하게 나온다.
            _touch = contact > float(self.cfg.participation_force_threshold)
            if True:
                # 계층 보상(λμνρ) — 수식은 자매 공유 파일 하나뿐. 여기는 입력
                # 조립만 하며, 로봇 종속 적응(미러 부호·폐쇄도·표면 거리)이 모인다.
                from .rewards_stage import compute_stage_rewards
                _R_tcp = matrix_from_quat(
                    self.robot.data.body_quat_w[:, self._tcp_idx])
                # 파지중심 = palm 부착 상수(인벨롭 목표 — 손끝 평균이면 핀치가 접근 보상을 먹는다).
                _R_palm = matrix_from_quat(
                    self.robot.data.body_quat_w[:, self.palm_idx])
                _gc_palm = palm_pos + torch.einsum(
                    "bij,j->bi", _R_palm, self._grasp_center_local)
                # ── 접촉 마스크 — 자매와 **같은 단일 임계**(stage_contact_threshold).
                #   구판은 참여 0.1N/감쌈 1.0N 두 임계를 섞어 썼다 — 자매는 0.1N 하나다.
                _thr_c = float(self.cfg.stage_contact_threshold)
                _mid_c = mid_f > _thr_c
                _dist_c = dist_f > _thr_c
                _tip_c = self._tip_force > _thr_c
                _touch_c = (_mid_c | _dist_c | _tip_c)[:, self._usable_t]
                _deep_all = _mid_c & _dist_c
                _deep_c = _deep_all[:, self._wrap_f_t]
                # ── persist — 자매 규약: deep 접촉 ≥1 연속 스텝 / 기준스텝 ──────────
                _ndeep = _deep_c.sum(dim=-1)
                self._persist_buf = torch.where(
                    _ndeep >= 1, self._persist_buf + 1,
                    torch.zeros_like(self._persist_buf))
                _persist = (self._persist_buf.float()
                            / max(float(self.cfg.stage_contact_persistence_steps), 1.0)
                            ).clamp(max=1.0)
                # ── 코리더 래치 — 자매 08.26 승인분. 자매는 per-env difficulty 로
                #   보간하는데 우리는 그 축이 없어 **ADR 진행률**(전역)로 보간한다 —
                #   "난이도가 오르면 보상 요구도 조여진다"는 같은 방향이다.
                _cfr = float(self.adr.progress)
                _cor_xy = (float(self.cfg.stage_corridor_xy_m[0])
                           + _cfr * (float(self.cfg.stage_corridor_xy_m[1])
                                     - float(self.cfg.stage_corridor_xy_m[0])))
                _cor_tilt = (float(self.cfg.stage_corridor_tilt_deg[0])
                             + _cfr * (float(self.cfg.stage_corridor_tilt_deg[1])
                                       - float(self.cfg.stage_corridor_tilt_deg[0])))
                self._corridor_latch |= (xy_disp > _cor_xy) | (tilt > _cor_tilt)
                _ref_up = torch.zeros_like(palm_pos)
                _ref_up[:, 2] = 1.0          # 이 자산은 베이스 +z ≡ world +z
                total, terms, gate, envelope_frac = compute_stage_rewards(
                    palm_pos=palm_pos,
                    grasp_center_pos=_gc_palm,
                    object_pos=obj_pos,
                    goal_pos=self.goal_pos,
                    tip_c=_tip_c,
                    persist_frac=_persist,
                    wrap_c=(_mid_c | _dist_c)[:, self._grp_b_env_t],
                    deep_c=_deep_c,
                    oppose=(contact > _thr_c)[:, self._grp_a].any(dim=-1),
                    height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
                    tilt_deg=tilt,
                    xy_disp=xy_disp,
                    touch_c=_touch_c,
                    thumb_force=contact[:, self._grp_a].sum(dim=-1),
                    palm_x=_R_tcp[:, :, 0],
                    # ★로봇 적응: 좌손은 미러라 palm_y 가 아래를 향한다(부팅 실측
                    #   cos(palm_y,up) 우 +1 / 좌 −1). 수식(clamp^4)은 자매와 동일하게
                    #   두고 **입력에 부호를 곱해** 넘긴다 — roll_q≡0 사망 방지.
                    palm_y=self._palm_y_sign * _R_tcp[:, :, 1],
                    ref_up=_ref_up,
                    obj_up=matrix_from_quat(
                        self.object.data.root_quat_w)[:, :, 2],
                    obj_speed=self.object.data.root_lin_vel_w.norm(dim=-1),
                    corridor_ok=(~self._corridor_latch).float(),
                    # ★로봇 적응: 자매는 synergy 폐쇄 지령의 평균을 넘긴다. 우리는
                    #   tip-IK 라 **fabric 손 관절의 실제 폐쇄도**(홈→닫힘한계 정규화,
                    #   유효 관절 평균)로 같은 의미를 만든다 — close_bridge 는
                    #   λ 상태에서 "손을 오므리는 진행" 자체에 소액을 주는 항이다.
                    syn_close_thumb=(
                        ((self.fabric_q[:, self.profile.num_arm_joints:]
                          - self._fab_hand_home) / self._close_den)
                        .clamp(0.0, 1.0)[:, self._close_thumb_m].mean(dim=-1)),
                    syn_close_fingers=(
                        ((self.fabric_q[:, self.profile.num_arm_joints:]
                          - self._fab_hand_home) / self._close_den)
                        .clamp(0.0, 1.0)[:, self._close_fingers_m].mean(dim=-1)),
                    # ★손가락별 gradient — **지령** 손끝에서 컵 **표면**까지의 거리.
                    #   d = sqrt((radial-R)^2 + relu(|h|-H)^2), 컵 축 기준 분해.
                    #   실 손끝 기준은 하위호환으로 수식에 남아 있으나 넘기지 않는다.
                    tip_cmd_surf_dist=self._tip_cmd_surface_dist(obj_pos),
                    actions=self.actions,
                    prev_actions=self.prev_actions,
                    cfg=self.cfg,
                )
                enclosure_dist = terms["_d_gc"]
                # ── 단계 판정 — 자매 규약(08.26 동일 세팅): **에피소드 누적 hit**.
                #   순간 게이트 평균이 아니라 "이 에피소드에서 한 번이라도 열렸나"를
                #   리셋 때 기록한다. ⑤ 정지는 목표 5cm·직립 10°·저속을 hold_steps
                #   연속 유지해야 hit — 스쳐 지나가는 것을 정지로 안 센다.
                _goal_d_now = torch.norm(obj_pos - self.goal_pos, dim=-1)
                _spd_now = self.object.data.root_lin_vel_w.norm(dim=-1)
                _tr_now = ((_goal_d_now <= float(self.cfg.stage_stay_pos_tol_m))
                           & (tilt <= float(self.cfg.stage_stay_tilt_deg))
                           & (_spd_now < float(self.cfg.stage_stay_speed_ref)))
                self._stay_run = torch.where(
                    _tr_now, self._stay_run + 1, torch.zeros_like(self._stay_run))
                self._stage_hit |= torch.stack([
                    terms["_lam"] > 0.5,
                    terms["_mu"] > 0.5,
                    terms["_nu"] > 0.5,
                    terms["_rho"] > 0.5,
                    self._stay_run >= int(self.cfg.stage_stay_hold_steps),
                ], dim=1)
                # 순간 게이트도 남긴다(이름 분리) — hit 는 리셋 주기라 굼뜨다.
                for _k, _tag in (("_lam", "approach"), ("_mu", "grasp"),
                                 ("_nu", "lift"), ("_rho", "transfer")):
                    self.extras[f"task/gate_now/{_tag}"] = terms[_k].mean()
                # 진단 키는 자매 함수의 것 그대로다(_grasp_q, _touch_frac, …).
                for _k, _tag in (("_grasp_q", "grip_q"), ("_touch_frac", "touch_f"),
                                 ("_deep_frac", "deep_f"), ("_align", "palm_align"),
                                 ("_orient_q", "orient_q"), ("_U", "upright_q"),
                                 ("_U_tol", "tilt_tol_q"), ("_H", "lift_q"),
                                 ("_S", "still_q"), ("_opp_soft", "opp_soft"),
                                 ("_envelope", "stage_envelope"),
                                 ("_z_ok", "z_ok"), ("_dz_gc", "dz_gc"),
                                 ("_dxy_gc", "dxy_gc"),
                                 ("_succ_soft", "succ_soft")):
                    self.extras[f"task/{_tag}"] = terms[_k].mean()
                # 코리더 — 몰수가 얼마나 걸리는지 상설 감시(자매 규약).
                self.extras["task/corridor_ok"] = (
                    (~self._corridor_latch).float().mean())
                _rdq = xy_disp / float(self.cfg.stage_disp_limit)
                self.extras["task/disp_q"] = (1.0 / (1.0 + _rdq * _rdq)).mean()
                self.extras["task/corridor_xy_bound_m"] = torch.as_tensor(_cor_xy)
                self.extras["task/corridor_tilt_bound_deg"] = torch.as_tensor(_cor_tilt)
                # ── 자세·물체 위치를 **보상과 같이** 읽는다(08.26 사용자 요청) ──
                #   보상 숫자만으로는 "어떤 자세에서 그 값이 나왔는지"를 못 본다.
                #   실제로 orient_q 0.95·align 0.99 가 **손날 자세**였고, 지표만
                #   보고는 3 주기를 정상으로 오독했다. palm_ee 6-DOF 와 컵 위치를
                #   같은 스텝에 남겨 보상과 나란히 읽는다.
                #   ★핵심은 `normal_yaw_err_deg` — 손바닥 법선이 컵에서 방위각으로
                #     몇 도 어긋났는가. 수정 전 보상 최적해가 **52.8°** 였다.
                #   ★기준점 차이를 적어둔다: 보상의 `align` 은 **palm 원점**에서,
                #     이 로깅은 **palm_ee**(48.8mm 앞)에서 잰다. 같은 벡터(palm_x)를
                #     보지만 원점이 달라 몇 도 차이가 난다. 보상 자신의 각도는
                #     `acos(task/palm_align)` 이다.
                from isaaclab.utils.math import euler_xyz_from_quat
                _tcp_pos = self._local(
                    self.robot.data.body_pos_w[:, self._tcp_idx])
                _rr, _pp, _yy = euler_xyz_from_quat(
                    self.robot.data.body_quat_w[:, self._tcp_idx])

                def _deg(a):                       # (-180,180] 로 감아 평균이 튀지 않게
                    return torch.atan2(torch.sin(a), torch.cos(a)) * (180.0 / math.pi)

                _nx = _R_tcp[:, :, 0]              # 손바닥 법선
                _to = obj_pos - _tcp_pos
                _toh, _nxh = _to.clone(), _nx.clone()
                _toh[:, 2] = 0.0
                _nxh[:, 2] = 0.0
                _yaw_err = torch.rad2deg(torch.acos(
                    torch.nn.functional.cosine_similarity(
                        _nxh, _toh, dim=-1, eps=1e-6).clamp(-1.0, 1.0)))
                _pitch = torch.rad2deg(torch.asin(_nx[:, 2].clamp(-1.0, 1.0)))
                for _tag, _v in (
                    ("palm/x", _tcp_pos[:, 0]), ("palm/y", _tcp_pos[:, 1]),
                    ("palm/z", _tcp_pos[:, 2]),
                    ("palm/roll_deg", _deg(_rr)), ("palm/pitch_deg", _deg(_pp)),
                    ("palm/yaw_deg", _deg(_yy)),
                    ("obj/x", obj_pos[:, 0]), ("obj/y", obj_pos[:, 1]),
                    ("obj/z", obj_pos[:, 2]),
                    ("palm/to_obj_mm", _to.norm(dim=-1) * 1000.0),
                    ("palm/normal_yaw_err_deg", _yaw_err),
                    ("palm/normal_pitch_deg", _pitch),
                ):
                    self.extras[f"task/pose/{_tag}"] = _v.mean()
                # ★★side-to-side 접근을 **직독**한다 — palm_ZX 평면이 world_z 와
                #   수직인가(= palm_y 가 연직인가). 이전에는 평균 orient_q 에서
                #   역산해야 했는데, 비선형 곱의 평균이라 per-env 분포를 못 봤다.
                #   p95 를 같이 남긴다 — 평균만 보면 꼬리가 숨는다.
                # ZX 기울기 — 수식이 자매 것이라 diag 키가 없다. env 에서 직접 계산
                #   (같은 palm_y 입력 — 부호는 |cos| 라 미러 무관).
                _zx = torch.rad2deg(torch.acos(
                    torch.nn.functional.cosine_similarity(
                        _R_tcp[:, :, 1], _ref_up, dim=-1, eps=1e-6
                    ).abs().clamp(0.0, 1.0)))
                self.extras["task/pose/palm/zx_tilt_deg"] = _zx.mean()
                self.extras["task/pose/palm/zx_tilt_p95_deg"] = torch.quantile(
                    _zx, 0.95)
                for _k in [k for k in terms if k.startswith("_")]:
                    terms.pop(_k)
            # ── 감쌈 로깅: 손가락별 **손바닥면** 감쌈(원하는 값)과 방향무관 접촉,
            #    그리고 필터 전 값 — raw 와의 차이가 곧 손등 접촉 비중이다.
            self.extras["task/envelope_frac_raw"] = envelope_fraction_graded(
                mid_f[:, self._env_f], dist_f[:, self._env_f],
                float(self.cfg.contact_force_threshold)).mean()
            _thr = float(self.cfg.contact_force_threshold)
            _wrap_pal = 0.5 * ((mid_f > _thr).float() + (dist_f > _thr).float())
            for _i, _fg in enumerate(self._fingers):
                self.extras[f"task/wrap/{_fg}"] = _wrap_pal[:, _i].mean()
                self.extras[f"task/touch/{_fg}"] = _touch[:, _i].float().mean()
            # ── 접근 **방향** 진단 (08.24) ────────────────────────────────────────
            # 보상은 거리(`d_grasp`)만 보므로 위에서 덮든 옆에서 오든 점수가 같다.
            # 측면 파지가 성립하려면 ①손바닥이 컵을 향하고 ②palm 접근축이 컵 축에
            # 수직이며 ③손끝이 컵 둘레로 퍼져야 한다. 셋을 상시 로깅해 "접근은
            # 하는데 파지가 안 되는" 구간에서 자세 탓인지 바로 가른다.
            from isaaclab.utils.math import quat_apply as _qa
            _pq = self.robot.data.body_quat_w[:, self.palm_idx]
            _pp = self.robot.data.body_pos_w[:, self.palm_idx]
            _obj_w = obj_pos + self.scene.env_origins
            _gc_dir = torch.nn.functional.normalize(
                self._grasp_center_local, dim=0).unsqueeze(0).expand(self.num_envs, 3)
            _fwd = _qa(_pq, _gc_dir)                       # palm 접근축(world)
            _to_obj = torch.nn.functional.normalize(_obj_w - _pp, dim=-1)
            # ① 접근축과 (컵−palm) 사이 각[deg]. 0=정면.
            # ★주의 — 이 "접근축"은 **파지중심 오프셋 방향**(법선에서 52.8°)이다.
            #   보상의 align 도 같은 오프셋 방위를 본다(법선 판은 d_gc=0 과 양립
            #   불가로 롤백됨). 법선 기준 값은 normal_yaw_err_deg 쪽.
            #   법선 기준 값은 `task/pose/palm/normal_yaw_err_deg` 를 봐라.
            self.extras["dir/palm_to_obj_deg"] = torch.rad2deg(torch.acos(
                (_fwd * _to_obj).sum(-1).clamp(-1.0, 1.0))).mean()
            # ② 접근축 ↔ 컵 축 각[deg]. 측면 파지=90° 근처 · 위에서 덮기=0° 근처.
            _cup = self._object_axis_w()
            self.extras["dir/palm_vs_cup_axis_deg"] = torch.rad2deg(torch.acos(
                (_fwd * _cup).sum(-1).abs().clamp(max=1.0))).mean()
            # ③ 손끝의 컵 둘레 방위각 폭[deg] — 감싸면 크고 한쪽에서 밀면 작다.
            _v = (tips + self.scene.env_origins[:, None, :]) - _obj_w[:, None, :]
            _vp = _v - (_v * _cup[:, None, :]).sum(-1, keepdim=True) * _cup[:, None, :]
            _e1 = torch.nn.functional.normalize(_vp[:, 0, :], dim=-1)
            _e2 = torch.cross(_cup, _e1, dim=-1)
            _phi = torch.atan2((_vp * _e2[:, None, :]).sum(-1),
                               (_vp * _e1[:, None, :]).sum(-1))
            self.extras["dir/tip_azimuth_span_deg"] = torch.rad2deg(
                _phi.max(dim=1).values - _phi.min(dim=1).values).mean()
        else:
            total, terms, gate, envelope_frac = compute_grasp_sensor_rewards(
                palm_pos=palm_pos,
                fingertip_pos=tips,
                object_pos=obj_pos,
                goal_pos=self.goal_pos,
                group_a_force=contact[:, self._grp_a],
                group_b_force=contact[:, self._grp_b],
                group_a_tip_idx=self._grp_a,
                group_b_env_tip_idx=self._grp_b_env,
                env_mid_force=mid_f[:, self._env_f],
                env_dist_force=dist_f[:, self._env_f],
                object_tilt_deg=tilt,
                object_up=up_cos,
                height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
                actions=self.actions,
                prev_actions=self.prev_actions,
                cfg=self.cfg,
            )
            enclosure_dist = terms["_d_side"]
        up_q = up_cos.clamp(0.0, 1.0) ** float(self.cfg.upright_exponent)
        # ★지터는 prev_actions 갱신 **전에** 재야 한다(갱신 후엔 항상 0).
        #   절대 액션이라 정책 노이즈가 곧 목표 순간이동이 된다 — 이 값이 크면
        #   Fabrics 가 저역통과시켜 정책 액션이 실제 palm 위치에 거의 영향을 못 준다.
        self.extras["action/arm_step_delta"] = (
            self.actions[:, :6] - self.prev_actions[:, :6]).abs().mean()
        self.extras["action/hand_step_delta"] = (
            self.actions[:, 6:] - self.prev_actions[:, 6:]).abs().mean()
        self.extras["action/arm_abs_mean"] = self.actions[:, :6].abs().mean()
        # ★슬루를 걸면 "액션 지터"와 "실제 지령 지터"가 갈린다. 액션은 여전히 떨 수
        #   있지만 지령이 안 떨면 팔은 정상 추종한다 — 그 분리를 봐야 판정이 된다.
        self.extras["action/cmd_step_mm"] = (
            self.palm_cmd[:, :3] - self._prev_cmd[:, :3]).norm(dim=-1).mean() * 1000.0
        self._prev_cmd.copy_(self.palm_cmd)
        self.prev_actions.copy_(self.actions)
        # ★★08.23 성공 판정 3조건 AND (자매 트랙 규약): goal 근접 AND 감쌈 AND 직립.
        #   구 판정은 goal 거리 하나뿐이라 **감쌈 0.21 로 이송만 한** 좌팔이 0.58 로
        #   집계됐다 — 지표가 우선순위 ①②를 못 보고 있었다. ADR 승급 기준도 이 값이다.
        _pass_pos = goal_dist < float(self.cfg.success_pos_tolerance)
        # ★08.26 동일 세팅 — stage 모드의 감쌈 하한은 트랙 전용(자매 규약: 공유 계약
        #   상수와 분리). env_frac 도 자매 함수의 4번째 반환(wrap4)이라 의미가 같다.
        _env_min = float(self.cfg.stage_success_envelope_min
                         if True
                         else self.cfg.success_envelope_min)
        _pass_env = envelope_frac >= _env_min
        _pass_tilt = tilt < float(self.cfg.success_tilt_max_deg)
        self._goal_reached_now = _pass_pos & _pass_env & _pass_tilt
        # ★조건별 개별 통과율 — AND 만 보면 어느 조건이 병목인지 알 수 없다.
        self.extras["task/pass_pos"] = _pass_pos.float().mean()
        self.extras["task/pass_envelope"] = _pass_env.float().mean()
        self.extras["task/pass_tilt"] = _pass_tilt.float().mean()
        self.extras["task/goal_reached_loose"] = _pass_pos.float().mean()

        # ---- ADR 승급 — 자매 08.25 완화 규약(08.26 동일 세팅) -----------------------
        # ★goal 위치까지 요구하면 성공률 0 으로 난이도가 한 번도 안 오른다(자매 실측:
        #   lstm_test5~7 difficulty 0.0000 고착). 승급은 **리프트 성공**(들었고·감쌈
        #   절반·파지 게이트)으로 완화하고, 엄격 판정은 task/goal_reached 로 계속 본다.
        #   ADR 이 안 오르면 코리더(느슨한 시작값)도 영영 안 조여진다 — 같은 축이다.
        if True:
            self._lift_success_now = (
                (obj_pos[:, 2] - self.object_spawn_pos[:, 2] >= 0.05)
                & (envelope_frac >= 0.5) & gate)
        else:
            self._lift_success_now = self._goal_reached_now
        self.extras["task/lift_success"] = self._lift_success_now.float().mean()
        if self.adr.maybe_increment(self._lift_success_now.float().mean()):
            em = getattr(self, "event_manager", None)
            if em is not None:
                em.reset(env_ids=self.robot._ALL_INDICES)
                em.apply(env_ids=self.robot._ALL_INDICES, mode="reset",
                         global_env_step_count=0)

        # ---- 로깅 ----------------------------------------------------------------
        for k, v in terms.items():
            # `_` 로 시작하는 키는 보상이 아니라 진단 원값이다(d_palm/d_side/gate_eff).
            self.extras[(f"task/{k[1:]}" if k.startswith("_") else f"reward/{k}")] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/contact_gate"] = gate.float().mean()
        self.extras["task/envelope_frac"] = envelope_frac.mean()
        # ★대조: 전 마디 동시접촉(0.5N) 기준. 느슨한 쪽만 오르면 "받치기"다.
        self.extras["task/envelope_strict"] = strict_env.mean()
        self.extras["task/grip_frac"] = grip_frac.mean()
        self.extras["task/persistence"] = persistence.mean()
        # 손가락이 풀린 env 비율 — 0 이면 접근 단계에 갇혀 파지를 시도조차 못 한다.
        self.extras["task/enclosure_dist"] = enclosure_dist.mean()
        self.extras["task/upright_quality"] = up_q.mean()
        # ★접촉력 **원값**. 이게 없으면 grip_frac=0 일 때 "진짜 미접촉"인지
        #   "임계(contact_force_threshold) 바로 아래"인지 구분할 수단이 없다
        #   — fab_test1 에서 실제로 그 구분이 안 돼 probe 를 따로 돌려야 했다.
        _best = contact.max(dim=1).values
        self.extras["contact/force_max"] = contact.max()
        self.extras["contact/force_best_finger"] = _best.mean()
        _g_thr = float(self.cfg.contact_force_threshold)
        self.extras["contact/n_over_thr"] = (contact > _g_thr).float().sum(dim=1).mean()
        self.extras["contact/n_over_tenth"] = (contact > _g_thr * 0.1).float().sum(dim=1).mean()
        self.extras["contact/group_a"] = contact[:, self._grp_a].max(dim=1).values.mean()
        self.extras["contact/group_b"] = contact[:, self._grp_b].max(dim=1).values.mean()
        # 손끝 최소거리 — approach 의 합만으론 "닿기 직전"인지 알 수 없다
        self.extras["task/tip_min_dist"] = (
            (tips - obj_pos[:, None, :]).norm(dim=-1).min(dim=1).values.mean()
        )
        self.extras["task/palm_dist"] = (palm_pos - obj_pos).norm(dim=-1).mean()
        # 손가락 상호 최소거리 — 마디 반경 ~10mm 이므로 20mm 미만이면 겹침이다.
        # 이 값이 작으면 envelope_frac 이 물리적으로 불가능한 감쌈을 세고 있다는 뜻.
        _fp = self.robot.data.body_pos_w[:, self._fl_t]
        _fd = torch.cdist(_fp, _fp).masked_fill(~self._fl_diff.unsqueeze(0), float("inf"))
        _fmin = _fd.view(self.num_envs, -1).min(dim=1).values
        self.extras["hand/finger_min_dist"] = _fmin.mean()
        self.extras["hand/overlap_frac"] = (_fmin < 0.020).float().mean()
        self.extras["task/object_height_delta"] = (
            obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean()
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/goal_reached"] = self._goal_reached_now.float().mean()
        # ★측정 공백 해소: grasp_v1 은 이 값을 보상에 쓰면서 로깅하지 않았다.
        self.extras["obj/xy_displacement"] = xy_disp.mean()
        self.extras["obj/xy_disp_p95"] = torch.quantile(xy_disp, 0.95)
        self.extras["obj/tilt_deg"] = tilt.mean()
        # Fabrics 추종 오차 — 정상상태 오차(78mm 이력)를 학습 중에도 본다.
        perr = (self.palm_targets[:, :3] - palm_pos).norm(dim=-1)
        self.extras["fabric/palm_err_mean"] = perr.mean()
        self.extras["fabric/palm_err_p95"] = torch.quantile(perr, 0.95)
        # ★★위 perr 은 **계산 오차 + 물리 추종**의 합이다. 둘은 성질이 완전히 달라
        #   섞어 보면 진단이 안 된다 — A 가 크면 fabric 이 목표를 못 푼 것(제어기
        #   설계 문제), B 가 크면 로봇이 계획을 못 따라간 것(관성·게인·토크).
        #   fabric 은 IK 솔버가 아니라 2 차 동역학계라 A 도 0 이 아니다.
        #   ★★_palm_frame(fabric taskmap) 출력은 **이미 로봇 베이스 기준**이다 — world
        #   로 보고 _local() 로 env_origins 를 빼면 43m 어긋난다(실측 43,285mm).
        #   판별 근거: perr(=|palm_targets − palm_pos|)가 26mm 로 성립하므로
        #   palm_targets 와 palm_pos 는 같은 env-local 계이고, 여기에 _local() 을 한 번
        #   더 먹인 _plan 만 튀었다. fabric 은 단일 로봇 모델을 전 env 가 공유한다.
        _plan, _ = self._palm_frame(self.fabric_q)
        self.extras["fabric/plan_err_mean"] = (
            self.palm_targets[:, :3] - _plan).norm(dim=-1).mean()      # A 정책목표→계획
        self.extras["fabric/track_err_mean"] = (
            _plan - palm_pos).norm(dim=-1).mean()                      # B 계획→실제
        # ★"정책이 컵을 겨냥은 하는가"와 "겨냥은 하는데 못 가는가"를 가른다.
        #   target_to_obj 작고 palm_dist 크면 → 추종/도달 문제
        #   target_to_obj 자체가 크면        → 정책이 컵을 안 겨냥 (탐색 문제)
        if self._hand_fabric:
            # ★손 제어를 두 층으로 갈라 잰다 — 어디서 막히는지 이 둘이 가른다.
            #   cmd_err : 정책이 지시한 관절 목표 → fabric attractor 가 실제로 만든
            #             `fabric_q`. 크면 **fabric 내부**가 목표를 못 따라간다
            #             (처방: hand_attractor_gain 상향).
            #   track_err: `fabric_q` → PhysX PD 가 실현한 실제 관절. 크면 **물리 쪽**
            #             (stiffness/effort)이 못 따라간다.
            #   손끝 IK 가 막혔던 곳이 정확히 cmd_err 에 해당한다(추종오차 85mm).
            _fq_hand = self.fabric_q[:, self.profile.num_arm_joints:]
            _cmd = self._fabric_hand_cmd
            self.extras["hand/cmd_err_rad"] = (_fq_hand - _cmd).abs().mean()
            # ★외전 이탈 — hand_control="pd" 는 외전을 init 에 **하드 고정**하지만
            #   "fabric" 은 fabric 이 손 20개를 전부 소유하므로 외전도 움직일 수 있다.
            #   목표는 init 으로 주지만 body_repulsion·joint_limit·cspace attractor 가
            #   함께 작용하면 밀려난다. 손가락이 벌어지면 감쌈이 약해진다
            #   (실측 fmin: fabric 27~28mm vs pd 23~24mm).
            _froz = self._frozen_t
            if _froz.numel() > 0:
                self.extras["hand/abduction_dev_rad"] = (
                    self.robot.data.joint_pos[:, _froz]
                    - self._default_q[:, _froz]).abs().mean()
            self.extras["hand/track_err_rad"] = (
                self.robot.data.joint_pos[:, self._hand_t]
                - _fq_hand[:, self._hand_from_fab]).abs().mean()
        self.extras["fabric/target_to_obj"] = (
            self.palm_targets[:, :3] - obj_pos).norm(dim=-1).mean()
        tau = self.robot.data.applied_torque[:, self._hand_t].abs()
        self.extras["debug/hand/torque_mean"] = tau.mean()
        self.extras["debug/hand/torque_max"] = tau.max()
        self.extras.update(self.adr.log_dict())
        self.extras["adr/goal_xy_radius"] = self.adr.get_param("goal", "xy_radius")
        self.extras["adr/trigger_metric"] = self._goal_reached_now.float().mean()

        # ── 학습 로그(stdout)에 주기적으로 한 줄 남긴다 ────────────────────────
        # TFEvents 를 파싱하지 않고 `grep METRICS <train.log>` 만으로 추적할 수 있게.
        # ★접촉 관련 원값을 반드시 포함한다 — 0 이 "진짜 0"인지 "임계 아래"인지
        #   구분 못 해 fab_test1~4 를 날린 이력이 있다.
        self._log_tick = getattr(self, "_log_tick", 0) + 1
        _every = int(getattr(self.cfg, "console_log_interval", 600))
        if _every > 0 and self._log_tick % _every == 0:
            print(
                f"[METRICS] step={self._log_tick:>8d}"
                f" rew={total.mean():+.3f}"
                f" approach={terms['approach'].mean():+.3f}"
                f" gate={gate.float().mean():.3f}"
                f" grip={grip_frac.mean():.3f}"
                f" env={envelope_frac.mean():.3f}"
                # ★max 만 찍으면 오독한다 — 2048x5 중 하나의 스파이크가 전형값처럼 보인다.
                #   실제로 그랬다(513N 보고 → 전형은 20N). 전형값을 먼저 놓는다.
                f" F={_best.mean():.1f}N/Fmax={contact.max():.0f}N"
                f" nF>thr={(contact > _g_thr).float().sum(dim=1).mean():.2f}"
                f" dz={(obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean():+.4f}"
                f" goal={self._goal_reached_now.float().mean():.3f}"
                f" tip={((tips - obj_pos[:, None, :]).norm(dim=-1).min(dim=1).values).mean():.3f}"
                f" tgt2obj={(self.palm_targets[:, :3] - obj_pos).norm(dim=-1).mean():.3f}"
                f" perr={perr.mean():.3f}"
                # ★prev_actions 는 위에서 이미 갱신됐다 — 여기서 다시 빼면 **항상 0** 이다.
                #   extras 에 미리 담아둔(갱신 전 계산) 값을 쓴다. fab_test5 로그의
                #   jit 필드가 그래서 0 으로 찍혔다(TFEvents 쪽은 정상).
                f" jit={float(self.extras['action/arm_step_delta']):.3f}"
                f" fmin={float(self.extras['hand/finger_min_dist'])*1000:.0f}mm"
                f" ovl={float(self.extras['hand/overlap_frac']):.2f}",
                flush=True,
            )
        return total

    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Fabrics 의 JointLimitRepulsion 이 관절한계를 담당하므로 한계 초과는 종료 사유가
        # 아니다.
        # ★08.22 리스폰 → **종료** 전환(grasp_v1 `_get_dones` 와 동일 구성:
        #   fallen | out_xy | tipped | 물리위반). 쓰러진/떨어진 컵을 방치하면 회복 불가
        #   상태의 전이가 배치를 희석하고 value 오차가 GAE 로 번진다.
        # ★감시: agn_test2 에서는 같은 종료가 "접근 회피"를 가르쳤다(reaching 0.056→0.005
        #   + episode_lengths 상승). 그 시그니처(approach↓ + eplen↑ 동시)가 보이면
        #   이 결정을 되짚는다. grasp_v1 은 같은 종료로 98% 까지 갔으므로 선례는 있다.
        qd = self.robot.data.joint_vel[:, self._arm_t]
        runaway = (qd.abs() > float(self.cfg.runaway_joint_vel)).any(dim=-1)

        obj = self._local(self.object.data.root_pos_w)
        fallen = obj[:, 2] < float(self.cfg.object_min_z)
        out_xy = (obj[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1) > float(
            self.cfg.object_out_of_bounds_xy)
        tipped = self._object_tilt_deg() > float(self.cfg.tipping_termination_deg)

        terminated = runaway | fallen | out_xy | tipped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        self.extras["term/fallen"] = fallen.float().mean()
        self.extras["term/out_xy"] = out_xy.float().mean()
        self.extras["term/tipped"] = tipped.float().mean()
        return terminated, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        p = self.profile

        # ★★KUKA 고정(08.25) `robot_spawn` ADR — 리셋 자세/속도에 노이즈를 준다.
        #   같은 홈에서만 출발하면 정책이 그 궤적 하나에 과적합된다(원본은 0 → 0.35 rad).
        # ★★08.27 팔은 **항상 홈**에서 시작한다. 역순 커리큘럼은 컵을 옮겨서 만든다
        #   (아래 스폰 보간) — 팔을 IK 로 텔레포트하던 이전 판이 h7 우팔 데드락의
        #   원인이었다. 시작 자세가 하나뿐이므로 시작분포가 성공 지표에 묶이지 않는다.
        q0 = self._default_q[env_ids].clone()
        _pn = float(self.adr.get_param("robot_spawn", "joint_pos_noise"))
        _vn = float(self.adr.get_param("robot_spawn", "joint_vel_noise"))
        qd0 = torch.zeros_like(q0)
        if _pn > 0.0:
            q0 = q0 + _pn * 2.0 * (torch.rand_like(q0) - 0.5)
            q0 = q0.clamp(self.robot.data.soft_joint_pos_limits[env_ids, :, 0],
                          self.robot.data.soft_joint_pos_limits[env_ids, :, 1])
        if _vn > 0.0:
            qd0 = _vn * 2.0 * (torch.rand_like(qd0) - 0.5)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        # ★원본은 속도 목표도 리셋 상태로 되돌린다 — 안 하면 직전 에피소드 목표가 샌다.
        self.robot.set_joint_velocity_target(qd0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[:, self._hand_t]
        # 자매 보상 상태 — 에피소드 경계에서 반드시 지운다. 코리더 래치가
        # 넘어가면 이전 에피소드의 위반이 새 에피소드의 ν 를 몰수한다.
        self._persist_buf[env_ids] = 0
        self._corridor_latch[env_ids] = False
        # 단계 hit 는 **리셋 때** 기록한다(자매 규약 — 에피소드 단위 지표).
        for _i, _nm in enumerate(("approach", "grasp", "lift", "transfer", "stay")):
            self.extras[f"task/stage/{_nm}"] = (
                self._stage_hit[env_ids, _i].float().mean())
        self._stage_hit[env_ids] = False
        self._stay_run[env_ids] = 0
        # ★원본과 동일: fabric 상태를 리셋된 **실제** 관절 상태로 동기화한다.
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = qd0[:, self._fab_t]
        self.fabric_qdd[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._gate_hold[env_ids] = 0.0
        # ★슬루 지령도 홈으로 되돌린다 — 안 하면 이전 에피소드 지령이 새어 들어온다.
        self.palm_cmd[env_ids] = self.home_palm[env_ids]
        self._goal_reached_now[env_ids] = False

        rng = self.adr.get_param("spawn", "xy_range")
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng
        spawn = torch.zeros(n, 3, device=self.device)
        # ★08.26 트랙 전용 스폰 오버라이드(도달 지도) — 프로필 상수는 자매 공유라
        #   여기서 갈아끼운다. cfg 는 (x, |y|) 로 적고 y 부호는 프로필 스폰 중심의
        #   부호(좌우 미러)를 따른다.
        _ovr_sp = getattr(self.cfg, "object_spawn_center_override", None)
        if _ovr_sp is not None:
            _sy = 1.0 if float(p.object_spawn_center[1]) >= 0.0 else -1.0
            _scx, _scy = float(_ovr_sp[0]), _sy * abs(float(_ovr_sp[1]))
        else:
            _scx, _scy = p.object_spawn_center[0], p.object_spawn_center[1]
        # ★★역순 커리큘럼(08.27) — **컵**이 손 앞에서 테이블 스폰으로 물러난다.
        #   f = adr("reset_near","frac") ∈ [0,1] : 0 = 홈 파지중심 바로 아래(Z 만 내리면
        #   되는 상태) → 1 = 테이블 스폰(만렙 = 배포 분포 계약). 팔은 항상 홈이다.
        #   ★스폰 노이즈는 보간 **뒤**에 더한다 — f=0 에서도 시작 다양성이 살아 있어야
        #     한다(다양성을 성공에 게이팅하면 나쁜 시드가 영구 고착한다: h7 우팔).
        _f = (float(self.adr.get_param("reset_near", "frac"))
              if bool(getattr(self.cfg, "curriculum_reset_near", True)) else 1.0)
        spawn[:, 0] = (1.0 - _f) * self._cup_near_xy[0] + _f * _scx + offs[:, 0]
        spawn[:, 1] = (1.0 - _f) * self._cup_near_xy[1] + _f * _scy + offs[:, 1]
        spawn[:, 2] = self._object_rest_z[env_ids] + float(self.cfg.object_spawn_pad)
        self.object_spawn_pos[env_ids] = spawn
        # ---- goal: 스폰과 독립 오프셋 (반경은 ADR 축, 0 이면 구 고정 goal 과 동치) ----
        # ★스폰의 결정론적 함수로 두면 goal obs 가 변별력 없는 입력이 돼 정책이 무시한다
        #   (배포에서 사용자 지정 위치에 반응하지 않음). 이송 학습의 전제 조건.
        g_xy = self.adr.get_param("goal", "xy_radius")
        g_z = self.adr.get_param("goal", "z_radius")
        goal = spawn.clone()
        goal[:, 2] += float(self.cfg.goal_height_offset)
        goal[:, :2] += (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * g_xy
        goal[:, 2] += (torch.rand(n, device=self.device) - 0.5) * 2.0 * g_z
        # palm 박스 안쪽으로 클램프 — 박스 밖 goal 은 도달 불가(액션 포화 학습 재발 경로).
        _m = float(self.cfg.goal_box_margin)
        # palm_lo/hi 는 (1,6) — env 브로드캐스트로 클램프한다
        goal = torch.max(torch.min(goal, self.palm_hi[0, :3] - _m), self.palm_lo[0, :3] + _m)
        self.goal_pos[env_ids] = goal

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        # ★★KUKA 고정(08.25) `object_spawn.rotation` ADR — x·y 축 회전 노이즈(0 → 1 rad).
        #   자세가 항상 같으면 정책이 한 접근각에만 맞춘다. z 축은 컵이 회전대칭이라 뺀다.
        _rot = float(self.adr.get_param("spawn", "rotation"))
        if _rot > 0.0:
            _ax = torch.zeros(n, 3, device=self.device)
            _ax[:, :2] = _rot * 2.0 * (torch.rand(n, 2, device=self.device) - 0.5)
            _ang = _ax.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            root[:, 3:7] = torch.cat(
                [torch.cos(0.5 * _ang), (_ax / _ang) * torch.sin(0.5 * _ang)], dim=-1)
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
