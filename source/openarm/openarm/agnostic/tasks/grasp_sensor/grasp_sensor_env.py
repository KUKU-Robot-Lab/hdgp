"""robot-agnostic grasp-sensor 환경 (direct).

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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, matrix_from_quat

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

from .grasp_sensor_env_cfg import GraspSensorEnvCfg
from .rewards import compute_grasp_sensor_rewards
from .rewards_tip_cyl import compute_tip_cyl_rewards
from .robot_profiles import PROFILES

_GRAVITY = 9.81


class GraspSensorEnv(DirectRLEnv):
    cfg: GraspSensorEnvCfg

    def __init__(self, cfg: GraspSensorEnvCfg, render_mode: str | None = None, **kwargs):
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
        # 커리큘럼 승급 전용(리프트 성공) — goal 위치까지 요구하면 승급이 영구 정지한다.
        self._lift_success_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # _get_dones 가 매 스텝 갱신, _get_rewards 가 같은 스텝에 재사용(dones 가 먼저다)
        self._tilt_deg_buf = torch.zeros(self.num_envs, device=self.device)
        # ★접촉 지속 카운터(grasp_v1 reward_contact_hold_buf). 끊기면 0 으로 리셋되므로
        #   "닿았다 뗐다"로는 못 채운다. 리셋 때 반드시 0 으로 — 안 하면 이전 에피소드의
        #   지속치가 새 에피소드에 그대로 보상으로 지급된다.
        self._persist_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obj_up_buf = torch.ones(self.num_envs, device=self.device)
        # ★★단계별 성공 플래그(08.25 사용자 요청). env 마다 **에피소드 동안 OR 누적**
        #   하고 리셋 시점에만 평균을 기록한다 — 스텝마다 안 찍으므로 비용이 없다.
        #   approach 성공을 **거리가 아니라 접촉 발생**으로 정의한 게 의도적이다:
        #   물체 크기가 달라지면 "닿기 직전 거리"가 달라져 하드 임계가 다물체에서 깨진다.
        self._stage_names = ("approach", "grasp", "lift", "transfer", "stay")
        self._stage_hit = torch.zeros(
            self.num_envs, len(self._stage_names), dtype=torch.bool, device=self.device)
        self._stay_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        # 손바닥면 법선(감쌈 = 손바닥 접촉만 인정). 프로필 미정의는 fail-loud —
        # 기본축을 가정하면 판정이 **조용히 뒤집혀** 손등 파지를 감쌈으로 계속 센다.
        if self.cfg.require_palmar_contact:
            _missing = [f for f in fingers if f not in p.palmar_axis_local]
            if _missing:
                raise RuntimeError(
                    f"[{p.name}] palmar_axis_local 미정의: {_missing} — "
                    "손바닥/손등 구분 불가. URDF 의 cross(굴곡축, 장축)으로 실측하거나 "
                    "cfg.require_palmar_contact=False 로 구 판정(크기만)을 명시할 것"
                )
            self._palmar_axes = torch.tensor(
                [p.palmar_axis_local[f] for f in fingers], device=self.device, dtype=torch.float32)
            # 마디별 body id — 감쌈 판정에 쓰는 (중간, 원위) 두 링크만.
            _bn = self.robot.body_names
            self._wrap_body_ids = torch.tensor(
                [[_bn.index(b[0]), _bn.index(b[1] if len(b) >= 3 else b[0])]
                 for b in (p.finger_sensor_bodies[f] for f in fingers)],
                device=self.device, dtype=torch.long)   # (F, 2) = (mid, dist)
        # 그룹 인덱스를 fingertip_bodies 에도 그대로 쓴다(접근 보상) — 두 목록의
        # 손가락 순서가 같아야 성립하므로 fail-loud 로 강제한다.
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"finger_sensor_bodies({len(fingers)}) 의 손가락 수가 달라 그룹 인덱스를 공유할 수 없다"
            )

        self._init_home_palm()

        if self._hand_fabric:
            print(f"[grasp_sensor] 손 제어=fabric(direct, gain={self.cfg.hand_attractor_gain}) "
                  f"· hand_repulsion={bool(self.cfg.use_hand_repulsion)} "
                  f"· PhysX self-collision={self.cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions}",
                  flush=True)

        print(f"[grasp_sensor] profile={p.name} arm={len(self.arm_ids)} hand={len(self.hand_ids)} "
              f"tips={len(self.tip_ids)} action={self.cfg.action_space} obs={self.cfg.observation_space} "
              f"fabric={p.fabric_robot_dir}",
              flush=True)

    # ------------------------------------------------------------------
    def _tip_palm_frame(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fabric FK 로 (palm 원점 (N,3), 회전 (N,3,3) — 열이 palm x/y/z 축).

        프레임 이름을 태스크가 알 필요가 없다 — fabric 의 "palm" taskmap(7점)이
        원점·세 축 보조점을 자기 규약으로 뽑는다(자매 트랙 `_palm_frame` 과 동일).
        """
        pts, _ = self.fabric.get_taskmap("palm")(q, None)
        if pts.shape[1] < 18:
            raise RuntimeError("palm taskmap 이 1점 모드 — tip_cyl 은 6-DOF palm 규약 필요")
        o = pts[:, :3]
        ax = torch.stack([
            torch.nn.functional.normalize(pts[:, 3:6] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 9:12] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 15:18] - o, dim=1),
        ], dim=-1)
        return o, ax

    def _setup_synergy(self) -> None:
        """시너지 그립 배선 — 관절 목표를 직접 보간해 파워그립을 구조적으로 보장한다.

        ★★관절 순서 함정(이 트랙에서 실제로 밟았다): 프로필의 자세 배열은
          **finger-major**([thumb_1..4, index_1..4, …])인데 articulation 은
          **관절번호-major**(index_1, middle_1, pinky_1, ring_1, thumb_1, index_2, …)다.
          슬라이스로 대응시키면 손 전체가 **조용히** 엉뚱한 자세로 움직인다.
          여기서 이름으로 한 번만 매핑하고, 이후 전부 이 인덱스를 쓴다.
        """
        p = self.profile
        for _f in ("hand_joint_names", "hand_open_pose", "hand_grip_pose"):
            if not getattr(p, _f):
                raise RuntimeError(
                    f"[{p.name}] hand_control='synergy' 인데 프로필에 {_f} 가 없다")
        n = len(p.hand_joint_names)
        if len(p.hand_open_pose) != n or len(p.hand_grip_pose) != n:
            raise RuntimeError(
                f"[{p.name}] 자세 배열 길이 불일치: names {n} / open "
                f"{len(p.hand_open_pose)} / grip {len(p.hand_grip_pose)}")
        jn = self.robot.data.joint_names
        # 프로필 순서 k → articulation 인덱스. **이름으로** 찾는다.
        self._syn_ids = [jn.index(nm) for nm in p.hand_joint_names]
        self._syn_open = torch.tensor(p.hand_open_pose, device=self.device)
        self._syn_grip = torch.tensor(p.hand_grip_pose, device=self.device)
        # 관절 k 를 모는 채널과, 그 관절이 속한 손가락.
        fingers = list(p.finger_sensor_bodies.keys())
        ch, fi = [], []
        for nm in p.hand_joint_names:
            _sfx = nm.rsplit("_", 1)[1]
            if _sfx not in p.hand_channel_of_joint:
                raise RuntimeError(f"[{p.name}] hand_channel_of_joint 에 접미사 {_sfx} 없음")
            ch.append(int(p.hand_channel_of_joint[_sfx]))
            _hit = [i for i, f in enumerate(fingers) if f"_{f}_" in nm]
            if len(_hit) != 1:
                raise RuntimeError(f"[{p.name}] 관절 {nm} 의 손가락을 특정 못함: {_hit}")
            fi.append(_hit[0])
        self._syn_ch = torch.tensor(ch, device=self.device, dtype=torch.long)
        self._syn_fi = torch.tensor(fi, device=self.device, dtype=torch.long)
        self._syn_nch = len(set(ch))
        self._syn_freeze = torch.tensor(
            [nm.rsplit("_", 1)[1] in p.hand_freeze_suffixes for nm in p.hand_joint_names],
            device=self.device)
        # 폐쇄도 버퍼 — 관절별 독립 진행도. 접촉 동결이 관절마다 따로 걸리므로 관절 단위다.
        self._syn_close = torch.zeros(self.num_envs, n, device=self.device)
        # 손 PD 속도 피드포워드용 — 첫 스텝의 이전 목표는 홈 자세(속도 0)다.
        self._syn_target = self.robot.data.joint_pos[:, self._syn_ids].clone()
        self._syn_vel = torch.zeros(self.num_envs, n, device=self.device)
        self._policy_dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        _lim = self.robot.data.soft_joint_pos_limits[0, self._syn_ids, :]
        self._syn_lo, self._syn_hi = _lim[:, 0].contiguous(), _lim[:, 1].contiguous()
        _grip_clamped = self._syn_grip.clamp(self._syn_lo, self._syn_hi)
        print(f"[grasp_sensor] synergy: 관절 {n}개 · 채널 {self._syn_nch} · "
              f"동결 {int(self._syn_freeze.sum())}개 · "
              f"grip 한계clamp {int((self._syn_grip != _grip_clamped).sum())}개", flush=True)

    def _synergy_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """액션(손가락×채널) → 관절 목표. 프로필 순서 (N, n)."""
        p = self.profile
        nf = len(p.finger_sensor_bodies)
        a = a_hand.view(self.num_envs, nf, self._syn_nch)
        if bool(getattr(self.cfg, "couple_four_fingers", False)):
            # 대향 그룹(엄지)은 독립, 나머지는 채널별 평균 — "특정 손가락만 안 닫힘"을
            # 액션 공간에서 제거한다. 접촉 동결은 관절별로 남아 형상 적응은 유지된다.
            _a_idx = self._group_a_idx
            _mask = torch.ones(nf, dtype=torch.bool, device=a.device)
            _mask[_a_idx] = False
            _common = a[:, _mask, :].mean(dim=1, keepdim=True)
            a = torch.where(_mask.view(1, nf, 1), _common.expand(-1, nf, -1), a)
        cmd = 0.5 * (a.clamp(-1.0, 1.0) + 1.0)                    # 절대 폐쇄도 [0,1]
        cmd_j = cmd[:, self._syn_fi, self._syn_ch]                # (N, n) 관절로 전개
        rate = float(self.cfg.synergy_close_speed)
        delta = (cmd_j - self._syn_close).clamp(-rate, rate)      # 변화율 상한(감소 가능)
        if bool(getattr(self.cfg, "synergy_contact_freeze", True)):
            _c = self._contact_forces_split()
            _thr = float(self.cfg.stage_contact_threshold)
            # 그 손가락의 원위마디 또는 팁이 닿았으면 동결 대상 관절을 멈춘다.
            _tipf = self._tip_contact_forces()
            _hold = ((_c[1] > _thr) | (_tipf > _thr))[:, self._syn_fi]   # (N, n)
            delta = delta * (~(_hold & self._syn_freeze)).float()
        self._syn_close = (self._syn_close + delta).clamp(0.0, 1.0)
        tgt = torch.lerp(self._syn_open.unsqueeze(0), self._syn_grip.unsqueeze(0),
                         self._syn_close)
        return tgt.clamp(self._syn_lo.unsqueeze(0), self._syn_hi.unsqueeze(0))

    def _tip_contact_forces(self) -> torch.Tensor:
        """손가락별 **팁만** 접촉력 (N, F). 규약: finger_sensor_bodies 마지막 원소=팁."""
        out = []
        for finger in self._finger_names:
            s = self._finger_sensors[finger][-1]
            fm = s.data.force_matrix_w
            out.append(fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1))
        return torch.stack(out, dim=1)

    def _setup_tip_cyl(self) -> None:
        """원통 액션의 기하 상수를 부팅 시 **홈 자세 FK 로 실측**한다(하드코딩 금지 —
        자산이 바뀌면 파지 기하도 바뀐다).

        원통 규약(probe_tip_workspace_cyl 08.24 실측, 32,768표본):
          축 = palm **y**(손가락 θ 산포 최소 20~21°, x/z 는 31~42°)
          원점 = 홈 손끝 5점의 (x,z)평면 중심 = 파지 중심(실측 palm+(91,118)mm)
          θ 공칭 = 홈 자세 각도 — 엄지 −84° vs 검지·중지·약지 63~73°(대향 ~150°),
                   넷은 z(축방향)로 분리(+27/+2/−22mm). "θ 고정"이 겹침을 구조로 차단.
        """
        # ★fabric taskmap 은 batch_size=num_envs 로 고정 컴파일 — 배치 1 로 부르면
        #   reshape 이 깨진다. 전 env 가 같은 홈이므로 전체 배치로 부르고 [0] 만 쓴다.
        q0 = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        o, R = self._tip_palm_frame(q0)
        tips, _ = self.fabric._fingertip_taskmap(q0, None)
        rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2),
                           tips.reshape(self.num_envs, 5, 3) - o[:, None, :])[0]  # (5,3)
        # 평면 = (x,z)=인덱스(0,2), 축 = y=인덱스1
        c = rel[:, [0, 2]].mean(dim=0)                                   # 파지 중심 (2,)
        d = rel[:, [0, 2]] - c
        self._cyl_center = c                                             # palm-local (x,z)
        self._cyl_theta = torch.atan2(d[:, 1], d[:, 0])                  # (5,) 손가락별 공칭각
        self._cyl_z_nom = rel[:, 1].clone()                              # (5,) 축방향 공칭
        self._tip_span = float(self.cfg.tip_action_span) * float(self.cfg.tip_diameter)
        self._tip_target_w = torch.zeros(self.num_envs, 15, device=self.device)

        # ★파지중심을 **palm body 프레임 상수**로 환산해 둔다(보상 ① 의 기준점).
        #   fabric q 는 *지령*이라 추종오차(fabric/joint_err_max 실측 0.65~1.02rad)만큼
        #   실측과 어긋난다 — 보상은 실측 palm 을 따라야 하므로 여기서 한 번만 환산하고
        #   런타임엔 실측 body pose 로 복원한다. c 는 fabric palm 프레임 기준이므로
        #   **같은 프레임에서** 월드로 올린 뒤 body 프레임으로 내린다(프레임 혼용 금지).
        # ★`_wrap_body_ids` 는 이 함수보다 **뒤에** 만들어지므로 여기서 존재를 볼 수 없다.
        #   대신 그것을 만드는 조건(cfg 플래그)을 본다 — tip_cyl 보상 ② 는 마디 위치가 필수.
        if not self.cfg.require_palmar_contact:
            raise RuntimeError(
                f"[{self.profile.name}] tip_cyl 보상은 마디(mid/dist) 위치가 필요하다 — "
                "cfg.require_palmar_contact=True 여야 _wrap_body_ids 가 생성된다"
            )
        # ★상수 정합 fail-loud. 이 저장소에는 오타로 손실항이 12,652 iter 동안 조용히
        #   꺼져 있던 이력이 있다 — 신설 필드는 전부 여기서 존재·정합을 검증한다.
        _c = self.cfg
        # ★★08.25 5단계 재편 어서션. 구 grasp_quality 4항 합(=1) 검사는 폐기했다 —
        #   G 가 `five_frac · near_q` 로 바뀌어 혼합비 자체가 없다.
        #   저장소에 "오타로 손실항이 12,652 iter 동안 조용히 꺼져 있던" 이력이 있어
        #   신설 상수는 전부 존재·범위를 부팅에서 확인한다(fail-loud).
        for _nm in ("stage_grasp_near_tau", "stage_stay_speed_ref", "stage_stay_hold_steps",
                    "stage_perp_exponent", "stage_roll_exponent", "stage_orient_floor",
                    "stage_transfer_weight", "stage_stay_weight"):
            if not hasattr(_c, _nm):
                raise RuntimeError(f"[{self.profile.name}] 5단계 상수 누락: {_nm}")
        if not (0.0 <= float(_c.stage_orient_floor) < 1.0):
            raise RuntimeError(
                f"[{self.profile.name}] stage_orient_floor={_c.stage_orient_floor} ∉ [0,1)")
        if float(_c.stage_grasp_near_tau) <= 0 or float(_c.stage_stay_speed_ref) <= 0:
            raise RuntimeError(
                f"[{self.profile.name}] near_tau/stay_speed_ref 는 양수여야 한다")
        # ★가중 사다리가 단조 증가여야 한다 — 인자가 깊어질수록 곱이 작아지므로
        #   상한이 안 커지면 뒤 단계가 앞 단계를 못 이긴다(lstm_test8: 곱이 0.008 로 소멸).
        _ladder = [float(_c.stage_approach_weight), float(_c.stage_grasp_weight),
                   float(_c.stage_lift_weight), float(_c.stage_transfer_weight),
                   float(_c.stage_stay_weight)]
        if any(b < a for a, b in zip(_ladder, _ladder[1:])):
            raise RuntimeError(
                f"[{self.profile.name}] 단계 가중이 단조 증가가 아니다: {_ladder}")
        if float(_c.stage_success_height) > float(_c.goal_height_offset):
            raise RuntimeError(
                f"[{self.profile.name}] stage_success_height({_c.stage_success_height}) 가 "
                f"goal_height_offset({_c.goal_height_offset}) 보다 크다 — 성공 도달 불가")
        if float(_c.stage_lift_height_ref) > float(_c.goal_height_offset) + 1e-9:
            raise RuntimeError(
                f"[{self.profile.name}] stage_lift_height_ref 가 목표 높이를 넘는다 — "
                "포화점이 목표 밖이면 목표 근처 gradient 가 0 이다")
        # ★`_group_b_env_idx` 는 이 함수보다 뒤에 만들어진다(앞의 _wrap_body_ids 와 같은
        #   순서 함정) — 프로필에서 직접 센다.
        _p = self.profile
        _nb = len([f for f in _p.contact_group_b if f in _p.envelope_fingers])
        _q = float(_c.stage_success_envelope_min) * _nb
        if abs(_q - round(_q)) > 1e-6:
            raise RuntimeError(
                f"[{self.profile.name}] success_envelope_min({_c.success_envelope_min}) 가 "
                f"1/{_nb} 의 정수배가 아니다 — 도달 불가 임계가 된다")
        # ★파지중심 = **대향 중점**(엄지 팁과 4지 팁 중심의 중점). 5점 단순평균 c 는
        #   엄지 1 대 4지 4 의 가중이라 4지 쪽으로 0.3·(엄지−4지중심) 만큼 치우친다.
        #   그 치우친 점에 컵을 두면 4지는 표면 안쪽까지 파고들고(실측 31.6N) 엄지는
        #   r 바닥(≈63mm)이 표면 밖 18mm 라 **영원히 못 닿는다**(실측 thumb touch 0.00).
        #   대향 중점으로 옮기면 실측상 엄지 100% 접촉 + 4지 감쌈 1.00 이 동시 성립한다
        #   (오프셋 스윕 T-1: 28mm 에서 G 최대, 유도값 0.3×|엄지−4지| ≈ 30mm 와 일치).
        _fg = list(self.profile.finger_sensor_bodies.keys())
        _ti = _fg.index(self.profile.contact_group_a[0])
        _oth = [i for i in range(len(_fg)) if i != _ti]
        _opp_mid = 0.5 * (rel[_ti] + rel[_oth].mean(dim=0))              # (3,) palm-local
        # 파지중심의 fabric-palm-local 좌표(축=y, 평면=(x,z)) — 보상 ① 의 기준점.
        # ★fabric↔env-local 원점 정합은 여기서 못 한다: 이 시점의 body_pos_w 는 **stale**
        #   이다(로봇이 아직 홈에 안 놓임 — _init_home_palm 주석의 함정). 정합은
        #   물리를 돌린 뒤 _init_home_palm 끝에서 손끝 5점으로 실측한다.
        # ★★대향 중점 자체를 쓰면 안 된다(08.25 실측으로 반증). 홈 자세는 손이 **펴져**
        #   있어 엄지가 멀리 뻗어 있고, 그 중점(5점평균 대비 72mm)은 4지가 전혀 못 닿는
        #   허공이다(오프셋 스윕: 60mm 에서 이미 wrap4=0.00). 파지는 손가락이 **오므렸을
        #   때** 만나는 지점에서 일어나며 그것은 5점평균에 훨씬 가깝다.
        #   실측 최적 = 28mm = 대향중점까지 거리의 0.39 — 이 비율을 상수로 둔다.
        _c3 = torch.tensor([float(c[0]), float(self._cyl_z_nom.mean()), float(c[1])],
                           device=self.device)
        # ★★08.25 파지중심을 **자유 컵 실측값**으로 덮어쓴다. 구 유도(대향중점 보간
        #   frac=0.67 → [55, 2, 103]mm)는 컵을 매 스텝 `write_root_state_to_sim` 으로
        #   **붙잡아 놓고** 잰 것이라 무효였다. 컵이 자유롭게 움직이면 손이 닫히면서
        #   컵을 쓸어내려 실제로 무는 지점이 훨씬 손바닥 쪽이다:
        #     닫힘 후 컵 palm-local — ff=0 [58, 1, 64]mm · ff=1 [56, −3, 64]mm
        #     (두 독립 측정에서 z=64mm 일치. 구 값과 z 차이 −39mm)
        #   ★대향중점 보간선은 이 점을 지나지 않는다(어떤 frac 으로도 재현 불가) —
        #     그래서 frac 조정이 아니라 직접 지정이다.
        #   자산·손 제어·폐쇄 속도가 바뀌면 probe_seqclose 로 반드시 재측정할 것.
        _ovr = getattr(self.cfg, "stage_gc_local_override", None)
        if _ovr is not None:
            self._gc_local = torch.tensor([float(v) for v in _ovr], device=self.device)
            _src = "자유 컵 실측(override)"
        else:
            self._gc_local = _c3 + float(
                self.cfg.stage_gc_opposition_frac) * (_opp_mid - _c3)
            _src = f"대향중점 보간 frac={float(self.cfg.stage_gc_opposition_frac):.2f}"
        _shift = float(torch.norm(self._gc_local - _c3))
        if not (0.010 <= _shift <= 0.120):
            raise RuntimeError(
                f"[{self.profile.name}] 파지중심 이동량 {_shift*1000:.0f}mm 이 상식 범위"
                "(10~120mm) 밖이다 — 자산이 바뀌었으면 probe_seqclose 로 재실측할 것")
        print(f"[grasp_sensor] 파지중심 palm-local "
              f"{[round(float(v) * 1000) for v in self._gc_local]}mm "
              f"(5점평균 대비 {_shift * 1000:.0f}mm · {_src})", flush=True)
        _th = (self._cyl_theta * 180 / math.pi).tolist()
        print(f"[grasp_sensor] tip_cyl 기하: 중심 palm+({float(c[0])*1000:.0f},{float(c[1])*1000:.0f})mm "
              f"θ°={[round(v) for v in _th]} z₀mm={[round(float(v)*1000) for v in self._cyl_z_nom]} "
              f"r={float(self.cfg.tip_r_center)*1000:.0f}±{self._tip_span*1000:.0f}mm", flush=True)

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

        if self._tip_cyl or self._synergy:
            # ★fabric FK 프레임과 sim env-local 은 **원점이 다르다**(실측 544mm — 첫
            #   배선에서 palm body 프레임과 섞어 파지중심이 632~728mm 로 나왔다).
            #   같은 물리점(손끝 5개)을 양쪽에서 읽어 상수 오프셋을 실측한다. 회전까지
            #   다르면 평행이동으로 못 잇으므로 산포를 보고 fail-loud.
            q0f = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
            tips_fab = self.fabric._fingertip_taskmap(q0f, None)[0].reshape(
                self.num_envs, 5, 3)[0]
            tips_sim = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                        - self.scene.env_origins[:, None, :])[0]
            delta = tips_sim - tips_fab                                   # (5,3)
            spread = float(delta.std(dim=0).max())
            if spread > 2e-3:
                raise RuntimeError(
                    f"[{self.profile.name}] fabric↔env 프레임이 순수 평행이동이 아니다 "
                    f"(손끝 오프셋 산포 {spread * 1000:.1f}mm > 2mm) — 회전 정합 필요"
                )
            self._fab_to_env = delta.mean(dim=0)                          # (3,) 상수
            print(f"[grasp_sensor] fabric→env 오프셋 = "
                  f"{[round(float(v) * 1000) for v in self._fab_to_env]}mm "
                  f"(산포 {spread * 1000:.2f}mm) · 파지중심 palm-local "
                  f"{[round(float(v) * 1000) for v in self._gc_local]}mm", flush=True)

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
        print(f"[grasp_sensor] 홈 palm={[round(v, 4) for v in home.tolist()]} | "
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
    def _palm_ee_R(self) -> torch.Tensor:
        """palm_ee 회전행렬 (N,3,3). 열 0 = **손바닥 법선(+x)**, 열 1 = +y.

        ★실측(probe_palmee): `r_hl_palm` / `r_hl_palm_ee` / `r_hl_palm_alias` /
          `r_hl_base` 네 body 가 **회전은 완전히 동일**하고 위치만 다르다(palm_ee 가
          palm 보다 40mm 앞). 따라서 자세 항에는 palm_idx 를 그대로 써도 같다.
          홈에서 +x = [0.002, 1.000, −0.002] 로 컵 축(+z)과 −0.0025 = 이미 수직이다.
        """
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])

    def _base_up_vec(self) -> torch.Tensor:
        """로봇 **베이스** +z 의 월드 방향 (N,3) — 자세 항의 기준축.

        ★★08.25 자세 기준을 컵(`obj_up`) → **베이스**로 바꿨다(사용자 지적).
          컵 기준이면 **컵이 기울수록 기울인 접근이 정당해지는 되먹임**이 생긴다:
          손이 밀어 컵을 기울임 → 기준축이 따라 기움 → 그 자세가 perp/roll 만점 →
          더 밀어도 벌점 없음. 실제로 lstm_test11 에서 컵이 0.54° → 7.93° 로 단조
          증가했다. 베이스는 안 움직이므로 그 고리가 구조적으로 끊긴다.
        ★프로필이 베이스를 기울여 장착해도 맞도록 root_quat 에서 계산한다
          (현 자산은 rot=[1,0,0,0] 이라 월드 +z 와 같다).
        """
        return quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))

    def _obj_up_vec(self) -> torch.Tensor:
        """물체 local +z 의 월드 방향 (N,3) — 컵 축. 인식 pose 로 실기에서도 나온다.

        ★`_obj_up_buf` 는 `_get_dones` 에서 갱신되는데 그건 `_get_rewards` **뒤**라
          한 스텝 낡은 값이다. 보상은 여기서 즉석 계산한 값을 쓴다.
        """
        return quat_apply(
            self.object.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))

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
        # 손 제어 경로 — cfg.hand_control 주석에 세 모드의 실측 근거가 있다.
        _hc = str(self.cfg.hand_control)
        if _hc not in ("pd", "fabric", "tip_cyl", "synergy"):
            raise RuntimeError(
                f"hand_control={_hc!r} 미지원. 'pd'/'fabric'/'tip_cyl'/'synergy' 만."
            )
        self._hand_fabric = _hc == "fabric"
        self._tip_cyl = _hc == "tip_cyl"
        self._synergy = _hc == "synergy"
        if self._synergy:
            self._setup_synergy()
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
            # ★★`use_hand_fabric` 이 False 면 `construct_fabric` 이 `add_hand_fabric()` 을
            #   **아예 부르지 않는다** — hand_mode 를 아무리 direct 로 줘도 손 attractor 가
            #   없어 fabric_q 의 손 구간이 홈에서 안 움직인다. 08.23 실측으로 잡았다:
            #   완전 폐합을 명령해도 손가락 구 최소거리가 반발 ON/OFF 양쪽 24.82mm 로
            #   **완전히 동일**했고 min==mean 이었다(= 아무것도 안 변한 것).
            #   hand_mode 는 add_hand_fabric **안에서** 맵을 고르는 인자일 뿐이다.
            use_hand_fabric=self._hand_fabric,
            # tip_cyl: 손가락별 손끝 attractor 5개(각자 그 손가락 4관절만 — 간섭이
            # metric 층에서 사라진다. 실측: 겹침 목표에서도 타 손가락 2.6~4.6mm).
            tip_per_finger=self._tip_cyl,
            # ★손 20-DOF 를 fabric 이 소유한다(hand_mode="direct" = 20x20 항등).
            #   앞의 팔 7열이 0 이라 LinearMap 의 Jacobian 이 그대로 마스킹 —
            #   손 제어가 팔을 미는 결합이 구조적으로 생길 수 없다.
            hand_mode="direct" if self._hand_fabric else "pca",
            hand_attractor_gain=self.cfg.hand_attractor_gain,
            use_hand_repulsion=bool(self.cfg.use_hand_repulsion),
            # ★Kuka 패턴의 body 반발 쌍(손↔팔뚝)을 실제로 건다. 공유 fabric 클래스의
            #   기본값은 False 라 이 인자를 주는 트랙만 거동이 바뀐다.
            use_body_repulsion_pairs=bool(self.cfg.use_body_repulsion_pairs),
            robot_dir_name=p.fabric_robot_dir,
            robot_name=p.fabric_robot_dir,
            **({"fabric_params_filename": p.fabric_params_filename}
               if p.fabric_params_filename else {}),
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
        # 역치환(fabric 손 구간 순서 → articulation hand 순서). fabric 이 손을 소유하면
        # 이 방향으로 목표를 되읽는다.
        self._fab_to_hand_local = torch.argsort(self._hand_to_fab_local)
        # ★synergy 자세(프로필 finger-major 순서) → fabric 손 구간 순서.
        #   `_setup_synergy` 는 이 블록보다 **먼저** 실행되므로 여기서 만든다(순서 함정).
        #   articulation 인덱스로 대조해 이름 기반 매핑을 유지한다.
        if getattr(self, "_synergy", False):
            _syn_pos = {int(j): k for k, j in enumerate(self._syn_ids)}
            _fab_hand = self._fab_t[p.num_arm_joints:].tolist()
            _missing = [int(j) for j in _fab_hand if int(j) not in _syn_pos]
            if _missing:
                raise RuntimeError(
                    f"[{p.name}] synergy 자세에 없는 fabric 손 관절 {_missing} — "
                    "hand_joint_names 가 손 20관절을 모두 덮어야 한다")
            self._syn_to_fab_idx = torch.tensor(
                [_syn_pos[int(j)] for j in _fab_hand],
                device=self.device, dtype=torch.long)
        self.fabric_q = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, self.fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        # use_hand_fabric=False 라 무시되지만 원본 계약(B,5 PCA)은 지킨다.
        # direct 모드는 손 20-DOF 를 그대로, pca 경로는 원본 계약(B,5)을 지킨다.
        self._fabric_hand_cmd = torch.zeros(
            self.num_envs, p.num_hand_joints if self._hand_fabric else 5, device=self.device)
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

        # 파지중심(_gc_local)은 손 제어 모드와 무관한 **손 기하 상수**라 synergy 도 필요하다.
        if self._tip_cyl or self._synergy:
            self._setup_tip_cyl()
        if not p.palm_box_verified:
            print(f"[grasp_sensor] ⚠ palm_box 미검증({p.name}) — P-2 로 도달성 확인 후 승격할 것",
                  flush=True)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

        # ---- 팔: **절대 목표**(DEXTRAH 원본과 동일) ----------------------------------
        # ★★08.25 델타 누산 → 절대 매핑. DEXTRAH 는 스케일·누산을 아예 쓰지 않는다:
        #     palm_pose_targets = compute_absolute_action(a, PALM_POSE_MINS, PALM_POSE_MAXS)
        #     scale(x, lo, hi) = 0.5·(x + 1)·(hi − lo) + lo
        #   누산 방식의 문제는 **목표가 상태**라는 것이다. 액션이 한동안 포화하면 목표가
        #   박스 끝까지 달아나고, 되돌리는 데 다시 여러 스텝이 걸린다(이력 의존).
        #   실측: arm_pos_scale 0.01 에서 액션 1.0 의 달성률 28.2% · palm_err 151mm,
        #   arm_rot_scale 0.05 에서 달성률 13.8~25.3% · euler 오차 최대 84°.
        #   초과분은 이동이 아니라 목표 인플레로만 쌓여 그 구간 액션에 gradient 가 없다.
        #   절대 매핑에는 이 상태가 없다 — 목표는 항상 박스 안이고 한 스텝에 어디로든
        #   간다. **속도 제한은 fabric 의 attractor 동역학이 맡는다**(그게 fabric 의 역할).
        #   ★a=0 은 "유지"가 아니라 **박스 중심**이다. DEXTRAH 도 그 성질을 알고 박스를
        #     중심이 쓸만한 자세가 되도록 잡는다(open_tesollo 주석: "action=0 → palm z
        #     target=0.425 … z_max=0.95 면 center=0.625 라 팔이 올라가는 문제 수정").
        #     우리 박스 중심 = (0.375, −0.165, 0.45) · 회전 (90, 0, 90)° = 홈 회전과 일치.
        #   구 `arm_pos_scale`·`arm_rot_scale` 은 이 경로에서 쓰이지 않는다.
        self.palm_targets = (
            0.5 * (self.actions[:, :6] + 1.0) * (self._palm_hi - self._palm_lo)
            + self._palm_lo)

        # ---- 손 ---------------------------------------------------------------------
        if self._synergy:
            # 관절공간 시너지 — 목표를 직접 보간하므로 말아 쥐는 것이 보장된다.
            _syn_prev = self._syn_target
            self._syn_target = self._synergy_targets(self.actions[:, 6:])
            # ★★손 속도 피드포워드(08.25 3차 감사). Kuka 는 팔·손 **23관절 전체**에
            #   속도 목표를 준다. 우리 손은 fabric 밖(램프)이라 `fabric_qd` 를 쓸 수
            #   없다 — fabric_q 의 손 구간을 매 스텝 덮어쓰므로 그 qd 는 램프의
            #   도함수가 아니다. 램프 자체의 도함수를 쓴다(정책 dt = decimation·sim.dt).
            self._syn_vel = (self._syn_target - _syn_prev) / self._policy_dt
            # ★fabric 의 손 상태를 실제 손 자세로 덮어쓴다. 안 그러면 fabric 이
            #   **다른 손**으로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다
            #   (자매 트랙 경고). use_hand_fabric=False 라 `_fabric_hand_cmd`(PCA 5D)는
            #   무시되므로, grasp_v1 처럼 상태를 직접 동기화하는 것이 유일한 경로다.
            #   프로필 순서(finger-major) → fabric 순서 이름 매핑을 반드시 거친다.
            self.fabric_q[:, self.profile.num_arm_joints:] = self._syn_to_fab(
                self._syn_target)
        elif self._tip_cyl:
            # 원통 좌표 (r, z) × 5 → palm-local xyz → 월드(현재 fabric palm 프레임).
            # 절대 매핑(누산 아님): a=±1 → 공칭 ± span. θ 는 부팅 실측 공칭값 고정.
            a = self.actions[:, 6:].view(self.num_envs, 5, 2)
            r = (float(self.cfg.tip_r_center) + a[:, :, 0] * self._tip_span).clamp(min=0.005)
            zc = self._cyl_z_nom[None, :] + a[:, :, 1] * self._tip_span
            local = torch.empty(self.num_envs, 5, 3, device=self.device)
            local[:, :, 0] = self._cyl_center[0] + r * torch.cos(self._cyl_theta)[None, :]
            local[:, :, 2] = self._cyl_center[1] + r * torch.sin(self._cyl_theta)[None, :]
            local[:, :, 1] = zc
            # ★기준은 지령 palm 이 아니라 **fabric 이 실제로 도달한 palm** — 지령을 쓰면
            #   추종오차만큼 손끝 목표가 컵에서 어긋나고 그 오차는 파지 순간에 최대다
            #   (자매 트랙 규약).
            o, R = self._tip_palm_frame(self.fabric_q.detach())
            self._tip_target_w = (
                o[:, None, :] + torch.einsum("bij,bkj->bki", R, local)
            ).reshape(self.num_envs, 15)
        else:
            # 절대 목표 누산(자유 관절만) + 관절한계 clamp
            _delta = torch.zeros_like(self.hand_targets)
            _delta[:, self._hand_free_local] = self.actions[:, 6:] * float(self.cfg.hand_joint_scale)
            self.hand_targets = (self.hand_targets + _delta).clamp(self._hand_lo, self._hand_hi)

        # ---- Fabrics: 목표 주입 + 적분 (정책 스텝당 **한 번**) ------------------------
        if self._hand_fabric:
            # ★손 20개를 **전부** fabric 에 넘긴다(정책이 안 건드리는 외전 포함 —
            #   hand_targets 는 init 값으로 시작해 자유 관절만 누산되므로 그대로 맞다).
            #   일부만 넘기면 fabric 이 아는 손과 실제가 어긋나 hand_repulsion 이
            #   **틀린 형상으로** 회피한다 — 그게 이 배선으로 고치려는 결함이다.
            #   fabric_q 의 손 구간은 덮어쓰지 않는다: 이제 fabric 이 손의 plant 다.
            self._fabric_hand_cmd = self._hand_to_fabric(self.hand_targets)
        elif not self._tip_cyl:
            # 구 배선: fabric 은 손을 FK 로 보기만 한다. 손 자세를 안 넣으면 손이 접힌
            # 줄 모르고 없는 충돌을 피하려 팔을 민다.
            self.fabric_q[:, self.profile.num_arm_joints:] = self._hand_to_fabric(self.hand_targets)
        self.fabric.set_features(
            self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self._world_ids, self._world_indicator, self._fabric_damping,
            tip_target=self._tip_target_w if self._tip_cyl else None,
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

    def _syn_to_fab(self, syn_q: torch.Tensor) -> torch.Tensor:
        """synergy 자세(프로필 finger-major 순서) → fabric 손 구간 순서."""
        return syn_q[:, self._syn_to_fab_idx]

    def _apply_action(self) -> None:
        # Fabrics 가 만든 참조 궤적을 팔 PD 목표로. fabric_q 는 **오픈루프 plant** —
        # 실측 관절로 되돌려 동기화하면 팔이 명령을 못 따라간다(E1 사고 2건).
        arm_target = self.fabric_q[:, : self.profile.num_arm_joints]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        # ★★속도 피드포워드(08.25 복구). 여기에 0 을 넣으면 감쇠항 kd·(0 − q̇) 이
        #   참조 궤적의 움직임을 반대로 밀어 err ≈ (kd/kp)·q̇ 의 상시 지연이 생긴다.
        #   fabric 이 이미 `fabric_qd` 를 계산해 두는데 그걸 버리고 있었다.
        #   DEXTRAH 원본과 동일한 배선이다(cfg 주석에 산수·실측 근거).
        self.robot.set_joint_velocity_target(
            float(self.cfg.fabric_velocity_ff_scale)
            * self.fabric_qd[:, : self.profile.num_arm_joints],
            joint_ids=self.arm_ids)
        # ★손 PD 지령은 정책 목표가 아니라 **fabric 이 만든 참조 궤적**이다. 그래야
        #   hand_repulsion 이 계획에서 지운 관통 해가 실제 지령에도 없다.
        # ★★`self.hand_targets` 를 덮어쓰면 안 된다 — 이 트랙의 손 액션은 **누산**이라
        #   (hand_targets += delta) 실측값을 되먹이면 액션 의미가 바뀐다. 자매 트랙은
        #   절대 매핑이라 그대로 덮어도 되지만 여기서는 지령만 갈아끼운다.
        if self._synergy:
            # ★손은 fabric 밖이다 — 관절 목표를 **이름으로 찾은 인덱스**에 직접 준다.
            #   fabric 은 같은 자세를 `_fabric_hand_cmd` 로 받아 충돌 모델만 동기화한다.
            self.robot.set_joint_position_target(self._syn_target, joint_ids=self._syn_ids)
            # Kuka 와 동일하게 손에도 속도 목표를 준다 — 없으면 감쇠항 kd·(0 − q̇) 가
            # 닫는 동작을 상시 반대로 밀어 err ≈ (kd/kp)·q̇ 의 과도 지연이 생긴다.
            self.robot.set_joint_velocity_target(
                float(self.cfg.hand_velocity_ff_scale) * self._syn_vel,
                joint_ids=self._syn_ids)
            return
        _hand_cmd = (
            self.fabric_q[:, self.profile.num_arm_joints:][:, self._fab_to_hand_local]
            if (self._hand_fabric or self._tip_cyl) else self.hand_targets
        )
        self.robot.set_joint_position_target(_hand_cmd, joint_ids=self.hand_ids)
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

    def _palmar_mask(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """마디가 물체를 **손바닥면으로** 마주보고 있는가. bool (N, F, 2) = (mid, dist).

        판정: dot(손바닥법선_world, 물체중심 − 마디위치) > 0.
        손바닥 법선은 프로필의 국소축을 마디 자세로 회전시켜 얻는다.

        ★왜 필요한가(08.23 lstm_test3 ep5000 실측): 접촉센서는 링크에 붙어 있어
          손바닥면이든 손등이든 같은 크기를 낸다. 정책은 `middle_2` 를 하한 0 에 붙여
          밑동을 안 편 채 `_3`/`_4` 만 최대로 말아 갈고리를 만들고, 그 **등**으로 컵을
          밀었다 — middle_4 접촉 시간의 100%(부호 −19.7mm)가 손등. envelope_frac 은
          이를 감쌈으로 세어 0.746 을 보고했지만 손바닥 접촉만 세면 ≈0.55 로 성공
          임계 0.6 미달이다. 즉 지표가 실패를 성공으로 통과시키고 있었다.

        ★왜 힘 벡터가 아니라 기하인가: `force_matrix_w` 는 방향을 갖고 있어 더
          국소적인 판정이 가능하지만, "링크에 작용하는 힘"의 부호 규약을 실측으로
          확정하지 못했다(부호가 뒤집히면 판정이 통째로 반대가 되는데 조용하다).
          기하 판정은 probe 로 검증됐다 — 손바닥 접촉 +30.6/+45.0mm, 손등 −19.7mm 로
          분리가 깨끗했다. 힘 기반은 부호를 실측한 뒤 대조 지표로 먼저 붙일 것.
        """
        pos = self.robot.data.body_pos_w[:, self._wrap_body_ids]      # (N, F, 2, 3)
        quat = self.robot.data.body_quat_w[:, self._wrap_body_ids]    # (N, F, 2, 4)
        axes = self._palmar_axes[None, :, None, :].expand_as(pos)     # (N, F, 2, 3)
        palmar_w = quat_apply(quat.reshape(-1, 4), axes.reshape(-1, 3)).view_as(pos)
        to_obj = (obj_pos + self.scene.env_origins)[:, None, None, :] - pos
        return (palmar_w * to_obj).sum(dim=-1) > 0.0

    def _env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        joint_pos = torch.cat([q[:, self._arm_ids_t], q[:, self._hand_ids_t]], dim=1)
        # ★★측정 속도는 policy obs 에서 0 이 된다(Kuka `observation_annealing` 계수
        #   (0., 0.) 과 동일 — 시작·종단이 둘 다 0 이라 전 구간 무효화다). 정책이 보는
        #   속도 정보는 `fabric_qd`(참조 속도)뿐이다. critic 은 아래에서 **원값**을 받는다
        #   (Kuka critic 도 annealing 을 안 곱한 `robot_dof_vel` 을 쓴다).
        joint_vel = float(self.cfg.obs_measured_velocity_scale) * torch.cat(
            [qd[:, self._arm_ids_t], qd[:, self._hand_ids_t]], dim=1)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        # ★★08.25 쿼터니언(4D) → **회전행렬 두 열**(palm_x, palm_y = 6D).
        #   쿼터니언은 q 와 −q 가 같은 자세라 부호 이중성이 있고, 신경망이 그
        #   불연속을 따로 배워야 한다. 그리고 보상의 자세 항(perp_q·roll_q)이
        #   정확히 이 두 축의 함수라 관측과 보상이 같은 양을 본다.
        _pR = self._palm_ee_R()
        palm_ax = torch.cat([_pR[:, :, 0], _pR[:, :, 1]], dim=1)
        tips = (
            self.robot.data.body_pos_w[:, self._tip_ids_t]
            - self.scene.env_origins[:, None, :]
        ).reshape(self.num_envs, -1)
        obj_pos = self._env_local(self.object.data.root_pos_w)
        obj_quat = self.object.data.root_quat_w
        # ★clip_observations=5.0 이 raw 접촉력을 5N 에서 자른다(env 의 clamp(20) 은 무효였다).
        #   5로 나눠 20N 에서 4.0 으로 포화시키면 의미(포화점 20N)는 유지하고 클립에 안 걸린다.
        contact = (self._contact_forces() / 5.0).clamp(max=4.0)
        # ★★fabric 상태(08.25 신설, DEXTRAH Kuka policy obs 동일). 액션이 **절대 목표**
        #   이므로 정책은 참조 궤적의 현재 위치를 알아야 한다 — 이게 없으면 "어디로
        #   가라"만 내고 "지금 어디쯤인지"를 모른다. 관절 순서는 fabric 순서 그대로다
        #   (obs 규약은 내부 일관성만 있으면 되고, 정책이 학습으로 대응을 배운다).
        # ★★08.25 `fabric_qd`·`fabric_qdd` 54D 제거(사용자 결정). 0 처리되어서가
        #   아니라 **중복**이라서다:
        #     ① Kuka 는 둘 다 annealing 계수(0,0)를 곱해 죽인다(env.py:1376-1387).
        #        Kuka 정책은 속도·가속도를 어디서도 안 받고 위치만으로 학습해
        #        teacher→student 를 끝까지 성공시켰다.
        #     ② LSTM 이 있으므로 `fabric_q` 시퀀스에서 속도가 은닉상태로 나온다.
        #     ③ 실측: 손 20관절 구간은 **상수**(qd 0.1313~0.1318, 25%는 정확히 0) —
        #        `fabric_q[:, arm:]` 를 매 스텝 덮어써 적분기 속도 상태가 잔재가 된다.
        #     ④ `qdd` 는 가속도 한계(7.5/10.0)에 상시 포화라 정보량이 더 낮다.
        #   ★`fabric_q` 는 남긴다 — **절대 액션 매핑의 짝**이다. 정책이 참조 궤적의
        #     현재 위치를 모르면 절대 목표를 낼 수 없다. Kuka 도 q 만은 원값으로 준다.
        #   ★fabric 내부에서는 qd 가 살아 있다(적분·반발 속도게이트) — obs 에서만 뺀다.
        fab = self.fabric_q
        # ★★08.25 `contact` 5D 를 policy obs 에서 빼 **critic 전용**으로 옮겼다
        #   (사용자 결정, Kuka 배선과 동일 — Kuka 도 hand_forces 는 critic 에만 둔다).
        # ★★08.25 `obj_quat` 4D 도 policy 에서 뺐다(사용자 결정) — 물체 **자세**는
        #   실기 인식으로 안정적으로 얻기 어렵다. 위치(obj_pos)만 정책에 준다.
        #   보상은 obj_up 을 계속 쓴다(privileged 는 보상·critic 에 허용).
        obs = torch.cat([
            joint_pos, joint_vel, palm_pos, palm_ax, tips,
            obj_pos, self.goal_pos, self.actions, fab,
        ], dim=1)
        # ★measured_joint_torque — Kuka critic obs 동일(privileged). 팔+손 순서로
        #   policy obs 의 joint_pos/vel 과 같은 정렬을 쓴다.
        _tau = self.robot.data.applied_torque
        state = torch.cat([
            obs,
            # ★접촉력·물체 자세는 critic 전용(privileged). policy 는 못 본다.
            contact,
            obj_quat,
            # ★구 `joint_vel_true` 27D 는 삭제 — policy obs 의 joint_vel 이 이제
            #   실측 원값(scale 1.0)이라 critic 에 같은 값을 또 넣을 이유가 없다.
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.difficulty.float().unsqueeze(1) / float(self.cfg.curriculum_max_level),
            torch.cat([_tau[:, self._arm_ids_t], _tau[:, self._hand_ids_t]], dim=1),
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
        mid_raw, dist_raw = self._contact_forces_split()
        # ★감쌈은 **손바닥 접촉만** 인정한다(08.23 reward-audit ACCEPT). 손등으로 민
        #   마디의 힘을 0 으로 눌러 envelope_frac 계산에서 빠지게 한다.
        #   대향 게이트(contact)와 obs 는 건드리지 않는다 — 실기 센서도 방향을 모르므로
        #   obs 를 바꾸면 s2r 규약이 함께 흔들린다. 여기서 고치는 건 **판정**뿐이다.
        if self.cfg.require_palmar_contact:
            _pal = self._palmar_mask(obj_pos)
            mid_f = mid_raw * _pal[:, :, 0].float()
            dist_f = dist_raw * _pal[:, :, 1].float()
        else:
            mid_f, dist_f = mid_raw, dist_raw
        # 물체 기울기 — 같은 스텝에 _get_dones 가 계산·캐시한 값(dones 가 rewards 보다 먼저)
        tilt_deg = self._tilt_deg_buf
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        if self._tip_cyl or self._synergy:
            # ★08.24 총 재설계 — 5항+정규화, 게이트 1개(공유 세션 분리: 구 함수는
            #   grasp_lift_fabric 이 공유하므로 불변, tip_cyl 경로만 새 함수).
            # 파지중심(팔 전용 기준점). ★FK 입력은 fabric_q(지령)가 아니라 **실측 관절** —
            #   보상은 실제로 간 곳을 봐야 한다. 프레임은 원통 기하와 같은 fabric-palm 을
            #   쓰고, 마지막에 실측 상수로 env-local 로 옮긴다(프레임 혼용 금지).
            _qa = self.robot.data.joint_pos[:, self._fab_t].contiguous()
            _o, _R = self._tip_palm_frame(_qa)
            _gc = (_o + torch.einsum("bij,j->bi", _R, self._gc_local)
                   + self._fab_to_env)
            _thr = float(self.cfg.stage_contact_threshold)
            # ★거리 버퍼(_wrap_dist / _tip_d)는 구 `reach` 항 전용이었고 08.25 에 그 항을
            #   폐기하면서 함께 제거했다 — 보상은 이제 접촉 개수만 센다.
            _mid_c = mid_f > _thr
            _dist_c = dist_f > _thr
            _any_c = contact > _thr
            # ★08.25 grasp_v1 구조 — 거리 항(`pull_dist`/`touched`) 폐기. 접촉만 센다.
            #   팁 접촉은 손가락별 팁 센서(엄지 포함 5지).
            _tip_c = self._tip_contact_forces() > _thr
            # 접촉 지속 — 접촉 손가락 수가 임계 이상인 스텝을 세고 정규화(grasp_v1).
            #   끊기면 0 으로 리셋되므로 "닿았다 뗐다"로는 못 채운다.
            _ncon = (_mid_c | _dist_c | _tip_c).sum(dim=-1)
            self._persist_buf = torch.where(
                _ncon >= int(self.cfg.stage_persistence_min_contacts),
                self._persist_buf + 1,
                torch.zeros_like(self._persist_buf))
            _persist = (self._persist_buf.float()
                        / max(float(self.cfg.stage_contact_persistence_steps), 1.0)
                        ).clamp(max=1.0)
            _b = self._group_b_env_idx
            total, terms, gate, env_frac = compute_tip_cyl_rewards(
                palm_pos=palm_pos,
                grasp_center_pos=_gc,
                object_pos=obj_pos,
                goal_pos=self.goal_pos,
                tip_c=_tip_c,
                persist_frac=_persist,
                wrap_c=(_mid_c | _dist_c)[:, _b],
                deep_c=(_mid_c & _dist_c)[:, _b],
                oppose=_any_c[:, self._group_a_idx].any(dim=-1),
                height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
                tilt_deg=tilt_deg,
                xy_disp=torch.norm(
                    obj_pos[:, :2] - self.object_spawn_pos[:, :2], dim=-1),
                # ★5지 전부의 손바닥면 접촉 — 엄지 하나 터치는 five_frac 0.2 로 죽는다.
                five_c=(_mid_c | _dist_c | _tip_c),
                # ★자세: palm_ee **+x 가 손바닥 법선**(실측 확정, 네 palm 계열 body 가
                #   회전은 동일하고 위치만 다르다). 컵 축과 수직이어야 한다.
                palm_x=self._palm_ee_R()[:, :, 0],
                palm_y=self._palm_ee_R()[:, :, 1],
                # ★자세 항의 기준은 **베이스**(컵이 아니라). 컵 기준이면 기울인 컵이
                #   기울인 접근을 정당화하는 되먹임이 생긴다.
                ref_up=self._base_up_vec(),
                obj_up=self._obj_up_vec(),
                obj_speed=torch.norm(self.object.data.root_lin_vel_w, dim=-1),
                actions=self.actions,
                prev_actions=self.prev_actions,
                cfg=self.cfg,
            )
        else:
            total, terms, gate, env_frac = compute_grasp_sensor_rewards(
                object_tilt_deg=tilt_deg,
                object_up=self._obj_up_buf,
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
        _pass_pos = goal_dist < float(self.cfg.success_pos_tolerance)
        _env_min = float(self.cfg.stage_success_envelope_min if self._tip_cyl
                         else self.cfg.success_envelope_min)
        _pass_env = env_frac >= _env_min
        _pass_tilt = tilt_deg < float(self.cfg.success_tilt_max_deg)
        self._goal_reached_now = _pass_pos & _pass_env & _pass_tilt
        # ★커리큘럼 승급 기준(08.25): goal 위치까지 요구하면 성공률이 0 이라 난이도가
        #   한 번도 안 올랐다(lstm_test5~7 전부 difficulty_mean 0.0000 고착 = 스폰반경
        #   0.02m 최저 난이도만 시도). 승급은 **리프트 성공**으로 완화하고, 위치까지
        #   포함한 엄격 판정은 task/goal_reached 로 계속 로깅한다.
        if self._tip_cyl or self._synergy:
            self._lift_success_now = (
                (obj_pos[:, 2] - self.object_spawn_pos[:, 2] >= 0.05)
                & (env_frac >= 0.5)
                & gate
            )
        else:
            self._lift_success_now = self._goal_reached_now
        # ★조건별 개별 통과율 — AND 결과만 보면 어느 조건이 병목인지 알 수 없다
        #   (lstm_test2: 위치 0.845·감쌈 0.778 인데 AND 0.023 → 상관을 못 읽었다).
        self.extras["task/pass_pos"] = _pass_pos.float().mean()
        self.extras["task/pass_env"] = _pass_env.float().mean()
        self.extras["task/pass_tilt"] = _pass_tilt.float().mean()

        # ★컵 단독 리스폰은 폐기됐다(08.22, 사용자 지시). 텔레포트 전이(s→s′ 불연속)가
        #   학습 데이터에 그대로 들어가 value/LSTM 이 비마르코프 점프를 학습했고, 손이
        #   스폰 지대에 있으면 컵이 손가락 위로 겹쳐 소환됐다("순간이동 관통").
        #   전도/낙하/이탈은 이제 _get_dones 의 **truncation** 이 env 전체 리셋으로 처리한다.
        self.extras["task/object_tilt_deg"] = tilt_deg.mean()
        self.extras["task/tipped_rate"] = (
            tilt_deg > float(self.cfg.tilt_reset_deg)).float().mean()

        # ---- 로깅 ----
        # 접근 기하 진단(가중 전 원거리) — 보상 함수마다 싣는 키가 다르므로 관용 pop.
        #   구(공유) 함수: _d_palm/_d_side/_gate_eff · tip_cyl 함수: _d_max
        for _k, _tag in (("_d_palm", "task/d_palm"), ("_d_side", "task/d_side"),
                         ("_gate_eff", "task/gate_eff"), ("_d_max", "task/d_max"),
                         ("_d_gc", "task/d_graspcenter"), ("_grip_dist", "task/grip_dist"),
                         ("_touch_frac", "task/touch_frac"), ("_align", "task/palm_align"),
                         ("_G", "task/grip_q"), ("_H", "task/lift_q"), ("_U", "task/upright_q"),
                         ("_deep4", "task/deep4"), ("_tip_frac", "task/tip_frac"),
                         ("_full_tip", "task/full_tip"), ("_persist", "task/persist"),
                         ("_envelope", "task/envelope"), ("_grasp_q", "task/grasp_q"),
                         ("_envelope", "task/envelope"), ("_grasp_q", "task/grasp_q"),
                         ("_five_frac", "task/five_frac"), ("_near_q", "task/near_q"),
                         ("_perp_q", "task/perp_q"), ("_roll_q", "task/roll_q"),
                         ("_orient_q", "task/orient_q"), ("_T", "task/track_q"),
                         ("_S", "task/stay_q")):
            _v = terms.pop(_k, None)
            if _v is not None:
                self.extras[_tag] = _v.mean()
        # ---- 단계별 성공 누적 (리셋 때만 기록) -----------------------------------
        _five = (_mid_c | _dist_c | _tip_c)
        _dgc = torch.norm(_gc - obj_pos, dim=-1) if (self._tip_cyl or self._synergy) \
            else torch.full_like(goal_dist, 1e9)
        _h = obj_pos[:, 2] - self.object_spawn_pos[:, 2]
        _spd = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        _tr = (goal_dist <= float(self.cfg.success_pos_tolerance)) & (
            tilt_deg <= float(self.cfg.success_tilt_max_deg))
        self._stay_run = torch.where(
            _tr & (_spd < float(self.cfg.stage_stay_speed_ref)),
            self._stay_run + 1, torch.zeros_like(self._stay_run))
        _now = torch.stack([
            _five.any(dim=-1),                                   # ① 닿을 만큼 갔다
            _five.all(dim=-1) & (_dgc <= float(self.cfg.stage_grasp_near_tau)),   # ② 5지+밀착
            _h >= 0.05,                                          # ③ 리프트
            _tr,                                                 # ④ 이송+직립
            self._stay_run >= int(self.cfg.stage_stay_hold_steps),  # ⑤ 정지 유지
        ], dim=1)
        self._stage_hit |= _now
        # ★컵 밀림 — approach 를 음수로 만드는 유일한 양이다(가중 25.0 은 approach
        #   상한 2.0 의 12.5배라 50mm 만 밀어도 상한의 31% 를 잃는다). 부팅 실측에서
        #   스크립트 서보가 309mm 밀자 approach 가 −5.12 까지 갔다. 정책이 "컵 근처에
        #   가지 마라"를 배우기 시작하면 여기서 먼저 보인다.
        self.extras["task/xy_disp"] = torch.norm(
            obj_pos[:, :2] - self.object_spawn_pos[:, :2], dim=-1).mean()
        self.extras["task/envelope_frac"] = env_frac.mean()
        # 게이트 참인 env 의 감쌈 — "접촉은 됐는데 감쌈이 안 되는" 상태의 직접 지표
        _gf = gate.float()
        self.extras["task/envelope_at_gate"] = (
            (env_frac * _gf).sum() / _gf.sum().clamp(min=1.0))
        # ★구 판정(크기만)을 나란히 남긴다 — 둘의 차이가 곧 **손등 접촉 비중**이다.
        #   lstm_test3 ep5000 기준 raw 0.746 vs 손바닥만 ≈0.55 였다. 이 격차가 다시
        #   벌어지면 정책이 또 등으로 밀기 시작했다는 뜻이고, 구 로그와의 비교선도 된다.
        _thr = float(self.cfg.contact_force_threshold)
        _raw_wrap = ((mid_raw > _thr) | (dist_raw > _thr)).float()
        self.extras["task/envelope_frac_raw"] = _raw_wrap[:, self._env_finger_idx].mean()
        # 손가락별 감쌈 — 지금까지 어느 손가락이 빠지는지 로깅이 없어 tip 거리로
        # 추론해야 했다(08.23: 약지 0.20·새끼 0 을 probe 로야 알았다).
        _pal_wrap = ((mid_f > _thr) | (dist_f > _thr)).float()
        # ★touch(방향무관) 와 wrap(손바닥면) 을 **나란히** 남긴다 — 보상 ② 를 끄는 건
        #   touch, ③ 을 주는 건 wrap 이라 둘의 격차가 곧 "닿았지만 손등"의 양이다.
        _touch = (contact > _thr).float()
        for _i, _f in enumerate(self._finger_names):
            self.extras[f"task/wrap/{_f}"] = _pal_wrap[:, _i].mean()
            self.extras[f"task/touch/{_f}"] = _touch[:, _i].mean()
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
        _jerr = (self.fabric_q[:, : self.profile.num_arm_joints]
                 - self.robot.data.joint_pos[:, self._arm_ids_t]).abs()
        self.extras["fabric/joint_err_max"] = _jerr.max()
        # ★평균이 없어서 max(env×관절 전체 최대)만 보고 있었다 — 한 env 가 끌어올려도
        #   구분이 안 된다. lstm_test9 에서 max 0.5~1.3 rad 이 나왔는데 대표성을
        #   판정할 수가 없었다. fabric 가상 로봇과 실제 로봇의 상시 괴리를 보는 지표다:
        #   보상 기하(_tip_palm_frame(fabric_q))와 접촉(robot.data.*)이 이 값만큼
        #   **서로 다른 로봇**을 보고 있다.
        self.extras["fabric/joint_err_mean"] = _jerr.mean()
        self.extras["fabric/joint_err_p95"] = torch.quantile(
            _jerr.max(dim=-1).values, 0.95)
        # 실제 palm 속도 — 지령(arm_pos_scale/정책스텝)이 팔이 낼 수 있는 속도를
        # 넘으면 목표만 앞서 나가고 액션의 그만큼이 버려진다(구 leash 제거 경위).
        _pp = self._palm_pose_6d()[:, :3]
        if getattr(self, "_prev_palm_pos", None) is not None:
            # 방금 리셋된 env 는 홈으로 텔레포트돼 변위가 튄다 — 빼지 않으면 속도가
            # 과대평가되고, 그건 "팔이 지령을 따라간다"는 **틀린 안심**을 준다.
            _live = self.episode_length_buf > 1
            _spd = torch.norm(_pp - self._prev_palm_pos, dim=-1)
            self.extras["fabric/palm_speed_mean"] = (
                (_spd * _live).sum() / _live.float().sum().clamp(min=1.0))
        self._prev_palm_pos = _pp.detach().clone()
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
        # ★파지중심 잔차의 **방향 분해**(08.25 신설). lstm_test9 에서 d_gc 가 55mm 에
        #   고착했는데 그게 "덜 들어갔다"인지 "옆으로 어긋났다"인지 구분할 지표가 없어
        #   1,300 에폭을 원인 미상으로 보냈다. 접촉이 소지·약지에만 나고 검지·중지가
        #   정확히 0 인 것이 측면 오차를 시사했지만 **직접 잰 값이 없었다**.
        #   axial = 접근축(palm→파지중심) 성분, lateral = 그에 수직인 성분.
        if self._tip_cyl or self._synergy:
            # ★FK 입력은 보상과 **같은 실측 관절**이어야 한다(fabric_q 를 쓰면 진단
            #   좌표와 보상 좌표가 어긋나 서로 다른 로봇을 재게 된다).
            _po, _pR = self._tip_palm_frame(
                self.robot.data.joint_pos[:, self._fab_t].contiguous())
            _gcl = (_po + torch.einsum("bij,j->bi", _pR, self._gc_local)
                    + self._fab_to_env)
            _u = torch.nn.functional.normalize(_gcl - _po, dim=-1)
            _e = obj_pos - _gcl
            _ax = (_e * _u).sum(dim=-1)
            self.extras["task/gc_err_axial"] = _ax.mean()
            self.extras["task/gc_err_lateral"] = torch.norm(
                _e - _ax.unsqueeze(-1) * _u, dim=-1).mean()
            # 컵을 손바닥 좌표로 — 어느 손가락 쪽에 붙어 있는지가 좌표로 보인다.
            _cl = torch.einsum("bji,bj->bi", _pR, obj_pos - _po - self._fab_to_env)
            for _k, _ax_nm in enumerate("xyz"):
                self.extras[f"task/cup_palm_{_ax_nm}"] = _cl[:, _k].mean()
        if self._synergy:
            # ★정책의 실제 폐쇄도 명령. 이번 런에서 "정책이 손을 닫고 있는가"를
            #   묻는 데 필요한 값이었는데 로깅 키가 아예 없었다.
            self.extras["task/syn_close"] = self._syn_close.mean()
            self.extras["task/syn_close_max"] = self._syn_close.max()
            # ★★08.25 손가락별 폐쇄도. 20관절 평균 하나로는 "엄지만 안 닫힌다"를
            #   반증도 확증도 못 한다(사용자 렌더링 관찰 — 4지는 함께 오므리는데
            #   엄지는 오히려 펴진다). couple_four_fingers 로 4지가 묶여 있어
            #   엄지만 독립 분산을 갖는 구조라, 분리 계측이 없으면 진단이 막힌다.
            for _fi, _fn in enumerate(self._finger_names):
                _m = (self._syn_fi == _fi)
                self.extras[f"task/syn_close/{_fn}"] = self._syn_close[:, _m].mean()
            # ★엄지 실측 관절각. 개방 자세가 `_3 = −0.5 rad`(역굴곡)이라 폐쇄도 0 은
            #   "가만히"가 아니라 **뒤로 젖혀진 자세**다. 지령이 아니라 실제 각도를 본다.
            _q = self.robot.data.joint_pos
            for _sfx in ("3", "4"):
                _idx = [k for k, nm in enumerate(self.profile.hand_joint_names)
                        if "thumb" in nm and nm.endswith(f"_{_sfx}")]
                if _idx:
                    self.extras[f"task/thumb_q{_sfx}"] = _q[:, self._syn_ids[_idx[0]]].mean()
        # ★★08.25 palm 축별 추종오차 — 액션이 **절대 매핑**으로 바뀐 뒤 축별로 본 적이
        #   없다. 박스가 곧 액션 공간이므로 특정 축만 못 따라가면 그 축 액션 구간이
        #   통째로 gradient 를 잃는다. 지령(`palm_targets`)과 **실측** palm 의 차다.
        _pp = self._palm_pose_6d()
        _perr = _pp[:, :3] - self.palm_targets[:, :3]
        for _k, _ax_nm in enumerate("xyz"):
            self.extras[f"fabric/palm_ee_{_ax_nm}_err"] = _perr[:, _k].mean()
            self.extras[f"fabric/palm_ee_{_ax_nm}_abs"] = _perr[:, _k].abs().mean()
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
        # ---- 전도/낙하/이탈 = env 전체 리셋, 단 **terminated 로** ---------------------
        # ★★lstm_test1 ep1600 붕괴의 근본 원인이 여기였다(반증된 설계는 아래 주석 참조).
        #   실패를 truncated 로 내보내면 IsaacLab wrapper 가 그대로 extras["time_outs"]
        #   에 싣고(rl_games.py:284), rl_games 가 value_bootstrap 으로
        #   `shaped_rewards += gamma * V(s_t) * time_outs` 를 더한다(a2c_common.py:777).
        #   → **컵을 쓰러뜨릴 때마다 γ·V(s) 를 보너스로 지급**. 정책이 잘할수록 V 가
        #   커져 "잘 잡았다가 넘어뜨리기"가 계속 드는 것보다 이득이 되는 자기강화 루프.
        #   실측: ep1550→1600 에 실제 보상 3307→103 (32배 붕괴)인데 정책이 최적화하는
        #   shaped_rewards 는 72.8→79.4 로 **상승**. 에피소드 길이 473→69, tipped_rate
        #   0→0.019(≈1/55 = 에피소드 길이와 정확히 일치) — 안정적 익스플로잇 수렴.
        # → 실패는 terminated(부트스트랩 없음, 이후 가치 0). 보상이 전 항 비음수라
        #   조기 종료는 그 자체로 손해이고, 접근 회피(agn_test2)는 그때 없던 dense
        #   approach(2.0, 무게이트)가 막는다. bootstrap 은 **진짜 시간 만기에만**.
        _obj_z = quat_apply(
            self.object.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3),
        )
        # 물체 local +z 와 world +z 의 정렬 = cos(기울기). upright 보상이 직접 쓴다.
        self._obj_up_buf = _obj_z[:, 2].clamp(-1.0, 1.0)
        self._tilt_deg_buf = torch.rad2deg(torch.acos(self._obj_up_buf))
        tipped = self._tilt_deg_buf > float(self.cfg.tilt_reset_deg)
        terminated = self._abnormal_buf | fell | tipped | out_xy
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        truncated = timeout
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
        succ = self._lift_success_now[env_ids]
        self.difficulty[env_ids] = (
            self.difficulty[env_ids] + torch.where(succ, 1, -1)
        ).clamp(0, int(self.cfg.curriculum_max_level))
        self._goal_reached_now[env_ids] = False
        self._lift_success_now[env_ids] = False

        # ---- 로봇: 프로필 init 자세 ---------------------------------------------------
        q0 = self._default_q[env_ids].clone()
        qd0 = torch.zeros_like(q0)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[:, self._hand_ids_t]
        if self._synergy:
            # 폐쇄도는 에피소드마다 완전 개방에서 시작한다 — 남기면 이전 에피소드의
            # 폐쇄가 살아남아 리셋 직후 손이 이미 쥔 상태가 된다.
            self._syn_close[env_ids] = 0.0
            # 속도 피드포워드도 함께 리셋 — 안 하면 리셋 직후 한 스텝 동안 이전
            # 에피소드의 목표와 홈 자세의 차이가 거대한 가짜 속도로 들어간다.
            self._syn_target[env_ids] = q0[:, self._syn_ids]
            self._syn_vel[env_ids] = 0.0
        # 접촉 지속 카운터도 반드시 리셋 — 남기면 이전 에피소드의 지속치가 새 에피소드
        # 첫 스텝부터 보상으로 지급된다(grasp_v1 도 같은 자리에서 0 으로 만든다).
        # ★단계별 성공률 — 끝난 에피소드의 결과를 여기서만 기록하고 비운다.
        for _i, _nm in enumerate(self._stage_names):
            self.extras[f"task/stage/{_nm}"] = self._stage_hit[env_ids, _i].float().mean()
        self._stage_hit[env_ids] = False
        self._stay_run[env_ids] = 0
        self._persist_buf[env_ids] = 0
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
