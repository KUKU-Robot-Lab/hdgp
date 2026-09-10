"""grasp_fj_rh — Track B 를 RH56F1(언더액추 손)에 얹는다.

`GraspFJEnv`(Track B) 를 상속하고 **세 가지만** 덮는다.

1. `_arm_slot_width` — per_finger 손 레이아웃 검사가 읽는 팔 슬롯 폭(A 의 palm 6 → B 의 관절 7).
2. `_setup_fabrics` — 부모 부트스트랩 뒤에 **언더액추 계약 검사**를 붙인다.
3. `_hand_command` — 손을 **SimToolReal 원본 방식**(관절 절대 매핑 + EMA)으로 되돌린다.
4. `_log_fabric_metrics` — `ctrl/*` 에 mimic 건전성 지표를 더한다.

★왜 손을 덮나(09.08 사용자 확정 "SimToolReal 방식 그대로, 로봇만 RH56F1"): 원본
  (`repo/simtoolreal/.../env.py:3831-3846`)은 손 액션을 관절 범위로 직접 스케일하고 EMA 만
  건다 — 속도 제한도, 게이트도 없다. 부모(grasp_kp)는 시너지 폐쇄도를 스텝당 0.005 로 제한하고
  `close_gate` 를 곱하는데, 이 손은 케이지 반경이 53mm 라 게이트가 17,312 epoch 내내 0.001
  이었다. 게이트가 0 이면 닫는 방향 변화량이 0 이 되어 **손이 구조적으로 못 닫힌다**.

★왜 mimic 을 부팅에서 검사하나: 이 손의 결합(종속 6관절)은 **자산이 들고 있고**, 우리
  액추에이터는 그 관절을 0/0 으로 둔다. 결합이 USD 에서 사라지면(09.02 실측: headless
  빌드가 URDF `<mimic>` 12개를 통째로 잃었다) 종속관절은 제약도 드라이브도 없는
  **자유 관절**이 되어 손가락이 흐물거린다 — 지표에는 "파지가 안 된다"로만 보인다.
  그래서 결합의 존재를 USD 속성으로 직접 확인하고, 없으면 부팅에서 죽인다.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import torch

from ..grasp_fj.grasp_fj_env import GraspFJEnv
from ..grasp_fj.fj_core_cfg import _ASSETS_DIR
from .grasp_fj_rh_env_cfg import GraspFJRHEnvCfg

#: PhysX mimic 제약이 종속관절 prim 에 남기는 속성. 규약은 `q_dep + gearing·q_ref + offset = 0`
#: 이라 **gearing = −multiplier** 다(임포터 정상 동작).
_MIMIC_GEARING_ATTR = "physxMimicJoint:rotZ:gearing"


class GraspFJRHEnv(GraspFJEnv):
    cfg: GraspFJRHEnvCfg

    # ------------------------------------------------------------------
    # 액션 레이아웃
    # ------------------------------------------------------------------
    def _arm_slot_width(self) -> int:
        """팔이 쓰는 액션 슬롯 수 = 관절 수(B). mixin 기본값 6(palm)을 덮는다."""
        return int(self.cfg._arm_action_dim(self.profile))

    # ------------------------------------------------------------------
    # 부트스트랩 — 부모(B) 그대로 + 언더액추 계약 검사
    # ------------------------------------------------------------------
    def _setup_fabrics(self) -> None:
        super()._setup_fabrics()                 # 시너지·인덱스·팔 목표 버퍼(fabric 은 None)
        self._mimic_pairs = self._load_mimic_pairs()
        self._assert_mimic_constraints_present()
        self._assert_hand_pose_usable()
        self._widen_dependent_joint_limits()

    def _load_mimic_pairs(self) -> dict[str, tuple[str, float]]:
        """자산 URDF 의 `<mimic>` 표에서 **이 손의** 종속→(리더, 배율)을 읽는다.

        왜 코드에 표를 안 적나: 관절 이름·배율은 로봇 종속이고 자산이 진실원천이다.
        USD 옆의 URDF 는 같은 생성기가 함께 낸 파일이라 둘이 갈릴 수 없다.
        """
        # ★자산 **폴더**에서 찾는다 — 파일명에서 파생하면 오버레이 변형(`*_right.usda`,
        #   오른팔 전용)에서 없는 URDF 를 가리키고, 확장자만 바꾸면 usda 를 XML 로 파싱해
        #   `ParseError` 로 죽는다(09.07 실측). 오버레이는 base 를 subLayer 로 물으므로
        #   같은 폴더의 URDF 가 곧 그 자산의 URDF 다.
        _dir = os.path.dirname(os.path.join(_ASSETS_DIR, self.profile.usd_relpath))
        _cands = sorted(glob.glob(os.path.join(_dir, "*.urdf")))
        if len(_cands) != 1:
            raise RuntimeError(
                f"[grasp_fj_rh] 자산 URDF 를 특정 못했다({_dir}): {_cands} — "
                "언더액추 계약을 확인할 수 없다")
        urdf = _cands[0]
        side = self.profile.hand_joint_names[0].split("_hj_")[0]      # 'r' | 'l'
        names = set(self.robot.data.joint_names)
        pairs: dict[str, tuple[str, float]] = {}
        for j in ET.parse(urdf).getroot().iter("joint"):
            m = j.find("mimic")
            dep = j.get("name")
            if m is None or not dep or not dep.startswith(f"{side}_hj_") or dep not in names:
                continue
            pairs[dep] = (m.get("joint"), float(m.get("multiplier", 1.0)))
        if not pairs:
            raise RuntimeError(
                f"[grasp_fj_rh] URDF 에 {side} 손 mimic 이 하나도 없다 ({urdf}) — 이 트랙은 "
                "언더액추 손 전용이다")
        return pairs

    def _assert_mimic_constraints_present(self) -> None:
        """USD 에 PhysX mimic 제약이 **실제로** 붙어 있는지 확인한다(없으면 부팅 실패).

        ★"API 가 붙었다"가 아니라 gearing 값까지 본다 — 배율이 URDF 와 다르면 결합비가
          조용히 달라져 손이 다른 손이 된다.
        """
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        root = str(self.cfg.robot_cfg.prim_path).replace("env_.*", "env_0")
        prim = stage.GetPrimAtPath(root)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"[grasp_fj_rh] 로봇 prim 을 못 찾았다: {root}")
        found: dict[str, float] = {}
        for p in stage.Traverse():
            if not str(p.GetPath()).startswith(root):
                continue
            a = p.GetAttribute(_MIMIC_GEARING_ATTR)
            if a and a.Get() is not None:
                found[p.GetName()] = float(a.Get())
        bad = []
        for dep, (_lead, mult) in self._mimic_pairs.items():
            got = found.get(dep)
            if got is None:
                bad.append(f"{dep}: mimic 제약 없음")
            elif abs(got + mult) > 1e-3:          # gearing = −multiplier
                bad.append(f"{dep}: gearing {got:.4f} ≠ −배율 {-mult:.4f}")
        if bad:
            raise RuntimeError(
                "[grasp_fj_rh] USD 언더액추 결합이 깨졌다 — 종속관절이 드라이브도 제약도 없는 "
                "자유 관절이 된다(09.02: headless 빌드가 mimic 12개를 통째로 잃었다). "
                + " · ".join(bad))
        # 진단용 인덱스 — 로그의 mimic 오차는 이 짝으로 잰다.
        jn = self.robot.data.joint_names
        self._mim_dep_t = torch.tensor([jn.index(d) for d in self._mimic_pairs],
                                       device=self.device, dtype=torch.long)
        self._mim_lead_t = torch.tensor([jn.index(v[0]) for v in self._mimic_pairs.values()],
                                        device=self.device, dtype=torch.long)
        self._mim_mult = torch.tensor([v[1] for v in self._mimic_pairs.values()],
                                      device=self.device)
        print(f"[grasp_fj_rh] 언더액추 확인 — 구동 {len(self._syn_ids)} · "
              f"종속 {len(self._mimic_pairs)}(PhysX mimic, gearing 대조 통과)", flush=True)

    def _assert_hand_pose_usable(self) -> None:
        """자세표가 **실제로 움직이는가**를 부팅에서 본다(09.02 조용한 실패 재발 차단).

        ① grip 이 관절 한계 밖이면 지령이 잘려 그 손가락은 폐쇄도 1 에서도 안 움직인다.
        ② open == grip 이면 그 관절은 시너지가 아예 안 건드린다.
        둘 다 지표에 "파지가 약하다"로만 보이고 이유가 안 나온다.
        """
        _clamped = self._syn_grip.clamp(self._syn_lo, self._syn_hi)
        bad = [f"{n}: grip {float(self._syn_grip[i]):.3f} → 한계클램프 {float(_clamped[i]):.3f}"
               for i, n in enumerate(self.profile.hand_joint_names)
               if abs(float(self._syn_grip[i] - _clamped[i])) > 1e-6]
        if bad:
            raise RuntimeError(
                "[grasp_fj_rh] grip 자세가 관절 한계 밖이라 잘린다 — 이 손은 tesollo 처럼 "
                "과지령으로 한계에 눌러 붙이는 규약을 쓰지 않는다: " + " · ".join(bad))
        dead = [n for i, n in enumerate(self.profile.hand_joint_names)
                if not bool(self._syn_movable[i])]
        if dead:
            raise RuntimeError(
                f"[grasp_fj_rh] open == grip 인 손 관절 {dead} — 액션 슬롯이 있는데 아무것도 "
                "안 움직인다(노브가 자세표를 덮어썼는지 확인하라)")

    # ------------------------------------------------------------------
    # 손 — SimToolReal 원본 매핑(관절 절대 + EMA)
    # ------------------------------------------------------------------
    def _hand_command(self) -> None:
        """`q_hand* = EMA(scale(a, 관절하한, 관절상한))` — 원본 `pre_physics_step` 과 같은 식.

        원본(IsaacGymEnvs `SimToolReal`):
            cur[7:] = scale(a[7:], lower, upper)
            cur[7:] = hand_moving_average · cur + (1 − hand_moving_average) · prev
            cur[7:] = clamp(cur, lower, upper)
        속도 목표는 주지 않는다(위치 목표만 쓴다) — `hand_velocity_ff_scale=0` 이 그 계약이다.

        ★부모의 시너지 경로(폐쇄도 lerp · `synergy_close_speed` · `close_gate` · 막힘 홀드)는
          여기서 **타지 않는다**. 그 장치들은 `grasp_s2r` 계보의 것이고 원본에 없다.
        ★액션 순서 = `profile.hand_joint_names` 순서 = `self._syn_ids` 순서다(부모가 이름으로
          매핑해 둔 그대로). 그래서 슬라이스가 아니라 그 인덱스로 쓴다.
        """
        _prev = self._syn_target
        a = self.actions[:, self._hand_action_offset:].clamp(-1.0, 1.0)
        _raw = self._syn_lo + 0.5 * (a + 1.0) * (self._syn_hi - self._syn_lo)
        _m = float(self.cfg.hand_ema)
        self._syn_target = (_m * _raw + (1.0 - _m) * _prev).clamp(self._syn_lo, self._syn_hi)
        # 원본은 속도 목표가 없다. 부모 `_apply_action` 이 이 값을 쓰지만 스케일이 0 이다.
        self._syn_vel = torch.zeros_like(self._syn_target)
        # ---- 진단만 (제어에는 안 쓴다) ----
        # 폐쇄도: open→grip 구간에서 지금 목표가 어디쯤인가. 가동폭 0 관절은 제외한다.
        _span = self._syn_grip - self._syn_open
        _cl = (self._syn_target - self._syn_open) / torch.where(
            _span.abs() > 1e-6, _span, torch.ones_like(_span))
        self._syn_close = torch.where(self._syn_movable.unsqueeze(0), _cl.clamp(0.0, 1.0),
                                      torch.zeros_like(_cl))
        # 케이지↔물체 거리는 게이트가 꺼져도 계속 찍는다 — 접근이 되는지 보는 1차 지표다.
        _obj = self._env_local(self.object.data.root_pos_w)
        _palm = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _cage = _palm + (self._palm_ee_R() @ self._cage_offset_palm)
        self._cage_ctr_dist = self._banded_dist(_cage - _obj)
        self._close_gate = torch.ones(self.num_envs, device=self.device)

    def _widen_dependent_joint_limits(self) -> None:
        """종속관절 관절한계를 넓혀 **mimic 제약과 한계가 동시에 만족 불가**가 되는 것을 막는다.

        ★09.07 rh_b1 실측: 손 최하단이 테이블(0.205)을 3~5cm 파고든 epoch 에서만
          `ctrl/mimic_err_max` 가 0.15 → 400~790 rad 로 폭주했고, 그렇지 않은 epoch 은
          0.15 대에 머물렀다(상관 118/222 epoch). 09.02 와 같은 메커니즘이다 —
          접촉이 구동관절을 하한 밑으로 역구동하면 종속 요구치가 종속 한계 밖이 된다.
        ★한계 확장은 물리적으로 공짜다: 종속 위치를 정하는 권한은 mimic 제약이고 관절한계는
          백스톱이다. 넓히기만 하고 좁히지 않는다.
        """
        m = float(self.cfg.mimic_dep_limit_margin_rad)
        if m <= 0.0:
            print("[grasp_fj_rh] 종속관절 한계 확장 OFF (자산 값 그대로)", flush=True)
            return
        lim = self.robot.data.joint_pos_limits[:, self._mim_dep_t, :].clone()
        lim[..., 0] -= m
        lim[..., 1] += m
        self.robot.write_joint_limits_to_sim(lim, joint_ids=self._mim_dep_t.tolist())
        _lo = float(lim[0, :, 0].min())
        _hi = float(lim[0, :, 1].max())
        print(f"[grasp_fj_rh] 종속 {int(self._mim_dep_t.numel())}관절 한계를 ±{m} rad 확장 "
              f"→ [{_lo:.3f}, {_hi:.3f}] (mimic 제약이 위치를 정하고 한계는 백스톱)", flush=True)

    # ------------------------------------------------------------------
    # 로그 — B 의 ctrl/* 에 mimic 건전성 2개를 더한다
    # ------------------------------------------------------------------
    def _log_fabric_metrics(self) -> None:
        super()._log_fabric_metrics()
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        # ★결합 오차 — 접촉이 종속관절을 제약 밖으로 밀면 여기서 먼저 보인다.
        #   09.07 실측: 자유 폐쇄 ≤0.05 rad · 자유 물체 파지 ≤0.04 rad ·
        #   고정 물체를 짓누르면 1.05 rad 까지 벌어진다(그 상태가 곧 발산 직전이다).
        _err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs()
        self.extras["ctrl/mimic_err_max"] = _err.max()
        self.extras["ctrl/hand_dep_qd_max"] = qd[:, self._mim_dep_t].abs().max()
        # ★막힘 게이트(`synergy_hold_mode="blocked"`)는 이 트랙에서 임계를 바꿨는데
        #   (1.00 → 0.25) A/B 는 이 지표를 접촉 진단 경로에서만 찍어 여기선 안 나온다.
        #   바꾼 게이트는 측정 가능해야 한다 — 값이 0 에 붙어 있으면 임계가 너무 크고,
        #   접촉 전부터 1 에 가까우면 너무 작다.
        self.extras["ctrl/hand_blocked_frac"] = self._hand_blocked().float().mean()
        # ★관절별 분해 — `syn_close` 하나로는 "닫는 걸 못 배웠다"와 "엄지만 반대로 가서
        #   평균이 깎였다"가 구분되지 않는다. 지령(cmd)·실측(q)·종속추종(dep) 을 관절별로
        #   따로 찍어 세 층 중 어디가 막혔는지 본다.
        _qh = q[:, self._syn_ids]
        for _i, _nm in enumerate(self.profile.hand_joint_names):
            _s = _nm.split("_hj_")[-1]
            self.extras[f"handq/cmd_{_s}"] = self._syn_target[:, _i].mean()
            self.extras[f"handq/q_{_s}"] = _qh[:, _i].mean()
        for _k, (_dep, (_lead, _m)) in enumerate(self._mimic_pairs.items()):
            _s = _dep.split("_hj_")[-1]
            self.extras[f"handq/dep_{_s}"] = q[:, self._mim_dep_t[_k]].mean()
        # ★감쌈인가 누르기인가 — 관절각만으로는 안 갈린다. 엄지팁과 4지팁 중심이 물체를
        #   **사이에 두고 마주보는지**를 부호로 잰다. dot<0 이면 물체가 둘 사이에 있다.
        # ★`_env_local` 은 (N,3) 전용이다 — (N,5,3) 을 주면 env 2개 이상에서
        #   브로드캐스트가 깨진다(N=1 만 우연히 통과). 원점을 직접 뺀다.
        _tips = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                 - self.scene.env_origins[:, None, :])              # (N, 5, 3)
        _obj = self._env_local(self.object.data.root_pos_w).unsqueeze(1)          # (N, 1, 3)
        _v = _tips - _obj
        _vt = _v[:, 0, :]                       # 엄지 (fingertip_bodies 첫 항)
        _vf = _v[:, 1:, :].mean(dim=1)          # 4지 중심
        _dot = (_vt * _vf).sum(dim=-1) / (_vt.norm(dim=-1) * _vf.norm(dim=-1) + 1e-9)
        self.extras["grip/opposed_frac"] = (_dot < 0.0).float().mean()
        self.extras["grip/thumb_flex_cos"] = _dot.mean()
        self.extras["grip/thumb_tip_d"] = _vt.norm(dim=-1).mean()
        self.extras["grip/flex_tip_d"] = _v[:, 1:, :].norm(dim=-1).mean()
        # 손끝이 물체 **위**에 떠 있는가(헛돌기) — 물체 중심 대비 팁 z 편차.
        self.extras["grip/tip_above_obj_z"] = (_tips[:, :, 2] - _obj[:, :, 2]).mean()
        # ★손이 물체를 품었는가 — 케이지 중심과 물체의 거리. 이 트랙의 판정 기준선:
        #   손안 판정(09.08)에서 파지가 성립한 값 ≈ 0.005m · 대본 접근이 멈춘 값 0.032m ·
        #   물체 반경 0.0286m(케이지 여유 0.024m). 이 값이 여유 밖이면 아무리 닫아도
        #   물체가 손 안에 없다. 손끝 거리(ft_dist)와 달리 **팔이 한 일**만 본다.
        self.extras["grip/cage_obj_d"] = self._cage_ctr_dist.mean()
