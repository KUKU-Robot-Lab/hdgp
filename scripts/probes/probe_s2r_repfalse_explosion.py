#!/usr/bin/env python3
"""`replicate_physics=False` 폭주 메커니즘 분해 프로브 — grasp_s2r.

배경 (08.29)
------------
cup_family(MultiAsset → replicate_physics=False)에서 리셋 직후 1스텝 만에
arm qd 2,000~4,700 rad/s 로 폭주한다(`episode_lengths` 1.2). IsaacLab
0.54.3↔0.45.9(2.2.1) 모두 동일 붕괴 → 프레임워크 버전이 아니라 우리
씬/자산/파싱 경로의 문제다. 진단 지표는 1024 env 평균이라 메커니즘이
안 보인다 — 소수 env 로 스텝별 물리 원시값을 직접 덤프한다.

무엇을 가르는가
---------------
리셋 기입 직후(스텝 전)와 스텝별로 PhysX 뷰에서 직접 읽는다:
  1. 기입 정합: write 직후 get_dof_positions() == default_q 인가
     → 어긋나면 "기입이 안 먹는" 계열, 맞으면 "스텝에서 폭발" 계열.
  2. 접촉 동반 여부: 손끝 ContactSensor net force 가 폭주와 함께 튀는가
     → 튀면 충돌/필터링 계열, 0 이면 드라이브/조인트 계열.
  3. 어느 관절이 먼저 터지는가(argmax |qd|) — 팔/손/양쪽.

사용
----
  RUN_LABEL=probe_repfalse python scripts/probes/probe_s2r_repfalse_explosion.py \
      --num_envs 8 --bank cup_family --steps 20
  # 대조: --bank single_cup --replicate true
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--bank", default="cup_family", choices=["single_cup", "cup_family"])
parser.add_argument("--replicate", default="auto", choices=["auto", "true", "false"],
                    help="auto = bank 가 정함(cup_family→False)")
parser.add_argument("--audit", action="store_true",
                    help="부팅 직후 env0 vs 클론의 USD 물성·드라이브 게인을 대조하고 종료")
parser.add_argument("--no_filter", action="store_true",
                    help="scene.filter_collisions 를 no-op 으로 — CollisionGroup 이 "
                         "articulation 인접링크 자동필터를 깨는지 검증(DEXTRAH 는 미호출)")
parser.add_argument("--selfcol", default=None, choices=["on", "off"],
                    help="로봇 자기충돌 강제 — 손 hull 겹침 폭발 가설 검증 "
                         "(USD 저작·grasp_v2·DEXTRAH 는 전부 OFF)")
parser.add_argument("--depen", type=float, default=None,
                    help="max_depenetration_velocity 덮기 (기본 1000 → 예: 5.0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env import GraspS2REnv  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env_cfg import (  # noqa: E402
    GraspS2RTesolloRightEnvCfg,
)


def main() -> None:
    if args.no_filter:
        from isaaclab.scene import InteractiveScene

        InteractiveScene.filter_collisions = lambda self, **kw: print(
            "[probe] filter_collisions SKIPPED (no-op)", flush=True)
    cfg = GraspS2RTesolloRightEnvCfg()
    if args.selfcol is not None:
        cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = \
            args.selfcol == "on"
        print(f"[probe] enabled_self_collisions={args.selfcol == 'on'}", flush=True)
    if args.depen is not None:
        cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity = float(args.depen)
        # multi 스펙은 _object_spawn_base 를 shallow-replace 하므로 중첩 rigid_props
        # 인스턴스를 직접 고치면 전 종에 전파된다.
        cfg.object_cfg.spawn.rigid_props.max_depenetration_velocity = float(args.depen)
        _b = getattr(cfg, "_object_spawn_base", None)
        if _b is not None:
            _b.rigid_props.max_depenetration_velocity = float(args.depen)
        print(f"[probe] max_depenetration_velocity={args.depen}", flush=True)
    cfg.scene.num_envs = int(args.num_envs)
    cfg.object_bank = args.bank
    cfg.enable_events = False          # 이벤트는 무죄로 판명 — 잡음 제거
    cfg.episode_length_s = 1000.0      # 프로브 중 리셋 오염 방지(관례)
    if args.replicate != "auto":
        # ★finalize_after_overrides(_apply_object_bank)가 multi 에서 False 로
        #   강제하므로, 명시 지정은 그 뒤에 반영돼야 한다 → env __init__ 이
        #   finalize 를 부른 뒤에는 늦다. 여기서 미리 bank 를 적용해 두고 덮는다.
        cfg.finalize_after_overrides()
        cfg.scene.replicate_physics = args.replicate == "true"
    env = GraspS2REnv(cfg, render_mode=None)
    robot = env.unwrapped.robot
    view = robot.root_physx_view

    n = env.unwrapped.num_envs
    arm_ids = env.unwrapped._arm_ids_t
    default_q = robot.data.default_joint_pos

    if args.audit:
        # ---- 0) env0(원본) vs 클론 — spawn 수정이 클론에 실렸는가 ---------------
        #   프로브 실측(08.29): env0 만 건강(qd 0.05)·클론은 l_aj_1 14 rad/s 요동
        #   + 110 kN 접촉 스파이크 → "수정이 env0 에만 적용" 가설을 USD 로 확정한다.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        keys = ("physxRigidBody:disableGravity",
                "physxRigidBody:maxDepenetrationVelocity")
        art_key = "physxArticulation:enabledSelfCollisions"
        for e in range(min(n, 4)):
            root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
            vals: dict[str, object] = {}
            if root and root.IsValid():
                a = root.GetAttribute(art_key)
                vals[art_key.split(":")[-1]] = a.Get() if a and a.HasValue() else "∅"
                for prim in stage.Traverse():
                    p = str(prim.GetPath())
                    if not p.startswith(f"/World/envs/env_{e}/Robot"):
                        continue
                    for k in keys:
                        a = prim.GetAttribute(k)
                        if a and a.HasValue() and k.split(":")[-1] not in vals:
                            vals[k.split(":")[-1]] = (a.Get(), p.rsplit("/", 1)[-1])
                    if len(vals) >= 3:
                        break
            print(f"[audit] env_{e}: {vals}", flush=True)
        stiff = view.get_dof_stiffnesses()
        damp = view.get_dof_dampings()
        d_s = (stiff - stiff[0:1]).abs().max(dim=1).values
        d_d = (damp - damp[0:1]).abs().max(dim=1).values
        print(f"[audit] 드라이브 게인 env0 대비 최대편차: stiffness={d_s.tolist()} "
              f"damping={d_d.tolist()}", flush=True)
        env.close()
        return

    env.reset()

    # ---- 1) 리셋 기입 정합 — 스텝 전에 PhysX 에서 직접 읽는다 -------------------
    q_now = view.get_dof_positions().to(default_q.device)
    qd_now = view.get_dof_velocities().to(default_q.device)
    dev = (q_now - default_q).abs()
    print(f"[probe] replicate_physics={cfg.scene.replicate_physics} bank={args.bank} n={n}",
          flush=True)
    print(f"[probe] 리셋 기입 직후(스텝 전): q_dev max={dev.max().item():.5f} rad "
          f"mean={dev.mean().item():.5f} | qd max={qd_now.abs().max().item():.5f} rad/s",
          flush=True)

    joint_names = robot.joint_names
    obj_p = env.unwrapped.object.data.root_pos_w
    origins = env.unwrapped.scene.env_origins
    obj_local = obj_p - origins
    print(f"[probe] 리셋 직후 컵 pos(env-local): "
          f"{[[round(float(c), 3) for c in row] for row in obj_local]}", flush=True)
    palm_p = robot.data.body_pos_w[:, env.unwrapped.palm_idx] - origins
    print(f"[probe] palm pos(env-local): "
          f"{[[round(float(c), 3) for c in row] for row in palm_p]}", flush=True)
    tbl = getattr(env.unwrapped, "table", None)
    if tbl is not None:
        t_local = tbl.data.root_pos_w - origins
        print(f"[probe] 테이블 root(env-local): "
              f"{[[round(float(c), 3) for c in row] for row in t_local]}", flush=True)
    print(f"[probe] env_origins: "
          f"{[[round(float(c), 2) for c in row] for row in origins[: min(n, 8)]]}",
          flush=True)
    tips = robot.data.body_pos_w[:, env.unwrapped._tip_ids_t] - origins.unsqueeze(1)
    for e in range(min(n, 4)):
        d = (tips[e] - obj_local[e].unsqueeze(0)).norm(dim=-1)
        print(f"[probe] env{e} 손끝↔컵중심 거리(mm): "
              f"{[round(float(x) * 1000) for x in d]} | 컵 {obj_local[e].tolist()}",
              flush=True)
    zeros = torch.zeros(n, env.unwrapped.cfg.action_space, device=env.unwrapped.device)

    # ---- 2) 스텝별 폭주 추적 ----------------------------------------------------
    for t in range(int(args.steps)):
        env.step(zeros)
        q = robot.data.joint_pos
        qd = robot.data.joint_vel
        qd_abs = qd.abs()
        flat = qd_abs.argmax().item()
        e_i, j_i = divmod(flat, qd.shape[1])
        # 접촉 귀속: 센서별 최대 net force — 어느 센서·어느 env 가 관통 중인가.
        f_tip = -1.0
        f_who = "-"
        sensors = env.unwrapped.scene.sensors
        for s_name, s in sensors.items():
            try:
                f = s.data.net_forces_w.norm(dim=-1)   # (N, B)
            except (AttributeError, RuntimeError):
                continue
            fm = float(f.max())
            if fm > f_tip:
                f_tip = fm
                fe, fb = divmod(int(f.argmax()), f.shape[1])
                f_who = f"{s_name}[env{fe},b{fb}]"
        arm_qd_max = qd_abs[:, arm_ids].max().item()
        # 병든 env 규모 — qd 가 abnormal 임계(20)를 넘는 env 수와 그 최악값 분포.
        _env_qd = qd_abs.max(dim=1).values
        n_sick = int((_env_qd > 20.0).sum())
        # ★이탈 바디 특정 — env-local 홈은 전 env 동일해야 하므로 per-body 중앙값
        #   대비 이탈이 큰 (env, body) 가 곧 "실제로 움직인 몸"이다.
        _bp = robot.data.body_pos_w - origins.unsqueeze(1)      # (N, B, 3)
        _med = _bp.median(dim=0).values                          # (B, 3)
        _dev_b = (_bp - _med.unsqueeze(0)).norm(dim=-1)          # (N, B)
        _flat_top = _dev_b.flatten().topk(5)
        _names = robot.body_names
        _tops = []
        for v, fi in zip(_flat_top.values.tolist(), _flat_top.indices.tolist()):
            be, bb = divmod(fi, _dev_b.shape[1])
            _tops.append(f"env{be}/{_names[bb]}:{v * 1000:.0f}mm")
        print(f"[probe]      이탈바디 top5: {' '.join(_tops)}", flush=True)
        print(f"[probe] t={t:3d} qd_max={qd_abs.max().item():10.2f} rad/s "
              f"(env{e_i} {joint_names[j_i]}) arm_qd={arm_qd_max:10.2f} "
              f"q@argmax={q[e_i, j_i]:8.2f} contactF={f_tip:9.2f} N @{f_who} "
              f"sick(qd>20)={n_sick}/{n}", flush=True)
        if t in (0, 4, int(args.steps) - 1):
            dev_t = (q - default_q).abs()[:, arm_ids]
            print(f"[probe]      arm q_dev: max={dev_t.max().item():.4f} "
                  f"mean={dev_t.mean().item():.4f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
