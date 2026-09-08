#!/usr/bin/env python3
"""grasp_fj_rh 대본 파지 — **정책 없이** 접근·폐쇄·리프트를 손으로 시켜 본다.

왜 필요한가
-----------
학습이 `lifted 0 · close_gate 0 · reward 상수`로 정체할 때 원인은 둘 중 하나다.

  (A) 과제는 풀 수 있는데 **탐색이 첫 접촉을 못 만든다**(경사 부재)
  (B) 이 손·이 물체로는 **애초에 파지가 성립하지 않는다**(도달·게이트·기하)

둘은 학습 곡선에서 똑같이 보이므로, 팔을 직접 몰아 봐야 갈린다. 선행 rh56f1 트랙이
07.23 에 같은 함정에 빠졌다 — reward 를 일곱 번 고치고 나서야 "손이 물체 위 12cm 에
떠 있어 접촉이 원천 불가"였음이 grip_probe 로 드러났다.

무엇을 하나
-----------
env 를 그대로 띄우고(보상·게이트·관측 전부 실제 경로) 액션만 대본으로 준다.

  1) 접근: 케이지 중심이 물체 파지중심에 오도록 팔 관절 목표를 몬다(야코비안 1회 해).
  2) 폐쇄: 손 액션 +1(완전 폐쇄 지령).
  3) 리프트: 케이지를 +z 로 올린다.

각 구간에서 `ft_dist · close_gate · syn_close · dz · lifted · 물체 z` 를 찍는다.
★대본이 성공하면 (A)다 — 보상·탐색 문제. 실패하면 (B)다 — 기하·게이트를 고쳐야 한다.

사용
----
    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_fjrh_scripted.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--approach-steps", type=int, default=120)
parser.add_argument("--close-steps", type=int, default=150)
parser.add_argument("--lift-steps", type=int, default=150)
parser.add_argument("--lift-height", type=float, default=0.12)
parser.add_argument("--profile", default="rh56f1_right_only")
parser.add_argument("--inhand", type=int, default=0,
                    help=">0 이면 **손안 판정 시험**. 팔·접근·보상을 전부 빼고 물체를 케이지 중심에 "
                         "직접 놓은 뒤(이 스텝 수만큼 운동학적으로 붙잡아 둔다) 손을 닫고 놓아준다. "
                         "이후 물체가 케이지를 따라오면 파지 성립, 상판으로 떨어지면 불성립이다. "
                         "접근·리셋·탐색이 개입하지 않으므로 하드웨어 능력만 잰다.")
parser.add_argument("--inhand-watch", type=int, default=200,
                    help="놓아준 뒤 관찰 스텝 수.")
parser.add_argument("--hand-approach", default=None,
                    help="접근 구간 손 액션 6개(정규화 -1..1, 쉼표). 미지정 = 기존 close 0.0 균일. "
                         "★엄지를 접근 때부터 대향 위치에 두어 폐쇄 순간의 120° 휩쓸기를 없애는 용도.")
parser.add_argument("--hand-close", default=None,
                    help="폐쇄/리프트 구간 손 액션 6개(정규화 -1..1, 쉼표). 미지정 = 기존 close 1.0 균일.")
parser.add_argument("--blocked-thr", type=float, default=-1.0,
                    help="blocked_err_thr_rad 덮어쓰기(-1 = cfg 기본값). 토크 포화 오차 "
                         "effort/kp = 1.0/5.0 = 0.2 rad 보다 충분히 커야 접촉 뒤에도 계속 조인다")
parser.add_argument("--hold-mode", default="", help="synergy_hold_mode 덮어쓰기(blocked|none)")
parser.add_argument("--grasp-z", type=float, default=None, type_=None) if False else parser.add_argument(
    "--grasp-z", type=float, default=None,
    help="object_grasp_z_offset 덮어쓰기(음수 허용 — 물체 원점 아래를 잡으면 전도 모멘트가 준다)")
parser.add_argument("--obj-ang-damp", type=float, default=-1.0, help="물체 각감쇠 덮어쓰기")
parser.add_argument("--bank", default="", help="object_bank 덮어쓰기")
parser.add_argument("--obj-mass", type=float, default=-1.0, help="물체 질량 덮어쓰기")
parser.add_argument("--random-steps", type=int, default=0,
                    help=">0 이면 **균등 무작위 액션**으로 그만큼 돌며 관절별 이동량과 손의 "
                         "이동/회전 비를 잰다. 정책이 학습되지 않은 상태(σ≈1)의 대리 측정이다")
parser.add_argument("--replicate", default="", choices=("", "true", "false"),
                    help="scene.replicate_physics 강제. 단일 물체 경로는 True 로 남는데, "
                         "다물체 경로(False)와 물리가 갈리는지 가르는 스위치다")
parser.add_argument("--k-arm", type=float, default=-1.0,
                    help="k_arm 덮어쓰기(슬루 = 0.1·k/dt). 손이 작을수록 스텝당 이동이 "
                         "여유(케이지반경−물체반경 = 24mm)에 비해 커진다 — 그게 전도의 원인이다")
parser.add_argument("--approach-gain", type=float, default=1.0,
                    help="접근 액션 배율. 1.0 이면 매 스텝 슬루 상한(1 rad/s)까지 민다 — "
                         "지금 대본은 사실상 물체를 들이받는다. 낮추면 정밀 접근이 된다")
parser.add_argument("--friction", type=float, default=-1.0,
                    help="물체·작업면 마찰 덮어쓰기. 전도 판정 h > r/μ — μ 를 낮추면 밀려서 "
                         "**미끄러진다**(회복 가능), 높으면 넘어진다(에피소드 종료)")
parser.add_argument("--tilt-reset", type=float, default=-1.0,
                    help="tilt_reset_deg 덮어쓰기. 179 로 두면 전도 종료가 사라져 **리셋 되먹임 없이** "
                         "접촉의 실제 결과를 볼 수 있다")
parser.add_argument("--zero-steps", type=int, default=0,
                    help=">0 이면 **액션 0** 으로 그만큼 돌며 종료 사유를 누적한다 — 가만히 두는데도"
                         " 리셋되면 접근·파지 이전에 씬이 깨져 있다는 뜻이다")
parser.add_argument("--video", action="store_true", help="rgb_array 로 녹화한다(--enable_cameras 필요)")
parser.add_argument("--video-dir", default="/home/user/rl_ws/our_source/rh56f1_fj_video",
                    help="영상 저장 폴더 — 로그 밑은 파일명이 전부 같아 구분이 안 된다(메모리 규약)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
_app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env import GraspFJRHEnv  # noqa: E402
from openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env_cfg import (  # noqa: E402
    GraspFJRH56F1RightEnvCfg,
)


def _cage_jacobian(env, arm_t):
    """케이지 중심의 위치 야코비안 (3, n_arm) = J_v − r × J_w (r = palm→케이지, 월드)."""
    R = matrix_from_quat(env.robot.data.body_quat_w[:, env.palm_idx])[0]
    r = R @ env._cage_offset_palm
    jac = env.robot.root_physx_view.get_jacobians()[0]
    bi = env.palm_idx - 1 if jac.shape[0] == len(env.robot.data.body_names) - 1 else env.palm_idx
    J = jac[bi][:, arm_t]
    Jc = J[:3] - torch.stack([torch.linalg.cross(r, J[3:][:, k]) for k in range(J.shape[1])], dim=1)
    return Jc, r


def _report(env, tag: str) -> None:
    obj = env._env_local(env.object.data.root_pos_w)
    palm = env._env_local(env.robot.data.body_pos_w[:, env.palm_idx])
    R = env._palm_ee_R()
    cage = palm + (R @ env._cage_offset_palm)
    d = (cage - obj).norm(dim=-1)
    tips = env.robot.data.body_pos_w[:, env._tip_ids_t]
    ft = (tips - env.object.data.root_pos_w.unsqueeze(1)).norm(dim=-1).mean(dim=1)
    dz = obj[:, 2] - env._settled_z if hasattr(env, "_settled_z") else obj[:, 2] * 0
    # ★진단: 폐쇄가 안 오를 때 원인을 가르는 세 값 — 막힘 판정 · 에피소드 길이(리셋 여부) ·
    #   관절별 폐쇄도. 리셋이 잦으면 `_syn_close` 가 0 으로 되감기므로 게이트·임계를 봐도 헛짚는다.
    _blk = env._hand_blocked().float().mean()
    _err = (env._syn_target - env.robot.data.joint_pos[:, env._syn_ids]).abs().mean(dim=0)
    _sc = env._syn_close.mean(dim=0)
    print(f"[scripted] {tag:12s} cage−obj {float(d.mean()):.4f} m · ft_dist {float(ft.mean()):.4f}"
          f" · close_gate {float(env._close_gate.mean()):.3f}"
          f" · syn_close {float(env._syn_close.mean()):.3f}"
          f" · blocked {float(_blk):.3f} · ep_len {float(env.episode_length_buf.float().mean()):.0f}"
          f" · obj_z {float(obj[:, 2].mean()):.4f}"
          f" · lifted {float(env._latched.float().mean()):.3f}", flush=True)
    # ★물체가 상판을 **파고드는지** 직접 잰다. 세워져 있으면 원점 = 상판 + 원점오프셋,
    #   누우면 원점 = 상판 + 반지름. 그보다 낮으면 관통이다(사용자 지적 09.08).
    _tz = float(env.cfg.table_surface_z)
    _tilt = float(env._tilt_deg.mean())
    _off = float(env._obj_origin_off.mean()) if hasattr(env, "_obj_origin_off") else 0.0599
    _r = 0.044 * 0.65
    _expect = _tz + (_off if _tilt < 45 else _r)
    _pen = _expect - float(obj[:, 2].mean())
    print(f"[scripted] {'':12s} 기울기 {_tilt:.1f}° · 물체 z {float(obj[:, 2].mean()):.4f} "
          f"(기대 {_expect:.4f}, 상판 {_tz}) → 상판 관통 {_pen * 1000:+.1f}mm"
          f"{'  ★박혀 있다' if _pen > 0.005 else ''}", flush=True)
    print(f"[scripted] {'':12s} 관절별 close {[round(float(v), 3) for v in _sc]} · "
          f"|target−q| {[round(float(v), 3) for v in _err]}", flush=True)
    # ★종료 사유 — ep_len 이 짧으면 폐쇄·접근이 매번 되감긴다. 원인을 여기서 바로 읽는다.
    _dn = {k.split("/")[-1]: round(float(v), 4) for k, v in env.extras.items()
           if k.startswith("done/") and float(v) > 0.0}
    if _dn:
        print(f"[scripted] {'':12s} 종료: {_dn}", flush=True)


def _u(env):
    """RecordVideo 래퍼를 벗겨 실제 env 를 준다."""
    return getattr(env, "unwrapped", env)


def main() -> None:
    cfg = GraspFJRH56F1RightEnvCfg()
    cfg.profile_name = args.profile
    cfg.scene.num_envs = int(args.num_envs)
    if float(args.blocked_thr) >= 0.0:
        cfg.blocked_err_thr_rad = float(args.blocked_thr)
    if args.hold_mode:
        cfg.synergy_hold_mode = args.hold_mode
    if args.bank:
        cfg.object_bank = args.bank
    if args.grasp_z is not None:
        cfg.object_grasp_z_offset = float(args.grasp_z)
    if float(args.obj_ang_damp) >= 0.0:
        cfg.object_angular_damping = float(args.obj_ang_damp)
    if float(args.obj_mass) > 0.0:
        cfg.object_mass_override = float(args.obj_mass)
    if float(args.tilt_reset) > 0.0:
        cfg.tilt_reset_deg = float(args.tilt_reset)
    if args.replicate:
        cfg.scene.replicate_physics = (args.replicate == "true")
    if float(args.k_arm) > 0.0:
        cfg.k_arm = float(args.k_arm)
        cfg.arm_slew_rad_s = float(args.k_arm) * float(cfg.arm_ema) / (float(cfg.sim.dt) * int(cfg.decimation))
    if float(args.friction) > 0.0:
        _f = float(args.friction)
        cfg.object_friction_range = (_f, _f)
        cfg.surface_friction = _f
    if args.video:
        # ★기본 시점은 씬 전체라 로봇이 점으로 보인다 — 파지 지점을 클로즈업한다.
        cfg.viewer.eye = (0.85, -0.62, 0.55)
        cfg.viewer.lookat = (0.38, -0.20, 0.30)
        cfg.viewer.resolution = (1280, 720)
    cfg.finalize_after_overrides()
    env = GraspFJRHEnv(cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, name_prefix="fjrh_scripted",
            step_trigger=lambda s: s == 0,
            video_length=int(args.approach_steps + args.close_steps + args.lift_steps),
            disable_logger=True)
        print(f"[scripted] 녹화 → {args.video_dir}", flush=True)
    env.reset()

    e = _u(env)
    n_arm = int(e.profile.num_arm_joints)
    arm_t = e._arm_ids_t
    n_act = int(cfg.action_space)
    gain = float(cfg.arm_ema) * float(cfg.k_arm)          # 스텝당 목표 변화 = gain · a
    act = torch.zeros(e.num_envs, n_act, device=e.device)

    tally: dict = {"resets": 0}

    def _parse_hand(txt):
        """쉼표 6개 → (1, 6) 정규화 액션. 관절 순서는 profile.hand_joint_names 그대로."""
        v = [float(x) for x in txt.split(",")]
        if len(v) != n_act - n_arm:
            raise SystemExit(f"손 액션은 {n_act - n_arm}개여야 한다, got {len(v)}")
        return torch.tensor(v, device=e.device).unsqueeze(0)

    _HA = _parse_hand(args.hand_approach) if args.hand_approach else None
    _HC = _parse_hand(args.hand_close) if args.hand_close else None

    def _drive(goal_offset, steps: int, close: float, tag: str, hand=None):
        """케이지를 '물체 + offset' 으로 몰면서 손을 지령한다.

        `hand` 가 주어지면 관절별 정규화 액션을 그대로 쓰고, 없으면 기존처럼
        스칼라 `close` 를 6관절에 균일 적용한다(대각선만 훑는 구판 거동).
        """
        for i in range(steps):
            obj = e._env_local(e.object.data.root_pos_w)
            palm = e._env_local(e.robot.data.body_pos_w[:, e.palm_idx])
            cage = palm + (e._palm_ee_R() @ e._cage_offset_palm)
            err = (obj + torch.tensor(goal_offset, device=e.device)) - cage        # (N,3)
            Jc, _ = _cage_jacobian(e, arm_t)
            dq = torch.linalg.lstsq(
                Jc @ Jc.T + 1e-4 * torch.eye(3, device=e.device), err.T).solution
            dq = (Jc.T @ dq).T                                                     # (N, n_arm)
            act[:, :n_arm] = (dq / gain).clamp(-1.0, 1.0) * float(args.approach_gain)
            act[:, n_arm:] = hand if hand is not None else (2.0 * close - 1.0)
            _prev = e.episode_length_buf.clone()
            env.step(act)
            tally["resets"] += int((e.episode_length_buf < _prev).sum())
            for _k, _v in e.extras.items():
                if _k.startswith("done/") and not _k.endswith("qd_max") and float(_v) > 0.0:
                    _n = _k.split("/")[-1]
                    tally[_n] = tally.get(_n, 0.0) + float(_v) * e.num_envs
            if i % 50 == 0 or i == steps - 1:
                _report(e, f"{tag}+{i}")

    if int(args.random_steps) > 0:
        import math as _m
        jn = e.robot.data.joint_names
        names = [jn[int(i)] for i in arm_t]
        q0 = e.robot.data.joint_pos[:, arm_t].clone()
        p0 = e._env_local(e.robot.data.body_pos_w[:, e.palm_idx]).clone()
        R0 = e._palm_ee_R().clone()
        travel = torch.zeros(len(names), device=e.device)
        prev = q0.clone()
        for i in range(int(args.random_steps)):
            a = (torch.rand(e.num_envs, n_act, device=e.device) * 2.0 - 1.0)
            env.step(a)
            q = e.robot.data.joint_pos[:, arm_t]
            travel += (q - prev).abs().mean(dim=0)
            prev = q.clone()
        p1 = e._env_local(e.robot.data.body_pos_w[:, e.palm_idx])
        R1 = e._palm_ee_R()
        # 손 회전각 = trace 로부터 (R0ᵀR1 의 회전각)
        _rel = torch.matmul(R0.transpose(1, 2), R1)
        _tr = _rel[:, 0, 0] + _rel[:, 1, 1] + _rel[:, 2, 2]
        _ang = torch.acos(((_tr - 1.0) / 2.0).clamp(-1.0, 1.0)) * 180.0 / _m.pi
        _disp = (p1 - p0).norm(dim=-1)
        print(f"[scripted] 무작위 {args.random_steps} 스텝 (env {e.num_envs}개)", flush=True)
        print("[scripted] 관절별 누적 이동(rad): "
              + " · ".join(f"{n.split('_')[-1]} {float(v):.2f}" for n, v in zip(names, travel)),
              flush=True)
        print(f"[scripted] 손: 이동 {float(_disp.mean()) * 1000:.0f}mm · 회전 {float(_ang.mean()):.0f}° "
              f"(최대 {float(_ang.max()):.0f}°)", flush=True)
        _lim = e.robot.data.soft_joint_pos_limits[0, arm_t]
        _q = e.robot.data.joint_pos[0, arm_t]
        print("[scripted] 관절별 한계 대비 위치: "
              + " · ".join(f"{n.split('_')[-1]} {float((_q[k] - _lim[k,0]) / (_lim[k,1] - _lim[k,0])):.2f}"
                           for k, n in enumerate(names)), flush=True)
        _report(e, "random-final")
        env.close()
        return

    if int(args.inhand) > 0:
        # ★손안 판정: 접근·리셋·보상을 배제하고 "닫으면 잡히는가"만 본다.
        #   물체를 케이지 중심에 놓고, 손가락이 도달할 때까지 운동학적으로 붙잡았다가 놓는다.
        if _HC is None:
            raise SystemExit("--inhand 에는 --hand-close 가 필요하다")
        _hold = int(args.inhand)

        def _cage_world():
            palm = e.robot.data.body_pos_w[:, e.palm_idx]
            return palm + (e._palm_ee_R() @ e._cage_offset_palm)

        for i in range(30):                      # 홈에서 정착
            env.step(torch.zeros(e.num_envs, n_act, device=e.device))
        _report(e, "inhand-settle")

        for i in range(_hold + int(args.inhand_watch)):
            act[:, :n_arm] = 0.0
            act[:, n_arm:] = _HC
            if i < _hold:                        # 손가락이 올 때까지 물체를 제자리에 고정
                st = e.object.data.root_state_w.clone()
                st[:, :3] = _cage_world()
                st[:, 7:] = 0.0
                e.object.write_root_state_to_sim(st)
            env.step(act)
            if i in (0, _hold - 1, _hold, _hold + 30, _hold + 100,
                     _hold + int(args.inhand_watch) - 1):
                _report(e, ("hold" if i < _hold else "free") + f"+{i}")
        # 판정: 물체가 케이지를 따라왔는가 · 상판으로 돌아갔는가
        _obj = e._env_local(e.object.data.root_pos_w)
        _cage = e._env_local(e.robot.data.body_pos_w[:, e.palm_idx]) + (e._palm_ee_R() @ e._cage_offset_palm)
        _d = (_cage - _obj).norm(dim=-1)
        _rest = 0.205 + float(cfg.object_grasp_z_offset)
        print(f"[scripted] 손안 판정: 케이지−물체 {float(_d.mean())*1000:.1f}mm "
              f"(env별 {[round(float(v)*1000) for v in _d]}) · "
              f"물체 z {float(_obj[:, 2].mean()):.4f} · 케이지 z {float(_cage[:, 2].mean()):.4f} · "
              f"상판 안착 z ≈ {_rest:.4f}", flush=True)
        env.close()
        return

    if int(args.zero_steps) > 0:
        # ★가만히 두기 시험 — 액션 0. 여기서 리셋이 나면 원인은 정책·접근이 아니다.
        tally, resets = {}, 0
        prev = e.episode_length_buf.clone()
        for i in range(int(args.zero_steps)):
            env.step(torch.zeros(e.num_envs, n_act, device=e.device))
            for k, v in e.extras.items():
                if k.startswith("done/") and not k.endswith("qd_max") and float(v) > 0.0:
                    tally[k.split("/")[-1]] = tally.get(k.split("/")[-1], 0.0) + float(v)
            now = e.episode_length_buf
            resets += int((now < prev).sum())
            prev = now.clone()
            if i % 100 == 0:
                _report(e, f"zero+{i}")
        print(f"[scripted] 무액션 {args.zero_steps} 스텝: 리셋 {resets}회 "
              f"(env {e.num_envs}개) · 종료사유 누적 { {k: round(v, 3) for k, v in tally.items()} }",
              flush=True)
        _report(e, "zero-final")
        env.close()
        return

    print(f"[scripted] hold_mode={cfg.synergy_hold_mode} · blocked_thr={cfg.blocked_err_thr_rad}"
          f" · 손 kp/effort → 포화오차 ≈ 0.2 rad", flush=True)
    _report(e, "reset")
    # ★파지중심은 물체 원점 + `object_grasp_z_offset` — env 가 컵 파지 높이로 쓰는 값이다.
    _gz = float(cfg.object_grasp_z_offset)
    _drive((0.0, 0.0, _gz), args.approach_steps, 0.0, "approach", _HA)
    _drive((0.0, 0.0, _gz), args.close_steps, 1.0, "close", _HC)
    _drive((0.0, 0.0, _gz + float(args.lift_height)), args.lift_steps, 1.0, "lift", _HC)
    _report(e, "final")
    print(f"[scripted] 누적: { {k: (int(v) if k == 'resets' else round(v, 1)) for k, v in tally.items()} }"
          f" · bank={cfg.object_bank} grasp_z={cfg.object_grasp_z_offset} "
          f"ang_damp={cfg.object_angular_damping} mass={cfg.object_mass_override or 0.134}",
          flush=True)
    print("[scripted] 판정: 대본으로 들리면 (A) 탐색·보상 문제 · 안 들리면 (B) 기하·게이트 문제",
          flush=True)
    env.close()


main()
_app.close()
